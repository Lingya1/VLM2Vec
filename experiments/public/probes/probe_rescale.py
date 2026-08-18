"""干预实验：把循环块的残差贡献缩回去，掉的分能不能回来？

诊断说的是：pre-norm 残差流没有跨圈缩放，范数随圈数线性膨胀，suffix + 最终 RMSNorm
在一个它没见过的尺度上工作，于是"读出改善量 × ||h|| ≈ 常数"，相对贡献按 1/||h|| 衰减。

这是观测性证据。本脚本做干预：在循环块出口处令

    h_out  <-  h_in + alpha * (h_out - h_in)

其中 h_in 是循环块的入口状态（第 loop_start 层的输入，即 prefix 出口），
h_out 是最后一圈的出口。alpha=1 就是原样，alpha=1/T 是把 T 圈的累计残差摊回一圈的量级。

判据：若 alpha<1 能在不改任何权重的前提下把 R@1 与 margin 拉回来，残差尺度就是因果病灶；
若毫无变化，说明诊断错了，膨胀只是伴随现象。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/zhoutuowen/VLM2Vec")
sys.path.insert(0, "/home/zhoutuowen/VLM2Vec/experiments/public/train")

from probe_layerwise_separation import _open_images, load_pairs
from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import QWEN2_VL, load_processor, process_vlm_inputs_fns
from src.utils.basic_utils import batch_to_device


class Rescaler:
    """在循环块入口记下 h_in，在最后一圈出口把残差缩放后写回。"""

    def __init__(self, num_loops, alpha):
        self.num_loops = num_loops
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.h_in = None
        self.n_exit = 0

    def pre(self, module, args, kwargs):
        if self.h_in is None:
            h = args[0] if args else kwargs["hidden_states"]
            self.h_in = h.detach().clone()
        return None

    def post(self, module, args, output):
        self.n_exit += 1
        if self.n_exit < self.num_loops or self.alpha == 1.0:
            return output
        h = output[0]
        new = self.h_in.to(h.dtype) + self.alpha * (h - self.h_in.to(h.dtype))
        return (new,) + tuple(output[1:])


@torch.no_grad()
def encode(model, processor, items, data_args, bs, resc):
    fn = process_vlm_inputs_fns[QWEN2_VL]
    out = []
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        batch = {"text": [t for t, _ in chunk], "images": [_open_images(p) for _, p in chunk]}
        inputs = fn(batch, processor=processor, max_length=data_args.max_len)
        inputs = batch_to_device(inputs, model.device)
        if resc:
            resc.reset()
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            out.append(model.encode_input(inputs).float().cpu())
    return torch.cat(out).numpy()


def metrics(q, c, dup):
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    sims = q @ c.T
    pos = np.diag(sims).copy()
    neg = sims.astype(np.float64).copy()
    neg[dup] = np.nan
    margin = float(np.nanmean(pos - np.nanmax(neg, axis=1)))
    n = len(q)
    qq = q @ q.T
    an = float(qq[np.triu_indices(n, 1)].mean())
    top1 = np.argsort(-sims, axis=1)[:, 0]
    r1 = float(np.mean([dup[i, top1[i]] for i in range(n)]))
    return margin, an, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",
                    default="/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--image_dir", default="/home/zhoutuowen/data/MMEB-train")
    ap.add_argument("--subset", default="OK-VQA")
    ap.add_argument("--num_pairs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    rp = os.path.join(args.checkpoint_path, "reloop.pt")
    st = (torch.load(rp, map_location="cpu", weights_only=False) if os.path.exists(rp)
          else {"reloop_t": 1, "reloop_m": 0, "reloop_loop_start": 17, "reloop_loop_end": 27})
    import glob as _g
    full = (os.path.exists(os.path.join(args.checkpoint_path, "model.safetensors"))
            or bool(_g.glob(os.path.join(args.checkpoint_path, "model-*.safetensors"))))
    ma = ModelArguments(model_name=args.checkpoint_path if full else args.model_name,
                        checkpoint_path=args.checkpoint_path, model_backbone=QWEN2_VL,
                        lora=not full, pooling="eos", normalize=True,
                        reloop_t=st["reloop_t"], reloop_m=st["reloop_m"],
                        reloop_loop_start=st["reloop_loop_start"],
                        reloop_loop_end=st["reloop_loop_end"])
    da = DataArguments()
    pr = load_processor(ma, da)
    model = MMEBModel.load(ma, is_trainable=False, processor=pr).to("cuda", dtype=torch.bfloat16).eval()
    sched = None
    for _, mod in model.named_modules():
        if hasattr(mod, "recurrence") and mod.recurrence is not None:
            sched = mod.recurrence
            break
    LS, LE = st["reloop_loop_start"], st["reloop_loop_end"]
    dec = model.encoder.model
    layers = dec.layers if hasattr(dec, "layers") else dec.language_model.layers

    resc = Rescaler(1, 1.0)
    layers[LS].register_forward_pre_hook(resc.pre, with_kwargs=True)
    layers[LE - 1].register_forward_hook(resc.post)

    q, c = load_pairs(args.subset, args.num_pairs, args.image_dir, 42)
    keys = np.array([f"{t}||{p}" for t, p in c])
    dup = keys[None, :] == keys[:, None]

    print(f"\n=== {os.path.basename(args.checkpoint_path)} (训练深度 T={st['reloop_t']}) "
          f"@ {args.subset} n={len(q)} ===")
    print(f"{'推理T':>5s} {'alpha':>12s} {'margin':>8s} {'aniso':>8s} {'R@1':>8s}")
    base = {}
    for T in args.loops:
        sched.num_loops = T
        resc.num_loops = T
        for name, a in [("1.0 (原样)", 1.0), ("1/sqrt(T)", 1.0 / np.sqrt(T)), ("1/T", 1.0 / T)]:
            if T == 1 and a != 1.0:
                continue
            resc.alpha = float(a)
            qe = encode(model, pr, q, da, args.batch_size, resc)
            ce = encode(model, pr, c, da, args.batch_size, resc)
            m, an, r1 = metrics(qe, ce, dup)
            if T == 1:
                base = {"m": m, "an": an, "r1": r1}
            flag = ""
            if base and a != 1.0:
                flag = f"   (相对 T=1 基准 R@1 {r1-base['r1']:+.4f})"
            print(f"{T:>5d} {name:>12s} {m:>8.4f} {an:>8.4f} {r1:>8.4f}{flag}")


if __name__ == "__main__":
    main()
