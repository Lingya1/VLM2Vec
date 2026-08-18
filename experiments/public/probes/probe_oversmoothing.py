"""循环块是不是一个 token 级的 over-smoothing 算子？

动机：已有的测量报的是"样本之间"的各向异性（不同输入的 embedding 互相靠拢）。但 transformer
文献里 rank collapse / over-smoothing 讲的是"序列内 token 之间"互相靠拢，是另一个量。若不区分，
机制叙述就只是对掉分的重新描述，而不是解释。

本脚本在每一圈的出口测序列内的 token 几何：

  tok_cos    序列内非填充 token 两两余弦均值（over-smoothing 的标准口径）
  tok_rank   token 矩阵的 participation ratio，衡量占了几个有效方向
  res_ratio  ||H - 1·h̄ᵀ||_F / ||H||_F，偏离"所有 token 都相同"的秩一矩阵有多远，
             越小越接近完全坍缩（Dong et al. 2021 的 res 口径）

判据：若 tok_cos 随圈数单调升、res_ratio 单调降，那么样本级各向异性就是 token 级坍缩的下游
后果，机制链条闭合；若 token 级几何基本不动，那样本级的现象另有原因，"收缩算子"的说法要改。
"""
import argparse
import json
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


def find_schedule(model):
    for name, mod in model.named_modules():
        if hasattr(mod, "recurrence") and mod.recurrence is not None:
            return mod.recurrence, name
    raise RuntimeError("没找到 recurrence 调度")


def build_mask(attention_mask, seq_len, num_registers):
    """寄存器追加在序列末尾且恒为有效位（reloop.py RetrievalRegisters.extend_inputs），
    而 inputs["attention_mask"] 是扩展前的。直接拿它去索引 hidden_states 会长度不匹配，
    退化成"全部位置都算"，于是 padding 位（左侧，彼此几乎相同）会把 tok_cos 灌高。"""
    am = attention_mask.astype(bool)
    if am.shape[1] == seq_len:
        return am
    pad = seq_len - am.shape[1]
    assert pad == num_registers, f"长度差 {pad} 与寄存器数 {num_registers} 不符"
    return np.concatenate([am, np.ones((am.shape[0], pad), dtype=bool)], axis=1)


def token_geometry(H, mask):
    """H: [B,L,D]  mask: [B,L] bool。逐序列算，再对 batch 求均值。"""
    cos, rank, res = [], [], []
    for b in range(H.shape[0]):
        X = H[b][mask[b]].astype(np.float64)
        if len(X) < 3:
            continue
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        C = Xn @ Xn.T
        n = len(X)
        cos.append(C[np.triu_indices(n, 1)].mean())
        mu = X.mean(0, keepdims=True)
        # 参与比必须定义在特征值（=奇异值平方）上并先中心化；
        # 用奇异值直接算会把绝对值抬高数倍，虽然方向不变。两种都报出来以便对照。
        s = np.linalg.svd(X, compute_uv=False)
        sc = np.linalg.svd(X - mu, compute_uv=False)
        lam, lamc = s ** 2, sc ** 2
        rank.append((s.sum() ** 2 / (s ** 2).sum(),                 # 原实现（有误）
                     lam.sum() ** 2 / (lam ** 2).sum(),             # 特征值、未中心化
                     lamc.sum() ** 2 / (lamc ** 2).sum()))          # 中心化 PR（标准）
        res.append(np.linalg.norm(X - mu) / (np.linalg.norm(X) + 1e-8))
    r = np.array(rank)
    return float(np.mean(cos)), tuple(r.mean(0)), float(np.mean(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",
                    default="/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--image_dir", default="/home/zhoutuowen/data/MMEB-train")
    ap.add_argument("--subset", default="OK-VQA")
    ap.add_argument("--num_pairs", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_t", type=int, default=8)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    reloop_pt = os.path.join(args.checkpoint_path, "reloop.pt")
    state = (torch.load(reloop_pt, map_location="cpu", weights_only=False)
             if os.path.exists(reloop_pt)
             else {"reloop_t": 1, "reloop_m": 0, "reloop_loop_start": 17, "reloop_loop_end": 27})
    print(f"checkpoint 拓扑: {state}")

    import glob as _glob
    is_fullft = (os.path.exists(os.path.join(args.checkpoint_path, "model.safetensors"))
                 or bool(_glob.glob(os.path.join(args.checkpoint_path, "model-*.safetensors"))))
    load_from = args.checkpoint_path if is_fullft else args.model_name

    model_args = ModelArguments(
        model_name=load_from, checkpoint_path=args.checkpoint_path,
        model_backbone=QWEN2_VL, lora=not is_fullft, pooling="eos", normalize=True,
        reloop_t=state["reloop_t"], reloop_m=state["reloop_m"],
        reloop_loop_start=state["reloop_loop_start"], reloop_loop_end=state["reloop_loop_end"])
    data_args = DataArguments()
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model = model.to("cuda", dtype=torch.bfloat16).eval()
    sched, _ = find_schedule(model)
    LS, LE = state["reloop_loop_start"], state["reloop_loop_end"]

    decoder = model.encoder.model
    layers = decoder.layers if hasattr(decoder, "layers") else decoder.language_model.layers
    captured = []
    layers[LE - 1].register_forward_hook(
        lambda m, a, out: captured.append(out[0].detach().float().cpu().numpy()))

    queries, _ = load_pairs(args.subset, args.num_pairs, args.image_dir, 42)
    process_fn = process_vlm_inputs_fns[QWEN2_VL]

    rows = []
    for t in range(1, args.max_t + 1):
        sched.num_loops = t
        per_loop = [[] for _ in range(t)]
        for start in range(0, len(queries), args.batch_size):
            chunk = queries[start:start + args.batch_size]
            batch = {"text": [x for x, _ in chunk], "images": [_open_images(p) for _, p in chunk]}
            inputs = process_fn(batch, processor=processor, max_length=data_args.max_len)
            inputs = batch_to_device(inputs, model.device)
            captured.clear()
            with torch.no_grad(), torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                model.encode_input(inputs)
            assert len(captured) == t, f"期望捕获 {t} 圈，实际 {len(captured)} 圈"
            am = inputs["attention_mask"].cpu().numpy()
            for k, H in enumerate(captured):
                m = build_mask(am, H.shape[1], state["reloop_m"])
                if state["reloop_m"] > 0:
                    m = m.copy()
                    m[:, -state["reloop_m"]:] = False   # 只统计真实 token，寄存器另算
                per_loop[k].append(token_geometry(H, m))
        agg = [(float(np.mean([r[0] for r in per_loop[k]])),
                tuple(np.mean([r[1] for r in per_loop[k]], axis=0)),
                float(np.mean([r[2] for r in per_loop[k]]))) for k in range(t)]
        rows.append({"T": t, "loops": [{"loop": k + 1, "tok_cos": a[0], "tok_rank": a[1][0],
                                        "tok_rank_eig": a[1][1], "tok_rank_ctr": a[1][2],
                                        "res_ratio": a[2]} for k, a in enumerate(agg)]})
        print(f"\n--- 推理 T={t} ---")
        print(f"{'圈':>3s} {'tok_cos':>9s} {'orig':>8s} {'eig':>8s} {'ctrPR':>8s} {'res_ratio':>10s}")
        for k, a in enumerate(agg):
            print(f"{k+1:>3d} {a[0]:>9.4f} {a[1][0]:>8.2f} {a[1][1]:>8.2f} {a[1][2]:>8.2f} {a[2]:>10.4f}")

    print(f"\n=== 汇总：每次运行的最后一圈出口（即真正喂给 suffix 的状态） ===")
    print(f"{'推理T':>5s} {'tok_cos':>9s} {'orig':>8s} {'eig':>8s} {'ctrPR':>8s} {'res_ratio':>10s}")
    for r in rows:
        a = r["loops"][-1]
        print(f"{r['T']:>5d} {a['tok_cos']:>9.4f} {a['tok_rank']:>8.2f} {a['tok_rank_eig']:>8.2f} {a['tok_rank_ctr']:>8.2f} {a['res_ratio']:>10.4f}")

    print(f"\n=== 单次运行内部逐圈演化（取 T={args.max_t} 那一行） ===")
    print(f"{'圈':>3s} {'tok_cos':>9s} {'orig':>8s} {'eig':>8s} {'ctrPR':>8s} {'res_ratio':>10s}")
    for a in rows[-1]["loops"]:
        print(f"{a['loop']:>3d} {a['tok_cos']:>9.4f} {a['tok_rank']:>8.2f} {a['tok_rank_eig']:>8.2f} {a['tok_rank_ctr']:>8.2f} {a['res_ratio']:>10.4f}")

    if args.output:
        json.dump({"checkpoint": args.checkpoint_path, "subset": args.subset,
                   "trained_t": state["reloop_t"], "rows": rows}, open(args.output, "w"), indent=2)
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
