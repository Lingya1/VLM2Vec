"""同一份权重、只改推理圈数：循环本身是不是一个"无差别吸引"算子？

缓存 embedding 上的分析已经给出：T=4 训出来的模型相比 T=1，正样本余弦升高 (+0.048)，
但负样本升得更多 (+0.069)，于是净判别性 10/10 子集为负。问题是那个对比换了整套权重，
"多循环了几圈"和"这套权重是在那个圈数下训的"混在一起。

本脚本把权重钉死，唯一自变量是推理时的圈数，逐圈报：

  pos        正样本余弦（对齐度）
  neg_mean   负样本余弦均值（背景拥挤度）
  neg_max    最强干扰项余弦
  margin     pos - neg_max，检索真正吃的量
  aniso      query 两两余弦均值，与缓存 embedding 上的口径一致
  eff_rank   participation ratio，表示占了多少个有效维度

判据：若 neg_mean 的增速持续快于 pos，且 margin 随圈数单调下降，那么"循环是无差别收缩"
就不再是两个模型之间的相关性，而是循环这个操作本身的性质。
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


@torch.no_grad()
def encode(model, processor, items, data_args, batch_size, desc):
    process_fn = process_vlm_inputs_fns[QWEN2_VL]
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = {"text": [t for t, _ in chunk],
                 "images": [_open_images(p) for _, p in chunk]}
        inputs = process_fn(batch, processor=processor, max_length=data_args.max_len)
        inputs = batch_to_device(inputs, model.device)
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            reps = model.encode_input(inputs)
        out.append(reps.float().cpu())
        print(f"\r  {desc}: {min(start + batch_size, len(items))}/{len(items)}", end="", flush=True)
    print()
    return torch.cat(out, dim=0).numpy()


def geometry(q, c, dup_mask):
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    sims = q @ c.T
    pos = np.diag(sims).copy()

    neg = sims.copy().astype(np.float64)
    neg[dup_mask] = np.nan
    neg_mean = np.nanmean(neg, axis=1)
    neg_max = np.nanmax(neg, axis=1)

    n = len(q)
    qq = q @ q.T
    aniso = float(qq[np.triu_indices(n, 1)].mean())

    s = np.linalg.svd(q, compute_uv=False)
    eff_rank = float(s.sum() ** 2 / (s ** 2).sum())

    ranked = np.argsort(-sims, axis=1)
    hit = np.array([dup_mask[i, ranked[i, 0]] for i in range(n)])

    return {"pos": float(pos.mean()), "neg_mean": float(neg_mean.mean()),
            "neg_max": float(neg_max.mean()), "margin": float((pos - neg_max).mean()),
            "aniso": aniso, "eff_rank": eff_rank, "recall@1": float(hit.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",
                    default="/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--image_dir", default="/home/zhoutuowen/data/MMEB-train")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--num_pairs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_t", type=int, default=6)
    ap.add_argument("--loop_start", type=int, default=17)
    ap.add_argument("--loop_end", type=int, default=27)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    reloop_pt = os.path.join(args.checkpoint_path, "reloop.pt")
    if os.path.exists(reloop_pt):
        state = torch.load(reloop_pt, map_location="cpu", weights_only=False)
    else:
        state = {"reloop_t": 1, "reloop_m": 0,
                 "reloop_loop_start": args.loop_start, "reloop_loop_end": args.loop_end}
        print("未找到 reloop.pt，按 T=1/M=0 处理")
    print(f"checkpoint 自带拓扑: {state}")

    import glob as _glob
    is_fullft = (os.path.exists(os.path.join(args.checkpoint_path, "model.safetensors"))
                 or bool(_glob.glob(os.path.join(args.checkpoint_path, "model-*.safetensors"))))
    load_from = args.checkpoint_path if is_fullft else args.model_name
    print(f"权重类型: {'全参' if is_fullft else 'LoRA'}，从 {load_from} 读取")

    model_args = ModelArguments(
        model_name=load_from, checkpoint_path=args.checkpoint_path,
        model_backbone=QWEN2_VL, lora=not is_fullft, pooling="eos", normalize=True,
        reloop_t=state["reloop_t"], reloop_m=state["reloop_m"],
        reloop_loop_start=state["reloop_loop_start"],
        reloop_loop_end=state["reloop_loop_end"])
    data_args = DataArguments()
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model = model.to("cuda", dtype=torch.bfloat16).eval()

    with torch.no_grad():
        fp = sum(float(p.float().abs().sum()) for n, p in model.encoder.named_parameters()
                 if "layers.20." in n or "layers.26." in n)
    print(f"权重指纹(层20+层26 绝对值和): {fp:.6f}")

    try:
        sched, where = find_schedule(model)
    except RuntimeError:
        from src.model.reloop import attach_recurrence
        model.encoder.config.use_cache = False
        if hasattr(model.encoder.config, "text_config"):
            model.encoder.config.text_config.use_cache = False
        sched = attach_recurrence(model.encoder, state["reloop_loop_start"],
                                  state["reloop_loop_end"], 1)
        where = "手动挂载"
    print(f"schedule 挂在 {where}: {sched}")

    queries, candidates = load_pairs(args.subset, args.num_pairs, args.image_dir, args.seed)
    print(f"{args.subset}: 取到 {len(queries)} 对")
    keys = np.array([f"{t}||{p}" for t, p in candidates])
    dup_mask = keys[None, :] == keys[:, None]

    rows = []
    for t in range(1, args.max_t + 1):
        sched.num_loops = t
        print(f"\n--- 推理 T={t} ---")
        q = encode(model, processor, queries, data_args, args.batch_size, f"T={t} query")
        c = encode(model, processor, candidates, data_args, args.batch_size, f"T={t} cand ")
        g = geometry(q, c, dup_mask)
        g["T"] = t
        rows.append(g)

    print(f"\n=== {args.subset} @ {os.path.basename(args.checkpoint_path)} "
          f"(权重训练时 T={state['reloop_t']}，下面只改推理圈数) ===")
    hdr = f"{'T':>2s} {'pos':>8s} {'neg_mean':>9s} {'neg_max':>8s} {'margin':>8s} {'aniso':>8s} {'eff_rank':>9s} {'R@1':>7s}"
    print(hdr)
    for r in rows:
        print(f"{r['T']:>2d} {r['pos']:>8.4f} {r['neg_mean']:>9.4f} {r['neg_max']:>8.4f} "
              f"{r['margin']:>8.4f} {r['aniso']:>8.4f} {r['eff_rank']:>9.1f} {r['recall@1']:>7.4f}")

    b = rows[0]
    print(f"\n相对 T=1 的累计变化（正 = 变大）:")
    print(f"{'T':>2s} {'Δpos':>9s} {'Δneg_mean':>10s} {'Δmargin':>9s} {'Δaniso':>9s}")
    for r in rows[1:]:
        print(f"{r['T']:>2d} {r['pos']-b['pos']:>+9.4f} {r['neg_mean']-b['neg_mean']:>+10.4f} "
              f"{r['margin']-b['margin']:>+9.4f} {r['aniso']-b['aniso']:>+9.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"checkpoint": args.checkpoint_path, "subset": args.subset,
                       "trained_t": state["reloop_t"], "num_pairs": len(queries),
                       "rows": rows}, f, indent=2)
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
