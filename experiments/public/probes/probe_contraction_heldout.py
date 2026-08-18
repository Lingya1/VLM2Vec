"""§4 逐圈扫描的留出集版本。

原版 probe_contraction.py 从 MMEB-**train** 取对，而 OK-VQA 就在 vqa6 的训练混合里，
所以那里的 R@1 混进了记忆成分（统计审查的致命项之一）。这里唯一改动的是数据来源：
改从 ziyjiang/MMEB_Test_Instruct 的 test split 取对，其余（模型加载、编码、几何量、
逐圈扫描）与原版逐行一致，这样两份结果可以直接对读。

顺带修掉 eff_rank 的实现：原版把参与比公式套在奇异值上且未中心化。这里同时输出
原定义（便于与旧表对照）与标准的中心化特征值参与比。

用法：
  python probe_contraction_heldout.py --checkpoint_path <ckpt> --subset OK-VQA --max_t 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/zhoutuowen/VLM2Vec")
sys.path.insert(0, "/home/zhoutuowen/VLM2Vec/experiments/public/train")

from probe_layerwise_separation import _open_images
from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import (QWEN2_VL, load_processor,
                                 process_input_text, process_vlm_inputs_fns)
from src.utils.basic_utils import batch_to_device

TEST_HF_PATH = "ziyjiang/MMEB_Test_Instruct"
IMAGE_ROOT = "/home/zhoutuowen/data/MMEB-V2/image-tasks"


def load_pairs_heldout(subset, num_pairs, seed):
    """按 image_qa_dataset.py 的口径构造 query/positive 对，但取 test split。

    与评测保持一致的两处细节：指令前缀的拼法、以及末尾补的换行。对不上的话
    输入分布就与评测不同，几何量无从与缓存 embedding 对读。
    """
    from datasets import load_dataset
    ds = load_dataset(TEST_HF_PATH, subset, split="test")
    ds = ds.shuffle(seed=seed)
    if num_pairs < ds.num_rows:
        ds = ds.select(range(num_pairs))

    queries, candidates = [], []
    for row in ds:
        qry_inst = "\n" + (row["qry_inst"] or "").replace("<|image_1|>", "").strip()
        qry_text = process_input_text(qry_inst, QWEN2_VL, text=row["qry_text"],
                                      add_image_token=True)
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        img = row["qry_img_path"]
        queries.append((qry_text, os.path.join(IMAGE_ROOT, img) if img else None))
        candidates.append((row["tgt_text"][0], None))
    return queries, candidates


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
    eff_rank_old = float(s.sum() ** 2 / (s ** 2).sum())
    qc = q - q.mean(0, keepdims=True)
    lam = np.linalg.svd(qc, compute_uv=False) ** 2
    eff_rank = float(lam.sum() ** 2 / (lam ** 2).sum())

    ranked = np.argsort(-sims, axis=1)
    hit = np.array([dup_mask[i, ranked[i, 0]] for i in range(n)])

    return {"pos": float(pos.mean()), "neg_mean": float(neg_mean.mean()),
            "neg_max": float(neg_max.mean()), "margin": float((pos - neg_max).mean()),
            "aniso": aniso, "eff_rank": eff_rank, "eff_rank_old": eff_rank_old,
            "recall@1": float(hit.mean()), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",
                    default="/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--num_pairs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_t", type=int, default=8)
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

    queries, candidates = load_pairs_heldout(args.subset, args.num_pairs, args.seed)
    print(f"{args.subset} (test split): 取到 {len(queries)} 对")
    keys = np.array([f"{t}||{p}" for t, p in candidates])
    dup_mask = keys[None, :] == keys[:, None]
    print(f"唯一候选 {len(set(keys))} / {len(keys)}（重复答案已按 dup_mask 记为命中）")

    rows, embs = [], {}
    for t in range(1, args.max_t + 1):
        sched.num_loops = t
        print(f"\n--- 推理 T={t} ---")
        q = encode(model, processor, queries, data_args, args.batch_size, f"T={t} query")
        c = encode(model, processor, candidates, data_args, args.batch_size, f"T={t} cand ")
        g = geometry(q, c, dup_mask)
        g["T"] = t
        rows.append(g)
        embs[t] = (q.astype(np.float32), c.astype(np.float32))

    # 共模论证缺的一环：聚合均值相容不等于逐样本变换保序。把 embedding 存下来，
    # 这样可以离线直接测每个 query 内候选打分的秩相关，而不必再占 GPU。
    if args.output:
        np.savez_compressed(args.output.replace(".json", "_emb.npz"),
                            **{f"q{t}": v[0] for t, v in embs.items()},
                            **{f"c{t}": v[1] for t, v in embs.items()},
                            dup=dup_mask)

    print(f"\n=== {args.subset} [留出集] @ {os.path.basename(args.checkpoint_path)} "
          f"(权重训练时 T={state['reloop_t']}，下面只改推理圈数) ===")
    print(f"{'T':>2s} {'pos':>8s} {'neg_mean':>9s} {'neg_max':>8s} {'margin':>8s} "
          f"{'aniso':>8s} {'eff_rank':>9s} {'(旧口径)':>9s} {'R@1':>7s}")
    for r in rows:
        print(f"{r['T']:>2d} {r['pos']:>8.4f} {r['neg_mean']:>9.4f} {r['neg_max']:>8.4f} "
              f"{r['margin']:>8.4f} {r['aniso']:>8.4f} {r['eff_rank']:>9.1f} "
              f"{r['eff_rank_old']:>9.1f} {r['recall@1']:>7.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"checkpoint": args.checkpoint_path, "subset": args.subset,
                       "split": "test", "trained_t": state["reloop_t"], "rows": rows},
                      f, indent=2)
        print(f"\n已写入 {args.output}")


if __name__ == "__main__":
    main()
