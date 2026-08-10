"""逐层正负分离度探针：在本机自训的 UME checkpoint 上复核 ReLoop-UME 的三段式观察。

ReLoop-UME(arXiv 2607.28751)的全部方法设计都挂在一个经验观察上：在训练好的单次前向
UME 模型里，逐层测正负相似度分离度 S_l，会看到"前段平 / 中后段持续上升 / 末层跳变"的
三段式，于是把参数共享的递归块放在中间那一段。它给 Qwen2-VL-2B 的划分是 0-16 / 17-26 / 27。

这个观察只需要一个训好的 checkpoint，不依赖论文代码，所以可以独立复核。本脚本做三件事：

1. **复现 S_l 曲线。** 看本机 checkpoint(image-only LoRA，20 子集)是否也落在 17-26。
   论文附录 C.2 的边界规则是确定性的三段折线变点拟合，这里按同一套约束实现。

2. **按任务族分别出曲线。** 论文只报了跨 backbone 的平均曲线，没有报不同任务族是否共享
   同一个形成区间。若 CLS 与 RET 的上升段明显错位，"一个全局固定区间"这个前提就有问题，
   而这正是它相对 PLUME 的主要卖点所在。

3. **拆开末层跳变。** 论文用"层 27 出现陡升"论证末层是 Embedding Mapping 阶段、不该被
   循环(附录 E "Why the terminal mapping layer is not looped")。但 readout 恰好只在最后
   一层之后过了一次 final RMSNorm，而 RMSNorm 的 per-channel 权重在 L2 归一化后并不会被
   约掉。本脚本同时给出两条曲线：raw(全部层一律取 decoder layer 的原始输出)与
   readout(末层换成过了 final norm 的版本)，把跳变里"层 27 的计算"与"final norm"分开。

负样本口径：逐子集独立算。跨子集的负样本(拿 ImageNet 的类名去当 MSCOCO 检索的负例)几乎
必然可分，混在一起会把 S_l 整体抬高并抹平层间差异。同一子集内还会剔除与正样本文本+图像
完全相同的候选(CLS 子集里类名大量重复，这些是真的假负例)。

用法:
    cd /home/zhoutuowen/VLM2Vec
    PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        /home/zhoutuowen/anaconda3/envs/vlm2vec/bin/python \
        experiments/public/train/probe_layerwise_separation.py \
        --checkpoint_path output/Qwen2vl_2B.imageonly.lora16.BS256.4A40 \
        --num_pairs 500 --batch_size 8
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset

from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import (QWEN2_VL, VLM_IMAGE_TOKENS, load_processor,
                                 process_vlm_inputs_fns)
from src.utils.basic_utils import batch_to_device

PHI3V_IMAGE_TOKEN = "<|image_1|>"

# 每个任务族取一个代表子集。MMEB 的 Image 侧四族：分类/问答/检索/定位。
DEFAULT_SUBSETS = {
    "ImageNet_1K": "I-CLS",
    "OK-VQA": "I-QA",
    "MSCOCO_t2i": "I-RET",
    "MSCOCO": "I-VG",
}


@dataclass
class LayerTaps:
    """挂在每个 decoder layer 上的输出捕获器。

    直接用 hook 而不是 output_hidden_states，有两个原因：其一，HF 的 hidden_states 里
    最后一项是过了 final norm 的，与前 L-1 项口径不一致，而这恰好是本脚本要检验的地方；
    其二，hook 拿到的一定是 decoder layer 自身的输出，不依赖各版本 transformers 对
    all_hidden_states 的拼装顺序。
    """

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.buffer = [None] * self.num_layers
        self._handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(layers)
        ]

    def _make_hook(self, idx):
        def hook(module, args, output):
            self.buffer[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    def clear(self):
        self.buffer = [None] * self.num_layers

    def remove(self):
        for h in self._handles:
            h.remove()


def find_decoder_layers(encoder):
    """定位语言侧 decoder 的 ModuleList。

    LoRA 已经 merge_and_unload，拿回来的是 base 类；但 Qwen2-VL 在不同 transformers 版本
    下层的挂载路径不同(model.layers / model.language_model.layers)，逐个试。
    """
    candidates = [
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
    ]
    for path in candidates:
        obj = encoder
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            return obj, path
    raise RuntimeError(
        f"未能在 {type(encoder).__name__} 上找到 decoder 层，试过：{candidates}"
    )


def load_pairs(subset, num_pairs, image_dir, seed):
    """从 MMEB-train 取 query-positive 对，返回 (query 项, candidate 项) 两个列表。

    与 mmeb_dataset.py 一致地先整体置换再截断：部分子集(SUN397/ImageNet)在 Arrow 里按类别
    排序，直接取前 N 条只会覆盖到排在最前的几十个类。
    """
    ds = load_dataset(image_dir, subset, split="original")
    ds = ds.shuffle(seed=seed)
    if num_pairs < ds.num_rows:
        ds = ds.select(range(num_pairs))

    queries, candidates = [], []
    for row in ds:
        qry_text = (row["qry"] or "").replace(PHI3V_IMAGE_TOKEN, VLM_IMAGE_TOKENS[QWEN2_VL])
        pos_text = (row["pos_text"] or "").replace(PHI3V_IMAGE_TOKEN, VLM_IMAGE_TOKENS[QWEN2_VL])
        qry_img = row["qry_image_path"] or None
        pos_img = row["pos_image_path"] or None
        if (not qry_text and not qry_img) or (not pos_text and not pos_img):
            continue
        queries.append((qry_text, os.path.join(image_dir, qry_img) if qry_img else None))
        candidates.append((pos_text, os.path.join(image_dir, pos_img) if pos_img else None))
    return queries, candidates


def _open_images(path):
    from PIL import Image
    if path is None:
        return None
    with Image.open(path) as img:
        return [img.convert("RGB")]


@torch.no_grad()
def encode_layerwise(model, processor, items, taps, data_args, batch_size, desc):
    """对一组 (text, image_path) 编码，返回每层 readout 位置的向量。

    输出 shape [N, num_layers + 1, d]，最后一维索引 0..num_layers-1 是各 decoder layer 的
    原始输出，索引 num_layers 是过了 final norm 的 readout(模型真正用来算相似度的那个)。
    """
    process_fn = process_vlm_inputs_fns[QWEN2_VL]
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = {
            "text": [t for t, _ in chunk],
            "images": [_open_images(p) for _, p in chunk],
        }
        inputs = process_fn(batch, processor=processor, max_length=data_args.max_len)
        inputs = batch_to_device(inputs, model.device)

        taps.clear()
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            result = model.encoder(**inputs, return_dict=True, output_hidden_states=True)

        attn = inputs["attention_mask"]
        # 与 MMEBModel._pooling 完全一致地定位 readout 位置，避免两套口径。
        left_padding = (attn[:, -1].sum() == attn.shape[0])
        bsz = attn.shape[0]
        if left_padding:
            idx = torch.full((bsz,), attn.shape[1] - 1, device=attn.device, dtype=torch.long)
        else:
            idx = attn.sum(dim=1) - 1
        rows = torch.arange(bsz, device=attn.device)

        per_layer = [h[rows, idx].float() for h in taps.buffer]
        per_layer.append(result.hidden_states[-1][rows, idx].float())
        out.append(torch.stack(per_layer, dim=1).cpu())

        done = min(start + batch_size, len(items))
        print(f"\r  {desc}: {done}/{len(items)}", end="", flush=True)
    print()
    return torch.cat(out, dim=0).numpy()


def separation_curve(q_vecs, c_vecs, quantiles, dup_mask):
    """S_l = mean_i [ s+_i,l - Q_q(s-_i,l) ]，逐层算。

    q_vecs / c_vecs: [N, L, d]。dup_mask[i, j] 为 True 表示候选 j 与查询 i 的正样本内容
    完全相同，算负样本分位数时要排除，否则 CLS 子集里重复的类名会把 Q_q 顶到 1 附近。
    """
    n, num_layers, _ = q_vecs.shape
    curves = {q: np.zeros(num_layers) for q in quantiles}
    for l in range(num_layers):
        qn = q_vecs[:, l, :]
        cn = c_vecs[:, l, :]
        qn = qn / (np.linalg.norm(qn, axis=1, keepdims=True) + 1e-8)
        cn = cn / (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-8)
        sims = qn @ cn.T
        pos = np.diag(sims).copy()
        neg = sims.copy()
        neg[dup_mask] = np.nan
        for q in quantiles:
            negq = np.nanquantile(neg, q, axis=1)
            curves[q][l] = float(np.mean(pos - negq))
    return curves


def _lsq_line(xs, ys):
    """返回 (斜率, 残差平方和)。单点段没有斜率，按 0 处理并记 0 残差。"""
    if len(xs) < 2:
        return 0.0, 0.0
    a, b = np.polyfit(xs, ys, 1)
    resid = float(np.sum((ys - (a * xs + b)) ** 2))
    return float(a), resid


def fit_three_segments(curves, min_prefix=4, min_middle=4):
    """论文附录 C.2 的确定性变点拟合：三段折线，中段斜率为正且大于首尾两段。

    每条曲线先归一化到 [0,1]，对所有 quantile 汇总残差；并列时先比最差 quantile 的残差，
    再比中段长度(取短的)。
    """
    normed = {}
    for q, y in curves.items():
        lo, hi = float(np.min(y)), float(np.max(y))
        normed[q] = (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)

    num_layers = len(next(iter(normed.values())))
    best_key, best_ab = None, None
    for a in range(min_prefix, num_layers):
        for b in range(a + min_middle - 1, num_layers - 1):
            total, worst, ok = 0.0, 0.0, True
            for y in normed.values():
                res_q, slopes = 0.0, []
                for s, e in ((0, a - 1), (a, b), (b + 1, num_layers - 1)):
                    xs = np.arange(s, e + 1, dtype=float)
                    slope, resid = _lsq_line(xs, y[s:e + 1])
                    slopes.append(slope)
                    res_q += resid
                if not (slopes[1] > 0 and slopes[1] > slopes[0] and slopes[1] > slopes[2]):
                    ok = False
                    break
                total += res_q
                worst = max(worst, res_q)
            if not ok:
                continue
            key = (total, worst, b - a)
            if best_key is None or key < best_key:
                best_key, best_ab = key, (a, b)
    return best_ab, (best_key[0] if best_key else None)


def ascii_curve(y, width=56, height=12):
    """无 matplotlib 时也能看趋势。横轴是层索引，纵轴线性缩放。"""
    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo if hi > lo else 1.0
    rows = []
    for r in range(height, -1, -1):
        level = lo + span * r / height
        line = "".join(
            "*" if y[int(round(c * (len(y) - 1) / (width - 1)))] >= level else " "
            for c in range(width)
        )
        rows.append(f"{level:+7.4f} |{line}")
    rows.append(" " * 8 + "+" + "-" * width)
    rows.append(" " * 9 + f"layer 0{' ' * (width - 16)}layer {len(y) - 1}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--checkpoint_path", default="output/Qwen2vl_2B.imageonly.lora16.BS256.4A40")
    ap.add_argument("--image_dir", default="/home/zhoutuowen/data/MMEB-train")
    ap.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    ap.add_argument("--num_pairs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quantiles", default="0.5,0.6,0.7,0.8")
    ap.add_argument("--output", default="output/layerwise_separation.json")
    args = ap.parse_args()

    quantiles = [float(q) for q in args.quantiles.split(",")]
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    model_args = ModelArguments(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        model_backbone=QWEN2_VL,
        lora=True,
        pooling="eos",
        normalize=True,
    )
    data_args = DataArguments()

    print(f"加载 processor 与模型：{args.checkpoint_path}")
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model = model.to("cuda", dtype=torch.bfloat16).eval()

    layers, layer_path = find_decoder_layers(model.encoder)
    taps = LayerTaps(layers)
    print(f"decoder 层：{len(layers)} 层，挂载路径 {layer_path}")

    report = {
        "checkpoint": args.checkpoint_path,
        "num_layers": len(layers),
        "num_pairs": args.num_pairs,
        "quantiles": quantiles,
        "subsets": {},
    }

    try:
        for subset in subsets:
            family = DEFAULT_SUBSETS.get(subset, "?")
            print(f"\n=== {subset} ({family}) ===")
            queries, candidates = load_pairs(subset, args.num_pairs, args.image_dir, args.seed)
            print(f"  取到 {len(queries)} 对")

            # 内容完全相同的候选算作假负例：CLS 子集的正样本就是类名，重复率极高。
            keys = [f"{t}||{p}" for t, p in candidates]
            key_arr = np.array(keys)
            dup_mask = key_arr[None, :] == key_arr[:, None]

            q_vecs = encode_layerwise(model, processor, queries, taps, data_args,
                                      args.batch_size, "query")
            c_vecs = encode_layerwise(model, processor, candidates, taps, data_args,
                                      args.batch_size, "cand ")

            num_layers = len(layers)
            # raw：全部 num_layers 层一律取 decoder layer 原始输出，口径一致。
            raw_curves = separation_curve(q_vecs[:, :num_layers], c_vecs[:, :num_layers],
                                          quantiles, dup_mask)
            # readout：末层换成过了 final norm 的版本，即模型真正用的 embedding。
            ro_q = np.concatenate([q_vecs[:, :num_layers - 1], q_vecs[:, num_layers:]], axis=1)
            ro_c = np.concatenate([c_vecs[:, :num_layers - 1], c_vecs[:, num_layers:]], axis=1)
            ro_curves = separation_curve(ro_q, ro_c, quantiles, dup_mask)

            ab_raw, _ = fit_three_segments(raw_curves)
            ab_ro, _ = fit_three_segments(ro_curves)

            main_q = 0.8 if 0.8 in quantiles else quantiles[-1]
            y = ro_curves[main_q]
            print(f"\n  S_l 曲线 (readout 口径, q={main_q}):")
            print(ascii_curve(y))
            print(f"  变点拟合  raw: {ab_raw}   readout: {ab_ro}   (论文 Qwen2-VL-2B: (17, 26))")
            print(f"  末层跳变分解: 层{num_layers-2}->层{num_layers-1}(未过norm) "
                  f"{raw_curves[main_q][-1] - raw_curves[main_q][-2]:+.4f}, "
                  f"层{num_layers-2}->readout(过norm) "
                  f"{ro_curves[main_q][-1] - ro_curves[main_q][-2]:+.4f}")

            report["subsets"][subset] = {
                "family": family,
                "num_pairs": len(queries),
                "curves_raw": {str(q): raw_curves[q].tolist() for q in quantiles},
                "curves_readout": {str(q): ro_curves[q].tolist() for q in quantiles},
                "breakpoints_raw": ab_raw,
                "breakpoints_readout": ab_ro,
            }
    finally:
        taps.remove()

    # 跨子集的宏平均曲线：论文报的是跨独立训练模型的平均，这里没有多个模型，
    # 改成跨任务族平均，用来看"全局固定区间"这个假设在任务族之间是否稳定。
    if report["subsets"]:
        pooled = {}
        for q in quantiles:
            stack = np.stack([np.array(s["curves_readout"][str(q)])
                              for s in report["subsets"].values()])
            pooled[q] = stack.mean(axis=0)
        ab_pooled, _ = fit_three_segments(pooled)
        report["pooled_readout"] = {str(q): pooled[q].tolist() for q in quantiles}
        report["breakpoints_pooled"] = ab_pooled
        print(f"\n=== 跨任务族宏平均 ===")
        print(ascii_curve(pooled[0.8 if 0.8 in quantiles else quantiles[-1]]))
        print(f"  变点拟合: {ab_pooled}   (论文 Qwen2-VL-2B: (17, 26))")
        print("  各子集: " + ", ".join(
            f"{k}={v['breakpoints_readout']}" for k, v in report["subsets"].items()))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n结果写入 {args.output}")


if __name__ == "__main__":
    sys.exit(main())
