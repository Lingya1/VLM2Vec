"""逐子集检索诊断。

训练日志只有一个混合了 6 个子集的全局 InfoNCE，看不出是哪个子集拖后腿；
而 hit@1 在不同子集之间也不可直接比较——候选池大小差一个数量级时，
同样的表示质量会给出完全不同的 hit@1。这里从评测存下的向量里重算：

  loss_pool   在该子集真实候选池上的 InfoNCE（温度与训练一致，0.02）
  loss_rand   同样池子下随机猜的 InfoNCE = ln(池大小)，作为可比的参照系
  margin      正样本余弦 减去 最强负样本余弦，表示质量的直接读数，与池大小无关

用法: python3 experiments/public/train/per_subset_diag.py <eval 目录>
"""
import json
import os
import pickle
import sys

import numpy as np

TEMP = 0.02


def diag(d, name):
    with open(os.path.join(d, f"{name}_qry"), "rb") as f:
        qry = np.asarray(pickle.load(f), dtype=np.float32)
    with open(os.path.join(d, f"{name}_tgt"), "rb") as f:
        cand_dict = pickle.load(f)
    infos = [json.loads(l) for l in open(os.path.join(d, f"{name}_info.jsonl"))]

    keys = list(cand_dict.keys())
    key_ix = {k: i for i, k in enumerate(keys)}
    cand = np.stack([np.asarray(cand_dict[k], dtype=np.float32) for k in keys])

    sims = qry @ cand.T  # 向量在评测时已归一化，点积即余弦

    losses, pos_sims, margins, top_negs = [], [], [], []
    for i, info in enumerate(infos):
        labels = info["label_name"]
        labels = labels if isinstance(labels, list) else [labels]
        gold = [key_ix[l] for l in labels if l in key_ix]
        if not gold:
            continue
        row = sims[i]
        # 多正样本时取最强的那个当作 InfoNCE 的目标，其余从负样本里剔除
        g = max(gold, key=lambda j: row[j])
        logits = row / TEMP
        m = logits.max()
        losses.append(float(m + np.log(np.exp(logits - m).sum()) - logits[g]))
        neg = np.delete(row, gold)
        pos_sims.append(float(row[g]))
        top_negs.append(float(neg.max()))
        margins.append(float(row[g] - neg.max()))

    score = json.load(open(os.path.join(d, f"{name}_score.json")))
    return dict(
        name=name,
        n_pool=len(keys),
        n_qry=len(losses),
        hit1=score["hit@1"] * 100,
        hit5=score["hit@5"] * 100,
        loss=float(np.mean(losses)),
        loss_rand=float(np.log(len(keys))),
        pos=float(np.mean(pos_sims)),
        top_neg=float(np.mean(top_negs)),
        margin=float(np.mean(margins)),
    )


def main():
    d = sys.argv[1]
    names = sorted(f[: -len("_score.json")] for f in os.listdir(d) if f.endswith("_score.json"))
    rows = [diag(d, n) for n in names]
    rows.sort(key=lambda r: r["hit1"])

    # 候选池大小在子集之间差近十倍，hit@1 和裸 loss 都不可横向比较；
    # 减去同池随机猜的 ln(N) 之后剩下的 nat 数才是与池大小无关的判别力读数。
    for r in rows:
        r["gain"] = r["loss_rand"] - r["loss"]
    rows.sort(key=lambda r: r["gain"])

    print(f"{'子集':<17}{'池大小':>8}{'hit@1':>8}{'hit@5':>8}{'loss':>8}{'随机':>7}"
          f"{'降低nat':>9}{'正样本':>8}{'最强负':>8}{'margin':>8}")
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:<17}{r['n_pool']:>8}{r['hit1']:>8.1f}{r['hit5']:>8.1f}"
              f"{r['loss']:>8.2f}{r['loss_rand']:>7.2f}{r['gain']:>9.2f}"
              f"{r['pos']:>8.3f}{r['top_neg']:>8.3f}{r['margin']:>8.3f}")
    print()
    print(f"{'均值':<17}{'':>8}{np.mean([r['hit1'] for r in rows]):>8.1f}"
          f"{np.mean([r['hit5'] for r in rows]):>8.1f}{np.mean([r['loss'] for r in rows]):>8.2f}"
          f"{'':>7}{np.mean([r['gain'] for r in rows]):>9.2f}")


if __name__ == "__main__":
    main()
