"""共模论证缺的一环：聚合均值相容 ≠ 逐样本变换保序。

α² 是由两个标量（pos 均值、neg 均值）定出来的，无穷多个**非保序**的逐样本变换
会给出完全一样的均值。所以"共模是保序的 → 不预测 R@1 变化"这条推理中间是断的。

这里用缓存 embedding 直接测：对每个 query，比较第 1 圈与第 t 圈下
它对全部候选打分的秩相关（Kendall τ / Spearman ρ），以及 top-1 是否改变。
若共模模型成立，τ 应接近 1。
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr

d = np.load("/home/zhoutuowen/VLM2Vec/output/heldout_sweep_T4M5_okvqa_emb.npz")
dup = d["dup"]
Ts = sorted(int(k[1:]) for k in d.files if k.startswith("q"))


def scores(t):
    q = d[f"q{t}"]; c = d[f"c{t}"]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return q @ c.T


S = {t: scores(t) for t in Ts}
n = S[1].shape[0]


def hits(s):
    pred = s.argmax(1)
    return np.array([dup[i, pred[i]] if dup.ndim == 2 else pred[i] == i for i in range(n)])


print("每个 query 内，候选打分排序相对第 1 圈的稳定性（n=300 query × 300 候选）")
print(f"{'圈':>3s} {'Kendall τ':>12s} {'Spearman ρ':>12s} {'top-1 不变':>10s} "
      f"{'τ<0.99 的比例':>13s} {'R@1':>8s}")
h1 = hits(S[1])
base_top1 = S[1].argmax(1)
for t in Ts:
    taus, rhos = [], []
    for i in range(n):
        taus.append(kendalltau(S[1][i], S[t][i]).statistic)
        rhos.append(spearmanr(S[1][i], S[t][i]).statistic)
    taus = np.array(taus)
    same_top1 = (S[t].argmax(1) == base_top1).mean()
    print(f"{t:>3d} {np.mean(taus):>12.4f} {np.mean(rhos):>12.4f} {same_top1:>10.1%} "
          f"{(taus < 0.99).mean():>13.1%} {hits(S[t]).mean():>8.4f}")

print()
print("关键判据：若前几圈'几乎纯共模'，则那几圈的 τ 应≈1 且 top-1 应几乎不变。")
print()

# 唯一显著的 R@1 变化落在哪一段？逐圈对第 1 圈做 McNemar
from math import comb
print("相对第 1 圈的配对翻转（McNemar 精确检验）")
print(f"{'圈':>3s} {'b(1错→t对)':>11s} {'c(1对→t错)':>11s} {'净':>5s} {'双侧 p':>9s}")
for t in Ts[1:]:
    ht = hits(S[t])
    b = int((~h1 & ht).sum()); c = int((h1 & ~ht).sum())
    m = b + c
    p = min(1.0, sum(comb(m, i) for i in range(max(b, c), m + 1)) / 2 ** m * 2) if m else 1.0
    print(f"{t:>3d} {b:>11d} {c:>11d} {b-c:>5d} {p:>9.4f}")
