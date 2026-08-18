"""zero-shot 分层效应的留一敏感性检验。

新主张（§1.2）：深度特异地伤害 zero-shot，in-domain −0.72 vs zero-shot −2.60，
分层差 −1.88。最明显的攻击点是 ScienceQA 一个子集就 −5.40，可能它撑起了整个效应。
这里逐一剔除每个子集重算，看结论有多依赖单点。

同时对种子轴做同样的事，因为"种子轴无此模式"这个否证同样可能被单点驱动。
"""
import json
import os
from math import erf, sqrt

import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output"
IN_DOMAIN = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"]
ZERO_SHOT = ["ScienceQA", "VizWiz", "GQA", "TextVQA"]

AXES = {
    "种子": (f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s42/eval_vqa10",
            f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s43/eval_vqa10"),
    "深度": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa",
            f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M5.s42/eval_vqa"),
    "寄存器": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M1.s42/eval_vqa",
             f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa"),
}


def hits(d, sub):
    p = f"{d}/{sub}_pred.jsonl"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return np.array([1 if json.loads(l)["prediction"][0] in set(json.loads(l)["label"])
                         else 0 for l in f])


def deltas(da, db):
    out = {}
    for s in IN_DOMAIN + ZERO_SHOT:
        a, b = hits(da, s), hits(db, s)
        if a is not None and b is not None:
            out[s] = (b.mean() - a.mean()) * 100
    return out


def strat(d, drop=None):
    idv = [d[s] for s in IN_DOMAIN if s in d and s != drop]
    zsv = [d[s] for s in ZERO_SHOT if s in d and s != drop]
    if not idv or not zsv:
        return None
    return np.mean(zsv) - np.mean(idv), np.mean(idv), np.mean(zsv)


D = {k: deltas(*v) for k, v in AXES.items()}

print("=== 留一敏感性：分层差 (zero-shot 均值 − in-domain 均值) ===")
print(f"{'剔除的子集':<20s} " + "".join(f"{k:>10s}" for k in D))
full = {k: strat(v)[0] for k, v in D.items()}
print(f"{'（全部 10 个）':<20s} " + "".join(f"{full[k]:>+10.2f}" for k in D))
print("-" * 52)
for s in IN_DOMAIN + ZERO_SHOT:
    tag = f"{s}{' [zs]' if s in ZERO_SHOT else ''}"
    line = f"{tag:<20s} "
    for k in D:
        r = strat(D[k], drop=s)
        line += f"{r[0]:>+10.2f}" if r else f"{'—':>10s}"
    print(line)

print()
print("=== 深度轴：分层差的符号在留一下是否稳定 ===")
vals = [strat(D["深度"], drop=s)[0] for s in IN_DOMAIN + ZERO_SHOT]
print(f"  10 次留一的分层差范围: [{min(vals):+.2f}, {max(vals):+.2f}]，"
      f"全部为负: {all(v < 0 for v in vals)}")
sv = [strat(D["种子"], drop=s)[0] for s in IN_DOMAIN + ZERO_SHOT]
print(f"  种子轴同样操作:        [{min(sv):+.2f}, {max(sv):+.2f}]，"
      f"全部为正: {all(v > 0 for v in sv)}")

print()
print("=== 换一种不依赖分组均值的检验：秩和（Mann-Whitney U）===")
from itertools import product
for k in D:
    idv = [D[k][s] for s in IN_DOMAIN if s in D[k]]
    zsv = [D[k][s] for s in ZERO_SHOT if s in D[k]]
    u = sum(1 for a, b in product(zsv, idv) if a < b) + \
        0.5 * sum(1 for a, b in product(zsv, idv) if a == b)
    n1, n2 = len(zsv), len(idv)
    print(f"  {k:<6s} U = {u:.1f} / {n1 * n2}  "
          f"(zero-shot 低于 in-domain 的成对比例 {u / (n1 * n2):.1%})")
print("  注：n1=4, n2=6 时精确检验的双侧最小可达 p 约 0.0095（U=0 或 24）。")


def perm_p(d, n=100000, seed=0):
    """置换检验：把 10 个子集的 Δ 随机分成 4+6，看分层差有多极端。

    这不依赖正态假设，也不假设子集独立于分组以外的任何结构。
    """
    rng = np.random.default_rng(seed)
    vals = np.array([d[s] for s in IN_DOMAIN + ZERO_SHOT if s in d])
    obs = strat(d)[0]
    cnt = 0
    for _ in range(n):
        p = rng.permutation(len(vals))
        zs, idx = vals[p[:4]], vals[p[4:]]
        if abs(zs.mean() - idx.mean()) >= abs(obs):
            cnt += 1
    return obs, cnt / n


print()
print("=== 置换检验（把 10 个 Δ 随机重分成 4+6，10 万次）===")
for k in D:
    obs, p = perm_p(D[k])
    print(f"  {k:<6s} 观测分层差 {obs:+.2f}，双侧 p = {p:.4f}")
