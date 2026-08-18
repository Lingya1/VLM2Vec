"""eff_rank 的实现被代码审查判定有误：拿奇异值当特征值用了参与比公式，且未中心化。

这里在缓存 embedding 上同时算四种定义，检查 §3 里"T=4 相对 T=1 有效秩 −51、10/10 一致"
这个结论会不会因为换成正确定义而翻转。

  pr_sv_uc   (Σs)²/Σs²        —— 原实现（奇异值 + 未中心化）
  pr_ev_uc   (Σs²)²/Σs⁴       —— 参与比用特征值 λ=s²，仍未中心化（二阶矩口径）
  pr_ev_c    同上但先减均值    —— 标准协方差参与比，这是文献口径
  erank      exp(熵(s²))      —— Roy & Vetterli 的 effective rank，另一族定义
"""
import json
import os
import pickle

import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output"
T1 = f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa"
T4 = f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M5.s42/eval_vqa"
SUBSETS = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
           "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"]


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    x = obj[0] if isinstance(obj, tuple) else obj
    return np.asarray(x, dtype=np.float64)


def defs(x):
    """x: (n, d)，已 L2 归一化。返回四种有效维度定义。"""
    out = {}
    s = np.linalg.svd(x, compute_uv=False)
    out["pr_sv_uc"] = s.sum() ** 2 / (s ** 2).sum()

    lam = s ** 2
    out["pr_ev_uc"] = lam.sum() ** 2 / (lam ** 2).sum()

    xc = x - x.mean(0, keepdims=True)
    sc = np.linalg.svd(xc, compute_uv=False)
    lc = sc ** 2
    out["pr_ev_c"] = lc.sum() ** 2 / (lc ** 2).sum()

    p = lc / lc.sum()
    p = p[p > 0]
    out["erank"] = float(np.exp(-(p * np.log(p)).sum()))
    return {k: float(v) for k, v in out.items()}


rows = []
for sub in SUBSETS:
    a, b = f"{T1}/{sub}_qry", f"{T4}/{sub}_qry"
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"跳过 {sub}（缺缓存）")
        continue
    xa, xb = load(a), load(b)
    xa /= np.linalg.norm(xa, axis=1, keepdims=True)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    rows.append((sub, len(xa), defs(xa), defs(xb)))

keys = ["pr_sv_uc", "pr_ev_uc", "pr_ev_c", "erank"]
print(f"{'子集':<18s} {'n':>5s} " + " ".join(f"{'Δ' + k:>12s}" for k in keys))
for sub, n, da, db in rows:
    print(f"{sub:<18s} {n:>5d} " + " ".join(f"{db[k] - da[k]:>+12.2f}" for k in keys))

print()
print(f"{'定义':<12s} {'T1 均值':>10s} {'T4 均值':>10s} {'Δ均值':>10s} {'为负子集':>10s}")
for k in keys:
    d = [db[k] - da[k] for _, _, da, db in rows]
    m1 = np.mean([da[k] for _, _, da, _ in rows])
    m4 = np.mean([db[k] for _, _, _, db in rows])
    print(f"{k:<12s} {m1:>10.1f} {m4:>10.1f} {np.mean(d):>+10.2f} "
          f"{sum(1 for v in d if v < 0):>7d}/{len(d)}")
