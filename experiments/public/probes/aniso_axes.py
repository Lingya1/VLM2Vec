"""§3 声称"各向异性 +0.061，是种子噪声 0.0097 的 6.7 倍"。

那个 0.0097 取自单个 OK-VQA 子集。现在种子对在全部 10 个子集上都有缓存 embedding，
可以在同一口径下把三条轴并排量一遍，看 6.7 倍这个说法还剩多少。
"""
import os
import pickle

import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output"
IN_DOMAIN = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"]
ZERO_SHOT = ["ScienceQA", "VizWiz", "GQA", "TextVQA"]
SUBSETS = IN_DOMAIN + ZERO_SHOT

AXES = {
    "种子 s42→s43": (f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s42/eval_vqa10",
                    f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s43/eval_vqa10"),
    "深度 T1→T4": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa",
                  f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M5.s42/eval_vqa"),
    "寄存器 M1→M5": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M1.s42/eval_vqa",
                   f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa"),
}


def stats(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        o = pickle.load(f)
    x = np.asarray(o[0] if isinstance(o, tuple) else o, dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    n = len(x)
    g = x @ x.T
    aniso = float(g[np.triu_indices(n, 1)].mean())
    xc = x - x.mean(0, keepdims=True)
    lam = np.linalg.svd(xc, compute_uv=False) ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    return aniso, pr


print(f"{'轴':<14s} {'Δ各向异性':>11s} {'为正':>6s} {'ΔPR(中心化)':>13s} {'为负':>6s} "
      f"{'Δaniso in-dom':>14s} {'Δaniso zs':>11s}")
res = {}
for name, (da, db) in AXES.items():
    da_, db_, subs = [], [], []
    for s in SUBSETS:
        A, B = stats(f"{da}/{s}_qry"), stats(f"{db}/{s}_qry")
        if A is None or B is None:
            continue
        da_.append(A); db_.append(B); subs.append(s)
    if not subs:
        print(f"{name:<14s} 数据不全")
        continue
    dan = np.array([b[0] - a[0] for a, b in zip(da_, db_)])
    dpr = np.array([b[1] - a[1] for a, b in zip(da_, db_)])
    idm = np.mean([d for d, s in zip(dan, subs) if s in IN_DOMAIN])
    zsm = np.mean([d for d, s in zip(dan, subs) if s in ZERO_SHOT])
    print(f"{name:<14s} {dan.mean():>+11.4f} {int((dan > 0).sum()):>4d}/{len(dan)} "
          f"{dpr.mean():>+13.1f} {int((dpr < 0).sum()):>4d}/{len(dpr)} "
          f"{idm:>+14.4f} {zsm:>+11.4f}")
    res[name] = dan

if "种子 s42→s43" in res and "深度 T1→T4" in res:
    s, t = res["种子 s42→s43"], res["深度 T1→T4"]
    print()
    print(f"深度 / 种子 的各向异性变化幅度比 = {abs(t.mean()) / max(abs(s.mean()), 1e-12):.2f}×")
    print(f"  （初稿声称 6.7×，但那个分母取自单个 OK-VQA 子集）")
    print(f"  种子轴逐子集 |Δaniso| 最大 = {np.abs(s).max():.4f}")
    print(f"  深度轴逐子集 |Δaniso| 最小 = {np.abs(t).min():.4f}")
