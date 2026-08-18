"""判决性检验：种子噪声与深度效应，在**同一套 10 个子集、同一评测协议**下直接对比。

统计审查判"不通过"的第一与第三条致命项是：
  (1) 把 10 个子集当成 10 个独立样本，实际是同一对 checkpoint 上的重复测量；
      种子噪声在 checkpoint 层面是共模的，宏平均削不掉它。
  (2) 噪声地板由 n=1（单个种子对、单个子集）反推，σ 的 95% CI 宽达 [0.38, 27.1]。

本脚本的判据在跑之前就写死：
  - 若**种子对**也给出接近 8/10 的同号率、且宏平均漂移接近 1.2，
    则 §1 靠"符号一致性"支撑深度效应的论证当场作废。
  - 若种子对的符号是散的（例如 5/10 上下）而深度对是齐的，则该论证被大幅加强。

注意作用域：种子对是 okvqa/T1M0/94 步的权重，深度对是 vqa6/M5/410 步的权重，
两者训练配置不同。所以这给出的是"同协议下换种子能造成多大的子集级同号漂移"这一
**量级参照**，不是深度对自身的重复。这一点在解读时不能含糊。
"""
import json
import os

import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output"
SUBSETS = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
           "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"]
IN_DOMAIN = {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"}

PAIRS = {
    "种子 (okvqa T1M0 s42→s43)": (
        f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s42/eval_vqa10",
        f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s43/eval_vqa10"),
    "深度 (vqa6 M5 T1→T4)": (
        f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa",
        f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M5.s42/eval_vqa"),
    "寄存器 (vqa6 T1 M1→M5)": (
        f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M1.s42/eval_vqa",
        f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa"),
}


def hits(d, sub):
    p = f"{d}/{sub}_pred.jsonl"
    if not os.path.exists(p):
        return None
    out = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out.append(1 if r["prediction"][0] in set(r["label"]) else 0)
    return np.array(out)


def sign_test_p(k, n):
    """双侧符号检验（精确二项，p=0.5）。k = 同号数中较多的一边。"""
    from math import comb
    k = max(k, n - k)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_z(a, b):
    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    if b01 + b10 == 0:
        return 0.0, b01, b10
    return (b01 - b10) / np.sqrt(b01 + b10), b01, b10


def boot_ci(deltas_per_query, n=10000, seed=0):
    """按 query 配对自助，对 10 个子集的宏平均做 CI。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        vals = []
        for d in deltas_per_query:
            idx = rng.integers(0, len(d), len(d))
            vals.append(d[idx].mean())
        out.append(np.mean(vals) * 100)
    return np.percentile(out, [2.5, 97.5]), float(np.mean(np.array(out) > 0))


print("=" * 92)
results = {}
for name, (da, db) in PAIRS.items():
    rows, per_q = [], []
    for sub in SUBSETS:
        ha, hb = hits(da, sub), hits(db, sub)
        if ha is None or hb is None:
            continue
        n = min(len(ha), len(hb))
        ha, hb = ha[:n], hb[:n]
        z, b01, b10 = mcnemar_z(ha, hb)
        rows.append((sub, n, ha.mean() * 100, hb.mean() * 100,
                     (hb.mean() - ha.mean()) * 100, z, b01, b10))
        per_q.append(hb.astype(float) - ha.astype(float))
    if not rows:
        print(f"[等待] {name}: 评测尚未完成\n")
        continue

    d = np.array([r[4] for r in rows])
    neg = int((d < 0).sum())
    print(f"### {name}   （{len(rows)} 个子集）")
    print(f"{'子集':<18s} {'n':>5s} {'A':>7s} {'B':>7s} {'Δ':>7s} "
          f"{'McNemar z':>10s} {'0→1':>5s} {'1→0':>5s}")
    for sub, n, a, b, dd, z, b01, b10 in rows:
        mark = "*" if abs(z) > 1.96 else " "
        tag = "" if sub in IN_DOMAIN else " [zs]"
        print(f"{sub + tag:<18s} {n:>5d} {a:>7.2f} {b:>7.2f} {dd:>+7.2f} "
              f"{z:>9.2f}{mark} {b01:>5d} {b10:>5d}")

    ci, ppos = boot_ci(per_q)
    allq_a = np.concatenate([h for h in
                             [hits(da, s) for s in SUBSETS] if h is not None])
    allq_b = np.concatenate([h for h in
                             [hits(db, s) for s in SUBSETS] if h is not None])
    n = min(len(allq_a), len(allq_b))
    zt, tb01, tb10 = mcnemar_z(allq_a[:n], allq_b[:n])

    print(f"\n  宏平均 Δ = {d.mean():+.2f}   逐子集 SD = {d.std(ddof=1):.2f}   "
          f"|Δ| 中位 = {np.median(np.abs(d)):.2f}")
    print(f"  为负 {neg}/{len(d)}   符号检验双侧 p = {sign_test_p(neg, len(d)):.3f}")
    print(f"  配对自助 95% CI = [{ci[0]:+.2f}, {ci[1]:+.2f}]   P(Δ>0) = {ppos:.4f}")
    from math import erf, sqrt
    pz = 2 * (1 - 0.5 * (1 + erf(abs(zt) / sqrt(2))))
    print(f"  合并 McNemar（N={n}）: z = {zt:+.2f}，p = {pz:.2e}  "
          f"(0→1: {tb01}, 1→0: {tb10})")
    print(f"  子集内 |Δ| 最大 = {np.abs(d).max():.2f}（{rows[int(np.argmax(np.abs(d)))][0]}）")
    print()
    results[name] = d

if "种子 (okvqa T1M0 s42→s43)" in results and "深度 (vqa6 M5 T1→T4)" in results:
    s, t = results["种子 (okvqa T1M0 s42→s43)"], results["深度 (vqa6 M5 T1→T4)"]
    print("=" * 92)
    print("### 判决")
    print(f"  种子：{int((s < 0).sum())}/{len(s)} 同号（负），宏平均 {s.mean():+.2f}，"
          f"逐子集 SD {s.std(ddof=1):.2f}")
    print(f"  深度：{int((t < 0).sum())}/{len(t)} 同号（负），宏平均 {t.mean():+.2f}，"
          f"逐子集 SD {t.std(ddof=1):.2f}")
    print()
    sm = max(int((s < 0).sum()), len(s) - int((s < 0).sum()))
    tm = max(int((t < 0).sum()), len(t) - int((t < 0).sum()))
    print(f"  同号率：种子 {sm}/{len(s)} vs 深度 {tm}/{len(t)}")
    print(f"  宏平均幅度比：|深度| / |种子| = {abs(t.mean()) / max(abs(s.mean()), 1e-9):.2f}×")
    if sm >= 8 and abs(s.mean()) > 0.8:
        print("\n  → 种子也给出高同号率与相当幅度的漂移。§1 的符号一致性论证作废。")
    elif sm <= 6:
        print("\n  → 种子的符号是散的，而深度是齐的。§1 的符号一致性论证被加强。")
    else:
        print("\n  → 介于两者之间，需按幅度而非符号来判。")
