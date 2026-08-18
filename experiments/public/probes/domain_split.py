"""深度与种子的**定性**差别：深度是否特异地伤害域外迁移？

10 个子集里 6 个在 vqa6 训练混合里（in-domain），4 个从没训过（zero-shot）。
如果深度只是"另一次随机重训"，它对两组的影响应当没有系统差别；
如果深度特异地伤害泛化，zero-shot 那组会明显更差，而种子轴不该有这个模式。

这是一个比"宏平均差多少"更难被 n=1 噪声伪造的判据：它要求噪声不仅幅度对得上，
还要恰好按 in-domain / zero-shot 分层。
"""
import json
import os
from math import erf, sqrt

import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output"
IN_DOMAIN = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"]
ZERO_SHOT = ["ScienceQA", "VizWiz", "GQA", "TextVQA"]

AXES = {
    "种子 s42→s43": (f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s42/eval_vqa10",
                    f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s43/eval_vqa10"),
    "深度 T1→T4": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa",
                  f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M5.s42/eval_vqa"),
    "寄存器 M1→M5": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M1.s42/eval_vqa",
                   f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa"),
}


def hits(d, sub):
    p = f"{d}/{sub}_pred.jsonl"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return np.array([1 if json.loads(l)["prediction"][0] in set(json.loads(l)["label"])
                         else 0 for l in f])


def mcnemar(a, b):
    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    if b01 + b10 == 0:
        return 0.0, 1.0
    z = (b01 - b10) / sqrt(b01 + b10)
    return z, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


print(f"{'轴':<14s} {'in-domain':>11s} {'zero-shot':>11s} {'差 (zs − id)':>13s} "
      f"{'zs 组 McNemar':>15s}")
store = {}
for name, (da, db) in AXES.items():
    per = {}
    for sub in IN_DOMAIN + ZERO_SHOT:
        ha, hb = hits(da, sub), hits(db, sub)
        if ha is None or hb is None:
            continue
        per[sub] = (hb.mean() - ha.mean()) * 100

    idv = [per[s] for s in IN_DOMAIN if s in per]
    zsv = [per[s] for s in ZERO_SHOT if s in per]
    if not idv or not zsv:
        print(f"{name:<14s} 数据不全")
        continue

    za = np.concatenate([hits(da, s) for s in ZERO_SHOT if s in per])
    zb = np.concatenate([hits(db, s) for s in ZERO_SHOT if s in per])
    z, p = mcnemar(za, zb)
    print(f"{name:<14s} {np.mean(idv):>+11.2f} {np.mean(zsv):>+11.2f} "
          f"{np.mean(zsv) - np.mean(idv):>+13.2f} {f'z={z:+.2f} p={p:.1e}':>15s}")
    store[name] = (idv, zsv, per)

print()
print("逐子集明细（zero-shot 用 [zs] 标出）")
hdr = f"{'子集':<18s}" + "".join(f"{k:>15s}" for k in store)
print(hdr)
for s in IN_DOMAIN + ZERO_SHOT:
    tag = " [zs]" if s in ZERO_SHOT else ""
    line = f"{s + tag:<18s}"
    for k in store:
        line += f"{store[k][2].get(s, float('nan')):>+15.2f}"
    print(line)

print()
print("配对自助：zero-shot 组的宏平均与 in-domain 组之差（逐 query 重采样，10000 次）")
rng = np.random.default_rng(0)
for name, (da, db) in AXES.items():
    if name not in store:
        continue
    dq = {}
    for sub in IN_DOMAIN + ZERO_SHOT:
        ha, hb = hits(da, sub), hits(db, sub)
        if ha is not None and hb is not None:
            dq[sub] = hb.astype(float) - ha.astype(float)
    out = []
    for _ in range(10000):
        idm = np.mean([dq[s][rng.integers(0, len(dq[s]), len(dq[s]))].mean()
                       for s in IN_DOMAIN if s in dq])
        zsm = np.mean([dq[s][rng.integers(0, len(dq[s]), len(dq[s]))].mean()
                       for s in ZERO_SHOT if s in dq])
        out.append((zsm - idm) * 100)
    lo, hi = np.percentile(out, [2.5, 97.5])
    print(f"  {name:<14s} 差 = {np.mean(out):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"P(差>0) = {np.mean(np.array(out) > 0):.4f}")
