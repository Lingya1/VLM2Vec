"""G-A 的零成本检验：T=4 沿用了为 T=1 调的 lr=1e-5，权重共享把有效深度 ×4。

如果学习率对 T=4 失配，训练日志里应当看得到：梯度范数系统性更大、被裁剪（clip=1.0）的
步数更多、以及裁剪后的有效步长被压缩。这些量全都已经落盘，不需要重训。

判据：
  - 若 T=4 的 grad_norm 分布与 T=1 基本重合 → 学习率失配的证据弱，G-A 可以降级
  - 若 T=4 显著更大 / 裁剪率显著更高 → 有效学习率确实被扭曲，G-A 坐实
"""
import re
import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output"
CLIP = 1.0
RUNS = [
    ("vqa6 T1M5", f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/train.log"),
    ("vqa6 T4M5", f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M5.s42/train.log"),
    ("vqa6 T1M1", f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M1.s42/train.log"),
    ("vqa6 T4M1", f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M1.s42/train.log"),
    ("okvqa T1M5", f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M5.s42/train.log"),
    ("okvqa T4M5", f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T4.M5.s42/train.log"),
    ("okvqa T1M0", f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s42/train.log"),
    ("okvqa T4M0", f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T4.M0.s42/train.log"),
]

PAT_G = re.compile(r"'grad_norm': ([0-9.eE+-]+)")
PAT_L = re.compile(r"\{'loss': ([0-9.eE+-]+)")

res = {}
for name, path in RUNS:
    try:
        with open(path) as f:
            txt = f.read()
    except FileNotFoundError:
        continue
    g = np.array([float(x) for x in PAT_G.findall(txt)])
    l = np.array([float(x) for x in PAT_L.findall(txt)])
    if len(g) == 0:
        continue
    res[name] = (g, l)

print(f"{'运行':<12s} {'步数':>5s} {'中位':>9s} {'均值':>9s} {'p90':>9s} {'最大':>10s} "
      f"{'裁剪率':>8s} {'后半裁剪':>9s} {'末段loss':>9s}")
for name, (g, l) in res.items():
    half = g[len(g) // 2:]
    print(f"{name:<12s} {len(g):>5d} {np.median(g):>9.2f} {g.mean():>9.2f} "
          f"{np.percentile(g, 90):>9.2f} {g.max():>10.1f} "
          f"{(g > CLIP).mean():>7.1%} {(half > CLIP).mean():>8.1%} "
          f"{l[-20:].mean():>9.4f}")

print()
print("配对：同数据同 M，只差 T")
for a, b in [("vqa6 T1M5", "vqa6 T4M5"), ("vqa6 T1M1", "vqa6 T4M1"),
             ("okvqa T1M5", "okvqa T4M5"), ("okvqa T1M0", "okvqa T4M0")]:
    if a not in res or b not in res:
        continue
    ga, gb = res[a][0], res[b][0]
    ha, hb = ga[len(ga) // 2:], gb[len(gb) // 2:]
    print(f"  {a:>11s} → {b:<11s}  中位 {np.median(ga):7.2f} → {np.median(gb):7.2f} "
          f"({np.median(gb) / np.median(ga):4.2f}×)   "
          f"后半裁剪率 {(ha > CLIP).mean():5.1%} → {(hb > CLIP).mean():5.1%}")

print()
print("有效步长的代理量：裁剪后 min(g,1)/g 的均值 = 梯度方向被保留的比例")
print("（=1 表示从未裁剪；越小表示 AdamW 实际迈的步越被压缩）")
for name, (g, _) in res.items():
    eff = np.minimum(g, CLIP) / g
    half = eff[len(eff) // 2:]
    print(f"  {name:<12s} 全程 {eff.mean():.4f}   后半 {half.mean():.4f}")
