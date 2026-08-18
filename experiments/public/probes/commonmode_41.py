"""对 §4.1（T=1 权重强行多循环）做与 §4.2 相同的共模分解。

必要性：§4.2 发现 T=4 权重前 5 圈的几何变化几乎纯共模（保序、不影响排序），
而 §0/§4.6 又用"最终 embedding 各向异性上升"当作读出失效的证据。
若 §4.1 的各向异性上升同样是共模，那 §0 的机制叙述就与 §4.2 自相矛盾。
这里直接算，看它到底是不是。

模型：cos → α² + (1−α²)·cos，用 neg_mean 定 α²，再去预测 pos。
残差 = 实测 pos − 预测 pos。残差显著为负 = 存在共模解释不掉的真实判别性损失。
"""
import json
import os

import numpy as np

FILES = {
    "T=1 权重 (§4.1)": "/home/zhoutuowen/VLM2Vec/output/contraction_vqa6_T1M5_okvqa.json",
    "T=4 权重 (§4.2)": "/home/zhoutuowen/VLM2Vec/output/contraction_vqa6_T4M5_okvqa_T12.json",
    "T=4 权重 留出集": "/home/zhoutuowen/VLM2Vec/output/heldout_sweep_T4M5_okvqa.json",
    "T=1 权重 留出集": "/home/zhoutuowen/VLM2Vec/output/heldout_sweep_T1M5_okvqa.json",
}


def analyse(name, path, ref_idx=0):
    if not os.path.exists(path):
        print(f"[跳过] {name}: 缺 {os.path.basename(path)}\n")
        return
    rows = json.load(open(path))["rows"]
    rows = sorted(rows, key=lambda r: r["T"])
    ref = rows[ref_idx]
    p0, n0 = ref["pos"], ref["neg_mean"]

    print(f"=== {name}  (基准 T={ref['T']}) ===")
    print(f"{'T':>3s} {'a2':>8s} {'pos预测':>9s} {'pos实测':>9s} {'残差':>9s} "
          f"{'去共模pos':>10s} {'aniso':>8s} {'R@1':>8s}")
    dec, r1 = [], []
    for r in rows:
        denom = 1.0 - n0
        a2 = (r["neg_mean"] - n0) / denom if abs(denom) > 1e-9 else 0.0
        a2c = min(max(a2, -0.99), 0.99)
        pred = a2c + (1 - a2c) * p0
        d = (r["pos"] - a2c) / (1 - a2c)
        dec.append(d)
        r1.append(r["recall@1"])
        print(f"{r['T']:>3d} {a2c:>8.4f} {pred:>9.4f} {r['pos']:>9.4f} "
              f"{r['pos'] - pred:>+9.4f} {d:>10.4f} {r['aniso']:>8.4f} {r['recall@1']:>8.4f}")

    an = [r["aniso"] for r in rows]
    if len(rows) > 2:
        print(f"\n  r(各向异性, R@1)   = {np.corrcoef(an, r1)[0, 1]:+.3f}")
        print(f"  r(去共模 pos, R@1) = {np.corrcoef(dec, r1)[0, 1]:+.3f}")
    res = [r["pos"] - (min(max((r["neg_mean"] - n0) / (1 - n0), -0.99), 0.99)
                       + (1 - min(max((r["neg_mean"] - n0) / (1 - n0), -0.99), 0.99)) * p0)
           for r in rows]
    print(f"  残差幅度：最大 |res| = {max(abs(x) for x in res):.4f}，"
          f"末点 = {res[-1]:+.4f}")
    print()


for k, v in FILES.items():
    analyse(k, v)
