"""汇总 OK-VQA 上的 T x M 网格，全参微调与 LoRA 并排。

网格是 T in {1,4} x M in {0,1,5}。三列各自回答一个问题：
  M=0 -> M=1  只改读出位置（因果掩码下 register 影响不到任何真实 token，
              加一个 register 唯一的作用就是把池化位置从末个真实 token 挪到 register 上）
  M=1 -> M=5  只加 register 数量（多出来的是 register 之间的级联聚合）
  T=1 -> T=4  只加循环深度

判读的前提是噪声基线：同配置换种子的差 |A42-A43|。任何小于它的差都读不出东西。
"""
import json
import os
import re

BASE = "/home/zhoutuowen/VLM2Vec/output"
PARADIGMS = [("全参微调", "Qwen2vl_2B.reloopft.okvqa"), ("LoRA(DoRA r16)", "Qwen2vl_2B.reloop.okvqa")]
TS, MS = [1, 4], [0, 1, 5]


def score(prefix, t, m, seed=42):
    f = f"{BASE}/{prefix}.T{t}.M{m}.s{seed}/eval__okvqa/OK-VQA_score.json"
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))["hit@1"] * 100
    except Exception:
        return None


def final_loss(prefix, t, m, seed=42, tail=10):
    """取最后若干步 loss 的均值。单步 loss 抖动大，末步的值不能代表收敛水平。"""
    f = f"{BASE}/{prefix}.T{t}.M{m}.s{seed}/train.log"
    if not os.path.exists(f):
        return None
    vals = re.findall(r"'loss': ([0-9.]+)", open(f, errors="ignore").read())
    if not vals:
        return None
    v = [float(x) for x in vals[-tail:]]
    return sum(v) / len(v)


def fmt(x, w=7, p=2):
    return f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-"


def main():
    print("=" * 74)
    print("OK-VQA  hit@1 (%)   —   T x M 网格")
    print("=" * 74)

    grids = {}
    for label, prefix in PARADIGMS:
        g = {(t, m): score(prefix, t, m) for t in TS for m in MS}
        grids[label] = g
        noise = None
        a42, a43 = score(prefix, 1, 0, 42), score(prefix, 1, 0, 43)
        if a42 is not None and a43 is not None:
            noise = abs(a42 - a43)

        print(f"\n【{label}】")
        print("            " + "".join(f"   M={m}   " for m in MS))
        for t in TS:
            row = "".join(fmt(g[(t, m)], 9) for m in MS)
            print(f"    T={t}   {row}")
        for t in TS:
            ls = "".join(fmt(final_loss(prefix, t, m), 9, 3) for m in MS)
            print(f"  loss T={t} {ls}")
        if noise is not None:
            print(f"    噪声基线 |T1M0(s42) - T1M0(s43)| = {noise:.2f}  ——  小于它的差读不出东西")
        else:
            print("    噪声基线: 缺种子重复，下面所有差值都没有参照物")

    print("\n" + "=" * 74)
    print("三个轴各自的效应（正数表示该改动带来提升）")
    print("=" * 74)
    for label, _ in PARADIGMS:
        g = grids[label]
        print(f"\n【{label}】")

        def d(a, b):
            x, y = g.get(a), g.get(b)
            return f"{x - y:+6.2f}" if (x is not None and y is not None) else "     -"

        for t in TS:
            print(f"    T={t}:  只改读出 (M0->M1) {d((t,1),(t,0))}   "
                  f"再加 register (M1->M5) {d((t,5),(t,1))}   "
                  f"合计 (M0->M5) {d((t,5),(t,0))}")
        for m in MS:
            print(f"    M={m}:  只加深度 (T1->T4) {d((4,m),(1,m))}    [算力约 2.07x]")

        b, a = g.get((1, 5)), g.get((1, 0))
        dd, c = g.get((4, 5)), g.get((4, 0))
        if None not in (a, b, c, dd):
            print(f"    交互项 (D-C)-(B-A) = {(dd - c) - (b - a):+.2f}   "
                  "负值意味着换过读出之后再加深度反而互相抵消")

    print("\n" + "=" * 74)
    print("全参 vs LoRA：同一格的差（正数表示全参更好）")
    print("=" * 74)
    ft, lo = grids["全参微调"], grids["LoRA(DoRA r16)"]
    for t in TS:
        cells = []
        for m in MS:
            x, y = ft.get((t, m)), lo.get((t, m))
            cells.append(f"M={m} {x - y:+6.2f}" if (x is not None and y is not None) else f"M={m}      -")
        print(f"    T={t}:  " + "   ".join(cells))
    print("\n读法：若两种训练方式给出同号同序的效应，说明之前用 LoRA 得到的结论不是低秩"
          "\n约束的产物；若符号翻转，则说明 LoRA 确实压制了循环需要的权重更新。")


if __name__ == "__main__":
    main()
