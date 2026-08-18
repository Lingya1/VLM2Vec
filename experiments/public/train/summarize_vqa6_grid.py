"""汇总 VQA6 四格（M in {1,5} x T in {1,4}），并与 OK-VQA 那一轮的同名效应对照。

这一轮唯一变的是数据：唯一样本 9009 -> 59009。所以要看的是同一个效应在两种数据量下
的差别，而不是绝对分（全局批 192 -> 144，负样本池变了，绝对分不跨轮比）。

评测覆盖 10 个 VQA 子集，其中 4 个训练里没出现。分开报"见过"与"没见过"两组：
循环深度若真在做多步推导，最该在没见过的分布上显现。
"""
import glob
import json
import os

BASE = "/home/zhoutuowen/VLM2Vec/output"
PREFIX = "Qwen2vl_2B.reloopft.vqa6"
EVAL_DIR = "eval_vqa"

TRAINED = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"]
CELLS = [(1, 1), (1, 5), (4, 1), (4, 5)]

# OK-VQA 那一轮（9009 条、全局批 192、2 epoch）的同名效应，用来对照数据放大的影响
PREV = {"readout_T1": 10.1, "readout_T4": 4.5, "reg_T1": 1.9, "reg_T4": -0.3,
        "depth_M1": -1.0, "depth_M5": -3.2, "noise": 1.2}


def scores(t, m):
    d = f"{BASE}/{PREFIX}.T{t}.M{m}.s42/{EVAL_DIR}"
    out = {}
    for f in glob.glob(f"{d}/*_score.json"):
        name = os.path.basename(f)[: -len("_score.json")]
        try:
            out[name] = json.load(open(f))["hit@1"] * 100
        except Exception:
            pass
    return out


def mean(d, keys):
    v = [d[k] for k in keys if k in d]
    return sum(v) / len(v) if v else None


def fmt(x, w=8, p=2):
    return f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-"


def main():
    data = {c: scores(*c) for c in CELLS}
    have = {c: d for c, d in data.items() if d}
    if not have:
        print("还没有任何评测结果。")
        return

    all_subsets = sorted({k for d in data.values() for k in d})
    unseen = [s for s in all_subsets if s not in TRAINED]

    print("=" * 78)
    print("VQA6 四格 · 逐子集 hit@1 (%)")
    print("=" * 78)
    head = "".join(f"  T{t}M{m}  " for t, m in CELLS)
    print(f"{'子集':<22}{head}")
    for s in all_subsets:
        mark = "" if s in TRAINED else "  *未见"
        print(f"{s:<22}" + "".join(fmt(data[c].get(s)) for c in CELLS) + mark)

    print("-" * 78)
    for label, keys in [("训练见过的 6 个", TRAINED), ("训练未见的", unseen), ("全部", all_subsets)]:
        if not keys:
            continue
        print(f"{label + ' 均值':<22}" + "".join(fmt(mean(data[c], keys)) for c in CELLS))

    print("\n" + "=" * 78)
    print("两个轴的效应，与 OK-VQA 那一轮对照（正数表示该改动带来提升）")
    print("=" * 78)

    def eff(group, a, b):
        x, y = mean(data[a], group), mean(data[b], group)
        return x - y if (x is not None and y is not None) else None

    def line(name, group, a, b, prev_key):
        v = eff(group, a, b)
        p = PREV[prev_key]
        s = f"    {name:<30}{fmt(v, 8)}"
        if v is not None:
            s += f"    OK-VQA 那轮 {p:+.1f}    变化 {v - p:+.1f}"
        print(s)

    for label, keys in [("训练见过的 6 个", TRAINED), ("训练未见的", unseen)]:
        if not keys:
            continue
        print(f"\n【{label}】")
        line("加 register M1->M5 (T=1)", keys, (1, 5), (1, 1), "reg_T1")
        line("加 register M1->M5 (T=4)", keys, (4, 5), (4, 1), "reg_T4")
        line("只加深度 T1->T4 (M=1)", keys, (4, 1), (1, 1), "depth_M1")
        line("只加深度 T1->T4 (M=5)", keys, (4, 5), (1, 5), "depth_M5")

    print("\n" + "=" * 78)
    print(f"读法：OK-VQA 那一轮的噪声下限是 {PREV['noise']:.1f} 分（同配置换种子）。这一轮没跑种子重复，")
    print("      沿用该量级作参照。关键看深度那两行：若 T1->T4 仍为负且幅度相当，则")
    print("      '数据太少' 这个解释被排除；若明显收敛到零甚至转正，则说明循环需要更多数据。")
    print("      未见子集那一组更关键 —— 多步推导的价值本该在没见过的分布上才显现。")


if __name__ == "__main__":
    main()
