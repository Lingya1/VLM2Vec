"""汇总 grounding 2x2 的结果，并直接算出判读所需的三个量。

为什么不只打一张分数表：这一轮的结论完全取决于交互项的符号，以及基线到底有没有拟合。
这两件事都不是从绝对分能看出来的，所以脚本自己把 ΔT / ΔM / 交互项和各格的末段 loss
斜率一起算出来，免得事后凭一列分数挑解释。

判读锚点写在输出里，来自已完成的两轮和论文附录，不要在这里改动它们。
"""
import glob
import json
import os
import re
import statistics

import numpy as np

PREFIX = os.environ.get("EXP_PREFIX", "Qwen2vl_2B.reloopft.ground")
EVAL_DIR = "eval_grounding"

# name -> (T, M)
CELLS = {"A42": (1, 0), "B42": (1, 5), "C42": (4, 0), "D42": (4, 5)}
IND = ["MSCOCO"]
OOD = ["RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"]


def cell_dir(t, m, seed=42):
    return f"output/{PREFIX}.T{t}.M{m}.s{seed}"


def scores(d):
    out = {}
    for f in glob.glob(os.path.join(d, EVAL_DIR, "*_score.json")):
        name = os.path.basename(f)[: -len("_score.json")]
        out[name] = json.load(open(f))["hit@1"] * 100
    return out


def loss_tail(d):
    """末段训练 loss 与斜率。斜率是本轮唯一能判断基线是否欠拟合的量。"""
    log = os.path.join(d, "train.log")
    if not os.path.exists(log):
        return None, None
    ls = [float(x) for x in re.findall(r"'loss': ([0-9.]+)", open(log, errors="ignore").read())]
    if len(ls) < 20:
        return None, None
    tail = ls[-max(10, len(ls) // 5):]
    k, _ = np.polyfit(np.arange(len(tail)), tail, 1)
    return float(np.mean(tail)), float(k)


got = {}
for name, (t, m) in CELLS.items():
    d = cell_dir(t, m)
    s = scores(d)
    if s:
        got[name] = {"dir": d, "s": s, "loss": loss_tail(d)}

if not got:
    print(f"还没有任何结果。找的是 output/{PREFIX}.T*.M*.s42/{EVAL_DIR}/*_score.json")
    raise SystemExit(0)

subsets = IND + OOD
print("=" * 78)
print(f"Grounding 2x2  ({PREFIX})")
print("=" * 78)
print(f"{'格':<6}{'T':>3}{'M':>3}  " + "".join(f"{n[:15]:>16}" for n in subsets))
print("-" * 78)
for name in ["A42", "B42", "C42", "D42"]:
    if name not in got:
        print(f"{name:<6}{CELLS[name][0]:>3}{CELLS[name][1]:>3}  (无结果)")
        continue
    s = got[name]["s"]
    t, m = CELLS[name]
    row = "".join(f"{s.get(n, float('nan')):>16.2f}" for n in subsets)
    print(f"{name:<6}{t:>3}{m:>3}  " + row)

print()
print(f"{'格':<6}{'域内(MSCOCO)':>14}{'零样本(3子集)':>16}{'四子集均值':>13}"
      f"{'末段loss':>11}{'斜率/步':>11}{'拟合状态':>11}")
print("-" * 78)
agg = {}
for name in ["A42", "B42", "C42", "D42"]:
    if name not in got:
        continue
    s = got[name]["s"]
    ind = statistics.mean(s[n] for n in IND if n in s) if any(n in s for n in IND) else float("nan")
    ood = statistics.mean(s[n] for n in OOD if n in s) if any(n in s for n in OOD) else float("nan")
    alls = statistics.mean(s.values())
    agg[name] = {"ind": ind, "ood": ood, "all": alls}
    lo, k = got[name]["loss"]
    state = "-" if k is None else ("仍在降" if k < -0.003 else ("微降" if k < -0.0005 else "已平"))
    lo_s = f"{lo:.3f}" if lo is not None else "-"
    k_s = f"{k:+.5f}" if k is not None else "-"
    print(f"{name:<6}{ind:>14.2f}{ood:>16.2f}{alls:>13.2f}{lo_s:>11}{k_s:>11}{state:>11}")


def delta(a, b, key):
    if a in agg and b in agg:
        return agg[b][key] - agg[a][key]
    return None


print()
print("=" * 78)
print("判读")
print("=" * 78)
for key, label in [("ood", "零样本三子集"), ("ind", "域内 MSCOCO"), ("all", "四子集均值")]:
    dt0 = delta("A42", "C42", key)
    dt5 = delta("B42", "D42", key)
    dm1 = delta("A42", "B42", key)
    dm4 = delta("C42", "D42", key)
    inter = None if (dm1 is None or dm4 is None) else dm4 - dm1
    f = lambda v: "  缺格" if v is None else f"{v:+6.2f}"
    print(f"\n{label}:")
    print(f"  深度增益  ΔT@M=0 = C42-A42 {f(dt0)}     ΔT@M=5 = D42-B42 {f(dt5)}")
    print(f"  读出增益  ΔM@T=1 = B42-A42 {f(dm1)}     ΔM@T=4 = D42-C42 {f(dm4)}")
    print(f"  交互项    {f(inter)}")

print()
print("-" * 78)
print("参照锚点（已完成的两轮 + 论文附录，勿改）")
print("-" * 78)
print("  论文附录 2x2 (MMEB-V2 All):   M=0: 60.6 -> 61.8 (ΔT +1.2)")
print("                                M=5: 60.6 -> 63.2 (ΔT +2.6)   交互项 +1.4")
print("  论文 image grounding:          83.9  对 VLM2Vec-V2 的 77.3 (+6.6)")
print("  第一轮 OK-VQA (18k 样本):      ΔT@M=0 +4.6   ΔT@M=5 -3.2   交互项 -7.8")
print("                                 但 T1M0 loss 0.767 斜率 -0.0075，基线欠拟合")
print("  第二轮 VQA6 (59k 样本):        ΔT@M=5 -1.47（十子集一致为负）")
print("                                 三格 loss 均收敛到 0.43 且已平")
print("  种子噪声下限:                  1.2 分 (OK-VQA T1M0 两个种子 52.2 vs 51.0)")
print()
print("怎么下结论：")
print("  1. 先看 A42 的斜率。若 < -0.003，基线仍欠拟合，本轮 ΔM 不能当机制证据，")
print("     要加 TARGET_SAMPLES 重跑，否则会重复第一轮那个 +10 分的假象。")
print("  2. 再看零样本那一列的 ΔT@M=5。转正且超过 1.2 分 -> VQA6 的负号是任务不对，")
print("     结构有效，这条线可以继续。仍为负 -> 是结构问题，与数据无关。")
print("  3. 交互项的符号是与论文对齐与否的判据：论文 +1.4（互补），")
print("     我们第一轮 -7.8（互相替代）。若这一轮仍显著为负，说明")
print("     'register 与深度是同一个缺陷的两种修法' 在任务之间是稳定的。")
