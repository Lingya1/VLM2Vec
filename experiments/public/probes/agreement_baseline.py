"""§5 的"随机同错基线"被判为稻草人。正确的参照是**种子对的同错率**。

初稿写："两者都错时犯同一个错的比例 64.9%，而随机同错概率仅 0.015%~0.14%"。
两个共享 backbone、训练数据、且判断一致率 88.8% 的模型本来就不会均匀随机地犯错，
所以那个基线不提供任何信息。这里换成三条轴的并排比较，全部用同一套子集、同一评测协议。
"""
import json
import os

import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output"
SUBSETS = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
           "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"]

AXES = {
    "种子 s42→s43": (f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s42/eval_vqa10",
                    f"{BASE}/Qwen2vl_2B.reloopft.okvqa.T1.M0.s43/eval_vqa10"),
    "深度 T1→T4  ": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa",
                    f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T4.M5.s42/eval_vqa"),
    "寄存器 M1→M5": (f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M1.s42/eval_vqa",
                    f"{BASE}/Qwen2vl_2B.reloopft.vqa6.T1.M5.s42/eval_vqa"),
}


def load(d, sub):
    p = f"{d}/{sub}_pred.jsonl"
    if not os.path.exists(p):
        return None
    top1, ok, npool = [], [], []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            lab = set(r["label"])
            top1.append(r["prediction"][0])
            ok.append(r["prediction"][0] in lab)
            npool.append(len(r["prediction"]))
    return np.array(top1, dtype=object), np.array(ok), int(np.median(npool))


print(f"{'轴':<14s} {'子集':>3s} {'N':>6s} {'池':>5s} {'判断一致':>9s} {'top1相同':>9s} "
      f"{'都错':>7s} {'都错时同错':>11s} {'Δ分':>7s}")
for name, (da, db) in AXES.items():
    ta, tb, oa, ob, pool = [], [], [], [], []
    ns = 0
    for s in SUBSETS:
        A, B = load(da, s), load(db, s)
        if A is None or B is None:
            continue
        n = min(len(A[0]), len(B[0]))
        ta.append(A[0][:n]); oa.append(A[1][:n])
        tb.append(B[0][:n]); ob.append(B[1][:n])
        pool.append(A[2]); ns += 1
    if ns == 0:
        print(f"{name:<14s} 评测尚未完成")
        continue
    ta, tb = np.concatenate(ta), np.concatenate(tb)
    oa, ob = np.concatenate(oa), np.concatenate(ob)

    both_wrong = (~oa) & (~ob)
    same_err = (ta[both_wrong] == tb[both_wrong]).mean() if both_wrong.sum() else float("nan")
    print(f"{name:<14s} {ns:>3d} {len(ta):>6d} {int(np.median(pool)):>5d} "
          f"{(oa == ob).mean():>8.1%} {(ta == tb).mean():>8.1%} "
          f"{both_wrong.mean():>6.1%} {same_err:>10.1%} "
          f"{(ob.mean() - oa.mean()) * 100:>+7.2f}")

print()
print("读法：同错率必须与**种子轴**比，而不是与均匀随机比。")
print("若深度轴的同错率与种子轴接近，则'深度只是把函数挪到旁边'成立；")
print("若深度轴显著更低，说明深度确实改变了模型犯错的方式（不论好坏）。")
