"""§2.3 把"分侧变换、CSLS/Sinkhorn 最可能奏效"列为未测。这里测掉它。

若 hubness 修复能抹平深度赤字，那么 §2.3 的"几何 artifact"假说就该升格；
若不能，则 §2.3 的适用范围可以从"我们扫过的那一族"扩到"包括标准 hubness 修复"。
"""
import glob, json, os, pickle
import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.reloopft.{}.s42/eval_vqa"


def evaluate(tag, mode, k=10):
    accs = {}
    for f in sorted(glob.glob(BASE.format(tag) + "/*_tgt")):
        sub = os.path.basename(f)[:-4]
        Q = np.asarray(pickle.load(open(f.replace("_tgt", "_qry"), "rb")), dtype=np.float32)
        T = pickle.load(open(f, "rb"))
        info = [json.loads(l) for l in open(f.replace("_tgt", "_info.jsonl"))]
        keys = list(T.keys()); idx = {kk: i for i, kk in enumerate(keys)}
        C = np.stack([np.asarray(T[kk]) for kk in keys]).astype(np.float32)

        if mode == "sided_center":            # query 侧与候选侧各自去均值
            Q = Q - Q.mean(0, keepdims=True)
            C = C - C.mean(0, keepdims=True)

        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        S = Qn @ Cn.T

        if mode == "csls":                    # 每个候选减去它对 query 的 k 近邻均值（hubness 惩罚）
            r = np.sort(S, axis=0)[-k:].mean(0)
            S = 2 * S - r[None, :]

        hit = 0
        for i, o in enumerate(info):
            cand = [idx[c] for c in o["cand_names"] if c in idx]
            gold = idx.get(o["label_name"])
            if gold is None or not cand:
                continue
            hit += (cand[int(np.argmax(S[i, cand]))] == gold)
        accs[sub] = hit / len(info) * 100
    return accs


print(f"{'模式':16s} {'M=5 ΔT':>9s} {'M=1 ΔT':>9s}   （ΔT = T4 − T1 的宏平均）")
for mode in ["raw", "sided_center", "csls"]:
    row = []
    for m in ["M5", "M1"]:
        a = evaluate(f"vqa6.T1.{m}", mode)
        b = evaluate(f"vqa6.T4.{m}", mode)
        subs = sorted(set(a) & set(b))
        row.append(np.mean([b[s] - a[s] for s in subs]))
    print(f"{mode:16s} {row[0]:>+9.2f} {row[1]:>+9.2f}")

print("\n分子集看 CSLS 是否真的修掉了 hubness（M=5 的 T1 臂，raw → csls）：")
r0 = evaluate("vqa6.T1.M5", "raw"); r1 = evaluate("vqa6.T1.M5", "csls")
for s in sorted(r0):
    print(f"  {s:16s} {r0[s]:6.2f} → {r1[s]:6.2f}  ({r1[s]-r0[s]:+.2f})")
