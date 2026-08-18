"""两侧深度混配:query 用 T=a 的 embedding、候选用 T=b 的,算所有 (a,b) 组合的 hit@1。

要回答的问题
------------
强制 T 扫描发现 D42 在 T=3 读出优于其训练深度 T=4。但那是两侧同步变深的结果,
分不出"多走一圈的损害"落在哪一侧:
  - 若 (Tq=3, Tc=4) ≈ (3,3) 而 (4,3) ≈ (4,4)，损害在 query 侧;
  - 若矩阵大致对称、峰在 (3,3)，两侧同等受损;
  - 若混配(3,4)/(4,3) 反而比匹配的 (4,4) 还差，说明每走一圈整个嵌入空间在
    整体旋转——两侧必须停在同一圈才对得上，这对"部署时只砍 query 侧省算力"
    这类想法是直接否定。

不需要 GPU:强制 T 评测已把两侧向量存盘,这里只做矩阵乘法。

用法: python3 experiments/public/eval/mixmatch_depth.py <checkpoint目录>
"""
import json
import os
import pickle
import sys

import numpy as np

CKPT = sys.argv[1]
# 深度 -> 评测目录。native 深度的结果在网格评测目录 eval_grounding 里。
NATIVE_T = 4 if ".T4." in CKPT else 1
DEPTH_DIRS = {}
for t in [1, 2, 3, 4, 6]:
    d = os.path.join(CKPT, f"eval_ground_forceT{t}")
    if os.path.isdir(d):
        DEPTH_DIRS[t] = d
native = os.path.join(CKPT, "eval_grounding")
if os.path.isdir(native):
    DEPTH_DIRS[NATIVE_T] = native

SUBSETS = ["MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"]


def load(d, name):
    with open(os.path.join(d, f"{name}_qry"), "rb") as f:
        qry = np.asarray(pickle.load(f), dtype=np.float32)
    with open(os.path.join(d, f"{name}_tgt"), "rb") as f:
        cand = pickle.load(f)
    infos = [json.loads(l) for l in open(os.path.join(d, f"{name}_info.jsonl"))]
    labels, pools = [], []
    for info in infos:
        l = info["label_name"]
        labels.append(l if isinstance(l, list) else [l])
        pools.append(info["cand_names"])
    return qry, cand, labels, pools


def hit1(qry, cand_dict, labels, pools, key_order):
    """官方口径:每个 query 只在它自己的候选列表(约 1000 个)里检索。
    全局去重池有 8000+,直接在全局池里算会把所有数字压低 25-30 分且不可与
    summary.txt 对照——这一点已实测验证。"""
    cand = np.stack([np.asarray(cand_dict[k], dtype=np.float32) for k in key_order])
    key_ix = {k: i for i, k in enumerate(key_order)}
    sims = qry @ cand.T
    ok = 0
    n = 0
    for i, (ls, pool) in enumerate(zip(labels, pools)):
        cols = np.fromiter((key_ix[c] for c in pool if c in key_ix), dtype=np.int64)
        gold = {key_ix[l] for l in ls if l in key_ix}
        if not gold or cols.size == 0:
            continue
        n += 1
        top = cols[sims[i, cols].argmax()]
        ok += int(top in gold)
    return ok / n * 100


ts = sorted(DEPTH_DIRS)
print(f"checkpoint: {CKPT}   可用深度: {ts} (native T={NATIVE_T})")

per_subset = {}
for name in SUBSETS:
    data = {}
    for t, d in DEPTH_DIRS.items():
        if os.path.exists(os.path.join(d, f"{name}_qry")):
            data[t] = load(d, name)
    if len(data) < 2:
        continue
    # 一致性检查:各深度的查询数、标签序列、候选键集合必须一致,否则不可混配
    t0 = sorted(data)[0]
    ref_labels = data[t0][2]
    ref_pools = data[t0][3]
    ref_keys = set(data[t0][1].keys())
    for t in data:
        assert data[t][2] == ref_labels, f"{name}: T={t} 的标签序列与 T={t0} 不一致"
        assert set(data[t][1].keys()) == ref_keys, f"{name}: T={t} 的候选池与 T={t0} 不一致"
    key_order = sorted(ref_keys)

    mat = {}
    for tq in data:
        for tc in data:
            mat[(tq, tc)] = hit1(data[tq][0], data[tc][1], ref_labels, ref_pools, key_order)
    per_subset[name] = (sorted(data), mat)

for name, (avail, mat) in per_subset.items():
    print(f"\n=== {name} ===  行=query深度, 列=候选深度")
    print("      " + "".join(f"  Tc={t:<4}" for t in avail))
    for tq in avail:
        row = "".join(f"  {mat[(tq, tc)]:6.1f}" for tc in avail)
        diag = " <- 匹配" if False else ""
        print(f"Tq={tq:<2}{row}{diag}")

# 零样本三子集均值矩阵
zs_names = [n for n in per_subset if n != "MSCOCO"]
if zs_names:
    avail = per_subset[zs_names[0]][0]
    if all(per_subset[n][0] == avail for n in zs_names):
        print(f"\n=== 零样本三子集均值 ===  行=query深度, 列=候选深度")
        print("      " + "".join(f"  Tc={t:<4}" for t in avail))
        for tq in avail:
            row = "".join(
                f"  {np.mean([per_subset[n][1][(tq, tc)] for n in zs_names]):6.1f}"
                for tc in avail)
            print(f"Tq={tq:<2}{row}")
