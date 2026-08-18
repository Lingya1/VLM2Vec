"""每条 query 的 oracle 深度分析——P2 (SUFFICE) §8.1 预注册门控实验的免训练版。

问题:自适应深度有没有天花板可吃?
  Oracle-A  两侧同深度(需要多深度索引):每条 query 取五个匹配深度里任一命中即算命中
  Oracle-B  库侧固定在 T=3(离线索引的现实约束,P2 §2/§7),query 侧自适应
  最优固定  对照:全体 query 用同一个最优深度

按 P2 §8.1 的判据:若 oracle 不能明显 Pareto 优于最优固定深度,自适应深度没有头部空间。
注意 oracle 数字天然带多重比较通胀(5 个深度里蒙对一个也算),所以额外报告
"深度带结构":逐 query 看它在哪些深度上命中,连续区间(如只在浅层/只在深层命中)
是真深度敏感,零散命中(如只在 {1,6})更像边缘样本的抖动。

用法: python3 experiments/public/eval/oracle_depth.py <checkpoint目录>
"""
import json
import os
import pickle
import sys

import numpy as np

CKPT = sys.argv[1]
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
FIXED_GALLERY_T = 3


def load(d, name):
    qry = np.asarray(pickle.load(open(os.path.join(d, f"{name}_qry"), "rb")), dtype=np.float32)
    cand = pickle.load(open(os.path.join(d, f"{name}_tgt"), "rb"))
    infos = [json.loads(l) for l in open(os.path.join(d, f"{name}_info.jsonl"))]
    labels = [i["label_name"] if isinstance(i["label_name"], list) else [i["label_name"]]
              for i in infos]
    pools = [i["cand_names"] for i in infos]
    return qry, cand, labels, pools


def hit_vector(qry, cand_dict, labels, pools, key_order):
    """逐 query 的命中 0/1 向量(官方口径:每条 query 限自己的候选列表)。"""
    cand = np.stack([np.asarray(cand_dict[k], dtype=np.float32) for k in key_order])
    key_ix = {k: i for i, k in enumerate(key_order)}
    sims = qry @ cand.T
    out = np.zeros(len(labels), dtype=bool)
    for i, (ls, pool) in enumerate(zip(labels, pools)):
        cols = np.fromiter((key_ix[c] for c in pool if c in key_ix), dtype=np.int64)
        gold = {key_ix[l] for l in ls if l in key_ix}
        if not gold or cols.size == 0:
            continue
        out[i] = cols[sims[i, cols].argmax()] in gold
    return out


ts = sorted(DEPTH_DIRS)
print(f"checkpoint: {CKPT}   深度: {ts}   库侧固定深度(Oracle-B): T={FIXED_GALLERY_T}")
print()

agg = {"fixed": [], "oa": [], "ob": [], "n": []}
for name in SUBSETS:
    data = {t: load(d, name) for t, d in DEPTH_DIRS.items()
            if os.path.exists(os.path.join(d, f"{name}_qry"))}
    if len(data) < len(ts):
        continue
    key_order = sorted(data[ts[0]][1].keys())
    labels, pools = data[ts[0]][2], data[ts[0]][3]

    # 匹配深度的命中矩阵 (n_depth, n_query)
    H = np.stack([hit_vector(data[t][0], data[t][1], labels, pools, key_order) for t in ts])
    # 库侧固定 T=3、query 侧扫深度
    HB = np.stack([hit_vector(data[t][0], data[FIXED_GALLERY_T][1], labels, pools, key_order)
                   for t in ts])

    n = H.shape[1]
    fixed_best_t = ts[int(H.mean(1).argmax())]
    fixed_best = H.mean(1).max() * 100
    oracle_a = H.any(0).mean() * 100
    oracle_b = HB.any(0).mean() * 100

    always = H.all(0).mean() * 100
    never = (~H.any(0)).mean() * 100
    dep = 100 - always - never

    # 深度带结构:在深度序 [1,2,3,4,6] 上命中模式是否为连续区间
    dep_mask = H.any(0) & ~H.all(0)
    patt = H[:, dep_mask].T  # (n_dep_query, n_depth)
    def contiguous(row):
        idx = np.flatnonzero(row)
        return idx[-1] - idx[0] + 1 == idx.size
    contig = np.array([contiguous(r) for r in patt])
    shallow_only = np.array([r[0] and not r[-1] and contiguous(r) for r in patt])
    deep_only = np.array([r[-1] and not r[0] and contiguous(r) for r in patt])

    print(f"=== {name} (n={n}) ===")
    print(f"  最优固定深度 T={fixed_best_t}: {fixed_best:.1f}   "
          f"Oracle-A(两侧同深): {oracle_a:.1f} (+{oracle_a - fixed_best:.1f})   "
          f"Oracle-B(库固定T=3): {oracle_b:.1f} (+{oracle_b - H.mean(1).max() * 100:.1f})")
    print(f"  全深度都对 {always:.1f}%   全深度都错 {never:.1f}%   深度敏感 {dep:.1f}%")
    if dep_mask.sum() > 0:
        print(f"  深度敏感者中: 连续带 {contig.mean() * 100:.0f}%"
              f" (浅带 {shallow_only.mean() * 100:.0f}%, 深带 {deep_only.mean() * 100:.0f}%)"
              f"   零散 {(~contig).mean() * 100:.0f}%")
    agg["fixed"].append(fixed_best); agg["oa"].append(oracle_a)
    agg["ob"].append(oracle_b); agg["n"].append(n)
    print()

if agg["n"]:
    print("=" * 70)
    print(f"四子集均值:  最优固定 {np.mean(agg['fixed']):.1f}   "
          f"Oracle-A {np.mean(agg['oa']):.1f} (+{np.mean(agg['oa']) - np.mean(agg['fixed']):.1f})   "
          f"Oracle-B {np.mean(agg['ob']):.1f}")
    print()
    print("判读(对应 P2 §8.1 的 kill condition):")
    print("  - Oracle-A 相对最优固定的增量就是自适应深度的理论天花板(含多重比较通胀,")
    print("    真实可实现值必定更低,门控头还要能预测对);")
    print("  - 深度敏感占比与'连续带'占比高,才说明存在真实的逐样本最优深度结构;")
    print("  - Oracle-B 度量部署约束下(库离线单深度)的版本,若明显低于 Oracle-A,")
    print("    说明跨深度失配吃掉了自适应收益,P2 §7 的跨深度对齐损失是先决条件。")
