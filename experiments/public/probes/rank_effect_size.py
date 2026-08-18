"""把 §3.1 的分位排名位移换算成"名次"时，必须用每个子集自己的候选池大小。

文档此前两处都写错了：§3.1 说统一 1000，§5 也说 1000。实际是每子集的唯一候选数，
范围 736~6850。同时区分两件事：
  (a) 全体 query 的平均名次位移  —— 分位差 × 池大小
  (b) hit@1 的变化               —— 只由跨过第 1 名那条线的极少数 query 贡献
两者不是同一件事，(a) 既不能支撑也不能反驳 (b)。
"""
import glob, os, pickle
import numpy as np

BASE = "/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.reloopft.vqa6.{}.s42/eval_vqa"


def load(tag, sub):
    d = BASE.format(tag)
    with open(f"{d}/{sub}_qry", "rb") as f:
        q = pickle.load(f)
    with open(f"{d}/{sub}_tgt", "rb") as f:
        t = pickle.load(f)
    return q, t


def qrank(tag, sub):
    q, t = load(tag, sub)
    keys = list(t.keys())
    C = np.stack([np.asarray(t[k]) for k in keys]).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    idx = {k: i for i, k in enumerate(keys)}
    Q, gold = [], []
    for k, v in q.items():
        reps, tgts = v if isinstance(v, tuple) else (v, None)
        break
    # qry 结构：dict[qid] -> (rep, gold_text)；逐条取
    for qid, v in q.items():
        rep, g = (v[0], v[1]) if isinstance(v, (tuple, list)) and len(v) == 2 else (v, None)
        if g is None or g not in idx:
            continue
        Q.append(np.asarray(rep)); gold.append(idx[g])
    Q = np.stack(Q).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
    S = Q @ C.T
    gold = np.array(gold)
    gs = S[np.arange(len(gold)), gold]
    rank = (S > gs[:, None]).sum(1)                 # 0-based 名次
    return rank / len(keys), len(keys), rank


subs = [os.path.basename(f)[:-4] for f in
        sorted(glob.glob(BASE.format("T1.M5") + "/*_tgt"))]

print(f"{'子集':16s} {'池大小':>7s} {'Δ分位':>10s} {'Δ名次':>9s} {'Δ名次(中位)':>11s}")
dq, dr, pools = [], [], []
for s in subs:
    try:
        a, n, ra = qrank("T1.M5", s)
        b, _, rb = qrank("T4.M5", s)
    except Exception as e:
        print(f"{s:16s}  跳过 ({type(e).__name__})"); continue
    m = min(len(a), len(b))
    d = (b[:m] - a[:m]).mean()
    dn = (rb[:m] - ra[:m])
    print(f"{s:16s} {n:>7d} {d:>+10.5f} {dn.mean():>+9.1f} {np.median(dn):>+11.1f}")
    dq.append(d); dr.append(dn.mean()); pools.append(n)

print()
print(f"宏平均 Δ分位 = {np.mean(dq):+.5f}   （文档报的是 −0.00211）")
print(f"池大小 {min(pools)} ~ {max(pools)}，均值 {np.mean(pools):.0f}")
print(f"折算平均名次位移：按各子集自己的池 = {np.mean(dr):+.1f} 名")
print(f"  若错按统一 1000 折算 = {np.mean(dq)*1000:+.1f} 名（文档旧口径）")
