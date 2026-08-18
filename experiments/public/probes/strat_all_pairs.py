"""把 zero-shot 分层的推断放到正确的层级上，并把两对深度实验都算进来。

终审指出：文档报的 [−3.23,−0.52] 是在**子集内部对 query 重抽样**得到的，
它只控制 query 抽样误差；而"深度特异地伤害 zero-shot"是关于**子集分组**的主张，
威胁它的是子集之间的散布。这里同时报两个层级，并做 C(10,4)=210 的精确置换检验。
"""
import glob, json, os, itertools
import numpy as np
from scipy import stats

ID = {"OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"}  # 训练混合内
BASE = "/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.reloopft.{}.s42/eval_vqa"


def scores(tag):
    out = {}
    for f in glob.glob(BASE.format(tag) + "/*_score.json"):
        s = os.path.basename(f)[:-11]
        v = json.load(open(f))
        out[s] = (v if isinstance(v, (int, float)) else list(v.values())[0]) * 100
    return out


def strat(a, b, label):
    subs = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in subs])
    isid = np.array([s in ID for s in subs])
    din, dzs = d[isid], d[~isid]
    diff = dzs.mean() - din.mean()

    # 子集层：Welch t
    t, p_t = stats.ttest_ind(dzs, din, equal_var=False)
    df = (dzs.var(ddof=1)/len(dzs) + din.var(ddof=1)/len(din))**2 / (
        (dzs.var(ddof=1)/len(dzs))**2/(len(dzs)-1) + (din.var(ddof=1)/len(din))**2/(len(din)-1))
    se = np.sqrt(dzs.var(ddof=1)/len(dzs) + din.var(ddof=1)/len(din))
    crit = stats.t.ppf(0.975, df)

    # 精确置换：C(10,4) 种把 4 个子集标成 zero-shot 的分法
    k = (~isid).sum()
    perms = [np.array(c) for c in itertools.combinations(range(len(subs)), k)]
    stat = []
    for c in perms:
        m = np.zeros(len(subs), bool); m[c] = True
        stat.append(d[m].mean() - d[~m].mean())
    stat = np.array(stat)
    p_perm = (np.abs(stat) >= abs(diff) - 1e-12).mean()
    rank = int((np.abs(stat) >= abs(diff) - 1e-12).sum())

    print(f"\n=== {label} ===")
    print(f"  宏平均 {d.mean():+.2f}   in-domain {din.mean():+.2f} (n=6)   zero-shot {dzs.mean():+.2f} (n=4)")
    print(f"  分层差 {diff:+.2f}")
    print(f"  子集层 Welch: t={t:+.2f}, df={df:.2f}, 95% CI [{diff-crit*se:+.2f}, {diff+crit*se:+.2f}], p={p_t:.3f}")
    print(f"  精确置换 (C(10,{k})={len(perms)}): p={p_perm:.3f}，排名 {rank}/{len(perms)}")
    print(f"  组内 SD: in-domain {din.std(ddof=1):.2f}, zero-shot {dzs.std(ddof=1):.2f}")
    # 留一
    loo = []
    for i in range(len(subs)):
        m = np.ones(len(subs), bool); m[i] = False
        dd, ii = d[m], isid[m]
        loo.append((subs[i], dd[~ii].mean() - dd[ii].mean()))
    worst = min(loo, key=lambda x: abs(x[1]))
    print(f"  留一：范围 [{min(x[1] for x in loo):+.2f}, {max(x[1] for x in loo):+.2f}]，"
          f"影响最大的是去掉 {worst[0]} → {worst[1]:+.2f}")
    return d, diff


pairs = [("vqa6.T1.M5", "vqa6.T4.M5", "深度 M=5（头条）"),
         ("vqa6.T1.M1", "vqa6.T4.M1", "深度 M=1（文档从未报告）"),
         ("vqa6.T1.M5", "vqa6.T1.M5", None)]
ds = []
for a, b, lab in pairs:
    if lab is None: continue
    ds.append(strat(scores(a), scores(b), lab)[0])

# 两对的一致性
d5, d1 = ds
print(f"\n两对逐子集 Δ 的一致性：同号 {int((np.sign(d5)==np.sign(d1)).sum())}/10，r={np.corrcoef(d5,d1)[0,1]:+.2f}")

# 关键反驳：分层统计量是不是比宏平均"更难伪造"？
sd = d5.std(ddof=1)
sd_macro = sd / np.sqrt(10)
sd_strat = sd * np.sqrt(1/6 + 1/4)
print(f"\n=== 「分层判据比宏平均难伪造」这句话的检验 ===")
print(f"  逐子集 SD = {sd:.2f}")
print(f"  SD(宏平均) = {sd_macro:.2f}      SD(分层差) = {sd_strat:.2f}")
print(f"  分层统计量的噪声是宏平均的 {sd_strat/sd_macro:.2f} 倍 —— 它更容易被偶然造出来，不是更难")
