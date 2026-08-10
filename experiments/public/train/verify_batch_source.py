"""校验 interleave 的全局批构成是否符合设计意图。

修复前的实测（本脚本首版跑出，已作为基线记录在此）：
    4 卡 / 每卡 64 / interleave_batch 32 时，每个全局批只含 **1.00** 个源，
    且连续两个 step 都是同一个源 —— 同源连续长度实际为 512 而非配置的 128。

两个成因：
  1. RandomlyCyclingMultiSourcesBatchesIterable.shard_data_sources 把同一个 generator
     原样传给每个分片，所有 rank 与 dataloader worker 抽到完全相同的源序列。
  2. mixed_dataset.py 把 interleave_batch_size 乘了 world_size，但这个计数发生在单个
     rank 自己的数据流里（IterableDataset 下每个 worker 独立产出整批），
     导致一个同源块横跨 world_size 个 step。
  3. 附带：num_shards 只按 dataloader worker 数算，4 卡切完每卡只剩 1 个分片，
     每卡 4 个 worker 里只有 1 个真正在取数。

本脚本对着解析期望做确定性校验，不依赖 GPU 与真实数据。

用法:
    cd /home/zhoutuowen/VLM2Vec
    PYTHONNOUSERSITE=1 PYTHONPATH=. python experiments/public/train/verify_batch_source.py
"""
import collections
import itertools

from datasets import Dataset
from datasets.distributed import split_dataset_by_node

from src.data.dataset.hf_datasets import interleave_datasets

WORLD_SIZE = 4
PER_DEVICE = 64
INTERLEAVE_BATCH = 32  # --interleave_batch_size，修复后按卡计
NUM_WORKERS = 4
NUM_SOURCES = 6
ROWS_PER_SOURCE = 40000
NUM_STEPS = 40

GLOBAL_BATCH = PER_DEVICE * WORLD_SIZE
NUM_SHARDS = NUM_WORKERS * WORLD_SIZE
BLOCKS_PER_GLOBAL_BATCH = GLOBAL_BATCH // INTERLEAVE_BATCH
BLOCKS_PER_RANK_STEP = PER_DEVICE // INTERLEAVE_BATCH


def build():
    datasets_ = []
    for s in range(NUM_SOURCES):
        d = Dataset.from_dict(
            {"src": [s] * ROWS_PER_SOURCE, "i": list(range(ROWS_PER_SOURCE))}
        )
        datasets_.append(d.to_iterable_dataset(num_shards=NUM_SHARDS))
    probs = [1.0 / NUM_SOURCES] * NUM_SOURCES
    return interleave_datasets(
        datasets_,
        probabilities=probs,
        batch_size=INTERLEAVE_BATCH,
        seed=42,
        stopping_strategy="all_exhausted",
    )


def run_lengths(seq):
    out, cur = [], 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            cur += 1
        else:
            out.append(cur)
            cur = 1
    out.append(cur)
    return out


def main():
    print(
        f"global_batch={GLOBAL_BATCH}  per_device={PER_DEVICE}  world_size={WORLD_SIZE}  "
        f"num_shards={NUM_SHARDS}  可用源数={NUM_SOURCES}"
    )
    print(
        f"设计意图：每个全局批 {BLOCKS_PER_GLOBAL_BATCH} 个同源块，块长 {INTERLEAVE_BATCH}；"
        f"每卡每步 {BLOCKS_PER_RANK_STEP} 块\n"
    )

    ds = build()
    per_rank = [
        iter(split_dataset_by_node(ds, rank=r, world_size=WORLD_SIZE))
        for r in range(WORLD_SIZE)
    ]

    distinct_per_batch = []
    all_run_lengths = []
    steps_with_identical_ranks = 0
    rank_streams = [[] for _ in range(WORLD_SIZE)]

    for _ in range(NUM_STEPS):
        batch, first_of_each_rank = [], []
        for r in range(WORLD_SIZE):
            seq = [ex["src"] for ex in itertools.islice(per_rank[r], PER_DEVICE)]
            rank_streams[r].extend(seq)
            first_of_each_rank.append(seq[0])
            batch.extend(seq)
        distinct_per_batch.append(len(set(batch)))
        if len(set(first_of_each_rank)) == 1:
            steps_with_identical_ranks += 1

    for r in range(WORLD_SIZE):
        all_run_lengths.extend(run_lengths(rank_streams[r]))

    # 校验 1：单卡数据流里，换源只发生在 interleave_batch 的整数倍边界上
    bad = [n for n in all_run_lengths[:-WORLD_SIZE] if n % INTERLEAVE_BATCH != 0]
    check_block = not bad
    print(
        f"[1] 块长对齐：{len(all_run_lengths)} 个同源段，"
        f"{'全部' if check_block else f'有 {len(bad)} 个不'}是 {INTERLEAVE_BATCH} 的整数倍"
    )

    # 校验 2：各 rank 的源序列已去相关（修复前 4 卡永远同源，该值应为 100%）
    identical_ratio = steps_with_identical_ranks / NUM_STEPS
    chance = (1.0 / NUM_SOURCES) ** (WORLD_SIZE - 1)
    check_decorr = identical_ratio < 0.2
    print(
        f"[2] 跨卡去相关：{steps_with_identical_ranks}/{NUM_STEPS} "
        f"({identical_ratio:.0%}) 的 step 里 4 卡起始源相同，"
        f"随机情况下应约 {chance:.1%}，修复前为 100%"
    )

    # 校验 3：每个全局批的不同源数是否贴合解析期望
    expected = NUM_SOURCES * (1 - (1 - 1 / NUM_SOURCES) ** BLOCKS_PER_GLOBAL_BATCH)
    observed = sum(distinct_per_batch) / len(distinct_per_batch)
    check_distinct = abs(observed - expected) < 0.5
    print(
        f"[3] 全局批源多样性：实测均值 {observed:.2f}，"
        f"解析期望 {expected:.2f} = S(1-(1-1/S)^B)，S={NUM_SOURCES} B={BLOCKS_PER_GLOBAL_BATCH}"
    )
    print(f"    分布：{dict(sorted(collections.Counter(distinct_per_batch).items()))}")

    print()
    if check_block and check_decorr and check_distinct:
        print("三项校验全部通过，修复生效。")
    else:
        failed = [
            n
            for n, ok in zip(["块长对齐", "跨卡去相关", "源多样性"],
                             [check_block, check_decorr, check_distinct])
            if not ok
        ]
        print(f"未通过：{', '.join(failed)}")


if __name__ == "__main__":
    main()
