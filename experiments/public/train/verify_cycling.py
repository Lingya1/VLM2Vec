"""自检 interleave_datasets 的 all_exhausted 是否真的在循环重采样。

datasets>=4 的 ArrowExamplesIterable / MappedExamplesIterable 是一次性的，耗尽后
重新包一层 _HasNextIterator 会立刻再次 StopIteration，导致小数据集被静默判定为
永久耗尽 —— all_exhausted 名义上循环，实际退化成"用完即退场"。
hf_datasets.py 里在重建迭代器前显式调用了 _init_state_dict() 来修这个问题。

判据：小数据集(10 条)与大数据集(100 条)等权混合，取 200 条。
  修复后  两者各约 100 条，小数据集被循环约 10 遍
  未修复  小数据集只出现 10 条然后彻底消失

用法: python experiments/public/train/verify_cycling.py
"""
import collections

from datasets import Dataset

from src.data.dataset.hf_datasets import interleave_datasets


def main():
    small = Dataset.from_dict({"v": [f"S{i}" for i in range(10)]}).to_iterable_dataset()
    large = Dataset.from_dict({"v": [f"L{i}" for i in range(100)]}).to_iterable_dataset()

    mixed = interleave_datasets(
        [small, large], probabilities=[0.5, 0.5], batch_size=2,
        seed=42, stopping_strategy="all_exhausted",
    )

    n = 200
    got = [x["v"] for _, x in zip(range(n), iter(mixed))]
    c = collections.Counter(v[0] for v in got)
    passes = c["S"] / 10

    print(f"取 {n} 条：small={c['S']}  large={c['L']}")
    print(f"small(10 条) 被循环 {passes:.1f} 遍")
    if c["S"] <= 10:
        raise SystemExit(
            f"循环重采样未生效：small 只出现 {c['S']} 条（等于其总条数），说明耗尽后没有重新开始。\n"
            "检查 hf_datasets.py 中 CyclingMultiSourcesBatchesIterable.__iter__ 里的 _init_state_dict() 调用。"
        )
    print("循环重采样正常")


if __name__ == "__main__":
    main()
