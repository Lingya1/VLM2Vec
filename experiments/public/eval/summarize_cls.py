#!/usr/bin/env python
"""汇总 MMEB 评测结果：按 in-domain / zero-shot 分组打印 hit@1 并给出均值。

用法: python summarize_cls.py <eval_output_dir> [task]
  task 取 cls（默认）或 vqa。省略时为 cls，保持与既有 CLS 脚本的调用方式兼容。
"""
import json
import os
import sys

TASK_GROUPS = {
    "cls": {
        "in_domain": ["ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397"],
        "zero_shot": ["Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"],
    },
    "vqa": {
        "in_domain": ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"],
        "zero_shot": ["ScienceQA", "VizWiz", "GQA", "TextVQA"],
    },
}


def load_hit1(out_dir, name):
    path = os.path.join(out_dir, f"{name}_score.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("hit@1")


def render(out_dir, group_name, names, width):
    rows, vals = [], []
    for name in names:
        hit1 = load_hit1(out_dir, name)
        if hit1 is None:
            rows.append(f"  {name:<{width}s} (未完成)")
        else:
            rows.append(f"  {name:<{width}s} {hit1 * 100:6.2f}")
            vals.append(hit1)
    print(f"\n{group_name}  (hit@1 %)")
    print("\n".join(rows))
    if vals:
        print(f"  {'-- 均值':<{width}s} {sum(vals) / len(vals) * 100:6.2f}   ({len(vals)}/{len(names)} 个子集)")
    return vals


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = sys.argv[1]
    task = (sys.argv[2] if len(sys.argv) > 2 else "cls").lower()
    if task not in TASK_GROUPS:
        print(f"未知的 task: {task}（可选 {'/'.join(TASK_GROUPS)}）")
        sys.exit(1)

    groups = TASK_GROUPS[task]
    width = max(len(n) for n in groups["in_domain"] + groups["zero_shot"]) + 1
    in_vals = render(out_dir, "in-domain（训练过）", groups["in_domain"], width)
    zs_vals = render(out_dir, "zero-shot（未训练）", groups["zero_shot"], width)
    allv = in_vals + zs_vals
    if allv:
        print(f"\n全部 {len(allv)} 个子集总均值: {sum(allv) / len(allv) * 100:.2f}")


if __name__ == "__main__":
    main()
