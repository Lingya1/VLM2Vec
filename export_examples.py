"""
导出 FashionIQ 各排名区间的示例到 example 文件夹
包括：查询图片、正确目标图片、模型检索到的图片、查询指令文本
"""

import json
import os
import shutil
import random
import argparse


def load_predictions(eval_dir, dataset_name):
    preds, infos = [], []
    with open(os.path.join(eval_dir, f"{dataset_name}_pred.jsonl")) as f:
        for line in f:
            preds.append(json.loads(line))
    with open(os.path.join(eval_dir, f"{dataset_name}_info.jsonl")) as f:
        for line in f:
            infos.append(json.loads(line))
    return preds, infos


def load_hf_dataset(dataset_name):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    return load_dataset("ziyjiang/MMEB_Test_Instruct", dataset_name, split="test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FashionIQ")
    parser.add_argument("--eval_dir", type=str,
                        default="/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.imageonly.lora16.BS256.4A40/eval_image")
    parser.add_argument("--image_root", type=str,
                        default="/home/zhoutuowen/data/MMEB-V2/image-tasks")
    parser.add_argument("--output_dir", type=str,
                        default="/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.imageonly.lora16.BS256.4A40/examples")
    parser.add_argument("--samples_per_bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    preds, infos = load_predictions(args.eval_dir, args.dataset)
    ds = load_hf_dataset(args.dataset)

    buckets = [
        ("top1", 1, 1),
        ("top2-5", 2, 5),
        ("top6-10", 6, 10),
        ("top11-20", 11, 20),
        ("top21-50", 21, 50),
        ("top51-100", 51, 100),
        ("top100+", 101, 99999),
    ]

    rank_to_idx = {}
    for i, pred in enumerate(preds):
        label = pred["label"][0]
        for r, p in enumerate(pred["prediction"]):
            if p == label:
                rank_to_idx[i] = r + 1
                break

    bucket_indices = {name: [] for name, _, _ in buckets}
    for idx, rank in rank_to_idx.items():
        for name, lo, hi in buckets:
            if lo <= rank <= hi:
                bucket_indices[name].append(idx)
                break

    dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    all_examples = []

    for bucket_name, lo, hi in buckets:
        indices = bucket_indices[bucket_name]
        if not indices:
            continue

        n = min(args.samples_per_bucket, len(indices))
        sampled = random.sample(indices, n)

        bucket_dir = os.path.join(dataset_dir, bucket_name)
        os.makedirs(bucket_dir, exist_ok=True)

        for si, idx in enumerate(sampled):
            pred = preds[idx]
            row = ds[idx]
            rank = rank_to_idx[idx]

            example_dir = os.path.join(bucket_dir, f"sample_{si+1}_rank{rank}")
            os.makedirs(example_dir, exist_ok=True)

            qry_img_src = os.path.join(args.image_root, row["qry_img_path"])
            tgt_img_paths = row["tgt_img_path"]
            correct_img_src = os.path.join(args.image_root, tgt_img_paths[0])

            top1_pred_name = pred["prediction"][0].rstrip(":")
            retrieved_img_src = os.path.join(args.image_root, top1_pred_name)

            qry_ext = os.path.splitext(qry_img_src)[1] or ".jpg"
            cor_ext = os.path.splitext(correct_img_src)[1] or ".jpg"
            ret_ext = os.path.splitext(retrieved_img_src)[1] or ".jpg"

            qry_dst = os.path.join(example_dir, f"1_query{qry_ext}")
            cor_dst = os.path.join(example_dir, f"2_correct_target{cor_ext}")
            ret_dst = os.path.join(example_dir, f"3_model_retrieved{ret_ext}")

            for src, dst in [(qry_img_src, qry_dst), (correct_img_src, cor_dst), (retrieved_img_src, ret_dst)]:
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"  [WARNING] 图片不存在: {src}")

            info = {
                "dataset": args.dataset,
                "sample_index": idx,
                "rank_bucket": bucket_name,
                "correct_answer_rank": rank,
                "query_instruction": row.get("qry_inst", "").strip(),
                "query_text": row.get("qry_text", ""),
                "query_image": row.get("qry_img_path", ""),
                "correct_target_image": tgt_img_paths[0] if tgt_img_paths else "",
                "model_top1_retrieved": top1_pred_name,
                "model_top5": [p.rstrip(":") for p in pred["prediction"][:5]],
                "is_correct": rank == 1,
                "files": {
                    "query": f"1_query{qry_ext}",
                    "correct": f"2_correct_target{cor_ext}",
                    "retrieved": f"3_model_retrieved{ret_ext}",
                }
            }

            with open(os.path.join(example_dir, "info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            readme_lines = [
                f"# {args.dataset} - {bucket_name} (排名 #{rank})",
                "",
                f"**样本索引**: {idx}",
                f"**正确答案排名**: {rank}",
                "",
                "## 查询",
                f"**指令**: {row.get('qry_inst', '').strip()}",
                f"**文本描述**: {row.get('qry_text', '')}",
                "",
                f"**查询图片**: ![query](1_query{qry_ext})",
                "",
                "## 正确目标",
                f"**图片路径**: {tgt_img_paths[0]}",
                f"![correct](2_correct_target{cor_ext})",
                "",
                "## 模型检索结果 (Top-1)",
                f"**图片路径**: {top1_pred_name}",
                f"![retrieved](3_model_retrieved{ret_ext})",
                "",
                "## 模型 Top-5",
            ]
            for r, p in enumerate(pred["prediction"][:5], 1):
                marker = " **<-- 正确答案**" if p == pred["label"][0] else ""
                readme_lines.append(f"{r}. `{p.rstrip(':')}`{marker}")

            with open(os.path.join(example_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(readme_lines))

            all_examples.append(info)
            print(f"  [{bucket_name}] sample_{si+1}: idx={idx}, rank={rank}")

    summary_lines = [
        f"# {args.dataset} 评估错误分析示例",
        "",
        f"总样本数: {len(preds)}",
        f"hit@1: {sum(1 for r in rank_to_idx.values() if r == 1) / len(preds) * 100:.1f}%",
        "",
        "## 各排名区间样本数",
        "",
        "| 区间 | 样本数 | 占比 | 抽样数 |",
        "|------|--------|------|--------|",
    ]
    for name, lo, hi in buckets:
        cnt = len(bucket_indices[name])
        pct = cnt / len(preds) * 100
        sampled_cnt = min(args.samples_per_bucket, cnt)
        summary_lines.append(f"| {name} | {cnt} | {pct:.1f}% | {sampled_cnt} |")

    summary_lines += [
        "",
        "## 目录结构",
        "每个示例文件夹包含:",
        "- `1_query.jpg` - 查询图片",
        "- `2_correct_target.jpg` - 正确目标图片",
        "- `3_model_retrieved.jpg` - 模型检索到的图片 (Top-1)",
        "- `info.json` - 详细信息 (指令、文本、排名等)",
        "- `README.md` - 可视化说明",
    ]

    with open(os.path.join(dataset_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    with open(os.path.join(dataset_dir, "all_examples.json"), "w", encoding="utf-8") as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)

    print(f"\n完成！示例已保存到: {dataset_dir}")
    print(f"共导出 {len(all_examples)} 个示例")


if __name__ == "__main__":
    main()
