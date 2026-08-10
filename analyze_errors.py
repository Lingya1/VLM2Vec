"""
MMEB 评估错误案例分析脚本
用法: HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1 python analyze_errors.py --dataset FashionIQ --num_samples 10
"""

import json
import argparse
import os
import random

def load_predictions(eval_dir, dataset_name):
    pred_path = os.path.join(eval_dir, f"{dataset_name}_pred.jsonl")
    info_path = os.path.join(eval_dir, f"{dataset_name}_info.jsonl")
    preds, infos = [], []
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))
    with open(info_path) as f:
        for line in f:
            infos.append(json.loads(line))
    return preds, infos

def load_hf_dataset(dataset_name):
    from datasets import load_dataset
    ds = load_dataset("ziyjiang/MMEB_Test_Instruct", dataset_name, split="test")
    return ds

def analyze_dataset(dataset_name, eval_dir, image_root, num_samples=10, seed=42):
    preds, infos = load_predictions(eval_dir, dataset_name)
    ds = load_hf_dataset(dataset_name)

    total = len(preds)
    correct, wrong_indices = 0, []
    for i, pred in enumerate(preds):
        top1 = pred["prediction"][0]
        label = pred["label"][0]
        if top1 == label:
            correct += 1
        else:
            wrong_indices.append(i)

    print(f"{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"总样本数: {total}")
    print(f"正确: {correct} ({correct/total*100:.1f}%)")
    print(f"错误: {len(wrong_indices)} ({len(wrong_indices)/total*100:.1f}%)")
    print(f"{'='*70}\n")

    random.seed(seed)
    sampled = random.sample(wrong_indices, min(num_samples, len(wrong_indices)))

    for rank, idx in enumerate(sampled, 1):
        pred = preds[idx]
        row = ds[idx]

        top1_pred = pred["prediction"][0]
        label = pred["label"][0]
        top5_preds = pred["prediction"][:5]

        label_rank = -1
        for r, p in enumerate(pred["prediction"]):
            if p == label:
                label_rank = r + 1
                break

        print(f"--- 错误案例 {rank}/{len(sampled)} (样本 #{idx}) ---")
        print(f"  查询指令: {row.get('qry_inst', 'N/A').strip()}")
        print(f"  查询文本: {row.get('qry_text', 'N/A')}")

        qry_img = row.get("qry_img_path", "")
        if qry_img:
            full_qry_img = os.path.join(image_root, qry_img)
            print(f"  查询图像: {full_qry_img}")

        print(f"  正确目标: {label}")
        if "tgt_img_path" in row:
            tgt_paths = row["tgt_img_path"]
            if isinstance(tgt_paths, list) and len(tgt_paths) > 0:
                correct_img = os.path.join(image_root, tgt_paths[0])
                print(f"  正确图像: {correct_img}")

        print(f"  模型 Top-1: {top1_pred}")
        if top1_pred.startswith(dataset_name + "/") or "/" in top1_pred:
            pred_img_name = top1_pred.rstrip(":")
            pred_img = os.path.join(image_root, pred_img_name)
            print(f"  错误图像: {pred_img}")

        print(f"  正确答案排名: {label_rank if label_rank > 0 else '未在 Top-K 中找到'}")
        print(f"  模型 Top-5:")
        for r, p in enumerate(top5_preds, 1):
            marker = " <-- 正确" if p == label else ""
            print(f"    {r}. {p}{marker}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FashionIQ")
    parser.add_argument("--eval_dir", type=str,
                        default="/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.imageonly.lora16.BS256.4A40/eval_image")
    parser.add_argument("--image_root", type=str,
                        default="/home/zhoutuowen/data/MMEB-V2/image-tasks")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    analyze_dataset(args.dataset, args.eval_dir, args.image_root, args.num_samples, args.seed)
