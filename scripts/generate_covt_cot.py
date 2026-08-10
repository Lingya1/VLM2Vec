"""
Offline generation of visual reasoning chains (CoT) from CoVT model on MMEB-train.

CoVT generates mixed text + visual token reasoning chains containing:
  - <|sam_pad|> x8  (segmentation features)
  - <|depth_pad|> x4 (depth features)
  - <|dino_pad|> x4  (DINO features)

These chains are cached to disk so the VLM2Vec training pipeline can
concatenate them to original inputs and use the EOS hidden state as embedding.

Usage:
    # Single GPU
    python scripts/generate_covt_cot.py \
        --model_path weights_model/weights_models/CoVT \
        --data_dir data/MMEB-train \
        --output_dir data/MMEB-train-covt-cot \
        --gpu_id 0

    # Multi-GPU (shard by subset index)
    python scripts/generate_covt_cot.py --gpu_id 0 --shard_id 0 --num_shards 4
    python scripts/generate_covt_cot.py --gpu_id 1 --shard_id 1 --num_shards 4
    ...
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


ALL_SUBSETS = [
    "ImageNet_1K", "N24News", "HatefulMemes", "VOC2007", "SUN397",
    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
    "Visual7W", "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t",
    "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "MSCOCO",
]

REASONING_PROMPT = (
    "Use segmentation, depth map, and perception feature information "
    "of the image to answer this question."
)


def extract_task_instruction(qry_text: str) -> str:
    """Extract the task instruction from MMEB query text, stripping image tokens."""
    text = qry_text.replace("<|image_1|>", "").strip()
    if not text:
        text = "Describe the given image."
    return text


def build_messages(image_path: str, task_instruction: str):
    """Build Qwen2.5-VL chat messages for CoVT generation."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": f"{task_instruction} {REASONING_PROMPT}"},
            ],
        }
    ]


def collect_generation_tasks(df: pd.DataFrame, data_dir: str, subset_name: str, processed_keys: set):
    """Collect all (index, field, image_path, instruction) tuples that need CoT generation."""
    tasks = []
    for idx in range(len(df)):
        row = df.iloc[idx]

        qry_img = str(row.get("qry_image_path", "") or "").strip()
        if qry_img:
            key = f"{idx}_qry"
            if key not in processed_keys:
                tasks.append({
                    "index": idx,
                    "field": "qry",
                    "image_path": os.path.join(data_dir, qry_img),
                    "instruction": extract_task_instruction(str(row["qry"])),
                })

        pos_img = str(row.get("pos_image_path", "") or "").strip()
        if pos_img:
            key = f"{idx}_pos"
            if key not in processed_keys:
                pos_text = str(row.get("pos_text", "") or "")
                tasks.append({
                    "index": idx,
                    "field": "pos",
                    "image_path": os.path.join(data_dir, pos_img),
                    "instruction": extract_task_instruction(pos_text) if pos_text else "Represent the given image.",
                })

    return tasks


def generate_cot(model, processor, image_path: str, task_instruction: str, max_new_tokens: int):
    """Run CoVT generation for a single image and return the generated token IDs + text."""
    messages = build_messages(image_path, task_instruction)

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    cot_with_special = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    cot_clean = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    tok = processor.tokenizer
    token_ids = generated_ids.cpu().tolist()
    sam_count = token_ids.count(tok.convert_tokens_to_ids("<|sam_pad|>"))
    depth_count = token_ids.count(tok.convert_tokens_to_ids("<|depth_pad|>"))
    dino_count = token_ids.count(tok.convert_tokens_to_ids("<|dino_pad|>"))

    return {
        "cot_token_ids": token_ids,
        "cot_with_special_tokens": cot_with_special,
        "cot_clean": cot_clean,
        "num_tokens": len(token_ids),
        "num_sam": sam_count,
        "num_depth": depth_count,
        "num_dino": dino_count,
    }


def load_processed_keys(output_path: str) -> set:
    """Load already-processed keys from an existing JSONL file for resuming."""
    keys = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    keys.add(f"{item['index']}_{item['field']}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def main():
    parser = argparse.ArgumentParser(description="Offline CoVT visual CoT generation on MMEB-train")
    parser.add_argument("--model_path", type=str, default="weights_model/weights_models/CoVT")
    parser.add_argument("--data_dir", type=str, default="data/MMEB-train")
    parser.add_argument("--output_dir", type=str, default="data/MMEB-train-covt-cot")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="CoVT visual CoT is typically ~60 tokens; 128 gives ample buffer")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--subsets", nargs="+", default=None, help="Specific subsets to process")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-GPU")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    print(f"Loading CoVT model from {args.model_path} on GPU {args.gpu_id}...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu_id}",
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    subsets = args.subsets or ALL_SUBSETS
    # Multi-GPU sharding: each shard handles a slice of subsets
    if args.num_shards > 1:
        subsets = [s for i, s in enumerate(subsets) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: processing {subsets}")

    os.makedirs(args.output_dir, exist_ok=True)

    total_generated = 0
    total_errors = 0

    for subset_name in subsets:
        parquet_path = os.path.join(args.data_dir, subset_name, "original-00000-of-00001.parquet")
        if not os.path.exists(parquet_path):
            print(f"[SKIP] {subset_name}: parquet not found at {parquet_path}")
            continue

        output_path = os.path.join(args.output_dir, f"{subset_name}.jsonl")
        processed_keys = load_processed_keys(output_path)

        df = pd.read_parquet(parquet_path)
        tasks = collect_generation_tasks(df, args.data_dir, subset_name, processed_keys)

        print(f"\n{'='*60}")
        print(f"Subset: {subset_name} | Total rows: {len(df)} | "
              f"Already done: {len(processed_keys)} | To process: {len(tasks)}")
        print(f"{'='*60}")

        if not tasks:
            continue

        with open(output_path, "a") as fout:
            for task in tqdm(tasks, desc=subset_name):
                try:
                    cot_result = generate_cot(
                        model, processor,
                        task["image_path"],
                        task["instruction"],
                        args.max_new_tokens,
                    )
                    record = {
                        "index": int(task["index"]),
                        "field": task["field"],
                        "subset": subset_name,
                        "instruction": task["instruction"],
                        **cot_result,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    total_generated += 1

                except Exception as e:
                    record = {
                        "index": int(task["index"]),
                        "field": task["field"],
                        "subset": subset_name,
                        "error": str(e),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    total_errors += 1
                    if total_errors <= 10:
                        print(f"  [ERROR] idx={task['index']} field={task['field']}: {e}")

    print(f"\n{'='*60}")
    print(f"Done! Generated: {total_generated} | Errors: {total_errors}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
