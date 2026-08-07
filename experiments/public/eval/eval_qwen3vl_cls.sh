#!/bin/bash

# === Qwen3-VL LoRA 在 MMEB 上的单卡评测 ===
#
# 用法:
#   bash experiments/public/eval/eval_qwen3vl_cls.sh <CHECKPOINT_DIR>
#
# 环境变量:
#   GPU        使用哪张卡（默认 0）
#   CONFIG     评测配置（默认 experiments/public/eval/cls.yaml）
#   OUT_NAME   输出子目录名（默认 eval_cls），结果写到 <CHECKPOINT_DIR>/<OUT_NAME>
#   TASK       汇总时的分组，cls 或 vqa（默认 cls）
#   LORA       是否加载 LoRA（默认 true）。设为 false 可评测未微调的原始底座，
#              此时把 CHECKPOINT_DIR 指向基座模型目录即可。
#
# 例：评 VQA
#   GPU=1 CONFIG=experiments/public/eval/vqa.yaml OUT_NAME=eval_vqa TASK=vqa \
#     bash experiments/public/eval/eval_qwen3vl_cls.sh output/<run>

set -e

QWEN3_ENV=/home/zhoutuowen/anaconda3/envs/vlm2vec_qwen3
export PATH="$QWEN3_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export WANDB_DISABLED=true

cd /home/zhoutuowen/VLM2Vec

CKPT=${1:?"用法: bash eval_qwen3vl_cls.sh <CHECKPOINT_DIR>"}
BASE_MODEL=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen3-VL-2B-Instruct
DATA_BASEDIR=/home/zhoutuowen/data/MMEB-V2
GPU=${GPU:-0}
CONFIG=${CONFIG:-experiments/public/eval/cls.yaml}
OUT_NAME=${OUT_NAME:-eval_cls}
OUTPUT_PATH="$CKPT/$OUT_NAME"
mkdir -p "$OUTPUT_PATH"

echo "================================================="
echo "MMEB CLS Evaluation (Qwen3-VL LoRA, GPU=$GPU)"
echo "Base Model:  $BASE_MODEL"
echo "Checkpoint:  $CKPT"
echo "Config:      $CONFIG"
echo "Output:      $OUTPUT_PATH"
echo "================================================="

CUDA_VISIBLE_DEVICES=$GPU python \
    eval.py \
    --pooling eos \
    --normalize true \
    --per_device_eval_batch_size 16 \
    --model_backbone "qwen3_vl" \
    --model_name "$BASE_MODEL" \
    --checkpoint_path "$CKPT" \
    --lora "${LORA:-true}" \
    --dataset_config "$CONFIG" \
    --encode_output_path "$OUTPUT_PATH" \
    --data_basedir "$DATA_BASEDIR" \
    2>&1 | tee "$OUTPUT_PATH/eval.log"

echo ""
python experiments/public/eval/summarize_cls.py "$OUTPUT_PATH" "${TASK:-cls}" | tee "$OUTPUT_PATH/summary.txt"
