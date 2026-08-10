#!/bin/bash

# === MMEB 图像评估脚本 (适配当前环境) ===

# 防止加载 ~/.local 下的旧包干扰
export PYTHONNOUSERSITE=1

# CUDA 路径
export PATH=/usr/local/cuda-12.2/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.2

# HuggingFace 缓存路径（评估数据集需要从 HF 下载标注）
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets

# WandB 关闭
export WANDB_DISABLED=true

cd /home/zhoutuowen/VLM2Vec

# ==============================================================================
# 配置
# ==============================================================================

# 基础模型路径
BASE_MODEL=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct

# LoRA checkpoint 路径（训练好的模型）
CHECKPOINT_PATH=/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.imageonly.lora16.BS256.4A40

# 评估数据根目录
DATA_BASEDIR=/home/zhoutuowen/data/MMEB-V2

# 评估结果输出目录
OUTPUT_PATH=/home/zhoutuowen/VLM2Vec/output/Qwen2vl_2B.imageonly.lora16.BS256.4A40/eval_image
mkdir -p $OUTPUT_PATH

# GPU 配置（单卡评估，可通过环境变量覆盖，如: CUDA_VISIBLE_DEVICES=5 bash eval_image_4a40.sh）
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ==============================================================================
# 运行评估
# ==============================================================================

echo "================================================="
echo "MMEB Image Evaluation (Single GPU)"
echo "Base Model: $BASE_MODEL"
echo "LoRA Checkpoint: $CHECKPOINT_PATH"
echo "Data Dir: $DATA_BASEDIR"
echo "Output: $OUTPUT_PATH"
echo "================================================="

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python \
    eval.py \
    --pooling eos \
    --normalize true \
    --per_device_eval_batch_size 16 \
    --model_backbone "qwen2_vl" \
    --model_name "$BASE_MODEL" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --lora true \
    --dataset_config "experiments/public/eval/image.yaml" \
    --encode_output_path "$OUTPUT_PATH" \
    --data_basedir "$DATA_BASEDIR" \
    2>&1 | tee $OUTPUT_PATH/eval.log

echo ""
echo "================================================="
echo "Evaluation finished! Results saved to: $OUTPUT_PATH"
echo "================================================="

# 打印所有 score 文件
echo ""
echo "Score files:"
for f in $OUTPUT_PATH/*_score.json; do
    if [ -f "$f" ]; then
        echo "--- $(basename $f) ---"
        cat "$f"
        echo ""
    fi
done
