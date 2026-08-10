#!/bin/bash

# === 4 x A40 快速训练脚本 ===
# 优化：降低分辨率 + 精简数据集 + 增大chunk + 减少步数
# 预计时间：~6-10 小时（原始 ~40-70 小时）

MODEL_NAME=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct

# 防止加载 ~/.local 下的旧包
export PYTHONNOUSERSITE=1

# HuggingFace 离线模式
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# WandB 关闭
export WANDB_DISABLED=true

# 实验名称和输出路径
EXP_NAME=Qwen2vl_2B.imageonly.fast.lora16.4A40
EXP_DIR=/home/zhoutuowen/VLM2Vec/output/$EXP_NAME
mkdir -p $EXP_DIR

cd /home/zhoutuowen/VLM2Vec

# 关键优化点：
#   1. resize_max_pixels: 降到 200704 (28*28*256)，token 数大幅减少
#   2. gc_chunk_size: 4 -> 8，前向传播次数减半
#   3. 精简数据集：8个核心子集
#   4. max_steps: 5000 -> 2000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=2207 \
    --max_restarts=0 \
    train.py \
    --lora \
    --lora_r 16 \
    --model_name $MODEL_NAME \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataloader_num_workers 4 \
    --dataset_config experiments/public/train/train_image_4a40_fast.yaml \
    --run_name $EXP_NAME \
    --output_dir $EXP_DIR \
    --grad_cache True \
    --per_device_train_batch_size 64 \
    --gc_q_chunk_size 8 \
    --gc_p_chunk_size 8 \
    --interleave_batch_size 32 \
    --resize_max_pixels 200704 \
    --lr_scheduler_type linear \
    --learning_rate 5e-5 \
    --max_steps 2000 \
    --warmup_steps 50 \
    --save_steps 500 \
    --logging_steps 1 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from auto \
    --report_to none \
    2>&1 | tee $EXP_DIR/train.log
