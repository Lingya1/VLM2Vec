#!/bin/bash

# === 4 x A40 (48GB each) 训练脚本 ===

# 模型路径（本地路径或 HuggingFace 模型名）
# 如果已手动下载到本地，改成本地路径：
#   MODEL_NAME=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct
# 如果让训练时自动从 HuggingFace 下载：
MODEL_NAME=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct

# 防止加载 ~/.local 下的旧包（librosa 等）干扰
export PYTHONNOUSERSITE=1

# HuggingFace 缓存路径 + 离线模式
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# WandB 设置（如不需要可改为 none）
export WANDB_DISABLED=true

# 实验名称和输出路径
EXP_NAME=Qwen2vl_2B.imageonly.lora16.BS256.4A40
EXP_DIR=/home/zhoutuowen/VLM2Vec/output/$EXP_NAME
mkdir -p $EXP_DIR

cd /home/zhoutuowen/VLM2Vec

# 训练命令
# 相比原始 8xH100 的调整：
#   - nproc_per_node: 8 -> 4
#   - CUDA_VISIBLE_DEVICES: 4张卡
#   - per_device_train_batch_size: 128 -> 64（A40显存48GB，约为H100的60%）
#   - gc_q/p_chunk_size: 8 -> 4（减小GradCache分块大小，降低峰值显存）
#   - interleave_batch_size: 64 -> 32
#   - 总有效batch size: 64 x 4 = 256（原始为128 x 8 = 1024）
#   - 可酌情增大 max_steps 来补偿更小的 batch size

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 torchrun \
    --nproc_per_node=6 \
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
    --dataset_config experiments/public/train/train_image_4a40.yaml \
    --run_name $EXP_NAME \
    --output_dir $EXP_DIR \
    --grad_cache True \
    --per_device_train_batch_size 64 \
    --gc_q_chunk_size 4 \
    --gc_p_chunk_size 4 \
    --interleave_batch_size 32 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --max_steps 2500 \
    --warmup_steps 100 \
    --save_steps 200 \
    --logging_steps 1 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from auto \
    --report_to none \
    2>&1 | tee $EXP_DIR/train.log
