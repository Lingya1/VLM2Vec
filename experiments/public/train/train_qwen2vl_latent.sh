#!/bin/bash
# Qwen2-VL-2B + 隐式推理瓶颈。K x beta 网格的每一格都跑这个脚本，只改环境变量。
#
# LATENT_K=0 时整套机制不接入，等价于判别式基线，因此对照组与实验组共用一份代码，
# 不存在"两条代码路径实现不一致"的解释空间。
# LATENT_BETA=0 时接入 reason token 但不加率项，这一列就是 LaME 式的无率正则瓶颈。
#
# 用法：
#   # 实现验证（HatefulMemes 单子集，约 18 分钟）
#   SMOKE=1 LATENT_K=8 LATENT_BETA=1e-4 bash experiments/public/train/train_qwen2vl_latent.sh
#
#   # 网格里的一格
#   DATA_CONFIG=experiments/public/train/train_vqa6_10k.yaml \
#   LATENT_K=8 LATENT_BETA=0 bash experiments/public/train/train_qwen2vl_latent.sh

set -e

MODEL_NAME=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct

CONDA_ENV=/home/zhoutuowen/anaconda3/envs/vlm2vec_qwen3
export PATH="$CONDA_ENV/bin:$PATH"
# ~/.local 下有一份与当前 numpy 二进制不兼容的 sklearn，会在 import transformers 时炸
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

GPUS=${GPUS:-6,7}
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

PER_DEVICE=${PER_DEVICE:-64}
HOMOGENEOUS=${HOMOGENEOUS:-64}
# 压测结论（真实最长序列 1314 token）：chunk 16 峰值 15.1 GB / 185.5 s，chunk 32 峰值
# 24.7 GB / 186.3 s —— 多用 64% 显存换不到任何速度。所以固定用 16，不必再扫。
GC_CHUNK=${GC_CHUNK:-16}
GRAD_CKPT=${GRAD_CKPT:-True}

# 瓶颈超参
LATENT_K=${LATENT_K:-8}
LATENT_BETA=${LATENT_BETA:-0}
LATENT_FREE_BITS=${LATENT_FREE_BITS:-0.02}
LATENT_SIZE_ARG=""
[ -n "${LATENT_SIZE:-}" ] && LATENT_SIZE_ARG="--latent_size $LATENT_SIZE"

if [ "${SMOKE:-0}" = "1" ]; then
    DATA_CONFIG=${DATA_CONFIG:-experiments/public/train/_hatefulmemes_smoke.yaml}
    # 8500 条 / 全局批 128 约 66 步
    MAX_STEPS=${MAX_STEPS:-66}
    EXP_NAME=${EXP_NAME:-Qwen2vl_2B.latent.smoke.K$LATENT_K.b$LATENT_BETA}
    REPORT_TO=none
    SAVE_ARGS="--save_strategy no"
else
    DATA_CONFIG=${DATA_CONFIG:-experiments/public/train/train_vqa6_10k.yaml}
    MAX_STEPS=${MAX_STEPS:-460}
    EXP_NAME=${EXP_NAME:-Qwen2vl_2B.latent.K$LATENT_K.b$LATENT_BETA}
    REPORT_TO=wandb
    # /home 长期贴在 100%，此前已因写满导致源码被删。LoRA checkpoint 每个约 235 MB。
    SAVE_ARGS="--save_steps $MAX_STEPS --save_total_limit 1"
fi

# beta 的退火：前 15% 步保持 0 让对比目标先把表示拉开，再用 35% 的步数线性升到目标值。
# 直接上固定 beta 会在表示还没学起来时把后验压塌到先验（KL->0，z 与输入无关），
# 这是 VIB 的经典失败模式；未训练时实测率就有 3 万 nats 量级，压力相当大。
BETA_DELAY=${BETA_DELAY:-$((MAX_STEPS * 15 / 100))}
BETA_WARMUP=${BETA_WARMUP:-$((MAX_STEPS * 35 / 100))}

EXP_DIR=/home/zhoutuowen/VLM2Vec/output/$EXP_NAME
mkdir -p "$EXP_DIR"

unset WANDB_DISABLED WANDB_MODE
cd /home/zhoutuowen/VLM2Vec

echo "=== $EXP_NAME ==="
echo "卡: $GPUS ($NPROC 张)   全局批: $((PER_DEVICE * NPROC))   步数: $MAX_STEPS"
echo "瓶颈: K=$LATENT_K  beta=$LATENT_BETA  free_bits=$LATENT_FREE_BITS  delay=$BETA_DELAY  warmup=$BETA_WARMUP"
echo "数据: $DATA_CONFIG   输出: $EXP_DIR"

set +e
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=${PORT:-2433} \
    --max_restarts=0 \
    train.py \
    --lora \
    --lora_r 16 \
    --model_name $MODEL_NAME \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --latent_k $LATENT_K \
    --latent_beta $LATENT_BETA \
    --latent_free_bits $LATENT_FREE_BITS \
    --latent_beta_delay $BETA_DELAY \
    --latent_beta_warmup $BETA_WARMUP \
    $LATENT_SIZE_ARG \
    --dataloader_num_workers 4 \
    --dataset_config "$DATA_CONFIG" \
    --dataset_size_alpha 0.5 \
    --run_name $EXP_NAME \
    --output_dir "$EXP_DIR" \
    --grad_cache True \
    --per_device_train_batch_size $PER_DEVICE \
    --gc_q_chunk_size $GC_CHUNK \
    --gc_p_chunk_size $GC_CHUNK \
    --gradient_checkpointing $GRAD_CKPT \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --homogeneous_batch_size_per_device $HOMOGENEOUS \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --max_steps $MAX_STEPS \
    --warmup_steps $((MAX_STEPS / 10)) \
    --logging_steps 1 \
    $SAVE_ARGS \
    --save_safetensors True \
    --remove_unused_columns False \
    --report_to $REPORT_TO \
    2>&1 | tee "$EXP_DIR/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "训练失败，退出码 $TRAIN_STATUS。日志: $EXP_DIR/train.log"
    exit "$TRAIN_STATUS"
fi
echo "训练完成: $EXP_DIR"
