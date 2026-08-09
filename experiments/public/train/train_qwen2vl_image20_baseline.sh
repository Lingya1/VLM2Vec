#!/bin/bash
# Qwen2-VL-2B 20 子集图像判别式基线（30k 截断版）。
#
# 用途：这是 latent 率项方法的对照组。后续 K x beta 网格的每一格都跑这套配方，
# 只改瓶颈相关的开关，这样 delta 才可解释。
#
# 规模选择：全量配方（train_image_4a40.yaml，1,068,173 条 / 2086 步）8 卡实测 87 s/step，
# 约 52 小时。这是共享主机，那个占用不合适，所以改成每子集截断 30k（473,428 条 / 925 步）
# 并只用 4 卡，留一半卡给别人。总算力从 416 GPU 小时降到约 185。
# 代价：绝对分不再可与 LaME 报告的 68.5 直接对比；可上报的绝对分在全评测机器上另跑。
#
# 全局批仍为 512（每卡 128 x 4 卡），与 LaME 一致，因此对比学习的负样本池没有缩水。
# 同源块长仍 64，每个全局批 8 个块，与 8 卡版逐块等价。
#
# 依赖三处已验证的修复：分片随机流去相关、块长按卡计、num_shards 乘 world_size
# （见 experiments/public/train/verify_batch_source.py）
#
# 用法：
#   bash experiments/public/train/train_qwen2vl_image20_baseline.sh          # 正式训练
#   SMOKE=1 bash experiments/public/train/train_qwen2vl_image20_baseline.sh  # 12 步冒烟

set -e

MODEL_NAME=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct

# vlm2vec_qwen3 的 transformers 4.57.6 同时支持 vendored Qwen2-VL 与原生 Qwen3-VL，
# 两个 backbone 共用一个环境，避免后续对照实验跨环境带来的版本差异。
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

# 5 号卡上有别的用户的进程，避开
GPUS=${GPUS:-0,1,2,3}
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

# 全局批 512 = 每卡 128 x 4 卡。同源块长 64，每卡每步 2 块，每个全局批 8 块。
# 峰值显存由 GradCache 的分块大小决定而非 per_device，所以从 64 提到 128 只多占
# 常驻的输入张量（128 张图的 pixel_values 约 385 MB），激活峰值不变。
PER_DEVICE=${PER_DEVICE:-128}
HOMOGENEOUS=${HOMOGENEOUS:-64}
# GradCache 分块：块越小峰值显存越低但 GPU 利用率越差。
# 实测 per_device=64 且不开梯度检查点时，chunk 8/16/32 全部 OOM —— 因为 DocVQA 一类
# 子集的查询长到 1302 token，一个设备批就是 8.3 万 token，而同源块又让整批都是它。
# 开梯度检查点后用激活重算换显存，才能把分块放大到有意义的规模。
GC_CHUNK=${GC_CHUNK:-16}
GRAD_CKPT=${GRAD_CKPT:-True}

DATA_CONFIG=${DATA_CONFIG:-experiments/public/train/train_image20_30k.yaml}

if [ "${SMOKE:-0}" = "1" ]; then
    EXP_NAME=${EXP_NAME:-Qwen2vl_2B.image20.smoke}
    MAX_STEPS=${MAX_STEPS:-12}
    REPORT_TO=none
    SAVE_ARGS="--save_strategy no"
else
    EXP_NAME=${EXP_NAME:-Qwen2vl_2B.image20.30k.a0.5.BS512.4A40}
    # 20 个子集按 30k 截断后共 473,428 条，全局批 512 -> 一个 epoch 约 925 步
    MAX_STEPS=${MAX_STEPS:-925}
    REPORT_TO=wandb
    # /home 长期贴在 100%（现余 21 GB），且此前已因写满导致源码被删。
    # LoRA checkpoint 每个约 235 MB，只留 2 个。
    SAVE_ARGS="--save_steps 300 --save_total_limit 2"
fi

EXP_DIR=/home/zhoutuowen/VLM2Vec/output/$EXP_NAME
mkdir -p "$EXP_DIR"

# 冒烟测试遗留的 WANDB_DISABLED 会和 --report_to wandb 直接冲突并报错
unset WANDB_DISABLED WANDB_MODE

cd /home/zhoutuowen/VLM2Vec

echo "=== $EXP_NAME ==="
echo "卡: $GPUS ($NPROC 张)   全局批: $((PER_DEVICE * NPROC))   同源块长(每卡): $HOMOGENEOUS"
echo "步数: $MAX_STEPS   输出: $EXP_DIR"

set +e
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=${PORT:-2411} \
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
    --warmup_steps 100 \
    --logging_steps 1 \
    $SAVE_ARGS \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from auto \
    --report_to $REPORT_TO \
    2>&1 | tee "$EXP_DIR/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [ "$TRAIN_STATUS" -ne 0 ]; then
    echo "训练失败，退出码 $TRAIN_STATUS。日志: $EXP_DIR/train.log"
    exit "$TRAIN_STATUS"
fi
echo "训练完成: $EXP_DIR"
