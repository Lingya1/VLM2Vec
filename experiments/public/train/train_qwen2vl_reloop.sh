#!/bin/bash
# Qwen2-VL-2B + ReLoop 式循环深度与检索寄存器。A 与 D 两格共用这一个脚本，只改 T 和 M。
#
# RELOOP_T=1 RELOOP_M=0 时整套机制不接入，逐元素等价于判别式基线（断言见
# verify_reloop_identity.py），因此对照组与实验组不存在"两条代码路径实现不一致"的
# 解释空间。这一点是本次实验唯一观测量 D-A 可解释的前提。
#
# 用法：
#   # 先跑冒烟，量 s/step 与显存峰值，再定正式步数
#   SMOKE=1 RELOOP_T=1 RELOOP_M=0 bash experiments/public/train/train_qwen2vl_reloop.sh
#   SMOKE=1 RELOOP_T=4 RELOOP_M=5 bash experiments/public/train/train_qwen2vl_reloop.sh
#
#   # 正式两格
#   RELOOP_T=1 RELOOP_M=0 bash experiments/public/train/train_qwen2vl_reloop.sh
#   RELOOP_T=4 RELOOP_M=5 bash experiments/public/train/train_qwen2vl_reloop.sh

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

# 0-4 号卡上有别的用户的进程，只有 5/6/7 空着
GPUS=${GPUS:-5,6,7}
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

# 全局批 384 = 每卡 128 x 3 卡。同源块长 64，每卡每步 2 块，每个全局批 6 块。
# 注意这与 4 卡基线的 512 不同，负样本池变了，所以绝对分不能跨配置比；A/D 之间可比。
PER_DEVICE=${PER_DEVICE:-128}
HOMOGENEOUS=${HOMOGENEOUS:-64}
# GradCache 分块对 InfoNCE 是数学精确的，只是显存/速度旋钮，不改梯度。因此 T=4 那格若
# 顶不住可以单独降到 8，不破坏与 T=1 的可比性；必须一致的是全局批、步数、数据顺序与种子。
GC_CHUNK=${GC_CHUNK:-16}
GRAD_CKPT=${GRAD_CKPT:-True}

# ReLoop 的两个轴
RELOOP_T=${RELOOP_T:-1}
RELOOP_M=${RELOOP_M:-0}
RELOOP_READOUT=${RELOOP_READOUT:-last}
# 循环区间留空则用默认 [num_layers-11, num_layers-1)，28 层模型即 [17, 27)
LOOP_ARGS=""
[ -n "${RELOOP_LOOP_START:-}" ] && LOOP_ARGS="$LOOP_ARGS --reloop_loop_start $RELOOP_LOOP_START"
[ -n "${RELOOP_LOOP_END:-}" ] && LOOP_ARGS="$LOOP_ARGS --reloop_loop_end $RELOOP_LOOP_END"

# 种子必须在两格之间一致，否则 D-A 里混进了初始化与数据顺序的差异
SEED=${SEED:-42}

DATA_CONFIG=${DATA_CONFIG:-experiments/public/train/train_reloop4.yaml}

if [ "${SMOKE:-0}" = "1" ]; then
    EXP_NAME=${EXP_NAME:-Qwen2vl_2B.reloop.smoke.T$RELOOP_T.M$RELOOP_M}
    MAX_STEPS=${MAX_STEPS:-12}
    REPORT_TO=none
    # 冒烟也走一次存盘。不这么做的话冒烟覆盖不到 _save 那条路径，而 register 权重与循环
    # 拓扑恰好是在那里落盘的：漏了会在正式训练的最后一步才炸，等于白跑十几个小时。
    SAVE_ARGS="--save_steps $MAX_STEPS --save_total_limit 1"
else
    EXP_NAME=${EXP_NAME:-Qwen2vl_2B.reloop4.T$RELOOP_T.M$RELOOP_M.s$SEED.BS384.3A40}
    # 84009 条 / 全局批 384 -> 一个 epoch 约 219 步
    MAX_STEPS=${MAX_STEPS:-219}
    REPORT_TO=wandb
    # /home 上 output/ 已占 22G。DoRA checkpoint 每个约 235 MB，只留最后一个。
    SAVE_ARGS="--save_steps $MAX_STEPS --save_total_limit 1"
fi

EXP_DIR=/home/zhoutuowen/VLM2Vec/output/$EXP_NAME
mkdir -p "$EXP_DIR"

# 冒烟测试遗留的 WANDB_DISABLED 会和 --report_to wandb 直接冲突并报错
unset WANDB_DISABLED WANDB_MODE

cd /home/zhoutuowen/VLM2Vec

echo "=== $EXP_NAME ==="
echo "卡: $GPUS ($NPROC 张)   全局批: $((PER_DEVICE * NPROC))   同源块长(每卡): $HOMOGENEOUS"
echo "ReLoop: T=$RELOOP_T  M=$RELOOP_M  readout=$RELOOP_READOUT ${LOOP_ARGS:+(区间$LOOP_ARGS)}"
echo "步数: $MAX_STEPS   种子: $SEED   GC 分块: $GC_CHUNK"
echo "数据: $DATA_CONFIG   输出: $EXP_DIR"

set +e
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=${PORT:-2455} \
    --max_restarts=0 \
    train.py \
    --lora \
    --lora_r 16 \
    --model_name $MODEL_NAME \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --reloop_t $RELOOP_T \
    --reloop_m $RELOOP_M \
    --reloop_readout $RELOOP_READOUT \
    $LOOP_ARGS \
    --seed $SEED \
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
