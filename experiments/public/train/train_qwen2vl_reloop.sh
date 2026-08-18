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

# train.py 里 wandb.init 的 mode="online" 是硬编码的，WANDB_MODE=offline 关不掉它，
# 只有 --report_to 能。默认 90 s 的 init 超时在这台机器上会直接失败（run 其实建成功了，
# 只是调用没在 90 s 内返回），所以放宽到 300 s。真正怕阻塞的长跑请直接 REPORT_TO=none：
# logging_steps=1 已经把每步 loss 写进 train.log，曲线事后可还原。
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

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

# 每张图最多几个视觉 token（乘 28*28 得到 resize_max_pixels：一个合并后的 token 对应
# 14x14 的 patch 再做 2x2 合并，正好 784 个像素）。默认 1280 是仓库原值，改了会让此前
# 所有结果不可比，所以只在需要时显式覆盖。
# 实测：OK-VQA/A-OKVQA 中位 366、Visual7W 255、ChartQA 178，都远在 640 以下，压到 640
# 对它们逐比特不变；只有 DocVQA/InfographicsVQA（原图 3~3.75 MP，被上限顶到 1248）会减半。
VISION_TOKENS=${VISION_TOKENS:-1280}
MAX_PIXELS=$((28 * 28 * VISION_TOKENS))

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
    REPORT_TO=${REPORT_TO:-wandb}
    # /home 上 output/ 已占 22G。DoRA checkpoint 每个约 235 MB，只留最后一个。
    # 长跑（RET 全量 7446 步 = 20+ 小时）必须设 SAVE_STEPS 做中途落盘：8/14 那次在
    # 1993 步被 NCCL 超时杀掉，5.8 小时无任何产物。save_only_model 下这些中间份
    # 不能续训（没有优化器状态），但至少可以拿去评测/抢救。
    SAVE_ARGS="--save_steps ${SAVE_STEPS:-$MAX_STEPS} --save_total_limit ${SAVE_LIMIT:-1}"
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

# FULL_FT=1 走论文的配方：全参微调 + 学习率 1e-5 + τ=0.05 + DeepSpeed ZeRO，不用 LoRA。
# 此时必须关掉 GradCache：它直接对原始张量调 loss.backward()，绕过 engine.backward()，
# 于是 ZeRO 的梯度切分与跨卡归约不会发生。那种失败不报错——各卡各自更新、loss 照样降，
# 结果却毫无意义。关掉后走 MMEBModel.forward，它自己做 all_gather，负样本池仍是整个全局批。
if [ "${FULL_FT:-0}" = "1" ]; then
    TUNE_ARGS="--temperature ${TEMPERATURE:-0.05}"
    LR=${LR:-1e-5}
    GRAD_CACHE=${GRAD_CACHE:-False}
    DS_CONFIG=${DS_CONFIG:-experiments/public/train/ds_zero${ZERO_STAGE:-3}.json}
    # save_only_model：全参 checkpoint 光优化器状态就约 26 GB，而我们只需要末态权重去评测。
    # 代价是不能续训。
    # 论文的优化器配方：AdamW β=(0.9,0.95)、weight decay 0.1、梯度裁剪 1.0、5% warmup、cosine。
    # β2=0.95 而非默认 0.999：二阶矩窗口更短，对大批次对比学习里梯度尺度的突变响应更快。
    # 这几项只在 FULL_FT 分支生效，LoRA 那批已完成实验的代码路径保持原样，否则新旧不可比。
    EXTRA_ARGS="--deepspeed $DS_CONFIG --save_only_model True --vision_lr ${VISION_LR:-2e-6}"
    EXTRA_ARGS="$EXTRA_ARGS --weight_decay ${WEIGHT_DECAY:-0.1} --adam_beta1 0.9 --adam_beta2 0.95"
    EXTRA_ARGS="$EXTRA_ARGS --max_grad_norm 1.0"
    WARMUP_STEPS=$((MAX_STEPS * 5 / 100))
else
    TUNE_ARGS="--lora --lora_r ${LORA_R:-16} --temperature ${TEMPERATURE:-0.02}"
    LR=${LR:-1e-4}
    GRAD_CACHE=${GRAD_CACHE:-True}
    EXTRA_ARGS=""
    WARMUP_STEPS=$((MAX_STEPS / 10))
fi
echo "训练方式: $([ "${FULL_FT:-0}" = "1" ] && echo 全参微调 || echo LoRA)   学习率: $LR"
echo "GradCache: $GRAD_CACHE   warmup: $WARMUP_STEPS/$MAX_STEPS 步   ${EXTRA_ARGS:+DeepSpeed: $DS_CONFIG}"

set +e
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=${PORT:-2455} \
    --max_restarts=0 \
    train.py \
    $TUNE_ARGS \
    --model_name $MODEL_NAME \
    --bf16 \
    --pooling eos \
    --normalize True \
    --resize_max_pixels $MAX_PIXELS \
    --reloop_t $RELOOP_T \
    --reloop_m $RELOOP_M \
    --reloop_readout $RELOOP_READOUT \
    $LOOP_ARGS \
    --seed $SEED \
    --dataloader_num_workers 4 \
    --ddp_timeout ${DDP_TIMEOUT:-10800} \
    --dataset_config "$DATA_CONFIG" \
    --dataset_size_alpha 0.5 \
    --run_name $EXP_NAME \
    --output_dir "$EXP_DIR" \
    --grad_cache $GRAD_CACHE \
    $EXTRA_ARGS \
    --per_device_train_batch_size $PER_DEVICE \
    --gc_q_chunk_size $GC_CHUNK \
    --gc_p_chunk_size $GC_CHUNK \
    --gradient_checkpointing $GRAD_CKPT \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --homogeneous_batch_size_per_device $HOMOGENEOUS \
    --lr_scheduler_type cosine \
    --learning_rate $LR \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
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
