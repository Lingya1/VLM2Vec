#!/bin/bash
# 测 register 之间是否坍缩。参数与 eval_reloop.sh 保持一致，只是换成跑探针。
# 用法: bash experiments/public/eval/probe_register_collapse.sh <ckpt_dir> [gpu]
set -e
cd /home/zhoutuowen/VLM2Vec

export PATH="/home/zhoutuowen/anaconda3/envs/vlm2vec_qwen3/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true

CKPT=${1:?用法: bash probe_register_collapse.sh <ckpt_dir> [gpu]}
GPU=${2:-5}
BASE_MODEL=${BASE_MODEL:-/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct}
DATA_BASEDIR=${DATA_BASEDIR:-/home/zhoutuowen/data/MMEB-V2}
CONFIG=${CONFIG:-experiments/public/eval/_okvqa.yaml}

# 与 eval_reloop.sh 同样按目录内容判断权重类型，而不是靠调用方传对开关：
# 传错不会报错，只会静默加载错的权重集合，测出来的隐状态无从解释。
if [ -f "$CKPT/adapter_model.safetensors" ]; then
    TUNE_ARGS="--lora true"
    LOAD_FROM=$BASE_MODEL
else
    TUNE_ARGS=""
    LOAD_FROM=$CKPT
fi

# 必须与训练同值。探针看 1248 token 的文档图、训练看 640 的，测到的隐状态属于
# 另一个输入分布，坍缩与否的读数不可用。
VISION_TOKENS=${VISION_TOKENS:-640}
MAX_PIXELS=$((28 * 28 * VISION_TOKENS))

# GPU=cpu 时走 CPU，用于卡被占满但又想立刻拿到读数的情况（8 个样本量级可接受）
if [ "$GPU" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$GPU
fi

python experiments/public/eval/probe_register_collapse.py \
    --pooling eos \
    --normalize true \
    --resize_max_pixels $MAX_PIXELS \
    --per_device_eval_batch_size 16 \
    --model_backbone "qwen2_vl" \
    --model_name "$LOAD_FROM" \
    --checkpoint_path "$CKPT" \
    $TUNE_ARGS \
    --dataset_config "$CONFIG" \
    --encode_output_path "/tmp/probe_$(basename "$CKPT")" \
    --data_basedir "$DATA_BASEDIR"
