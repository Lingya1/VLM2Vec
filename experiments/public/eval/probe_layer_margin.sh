#!/bin/bash
# 逐层判别性曲线（probe_layer_margin.py 的运行器）。参数约定与 eval_reloop.sh 一致。
# 用法: CONFIG=experiments/public/eval/grounding.yaml \
#       bash experiments/public/eval/probe_layer_margin.sh <ckpt_dir> [gpu]
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

CKPT=${1:?用法: bash probe_layer_margin.sh <ckpt_dir> [gpu]}
GPU=${2:-0}
BASE_MODEL=${BASE_MODEL:-/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct}
DATA_BASEDIR=${DATA_BASEDIR:-/home/zhoutuowen/data/MMEB-V2}
CONFIG=${CONFIG:-experiments/public/eval/grounding.yaml}

if [ -f "$CKPT/adapter_model.safetensors" ]; then
    TUNE_ARGS="--lora true"
    LOAD_FROM=$BASE_MODEL
else
    TUNE_ARGS=""
    LOAD_FROM=$CKPT
fi

# 与训练同值，理由同 probe_register_collapse.sh
VISION_TOKENS=${VISION_TOKENS:-640}
MAX_PIXELS=$((28 * 28 * VISION_TOKENS))

if [ "$GPU" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$GPU
fi

python experiments/public/eval/probe_layer_margin.py \
    --pooling eos \
    --normalize true \
    --resize_max_pixels $MAX_PIXELS \
    --per_device_eval_batch_size 16 \
    --model_backbone "qwen2_vl" \
    --model_name "$LOAD_FROM" \
    --checkpoint_path "$CKPT" \
    $TUNE_ARGS \
    --dataset_config "$CONFIG" \
    --encode_output_path "/tmp/layerprobe_$(basename "$CKPT")" \
    --data_basedir "$DATA_BASEDIR"
