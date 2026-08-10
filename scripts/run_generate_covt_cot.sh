#!/bin/bash
# Offline generation of CoVT visual reasoning chains on MMEB-train
#
# Single GPU:   bash scripts/run_generate_covt_cot.sh
# Multi-GPU:    bash scripts/run_generate_covt_cot.sh multi
# Specific subsets: bash scripts/run_generate_covt_cot.sh --subsets OK-VQA A-OKVQA

set -e

MODEL_PATH="weights_model/weights_models/CoVT"
DATA_DIR="data/MMEB-train"
OUTPUT_DIR="data/MMEB-train-covt-cot"
MAX_NEW_TOKENS=128

cd /home/zhoutuowen

source /home/zhoutuowen/anaconda3/etc/profile.d/conda.sh
conda activate vlm2vec
export PYTHONNOUSERSITE=1

mkdir -p logs

if [ "$1" = "multi" ]; then
    # Multi-GPU: launch on free GPUs (3 and 5 currently available)
    GPUS=(3 5)
    NUM_SHARDS=${#GPUS[@]}
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        GPU_ID=${GPUS[$SHARD_ID]}
        echo "Launching shard ${SHARD_ID}/${NUM_SHARDS} on physical GPU ${GPU_ID}..."
        CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/generate_covt_cot.py \
            --model_path ${MODEL_PATH} \
            --data_dir ${DATA_DIR} \
            --output_dir ${OUTPUT_DIR} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --gpu_id 0 \
            --shard_id ${SHARD_ID} \
            --num_shards ${NUM_SHARDS} \
            > logs/covt_cot_shard${SHARD_ID}.log 2>&1 &
        echo "  PID: $!"
    done
    echo "All shards launched. Monitor with: tail -f logs/covt_cot_shard*.log"
    wait
else
    # Single GPU on physical GPU 3 (free)
    GPU_ID=${GPU_ID:-3}
    echo "Single GPU generation on physical GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/generate_covt_cot.py \
        --model_path ${MODEL_PATH} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --gpu_id 0 \
        "$@"
fi
