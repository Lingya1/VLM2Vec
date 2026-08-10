#!/bin/bash
# 用 VLM2Vec 官方发布的 VLM2Vec-V2.0（Qwen2-VL-2B + LoRA）跑 MMEB 图像分类评测。
#
# 用途：验证"指令-标签先验绑定"这个机制在真正全量训练过的已发布模型上还剩多少。
# 我们此前观察到的 ObjectNet 34.3 -> 60.5 是在只训了 5 个 CLS 子集、VOC2007 被严重
# 过采样的模型上测的，不能代表全量训练的情形。
#
# 关键：默认关掉 vision 边界符的修复。VLM2Vec-V2 是在没有边界符的条件下训练的，
# 评测时补上属于训练-测试不一致，会把结论带偏。要做边界符本身的 A/B，请在训练侧做。
#
# 用法:
#   CONFIG=experiments/public/eval/_instr_probe_base.yaml OUT_NAME=probe_base \
#     bash experiments/public/eval/eval_vlm2vecv2_cls.sh

set -e

export PATH="/home/zhoutuowen/anaconda3/envs/vlm2vec_qwen3/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export WANDB_DISABLED=true
export VLM2VEC_NO_VISION_BOUNDARY=${VLM2VEC_NO_VISION_BOUNDARY:-1}

cd /home/zhoutuowen/VLM2Vec

BASE_MODEL=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct
CKPT=${CKPT:-/tmp/hfmodels/VLM2Vec-V2.0}
DATA_BASEDIR=/home/zhoutuowen/data/MMEB-V2
GPU=${GPU:-0}
CONFIG=${CONFIG:-experiments/public/eval/_instr_probe_base.yaml}
OUT_NAME=${OUT_NAME:-probe_base}
OUTPUT_PATH=/home/zhoutuowen/VLM2Vec/output/vlm2vecv2_probe/$OUT_NAME
mkdir -p "$OUTPUT_PATH"

echo "================================================="
echo "模型:     $CKPT (LoRA on $(basename $BASE_MODEL))"
echo "配置:     $CONFIG"
echo "边界符修复: $([ "$VLM2VEC_NO_VISION_BOUNDARY" = "1" ] && echo '关（与其训练条件一致）' || echo '开')"
echo "输出:     $OUTPUT_PATH"
echo "================================================="

CUDA_VISIBLE_DEVICES=$GPU python \
    eval.py \
    --pooling eos \
    --normalize true \
    --per_device_eval_batch_size 16 \
    --model_backbone "qwen2_vl" \
    --model_name "$BASE_MODEL" \
    --checkpoint_path "$CKPT" \
    --lora true \
    --dataset_config "$CONFIG" \
    --encode_output_path "$OUTPUT_PATH" \
    --data_basedir "$DATA_BASEDIR" \
    2>&1 | tee "$OUTPUT_PATH/eval.log"

echo ""
for f in "$OUTPUT_PATH"/*_score.json; do
    [ -e "$f" ] || continue
    printf "  %-14s %s\n" "$(basename "$f" _score.json)" "$(python -c "import json,sys; print(f\"{json.load(open('$f'))['hit@1']*100:.2f}\")")"
done | tee "$OUTPUT_PATH/summary.txt"
