#!/bin/bash
# 评测 ReLoop 训练出的 checkpoint。
#
# 不需要在命令行上重复 --reloop_t / --reloop_m：checkpoint 目录里的 reloop.pt 同时存了
# register 权重与循环拓扑，MMEBModel.load 会以它为准。这样就不存在"评测深度与训练深度
# 对不齐"这种不报错但分数不可解释的情形。
#
# 边界符：训练侧是开着的（processor.py 的默认行为），所以评测也必须开着，否则是
# 训练-测试不一致。这里显式设成 0 以防环境里残留了 1。
#
# 用法：
#   CKPT=output/Qwen2vl_2B.reloop.okvqa.T1.M0.s42 bash experiments/public/eval/eval_reloop.sh
#   CKPT=... CONFIG=experiments/public/eval/vqa.yaml bash experiments/public/eval/eval_reloop.sh

set -e

export PATH="/home/zhoutuowen/anaconda3/envs/vlm2vec_qwen3/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export WANDB_DISABLED=true
# 这台机器连不上 huggingface.co，不设离线标志的话每个数据集文件都要熬完五轮重试退避
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# 训练侧开着边界符修复，评测必须一致
export VLM2VEC_NO_VISION_BOUNDARY=0

cd /home/zhoutuowen/VLM2Vec

BASE_MODEL=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct
DATA_BASEDIR=/home/zhoutuowen/data/MMEB-V2
CONFIG=${CONFIG:-experiments/public/eval/_okvqa.yaml}
GPU=${GPU:-5}

# 必须与训练用的同一个值：训练看 640 token 的文档图、评测看 1248 token 的，
# 输入分布就对不上了，而这种不一致不会报错，只会让分数无从解释。
VISION_TOKENS=${VISION_TOKENS:-1280}
MAX_PIXELS=$((28 * 28 * VISION_TOKENS))

if [ -z "${CKPT:-}" ]; then
    echo "必须给 CKPT，例如 CKPT=output/Qwen2vl_2B.reloop.okvqa.T1.M0.s42"
    exit 1
fi
[ -d "$CKPT" ] || { echo "找不到 checkpoint 目录: $CKPT"; exit 1; }

# 按 checkpoint 里有没有 adapter 自动判断是 LoRA 还是全参，而不是靠调用方传对开关：
# 传错时不会报错，而是静默加载错的权重集合，分数无从解释。全参时 model_name 必须指向
# checkpoint 本身，否则读回的是原始基座。
if [ -f "$CKPT/adapter_model.safetensors" ]; then
    TUNE_ARGS="--lora true"
    LOAD_FROM=$BASE_MODEL
    echo "权重类型:   LoRA adapter"
elif [ -f "$CKPT/model.safetensors" ] || ls "$CKPT"/model-*.safetensors >/dev/null 2>&1; then
    TUNE_ARGS=""
    LOAD_FROM=$CKPT
    echo "权重类型:   全参微调"
else
    echo "在 $CKPT 里既找不到 adapter_model.safetensors 也找不到 model.safetensors"
    exit 1
fi

OUTPUT_PATH=${OUTPUT_PATH:-$CKPT/eval_$(basename "$CONFIG" .yaml)}
mkdir -p "$OUTPUT_PATH"

echo "================================================="
echo "checkpoint: $CKPT"
if [ -f "$CKPT/reloop.pt" ]; then
    python -c "
import torch
s = torch.load('$CKPT/reloop.pt', map_location='cpu', weights_only=False)
print(f\"ReLoop 拓扑(取自 checkpoint): T={s['reloop_t']} M={s['reloop_m']} \"
      f\"loop=[{s['reloop_loop_start']},{s['reloop_loop_end']}) readout={s['reloop_readout']}\")"
else
    echo "ReLoop 拓扑: 无 reloop.pt，按判别式基线评测（T=1, M=0）"
fi
echo "配置:       $CONFIG"
echo "输出:       $OUTPUT_PATH"
echo "================================================="

CUDA_VISIBLE_DEVICES=$GPU python \
    eval.py \
    --pooling eos \
    --normalize true \
    --resize_max_pixels $MAX_PIXELS \
    --per_device_eval_batch_size 16 \
    --model_backbone "qwen2_vl" \
    --model_name "$LOAD_FROM" \
    --checkpoint_path "$CKPT" \
    $TUNE_ARGS \
    --dataset_config "$CONFIG" \
    --encode_output_path "$OUTPUT_PATH" \
    --data_basedir "$DATA_BASEDIR" \
    2>&1 | tee "$OUTPUT_PATH/eval.log"

echo ""
for f in "$OUTPUT_PATH"/*_score.json; do
    [ -e "$f" ] || continue
    printf "  %-18s %s\n" "$(basename "$f" _score.json)" \
        "$(python -c "import json; print(f\"{json.load(open('$f'))['hit@1']*100:.2f}\")")"
done | tee "$OUTPUT_PATH/summary.txt"
