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
# 训练侧开着边界符修复，评测必须一致
export VLM2VEC_NO_VISION_BOUNDARY=0

cd /home/zhoutuowen/VLM2Vec

BASE_MODEL=/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct
DATA_BASEDIR=/home/zhoutuowen/data/MMEB-V2
CONFIG=${CONFIG:-experiments/public/eval/_okvqa.yaml}
GPU=${GPU:-5}

if [ -z "${CKPT:-}" ]; then
    echo "必须给 CKPT，例如 CKPT=output/Qwen2vl_2B.reloop.okvqa.T1.M0.s42"
    exit 1
fi
[ -d "$CKPT" ] || { echo "找不到 checkpoint 目录: $CKPT"; exit 1; }

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
    printf "  %-18s %s\n" "$(basename "$f" _score.json)" \
        "$(python -c "import json; print(f\"{json.load(open('$f'))['hit@1']*100:.2f}\")")"
done | tee "$OUTPUT_PATH/summary.txt"
