#!/bin/bash
# ReLoop 在 OK-VQA 单子集上的快速筛查：三格串行训练 + 评测。
#
# 三格的设计意图：
#   A42 = T=1, M=0, seed 42   基线
#   A43 = T=1, M=0, seed 43   基线的种子重复 —— |A42-A43| 就是跑间噪声
#   D42 = T=4, M=5, seed 42   完整机制
#
# 只有拿到 |A42-A43|，D42-A42 才有参照物。单子集下这个方差基线才付得起（约 75 min/格）；
# 换到 VQA6 就是 6~8 小时一格，那时再补方差就太贵了。所以顺序上必须先在这里补。
#
# 除 T/M/seed 外三格完全一致：同数据、同步数、同全局批、同 GC 分块、同学习率调度。
#
# 预计：A 约 75 min，D 约 107 min，三格约 4.3 h，加评测每格 10~15 min。
#
# 用法：
#   bash experiments/public/train/run_reloop_okvqa.sh              # 全跑
#   CELLS="A42 D42" bash experiments/public/train/run_reloop_okvqa.sh   # 只跑指定格
#   SKIP_EVAL=1 bash experiments/public/train/run_reloop_okvqa.sh       # 只训练不评测

set -e
cd /home/zhoutuowen/VLM2Vec

GPUS=${GPUS:-5,6,7}
PER_DEVICE=${PER_DEVICE:-64}
# 9009 条 / 全局批 192 = 每 epoch 47 步，取 2 个 epoch
MAX_STEPS=${MAX_STEPS:-94}
DATA_CONFIG=${DATA_CONFIG:-experiments/public/train/train_okvqa.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-experiments/public/eval/_okvqa.yaml}
CELLS=${CELLS:-"A42 A43 D42"}

# 单子集下同源块无意义（只有一个源），设成等于 per_device 即每步整批同源
HOMOGENEOUS=${HOMOGENEOUS:-64}

run_cell () {
    local name=$1 t=$2 m=$3 seed=$4
    local exp="Qwen2vl_2B.reloop.okvqa.T$t.M$m.s$seed"
    local dir="output/$exp"

    if [ -f "$dir/adapter_model.safetensors" ]; then
        echo ">>> $name 已存在，跳过训练: $dir"
    else
        echo ">>> $name 开始训练 (T=$t M=$m seed=$seed)"
        EXP_NAME=$exp GPUS=$GPUS PER_DEVICE=$PER_DEVICE HOMOGENEOUS=$HOMOGENEOUS \
        MAX_STEPS=$MAX_STEPS SEED=$seed RELOOP_T=$t RELOOP_M=$m \
        DATA_CONFIG=$DATA_CONFIG PORT=$((2460 + seed % 100)) \
            bash experiments/public/train/train_qwen2vl_reloop.sh
    fi

    if [ "${SKIP_EVAL:-0}" != "1" ]; then
        echo ">>> $name 开始评测"
        CKPT=$dir CONFIG=$EVAL_CONFIG GPU=${GPUS%%,*} \
            bash experiments/public/eval/eval_reloop.sh
    fi
}

for cell in $CELLS; do
    case $cell in
        A42) run_cell A42 1 0 42 ;;
        A43) run_cell A43 1 0 43 ;;
        D42) run_cell D42 4 5 42 ;;
        *) echo "未知的格: $cell（可选 A42 A43 D42）"; exit 1 ;;
    esac
done

echo
echo "================= 汇总 ================="
printf "%-34s %s\n" "实验" "OK-VQA hit@1"
for cell in $CELLS; do
    case $cell in
        A42) exp=Qwen2vl_2B.reloop.okvqa.T1.M0.s42 ;;
        A43) exp=Qwen2vl_2B.reloop.okvqa.T1.M0.s43 ;;
        D42) exp=Qwen2vl_2B.reloop.okvqa.T4.M5.s42 ;;
    esac
    f="output/$exp/eval__okvqa/OK-VQA_score.json"
    if [ -f "$f" ]; then
        printf "%-34s %s\n" "$cell ($exp)" \
            "$(python -c "import json; print(f\"{json.load(open('$f'))['hit@1']*100:.2f}\")")"
    else
        printf "%-34s %s\n" "$cell ($exp)" "(无结果)"
    fi
done
echo
echo "读法：先看 |A42-A43| 有多大，那是噪声下限；D42-A42 要明显超过它才算信号。"
