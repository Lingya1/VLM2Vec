#!/bin/bash
# OK-VQA 上全参微调的 T x M 网格，无人值守跑完 7 格并自动汇总。
#
# 网格：T in {1,4} x M in {0,1,5}，外加 T1M0 的种子重复作噪声基线。
# 两路并行：T=1 的四格在 GPU 5,6,7；T=4 的三格在 GPU 0,1,2。两路步时不同但总时长接近。
#
# 每路内部按信息量排序，先跑完的就是最关键的格：半夜出事时保住的是 2x2 核心。
#
# 论文配方（arXiv 2607.28751 第 443-451 行）：全参微调、LM/merger 1e-5、视觉编码器 2e-6、
# τ=0.05、AdamW β=(0.9,0.95)、weight decay 0.1、梯度裁剪 1.0、5% warmup、cosine、
# BF16 + 梯度检查点 + ZeRO-3。与论文仍有的差距：步数 94 vs 5000，数据 1 个子集 vs 24 个，
# 全局批 192 vs 256（3 卡整除的上限，每卡 96 会 OOM）。
#
# 用法：
#   nohup bash experiments/public/train/run_fullft_grid.sh > /tmp/fullft_grid.log 2>&1 &
#   汇总在 output/FULLFT_GRID_SUMMARY.txt

cd /home/zhoutuowen/VLM2Vec

PY=${PY:-$HOME/anaconda3/envs/vlm2vec_qwen3/bin/python}
SUMMARY=output/FULLFT_GRID_SUMMARY.txt

COMMON=(
    FULL_FT=1
    ZERO_STAGE=3
    PER_DEVICE=64
    HOMOGENEOUS=64
    MAX_STEPS=94
    EXP_PREFIX=Qwen2vl_2B.reloopft.okvqa
    DATA_CONFIG=experiments/public/train/train_okvqa.yaml
    REPORT_TO=none
)

echo "=========================================================="
echo "全参微调 T x M 网格   启动于 $(date '+%F %T')"
echo "  流1 (GPU 5,6,7): T1M5 -> T1M0 -> T1M1 -> T1M0(s43)"
echo "  流2 (GPU 0,1,2): T4M5 -> T4M0 -> T4M1"
echo "  预计 T=1 每格约 32 min、T=4 每格约 43 min，总墙钟约 2.2 h"
echo "=========================================================="

nohup env "${COMMON[@]}" CELLS="B42 A42 B1_42 A43" GPUS=5,6,7 PORT_BASE=2860 \
    bash experiments/public/train/run_reloop_okvqa.sh > /tmp/ft_stream1.log 2>&1 &
S1=$!
sleep 10
nohup env "${COMMON[@]}" CELLS="D42 C42 D1_42" GPUS=0,1,2 PORT_BASE=2960 \
    bash experiments/public/train/run_reloop_okvqa.sh > /tmp/ft_stream2.log 2>&1 &
S2=$!

echo "流1 pid=$S1  流2 pid=$S2"

# 每 10 分钟打一行进度，日志里能看出卡在哪一格
(
    while kill -0 $S1 2>/dev/null || kill -0 $S2 2>/dev/null; do
        sleep 600
        echo "--- $(date '+%T') 流1: $(grep -oE '[0-9]+/94' /tmp/ft_stream1.log | tail -1)" \
             "  流2: $(grep -oE '[0-9]+/94' /tmp/ft_stream2.log | tail -1)"
    done
) &
WATCH=$!

wait $S1
echo "流1 结束于 $(date '+%T')"
wait $S2
echo "流2 结束于 $(date '+%T')"
kill $WATCH 2>/dev/null

echo
echo "两路都结束，开始汇总..."
$PY experiments/public/train/summarize_fullft_grid.py | tee $SUMMARY
echo
echo "汇总已写入 $SUMMARY"
echo "完成于 $(date '+%F %T')"
