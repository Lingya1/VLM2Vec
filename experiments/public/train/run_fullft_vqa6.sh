#!/bin/bash
# VQA6 上的四格：M in {1,5} x T in {1,4}，全参微调，单路串行，无人值守跑完并自动汇总。
#
# 这一轮要回答的是上一轮留下的唯一未排除项：深度的负效应是不是因为数据太少。
# 上一轮在 OK-VQA(9009 条)上，M>=1 的四格训练 loss 末段斜率已经 <0.001/步，
# 也就是数据被榨干了，再加步数只是多背几遍；所以这次动的是数据而不是步数。
#
# 相对上一轮唯一变的是数据：唯一样本 9009 -> 59009（6.5 倍），六个 QA 子集。
# 评测用 vqa.yaml 的 10 个子集，其中 ScienceQA / VizWiz / GQA / TextVQA 训练里没出现，
# 循环深度若真有价值，最该显现的就是这种没见过的分布。
#
# 为什么单路：GPU 0-3 被别的用户的 Ray 作业占着（每卡 31 GB），4 也是别人的，
# 只有 5/6/7 可用。两路并行会直接 OOM —— 这已经发生过一次。
#
# 格的顺序按信息量：先 M=5 的两格拿到深度效应（论文设置），再补 M=1 的两格看
# register 数量。中途出事时保住的是能回答核心问题的那两格。
#
# 配置来自实测：
#   - 每图 640 个视觉 token（压缩只作用于 DocVQA/InfographicsVQA，1248->约 635；
#     OK-VQA/A-OKVQA/Visual7W/ChartQA 逐 token 不变，已逐元素核对过）
#   - 每卡 48、全局批 144。每卡 64 在这个混合下差 240 MB，OOM。
#   - T=1 20.24 s/it，T=4 26.74 s/it（14 步冒烟实测）
#   - 410 步 = 1 个 epoch
#
# 注意：全局批 144 与 OK-VQA 那一轮的 192 不同，负样本池变了，绝对分不跨轮比；
# 四格之间一致，观测量是格与格之差。
#
# 用法：
#   nohup bash experiments/public/train/run_fullft_vqa6.sh > /tmp/vqa6_grid.log 2>&1 &
#   汇总在 output/VQA6_GRID_SUMMARY.txt
#   只跑前两格：CELLS="B42 D42" bash experiments/public/train/run_fullft_vqa6.sh

cd /home/zhoutuowen/VLM2Vec

PY=${PY:-$HOME/anaconda3/envs/vlm2vec_qwen3/bin/python}
SUMMARY=${SUMMARY:-output/VQA6_GRID_SUMMARY.txt}
CELLS=${CELLS:-"B42 D42 B1_42 D1_42"}
GPUS=${GPUS:-5,6,7}
# 第二条流必须换端口，否则两个 torchrun 的 rendezvous 撞在一起，
# 后起的那个会连上前一个的 store 并卡死在 rendezvous 上，不报错也不推进。
PORT_BASE=${PORT_BASE:-2880}

echo "=========================================================="
echo "VQA6 四格（单路串行）  启动于 $(date '+%F %T')"
echo "  GPU $GPUS   队列: $CELLS"
echo "  T=1 约 2.4 h/格，T=4 约 3.0 h/格，四格训练约 10.8 h"
echo "  加 10 子集评测，预计总计约 14 h"
echo "=========================================================="

env FULL_FT=1 ZERO_STAGE=3 VISION_TOKENS=640 PER_DEVICE=48 HOMOGENEOUS=48 \
    MAX_STEPS=410 EXP_PREFIX=Qwen2vl_2B.reloopft.vqa6 \
    DATA_CONFIG=experiments/public/train/train_vqa6_10k.yaml \
    EVAL_CONFIG=experiments/public/eval/vqa.yaml \
    REPORT_TO=none CELLS="$CELLS" GPUS=$GPUS PORT_BASE=$PORT_BASE \
    bash experiments/public/train/run_reloop_okvqa.sh

echo
echo "队列跑完，开始汇总..."
$PY experiments/public/train/summarize_vqa6_grid.py | tee $SUMMARY
echo
echo "汇总已写入 $SUMMARY"
echo "完成于 $(date '+%F %T')"
