#!/bin/bash
# 测试期深度扫描：对已训练的 checkpoint，在评测时把循环圈数 T 强制改成别的值。
#
# 三个可分离的问题，各由一组 (checkpoint, 强制T) 回答：
#   D42 (训练T=4) 在 T=1,2,3,4,6 下评   -> 训练出的模型是否真在用它的 4 圈？
#       峰在 4：模型确实利用了循环，只是循环形态的净值不如单遍（预算/成本问题）
#       峰在 1-2：后几圈从未被学会，纯粹是噪声（迭代本身没立起来）
#       T=6 外推：Parcae 说 test-time 天花板由训练深度决定，看回落多陡
#   B42 (训练T=1) 在 T=2,4 下评          -> 未经迭代训练的块被硬迭代的代价（幼稚迭代成本）
#   C42 (训练T=4,M=0) 在 T=1,2 下评      -> 为可迭代付出的代价：它的单遍形态比 A42 差多少
#
# 结果写进 $CKPT/eval_ground_forceT$T/，不碰网格的 eval_grounding/。
#
# 用法（两张空卡各起一个实例）:
#   JOBS="D42:1 D42:2 D42:3 D42:6" GPU=0 nohup bash experiments/public/eval/sweep_force_t.sh > /tmp/sweep_gpu0.log 2>&1 &
#   JOBS="B42:2 B42:4 C42:1 C42:2" GPU=3 nohup bash experiments/public/eval/sweep_force_t.sh > /tmp/sweep_gpu3.log 2>&1 &
set -e
cd /home/zhoutuowen/VLM2Vec

GPU=${GPU:-0}
CONFIG=${CONFIG:-experiments/public/eval/grounding.yaml}
PREFIX=${PREFIX:-Qwen2vl_2B.reloopft.ground}
JOBS=${JOBS:?给 JOBS，如 \"D42:1 D42:2\"}

dir_of () {
    case $1 in
        A42) echo "output/$PREFIX.T1.M0.s42" ;;
        B42) echo "output/$PREFIX.T1.M5.s42" ;;
        C42) echo "output/$PREFIX.T4.M0.s42" ;;
        D42) echo "output/$PREFIX.T4.M5.s42" ;;
        *) echo "未知格 $1" >&2; exit 1 ;;
    esac
}

for job in $JOBS; do
    cell=${job%%:*}; t=${job##*:}
    ckpt=$(dir_of $cell)
    out="$ckpt/eval_ground_forceT$t"
    if [ -f "$out/summary.txt" ]; then
        echo "[$(date '+%F %T')] $cell T=$t 已有结果，跳过"
        continue
    fi
    echo "[$(date '+%F %T')] $cell 强制 T=$t 评测开始 (GPU $GPU)"
    RELOOP_FORCE_T=$t OUTPUT_PATH=$out CKPT=$ckpt CONFIG=$CONFIG \
        GPU=$GPU VISION_TOKENS=640 \
        bash experiments/public/eval/eval_reloop.sh
    echo "[$(date '+%F %T')] $cell T=$t 完成"
done
echo "[$(date '+%F %T')] 本实例全部完成: $JOBS"
