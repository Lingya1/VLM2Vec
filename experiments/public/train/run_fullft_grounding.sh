#!/bin/bash
# Visual grounding 上的 2x2：M in {0,5} x T in {1,4}，全参微调，单路串行，无人值守。
#
# 这一轮要回答的问题
# ------------------
# VQA6 上循环深度一致为负（十子集 −1.47）。但那一轮同时暴露了两个问题，使得结论不能
# 外推到论文的主张：
#   1. 我们的 T1M0 基线欠拟合（训练 loss 0.767 且斜率 −0.0075 仍在陡降，而带 register
#      的格子都在 0.24）。于是 register 和深度都在扮演"让 InfoNCE 变得可拟合"的优化辅助，
#      谁先上谁拿走全部功劳 —— 交互项因此从论文的 +1.4 翻成我们的 −7.8。
#   2. 评测里完全没有论文效应最强的那类任务。论文增益集中在 VisDoc-OOD（65.3 对基线
#      39.4）和 image grounding（83.9 对 77.3），而我们评的十个子集全是 image-VQA，
#      DocVQA 还已经 88.7 顶到天花板。
#
# grounding 同时改善这两点：效应大（6.6 分，是 All 平均 2.6 分的两倍半，且远超我们
# 1.2 分的种子噪声），且 MMEB 只有一个域内训练子集配三个零样本评测子集，一次训练就拿到
# 域内与 OOD 两条读数。
#
# 判读方式（这是重点，先写下来免得事后挑解释）
# ------------------------------------------
#   ΔT@M=5 = D42 − B42   转正 -> VQA6 的负号是任务不对，结构本身有效
#                        仍为负 -> 结构问题，不是数据问题
#   ΔT@M=0 = C42 − A42   VQA6 那轮这一格是 +4.6（与论文 +1.2 同号），
#                        是判断"递归实现有没有坏"的锚点，必须保留
#   A42 的末段 loss 斜率  这一轮是否又是欠拟合基线的判据。若斜率仍 < −0.003，
#                        则本轮的 ΔM 同样不能当机制证据，必须加预算重跑
#
# 格的顺序按信息量：B42/D42 先跑，它们直接回答"数据还是结构"；A42/C42 补 M=0 那一列
# 用于解释。中途出事时保住的是能回答核心问题的那两格。
#
# 配置说明
# --------
#   - 与 VQA 的关键区别：query 与 target 两侧都带图（query 是整图 + 指称短语，target 是
#     物体裁剪图），每样本视觉 token 明显更多，所以 PER_DEVICE 不能沿用 VQA6 的 48，
#     由下面的冒烟段实测标定。
#   - VISION_TOKENS=640 在 grounding 上不绑定：COCO 图最长边 640，640x480/784 约 392
#     个 token，裁剪图更小。设着只是给异常大图一个上限，并与 VQA6 那轮保持同一个旋钮值。
#   - 6 万条对齐 VQA6 的 59009 条；步数由 batch 反算，保证四格看到的样本数一致。
#
# 用法：
#   nohup bash experiments/public/train/run_fullft_grounding.sh > /tmp/grounding_grid.log 2>&1 &
#   WAIT_FOR_IDLE=0 ... 跳过等待前一队列
#   CELLS="B42 D42" ... 只跑前两格

cd /home/zhoutuowen/VLM2Vec

PY=${PY:-$HOME/anaconda3/envs/vlm2vec_qwen3/bin/python}
GPUS=${GPUS:-5,6,7}
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
CELLS=${CELLS:-"B42 D42 A42 C42"}
SUMMARY=${SUMMARY:-output/GROUNDING_GRID_SUMMARY.txt}
# 与 VQA6 那一轮的 2880 错开，避免两个 torchrun 的 rendezvous 撞端口后静默卡死
PORT_BASE=${PORT_BASE:-2920}
TARGET_SAMPLES=${TARGET_SAMPLES:-60000}
VISION_TOKENS=${VISION_TOKENS:-640}
EXP_PREFIX=${EXP_PREFIX:-Qwen2vl_2B.reloopft.ground}

echo "=========================================================="
echo "Grounding 2x2（全参微调，单路串行）  启动于 $(date '+%F %T')"
echo "  GPU $GPUS   队列: $CELLS"
echo "=========================================================="

# ---------- 1. 等前一条队列腾出卡 ----------
# D1_42 还在训练加评测。这里不去 kill 任何东西，只等自己的 train.py/eval.py 都退干净。
# 用 pgrep -u 限定本人，免得把别的用户的进程当成自己的而永远等下去。
if [ "${WAIT_FOR_IDLE:-1}" = "1" ]; then
    echo "[$(date '+%F %T')] 等待当前队列（D1_42 训练+评测）结束..."
    while pgrep -u "$(id -u)" -f "train\.py|eval\.py" >/dev/null 2>&1; do
        sleep 120
    done
    echo "[$(date '+%F %T')] 卡已空出，继续"
fi

# ---------- 2. 冒烟标定 per-device batch ----------
# 用最吃显存的那格（T=4 M=5）去试，从大到小取第一个活下来的。
# 不这么做的话只有两种选择：拍一个保守值白扔一半吞吐，或者拍大了在第三小时 OOM。
if [ -n "${PER_DEVICE:-}" ]; then
    echo "[$(date '+%F %T')] 跳过冒烟，沿用给定的 PER_DEVICE=$PER_DEVICE"
else
    for cand in 40 32 24 16; do
        echo "[$(date '+%F %T')] 冒烟 per-device=$cand (T=4 M=5, 12 步)..."
        rm -rf output/Qwen2vl_2B.reloop.smoke.T4.M5
        if env SMOKE=1 FULL_FT=1 ZERO_STAGE=3 RELOOP_T=4 RELOOP_M=5 \
                PER_DEVICE=$cand HOMOGENEOUS=$cand VISION_TOKENS=$VISION_TOKENS \
                DATA_CONFIG=experiments/public/train/train_grounding.yaml \
                GPUS=$GPUS PORT=$((PORT_BASE + 90)) REPORT_TO=none \
                bash experiments/public/train/train_qwen2vl_reloop.sh \
                > /tmp/ground_smoke_$cand.log 2>&1; then
            PER_DEVICE=$cand
            SPIT=$(grep -oE "[0-9.]+s/it" /tmp/ground_smoke_$cand.log | tail -1)
            echo "[$(date '+%F %T')] per-device=$cand 通过，T=4 实测 ${SPIT:-未取到}"
            break
        fi
        echo "[$(date '+%F %T')] per-device=$cand 失败（多半是 OOM），降一档"
    done
    if [ -z "${PER_DEVICE:-}" ]; then
        echo "连 16 都放不下，停下来人工看 /tmp/ground_smoke_16.log"
        exit 1
    fi
fi

GLOBAL=$((PER_DEVICE * NGPU))
MAX_STEPS=$((TARGET_SAMPLES / GLOBAL))
echo "=========================================================="
echo "  每卡 $PER_DEVICE  全局批 $GLOBAL  步数 $MAX_STEPS ($TARGET_SAMPLES 样本)"
echo "=========================================================="

# ---------- 3. 四格 ----------
env FULL_FT=1 ZERO_STAGE=3 VISION_TOKENS=$VISION_TOKENS \
    PER_DEVICE=$PER_DEVICE HOMOGENEOUS=$PER_DEVICE \
    MAX_STEPS=$MAX_STEPS EXP_PREFIX=$EXP_PREFIX \
    DATA_CONFIG=experiments/public/train/train_grounding.yaml \
    EVAL_CONFIG=experiments/public/eval/grounding.yaml \
    REPORT_TO=none CELLS="$CELLS" GPUS=$GPUS PORT_BASE=$PORT_BASE \
    bash experiments/public/train/run_reloop_okvqa.sh

# ---------- 4. 汇总 ----------
echo
echo "队列跑完，开始汇总..."
EXP_PREFIX=$EXP_PREFIX $PY experiments/public/train/summarize_grounding_grid.py | tee $SUMMARY
echo
echo "汇总已写入 $SUMMARY"
echo "完成于 $(date '+%F %T')"
