#!/bin/bash
# RET 全量（8 子集 595,700 条，不截断）单格全参微调 + 自动评测，一条龙后台跑。
#
# 这轮回答的问题：此前 OK-VQA/VQA6/grounding 三轮的共同批评是训练预算太小
# （1.8 万～6 万条），结论够不到论文 128 万条的作用域。RET 全量把预算抬到
# 论文的一半左右，且候选侧是真实图片/长句——此前所有证据都指向"候选内容
# 越丰富，迭代机制越有戏"。这格是 T=3 M=5；对照格（T=1 M=0/M=5）等这格
# 出分后再定，避免空烧 21 小时。
#
# 关键开关 RELOOP_DUMMY_VISION=1：RET 混合里 t2i 的 query 是纯文本，i2t 的
# query 带图。同质批（每卡整批来自单一子集）下两卡会走不同的视觉塔分支，
# ZeRO-3 的参数 allgather 顺序错位，第 0 步 NCCL 死锁（8/13 冒烟实测：两 rank
# 停在同一 SeqNum、NumelIn 一个 768 一个 752640）。开关让无图前向也过一张
# 零值假图（输出乘 0 加回，数值逐比特不变），顺序恒同。8/14 冒烟 12 步通过。
#
# 步数：595700 / (40*2) = 7446 步 = 恰好一个 epoch。冒烟实测 10.3 s/步，
# 训练约 21.2 小时；评测 12 子集拆 A/B 两片在两张卡上并行。
#
# 用法：  nohup bash experiments/public/train/run_fullft_ret.sh > /tmp/ret_full.log 2>&1 &

set -u
cd /home/zhoutuowen/VLM2Vec

GPUS=${GPUS:-0,3}
GPU_A=${GPUS%%,*}
GPU_B=${GPUS##*,}
T=${T:-3}
M=${M:-5}
EXP_NAME=Qwen2vl_2B.reloop.ret_full.T$T.M$M.s42.BS80.2A40
DIR=output/$EXP_NAME

echo "[$(date '+%F %T')] === RET 全量 T=$T M=$M，卡 $GPUS ==="

if [ -f "$DIR/model.safetensors" ] || ls "$DIR"/model-*.safetensors >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] 已有训练产物，跳过训练"
else
    # 8/14 失败复盘：/home 共享盘 I/O 争抢导致 dataloader 反复停顿 4-18 分钟，
    # 第 1993 步一次停顿超过 NCCL 默认 30 分钟超时，rank0 被看门狗杀掉，5.8 小时
    # 无产物。对策：超时放宽到 3 小时（DDP_TIMEOUT，观察到的最长停顿 18 分钟，
    # 3 小时足以存活）、每 1000 步存一次权重（SAVE_STEPS，只留 2 份，再崩也能
    # 从中间产物评测）。数据仍从 /home 读——争抢期会慢，但不会死。
    RELOOP_DUMMY_VISION=1 FULL_FT=1 ZERO_STAGE=3 \
    GPUS=$GPUS PER_DEVICE=40 HOMOGENEOUS=40 \
    RELOOP_T=$T RELOOP_M=$M SEED=42 \
    DATA_CONFIG=experiments/public/train/train_ret_full.yaml \
    MAX_STEPS=7446 REPORT_TO=none PORT=2470 \
    DDP_TIMEOUT=10800 SAVE_STEPS=1000 SAVE_LIMIT=2 \
    EXP_NAME=$EXP_NAME \
    bash experiments/public/train/train_qwen2vl_reloop.sh
    status=$?
    if [ $status -ne 0 ]; then
        echo "[$(date '+%F %T')] 训练失败（退出码 $status），不评测"; exit $status
    fi
fi

echo "[$(date '+%F %T')] 训练完成，开始双卡分片评测"
mkdir -p "$DIR/eval_ret"
CKPT=$DIR CONFIG=experiments/public/eval/ret_shardA.yaml GPU=$GPU_A \
    OUTPUT_PATH=$DIR/eval_ret bash experiments/public/eval/eval_reloop.sh \
    > "$DIR/eval_ret/shardA.log" 2>&1 &
pa=$!
CKPT=$DIR CONFIG=experiments/public/eval/ret_shardB.yaml GPU=$GPU_B \
    OUTPUT_PATH=$DIR/eval_ret bash experiments/public/eval/eval_reloop.sh \
    > "$DIR/eval_ret/shardB.log" 2>&1 &
pb=$!
wait $pa; ra=$?
wait $pb; rb=$?
echo "[$(date '+%F %T')] 评测结束 (shardA=$ra shardB=$rb)"

echo "=== $EXP_NAME RET 12 子集 hit@1 ==="
for f in "$DIR"/eval_ret/*_score.json; do
    [ -e "$f" ] || continue
    printf "  %-18s %s\n" "$(basename "$f" _score.json)" \
        "$(/home/zhoutuowen/anaconda3/envs/vlm2vec_qwen3/bin/python -c "import json; print(f\"{json.load(open('$f'))['hit@1']*100:.2f}\")")"
done | tee "$DIR/eval_ret/summary.txt"
echo "[$(date '+%F %T')] 全部完成"
