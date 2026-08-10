#!/bin/bash
# GradCache 分块大小的显存压力测试。
#
# 为什么需要它：此前判断分块大小用的是 5 步冒烟，而冒烟抽到的子集是随机的，覆盖不到
# 最坏情况，用它下"chunk 64 安全"的结论是不成立的。本脚本改用 _stress_worst.yaml
# （只含 N24News 与 WebQA，两侧序列都接近全集上限 1495 token），让每一步都是最坏情况，
# 再横扫分块大小看峰值显存。
#
# 峰值取自 nvidia-smi 的轮询而非 torch.cuda.max_memory_allocated，因为真正会导致 OOM 的
# 是含缓存与碎片在内的实际占用，而 expandable_segments 下两者能差出若干 GB。
#
# 用法：
#   GPUS=6,7 bash experiments/public/train/stress_gc_chunk.sh
#   GPUS=6,7 CHUNKS="16 32" bash experiments/public/train/stress_gc_chunk.sh

set -e

GPUS=${GPUS:-6,7}
CHUNKS=${CHUNKS:-"16 32 64 128"}
PER_DEVICE=${PER_DEVICE:-128}
STEPS=${STEPS:-3}
PORT=${PORT:-2450}

FIRST_GPU=$(echo "$GPUS" | cut -d, -f1)
RESULT=/tmp/stress_gc_chunk.txt
: > "$RESULT"

echo "压力测试：每卡 $PER_DEVICE 条，全部来自最长的两个子集，$STEPS 步"
echo "卡: $GPUS    分块候选: $CHUNKS"
echo

for CHUNK in $CHUNKS; do
    echo "--- chunk=$CHUNK ---"
    LOG=/tmp/stress_chunk${CHUNK}.out

    # 后台轮询显存，取整个跑动期间的最大值
    ( PEAK=0
      while true; do
          CUR=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$FIRST_GPU" 2>/dev/null | tr -d ' ')
          [ -n "$CUR" ] && [ "$CUR" -gt "$PEAK" ] 2>/dev/null && PEAK=$CUR && echo "$PEAK" > /tmp/stress_peak_${CHUNK}
          sleep 2
      done ) &
    MONITOR=$!

    set +e
    SMOKE=1 MAX_STEPS=$STEPS GPUS="$GPUS" PER_DEVICE=$PER_DEVICE GC_CHUNK=$CHUNK PORT=$PORT \
        DATA_CONFIG=experiments/public/train/_stress_worst.yaml \
        EXP_NAME=stress.chunk$CHUNK \
        bash experiments/public/train/train_qwen2vl_image20_baseline.sh > "$LOG" 2>&1
    STATUS=$?
    set -e

    kill $MONITOR 2>/dev/null || true
    wait $MONITOR 2>/dev/null || true

    PEAK=$(cat /tmp/stress_peak_${CHUNK} 2>/dev/null || echo 0)
    SEQ=$(grep -a -oE "processed_qry_inputs\['input_ids'\]\.shape=torch\.Size\(\[$PER_DEVICE, [0-9]+\]\)" "$LOG" \
          | grep -oE "[0-9]+\]\)$" | tr -d '])' | sort -n | tail -1)
    STEPTIME=$(grep -a -oE "[0-9]+/$STEPS \[[0-9:]+<[0-9:?]+, *[0-9.]+s/it\]" "$LOG" | tail -1 | grep -oE "[0-9.]+s/it")

    if [ "$STATUS" -ne 0 ]; then
        if grep -qa "OutOfMemory" "$LOG"; then
            VERDICT="OOM"
        else
            VERDICT="失败(退出码 $STATUS)"
        fi
    else
        VERDICT="通过"
    fi

    printf "chunk=%-4s %-14s 峰值显存=%6s MiB  最长序列=%-5s  步时=%s\n" \
        "$CHUNK" "$VERDICT" "$PEAK" "${SEQ:-n/a}" "${STEPTIME:-n/a}" | tee -a "$RESULT"

    rm -rf "/home/zhoutuowen/VLM2Vec/output/stress.chunk$CHUNK" 2>/dev/null || true
    sleep 10
done

echo
echo "=== 汇总 ==="
cat "$RESULT"
echo
echo "A40 单卡 48 GB (49140 MiB)。留出余量后，建议取通过项里峰值不超过 40000 MiB 的最大分块。"
