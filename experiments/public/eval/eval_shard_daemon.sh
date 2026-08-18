#!/bin/bash
# 在空闲卡上替主评测进程分担尾部子集，缩短每格的评测墙钟时间。
#
# 为什么按子集分片而不是数据并行：eval.py 支持 torchrun 多卡，但那条路径在本仓库这个
# 配置下没验证过，而 B42 是单卡评的。四格之间要比的差值只有 1-2 分，正好落在
# gather/padding 类 bug 能造成的量级里。按子集分片则每个进程走的是与 B42 逐比特相同的
# 单卡代码路径，只是各自负责不同子集，不存在可比性风险。
#
# 为什么只分尾部：主进程按 vqa.yaml 顺序串行推进，eval.py 靠"_qry 文件是否存在"跳过
# 已完成的子集。把靠前的子集分出去，主进程可能在分片进程写文件的中途去读，读到半个
# pickle。只接管尾部四个（ScienceQA/VizWiz/GQA/TextVQA），主进程要二十分钟后才会走到，
# 时间差足够。
#
# 为什么直接调 eval.py 而不是 eval_reloop.sh：后者跑完会在目录里写 summary.txt，
# 而 summary.txt 正是本脚本判断"这一格评完了"的标志，分片进程写它会造成误判。
#
# 用法（无人值守）：
#   nohup bash experiments/public/eval/eval_shard_daemon.sh > /tmp/eval_shard.log 2>&1 &

cd /home/zhoutuowen/VLM2Vec

export PATH="/home/zhoutuowen/anaconda3/envs/vlm2vec_qwen3/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME=/home/zhoutuowen/.cache/huggingface
export HF_DATASETS_CACHE=/home/zhoutuowen/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
# 训练侧开着边界符修复，评测必须一致，否则输入分布对不上且不会报错
export VLM2VEC_NO_VISION_BOUNDARY=0

DATA_BASEDIR=/home/zhoutuowen/data/MMEB-V2
# 必须与训练同值：训练看 640 token 的文档图、评测看 1248 token 的，输入分布就对不上了
VISION_TOKENS=${VISION_TOKENS:-640}
MAX_PIXELS=$((28 * 28 * VISION_TOKENS))
GPU_A=${GPU_A:-6}
GPU_B=${GPU_B:-7}

# 队列顺序必须与主编排一致，否则会在还没训完的格上空等
CELLS=${CELLS:-"Qwen2vl_2B.reloopft.vqa6.T4.M5.s42 \
                Qwen2vl_2B.reloopft.vqa6.T1.M1.s42 \
                Qwen2vl_2B.reloopft.vqa6.T4.M1.s42"}

run_shard () {
    local ckpt=$1 cfg=$2 gpu=$3
    CUDA_VISIBLE_DEVICES=$gpu python eval.py \
        --pooling eos \
        --normalize true \
        --resize_max_pixels $MAX_PIXELS \
        --per_device_eval_batch_size 16 \
        --model_backbone "qwen2_vl" \
        --model_name "$ckpt" \
        --checkpoint_path "$ckpt" \
        --dataset_config "$cfg" \
        --encode_output_path "$ckpt/eval_vqa" \
        --data_basedir "$DATA_BASEDIR" \
        > "$ckpt/eval_vqa/shard_$(basename "$cfg" .yaml).log" 2>&1
}

for exp in $CELLS; do
    dir="output/$exp"
    echo "[$(date '+%F %T')] 等待 $exp 训练完成..."
    # model.safetensors 只在训练结束时写出，用它判断训练是否收尾；
    # summary.txt 由主评测进程在最后写出，出现即说明这一格已经评完，无需再分担。
    while [ ! -f "$dir/model.safetensors" ] && ! ls "$dir"/model-*.safetensors >/dev/null 2>&1; do
        [ -f "$dir/eval_vqa/summary.txt" ] && break
        sleep 60
    done

    if [ -f "$dir/eval_vqa/summary.txt" ]; then
        echo "[$(date '+%F %T')] $exp 已评完，跳过"
        continue
    fi

    mkdir -p "$dir/eval_vqa"
    echo "[$(date '+%F %T')] $exp 分片评测开始：GPU $GPU_A 跑 shardA，GPU $GPU_B 跑 shardB"
    run_shard "$dir" experiments/public/eval/vqa_shardA.yaml $GPU_A &
    pa=$!
    run_shard "$dir" experiments/public/eval/vqa_shardB.yaml $GPU_B &
    pb=$!
    wait $pa; wait $pb
    echo "[$(date '+%F %T')] $exp 分片完成，已产出: $(ls "$dir"/eval_vqa/*_score.json 2>/dev/null | wc -l) 个子集"
done

echo "[$(date '+%F %T')] 全部格处理完毕"
