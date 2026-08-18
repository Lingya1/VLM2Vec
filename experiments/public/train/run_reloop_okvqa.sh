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
# 默认不接 wandb：它的 init 在这台机器上会超过 90 s 的默认超时而整个跑挂掉，而这是个
# 4 小时以上的无人值守任务，不该依赖外部服务。每步 loss 有 logging_steps=1 写进 train.log，
# 曲线用 summarize_reloop_okvqa.py 还原。要接就 REPORT_TO=wandb。
REPORT_TO=${REPORT_TO:-none}
# 非交互 shell 的 PATH 里没有裸 python，汇总段取分数要用环境里的解释器
PY=${PY:-$HOME/anaconda3/envs/vlm2vec_qwen3/bin/python}
# 两个实例并行跑时必须给不同的 PORT_BASE，否则 torchrun 的 rendezvous 端口会撞
PORT_BASE=${PORT_BASE:-2460}

# 单子集下同源块无意义（只有一个源），设成等于 per_device 即每步整批同源
HOMOGENEOUS=${HOMOGENEOUS:-64}

# 全参跑用不同前缀，免得和 LoRA 那批结果互相覆盖
EXP_PREFIX=${EXP_PREFIX:-Qwen2vl_2B.reloop.okvqa}

# 每图最多几个视觉 token。1280 是仓库原值；文档子集（DocVQA/InfographicsVQA）会被顶到
# 约 1248，在 3xA40 上放不下，压到 640 只影响这两个子集，自然图子集逐 token 不变。
VISION_TOKENS=${VISION_TOKENS:-1280}

run_cell () {
    local name=$1 t=$2 m=$3 seed=$4
    local exp="$EXP_PREFIX.T$t.M$m.s$seed"
    local dir="output/$exp"

    if [ -f "$dir/adapter_model.safetensors" ] || [ -f "$dir/model.safetensors" ]; then
        echo ">>> $name 已存在，跳过训练: $dir"
    else
        echo ">>> $name 开始训练 (T=$t M=$m seed=$seed)"
        EXP_NAME=$exp GPUS=$GPUS PER_DEVICE=$PER_DEVICE HOMOGENEOUS=$HOMOGENEOUS \
        MAX_STEPS=$MAX_STEPS SEED=$seed RELOOP_T=$t RELOOP_M=$m REPORT_TO=$REPORT_TO \
        DATA_CONFIG=$DATA_CONFIG PORT=$((PORT_BASE + seed % 100)) \
        FULL_FT=${FULL_FT:-0} ZERO_STAGE=${ZERO_STAGE:-3} \
        VISION_TOKENS=$VISION_TOKENS \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            bash experiments/public/train/train_qwen2vl_reloop.sh
    fi

    if [ "${SKIP_EVAL:-0}" != "1" ]; then
        echo ">>> $name 开始评测"
        # VISION_TOKENS 必须与训练同值，否则训练看压缩过的文档图、评测看原始分辨率，
        # 输入分布对不上而且不会报错。
        CKPT=$dir CONFIG=$EVAL_CONFIG GPU=${GPUS%%,*} VISION_TOKENS=$VISION_TOKENS \
            bash experiments/public/eval/eval_reloop.sh
    fi
}

# 无人值守时单格失败不能拖垮整队：剩下的格照跑，失败的记下来在汇总里显示。
# 没有这层包装的话，set -e 会让一次 OOM 吃掉整晚。
FAILED=""
run_cell_safe () {
    if ! run_cell "$@"; then
        echo "!!! $1 失败（已跳过，队列继续）"
        FAILED="$FAILED $1"
    fi
}

for cell in $CELLS; do
    case $cell in
        A42) run_cell_safe A42 1 0 42 ;;
        A43) run_cell_safe A43 1 0 43 ;;
        # B 与 C 把 D 的两个改动拆开：D 同时换了读出位置（追加 register 后从末位池化）和
        # 循环深度，两者绑在一起无法归因。B 只改读出，C 只加深度。
        B42) run_cell_safe B42 1 5 42 ;;
        # M=1 把"追加多少个 register"与"读出位置变了"拆开：因果掩码下 register 影响不到
        # 任何真实 token，M 的唯一通路就是读出位置，M=5 相对 M=1 只多了 register 之间的
        # 级联聚合。若 B1 与 B 同分，则 register 数量无意义，整个机制退化为一行池化改动。
        B1_42) run_cell_safe B1_42 1 1 42 ;;
        C42) run_cell_safe C42 4 0 42 ;;
        # D1 是 D 在 M 轴上的对照：若 D1≈D，则第 2~5 个 register 在深度下同样是冗余的，
        # 循环带来的增益与"多少个工作位"无关，只与读出位置有关。
        D1_42) run_cell_safe D1_42 4 1 42 ;;
        D42) run_cell_safe D42 4 5 42 ;;
        *) echo "未知的格: $cell（可选 A42 A43 B1_42 B42 C42 D1_42 D42）"; exit 1 ;;
    esac
done

echo
echo "================= 汇总 ================="
printf "%-34s %s\n" "实验" "OK-VQA hit@1"
for cell in $CELLS; do
    case $cell in
        A42) exp=$EXP_PREFIX.T1.M0.s42 ;;
        A43) exp=$EXP_PREFIX.T1.M0.s43 ;;
        B42) exp=$EXP_PREFIX.T1.M5.s42 ;;
        B1_42) exp=$EXP_PREFIX.T1.M1.s42 ;;
        C42) exp=$EXP_PREFIX.T4.M0.s42 ;;
        D1_42) exp=$EXP_PREFIX.T4.M1.s42 ;;
        D42) exp=$EXP_PREFIX.T4.M5.s42 ;;
    esac
    # 评测目录随 EVAL_CONFIG 而变；多子集时取各子集 hit@1 的均值。
    d="output/$exp/eval_$(basename "$EVAL_CONFIG" .yaml)"
    if ls "$d"/*_score.json >/dev/null 2>&1; then
        printf "%-34s %s\n" "$cell ($exp)" \
            "$($PY -c "
import json, glob
v = [json.load(open(f))['hit@1'] for f in glob.glob('$d/*_score.json')]
print(f'{sum(v)/len(v)*100:.2f}  ({len(v)} 个子集均值)')")"
    else
        printf "%-34s %s\n" "$cell ($exp)" "(无结果)"
    fi
done
echo
echo "读法：先看 |A42-A43| 有多大，那是噪声下限；D42-A42 要明显超过它才算信号。"
