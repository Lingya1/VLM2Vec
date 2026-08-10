#!/bin/bash
# 指令措辞的距离梯度：在同一个子集上，从"只改一个词"到"完全改写"逐档替换查询指令，
# 看性能随措辞距离怎么掉。
#
# 要区分的是两种截然不同的解释：
#   (a) 模型把指令当作精确匹配的字符串键 —— 连只改一个词都会大幅掉分；
#   (b) 只是普通的 prompt 敏感 —— 只有语义距离较远的改写才掉分。
# 前者是有分量的发现，后者已经被纯文本的工作写过（arXiv 2605.22544）。
#
# 用法:
#   GPU=0 SUBSET=SUN397 bash experiments/public/eval/probe_instruction_gradient.sh

set -e
cd /home/zhoutuowen/VLM2Vec

GPU=${GPU:-0}
SUBSET=${SUBSET:-SUN397}
TMPDIR_YAML=/tmp/instr_grad_$SUBSET
mkdir -p "$TMPDIR_YAML"

# 按与原句的距离从近到远排列。原句是 "Identify the scene shown in the image."
declare -a VARIANTS=(
  "original|"
  "case|identify the scene shown in the image."
  "nodot|Identify the scene shown in the image"
  "oneword|Identify the scene depicted in the image."
  "synonym|Recognize the scene shown in the image."
  "reorder|In the image, identify the scene shown."
  "rewrite|What kind of environment is depicted here?"
)
if [ "$SUBSET" = "VOC2007" ]; then
  VARIANTS=(
    "original|"
    "case|identify the object shown in the image."
    "nodot|Identify the object shown in the image"
    "oneword|Identify the object depicted in the image."
    "synonym|Recognize the object shown in the image."
    "reorder|In the image, identify the object shown."
    "rewrite|Name the specific thing pictured."
  )
fi

for entry in "${VARIANTS[@]}"; do
  tag="${entry%%|*}"
  inst="${entry#*|}"
  out="grad_${SUBSET}_${tag}"
  if [ -s "/home/zhoutuowen/VLM2Vec/output/vlm2vecv2_probe/$out/summary.txt" ]; then
    echo "跳过已完成的 $tag"; continue
  fi
  yaml="$TMPDIR_YAML/$tag.yaml"
  {
    echo "$SUBSET:"
    echo "    dataset_parser: image_cls"
    echo "    dataset_name: $SUBSET"
    echo "    dataset_split: test"
    echo "    image_root: image-tasks/"
    echo "    eval_type: local"
    [ -n "$inst" ] && echo "    qry_inst_override: \"$inst\""
  } > "$yaml"

  echo "=== $tag: ${inst:-（原始指令）} ==="
  GPU=$GPU CONFIG="$yaml" OUT_NAME="$out" \
    bash experiments/public/eval/eval_vlm2vecv2_cls.sh > "/tmp/grad_${SUBSET}_${tag}.out" 2>&1
done

echo
echo "======== $SUBSET 指令措辞梯度 ========"
for entry in "${VARIANTS[@]}"; do
  tag="${entry%%|*}"
  inst="${entry#*|}"
  s=$(cat "/home/zhoutuowen/VLM2Vec/output/vlm2vecv2_probe/grad_${SUBSET}_${tag}/summary.txt" 2>/dev/null | awk '{print $2}')
  printf "  %-10s %6s   %s\n" "$tag" "${s:-n/a}" "${inst:-（原始指令）}"
done
