"""探测视频嵌入是否对帧序不敏感（bag-of-frames 坍塌），以及边界符是否是成因。

动机
----
统一多模态嵌入用 InfoNCE 训练，而多数视频-文本配对的语义由"帧的集合"就能决定，不需要
顺序。若表示本身不编码时序，模型在需要时序理解的检索任务上就没有可用信号。

此外 Qwen2-VL 的 M-RoPE 依赖 <|vision_start|> / <|vision_end|> 这对边界符来判定哪一段
token 属于视觉、进而分配三维（时间/高/宽）位置索引。VLM2Vec 系列在构造视频输入时只放了
裸的 <|video_pad|>，没有这对边界符。若如此，视觉 token 会退化为一维文本位置编码，帧间
的时间维直接消失——这会是"时序坍塌"的一个具体机制解释。

设计
----
对每段视频取 8 帧，构造四个变体并各自编码：
  orig  原序
  rev   逆序
  shuf  随机置换
  swap  半数帧换成另一段视频的帧（内容对照）
再加一条 diff（完全另一段视频）用来标定"低相似"是多低。

判读：
  cos(orig, shuf) 接近 1 而 cos(orig, swap) 明显更低  -> 对顺序不敏感但对内容敏感，
  即 bag-of-frames 坍塌。若两者都接近 1，说明表示本身就没区分度，结论不成立。

两种输入格式各跑一遍：
  bare      "<|video_pad|>..."                              （VLM2Vec 现状）
  bounded   "<|vision_start|><|video_pad|><|vision_end|>..." （补上边界符）
若 bare 下顺序不敏感而 bounded 下敏感，则边界符缺失就是成因。

用法:
    cd /home/zhoutuowen/VLM2Vec
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. \
        python experiments/public/train/probe_temporal_collapse.py --num_videos 60
"""
import argparse
import os
import random

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = "/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct"
FRAME_ROOT = "/tmp/vidsample/data/ziyan/video_retrieval"

# VLM2Vec 的 process_input_text 构造视频输入的方式是 video_token + " " + instruction，
# 即裸的 <|video_pad|>，不带 vision 边界符。bare 就是它线上真实使用的格式。
INSTRUCTION = "Represent the given video."
PROMPTS = {
    "bare": f"<|video_pad|> {INSTRUCTION}",
    "bounded": f"<|vision_start|><|video_pad|><|vision_end|> {INSTRUCTION}",
}


def list_videos(dataset, limit):
    root = os.path.join(FRAME_ROOT, dataset, "frames")
    vids = sorted(os.listdir(root))[:limit]
    return [os.path.join(root, v) for v in vids]


def load_frames(vdir, num_frames, size):
    files = sorted(f for f in os.listdir(vdir) if f.endswith((".jpg", ".png")))
    if len(files) < num_frames:
        return None
    # 均匀采样，与常见的视频嵌入实现一致
    idx = [round(i * (len(files) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [Image.open(os.path.join(vdir, files[i])).convert("RGB").resize((size, size))
            for i in idx]


@torch.no_grad()
def encode(model, processor, frames, prompt, device):
    # 这里手工组装而不复用 VLM2Vec 的 process_fn，有两个原因：
    #   1. 它的视频分支用 return_tensors="np"，而 transformers 4.57.6 的视频处理器只收
    #      PyTorch 张量，会直接报错（与此前 Qwen3-VL 图像那次同类）；
    #   2. 它会自动补 vision 边界符，而 bare 这一档要复现的正是"没有边界符"的原状。
    # vendored 的 Qwen2-VL forward 期望 pixel_values_videos / video_grid_thw 是逐样本的
    # list，每项形状 (n_video, 3)；直接传 processor 的二维输出会在 rot_pos_emb 里被索引
    # 成 0 维张量。
    out = processor(text=[prompt], videos=[frames], return_tensors="pt")
    inputs = {
        "input_ids": out["input_ids"].to(device),
        "attention_mask": out["attention_mask"].to(device),
        "pixel_values_videos": [out["pixel_values_videos"].to(device)],
        "video_grid_thw": [out["video_grid_thw"].to(device)],
    }
    # 走 encode_input 才能拿到"它对外输出的那个嵌入"（eos 池化 + 归一化）；
    # 直接读 hidden_states 会漏掉归一化，相似度的量纲对不上。
    rep = model.encode_input(inputs)
    return rep[0].float()


def load_model(checkpoint, device):
    """checkpoint 为空时用基座，否则挂上 VLM2Vec 的 LoRA。"""
    from src.arguments import ModelArguments
    from src.model.model import MMEBModel

    model_args = ModelArguments(
        model_name=MODEL, model_backbone="qwen2_vl", pooling="eos", normalize=True,
        lora=bool(checkpoint), checkpoint_path=checkpoint or None,
    )
    model = MMEBModel.load(model_args, is_trainable=False)
    return model.to(device, dtype=torch.bfloat16).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_videos", type=int, default=60)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--dataset", default="MSR-VTT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="/tmp/hfmodels/VLM2Vec-V2.0",
                    help="VLM2Vec LoRA 路径；传空串则用未训练的基座（其隐状态各向异性，"
                         "相似度数值没有解释力，只适合调试）")
    args = ap.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL)
    model = load_model(args.checkpoint, device)
    print(f"模型: {args.checkpoint or '基座 Qwen2-VL-2B（未训练）'}")

    vdirs = list_videos(args.dataset, args.num_videos * 2)
    clips = []
    for v in vdirs:
        f = load_frames(v, args.num_frames, args.frame_size)
        if f is not None:
            clips.append(f)
        if len(clips) >= args.num_videos + 1:
            break
    print(f"{args.dataset}: 取到 {len(clips)} 段视频，每段 {args.num_frames} 帧 "
          f"({args.frame_size}x{args.frame_size})\n")

    results = {}
    for name, prompt in PROMPTS.items():
        sims = {"rev": [], "shuf": [], "swap": [], "diff": []}
        for i in range(len(clips) - 1):
            f = clips[i]
            other = clips[i + 1]

            e_orig = encode(model, processor, f, prompt, device)

            e_rev = encode(model, processor, list(reversed(f)), prompt, device)

            perm = f[:]
            random.shuffle(perm)
            e_shuf = encode(model, processor, perm, prompt, device)

            # 半数帧换成另一段视频的，用来确认表示对内容是敏感的
            mixed = f[: args.num_frames // 2] + other[args.num_frames // 2:]
            e_swap = encode(model, processor, mixed, prompt, device)

            e_diff = encode(model, processor, other, prompt, device)

            sims["rev"].append(float(e_orig @ e_rev))
            sims["shuf"].append(float(e_orig @ e_shuf))
            sims["swap"].append(float(e_orig @ e_swap))
            sims["diff"].append(float(e_orig @ e_diff))

        results[name] = {k: sum(v) / len(v) for k, v in sims.items()}

        print(f"=== {name} ===")
        for k, v in results[name].items():
            print(f"   cos(orig, {k:4s}) = {v:.4f}")
        print()

    print("=== 判读 ===")
    for name, r in results.items():
        # 顺序敏感度相对内容敏感度的比例：越接近 0 说明顺序被完全忽略
        order_drop = 1.0 - r["shuf"]
        content_drop = 1.0 - r["swap"]
        ratio = order_drop / content_drop if content_drop > 1e-6 else float("nan")
        print(f"{name:8s} 顺序引起的变化 {order_drop:.4f}，内容引起的变化 {content_drop:.4f}，"
              f"占比 {ratio * 100:.1f}%")
    print("\n占比越低说明表示越像 bag-of-frames；若 bounded 明显高于 bare，"
          "则边界符缺失是时序信息丢失的成因之一。")


if __name__ == "__main__":
    main()
