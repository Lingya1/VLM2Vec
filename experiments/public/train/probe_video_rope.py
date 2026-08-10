"""直接查看视频输入在有/无 vision 边界符两种情况下拿到的 M-RoPE 位置索引。

相似度探针只能间接反映位置编码的差异；位置索引本身是可以直接打印的，是关于机制的
最直接证据。Qwen2-VL 的 get_rope_index 返回 (3, batch, seq)，三行分别是时间、高、宽。

判读要点：
  - 若视觉段三行完全相同且逐位递增，说明退化为一维顺序编码（与文本 token 无异）；
  - 若时间行按帧分组阶梯上升、高宽行在帧内平铺且帧间重复，说明三维 M-RoPE 生效。
"""
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL = "/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct"
FRAME_DIR = "/tmp/vidsample/data/ziyan/video_retrieval/MSR-VTT/frames"

PROMPTS = {
    "bare": "<|video_pad|>\nRepresent the given video.",
    "bounded": "<|vision_start|><|video_pad|><|vision_end|>\nRepresent the given video.",
}


def main():
    # 必须用 VLM2Vec 自己 vendor 的那份实现，而不是 transformers 原生的：结论是关于
    # "VLM2Vec 线上跑的是什么"，换一份实现就不算数了。该类在新版 transformers 下不能
    # 直接 from_pretrained（_init_weights 读 vision_config.initializer_range 会失败），
    # 所以经 MMEBModel.load 这条项目自己的加载路径拿到它。
    from src.arguments import ModelArguments
    from src.model.model import MMEBModel

    processor = AutoProcessor.from_pretrained(MODEL)
    wrapper = MMEBModel.load(
        ModelArguments(model_name=MODEL, model_backbone="qwen2_vl", pooling="eos",
                       normalize=True, lora=False),
        is_trainable=False)
    model = wrapper.encoder
    assert hasattr(model, "get_rope_index"), type(model).__name__

    vid = sorted(os.listdir(FRAME_DIR))[0]
    files = sorted(f for f in os.listdir(os.path.join(FRAME_DIR, vid)))[:8]
    # 用小尺寸让每帧的 token 数少到能整屏打印出来
    frames = [Image.open(os.path.join(FRAME_DIR, vid, f)).convert("RGB").resize((56, 56))
              for f in files]

    video_token_id = model.config.video_token_id
    for name, prompt in PROMPTS.items():
        inputs = processor(text=[prompt], videos=[frames], return_tensors="pt")
        ids = inputs["input_ids"]
        pos, _ = model.get_rope_index(
            ids, image_grid_thw=None, video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs["attention_mask"])

        vis = (ids[0] == video_token_id).nonzero().flatten()
        s, e = int(vis[0]), int(vis[-1])
        grid = inputs["video_grid_thw"][0].tolist()
        print(f"=== {name} ===")
        print(f"  序列长 {ids.shape[1]}，视觉段 [{s}, {e}] 共 {e - s + 1} 个 token，"
              f"video_grid_thw={grid}")
        for row, label in enumerate(("时间", "高  ", "宽  ")):
            seg = pos[row, 0, s:e + 1].tolist()
            print(f"  {label}: {seg}")
        t, h, w = pos[:, 0, s:e + 1]
        print(f"  三行是否完全相同（相同=退化为一维）: {bool(torch.equal(t, h) and torch.equal(h, w))}")
        print(f"  时间行的不同取值个数: {len(set(t.tolist()))}（应等于时间分组数 {grid[0]}）")
        print()


if __name__ == "__main__":
    main()
