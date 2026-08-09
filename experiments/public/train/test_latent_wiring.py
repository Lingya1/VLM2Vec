"""在真实 Qwen2-VL-2B 上验证 latent 瓶颈的模型侧接线。

单测（test_latent_bottleneck.py）只覆盖了瓶颈模块自身的数学；这里验证的是它与 backbone
的接缝，也就是最容易静默出错的部分：

  - reason token 的嵌入有没有真的落到末尾 K 个位置（左填充下才成立的假设）
  - 追加的占位 token 有没有破坏视觉分支（Qwen2-VL 靠 input_ids 定位图像占位符再
    masked_scatter 视觉特征，注入方式选错会让图像特征丢失）
  - 梯度能否穿过 hook 回到 reason_embed 与 mu/logvar 头
  - 训练采样与推理取均值两条分支的行为

用法:
    cd /home/zhoutuowen/VLM2Vec
    CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 PYTHONPATH=. \
        python experiments/public/train/test_latent_wiring.py
"""
import torch
from PIL import Image

from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import QWEN2_VL, load_processor, process_vlm_inputs_fns

MODEL = "/home/zhoutuowen/weights_model/weights_models/Qwen/Qwen2-VL-2B-Instruct"
K = 4

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def build_batch(processor, device, batch_size=2):
    img = Image.new("RGB", (112, 112), color=(120, 180, 90))
    texts = ["<|image_pad|>\nWhat is shown in the image?"] * batch_size
    inputs = process_vlm_inputs_fns[QWEN2_VL](
        {"text": texts, "images": [img] * batch_size}, processor=processor, max_length=512
    )
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_args = ModelArguments(
        model_name=MODEL, model_backbone="qwen2_vl", pooling="eos", normalize=True,
        lora=False, latent_k=K, latent_beta=1e-3, latent_free_bits=0.0,
    )
    data_args = DataArguments()
    processor = load_processor(model_args, data_args)

    print("加载模型…")
    model = MMEBModel.build(model_args).to(device)
    model.train()

    batch = build_batch(processor, device)
    seq_len = batch["input_ids"].shape[1]

    print("\n=== 序列扩展与视觉分支 ===")
    ext_ids, ext_mask = model.latent.extend_inputs(batch["input_ids"], batch["attention_mask"])
    check("追加了 K 个位置", ext_ids.shape[1] == seq_len + K,
          f"{seq_len} -> {ext_ids.shape[1]}")
    check("追加位置的 attention_mask 为 1", bool(ext_mask[:, -K:].all()))
    img_token_id = model.config.image_token_id
    check("占位 token 不与图像占位符冲突",
          model.latent.placeholder_token_id != img_token_id,
          f"placeholder={model.latent.placeholder_token_id}, image_token={img_token_id}")

    print("\n=== 前向 ===")
    reps = model.encode_input(batch)
    check("表示形状正确", reps.shape == (2, model.latent.bottleneck.latent_size),
          f"{tuple(reps.shape)}")
    check("表示无 NaN/Inf", bool(torch.isfinite(reps).all()))

    rate = model.pop_rate_loss()
    check("率被累积且为正", rate is not None and float(rate) > 0,
          f"rate={float(rate):.2f} nats")
    # 这里只报数不做断言。未训练模型的率没有解释力：to_mu 是随机初始化的线性层作用在
    # 幅度不小的隐状态上，mu 很大；init_logvar=-3 又让后验一开始就远离先验，光这一项
    # 每维就约 1.02 nats，乘 K*d 维已是数千 nats。真正的 P0 读数要在训练后取。
    #
    # 另外结构上界的绝对值取决于 LaME 未给出的 SNR（取 1 得 4259，取 100 得 28000），
    # 所以 P0 的判据不是"率低于上界"，而是"率/上界的比值在 K 上近似不变"——上界对 K
    # 线性，SNR 在比值里被约掉。见 plan 的 P0。
    bound = model.latent.structural_bound_nats()
    print(f"        [报数] 未训练时 rate={float(rate):.0f} nats，"
          f"SNR=1 的结构上界={bound:.0f} nats，比值={float(rate)/bound:.2f}")

    print("\n=== 梯度能否穿过 hook ===")
    model.zero_grad()
    reps = model.encode_input(batch)
    loss = reps.sum() + 1e-3 * model.pop_rate_loss()
    loss.backward()
    g_embed = model.latent.reason_embed.grad
    check("梯度到达 reason_embed", g_embed is not None and g_embed.abs().sum() > 0,
          f"grad 范数 {g_embed.norm():.3e}" if g_embed is not None else "grad 为 None")
    g_mu = model.latent.bottleneck.to_mu.weight.grad
    check("梯度到达 mu 头", g_mu is not None and g_mu.abs().sum() > 0,
          f"grad 范数 {g_mu.norm():.3e}" if g_mu is not None else "grad 为 None")
    g_lv = model.latent.bottleneck.to_logvar.weight.grad
    check("梯度到达 logvar 头", g_lv is not None and g_lv.abs().sum() > 0,
          f"grad 范数 {g_lv.norm():.3e}" if g_lv is not None else "grad 为 None")

    print("\n=== reason token 确实参与计算 ===")
    # 把 reason_embed 换成另一组值，输出必须变化。若不变，说明 hook 没生效或
    # 读出位置取错了（例如仍在读 EOS），这是最隐蔽的一类接线错误。
    model.eval()
    with torch.no_grad():
        base = model.encode_input(batch).clone()
        saved = model.latent.reason_embed.data.clone()
        model.latent.reason_embed.data.normal_(0, 0.5)
        perturbed = model.encode_input(batch)
        delta = (base - perturbed).abs().max()
        model.latent.reason_embed.data.copy_(saved)
    check("改动 reason_embed 会改变输出", float(delta) > 1e-4, f"最大变化 {float(delta):.4e}")

    print("\n=== 训练采样 vs 推理取均值 ===")
    model.eval()
    with torch.no_grad():
        a, b = model.encode_input(batch), model.encode_input(batch)
    check("推理两次结果一致（取均值不采样）", torch.allclose(a, b, atol=1e-5),
          f"最大差 {float((a-b).abs().max()):.2e}")
    model.train()
    with torch.no_grad():
        c, d = model.encode_input(batch), model.encode_input(batch)
    check("训练两次结果不同（确实在采样）", not torch.allclose(c, d, atol=1e-5),
          f"最大差 {float((c-d).abs().max()):.2e}")

    print("\n=== no_grad 下不累积率 ===")
    model.pop_rate_loss()
    with torch.no_grad():
        model.encode_input(batch)
    check("no_grad 前向不污染累积器（否则等效 beta 被放大）",
          model.pop_rate_loss() is None)

    print("\n=== 评测路径：整模型转 bf16 ===")
    # eval.py 会做 model.to(device, dtype=torch.bfloat16)，把瓶颈头也一起转成 bf16。
    # 若 readout 里硬转 fp32，就会因为输入与权重 dtype 不一致直接抛错——训练时不触发，
    # 只在评测时炸，是最容易漏掉的一类。
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    try:
        with torch.no_grad():
            reps_bf16 = model.encode_input(batch)
        ok = torch.isfinite(reps_bf16).all() and reps_bf16.shape == (2, model.latent.bottleneck.latent_size)
        check("bf16 下前向正常", bool(ok), f"dtype={reps_bf16.dtype}")
    except Exception as e:
        check("bf16 下前向正常", False, f"{type(e).__name__}: {e}")

    print(f"\n通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项:", ", ".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
