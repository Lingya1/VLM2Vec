"""验证 ReLoop 改动在 T=1 / M=0 时与基线逐元素等价，以及机制开启后确实生效。

为什么这个验证是必须的
----------------------
本次实验唯一的观测量是 D-A 这个差值（D 是 T=4/M=5，A 是 T=1/M=0）。如果 A 这一格因为
改动引入了任何偏移，差值就同时包含了"机制的效果"与"实现的副作用"，实验结论不成立。
所以这里要的不是"数值接近"，而是逐元素相同。

用小 config 而不是真实的 2B 权重：这几条断言检的是层调度与注入的机械正确性，与权重无关，
小模型能在 CPU 上秒级跑完，可以在每次改动后无成本地重跑。真实权重下的等价性由第 1 条
断言的传递性保证——调度产出的是同一批模块对象、同一个顺序。

用法:
    cd /home/zhoutuowen/VLM2Vec
    PYTHONNOUSERSITE=1 PYTHONPATH=. python experiments/public/train/verify_reloop_identity.py
"""
import sys

import torch

from src.model.reloop import RecurrenceSchedule, RetrievalRegisters
from src.model.vlm_backbone.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from src.model.vlm_backbone.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

PASS, FAIL = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m"
failures = []


def check(name, ok, detail=""):
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def build_tiny_model(num_layers=6):
    config = Qwen2VLConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        # mrope_section 必须求和为 head_dim/2；这里 head_dim = 64/4 = 16，故三段和为 8。
        # config 会把 type=mrope 规范化成 default（mrope 用的就是默认 RoPE 计算）。
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    torch.manual_seed(0)
    model = Qwen2VLModel(config)
    model.eval()
    return model, config


def forward(model, input_ids, attention_mask):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, return_dict=True)
    return out.last_hidden_state


def main():
    # --- 纯逻辑：调度展开是否正确 -------------------------------------------------
    s1 = RecurrenceSchedule(num_layers=28, loop_start=17, loop_end=27, num_loops=1)
    check("T=1 的调度等于原始逐层顺序",
          s1.indices == list(range(28)) and s1.is_identity(),
          f"applications={s1.num_applications}")

    s4 = RecurrenceSchedule(num_layers=28, loop_start=17, loop_end=27, num_loops=4)
    # prefix 17 层 + loop 10 层 x 4 + suffix 1 层
    check("T=4 的调度展开为 58 次层调用",
          s4.num_applications == 58 and not s4.is_identity(),
          f"applications={s4.num_applications}")
    check("T=4 的调度里 loop 块按整块重复而非逐层重复",
          s4.indices[17:27] == list(range(17, 27)) and s4.indices[27:37] == list(range(17, 27)))

    for bad in [dict(loop_start=-1, loop_end=5), dict(loop_start=5, loop_end=5),
                dict(loop_start=0, loop_end=29)]:
        try:
            RecurrenceSchedule(num_layers=28, num_loops=2, **bad)
            check(f"非法 loop 区间 {bad} 应当报错", False)
        except ValueError:
            check(f"非法 loop 区间 {bad} 被拦下", True)

    # --- 前向：identity 调度必须逐元素等于不挂调度 --------------------------------
    model, config = build_tiny_model(num_layers=6)
    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    # 左填充，与 Qwen2-VL 在 build() 里设的 padding_side 一致
    attention_mask[0, :3] = 0

    baseline = forward(model, input_ids, attention_mask)

    model.recurrence = RecurrenceSchedule(num_layers=6, loop_start=3, loop_end=5, num_loops=1)
    identity = forward(model, input_ids, attention_mask)
    check("挂上 T=1 调度后前向逐元素等于基线",
          torch.equal(baseline, identity),
          f"max|diff|={(baseline - identity).abs().max().item():.3e}")

    model.recurrence = RecurrenceSchedule(num_layers=6, loop_start=3, loop_end=5, num_loops=3)
    looped = forward(model, input_ids, attention_mask)
    check("T=3 确实改变了输出（机制真的接上了）",
          not torch.allclose(baseline, looped, atol=1e-5),
          f"max|diff|={(baseline - looped).abs().max().item():.3e}")

    # --- KV cache 必须被拦死 ------------------------------------------------------
    # DynamicCache 按固定 layer_idx 索引，循环同一层会往同一槽位反复 concat KV：
    # 不报错，但注意力会读到自己上一圈的旧状态，是最难发现的一类错误。
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        check("T>1 且 use_cache=True 应当报错", False)
    except ValueError:
        check("T>1 且 use_cache=True 被拦下", True)

    model.recurrence = None
    restored = forward(model, input_ids, attention_mask)
    check("撤掉调度后恢复到基线输出", torch.equal(baseline, restored))

    # --- 检索寄存器 --------------------------------------------------------------
    regs = RetrievalRegisters(hidden_size=64, num_registers=5)
    ext_ids, ext_mask = regs.extend_inputs(input_ids, attention_mask)
    check("extend_inputs 在末尾追加 M 个位置",
          ext_ids.shape == (2, 17) and ext_mask.shape == (2, 17),
          f"{tuple(input_ids.shape)} -> {tuple(ext_ids.shape)}")
    check("追加的位置全部可见（mask=1）", bool(ext_mask[:, -5:].all()))
    check("原有 token 与 mask 未被改动",
          torch.equal(ext_ids[:, :12], input_ids) and torch.equal(ext_mask[:, :12], attention_mask))

    embeds = torch.zeros(2, 17, 64)
    injected = regs.inject(embeds)
    check("inject 只改写末尾 M 个位置",
          torch.equal(injected[:, :12], embeds[:, :12]) and bool((injected[:, -5:] != 0).any()))
    check("inject 对每个样本写入同一组 register",
          torch.equal(injected[0, -5:], injected[1, -5:]))
    check("inject 不原地修改输入（保住 autograd 图）", torch.equal(embeds, torch.zeros(2, 17, 64)))
    check("register 参与梯度", regs.register_embed.requires_grad)

    # 占位 token id 若落在视觉 token 上，get_rope_index 会按视觉分支算 M-RoPE，位置编码错乱
    check("占位 token 默认不是视觉 token id",
          regs.placeholder_token_id not in {151652, 151653, 151654, 151655, 151656},
          f"placeholder_token_id={regs.placeholder_token_id}")

    print()
    if failures:
        print(f"{len(failures)} 条断言未通过：" + "，".join(failures))
        return 1
    print("全部通过。T=1/M=0 与基线逐元素等价，机制开启后生效，cache 冲突已拦。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
