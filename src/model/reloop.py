"""ReLoop 式的循环深度与检索寄存器。

复现目标
--------
ReLoop-UME 的主张是：把 UME 的推理预算做在**深度**上而不是 token 工作区上——复用一个
参数共享的中段层块 T 次，再配一小组可学习的检索寄存器作为固定大小的证据工作区。
论文未开源，本模块是按其描述在 VLM2Vec 上的最小复现，用来回答一个前置问题：这个机制
在我们自己的 pipeline 与规模下到底有没有增益。

两个轴是独立开关，因此对照组与实验组共用一条代码路径：

    T=1, M=0  ->  逐元素等价于原判别式基线
                  （断言见 experiments/public/train/verify_reloop_identity.py）
    T=4, M=5  ->  完整机制

之所以坚持"退化到基线"而不是维护两份代码，是因为 D-A 这个差值是本次实验唯一的观测量。
若对照组走另一条代码路径，任何差值都存在"两边实现不一致"的解释空间，实验就白跑了。

关于 register 的读出位置
------------------------
register 追加在序列末尾，而 Qwen2-VL 在 build() 里设了 padding_side="left"，因此末尾
M 个位置对所有样本都对齐，MMEBModel._pooling 取的位置 -1 恰好是最后一个 register。
M=0 时位置 -1 退化成原来的 eos token，两种配置的读出规则是同一条，不需要分支。

备选读出是对 M 个 register 取均值。没有默认用它，是因为均值会让"读出宽度"随 M 变化，
M 的消融就同时改了两件事；取末位则只改工作区容量。reloop_readout=mean 可以切换。
"""
import torch
from torch import nn


class RetrievalRegisters(nn.Module):
    """M 个可学习的检索寄存器，追加在输入序列末尾。

    注入方式与 LatentReasoner 相同：先在序列末尾补 M 个占位 token，再用 embedding 层的
    forward hook 把这 M 个位置的嵌入替换成可学习向量。不直接传 inputs_embeds 的原因是
    Qwen2-VL 的 forward 要靠 input_ids 定位图像占位符再把视觉特征 masked_scatter 进去，
    绕过 input_ids 视觉分支就失效了。

    占位 token 的 id 不影响结果（嵌入整个被替换），但必须是普通文本 token：若误用
    image_pad/video_pad 的 id，get_rope_index 会把它当视觉 token 去算 M-RoPE，位置编码
    会错乱。
    """

    def __init__(self, hidden_size: int, num_registers: int, placeholder_token_id: int = 0):
        super().__init__()
        if num_registers <= 0:
            raise ValueError(f"num_registers 必须为正，收到 {num_registers}；M=0 应当不构造本模块")
        self.num_registers = num_registers
        self.placeholder_token_id = placeholder_token_id
        self.register_embed = nn.Parameter(torch.empty(num_registers, hidden_size))
        # std 0.02 与 Qwen2 的 initializer_range 一致。register 的嵌入要和真实 token 嵌入
        # 处于同一尺度，否则前几层的注意力会被这几个异常大/小的位置主导。
        nn.init.normal_(self.register_embed, std=0.02)

    def extend_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        b = input_ids.shape[0]
        ph = torch.full((b, self.num_registers), self.placeholder_token_id,
                        dtype=input_ids.dtype, device=input_ids.device)
        keep = torch.ones((b, self.num_registers),
                          dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([input_ids, ph], dim=1), torch.cat([attention_mask, keep], dim=1)

    def inject(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """替换末尾 M 个位置的嵌入。返回新张量而非原地改，以免破坏 autograd 图。"""
        out = inputs_embeds.clone()
        out[:, -self.num_registers:, :] = self.register_embed.to(out.dtype).unsqueeze(0).expand(
            out.shape[0], -1, -1)
        return out


class RecurrenceSchedule:
    """解码器层的执行顺序：prefix 一次、loop 块 T 次、suffix 一次。

    只描述"按什么顺序调用哪些层"，不持有任何参数——参数共享正是通过重复调用同一批
    ModuleList 元素实现的，这与 ReLoop 的做法一致，也是循环深度不增加参数量的原因。

    Args:
        num_layers: 解码器总层数。
        loop_start: loop 块的起始层下标（含）。
        loop_end: loop 块的结束层下标（不含）。
        num_loops: T，loop 块重复次数。T=1 时调度退化为原始顺序。
    """

    def __init__(self, num_layers: int, loop_start: int, loop_end: int, num_loops: int):
        if not 0 <= loop_start < loop_end <= num_layers:
            raise ValueError(
                f"loop 区间非法：[{loop_start}, {loop_end}) 不在 [0, {num_layers}] 内，"
                "且要求 loop_start < loop_end")
        if num_loops < 1:
            raise ValueError(f"num_loops 必须 >= 1，收到 {num_loops}")
        self.num_layers = num_layers
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.num_loops = num_loops

    @property
    def indices(self):
        """展开后的层下标序列。T=1 时恰为 range(num_layers)。"""
        prefix = list(range(0, self.loop_start))
        block = list(range(self.loop_start, self.loop_end))
        suffix = list(range(self.loop_end, self.num_layers))
        return prefix + block * self.num_loops + suffix

    @property
    def num_applications(self) -> int:
        return len(self.indices)

    def is_identity(self) -> bool:
        """是否与不加循环的原始顺序完全一致。"""
        return self.indices == list(range(self.num_layers))

    def __repr__(self):
        return (f"RecurrenceSchedule(prefix=[0,{self.loop_start}), "
                f"loop=[{self.loop_start},{self.loop_end}) x {self.num_loops}, "
                f"suffix=[{self.loop_end},{self.num_layers}), "
                f"applications={self.num_applications})")


def attach_recurrence(encoder, loop_start: int, loop_end: int, num_loops: int):
    """把循环调度挂到 backbone 的解码器上，返回挂上去的 schedule。

    encoder 可能被 PEFT 包了一层，也可能是 Qwen2VLForConditionalGeneration，所以这里
    向下找到那个持有 `layers` ModuleList 的模块，而不是假定某个固定路径。
    """
    decoder = _find_decoder(encoder)
    schedule = RecurrenceSchedule(
        num_layers=len(decoder.layers),
        loop_start=loop_start,
        loop_end=loop_end,
        num_loops=num_loops,
    )
    decoder.recurrence = schedule
    return schedule


def _find_decoder(module):
    """找到持有 decoder layers 的那个模块。"""
    for m in module.modules():
        layers = getattr(m, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0 and hasattr(m, "embed_tokens"):
            return m
    raise AttributeError(
        f"在 {type(module).__name__} 下找不到带 layers + embed_tokens 的解码器模块，"
        "无法接入循环深度")
