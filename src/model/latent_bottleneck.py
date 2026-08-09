"""随机化的隐式推理瓶颈与变分率项。

立论
----
LaME 把"面向嵌入的隐式推理"表述为信息瓶颈，但它在 §3.2 与附录 A 两处明确声明不使用
率正则，只拿 reason token 的个数 K 当硬容量上界:

    I(Z;X) <= K * d * log(1 + SNR)

这个上界在实践中是空的。Qwen2-VL-2B 的 d=1536，K=1 时右端已达数千 nats，远高于任何
真实的 I(Z;X)。关键在于它在 K 从 1 到 32 的**整个区间里都是松弛的**，因此压缩程度并不
随 K 改变，K 实际控制的是参数量、读出端宽度与优化难度，而不是信息量。

需要小心一个看似顺理成章但错误的反驳："放松约束不该让性能变差，所以约束没生效"。这条
不成立 —— IB 里压缩本身就是正则化，容量放松导致泛化变差正是预期行为，倒 U 曲线与
"容量约束在起作用"是相容的，不能拿它当反证。站得住的只有上面那条量级论证，而它是可测的
（见 P0：实测率与结构上界的比值）。

三条佐证：

  1. LaME 与 BToks 在架构差异极大的两套实现上各自测出同一条倒 U（前者 K=8 最优、K=16
     退化，后者 K=4 最优 58.96、单调滑到 K=32 的 57.99），且解释几乎逐字相同，都只是
     定性断言，没有任何对实际信息量的测量；
  2. 整套机制净增益只有 +0.6（判别式基线 68.5 -> 完整版 69.3）；
  3. 把解码目标换成完整 CoT 后掉回 63.8，与不要解码头完全相同 —— 瓶颈的行为由监督
     信号的信息密度决定，而不由容量决定。

本模块补上缺失的那一项：把确定性的 reason token 隐状态变成随机隐变量，目标函数里加上
真正的变分率项 beta * KL(q(z|x) || p(z))。这样"每个样本该用多少容量"成为率失真权衡的
连续可微副产物，而不是一个需要单独优化的离散路由决策 —— 后者正是 TTE-Flash 把 think
budget 做成离散自适应决策后反而掉分（68.3 -> 66）的原因，作者自述 "adaptive think
introduces a harder optimization problem"。

推理时取后验均值，不采样，因此不增加任何推理开销，保留 LaME 的单次前向与吞吐优势。
"""
import math

import torch
from torch import nn


class StochasticBottleneck(nn.Module):
    """把 reason token 的隐状态映射为高斯后验并采样，同时给出每样本的率。

    Args:
        hidden_size: backbone 隐藏维度 d。
        latent_size: 隐变量维度。默认与 hidden_size 相同；调小可以在不改 K 的前提下
            单独收紧容量，是 K 之外的第二个消融轴。
        free_bits: 每维 KL 的下限（nats）。早期 beta 还小的时候后验容易整体塌到先验上
            （KL -> 0，z 与输入无关），free bits 让每一维在降到该阈值之后就不再被率项
            继续压，从而保住一部分可用容量。0 表示关闭。
        init_logvar: logvar 头输出的初始偏置。取负值让初始后验方差偏小，训练早期更接近
            确定性瓶颈，避免噪声淹没还没学起来的表示。代价是初始率随之抬高：仅这一项
            每维就贡献 0.5*(exp(v) - 1 - v) nats，v=-3 时约 1.02，v=-1 时约 0.18。
            默认取 -1 是两者的折中——既保留可回传的非零梯度（v=0 恰是率项的驻点），
            又不让初始率把 beta 的标定空间挤掉。
    """

    def __init__(
        self,
        hidden_size: int,
        latent_size: int = None,
        free_bits: float = 0.0,
        init_logvar: float = -1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size or hidden_size
        self.free_bits = free_bits

        # 先归一化再进头部。Qwen2-VL 末层隐状态的范数在 200 量级，默认初始化的 Linear
        # 直接作用其上会让 |mu| 一开始就有 3 左右，初始率高达每维数 nats（K=8 时约
        # 7 万 nats），率项在训练最开始就压过对比损失，beta 完全没法按"占目标函数多少
        # 比例"来标定。归一化把输入尺度与 backbone 解耦，初始率降到千 nats 量级。
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.to_mu = nn.Linear(hidden_size, self.latent_size)
        self.to_logvar = nn.Linear(hidden_size, self.latent_size)

        # mu 头按常规初始化；logvar 头置零权重加负偏置，使初始 sigma = exp(init_logvar/2)
        # 对所有样本一致且较小。若让 logvar 随机初始化，不同维度的噪声尺度差异会在
        # 训练最开始就把某些维压死，形成不可逆的维度塌缩。
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, init_logvar)

    def forward(self, h_reason: torch.Tensor, deterministic: bool = None):
        """
        Args:
            h_reason: (B, K, d) reason token 的最后一层隐状态。
            deterministic: 取 True 时直接返回均值。默认跟随 self.training，
                即训练采样、推理取均值。

        Returns:
            z:  (B, K, latent_size) 隐变量。
            kl: (B,) 每个样本的率，单位 nats。已按 free bits 截断。
            stats: 诊断用的标量字典，不参与反向。
        """
        if deterministic is None:
            deterministic = not self.training

        # 对齐到本模块参数的 dtype。训练时参数是 fp32、隐状态是 bf16，走的是升精度；
        # 评测脚本会对整个模型做 .to(dtype=bfloat16)，那时参数变 bf16，若这里仍硬转 fp32
        # 就会因为输入与权重 dtype 不一致直接抛错。
        h_reason = h_reason.to(self.to_mu.weight.dtype)
        h_reason = self.pre_norm(h_reason)
        mu = self.to_mu(h_reason)
        # 不夹住 logvar 的话，exp() 在 bf16 下很容易溢出成 inf，KL 直接变 nan
        logvar = self.to_logvar(h_reason).clamp(min=-10.0, max=10.0)

        if deterministic:
            z = mu
        else:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        # 与标准正态先验的逐维 KL: 0.5 * (mu^2 + sigma^2 - 1 - log sigma^2)
        # 无论参数是什么精度，率一律在 fp32 里累加：要对 K*d 个维度求和（K=8 时上万项），
        # bf16 只有 8 位尾数，逐项累加的舍入误差会把率算偏一个可观的比例。
        mu32, logvar32 = mu.float(), logvar.float()
        kl_per_dim = 0.5 * (mu32.pow(2) + logvar32.exp() - 1.0 - logvar32)

        # free bits 只改梯度不改度量：截断后的值用于损失，未截断的值用于上报实测率，
        # 否则 P2 的"率-难度相关性"会被下限抹平。
        raw_kl = kl_per_dim.sum(dim=(-2, -1))
        if self.free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl = kl_per_dim.sum(dim=(-2, -1))

        stats = {
            "rate_nats": raw_kl.detach().mean(),
            "rate_nats_clamped": kl.detach().mean(),
            "posterior_sigma": torch.exp(0.5 * logvar).detach().mean(),
            "mu_abs": mu.detach().abs().mean(),
        }
        return z, kl, stats


class BetaScheduler:
    """率项系数 beta 的退火。

    直接用固定的 beta 开训会让后验在表示还没学起来时就塌到先验（KL -> 0，z 变成常量），
    这是 VIB 的经典失败模式。先给一段 beta=0 的自由期让对比目标把表示拉开，再线性升到
    目标值，配合 free bits 基本可以避免。
    """

    def __init__(self, beta: float, warmup_steps: int = 0, delay_steps: int = 0):
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.delay_steps = delay_steps

    def value(self, step: int) -> float:
        if self.beta == 0.0:
            return 0.0
        if step < self.delay_steps:
            return 0.0
        if self.warmup_steps <= 0:
            return self.beta
        progress = (step - self.delay_steps) / self.warmup_steps
        return self.beta * min(1.0, max(0.0, progress))


class LatentReasoner(nn.Module):
    """K 个可学习的 reason token，加上它们隐状态之上的随机瓶颈。

    注入方式：在输入序列末尾追加 K 个占位 token，再用 embedding 层的 forward hook 把这
    K 个位置的嵌入替换成可学习向量。之所以不直接传 inputs_embeds，是因为 Qwen2-VL 的
    forward 要靠 input_ids 定位图像占位符再把视觉特征 masked_scatter 进去，绕过 input_ids
    会让视觉分支失效。

    之所以能固定取末尾 K 个位置，是因为 Qwen2-VL 在 build() 里设了 padding_side="left"：
    左填充下每条样本的真实内容都靠右对齐，追加在末尾的 reason token 对所有样本都落在
    同一组下标上，不需要按 attention_mask 逐样本算偏移。若哪天改成右填充，这里必须改成
    按每条样本的有效长度定位，否则 reason token 会落进 padding 区。

    占位 token 的 id 本身不影响结果（嵌入会被整个替换掉），但必须是普通文本 token：
    若误用 image_pad/video_pad 的 id，get_rope_index 会把它当成视觉 token 去算 M-RoPE，
    位置编码会错乱。
    """

    def __init__(
        self,
        hidden_size: int,
        num_tokens: int,
        latent_size: int = None,
        free_bits: float = 0.0,
        init_logvar: float = -3.0,
        placeholder_token_id: int = 0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.placeholder_token_id = placeholder_token_id
        self.reason_embed = nn.Parameter(torch.empty(num_tokens, hidden_size))
        nn.init.normal_(self.reason_embed, std=0.02)
        self.bottleneck = StochasticBottleneck(hidden_size, latent_size, free_bits, init_logvar)

    def extend_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        b = input_ids.shape[0]
        ph = torch.full((b, self.num_tokens), self.placeholder_token_id,
                        dtype=input_ids.dtype, device=input_ids.device)
        keep = torch.ones((b, self.num_tokens),
                          dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([input_ids, ph], dim=1), torch.cat([attention_mask, keep], dim=1)

    def inject(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """替换末尾 K 个位置的嵌入。返回新张量而非原地改，以免破坏 autograd 图。"""
        out = inputs_embeds.clone()
        out[:, -self.num_tokens:, :] = self.reason_embed.to(out.dtype).unsqueeze(0).expand(
            out.shape[0], -1, -1)
        return out

    def readout(self, last_hidden_state: torch.Tensor, deterministic: bool = None):
        # KL 里有 exp 与 log，bf16 下精度不够且容易溢出，这里统一升到 fp32 再算
        h = last_hidden_state[:, -self.num_tokens:, :].float()
        return self.bottleneck(h, deterministic=deterministic)

    def structural_bound_nats(self, snr: float = 1.0) -> float:
        """LaME 声称的结构上界 K * d * log(1 + SNR)，用于 P0 里与实测率作比。

        SNR 论文未给具体取值，这里默认 1.0 取其最保守的一档；即便如此，K=1、d=1536 时
        右端也已是 1064 nats 量级，而实测率通常在个位到几十 nats。
        """
        return self.num_tokens * self.bottleneck.latent_size * math.log(1.0 + snr)


def gaussian_kl_reference(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """解析 KL，仅供单元测试与蒙特卡洛估计对拍。"""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=(-2, -1))


def monte_carlo_kl(mu: torch.Tensor, logvar: torch.Tensor, num_samples: int = 200000) -> torch.Tensor:
    """用采样估计 KL(q||p)，与解析式对拍以确认公式没写错。"""
    std = torch.exp(0.5 * logvar)
    total = torch.zeros(mu.shape[0], device=mu.device, dtype=torch.float64)
    for _ in range(num_samples):
        eps = torch.randn_like(mu)
        z = mu + std * eps
        log_q = (-0.5 * eps.pow(2) - 0.5 * math.log(2 * math.pi) - 0.5 * logvar).sum(dim=(-2, -1))
        log_p = (-0.5 * z.pow(2) - 0.5 * math.log(2 * math.pi)).sum(dim=(-2, -1))
        total += (log_q - log_p).double()
    return total / num_samples
