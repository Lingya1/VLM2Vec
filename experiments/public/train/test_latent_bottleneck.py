"""随机瓶颈与率项的单元测试（纯 CPU，不需要 GPU 与真实数据）。

跑法:
    cd /home/zhoutuowen/VLM2Vec
    PYTHONNOUSERSITE=1 PYTHONPATH=. python experiments/public/train/test_latent_bottleneck.py
"""
import torch

from src.model.latent_bottleneck import (
    BetaScheduler,
    StochasticBottleneck,
    gaussian_kl_reference,
    monte_carlo_kl,
)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def test_kl_formula():
    """解析 KL 必须和蒙特卡洛估计一致，否则率项本身就是错的。"""
    torch.manual_seed(0)
    mu = torch.randn(3, 2, 4) * 0.7
    logvar = torch.randn(3, 2, 4).clamp(-1.5, 1.5)

    analytic = gaussian_kl_reference(mu, logvar)
    mc = monte_carlo_kl(mu, logvar, num_samples=20000)
    rel = ((analytic.double() - mc).abs() / analytic.double().abs()).max().item()
    check("解析 KL 与蒙特卡洛一致", rel < 0.02, f"最大相对误差 {rel:.4f}")

    # 后验等于先验时率必须恰好为 0
    zero = gaussian_kl_reference(torch.zeros(2, 2, 4), torch.zeros(2, 2, 4))
    check("后验=先验时率为 0", torch.allclose(zero, torch.zeros(2), atol=1e-6),
          f"得到 {zero.tolist()}")


def test_reparameterization():
    """训练时必须采样且可导，推理时必须确定。"""
    torch.manual_seed(0)
    # 用默认的 init_logvar=-3。注意 d(KL)/d(logvar) = 0.5*(exp(logvar)-1)，在 logvar=0
    # 处恰好为零，所以若用 init_logvar=0 建这个用例，logvar 头的梯度会是 0 —— 那是驻点
    # 而非缺陷。默认的 -3 给出 0.5*(e^-3 - 1) ≈ -0.475，是把 sigma 往先验方向推的力。
    bn = StochasticBottleneck(hidden_size=8, latent_size=6)
    h = torch.randn(4, 3, 8, requires_grad=True)

    bn.train()
    z1, kl, _ = bn(h)
    z2, _, _ = bn(h)
    check("训练时两次前向的 z 不同（确实在采样）", not torch.allclose(z1, z2))
    check("率对每个样本各一个标量", kl.shape == (4,), f"shape={tuple(kl.shape)}")

    kl.sum().backward()
    check("率项可回传到输入", h.grad is not None and h.grad.abs().sum() > 0)
    check("率项可回传到 logvar 头",
          bn.to_logvar.weight.grad is not None and bn.to_logvar.weight.grad.abs().sum() > 0,
          f"grad 范数 {bn.to_logvar.weight.grad.abs().sum().item():.3e}")

    # 把上面那条注释里的性质直接测出来，避免以后有人把 init_logvar 改成 0 后困惑于
    # "logvar 学不动"
    bn0 = StochasticBottleneck(hidden_size=8, latent_size=6, init_logvar=0.0)
    bn0.train()
    _, kl0, _ = bn0(torch.randn(4, 3, 8))
    kl0.sum().backward()
    check("logvar=0 是率项的驻点（sigma=1 即先验方差，率不再推动它）",
          bn0.to_logvar.weight.grad.abs().sum().item() < 1e-6)

    bn.eval()
    z3, _, _ = bn(h)
    z4, _, _ = bn(h)
    check("推理时 z 确定（取均值，不增加推理开销）", torch.allclose(z3, z4))
    # 参考值要跟着走一遍 pre_norm：头部前有 LayerNorm 把输入尺度与 backbone 解耦
    check("推理时的 z 就是 mu", torch.allclose(z3, bn.to_mu(bn.pre_norm(h))))


def test_free_bits():
    """free bits 应当只托住损失里的率，不污染上报的实测率。"""
    torch.manual_seed(0)
    lam = 0.05
    bn = StochasticBottleneck(hidden_size=8, latent_size=6, free_bits=lam, init_logvar=0.0)
    bn.train()
    # 让后验非常接近先验：mu 头置零、logvar 头置零 -> 每维 KL = 0
    torch.nn.init.zeros_(bn.to_mu.weight)
    torch.nn.init.zeros_(bn.to_mu.bias)
    torch.nn.init.zeros_(bn.to_logvar.bias)

    h = torch.randn(5, 3, 8)
    _, kl, stats = bn(h)
    expected_floor = lam * 3 * 6  # K x latent_size 维，每维托底 lam
    check("率被 free bits 托住不再降为 0",
          torch.allclose(kl, torch.full((5,), expected_floor), atol=1e-5),
          f"得到 {kl[0].item():.4f}，期望 {expected_floor:.4f}")
    check("上报的实测率不含 free bits 托底（P2 分析需要真实值）",
          stats["rate_nats"].item() < 1e-5,
          f"rate_nats={stats['rate_nats'].item():.2e}")


def test_beta_schedule():
    s = BetaScheduler(beta=1e-3, warmup_steps=100, delay_steps=50)
    vals = [s.value(t) for t in (0, 49, 50, 100, 150, 200, 1000)]
    check("beta 在自由期内恒为 0", vals[0] == 0.0 and vals[1] == 0.0)
    check("beta 线性升到目标值", abs(vals[3] - 5e-4) < 1e-9, f"step100 -> {vals[3]:.2e}")
    check("beta 升满后保持", vals[4] == 1e-3 and vals[6] == 1e-3)
    check("beta=0 时全程为 0", all(BetaScheduler(0.0, 100).value(t) == 0.0 for t in (0, 500)))


def test_capacity_monotonicity():
    """核心预测 P1 的机制自检。

    率项存在时，加大 K 不应自动带来更大的实测率 —— 率由 beta 与任务需求共同决定，
    而不是由容量决定。这里用一个合成的率失真优化验证该机制：固定 beta 下最优率与 K
    无关（多出来的维度会被压到先验上），而 beta=0 时率随 K 线性发散。
    """
    results = {}
    for K in (2, 4, 8, 16):
        for beta in (0.0, 0.1):
            torch.manual_seed(0)
            bn = StochasticBottleneck(hidden_size=8, latent_size=4, init_logvar=0.0)
            bn.train()
            h = torch.randn(64, K, 8)
            target = torch.randn(64, 3)
            head = torch.nn.Linear(K * 4, 3)
            opt = torch.optim.Adam(list(bn.parameters()) + list(head.parameters()), lr=0.05)
            for _ in range(300):
                z, kl, _ = bn(h)
                pred = head(z.reshape(z.shape[0], -1))
                loss = torch.nn.functional.mse_loss(pred, target) + beta * kl.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                _, kl, _ = bn(h)
            results[(K, beta)] = kl.mean().item()

    rates_b0 = [results[(K, 0.0)] for K in (2, 4, 8, 16)]
    rates_b1 = [results[(K, 0.1)] for K in (2, 4, 8, 16)]
    print(f"      beta=0   各 K 的实测率: {[f'{r:.1f}' for r in rates_b0]}")
    print(f"      beta=0.1 各 K 的实测率: {[f'{r:.1f}' for r in rates_b1]}")

    growth_b0 = rates_b0[-1] / max(rates_b0[0], 1e-6)
    growth_b1 = rates_b1[-1] / max(rates_b1[0], 1e-6)
    check("无率项时容量随 K 发散", growth_b0 > 2.0, f"K=16/K=2 的率之比 {growth_b0:.1f}x")
    check("有率项时容量不随 K 发散", growth_b1 < growth_b0 / 2,
          f"K=16/K=2 的率之比 {growth_b1:.1f}x（对照 {growth_b0:.1f}x）")


def main():
    for name, fn in [
        ("KL 公式正确性", test_kl_formula),
        ("重参数化与推理确定性", test_reparameterization),
        ("free bits", test_free_bits),
        ("beta 退火", test_beta_schedule),
        ("容量-率解耦（P1 机制自检）", test_capacity_monotonicity),
    ]:
        print(f"\n=== {name} ===")
        fn()

    print(f"\n通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项: " + ", ".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
