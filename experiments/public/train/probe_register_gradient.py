"""测检索寄存器在真实训练路径里到底有没有被优化，以及走了多远。

为什么需要单独测
----------------
落盘的 register 权重统计量与初始化几乎一致（std 0.0200 对 0.0200，各行 L2 norm 均值
0.7838 对理论 0.784），但这既可能是"根本没进优化器"，也可能是"进了优化器但梯度方向
不一致、94 步下等效随机游走"。两者对结论的影响完全不同：前者是实现缺陷，后者说明
register 的取值本身不承重、机制退化成"在一个追加位置上做池化"。

而这个区分又直接决定"LoRA 而非全参训练是否是差异来源"这个问题怎么答：若 register
拿不到梯度，那它与 LoRA 秩无关；若拿得到但没动，说明可训练性不是瓶颈。

做法：包一层 training_step，在每步优化器更新之后读三个量 ——
  grad_norm       该步 register 的梯度范数。恒为 None/0 即没有梯度通路。
  |Δ| since init  相对第 0 步之前快照的累计位移范数。
  相对位移        |Δ| / |init|，这是唯一可跨维度解读的量。

用法:
    cd /home/zhoutuowen/VLM2Vec
    bash experiments/public/train/probe_register_gradient.sh
"""
import sys

import torch

import src.trainer as trainer_mod


def _install_probe():
    trainer_cls = trainer_mod.GradCacheLateProcessTrainer
    original = trainer_cls.training_step
    state = {'init': None}

    def probed(self, model, inputs, *args, **kwargs):
        inner = model.module if hasattr(model, 'module') else model
        reg = getattr(getattr(inner, 'reloop', None), 'register_embed', None)
        if reg is None:
            return original(self, model, inputs, *args, **kwargs)

        if state['init'] is None:
            state['init'] = reg.detach().float().clone()
            print(f"[REGPROBE] init: norm={state['init'].norm():.6f} "
                  f"std={state['init'].std():.6f} requires_grad={reg.requires_grad} "
                  f"in_optimizer={_in_optimizer(self, reg)}", flush=True)

        loss = original(self, model, inputs, *args, **kwargs)

        g = reg.grad
        cur = reg.detach().float()
        delta = (cur - state['init']).norm()
        print(f"[REGPROBE] step={self.state.global_step} "
              f"grad_norm={'None' if g is None else f'{g.detach().float().norm():.3e}'} "
              f"|delta|={delta:.3e} rel={delta / state['init'].norm():.3e} "
              f"norm={cur.norm():.6f}", flush=True)
        return loss

    trainer_cls.training_step = probed


def _in_optimizer(trainer, param):
    """register 是否真的落在某个 param group 里。没落进去 = 永远不会被更新。"""
    opt = getattr(trainer, 'optimizer', None)
    if opt is None:
        return 'optimizer-not-built-yet'
    return any(param is p for group in opt.param_groups for p in group['params'])


if __name__ == '__main__':
    _install_probe()
    import train
    sys.exit(train.main())
