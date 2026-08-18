"""汇总 OK-VQA 三格筛查的结果：loss 轨迹、hit@1、以及 register 是否真的学到东西。

三个读数各自回答一件事：
  |A42 - A43|      这套 pipeline 的跑间噪声。D42-A42 必须明显超过它才算信号。
  D42 - A42        机制的效应。
  register 范数     只作参考，不能当"有没有学"的判据：范数对旋转不变，参数整体转向也不
                   变。判断梯度是否真的落到 register 上，要看 checkpoint 的 optimizer.pt
                   里那个 (M, hidden) 参数的矩估计——一阶矩与二阶矩开方的比值反映梯度方向
                   在步间是否一致，比值远小于 1 说明信号弱且噪，即便有梯度也学不动。

用法:
    cd /home/zhoutuowen/VLM2Vec
    PYTHONNOUSERSITE=1 PYTHONPATH=. \
      ~/anaconda3/envs/vlm2vec_qwen3/bin/python \
      experiments/public/train/summarize_reloop_okvqa.py
"""
import json
import math
import os
import re

CELLS = [
    ("A42", "Qwen2vl_2B.reloop.okvqa.T1.M0.s42"),
    ("A43", "Qwen2vl_2B.reloop.okvqa.T1.M0.s43"),
    ("B1_42", "Qwen2vl_2B.reloop.okvqa.T1.M1.s42"),
    ("B42", "Qwen2vl_2B.reloop.okvqa.T1.M5.s42"),
    ("C42", "Qwen2vl_2B.reloop.okvqa.T4.M0.s42"),
    ("D42", "Qwen2vl_2B.reloop.okvqa.T4.M5.s42"),
]
# 各格的推理成本，用层调用次数之比表示（基线 28 次；T=4 时 prefix17 + 10x4 + suffix1 = 58）
COST = {"A42": 1.00, "A43": 1.00, "B1_42": 1.00, "B42": 1.00, "C42": 2.07, "D42": 2.07}
OUT = "output"
EVAL_SUBDIR = "eval__okvqa"
LOSS_RE = re.compile(r"'loss': ([0-9.]+)")
INIT_NORM = 0.02 * math.sqrt(5 * 1536)


def read_losses(exp):
    path = os.path.join(OUT, exp, "train.log")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")
    return [float(m) for m in LOSS_RE.findall(text)]


def read_score(exp):
    path = os.path.join(OUT, exp, EVAL_SUBDIR, "OK-VQA_score.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def read_register_norm(exp):
    path = os.path.join(OUT, exp, "reloop.pt")
    if not os.path.exists(path):
        return None
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "register_embed" not in state:
        return None
    return float(state["register_embed"].float().norm())


def main():
    rows = []
    print(f"{'格':<7} {'成本':>5} {'步数':>5} {'loss 首':>8} {'loss 末5均':>10} "
          f"{'hit@1':>7} {'hit@5':>7}")
    for name, exp in CELLS:
        losses = read_losses(exp)
        score = read_score(exp)
        hit1 = score["hit@1"] * 100 if score else None
        hit5 = score["hit@5"] * 100 if score else None
        tail = sum(losses[-5:]) / len(losses[-5:]) if losses else None
        print(f"{name:<7} {COST.get(name, float('nan')):>5.2f} {len(losses):>5} "
              f"{losses[0] if losses else float('nan'):>8.3f} "
              f"{tail if tail is not None else float('nan'):>10.3f} "
              f"{hit1 if hit1 is not None else float('nan'):>7.2f} "
              f"{hit5 if hit5 is not None else float('nan'):>7.2f}")
        rows.append((name, hit1))

    scores = dict(rows)
    a42, a43 = scores.get("A42"), scores.get("A43")
    noise = abs(a42 - a43) if (a42 is not None and a43 is not None) else None
    a_mean = (a42 + a43) / 2 if noise is not None else a42

    print()
    if noise is None:
        print("跑间噪声 |A42-A43| = 缺少其中一格，无法估计")
    else:
        print(f"跑间噪声 |A42-A43| = {noise:.2f} 分（判读线：效应要明显超过它）")

    def report(label, hi, lo, cost_note):
        if hi is None or lo is None:
            print(f"{label:<28} 缺少对应格")
            return
        d = hi - lo
        tag = ""
        if noise is not None:
            if abs(d) <= noise:
                tag = "  <- 噪声内"
            elif abs(d) > 2 * noise:
                tag = "  <- 明显超噪声"
        print(f"{label:<28} {d:+6.2f} 分   {cost_note}{tag}")

    print()
    print("把 D 的两个改动拆开（相对 A 两格均值 "
          f"{a_mean:.2f} 分）:" if a_mean is not None else "拆解：缺少 A 格")
    report("只改读出 (B42-A)", scores.get("B42"), a_mean, "成本 1.00x")
    report("只加深度 (C42-A)", scores.get("C42"), a_mean, "成本 2.07x")
    report("ReLoop 全套 (D42-A)", scores.get("D42"), a_mean, "成本 2.07x")
    print()
    report("已换读出后再加深度 (D42-B42)", scores.get("D42"), scores.get("B42"),
           "成本 1.00x -> 2.07x")
    report("register 数 5 vs 1 (B42-B1_42)", scores.get("B42"), scores.get("B1_42"),
           "成本相同")

    print()
    norm = read_register_norm("Qwen2vl_2B.reloop.okvqa.T4.M5.s42")
    if norm is None:
        print("register 范数: 未找到 D42 的 reloop.pt")
    else:
        moved = abs(norm - INIT_NORM) / INIT_NORM
        print(f"register 范数 = {norm:.3f}（初始化期望 {INIT_NORM:.3f}，相对变化 {moved*100:.1f}%）")
        print("  注: 范数只是参考量，不足以判断是否学到东西（旋转不变）。"
              "要判断请看 checkpoint-*/optimizer.pt 里该参数的矩估计。")

    print()
    print("参照锚点（同机已有，非同配方，仅作刻度）:")
    print("  Qwen2-VL 20 子集基线 OK-VQA hit@1 = 58.80")
    print("  Qwen3-VL VQA6 两轮   OK-VQA hit@1 = 55.50")


if __name__ == "__main__":
    main()
