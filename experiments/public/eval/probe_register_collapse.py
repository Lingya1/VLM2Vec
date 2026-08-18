"""测 M 个检索寄存器在最后一层是否互相坍缩，以及读出位置是否各向异性。

因果掩码下这 M 个位置看到的上下文几乎相同（第 i 个只多看到前 i-1 个 register），
所以它们的隐状态很可能高度相似。这里量化三件事：

  组内两两余弦   同一个样本内、M 个 register 位置彼此的余弦。接近 1 就是坍缩，
                意味着 M>1 并没有提供 M 份不同的证据槽位。
  跨样本余弦     固定某个 register 位置，不同样本之间的余弦。这是检索真正在意的量：
                接近 1 说明所有样本被编码到几乎同一个方向上，判别性无从产生。
  末真实 token   同样算跨样本余弦，作为基线读出位置的参照。基线之所以差，
                预期就体现在这个数偏高。

用法:
    cd /home/zhoutuowen/VLM2Vec
    bash experiments/public/eval/probe_register_collapse.sh <ckpt_dir>
"""
import os
import sys

import torch
import yaml
from transformers import AutoConfig, HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.utils.basic_utils import print_master


def pairwise_cos_stats(x):
    """x: (n, d) -> 非对角余弦的均值/最大/最小"""
    n = torch.nn.functional.normalize(x.float(), dim=-1)
    cos = n @ n.T
    off = cos[~torch.eye(cos.shape[0], dtype=bool, device=cos.device)]
    return off.mean().item(), off.max().item(), off.min().item()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', backbone)
        setattr(training_args, 'model_backbone', backbone)

    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model.eval().to(training_args.device, dtype=torch.bfloat16)

    if model.reloop is None:
        print_master("该 checkpoint 没有 register（M=0），无组内相似度可测；"
                     "只报末真实 token 的跨样本余弦")
    m = model.reloop.num_registers if model.reloop is not None else 0

    with open(data_args.dataset_config) as f:
        dataset_configs = yaml.safe_load(f)
    name, task_config = next(iter(dataset_configs.items()))
    if data_args.data_basedir:
        for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
            if task_config.get(key):
                task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

    qry_dataset, _ = AutoEvalPairDataset.instantiate(
        model_args=model_args, data_args=data_args, **task_config)
    collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
    # 8 个就够看坍缩，再多会 OOM：output_hidden_states 会留下全部 28 层，
    # 而 encoder 是 ForConditionalGeneration，还要算 15 万词表的 logits
    n_probe = int(os.environ.get("N_PROBE", 8))
    batch = collator([qry_dataset[i] for i in range(n_probe)])
    if isinstance(batch, (list, tuple)):
        batch = batch[0]

    inputs = {k: (v.to(training_args.device) if torch.is_tensor(v) else v)
              for k, v in batch.items()}

    with torch.no_grad():
        model_input = model._prepare_model_input(inputs)
        if m > 0:
            model_input['input_ids'], model_input['attention_mask'] = \
                model.reloop.extend_inputs(model_input['input_ids'],
                                           model_input['attention_mask'])
        out = model.encoder(**model_input, return_dict=True, output_hidden_states=True)
        h = out.hidden_states[-1].float().cpu()
        del out
        torch.cuda.empty_cache()

    print_master(f"\n=== {os.path.basename(model_args.model_name)} ===")
    print_master(f"探针样本数={h.shape[0]}, 序列长={h.shape[1]}, M={m}")

    if m > 1:
        regs = h[:, -m:, :]
        stats = [pairwise_cos_stats(regs[i]) for i in range(regs.shape[0])]
        mean = sum(s[0] for s in stats) / len(stats)
        mx = max(s[1] for s in stats)
        mn = min(s[2] for s in stats)
        print_master(f"组内 {m} 个 register 两两余弦: 均值 {mean:+.4f} "
                     f"(样本间最大 {mx:+.4f}, 最小 {mn:+.4f})")
        print_master("  接近 +1 即坍缩：M 个位置没有承载 M 份不同的证据")

        # 聚合均值会把"相邻两个很像、首尾差很远"和"两两都一样像"混为一谈，
        # 而这两种情形对 M 的解释完全不同：前者是级联式细化，后者才是冗余副本。
        nreg = torch.nn.functional.normalize(regs, dim=-1)
        cosmat = torch.einsum('bid,bjd->bij', nreg, nreg).mean(0)
        print_master(f"  逐对余弦矩阵（{regs.shape[0]} 个样本上平均）:")
        for i in range(m):
            print_master("    " + "  ".join(f"{cosmat[i, j]:+.3f}" for j in range(m)))
        print_master(f"  首尾 register 1 vs {m}: {cosmat[0, m - 1]:+.4f}")

        # 有效秩：把每个样本的 M 个状态做 SVD，奇异值平方归一后的熵指数。
        # 接近 1 说明 M 个状态张成的其实是一条直线，即同一向量的缩放副本；
        # 接近 M 说明它们各自独立。这比余弦均值更直接地回答"M 个槽位值不值 M 份"。
        ers = []
        for i in range(regs.shape[0]):
            sv = torch.linalg.svdvals(nreg[i])
            pk = (sv ** 2) / (sv ** 2).sum()
            ers.append(torch.exp(-(pk * pk.clamp_min(1e-12).log()).sum()).item())
        print_master(f"  组内有效秩: 均值 {sum(ers) / len(ers):.2f} / {m}"
                     f"  (最小 {min(ers):.2f}, 最大 {max(ers):.2f})")

        print_master("每个 register 位置的跨样本余弦（越低越有判别性）:")
        for j in range(m):
            mu, _, _ = pairwise_cos_stats(regs[:, j, :])
            print_master(f"  register {j}: {mu:+.4f}")

    if m > 0:
        last_real = h[:, -(m + 1), :]
    else:
        last_real = h[:, -1, :]
    mu, _, _ = pairwise_cos_stats(last_real)
    print_master(f"末真实 token 的跨样本余弦: {mu:+.4f}（基线读出位置的参照）")

    # 左填充下 readout='last' 取的就是位置 -1，直接从 h 归一化即可，不必再跑一遍前向
    pooled = torch.nn.functional.normalize(h[:, -1, :], dim=-1)
    mu, mx, mn = pairwise_cos_stats(pooled)
    print_master(f"最终归一化表示的跨样本余弦: 均值 {mu:+.4f} "
                 f"(最大 {mx:+.4f}, 最小 {mn:+.4f})")


if __name__ == "__main__":
    main()
