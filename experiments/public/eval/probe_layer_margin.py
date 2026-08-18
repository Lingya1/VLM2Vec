"""逐层（含循环内逐次施加）的检索判别性曲线。

要回答的问题
------------
ReLoop-UME 的 §2.2 用逐层间隔 S_l 找到了 Qwen2-VL 的 "retrieval formation" 区间
（17–26），并据此决定循环哪一段。但那条曲线是在**全量混合、5000 步**的 checkpoint 上
测的。我们的模型是单任务、约 500 步训出来的，formation 区间未必还在原处——若已经
移位，循环 17–26 就是在循环错误的层段，这足以解释 ΔT 为负而与论文不矛盾。

其次，对 T=4 的 checkpoint，这条曲线在循环段会被展开成 4 段（共 58 个点）：
判别性若随 loop 单调上升，说明迭代确实在累积检索证据（论文的机制主张）；
若第 1 圈内到顶、后面横走或回落，说明后 3 圈是纯开销甚至在侵蚀表示。

测量定义（照搬论文）
--------------------
  S_a = mean_i [ sim(q_i, c_i) - Q80({sim(q_i, c_j)}_{j != i}) ]
a 是"第几次层调用"（T=1 时即层号）。sim 是 L2 归一化后的点积。读出位置取 -1
（左填充；M>0 时是最后一个 register，M=0 时是 eos）。中间层状态先过最终 RMSNorm
再归一化，使最后一个点与真实评测读出严格一致。附带每个 a 的池内 hit@1。

用法
----
  CKPT=output/... GPU=0 N_PROBE=160 bash experiments/public/eval/probe_layer_margin.sh
"""
import os

import torch
import yaml
from transformers import AutoConfig, HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor
from src.utils.basic_utils import print_master


def readout_per_application(model, collator, rows, side, device, batch_size, num_registers):
    """对一批样本前向，返回 (n_apps, N, d)：每次层调用后、读出位置的隐状态。

    all_hidden_states[k] 是第 k 次调用的输入（k=0 为嵌入层输出），末位是过了 RMSNorm 的
    最终输出。所以"第 k 次调用的输出"= hidden_states[k+1]（k 为最后一次时已带 norm）。
    这里把未过 norm 的中间态统一手动过一遍最终 RMSNorm，保证 58 个点定义一致，
    且最后一个点与 encode_input 的真实读出逐元素相同。
    """
    final_norm = None
    for m in model.encoder.modules():
        if hasattr(m, 'layers') and hasattr(m, 'norm') and hasattr(m, 'embed_tokens'):
            final_norm = m.norm
            break
    assert final_norm is not None, "找不到解码器的最终 RMSNorm"

    outs = []
    for s in range(0, len(rows), batch_size):
        batch = collator(rows[s:s + batch_size])
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            model_input = model._prepare_model_input(inputs)
            if num_registers > 0:
                model_input['input_ids'], model_input['attention_mask'] = \
                    model.reloop.extend_inputs(model_input['input_ids'],
                                               model_input['attention_mask'])
            out = model.encoder(**model_input, return_dict=True, output_hidden_states=True)
            hs = out.hidden_states  # 长度 n_apps+1
            n_apps = len(hs) - 1
            per_app = []
            for k in range(1, n_apps + 1):
                h = hs[k][:, -1, :]
                if k < n_apps:  # 中间态未过 norm，补上；末位已带 norm
                    h = final_norm(h)
                per_app.append(h.float().cpu())
            outs.append(torch.stack(per_app))  # (n_apps, b, d)
            del out, hs
            torch.cuda.empty_cache()
        print_master(f"  {side}: {min(s + batch_size, len(rows))}/{len(rows)}")
    return torch.cat(outs, dim=1)  # (n_apps, N, d)


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

    m = model.reloop.num_registers if model.reloop is not None else 0
    schedule = model.reloop_schedule

    with open(data_args.dataset_config) as f:
        dataset_configs = yaml.safe_load(f)
    name, task_config = next(iter(dataset_configs.items()))
    if data_args.data_basedir:
        for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
            if task_config.get(key):
                task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

    qry_dataset, _ = AutoEvalPairDataset.instantiate(
        model_args=model_args, data_args=data_args, **task_config)

    n_probe = int(os.environ.get("N_PROBE", 160))
    batch_size = int(os.environ.get("PROBE_BS", 8))

    # 每行的正例是其候选列表的第 0 位（data_prepare 里 label_name = cand_names[0]）。
    # 探针池 = N 个查询各自的正例；查询 i 的负例是其他查询的正例。去重后若正例撞名
    # （同图同物体），跳过该行以保持一一对应。
    qrows, crows, seen = [], [], set()
    for i in range(len(qry_dataset)):
        if len(qrows) >= n_probe:
            break
        row = qry_dataset[i]
        pos_name = row['dataset_infos']['label_name']
        if pos_name in seen:
            continue
        seen.add(pos_name)
        qrows.append(row)
        crows.append({'cand_text': [row['cand_text'][0]],
                      'cand_image': [row['cand_image'][0]],
                      'dataset_infos': {'cand_name': pos_name}})

    print_master(f"\n=== {os.path.basename(model_args.model_name)} on {name} ===")
    print_master(f"探针查询数={len(qrows)}, M={m}, schedule={schedule}")

    qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
    cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "cand")
    Q = readout_per_application(model, qry_collator, qrows, "qry",
                                training_args.device, batch_size, m)
    C = readout_per_application(model, cand_collator, crows, "cand",
                                training_args.device, batch_size, m)
    assert Q.shape[0] == C.shape[0], f"两侧层调用数不一致: {Q.shape[0]} vs {C.shape[0]}"

    n_apps, N, _ = Q.shape
    # 层调用序号 -> (层号, 第几圈)。schedule 为 None 时就是普通 28 层。
    if schedule is not None:
        idx = schedule.indices
        loop_of = []
        cnt = {}
        for a, layer in enumerate(idx):
            cnt[layer] = cnt.get(layer, 0) + 1
            loop_of.append((layer, cnt[layer]))
    else:
        loop_of = [(a, 1) for a in range(n_apps)]

    print_master(f"\n{'调用#':>5} {'层':>4} {'圈':>3} {'间隔S(Q80)':>11} {'池内hit@1':>10}")
    prev_loop = 1
    for a in range(n_apps):
        q = torch.nn.functional.normalize(Q[a], dim=-1)
        c = torch.nn.functional.normalize(C[a], dim=-1)
        sim = q @ c.T                      # (N, N)，对角为正例
        pos = sim.diag()
        offdiag = sim[~torch.eye(N, dtype=torch.bool)].view(N, N - 1)
        q80 = torch.quantile(offdiag.float(), 0.8, dim=1)
        margin = (pos - q80).mean().item()
        hit = (sim.argmax(dim=1) == torch.arange(N)).float().mean().item()
        layer, loop = loop_of[a]
        sep = "  <- 进入第 %d 圈" % loop if loop != prev_loop else ""
        prev_loop = loop
        print_master(f"{a:>5} {layer:>4} {loop:>3} {margin:>11.4f} {hit * 100:>9.1f}%{sep}")


if __name__ == "__main__":
    main()
