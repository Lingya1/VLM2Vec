import os
from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.model.processor import QWEN2_5_VL_TOKENSELECTION
from src.arguments import ModelArguments, TrainingArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, \
    backbone2model, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V

from src.arguments import ModelArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, INTERNVIDEO2, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, LamRA_QWEN2_5, COLPALI
from src.model.processor import QWEN3_VL
# 与 processor.py 同理：这些 vendored backbone 依赖已被移走的 transformers 内部符号，
# Qwen3-VL 路径用不到，逐个保护以免新环境下整个模块 import 失败。
try:
    from src.model.baseline_backbone.colpali import ColPali
except Exception:
    ColPali = None
try:
    from src.model.baseline_backbone.gme.gme_inference import GmeQwen2VL
except Exception:
    GmeQwen2VL = None
try:
    from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL
except Exception:
    LamRAQwen2VL = None
try:
    from src.model.baseline_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL
except Exception:
    LamRAQwen25VL = None
try:
    from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
except Exception:
    Phi3VForCausalLM = None
try:
    from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
except Exception:
    LlavaNextForConditionalGeneration = None

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


LATENT_WEIGHTS_NAME = "latent_reasoner.pt"
RELOOP_WEIGHTS_NAME = "reloop.pt"


def attach_reloop(model, model_args, config, state=None):
    """在已构好的 MMEBModel 上接入 ReLoop 的循环深度与检索寄存器。

    T=1 且 M=0 时什么都不做，模型逐元素等价于原判别式基线，所以对照组与实验组共用这一条
    代码路径。写成在模型外部接入而不是塞进两个构造函数，是为了让 build() 与 load() 复用
    同一段逻辑——评测路径与训练路径的接法不一致是最容易产生"分数对不上"的地方。

    Args:
        state: checkpoint 里读出的字典。给了就以它为准覆盖命令行参数，理由同 latent：
            评测时若 T/M 与训练时对不齐，会静默读到随机初始化的 register 或错误的深度。
    """
    from src.model.reloop import RetrievalRegisters, attach_recurrence

    # 用 getattr 而不是直接取属性：部分评测脚本自己拼 model_args，不一定带上这几个字段
    num_loops = getattr(model_args, 'reloop_t', 1) or 1
    num_registers = getattr(model_args, 'reloop_m', 0) or 0
    loop_start = getattr(model_args, 'reloop_loop_start', None)
    loop_end = getattr(model_args, 'reloop_loop_end', None)
    readout = getattr(model_args, 'reloop_readout', 'last')
    if state is not None:
        num_loops = state['reloop_t']
        num_registers = state['reloop_m']
        loop_start, loop_end = state['reloop_loop_start'], state['reloop_loop_end']
        readout = state['reloop_readout']

    # 测试期深度扫描的开关：只改 T，寄存器/区间/读出仍以 checkpoint 为准。
    # 用途是回答"训练在 T=4 的模型，推理时多走或少走几圈会怎样"——这是判断模型
    # 是否真在利用循环的最直接观测。训练路径从不设这个环境变量，默认行为不变。
    force_t = os.environ.get('RELOOP_FORCE_T')
    if force_t is not None:
        print_master(f"!!! RELOOP_FORCE_T={force_t}: 覆盖拓扑 T={num_loops} -> {force_t}（仅测试期扫描用）")
        num_loops = int(force_t)

    if num_loops == 1 and num_registers == 0:
        return

    if model.latent is not None:
        raise ValueError(
            "latent 瓶颈与 ReLoop 不能同时开：两者都靠 embedding hook 改写序列末尾若干个"
            "位置的嵌入，同时挂上会互相覆盖。请分开跑。")

    # 循环深度与 KV cache 互斥（cache 按固定 layer_idx 索引，重复调用同一层会污染槽位）。
    # 在此强制关掉，而不是只依赖各加载分支自觉：漏设的分支会一路跑到前向才炸，
    # 而那已经是在训练跑完、要出评测数字的时候了。
    config.use_cache = False
    if hasattr(config, 'text_config'):
        config.text_config.use_cache = False
    if hasattr(model.encoder, 'config'):
        model.encoder.config.use_cache = False

    hidden = getattr(config, 'hidden_size', None) or config.text_config.hidden_size
    num_layers = getattr(config, 'num_hidden_layers', None) or config.text_config.num_hidden_layers
    # 默认循环倒数第 2..11 层，末层留作不参与循环的 suffix。28 层模型即 [17, 27)。
    # 之所以不从第 0 层开始循环：前段层做的是模态对齐与局部特征提取，反复施加它们没有
    # "多轮关系推导"的含义；而我们的分层可分性探针显示检索判别性是在中后段才开始形成的。
    if loop_start is None:
        loop_start = max(0, num_layers - 11)
    if loop_end is None:
        loop_end = num_layers - 1

    schedule = attach_recurrence(model.encoder, loop_start, loop_end, num_loops)
    model.reloop_schedule = schedule
    model.reloop_readout = readout

    if num_registers > 0:
        registers = RetrievalRegisters(hidden_size=hidden, num_registers=num_registers)
        if state is not None:
            registers.load_state_dict({'register_embed': state['register_embed']})
        model.reloop = registers
        # 与 latent 同理：用 hook 而不是直接传 inputs_embeds，因为 Qwen2-VL 要靠 input_ids
        # 定位图像占位符再把视觉特征 scatter 进去。
        model.encoder.get_input_embeddings().register_forward_hook(
            lambda module, args, output: model.reloop.inject(output)
        )

    print_master(
        f"ReLoop: T={num_loops}, M={num_registers}, readout={readout}, {schedule}, "
        f"解码层调用次数 {schedule.num_applications} (基线 {num_layers})")


def build_latent_reasoner(model_args, config):
    """按参数造一个 LatentReasoner；latent_k=0 时返回 None，模型退化为原始判别式版本。"""
    if not getattr(model_args, 'latent_k', 0):
        return None
    from src.model.latent_bottleneck import LatentReasoner

    # Qwen2-VL 的 hidden_size 在顶层，Qwen3-VL 放在 text_config 下
    hidden = getattr(config, 'hidden_size', None)
    if hidden is None and hasattr(config, 'text_config'):
        hidden = config.text_config.hidden_size
    if hidden is None:
        raise ValueError(f"无法从 config 推断 hidden_size：{type(config).__name__}")

    reasoner = LatentReasoner(
        hidden_size=hidden,
        num_tokens=model_args.latent_k,
        latent_size=model_args.latent_size,
        free_bits=model_args.latent_free_bits,
        init_logvar=model_args.latent_init_logvar,
    )
    print_master(
        f"Latent bottleneck: K={model_args.latent_k}, d={hidden}, "
        f"latent_size={reasoner.bottleneck.latent_size}, beta={model_args.latent_beta}, "
        f"free_bits={model_args.latent_free_bits}, "
        f"structural bound={reasoner.structural_bound_nats():.0f} nats"
    )
    return reasoner


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 latent=None,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # 本类所有编码路径都只取 hidden_states[-1]，logits 一次都没读过（全仓唯一读
        # .logits 的是 colpali 那条无关分支）。跳过 lm_head 在数学上完全等价，省下的是
        # 每个位置到 15 万词表的投影——长序列子集下这一个张量就足以撑爆显存。
        if hasattr(self.encoder, 'lm_head'):
            self.encoder.skip_lm_head = True

        self.latent = latent
        # ReLoop 的两个开关，由 attach_reloop 在模型构好后接入
        self.reloop, self.reloop_schedule, self.reloop_readout = None, None, 'last'
        self._kl_sum, self._kl_calls, self.latent_stats = None, 0, {}
        if latent is not None:
            # 用 hook 而不是直接传 inputs_embeds：Qwen2-VL 要靠 input_ids 定位图像占位符
            # 再把视觉特征 scatter 进去，绕过 input_ids 视觉分支就失效了。
            # hook 挂在 embedding 层上，梯度检查点重算时会再次触发，与首次前向一致。
            self.encoder.get_input_embeddings().register_forward_hook(
                lambda module, args, output: self.latent.inject(output)
            )

    def pop_rate_loss(self):
        """取出并清空本次前向累积的率，单位 nats。

        GradCache 会分块多次调用 encode_input（查询侧与目标侧各一轮），这里对各次调用取
        平均而不是求和，使返回值与"每样本平均率"同量纲，不随分块数变化。
        """
        if self._kl_sum is None or self._kl_calls == 0:
            return None
        rate = self._kl_sum / self._kl_calls
        self._kl_sum, self._kl_calls = None, 0
        return rate

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _prepare_model_input(self, input):
        """去掉非张量字段，并把 Qwen3-VL 逐样本的视觉 list 拼成扁平张量。"""
        model_input = {k: v for k, v in input.items() if k not in ('texts', 'images')}
        if getattr(self, "model_backbone", None) != QWEN3_VL:
            return model_input
        device = model_input['input_ids'].device
        for pv_key, thw_key in (('pixel_values', 'image_grid_thw'),
                                ('pixel_values_videos', 'video_grid_thw')):
            pv, thw = model_input.get(pv_key), model_input.get(thw_key)
            if isinstance(pv, list):
                kept = [(p, t) for p, t in zip(pv, thw) if p is not None]
                if kept:
                    model_input[pv_key] = torch.cat([p for p, _ in kept], dim=0).to(device)
                    model_input[thw_key] = torch.cat([t for _, t in kept], dim=0).to(device)
                else:
                    model_input.pop(pv_key, None)
                    model_input.pop(thw_key, None)
        return model_input

    def _encode_latent(self, input):
        """经 K 个 reason token 与随机瓶颈得到表示，同时累积率。"""
        model_input = self._prepare_model_input(input)
        model_input['input_ids'], model_input['attention_mask'] = self.latent.extend_inputs(
            model_input['input_ids'], model_input['attention_mask'])

        out = self.encoder(**model_input, return_dict=True, output_hidden_states=True)
        z, kl, stats = self.latent.readout(out.hidden_states[-1])

        # 只在有梯度的那一遍累积。GradCache 会先做一遍 no_grad 前向拿表示，再分块重算，
        # 若两遍都累积，第二遍 pop 出来的是"无梯度旧值 + 本块新值"的混合：不报错，但率项
        # 的数值被放大，等效 beta 与配置值对不上。评测时全程 no_grad，同样不该累积。
        if torch.is_grad_enabled():
            kl_mean = kl.mean()
            self._kl_sum = kl_mean if self._kl_sum is None else self._kl_sum + kl_mean
            self._kl_calls += 1
        self.latent_stats = stats

        # 对 K 个 token 取均值而非拼接：拼接会让表示维度随 K 变化，K 的消融就同时改了
        # 嵌入维度，两个因素混在一起无法归因。
        reps = z.mean(dim=1)
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _encode_reloop(self, input):
        """追加 M 个检索寄存器后编码，从寄存器读出表示。

        循环深度不在这里体现——它挂在解码器的层调度上，对本方法透明。因此 M=0、T>1 的
        配置根本不会走到这里，走的是与基线完全相同的通用分支。
        """
        model_input = self._prepare_model_input(input)
        model_input['input_ids'], model_input['attention_mask'] = self.reloop.extend_inputs(
            model_input['input_ids'], model_input['attention_mask'])

        out = self.encoder(**model_input, return_dict=True, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]

        if self.reloop_readout == 'mean':
            reps = last_hidden[:, -self.reloop.num_registers:, :].mean(dim=1)
            if self.normalize:
                reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
            return reps
        # 取位置 -1，即最后一个 register。传扩展后的 mask 而不是原始 mask：左填充下
        # _pooling 只用 mask 判断填充方向，传原始 mask 也能得到同样结果，但两者长度不一致
        # 是个隐患，一旦将来改成右填充就会静默取错位置。
        return self._pooling(last_hidden, model_input['attention_mask'])

    def encode_input(self, input):
        if self.reloop is not None:
            return self._encode_reloop(input)
        if self.latent is not None:
            supported = {QWEN2_VL, QWEN2_5_VL, QWEN3_VL}
            backbone = getattr(self, "model_backbone", None)
            if backbone not in supported:
                raise NotImplementedError(
                    f"latent bottleneck 目前只在 {sorted(supported)} 上接过线，当前 backbone 是 {backbone}。"
                    "其余 backbone 的 padding 方向与视觉特征注入方式未经验证，直接套用会静默出错。"
                )
            return self._encode_latent(input)
        if getattr(self, "model_backbone", None) == INTERNVIDEO2:
            if "input_ids" in input.keys():
                # text side
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input["attention_mask"],
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                pooled_output = self.encoder.text_proj(pooled_text_embeds)
                pooled_output /= pooled_output.norm(dim=-1, keepdim=True)
                return pooled_output
            else:
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)
                vfeat = self.encoder.vision_proj(vfeat)
                vfeat /= vfeat.norm(dim=-1, keepdim=True)
                return vfeat
        elif getattr(self, "model_backbone", None) in [GME, LamRA, LamRA_QWEN2_5]:
            # pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            texts = [text.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + '\n', '') for text in input["texts"]] # we are actually passing video queries so this should not happen
            images = []
            for imgs in input['images']:
                # if multi images are given, select the middle frame only
                if isinstance(imgs, list):
                    imgs = imgs[len(imgs) // 2]
                    assert not isinstance(imgs, list) # make sure we have extracted the middle frame and it is no longer a list
                    images.append(imgs)
                else:
                    images.append(imgs)
            pooled_output = self.encoder.get_fused_embeddings(texts=texts, images=images)
            return pooled_output
        elif getattr(self, "model_backbone", None) == COLPALI:
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output
        elif getattr(self, "model_backbone", None) == QWEN3_VL:
            # process_fn 里 pixel_values / image_grid_thw 是逐样本的 list（含 None），
            # 原生 Qwen3-VL 的 forward 要的是拼接后的扁平张量：
            # pixel_values [总patch数, dim]、image_grid_thw [总图数, 3]。
            model_input = {k: v for k, v in input.items() if k not in ('texts', 'images')}
            # batch_to_device 只搬顶层张量，搬不动"张量组成的 list"，所以这里拼接完要
            # 显式对齐到 input_ids 所在的设备，否则视觉塔会收到 CPU 张量而权重在 GPU。
            device = model_input['input_ids'].device
            for pv_key, thw_key in (('pixel_values', 'image_grid_thw'),
                                    ('pixel_values_videos', 'video_grid_thw')):
                pv, thw = model_input.get(pv_key), model_input.get(thw_key)
                if isinstance(pv, list):
                    kept = [(p, t) for p, t in zip(pv, thw) if p is not None]
                    if kept:
                        model_input[pv_key] = torch.cat([p for p, _ in kept], dim=0).to(device)
                        model_input[thw_key] = torch.cat([t for _, t in kept], dim=0).to(device)
                    else:
                        model_input.pop(pv_key, None)
                        model_input.pop(thw_key, None)
            hidden_states = self.encoder(**model_input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            return self._pooling(hidden_states, input['attention_mask'])
        elif getattr(self, "model_backbone", None) == LLAVA_NEXT:
            input['pixel_values'] = input['pixel_values'].squeeze(dim=1)
            input['image_sizes'] = input['image_sizes'].squeeze(dim=1)
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output
        else:
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == QWEN3_VL:
            from transformers import AutoModelForImageTextToText
            config.use_cache = False
            base_model = AutoModelForImageTextToText.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False

            from .utils import parse_layer_type
            lm_qwen_layer = 28
            vis_qwen_layer = 32
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,
                vis_skip_layer=vis_skip_layer,
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        latent = build_latent_reasoner(model_args, config)
        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                latent=latent,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                latent=latent,
            )
        # encode_input 按 self.model_backbone 分发，必须在这里落地，
        # 否则 Qwen3-VL 会走到通用分支，pixel_values 还是 list 直接报错。
        model.model_backbone = model_backbone
        attach_reloop(model, model_args, config)
        return model


    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        # Loading the base model
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
        if model_args.model_backbone == QWEN3_VL:
            from transformers import AutoModelForImageTextToText
            base_model = AutoModelForImageTextToText.from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V}:
            # LoRA 的 checkpoint 里只有 adapter，骨干仍从基座读，再由下面的 PeftModel 叠上去；
            # 全参微调则必须从 checkpoint 读。这里原先写死 model_name，全参时传
            # --checkpoint_path 会静默读回原始基座，而 reloop.pt / latent.pt 又是按
            # checkpoint 正确加载的，于是拼出「原始骨干 + 训练好的寄存器」且不报错。
            base_weights = model_args.model_name if model_args.lora else model_name_or_path
            config = AutoConfig.from_pretrained(base_weights, trust_remote_code=True)
            # FlashAttention2 只能在 CUDA 上跑。卡被占满时想用 CPU 跑小规模探针，
            # 就得能换成 eager/sdpa；默认值不变，训练与评测的行为一字不改。
            attn_impl = os.environ.get("VLM2VEC_ATTN", "flash_attention_2")
            config._attn_implementation = attn_impl
            config.vision_config._attn_implementation = attn_impl
            # 判别式编码只做单次前向，不需要 KV cache；这里不关掉的话会保留预训练默认的
            # True，与循环深度冲突（cache 按固定 layer_idx 索引）。其余分支与 build() 都已关。
            config.use_cache = False
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                base_weights,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )
        elif model_args.model_backbone == PHI3V:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        elif model_args.model_backbone == INTERNVIDEO2:
            print_master(f'Loading backbone [{model_args.model_backbone}] from {"src/model/vlm_backbone/internvideo2/"}')
            config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/",
                                                trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained("src/model/vlm_backbone/internvideo2/", config=config,
                                                                                   trust_remote_code=True)
        elif model_args.model_backbone == GME:
            base_model = GmeQwen2VL(model_args.model_name, processor=kwargs['processor'])
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA:
            base_model = LamRAQwen2VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA_QWEN2_5:
            base_model = LamRAQwen25VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == COLPALI:
            base_model = ColPali.from_pretrained(model_args.model_name)
            setattr(base_model, 'config', config)
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # checkpoint 里存在 latent 权重就说明这是个带瓶颈的模型，据此决定是否接入，
        # 避免评测时还得手动把 latent_k 与训练时对齐（对不齐会静默读到随机初始化的头）。
        latent_ckpt = os.path.join(model_name_or_path, LATENT_WEIGHTS_NAME)
        latent = None
        if os.path.exists(latent_ckpt):
            state = torch.load(latent_ckpt, map_location='cpu')
            inferred_k = state['reason_embed'].shape[0]
            if getattr(model_args, 'latent_k', 0) and model_args.latent_k != inferred_k:
                print_master(f"覆盖 latent_k：命令行给的是 {model_args.latent_k}，"
                             f"checkpoint 里是 {inferred_k}，以 checkpoint 为准")
            model_args.latent_k = inferred_k
            model_args.latent_size = state['bottleneck.to_mu.bias'].shape[0]
            latent = build_latent_reasoner(model_args, config)
            latent.load_state_dict(state)
        elif getattr(model_args, 'latent_k', 0):
            latent = build_latent_reasoner(model_args, config)

        # Building the model on top of the base
        if model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                latent=latent,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                latent=latent,
            )

        model.model_backbone = model_args.model_backbone
        # checkpoint 里有 reloop.pt 就以它为准，命令行给的 T/M 只在从头训练时起作用
        reloop_ckpt = os.path.join(model_name_or_path, RELOOP_WEIGHTS_NAME)
        reloop_state = torch.load(reloop_ckpt, map_location='cpu') if os.path.exists(reloop_ckpt) else None
        attach_reloop(model, model_args, config, state=reloop_state)
        return model

    def reloop_state_dict(self):
        """ReLoop 需要落盘的全部内容：register 权重 + 拓扑。没接入时返回 None。

        拓扑必须一起存。循环深度 T 本身一个参数都不带，若只存权重，评测时就得靠命令行把
        T 与训练时对齐；对不齐是"同一份权重在不同深度下评测"，分数不可解释而且不会报错。
        """
        if self.reloop_schedule is None:
            return None
        state = {
            'reloop_t': self.reloop_schedule.num_loops,
            'reloop_m': self.reloop.num_registers if self.reloop is not None else 0,
            'reloop_loop_start': self.reloop_schedule.loop_start,
            'reloop_loop_end': self.reloop_schedule.loop_end,
            'reloop_readout': self.reloop_readout,
        }
        if self.reloop is not None:
            state['register_embed'] = self.reloop.register_embed.detach().cpu()
        return state

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        # reason token 与瓶颈头挂在 MMEBModel 上而不在 PEFT 里，save_pretrained 存不到它们。
        # 漏存的话，评测时会拿一组随机初始化的 reason token 去跑，分数会莫名其妙地低。
        if self.latent is not None:
            torch.save(self.latent.state_dict(), os.path.join(output_dir, LATENT_WEIGHTS_NAME))
        reloop_state = self.reloop_state_dict()
        if reloop_state is not None:
            torch.save(reloop_state, os.path.join(output_dir, RELOOP_WEIGHTS_NAME))

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self.encoder, "enable_input_require_grads"):
            self.encoder.enable_input_require_grads()
