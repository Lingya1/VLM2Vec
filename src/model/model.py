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

        self.latent = latent
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

    def encode_input(self, input):
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
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
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
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        # reason token 与瓶颈头挂在 MMEBModel 上而不在 PEFT 里，save_pretrained 存不到它们。
        # 漏存的话，评测时会拿一组随机初始化的 reason token 去跑，分数会莫名其妙地低。
        if self.latent is not None:
            torch.save(self.latent.state_dict(), os.path.join(output_dir, LATENT_WEIGHTS_NAME))

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
