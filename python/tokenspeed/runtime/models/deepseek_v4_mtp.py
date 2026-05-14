# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Inference-only DeepSeek V4 MTP / NextN draft model."""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.layer import MoELayer
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4DecoderLayer,
    DeepseekV4MegaMoEExperts,
    _fp8_linear,
    hc_head,
)
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)


_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _spec_layer_idx(config: PretrainedConfig, weight_name: str) -> Optional[int]:
    if getattr(config, "num_nextn_predict_layers", 0) <= 0:
        return None
    start = config.num_hidden_layers
    for idx in range(start, start + config.num_nextn_predict_layers):
        if weight_name.startswith(f"model.layers.{idx}."):
            return idx
    return None


def _find_mtp_layer_idx(name: str) -> int:
    parts = name.split(".")
    if len(parts) > 1 and parts[0] == "mtp":
        try:
            return int(parts[1])
        except ValueError:
            pass
    for part in parts:
        try:
            return int(part)
        except ValueError:
            continue
    return 0


class DeepseekV4MTPSharedHead(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV4MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        cache_layer_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.cache_layer_index = (
            layer_id if cache_layer_index is None else cache_layer_index
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("e_proj", prefix),
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("h_proj", prefix),
        )
        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_mult * config.hidden_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.shared_head = DeepseekV4MTPSharedHead(config)
        self.mtp_block = DeepseekV4DecoderLayer(
            config,
            layer_id,
            mapping,
            quant_config,
            add_prefix("mtp_block", prefix),
            cache_layer_index=self.cache_layer_index,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            raise ValueError("DeepSeek V4 MTP requires input_embeds.")
        input_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, input_embeds)
        input_embeds = self.enorm(input_embeds)
        previous_hidden_states = previous_hidden_states.view(
            -1, self.hc_mult, self.config.hidden_size
        )
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = _fp8_linear(
            self.h_proj,
            previous_hidden_states,
            (self.h_proj.output_size, self.h_proj.input_size),
        ) + _fp8_linear(
            self.e_proj,
            input_embeds,
            (self.e_proj.output_size, self.e_proj.input_size),
        ).unsqueeze(
            -2
        )

        return self.mtp_block(
            positions,
            hidden_states,
            ctx,
            out_cache_loc,
            input_ids,
        )

    def compute_logits_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.hc_mult, self.config.hidden_size)
        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return self.shared_head.norm(hidden_states)


class DeepseekV4MultiTokenPredictor(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        layers = {}
        for local_idx in range(self.num_mtp_layers):
            # Checkpoint layer ids remain global, while draft KV slots are compact.
            layer_idx = self.mtp_start_layer_idx + local_idx
            layers[str(layer_idx)] = DeepseekV4MultiTokenPredictorLayer(
                config,
                mapping,
                layer_idx,
                quant_config,
                add_prefix(f"layers.{layer_idx}", prefix),
                cache_layer_index=local_idx,
            )
        self.layers = nn.ModuleDict(layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        return self.layers[str(layer_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            ctx,
            out_cache_loc,
            input_embeds,
        )

    def compute_logits_hidden(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        return self.layers[str(layer_idx)].compute_logits_hidden(hidden_states)


class DeepseekV4ForCausalLMNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.model = DeepseekV4MultiTokenPredictor(
            config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if self.mapping.attn.has_dp:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(
                config,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
            )

    def get_hot_token_id(self):
        return None

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        input_lengths: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        captured_hidden_states: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
        **kwargs,
    ):
        del kwargs
        if captured_hidden_states is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("DeepSeek V4 MTP requires captured_hidden_states.")
            captured_hidden_states = torch.zeros(
                0,
                self.config.hc_mult * self.config.hidden_size,
                device=input_ids.device,
                dtype=self.model.embed_tokens.weight.dtype,
            )

        mtp_hidden_states = self.model(
            input_ids,
            positions,
            captured_hidden_states,
            ctx,
            out_cache_loc,
            input_embeds=input_embeds,
            spec_step_idx=spec_step_idx,
        ).flatten(1)
        logits_hidden_states = self.model.compute_logits_hidden(
            mtp_hidden_states,
            spec_step_idx,
        )
        logits_metadata = LogitsMetadata.from_forward_context(ctx, input_lengths)
        return self.logits_processor(
            input_ids,
            logits_hidden_states,
            self.lm_head,
            logits_metadata,
            aux_hidden_states=[mtp_hidden_states],
        )

    @staticmethod
    def _remap_weight_name(name: str) -> str:
        for old, new in {
            ".emb.tok_emb.weight": ".embed_tokens.weight",
            ".head.weight": ".shared_head.head.weight",
            ".norm.weight": ".shared_head.norm.weight",
        }.items():
            if old in name:
                name = name.replace(old, new)
        return name

    @staticmethod
    def _rewrite_spec_layer_name(spec_layer: int, name: str) -> str:
        spec_layer_weight_names = (
            "embed_tokens",
            "enorm",
            "hnorm",
            "h_proj",
            "e_proj",
            "shared_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        )
        shared_weight_names = ("embed_tokens",)
        is_spec_weight = any(
            weight_name in name for weight_name in spec_layer_weight_names
        )
        is_shared_weight = any(
            weight_name in name for weight_name in shared_weight_names
        )
        if not is_spec_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        elif is_shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def _map_checkpoint_name(self, raw_name: str) -> Optional[str]:
        if raw_name.startswith("mtp."):
            mtp_layer_idx = _find_mtp_layer_idx(raw_name)
            raw_name = raw_name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"model.layers.{self.config.num_hidden_layers + mtp_layer_idx}.",
                1,
            )
        spec_layer = _spec_layer_idx(self.config, raw_name)
        if spec_layer is None:
            return None
        name = self._remap_weight_name(raw_name)
        name = self._rewrite_spec_layer_name(spec_layer, name)
        if name.endswith(".shared_head.head.weight"):
            return None
        if name.endswith(".scale"):
            suffix = (
                ".weight_scale"
                if _EXPERT_SCALE_RE.search(name)
                else ".weight_scale_inv"
            )
            name = name.removesuffix(".scale") + suffix
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if ".ffn.gate.bias" in name:
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        return name

    def get_stacked_params_mapping(self):
        return [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = self.get_stacked_params_mapping()
        params_dict = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        loaded_params: set[str] = set()
        for raw_name, loaded_weight in weights:
            name = self._map_checkpoint_name(raw_name)
            if name is None:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = params_dict.get(mapped_name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if moe_loader.matches(name):
                    mapped_name = moe_loader.load(name, loaded_weight)
                    loaded_params.add(mapped_name)
                    continue
                param = params_dict.get(name)
                if param is None:
                    logger.debug("Skipping unmatched DeepSeek V4 MTP weight: %s", name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        missing_layers = []
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if not any(f"model.layers.{layer_idx}." in name for name in loaded_params):
                missing_layers.append(layer_idx)
        if missing_layers:
            raise ValueError(
                "DeepSeek V4 MTP weights missing for speculative layer(s) "
                f"{missing_layers}. Use a checkpoint that includes `mtp.*` "
                "weights or disable NEXTN speculative decoding."
            )
        self.post_load_weights()
        return loaded_params

    def post_load_weights(self):
        for module in self.modules():
            if isinstance(module, DeepseekV4Compressor):
                module.process_weights_after_loading()
            elif isinstance(module, DeepseekV4MegaMoEExperts):
                module.finalize_weights()
            elif isinstance(module, MoELayer):
                module.process_weights_after_loading(module)


EntryClass = [DeepseekV4ForCausalLMNextN]
