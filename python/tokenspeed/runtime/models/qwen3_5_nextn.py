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

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForCausalLM
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3_5ForConditionalGenerationNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # The MTP model is unquantized in the nvfp4 checkpoint.
        if quant_config and quant_config.get_name() == "nvfp4":
            quant_config = None

        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)
        config.num_hidden_layers = 1
        config.full_attention_interval = 1
        self.model = Qwen3_5ForCausalLM(
            config,
            mapping=self.mapping,
            quant_config=quant_config,
            prefix=add_prefix("mtp", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if self.mapping.attn.has_dp:
                self.lm_head = ReplicatedLinear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    prefix=add_prefix("lm_head", prefix),
                )
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
            skip_all_gather=self.mapping.attn.has_dp,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

    def get_hot_token_id(self):
        return None

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        if not self.config.tie_word_embeddings:
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
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ):
        if captured_hidden_states is None and not ctx.forward_mode.is_idle():
            raise ValueError("Qwen3.5 MTP requires captured_hidden_states.")

        if ctx.forward_mode.is_idle():
            # IDLE forward: skip MTP-specific ops, just run the inner model
            # for NCCL collective participation.
            hidden_states = torch.zeros(
                0,
                self.config.hidden_size * 2,
                device=input_ids.device,
                dtype=self.model.embed_tokens.weight.dtype,
            )
        else:
            assert input_embeds is None
            input_embeds = self.model.embed_tokens(input_ids)
            hidden_states = captured_hidden_states
            input_embeds = self.pre_fc_norm_embedding(input_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        hidden_states = self.fc(hidden_states)

        hidden_states, _ = self.model(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            input_embeds=hidden_states,
        )

        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, logits_metadata
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        num_experts = getattr(self.config, "num_experts", None)

        # Skip loading extra parameters for GPTQ/nvfp4 models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        moe_loader = None
        if num_experts is not None:
            # MoE expert weights, scales, and activation scales are handled
            # by the checkpoint loader.
            moe_loader = build_moe_checkpoint_loader(
                params_dict=params_dict,
                expert_schema=ExpertCheckpointSchema(
                    gate_proj_name="gate_proj",
                    down_proj_name="down_proj",
                    up_proj_name="up_proj",
                ),
                fused_schema=ExpertCheckpointSchema(
                    gate_up_fused_name="gate_up_proj",
                    down_proj_name="down_proj",
                ),
                num_experts=num_experts,
                ep_rank=self.mapping.moe.ep_rank,
                ep_size=self.mapping.moe.ep_size,
            )
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Only process MTP branch weights
            if "mtp" not in name:
                continue

            if name.startswith("mtp."):
                # Remove the mtp. prefix for processing
                name = name.replace("mtp.", "model.")

                name = name.replace("model.fc", "fc")
                name = name.replace("model.pre_fc", "pre_fc")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            # 1) Process stacked parameters (q_proj/k_proj/v_proj & gate_proj/up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-matching weights
                if weight_name not in name:
                    continue

                # Skip MoE experts.* here, handled separately below
                if "mlp.experts" in name:
                    continue

                name_mapped = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/nvfp4 models.
                if (
                    name_mapped.endswith(ignore_suffixes)
                    and name_mapped not in params_dict
                ):
                    continue

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                name = name_mapped
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith((".bias", "_bias")) and name not in params_dict:
                    continue
                if moe_loader is not None and moe_loader.matches(name):
                    mapped_name = moe_loader.load(name, loaded_weight)
                    loaded_params.add(mapped_name)
                    continue

                # Skip loading extra parameters for GPTQ/nvfp4 models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    logger.warning("MTP weight not in params_dict: %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForConditionalGenerationNextN(Qwen3_5ForConditionalGenerationNextN):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config, mapping=mapping, quant_config=quant_config, prefix=prefix
        )


EntryClass = [
    Qwen3_5ForConditionalGenerationNextN,
    Qwen3_5MoeForConditionalGenerationNextN,
]
