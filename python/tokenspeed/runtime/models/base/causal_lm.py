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

"""Base causal language model: model + lm_head + logits_processor."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base.transformer_model import BaseTransformerModel
from tokenspeed.runtime.utils import add_prefix


class BaseCausalLM(nn.Module):

    model_cls: type[BaseTransformerModel]

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.capture_aux_hidden_states: bool = False

        self.model = self.resolve_model(config, mapping, quant_config, prefix)
        self.lm_head = self.resolve_lm_head(config, quant_config, prefix)
        self.logits_processor = self.resolve_logits_processor(config)
        self.post_init()

    def resolve_model(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> BaseTransformerModel:

        return self.model_cls(
            config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

    def resolve_lm_head(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:

        if getattr(config, "tie_word_embeddings", False):
            return self.model.embed_tokens

        if self.mapping.attn.has_dp:
            return ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                prefix=add_prefix("lm_head", prefix),
            )

        return ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

    def resolve_logits_processor(self, config: PretrainedConfig) -> LogitsProcessor:

        return LogitsProcessor(
            config,
            skip_all_gather=self.mapping.attn.has_dp,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

    def post_init(self) -> None:
        """Hook for subclasses that need derived state after shared modules exist."""

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:

        self.capture_aux_hidden_states = True

        if layer_ids is None:

            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]

        else:

            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        model_kwargs = self.prepare_model_kwargs(ctx, input_ids, kwargs)

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            **model_kwargs,
        )
        logits_metadata = LogitsMetadata.from_forward_context(ctx)

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            logits_metadata,
            aux_hidden_states,
        )

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        """Hook for subclasses to pass model-specific tensors."""
        model_kwargs = {}
        for key in ("input_embeds", "inputs_embeds"):
            if kwargs.get(key) is not None:
                model_kwargs[key] = kwargs[key]
        return model_kwargs

    # Weight loading.

    def get_stacked_params_mapping(self) -> list[tuple[str, str, str]]:

        return []

    def get_skip_weight_names(self) -> list[str]:

        return ["rotary_emb.inv_freq"]

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> None:

        stacked_params_mapping = self.get_stacked_params_mapping()
        skip_patterns = self.get_skip_weight_names()
        params_dict: dict[str, nn.Parameter] = dict(self.named_parameters())

        for name, loaded_weight in weights:

            if any(pattern in name for pattern in skip_patterns):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:

                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)

                break

            else:

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:

        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:

        del self.model.embed_tokens.weight
        del self.lm_head.weight

        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
