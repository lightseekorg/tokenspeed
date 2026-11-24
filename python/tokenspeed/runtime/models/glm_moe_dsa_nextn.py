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

"""Inference-only NextN drafter for the shared GLM-MoE-DSA backbone."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import (
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.layers.attention.dsa.utils import (
    _prepare_dsa_topk_for_mtp_decode,
)
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_nextn import (
    DeepseekModelNextN,
    DeepseekNextNWeights,
)
from tokenspeed.runtime.models.deepseek_v3 import absorb_mla_kv_b_proj
from tokenspeed.runtime.models.glm5 import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    load_fused_indexer_projection,
    pad_fused_qkv_a_proj_weight_for_fp8_blockscale,
)


class GlmMoeDsaModelNextN(DeepseekModelNextN):
    decoder_cls = GlmMoeDsaDecoderLayer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = torch.where(positions.unsqueeze(-1) == 0, 0, hidden_states)
        if captured_hidden_states is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("GLM-MoE-DSA NextN requires captured_hidden_states.")
            captured_hidden_states = hidden_states

        hidden_states = self.eh_proj(
            torch.cat(
                (
                    self.enorm(hidden_states),
                    self.hnorm(captured_hidden_states),
                ),
                dim=-1,
            )
        )

        residual = None
        hidden_states, residual = self.decoder(
            positions,
            hidden_states,
            ctx,
            residual,
        )

        if not ctx.forward_mode.is_idle():
            hidden_states, _ = self.decoder.comm_manager.final_norm(
                hidden_states, residual, ctx, self.shared_head.norm
            )
        return hidden_states, None


class GlmMoeDsaForCausalLMNextN(DeepseekNextNWeights, GlmMoeDsaForCausalLM):
    compute_dsa_topk_first_step = True

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.index_share_for_mtp_iteration = bool(
            getattr(config, "index_share_for_mtp_iteration", False)
        )
        self._init_indexer_pairing_state()
        # ``q_a_proj`` / ``kv_a_proj_with_mqa`` checkpoint tensors waiting for
        # their partner before the fused projection is written. Kept on the
        # instance because a live update streams the checkpoint in chunks and
        # the pair may straddle a ``load_weights`` call.
        self._pending_a_proj: dict[str, torch.Tensor] = {}

        if quant_config is not None and quant_config.get_name() == "nvfp4":
            quant_config = None

        self.quant_config = quant_config

        self.model = GlmMoeDsaModelNextN(
            config, mapping=self.mapping, quant_config=quant_config
        )

        if self.mapping.attn.has_dp:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
            )
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=self.mapping.attn.has_dp,
            do_argmax=True,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            attn_tp_size=self.mapping.attn.tp_size,
            dp_lm_head_tp=False,
        )

    prepare_dsa_topk_for_mtp_decode = staticmethod(_prepare_dsa_topk_for_mtp_decode)

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The first NextN pass is the sparse catch-up over the whole target
        # verify window and keeps the target's post-verify lengths for it; the
        # accepted prefix is never published here (the drafter's step loop
        # publishes it for the one-token steps that follow).
        with report_collective_sizing(ctx, ctx.bs, ctx.global_bs):
            hidden_states, _ = self.model(
                input_ids,
                positions,
                ctx,
                captured_hidden_states=captured_hidden_states,
            )
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, logits_metadata
        )

    def get_hot_token_id(self) -> None:
        return None

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Shard preselection for ``load_weights`` (see DefaultModelLoader).

        Accepts a superset of the checkpoint names ``load_weights`` consumes:
        everything under the NextN layer prefix.
        """
        return self._nextn_layer_prefix(name) is not None

    def _nextn_param(
        self, params_dict: dict[str, nn.Parameter], name: str
    ) -> nn.Parameter | None:
        return self.get_param(params_dict, name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        super().load_weights(self._load_indexer_weights(weights))

    def _load_indexer_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Load indexer weights as they stream by; yield every other weight."""
        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        for raw_name, loaded_weight in weights:
            name = self._map_checkpoint_name(raw_name)
            if name is None or ".indexer." not in name:
                yield raw_name, loaded_weight
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = self.get_param(params_dict, name)
            if param is not None:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            load_fused_indexer_projection(
                name,
                loaded_weight,
                params_dict=params_dict,
                modules_dict=modules_dict,
                pending_fp8_wk=self._pending_fp8_wk,
                loaded_shards=self._loaded_fused_indexer_shards,
                weight_block_size=getattr(self.quant_config, "weight_block_size", None),
            )

    def post_load_weights(self) -> None:
        self_attn = self.model.decoder.self_attn
        pad_fused_qkv_a_proj_weight_for_fp8_blockscale(self_attn)
        absorb_mla_kv_b_proj(self_attn)


EntryClass = [GlmMoeDsaForCausalLMNextN]
