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

"""HY V4 MTP draft (``HYV4ForCausalLMNextN``).

The checkpoint stores one MTP block under ``model.mtp_layers.0``. It is a
single-stream decoder layer (no iHC weights) with its own gated sparse MLA,
learnable sink and MoE, wrapped by ``enorm``/``hnorm``/``eh_proj`` on the input
and ``final_layernorm`` on the output. The embedding and LM head are shared with
the target model, so the draft loads neither.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import (
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.hy_v4 import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    check_hyv4_quant_config,
    check_hyv4_runtime,
    finalize_hyv4_attention,
    load_hyv4_weights,
)

_MTP_CHECKPOINT_PREFIX = "model.mtp_layers.0"
# Wrapper tensors that live beside the decoder block in ``model``; everything
# else under the MTP prefix belongs to the decoder.
_MTP_WRAPPER_WEIGHTS = ("enorm", "hnorm", "eh_proj", "final_layernorm")


class HYV4ModelNextN(nn.Module):
    """HY V4 MTP block: fuse the next token with the target hidden state."""

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        check_hyv4_quant_config(quant_config)
        check_hyv4_runtime(mapping, enable_ihc=False)
        self.config = config
        self.mapping = mapping
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        # The draft owns a single attention layer, so its cache layer id is 0.
        # The checkpoint prefix lets quantization exclusion lists that name the
        # MTP block match its modules.
        self.decoder = HYV4DecoderLayer(
            config,
            0,
            mapping=mapping,
            quant_config=quant_config,
            prefix=_MTP_CHECKPOINT_PREFIX,
            is_nextn=True,
            alt_stream=self.alt_stream,
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        # Position 0 has no previous token, so its embedding is zeroed exactly as
        # in the reference MTP head.
        hidden_states = torch.where(positions.unsqueeze(-1) == 0, 0, hidden_states)
        if captured_hidden_states is None:
            if not ctx.forward_mode.is_idle():
                raise ValueError("HY V4 NextN requires captured_hidden_states.")
            captured_hidden_states = hidden_states

        hidden_states = self.eh_proj(
            torch.cat(
                (self.enorm(hidden_states), self.hnorm(captured_hidden_states)),
                dim=-1,
            )
        )
        hidden_states, residual = self.decoder(positions, hidden_states, ctx, None)
        if not ctx.forward_mode.is_idle():
            hidden_states, _ = self.decoder.comm_manager.final_norm(
                hidden_states, residual, ctx, self.final_layernorm
            )
        return hidden_states, None


class HYV4ForCausalLMNextN(HYV4ForCausalLM):
    """HY V4 MTP draft model sharing the target's embedding and LM head."""

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        # The MTP layer runs its own indexer at every draft step rather than
        # reusing the target's or the previous step's top-k.
        self.index_share_for_mtp_iteration = False
        if config.num_nextn_predict_layers != 1:
            raise ValueError(
                "HY V4 NextN supports exactly one MTP layer, got "
                f"{config.num_nextn_predict_layers}"
            )

        self.model = HYV4ModelNextN(config, mapping=mapping, quant_config=quant_config)
        if mapping.attn.has_dp:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                tp_rank=mapping.attn.tp_rank,
                tp_size=mapping.attn.tp_size,
                tp_group=mapping.attn.tp_group,
            )
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=mapping.attn.has_dp,
            do_argmax=True,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            dp_lm_head_tp=False,
        )

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        """Shard preselection: only the MTP block belongs to the draft."""
        return name.startswith(f"{_MTP_CHECKPOINT_PREFIX}.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def mtp_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if not name.startswith(f"{_MTP_CHECKPOINT_PREFIX}."):
                    continue
                suffix = name[len(_MTP_CHECKPOINT_PREFIX) + 1 :]
                head = suffix.split(".", 1)[0]
                if head in _MTP_WRAPPER_WEIGHTS:
                    yield f"model.{suffix}", tensor
                else:
                    yield f"model.decoder.{suffix}", tensor

        load_hyv4_weights(
            self,
            mtp_weights(),
            module_prefix="model",
            checkpoint_prefix="model",
            shared_params=("model.embed_tokens.", "lm_head."),
        )
        self.post_load_weights()

    def post_load_weights(self) -> None:
        finalize_hyv4_attention(self.model.decoder.self_attn)


EntryClass = [HYV4ForCausalLMNextN]
