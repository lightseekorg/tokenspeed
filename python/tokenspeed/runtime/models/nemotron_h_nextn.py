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

"""Nemotron-H multi-token prediction (MTP) draft head.

The head embeds the next token and the target's final hidden state, merges
them with ``eh_proj`` and runs the checkpoint's MTP blocks (one NoPE attention
and one latent MoE block) with the target's block code, then
``final_layernorm``. Embedding and LM head are the target's. The checkpoint
keeps the whole MTP subtree in BF16, so it is built without quantization.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.nemotron_h_config import NemotronHConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import (
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.nemotron_h import (
    MixerOutput,
    NemotronHBlock,
    NemotronHNorm,
    expert_checkpoint_loader,
    load_weight,
    require_single_tp_group,
)

# The checkpoint nests the merge projection and its norms under the first block.
_HEAD_RENAMES = (
    ("layers.0.eh_proj.", "eh_proj."),
    ("layers.0.enorm.", "enorm."),
    ("layers.0.hnorm.", "hnorm."),
)


def _mtp_param_name(checkpoint_name: str) -> str:
    """The head's parameter name for an ``mtp.*`` checkpoint tensor."""
    name = checkpoint_name[len("mtp.") :]
    if name.endswith(".final_layernorm.weight"):
        return "final_layernorm.weight"
    for old, new in _HEAD_RENAMES:
        if name.startswith(old):
            return new + name[len(old) :]
    return name


class NemotronHForCausalLMNextN(nn.Module):
    """The MTP head as an Eagle-style single-step draft model."""

    def __init__(
        self,
        config: NemotronHConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del quant_config, prefix
        require_single_tp_group(mapping)
        block_types = tuple(config.mtp_layers_block_type)
        if (
            config.num_nextn_predict_layers != 1
            or block_types[:1] != ("attention",)
            or block_types.count("attention") != 1
            or "mamba" in block_types
        ):
            # Draft narrowing and the one-layer draft cache assume this shape.
            raise NotImplementedError(
                "Nemotron-H MTP needs one predict layer that opens with its only "
                "attention block and has no Mamba2 blocks, got "
                f"{config.num_nextn_predict_layers} of {block_types}"
            )
        self.config = config
        self.mapping = mapping
        attn = mapping.attn
        hidden = config.hidden_size
        eps = config.layer_norm_epsilon
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            hidden,
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )
        self.enorm = RMSNorm(hidden, eps=eps)
        self.hnorm = RMSNorm(hidden, eps=eps)
        self.eh_proj = ReplicatedLinear(
            2 * hidden, hidden, bias=False, prefix="mtp.layers.0.eh_proj"
        )
        alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            NemotronHBlock(
                config,
                mapping,
                block_type,
                0 if block_type == "attention" else None,
                config.num_hidden_layers + i,
                None,
                f"mtp.layers.{i}",
                alt_stream,
            )
            for i, block_type in enumerate(block_types)
        )
        self.final_layernorm = NemotronHNorm(
            hidden, eps, mapping, config.num_hidden_layers + len(block_types)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            hidden,
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=False,
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )

    def get_hot_token_id(self) -> None:
        """The head drafts over the full vocabulary."""
        return None

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.embed_tokens.weight
        del self.lm_head.weight
        self.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        del positions
        if ctx.forward_mode.is_idle():
            merged = torch.zeros(
                0,
                2 * self.config.hidden_size,
                device=input_ids.device,
                dtype=self.embed_tokens.weight.dtype,
            )
        else:
            if captured_hidden_states is None:
                raise ValueError("Nemotron-H MTP requires captured_hidden_states")
            merged = torch.cat(
                [
                    self.enorm(self.embed_tokens(input_ids)),
                    self.hnorm(captured_hidden_states),
                ],
                dim=-1,
            )
        hidden_states, _ = self.eh_proj(merged)

        with report_collective_sizing(ctx, ctx.bs, ctx.global_bs):
            previous: MixerOutput = (hidden_states, None)
            residual = None
            for layer in self.layers:
                previous, residual = layer(previous, residual, ctx)
                if residual.shape[0] != previous[0].shape[0]:
                    # A narrowed step's attention kept only the live rows.
                    residual = residual.index_select(0, ctx.gather_ids)
            hidden_states, _, _ = self.final_layernorm.add_norm(
                previous, residual, None, ctx
            )

        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, logits_metadata
        )

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Only checkpoint shards holding MTP tensors need to be read."""
        return name.startswith("mtp.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        expert_loader = expert_checkpoint_loader(params, self.config, self.mapping)
        for checkpoint_name, loaded in weights:
            if checkpoint_name.startswith("mtp."):
                load_weight(
                    params, expert_loader, _mtp_param_name(checkpoint_name), loaded
                )


EntryClass = NemotronHForCausalLMNextN
