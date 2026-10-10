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

"""Native dots3-note MTP on the existing single-layer Eagle execution path.

The checkpoint's layer 46 runs as draft cache layer 0: SWA attention and a dense
MLP. It fuses its own embedding of x[t+1] with post-final-norm target h[t], at
position t. Its shared_head.norm is applied after the decoder residual sum;
that normalized output is also the recurrent hidden input for the next step.
Only the vocabulary projection is shared with the target, via set_head.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy

import torch
from torch import nn

from tokenspeed.runtime.execution.context import (
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.dots3_note import (
    Dots3NoteDecoderLayer,
    _Dots3NoteForCausalLM,
)
from tokenspeed.runtime.utils import add_prefix


class Dots3NoteModelNextN(nn.Module):
    def __init__(self, config, mapping, *, quant_config, prefix):
        super().__init__()
        if config.num_hidden_layers != 46:
            raise ValueError("dots3_note NextN requires the source 46-layer config")
        if (
            mapping.attn.qcp_size != 1
            or mapping.attn.dcp_size != 1
            or mapping.pp_size != 1
        ):
            raise NotImplementedError(
                "dots3_note NextN supports neither QCP, DCP nor PP"
            )
        self.mapping = mapping
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", prefix),
        )
        layer_config = copy(config)
        layer_config.num_hidden_layers = 1
        layer_config.layer_types = ["sliding_attention"]
        self.layers = nn.ModuleList(
            [
                Dots3NoteDecoderLayer(
                    layer_config,
                    0,
                    mapping,
                    is_nextn=True,
                    quant_config=quant_config,
                    prefix=add_prefix("layers.0", prefix),
                    alt_stream=None,
                )
            ]
        )
        # This is the MTP shared_head.norm, not the target's final norm.
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Fuse next-token embeddings and target/recurrent hidden rows at positions.

        Explicit input_embeds replaces only the embedding lookup. Returns the
        decoder's post-residual, post-norm hidden rows and no auxiliary taps.
        """
        if ctx.forward_mode.is_idle():
            hidden_states = self.enorm.weight.new_empty((0, self.enorm.weight.numel()))
        else:
            if captured_hidden_states is None:
                raise ValueError("dots3_note NextN requires captured_hidden_states")
            embeds = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            hidden_states = self.eh_proj(
                torch.cat(
                    (self.enorm(embeds), self.hnorm(captured_hidden_states)), dim=-1
                )
            )[0]
        layer = self.layers[0]
        hidden_states, residual = layer(positions, hidden_states, ctx, None)
        if not ctx.forward_mode.is_idle():
            hidden_states, _ = layer.comm_manager.final_norm(
                hidden_states, residual, ctx, self.norm
            )
        return hidden_states, None


class Dot3NoteForCausalLMNextN(_Dots3NoteForCausalLM):
    """Text-only auxiliary model: no VLM constructor arguments or encoders."""

    model_cls = Dots3NoteModelNextN

    def resolve_logits_processor(self, config):
        return LogitsProcessor(
            config,
            skip_all_gather=self.mapping.attn.has_dp,
            do_argmax=True,
            tp_rank=self.mapping.lm_head.tp_rank,
            tp_size=self.mapping.lm_head.tp_size,
            tp_group=self.mapping.lm_head.tp_group,
            dp_lm_head_tp=self.mapping.attn.has_dp and self.mapping.lm_head.has_tp,
        )

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ):
        """Return logits and normalized recurrent hidden states for Eagle."""
        narrowing = ctx.draft_narrowing is not None
        with report_collective_sizing(
            ctx,
            ctx.bs if narrowing else ctx.input_num_tokens,
            ctx.global_bs if narrowing else ctx.global_num_tokens,
        ):
            hidden_states, _ = self.model(
                input_ids,
                positions,
                ctx,
                input_embeds=input_embeds,
                captured_hidden_states=captured_hidden_states,
            )
            # Snapshot the narrowed DP row counts before the sizing scope clears them.
            logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            logits_metadata,
        )

    def get_hot_token_id(self):
        return None

    def set_head(self, head: nn.Parameter) -> None:
        """Bind the target vocabulary weight without replacing the MTP embedding."""
        if head.shape != self.lm_head.weight.shape:
            raise ValueError(
                "dots3_note NextN and target vocabulary head shapes differ"
            )
        self.lm_head.weight = head

    def set_embed_and_head(self, embed, head):
        raise NotImplementedError(
            "dots3_note NextN has its own embedding; use set_head(head)"
        )

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Select shards containing exactly block 46 or the dedicated MTP embedding."""
        return (
            name.startswith("model.layers.46.")
            or name == "model.mtp.embed_tokens.weight"
        )

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Remap the complete native MTP block through the dots checkpoint loader.

        Target tensors sharing a shard are ignored. Every draft parameter,
        including both fused MLP halves, must be supplied, and every serialized
        FP8 matrix must have its own FP32 128x128 scale grid in either stream order.
        The root lm_head is intentionally not loaded: factory binds the target head.
        """
        params = dict(self.named_parameters())
        names = {}
        for name in params:
            if not name.startswith("model."):
                continue
            if name == "model.embed_tokens.weight":
                source = "model.mtp.embed_tokens.weight"
            elif name == "model.norm.weight":
                source = "model.layers.46.shared_head.norm.weight"
            elif name.startswith("model.layers.0."):
                source = name.replace("model.layers.0.", "model.layers.46.", 1)
            else:
                source = name.replace("model.", "model.layers.46.", 1)
            if ".gate_up_proj." in source:
                for half in ("gate_proj", "up_proj"):
                    names[source.replace("gate_up_proj", half)] = name
            else:
                names[source] = name
        required = set(names)
        # These two checkpoint FP8 projections are materialized to BF16 by dots.
        for projection in ("kv_a_proj_with_mqa", "g_proj"):
            names[f"model.layers.46.self_attn.{projection}.weight_scale_inv"] = (
                f"model.layers.0.self_attn.{projection}.weight_scale_inv"
            )

        def remapped_weights():
            seen = set()
            fp8_shapes = {}
            scale_shapes = {}
            for source, weight in weights:
                if not self.checkpoint_weight_name_filter(source):
                    if source.startswith("model.mtp."):
                        raise KeyError(
                            f"Unsupported dots3_note MTP checkpoint weight: {source}"
                        )
                    continue
                if source not in names:
                    raise KeyError(
                        f"Unsupported dots3_note MTP checkpoint weight: {source}"
                    )
                if source in seen:
                    raise ValueError(
                        f"Duplicate dots3_note MTP checkpoint weight: {source}"
                    )
                seen.add(source)
                name = names[source]
                module_name, field = source.rsplit(".", 1)
                if field == "weight" and weight.dtype == torch.float8_e4m3fn:
                    fp8_shapes[module_name] = tuple(
                        (n + 127) // 128 for n in weight.shape
                    )
                if field == "weight_scale_inv":
                    if weight.dtype != torch.float32 or weight.ndim != 2:
                        raise ValueError(f"MTP requires FP32 matrix scales: {source}")
                    scale_shapes[module_name] = tuple(weight.shape)
                if ".mlp.gate_proj." in source:
                    name = name.replace("gate_up_proj", "gate_proj")
                elif ".mlp.up_proj." in source:
                    name = name.replace("gate_up_proj", "up_proj")
                yield name, weight
            if fp8_shapes != scale_shapes:
                raise ValueError(
                    "Incomplete or invalid dots3_note MTP FP8 weight/scale pairs"
                )
            missing = required - seen
            if missing:
                raise ValueError(
                    f"Missing dots3_note MTP checkpoint weights: {sorted(missing)}"
                )

        super().load_weights(remapped_weights())


class Dots3NoteForCausalLMNextN(Dot3NoteForCausalLMNextN):
    """Public checkpoint architecture alias."""


EntryClass = [Dot3NoteForCausalLMNextN, Dots3NoteForCausalLMNextN]
