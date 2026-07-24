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

"""Base transformer model: embed -> layers -> norm."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.base.comm_ops import FinalNormOp
from tokenspeed.runtime.models.base.compiler import (
    compile_decoder_layer,
    find_first_compute_input_group,
)
from tokenspeed.runtime.models.base.decoder_layer import (
    BaseDecoderLayer,
    CompiledDecoderLayer,
)
from tokenspeed.runtime.models.base.placement import ParallelGroup, PlacementType
from tokenspeed.runtime.moe.distribution_recorder import (
    get_global_expert_distribution_recorder,
)
from tokenspeed.runtime.utils import add_prefix, make_layers


class BaseTransformerModel(nn.Module):

    layer_cls: type[BaseDecoderLayer] = BaseDecoderLayer

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__()

        self.config = config
        self.quant_config = quant_config
        self.mapping = mapping
        self.padding_idx: int | None = getattr(config, "pad_token_id", None)
        self.vocab_size: int = config.vocab_size

        self.embed_tokens = self.resolve_embed(config, prefix)
        self.layers = self.resolve_layers(config, quant_config, prefix)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture: list[int] = []

        self._compile_decoder_stack()

        # Build the final norm op that handles cross-layer communication
        # after the last decoder layer (fused allreduce + norm, or separate
        # norm + all-gather for RSAG mode).
        self._final_norm_op = self._build_final_norm_op()

    def _compile_decoder_stack(self) -> None:
        """Compile only ``CompiledDecoderLayer`` instances."""
        prev_output_group = None
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, CompiledDecoderLayer):
                continue
            next_layer_input_group = None
            if idx + 1 < len(self.layers):
                next_layer = self.layers[idx + 1]
                if isinstance(next_layer, CompiledDecoderLayer):
                    next_exec_plan = next_layer.resolve_exec_plan()
                    next_layer_input_group = find_first_compute_input_group(
                        next_exec_plan
                    )
            compiled = compile_decoder_layer(
                layer=layer,
                exec_plan=layer.resolve_exec_plan(),
                mapping=self.mapping,
                prev_layer_output_group=prev_output_group,
                next_layer_input_group=next_layer_input_group,
            )
            layer.set_compiled(compiled)
            if compiled.final_placement is not None:
                prev_output_group = compiled.final_placement.group
            else:
                prev_output_group = None

    def _build_final_norm_op(self) -> FinalNormOp:
        """Create a FinalNormOp for the post-last-layer norm + comm."""
        last_layer = self.layers[-1] if len(self.layers) > 0 else None

        use_ar = True
        group_type = ParallelGroup.ATTN_TP
        if isinstance(last_layer, CompiledDecoderLayer):
            compiled = getattr(last_layer, "_compiled", None)
            if compiled is not None and compiled.final_placement is not None:
                use_ar = compiled.final_placement.type != PlacementType.SHARD
                group_type = compiled.final_placement.group

        return FinalNormOp(
            mapping=self.mapping,
            group_type=group_type,
            norm_module=self.norm,
            use_all_reduce_mode=use_ar,
            lm_head_group_type=ParallelGroup.ATTN_TP,
        )

    def resolve_embed(self, config: PretrainedConfig, prefix: str) -> nn.Module:
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )

    def resolve_layers(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.ModuleList:
        return make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: self.layer_cls(
                config=config,
                layer_id=idx,
                mapping=self.mapping,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:

        hidden_states = input_embeds
        residual = None

        if input_embeds is None:
            # When TP > 1 and fused allreduce+norm is available, skip the
            # NCCL allreduce in the embedding and let the first decoder layer
            # fuse it with the input layernorm via the fused all-reduce kernel.
            first_layer = self.layers[0]
            if isinstance(first_layer, CompiledDecoderLayer):
                first_compiled = first_layer._compiled
                fuse_embed_reduce = first_compiled.can_fuse_embed_reduce(
                    input_ids.shape[0]
                )
            elif isinstance(first_layer, BaseDecoderLayer):
                fuse_embed_reduce = (
                    self.mapping.attn.tp_size > 1
                    and first_layer.comm_manager.should_fuse(input_ids.shape[0])
                )
            else:
                fuse_embed_reduce = False
            hidden_states = self.embed_tokens(
                input_ids, reduce_results=not fuse_embed_reduce
            )
            if fuse_embed_reduce:
                residual = torch.zeros_like(hidden_states)

        aux_hidden_states: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):

            with get_global_expert_distribution_recorder().with_current_layer(i):

                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    ctx,
                    out_cache_loc,
                    residual,
                    aux_hidden_states=(
                        aux_hidden_states if i in self.layers_to_capture else None
                    ),
                )

        if not ctx.forward_mode.is_idle():

            if residual is None:
                raise RuntimeError("residual is required for non-idle forward mode.")

            if isinstance(layer, BaseDecoderLayer):
                hidden_states, final_residual = layer.comm_manager.final_norm(
                    hidden_states, residual, ctx, self.norm
                )
            else:
                hidden_states, final_residual = self._final_norm_op(
                    hidden_states, residual, ctx
                )

            # An id == num_layers (capture index num_layers + 1) selects the
            # final norm's output residual as an aux state, matching how each
            # layer type captures in-loop: BaseDecoderLayer gathers across
            # attn-TP, CompiledDecoderLayer appends raw.
            if (
                aux_hidden_states is not None
                and final_residual is not None
                and len(self.layers) + 1 in self.layers_to_capture
            ):
                if hasattr(layer, "comm_manager"):
                    final_residual = layer.comm_manager.gather_residual(
                        final_residual, ctx
                    )
                aux_hidden_states.append(final_residual.clone())

        return hidden_states, aux_hidden_states or None
