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
from tokenspeed.runtime.execution.context import ForwardContext, TargetCaptureSink
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.base.decoder_layer import BaseDecoderLayer
from tokenspeed.runtime.utils import add_prefix, make_layers


def notify_target_capture(
    sink: TargetCaptureSink,
    capture_idx_map: dict[int, int],
    layer_idx: int,
    aux_hidden_states: list[torch.Tensor],
) -> None:
    """Hand the tap capture layer ``layer_idx`` just appended to the
    forward's sink.

    Skipped when the capture layer appended no tap: an idle forward skips
    BaseDecoderLayer's attention block, so nothing was appended. Layers that
    append an empty tap on an idle forward still hand over the zero-row tensor.

    Args:
        sink: The forward's ``ctx.target_capture_sink``.
        capture_idx_map: Capture layer index to the draft's positional tap
            index; layers absent from it are not handed over.
        layer_idx: The capture layer that just ran.
        aux_hidden_states: The taps collected so far in this forward.
    """
    capture_idx = capture_idx_map.get(layer_idx)
    if capture_idx is None or len(aux_hidden_states) != capture_idx + 1:
        return
    sink.on_target_capture(capture_idx, aux_hidden_states[capture_idx])


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
        # DFLASH: each capture layer's positional tap index (the draft
        # concatenates taps in this order); set_dflash_layers_to_capture().
        self._dflash_capture_idx_map: dict[int, int] = {}

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
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:

        hidden_states = input_embeds
        residual = None

        if input_embeds is None:
            # When TP > 1 and fused allreduce+norm is available, skip the
            # NCCL allreduce in the embedding and let the first decoder layer
            # fuse it with the input layernorm via the fused all-reduce kernel.
            first_layer = self.layers[0]
            fuse_embed_reduce = (
                self.embed_tokens.tp_size > 1
                and first_layer.comm_manager.should_fuse(input_ids.shape[0])
            )
            hidden_states = self.embed_tokens(
                input_ids, reduce_results=not fuse_embed_reduce
            )
            if fuse_embed_reduce:
                residual = torch.zeros_like(hidden_states)

        aux_hidden_states: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):

            capture = i in self.layers_to_capture

            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states=aux_hidden_states if capture else None,
            )

            if capture and ctx.target_capture_sink is not None:
                notify_target_capture(
                    ctx.target_capture_sink,
                    self._dflash_capture_idx_map,
                    i,
                    aux_hidden_states,
                )

        if not ctx.forward_mode.is_idle():

            if residual is None:
                raise RuntimeError("residual is required for non-idle forward mode.")

            hidden_states, final_residual = layer.comm_manager.final_norm(
                hidden_states, residual, ctx, self.norm
            )

            # Capture index num_layers + 1 (EAGLE3's id num_layers, DFLASH's
            # last layer id) selects the final norm's output residual as an
            # aux state, gathered across attn-TP as the in-loop captures are.
            if (
                aux_hidden_states is not None
                and final_residual is not None
                and len(self.layers) + 1 in self.layers_to_capture
            ):
                final_residual = layer.comm_manager.gather_residual(final_residual, ctx)
                aux_hidden_states.append(final_residual.clone())
                if ctx.target_capture_sink is not None:
                    notify_target_capture(
                        ctx.target_capture_sink,
                        self._dflash_capture_idx_map,
                        len(self.layers) + 1,
                        aux_hidden_states,
                    )

        return hidden_states, aux_hidden_states or None
