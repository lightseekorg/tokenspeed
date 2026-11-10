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

"""Base decoder layer classes.

``BaseDecoderLayer`` uses CommManager for communication.
"""

from __future__ import annotations

from typing import Generic, TypeVar

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.quantization import QuantizationConfig as Q


def capture_layer_input(
    comm_manager: CommManager,
    residual: torch.Tensor,
    ctx: ForwardContext,
    aux_hidden_states: list[torch.Tensor] | None,
) -> None:
    """Append a capture layer's input to ``aux_hidden_states`` as full rows.

    Args:
        comm_manager: The layer's communication policy; gathers RSAG rows.
        residual: The residual right after the layer's opening (all-reduce +)
            add, i.e. the reduced layer input even when the previous layer's
            MLP all-reduce was fused into that add.
        ctx: The forward context.
        aux_hidden_states: The taps collected so far; ``None`` when this layer
            is not captured.
    """
    if aux_hidden_states is None:
        return
    # Under RSAG the residual is reduce-scattered across the attn TP group;
    # aux consumers (e.g. the EAGLE3 drafter) expect full rows. Clone because
    # later norms update the residual in place.
    aux_hidden_states.append(comm_manager.gather_residual(residual, ctx).clone())


_C = TypeVar("_C", bound=PretrainedConfig)


class BaseDecoderLayer(nn.Module, Generic[_C]):
    """Default decoder layer using CommManager for communication.

    Subclasses override ``resolve_attn()`` and ``resolve_mlp()``.
    """

    def __init__(
        self,
        config: _C,
        layer_id: int,
        mapping: Mapping,
        quant_config: Q | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__()

        self.config = config
        self.quant_config = quant_config
        self.layer_id = layer_id
        self.total_layers = config.num_hidden_layers
        self.mapping = mapping

        self.input_layernorm = self.resolve_norm()
        self.post_attention_layernorm = self.resolve_norm()

        self.self_attn = self.resolve_attn(prefix)
        self.mlp = self.resolve_mlp(prefix)

        self.comm_manager = CommManager(
            mapping=self.mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=self.is_prev_moe_layer,
            dense_batch_invariant=self.dense_batch_invariant,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            query_sharded=False,
        )

    @property
    def is_moe_layer(self) -> bool:

        return False

    @property
    def is_prev_moe_layer(self) -> bool:

        return self.is_moe_layer

    @property
    def dense_batch_invariant(self) -> bool:
        """Whether the dense MLP tail transposes rows instead of
        reduce-scattering head/channel partials (``--tp-batch-invariant``)."""

        return False

    def resolve_norm(self) -> nn.Module:

        return RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def resolve_attn(self, prefix: str) -> nn.Module:

        raise NotImplementedError

    def resolve_mlp(self, prefix: str) -> nn.Module:

        raise NotImplementedError

    def forward_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = self.comm_manager.input_reduce_norm(
            hidden_states, residual
        )

        capture_layer_input(self.comm_manager, residual, ctx, aux_hidden_states)

        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
        )

        hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
            hidden_states, residual, ctx
        )

        return hidden_states, residual

    def forward_idle_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        aux_hidden_states: list | None,
    ) -> None:
        """Run the attention half of an idle forward, which has no rows.

        Nothing by default; a layer whose attention joins collectives, or
        whose tap must stay aligned across ranks, runs them here.
        """

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:

        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)

        if self.is_moe_layer:

            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu
            )

        else:

            hidden_states = self.mlp(hidden_states)

        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )

        return hidden_states

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():

            hidden_states, residual = self.forward_attn(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states,
            )

        else:

            self.forward_idle_attn(positions, hidden_states, ctx, aux_hidden_states)

        hidden_states = self.forward_mlp(
            hidden_states,
            residual,
            ctx,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )

        return hidden_states, residual


class BaseMoEDecoderLayer(BaseDecoderLayer):

    @property
    def is_moe_layer(self) -> bool:

        return True
