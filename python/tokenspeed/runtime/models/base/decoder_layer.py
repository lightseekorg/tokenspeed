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

``BaseDecoderLayer`` uses CommManager for communication (the default path).
``CompiledDecoderLayer`` uses the compiler-driven path.
"""

from __future__ import annotations

from typing import Generic, List, Optional, Tuple, TypeVar

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.quantization import QuantizationConfig as Q
from tokenspeed.runtime.models.base.execution import (
    CompiledDecoderLayer as _CompiledRuntime,
)
from tokenspeed.runtime.models.base.execution import (
    ExecutionNode,
)
from tokenspeed.runtime.models.base.module_spec import ModuleKind, ModuleSpec
from tokenspeed.runtime.models.base.placement import ParallelGroup, Partial, Replicate


def _default_compute_output_placement(
    mapping: Mapping,
    group: ParallelGroup,
) -> Optional[Partial]:
    if group == ParallelGroup.ATTN_TP:
        has_parallel = mapping.has_attn_tp
    elif group == ParallelGroup.DENSE_TP:
        has_parallel = mapping.dense.has_tp
    elif group == ParallelGroup.MOE_TP_EP:
        has_parallel = mapping.moe.has_tp_ep
    else:
        raise ValueError(f"Unknown group: {group}")
    return Partial(group) if has_parallel else None


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
            prev_is_moe=self.is_moe_layer,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    @property
    def is_moe_layer(self) -> bool:

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
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = self.comm_manager.input_reduce_norm(
            hidden_states, residual
        )

        if aux_hidden_states is not None:

            aux_hidden_states.append(residual.clone())

        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )

        hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
            hidden_states, residual, ctx
        )

        return hidden_states, residual

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
        out_cache_loc: torch.Tensor,
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
                out_cache_loc,
                residual,
                aux_hidden_states,
            )

            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                ctx,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )

        else:

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


class CompiledDecoderLayer(nn.Module, Generic[_C]):
    """Compiler-driven decoder layer (opt-in).

    Instead of CommManager, the forward delegates to a
    ``_CompiledRuntime`` produced by the layer compiler.
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
        self.prefix = prefix

        self._compiled: Optional[_CompiledRuntime] = None
        self._exec_plan = self.build_execution_plan(prefix)

    @property
    def is_moe_layer(self) -> bool:
        return False

    def resolve_norm(self) -> nn.Module:
        return RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def build_execution_plan(self, prefix: str) -> List[ExecutionNode]:
        self.input_layernorm = self.resolve_norm()
        self.self_attn = self.resolve_attn(prefix)
        self.post_attention_layernorm = self.resolve_norm()
        self.mlp = self.resolve_mlp(prefix)

        return [
            ExecutionNode(
                module=self.input_layernorm,
                spec=self.norm_spec(captures_aux=True, skip_on_idle=True),
                name="input_layernorm",
            ),
            ExecutionNode(
                module=self.self_attn,
                spec=self.attn_spec(),
                name="self_attn",
            ),
            ExecutionNode(
                module=self.post_attention_layernorm,
                spec=self.norm_spec(),
                name="post_attention_layernorm",
            ),
            ExecutionNode(
                module=self.mlp,
                spec=self.mlp_spec(),
                name="mlp",
            ),
        ]

    def norm_spec(
        self,
        *,
        captures_aux: bool = False,
        skip_on_idle: bool = False,
    ) -> ModuleSpec:
        return ModuleSpec.from_kind(
            kind=ModuleKind.NORM,
            supports_fused_reduce_norm=True,
            captures_aux=captures_aux,
            skip_on_idle=skip_on_idle,
        )

    def attn_spec(self) -> ModuleSpec:
        input_placement = Replicate(ParallelGroup.ATTN_TP)
        return ModuleSpec.from_kind(
            input_placement=input_placement,
            output_placement=_default_compute_output_placement(
                self.mapping, ParallelGroup.ATTN_TP
            ),
            kind=ModuleKind.ATTENTION,
            skip_on_idle=True,
        )

    def mlp_spec(self) -> ModuleSpec:
        mlp_group = (
            ParallelGroup.MOE_TP_EP if self.is_moe_layer else ParallelGroup.DENSE_TP
        )
        kind = ModuleKind.MOE if self.is_moe_layer else ModuleKind.DENSE_MLP
        return ModuleSpec.from_kind(
            input_placement=Replicate(mlp_group),
            output_placement=_default_compute_output_placement(self.mapping, mlp_group),
            kind=kind,
        )

    def resolve_attn(self, prefix: str) -> nn.Module:
        raise NotImplementedError

    def resolve_mlp(self, prefix: str) -> nn.Module:
        raise NotImplementedError

    def resolve_exec_plan(self) -> List[ExecutionNode]:
        return self._exec_plan

    def set_compiled(self, compiled: _CompiledRuntime) -> None:
        self._compiled = compiled

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._compiled.forward(
            positions, hidden_states, ctx, out_cache_loc, residual, aux_hidden_states
        )


class CompiledMoEDecoderLayer(CompiledDecoderLayer):

    @property
    def is_moe_layer(self) -> bool:
        return True
