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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext

if TYPE_CHECKING:
    from tokenspeed.runtime.models.base.comm_ops import CommOp

from tokenspeed.runtime.models.base.module_spec import ModuleKind, ModuleSpec
from tokenspeed.runtime.models.base.placement import Placement


@dataclass(frozen=True, slots=True)
class ExecutionNode:
    module: nn.Module
    spec: ModuleSpec
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionState:
    hidden_states: torch.Tensor
    residual: torch.Tensor | None
    ctx: ForwardContext
    out_cache_loc: torch.Tensor


StepRunner = Callable[[ExecutionState, torch.Tensor], ExecutionState]


@dataclass
class ExecutionStep:
    runner: StepRunner
    module: nn.Module | None = None
    pre_comms: list[CommOp] = field(default_factory=list)
    post_comms: list[CommOp] = field(default_factory=list)
    spec: ModuleSpec = field(default_factory=ModuleSpec)
    kind: ModuleKind = ModuleKind.GENERIC
    captures_aux: bool = False
    skip_on_idle: bool = False
    name: str | None = None


class CompiledDecoderLayer(nn.Module):
    def __init__(
        self,
        steps: list[ExecutionStep],
        final_placement: Placement | None,
        mapping: Mapping,
    ) -> None:
        from tokenspeed.runtime.models.base.comm_ops import (
            AllGatherOp,
            ReduceScatterOp,
            ResidualAllGatherOp,
            ResidualSliceOp,
        )

        super().__init__()
        self.final_placement = final_placement
        self.mapping = mapping

        self.steps = steps
        self.comm_modules = nn.ModuleList()
        has_rsag_comms = False
        for step in steps:
            for comm in step.pre_comms:
                self.comm_modules.append(comm)
                if isinstance(
                    comm,
                    (
                        AllGatherOp,
                        ReduceScatterOp,
                        ResidualAllGatherOp,
                        ResidualSliceOp,
                    ),
                ):
                    has_rsag_comms = True
            for comm in step.post_comms:
                self.comm_modules.append(comm)
                if isinstance(
                    comm,
                    (
                        AllGatherOp,
                        ReduceScatterOp,
                        ResidualAllGatherOp,
                        ResidualSliceOp,
                    ),
                ):
                    has_rsag_comms = True
        self.has_rsag_comms = has_rsag_comms

    def can_fuse_embed_reduce(self, num_tokens: int) -> bool:
        from tokenspeed.runtime.models.base.comm_ops import FusedReduceNormOp

        if not self.steps:
            return False
        first_module = self.steps[0].module
        if isinstance(first_module, FusedReduceNormOp):
            return first_module._should_fuse(num_tokens)
        return False

    def _num_global_tokens(self, ctx: ForwardContext) -> int:
        if ctx.global_num_tokens is not None:
            return sum(ctx.global_num_tokens)
        return ctx.input_num_tokens

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_global_tokens = self._num_global_tokens(ctx)
        is_idle = ctx.forward_mode.is_idle() if ctx.forward_mode else False

        if num_global_tokens == 0:
            return hidden_states, residual
        if hidden_states.shape[0] == 0 and not self.has_rsag_comms:
            return hidden_states, residual

        state = ExecutionState(hidden_states, residual, ctx, out_cache_loc)

        for step in self.steps:
            if is_idle and step.skip_on_idle:
                continue

            for comm in step.pre_comms:
                hidden_states, residual = comm(
                    state.hidden_states, state.residual, state.ctx
                )
                state = ExecutionState(
                    hidden_states, residual, state.ctx, state.out_cache_loc
                )

            state = step.runner(state, positions)

            if (
                step.captures_aux
                and aux_hidden_states is not None
                and state.residual is not None
            ):
                aux_hidden_states.append(state.residual.clone())

            for comm in step.post_comms:
                hidden_states, residual = comm(
                    state.hidden_states, state.residual, state.ctx
                )
                state = ExecutionState(
                    hidden_states, residual, state.ctx, state.out_cache_loc
                )

        return state.hidden_states, state.residual
