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

"""CommOp: communication operations automatically inserted by the layer compiler.

Each ``CommOp`` is an ``nn.Module`` that performs a single communication
primitive (all-reduce, reduce-scatter, all-gather, or fused variants).
They are created by the compiler based on Placement transitions between
adjacent compute modules.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.models.base.placement import ParallelGroup

# ---------------------------------------------------------------------------
# Helpers for computing scattered token counts from ForwardContext
# ---------------------------------------------------------------------------


def _scatter_count(num_tokens: int, tp_size: int) -> List[int]:
    base, remainder = divmod(num_tokens, tp_size)
    return [base + 1] * remainder + [base] * (tp_size - remainder)


def _scattered_num_tokens_all(ctx: ForwardContext, mapping: Mapping) -> List[int]:
    if ctx.global_num_tokens is not None:
        scattered: List[int] = []
        for attn_dp_rank in range(mapping.attn.dp_size):
            # global_num_tokens is indexed by global rank with dp stride
            # tp_size * cp_size; cp peers report the same count.
            num_tokens = ctx.global_num_tokens[
                attn_dp_rank * mapping.attn.tp_size * mapping.attn.cp_size
            ]
            scattered.extend(_scatter_count(num_tokens, mapping.attn.tp_size))
        return scattered
    return _scatter_count(ctx.input_num_tokens, mapping.attn.tp_size)


def _group_scattered_num_tokens(
    ctx: ForwardContext,
    mapping: Mapping,
    group_type: ParallelGroup,
) -> List[int]:
    if group_type == ParallelGroup.ATTN_TP:
        start = mapping.attn.tp_size * mapping.attn.dp_rank
        end = start + mapping.attn.tp_size
        return _scattered_num_tokens_all(ctx, mapping)[start:end]
    elif group_type == ParallelGroup.DENSE_TP:
        start = mapping.dense.tp_size * mapping.dense.dp_rank
        end = start + mapping.dense.tp_size
        return _scattered_num_tokens_all(ctx, mapping)[start:end]
    elif group_type == ParallelGroup.MOE_TP_EP:
        tp_ep_size = mapping.moe.tp_ep_size
        # Without DP, all ranks share the batch and the scattered table needs
        # no global metadata, so the lookup below stays valid.
        if ctx.global_num_tokens is not None or not mapping.attn.has_dp:
            # After the attention reduce-scatter, each rank holds its
            # scattered share of its attn dp group's tokens, not the raw
            # global count; MoE collectives must size from those rows.
            scattered = _scattered_num_tokens_all(ctx, mapping)
            return [
                scattered[mapping.attn.scatter_index(rank)]
                for rank in mapping.moe.tp_ep_group
            ]
        # With DP but no gathered metadata, other dp groups' counts are
        # unknown; only the local rank's contribution can be reported.
        result = [0] * tp_ep_size
        result[mapping.moe.tp_ep_rank] = ctx.input_num_tokens
        return result
    else:
        raise ValueError(f"Unknown parallel group type: {group_type}")


# ---------------------------------------------------------------------------
# Group info
# ---------------------------------------------------------------------------


def _get_group_info(
    mapping: Mapping, group_type: ParallelGroup
) -> Tuple[int, Tuple[int, ...], bool]:
    """Return (rank, group, has_parallelism) for the given parallel group type."""
    if group_type == ParallelGroup.ATTN_TP:
        return mapping.attn.tp_rank, mapping.attn.tp_group, mapping.has_attn_tp
    elif group_type == ParallelGroup.DENSE_TP:
        return mapping.dense.tp_rank, mapping.dense.tp_group, mapping.dense.has_tp
    elif group_type == ParallelGroup.MOE_TP_EP:
        return mapping.moe.tp_ep_rank, mapping.moe.tp_ep_group, mapping.moe.has_tp_ep
    else:
        raise ValueError(f"Unknown parallel group type: {group_type}")


def _should_fuse_allreduce_norm(
    num_tokens: int,
    *,
    has_parallel: bool,
    use_all_reduce_mode: bool = True,
) -> bool:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    return (
        use_all_reduce_mode
        and has_parallel
        and global_server_args_dict.get("enable_allreduce_fusion", False)
        and num_tokens > 0
        and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
    )


# ---------------------------------------------------------------------------
# Communication Operations
# ---------------------------------------------------------------------------


class CommOp(nn.Module):
    """Base class for compiler-inserted communication operations."""

    def __init__(self, mapping: Mapping, group_type: ParallelGroup) -> None:
        super().__init__()
        self.mapping = mapping
        self.group_type = group_type
        rank, group, has_parallel = _get_group_info(mapping, group_type)
        self._rank = rank
        self._group = group
        self._has_parallel = has_parallel


class AllReduceOp(CommOp):
    """all_reduce: Partial -> Replicate."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self._has_parallel:
            return hidden_states, residual
        hidden_states = all_reduce(hidden_states, self._group)
        return hidden_states, residual


class ReduceScatterOp(CommOp):
    """reduce_scatter: Partial -> Shard."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self._has_parallel:
            return hidden_states, residual
        scattered_num_tokens = _group_scattered_num_tokens(
            ctx, self.mapping, self.group_type
        )
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self._group,
            scattered_num_tokens=scattered_num_tokens,
        )
        return hidden_states, residual


class AllGatherOp(CommOp):
    """all_gather: Shard -> Replicate."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self._has_parallel:
            return hidden_states, residual
        scattered_num_tokens = _group_scattered_num_tokens(
            ctx, self.mapping, self.group_type
        )
        hidden_states = token_all_gather(
            hidden_states,
            group=self._group,
            scattered_num_tokens=scattered_num_tokens,
        )
        return hidden_states, residual


class ResidualAllGatherOp(CommOp):
    """all_gather the residual: needed when transitioning from RSAG -> AR mode."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self._has_parallel or residual is None:
            return hidden_states, residual
        scattered_num_tokens = _group_scattered_num_tokens(
            ctx, self.mapping, self.group_type
        )
        residual = token_all_gather(
            residual,
            group=self._group,
            scattered_num_tokens=scattered_num_tokens,
        )
        return hidden_states, residual


class ResidualSliceOp(CommOp):
    """Slice residual when transitioning from AR -> RSAG mode.

    When the previous layer used all-reduce (residual has full tokens) but the
    current layer uses reduce-scatter (residual should be scattered), we need
    to slice the residual to keep only the local portion.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self._has_parallel or residual is None:
            return hidden_states, residual
        scattered_num_tokens = _group_scattered_num_tokens(
            ctx, self.mapping, self.group_type
        )
        offset = sum(scattered_num_tokens[: self._rank])
        residual = residual[offset : offset + hidden_states.size(0)]
        return hidden_states, residual


class FusedReduceNormOp(CommOp):
    """Fused allreduce + residual + RMSNorm.

    When conditions are met (all-reduce mode, small enough token count), this
    replaces separate allreduce + norm with a single fused kernel. Falls back
    to unfused path when fusion is not beneficial.
    """

    def __init__(
        self,
        mapping: Mapping,
        group_type: ParallelGroup,
        norm_module: nn.Module,
    ) -> None:
        super().__init__(mapping, group_type)
        self.norm_module = norm_module

    def _should_fuse(self, num_tokens: int) -> bool:
        return _should_fuse_allreduce_norm(
            num_tokens,
            has_parallel=self._has_parallel,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if residual is None:
            # First layer: no residual to fuse with, just norm
            residual = hidden_states
            hidden_states = self.norm_module(hidden_states)
            return hidden_states, residual

        if self._should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.norm_module.forward_with_allreduce_fusion(
                    self._rank,
                    self._group,
                    hidden_states,
                    residual,
                )
            )
        else:
            # Fusion not available — fall back to explicit allreduce + norm.
            # The hidden_states arriving here are Partial (unreduced) from
            # the preceding compute module's output.  We must allreduce
            # before applying the norm.
            if self._has_parallel:
                hidden_states = all_reduce(hidden_states, self._group)
            hidden_states, residual = self.norm_module(hidden_states, residual)
        return hidden_states, residual


class DeferredReduceOp(CommOp):
    """A marker that indicates allreduce is deferred to the downstream norm op.

    The reduce is always deferred — the downstream ``FusedReduceNormOp`` or
    ``FinalNormOp`` is responsible for performing the all-reduce (fused or
    explicit) before applying the norm.  This op is therefore a no-op at
    runtime; it exists so that the compiler can record the deferred state.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Always defer — the downstream norm op handles the reduce.
        return hidden_states, residual


class FinalNormOp(CommOp):
    """Final norm after last layer, optionally fusing deferred allreduce.

    Also handles the post-final-norm all-gather needed in RSAG mode for the
    LM head.
    """

    def __init__(
        self,
        mapping: Mapping,
        group_type: ParallelGroup,
        norm_module: nn.Module,
        use_all_reduce_mode: bool,
        lm_head_group_type: Optional[ParallelGroup] = None,
    ) -> None:
        super().__init__(mapping, group_type)
        self.norm_module = norm_module
        self.use_all_reduce_mode = use_all_reduce_mode
        # The LM head follows attn_tp sharding, so in RSAG mode the
        # all-gather must use the attn_tp group — which may differ from
        # group_type (e.g. when the last layer outputs on DENSE_TP).
        if lm_head_group_type is not None and lm_head_group_type != group_type:
            lm_rank, lm_group, lm_has_parallel = _get_group_info(
                mapping, lm_head_group_type
            )
            self._lm_head_group_type = lm_head_group_type
            self._lm_rank = lm_rank
            self._lm_group = lm_group
            self._lm_has_parallel = lm_has_parallel
        else:
            self._lm_head_group_type = group_type
            self._lm_rank = self._rank
            self._lm_group = self._group
            self._lm_has_parallel = self._has_parallel

    def _should_fuse(self, num_tokens: int) -> bool:
        return _should_fuse_allreduce_norm(
            num_tokens,
            has_parallel=self._has_parallel,
            use_all_reduce_mode=self.use_all_reduce_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        ctx: ForwardContext,
    ) -> torch.Tensor:
        if self._should_fuse(hidden_states.shape[0]):
            hidden_states, *_ = self.norm_module.forward_with_allreduce_fusion(
                self._rank,
                self._group,
                hidden_states,
                residual,
            )
        else:
            # The preceding DeferredReduceOp always defers, so we must
            # perform the all-reduce here before applying the norm.
            if self._has_parallel and self.use_all_reduce_mode:
                hidden_states = all_reduce(hidden_states, self._group)
            hidden_states, _ = self.norm_module(hidden_states, residual)
            # In RSAG mode, all-gather to restore tokens for the LM head.
            # Uses the LM head group (ATTN_TP) which may differ from the
            # scatter group when attn_tp != dense_tp.
            if self._lm_has_parallel and not self.use_all_reduce_mode:
                scattered_num_tokens = _group_scattered_num_tokens(
                    ctx, self.mapping, self._lm_head_group_type
                )
                hidden_states = token_all_gather(
                    hidden_states,
                    group=self._lm_group,
                    scattered_num_tokens=scattered_num_tokens,
                )
        return hidden_states
