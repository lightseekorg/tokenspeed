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

import torch

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext


class CommManager:
    """Manages communication patterns (all_reduce vs RSAG) for each decoder layer."""

    def __init__(
        self,
        mapping: Mapping,
        layer_id: int,
        is_moe: bool,
        prev_is_moe: bool,
        input_layernorm: torch.nn.Module | None = None,
        post_attn_layernorm: torch.nn.Module | None = None,
    ) -> None:
        self.mapping = mapping
        self.layer_id = layer_id
        self.is_moe = is_moe
        self.prev_is_moe = prev_is_moe
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm

    # ---- Scattered token counts ----

    @staticmethod
    def _scatter_count(num_tokens: int, tp_size: int) -> list[int]:
        base, remainder = divmod(num_tokens, tp_size)
        return [base + 1] * remainder + [base] * (tp_size - remainder)

    def get_num_tokens(self, ctx: ForwardContext):
        scattered = self.scattered_num_tokens(ctx)
        return sum(scattered), max(scattered)

    def scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        if ctx.global_num_tokens is not None:
            scattered = []
            for attn_dp_rank in range(self.mapping.attn.dp_size):
                num_tokens = ctx.global_num_tokens[
                    attn_dp_rank * self.mapping.attn.tp_size
                ]
                scattered.extend(
                    self._scatter_count(num_tokens, self.mapping.attn.tp_size)
                )
            return scattered
        return self._scatter_count(ctx.input_num_tokens, self.mapping.attn.tp_size)

    def attn_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.attn.tp_size * self.mapping.attn.dp_rank
        end = start + self.mapping.attn.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def dense_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.dense.tp_size * self.mapping.dense.dp_rank
        end = start + self.mapping.dense.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def moe_tp_ep_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        tp_ep_size = self.mapping.moe.tp_ep_size
        if ctx.global_num_tokens is not None:
            # DP path: global_num_tokens has one entry per world rank.
            # MoE group = tp_ep_size contiguous ranks starting at dp_rank * tp_ep_size.
            start = self.mapping.moe.dp_rank * tp_ep_size
            return list(ctx.global_num_tokens[start : start + tp_ep_size])
        # No global_num_tokens (e.g., drafter context or non-DP).
        # This rank has all input_num_tokens; other ranks in the EP group have 0.
        result = [0] * tp_ep_size
        result[self.mapping.moe.tp_ep_rank] = ctx.input_num_tokens
        return result

    # ---- Communication patterns ----

    def use_all_reduce(self, is_moe: bool):
        if is_moe:
            return self.mapping.attn.tp_size == self.mapping.moe.tp_ep_size
        return self.mapping.attn.tp_size == self.mapping.dense.tp_size

    def pre_attn_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if self.layer_id == 0:
            return hidden_states

        if not self.mapping.has_attn_tp:
            return hidden_states

        if self.use_all_reduce(self.prev_is_moe):
            return hidden_states

        return token_all_gather(
            hidden_states,
            rank=self.mapping.attn.tp_rank,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )

    def post_attn_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.has_attn_tp:
            return hidden_states, residual

        if self.use_all_reduce(self.is_moe):
            hidden_states = all_reduce(
                hidden_states, self.mapping.attn.tp_rank, self.mapping.attn.tp_group
            )
            # The output residual is expected to have attn_tp_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens.
            # Otherwise, if this layer experiences a RSAG -> AR switch, residual needs allgather.
            if self.layer_id > 0 and not self.use_all_reduce(self.prev_is_moe):
                residual = token_all_gather(
                    residual,
                    rank=self.mapping.attn.tp_rank,
                    group=self.mapping.attn.tp_group,
                    scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
                )
        else:
            token_list = self.attn_tp_group_scattered_num_tokens(ctx)
            hidden_states = token_reduce_scatter(
                hidden_states,
                rank=self.mapping.attn.tp_rank,
                group=self.mapping.attn.tp_group,
                scattered_num_tokens=token_list,
            )
            # The output residual is expected to have scattered_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens, so needs slice.
            # Otherwise, if this layer experiences a AR -> RSAG switch, residual needs slice.
            if self.layer_id == 0 or self.use_all_reduce(self.prev_is_moe):
                offset = sum(token_list[: self.mapping.attn.tp_rank])
                residual = residual[offset : offset + hidden_states.size(0)]

        return hidden_states, residual

    def pre_mlp_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if self.is_moe:
            return self.pre_moe_comm(hidden_states, ctx)
        else:
            return self.pre_dense_comm(hidden_states, ctx)

    def pre_dense_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.dense.has_tp:
            return hidden_states

        if self.use_all_reduce(is_moe=False):
            return hidden_states

        return token_all_gather(
            hidden_states,
            rank=self.mapping.dense.tp_rank,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )

    def pre_moe_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.moe.has_tp_ep:
            return hidden_states

        if self.use_all_reduce(is_moe=True):
            return hidden_states

        return token_all_gather(
            hidden_states,
            rank=self.mapping.moe.tp_ep_rank,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )

    def post_mlp_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.is_moe:
            return self.post_moe_comm(hidden_states, residual, ctx)
        else:
            return self.post_dense_comm(hidden_states, residual, ctx)

    def post_dense_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.dense.has_tp:
            return hidden_states, residual

        if self.use_all_reduce(is_moe=False):
            hidden_states = all_reduce(
                hidden_states, self.mapping.dense.tp_rank, self.mapping.dense.tp_group
            )
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            rank=self.mapping.dense.tp_rank,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def post_moe_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.moe.has_tp_ep:
            return hidden_states, residual

        if self.use_all_reduce(is_moe=True):
            hidden_states = all_reduce(
                hidden_states, self.mapping.moe.tp_ep_rank, self.mapping.moe.tp_ep_group
            )
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            rank=self.mapping.moe.tp_ep_rank,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def post_final_norm_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.has_attn_tp:
            return hidden_states, residual
        if self.use_all_reduce(self.is_moe):
            return hidden_states, residual
        hidden_states = token_all_gather(
            hidden_states,
            rank=self.mapping.attn.tp_rank,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    # ---- Fused allreduce+norm ----

    def use_all_reduce_norm_fusion(self) -> bool:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        return (
            self.use_all_reduce(self.is_moe)
            and self.mapping.has_attn_tp
            and global_server_args_dict.get("enable_allreduce_fusion", False)
        )

    def should_fuse(self, num_tokens: int) -> bool:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        return (
            self.use_all_reduce_norm_fusion()
            and num_tokens > 0
            and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
        )

    def input_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.input_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def post_attn_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.post_attn_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states, residual = self.post_attn_comm(hidden_states, residual, ctx)
            hidden_states, residual = self.post_attn_layernorm(hidden_states, residual)
        return hidden_states, residual

    def post_mlp_fused(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual = self.post_mlp_comm(hidden_states, residual, ctx)
        return hidden_states, residual

    def final_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        norm: torch.nn.Module,
    ):
        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, *_ = norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
            )
        else:
            hidden_states, _ = norm(hidden_states, residual)
            hidden_states, _ = self.post_final_norm_comm(hidden_states, residual, ctx)
        return hidden_states
