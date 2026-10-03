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

from collections.abc import Sequence

import torch

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    all_to_all_transpose,
    token_all_gather,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import Group, Mapping
from tokenspeed.runtime.execution.context import ForwardContext


def moe_input_row_segments(
    mapping: Mapping,
    *,
    padded_global_num_tokens: Sequence[int],
    live_global_num_tokens: Sequence[int],
) -> list[tuple[int, int]]:
    """``(rows, live_rows)`` of each segment of the MoE layers' input, in row order.

    Mirrors the row layout ``CommManager.post_attn_comm`` and ``pre_moe_comm``
    produce for a padded forward (a decode graph replayed at a ladder batch
    size, a prefill graph replayed at a bucket): every rank's rows are padded
    to the same count and only a prefix of each rank's rows carries real
    tokens. The MoE input is either this rank's own rows -- the whole
    attention-DP group's rows after an all-reduce, or this rank's contiguous
    reduce-scatter shard of them -- or the all-gather of such shards over the
    MoE TP-EP group, one segment per group rank in group order.

    Args:
        mapping: The parallel layout.
        padded_global_num_tokens: Rows every rank feeds the model, indexed by
            global rank (the ranks of one attention-DP group share a value).
        live_global_num_tokens: Real token rows per global rank, same index;
            a rank's live rows are the first ones of its padded rows.

    Returns:
        One ``(rows, live_rows)`` per segment, ``live_rows <= rows``, summing
        to the MoE input's row count.
    """
    attn = mapping.attn
    world_size = mapping.world_size
    if len(padded_global_num_tokens) != world_size:
        raise ValueError(
            f"padded_global_num_tokens has {len(padded_global_num_tokens)} "
            f"entries, world_size={world_size}"
        )
    if len(live_global_num_tokens) != world_size:
        raise ValueError(
            f"live_global_num_tokens has {len(live_global_num_tokens)} "
            f"entries, world_size={world_size}"
        )

    def dp_group_rows(rank: int) -> tuple[int, int, int]:
        """``(padded, live, tp_rank)`` of ``rank``'s attention-DP group."""
        dp_rank, tp_rank = divmod(attn.scatter_index(rank), attn.tp_size)
        # The count table is indexed by global rank with the DP stride.
        first = dp_rank * attn.tp_size * attn.cp_size
        padded = int(padded_global_num_tokens[first])
        live = int(live_global_num_tokens[first])
        if not 0 <= live <= padded:
            raise ValueError(
                f"rank {rank}: {live} live rows do not fit {padded} padded rows"
            )
        return padded, live, tp_rank

    def shard(rank: int) -> tuple[int, int]:
        """``rank``'s reduce-scatter shard: a contiguous run of its group's rows."""
        padded, live, tp_rank = dp_group_rows(rank)
        lengths = CommManager._scatter_count(padded, attn.tp_size)
        offset = sum(lengths[:tp_rank])
        return lengths[tp_rank], min(max(live - offset, 0), lengths[tp_rank])

    moe_all_reduce = attn.tp_size == mapping.moe.tp_ep_size
    if mapping.moe.has_tp_ep and not moe_all_reduce:
        return [shard(rank) for rank in mapping.moe.tp_ep_group]
    if mapping.has_attn_tp and not moe_all_reduce:
        return [shard(mapping.rank)]
    padded, live, _ = dp_group_rows(mapping.rank)
    return [(padded, live)]


def dp_group_row_counts(
    table: list[int] | None,
    group: Group,
    rank: int,
    num_rows: int,
) -> list[int]:
    """Per-rank row counts of ``group`` read from a world-indexed DP table.

    ``num_rows`` is the rows this rank holds; the table's entry for it must
    agree, so a layout mistake fails here instead of hanging in the
    collective. The caller picks the table by the forward phase (input rows
    or collective rows, see ``forward_input_row_table`` and
    ``forward_collective_row_table``), never by matching ``num_rows``: a rank
    with no rows of its own could not tell the tables apart, and every rank
    of the group must pick the same one.
    """
    if table is None:
        raise ValueError(
            "DP row counts are unavailable: the forward carries no global token "
            "table (attention DP is required)"
        )
    if table[rank] != num_rows:
        raise ValueError(
            f"rank {rank} holds {num_rows} rows but the forward's DP row table "
            f"gives it {table[rank]}"
        )
    return [table[peer] for peer in group]


def forward_input_row_table(ctx: ForwardContext) -> list[int] | None:
    """Per-rank input rows of the forward (the token counts)."""
    return ctx.global_num_tokens


def forward_collective_row_table(ctx: ForwardContext) -> list[int] | None:
    """Per-rank rows the forward's collectives size by: the sizing a model
    reported (a drafter narrowing to its live rows), else the input rows."""
    if ctx.collective_global_num_tokens is not None:
        return ctx.collective_global_num_tokens
    return ctx.global_num_tokens


class CommManager:
    """Manages communication patterns (all_reduce vs RSAG) for each decoder layer.

    ``dense_batch_invariant`` selects the TP-batch-invariant dense tail: the
    layer's ``down_proj`` is column-parallel on hidden and ``post_dense_comm``
    transposes its ``[T_full, H / W]`` output back to this rank's rows instead
    of reduce-scattering partial sums.
    """

    def __init__(
        self,
        mapping: Mapping,
        layer_id: int,
        is_moe: bool,
        prev_is_moe: bool,
        dense_batch_invariant: bool,
        input_layernorm: torch.nn.Module | None = None,
        post_attn_layernorm: torch.nn.Module | None = None,
    ) -> None:
        self.mapping = mapping
        self.layer_id = layer_id
        self.is_moe = is_moe
        self.prev_is_moe = prev_is_moe
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm
        # utils.env imports server_args, which imports this package: resolve
        # the launch options lazily, as the fusion predicates below do.
        from tokenspeed.runtime.utils.env import global_server_args_dict

        # --layer-boundary-norm: how input_reduce_norm and final_norm add the
        # residual (docs/design/numerics.md, alignment.trainer).
        self.layer_boundary_norm: str = global_server_args_dict["layer_boundary_norm"]
        # --moe-combine-order: whether the MoE leaf already combined the routed
        # output across the MoE TP-EP group (post_moe_comm).
        self.moe_combine_order: str = global_server_args_dict["moe_combine_order"]
        if dense_batch_invariant and (
            not mapping.dense.has_tp or self.use_all_reduce(is_moe=False)
        ):
            raise ValueError(
                "the batch-invariant dense tail replaces a token reduce-scatter "
                "and needs a dense TP group wider than attention TP"
            )
        self.dense_batch_invariant = dense_batch_invariant

    # ---- Scattered token counts ----

    @staticmethod
    def _scatter_count(num_tokens: int, tp_size: int) -> list[int]:
        base, remainder = divmod(num_tokens, tp_size)
        return [base + 1] * remainder + [base] * (tp_size - remainder)

    def get_num_tokens(self, ctx: ForwardContext):
        scattered = self.scattered_num_tokens(ctx)
        return sum(scattered), max(scattered)

    def scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        global_counts = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        if global_counts is not None:
            scattered = []
            for attn_dp_rank in range(self.mapping.attn.dp_size):
                # global_counts is indexed by global rank with dp stride
                # tp_size * cp_size; cp peers report the same count.
                num_tokens = global_counts[
                    attn_dp_rank * self.mapping.attn.tp_size * self.mapping.attn.cp_size
                ]
                scattered.extend(
                    self._scatter_count(num_tokens, self.mapping.attn.tp_size)
                )
            return scattered
        num_tokens = (
            ctx.collective_num_tokens
            if ctx.collective_num_tokens is not None
            else ctx.input_num_tokens
        )
        return self._scatter_count(num_tokens, self.mapping.attn.tp_size)

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
        global_counts = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        # Without DP, all ranks share the batch and the scattered table needs
        # no global metadata, so the lookup below stays valid.
        if global_counts is not None or not self.mapping.attn.has_dp:
            # After post_attn_comm reduce-scatter, each rank holds its
            # scattered share of its attn dp group's tokens, not the raw
            # global count; MoE collectives must size from those rows.
            scattered = self.scattered_num_tokens(ctx)
            return [
                scattered[self.mapping.attn.scatter_index(rank)]
                for rank in self.mapping.moe.tp_ep_group
            ]
        # With DP but no gathered metadata, other dp groups' counts are
        # unknown; only the local rank's contribution can be reported.
        num_tokens = (
            ctx.collective_num_tokens
            if ctx.collective_num_tokens is not None
            else ctx.input_num_tokens
        )
        result = [0] * tp_ep_size
        result[self.mapping.moe.tp_ep_rank] = num_tokens
        return result

    # ---- Communication patterns ----

    def use_all_reduce(self, is_moe: bool):
        if is_moe:
            return self.mapping.attn.tp_size == self.mapping.moe.tp_ep_size
        return self.mapping.attn.tp_size == self.mapping.dense.tp_size

    def needs_pre_attn_all_gather(self) -> bool:
        """Whether attention preparation must gather the previous layer's rows."""
        return (
            self.layer_id > 0
            and self.mapping.has_attn_tp
            and not self.use_all_reduce(self.prev_is_moe)
        )

    def pre_attn_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.needs_pre_attn_all_gather():
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )

    def gather_residual(self, residual: torch.Tensor, ctx: ForwardContext):
        """All-gather a residual left scattered by the previous layer's RSAG
        path (e.g. for aux hidden capture); no-op when rows are already full.

        Mirrors the pre_attn_comm gather conditions.
        """
        if not self.needs_pre_attn_all_gather():
            return residual
        return self.gather_scattered_rows(residual, ctx)

    # ---- Row layouts ----
    #
    # An all-reduce layer holds every row of its attention DP group on each
    # attention-TP rank; an RSAG layer holds this rank's scattered share. The
    # two helpers below convert between the layouts for a model whose MLPs do
    # not all follow one pattern (LongCat runs a MoE beside dense MLPs).

    def slice_scattered_rows(
        self, hidden_states: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """Keep this rank's scattered share of the full rows (no collective)."""
        token_list = self.attn_tp_group_scattered_num_tokens(ctx)
        if hidden_states.shape[0] != sum(token_list):
            raise RuntimeError(
                "slice_scattered_rows expects the full rows of the attention "
                f"DP group: got {hidden_states.shape[0]} rows for "
                f"scattered counts {token_list}"
            )
        offset = sum(token_list[: self.mapping.attn.tp_rank])
        return hidden_states[offset : offset + token_list[self.mapping.attn.tp_rank]]

    def gather_scattered_rows(
        self, hidden_states: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """All-gather the scattered shares back into full rows."""
        token_list = self.attn_tp_group_scattered_num_tokens(ctx)
        if hidden_states.shape[0] != token_list[self.mapping.attn.tp_rank]:
            raise RuntimeError(
                "gather_scattered_rows expects this rank's scattered share: "
                f"got {hidden_states.shape[0]} rows for scattered counts "
                f"{token_list} at attention-TP rank {self.mapping.attn.tp_rank}"
            )
        if sum(token_list) == 0:
            return hidden_states
        return token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=token_list,
        )

    def post_attn_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.has_attn_tp:
            return hidden_states, residual

        if self.use_all_reduce(self.is_moe):
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
            # The output residual is expected to have attn_tp_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens.
            # Otherwise, if this layer experiences a RSAG -> AR switch, residual needs allgather.
            if self.layer_id > 0 and not self.use_all_reduce(self.prev_is_moe):
                # Keep this residual-producing collective free of early PDL
                # triggers: the next gated combine-norm may preload its result.
                residual = token_all_gather(
                    residual,
                    group=self.mapping.attn.tp_group,
                    scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
                )
        else:
            token_list = self.attn_tp_group_scattered_num_tokens(ctx)
            hidden_states = token_reduce_scatter(
                hidden_states,
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
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            return hidden_states, residual
        if self.dense_batch_invariant:
            # The column-parallel down_proj already reduced the whole
            # intermediate dim on each rank: [T_full, H / W] holds every
            # token's hidden shard, so only the rows move -- no sum.
            hidden_states = all_to_all_transpose(
                hidden_states,
                group=self.mapping.dense.tp_group,
                input_split_sizes=self.dense_tp_group_scattered_num_tokens(ctx),
            )
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def post_moe_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        """Bring the routed-expert output back to this rank's layout.

        Under ``--moe-combine-order rank`` the MoE leaf returned this rank's
        partial and the group sums them here (all-reduce, or reduce-scatter
        in the RSAG layout). Under ``slot`` the leaf already folded every
        token's slots across the EP group, so the rows are complete on every
        rank and nothing is reduced; in the RSAG layout this rank still takes
        back its own token rows, as the reduce-scatter would have.
        """
        if not self.mapping.moe.has_tp_ep:
            return hidden_states, residual

        if self.moe_combine_order == "slot":
            if self.use_all_reduce(is_moe=True):
                return hidden_states, residual
            token_list = self.moe_tp_ep_group_scattered_num_tokens(ctx)
            offset = sum(token_list[: self.mapping.moe.tp_ep_rank])
            own = token_list[self.mapping.moe.tp_ep_rank]
            return hidden_states[offset : offset + own], residual

        if self.use_all_reduce(is_moe=True):
            hidden_states = all_reduce(hidden_states, self.mapping.moe.tp_ep_group)
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def needs_final_all_gather(self) -> bool:
        """Whether the model output must gather the final layer's rows."""
        return self.mapping.has_attn_tp and not self.use_all_reduce(self.is_moe)

    def post_final_norm_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.needs_final_all_gather():
            return hidden_states, residual
        hidden_states = token_all_gather(
            hidden_states,
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
        """Whether this launch's fused all-reduce+norm kernel runs here.

        The trainer-order switches veto it at the point that relies on the
        veto, not only in ``resolve_numerics``: the unfused boundary norm needs
        the bf16 ``hidden + residual`` materialized first, and the slot-order
        MoE combine returns complete rows a fused all-reduce would sum
        ``tp_size`` times.
        """
        from tokenspeed.runtime.utils.env import global_server_args_dict

        if self.layer_boundary_norm == "unfused" or self.moe_combine_order == "slot":
            return False
        return (
            self.use_all_reduce_norm_fusion()
            and num_tokens > 0
            and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
        )

    def _fused_add_norm(
        self,
        norm: torch.nn.Module,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the residual and normalize in one kernel (the fp32 sum feeds
        the norm), fused with the all-reduce when the launch allows it."""
        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
            )
        else:
            hidden_states, residual = norm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _unfused_add_norm(
        norm: torch.nn.Module,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The trainer's layer boundary: the bf16 sum is materialized and is
        both the new residual and the norm's input."""
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = norm(hidden_states)
        return hidden_states, residual

    def input_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ):
        """The norm that opens a physical layer, consuming the previous
        layer's output (``--layer-boundary-norm``)."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self.layer_boundary_norm == "unfused":
            hidden_states, residual = self._unfused_add_norm(
                self.input_layernorm, hidden_states, residual
            )
        else:
            hidden_states, residual = self._fused_add_norm(
                self.input_layernorm, hidden_states, residual
            )
        return hidden_states, residual

    def intra_layer_add_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        """An add+norm inside a physical layer on ``input_layernorm`` (LongCat's
        second attention branch). Not a layer boundary, so it keeps the fused
        form under every ``--layer-boundary-norm``, as the trainer does."""
        return self._fused_add_norm(self.input_layernorm, hidden_states, residual)

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if ctx.forward_mode.is_idle():
            return hidden_states, None

        if self.layer_boundary_norm == "unfused":
            hidden_states, residual_out = self._unfused_add_norm(
                norm, hidden_states, residual
            )
            hidden_states, _ = self.post_final_norm_comm(hidden_states, residual, ctx)
        elif self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual_out, *_ = norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
            )
        else:
            hidden_states, residual_out = norm(hidden_states, residual)
            hidden_states, _ = self.post_final_norm_comm(hidden_states, residual, ctx)

        return hidden_states, residual_out
