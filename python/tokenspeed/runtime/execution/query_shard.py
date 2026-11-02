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

"""The query shard of a prefill forward under query context parallelism.

Under ``--query-context-parallel-size N`` the ranks of the attention TP
group split one extend forward's rows between them, request by request in
zigzag: every request's new rows are cut into ``2N`` near-equal segments
(``zigzag_segment_lengths``) and rank ``r`` computes segments ``r`` and
``2N - 1 - r`` of every request. Every rank so holds an early and a late
part of every request -- under causal attention the two cost about the same
in total on every rank for sufficiently long requests. Short requests may
leave some ranks empty; those ranks still join every history collective.
This avoids assigning whole long requests to individual ranks and serializing
the group-by-group history gathers behind them.

A rank's rows are its segments in request order, so they are ascending
batch-global rows (the "span" is the scheduler's packed extend rows in
request order) but not one contiguous range. The collectives over the shard
group (all-gather, reduce-scatter, all-to-all with the per-rank counts
``row_counts``) concatenate the ranks' rows rank by rank: the *gather
order* of the span (``span_to_gather_cpu``). Every collective that only moves
or sums rows works in that order unchanged; whoever reads gathered rows by
their batch-global position (a KV write by slot, the n-gram hash, the sampled
rows of the model exit) puts them back into span order first
(``to_span_order``), and whoever reduce-scatters span-order rows puts them
into gather order (``to_gather_order``). The plan is built once per forward
from the request lengths: host integers plus one staged copy of the
row-index tensors on the device. It rides ``ForwardContext.query_shard`` and
the attention backend's extend metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tokenspeed.runtime.utils import is_pin_memory_available


def scatter_count(num_tokens: int, size: int) -> list[int]:
    """Split ``num_tokens`` rows over ``size`` ranks, the first ranks one up.

    Args:
        num_tokens: Rows to split.
        size: Ranks sharing them.

    Returns:
        Per-rank row counts; they differ by at most one and sum to
        ``num_tokens``.
    """
    base, remainder = divmod(num_tokens, size)
    return [base + 1] * remainder + [base] * (size - remainder)


def _segment_lengths(lengths: torch.Tensor, segments: int) -> torch.Tensor:
    """Remainder-first segment lengths for a batch: ``[bs, segments]`` on CPU."""
    base, remainder = lengths // segments, lengths % segments
    return base.unsqueeze(1) + (
        torch.arange(segments).unsqueeze(0) < remainder.unsqueeze(1)
    ).to(torch.int64)


def zigzag_segment_lengths(length: int, size: int) -> list[int]:
    """Cut one request's ``length`` new rows into ``2 * size`` segments.

    The first segments are one row longer when ``length`` does not divide
    evenly; segment ``j`` belongs to rank ``j`` for ``j < size`` and to rank
    ``2 * size - 1 - j`` otherwise. Uses the same CPU segment calculation
    as ``QueryShardPlan.from_forward``, for a single request.
    """
    lengths = torch.tensor([length], dtype=torch.int64)
    return _segment_lengths(lengths, 2 * size)[0].tolist()


@dataclass(frozen=True)
class RowSplit:
    """Where a list of batch-global rows lives under a query shard.

    Attributes:
        rows_per_rank: How many of the rows every rank holds: the per-rank
            counts of a gather of those rows over the shard group.
        local_ids: ``[rows_per_rank[rank]]`` host indices into this rank's
            shard rows, in gather order.
        to_rows_order: Host index that puts the gathered rows (gather order)
            back in the order the rows were given:
            ``gathered.index_select(0, to_rows_order)``.
    """

    rows_per_rank: tuple[int, ...]
    local_ids: torch.Tensor
    to_rows_order: torch.Tensor


def _exclusive_cumsum(values: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(values, dim=0) - values


@dataclass(frozen=True, eq=False)
class QueryShardPlan:
    """Which rows of one extend forward this rank computes.

    Attributes:
        size: Ranks of the query-context-parallel group.
        rank: This rank's position in that group.
        row_counts: Rows every rank owns.
        span_to_gather_cpu: ``[total_rows]`` host int64: the span rows in gather
            order, every rank's rows (ascending) concatenated rank by rank.
        gather_to_span_cpu: ``[total_rows]`` host int64: the inverse of
            ``span_to_gather_cpu``, retained for ``split_rows`` to reuse without
            rebuilding it or reading the device copy back.
        local_rows_per_request: This rank's rows of every request.
        sampled_rows_per_rank: How many of the forward's sampled rows (the
            last row of every request, ``cumsum(input_lengths) - 1``) every
            rank holds.
        local_row_ids: ``[local_rows]`` int64 on the forward's device: this
            rank's span rows, ascending. Index any per-row span buffer with
            it to get the shard's rows.
        gather_to_span: ``[total_rows]`` int64 on the device: puts rows in
            gather order back in span order (``to_span_order``).
        span_to_gather: ``[total_rows]`` int64 on the device:
            ``span_to_gather_cpu`` (``to_gather_order``).
        local_index_of_row: ``[total_rows]`` int64 on the device: a span
            row's index among this rank's rows, ``-1`` on another rank.
        local_sampled_index: ``[local_sampled_rows]`` int64 on the device:
            this rank's sampled rows as indices into its shard rows, derived
            from the last row of each request. QCP consumers select these
            directly instead of indexing the shard with full-layout gather ids.
        sampled_to_request: ``[sampled_rows_total]`` int64 on the device: puts
            the gathered sampled rows (gather order) in request order.
    """

    size: int
    rank: int
    row_counts: tuple[int, ...]
    span_to_gather_cpu: torch.Tensor
    gather_to_span_cpu: torch.Tensor
    local_rows_per_request: tuple[int, ...]
    sampled_rows_per_rank: tuple[int, ...]
    local_row_ids: torch.Tensor
    gather_to_span: torch.Tensor
    span_to_gather: torch.Tensor
    local_index_of_row: torch.Tensor
    local_sampled_index: torch.Tensor
    sampled_to_request: torch.Tensor

    def __post_init__(self) -> None:
        if self.size < 1 or not 0 <= self.rank < self.size:
            raise ValueError(
                f"query shard rank {self.rank} is outside a group of {self.size}"
            )
        if len(self.row_counts) != self.size:
            raise ValueError("query shard row_counts must name every rank")
        if len(self.sampled_rows_per_rank) != self.size:
            raise ValueError("query shard sampled_rows_per_rank must name every rank")
        if any(count < 0 for count in self.row_counts) or any(
            count < 0 for count in self.sampled_rows_per_rank
        ):
            raise ValueError("query shard row counts are non-negative")
        if self.span_to_gather_cpu.shape != (sum(self.row_counts),):
            raise ValueError("query shard span_to_gather_cpu must name every span row")
        if sum(self.local_rows_per_request) != self.row_counts[self.rank]:
            raise ValueError(
                "query shard local_rows_per_request must sum to this rank's rows"
            )

    @classmethod
    def from_forward(
        cls,
        *,
        total_tokens: int,
        input_lengths: Sequence[int],
        size: int,
        rank: int,
        device: torch.device | str,
    ) -> QueryShardPlan:
        """Plan the shard of a forward from its per-request input lengths.

        Args:
            total_tokens: Rows of the packed extend span, ``sum(input_lengths)``.
            input_lengths: New-token count of every request, request order.
            size: Ranks of the query-context-parallel group.
            rank: This rank's position in the group.
            device: Explicit target for row-index tensors (the forward's device);
                they are staged with one non-blocking copy.

        Returns:
            The plan every rank of the group derives identically.
        """
        lengths = torch.tensor(
            [int(length) for length in input_lengths], dtype=torch.int64
        )
        if int(lengths.sum()) != total_tokens:
            raise ValueError(
                f"query shard: input lengths sum to {int(lengths.sum())}, not "
                f"{total_tokens} rows"
            )
        if size < 1 or not 0 <= rank < size:
            raise ValueError(f"query shard rank {rank} is outside a group of {size}")
        bs = lengths.shape[0]
        segments = 2 * size
        seg_lens = _segment_lengths(lengths, segments)  # [bs, 2N]
        seg_starts = (_exclusive_cumsum(lengths).unsqueeze(1)) + (
            torch.cumsum(seg_lens, dim=1) - seg_lens
        )
        # Rank r's segments, in gather order: per request, segment r then
        # segment 2N - 1 - r.
        ranks = torch.arange(size)
        own = torch.stack((ranks, segments - 1 - ranks), dim=1)  # [N, 2]
        per_rank_lens = seg_lens[:, own]  # [bs, N, 2]
        row_counts = per_rank_lens.sum(dim=(0, 2))
        order = (
            torch.arange(bs).view(1, bs, 1) * segments + own.view(size, 1, 2)
        ).reshape(-1)
        ordered_lens = seg_lens.reshape(-1)[order]
        ordered_starts = seg_starts.reshape(-1)[order]
        rank_major_rows = torch.repeat_interleave(
            ordered_starts - _exclusive_cumsum(ordered_lens),
            ordered_lens,
            output_size=total_tokens,
        ) + torch.arange(total_tokens, dtype=torch.int64)
        gather_to_span = torch.empty_like(rank_major_rows)
        gather_to_span[rank_major_rows] = torch.arange(total_tokens, dtype=torch.int64)
        rank_offsets = _exclusive_cumsum(row_counts)
        first = int(rank_offsets[rank])
        local_rows = int(row_counts[rank])
        local_row_ids = rank_major_rows[first : first + local_rows]
        local_index_of_row = torch.full((total_tokens,), -1, dtype=torch.int64)
        local_index_of_row[local_row_ids] = torch.arange(local_rows, dtype=torch.int64)
        sampled = _split(
            torch.cumsum(lengths, dim=0) - 1,
            gather_to_span=gather_to_span,
            row_counts=row_counts,
            rank=rank,
        )
        host = (
            local_row_ids,
            gather_to_span,
            rank_major_rows,
            local_index_of_row,
            sampled.local_ids,
            sampled.to_rows_order,
        )
        staged = torch.cat(host)
        device = torch.device(device)
        if device.type != "cpu" and is_pin_memory_available():
            staged = staged.pin_memory()
        staged = staged.to(device, non_blocking=True)
        views = torch.split(staged, [t.shape[0] for t in host])
        return cls(
            size=size,
            rank=rank,
            row_counts=tuple(int(count) for count in row_counts),
            span_to_gather_cpu=rank_major_rows,
            gather_to_span_cpu=gather_to_span,
            local_rows_per_request=tuple(
                int(count) for count in per_rank_lens[:, rank].sum(dim=1)
            ),
            sampled_rows_per_rank=sampled.rows_per_rank,
            local_row_ids=views[0],
            gather_to_span=views[1],
            span_to_gather=views[2],
            local_index_of_row=views[3],
            local_sampled_index=views[4],
            sampled_to_request=views[5],
        )

    @property
    def total_rows(self) -> int:
        return sum(self.row_counts)

    @property
    def local_rows(self) -> int:
        return self.row_counts[self.rank]

    @property
    def local_row_ids_cpu(self) -> torch.Tensor:
        """This rank's span rows, ascending, on the host."""
        first = sum(self.row_counts[: self.rank])
        return self.span_to_gather_cpu[first : first + self.local_rows]

    @property
    def sampled_rows_total(self) -> int:
        return sum(self.sampled_rows_per_rank)

    @property
    def local_sampled_rows(self) -> int:
        return self.sampled_rows_per_rank[self.rank]

    def split_rows(self, rows: torch.Tensor) -> RowSplit:
        """Where each of ``rows`` lives: the split a collective over them needs.

        The prompt-logprob rows of a forward, whose activations every rank
        contributes from its shard and gathers, go through here; the sampled
        rows take the same split (``sampled_rows_per_rank``). Host arithmetic,
        no per-row Python work.

        Args:
            rows: Batch-global rows, a 1-D host tensor of distinct rows.

        Returns:
            The per-rank counts, this rank's local indices and the index that
            restores the given order after the gather.

        Raises:
            ValueError: ``rows`` are not a 1-D host tensor, contain duplicate
                rows, or name a row past the span.
        """
        return _split(
            rows,
            gather_to_span=self.gather_to_span_cpu,
            row_counts=torch.tensor(self.row_counts, dtype=torch.int64),
            rank=self.rank,
        )

    def to_span_order(self, gathered: torch.Tensor) -> torch.Tensor:
        """Span rows gathered over the shard group (gather order) in span order."""
        if gathered.shape[0] != self.total_rows:
            raise ValueError(
                f"query shard: {gathered.shape[0]} gathered rows for a span of "
                f"{self.total_rows}"
            )
        return gathered.index_select(0, self.gather_to_span)

    def to_gather_order(self, span: torch.Tensor) -> torch.Tensor:
        """Span-order rows in gather order, the layout a reduce-scatter takes."""
        if span.shape[0] != self.total_rows:
            raise ValueError(
                f"query shard: {span.shape[0]} rows for a span of {self.total_rows}"
            )
        return span.index_select(0, self.span_to_gather)

    def rows_for_collective(self, num_tokens: int | None) -> tuple[int, ...]:
        """Per-rank row counts of the rows a collective moves.

        A model that reports a collective sizing (``report_collective_sizing``)
        has narrowed its rows to the sampled rows (a draft's first step keeps
        one live row per request); otherwise the rows are the shard.

        Args:
            num_tokens: ``ctx.collective_num_tokens``: ``None`` for the shard
                rows, the batch-global sampled-row count after a narrowing.

        Returns:
            The per-rank table the collectives split by.

        Raises:
            ValueError: ``num_tokens`` names neither the shard nor the
                sampled rows.
        """
        if num_tokens is None or num_tokens == self.total_rows:
            return self.row_counts
        if num_tokens == self.sampled_rows_total:
            return self.sampled_rows_per_rank
        raise ValueError(
            f"query shard: a collective over {num_tokens} rows matches neither "
            f"the {self.total_rows} shard rows nor the {self.sampled_rows_total} "
            "sampled rows"
        )


def _inverse(permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.shape[0], dtype=torch.int64)
    return inverse


def _split(
    rows: torch.Tensor,
    *,
    gather_to_span: torch.Tensor,
    row_counts: torch.Tensor,
    rank: int,
) -> RowSplit:
    """``QueryShardPlan.split_rows`` over the plan's host tables."""
    if rows.dim() != 1 or rows.device.type != "cpu":
        raise ValueError("query shard: rows must be a 1-D host tensor")
    total = gather_to_span.shape[0]
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= total):
        raise ValueError(
            f"query shard: rows {int(rows.min())}..{int(rows.max())} exceed the "
            f"{total}-row span"
        )
    size = row_counts.shape[0]
    # A row's position in gather order names its rank: the ranks' rows are
    # consecutive runs of the gather order.
    position = gather_to_span[rows.to(torch.int64)]
    owner = torch.searchsorted(torch.cumsum(row_counts, dim=0), position, right=True)
    rows_per_rank = torch.bincount(owner, minlength=size)
    # A gather of these rows concatenates every rank's rows rank by rank,
    # each rank's in its own (ascending) order: the rows sorted by position.
    sorted_position, by_position = torch.sort(position, stable=True)
    if bool((sorted_position[1:] == sorted_position[:-1]).any()):
        raise ValueError("query shard: rows must be distinct")
    mine = by_position[owner[by_position] == rank]
    return RowSplit(
        rows_per_rank=tuple(int(count) for count in rows_per_rank),
        local_ids=position[mine] - int(row_counts[:rank].sum()),
        to_rows_order=_inverse(by_position),
    )


__all__ = [
    "QueryShardPlan",
    "RowSplit",
    "scatter_count",
    "zigzag_segment_lengths",
]
