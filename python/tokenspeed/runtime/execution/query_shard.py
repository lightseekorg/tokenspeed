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

Under ``--prefill-context-parallel-size N`` the ranks of the attention TP
group split one extend forward's rows between them: rank ``r`` computes the
batch-global contiguous rows ``[sum(c[:r]), sum(c[:r+1]))`` of the
scheduler's packed extend span, with ``c = scatter_count(total_tokens, N)``
-- the same split the reduce-scatter / all-gather communication path
already uses, so the per-rank row tables of ``CommManager`` describe the
shard and the final gather of sampled rows needs no permutation (rank order
is request order). The plan is plain host integers built once per forward
from the request lengths; it rides ``ForwardContext.query_shard`` and the
attention backend's extend metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


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


def _split_sorted_rows(
    row_counts: Sequence[int], rows: torch.Tensor
) -> tuple[int, ...]:
    """How many of the sorted batch-global ``rows`` fall in each rank's shard.

    The one row split of a query shard: the sampled rows of a forward (the
    last row of every request) and the prompt-logprob rows of a plan both
    go through here. Host arithmetic over the shard boundaries, no per-row
    Python work.

    Args:
        row_counts: Rows every rank owns, rank order (the shards).
        rows: Batch-global rows, a 1-D host tensor sorted ascending; rank
            order is then row order, so each rank's rows are one contiguous
            run of ``rows``.

    Returns:
        Per-rank counts summing to ``rows.shape[0]``.

    Raises:
        ValueError: ``rows`` are not a 1-D host tensor, are not sorted, or
            name a row past the shards' span.
    """
    if rows.dim() != 1 or rows.device.type != "cpu":
        raise ValueError("query shard: rows must be a 1-D host tensor")
    if rows.shape[0] > 1 and bool((rows[1:] < rows[:-1]).any()):
        # The split counts rows below each boundary; unsorted rows would be
        # miscounted silently, so refuse them here.
        raise ValueError("query shard: rows must be sorted ascending")
    bounds = torch.cumsum(torch.tensor(row_counts, dtype=rows.dtype), dim=0)
    # Rows below each shard's end; successive differences are the shards'.
    below = torch.searchsorted(rows, bounds)
    counts = torch.diff(below, prepend=below.new_zeros(1))
    if int(below[-1]) != rows.shape[0]:
        raise ValueError(
            f"query shard: rows up to {int(rows.max())} exceed the "
            f"{int(bounds[-1])}-row span"
        )
    return tuple(int(count) for count in counts)


@dataclass(frozen=True)
class QueryShardPlan:
    """Which rows of one extend forward this rank computes.

    Attributes:
        size: Ranks of the query-context-parallel group.
        rank: This rank's position in that group.
        row_counts: Rows every rank owns, ``scatter_count(total, size)``.
        sampled_rows_per_rank: How many of the forward's sampled rows (the
            last row of every request, ``cumsum(input_lengths) - 1``) fall in
            each rank's shard. Request order equals rank order, so the
            concatenation of every rank's local sampled rows is the batch's
            sampled rows in request order.
    """

    size: int
    rank: int
    row_counts: tuple[int, ...]
    sampled_rows_per_rank: tuple[int, ...]

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

    @classmethod
    def from_forward(
        cls,
        *,
        total_tokens: int,
        input_lengths: Sequence[int],
        size: int,
        rank: int,
    ) -> QueryShardPlan:
        """Plan the shard of a forward from its per-request input lengths.

        Args:
            total_tokens: Rows of the packed extend span, ``sum(input_lengths)``.
            input_lengths: New-token count of every request, request order.
            size: Ranks of the query-context-parallel group.
            rank: This rank's position in the group.

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
        row_counts = scatter_count(total_tokens, size)
        # The sampled rows (cumsum(input_lengths) - 1) are sorted by
        # construction: the same split the prompt-logprob rows take.
        sampled_rows = torch.cumsum(lengths, dim=0) - 1
        return cls(
            size=size,
            rank=rank,
            row_counts=tuple(row_counts),
            sampled_rows_per_rank=_split_sorted_rows(row_counts, sampled_rows),
        )

    @property
    def total_rows(self) -> int:
        return sum(self.row_counts)

    @property
    def local_start(self) -> int:
        """First batch-global row of this rank's shard."""
        return sum(self.row_counts[: self.rank])

    @property
    def local_end(self) -> int:
        """One past the last batch-global row of this rank's shard."""
        return self.local_start + self.row_counts[self.rank]

    @property
    def local_rows(self) -> int:
        return self.row_counts[self.rank]

    @property
    def local_slice(self) -> slice:
        """This rank's rows as a slice of the batch-global row axis."""
        return slice(self.local_start, self.local_end)

    @property
    def sampled_rows_total(self) -> int:
        return sum(self.sampled_rows_per_rank)

    @property
    def local_sampled_first(self) -> int:
        """Index into the batch's sampled rows of this rank's first one.

        The sampled rows are sorted by row, so the ones this rank owns are
        the contiguous run ``[local_sampled_first, local_sampled_first +
        sampled_rows_per_rank[rank])`` of ``gather_ids``
        (``local_rows_run(sampled_rows_per_rank)``).
        """
        return self.local_rows_run(self.sampled_rows_per_rank).start

    @property
    def local_sampled_rows(self) -> int:
        return self.sampled_rows_per_rank[self.rank]

    def local_sampled_ids(self, gather_ids: torch.Tensor) -> torch.Tensor:
        """This rank's sampled rows as indices into its shard.

        The one place the batch's ``gather_ids`` (full-layout rows, the last
        row of every request, sorted) are cut to a shard: ``ForwardContext``
        carries the full layout on every forward, the target's and the
        drafter's step 0 alike, and whoever selects local rows -- the logits
        processor's ``gather_sampled_rows``, a draft that narrows to live
        rows -- goes through here.

        Args:
            gather_ids: ``[bs]`` batch-global sampled rows (``ctx.gather_ids``).

        Returns:
            ``[sampled_rows_per_rank[rank]]`` rows re-based to the shard.
        """
        if gather_ids.shape[0] != self.sampled_rows_total:
            raise ValueError(
                f"query shard: {gather_ids.shape[0]} gather ids for a plan of "
                f"{self.sampled_rows_total} sampled rows; pass the batch's full "
                "layout, not a shard's slice"
            )
        run = self.local_rows_run(self.sampled_rows_per_rank)
        return gather_ids[run] - self.local_start

    def rows_per_rank(self, rows: torch.Tensor) -> tuple[int, ...]:
        """How many of ``rows`` fall in each rank's shard.

        The split a collective over those rows needs (the prompt-logprob rows
        of a forward, whose activations every rank contributes from its shard
        and gathers in row order), the same one ``sampled_rows_per_rank`` is
        built with: rows are batch-global and sorted, so rank order is row
        order and this rank's rows are the contiguous run
        ``[sum(counts[:rank]), sum(counts[:rank + 1]))`` of ``rows``
        (``local_rows_run``). Host arithmetic over the shard boundaries, no
        per-row Python work.

        Args:
            rows: Batch-global rows, a 1-D host tensor sorted ascending.

        Returns:
            Per-rank counts summing to ``rows.shape[0]``.

        Raises:
            ValueError: ``rows`` are not a sorted 1-D host tensor, or name a
                row past the shard span.
        """
        return _split_sorted_rows(self.row_counts, rows)

    def local_rows_run(self, rows_per_rank: Sequence[int]) -> slice:
        """This rank's run of a sorted row list split by ``rows_per_rank``."""
        first = sum(rows_per_rank[: self.rank])
        return slice(first, first + rows_per_rank[self.rank])

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


__all__ = ["QueryShardPlan", "scatter_count"]
