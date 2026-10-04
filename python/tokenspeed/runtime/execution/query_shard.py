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
        if sum(int(length) for length in input_lengths) != total_tokens:
            raise ValueError(
                f"query shard: input lengths sum to "
                f"{sum(int(length) for length in input_lengths)}, not "
                f"{total_tokens} rows"
            )
        row_counts = scatter_count(total_tokens, size)
        sampled = [0] * size
        bound = 0
        owner = 0
        end = row_counts[0]
        for length in input_lengths:
            bound += int(length)
            last_row = bound - 1
            while last_row >= end and owner + 1 < size:
                owner += 1
                end += row_counts[owner]
            sampled[owner] += 1
        return cls(
            size=size,
            rank=rank,
            row_counts=tuple(row_counts),
            sampled_rows_per_rank=tuple(sampled),
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
        sampled_rows_per_rank[rank])`` of ``gather_ids``.
        """
        return sum(self.sampled_rows_per_rank[: self.rank])

    @property
    def local_sampled_rows(self) -> int:
        return self.sampled_rows_per_rank[self.rank]

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
