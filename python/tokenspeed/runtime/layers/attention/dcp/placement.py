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

"""Cache-format-independent placement of logical token slots."""

from dataclasses import dataclass

import numpy as np
import torch
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_slots_to_local


@dataclass(frozen=True)
class CachePlacement:
    """Ownership geometry for one sharded cache group, independent of its writer."""

    block_granularity: int
    virtual_block_count: int
    group: tuple[int, ...]
    rank: int

    def __post_init__(self) -> None:
        if self.block_granularity <= 0 or self.virtual_block_count <= 0:
            raise ValueError("Cache placement geometry must be positive")
        if not self.group or not 0 <= self.rank < len(self.group):
            raise ValueError("Cache placement rank must index its nonempty group")


def resolve_cache_slots(
    loc: torch.Tensor, placement: CachePlacement | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return local slots and an ownership mask; None placement preserves slots.

    The mask applies to both source loads and destination stores. Foreign rows
    resolve to safe dummy addresses but must never be written. This function
    knows neither the cache format nor the attention kernel's page-table layout.
    """
    if placement is None:
        return loc, None
    return virtual_slots_to_local(
        loc,
        rows_per_page=placement.block_granularity,
        virtual_block_count=placement.virtual_block_count,
        degree=len(placement.group),
        rank=placement.rank,
    )


def cyclic_slot_owner(
    slots: torch.Tensor, placement: CachePlacement | None
) -> torch.Tensor:
    """Owner rank of every virtual slot under the cyclic block placement.

    The same rule ``virtual_slots_to_local`` applies for one rank, evaluated
    for every rank at once: block ``v`` of a sharded group lives on rank
    ``(v - 1) % degree``. The null block 0, negative slots and slots past the
    virtual capacity have no owner and resolve to ``-1``.

    Args:
        slots: Integer virtual slots, any shape.
        placement: The group's ownership geometry, or ``None`` for a
            replicated group (every valid slot is rank 0's).

    Returns:
        ``int64`` owner ranks of ``slots``' shape.
    """
    if placement is None:
        return torch.zeros_like(slots, dtype=torch.int64)
    rows = placement.block_granularity
    degree = len(placement.group)
    safe = slots.to(torch.int64).clamp_min(0)
    block = safe // rows
    valid = (slots >= rows) & (block < placement.virtual_block_count)
    return torch.where(valid, (block - 1) % degree, -1)


def owned_history_rows(
    block_table_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    *,
    block_granularity: int,
    placement: CachePlacement | None,
) -> torch.Tensor:
    """Count, on the host, how many history rows of each request every rank owns.

    The query-context-parallel history gather all-gathers each request
    group's cached rows with per-rank counts, and the counts must be known on
    the host without a device sync. Ownership is a fact of the scheduler
    block, so they follow from the host block table directly: block column
    ``col`` of a request covers
    ``min(block_granularity, seq_len - col * block_granularity)`` rows and
    belongs to rank ``(block - 1) % degree``.

    Args:
        block_table_cpu: ``[requests, columns]`` host int32 block table of the
            group (virtual scheduler block ids, batch-ordered; ``0`` is a
            hole, ``-1`` ragged padding).
        seq_lens_cpu: ``[requests]`` host total history lengths (prefix plus
            this chunk's rows, all of which the prologue has written).
        block_granularity: Rows per block of ``block_table_cpu``, the group's
            scheduler block granularity. Explicit because a replicated group
            has no placement to carry it; with a placement it must equal
            ``placement.block_granularity``.
        placement: The group's ownership geometry, or ``None`` when every
            rank holds every row (one owner).

    Returns:
        ``[degree, requests]`` int64 owned row counts.

    Raises:
        ValueError: the granularity is not positive or not the placement's;
            the table does not cover a history; or it has a hole below a
            request's length, so some rows have no owner and the gather could
            not reconstruct them.
    """
    if block_table_cpu.dim() != 2 or seq_lens_cpu.dim() != 1:
        raise ValueError("owned_history_rows takes a [requests, columns] table")
    if block_table_cpu.shape[0] != seq_lens_cpu.numel():
        raise ValueError(
            f"block table has {block_table_cpu.shape[0]} rows for "
            f"{seq_lens_cpu.numel()} requests"
        )
    if block_granularity <= 0:
        raise ValueError(f"block granularity must be positive, got {block_granularity}")
    if placement is not None and placement.block_granularity != block_granularity:
        raise ValueError(
            f"a block table of {block_granularity}-row blocks does not match "
            f"the placement {placement}"
        )
    degree = 1 if placement is None else len(placement.group)
    requests, columns = block_table_cpu.shape
    seq_lens = seq_lens_cpu.to(torch.int64)
    if columns * block_granularity < int(seq_lens.max().item() if requests else 0):
        raise ValueError(
            f"block table of {columns} blocks x {block_granularity} does not "
            f"cover a history of {int(seq_lens.max().item())} rows"
        )
    starts = torch.arange(columns, dtype=torch.int64) * block_granularity
    tokens = (seq_lens.unsqueeze(1) - starts.unsqueeze(0)).clamp_(0, block_granularity)
    block = block_table_cpu.to(torch.int64)
    if placement is None:
        owner = torch.where(block > 0, 0, degree)
    else:
        valid = (block > 0) & (block < placement.virtual_block_count)
        owner = torch.where(valid, (block - 1) % degree, degree)
    # Column ``degree`` is the sink for blocks without an owner.
    counts = torch.zeros((requests, degree + 1), dtype=torch.int64)
    counts.scatter_add_(1, owner, tokens)
    if int(counts[:, degree].sum().item()):
        raise ValueError(
            "history block table has holes below the request lengths; the "
            "query-context-parallel gather needs an owner for every row"
        )
    return counts[:, :degree].transpose(0, 1).contiguous()


@dataclass(frozen=True)
class PrefixChunkSegments:
    """The prefix-chunk loops' rows, cut into runs of one scheduler block.

    A segment is the rows one loop reads of one request inside one block:
    consecutive in the loops' rows, in virtual slots and in the owner's
    physical slots, so ownership and addresses are per segment, not per row.
    Segments tile the loops' rows in order (loop-major, then request-major,
    then position); a loop bound or a prefix end can cut a block, so a
    segment holds at most ``block_granularity`` rows. Host ``int64`` numpy
    arrays of ``[segments]`` unless noted.

    Attributes:
        loop: Prefix-chunk loop of the segment.
        row: Its first row among every loop's rows, loops in order (the
            concatenation of ``chunk_kv_indices_list``).
        rows: Row count, positive.
        owner: Owner rank's position in the placement group.
        local_slot: The owner's physical slot of its first row.
        loop_rows: ``[num_chunks]`` rows of every loop.
    """

    loop: np.ndarray
    row: np.ndarray
    rows: np.ndarray
    owner: np.ndarray
    local_slot: np.ndarray
    loop_rows: np.ndarray


def prefix_chunk_segments(
    block_table_cpu: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    *,
    chunk_len: int,
    num_chunks: int,
    placement: CachePlacement,
) -> PrefixChunkSegments:
    """Cut, on the host, every prefix-chunk loop's rows at scheduler blocks.

    Prefix-chunk loop ``j`` reads rows ``[j * chunk_len, (j + 1) * chunk_len)``
    of every request's cached prefix, request-major
    (``chunk.get_chunks_paged``), and row ``p`` of a request lies in block
    column ``p // block_granularity`` of its block table. The work is per
    segment, so a long prefix costs its block count, not its row count.

    Args:
        block_table_cpu: ``[requests, columns]`` host int32 block table of the
            group (virtual scheduler block ids of ``block_granularity`` rows,
            batch-ordered; ``0`` is a hole, ``-1`` ragged padding).
        prefix_lens_cpu: ``[requests]`` host cached-prefix lengths.
        chunk_len: Prefix rows every request contributes to one loop.
        num_chunks: The forward's prefix-chunk loop count.
        placement: The sharded group's ownership geometry.

    Returns:
        The segments.

    Raises:
        ValueError: the chunking is invalid; the table does not cover a
            prefix; or a prefix row lies in a block without an owner, so the
            gather could not rebuild it.
    """
    if block_table_cpu.dim() != 2 or prefix_lens_cpu.dim() != 1:
        raise ValueError("prefix_chunk_segments takes a [requests, columns] table")
    if block_table_cpu.shape[0] != prefix_lens_cpu.numel():
        raise ValueError(
            f"block table has {block_table_cpu.shape[0]} rows for "
            f"{prefix_lens_cpu.numel()} requests"
        )
    if chunk_len <= 0 or num_chunks < 0:
        raise ValueError(f"invalid prefix chunking: {num_chunks} x {chunk_len}")
    # Host arithmetic in numpy, a few passes over the segments: torch
    # dispatch would cost more than the arithmetic, once per forward.
    degree = len(placement.group)
    granularity = placement.block_granularity
    requests = block_table_cpu.shape[0]
    prefix_lens = prefix_lens_cpu.numpy().astype(np.int64)
    max_prefix = int(prefix_lens.max()) if requests else 0
    # Only the blocks holding prefix rows matter; the table may be wider.
    if block_table_cpu.shape[1] * granularity < max_prefix:
        raise ValueError(
            f"block table of {block_table_cpu.shape[1]} blocks x {granularity} "
            f"does not cover a prefix of {max_prefix} rows"
        )
    # Every (loop, request) pair, loop-major: its rows [start, end) of the
    # request's prefix. The values every segment of a pair inherits are
    # stacked so that one repeat spreads them: numpy's per-call cost, not
    # its arithmetic, dominates a short history.
    start = np.arange(num_chunks, dtype=np.int64)[:, None] * chunk_len
    end = np.minimum(start + chunk_len, prefix_lens[None, :])
    lens = np.maximum(end - start, 0)
    loop_rows = lens.sum(axis=1)
    pairs = np.empty((5, lens.size), dtype=np.int64)
    loop, request, first_column, shift, base = pairs
    loop.reshape(lens.shape)[...] = np.arange(num_chunks)[:, None]
    request.reshape(lens.shape)[...] = np.arange(requests)
    first_column.reshape(lens.shape)[...] = start // granularity
    lens, end = lens.ravel(), end.ravel()
    # The pair's first row among the loops' rows.
    np.cumsum(lens, out=base)
    base -= lens
    # Block column c of the pair starts at row c * granularity + shift.
    np.subtract(base, start.repeat(requests), out=shift)
    counts = np.where(lens > 0, (end - 1) // granularity - first_column + 1, 0)
    # Less the segments of the earlier pairs, so that adding a segment's
    # index gives its column.
    first_column -= np.cumsum(counts) - counts
    loop, request, first_column, shift, base = np.repeat(pairs, counts, axis=1)
    column = np.arange(loop.size, dtype=np.int64) + first_column
    column_row = column * granularity + shift
    # A segment starts at its column's first row; only a pair's first
    # segment can start later, at the pair's first row (base).
    row = np.maximum(column_row, base)
    # Segments tile the loops' rows: each ends where the next starts.
    rows = np.empty_like(row)
    if rows.size:
        np.subtract(row[1:], row[:-1], out=rows[:-1])
        rows[-1] = int(loop_rows.sum()) - row[-1]
    previous = np.subtract(block_table_cpu.numpy()[request, column], 1, dtype=np.int64)
    # Blocks 1 .. virtual_block_count - 1 have an owner: as unsigned, a hole
    # (0) or padding (-1) wraps past the bound.
    if not (previous.view(np.uint64) < placement.virtual_block_count - 1).all():
        raise ValueError(
            "prefix block table has holes below the prefix lengths; the "
            "prefix-chunk gather needs an owner for every row"
        )
    local_block, owner = np.divmod(previous, degree)
    # The owner's physical slot: past its null block 0, plus the first row's
    # offset in the block.
    local_slot = local_block + 1
    local_slot *= granularity
    local_slot += row - column_row
    return PrefixChunkSegments(
        loop=loop,
        row=row,
        rows=rows,
        owner=owner,
        local_slot=local_slot,
        loop_rows=loop_rows,
    )
