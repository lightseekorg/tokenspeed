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

"""Packed candidate views and rank-major selection for DCP collectives."""

import torch
from tokenspeed_kernel._triton import tl, triton

# Number of candidate scores processed by one Triton program.
_SCORE_BLOCK_SIZE = 256
# Bound each gather iteration's vector width; larger top-k values use more iterations.
_MAX_GATHER_BLOCK_SIZE = 2048


def pack_index_candidates(offsets: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Return contiguous int32 [rows, candidates, 2], preserving score bits.

    Inputs are int32 positions and FP32 scores of the same shape. Reuse their
    storage when the producer already emitted interleaved candidate views.
    """
    rows, width = offsets.shape
    if (
        offsets.stride() == (width * 2, 2)
        and scores.stride() == offsets.stride()
        and scores.untyped_storage().data_ptr() == offsets.untyped_storage().data_ptr()
        and scores.storage_offset() == offsets.storage_offset() + 1
    ):
        return offsets.as_strided((rows, width, 2), (width * 2, 2, 1))
    return torch.stack((offsets, scores.view(torch.int32)), dim=-1)


@triton.jit(do_not_specialize=["rows"])
def _extract_rank_major_scores(
    records,
    scores,
    starts,
    ends,
    rows,
    WIDTH: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_BOUNDS: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    if WRITE_BOUNDS:
        if tl.program_id(1) == 0:
            tl.store(starts + row, 0)
            tl.store(ends + row, DEGREE * WIDTH)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source, local = col // WIDTH, col % WIDTH
    address = ((source * rows + row) * WIDTH + local) * 2
    position = tl.load(records + address, col < DEGREE * WIDTH, other=-1)
    bits = tl.load(records + address + 1, col < DEGREE * WIDTH, other=0)
    score = bits.to(tl.float32, bitcast=True)
    valid = (position >= 0) & (score == score) & (score > -float("inf"))
    tl.store(
        scores + row * DEGREE * WIDTH + col,
        tl.where(valid, score, -float("inf")),
        col < DEGREE * WIDTH,
    )


@triton.jit(do_not_specialize=["rows"])
def _gather_selected_candidates(
    records,
    order,
    values,
    positions,
    counts,
    rows,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    count = tl.full((), 0, tl.int32)
    for start in range(0, TOPK, BLOCK):
        col = start + tl.arange(0, BLOCK)
        selected = tl.load(order + row * TOPK + col, col < TOPK, other=0)
        score = tl.load(values + row * TOPK + col, col < TOPK, other=-float("inf"))
        source, local = selected // WIDTH, selected % WIDTH
        position = tl.load(
            records + ((source * rows + row) * WIDTH + local) * 2, col < TOPK, other=-1
        )
        valid = (col < TOPK) & (score > -float("inf"))
        tl.store(
            positions + row * TOPK + col, tl.where(valid, position, -1), col < TOPK
        )
        count += tl.sum(valid.to(tl.int32))
    tl.store(counts + row, count)


@triton.jit(do_not_specialize=["rows"])
def _select_through_threshold(
    records,
    scores,
    order,
    positions,
    counts,
    rows,
    COLUMNS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Any exact top-k has the k-th largest score as minimum and keeps all above it.
    row = tl.program_id(0).to(tl.int64)
    threshold = tl.full((), float("inf"), tl.float32)
    for start in range(0, TOPK, TOPK_BLOCK):
        col = start + tl.arange(0, TOPK_BLOCK)
        selected = tl.load(order + row * TOPK + col, col < TOPK, other=-1)
        score = tl.load(
            scores + row * COLUMNS + selected,
            (col < TOPK) & (selected >= 0),
            other=float("inf"),
        )
        threshold = tl.minimum(threshold, tl.min(score, axis=0))
    above = tl.full((), 0, tl.int32)
    for start in range(0, TOPK, TOPK_BLOCK):
        col = start + tl.arange(0, TOPK_BLOCK)
        selected = tl.load(order + row * TOPK + col, col < TOPK, other=-1)
        score = tl.load(
            scores + row * COLUMNS + selected,
            (col < TOPK) & (selected >= 0),
            other=-float("inf"),
        )
        above += tl.sum((score > threshold).to(tl.int32))
    # Keep scores above it plus the first ties, in candidate order, so all ranks agree.
    count = tl.full((), 0, tl.int32)
    tied = tl.full((), 0, tl.int32)
    for start in range(0, COLUMNS, BLOCK):
        col = start + tl.arange(0, BLOCK)
        score = tl.load(
            scores + row * COLUMNS + col, col < COLUMNS, other=-float("inf")
        )
        tie = (col < COLUMNS) & (score == threshold)
        tie_rank = tied + tl.cumsum(tie.to(tl.int32), axis=0) - 1
        keep = ((score > threshold) | (tie & (tie_rank < TOPK - above))) & (
            score > -float("inf")
        )
        source, local = col // WIDTH, col % WIDTH
        position = tl.load(
            records + ((source * rows + row) * WIDTH + local) * 2, keep, other=-1
        )
        slot = count + tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(positions + row * TOPK + slot, position, keep)
        count += tl.sum(keep.to(tl.int32))
        tied += tl.sum(tie.to(tl.int32))
    for start in range(0, TOPK, TOPK_BLOCK):
        col = start + tl.arange(0, TOPK_BLOCK)
        tl.store(positions + row * TOPK + col, -1, (col >= count) & (col < TOPK))
    tl.store(counts + row, count)


@triton.jit(do_not_specialize=["rows"])
def _select_source_through_threshold(
    records,
    scores,
    order,
    positions,
    counts,
    rows,
    WIDTH: tl.constexpr,
    DEGREE: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    # _select_through_threshold's selection, one program per (row, source
    # rank): each counts the kept candidates of the sources before its own
    # and writes its own source's at that offset.
    row = tl.program_id(0).to(tl.int64)
    source = tl.program_id(1)
    COLUMNS: tl.constexpr = DEGREE * WIDTH
    threshold = tl.full((), float("inf"), tl.float32)
    for start in range(0, TOPK, TOPK_BLOCK):
        col = start + tl.arange(0, TOPK_BLOCK)
        selected = tl.load(order + row * TOPK + col, col < TOPK, other=-1)
        score = tl.load(
            scores + row * COLUMNS + selected,
            (col < TOPK) & (selected >= 0),
            other=float("inf"),
        )
        threshold = tl.minimum(threshold, tl.min(score, axis=0))
    local = tl.arange(0, WIDTH)
    above = tl.full((), 0, tl.int32)
    above_before = tl.full((), 0, tl.int32)
    tied_before = tl.full((), 0, tl.int32)
    for other in tl.static_range(DEGREE):
        score = tl.load(scores + row * COLUMNS + other * WIDTH + local)
        source_above = tl.sum((score > threshold).to(tl.int32), axis=0)
        above += source_above
        if other < source:
            above_before += source_above
            tie = (score == threshold) & (score > -float("inf"))
            tied_before += tl.sum(tie.to(tl.int32), axis=0)
    quota = TOPK - above
    score = tl.load(scores + row * COLUMNS + source * WIDTH + local)
    live = score > -float("inf")
    tie = (score == threshold) & live
    tie_rank = tied_before + tl.cumsum(tie.to(tl.int32), axis=0) - 1
    keep = ((score > threshold) | (tie & (tie_rank < quota))) & live
    before = above_before + tl.minimum(tied_before, tl.maximum(quota, 0))
    slot = before + tl.cumsum(keep.to(tl.int32), axis=0) - 1
    position = tl.load(
        records + ((source * rows + row) * WIDTH + local) * 2, keep, other=-1
    )
    tl.store(positions + row * TOPK + slot, position, keep)
    if source == DEGREE - 1:
        count = before + tl.sum(keep.to(tl.int32), axis=0)
        for start in range(0, TOPK, TOPK_BLOCK):
            col = start + tl.arange(0, TOPK_BLOCK)
            tl.store(positions + row * TOPK + col, -1, (col >= count) & (col < TOPK))
        tl.store(counts + row, count)


def rank_major_scores(records: torch.Tensor) -> torch.Tensor:
    """FP32 [rows, shards * candidates] scores of int32 [shards, rows,
    candidates, 2] records, rank-major; invalid candidates score ``-inf``."""
    degree, rows, width, _ = records.shape
    scores = torch.empty(
        (rows, degree * width), device=records.device, dtype=torch.float32
    )
    if rows:
        _extract_rank_major_scores[
            (rows, triton.cdiv(degree * width, _SCORE_BLOCK_SIZE))
        ](
            records,
            scores,
            scores,
            scores,
            rows,
            WIDTH=width,
            DEGREE=degree,
            BLOCK=_SCORE_BLOCK_SIZE,
        )
    return scores


def rank_major_scores_with_bounds(
    records: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`rank_major_scores` and int32 [rows] row bounds over all of
    them (zero starts, full-width ends), written by the same launch."""
    degree, rows, width, _ = records.shape
    scores = torch.empty(
        (rows, degree * width), device=records.device, dtype=torch.float32
    )
    starts = torch.empty((rows,), device=records.device, dtype=torch.int32)
    ends = torch.empty((rows,), device=records.device, dtype=torch.int32)
    if rows:
        _extract_rank_major_scores[
            (rows, triton.cdiv(degree * width, _SCORE_BLOCK_SIZE))
        ](
            records,
            scores,
            starts,
            ends,
            rows,
            WIDTH=width,
            DEGREE=degree,
            BLOCK=_SCORE_BLOCK_SIZE,
            WRITE_BOUNDS=True,
        )
    return scores, starts, ends


def select_through_threshold(
    records: torch.Tensor, scores: torch.Tensor, order: torch.Tensor, *, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic top-k positions from any exact top-k ``order`` of
    ``scores`` (:func:`rank_major_scores` of ``records``).

    Keeps every candidate scoring above the k-th largest score and, among the
    candidates tied at it, the first ones in candidate order. Returns int32
    [rows, topk] positions in candidate order with -1 after the valid ones,
    and int32 [rows] counts of valid positions.
    """
    _, rows, width, _ = records.shape
    positions = torch.empty((rows, topk), device=records.device, dtype=torch.int32)
    counts = torch.empty((rows,), device=records.device, dtype=torch.int32)
    if rows:
        _select_through_threshold[(rows,)](
            records,
            scores,
            order,
            positions,
            counts,
            rows,
            COLUMNS=scores.shape[1],
            WIDTH=width,
            TOPK=topk,
            TOPK_BLOCK=triton.next_power_of_2(min(topk, _MAX_GATHER_BLOCK_SIZE)),
            BLOCK=_MAX_GATHER_BLOCK_SIZE,
        )
    return positions, counts


def select_through_threshold_by_source(
    records: torch.Tensor, scores: torch.Tensor, order: torch.Tensor, *, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`select_through_threshold`, bit for bit, with one program per
    row and source rank instead of one per row.

    Each program rereads its row's scores to count the sources before it, so
    this pays off while ``rows * shards`` programs fit a few waves of the
    device; the caller picks between the two."""
    degree, rows, width, _ = records.shape
    if width & (width - 1):
        raise ValueError("per-source selection needs a power-of-two width")
    positions = torch.empty((rows, topk), device=records.device, dtype=torch.int32)
    counts = torch.empty((rows,), device=records.device, dtype=torch.int32)
    if rows:
        _select_source_through_threshold[(rows, degree)](
            records,
            scores,
            order,
            positions,
            counts,
            rows,
            WIDTH=width,
            DEGREE=degree,
            TOPK=topk,
            TOPK_BLOCK=triton.next_power_of_2(min(topk, _MAX_GATHER_BLOCK_SIZE)),
            num_warps=4,
        )
    return positions, counts


def merge_packed_index_candidates(
    records: torch.Tensor, *, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select global logical positions from int32 [shards, rows, candidates, 2].

    The contiguous records keep the collective's source-rank layout. Filtering
    and selection use the same rank/candidate order as a row-major exchange.
    Returns int32 [rows, topk] positions and [rows] counts; padding is -1.
    """
    _, rows, width, _ = records.shape
    positions = torch.empty((rows, topk), device=records.device, dtype=torch.int32)
    counts = torch.empty((rows,), device=records.device, dtype=torch.int32)
    if rows == 0:
        return positions, counts
    values, order = torch.topk(rank_major_scores(records), topk, dim=-1)
    _gather_selected_candidates[(rows,)](
        records,
        order,
        values,
        positions,
        counts,
        rows,
        WIDTH=width,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(min(topk, _MAX_GATHER_BLOCK_SIZE)),
    )
    return positions, counts
