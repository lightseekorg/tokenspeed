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
    rows,
    WIDTH: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
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


def merge_packed_index_candidates(
    records: torch.Tensor, *, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select global logical positions from int32 [shards, rows, candidates, 2].

    The contiguous records keep the collective's source-rank layout. Filtering
    and selection use the same rank/candidate order as a row-major exchange.
    Returns int32 [rows, topk] positions and [rows] counts; padding is -1.
    """
    degree, rows, width, _ = records.shape
    scores = torch.empty(
        (rows, degree * width), device=records.device, dtype=torch.float32
    )
    positions = torch.empty((rows, topk), device=records.device, dtype=torch.int32)
    counts = torch.empty((rows,), device=records.device, dtype=torch.int32)
    if rows == 0:
        return positions, counts
    _extract_rank_major_scores[(rows, triton.cdiv(degree * width, _SCORE_BLOCK_SIZE))](
        records,
        scores,
        rows,
        WIDTH=width,
        DEGREE=degree,
        BLOCK=_SCORE_BLOCK_SIZE,
    )
    values, order = torch.topk(scores, topk, dim=-1)
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
