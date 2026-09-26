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

"""Compact owned Index-K pages without changing logical selection positions."""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _compact(
    Table,
    Requests,
    Causal,
    Pages,
    Positions,
    Lengths,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(Requests + row)
    length = tl.load(Causal + row)
    col = tl.arange(0, BLOCK)
    page = tl.load(
        Table + req * STRIDE + col,
        mask=(req >= 0) & (req < ROWS) & (col < COLS),
        other=-1,
    )
    valid = (page > 0) & (col * PAGE < length)
    index = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(Pages + row * COLS + index, page, mask=valid)
    tl.store(Positions + row * COLS + index, col, mask=valid)
    count = tl.sum(tl.where(valid, tl.minimum(PAGE, length - col * PAGE), 0))
    tl.store(Lengths + row, count)


@triton.jit
def _mask_scores(
    Logits,
    Positions,
    Lengths,
    Causal,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    COLS: tl.constexpr,
    PAGE: tl.constexpr,
    INITIAL: tl.constexpr,
    LOCAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Lengths + row)
    valid = (col < WIDTH) & (col < length)
    page = tl.load(Positions + row * COLS + col // PAGE, mask=valid, other=-1)
    logical = page * PAGE + col % PAGE
    causal = tl.load(Causal + row)
    score = tl.load(Logits + row * STRIDE + col, mask=valid, other=-float("inf"))
    finite = (score == score) & (tl.abs(score) != float("inf"))
    score = tl.where(finite, score, -float("inf"))
    forced = (logical < INITIAL) | (logical >= causal - LOCAL)
    score = tl.where(valid & forced, float("inf"), score)
    tl.store(
        Logits + row * STRIDE + col,
        tl.where(valid, score, -float("inf")),
        mask=col < WIDTH,
    )


def compact_index_pages(
    table: torch.Tensor,
    requests: torch.Tensor,
    causal_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact physical pages, logical page positions and local lengths.

    Inputs use one request ID and causal length per query. Table entries <= 0
    are absent pages. Outputs have fixed shapes for CUDA graph replay; lengths
    exclude invalid requests, empty shards and the unused portion of a tail.
    """
    shape = (requests.numel(), table.shape[1])
    pages = torch.zeros(shape, device=table.device, dtype=torch.int32)
    positions = torch.zeros_like(pages)
    lengths = torch.empty((requests.numel(), 1), device=table.device, dtype=torch.int32)
    _compact[(requests.numel(),)](
        table,
        requests,
        causal_lens,
        pages,
        positions,
        lengths,
        table.shape[0],
        table.shape[1],
        table.stride(0),
        page_size,
        triton.next_power_of_2(table.shape[1]),
    )
    return pages, positions, lengths


def mask_index_scores(
    logits: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    causal_lens: torch.Tensor,
    page_size: int,
    initial_tokens: int,
    local_tokens: int,
) -> None:
    """Mask padding and mark mandatory global windows in compact logits in place."""
    _mask_scores[(logits.shape[0], triton.cdiv(logits.shape[1], 256))](
        logits,
        positions,
        lengths,
        causal_lens,
        logits.shape[1],
        logits.stride(0),
        positions.shape[1],
        page_size,
        initial_tokens,
        local_tokens,
        256,
    )


@triton.jit
def _gather_candidates(
    Offsets,
    Logits,
    Positions,
    Logical,
    Scores,
    TOPK: tl.constexpr,
    LOGIT_STRIDE: tl.constexpr,
    COLS: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = tl.load(Offsets + row * TOPK + col, mask=col < TOPK, other=-1)
    valid = (col < TOPK) & (offset >= 0)
    score = tl.load(
        Logits + row * LOGIT_STRIDE + offset, mask=valid, other=-float("inf")
    )
    page = tl.load(Positions + row * COLS + offset // PAGE, mask=valid, other=-1)
    valid = valid & (score > -float("inf"))
    logical = page * PAGE + offset % PAGE
    tl.store(Logical + row * TOPK + col, tl.where(valid, logical, -1), mask=col < TOPK)
    tl.store(
        Scores + row * TOPK + col,
        tl.where(valid, score, -float("inf")),
        mask=col < TOPK,
    )


def candidate_topk_offsets(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Select unsorted candidate offsets, padding short rows with -1.

    Scores must already mask invalid/NaN candidates to -inf. Equal-score
    ordering is unspecified; candidate gathering removes masked selections.
    """
    offsets = torch.topk(
        scores, min(topk, scores.shape[1]), dim=-1, sorted=False
    ).indices
    if offsets.shape[1] < topk:
        offsets = torch.nn.functional.pad(
            offsets, (0, topk - offsets.shape[1]), value=-1
        )
    return offsets


def gather_index_candidates(
    offsets: torch.Tensor,
    logits: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map compact Top-K offsets to logical positions and gather FP32 scores."""
    logical = torch.empty(offsets.shape, device=offsets.device, dtype=torch.int32)
    scores = torch.empty(offsets.shape, device=offsets.device, dtype=torch.float32)
    _gather_candidates[(offsets.shape[0], triton.cdiv(offsets.shape[1], 256))](
        offsets,
        logits,
        positions,
        logical,
        scores,
        offsets.shape[1],
        logits.stride(0),
        positions.shape[1],
        page_size,
        256,
    )
    return logical, scores
