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
    # Table and logits geometry follows the batch; runtime so every batch
    # shape shares one binary.
    ROWS,
    COLS,
    STRIDE,
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


@triton.jit(do_not_specialize=["WIDTH", "STRIDE", "COL_STRIDE", "COLS"])
def _mask_scores(
    Logits,
    Positions,
    Lengths,
    Causal,
    # Table and logits geometry follows the batch; runtime so every batch
    # shape shares one binary.
    WIDTH,
    STRIDE,
    COL_STRIDE,
    COLS,
    PAGE: tl.constexpr,
    INITIAL: tl.constexpr,
    LOCAL: tl.constexpr,
    MASK_TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    length = tl.load(Lengths + row)
    # Length-aware selection never reads the tail. The full-width fallback
    # still needs it initialized, including after an empty graph replay.
    if MASK_TAIL or start < length:
        col = start + tl.arange(0, BLOCK)
        valid = (col < WIDTH) & (col < length)
        page = tl.load(Positions + row * COLS + col // PAGE, mask=valid, other=-1)
        logical = page * PAGE + col % PAGE
        causal = tl.load(Causal + row)
        score = tl.load(
            Logits + row * STRIDE + col * COL_STRIDE, mask=valid, other=-float("inf")
        )
        finite = (score == score) & (tl.abs(score) != float("inf"))
        score = tl.where(finite, score, -float("inf"))
        forced = (logical < INITIAL) | (logical >= causal - LOCAL)
        score = tl.where(valid & forced, float("inf"), score)
        tl.store(
            Logits + row * STRIDE + col * COL_STRIDE,
            tl.where(valid, score, -float("inf")),
            mask=(col < WIDTH) & (MASK_TAIL | valid),
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
    *,
    mask_tail: bool,
) -> None:
    """Mark mandatory windows; mask the tail only for full-width consumers."""
    _mask_scores[(logits.shape[0], triton.cdiv(logits.shape[1], 256))](
        logits,
        positions,
        lengths,
        causal_lens,
        logits.shape[1],
        logits.stride(0),
        logits.stride(1),
        positions.shape[1],
        page_size,
        initial_tokens,
        local_tokens,
        mask_tail,
        256,
    )


@triton.jit(
    do_not_specialize=[
        "WIDTH",
        "LOGIT_STRIDE",
        "LOGIT_COL_STRIDE",
        "COLS",
        "OFFSET_STRIDE",
        "VALUE_STRIDE",
    ]
)
def _gather_candidates(
    Offsets,
    Logits,
    Positions,
    Lengths,
    Values,
    Records,
    TOPK: tl.constexpr,
    # Table and logits geometry follows the batch; runtime so every batch
    # shape shares one binary.
    LOGIT_STRIDE,
    LOGIT_COL_STRIDE,
    WIDTH,
    COLS,
    OFFSET_STRIDE,
    VALUE_STRIDE,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = tl.load(Offsets + row * OFFSET_STRIDE + col, mask=col < TOPK, other=-1)
    end = WIDTH
    if Lengths is not None:
        end = tl.minimum(end, tl.load(Lengths + row))
    valid = (col < TOPK) & (offset >= 0) & (offset < end)
    if Values is None:
        score = tl.load(
            Logits + row * LOGIT_STRIDE + offset * LOGIT_COL_STRIDE,
            mask=valid,
            other=-float("inf"),
        )
    else:
        score = tl.load(
            Values + row * VALUE_STRIDE + col, mask=valid, other=-float("inf")
        )
    page = tl.load(Positions + row * COLS + offset // PAGE, mask=valid, other=-1)
    valid = valid & (score > -float("inf"))
    logical = page * PAGE + offset % PAGE
    base = (row * TOPK + col) * 2
    tl.store(Records + base, tl.where(valid, logical, -1), mask=col < TOPK)
    tl.store(
        Records + base + 1,
        tl.where(valid, score, -float("inf")).to(tl.int32, bitcast=True),
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
    *,
    lengths: torch.Tensor | None = None,
    selected_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local offsets to interleaved logical-position/FP32-score views.

    Optional lengths bound reads when the logits tail is uninitialized.
    Optional selected_scores reuse values already returned by the selector.
    The two outputs share contiguous int32 records for candidate exchange.
    """
    records = torch.empty((*offsets.shape, 2), device=offsets.device, dtype=torch.int32)
    _gather_candidates[(offsets.shape[0], triton.cdiv(offsets.shape[1], 256))](
        offsets,
        logits,
        positions,
        lengths,
        selected_scores,
        records,
        offsets.shape[1],
        logits.stride(0),
        logits.stride(1),
        logits.shape[1],
        positions.shape[1],
        offsets.stride(0),
        0 if selected_scores is None else selected_scores.stride(0),
        page_size,
        256,
    )
    return records[..., 0], records[..., 1].view(torch.float32)
