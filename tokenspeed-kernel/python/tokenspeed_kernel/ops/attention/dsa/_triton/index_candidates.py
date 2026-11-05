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
from tokenspeed_kernel.platform import current_platform


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
    # 64-bit row offsets: row * stride overflows int32 on a 1M-wide decode table.
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(Requests + row)
    length = tl.load(Causal + row)
    col = tl.arange(0, BLOCK)
    page = tl.load(
        Table + req.to(tl.int64) * STRIDE + col,
        mask=(req >= 0) & (req < ROWS) & (col < COLS),
        other=-1,
    )
    valid = (page > 0) & (col * PAGE < length)
    index = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(Pages + row * COLS + index, page, mask=valid)
    tl.store(Positions + row * COLS + index, col, mask=valid)
    count = tl.sum(tl.where(valid, tl.minimum(PAGE, length - col * PAGE), 0))
    tl.store(Lengths + row, count)


@triton.jit(do_not_specialize=["ROWS", "COLS", "STRIDE", "OUT_COLS"])
def _compact_bounded(
    Table,
    Requests,
    Causal,
    Pages,
    Positions,
    Lengths,
    Overflow,
    # Runtime, unspecialized batch geometry: every batch shape shares one binary.
    ROWS,
    COLS,
    STRIDE,
    OUT_COLS,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Keep only the first OUT_COLS valid pages: Lengths covers them, Overflow the rest.
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(Requests + row)
    length = tl.load(Causal + row)
    # Columns from ceil(length / PAGE) on start at or past the causal length.
    end = tl.minimum(COLS, (tl.maximum(length, 0) + PAGE - 1) // PAGE)
    end = tl.where((req >= 0) & (req < ROWS), end, 0)
    table = Table + req.to(tl.int64) * STRIDE
    found = 0
    tokens = 0
    for start in range(0, end, BLOCK):
        col = start + tl.arange(0, BLOCK)
        page = tl.load(table + col, mask=col < end, other=-1)
        valid = page > 0
        index = found + tl.cumsum(valid.to(tl.int32), axis=0) - 1
        stored = valid & (index < OUT_COLS)
        tl.store(Pages + row * OUT_COLS + index, page, mask=stored)
        tl.store(Positions + row * OUT_COLS + index, col, mask=stored)
        tokens += tl.sum(
            tl.where(stored, tl.minimum(PAGE, length - col * PAGE), 0), axis=0
        )
        found += tl.sum(valid.to(tl.int32), axis=0)
    tl.store(Lengths + row, tokens)
    if Overflow is not None:
        tl.store(Overflow + row, tl.maximum(found - OUT_COLS, 0))


@triton.jit
def _mask_block(
    Logits,
    Positions,
    Causal,
    row,
    start,
    length,
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
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(Lengths + row)
    if MASK_TAIL:
        # Full-width selection reads the tail: initialize it, even on an empty replay.
        _mask_block(
            Logits,
            Positions,
            Causal,
            row,
            tl.program_id(1) * BLOCK,
            length,
            WIDTH,
            STRIDE,
            COL_STRIDE,
            COLS,
            PAGE,
            INITIAL,
            LOCAL,
            MASK_TAIL,
            BLOCK,
        )
    else:
        # Selection never reads the tail; a replay-fixed grid strides the valid prefix.
        for start in range(
            tl.program_id(1) * BLOCK, length, tl.num_programs(1) * BLOCK
        ):
            _mask_block(
                Logits,
                Positions,
                Causal,
                row,
                start,
                length,
                WIDTH,
                STRIDE,
                COL_STRIDE,
                COLS,
                PAGE,
                INITIAL,
                LOCAL,
                MASK_TAIL,
                BLOCK,
            )


# Table columns one bounded-compaction program reads per iteration.
_COMPACT_BLOCK = 1024


def _launch_compact_bounded(
    table: torch.Tensor,
    requests: torch.Tensor,
    causal_lens: torch.Tensor,
    page_size: int,
    pages: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    overflow: torch.Tensor | None,
) -> None:
    """Compact each row's pages into ``pages`` / ``positions`` (``[rows,
    columns]``, zero-filled by the caller), at most ``columns`` per row."""
    _compact_bounded[(requests.numel(),)](
        table,
        requests,
        causal_lens,
        pages,
        positions,
        lengths,
        overflow,
        table.shape[0],
        table.shape[1],
        table.stride(0),
        pages.shape[1],
        page_size,
        _COMPACT_BLOCK,
    )


def compact_index_pages(
    table: torch.Tensor,
    requests: torch.Tensor,
    causal_lens: torch.Tensor,
    page_size: int,
    *,
    columns: int | None = None,
    overflow: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact physical pages, logical page positions and local lengths.

    Inputs use one request ID and causal length per query. Table entries <= 0
    are absent pages. Outputs have fixed shapes for CUDA graph replay; lengths
    exclude invalid requests, empty shards and the unused portion of a tail.

    With ``columns`` the outputs are ``[rows, columns]`` instead of the
    table's width: a row keeps its first ``columns`` pages, its length counts
    only their tokens, and ``overflow`` (an optional ``[rows]`` int32 output)
    receives the number of pages it left out, zero when they all fit. One
    binary serves every table and output width. Without ``columns`` (decode's
    static width) the full-width kernel runs.
    """
    if columns is not None:
        if columns <= 0:
            raise ValueError(f"compact index pages need columns > 0, got {columns}")
        rows = requests.numel()
        pages = torch.zeros((rows, columns), device=table.device, dtype=torch.int32)
        positions = torch.zeros_like(pages)
        lengths = torch.empty((rows, 1), device=table.device, dtype=torch.int32)
        if overflow is not None and (
            overflow.shape != (rows,)
            or overflow.dtype != torch.int32
            or not overflow.is_contiguous()
        ):
            raise ValueError("overflow must be a contiguous [rows] int32 tensor")
        _launch_compact_bounded(
            table,
            requests,
            causal_lens,
            page_size,
            pages,
            positions,
            lengths,
            overflow,
        )
        return pages, positions, lengths
    if overflow is not None:
        raise ValueError("only pages compacted to columns can overflow")
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


# Scores per iteration; the prefix mask has few programs per row, so it loads wider.
_TAIL_MASK_BLOCK = 256
_PREFIX_MASK_BLOCK = 1024


def _launch_mask_scores(
    logits: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    causal_lens: torch.Tensor,
    page_size: int,
    initial_tokens: int,
    local_tokens: int,
    *,
    mask_tail: bool,
    programs: int,
    block: int,
    num_warps: int = 4,
) -> None:
    """Launch ``programs`` mask programs per row, ``block`` scores at a time."""
    _mask_scores[(logits.shape[0], programs)](
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
        block,
        num_warps=num_warps,
    )


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
    rows = logits.shape[0]
    if mask_tail:
        block = _TAIL_MASK_BLOCK
        programs = triton.cdiv(logits.shape[1], block)
    else:
        # Size the grid to the device, not the width; static shapes keep replay valid.
        block = _PREFIX_MASK_BLOCK
        target = triton.cdiv(8 * current_platform().sm_count, max(rows, 1))
        programs = min(max(target, 1), triton.cdiv(logits.shape[1], block))
    _launch_mask_scores(
        logits,
        positions,
        lengths,
        causal_lens,
        page_size,
        initial_tokens,
        local_tokens,
        mask_tail=mask_tail,
        programs=programs,
        block=block,
    )


@triton.jit(
    do_not_specialize=[
        "WIDTH",
        "LOGIT_STRIDE",
        "LOGIT_COL_STRIDE",
        "COLS",
        "OFFSET_STRIDE",
        "VALUE_STRIDE",
        "SCORE_STRIDE",
    ]
)
def _gather_candidates(
    Offsets,
    Logits,
    Positions,
    Lengths,
    Values,
    Records,
    Scores,
    TOPK: tl.constexpr,
    # Table and logits geometry follows the batch; runtime so every batch
    # shape shares one binary.
    LOGIT_STRIDE,
    LOGIT_COL_STRIDE,
    WIDTH,
    COLS,
    OFFSET_STRIDE,
    VALUE_STRIDE,
    SCORE_STRIDE,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
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
    if Scores is None:
        base = (row * TOPK + col) * 2
        tl.store(Records + base, tl.where(valid, logical, -1), mask=col < TOPK)
        tl.store(
            Records + base + 1,
            tl.where(valid, score, -float("inf")).to(tl.int32, bitcast=True),
            mask=col < TOPK,
        )
    else:
        tl.store(
            Records + row * TOPK + col, tl.where(valid, logical, -1), mask=col < TOPK
        )
        tl.store(
            Scores + row * SCORE_STRIDE + col,
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
    *,
    lengths: torch.Tensor | None = None,
    selected_scores: torch.Tensor | None = None,
    scores_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local offsets to interleaved logical-position/FP32-score views.

    Optional lengths bound reads when the logits tail is uninitialized.
    Optional selected_scores reuse values already returned by the selector.
    The two outputs share contiguous int32 records for candidate exchange,
    unless ``scores_out`` (FP32 ``[rows, topk]`` with contiguous rows, such
    as a collective's send buffer) takes the scores: the positions are then
    their own contiguous int32 matrix and ``scores_out`` is returned.
    """
    if scores_out is None:
        records = torch.empty(
            (*offsets.shape, 2), device=offsets.device, dtype=torch.int32
        )
    else:
        if (
            scores_out.shape != offsets.shape
            or scores_out.dtype != torch.float32
            or scores_out.stride(1) != 1
        ):
            raise ValueError("candidate scores need an FP32 [rows, topk] output")
        records = torch.empty(offsets.shape, device=offsets.device, dtype=torch.int32)
    _gather_candidates[(offsets.shape[0], triton.cdiv(offsets.shape[1], 256))](
        offsets,
        logits,
        positions,
        lengths,
        selected_scores,
        records,
        scores_out,
        offsets.shape[1],
        logits.stride(0),
        logits.stride(1),
        logits.shape[1],
        positions.shape[1],
        offsets.stride(0),
        0 if selected_scores is None else selected_scores.stride(0),
        0 if scores_out is None else scores_out.stride(0),
        page_size,
        256,
    )
    if scores_out is not None:
        return records, scores_out
    return records[..., 0], records[..., 1].view(torch.float32)
