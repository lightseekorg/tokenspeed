# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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


from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _e2m1_decode,
    _layout,
    _planar_offset,
    _scale_bytes,
)

DEEPSEEK_V4_HEAD_DIM = 512
_SELECTED_ATTENTION_BLOCK_H = 16
_SELECTED_ATTENTION_BLOCK_N = 32
_SPARSE_MAX_SPLITS = 8
_GATHER_SELECTED_ROWS = 16


@triton.jit
def _selected_attention_init(
    sink_ptr,
    heads,
    head_mask,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Online-softmax state for BLOCK_H heads: (running max, denominator, accumulator).

    A whole-row program starts from its heads' sinks; a span program (SPLIT)
    starts empty and the span combine adds the sink.
    """
    if sink_ptr is not None and not SPLIT:
        max_logit = tl.load(sink_ptr + heads, mask=head_mask, other=0.0).to(tl.float32)
        denominator = tl.full((BLOCK_H,), 1.0, tl.float32)
    else:
        max_logit = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        denominator = tl.zeros((BLOCK_H,), tl.float32)
    return max_logit, denominator, tl.zeros((BLOCK_H, HEAD_DIM), tl.float32)


@triton.jit
def _selected_attention_step(
    q, kv, valid, max_logit, denominator, accumulator, softmax_scale
):
    """Fold one tile of shared K/V rows into the state.

    The row is both key and value, so both products run on tensor cores with
    the heads as the M dimension: S = Q K^T, then O += P K.
    """
    logits = tl.dot(q, tl.trans(kv)) * softmax_scale
    logits = tl.where(valid[None, :], logits, -float("inf"))
    next_max = tl.maximum(max_logit, tl.max(logits, axis=1))
    # Empty shards/tiles must not introduce exp(-inf - -inf).
    safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
    previous_scale = tl.exp(max_logit - safe_max)
    probabilities = tl.exp(logits - safe_max[:, None])
    probabilities = tl.where(valid[None, :], probabilities, 0.0)
    denominator = denominator * previous_scale + tl.sum(probabilities, axis=1)
    accumulator = accumulator * previous_scale[:, None] + tl.dot(
        probabilities.to(kv.dtype), kv
    )
    return next_max, denominator, accumulator


@triton.jit
def _selected_attention_store(
    max_logit,
    denominator,
    accumulator,
    token_idx,
    heads,
    head_mask,
    num_heads,
    out_ptr,
    out_stride_token,
    out_stride_head,
    lse_ptr,
    part_ptr,
    part_lse_ptr,
    HEAD_DIM: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Write the output and LSE, or with SPLIT this span's partial and LSE."""
    dim = tl.arange(0, HEAD_DIM)
    logsumexp = tl.where(
        denominator > 0.0, max_logit + tl.log(denominator), -float("inf")
    )
    output = tl.where(
        denominator[:, None] > 0.0, accumulator / denominator[:, None], 0.0
    )
    if SPLIT:
        part_row = (token_idx.to(tl.int64) * num_heads + heads) * tl.num_programs(
            2
        ) + tl.program_id(2)
        tl.store(part_lse_ptr + part_row, logsumexp, mask=head_mask)
        tl.store(
            part_ptr + part_row[:, None] * HEAD_DIM + dim[None, :],
            output,
            mask=head_mask[:, None],
        )
    else:
        if lse_ptr is not None:
            tl.store(
                lse_ptr + token_idx.to(tl.int64) * num_heads + heads,
                logsumexp,
                mask=head_mask,
            )
        tl.store(
            out_ptr
            + token_idx * out_stride_token
            + heads[:, None] * out_stride_head
            + dim[None, :],
            output,
            mask=head_mask[:, None],
        )


@triton.jit(do_not_specialize=["num_kv_rows", "indices_stride_token", "split_cols"])
def _dsv4_sparse_attention_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lens_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    num_heads,
    q_stride_token,
    q_stride_head,
    kv_stride_row,
    indices_stride_token,
    out_stride_token,
    out_stride_head,
    softmax_scale,
    num_kv_rows,
    part_ptr,
    part_lse_ptr,
    split_cols,
    TOPK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One query token x BLOCK_H heads x one span of selected rows per program.

    A program loads each selected row of its span once for all its heads. With
    SPLIT, the selected rows are cut into ``split_cols`` spans whose partials
    ``_dsv4_sparse_attention_combine_kernel`` merges with the sink.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < num_heads
    dim = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + token_idx * q_stride_token
        + heads[:, None] * q_stride_head
        + dim[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    max_logit, denominator, accumulator = _selected_attention_init(
        sink_ptr, heads, head_mask, BLOCK_H, HEAD_DIM, SPLIT
    )
    valid_len = tl.minimum(tl.maximum(tl.load(lens_ptr + token_idx), 0), TOPK)
    span_begin = tl.program_id(2) * split_cols
    span_end = tl.minimum(valid_len, span_begin + split_cols)
    offsets = tl.arange(0, BLOCK_N)
    for start in range(span_begin, span_end, BLOCK_N):
        cols = start + offsets
        valid = cols < span_end
        rows = tl.load(
            indices_ptr + token_idx * indices_stride_token + cols,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (rows >= 0) & (rows < num_kv_rows)
        rows = tl.where(valid, rows, 0)
        kv = tl.load(
            kv_ptr + rows[:, None] * kv_stride_row + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        max_logit, denominator, accumulator = _selected_attention_step(
            q, kv, valid, max_logit, denominator, accumulator, softmax_scale
        )
    _selected_attention_store(
        max_logit,
        denominator,
        accumulator,
        token_idx,
        heads,
        head_mask,
        num_heads,
        out_ptr,
        out_stride_token,
        out_stride_head,
        lse_ptr,
        part_ptr,
        part_lse_ptr,
        HEAD_DIM,
        SPLIT,
    )


@triton.jit(do_not_specialize=["splits"])
def _dsv4_sparse_attention_combine_kernel(
    part_ptr,
    part_lse_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    num_heads,
    splits,
    out_stride_token,
    out_stride_head,
    HEAD_DIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    """Merge one (token, head)'s span partials and its sink."""
    token_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)
    split = tl.arange(0, MAX_SPLITS)
    split_mask = split < splits
    part_row = (token_idx.to(tl.int64) * num_heads + head_idx) * splits + split
    span_lse = tl.load(part_lse_ptr + part_row, mask=split_mask, other=-float("inf"))
    max_logit = tl.max(span_lse, axis=0)
    if sink_ptr is not None:
        sink = tl.load(sink_ptr + head_idx).to(tl.float32)
        max_logit = tl.maximum(max_logit, sink)
    safe_max = tl.where(max_logit == -float("inf"), 0.0, max_logit)
    weights = tl.where(split_mask, tl.exp(span_lse - safe_max), 0.0)
    denominator = tl.sum(weights, axis=0)
    if sink_ptr is not None:
        denominator += tl.exp(sink - safe_max)
    dim = tl.arange(0, HEAD_DIM)
    parts = tl.load(
        part_ptr + part_row[:, None] * HEAD_DIM + dim[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(parts * weights[:, None], axis=0)
    output = tl.where(denominator > 0.0, merged / denominator, 0.0)
    tl.store(
        out_ptr + token_idx * out_stride_token + head_idx * out_stride_head + dim,
        output,
    )
    if lse_ptr is not None:
        tl.store(
            lse_ptr + token_idx.to(tl.int64) * num_heads + head_idx,
            tl.where(denominator > 0.0, safe_max + tl.log(denominator), -float("inf")),
        )


def _selected_attention_spans(
    tokens: int, heads: int, width: int, device: torch.device
) -> tuple[int, int, torch.Tensor | None, torch.Tensor | None]:
    """Plan the selected-row spans of a selected-attention launch.

    Few tokens leave SMs idle, so their selected rows are cut into spans that
    fill about two programs per SM; each span keeps at least two row tiles.

    Returns:
        ``(spans, span_cols, partials, partial_lse)``: the span count (the
        launch's third grid dimension), the rows per span (a multiple of
        ``_SELECTED_ATTENTION_BLOCK_N``), and FP32 scratch for the span
        partials ``[tokens, heads, spans, 512]`` and their LSE, both None for
        one span.
    """
    programs = tokens * triton.cdiv(heads, _SELECTED_ATTENTION_BLOCK_H)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    by_occupancy = triton.cdiv(2 * sms, programs)
    by_width = max(1, width // (2 * _SELECTED_ATTENTION_BLOCK_N))
    spans = max(1, min(_SPARSE_MAX_SPLITS, by_occupancy, by_width))
    span_cols = (
        triton.cdiv(triton.cdiv(width, spans), _SELECTED_ATTENTION_BLOCK_N)
        * _SELECTED_ATTENTION_BLOCK_N
    )
    spans = triton.cdiv(width, span_cols)
    if spans == 1:
        return 1, span_cols, None, None
    partials = torch.empty(
        (tokens, heads, spans, DEEPSEEK_V4_HEAD_DIM), dtype=torch.float32, device=device
    )
    partial_lse = torch.empty(
        (tokens, heads, spans), dtype=torch.float32, device=device
    )
    return spans, span_cols, partials, partial_lse


def _combine_selected_attention_spans(
    partials: torch.Tensor,
    partial_lse: torch.Tensor,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor | None,
) -> None:
    """Merge span partials from ``_selected_attention_spans`` and the sink into ``out``/``lse``."""
    tokens, heads, spans, _ = partials.shape
    _dsv4_sparse_attention_combine_kernel[(tokens, heads)](
        partials,
        partial_lse,
        attn_sink,
        out,
        lse,
        heads,
        spans,
        out.stride(0),
        out.stride(1),
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        MAX_SPLITS=_SPARSE_MAX_SPLITS,
        num_warps=4,
    )


def triton_dsv4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run selected shared-KV attention for DeepSeek V4 geometry."""

    if q.dim() != 3 or q.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected q [tokens, heads, 512], got {tuple(q.shape)}")
    kv_2d = kv.reshape(-1, kv.shape[-1])
    if kv_2d.shape[-1] != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(f"expected kv rows of width 512, got {tuple(kv.shape)}")
    indices_2d = indices.reshape(indices.shape[0], -1).contiguous()
    lens = lens.reshape(-1).contiguous()
    if indices_2d.shape[0] != q.shape[0] or lens.shape[0] != q.shape[0]:
        raise ValueError("selected-attention metadata must have one row per query")
    if attn_sink is not None and attn_sink.numel() < q.shape[1]:
        raise ValueError("attention sink must provide one value per query head")

    output = out if out is not None else torch.empty_like(q)
    tokens, heads, width = q.shape[0], q.shape[1], indices_2d.shape[1]
    spans, span_cols, partials, partial_lse = _selected_attention_spans(
        tokens, heads, width, q.device
    )
    _dsv4_sparse_attention_kernel[
        (tokens, triton.cdiv(heads, _SELECTED_ATTENTION_BLOCK_H), spans)
    ](
        q,
        kv_2d,
        indices_2d,
        lens,
        attn_sink,
        output,
        lse,
        heads,
        q.stride(0),
        q.stride(1),
        kv_2d.stride(0),
        indices_2d.stride(0),
        output.stride(0),
        output.stride(1),
        softmax_scale,
        kv_2d.shape[0],
        partials,
        partial_lse,
        span_cols,
        TOPK=width,
        HEAD_DIM=DEEPSEEK_V4_HEAD_DIM,
        BLOCK_H=_SELECTED_ATTENTION_BLOCK_H,
        BLOCK_N=_SELECTED_ATTENTION_BLOCK_N,
        SPLIT=spans > 1,
        num_warps=4,
        num_stages=3,
    )
    if spans > 1:
        _combine_selected_attention_spans(partials, partial_lse, attn_sink, output, lse)
    return output


@triton.jit
def _dequantize_rows(
    C,
    slot,
    valid,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    QUANTIZED: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    FORMAT: tl.constexpr,
):
    """Decode the paged rows at ``slot`` [N] into FP32 [N, D]; invalid rows are zero."""
    slot = tl.where(valid, slot, 0)[:, None]
    row_valid = valid[:, None]
    d = tl.arange(0, D)[None, :]
    base = C + (slot // PAGE_ROWS) * CP
    page_row = slot % PAGE_ROWS
    ROPE: tl.constexpr = D - QUANTIZED
    DATA_BYTES: tl.constexpr = VALUES + ROPE * 2
    ROW_BYTES: tl.constexpr = DATA_BYTES + SCALE_BYTES
    data_byte = page_row * DATA_BYTES + tl.where(VALUES == D, d, d // 2)
    scale_byte_offset = PAGE_ROWS * DATA_BYTES + page_row * SCALE_BYTES + d // GROUP
    if FORMAT == "global" or FORMAT == "index":
        byte = tl.load(
            base + _planar_offset(data_byte, CR, CB, ROW_BYTES), row_valid, other=0
        )
        scale_byte = tl.load(
            base + _planar_offset(scale_byte_offset, CR, CB, ROW_BYTES),
            row_valid,
            other=0,
        )
        if FORMAT == "global":
            scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            # E8M0 byte 0 represents 2**-127 (a float32 subnormal).
            scale = tl.where(
                scale_byte == 0,
                2.0**-127,
                (scale_byte.to(tl.int32) << 23).to(tl.float32, bitcast=True),
            )
        value = _e2m1_decode((byte >> ((d % 2) * 4)) & 15) * scale
    elif FORMAT == "index_v4":
        byte = tl.load(
            base + _planar_offset(page_row * DATA_BYTES + d, CR, CB, ROW_BYTES),
            row_valid,
            other=0,
        )
        # Reassemble the row's single little-endian FP32 scale.
        b = tl.arange(0, 4)[None, :]
        raw = tl.load(
            base
            + _planar_offset(
                PAGE_ROWS * DATA_BYTES + page_row * SCALE_BYTES + b,
                CR,
                CB,
                ROW_BYTES,
            ),
            row_valid,
            other=0,
        ).to(tl.int32)
        scale = tl.sum(raw << (b * 8), axis=1).to(tl.float32, bitcast=True)
        value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale[:, None]
    else:
        quantized = row_valid & (d < QUANTIZED)
        byte = tl.load(
            base + _planar_offset(page_row * DATA_BYTES + d, CR, CB, ROW_BYTES),
            quantized,
            other=0,
        )
        scale_byte = tl.load(
            base + _planar_offset(scale_byte_offset, CR, CB, ROW_BYTES),
            quantized,
            other=0,
        )
        scale = tl.where(
            scale_byte == 0,
            2.0**-127,
            (scale_byte.to(tl.int32) << 23).to(tl.float32, bitcast=True),
        )
        value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
        if ROPE > 0:
            # Halves were stored little-endian; rebuild the BF16 word in place.
            rope = row_valid & (d >= QUANTIZED)
            rope_byte = (
                page_row * DATA_BYTES + VALUES + tl.maximum(d - QUANTIZED, 0) * 2
            )
            lo = tl.load(
                base + _planar_offset(rope_byte, CR, CB, ROW_BYTES),
                rope,
                other=0,
            )
            hi = tl.load(
                base + _planar_offset(rope_byte + 1, CR, CB, ROW_BYTES),
                rope,
                other=0,
            )
            word = (hi.to(tl.uint16) << 8) | lo.to(tl.uint16)
            value = tl.where(
                d < QUANTIZED,
                value,
                word.to(tl.bfloat16, bitcast=True).to(tl.float32),
            )
    return tl.where(row_valid, value, 0.0)


@triton.jit
def _gather_segment_rows(
    cache,
    slots_ptr,
    lens_ptr,
    token_idx,
    slots_stride,
    capacity,
    cols,
    WIDTH: tl.constexpr,
    CP,
    CR: tl.constexpr,
    CB: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    QUANTIZED: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    FORMAT: tl.constexpr,
):
    """Decoded rows of one cache segment at segment-local columns ``cols``, and validity."""
    length = tl.minimum(tl.maximum(tl.load(lens_ptr + token_idx), 0), WIDTH)
    valid = (cols >= 0) & (cols < length)
    slot = tl.load(
        slots_ptr + token_idx * slots_stride + cols, mask=valid, other=-1
    ).to(tl.int64)
    valid = valid & (slot >= 0) & (slot < capacity)
    rows = _dequantize_rows(
        cache,
        slot,
        valid,
        CP,
        CR,
        CB,
        PAGE_ROWS,
        D,
        GROUP,
        VALUES,
        QUANTIZED,
        SCALE_BYTES,
        FORMAT,
    )
    return rows, valid


@triton.jit
def _gather_selected_rows_kernel(
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    kv_ptr,
    indices_ptr,
    swa_slots_stride,
    global_slots_stride,
    swa_capacity,
    global_capacity,
    SWA_CP,
    GLOBAL_CP,
    SWA_WIDTH: tl.constexpr,
    GLOBAL_WIDTH: tl.constexpr,
    SWA_CR: tl.constexpr,
    SWA_CB: tl.constexpr,
    SWA_GROUP: tl.constexpr,
    SWA_VALUES: tl.constexpr,
    SWA_QUANTIZED: tl.constexpr,
    SWA_SCALE_BYTES: tl.constexpr,
    GLOBAL_CR: tl.constexpr,
    GLOBAL_CB: tl.constexpr,
    GLOBAL_GROUP: tl.constexpr,
    GLOBAL_VALUES: tl.constexpr,
    GLOBAL_QUANTIZED: tl.constexpr,
    GLOBAL_SCALE_BYTES: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    HAS_GLOBAL: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Decode ROWS selected rows of one query into its BF16 workspace rows.

    The query's selection is its SWA slots followed by its global slots; row
    ``c`` lands at workspace row ``token * width + c`` and its index entry is
    that row, or -1 when the selection entry is masked.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    WIDTH: tl.constexpr = SWA_WIDTH + GLOBAL_WIDTH if HAS_GLOBAL else SWA_WIDTH
    start = tl.program_id(1) * ROWS
    cols = start + tl.arange(0, ROWS)
    in_width = cols < WIDTH
    # Program row blocks never straddle segments when SWA_WIDTH % ROWS == 0;
    # the both-segment branch keeps other widths correct.
    if start + ROWS <= SWA_WIDTH:
        rows, valid = _gather_segment_rows(
            swa_cache,
            swa_slots,
            swa_lens,
            token_idx,
            swa_slots_stride,
            swa_capacity,
            cols,
            SWA_WIDTH,
            SWA_CP,
            SWA_CR,
            SWA_CB,
            PAGE_ROWS,
            512,
            SWA_GROUP,
            SWA_VALUES,
            SWA_QUANTIZED,
            SWA_SCALE_BYTES,
            "swa",
        )
    elif start >= SWA_WIDTH:
        rows, valid = _gather_segment_rows(
            global_cache,
            global_slots,
            global_lens,
            token_idx,
            global_slots_stride,
            global_capacity,
            tl.where(in_width, cols - SWA_WIDTH, -1),
            GLOBAL_WIDTH,
            GLOBAL_CP,
            GLOBAL_CR,
            GLOBAL_CB,
            PAGE_ROWS,
            512,
            GLOBAL_GROUP,
            GLOBAL_VALUES,
            GLOBAL_QUANTIZED,
            GLOBAL_SCALE_BYTES,
            "global",
        )
    else:
        rows, valid = _gather_segment_rows(
            swa_cache,
            swa_slots,
            swa_lens,
            token_idx,
            swa_slots_stride,
            swa_capacity,
            tl.where(cols < SWA_WIDTH, cols, -1),
            SWA_WIDTH,
            SWA_CP,
            SWA_CR,
            SWA_CB,
            PAGE_ROWS,
            512,
            SWA_GROUP,
            SWA_VALUES,
            SWA_QUANTIZED,
            SWA_SCALE_BYTES,
            "swa",
        )
        if HAS_GLOBAL:
            tail, tail_valid = _gather_segment_rows(
                global_cache,
                global_slots,
                global_lens,
                token_idx,
                global_slots_stride,
                global_capacity,
                tl.where(in_width, cols - SWA_WIDTH, -1),
                GLOBAL_WIDTH,
                GLOBAL_CP,
                GLOBAL_CR,
                GLOBAL_CB,
                PAGE_ROWS,
                512,
                GLOBAL_GROUP,
                GLOBAL_VALUES,
                GLOBAL_QUANTIZED,
                GLOBAL_SCALE_BYTES,
                "global",
            )
            rows += tail
            valid = valid | tail_valid
    workspace_row = token_idx.to(tl.int64) * WIDTH + cols
    d = tl.arange(0, 512)
    tl.store(
        kv_ptr + workspace_row[:, None] * 512 + d[None, :],
        rows.to(tl.bfloat16),
        mask=in_width[:, None],
    )
    tl.store(
        indices_ptr + workspace_row,
        tl.where(valid, workspace_row, -1).to(tl.int32),
        mask=in_width,
    )


def _gather_selected_rows(
    swa_cache, swa_slots, swa_lens, global_cache, global_slots, global_lens
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode every query's selected rows into a BF16 workspace in one launch.

    Returns:
        ``(kv, indices)``: workspace rows ``[tokens * width, 512]`` (SWA
        selection, then global) and ``[tokens, width]`` int32 workspace row
        indices, -1 for masked selection entries.
    """
    tokens = swa_slots.shape[0]
    has_global = global_cache is not None
    swa_width = swa_slots.shape[1]
    global_width = global_slots.shape[1] if has_global else 0
    width = swa_width + global_width
    kv = torch.empty(
        (tokens * width, 512), dtype=torch.bfloat16, device=swa_slots.device
    )
    indices = torch.empty((tokens, width), dtype=torch.int32, device=swa_slots.device)
    _, swa_group, swa_values, _, swa_quantized = _layout("swa")
    _, global_group, global_values, _, global_quantized = _layout("global")
    # Without a global segment its arguments repeat the SWA ones, never read.
    g_cache = global_cache if has_global else swa_cache
    g_slots = global_slots if has_global else swa_slots
    g_lens = global_lens if has_global else swa_lens
    _gather_selected_rows_kernel[(tokens, triton.cdiv(width, _GATHER_SELECTED_ROWS))](
        swa_cache,
        swa_slots,
        swa_lens,
        g_cache,
        g_slots,
        g_lens,
        kv,
        indices,
        swa_slots.stride(0),
        g_slots.stride(0),
        swa_cache.shape[0] * swa_cache.shape[1],
        g_cache.shape[0] * g_cache.shape[1],
        swa_cache.stride(0),
        g_cache.stride(0),
        SWA_WIDTH=swa_width,
        GLOBAL_WIDTH=global_width,
        SWA_CR=swa_cache.stride(1),
        SWA_CB=swa_cache.stride(2),
        SWA_GROUP=swa_group,
        SWA_VALUES=swa_values,
        SWA_QUANTIZED=swa_quantized,
        SWA_SCALE_BYTES=_scale_bytes("swa"),
        GLOBAL_CR=g_cache.stride(1),
        GLOBAL_CB=g_cache.stride(2),
        GLOBAL_GROUP=global_group,
        GLOBAL_VALUES=global_values,
        GLOBAL_QUANTIZED=global_quantized,
        GLOBAL_SCALE_BYTES=_scale_bytes("global"),
        PAGE_ROWS=swa_cache.shape[1],
        HAS_GLOBAL=has_global,
        ROWS=_GATHER_SELECTED_ROWS,
        num_warps=4,
    )
    return kv, indices


def selected_attention_gather(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    query_chunk_size,
    lse,
):
    """Gather each selected KV row once, then run split attention and LSE merge.

    Inputs use V4.1 page-planar caches. Scratch is bounded to at most 32
    queries at a time; masked entries remain -1 and contribute no softmax mass.
    """
    width = swa_slots.shape[1] + (0 if global_slots is None else global_slots.shape[1])
    if width == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(-float("inf"))
        return out
    chunk = min(query_chunk_size, 32)
    sink = None if attn_sink is None else attn_sink[: q.shape[1]].contiguous()
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        kv, indices = _gather_selected_rows(
            swa_cache,
            swa_slots[start:end].contiguous(),
            swa_lens[start:end].contiguous(),
            global_cache,
            None if global_slots is None else global_slots[start:end].contiguous(),
            None if global_lens is None else global_lens[start:end].contiguous(),
        )
        lengths = torch.full((end - start,), width, dtype=torch.int32, device=q.device)
        triton_dsv4_prefill(
            q[start:end].contiguous(),
            kv,
            indices,
            lengths,
            sink,
            softmax_scale,
            out[start:end],
            None if lse is None else lse[start:end],
        )
    return out
