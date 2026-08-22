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

"""DSA indexer logits and top-k Gluon kernels for AMD GFX1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.indexing import (
    _check_packed_fp8_inputs,
    _dsa_decode_logits_fp8_kernel,
    _dsa_prefill_logits_fp8_kernel,
)

__all__ = [
    "gluon_dsa_decode_topk_fp8_gfx1250",
    "gluon_dsa_prefill_topk_fp8_gfx1250",
]

_RADIX_BITS = (12, 12, 8)
_MAX_BUCKETS = gl.constexpr(1 << max(_RADIX_BITS))
_TOPK_BLOCK_N = 2048
_TOPK_NUM_WARPS = 8
_TOPK_WAVES_PER_EU = 1


@gluon.constexpr_function
def _vector_layout(
    NUM_WARPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    return gl.BlockedLayout([LOAD_ELEMS], [32], [NUM_WARPS], [0])


@gluon.jit
def _fp32_to_topk_key(x):
    """Map descending FP32 order to ascending unsigned integer order."""
    bits = x.to(gl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ gl.where(sign != 0, 0, 0x7FFFFFFF)


@gluon.jit
def _topk_add(a, b):
    return a + b


@gluon.jit
def _accumulate_histogram_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    prefix,
    shared_histogram,
    shift: gl.constexpr,
    radix_bits: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    FIRST_PASS: gl.constexpr,
):
    offsets = tile_start + gl.arange(0, BLOCK_N, layout=value_layout)
    valid = offsets < candidate_len
    values = gl.load(
        candidate_logits + offsets,
        mask=valid,
        other=-float("inf"),
    )
    keys = _fp32_to_topk_key(values)
    if FIRST_PASS:
        prefix_match = valid
    else:
        prefix_match = valid & ((keys >> (shift + radix_bits)) == prefix)
    buckets = (keys >> shift) & ((1 << radix_bits) - 1)
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )


@gluon.jit
def _emit_topk_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    candidate_start,
    threshold,
    count_greater,
    remaining,
    shared_output_counters,
    out,
    row,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    offsets = tile_start + gl.arange(0, BLOCK_N, layout=value_layout)
    valid = offsets < candidate_len
    values = gl.load(
        candidate_logits + offsets,
        mask=valid,
        other=-float("inf"),
    )
    keys = _fp32_to_topk_key(values)
    greater = valid & (keys < threshold)
    equal = valid & (keys == threshold)
    reservation_mask = greater | equal
    counter = gl.where(greater, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        counter,
        axis=0,
        mask=reservation_mask,
    )
    logical_offsets = candidate_start + offsets.to(gl.int32)
    gl.store(
        out + row * out_stride + reservation,
        logical_offsets,
        mask=greater & (reservation < count_greater),
    )
    gl.store(
        out + row * out_stride + count_greater + reservation,
        logical_offsets,
        mask=equal & (reservation < remaining),
    )


@gluon.jit
def _dsa_wave32_radix_topk_kernel(
    logits,
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    value_layout: gl.constexpr = _vector_layout(gl.num_warps(), LOAD_ELEMS)
    histogram_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), _MAX_BUCKETS // (32 * gl.num_warps())
    )
    group_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), (_MAX_BUCKETS // 2) // (32 * gl.num_warps())
    )
    output_layout: gl.constexpr = _vector_layout(
        gl.num_warps(), topk // (32 * gl.num_warps())
    )
    histogram_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[_MAX_BUCKETS, 1]],
        [_MAX_BUCKETS],
        [0],
    )
    counter_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[2, 1]],
        [2],
        [0],
    )
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [_MAX_BUCKETS],
        histogram_shared_layout,
    )
    shared_output_counters = gl.allocate_shared_memory(
        gl.int32,
        [2],
        counter_shared_layout,
    )
    histogram_zeros = gl.zeros(
        [_MAX_BUCKETS],
        gl.int32,
        layout=histogram_layout,
    )
    counter_layout: gl.constexpr = _vector_layout(gl.num_warps(), 1)
    counter_zeros = gl.zeros([2], gl.int32, layout=counter_layout)

    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_start = gl.full([], 0, gl.int32)
        candidate_end = gl.load(row_ends + req).to(gl.int32)
        if q_len_per_req != 1:
            candidate_end = candidate_end - (q_len_per_req - 1) + q_offset
    else:
        req = row
        candidate_start = gl.load(row_starts + row).to(gl.int32)
        candidate_end = gl.load(row_ends + row).to(gl.int32)

    candidate_len = gl.maximum(candidate_end - candidate_start, 0)
    selected_count = gl.minimum(candidate_len, topk).to(gl.int32)
    output_offsets = gl.arange(0, topk, layout=output_layout)
    gl.store(out + row * out_stride + output_offsets, -1)
    gl.store(lens_out + row, selected_count)

    if candidate_len <= topk:
        valid = output_offsets < candidate_len
        logical_offsets = candidate_start + output_offsets.to(gl.int32)
        if IS_DECODE:
            block_idx = logical_offsets // page_size
            block_offset = logical_offsets - block_idx * page_size
            page = gl.load(
                block_table + req * block_table_cols + block_idx,
                mask=valid & (block_idx < block_table_cols),
                other=0,
            ).to(gl.int32)
            indices = page * page_size + block_offset
        else:
            indices = logical_offsets
        gl.store(
            out + row * out_stride + output_offsets,
            gl.where(valid, indices, -1),
        )
        return

    candidate_logits = logits + row * logits_stride + candidate_start
    bucket_offsets = gl.arange(0, _MAX_BUCKETS, layout=histogram_layout)
    prefix = gl.full([], 0, gl.uint32)
    remaining = gl.full([], topk, gl.int32)

    # The three-pass 12/12/8 radix schedule follows TokenSpeed's gfx950
    # selector, but all lane geometry and memory operations are Wave32-safe.
    for pass_index in gl.static_range(3):
        shared_histogram.store(histogram_zeros)
        gl.barrier()
        radix_bits = 12
        shift = 20
        if pass_index == 1:
            shift = 8
        elif pass_index == 2:
            radix_bits = 8
            shift = 0
        for tile_start in range(0, candidate_len, BLOCK_N):
            _accumulate_histogram_tile(
                candidate_logits,
                tile_start,
                candidate_len,
                prefix,
                shared_histogram,
                shift,
                radix_bits,
                value_layout,
                BLOCK_N,
                pass_index == 0,
            )
            gl.barrier()

        counts = shared_histogram.load(histogram_layout)
        count_pairs = counts.reshape([_MAX_BUCKETS // 2, 2])
        count_low, count_high = gl.split(count_pairs)
        count_low = gl.convert_layout(count_low, group_layout)
        count_high = gl.convert_layout(count_high, group_layout)
        group_counts = count_low + count_high
        cumulative = gl.associative_scan(group_counts, 0, _topk_add)
        before_group = cumulative - group_counts
        selected_group = (before_group < remaining) & (cumulative >= remaining)

        bucket_pairs = bucket_offsets.reshape([_MAX_BUCKETS // 2, 2])
        bucket_low, bucket_high = gl.split(bucket_pairs)
        bucket_low = gl.convert_layout(bucket_low, group_layout)
        bucket_high = gl.convert_layout(bucket_high, group_layout)
        select_low = before_group + count_low >= remaining
        selected_bucket = gl.where(select_low, bucket_low, bucket_high)
        selected_greater = before_group + gl.where(select_low, 0, count_low)
        packed = selected_bucket.to(gl.uint32) | (selected_greater.to(gl.uint32) << 12)
        packed = gl.sum(gl.where(selected_group, packed, 0), axis=0)
        prefix = (prefix << radix_bits) | (packed & 0xFFF)
        remaining -= ((packed >> 12) & 0xFFF).to(gl.int32)
        gl.barrier()

    shared_output_counters.store(counter_zeros)
    gl.barrier()
    count_greater = topk - remaining
    for tile_start in range(0, candidate_len, BLOCK_N):
        _emit_topk_tile(
            candidate_logits,
            tile_start,
            candidate_len,
            candidate_start,
            prefix,
            count_greater,
            remaining,
            shared_output_counters,
            out,
            row,
            out_stride,
            value_layout,
            BLOCK_N,
        )

    if IS_DECODE:
        gl.barrier()
        logical_offsets = gl.load(
            out + row * out_stride + output_offsets,
        ).to(gl.int32)
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_cols + block_idx,
            mask=(logical_offsets >= 0) & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        gl.store(
            out + row * out_stride + output_offsets,
            gl.where(
                logical_offsets >= 0,
                page * page_size + block_offset,
                -1,
            ),
        )


def _check_score_input_contract(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
) -> None:
    if weights.device != q.device or index_k_cache.device != q.device:
        raise ValueError("q, weights, and index_k_cache must be on the same device")
    if q.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("q and weights must have contiguous innermost dimensions")
    if index_k_cache.stride(-1) != 1:
        raise ValueError("index_k_cache pages must contain contiguous bytes")


def _check_topk_contract(topk: int) -> None:
    if topk not in (512, 1024, 2048):
        raise ValueError(
            f"DSA Gluon top-k supports topk=512, 1024, or 2048, got {topk}"
        )


def _dsa_topk_indices(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
    block_table: torch.Tensor | None = None,
    page_size: int = 1,
    q_len_per_req: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_table is None:
        is_decode = False
        block_table = row_starts
        block_table_cols = 0
    else:
        is_decode = True
        block_table_cols = block_table.shape[1]
    rows = logits.shape[0]
    _dsa_wave32_radix_topk_kernel[(rows,)](
        logits,
        block_table,
        row_starts,
        row_ends,
        out,
        lens_out,
        logits.stride(0),
        out.stride(0),
        block_table_cols,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        IS_DECODE=is_decode,
        BLOCK_N=_TOPK_BLOCK_N,
        LOAD_ELEMS=_TOPK_BLOCK_N // (32 * _TOPK_NUM_WARPS),
        num_warps=_TOPK_NUM_WARPS,
        waves_per_eu=_TOPK_WAVES_PER_EU,
    )
    return out, lens_out


def gluon_dsa_decode_topk_fp8_gfx1250(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    index_k_cache: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _check_topk_contract(topk)
    if q_len_per_req not in (1, 2, 3, 4, 5, 6):
        raise ValueError(
            f"DSA Gluon top-k supports q_len_per_req=1..6, got {q_len_per_req}"
        )
    if index_k_cache is None:
        raise RuntimeError("Gluon DSA paged top-k requires an FP8 index_k_cache")
    row_bytes, page_stride_bytes = _check_packed_fp8_inputs(
        q, index_k_cache, weights, int(page_size)
    )
    _check_score_input_contract(q, weights, index_k_cache)
    if seq_lens.dim() != 1:
        raise ValueError(
            f"seq_lens must be 1-D, got {tuple(seq_lens.shape)} for q={tuple(q.shape)}"
        )
    expected_tokens = int(seq_lens.numel()) * q_len_per_req
    if expected_tokens != q.shape[0]:
        raise ValueError(
            "q rows must equal seq_lens rows times q_len_per_req, got "
            f"q={tuple(q.shape)}, seq_lens={tuple(seq_lens.shape)}, "
            f"q_len_per_req={q_len_per_req}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < seq_lens.numel():
        raise ValueError(
            "block_table must have at least one row per request, got "
            f"block_table={tuple(block_table.shape)}, q={tuple(q.shape)}"
        )
    if seq_lens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("seq_lens and block_table must be int32")
    if seq_lens.device != q.device or block_table.device != q.device:
        raise ValueError("decode metadata must be on the same device as q")
    if not seq_lens.is_contiguous() or not block_table.is_contiguous():
        raise ValueError("seq_lens and block_table must be contiguous")
    if q.shape[0] == 0:
        empty_out = (
            torch.empty((0, topk), dtype=torch.int32, device=q.device)
            if out is None
            else out
        )
        empty_lens = (
            torch.empty((0,), dtype=torch.int32, device=q.device)
            if lens_out is None
            else lens_out
        )
        return empty_out, empty_lens

    max_seq_len = int(block_table.shape[1]) * int(page_size)
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    logits = torch.empty(
        (q.shape[0], max_seq_len),
        dtype=torch.float32,
        device=q.device,
    )
    block_n = 32
    _dsa_decode_logits_fp8_kernel[(q.shape[0], triton.cdiv(max_seq_len, block_n))](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        page_size=int(page_size),
        row_bytes=row_bytes,
        page_stride_bytes=page_stride_bytes,
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=float(softmax_scale),
        q_len_per_req=q_len_per_req,
        BLOCK_N=block_n,
        BLOCK_D=128,
        num_warps=4,
        waves_per_eu=1,
    )
    return _dsa_topk_indices(
        logits,
        seq_lens,
        seq_lens,
        block_table=block_table,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )


def gluon_dsa_prefill_topk_fp8_gfx1250(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _check_topk_contract(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Gluon DSA top-k requires an FP8 index_k_cache and page_size"
        )
    row_bytes, page_stride_bytes = _check_packed_fp8_inputs(
        q, index_k_cache, weights, int(page_size)
    )
    _check_score_input_contract(q, weights, index_k_cache)
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError(
            "row_starts/row_ends must be [tokens], got "
            f"row_starts={tuple(row_starts.shape)}, row_ends={tuple(row_ends.shape)}, "
            f"q={tuple(q.shape)}"
        )
    if (
        kv_workspace_slots.dtype != torch.int64
        or row_starts.dtype != torch.int32
        or row_ends.dtype != torch.int32
    ):
        raise TypeError(
            "kv_workspace_slots must be int64 and row_starts/row_ends must be int32"
        )
    if (
        kv_workspace_slots.device != q.device
        or row_starts.device != q.device
        or row_ends.device != q.device
    ):
        raise ValueError("prefill metadata must be on the same device as q")
    if not (
        kv_workspace_slots.is_contiguous()
        and row_starts.is_contiguous()
        and row_ends.is_contiguous()
    ):
        raise ValueError("prefill metadata must be contiguous")
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out

    seq_len_sum = int(kv_workspace_slots.numel())
    if seq_len_sum == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out
    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(seq_len_sum, 1) * 4))

    block_n = 32
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, seq_len_sum),
            dtype=torch.float32,
            device=q.device,
        )
        _dsa_prefill_logits_fp8_kernel[
            (end - start, triton.cdiv(seq_len_sum, block_n))
        ](
            q[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            weights.stride(0),
            weights.stride(1),
            logits.stride(0),
            seq_len_sum=seq_len_sum,
            page_size=int(page_size),
            row_bytes=row_bytes,
            page_stride_bytes=page_stride_bytes,
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            num_groups=q.shape[2] // 128,
            softmax_scale=float(softmax_scale),
            BLOCK_N=block_n,
            BLOCK_D=128,
            num_warps=4,
            waves_per_eu=1,
        )
        _dsa_topk_indices(
            logits,
            row_starts[start:end],
            row_ends[start:end],
            topk=topk,
            out=out[start:end],
            lens_out=lens_out[start:end],
        )
    return out, lens_out
