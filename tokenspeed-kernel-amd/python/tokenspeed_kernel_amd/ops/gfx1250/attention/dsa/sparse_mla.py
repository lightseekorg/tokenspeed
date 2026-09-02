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
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.standard_cache_logits import (
    _dsa_standard_decode_logits_kernel,
    _dsa_standard_prefill_logits_kernel,
)

__all__ = [
    "gluon_dsa_decode_topk_fp8_gfx1250",
    "gluon_dsa_decode_topk_standard_gfx1250",
    "gluon_dsa_prefill_topk_fp8_gfx1250",
    "gluon_dsa_prefill_topk_standard_gfx1250",
]

_RADIX_BITS = (12, 12, 8)
_RADIX0_BITS = gl.constexpr(_RADIX_BITS[0])
_RADIX1_BITS = gl.constexpr(_RADIX_BITS[1])
_RADIX2_BITS = gl.constexpr(_RADIX_BITS[2])
_MAX_BUCKETS = gl.constexpr(1 << max(_RADIX_BITS))
_TOPK_BLOCK_N = 2048
_TOPK_NUM_WARPS = 8
_TOPK_WAVES_PER_EU = 1
_DECODE_TOPK_BLOCK_N = 4096
_DECODE_TOPK_NUM_WARPS = 32
_DECODE_TOPK_WAVES_PER_EU = 1
_STANDARD_DECODE_BLOCK_N = 64
_STANDARD_DECODE_CHUNK_N = 64
_STANDARD_DECODE_NUM_WARPS = 4
_STANDARD_DECODE_WAVES_PER_EU = 2
_STANDARD_PREFILL_BLOCK_N = 128
_STANDARD_PREFILL_NUM_WARPS = 8
_STANDARD_PREFILL_WAVES_PER_EU = 1


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
    offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
    valid = offsets < candidate_len
    values = gl.amd.gfx1250.buffer_load(
        candidate_logits,
        offsets,
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
    block_table,
    tile_start,
    candidate_len,
    candidate_start,
    threshold,
    count_greater,
    remaining,
    shared_output_counters,
    out,
    row,
    req,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PREFIX_SHIFT: gl.constexpr,
    IS_DECODE: gl.constexpr,
):
    offsets = tile_start + gl.arange(0, BLOCK_N, layout=value_layout)
    offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
    valid = offsets < candidate_len
    values = gl.amd.gfx1250.buffer_load(
        candidate_logits,
        offsets,
        mask=valid,
        other=-float("inf"),
    )
    keys = _fp32_to_topk_key(values)
    compared_keys = keys if PREFIX_SHIFT == 0 else keys >> PREFIX_SHIFT
    greater = valid & (compared_keys < threshold)
    equal = valid & (compared_keys == threshold)
    reservation_mask = greater | equal
    counter = gl.where(greater, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        counter,
        axis=0,
        mask=reservation_mask,
    )
    logical_offsets = candidate_start + offsets.to(gl.int32)
    if IS_DECODE:
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_cols + block_idx,
            mask=reservation_mask & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        output_values = page * page_size + block_offset
    else:
        output_values = logical_offsets
    gl.store(
        out + row * out_stride + reservation,
        output_values,
        mask=greater & (reservation < count_greater),
    )
    gl.store(
        out + row * out_stride + count_greater + reservation,
        output_values,
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
    shared_output_counters.store(counter_zeros)
    gl.barrier()

    # The three-pass schedule resolves the full ordered FP32 key.
    for pass_index in gl.static_range(3):
        shared_histogram.store(histogram_zeros)
        gl.barrier()
        radix_bits = _RADIX0_BITS
        shift = 32 - _RADIX0_BITS
        if pass_index == 1:
            radix_bits = _RADIX1_BITS
            shift = 32 - _RADIX0_BITS - _RADIX1_BITS
        elif pass_index == 2:
            radix_bits = _RADIX2_BITS
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
        if pass_index == 1:
            selected_bucket_count = gl.where(select_low, count_low, count_high)
        packed = selected_bucket.to(gl.uint32) | (selected_greater.to(gl.uint32) << 12)
        if pass_index == 1:
            selected_bucket_complete = selected_bucket_count == (
                remaining - selected_greater
            )
            packed |= selected_bucket_complete.to(gl.uint32) << 23
        packed = gl.sum(gl.where(selected_group, packed, 0), axis=0)
        prefix = (prefix << radix_bits) | (packed & 0xFFF)
        remaining -= ((packed >> 12) & 0x7FF).to(gl.int32)
        gl.barrier()
        if pass_index == 1:
            if ((packed >> 23) & 1) != 0:
                count_greater = topk - remaining
                for tile_start in range(0, candidate_len, BLOCK_N):
                    _emit_topk_tile(
                        candidate_logits,
                        block_table,
                        tile_start,
                        candidate_len,
                        candidate_start,
                        prefix,
                        count_greater,
                        remaining,
                        shared_output_counters,
                        out,
                        row,
                        req,
                        block_table_cols,
                        page_size,
                        out_stride,
                        value_layout,
                        BLOCK_N,
                        shift,
                        IS_DECODE,
                    )
                return

    count_greater = topk - remaining
    for tile_start in range(0, candidate_len, BLOCK_N):
        _emit_topk_tile(
            candidate_logits,
            block_table,
            tile_start,
            candidate_len,
            candidate_start,
            prefix,
            count_greater,
            remaining,
            shared_output_counters,
            out,
            row,
            req,
            block_table_cols,
            page_size,
            out_stride,
            value_layout,
            BLOCK_N,
            0,
            IS_DECODE,
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
    if not index_k_cache.is_contiguous():
        raise ValueError("index_k_cache must be contiguous")


def _check_standard_scorer_inputs(
    q: torch.Tensor,
    q_scales: torch.Tensor | None,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_size: int,
) -> tuple[int, int, bool]:
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(
            f"standard-cache DSA scorer expects BF16 or FP8 q, got {q.dtype}"
        )
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"DSA weights must be BF16 or FP32, got {weights.dtype}")
    if q.dim() != 3 or q.shape[1] not in (32, 64) or q.shape[2] != 128:
        raise ValueError(
            "standard-cache DSA scorer requires q=[tokens, 32|64, 128], got "
            f"{tuple(q.shape)}"
        )
    if weights.shape != q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(q.shape[:2])}, got {tuple(weights.shape)}"
        )
    if q.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("q and weights must have contiguous innermost dimensions")
    if page_size != 64:
        raise ValueError(
            f"standard-cache DSA scorer requires page_size=64, got {page_size}"
        )
    if index_k_cache.dtype != torch.uint8:
        raise TypeError("index_k_cache must be a uint8 tensor")
    row_bytes = 128 + 4
    if index_k_cache.dim() != 2:
        raise ValueError(
            "index_k_cache must be a packed slot matrix or page-planar matrix, "
            f"got shape {tuple(index_k_cache.shape)}"
        )
    page_bytes = page_size * row_bytes
    if index_k_cache.shape[1] == row_bytes:
        if not index_k_cache.is_contiguous():
            raise ValueError("packed index_k_cache must be contiguous")
        if index_k_cache.shape[0] % page_size:
            raise ValueError("index_k_cache slot count must be page aligned")
        page_stride_bytes = page_bytes
    elif (
        index_k_cache.shape[1] >= page_bytes
        and index_k_cache.stride(1) == 1
        and index_k_cache.stride(0) >= page_bytes
    ):
        page_stride_bytes = index_k_cache.stride(0)
    else:
        raise ValueError(
            "index_k_cache must be contiguous [slots, row_bytes] or page-planar "
            f"[pages, at least {page_bytes} bytes], got "
            f"shape={tuple(index_k_cache.shape)}, stride={index_k_cache.stride()}"
        )
    if index_k_cache.storage_offset() % 4 or page_stride_bytes % 4:
        raise ValueError("index_k_cache page storage must be float32 aligned")
    if weights.device != q.device or index_k_cache.device != q.device:
        raise ValueError("q, weights, and index_k_cache must be on the same device")
    q_is_fp8 = q.dtype == torch.float8_e4m3fn
    if q_is_fp8:
        if q_scales is None:
            raise ValueError("FP8 q requires per-token/head q_scales")
        if (
            q_scales.dtype != torch.float32
            or q_scales.shape != q.shape[:2]
            or q_scales.device != q.device
            or q_scales.stride(-1) != 1
        ):
            raise ValueError(
                "q_scales must be FP32 [tokens, heads] with a contiguous head axis"
            )
    elif q_scales is not None:
        raise ValueError("q_scales is only valid with FP8 q")
    return row_bytes, page_stride_bytes, q_is_fp8


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
    if is_decode:
        block_n = _DECODE_TOPK_BLOCK_N
        num_warps = min(_DECODE_TOPK_NUM_WARPS, topk // 32)
        waves_per_eu = _DECODE_TOPK_WAVES_PER_EU
    else:
        block_n = _TOPK_BLOCK_N
        num_warps = _TOPK_NUM_WARPS
        waves_per_eu = _TOPK_WAVES_PER_EU
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
        BLOCK_N=block_n,
        LOAD_ELEMS=block_n // (32 * num_warps),
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
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
        raise RuntimeError("Gluon DSA paged top-k requires packed FP8 index_k_cache")
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
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
            "Gluon DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
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


def gluon_dsa_decode_topk_standard_gfx1250(
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
    q_scales: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score standard FP8 index keys with Wave32 WMMA and return global slots."""
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _check_topk_contract(topk)
    if q_len_per_req not in (1, 2, 3, 4, 5, 6):
        raise ValueError(
            f"DSA Gluon top-k supports q_len_per_req=1..6, got {q_len_per_req}"
        )
    if index_k_cache is None:
        raise RuntimeError("standard-cache DSA scorer requires index_k_cache")
    row_bytes, page_stride_bytes, q_is_fp8 = _check_standard_scorer_inputs(
        q,
        q_scales,
        weights,
        index_k_cache,
        int(page_size),
    )
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got {tuple(seq_lens.shape)}")
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
            f"{tuple(block_table.shape)}"
        )
    if seq_lens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise TypeError("seq_lens and block_table must be int32")
    if seq_lens.device != q.device or block_table.device != q.device:
        raise ValueError("decode metadata must be on the same device as q")
    if not seq_lens.is_contiguous() or not block_table.is_contiguous():
        raise ValueError("seq_lens and block_table must be contiguous")
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out

    max_candidates = int(block_table.shape[1]) * int(page_size)
    if max_candidates == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out
    logits = torch.empty(
        (q.shape[0], max_candidates),
        dtype=torch.float32,
        device=q.device,
    )
    block_n = _STANDARD_DECODE_BLOCK_N
    chunk_n = _STANDARD_DECODE_CHUNK_N
    num_warps = _STANDARD_DECODE_NUM_WARPS
    q_scale_arg = q_scales if q_scales is not None else weights
    cache_span = (
        0
        if index_k_cache.shape[0] == 0
        else (index_k_cache.shape[0] - 1) * index_k_cache.stride(0)
        + index_k_cache.shape[1]
    )
    grid = (q.shape[0], triton.cdiv(max_candidates, chunk_n))
    _dsa_standard_decode_logits_kernel[grid](
        q,
        q_scale_arg,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_scale_arg.stride(0),
        q_scale_arg.stride(1),
        weights.stride(0),
        weights.stride(1),
        block_table.stride(0),
        logits.stride(0),
        float(softmax_scale),
        max_candidates,
        q_len_per_req,
        PAGE_SIZE=int(page_size),
        ROW_BYTES=row_bytes,
        PAGE_STRIDE_BYTES=page_stride_bytes,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        CHUNK_N=chunk_n,
        NUM_WARPS=num_warps,
        Q_IS_FP8=q_is_fp8,
        USE_BUFFER_LOAD=cache_span < 2**31,
        USE_BUFFER_STORE=logits.nbytes < 2**31,
        num_warps=num_warps,
        waves_per_eu=_STANDARD_DECODE_WAVES_PER_EU,
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


def gluon_dsa_prefill_topk_standard_gfx1250(
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
    q_scales: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score standard FP8 workspace keys with Wave32 WMMA and return row ids."""
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _check_topk_contract(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError("standard-cache DSA scorer requires cache and page_size")
    row_bytes, page_stride_bytes, q_is_fp8 = _check_standard_scorer_inputs(
        q,
        q_scales,
        weights,
        index_k_cache,
        int(page_size),
    )
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError("row_starts and row_ends must have one element per q row")
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

    workspace_rows = int(kv_workspace_slots.numel())
    if workspace_rows == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out
    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(workspace_rows, 1) * 4))

    block_n = _STANDARD_PREFILL_BLOCK_N
    num_warps = _STANDARD_PREFILL_NUM_WARPS
    q_scale_arg = q_scales if q_scales is not None else weights
    cache_span = (
        0
        if index_k_cache.shape[0] == 0
        else (index_k_cache.shape[0] - 1) * index_k_cache.stride(0)
        + index_k_cache.shape[1]
    )
    dummy_table = row_starts
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, workspace_rows),
            dtype=torch.float32,
            device=q.device,
        )
        _dsa_standard_prefill_logits_kernel[(end - start, 1)](
            q[start:end],
            q_scale_arg[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            dummy_table,
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q_scale_arg.stride(0),
            q_scale_arg.stride(1),
            weights.stride(0),
            weights.stride(1),
            logits.stride(0),
            float(softmax_scale),
            workspace_rows,
            PAGE_SIZE=int(page_size),
            ROW_BYTES=row_bytes,
            PAGE_STRIDE_BYTES=page_stride_bytes,
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            BLOCK_N=block_n,
            NUM_WARPS=num_warps,
            Q_IS_FP8=q_is_fp8,
            USE_BUFFER_LOAD=cache_span < 2**31,
            USE_BUFFER_STORE=logits.nbytes < 2**31,
            num_warps=num_warps,
            waves_per_eu=_STANDARD_PREFILL_WAVES_PER_EU,
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
