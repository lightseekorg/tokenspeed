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

"""DSA top-k Gluon kernels for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    _RADIX_TOPK_MIN_COLS,
    _topk_with_padding as _triton_topk_with_padding,
    local_topk_to_global_slots as _triton_local_topk_to_global_slots,
)
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "gluon_dsa_decode_topk_fp8_gfx950",
    "gluon_dsa_prefill_topk_fp8_gfx950",
]


@gluon.constexpr_function
def _vector_layout(
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    return gl.BlockedLayout([LOAD_ELEMS], [64], [NUM_WARPS], [0])


@gluon.jit
def _fp32_to_ordered_key(x):
    bits = x.to(gl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ gl.where(sign != 0, 0xFFFFFFFF, 0x80000000)


@gluon.jit
def _topk_add(a, b):
    return a + b


@gluon.jit
def _find_topk_threshold_key(
    values,
    valid,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    layout: gl.constexpr,
):
    keys = _fp32_to_ordered_key(values)
    prefix = gl.full((), 0, dtype=gl.uint32)
    remaining = gl.full((), topk, dtype=gl.int32)

    for shift in gl.static_range(28, -1, -4):
        if shift == 28:
            prefix_match = valid
        else:
            prefix_match = valid & ((keys >> (shift + 4)) == prefix)
        bucket = (keys >> shift) & 0xF
        cumulative = gl.full((), 0, dtype=gl.int32)
        selected = gl.full((), 0, dtype=gl.uint32)
        selected_remaining = remaining
        found = gl.full((), 0, dtype=gl.int32)

        for bucket_id in gl.static_range(15, -1, -1):
            in_bucket = prefix_match & (bucket == bucket_id)
            count = gl.sum(
                gl.where(
                    in_bucket,
                    gl.full([BLOCK_N], 1, gl.int32, layout=layout),
                    gl.full([BLOCK_N], 0, gl.int32, layout=layout),
                ),
                axis=0,
            ).to(gl.int32)
            take = (found == 0) & (remaining <= cumulative + count)
            selected = gl.where(take, bucket_id, selected)
            selected_remaining = gl.where(
                take, remaining - cumulative, selected_remaining
            )
            cumulative += gl.where(found == 0, count, 0)
            found = gl.where(take, 1, found)

        prefix = (prefix << 4) | selected
        remaining = selected_remaining

    return prefix


@gluon.jit
def _dsa_decode_select_topk_kernel(
    logits,
    block_table,
    seq_lens,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    topk_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    top_offsets = gl.arange(0, topk, layout=topk_layout)
    seq_len = gl.load(seq_lens + row).to(gl.int32)
    lens = gl.minimum(seq_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(out + row * out_stride + top_offsets, -1)

    if seq_len <= topk:
        valid_top = top_offsets < seq_len
        local = top_offsets.to(gl.int32)
        block_idx = local // page_size
        block_offset = local - block_idx * page_size
        page = gl.load(
            block_table + row * block_table_stride + block_idx,
            mask=valid_top & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        slots = page * page_size + block_offset
        gl.store(
            out + row * out_stride + top_offsets,
            gl.where(valid_top, slots, -1),
            mask=top_offsets < topk,
        )
        return

    valid = offsets < seq_len
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    threshold = _find_topk_threshold_key(values, valid, topk, BLOCK_N, layout)
    keys = _fp32_to_ordered_key(values)
    selected = valid & (keys >= threshold)
    selected_i32 = selected.to(gl.int32)
    selected_pos = gl.associative_scan(selected_i32, 0, _topk_add) - 1
    write = selected & (selected_pos < topk)
    local = offsets.to(gl.int32)
    block_idx = local // page_size
    block_offset = local - block_idx * page_size
    page = gl.load(
        block_table + row * block_table_stride + block_idx,
        mask=write & (block_idx < block_table_cols),
        other=0,
    ).to(gl.int32)
    slots = page * page_size + block_offset
    gl.store(out + row * out_stride + selected_pos, slots, mask=write)


@gluon.jit
def _dsa_prefill_select_topk_kernel(
    logits,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    topk_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    top_offsets = gl.arange(0, topk, layout=topk_layout)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(row_end - row_start, 0)
    lens = gl.minimum(candidate_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(out + row * out_stride + top_offsets, -1)

    if candidate_len <= topk:
        local = row_start + top_offsets.to(gl.int32)
        valid_top = top_offsets < candidate_len
        gl.store(
            out + row * out_stride + top_offsets,
            gl.where(valid_top, local, -1),
            mask=top_offsets < topk,
        )
        return

    valid = (offsets >= row_start) & (offsets < row_end)
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    threshold = _find_topk_threshold_key(values, valid, topk, BLOCK_N, layout)
    keys = _fp32_to_ordered_key(values)
    selected = valid & (keys >= threshold)
    selected_i32 = selected.to(gl.int32)
    selected_pos = gl.associative_scan(selected_i32, 0, _topk_add) - 1
    write = selected & (selected_pos < topk)
    gl.store(out + row * out_stride + selected_pos, offsets.to(gl.int32), mask=write)


@gluon.jit
def _dsa_decode_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    block_table_stride: gl.constexpr,
    logits_stride: gl.constexpr,
    page_size: gl.constexpr,
    row_bytes: gl.constexpr,
    max_seq_len: gl.constexpr,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    num_groups: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    token = gl.program_id(0)
    block_id = gl.program_id(1)
    offsets = block_id * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=_vector_layout(BLOCK_N, gl.num_warps(), 1)
    )
    seq_len = gl.load(seq_lens + token).to(gl.int32)
    valid = (offsets < seq_len) & (offsets < max_seq_len)
    block_idx = offsets // page_size
    block_offset = offsets - block_idx * page_size
    page = gl.load(
        block_table + token * block_table_stride + block_idx,
        mask=valid,
        other=0,
    ).to(gl.int64)
    page_bytes = page_size * row_bytes
    fp8_base = page * page_bytes + block_offset.to(gl.int64) * head_dim
    scale_base = (
        page * (page_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset.to(gl.int64) * num_groups
    )
    scores = gl.full(
        [BLOCK_N],
        value=0.0,
        dtype=gl.float32,
        layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
    )

    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(weights + token * num_heads + head).to(gl.float32)
        head_score = gl.full(
            [BLOCK_N],
            value=0.0,
            dtype=gl.float32,
            layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
        )
        for dim in gl.static_range(0, head_dim):
            q_val = gl.load(q + (token * num_heads + head) * head_dim + dim).to(
                gl.float32
            )
            k_val = gl.load(
                index_k_fp8 + fp8_base + dim,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            k_scale = gl.load(
                index_k_scale + scale_base + dim // 128,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            head_score += k_val * k_scale * q_val
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = gl.where(valid, scores, -float("inf"))
    gl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < max_seq_len,
    )


@gluon.jit
def _dsa_prefill_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: gl.constexpr,
    seq_len_sum: gl.constexpr,
    page_size: gl.constexpr,
    row_bytes: gl.constexpr,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    num_groups: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    token = gl.program_id(0)
    block_id = gl.program_id(1)
    offsets = block_id * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=_vector_layout(BLOCK_N, gl.num_warps(), 1)
    )
    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < seq_len_sum)
    slots = gl.load(kv_workspace_slots + offsets, mask=offsets < seq_len_sum, other=0)
    page = slots // page_size
    block_offset = slots - page * page_size
    page_bytes = page_size * row_bytes
    fp8_base = page * page_bytes + block_offset * head_dim
    scale_base = (
        page * (page_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset * num_groups
    )
    scores = gl.full(
        [BLOCK_N],
        value=0.0,
        dtype=gl.float32,
        layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
    )

    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(weights + token * num_heads + head).to(gl.float32)
        head_score = gl.full(
            [BLOCK_N],
            value=0.0,
            dtype=gl.float32,
            layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
        )
        for dim in gl.static_range(0, head_dim):
            q_val = gl.load(q + (token * num_heads + head) * head_dim + dim).to(
                gl.float32
            )
            k_val = gl.load(
                index_k_fp8 + fp8_base + dim,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            k_scale = gl.load(
                index_k_scale + scale_base + dim // 128,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            head_score += k_val * k_scale * q_val
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = gl.where(valid, scores, -float("inf"))
    gl.store(
        logits + token * logits_stride + offsets, scores, mask=offsets < seq_len_sum
    )


def _check_packed_fp8_inputs(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    page_size: int,
) -> int:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"DSA Gluon top-k expects BF16 q, got {q.dtype}")
    if weights.dtype != torch.float32:
        raise TypeError(f"DSA Gluon top-k expects FP32 weights, got {weights.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {tuple(q.shape)}")
    if weights.shape != q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(q.shape[:2])}, got {tuple(weights.shape)}"
        )
    if q.shape[2] != 128:
        raise ValueError(f"DSA Gluon top-k supports head_dim=128, got {q.shape[2]}")
    if page_size != 64:
        raise ValueError(f"DSA Gluon top-k supports page_size=64, got {page_size}")
    if index_k_cache.dtype != torch.uint8:
        raise TypeError(
            "DSA Gluon FP8 top-k expects uint8 packed index_k_cache, got "
            f"{index_k_cache.dtype}"
        )
    num_groups = q.shape[2] // 128
    row_bytes = q.shape[2] + num_groups * 4
    if index_k_cache.dim() != 2 or index_k_cache.shape[1] != row_bytes:
        raise ValueError(
            "packed index_k_cache must have shape [slots, row_bytes="
            f"{row_bytes}], got {tuple(index_k_cache.shape)}"
        )
    if index_k_cache.shape[0] % page_size != 0:
        raise ValueError("packed index_k_cache slot count must be page aligned")
    return row_bytes


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _load_elems(block: int, num_warps: int) -> int:
    return max(1, triton.cdiv(int(block), 64 * int(num_warps)))


def _validate_topk(topk: int) -> None:
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk & (topk - 1):
        raise ValueError(f"DSA Gluon top-k requires power-of-two topk, got {topk}")


def _use_triton_radix_topk(cols: int) -> bool:
    return int(cols) >= _RADIX_TOPK_MIN_COLS


def gluon_dsa_decode_topk_fp8_gfx950(
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
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del plan, q_len_per_req
    topk = int(topk)
    _validate_topk(topk)
    if index_k_cache is None:
        raise RuntimeError("Gluon DSA paged top-k requires packed FP8 index_k_cache")
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
    if seq_lens.dim() != 1 or seq_lens.numel() != q.shape[0]:
        raise ValueError(
            "seq_lens must be [tokens], got "
            f"{tuple(seq_lens.shape)} for q={tuple(q.shape)}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < q.shape[0]:
        raise ValueError(
            "block_table must have at least one row per token, got "
            f"block_table={tuple(block_table.shape)}, q={tuple(q.shape)}"
        )
    if q.shape[0] == 0:
        empty_out = (
            torch.empty((0, int(topk)), dtype=torch.int32, device=q.device)
            if out is None
            else out
        )
        empty_lens = (
            torch.empty((0,), dtype=torch.int32, device=q.device)
            if lens_out is None
            else lens_out
        )
        return empty_out, empty_lens
    if not q.is_cuda:
        raise RuntimeError("DSA Gluon FP8 decode top-k requires CUDA tensors")

    q = q.contiguous()
    index_k_cache = index_k_cache.contiguous()
    weights = weights.contiguous()
    seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
    max_seq_len = int(block_table.shape[1]) * int(page_size)
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    block_n = 64
    _dsa_decode_logits_fp8_kernel[(q.shape[0], triton.cdiv(max_seq_len, block_n))](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        block_table.stride(0),
        logits.stride(0),
        page_size=int(page_size),
        row_bytes=row_bytes,
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=float(softmax_scale),
        BLOCK_N=block_n,
        num_warps=4,
    )
    if _use_triton_radix_topk(max_seq_len):
        local_topk_offsets = _triton_topk_with_padding(logits, topk)
        return _triton_local_topk_to_global_slots(
            local_topk_offsets=local_topk_offsets,
            block_table=block_table,
            block_size=int(page_size),
            seq_lens=seq_lens,
            out=out,
            lens_out=lens_out,
        )

    select_warps = 8
    select_block = _next_power_of_2(max(max_seq_len, topk))
    _dsa_decode_select_topk_kernel[(q.shape[0],)](
        logits,
        block_table,
        seq_lens,
        out,
        lens_out,
        logits.stride(0),
        block_table.stride(0),
        out.stride(0),
        block_table.shape[1],
        page_size=int(page_size),
        topk=topk,
        BLOCK_N=select_block,
        LOAD_ELEMS=_load_elems(select_block, select_warps),
        TOPK_LOAD_ELEMS=_load_elems(topk, select_warps),
        num_warps=select_warps,
    )
    return out, lens_out


def gluon_dsa_prefill_topk_fp8_gfx950(
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
    _validate_topk(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Gluon DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
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
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out
    if not q.is_cuda:
        raise RuntimeError("DSA Gluon FP8 prefill top-k requires CUDA tensors")

    q = q.contiguous()
    index_k_cache = index_k_cache.contiguous()
    weights = weights.contiguous()
    kv_workspace_slots = kv_workspace_slots.to(
        device=q.device, dtype=torch.int64
    ).contiguous()
    row_starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    row_ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    seq_len_sum = int(kv_workspace_slots.numel())
    if seq_len_sum == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out

    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(seq_len_sum, 1) * 4))
    block_n = 64
    select_warps = 8
    use_triton_radix_topk = _use_triton_radix_topk(seq_len_sum)
    if use_triton_radix_topk:
        candidate_lens = (row_ends - row_starts).clamp_min(0)
        lens_out.copy_(
            torch.minimum(candidate_lens, torch.full_like(candidate_lens, topk))
        )
    else:
        select_block = _next_power_of_2(max(seq_len_sum, topk))
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, seq_len_sum), dtype=torch.float32, device=q.device
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
            logits.stride(0),
            seq_len_sum=seq_len_sum,
            page_size=int(page_size),
            row_bytes=row_bytes,
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            num_groups=q.shape[2] // 128,
            softmax_scale=float(softmax_scale),
            BLOCK_N=block_n,
            num_warps=4,
        )
        if use_triton_radix_topk:
            workspace_indices = _triton_topk_with_padding(logits, topk)
            valid = (workspace_indices >= row_starts[start:end, None]) & (
                workspace_indices < row_ends[start:end, None]
            )
            out[start:end].copy_(torch.where(valid, workspace_indices, -1))
            continue

        _dsa_prefill_select_topk_kernel[(end - start,)](
            logits,
            row_starts[start:end],
            row_ends[start:end],
            out[start:end],
            lens_out[start:end],
            logits.stride(0),
            out.stride(0),
            topk=topk,
            BLOCK_N=select_block,
            LOAD_ELEMS=_load_elems(select_block, select_warps),
            TOPK_LOAD_ELEMS=_load_elems(topk, select_warps),
            num_warps=select_warps,
        )
    return out, lens_out
