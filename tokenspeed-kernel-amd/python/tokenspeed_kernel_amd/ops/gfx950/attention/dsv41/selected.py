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

"""Fused two-reader DeepSeek V4.1 selected attention for GFX950.

Reads page-planar SWA (E4M3/E8M0, 528B) and global (E2M1/E4M3, 288B) caches
directly. One online softmax includes both segments and a single sink. This
replaces the portable gather-then-``dsv4_prefill`` path.
"""

from __future__ import annotations

import math
import os

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["launch_gluon_dsv41_selected_attention_gfx950"]

_SWA_ROW_BYTES = 528
_GLOBAL_ROW_BYTES = 288
_PAGE_ROWS = 64
_HEAD_DIM = 512


@gluon.jit
def _e2m1_decode(code):
    a = code & 7
    value = gl.where(
        a < 4,
        a.to(gl.float32) * 0.5,
        gl.where(a == 4, 2.0, gl.where(a == 5, 3.0, gl.where(a == 6, 4.0, 6.0))),
    )
    return gl.where((code & 8) != 0, -value, value)


@gluon.jit
def _planar_offset(byte, CR: gl.constexpr, CB: gl.constexpr, ROW_BYTES: gl.constexpr):
    if CR == ROW_BYTES * CB:
        return byte * CB
    return (byte // ROW_BYTES) * CR + (byte % ROW_BYTES) * CB


@gluon.jit
def _load_segment_slot(
    slots,
    token_idx,
    positions,
    segment_len,
    capacity,
    stride_slots: tl.int64,
    WIDTH: gl.constexpr,
):
    slot = gl.load(
        slots + token_idx.to(tl.int64) * stride_slots + positions,
        mask=(positions < segment_len) & (positions < WIDTH),
        other=-1,
    ).to(gl.int64)
    valid = (
        (positions < segment_len)
        & (positions < WIDTH)
        & (slot >= 0)
        & (slot < capacity)
    )
    return gl.where(valid, slot, 0), valid


@gluon.jit
def _load_v41_tile(
    cache_u8,
    slots,
    valid,
    dims,
    page_stride: tl.int64,
    PAGE_ROWS: gl.constexpr,
    GROUP: gl.constexpr,
    VALUES: gl.constexpr,
    CR: gl.constexpr,
    CB: gl.constexpr,
    IS_SWA: gl.constexpr,
):
    D: gl.constexpr = 512
    ROW_BYTES: gl.constexpr = VALUES + D // GROUP
    page = slots // PAGE_ROWS
    row = slots % PAGE_ROWS
    data_idx = dims if IS_SWA else dims // 2
    data_byte = row[None, :] * VALUES + data_idx[:, None]
    scale_byte = (
        PAGE_ROWS * VALUES + row[None, :] * (D // GROUP) + dims[:, None] // GROUP
    )
    mask = valid[None, :]
    packed = gl.load(
        cache_u8
        + page[None, :] * page_stride
        + _planar_offset(data_byte, CR, CB, ROW_BYTES),
        mask=mask,
        other=0,
    )
    scale_u8 = gl.load(
        cache_u8
        + page[None, :] * page_stride
        + _planar_offset(scale_byte, CR, CB, ROW_BYTES),
        mask=mask,
        other=0,
    )
    if IS_SWA:
        value = packed.to(tl.float8e4nv, bitcast=True).to(gl.float32)
        scale = gl.where(
            scale_u8 == 0,
            2.0**-127,
            (scale_u8.to(gl.int32) << 23).to(gl.float32, bitcast=True),
        )
    else:
        value = _e2m1_decode((packed.to(gl.int32) >> ((dims[:, None] % 2) * 4)) & 15)
        scale = scale_u8.to(tl.float8e4nv, bitcast=True).to(gl.float32)
    return gl.where(mask, (value * scale).to(gl.bfloat16), 0.0)


@gluon.jit
def gluon_dsv41_selected_attention_gfx950(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    out,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    swa_page_stride: tl.int64,
    swa_slot_stride: tl.int64,
    global_page_stride: tl.int64,
    global_slot_stride: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    softmax_scale: tl.float32,
    swa_capacity: tl.int64,
    global_capacity: tl.int64,
    num_heads: tl.int32,
    SWA_WIDTH: gl.constexpr,
    GLOBAL_WIDTH: gl.constexpr,
    HAS_GLOBAL: gl.constexpr,
    SWA_CR: gl.constexpr,
    SWA_CB: gl.constexpr,
    GLOBAL_CR: gl.constexpr,
    GLOBAL_CB: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    mfma_score: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mfma_value: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    q_threads_d: gl.constexpr = min(64, HEAD_DIM // 8)
    q_threads_h: gl.constexpr = 64 // q_threads_d
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[q_threads_h, q_threads_d],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    kv_threads_d: gl.constexpr = min(64, HEAD_DIM // 8)
    kv_threads_k: gl.constexpr = 64 // kv_threads_d
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[kv_threads_d, kv_threads_k],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    out_layout: gl.constexpr = q_load_layout
    q_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [BLOCK_H, HEAD_DIM],
        [1, 0],
    )
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [HEAD_DIM, TILE_K],
        [0, 1],
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_score,
        k_width=8,
    )
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_score,
        k_width=8,
    )
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_value,
        k_width=4,
    )
    v_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_value,
        k_width=4,
    )

    token_idx = gl.program_id(axis=0)
    head_group_idx = gl.program_id(axis=1)
    head_offset = head_group_idx * BLOCK_H
    swa_len = gl.minimum(
        gl.maximum(gl.load(swa_lens + token_idx).to(tl.int32), 0), SWA_WIDTH
    )
    if HAS_GLOBAL:
        global_len = gl.minimum(
            gl.maximum(gl.load(global_lens + token_idx).to(tl.int32), 0),
            GLOBAL_WIDTH,
        )
    else:
        global_len = 0
    swa_tiles = gl.cdiv(swa_len, TILE_K)
    global_tiles = gl.cdiv(global_len, TILE_K)
    num_tiles = gl.maximum(swa_tiles + global_tiles, 1)

    q_heads = head_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, q_load_layout),
    )
    q_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    q_offsets = (
        token_idx.to(tl.int64) * stride_q_t
        + q_heads[:, None].to(tl.int64) * stride_q_h
        + q_dims[None, :].to(tl.int64)
    )
    q_shared = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [BLOCK_H, HEAD_DIM],
        layout=q_shared_layout,
    )
    q_shared.store(
        gl.load(q + q_offsets, mask=(q_heads < num_heads)[:, None], other=0.0)
    )
    kv_shared = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [HEAD_DIM, TILE_K],
        layout=kv_shared_layout,
    )
    q_dot = q_shared.load(q_dot_layout)

    score_heads = head_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, mfma_score),
    )
    valid_heads = score_heads < num_heads
    max_value = gl.load(attn_sink + score_heads, mask=valid_heads, other=0.0).to(
        gl.float32
    )
    denominator = gl.full(
        [BLOCK_H],
        1.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, mfma_score),
    )
    accumulator = gl.zeros(
        [BLOCK_H, HEAD_DIM],
        dtype=gl.float32,
        layout=mfma_value,
    )

    local_k_load = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, kv_load_layout))
    kv_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, kv_load_layout))

    for tile_idx in range(num_tiles):
        if tile_idx < swa_tiles:
            positions = tile_idx * TILE_K + local_k_load
            slots_load, valid_load = _load_segment_slot(
                swa_slots,
                token_idx,
                positions,
                swa_len,
                swa_capacity,
                swa_slot_stride,
                SWA_WIDTH,
            )
            kv_values = _load_v41_tile(
                swa_cache,
                slots_load,
                valid_load,
                kv_dims,
                swa_page_stride,
                64,
                32,
                512,
                SWA_CR,
                SWA_CB,
                True,
            )
        else:
            positions = (tile_idx - swa_tiles) * TILE_K + local_k_load
            if HAS_GLOBAL:
                slots_load, valid_load = _load_segment_slot(
                    global_slots,
                    token_idx,
                    positions,
                    global_len,
                    global_capacity,
                    global_slot_stride,
                    GLOBAL_WIDTH,
                )
                kv_values = _load_v41_tile(
                    global_cache,
                    slots_load,
                    valid_load,
                    kv_dims,
                    global_page_stride,
                    64,
                    16,
                    256,
                    GLOBAL_CR,
                    GLOBAL_CB,
                    False,
                )
            else:
                valid_load = local_k_load < 0
                kv_values = gl.zeros(
                    [HEAD_DIM, TILE_K],
                    dtype=q.dtype.element_ty,
                    layout=kv_load_layout,
                )

        valid_mfma = gl.convert_layout(valid_load, gl.SliceLayout(0, mfma_score))
        kv_shared.store(kv_values)

        k_dot = kv_shared.load(k_dot_layout)
        v_dot = kv_shared.permute([1, 0]).load(v_dot_layout)
        scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_score)
        scores = gl.amd.cdna4.mfma(q_dot, k_dot, scores) * softmax_scale
        scores = gl.where(
            valid_heads[:, None] & valid_mfma[None, :],
            scores,
            -float("inf"),
        )

        tile_max = gl.max(scores, axis=1)
        next_max = gl.maximum(max_value, tile_max)
        safe_next_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        previous_scale = gl.exp(max_value - safe_next_max)
        probabilities = gl.exp(scores - safe_next_max[:, None])
        denominator = previous_scale * denominator + gl.sum(probabilities, axis=1)
        accumulator_scale = gl.convert_layout(
            previous_scale,
            gl.SliceLayout(1, mfma_value),
        )
        accumulator *= accumulator_scale[:, None]
        p_dot = gl.convert_layout(probabilities.to(q.dtype.element_ty), p_dot_layout)
        accumulator = gl.amd.cdna4.mfma(p_dot, v_dot, accumulator)
        max_value = next_max

    denominator_value = gl.convert_layout(denominator, gl.SliceLayout(1, mfma_value))
    safe_denominator = gl.where(denominator_value > 0.0, denominator_value, 1.0)
    accumulator /= safe_denominator[:, None]
    accumulator = gl.where(denominator_value[:, None] > 0.0, accumulator, 0.0)

    out_heads = head_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, out_layout),
    )
    out_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, out_layout))
    out_offsets = (
        token_idx.to(tl.int64) * stride_o_t
        + out_heads[:, None].to(tl.int64) * stride_o_h
        + out_dims[None, :].to(tl.int64)
    )
    output = gl.convert_layout(accumulator.to(out.dtype.element_ty), out_layout)
    gl.store(out + out_offsets, output, mask=(out_heads < num_heads)[:, None])


def _tile_k() -> int:
    value = int(os.environ.get("TOKENSPEED_DSV41_TILE_K", "32"))
    if value not in (16, 32, 64):
        raise ValueError("TOKENSPEED_DSV41_TILE_K must be 16, 32, or 64")
    return value


def _output(out, shape, dtype, device):
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must have the specified shape, dtype, device and be contiguous"
        )
    return out


def _cache(cache, row_bytes, name):
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1:] != (64, row_bytes)
    ):
        raise ValueError(f"{name} cache must be uint8 [pages, 64, {row_bytes}]")


def _integers(x, shape, name):
    if x.shape != shape or x.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32/int64 with shape {shape}")


def launch_gluon_dsv41_selected_attention_gfx950(
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
    schedule,
    prefill_kv,
    prefill_indices,
):
    """Fused SWA+global selected attention over V4.1 page-planar caches."""
    attn_sink = attn_sink[: q.shape[1]]
    if prefill_kv is not None:
        if prefill_indices is None:
            raise ValueError("prefill_kv requires prefill_indices")
        from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

        out = _output(out, q.shape, q.dtype, q.device)
        lengths = torch.full(
            (q.shape[0],),
            prefill_indices.shape[-1],
            dtype=torch.int32,
            device=q.device,
        )
        dsv4_prefill(
            q=q.contiguous(),
            kv=prefill_kv,
            indices=prefill_indices,
            lens=lengths,
            attn_sink=attn_sink.contiguous(),
            softmax_scale=softmax_scale,
            out=out,
            override=None,
            solution=None,
        )
        return out
    if prefill_indices is not None:
        raise ValueError("prefill_indices requires prefill_kv")
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16 or q.shape[1] < 1:
        raise ValueError("q must be BF16 [tokens, heads, 512]")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    extras = (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        attn_sink,
        out,
    )
    if not q.is_cuda or any(t.device != q.device for t in extras if t is not None):
        raise ValueError("all tensors must share a CUDA/ROCm device")
    if attn_sink.shape != (q.shape[1],) or attn_sink.dtype != torch.float32:
        raise ValueError("attn_sink must be FP32 [heads]")
    _cache(swa_cache, _SWA_ROW_BYTES, "swa")
    if swa_slots.ndim != 2 or swa_slots.shape[0] != q.shape[0]:
        raise ValueError("slots must have one row per query")
    _integers(swa_slots, swa_slots.shape, "slots")
    _integers(swa_lens, (q.shape[0],), "lens")
    has_global = global_cache is not None
    if not has_global:
        if global_slots is not None or global_lens is not None:
            raise ValueError(
                "absent global cache requires None global slots and lengths"
            )
    else:
        if global_slots is None or global_lens is None:
            raise ValueError("global cache requires global slots and lengths")
        _cache(global_cache, _GLOBAL_ROW_BYTES, "global")
        if global_slots.ndim != 2 or global_slots.shape[0] != q.shape[0]:
            raise ValueError("slots must have one row per query")
        _integers(global_slots, global_slots.shape, "slots")
        _integers(global_lens, (q.shape[0],), "lens")
    out = _output(out, q.shape, q.dtype, q.device)
    width = swa_slots.shape[1] + (global_slots.shape[1] if has_global else 0)
    if q.shape[0] == 0 or width == 0:
        return out.zero_()

    scale = float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    q = q.contiguous()
    sink = attn_sink.contiguous()
    swa_slots_i = swa_slots.to(torch.int32).contiguous()
    swa_lens_i = swa_lens.to(torch.int32).contiguous()
    if has_global:
        global_slots_i = global_slots.to(torch.int32).contiguous()
        global_lens_i = global_lens.to(torch.int32).contiguous()
        global_cache_u8 = global_cache
        global_width = global_slots_i.shape[1]
        global_capacity = global_cache.shape[0] * _PAGE_ROWS
        global_page_stride = global_cache.stride(0)
        global_slot_stride = global_slots_i.stride(0)
        global_cr = global_cache.stride(1)
        global_cb = global_cache.stride(2)
    else:
        global_slots_i = swa_slots_i
        global_lens_i = swa_lens_i
        global_cache_u8 = swa_cache
        global_width = 0
        global_capacity = 0
        global_page_stride = 0
        global_slot_stride = 0
        global_cr = _GLOBAL_ROW_BYTES
        global_cb = 1

    grid = (q.shape[0], triton.cdiv(q.shape[1], 16))
    gluon_dsv41_selected_attention_gfx950[grid](
        q,
        swa_cache,
        swa_slots_i,
        swa_lens_i,
        global_cache_u8,
        global_slots_i,
        global_lens_i,
        sink,
        out,
        q.stride(0),
        q.stride(1),
        swa_cache.stride(0),
        swa_slots_i.stride(0),
        global_page_stride,
        global_slot_stride,
        out.stride(0),
        out.stride(1),
        scale,
        swa_cache.shape[0] * _PAGE_ROWS,
        global_capacity,
        q.shape[1],
        SWA_WIDTH=swa_slots_i.shape[1],
        GLOBAL_WIDTH=max(global_width, 1),
        HAS_GLOBAL=has_global,
        SWA_CR=swa_cache.stride(1),
        SWA_CB=swa_cache.stride(2),
        GLOBAL_CR=global_cr,
        GLOBAL_CB=global_cb,
        BLOCK_H=16,
        TILE_K=_tile_k(),
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    return out
