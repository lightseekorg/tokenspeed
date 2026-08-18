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

"""Direct page-planar DeepSeek V4 selected attention for AMD GFX950."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["gluon_deepseek_v4_paged_selected_attention_gfx950"]


@gluon.jit
def _load_segment_slot(
    slots,
    token_idx,
    positions,
    segment_len,
    capacity,
    WIDTH: gl.constexpr,
):
    slot = gl.load(
        slots + token_idx.to(tl.int64) * WIDTH + positions,
        mask=positions < segment_len,
        other=-1,
    ).to(gl.int64)
    valid = (positions < segment_len) & (slot >= 0) & (slot < capacity)
    return gl.where(valid, slot, 0), valid


@gluon.jit
def _load_page_planar_tile(
    cache_u8,
    cache_fp8,
    cache_bf16,
    slots,
    valid,
    dims,
    page_stride_bytes,
    page_size,
):
    page = slots // page_size
    position = slots % page_size
    token_base = page * page_stride_bytes + position * 576
    nope_dim = dims[:, None] < 448
    mask = valid[None, :]
    value = gl.load(
        cache_fp8 + token_base[None, :] + dims[:, None].to(gl.int64),
        mask=mask & nope_dim,
        other=0.0,
    ).to(gl.float32)
    exponent = gl.load(
        cache_u8
        + page[None, :] * page_stride_bytes
        + page_size * 576
        + position[None, :] * 8
        + dims[:, None].to(gl.int64) // 64,
        mask=mask & nope_dim,
        other=127,
    ).to(gl.float32)
    rope_offset = (
        token_base[None, :] + 448 + (dims[:, None].to(gl.int64) - 448) * 2
    ) // 2
    rope = gl.load(
        cache_bf16 + rope_offset,
        mask=mask & ~nope_dim,
        other=0.0,
    ).to(gl.float32)
    nope = value * gl.exp2(exponent - 127.0)
    return gl.where(nope_dim, nope, rope).to(gl.bfloat16)


@gluon.jit
def _deepseek_v4_paged_selected_attention_kernel(
    q,
    swa_cache_u8,
    swa_cache_fp8,
    swa_cache_bf16,
    swa_slots,
    swa_lens,
    attn_sink,
    extra_cache_u8,
    extra_cache_fp8,
    extra_cache_bf16,
    extra_slots,
    extra_lens,
    out,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    swa_page_stride_bytes: tl.int64,
    extra_page_stride_bytes: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    softmax_scale: tl.float32,
    num_heads: tl.int32,
    swa_page_size: gl.constexpr,
    extra_page_size: gl.constexpr,
    swa_capacity: tl.int64,
    extra_capacity: tl.int64,
    SWA_WIDTH: gl.constexpr,
    EXTRA_WIDTH: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
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
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
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
    raw_swa_len = gl.load(swa_lens + token_idx).to(gl.int32)
    swa_len = gl.minimum(gl.maximum(raw_swa_len, 0), SWA_WIDTH)
    if HAS_EXTRA:
        raw_extra_len = gl.load(extra_lens + token_idx).to(gl.int32)
        extra_len = gl.minimum(gl.maximum(raw_extra_len, 0), EXTRA_WIDTH)
    else:
        extra_len: gl.constexpr = 0

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
    q_values = gl.load(
        q + q_offsets,
        mask=(q_heads < num_heads)[:, None],
        other=0.0,
    )
    q_shared = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [BLOCK_H, HEAD_DIM],
        layout=q_shared_layout,
    )
    q_shared.store(q_values)

    kv_shared = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [HEAD_DIM, TILE_K],
        layout=kv_shared_layout,
    )

    gl.barrier()
    q_dot = q_shared.load(q_dot_layout)
    score_heads = head_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, mfma_score),
    )
    valid_heads = score_heads < num_heads
    max_value = gl.load(
        attn_sink + score_heads,
        mask=valid_heads,
        other=0.0,
    ).to(gl.float32)
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

    local_k_load = gl.arange(
        0,
        TILE_K,
        layout=gl.SliceLayout(0, kv_load_layout),
    )
    kv_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, kv_load_layout))
    swa_tiles = gl.cdiv(swa_len, TILE_K)
    if HAS_EXTRA:
        extra_tiles = gl.cdiv(extra_len, TILE_K)
    else:
        extra_tiles: gl.constexpr = 0
    num_tiles = swa_tiles + extra_tiles

    for tile_idx in range(num_tiles):
        if tile_idx < swa_tiles:
            positions = tile_idx * TILE_K + local_k_load
            slots_load, valid_load = _load_segment_slot(
                swa_slots,
                token_idx,
                positions,
                swa_len,
                swa_capacity,
                SWA_WIDTH,
            )
            kv_values = _load_page_planar_tile(
                swa_cache_u8,
                swa_cache_fp8,
                swa_cache_bf16,
                slots_load,
                valid_load,
                kv_dims,
                swa_page_stride_bytes,
                swa_page_size,
            )
        else:
            extra_tile_idx = tile_idx - swa_tiles
            positions = extra_tile_idx * TILE_K + local_k_load
            slots_load, valid_load = _load_segment_slot(
                extra_slots,
                token_idx,
                positions,
                extra_len,
                extra_capacity,
                EXTRA_WIDTH,
            )
            kv_values = _load_page_planar_tile(
                extra_cache_u8,
                extra_cache_fp8,
                extra_cache_bf16,
                slots_load,
                valid_load,
                kv_dims,
                extra_page_stride_bytes,
                extra_page_size,
            )

        valid_mfma = gl.convert_layout(
            valid_load,
            gl.SliceLayout(0, mfma_score),
        )
        kv_shared.store(kv_values)
        gl.barrier()

        k_dot = kv_shared.load(k_dot_layout)
        v_dot = kv_shared.permute([1, 0]).load(v_dot_layout)
        scores = gl.zeros(
            [BLOCK_H, TILE_K],
            dtype=gl.float32,
            layout=mfma_score,
        )
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
        p_dot = gl.convert_layout(
            probabilities.to(q.dtype.element_ty),
            p_dot_layout,
        )
        accumulator = gl.amd.cdna4.mfma(p_dot, v_dot, accumulator)
        max_value = next_max
        gl.barrier()

    denominator_value = gl.convert_layout(
        denominator,
        gl.SliceLayout(1, mfma_value),
    )
    safe_denominator = gl.where(denominator_value > 0.0, denominator_value, 1.0)
    accumulator /= safe_denominator[:, None]
    accumulator = gl.where(
        denominator_value[:, None] > 0.0,
        accumulator,
        0.0,
    )

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
    gl.store(
        out + out_offsets,
        output,
        mask=(out_heads < num_heads)[:, None],
    )


def _check_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _check_page_size(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _check_cache(
    name: str,
    cache: torch.Tensor,
    device: torch.device,
    page_size: int,
) -> None:
    if cache.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8, got {cache.dtype}")
    expected_width = page_size * 584
    if cache.dim() != 2 or cache.shape[0] == 0 or cache.shape[1] < expected_width:
        raise ValueError(
            f"{name} must have shape [nonzero pages, >= {expected_width}], got "
            f"{tuple(cache.shape)}"
        )
    if (
        cache.device != device
        or cache.stride(1) != 1
        or cache.stride(0) < cache.shape[1]
        or cache.stride(0) % 2 != 0
        or cache.storage_offset() % 2 != 0
    ):
        raise ValueError(
            f"{name} must be an aligned, unit-inner-stride page view on {device}"
        )


def _check_metadata(
    slots_name: str,
    slots: torch.Tensor,
    lens_name: str,
    lens: torch.Tensor,
    tokens: int,
    device: torch.device,
) -> None:
    if slots.dtype != torch.int32:
        raise TypeError(f"{slots_name} must be int32, got {slots.dtype}")
    if slots.dim() < 2 or slots.shape[0] != tokens:
        raise ValueError(f"{slots_name} must have one row per query token")
    if slots.device != device or not slots.is_contiguous():
        raise ValueError(f"{slots_name} must be contiguous and on {device}")
    if lens.dtype != torch.int32:
        raise TypeError(f"{lens_name} must be int32, got {lens.dtype}")
    if lens.shape != (tokens,):
        raise ValueError(f"{lens_name} must have shape [{tokens}]")
    if lens.device != device or not lens.is_contiguous():
        raise ValueError(f"{lens_name} must be contiguous and on {device}")


def _shares_storage(tensor: torch.Tensor, other: torch.Tensor) -> bool:
    if tensor.numel() == 0 or other.numel() == 0:
        return False
    return tensor.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()


def gluon_deepseek_v4_paged_selected_attention_gfx950(
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_page_size: int,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extra_kv_cache: torch.Tensor | None = None,
    extra_slots: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run direct page-planar DeepSeek V4 selected attention on GFX950.

    Args:
        q: Contiguous BF16 queries shaped `[tokens, heads, 512]`.
        swa_kv_cache: Contiguous uint8 SWA cache shaped
            `[pages, swa_page_size * 584]`. Each page stores all 576-byte
            token rows followed by eight exponent bytes per row.
        swa_slots: Contiguous int32 global SWA slots with one row per token.
        swa_lens: Contiguous int32 valid SWA widths shaped `[tokens]`.
        swa_page_size: Number of independently addressed rows per SWA page.
        attn_sink: Contiguous FP32 or BF16 sink logits, one per query head.
        softmax_scale: Finite scale applied to query-key dot products.
        extra_kv_cache: Optional page-planar cache for a second segment.
        extra_slots: Optional contiguous int32 global slots for that segment.
        extra_lens: Optional contiguous int32 valid widths for that segment.
        extra_page_size: Optional independently addressed rows per extra page.
        out: Optional exact contiguous BF16 destination shaped like `q`.

    Returns:
        The BF16 attention result shaped exactly like `q`. SWA and extra
        selections share one sink-seeded online-softmax state.
    """

    q = _check_tensor("q", q)
    swa_kv_cache = _check_tensor("swa_kv_cache", swa_kv_cache)
    swa_slots = _check_tensor("swa_slots", swa_slots)
    swa_lens = _check_tensor("swa_lens", swa_lens)
    attn_sink = _check_tensor("attn_sink", attn_sink)
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be BF16, got {q.dtype}")
    if q.dim() != 3 or q.shape[0] < 1 or q.shape[2] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    if not q.is_cuda:
        raise ValueError("q must be on an AMD GPU")
    if torch.version.hip is None:
        raise RuntimeError("GFX950 Gluon attention requires a ROCm build")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    device = q.device
    tokens = q.shape[0]

    swa_page_size = _check_page_size("swa_page_size", swa_page_size)
    _check_cache("swa_kv_cache", swa_kv_cache, device, swa_page_size)
    _check_metadata(
        "swa_slots",
        swa_slots,
        "swa_lens",
        swa_lens,
        tokens,
        device,
    )

    extra_values = (extra_kv_cache, extra_slots, extra_lens, extra_page_size)
    has_extra = any(value is not None for value in extra_values)
    if has_extra and not all(value is not None for value in extra_values):
        raise ValueError(
            "extra_kv_cache, extra_slots, extra_lens, and extra_page_size "
            "must be provided together"
        )
    if has_extra:
        extra_kv_cache = _check_tensor("extra_kv_cache", extra_kv_cache)
        extra_slots = _check_tensor("extra_slots", extra_slots)
        extra_lens = _check_tensor("extra_lens", extra_lens)
        extra_page_size = _check_page_size("extra_page_size", extra_page_size)
        _check_cache(
            "extra_kv_cache",
            extra_kv_cache,
            device,
            extra_page_size,
        )
        _check_metadata(
            "extra_slots",
            extra_slots,
            "extra_lens",
            extra_lens,
            tokens,
            device,
        )

    if attn_sink.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"attn_sink must be FP32 or BF16, got {attn_sink.dtype}")
    if attn_sink.dim() != 1 or attn_sink.numel() < q.shape[1]:
        raise ValueError("attn_sink must be 1-D with one value per query head")
    if attn_sink.device != device or not attn_sink.is_contiguous():
        raise ValueError(f"attn_sink must be contiguous and on {device}")

    try:
        scale = float(softmax_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("softmax_scale must be a finite real scalar") from error
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    if out is not None:
        out = _check_tensor("out", out)
        if out.dtype != torch.bfloat16:
            raise TypeError(f"out must be BF16, got {out.dtype}")
        if out.shape != q.shape:
            raise ValueError(
                f"out must have exact shape {tuple(q.shape)}, got {tuple(out.shape)}"
            )
        if out.device != device or not out.is_contiguous():
            raise ValueError(f"out must be contiguous and on {device}")
        aliases = [
            ("q", q),
            ("swa_kv_cache", swa_kv_cache),
            ("swa_slots", swa_slots),
            ("swa_lens", swa_lens),
            ("attn_sink", attn_sink),
        ]
        if has_extra:
            aliases.extend(
                [
                    ("extra_kv_cache", extra_kv_cache),
                    ("extra_slots", extra_slots),
                    ("extra_lens", extra_lens),
                ]
            )
        for name, tensor in aliases:
            if _shares_storage(out, tensor):
                raise ValueError(f"out must not alias {name}")

    output = out if out is not None else torch.empty_like(q)
    if q.shape[1] == 0:
        return output

    swa_slots_2d = swa_slots.view(tokens, swa_slots.numel() // tokens)
    if has_extra:
        extra_slots_2d = extra_slots.view(tokens, extra_slots.numel() // tokens)
        extra_cache_arg = extra_kv_cache
        extra_lens_arg = extra_lens
        extra_page_size_arg = extra_page_size
    else:
        extra_slots_2d = swa_slots_2d
        extra_cache_arg = swa_kv_cache
        extra_lens_arg = swa_lens
        extra_page_size_arg = swa_page_size

    swa_cache_fp8 = swa_kv_cache.view(torch.float8_e4m3fn)
    swa_cache_bf16 = swa_kv_cache.view(torch.bfloat16)
    extra_cache_fp8 = extra_cache_arg.view(torch.float8_e4m3fn)
    extra_cache_bf16 = extra_cache_arg.view(torch.bfloat16)
    grid = (tokens, triton.cdiv(q.shape[1], 16))
    _deepseek_v4_paged_selected_attention_kernel[grid](
        q,
        swa_kv_cache,
        swa_cache_fp8,
        swa_cache_bf16,
        swa_slots_2d,
        swa_lens,
        attn_sink,
        extra_cache_arg,
        extra_cache_fp8,
        extra_cache_bf16,
        extra_slots_2d,
        extra_lens_arg,
        output,
        q.stride(0),
        q.stride(1),
        swa_kv_cache.stride(0),
        extra_cache_arg.stride(0),
        output.stride(0),
        output.stride(1),
        scale,
        q.shape[1],
        swa_page_size,
        extra_page_size_arg,
        swa_kv_cache.shape[0] * swa_page_size,
        extra_cache_arg.shape[0] * extra_page_size_arg,
        SWA_WIDTH=swa_slots_2d.shape[1],
        EXTRA_WIDTH=extra_slots_2d.shape[1] if has_extra else 0,
        HAS_EXTRA=has_extra,
        BLOCK_H=16,
        # Segment-local dequantization keeps the 32-row tile scratch-free.
        TILE_K=32,
        HEAD_DIM=512,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return output
