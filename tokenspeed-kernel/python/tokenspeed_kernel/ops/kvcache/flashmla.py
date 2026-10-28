# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
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

"""Packed E4M3 latent rows consumed by FlashMLA's sparse attention kernels."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel._triton import tl, triton

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.attention.prologue.types import LatentNorm


def _cache_layout(cache: torch.Tensor, loc: torch.Tensor) -> tuple[int, int, int]:
    if (
        cache.dtype != torch.uint8
        or cache.ndim not in (2, 3, 4)
        or cache.shape[-1] != 656
        or cache.stride(-1) != 1
        or (cache.ndim >= 3 and cache.shape[-2] != 1)
        or any(s % 4 for s in cache.stride()[:-1])
        or cache.storage_offset() % 4
    ):
        raise ValueError("FlashMLA KV requires 4-byte-aligned uint8 rows of 656 bytes")
    if (
        loc.ndim != 1
        or loc.dtype not in (torch.int32, torch.int64)
        or not loc.is_contiguous()
        or loc.device != cache.device
    ):
        raise ValueError(
            "FlashMLA KV slots must be dense int32/int64 on the cache device"
        )
    page_size = cache.shape[1] if cache.ndim == 4 else 1
    row_stride = cache.stride(1) if cache.ndim == 4 else cache.stride(0)
    page_stride = cache.stride(0)
    if row_stride < 656 or page_stride < page_size * row_stride:
        raise ValueError("FlashMLA KV rows must not overlap")
    return page_size, row_stride, page_stride


@triton.jit
def _set_flashmla_kv_buffer(
    cache,
    scales,
    rope_cache,
    loc,
    nope,
    rope,
    write_mask,
    norm_weight,
    norm_eps,
    nope_stride,
    rope_stride,
    row_stride,
    page_stride,
    num_slots,
    PAGE_SIZE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SANITIZE: tl.constexpr,
    MASKED: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(loc + row).to(tl.int64)
    valid = (slot >= 0) & (slot < num_slots)
    if MASKED:
        valid = valid & tl.load(write_mask + row)
    if valid:
        channels = tl.arange(0, 512)
        values = tl.load(nope + row * nope_stride + channels).to(tl.float32)
        if HAS_NORM:
            # Keep the normalized latent in fp32 until the FP8 cache store.
            variance = tl.sum(values * values, axis=0) / 512.0
            weight = tl.load(norm_weight + channels).to(tl.float32)
            values = (values * tl.rsqrt(variance + norm_eps)) * weight
        offsets = channels.reshape(4, 128)
        values = values.reshape(4, 128)
        if SANITIZE:
            values = tl.where(values == values, values, 0.0)
            values = tl.minimum(tl.maximum(values, -3.38953139e38), 3.38953139e38)
        scale = tl.maximum(tl.max(tl.abs(values), axis=1) / 448.0, 1e-4)
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
        # At the largest BF16 scale, E4M3's 256 overflows on dequantization.
        # Its preceding value, 240, is finite; all lower scales keep the full range.
        limit = tl.where(scale < 2.0**120, 448.0, 240.0)[:, None]
        quantized = tl.minimum(tl.maximum(values / scale[:, None], -limit), limit)
        base = (slot // PAGE_SIZE) * page_stride + (slot % PAGE_SIZE) * row_stride
        tl.store(cache + base + offsets, quantized.to(tl.float8e4nv))
        tl.store(scales + base // 4 + 128 + tl.arange(0, 4), scale)
        r = tl.arange(0, 64)
        rope_values = tl.load(
            rope + row * rope_stride + r, mask=r < ROPE_DIM, other=0.0
        )
        if SANITIZE:
            rope_values = rope_values.to(tl.float32)
            rope_values = tl.where(rope_values == rope_values, rope_values, 0.0)
            rope_values = tl.minimum(
                tl.maximum(rope_values, -3.38953139e38), 3.38953139e38
            )
        tl.store(rope_cache + base // 2 + 264 + r, rope_values)


def set_flashmla_kv_buffer(
    cache: torch.Tensor,
    loc: torch.Tensor,
    nope: torch.Tensor,
    rope: torch.Tensor,
    *,
    sanitize: bool,
    write_mask: torch.Tensor | None,
    norm: LatentNorm | None = None,
) -> None:
    """Scatter BF16 keys into packed FlashMLA rows.

    Args:
        cache: Uint8 [slots, 656], [slots, 1, 656], or paged
            [pages, page_size, 1, 656] storage. Page and row strides may pad.
        loc: Dense int32/int64 destination slots. Out-of-range slots are skipped.
        nope: BF16 latent rows, [rows, 512] or [rows, 1, 512].
        rope: BF16 RoPE rows of width 0 or 64, otherwise shaped like nope.
        sanitize: Replace NaN/Inf with finite values before quantizing.
        write_mask: Dense bool [rows] mask, or None to write all valid slots.
        norm: Optional RMSNorm of the unnormalized latent, computed in fp32
            in the same kernel as quantization and scatter. None means nope
            is already normalized.
    """
    page_size, row_stride, page_stride = _cache_layout(cache, loc)
    if nope.shape[-1] != 512 or rope.shape[-1] not in (0, 64):
        raise ValueError("FlashMLA KV requires latent width 512 and RoPE width 0 or 64")
    if any(
        t.dtype != torch.bfloat16
        or t.device != cache.device
        or t.ndim not in (2, 3)
        or (t.ndim == 3 and t.shape[1] != 1)
        or t.shape[0] != loc.numel()
        or t.stride(-1) != 1
        for t in (nope, rope)
    ):
        raise ValueError("FlashMLA KV sources must be BF16 rows matching the slots")
    if write_mask is not None and (
        write_mask.shape != loc.shape
        or write_mask.dtype != torch.bool
        or write_mask.device != cache.device
        or not write_mask.is_contiguous()
    ):
        raise ValueError("FlashMLA KV write mask must be dense bool [rows]")
    if norm is not None and (
        norm.weight.shape != (512,)
        or not norm.weight.is_contiguous()
        or norm.weight.dtype not in (torch.bfloat16, torch.float32)
        or norm.weight.device != cache.device
        or not math.isfinite(norm.eps)
        or norm.eps < 0
    ):
        raise ValueError(
            "FlashMLA RMSNorm requires 512 dense weights and finite nonnegative eps"
        )
    if loc.numel() == 0:
        return
    nope = nope.reshape(loc.numel(), 512)
    rope = rope.reshape(loc.numel(), rope.shape[-1])
    _set_flashmla_kv_buffer[(loc.numel(),)](
        cache.view(torch.float8_e4m3fn),
        cache.view(torch.float32),
        cache.view(torch.bfloat16),
        loc,
        nope,
        rope,
        write_mask,
        None if norm is None else norm.weight,
        0.0 if norm is None else norm.eps,
        nope.stride(0),
        rope.stride(0),
        row_stride,
        page_stride,
        cache.shape[0] * page_size,
        PAGE_SIZE=page_size,
        ROPE_DIM=rope.shape[-1],
        SANITIZE=sanitize,
        MASKED=write_mask is not None,
        HAS_NORM=norm is not None,
        num_warps=4,
    )


@triton.jit
def _get_flashmla_kv_buffer(
    cache,
    scales,
    rope_cache,
    loc,
    nope,
    rope,
    row_stride,
    page_stride,
    num_slots,
    PAGE_SIZE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(loc + row).to(tl.int64)
    valid = (slot >= 0) & (slot < num_slots)
    base = (slot // PAGE_SIZE) * page_stride + (slot % PAGE_SIZE) * row_stride
    offsets = tl.arange(0, 512)
    values = tl.load(cache + base + offsets, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scales + base // 4 + 128 + offsets // 128, mask=valid, other=0)
    tl.store(nope + row * 512 + offsets, values * scale)
    if ROPE_DIM:
        r = tl.arange(0, 64)
        values = tl.load(rope_cache + base // 2 + 264 + r, mask=valid, other=0)
        tl.store(rope + row * ROPE_DIM + r, values)


def get_flashmla_kv_buffer(
    cache: torch.Tensor,
    loc: torch.Tensor,
    rope_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather packed rows, returning [rows, 1, 512] latent and RoPE keys.

    Args:
        cache: Packed byte rows, with the layout accepted by set_flashmla_kv_buffer.
        loc: Dense int32/int64 source slots. Out-of-range slots produce zeros.
        rope_dim: Logical RoPE width, 0 or 64.
        dtype: Floating dtype of the returned, dequantized rows.
    """
    page_size, row_stride, page_stride = _cache_layout(cache, loc)
    if rope_dim not in (0, 64) or not dtype.is_floating_point:
        raise ValueError(
            "FlashMLA KV gather requires RoPE width 0/64 and floating output"
        )
    nope = torch.empty((loc.numel(), 1, 512), dtype=dtype, device=cache.device)
    rope = torch.empty((loc.numel(), 1, rope_dim), dtype=dtype, device=cache.device)
    if loc.numel():
        _get_flashmla_kv_buffer[(loc.numel(),)](
            cache.view(torch.float8_e4m3fn),
            cache.view(torch.float32),
            cache.view(torch.bfloat16),
            loc,
            nope,
            rope,
            row_stride,
            page_stride,
            cache.shape[0] * page_size,
            PAGE_SIZE=page_size,
            ROPE_DIM=rope_dim,
            num_warps=4,
        )
    return nope, rope
