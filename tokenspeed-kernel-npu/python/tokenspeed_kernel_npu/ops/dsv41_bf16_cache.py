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


"""BF16 V4.1 cache copies and an explicit SWA quantize/dequantize writer."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_cache import _e4m3_encode
from tokenspeed_kernel_npu.ops.dsv41_pointwise import _decode_e4m3_bits


@triton.jit
def _copy_rows(
    Rows,
    Cache,
    Slots,
    Out,
    R0,
    R1,
    CP,
    CR,
    CC,
    SS,
    CAPACITY,
    COUNT,
    SCATTER: tl.constexpr,
):
    d = tl.arange(0, 512)
    for index in range(tl.program_id(0), COUNT, tl.num_programs(0)):
        row = index.to(tl.int64)
        slot = tl.load(Slots + row * SS).to(tl.int64)
        valid = (slot >= 0) & (slot < CAPACITY)
        # Scalar 64-bit page base avoids vector byte offsets wrapping at 2 GiB.
        base = Cache + (slot // 64) * CP + (slot % 64) * CR
        if SCATTER:
            if valid:
                value = tl.load(Rows + row * R0 + d * R1)
                tl.store(base + d * CC, value)
        else:
            value = tl.load(base + d * CC, valid, other=0)
            tl.store(Out + row * 512 + d, value)


@triton.jit
def _gather_rows(Cache, Slots, Out, CP, CR, CC, SS, CAPACITY, COUNT):
    dimensions = tl.arange(0, 512)
    # Four scalar page bases let the compiler schedule independent row copies
    # together without constructing unsafe large vector byte offsets.
    for block in range(tl.program_id(0), tl.cdiv(COUNT, 4), tl.num_programs(0)):
        first = (block * 4).to(tl.int64)
        for offset in tl.static_range(4):
            row = first + offset
            if row < COUNT:
                slot = tl.load(Slots + row * SS).to(tl.int64)
                valid = (slot >= 0) & (slot < CAPACITY)
                base = Cache + (slot // 64) * CP + (slot % 64) * CR
                value = tl.load(base + dimensions * CC, valid, other=0)
                tl.store(Out + row * 512 + dimensions, value)


@triton.jit
def _swa_qdq_scatter(
    Rows,
    Cache,
    Slots,
    Out,
    R0,
    R1,
    CP,
    CR,
    CC,
    SS,
    CAPACITY,
    TILE: tl.constexpr,
    HAS_OUT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + row * SS).to(tl.int64)
    valid = (slot >= 0) & (slot < CAPACITY)
    if valid or HAS_OUT:
        d = tl.program_id(1) * TILE + tl.arange(0, TILE)
        x = tl.load(Rows + row * R0 + d * R1).to(tl.float32).reshape(TILE // 32, 32)
        amax = tl.max(tl.abs(x), 1)
        scaled_max = tl.maximum(amax, 1.0e-4) * (1.0 / 448.0)
        bits = scaled_max.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
        normalized = tl.div_rn(x, scale[:, None]).reshape(TILE)
        code = _e4m3_encode(normalized)
        decoded = _decode_e4m3_bits(code).reshape(TILE // 32, 32)
        value = (decoded * scale[:, None]).reshape(TILE)
        base = Cache + (slot // 64) * CP + (slot % 64) * CR
        tl.store(base + d * CC, value, valid)
        if HAS_OUT:
            tl.store(Out + row * 512 + d, value)


def _check_cache(cache):
    if cache.dtype != torch.bfloat16 or cache.ndim != 3 or cache.shape[1:] != (64, 512):
        raise ValueError("BF16 KV cache must have shape [pages,64,512]")


def cache_scatter_bf16(rows, cache, slots):
    """Store [N,512] rows directly in BF16; invalid slots skip writes."""
    _check_cache(cache)
    if (
        rows.ndim != 2
        or rows.shape[1] != 512
        or slots.ndim != 1
        or slots.numel() != rows.shape[0]
    ):
        raise ValueError("BF16 cache scatter requires [N,512] rows and [N] slots")
    if rows.shape[0]:
        _copy_rows[(min(rows.shape[0], 256),)](
            rows,
            cache,
            slots,
            cache,
            *rows.stride(),
            *cache.stride(),
            slots.stride(0),
            cache.shape[0] * 64,
            rows.shape[0],
            True,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )


def cache_gather_bf16(cache, slots, out=None):
    """Read arbitrary slot shape as [...,512] BF16/FP32; invalid slots yield zero."""
    _check_cache(cache)
    shape = (*slots.shape, 512)
    if out is None:
        out = torch.empty(shape, device=cache.device, dtype=torch.bfloat16)
    if (
        out.shape != shape
        or not out.is_contiguous()
        or out.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError(
            "BF16 cache gather requires a contiguous [...,512] BF16/FP32 output"
        )
    indices = slots.contiguous().view(-1)
    # A batched MTE2/MTE3 copy avoids per-row launch/scheduling overhead.
    # Keep the scalar Triton path for unaligned/strided layouts .
    if (
        indices.numel() >= 4096
        and cache.stride(2) == 1
        and cache.data_ptr() % 32 == 0
        and cache.stride(0) % 16 == 0
        and cache.stride(1) % 16 == 0
        and indices.dtype in (torch.int32, torch.int64)
    ):
        from tokenspeed_kernel_npu.ops import dsv41_cache_gather_native as native

        native.gather_rows(cache, indices, out.view(-1, 512))
        return out
    if indices.numel():
        _gather_rows[(min(triton.cdiv(indices.numel(), 4), 48),)](
            cache,
            indices,
            out,
            *cache.stride(),
            1,
            cache.shape[0] * 64,
            indices.numel(),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out


def cache_scatter_swa_bf16(rows, cache, slots, out=None):
    """Apply SWA E4M3/E8M0 group32 QDQ and store valid slots in BF16.

    Optional ``out`` receives QDQ values for every input row, including rows
    whose cache slot is invalid. This preserves the RoPE writer's output.
    """
    _check_cache(cache)
    if (
        rows.ndim != 2
        or rows.shape[1] != 512
        or slots.ndim != 1
        or slots.numel() != rows.shape[0]
    ):
        raise ValueError("SWA cache scatter requires [N,512] rows and [N] slots")
    if out is not None and (
        out.shape != rows.shape
        or not out.is_contiguous()
        or out.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("SWA QDQ output must be contiguous [N,512] BF16/FP32")
    rows = rows.contiguous()
    if rows.shape[0]:
        _swa_qdq_scatter[(rows.shape[0], 4)](
            rows,
            cache,
            slots,
            out,
            *rows.stride(),
            *cache.stride(),
            slots.stride(0),
            cache.shape[0] * 64,
            128,
            out is not None,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out
