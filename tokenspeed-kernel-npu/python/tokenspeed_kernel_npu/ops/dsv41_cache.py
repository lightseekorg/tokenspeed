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

"""BF16 resident KV with optional V4.1 operand rounding.

The temporary SWA/global row codecs serve QDQ and DSpark. They preserve
the GPU codec rounding without requiring hardware FP8 casts.
"""

from functools import lru_cache

import torch
from tokenspeed_kernel_npu._triton import tl, triton
from tokenspeed_kernel_npu.ops.dsv41_pointwise import decode_rows

LAYOUTS = {
    "swa": (512, 32, 512, 528),
    "global": (512, 16, 256, 288),
}


@triton.jit
def _e4m3_encode(x):
    a = tl.minimum(tl.abs(x), 448.0)
    exponent = tl.maximum(((a.to(tl.int32, bitcast=True) >> 23) & 255) - 127, -6)
    factor = ((130 - exponent) << 23).to(tl.float32, bitcast=True)
    scaled = a * factor
    low = tl.floor(scaled)
    fraction = scaled - low
    rounded = low.to(tl.int32) + (
        (fraction > 0.5) | ((fraction == 0.5) & ((low.to(tl.int32) & 1) != 0))
    ).to(tl.int32)
    sign = (x.to(tl.int32, bitcast=True) >> 24) & 128
    return (((exponent + 6) * 8 + rounded) | sign).to(tl.uint8)


@lru_cache(maxsize=None)
@torch.compiler.assume_constant_result
def _e4m3_table(device):
    # A device constant initialized before graph capture, never per request.
    return (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
        .to(device)
    )


@triton.jit
def _e4m3_decode(code, table):
    return tl.load(table + code.to(tl.int32))


@triton.jit
def _e2m1_encode(x):
    a = tl.minimum(tl.abs(x), 6.0)
    # Nearest, ties to even *code*, including signed zero. V4's codec is not RNE.
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return (code | ((x.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)


@triton.jit
def _e2m1_decode(code):
    a = code & 7
    value = tl.where(
        a < 4,
        a.to(tl.float32) * 0.5,
        tl.where(a == 4, 2.0, tl.where(a == 5, 3.0, tl.where(a == 6, 4.0, 6.0))),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _pack_kernel(
    X,
    Packed,
    E4M3,
    XS0,
    XS1,
    PS0,
    PS1,
    GROUP: tl.constexpr,
    VALUES: tl.constexpr,
    FORMAT: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    d = tile * 128 + tl.arange(0, 128)
    x = tl.load(X + row * XS0 + d * XS1).to(tl.float32).reshape(128 // GROUP, GROUP)
    amax = tl.max(tl.abs(x), axis=1)
    if FORMAT == "global":
        scale_byte = _e4m3_encode(tl.div_rn(tl.maximum(amax, 6 * 2.0**-9), 6.0))
        scale = _e4m3_decode(scale_byte, E4M3)
    else:
        scaled_max = tl.maximum(amax, 0.0001) * (1.0 / 448.0)
        bits = scaled_max.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 8388607) != 0).to(tl.int32)
        scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
        scale_byte = (exponent + 127).to(tl.uint8)
    y = tl.div_rn(x, scale[:, None]).reshape(128)
    base = Packed + row * PS0
    if FORMAT == "global":
        lo, hi = tl.split(_e2m1_encode(y).reshape(64, 2))
        tl.store(base + (tile * 64 + tl.arange(0, 64)) * PS1, lo | (hi << 4))
    else:
        tl.store(base + d * PS1, _e4m3_encode(y))
    s = tile * (128 // GROUP) + tl.arange(0, 128 // GROUP)
    tl.store(base + (VALUES + s) * PS1, scale_byte)


def _pack(rows, packed, cache_format):
    dim, group, values, row_bytes = LAYOUTS[cache_format]
    if rows.ndim != 2 or rows.shape[1] != dim:
        raise ValueError("Incorrect KV row shape")
    if packed.shape != (rows.shape[0], row_bytes) or packed.dtype != torch.uint8:
        raise ValueError("Incorrect packed KV row shape or dtype")
    # Strided vector loads can exceed Ascend UB during FP4 quantization.
    rows = rows.contiguous()
    if rows.shape[0]:
        _pack_kernel[(rows.shape[0], 4)](
            rows,
            packed,
            _e4m3_table(rows.device),
            *rows.stride(),
            *packed.stride(),
            GROUP=group,
            VALUES=values,
            FORMAT=cache_format,
            num_warps=4,
            enable_fp_fusion=False,
            multibuffer=False,
        )


def cache_pack(rows, cache_format, out):
    """Encode BF16/FP32 rows into contiguous bytes in the original V4.1 layout."""
    if cache_format in ("swa_bf16", "global_bf16"):
        if rows.ndim != 2 or rows.shape[1] != 512:
            raise ValueError("Unquantized V4.1 KV requires [rows,512]")
        if out is None:
            out = torch.empty(rows.shape, device=rows.device, dtype=torch.bfloat16)
        if out.shape != rows.shape or out.dtype != torch.bfloat16:
            raise ValueError("Unquantized V4.1 packed rows require BF16 output")
        out.copy_(rows)
        return out
    cache_format = cache_format.removesuffix("_bf16_qdq")
    if out is None:
        out = torch.empty(
            (rows.shape[0], LAYOUTS[cache_format][3]),
            device=rows.device,
            dtype=torch.uint8,
        )
    _pack(rows, out, cache_format)
    return out


def cache_scatter(rows, cache, slots, cache_format):
    """Store BF16 rows, with optional QDQ; invalid slots skip writes."""
    if cache_format in ("swa_bf16", "global_bf16"):
        from tokenspeed_kernel_npu.ops.dsv41_bf16_cache import cache_scatter_bf16

        cache_scatter_bf16(rows, cache, slots)
        return
    if cache_format == "swa_bf16_qdq":
        from tokenspeed_kernel_npu.ops.dsv41_bf16_cache import cache_scatter_swa_bf16

        cache_scatter_swa_bf16(rows, cache, slots)
        return
    if cache_format == "global_bf16_qdq":
        from tokenspeed_kernel_npu.ops.dsv41_bf16_cache import cache_scatter_bf16

        decoded = cache_unpack(cache_pack(rows, "global", None), "global", None)
        cache_scatter_bf16(decoded, cache, slots)
        return
    raise ValueError("Ascend V4.1 requires BF16 KV/index storage")


def cache_unpack(packed, cache_format, out):
    """Decode strided packed rows into contiguous BF16/FP32 output."""
    if cache_format in ("swa_bf16", "global_bf16"):
        if packed.ndim != 2 or packed.shape[1] != 512 or packed.dtype != torch.bfloat16:
            raise ValueError("Unquantized V4.1 KV requires BF16 [rows,512]")
        if out is None:
            return packed.clone(memory_format=torch.contiguous_format)
        if out.shape != packed.shape or out.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Unquantized V4.1 KV output must be BF16/FP32 [rows,512]")
        out.copy_(packed)
        return out
    cache_format = cache_format.removesuffix("_bf16_qdq")
    dim, _, values, width = LAYOUTS[cache_format]
    if packed.ndim != 2 or packed.shape[1] != width or packed.dtype != torch.uint8:
        raise ValueError("Incorrect packed KV row shape or dtype")
    if out is None:
        out = torch.empty(
            (packed.shape[0], dim), device=packed.device, dtype=torch.bfloat16
        )
    if (
        out.shape != (packed.shape[0], dim)
        or out.dtype not in (torch.bfloat16, torch.float32)
        or not out.is_contiguous()
    ):
        raise ValueError("KV output must be contiguous BF16/FP32 [rows,512]")
    return decode_rows(
        packed[:, :values], packed[:, values:], None, out, fp4=cache_format == "global"
    )


def cache_gather(cache, slots, cache_format, out):
    """Decode arbitrary-shaped slot selections; invalid slots produce zero."""
    if cache_format in ("swa_bf16_qdq", "global_bf16_qdq", "swa_bf16", "global_bf16"):
        from tokenspeed_kernel_npu.ops.dsv41_bf16_cache import cache_gather_bf16

        return cache_gather_bf16(cache, slots, out)
    raise ValueError("Ascend V4.1 requires BF16 KV/index storage")
