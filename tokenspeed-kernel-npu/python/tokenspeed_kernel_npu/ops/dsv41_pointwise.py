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

"""Fused V4.1 pointwise steps, preserving FP32 arithmetic and final casts."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _rope_tail(
    X,
    P,
    Table,
    X0,
    XH,
    XD,
    PS,
    TS,
    TOTAL_ROWS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTARY: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    rr = tl.arange(0, TILE_ROWS)
    dd = tl.arange(0, BLOCK)
    pairs = tl.arange(0, BLOCK // 2)
    for start in range(
        tl.program_id(0) * TILE_ROWS, TOTAL_ROWS, tl.num_programs(0) * TILE_ROWS
    ):
        row = (start + rr).to(tl.int64)
        token, head = row // HEADS, row % HEADS
        live = row < TOTAL_ROWS
        pos = tl.maximum(tl.load(P + token * PS, live, 0).to(tl.int64), 0)
        base = token * X0 + head * XH + (DIM - ROTARY) * XD
        x = tl.load(
            X + base[:, None] + dd[None, :] * XD,
            live[:, None] & (dd[None, :] < ROTARY),
            0,
        ).to(tl.float32)
        even, odd = tl.split(tl.reshape(x, (TILE_ROWS, BLOCK // 2, 2)))
        mask = live[:, None] & (pairs[None, :] < ROTARY // 2)
        c = tl.load(Table + pos[:, None] * TS + pairs[None, :], mask, 0)
        s = tl.load(Table + pos[:, None] * TS + ROTARY // 2 + pairs[None, :], mask, 0)
        re = even * c - odd * s
        ro = odd * c + even * s
        value = tl.reshape(tl.join(re, ro), (TILE_ROWS, BLOCK))
        tl.store(
            X + base[:, None] + dd[None, :] * XD,
            value,
            live[:, None] & (dd[None, :] < ROTARY),
        )


def rope_inplace(values, positions, table):
    if values.ndim not in (2, 3) or values.stride(-1) != 1:
        raise ValueError("V4.1 rotary requires 2D/3D channel-contiguous values")
    width = table.shape[1]
    if width % 2 or width > values.shape[-1] or table.stride(1) != 1:
        raise ValueError("V4.1 rotary table must contain contiguous cosine/sine halves")
    heads = values.shape[1] if values.ndim == 3 else 1
    total_rows = values.shape[0] * heads
    tile_rows = (
        16
        if width == 64 and values.shape[-1] == 512 and heads > 1 and total_rows >= 256
        else 4
    )
    blocks = 48 if tile_rows == 16 else 256
    if total_rows:
        _rope_tail[(min(triton.cdiv(total_rows, tile_rows), blocks),)](
            values,
            positions,
            table,
            values.stride(0),
            values.stride(1) if values.ndim == 3 else 0,
            values.stride(-1),
            positions.stride(0),
            table.stride(0),
            values.shape[0] * heads,
            heads,
            values.shape[-1],
            width,
            triton.next_power_of_2(width),
            tile_rows,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return values


@triton.jit
def _reference_norm(
    X, W, O, X0, O0, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * X0 + d, d < N, 0).to(tl.float32)
    w = tl.load(W + d, d < N, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / N) + EPS)
    tl.store(O + row * O0 + d, w * (x * scale), d < N)


@triton.jit
def _reference_norm_rows(
    X,
    W,
    O,
    M,
    X0,
    O0,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    d = tl.arange(0, BLOCK)
    w = tl.load(W + d, d < N, 0).to(tl.float32)
    # Keep each original one-dimensional reduction. A two-dimensional batch
    # reduction changes rounding with this compiler near BF16 midpoints.
    for step in tl.static_range(ROW_TILE):
        row = tl.program_id(0).to(tl.int64) * ROW_TILE + step
        live = (row < M) & (d < N)
        x = tl.load(X + row * X0 + d, live, 0).to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / N) + EPS)
        tl.store(O + row * O0 + d, w * (x * scale), live)


def reference_rmsnorm(x, weight, eps, out):
    """FP32 mean/scale/weight followed by one cast, with explicit destination."""
    n = x.shape[-1]
    if x.stride(-1) != 1 or weight.shape != (n,) or not weight.is_contiguous():
        raise ValueError("reference RMSNorm requires contiguous channels and weight")
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError("reference RMSNorm destination must be contiguous and match x")
    rows = x.reshape(-1, n) if x.ndim != 2 else x
    if rows.numel():
        row_count = rows.shape[0]
        row_tile = 1
        if row_count >= 32:
            if n == 512:
                row_tile = 8 if row_count >= 128 else 4
            elif n in (1280, 5120):
                row_tile = 2
        if row_tile == 1:
            _reference_norm[(row_count,)](
                rows,
                weight,
                out,
                rows.stride(0),
                n,
                n,
                eps,
                triton.next_power_of_2(n),
                num_warps=1,
                enable_fp_fusion=False,
                multibuffer=False,
            )
        else:
            _reference_norm_rows[(triton.cdiv(row_count, row_tile),)](
                rows,
                weight,
                out,
                row_count,
                rows.stride(0),
                n,
                n,
                eps,
                triton.next_power_of_2(n),
                row_tile,
                num_warps=1,
                enable_fp_fusion=False,
                multibuffer=False,
            )
    return out


@triton.jit
def _decode_e4m3_bits(code):
    code = code.to(tl.int32)
    exponent, mantissa = (code >> 3) & 15, code & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    subnormal = (mantissa.to(tl.float32) * 0.001953125).to(tl.int32, bitcast=True)
    bits = tl.where(exponent == 0, subnormal, normal)
    bits = tl.where((exponent == 15) & (mantissa == 7), 0x7FC00000, bits)
    return (bits | ((code & 128) << 24)).to(tl.float32, bitcast=True)


@triton.jit
def _decode_e2m1_bits(code):
    mag = code & 7
    value = tl.where(
        mag < 4,
        mag.to(tl.float32) * 0.5,
        tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
    )
    # The frozen Ascend decoder canonicalizes FP4 negative zero to +0.
    sign = tl.where(mag != 0, (code & 8) << 28, 0)
    return (value.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _decode_rows(
    Codes,
    Scales,
    Valid,
    Out,
    CS0,
    CS1,
    SS0,
    SS1,
    ROWS,
    FP4: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    d = tl.arange(0, 512)
    for row_i in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        row = row_i.to(tl.int64)
        if FP4:
            code = tl.load(Codes + row * CS0 + (d // 2) * CS1).to(tl.int32)
            nibble = tl.where(d % 2 == 0, code & 15, code >> 4)
            value = _decode_e2m1_bits(nibble)
            sc = tl.load(Scales + row * SS0 + (d // 16) * SS1).to(tl.int32)
            scale = _decode_e4m3_bits(sc)
        else:
            code = tl.load(Codes + row * CS0 + d * CS1).to(tl.int32)
            value = _decode_e4m3_bits(code)
            sc = tl.load(Scales + row * SS0 + (d // 32) * SS1).to(tl.int32)
            bits = tl.where(sc == 0, 0x00400000, sc << 23)
            bits = tl.where(sc == 255, 0x7FC00000, bits)
            scale = bits.to(tl.float32, bitcast=True)
        decoded = value * scale
        if HAS_VALID:
            valid = tl.load(Valid + row) != 0
            decoded = tl.where(valid, decoded, 0.0)
        tl.store(Out + row * 512 + d, decoded)


def decode_rows(codes, scales, valid, out, *, fp4):
    """Decode native-gathered rows without changing their cache fetch strategy.

    Bit arithmetic retains subnormal, NaN and signed-zero codec semantics.
    Invalid slots become zero after decode; output is contiguous BF16 or FP32.
    """
    if not out.is_contiguous() or out.numel() != codes.shape[0] * 512:
        raise ValueError("decoded rows require a contiguous 512-channel destination")
    if codes.shape[0]:
        _decode_rows[(min(codes.shape[0], 256),)](
            codes,
            scales,
            valid,
            out,
            codes.stride(0),
            codes.stride(1),
            scales.stride(0),
            scales.stride(1),
            codes.shape[0],
            fp4,
            valid is not None,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return out
