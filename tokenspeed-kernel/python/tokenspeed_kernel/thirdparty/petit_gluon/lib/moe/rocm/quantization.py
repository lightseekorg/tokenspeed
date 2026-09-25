# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native MoE quantization helpers, including the standalone online routines."""

import triton.experimental.gluon as g
import triton.language as tl
from lib.gemm.rocm.amd_intrinsics import amdgcn_cvt_pk_fp8_f32, amdgcn_rcpf
from triton.experimental.gluon import language as l

kQuantBlockK = l.constexpr(128)
kFp8E4m3Max = l.constexpr(448.0)
kQuantFloor = l.constexpr(1.0e-6)


@g.jit
def ClampQuantAbsmax(absmax):
    # Elementwise predicate is Gluon's spelling of the native per-thread branch.
    finite = l.abs(absmax) < float("inf")
    return l.where(finite, l.maximum(absmax, kQuantFloor), kQuantFloor)


@g.jit
def QuantScaleFromAbsmax(absmax):
    clamped = ClampQuantAbsmax(absmax)
    return kFp8E4m3Max * amdgcn_rcpf(clamped)


@g.jit
def DequantScaleFromAbsmax(absmax):
    clamped = ClampQuantAbsmax(absmax)
    return clamped * amdgcn_rcpf(l.full((), kFp8E4m3Max, l.float32))


@g.jit
def OnlineQuantize1x128(out, inp, BLOCK_K: l.constexpr = kQuantBlockK):
    l.static_assert(BLOCK_K % 4 == 0)
    absmax = l.full((), 0, l.float32)
    for i in l.static_range(BLOCK_K):
        absmax = l.maximum(absmax, l.abs(l.load(inp + i)))
    qs = QuantScaleFromAbsmax(absmax)
    ds = DequantScaleFromAbsmax(absmax)
    for i in l.static_range(0, BLOCK_K, 4):
        qi = amdgcn_cvt_pk_fp8_f32(
            l.load(inp + i) * qs,
            l.load(inp + i + 1) * qs,
            l.full((), 0, l.uint32),
            False,
        )
        qi = amdgcn_cvt_pk_fp8_f32(
            l.load(inp + i + 2) * qs, l.load(inp + i + 3) * qs, qi, True
        )
        l.store(out.to(l.pointer_type(l.uint32)) + i // 4, qi)
    return ds


@g.jit
def OnlineQuantize2x128(
    inp, quant_scale, ROWS: l.constexpr = 2, COLS: l.constexpr = kQuantBlockK
):
    l.static_assert(ROWS == 2, "OnlineQuantize2x128 expects exactly 2 rows.")
    l.static_assert(COLS == kQuantBlockK, "OnlineQuantize2x128 expects kCols == 128.")
    out = ()
    for i in l.static_range(2):
        row = ()
        for k in l.static_range(2):
            s = quant_scale[k * 2 + i]
            for j in l.static_range(2):
                v = inp[k * 4 + j * 2 + i]
                u0 = amdgcn_cvt_pk_fp8_f32(
                    v[0] * s, v[1] * s, l.full((), 0, l.uint32), False
                )
                u1 = amdgcn_cvt_pk_fp8_f32(
                    v[2] * s, v[3] * s, l.full((), 0, l.uint32), False
                )
                row += ((u1 << 16) | u0,)
        out += (row,)
    return out


from typing import NamedTuple

from lib.gemm.rocm.amd_intrinsics import (
    amdgcn_cvt_scalef32_pk_fp4_f32,
    amdgcn_pk_mul_f32,
)


class MxFp4Scale(NamedTuple):
    byte: object
    packing_scale: object


@g.jit
def QuantizeFp8E4m3x4(value, scale):
    scale2 = (scale, scale)
    xy = amdgcn_pk_mul_f32(value[:2], scale2)
    zw = amdgcn_pk_mul_f32(value[2:], scale2)
    packed = amdgcn_cvt_pk_fp8_f32(xy[0], xy[1], 0, False)
    return amdgcn_cvt_pk_fp8_f32(zw[0], zw[1], packed, True)


class NativeMxFp4Quantization:
    __triton_builtin__ = True

    @g.jit
    def MaximumAbs(value):
        return l.maximum(
            l.maximum(l.abs(value[0]), l.abs(value[1])),
            l.maximum(l.abs(value[2]), l.abs(value[3])),
        )

    @g.jit
    def EncodeScale(max_abs):
        required = max_abs * (1.0 / 6.0)
        required_bits = required.to(l.uint32, bitcast=True)
        scale_byte = (required_bits >> 23) & 0xFF
        scale_byte += ((scale_byte < 0xFF) & ((required_bits & 0x7FFFFF) != 0)).to(
            l.uint32
        )
        bits = scale_byte << 23
        return MxFp4Scale(
            l.where(max_abs < 1.0e-12, 127, scale_byte),
            l.where(max_abs < 1.0e-12, 1.0, bits.to(l.float32, bitcast=True)),
        )

    @g.jit
    def Pack(values, scale, kVectors: l.constexpr):
        # The native destination is a register-array view at these call sites.
        l.static_assert(kVectors % 2 == 0, "native packing consumes pairs of float4")
        packed = ()
        for vector in l.static_range(0, kVectors, 2):
            a, b = values[vector], values[vector + 1]
            word = amdgcn_cvt_scalef32_pk_fp4_f32(0, a[0], a[1], scale.packing_scale, 0)
            word = amdgcn_cvt_scalef32_pk_fp4_f32(
                word, a[2], a[3], scale.packing_scale, 1
            )
            word = amdgcn_cvt_scalef32_pk_fp4_f32(
                word, b[0], b[1], scale.packing_scale, 2
            )
            word = amdgcn_cvt_scalef32_pk_fp4_f32(
                word, b[2], b[3], scale.packing_scale, 3
            )
            packed += (word,)
        return packed


class AiterMxFp4Quantization:
    __triton_builtin__ = True

    @g.jit
    def MaximumAbs(value):
        return l.maximum(
            l.maximum(
                l.abs(value[0]), l.abs(value[1]), propagate_nan=tl.PropagateNan.ALL
            ),
            l.maximum(
                l.abs(value[2]), l.abs(value[3]), propagate_nan=tl.PropagateNan.ALL
            ),
            propagate_nan=tl.PropagateNan.ALL,
        )

    @g.jit
    def EncodeScale(max_abs):
        bits = max_abs.to(l.uint32, bitcast=True)
        bits = (bits + 0x00400000) & 0xFF800000
        exponent = l.maximum(bits >> 23, 2)
        scale_byte = exponent - 2
        scale_bits = scale_byte << 23
        return MxFp4Scale(scale_byte, scale_bits.to(l.float32, bitcast=True))

    @g.jit
    def Pack(values, scale, kVectors: l.constexpr):
        packed = ()
        for vector in l.static_range(kVectors):
            value = values[vector]
            word = amdgcn_cvt_scalef32_pk_fp4_f32(
                0, value[0], value[1], scale.packing_scale, 0
            )
            word = amdgcn_cvt_scalef32_pk_fp4_f32(
                word, value[2], value[3], scale.packing_scale, 1
            )
            packed += (word.to(l.uint16),)
        return packed


@g.jit
def QuantizeMxFp4(values, max_abs, Quantization: l.constexpr, kVectors: l.constexpr):
    scale = Quantization.EncodeScale(max_abs)
    packed = Quantization.Pack(values, scale, kVectors)
    return packed, scale.byte
