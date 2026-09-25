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

import struct

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_fastmath import Fp16Trait, hmul2
from lib.gemm.rocm.amd_intrinsics import (
    HAS_AMD_BF8_PACK_CONVERSION,
    amdgcn_cvt_pk_f32_bf8,
    amdgcn_perm_b32,
    amdgcn_pk_mul_f32,
    bitreverse,
)
from lib.gemm.rocm.quantization.types import DataType
from triton.experimental.gluon import language as l

kFp8ScaleBias = l.constexpr(7)
kDataTypeFp16 = l.constexpr(int(DataType.Fp16))
kDataTypeFp8e8m0 = l.constexpr(int(DataType.Fp8e8m0))
kDataTypeFp8e5m2Fnuz = l.constexpr(int(DataType.Fp8e5m2Fnuz))


# Native detail namespace: template type parameters become explicit constexprs.
class DateTypeTrait:
    kExBits = {False: 5, True: 8}


class DequantizerToFp16Impl:
    __triton_builtin__ = True

    @g.jit
    def Dequant(v, SRC_EX: l.constexpr, FP16_EX: l.constexpr):
        kRightShift: l.constexpr = FP16_EX - SRC_EX
        kMask: l.constexpr = 0x70007000
        kSignMask: l.constexpr = 0x80008000
        first = (v & kSignMask) | ((v & kMask) >> kRightShift)
        v = v << 4
        second = (v & kSignMask) | ((v & kMask) >> kRightShift)
        return first, second


class DequantizerForFp8ScaleImpl:
    __triton_builtin__ = True

    @staticmethod
    def GlobalScaleFactor(bf16, intermediate, upscale):
        kSrcEx = 2
        kFp16Ex = DateTypeTrait.kExBits[bf16]
        kUpscale = upscale
        kScaleEx = 5
        kSrcBias = (1 << (kSrcEx - 1)) - 1
        kScaleBias = (1 << (kScaleEx - 1)) - 1
        kFp16Bias = (1 << (kFp16Ex - 1)) - 1
        kIntermediateConvertBias = int(intermediate == DataType.Fp8e5m2Fnuz)
        kUpscaleExpBiasRaw = ((1 << kFp16Ex) - 1) - ((1 << kScaleEx) - 1)
        kGSExpBias = (
            kFp16Bias
            - (kSrcBias - kIntermediateConvertBias)
            - (kUpscaleExpBiasRaw - kFp16Bias)
            - kScaleBias
            - kFp8ScaleBias.value
            if kUpscale
            else 0
        )
        kFp32Ex = 8
        kFp32Bias = (1 << (kFp32Ex - 1)) - 1
        kDequantExpBiasU32 = (kGSExpBias + kFp32Bias) << (32 - kFp32Ex - 1)
        return struct.unpack("<f", struct.pack("<I", kDequantExpBiasU32))[0]

    @g.jit
    def AdjustPackedScaleBias(s, BF16: l.constexpr, UPSCALE: l.constexpr):
        kFp16Ex: l.constexpr = 8 if BF16 else 5
        kScaleEx: l.constexpr = 5
        kFp16Bias: l.constexpr = (1 << (kFp16Ex - 1)) - 1
        kScaleBias: l.constexpr = (1 << (kScaleEx - 1)) - 1
        kUpscaleExpBiasRaw: l.constexpr = ((1 << kFp16Ex) - 1) - ((1 << kScaleEx) - 1)
        kReversePreprocessBias: l.constexpr = kFp16Bias - kScaleBias - kFp8ScaleBias
        if UPSCALE or kFp16Ex == 8:
            kScaleBiasU16: l.constexpr = (
                kUpscaleExpBiasRaw if UPSCALE else kReversePreprocessBias
            ) << (16 - kFp16Ex - 1)
            kScaleBiasU32: l.constexpr = (kScaleBiasU16 << 16) | kScaleBiasU16
            return s + kScaleBiasU32
        else:
            return s


@g.jit
def Fp4ToFp16(q):
    q = q.to(l.uint32)
    qr = bitreverse(q)
    return (
        q & 0x8E008E00,
        (q << 8) & 0x8E008E00,
        qr & 0x8E008E00,
        (qr << 8) & 0x8E008E00,
    )


@g.jit
def Fp4ToBf8(q):
    q = q.to(l.uint32)
    qr = bitreverse(q)
    return q & 0x8E8E8E8E, qr & 0x8E8E8E8E


class Dequantizer(DequantizerToFp16Impl):
    __triton_builtin__ = True

    @g.jit
    def Bias(high_precision, BF16: l.constexpr):
        kSrcEx: l.constexpr = 2
        kFp16Ex: l.constexpr = 8 if BF16 else 5
        kExpOffset: l.constexpr = 2 * (1 << (kFp16Ex - 1)) - (1 << (kSrcEx - 1)) - 1
        if BF16:
            v = l.full((), kExpOffset << (15 - kFp16Ex), l.uint16)
            return v.to(l.bfloat16, bitcast=True)
        else:
            off = l.where(high_precision, kExpOffset - kFp8ScaleBias, kExpOffset)
            v = (off << (15 - kFp16Ex)).to(l.uint16)
            return v.to(l.float16, bitcast=True)


class DequantizerForFp8Scale(DequantizerForFp8ScaleImpl):
    __triton_builtin__ = True

    @g.jit
    def Dequant(s, BF16: l.constexpr, INTERMEDIATE: l.constexpr):
        s = s.to(l.uint32) & 0xFFFF
        if BF16 and INTERMEDIATE == kDataTypeFp8e8m0:
            return ((s & 255) << 7) | (((s >> 8) & 255) << 23)
        else:
            l.static_assert(BF16 or INTERMEDIATE == kDataTypeFp16)
            v = amdgcn_perm_b32(l.full((), 0, l.uint32), s, 0x0C010C00)
            return v << (4 if BF16 else 7)

    @g.jit
    def DequantFullScale(
        s, BF16: l.constexpr, INTERMEDIATE: l.constexpr, UPSCALE: l.constexpr
    ):
        v = DequantizerForFp8Scale.Dequant(s, BF16, INTERMEDIATE)
        if BF16 and INTERMEDIATE == kDataTypeFp8e8m0:
            if UPSCALE:
                v += (kFp8ScaleBias | (kFp8ScaleBias << 16)) << 7
            return v
        else:
            return DequantizerForFp8ScaleImpl.AdjustPackedScaleBias(v, BF16, UPSCALE)

    @staticmethod
    def GlobalScaleFactor(bf16, intermediate, upscale):
        if bf16 and intermediate == DataType.Fp8e8m0:
            return 1.0
        return DequantizerForFp8ScaleImpl.GlobalScaleFactor(bf16, intermediate, upscale)


class UnifiedDequantizerForFp4Fp16:
    __triton_builtin__ = True

    @g.jit
    def DequantScales(s, HP: l.constexpr):
        return DequantizerForFp8Scale.DequantFullScale(s, False, kDataTypeFp16, not HP)

    @staticmethod
    def GlobalScaleFactor(high_precision):
        return DequantizerForFp8Scale.GlobalScaleFactor(
            False, DataType.Fp16, not high_precision
        )

    @g.jit
    def DequantWithScale(q, scale, HP: l.constexpr):
        s = scale.to(l.uint16, bitcast=True).to(l.uint32)
        s2 = s | (s << 16)
        bias = Dequantizer.Bias(HP, False).to(l.uint16, bitcast=True).to(l.uint32)
        bias2 = bias | (bias << 16)
        unpacked = Fp4ToFp16(q)
        out = ()
        for i in l.static_range(4):
            value = unpacked[i]
            if HP:
                value = hmul2(value, bias2, False)
            value = hmul2(value, s2, False)
            out += (value,)
        return out


class UnifiedDequantizerForFp4Bf16:
    __triton_builtin__ = True
    kUseBf8 = HAS_AMD_BF8_PACK_CONVERSION

    @g.jit
    def DequantWithScale(q, scale, HP: l.constexpr):
        s2 = Fp16Trait.ToFloat2(scale)
        if UnifiedDequantizerForFp4Bf16.kUseBf8:
            return UnifiedDequantizerForFp4Bf16.DequantWithScaleImplBf8Fnuz(q, s2, HP)
        else:
            return UnifiedDequantizerForFp4Bf16.DequantWithScaleImplFp16(q, s2, HP)

    @g.jit
    def DequantWithScaleImplFp16(q, s2, HP: l.constexpr):
        kBias: l.constexpr = 0x43000000 if HP else 0x46800000
        bias_f32_2 = l.full((), kBias | (kBias << 32), l.uint64)
        unpacked = Fp4ToFp16(q)
        out_f2 = ()
        for i in l.static_range(4):
            x = (
                (unpacked[i] & 65535)
                .to(l.uint16)
                .to(l.float16, bitcast=True)
                .to(l.float32)
            )
            y = (
                (unpacked[i] >> 16)
                .to(l.uint16)
                .to(l.float16, bitcast=True)
                .to(l.float32)
            )
            out_f2 += (
                x.to(l.uint32, bitcast=True).to(l.uint64)
                | (y.to(l.uint32, bitcast=True).to(l.uint64) << 32),
            )
        scaled = ()
        for i in l.static_range(4):
            value = out_f2[i]
            if HP:
                value = amdgcn_pk_mul_f32(value, bias_f32_2)
            scaled += (amdgcn_pk_mul_f32(value, s2),)
        out = ()
        for i in l.static_range(4):
            out += (
                amdgcn_perm_b32(
                    (scaled[i] >> 32).to(l.uint32), scaled[i].to(l.uint32), 0x07060302
                ),
            )
        return out

    @g.jit
    def DequantWithScaleImplBf8Fnuz(q, s2, HP: l.constexpr):
        kBias: l.constexpr = 0x43800000 if HP else 0x46800000
        bias_f32_2 = l.full((), kBias | (kBias << 32), l.uint64)
        bf8 = Fp4ToBf8(q)
        out_f2 = ()
        for i in l.static_range(2):
            out_f2 += (
                amdgcn_cvt_pk_f32_bf8(bf8[i], False),
                amdgcn_cvt_pk_f32_bf8(bf8[i], True),
            )
        scaled = ()
        for i in l.static_range(4):
            value = out_f2[i]
            if HP:
                value = amdgcn_pk_mul_f32(value, bias_f32_2)
            scaled += (amdgcn_pk_mul_f32(value, s2),)
        return (
            amdgcn_perm_b32(
                (scaled[1] >> 32).to(l.uint32),
                (scaled[0] >> 32).to(l.uint32),
                0x07060302,
            ),
            amdgcn_perm_b32(scaled[1].to(l.uint32), scaled[0].to(l.uint32), 0x07060302),
            amdgcn_perm_b32(
                (scaled[3] >> 32).to(l.uint32),
                (scaled[2] >> 32).to(l.uint32),
                0x07060302,
            ),
            amdgcn_perm_b32(scaled[3].to(l.uint32), scaled[2].to(l.uint32), 0x07060302),
        )


class UnifiedDequantizerForMxFp4Bf16(UnifiedDequantizerForFp4Bf16):
    __triton_builtin__ = True

    @g.jit
    def DequantScales(s, HP: l.constexpr):
        return DequantizerForFp8Scale.DequantFullScale(s, True, kDataTypeFp8e8m0, HP)

    @staticmethod
    def GlobalScaleFactor(high_precision):
        return 1.0 if high_precision else 32768.0


class UnifiedDequantizerForNvFp4Bf16(UnifiedDequantizerForFp4Bf16):
    __triton_builtin__ = True

    @g.jit
    def DequantScales(s, HP: l.constexpr):
        return DequantizerForFp8Scale.Dequant(s, False, kDataTypeFp16)

    @staticmethod
    def GlobalScaleFactor(high_precision):
        intermediate = (
            DataType.Fp8e5m2Fnuz
            if UnifiedDequantizerForFp4Bf16.kUseBf8
            else DataType.Fp16
        )
        return DequantizerForFp8Scale.GlobalScaleFactor(
            True, intermediate, not high_precision
        )
