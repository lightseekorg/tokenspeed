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

"""Native online row maximum, FP8 packing, and LDS shuffle."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    _pack_float2,
    _unpack_float2,
    amdgcn_cvt_pk_fp8_f32,
    amdgcn_pk_mul_f32,
    amdgcn_rcpf,
    kWarpSize,
)
from lib.moe.rocm.memory_ops import _load_uint2
from lib.tal.tensor.layout import Layout, Shape, Stride, make_coord
from triton.experimental.gluon import language as l


@g.jit
def _max4(v):
    # Native ComputeRowMax's local lambda; Gluon requires a named JIT helper.
    return l.maximum(
        l.maximum(l.abs(v[0]), l.abs(v[1])), l.maximum(l.abs(v[2]), l.abs(v[3]))
    )


class QuantizeAndShuffleFp8:
    __triton_builtin__ = True
    kNumWarps = 4
    kGroupN = 256
    kGroupM = 32
    kThreads = kNumWarps * kWarpSize.value
    kSubGroupSize = 16
    kKStages = kGroupN // 128
    kElementsPerThread = (kGroupM * kGroupN) // kThreads
    kElementsPerThreadVec4 = kElementsPerThread // 4
    assert kGroupN % 128 == 0
    # Native array extents; scalar words are used for the raw LDS allocation.
    FragH = (kElementsPerThreadVec4, 4)
    Quantized = (kElementsPerThreadVec4, 4)
    FragPacked = (kElementsPerThread // 4,)
    MaxShm = (kThreads, 2)
    Shm = (kThreads * kElementsPerThreadVec4,)
    ShmShape = Shape(Shape(2, kElementsPerThreadVec4 // 2), kNumWarps, Shape(16, 2, 2))
    ShmStride = Stride(
        Stride(
            (kElementsPerThreadVec4 // 2) * kNumWarps * kWarpSize.value,
            kNumWarps * kWarpSize.value,
        ),
        kWarpSize.value,
        Stride(2, 1, 32),
    )
    ShmWriteLayout = Layout(ShmShape, ShmStride)

    @g.jit
    def Run(shm_max, shm_q_h, h, tid, wid, wtid, ReadLayout: l.constexpr):
        quant_scale, dq_act = QuantizeAndShuffleFp8.ComputeRowMax(shm_max, h, tid)
        q = QuantizeAndShuffleFp8.Quantize(h, quant_scale)
        QuantizeAndShuffleFp8.WriteShm(shm_q_h, q, wid, wtid)
        l.barrier()
        out = QuantizeAndShuffleFp8.ReadShm(shm_q_h, wid, wtid, ReadLayout)
        l.barrier()
        return out, dq_act

    @g.jit
    def ComputeRowMax(shm_max, h, tid):
        kLocalMaxFloor: l.constexpr = 1e-6
        kFp8e4m3Max: l.constexpr = 240.0
        kKStages: l.constexpr = QuantizeAndShuffleFp8.kKStages
        kSubGroupSize: l.constexpr = QuantizeAndShuffleFp8.kSubGroupSize
        # Native shm_max is a float2 array; preserve each pair as one 64-bit access.
        shm_max = shm_max.to(l.pointer_type(l.uint64, 3))
        zero = l.full(tid.shape, 0, l.float32, tid.type.layout)
        qs2 = ((zero, zero), (zero, zero))
        ds2 = ((zero, zero), (zero, zero))
        for c in l.static_range(kKStages):
            # Expand native base=c*4 in tuple indices to retain constexpr values.
            lm = (
                l.maximum(
                    kLocalMaxFloor, l.maximum(_max4(h[c * 4]), _max4(h[c * 4 + 2]))
                ),
                l.maximum(
                    kLocalMaxFloor, l.maximum(_max4(h[c * 4 + 1]), _max4(h[c * 4 + 3]))
                ),
            )
            l.store(shm_max + tid, _pack_float2(lm))
            l.barrier()
            for i in l.static_range(16):
                v = _unpack_float2(l.load(shm_max + tid % kSubGroupSize + 16 * i))
                lm = (l.maximum(lm[0], v[0]), l.maximum(lm[1], v[1]))
            qs = (kFp8e4m3Max * amdgcn_rcpf(lm[0]), kFp8e4m3Max * amdgcn_rcpf(lm[1]))
            qs2 = qs2[:c] + (qs,) + qs2[c + 1 :]
            ds = (amdgcn_rcpf(qs2[c][0]), amdgcn_rcpf(qs2[c][1]))
            ds2 = ds2[:c] + (ds,) + ds2[c + 1 :]
            l.barrier()
        return qs2[0] + qs2[1], ds2[0] + ds2[1]

    @g.jit
    def Quantize(h, quant_scale):
        kElementsPerThreadVec4: l.constexpr = (
            QuantizeAndShuffleFp8.kElementsPerThreadVec4
        )
        q = ()
        for i in l.static_range(kElementsPerThreadVec4):
            # Native row=i%2 and half=i/4 must be expanded for tuple indexing.
            s = quant_scale[(i // 4) * 2 + i % 2]
            v = h[i]
            s2 = (s, s)
            xy = _unpack_float2(
                amdgcn_pk_mul_f32(_pack_float2(v[:2]), _pack_float2(s2))
            )
            zw = _unpack_float2(
                amdgcn_pk_mul_f32(_pack_float2(v[2:]), _pack_float2(s2))
            )
            qi = l.full((), 0, l.uint32)
            qi = amdgcn_cvt_pk_fp8_f32(xy[0], xy[1], qi, False)
            qi = amdgcn_cvt_pk_fp8_f32(zw[0], zw[1], qi, True)
            q += (qi,)
        return q

    @g.jit
    def WriteShm(shm_q_h, q, wid, wtid):
        layout: l.constexpr = QuantizeAndShuffleFp8.ShmWriteLayout
        kElementsPerThreadVec4: l.constexpr = (
            QuantizeAndShuffleFp8.kElementsPerThreadVec4
        )
        for i in l.static_range(kElementsPerThreadVec4):
            idx = layout(make_coord(i, wid, wtid))
            l.store(shm_q_h + idx, q[i])

    @g.jit
    def ReadShm(shm_q_h, wid, wtid, ShmReadLayout: l.constexpr):
        kElementsPerThreadVec4: l.constexpr = (
            QuantizeAndShuffleFp8.kElementsPerThreadVec4
        )
        layout: l.constexpr = ShmReadLayout
        # Gluon's uint64 pointer provides native uint2 (eight-byte) indexing.
        s = shm_q_h.to(l.pointer_type(l.uint64, 3))
        out = ()
        for i in l.static_range(kElementsPerThreadVec4):
            # Native views out[i] as two uint2 values, o[0] and o[1].
            o = (
                _load_uint2(s + layout(make_coord(i, 0, wtid))),
                _load_uint2(s + layout(make_coord(i, 1, wtid))),
            )
            out += (o[0] + o[1],)
        return out
