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

"""Native pipelined down projection and packed BF16 atomic output."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    _pack_float2,
    _unpack_float2,
    amdgcn_perm_b32,
    amdgcn_pk_mul_f32,
)
from lib.moe.rocm.fused_moe import ClearMat, HotLoopScheduler
from lib.tal.tensor.layout import Layout, Shape, Stride, make_coord
from triton.experimental.gluon import language as l


@g.jit
def ConditionalWrite(mask, base, vo, value, ID: l.constexpr):
    l.inline_asm_elementwise(
        "s_setvskip $1, "
        + ("0", "1", "2", "3", "4", "5", "6", "7")[ID]
        + "\n global_atomic_pk_add_bf16 $3, $4, $2\n global_atomic_pk_add_bf16 $3, $5, $2 offset:256\n s_setvskip 0, 0\n s_mov_b32 $0, 0;",
        "=s,s,s,v,v,v,~{memory}",
        [mask, base.to(l.uint64), vo.to(l.uint32), value[0], value[1]],
        l.int32,
        False,
        1,
    )


@g.jit
def MultRouteWeights(t, rw2):
    kAccumFragments: l.constexpr = len(t)
    l.static_assert(kAccumFragments % 2 == 0)
    for i in l.static_range(2):
        rw = rw2[i]
        rw_pk = _pack_float2((rw, rw))
        for j in l.static_range(kAccumFragments // 2):
            f = t[j * 2 + i]
            f = _unpack_float2(
                amdgcn_pk_mul_f32(_pack_float2(f[:2]), rw_pk)
            ) + _unpack_float2(amdgcn_pk_mul_f32(_pack_float2(f[2:]), rw_pk))
            t = t[: j * 2 + i] + (f,) + t[j * 2 + i + 1 :]
    return t


@g.jit
def ToBf16Rn(m):
    m_bits = ()
    for j in l.static_range(4):
        f = m[j]
        u = m[j].to(l.uint32, bitcast=True)
        rounded = u + 0x8000
        m_bits += (l.where(f != f, 0x7FFF0000, rounded).to(l.uint32),)
    o_x = amdgcn_perm_b32(m_bits[1], m_bits[0], 0x07060302)
    o_y = amdgcn_perm_b32(m_bits[3], m_bits[2], 0x07060302)
    return o_x, o_y


class FusedMoEBlockScaleFP8Stage2Op:
    __triton_builtin__ = True
    kNumWarps = 4
    kStage = 2
    kSubGroupSize = 16
    kSubGroupPadding = 2
    kSubGroupRowWords = 2 * kSubGroupSize + kSubGroupPadding
    kAccumFragments = 8
    kActivationFragments = 8
    kGroupDim = 256
    kTokenBatch = 8
    kTokenPairs = kTokenBatch // 2
    assert kTokenBatch % 2 == 0
    assert kAccumFragments % 2 == 0
    Shm = (kStage, kAccumFragments * kSubGroupSize * kSubGroupRowWords)
    WriteLayout = Layout(
        Shape(Shape(2, kTokenPairs), kNumWarps, Shape(16, 2)),
        Stride(
            Stride(kNumWarps * 16 * kSubGroupRowWords, 16 * kSubGroupRowWords),
            kTokenPairs * kSubGroupRowWords,
            Stride(2, kSubGroupRowWords),
        ),
    )
    kHalfStrideWords = 32 * kSubGroupRowWords
    ReadLayout = Layout(
        Shape(Shape(kTokenPairs, 2), kNumWarps, Shape(2, 16)),
        Stride(Stride(8, 2 * kHalfStrideWords), 2, Stride(1, kSubGroupRowWords)),
    )

    @g.jit
    def WriteShm(shm, stage, o, wid, wtid):
        layout: l.constexpr = FusedMoEBlockScaleFP8Stage2Op.WriteLayout
        Shm: l.constexpr = FusedMoEBlockScaleFP8Stage2Op.Shm
        for i in l.static_range(len(o)):
            base = layout(make_coord(i, wid, wtid))
            p = shm + stage * Shm[1] + base
            l.store(p, o[i][0])
            l.store(p + 1, o[i][1])

    @g.jit
    def ReadShm(shm, stage, wid, wtid):
        Op: l.constexpr = FusedMoEBlockScaleFP8Stage2Op
        layout: l.constexpr = Op.ReadLayout
        kTokenBatch: l.constexpr = Op.kTokenBatch
        o = ()
        for i in l.static_range(kTokenBatch):
            base = layout(make_coord(i, wid, wtid))
            p = shm + stage * Op.Shm[1] + base
            o += ((l.load(p), l.load(p + Op.kHalfStrideWords)),)
        return o

    @g.jit
    def WriteBack(out, o, tokens, invalid_token_mask, dim, d, wtid):
        vo = ()
        for i in l.static_range(len(tokens)):
            vo += ((tokens[i] * dim + d + wtid * 2) * 2,)
        ConditionalWrite(invalid_token_mask, out, vo[0], o[0], 0)
        ConditionalWrite(invalid_token_mask, out, vo[1], o[1], 1)
        ConditionalWrite(invalid_token_mask, out, vo[2], o[2], 2)
        ConditionalWrite(invalid_token_mask, out, vo[3], o[3], 3)
        ConditionalWrite(invalid_token_mask, out, vo[4], o[4], 4)
        ConditionalWrite(invalid_token_mask, out, vo[5], o[5], 5)
        ConditionalWrite(invalid_token_mask, out, vo[6], o[6], 6)
        ConditionalWrite(invalid_token_mask, out, vo[7], o[7], 7)

    @g.jit
    def Run(
        out,
        shm,
        state,
        dim,
        quant_h,
        dq_act,
        sorted_weights,
        tokens,
        invalid_token_mask,
        tid,
        wid,
        wtid,
        Trait: l.constexpr,
    ):
        Op: l.constexpr = FusedMoEBlockScaleFP8Stage2Op
        kGroupDim: l.constexpr = Op.kGroupDim
        kAccumFragments: l.constexpr = Trait.kAccumFragments
        kStage: l.constexpr = Op.kStage
        l.static_assert(Trait.kNumWarps == Op.kNumWarps)
        l.static_assert(kAccumFragments % 2 == 0)
        state = Trait.LoadStage(state, 0, tid, wid, wtid)
        zero = l.full([256], 0, l.uint32, l.BlockedLayout([1], [64], [4], [0]))
        FusedMoEBlockScaleFP8Stage2Op.WriteShm(
            shm, 1, ((zero, zero),) * kAccumFragments, wid, wtid
        )
        for d in range(0, dim, 2 * kGroupDim):
            for curr in l.static_range(kStage):
                tile_d = d + curr * kGroupDim
                if tile_d < dim:  # Native break, guarded because curr indexes tuples.
                    l.barrier()
                    t = ClearMat(tid, kAccumFragments)
                    HotLoopScheduler(64, 3, 1, 1, 2)
                    state = Trait.LoadStage(state, 1 - curr, tid, wid, wtid)
                    ret = FusedMoEBlockScaleFP8Stage2Op.ReadShm(
                        shm, 1 - curr, wid, wtid
                    )
                    t = Trait.Matmul(t, quant_h, dq_act, state, curr, wtid)
                    t = MultRouteWeights(t, sorted_weights)
                    o = ()
                    for i in l.static_range(kAccumFragments):
                        o += (ToBf16Rn(t[i]),)
                    FusedMoEBlockScaleFP8Stage2Op.WriteShm(shm, curr, o, wid, wtid)
                    if tile_d != 0:
                        FusedMoEBlockScaleFP8Stage2Op.WriteBack(
                            out,
                            ret,
                            tokens,
                            invalid_token_mask,
                            dim,
                            tile_d - kGroupDim,
                            wtid,
                        )
                    l.barrier()
        l.barrier()
        ret = FusedMoEBlockScaleFP8Stage2Op.ReadShm(
            shm, (dim // kGroupDim - 1) % kStage, wid, wtid
        )
        FusedMoEBlockScaleFP8Stage2Op.WriteBack(
            out, ret, tokens, invalid_token_mask, dim, dim - kGroupDim, wtid
        )
        return state[0][3].to(l.uint32)
