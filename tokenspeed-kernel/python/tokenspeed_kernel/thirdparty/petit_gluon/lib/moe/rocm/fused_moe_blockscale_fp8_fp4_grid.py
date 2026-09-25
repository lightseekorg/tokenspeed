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

"""Native MXFP4-weight MoE traits, configuration, and entry point."""

from typing import NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    amdgcn_ds_bpermute,
    amdgcn_pk_fma_f32,
    kWarpSize,
    mma_m16n16k32_bf8_fp8_f32,
)
from lib.gemm.rocm.quantization.dequant import Fp4ToBf8
from lib.moe.rocm.memory_ops import (
    InputLayout,
    MxFp4WeightLayout,
    _uninitialized_like,
    _uninitialized_uint4_array,
)
from lib.moe.rocm.ops.onestage_blockscale_fp8_stage1 import (
    FusedMoEBlockScaleFP8Stage1SingleBufferOp,
)
from lib.moe.rocm.ops.onestage_blockscale_fp8_stage2 import (
    FusedMoEBlockScaleFP8Stage2Op,
)
from lib.moe.rocm.ops.onestage_blockscale_quantization import QuantizeAndShuffleFp8
from lib.moe.rocm.ops.onestage_fused_moe_blockscale_fp8 import (
    LaunchOnestageFusedMoEBlockScaleFP8,
    OnestageFusedMoEBlockScaleFP8,
)
from lib.moe.rocm.warp_schedule import WarpSchedule
from lib.tal.tensor.layout import Layout, Shape, Stride
from triton.experimental.gluon import language as l


@g.jit
def Fma4(a, s, c):
    a2 = (a[:2], a[2:])
    c2 = (c[:2], c[2:])
    s2 = (s[:2], s[2:])
    r0 = amdgcn_pk_fma_f32(s2[0], c2[0], a2[0])
    r1 = amdgcn_pk_fma_f32(s2[1], c2[1], a2[1])
    return r0 + r1


@g.jit
def LoadFp8E8m0Scale(packed):
    u = ()
    for i in l.static_range(4):
        b = ((packed >> (i * 8)) & 0xFF).to(l.uint8)
        u += (b.to(l.uint32) << 23,)
    # Native union view and vector multiply, including BF8's 2^15 compensation.
    kExpBias: l.constexpr = (32768.0, 32768.0, 32768.0, 32768.0)
    f = ()
    for i in l.static_range(4):
        f += (u[i].to(l.float32, bitcast=True) * kExpBias[i],)
    return f


@g.jit
def MatmulBlockScaleFp4(t, w, x, x_scale, packed_scale, stage: l.constexpr):
    kLoadGlobal: l.constexpr = len(w)
    kActivationFragments: l.constexpr = len(x)
    tid = l.arange(0, 256, layout=l.BlockedLayout([1], [64], [4], [0]))
    wtid = tid % kWarpSize
    kFragmentsPerRow: l.constexpr = kActivationFragments // 2
    for i in l.static_range(kLoadGlobal):
        l.static_assert(kLoadGlobal == 4)
        for j in l.static_range(4):
            # ds_bpermute implements native __shfl's wave-local source index.
            s = amdgcn_ds_bpermute(((wtid & 48) + i * 4 + j) * 4, packed_scale)
            w_scale = LoadFp8E8m0Scale(s)
            qw = w[i][j]
            bf8 = Fp4ToBf8(qw)
            for row in l.static_range(2):
                # Native uint2 pointer view, with constexpr tuple indices.
                x_2 = (
                    x[row * kFragmentsPerRow + stage * 2][:2],
                    x[row * kFragmentsPerRow + stage * 2][2:],
                    x[row * kFragmentsPerRow + stage * 2 + 1][:2],
                    x[row * kFragmentsPerRow + stage * 2 + 1][2:],
                )
                x_s = x_scale[stage * 2 + row]
                x_s4 = (x_s, x_s, x_s, x_s)
                fs = (
                    w_scale[0] * x_s4[0],
                    w_scale[1] * x_s4[1],
                    w_scale[2] * x_s4[2],
                    w_scale[3] * x_s4[3],
                )
                zero = l.full(tid.shape, 0, l.float32, tid.type.layout)
                m_acc = (zero, zero, zero, zero)
                m_acc = mma_m16n16k32_bf8_fp8_f32(bf8, x_2[j], m_acc)
                value = Fma4(t[i * 2 + row], fs, m_acc)
                t = t[: i * 2 + row] + (value,) + t[i * 2 + row + 1 :]
    return t


# Production specialization of the native Config templates.
class FusedMoEMxFp4Config:
    __triton_builtin__ = True
    kGroupM = 32
    kGroupN = 256
    kGroupDim = 256
    kNumWarps = 4
    # Both projections consume the inter-dimension slice selected by blockIdx.x.
    kStage2GroupInterDim = kGroupDim


class _Stage1State(NamedTuple):
    input: object
    w1: object
    w3: object
    w1_tile: object
    scale_w1: object


class _Stage2State(NamedTuple):
    w2: object
    w2_tile: object
    scale_w2: object


class FusedMoEBlockScaleFP8Fp4Stage1Trait:
    __triton_builtin__ = True
    kNumWarps = FusedMoEMxFp4Config.kNumWarps
    kActivationFragments = FusedMoEMxFp4Config.kGroupDim // 32
    kKStages = FusedMoEMxFp4Config.kGroupDim // 128
    Input = InputLayout
    W13 = MxFp4WeightLayout
    State = _Stage1State
    kAccumFragments = 2 * W13.kLoadGlobal
    assert kAccumFragments == FusedMoEMxFp4Config.kGroupDim // 32

    @g.jit
    def Construct(input, w1, w3, tid):
        kKStages: l.constexpr = FusedMoEBlockScaleFP8Fp4Stage1Trait.kKStages
        tile = _uninitialized_uint4_array(tid, MxFp4WeightLayout.kLoadGlobal)
        scale = _uninitialized_like(tid, l.uint32)
        return _Stage1State(
            input,
            w1,
            w3,
            (tile,) * kKStages,
            (scale,) * kKStages,
        )

    @g.jit
    def PrefetchInput(state, shm_act, shm_scale, wid, wtid, token_select, tokens, m):
        input, w1, w3, w1_tile, scale_w1 = state
        input = InputLayout.FetchAsync(input, shm_act, wid, wtid, tokens)
        input = InputLayout.FetchScaleAsync(
            input, shm_scale, wid, wtid, token_select, m
        )
        return _Stage1State(input, w1, w3, w1_tile, scale_w1)

    @g.jit
    def LoadInitial(state, tid, wid, wtid):
        kKStages: l.constexpr = FusedMoEBlockScaleFP8Fp4Stage1Trait.kKStages
        input, w1, w3, w1_tile, scale_w1 = state
        for j in l.static_range(kKStages):
            tile, w1 = MxFp4WeightLayout.LoadTile(w1, j, wid, wtid)
            w1_tile = w1_tile[:j] + (tile,) + w1_tile[j + 1 :]
            scale, w1 = MxFp4WeightLayout.LoadScale(w1, tid)
            scale_w1 = scale_w1[:j] + (scale,) + scale_w1[j + 1 :]
            w1 = MxFp4WeightLayout.AdvanceStep(w1, 0, 1)
        return _Stage1State(input, w1, w3, w1_tile, scale_w1)

    @g.jit
    def ReadInput(shm_act, shm_scale, wtid):
        x = InputLayout.FetchToRegsFP4(shm_act, wtid)
        scale_x = InputLayout.FetchScaleToReg(shm_scale, wtid)
        return x, scale_x

    @g.jit
    def Matmul(t_gate, t_up, x, scale_x, state, tid, wid, wtid):
        kKStages: l.constexpr = FusedMoEBlockScaleFP8Fp4Stage1Trait.kKStages
        input, w1, w3, w1_tile, scale_w1 = state
        tile = _uninitialized_uint4_array(tid, MxFp4WeightLayout.kLoadGlobal)
        w3_tile = (tile,) * kKStages
        scale_w3 = (_uninitialized_like(tid, l.uint32),) * kKStages
        for j in l.static_range(kKStages):
            tile, w3 = MxFp4WeightLayout.LoadTile(w3, j, wid, wtid)
            w3_tile = w3_tile[:j] + (tile,) + w3_tile[j + 1 :]
            scale, w3 = MxFp4WeightLayout.LoadScale(w3, tid)
            scale_w3 = scale_w3[:j] + (scale,) + scale_w3[j + 1 :]
            w3 = MxFp4WeightLayout.AdvanceStep(w3, 0, 1)
            t_gate = MatmulBlockScaleFp4(t_gate, w1_tile[j], x, scale_x, scale_w1[j], j)
        for j in l.static_range(kKStages):
            tile, w1 = MxFp4WeightLayout.LoadTile(w1, j, wid, wtid)
            w1_tile = w1_tile[:j] + (tile,) + w1_tile[j + 1 :]
            scale, w1 = MxFp4WeightLayout.LoadScale(w1, tid)
            scale_w1 = scale_w1[:j] + (scale,) + scale_w1[j + 1 :]
            w1 = MxFp4WeightLayout.AdvanceStep(w1, 0, 1)
            t_up = MatmulBlockScaleFp4(t_up, w3_tile[j], x, scale_x, scale_w3[j], j)
        return t_gate, t_up, _Stage1State(input, w1, w3, w1_tile, scale_w1)


class FusedMoEBlockScaleFP8Fp4Stage2Trait:
    __triton_builtin__ = True
    kNumWarps = FusedMoEMxFp4Config.kNumWarps
    Schedule = WarpSchedule(FusedMoEMxFp4Config)
    kKStages = FusedMoEMxFp4Config.kGroupN // 128
    W2 = MxFp4WeightLayout
    State = _Stage2State
    kAccumFragments = 2 * W2.kLoadGlobal
    kActivationFragments = FusedMoEMxFp4Config.kGroupDim // 32
    kOutputPacksPerToken = FusedMoEMxFp4Config.kGroupN // 128
    assert kAccumFragments == Schedule.kAccumulatorFragments
    assert kActivationFragments == FusedMoEMxFp4Config.kGroupDim // 32
    assert kOutputPacksPerToken > 0

    @g.jit
    def Construct(w2, tid):
        kKStages: l.constexpr = FusedMoEBlockScaleFP8Fp4Stage2Trait.kKStages
        tile = _uninitialized_uint4_array(tid, MxFp4WeightLayout.kLoadGlobal)
        scale = _uninitialized_like(tid, l.uint32)
        return _Stage2State(w2, ((tile,) * kKStages,) * 2, ((scale,) * kKStages,) * 2)

    @g.jit
    def LoadStage(state, stage: l.constexpr, tid, wid, wtid):
        kKStages: l.constexpr = FusedMoEBlockScaleFP8Fp4Stage2Trait.kKStages
        w2, w2_tile, scale_w2 = state
        for j in l.static_range(kKStages):
            tile, w2 = MxFp4WeightLayout.LoadTile(w2, j, wid, wtid)
            tiles = w2_tile[stage][:j] + (tile,) + w2_tile[stage][j + 1 :]
            w2_tile = w2_tile[:stage] + (tiles,) + w2_tile[stage + 1 :]
            scale, w2 = MxFp4WeightLayout.LoadScale(w2, tid)
            scales = scale_w2[stage][:j] + (scale,) + scale_w2[stage][j + 1 :]
            scale_w2 = scale_w2[:stage] + (scales,) + scale_w2[stage + 1 :]
            w2 = MxFp4WeightLayout.AdvanceStep(w2, 0, 1)
        w2 = MxFp4WeightLayout.AdvanceStep(
            w2, FusedMoEMxFp4Config.kGroupN // 128, -kKStages
        )
        return _Stage2State(w2, w2_tile, scale_w2)

    @g.jit
    def Matmul(t, quant_h, dq_act, state, stage: l.constexpr, wtid):
        kKStages: l.constexpr = FusedMoEBlockScaleFP8Fp4Stage2Trait.kKStages
        for j in l.static_range(kKStages):
            t = MatmulBlockScaleFp4(
                t, state.w2_tile[stage][j], quant_h, dq_act, state.scale_w2[stage][j], j
            )
        return t


class FusedMoEBlockScaleFP8Fp4KernelTrait:
    __triton_builtin__ = True
    Scalar = l.float8e4nv
    kNumWarps = FusedMoEMxFp4Config.kNumWarps
    kThreads = kNumWarps * kWarpSize.value
    kTokenBatch = 8
    Input = InputLayout
    W13 = MxFp4WeightLayout
    W2 = MxFp4WeightLayout
    Stage1Trait = FusedMoEBlockScaleFP8Fp4Stage1Trait
    Stage1Op = FusedMoEBlockScaleFP8Stage1SingleBufferOp
    Stage2Trait = FusedMoEBlockScaleFP8Fp4Stage2Trait
    Stage2Op = FusedMoEBlockScaleFP8Stage2Op
    kElementsPerThread = (
        FusedMoEMxFp4Config.kGroupM * FusedMoEMxFp4Config.kGroupN
    ) // kThreads
    kElementsPerThreadVec4 = kElementsPerThread // 4
    QuantizationShuffleReadLayout = Layout(
        Shape(kElementsPerThreadVec4, 2, Shape(16, 4)),
        Stride(2 * kWarpSize.value, kWarpSize.value, Stride(1, 16)),
    )

    QuantizeAndShuffleOp = QuantizeAndShuffleFp8

    @g.jit
    def InitializeWeights(
        w13_base,
        w2,
        scales_w13,
        scales_w2,
        expert_id,
        tile_k,
        dim,
        inter_dim,
        unused_input_scale_blocks,
        unused_inter_dim_scale_blocks,
        w2_offset,
    ):
        # Kernel& members are passed/returned as SSA; uint4* arithmetic uses
        # native vector offsets multiplied by 16 for Gluon's byte pointers.
        Config: l.constexpr = FusedMoEMxFp4Config
        l.static_assert(
            Config.kGroupN == 256, "The scale requires 256 elements per group dim"
        )
        kScaleGroupK: l.constexpr = 128
        kScaleGroupN: l.constexpr = 64
        kLayoutN: l.constexpr = 16
        kWeightVecSize: l.constexpr = 16 * 2
        kRowGroupSize: l.constexpr = MxFp4WeightLayout.kRowGroupSize
        scales_w13 = scales_w13.to(l.pointer_type(l.uint32))
        scales_w2 = scales_w2.to(l.pointer_type(l.uint32))
        w1_ptr = (
            w13_base
            + (
                expert_id * (2 * inter_dim * dim) // kWeightVecSize
                + tile_k * Config.kGroupDim * dim // kWeightVecSize
            )
            * 16
        )
        w13_scale_words_per_expert = (2 * dim * inter_dim) // kRowGroupSize // 4
        w13_scale_words_per_col = (
            (dim // kScaleGroupK) * (Config.kGroupN // kScaleGroupN) * kWarpSize
        )
        scale_w1_ptr = (
            scales_w13
            + expert_id * w13_scale_words_per_expert
            + tile_k * w13_scale_words_per_col
        )
        w13_value_range = Config.kGroupDim * dim // 2
        w13_scale_range = w13_scale_words_per_col * 4
        w1 = MxFp4WeightLayout.Initialize(
            w1_ptr,
            w13_value_range,
            scale_w1_ptr.to(l.pointer_type(l.uint8)),
            w13_scale_range,
            dim,
        )
        w3 = MxFp4WeightLayout.Initialize(
            w1_ptr + (inter_dim * dim // kWeightVecSize) * 16,
            w13_value_range,
            (scale_w1_ptr + w13_scale_words_per_expert // 2).to(
                l.pointer_type(l.uint8)
            ),
            w13_scale_range,
            dim,
        )
        w2_value_k_tile_offset = (
            tile_k * Config.kStage2GroupInterDim * kLayoutN // kWeightVecSize
        )
        w2_ptr = (
            w2
            + (expert_id * (dim * inter_dim) // kWeightVecSize + w2_value_k_tile_offset)
            * 16
        )
        w2_scale_words_per_expert = (
            (dim * inter_dim) // MxFp4WeightLayout.kRowGroupSize // 4
        )
        w2_scale_k_tile_offset = (
            tile_k
            * (Config.kStage2GroupInterDim // kScaleGroupK)
            * (Config.kGroupN // kScaleGroupN)
            * kWarpSize
        )
        scale_w2_ptr = (
            scales_w2 + expert_id * w2_scale_words_per_expert + w2_scale_k_tile_offset
        )
        w2_value_range = (inter_dim * dim) // 2 - w2_value_k_tile_offset * 16
        w2_scale_range = w2_scale_words_per_expert * 4 - w2_scale_k_tile_offset * 4
        w2 = MxFp4WeightLayout.Initialize(
            w2_ptr,
            w2_value_range,
            scale_w2_ptr.to(l.pointer_type(l.uint8)),
            w2_scale_range,
            inter_dim,
        )
        return w1, w3, w2


class FusedMoEBlockScaleFP8Fp4Kernel(OnestageFusedMoEBlockScaleFP8):
    """Production specialization of the native Config/KernelTrait alias."""

    Config = FusedMoEMxFp4Config
    Trait = FusedMoEBlockScaleFP8Fp4KernelTrait


def FusedMoEBlockScaleFP8MXFP4Weight(
    out,
    act,
    w13,
    w2,
    sorted_token_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    scales_act,
    scales_w13,
    scales_w2,
    max_num_m_blocks,
    m,
    n,
    k,
    num_experts,
    stream,
    num_persistent_tgs,
):
    """Accumulate MXFP4-weight MoE into caller-zeroed BF16 ``out``.

    Buffers use native Petit layouts. ``m/n/k`` mean token/input/intermediate
    dimensions. ``num_experts`` bounds sorted expert IDs; ``stream`` selects
    execution stream. Return native status 0, 1 (shape), or 2 (null buffers).
    """
    if out is None or num_valid_ids is None:
        return 2
    if n == 0 or k == 0 or n % 128 or k % 128:
        return 1
    if any(
        p is None
        for p in (
            act,
            w13,
            w2,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            scales_act,
            scales_w13,
            scales_w2,
        )
    ):
        return 2
    LaunchOnestageFusedMoEBlockScaleFP8(
        out,
        act,
        w13,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        scales_act,
        scales_w13,
        scales_w2,
        max_num_m_blocks,
        m,
        n,
        k,
        num_experts,
        stream,
        num_persistent_tgs,
        FusedMoEBlockScaleFP8Fp4Kernel.Config,
        FusedMoEBlockScaleFP8Fp4Kernel.Trait,
    )
    return 0
