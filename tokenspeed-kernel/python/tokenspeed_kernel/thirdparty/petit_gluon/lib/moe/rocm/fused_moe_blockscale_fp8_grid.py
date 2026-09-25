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

"""Native FP8 MoE traits, configuration, and entry point."""

from typing import NamedTuple

import triton
import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    amdgcn_mov_dpp,
    kWarpSize,
    mma_m16n16k32_fp8_fp8_f32,
)
from lib.moe.rocm.fused_moe import Fma4
from lib.moe.rocm.memory_ops import (
    InputLayout,
    W2Layout,
    W13Layout,
    _uninitialized_like,
    _uninitialized_uint4_array,
)
from lib.moe.rocm.ops.onestage_blockscale_fp8_stage1 import (
    FusedMoEBlockScaleFP8Stage1DoubleBufferOp,
)
from lib.moe.rocm.ops.onestage_blockscale_fp8_stage2 import (
    FusedMoEBlockScaleFP8Stage2Op,
)
from lib.moe.rocm.ops.onestage_blockscale_quantization import QuantizeAndShuffleFp8
from lib.moe.rocm.ops.onestage_fused_moe_blockscale_fp8 import (
    LaunchOnestageFusedMoEBlockScaleFP8,
    OnestageFusedMoEBlockScaleFP8,
)
from lib.tal.tensor.layout import Layout, Shape, Stride
from triton.experimental.gluon import language as l


@g.jit
def GetDppValue(src, ctrl):
    kDppRowNewBcastBase: l.constexpr = 0x150
    l.static_assert(src.dtype.primitive_bitwidth == 32)

    bits = src.to(l.int32, bitcast=True)
    if ctrl == 0:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase, 0xF, 0xF, 0)
    elif ctrl == 1:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 1, 0xF, 0xF, 0)
    elif ctrl == 2:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 2, 0xF, 0xF, 0)
    else:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 3, 0xF, 0xF, 0)
    return dst.to(src.dtype, bitcast=True)


@triton.constexpr_function
def Stage1ScaleDppCtrl(stage, col):
    return (stage << 1) + (col >> 1)


@triton.constexpr_function
def Stage2ScaleDppCtrl(stage, col):
    return (stage << 1) + (col >> 1)


@g.jit
def Stage2DqIdx(i, k):
    return k * 2 + i


@g.jit
def MatmulBlockScaleFp8(
    t, w, x, x_scale, w_scale, stage: l.constexpr, ScaleDppCtrl: l.constexpr
):
    for i in l.static_range(4):
        # Native uint2 pointer views are immutable register tuples in Gluon.
        w_2 = (w[i * 2][:2], w[i * 2][2:], w[i * 2 + 1][:2], w[i * 2 + 1][2:])
        for row in l.static_range(2):
            # Expand native tg = row * 4: tuple indices must stay constexpr,
            # and a loop-local constexpr cannot be reassigned by Gluon.
            x_2 = (
                x[row * 4 + stage * 2][:2],
                x[row * 4 + stage * 2][2:],
                x[row * 4 + stage * 2 + 1][:2],
                x[row * 4 + stage * 2 + 1][2:],
            )
            zero = l.full(w[0][0].shape, 0, l.float32, w[0][0].type.layout)
            m_acc = (zero, zero, zero, zero)
            for j in l.static_range(4):
                m_acc = mma_m16n16k32_fp8_fp8_f32(w_2[j], x_2[j], m_acc)
            x_s = (
                (x_scale[0] if row == 0 else x_scale[1])
                if stage == 0
                else (x_scale[2] if row == 0 else x_scale[3])
            )
            w_sdpp = GetDppValue(w_scale, ScaleDppCtrl(stage, i))
            value = Fma4(t[i * 2 + row], w_sdpp * x_s, m_acc)
            t = t[: i * 2 + row] + (value,) + t[i * 2 + row + 1 :]
    return t


# These are the production FusedMoEConfig specializations of the C++ templates.
# The shared memory layouts currently implement this configuration only.
class FusedMoEConfig:
    __triton_builtin__ = True
    kGroupM = 32
    kGroupN = 256
    kGroupDim = 256
    kNumWarps = 4


# Gluon does not support member/subscript assignment in JIT code. Named SSA
# records hold the native members; methods return the updated record. Poison
# register values represent C++ members/locals left uninitialized by constructors.
# They must be overwritten before use, just as in the native implementation.
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


class FusedMoEBlockScaleFP8Stage1Trait:
    __triton_builtin__ = True
    Scalar = l.float8e4nv
    kNumWarps = FusedMoEConfig.kNumWarps
    kActivationFragments = 8
    Input = InputLayout
    W13 = W13Layout
    State = _Stage1State
    kAccumFragments = 8
    assert W13.kTileLoads == 8

    @g.jit
    def Construct(input, w1, w3, tid):
        # tid supplies only the per-thread register shape/layout.
        tile = _uninitialized_uint4_array(tid, W13Layout.kTileLoads)
        scale_w1 = _uninitialized_like(tid, l.float32)
        return _Stage1State(input, w1, w3, (tile, tile), scale_w1)

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
        input, w1, w3, w1_tile, scale_w1 = state
        tile, w1 = W13Layout.LoadTile(w1, 0, wid, wtid)
        w1_tile = (tile, w1_tile[1])
        tile, w1 = W13Layout.LoadTile(w1, 1, wid, wtid)
        w1_tile = (w1_tile[0], tile)
        scale_w1, w1 = W13Layout.LoadScale(w1, tid)
        w1 = W13Layout.AdvanceStep(w1, 0, 2)
        return _Stage1State(input, w1, w3, w1_tile, scale_w1)

    @g.jit
    def ReadInput(shm_act, shm_scale, wtid):
        x = InputLayout.FetchToRegs(shm_act, wtid)
        scale_x = InputLayout.FetchScaleToReg(shm_scale, wtid)
        return x, scale_x

    @g.jit
    def Matmul(t_gate, t_up, x, scale_x, state, tid, wid, wtid):
        input, w1, w3, w1_tile, scale_w1 = state
        tile = _uninitialized_uint4_array(tid, W13Layout.kTileLoads)
        w3_tile = (tile, tile)
        scale_w3 = _uninitialized_like(tid, l.float32)
        for j in l.static_range(2):
            tile, w3 = W13Layout.LoadTile(w3, j, wid, wtid)
            w3_tile = w3_tile[:j] + (tile,) + w3_tile[j + 1 :]
            if j == 0:
                scale_w3, w3 = W13Layout.LoadScale(w3, tid)
            t_gate = MatmulBlockScaleFp8(
                t_gate, w1_tile[j], x, scale_x, scale_w1, j, Stage1ScaleDppCtrl
            )
        w3 = W13Layout.AdvanceStep(w3, 0, 2)
        for j in l.static_range(2):
            tile, w1 = W13Layout.LoadTile(w1, j, wid, wtid)
            w1_tile = w1_tile[:j] + (tile,) + w1_tile[j + 1 :]
            if j == 0:
                scale_w1, w1 = W13Layout.LoadScale(w1, tid)
            t_up = MatmulBlockScaleFp8(
                t_up, w3_tile[j], x, scale_x, scale_w3, j, Stage1ScaleDppCtrl
            )
        w1 = W13Layout.AdvanceStep(w1, 0, 2)
        return t_gate, t_up, _Stage1State(input, w1, w3, w1_tile, scale_w1)


class FusedMoEBlockScaleFP8Stage2Trait:
    __triton_builtin__ = True
    kNumWarps = FusedMoEConfig.kNumWarps
    W2 = W2Layout
    State = _Stage2State
    kAccumFragments = 8
    kActivationFragments = 8
    kOutputPacksPerToken = 2
    assert W2.kTileLoads == 8

    @g.jit
    def Construct(w2, tid):
        # Native declares [2][2][kLoadGlobal], although LoadTile writes only
        # kTileLoads entries. Preserve the untouched tail as well as both stages.
        tile = _uninitialized_uint4_array(tid, W2Layout.kLoadGlobal)
        scale = _uninitialized_like(tid, l.float32)
        return _Stage2State(w2, ((tile, tile), (tile, tile)), (scale, scale))

    @g.jit
    def LoadStage(state, stage: l.constexpr, tid, wid, wtid):
        w2, w2_tile, scale_w2 = state
        tile, w2 = W2Layout.LoadTile(w2, 0, wid, wtid)
        # Immutable-tuple spelling of w2_tile[stage][0] = LoadTile(...).
        tile = tile + w2_tile[stage][0][W2Layout.kTileLoads :]
        w2_tile = w2_tile[:stage] + ((tile, w2_tile[stage][1]),) + w2_tile[stage + 1 :]
        tile, w2 = W2Layout.LoadTile(w2, 1, wid, wtid)
        tile = tile + w2_tile[stage][1][W2Layout.kTileLoads :]
        w2_tile = w2_tile[:stage] + ((w2_tile[stage][0], tile),) + w2_tile[stage + 1 :]
        scale, w2 = W2Layout.LoadScale(w2, tid)
        scale_w2 = scale_w2[:stage] + (scale,) + scale_w2[stage + 1 :]
        w2 = W2Layout.AdvanceStep(w2, 1, 0)
        return _Stage2State(w2, w2_tile, scale_w2)

    @g.jit
    def Matmul(t, quant_h, dq_act, state, stage: l.constexpr, wtid, dbg=False):
        # Native also leaves wtid and dbg unused.
        t = MatmulBlockScaleFp8(
            t,
            state.w2_tile[stage][0],
            quant_h,
            dq_act,
            state.scale_w2[stage],
            0,
            Stage2ScaleDppCtrl,
        )
        t = MatmulBlockScaleFp8(
            t,
            state.w2_tile[stage][1],
            quant_h,
            dq_act,
            state.scale_w2[stage],
            1,
            Stage2ScaleDppCtrl,
        )
        return t


class FusedMoEBlockScaleFP8KernelTrait:
    __triton_builtin__ = True
    Scalar = l.float8e4nv
    kNumWarps = FusedMoEConfig.kNumWarps
    kThreads = kNumWarps * kWarpSize.value
    kTokenBatch = 8
    Input = InputLayout
    W2 = W2Layout
    W13 = W13Layout
    Stage1Trait = FusedMoEBlockScaleFP8Stage1Trait
    Stage1Op = FusedMoEBlockScaleFP8Stage1DoubleBufferOp
    Stage2Trait = FusedMoEBlockScaleFP8Stage2Trait
    Stage2Op = FusedMoEBlockScaleFP8Stage2Op
    kElementsPerThread = (FusedMoEConfig.kGroupM * FusedMoEConfig.kGroupN) // kThreads
    kElementsPerThreadVec4 = kElementsPerThread // 4

    QuantizationShuffleReadLayout = Layout(
        Shape(kElementsPerThreadVec4, 2, Shape(16, kNumWarps)),
        Stride(2 * kWarpSize.value, 16, Stride(1, 32)),
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
        n_blocks,
        k_blocks,
        w2_offset,
    ):
        # Native Kernel& is passed as its dim_, inter_dim_, and retained W2
        # offset; the three updated layout members are returned to the caller.
        # Gluon has no uint4 pointer type. Keep native vector offsets, then
        # convert them to bytes at the pointer addition. Scales use uint32*.
        Config: l.constexpr = FusedMoEConfig
        Kernel: l.constexpr = OnestageFusedMoEBlockScaleFP8
        kVecSize: l.constexpr = 16 // (
            FusedMoEBlockScaleFP8KernelTrait.Scalar.primitive_bitwidth // 8
        )
        scales_w13 = scales_w13.to(l.pointer_type(l.uint32))
        scales_w2 = scales_w2.to(l.pointer_type(l.uint32))
        w1_ptr = (
            w13_base
            + (
                expert_id * (2 * inter_dim * dim) // kVecSize
                + tile_k * Config.kGroupDim * dim // kVecSize
            )
            * kVecSize
        )
        scale_w1_ptr = (
            scales_w13
            + expert_id * (2 * k_blocks * n_blocks)
            + tile_k
            * (Config.kGroupDim // Kernel.kScaleBlockSize)
            * (dim // Kernel.kScaleBlockSize)
        )
        w13_value_range = Config.kGroupDim * dim
        w13_scale_range = (Config.kGroupDim // Kernel.kScaleBlockSize) * n_blocks * 4
        w1 = W13Layout.Initialize(
            w1_ptr,
            w13_value_range,
            scale_w1_ptr.to(l.pointer_type(l.uint8)),
            w13_scale_range,
            dim,
        )
        w3 = W13Layout.Initialize(
            w1_ptr + (inter_dim * dim // kVecSize) * kVecSize,
            w13_value_range,
            (scale_w1_ptr + k_blocks * n_blocks).to(l.pointer_type(l.uint8)),
            w13_scale_range,
            dim,
        )

        # w2 is pre-shuffled as [rbi][cbi][kki][bni][kpi], blockN=16/blockK=32.
        # Advancing logical K by 256 advances cbi by 8 blocks = 4096 FP8 elements.
        w2_ptr = (
            w2
            + (
                expert_id * (dim * inter_dim) // kVecSize
                + tile_k * (Config.kGroupDim * 16) // kVecSize
            )
            * kVecSize
        )
        scale_w2_ptr = (
            scales_w2
            + expert_id * (n_blocks * k_blocks)
            + tile_k * (Config.kGroupDim // Kernel.kScaleBlockSize)
        )
        w2_value_range = dim * inter_dim - tile_k * Config.kGroupDim * 16
        w2_scale_range = (
            n_blocks * k_blocks * 4
            - tile_k * (Config.kGroupDim // Kernel.kScaleBlockSize) * 4
        )
        w2 = W2Layout.Initialize(
            w2_ptr,
            w2_value_range,
            scale_w2_ptr.to(l.pointer_type(l.uint8)),
            w2_scale_range,
            inter_dim,
            w2_offset,
        )
        return w1, w3, w2


class FusedMoEBlockScaleFP8Kernel(OnestageFusedMoEBlockScaleFP8):
    """Production specialization of the native Config/KernelTrait alias."""

    Config = FusedMoEConfig
    Trait = FusedMoEBlockScaleFP8KernelTrait


def FusedMoEBlockScaleFP8(
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
    stream,
    num_persistent_tgs,
):
    """Accumulate FP8 MoE into BF16 ``out``; buffers and dimensions follow native API.

    ``n`` is input/output width, ``k`` intermediate width, ``m`` token count.
    Routing and scales are already in Petit layout. ``out`` must be zeroed by
    the caller. Return 0 on success, 1 for invalid shape, 2 for null buffers.
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
        0,
        stream,
        num_persistent_tgs,
        FusedMoEBlockScaleFP8Kernel.Config,
        FusedMoEBlockScaleFP8Kernel.Trait,
    )
    return 0
