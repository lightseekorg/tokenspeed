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

"""Native fused MoE declarations and common device operations."""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import ClassVar, NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    HAS_AMD_SCHED_BARRIER,
    HAS_AMD_SCHED_GROUP_BARRIER,
    _pack_float2,
    _unpack_float2,
    amdgcn_exp2f,
    amdgcn_pk_fma_f32,
    amdgcn_pk_mul_f32,
    amdgcn_rcpf,
    amdgcn_sched_barrier,
    amdgcn_sched_group_barrier,
)
from triton.experimental.gluon import language as l


class FusedMoEDataType(IntEnum):
    kNone = 0
    kMxFp4 = 1
    kNvFp4 = 2
    kChannelScaleFp8 = 3
    kBlockScaleFp8 = 4
    kBf16 = 5


class FusedMoEWeightOrdering(IntEnum):
    kNativeMxFp4 = 0
    kPetitMxFp4 = 1
    kPetitFp8 = 2


class FusedMoEStages(IntEnum):
    kOneStage = 0
    kTwoStage = 1


class FusedMoEMfmaShape(IntEnum):
    kMfmaFp816x16x32 = 0
    kMfmaBf16MxFp4 = 1
    kMfmaScaleFp4MxFp4 = 2


class FusedMoEActivationFunction(IntEnum):
    kSiluDot = 0
    kOpenAISwiGLU = 1
    kKimiSitu = 2


class FusedMoEStage1Buffering(IntEnum):
    kSingleBuffer = 0
    kDoubleBuffer = 1


class FusedMoEWeightLoadPolicy(IntEnum):
    kCached = 0
    kNonTemporal = 1


class MegaMoETileShape(IntEnum):
    kN256 = 0
    kN128 = 1


class MegaMoEProducerGeometry(IntEnum):
    kCta56 = 0
    kCta64 = 1
    kCta128 = 2
    kCta192 = 3


class FusedMoEStage1TileShape(IntEnum):
    kM32N256 = 0
    kM64N512 = 1


class TokenMetadata(NamedTuple):
    token_topk_idx: object
    src_rank: object


@dataclass(frozen=True)
class FusedMoESolutionId:
    act_dtype: FusedMoEDataType
    weight_dtype: FusedMoEDataType
    bias_dtype: FusedMoEDataType
    weight_ordering: FusedMoEWeightOrdering
    mfma: FusedMoEMfmaShape
    stages: FusedMoEStages
    activation: FusedMoEActivationFunction
    stage1_buffering: FusedMoEStage1Buffering
    dim_div64: int = 0
    inter_dim_div64: int = 0
    weight_load_policy: FusedMoEWeightLoadPolicy = FusedMoEWeightLoadPolicy.kCached
    padding: int = 0
    kShapeAlignment: ClassVar[int] = 64
    kMaxShapeDiv64: ClassVar[int] = 0xFF

    def Dim(self):
        return self.dim_div64 * self.kShapeAlignment

    def InterDim(self):
        return self.inter_dim_div64 * self.kShapeAlignment

    @staticmethod
    def IsShapeEncodable(dim, inter_dim):
        return (
            dim != 0
            and inter_dim != 0
            and dim % 64 == 0
            and inter_dim % 64 == 0
            and dim // 64 <= 0xFF
            and inter_dim // 64 <= 0xFF
        )

    def WithShape(self, dim, inter_dim):
        return replace(
            self, dim_div64=(dim // 64) & 0xFF, inter_dim_div64=(inter_dim // 64) & 0xFF
        )

    def WithWeightLoadPolicy(self, policy):
        return replace(self, weight_load_policy=policy)

    def NumRanksLog2(self):
        return (self.Repr() >> 24) & 3

    def NumExperts(self):
        return (((self.Repr() >> 26) & 15) + ((self.Repr() >> 50) & 1) * 16 + 1) * 32

    def TopK(self):
        return ((self.Repr() >> 30) & 15) | (((self.Repr() >> 51) & 1) << 4)

    def HiddenSizeDiv64(self):
        return (self.Repr() >> 34) & 0xFF

    def W2TileShape(self):
        return MegaMoETileShape((self.Repr() >> 42) & 1)

    def MegaInterDimDiv64(self):
        return (((self.Repr() >> 43) & 0x1F) + 1) * 8

    def ProducerGeometry(self):
        return MegaMoEProducerGeometry((self.Repr() >> 48) & 3)

    def ProducerBlocks(self):
        return (56, 64, 128, 192)[self.ProducerGeometry()]

    def Stage1TileShape(self):
        return FusedMoEStage1TileShape((self.Repr() >> 41) & 1)

    def Stage1TileM(self):
        return 32 << self.Stage1TileShape()

    def Stage1TileN(self):
        return 256 << self.Stage1TileShape()

    def WithStage1TileShape(self, shape):
        kMask = 1 << 41
        return self.FromRepr((self.Repr() & ~kMask) | (int(shape) << 41))

    def WithMegaMoEConfig(
        self,
        num_ranks,
        experts,
        num_topk,
        hidden_size,
        inter_dim,
        producer_geometry,
        w2_tile_shape=MegaMoETileShape.kN256,
    ):
        rank_log2 = 0
        while num_ranks > 1:
            rank_log2 += 1
            num_ranks >>= 1
        return self.FromRepr(
            (self.Repr() & 0x00FFFFFF)
            | (rank_log2 << 24)
            | (((experts // 32 - 1) & 15) << 26)
            | ((experts // 32 - 1) >> 4 << 50)
            | ((num_topk & 15) << 30)
            | (num_topk >> 4 << 51)
            | (hidden_size // 64 << 34)
            | (int(w2_tile_shape) << 42)
            | (inter_dim // 512 - 1 << 43)
            | (int(producer_geometry) << 48)
        )

    def Repr(self):
        return (
            int(self.act_dtype)
            | (int(self.weight_dtype) << 4)
            | (int(self.bias_dtype) << 8)
            | (int(self.weight_ordering) << 12)
            | (int(self.mfma) << 14)
            | (int(self.stages) << 16)
            | (int(self.activation) << 20)
            | (int(self.stage1_buffering) << 23)
            | (self.dim_div64 << 24)
            | (self.inter_dim_div64 << 32)
            | (int(self.weight_load_policy) << 40)
            | (self.padding << 41)
        )

    @staticmethod
    def FromRepr(repr):
        # Keep raw enum integers: C++ FromRepr also accepts unknown enum values.
        return FusedMoESolutionId(
            repr & 15,
            repr >> 4 & 15,
            repr >> 8 & 15,
            repr >> 12 & 3,
            repr >> 14 & 3,
            repr >> 16 & 15,
            repr >> 20 & 7,
            repr >> 23 & 1,
            repr >> 24 & 255,
            repr >> 32 & 255,
            repr >> 40 & 1,
            repr >> 41 & 0x7FFFFF,
        )

    @staticmethod
    def MakeBase(
        act_dtype,
        weight_dtype,
        bias_dtype,
        weight_ordering,
        mfma,
        stages,
        activation,
        stage1_buffering,
    ):
        return FusedMoESolutionId(
            act_dtype,
            weight_dtype,
            bias_dtype,
            weight_ordering,
            mfma,
            stages,
            activation,
            stage1_buffering,
        )

    @staticmethod
    def MakeMegaBase(
        act_dtype,
        weight_dtype,
        bias_dtype,
        weight_ordering,
        mfma,
        stages,
        activation,
        stage1_buffering,
    ):
        return FusedMoESolutionId.MakeBase(
            act_dtype,
            weight_dtype,
            bias_dtype,
            weight_ordering,
            mfma,
            stages,
            activation,
            stage1_buffering,
        )

    @staticmethod
    def Make(
        act_dtype,
        weight_dtype,
        bias_dtype,
        weight_ordering,
        mfma,
        stages,
        activation,
        stage1_buffering,
        dim,
        inter_dim,
    ):
        return FusedMoESolutionId.MakeBase(
            act_dtype,
            weight_dtype,
            bias_dtype,
            weight_ordering,
            mfma,
            stages,
            activation,
            stage1_buffering,
        ).WithShape(dim, inter_dim)


@dataclass
class FusedMoE1StageParams:
    out: object
    act: object
    w13: object
    w2: object
    sorted_token_ids: object
    sorted_weights: object
    sorted_expert_ids: object
    num_valid_ids: object
    topk: int
    scales_act: object
    scales_w13: object
    scales_w2: object
    max_num_m_blocks: int
    m: int
    n: int
    k: int
    num_experts: int
    stream: object
    num_persistent_tgs: int
    w13_bias: object = None
    w2_bias: object = None


@dataclass
class FusedMoE2StageCommonParams:
    intermediate: object
    intermediate_bytes: int
    sorted_token_ids: object
    sorted_expert_ids: object
    num_valid_ids: object
    topk: int
    max_num_m_blocks: int
    m: int
    n: int
    k: int
    num_experts: int
    stream: object
    num_persistent_tgs: int


@dataclass
class FusedMoE2Stage1Params:
    common: object
    act: object
    w13: object
    scales_act: object
    scales_w13: object
    w13_bias: object = None


@dataclass
class FusedMoE2Stage2Params:
    common: object
    out: object
    w2: object
    sorted_weights: object
    scales_w2: object
    w2_bias: object = None


kFusedMoEErrorInvalidSolution = 1
kFusedMoEErrorUnsupported = 3

# Compatibility names for the previous public wrappers.
kFusedMoEErrorInvalidShape = 1
kFusedMoEErrorInvalidArgument = 2
kFusedMoEErrorUnsupportedArch = 3


@g.jit
def Fma4(a, s, c):
    a2 = (a[:2], a[2:])
    c2 = (c[:2], c[2:])
    s2 = (s, s)
    r0 = amdgcn_pk_fma_f32(s2, c2[0], a2[0])
    r1 = amdgcn_pk_fma_f32(s2, c2[1], a2[1])
    return r0 + r1


@g.jit
def SiluDot(gate, up):
    kMinusLog2e: l.constexpr = -1.4426950408889634
    kMinusLog2e2: l.constexpr = (kMinusLog2e, kMinusLog2e)
    kOne2: l.constexpr = (1.0, 1.0)
    g2 = (gate[:2], gate[2:])
    u2 = (up[:2], up[2:])
    # Immutable tuple writes retain the native vector/scalar instruction order.
    i2 = (
        (g2[0][0] * kMinusLog2e2[0], g2[0][1] * kMinusLog2e2[1]),
        (g2[1][0] * kMinusLog2e2[0], g2[1][1] * kMinusLog2e2[1]),
    )
    i2 = (
        (amdgcn_exp2f(i2[0][0]), amdgcn_exp2f(i2[0][1])),
        (amdgcn_exp2f(i2[1][0]), amdgcn_exp2f(i2[1][1])),
    )
    i2 = (
        (i2[0][0] + kOne2[0], i2[0][1] + kOne2[1]),
        (i2[1][0] + kOne2[0], i2[1][1] + kOne2[1]),
    )
    i2 = (
        (amdgcn_rcpf(i2[0][0]), amdgcn_rcpf(i2[0][1])),
        (amdgcn_rcpf(i2[1][0]), amdgcn_rcpf(i2[1][1])),
    )
    i2f = i2
    r0 = _unpack_float2(
        amdgcn_pk_mul_f32(
            amdgcn_pk_mul_f32(_pack_float2(gate[:2]), _pack_float2(i2f[0])),
            _pack_float2(u2[0]),
        )
    )
    r1 = _unpack_float2(
        amdgcn_pk_mul_f32(
            amdgcn_pk_mul_f32(_pack_float2(gate[2:]), _pack_float2(i2f[1])),
            _pack_float2(u2[1]),
        )
    )
    return r0 + r1


@g.jit
def HotLoopScheduler(
    kInstMFMA: l.constexpr,
    kInstVmemRead: l.constexpr = 0,
    kInstDsRead: l.constexpr = 0,
    kInstDsWrite: l.constexpr = 0,
    kInstVALU: l.constexpr = 0,
):
    if HAS_AMD_SCHED_GROUP_BARRIER and HAS_AMD_SCHED_BARRIER:
        kSchedGroupId: l.constexpr = 0
        kInstIssue: l.constexpr = kInstVmemRead + kInstDsRead + kInstDsWrite + kInstVALU
        if kInstMFMA > 0 and kInstIssue > 0:
            kInstMFMAPerIssue: l.constexpr = (
                4
                if kInstMFMA // kInstIssue > 12
                else (2 if kInstMFMA // kInstIssue > 6 else 1)
            )
            for i in l.static_range(kInstDsWrite):
                amdgcn_sched_group_barrier(0x200, 1, kSchedGroupId)
                amdgcn_sched_group_barrier(0x8, kInstMFMAPerIssue, kSchedGroupId)
            for i in l.static_range(kInstVmemRead):
                amdgcn_sched_group_barrier(0x20, 1, kSchedGroupId)
                amdgcn_sched_group_barrier(0x8, kInstMFMAPerIssue, kSchedGroupId)
            for i in l.static_range(kInstDsRead):
                amdgcn_sched_group_barrier(0x100, 1, kSchedGroupId)
                amdgcn_sched_group_barrier(0x8, kInstMFMAPerIssue, kSchedGroupId)
            for i in l.static_range(kInstVALU):
                amdgcn_sched_group_barrier(0x1, 1, kSchedGroupId)
                amdgcn_sched_group_barrier(0x8, kInstMFMAPerIssue, kSchedGroupId)
        amdgcn_sched_barrier(0)


@g.jit
def ClearMat(reference, FRAGMENTS: l.constexpr):
    if reference.type.is_block():
        zero = l.full(reference.shape, 0, l.float32, reference.type.layout)
    else:
        zero = l.full((), 0, l.float32)
    return ((zero, zero, zero, zero),) * FRAGMENTS


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
    """Accumulate FP8 block-scale MoE into caller-zeroed BF16 ``out``.

    ``act`` holds packed FP8 activations; ``w13`` holds shuffled gate/up weights
    and ``w2`` shuffled down weights. ``sorted_token_ids``, ``sorted_weights``
    and ``sorted_expert_ids`` hold padded route IDs, route weights and per-block
    expert IDs. ``num_valid_ids`` contains the padded route count and token
    count. ``topk`` is the number of routes per token. ``scales_act``,
    ``scales_w13`` and ``scales_w2`` are native-layout float32 block scales.
    ``max_num_m_blocks`` is routing capacity in blocks of 32; ``m``, ``n`` and
    ``k`` are token count, model width and intermediate width. ``stream`` is a
    Torch stream, HIP stream handle, or explicit None for the default stream.
    ``num_persistent_tgs`` is the requested threadgroup budget (0 disables it).
    Return 0 on launch, 1 for invalid shape, or 2 for null required buffers.
    """
    from lib.moe.rocm.fused_moe_blockscale_fp8_grid import (
        FusedMoEBlockScaleFP8 as invoke,
    )

    return invoke(
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
    )


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

    Arguments have the same meanings as ``FusedMoEBlockScaleFP8``, except
    ``w13``/``w2`` contain packed MXFP4, ``scales_w13``/``scales_w2`` contain
    shuffled E8M0 scale bytes, and ``num_experts`` bounds the sorted expert IDs
    (0 disables that bound). Return native status 0, 1 (shape), or 2 (argument).
    """
    from lib.moe.rocm.fused_moe_blockscale_fp8_fp4_grid import (
        FusedMoEBlockScaleFP8MXFP4Weight as invoke,
    )

    return invoke(
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
    )


# Host dispatch follows fused_moe.cc. Imports are deferred to avoid the native
# header dependency cycle between solution IDs, selectors, and kernel policies.
from functools import cache


@cache
def _TwoStageCallMap():
    from lib.moe.rocm.fused_moe_config_selector import (
        ConfigSelector,
        kFusedMoETwoStageMxFp4BiasSolutionId,
        kFusedMoETwoStageMxFp4SiluSolutionId,
    )

    calls = {}

    def RegisterTwoStageShape(base, dim, inter_dim, topks):
        cached = base.WithShape(dim, inter_dim)
        non_temporal = cached.WithWeightLoadPolicy(
            FusedMoEWeightLoadPolicy.kNonTemporal
        )
        for solution in (cached, non_temporal):
            calls[solution.Repr()] = {
                topk: ConfigSelector(solution, topk) for topk in topks
            }

    RegisterTwoStageShape(kFusedMoETwoStageMxFp4BiasSolutionId, 3072, 3072, (4,))
    RegisterTwoStageShape(
        kFusedMoETwoStageMxFp4BiasSolutionId.WithStage1TileShape(
            FusedMoEStage1TileShape.kM64N512
        ),
        3072,
        3072,
        (4,),
    )
    RegisterTwoStageShape(kFusedMoETwoStageMxFp4SiluSolutionId, 7168, 2048, (8, 9))
    RegisterTwoStageShape(kFusedMoETwoStageMxFp4SiluSolutionId, 7168, 3072, (7,))
    return calls


def TwoStageWorkspaceSize(Config, max_num_m_blocks, inter_dim):
    from lib.moe.rocm.fused_moe_2stage_kernel import TwoStageFusedMoEWorkspace

    if inter_dim != Config.kInterDim:
        return 0
    return TwoStageFusedMoEWorkspace(Config).Bytes(max_num_m_blocks, Config.kInterDim)


def FusedMoE2StageWorkspaceSize(max_num_m_blocks, inter_dim, solution_id):
    configs = _TwoStageCallMap().get(solution_id)
    return (
        0
        if configs is None
        else max(
            TwoStageWorkspaceSize(c, max_num_m_blocks, inter_dim)
            for c in configs.values()
        )
    )


def IsGfx950(stream, device):
    import torch

    return (
        torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx950"
    )


def RequiresNativeMxFp4(solution_id):
    return (
        FusedMoESolutionId.FromRepr(solution_id).weight_ordering
        == FusedMoEWeightOrdering.kNativeMxFp4
    )


@cache
def _TwoStageKernels(Config):
    from lib.moe.rocm.fused_moe_2stage_kernel import (
        MxFp4Stage1WorkspaceEpilogue,
        TwoStageFusedMoEStage2,
        TwoStageFusedMoEWorkspace,
    )
    from lib.moe.rocm.fused_moe_blockscale_fp8_kernel import FusedMoEStage1

    return (
        FusedMoEStage1(Config, MxFp4Stage1WorkspaceEpilogue(Config)),
        TwoStageFusedMoEWorkspace(Config),
        TwoStageFusedMoEStage2(Config),
    )


def InvokeTwoStage1(Config, params):
    import torch
    from lib.moe.rocm.fused_moe_2stage_kernel import (
        MxFp4Stage1WorkspaceEpilogue,
        TwoStageFusedMoEStage1Compute,
        TwoStageFusedMoEWorkspace,
    )
    from lib.moe.rocm.fused_moe_blockscale_fp8_kernel import FusedMoEStage1
    from lib.streams import native_stream

    common = params.common
    if any(
        p is None
        for p in (
            common.intermediate,
            common.num_valid_ids,
            params.act,
            params.w13,
            common.sorted_token_ids,
            common.sorted_expert_ids,
            params.scales_act,
            params.scales_w13,
        )
    ):
        return kFusedMoEErrorInvalidArgument
    if (common.topk, common.n, common.k) != (
        Config.kTopK,
        Config.kDim,
        Config.kInterDim,
    ):
        return kFusedMoEErrorInvalidArgument
    required = TwoStageWorkspaceSize(Config, common.max_num_m_blocks, common.k)
    if common.intermediate_bytes < required:
        return kFusedMoEErrorInvalidArgument
    if common.m == 0 or common.max_num_m_blocks == 0:
        return 0
    n_tiles = Config.kInterDim // Config.kStage1GroupN
    persistent = common.num_persistent_tgs > 0
    route_groups = common.max_num_m_blocks
    if persistent:
        workers = max(1, (common.num_persistent_tgs + n_tiles - 1) // n_tiles)
        route_groups = min(workers, route_groups)
    num_experts = common.num_experts if Config.kValidateExpertIds else 0
    kernel, workspace, _ = _TwoStageKernels(Config)
    with torch.cuda.device(params.act.device), torch.cuda.stream(
        native_stream(common.stream, params.act.device)
    ):
        TwoStageFusedMoEStage1Compute[(n_tiles, route_groups)](
            common.intermediate,
            params.act,
            params.w13,
            common.sorted_token_ids,
            common.sorted_expert_ids,
            common.num_valid_ids,
            params.scales_act,
            params.scales_w13,
            common.m,
            num_experts,
            common.max_num_m_blocks,
            params.w13_bias,
            kernel,
            workspace,
            persistent,
            num_warps=Config.kNumWarps,
            enable_fp_fusion=False,
        )
    return 0


def InvokeTwoStage2(Config, params):
    import torch
    from lib.moe.rocm.fused_moe_2stage_kernel import (
        TwoStageFusedMoEStage2,
        TwoStageFusedMoEStage2Compute,
    )
    from lib.streams import native_stream

    common = params.common
    if any(
        p is None
        for p in (
            common.intermediate,
            params.out,
            params.w2,
            common.sorted_token_ids,
            params.sorted_weights,
            common.sorted_expert_ids,
            common.num_valid_ids,
            params.scales_w2,
        )
    ):
        return kFusedMoEErrorInvalidArgument
    if (common.topk, common.n, common.k) != (
        Config.kTopK,
        Config.kDim,
        Config.kInterDim,
    ) or common.n % Config.kGroupN != 0:
        return kFusedMoEErrorInvalidArgument
    required = TwoStageWorkspaceSize(Config, common.max_num_m_blocks, common.k)
    if common.intermediate_bytes < required:
        return kFusedMoEErrorInvalidArgument
    if common.m == 0 or common.max_num_m_blocks == 0:
        return 0
    _, _, kernel = _TwoStageKernels(Config)
    num_experts = common.num_experts if Config.kValidateExpertIds else 0
    with torch.cuda.device(params.out.device), torch.cuda.stream(
        native_stream(common.stream, params.out.device)
    ):
        TwoStageFusedMoEStage2Compute[
            (common.n // Config.kGroupN, kernel.kPersistentWorkers)
        ](
            params.out,
            common.intermediate,
            params.w2,
            common.sorted_token_ids,
            params.sorted_weights,
            common.sorted_expert_ids,
            common.num_valid_ids,
            common.topk,
            params.scales_w2,
            num_experts,
            common.max_num_m_blocks,
            params.w2_bias,
            kernel,
            num_warps=Config.kNumWarps,
            enable_fp_fusion=False,
        )
    return 0


def FusedMoEMatmul2Stage1(params, solution_id):
    configs = _TwoStageCallMap().get(solution_id)
    if configs is None:
        return kFusedMoEErrorInvalidSolution
    if RequiresNativeMxFp4(solution_id) and not IsGfx950(params.common.stream, None):
        return kFusedMoEErrorUnsupported
    config = configs.get(params.common.topk)
    return (
        kFusedMoEErrorInvalidArgument
        if config is None
        else InvokeTwoStage1(config, params)
    )


def FusedMoEMatmul2Stage2(params, solution_id):
    configs = _TwoStageCallMap().get(solution_id)
    if configs is None:
        return kFusedMoEErrorInvalidSolution
    if RequiresNativeMxFp4(solution_id) and not IsGfx950(params.common.stream, None):
        return kFusedMoEErrorUnsupported
    config = configs.get(params.common.topk)
    return (
        kFusedMoEErrorInvalidArgument
        if config is None
        else InvokeTwoStage2(config, params)
    )
