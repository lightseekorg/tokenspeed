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
from typing import ClassVar

import triton.experimental.gluon as g
from lib.gemm.rocm.intrinsics import (
    HAS_AMD_SCHED_BARRIER,
    HAS_AMD_SCHED_GROUP_BARRIER,
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
    kClampedSiluDot = 3


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


kFusedMoEErrorInvalidSolution = 1
kFusedMoEErrorUnsupported = 3

# Compatibility names for the previous public wrappers.
kFusedMoEErrorInvalidShape = 1
kFusedMoEErrorInvalidArgument = 2
kFusedMoEErrorUnsupportedArch = 3


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
    zero = l.full(reference.shape, 0, l.float32, reference.type.layout)
    return ((zero, zero, zero, zero),) * FRAGMENTS


# Host dispatch follows fused_moe.cc. Imports are deferred to avoid the native
# header dependency cycle between solution IDs, selectors, and kernel policies.


def IsGfx950(stream, device):
    import torch

    return (
        torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx950"
    )
