"""Native MegaMoE symmetric-workspace layout and offsets."""

from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import lib.tal.host_device as _host_device_module
import triton.experimental.gluon as g
from lib.moe.rocm.memory_ops import MakeBufferResource
from lib.moe.rocm.ops.mxfp4_activation import MxFp4ActivationLayout
from lib.tal.device import DeviceTemplate, device_method
from lib.tal.host_device import host_device
from triton.experimental.gluon import language as l


class WorkspaceState(NamedTuple):
    br_: object
    rank_id_: object


@g.constexpr_function
def AlignUp(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


class MegaMoEWorkspace(DeviceTemplate):
    __triton_builtin__ = True

    def __init__(self, Layout):
        self.Layout = Layout
        self.kXGpuBarrierCounterOffset = 0
        self.kCacheLineBytes = 128
        self.kPageBytes = 4096
        self.kLargePageBytes = 2 * 1024 * 1024
        self.kXGpuEpochCounterOffset = self.kCacheLineBytes
        self.kXGpuEpochSignalBaseOffset = 2 * self.kCacheLineBytes
        self.kXGpuBarrierRecordBytes = (
            4 * 4096
            if Layout.kNumExperts > 512
            else (2 * 4096 if Layout.kNumExperts > 128 else 4096)
        )
        self.kMaxGridSyncSlots = getattr(Layout, "kGridSyncSlots", 2)
        self.kNumRanks = Layout.kNumRanks
        self.kNumExperts = Layout.kNumExperts
        self.kNumExpertsPerRank = self.kNumExperts // self.kNumRanks
        self.kMaxTokensPerRank = Layout.kMaxTokensPerRank
        self.kTopK = Layout.kTopK
        self.kDirectWorkShards = 8
        self.kDirectWorkHeadSets = 2
        self.kDirectWorkShardStride = 64
        self.kDirectWorkHeadBytes = (
            self.kDirectWorkHeadSets
            * self.kDirectWorkShards
            * self.kDirectWorkShardStride
        )
        self.kRankSymBufferBase = self.kNumRanks * self.kXGpuBarrierRecordBytes
        self.kSortedTokenBlock = 32
        self.kMaxExpertsPerToken = min(self.kTopK, self.kNumExpertsPerRank)
        self.kMaxPoolTokens = AlignUp(
            self.kNumRanks * self.kMaxTokensPerRank * self.kMaxExpertsPerToken
            + self.kNumExpertsPerRank * (self.kSortedTokenBlock - 1),
            self.kSortedTokenBlock,
        )
        self.kMaxPoolBlocks = self.kMaxPoolTokens // self.kSortedTokenBlock
        self.kL2ScaleRows = MxFp4ActivationLayout.PaddedScaleRows(self.kMaxPoolTokens)
        self.kL2ScaleCols = MxFp4ActivationLayout.ScaleCols(Layout.kInterDim)
        self.kL2ArrivalMaskBytes = self.kMaxPoolBlocks * 4
        self.kL2TokenBufferBytes = self.kMaxPoolTokens * Layout.kInterDim // 2
        self.kL2ScaleBufferBytes = self.kL2ScaleRows * self.kL2ScaleCols
        self.kDirectControlOffset = 3 * self.kCacheLineBytes
        self.kDirectEntryCountBytes = Layout.kNumSMs * 4
        self.kDirectPlanBaseBytes = self.kNumExperts * 8
        self.kDirectCountDoneBytes = 2 * self.kNumRanks * 4
        self.kDirectPlanReadyBytes = 2 * self.kNumRanks * 4
        self.kDirectPayloadReadyBytes = 2 * self.kNumExpertsPerRank * 4
        self.kDirectEpochGateBytes = 4
        self.kDirectLaunchReadyBytes = self.kNumRanks * 4
        self.kDirectControlBytes = (
            self.kDirectEntryCountBytes
            + self.kDirectPlanBaseBytes
            + self.kDirectCountDoneBytes
            + self.kDirectPlanReadyBytes
            + self.kDirectPayloadReadyBytes
            + self.kDirectEpochGateBytes
            + self.kDirectLaunchReadyBytes
        )
        self.kTokenMetadataBytes = self.kMaxPoolTokens * 8
        self.kRouteOutputReadyBytes = 2 * self.kMaxTokensPerRank * 4
        self.kL1PayloadArrivalMaskBytes = self.kMaxPoolBlocks * 4
        self.kL1TokenBufferBytes = self.kMaxPoolTokens * Layout.kInputTokenBytes
        self.kL1TokenWeightBytes = self.kMaxPoolTokens * 4
        self.kRankSlotRawBytes = (
            self.kNumRanks * self.kNumExpertsPerRank * 8
            + self.kMaxTokensPerRank * Layout.kRouteOutputBufferBytes
            + self.kRouteOutputReadyBytes
            + self.kL1PayloadArrivalMaskBytes
            + self.kTokenMetadataBytes
            + self.kL1TokenBufferBytes
            + self.kL1TokenWeightBytes
        )
        self.kSlotStride = AlignUp(self.kRankSlotRawBytes, self.kRankSymBufferBase)
        self.kLocalOffsetBase = AlignUp(
            self.kRankSymBufferBase + self.kNumRanks * self.kSlotStride,
            self.kLargePageBytes,
        )
        self.kLocalDataBytes64 = (
            self.kCacheLineBytes
            + self.kDirectWorkHeadBytes
            + self.kMaxTokensPerRank * self.kTopK * 4
            + self.kNumExperts * 8
            + self.kNumExpertsPerRank * 8
            + self.kNumExperts * self.kMaxTokensPerRank * 4
            + self.kMaxTokensPerRank * self.kTopK * 4
            + self.kMaxTokensPerRank * Layout.kInputTokenBytes
            + self.kL2ArrivalMaskBytes
            + self.kL2TokenBufferBytes
            + self.kL2ScaleBufferBytes
        )
        self.kLocalBytes64 = self.kLocalDataBytes64
        self.kLocalBytes = self.kLocalBytes64
        self.kWorkspaceBytes64 = self.kLocalOffsetBase + self.kLocalBytes64
        self.kWorkspaceBytes = self.kWorkspaceBytes64
        assert self.kMaxGridSyncSlots * 4 + 4 < self.kCacheLineBytes
        assert (
            self.kDirectControlOffset + self.kDirectControlBytes
            <= self.kXGpuBarrierRecordBytes
        )
        assert self.kRankSlotRawBytes <= 0xFFFFFFFF
        assert self.kNumExperts % self.kNumRanks == 0 and self.kNumRanks < 256
        assert self.kLocalBytes64 <= (1 << 32)
        assert self.kWorkspaceBytes64 < (1 << 32)

    @property
    def cache_key(self):
        return (
            sha256(
                Path(__file__).read_bytes()
                + Path(_host_device_module.__file__).read_bytes()
            ).hexdigest()
            + "MegaMoEWorkspace("
            + repr(sorted((k, v) for k, v in vars(self).items() if k != "Layout"))
            + ")"
        )

    @device_method
    def Initialize(self, buf_base, rank_id):
        return WorkspaceState(
            MakeBufferResource(buf_base, self.kWorkspaceBytes),
            (l.full((), 0, l.uint32) + rank_id).to(l.uint32),
        )

    @device_method
    def Rank(self, state):
        return state.rank_id_

    @host_device
    def WorkspaceBytes(self):
        return self.kWorkspaceBytes

    @host_device
    def XGpuBarrierRecordBytes(self):
        return self.kXGpuBarrierRecordBytes

    @host_device
    def RankSymBufferBase(self):
        return self.kRankSymBufferBase

    @host_device
    def RankSymBufferSlotBytes(self):
        return self.kSlotStride

    @host_device
    def LocalOffsetBase(self):
        return self.kLocalOffsetBase

    @host_device
    def XGpuBarrierCounterOffset(self, rank):
        return self.kXGpuBarrierCounterOffset + rank * self.kXGpuBarrierRecordBytes

    @host_device
    def XGpuBarrierSignalOffset(self, rank, phase):
        return self.XGpuBarrierCounterOffset(rank) + 4 + phase * 4

    @host_device
    def XGpuEpochCounterOffset(self, rank):
        return rank * self.kXGpuBarrierRecordBytes + self.kXGpuEpochCounterOffset

    @host_device
    def XGpuEpochSignalOffset(self, rank, source_rank):
        return (
            rank * self.kXGpuBarrierRecordBytes
            + self.kXGpuEpochSignalBaseOffset
            + source_rank * 4
        )

    @host_device
    def RecvCounterOffset(self, rank, src_rank, local_expert_idx):
        return (
            self.RankOffsetBase(rank)
            + (src_rank * self.kNumExpertsPerRank + local_expert_idx) * 8
        )

    @host_device
    def RouteOutputBufferOffset(self, rank):
        return (
            self.RecvCounterOffset(
                rank, self.kNumRanks - 1, self.kNumExpertsPerRank - 1
            )
            + 8
        )

    @host_device
    def RouteOutputReadyOffset(self, rank, parity, token):
        return (
            self.RouteOutputBufferOffset(rank)
            + self.kMaxTokensPerRank * self.Layout.kRouteOutputBufferBytes
            + (parity * self.kMaxTokensPerRank + token) * 4
        )

    @host_device
    def L1PayloadArrivalMaskOffset(self, rank, pool_block_index):
        return (
            self.RouteOutputReadyOffset(rank, 1, self.kMaxTokensPerRank - 1)
            + 4
            + pool_block_index * 4
        )

    @host_device
    def TokenMetadataOffset(self, rank, pool_token_index):
        return (
            self.L1PayloadArrivalMaskOffset(rank, self.kMaxPoolBlocks - 1)
            + 4
            + pool_token_index * 8
        )

    @host_device
    def L1TokenBufferOffset(self, rank, pool_token_index):
        return (
            self.TokenMetadataOffset(rank, self.kMaxPoolTokens - 1)
            + 8
            + pool_token_index * self.Layout.kInputTokenBytes
        )

    @host_device
    def L1TokenWeightsOffset(self, rank, pool_token_index):
        return (
            self.L1TokenBufferOffset(rank, self.kMaxPoolTokens - 1)
            + self.Layout.kInputTokenBytes
            + pool_token_index * 4
        )

    @host_device
    def DirectPushEntryCountOffset(self, rank, block):
        return (
            self.XGpuBarrierCounterOffset(rank) + self.kDirectControlOffset + block * 4
        )

    @host_device
    def DirectPushPlanBaseOffset(self, rank, source_rank, local_expert_idx, parity=0):
        return (
            self.DirectPushEntryCountOffset(rank, self.Layout.kNumSMs - 1)
            + 4
            + (source_rank * self.kNumExpertsPerRank + local_expert_idx) * 8
            + parity * 4
        )

    @host_device
    def DirectPushCountDoneOffset(self, rank, parity, source_rank):
        return (
            self.DirectPushPlanBaseOffset(
                rank, self.kNumRanks - 1, self.kNumExpertsPerRank - 1
            )
            + 8
            + (parity * self.kNumRanks + source_rank) * 4
        )

    @host_device
    def DirectPushPlanReadyOffset(self, rank, parity, destination_rank):
        return (
            self.DirectPushCountDoneOffset(rank, 1, self.kNumRanks - 1)
            + 4
            + (parity * self.kNumRanks + destination_rank) * 4
        )

    @host_device
    def DirectPushPayloadReadyOffset(self, rank, parity, local_expert_idx):
        return (
            self.DirectPushPlanReadyOffset(rank, 1, self.kNumRanks - 1)
            + 4
            + (parity * self.kNumExpertsPerRank + local_expert_idx) * 4
        )

    @host_device
    def DirectPushEpochGateOffset(self, rank):
        return (
            self.DirectPushPayloadReadyOffset(rank, 1, self.kNumExpertsPerRank - 1) + 4
        )

    @host_device
    def DirectPushLaunchReadyOffset(self, rank, source_rank):
        return self.DirectPushEpochGateOffset(rank) + 4 + source_rank * 4

    @host_device
    def GridSyncBarrierOffset(self):
        return self.kLocalOffsetBase

    @host_device
    def DirectPushWorkHeadOffset(self, shard, head_set=0):
        return (
            self.GridSyncBarrierOffset()
            + self.kCacheLineBytes
            + (head_set * self.kDirectWorkShards + shard) * self.kDirectWorkShardStride
        )

    @host_device
    def InputTokenTopKExpertIDOffset(self):
        return (
            self.GridSyncBarrierOffset()
            + self.kCacheLineBytes
            + self.kDirectWorkHeadBytes
        )

    @host_device
    def SendCounterOffset(self, expert_idx):
        return (
            self.InputTokenTopKExpertIDOffset()
            + self.kMaxTokensPerRank * self.kTopK * 4
            + expert_idx * 8
        )

    @host_device
    def RecvSumCounterOffset(self, local_expert_idx):
        return self.SendCounterOffset(self.kNumExperts - 1) + 8 + local_expert_idx * 8

    @host_device
    def RouteIndexOffset(self, destination, local_expert_idx, slot):
        return (
            self.RecvSumCounterOffset(self.kNumExpertsPerRank - 1)
            + 8
            + (
                (destination * self.kNumExpertsPerRank + local_expert_idx)
                * self.kMaxTokensPerRank
                + slot
            )
            * 4
        )

    @host_device
    def InputTokenTopKExpertWeightOffset(self):
        return (
            self.RouteIndexOffset(
                self.kNumRanks - 1,
                self.kNumExpertsPerRank - 1,
                self.kMaxTokensPerRank - 1,
            )
            + 4
        )

    @host_device
    def InputTokensOffset(self):
        return (
            self.InputTokenTopKExpertWeightOffset()
            + self.kMaxTokensPerRank * self.kTopK * 4
        )

    @host_device
    def L2ArrivalMaskOffset(self, pool_block_index):
        return (
            self.InputTokensOffset()
            + self.kMaxTokensPerRank * self.Layout.kInputTokenBytes
            + pool_block_index * 4
        )

    @host_device
    def L2TokenBufferOffset(self, pool_token_index):
        return (
            self.L2ArrivalMaskOffset(self.kMaxPoolBlocks - 1)
            + 4
            + pool_token_index * self.Layout.kInterDim // 2
        )

    @host_device
    def L2ScaleBufferOffset(self):
        return (
            self.L2TokenBufferOffset(self.kMaxPoolTokens - 1)
            + self.Layout.kInterDim // 2
        )

    @host_device
    def RankOffsetBase(self, rank):
        return self.kRankSymBufferBase + rank * self.kSlotStride
