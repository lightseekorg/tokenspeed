"""Native rank-local work assignment, including M64 stage-one tickets."""

from enum import IntEnum
from typing import NamedTuple

from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    amdgcn_ballot,
    amdgcn_ctz64,
    amdgcn_shuffle,
    amdgcn_wave_inclusive_add,
)
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class MegaMoEBlockPhase(IntEnum):
    kLinear1 = 0
    kLinear2 = 1


class Work(NamedTuple):
    phase: object
    expert_idx: object
    pool_block: object
    pool_row: object
    work_m: object
    tile: object


class SchedulerState(NamedTuple):
    workspace_: object
    expert_metadata_: object


class MegaMoEScheduler(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.Workspace = MegaMoEWorkspace(Config)
        self.kSortedTokenBlock = Config.kSortedTokenBlock
        self.kNumSMs, self.kNumRanks = Config.kNumSMs, Config.kNumRanks
        self.kNumExpertsPerRank = Config.kNumExperts // Config.kNumRanks
        self.kNumExpertsPerLane = (self.kNumExpertsPerRank + 63) // 64
        assert self.kNumExpertsPerLane <= 2

    @device_method
    def Construct(self, workspace):
        return SchedulerState(workspace, ())

    @device_method
    def FetchRecvSumPerExpert(self, state, wtid):
        tokens_per_expert = ()
        for i in l.static_range(self.kNumExpertsPerLane):
            value = (l.full((), 0, l.uint32), l.full((), 0, l.uint32))
            expert = i * 64 + wtid
            if expert < self.kNumExpertsPerRank:
                pending = l.full((), True, l.int1)
                while pending:
                    value = BufferResource.LoadU64(
                        state.workspace_.br_,
                        expert * 8,
                        self.Workspace.RecvSumCounterOffset(0),
                        BufferResource.kSC1Bit,
                    )
                    pending = value[1] != self.kNumSMs * self.kNumRanks
            tokens_per_expert += (value[0],)
        return SchedulerState(state.workspace_, tokens_per_expert)

    @device_method
    def GetWork(self, state, wtid, work_id):
        next = l.full((), 0, l.uint32)
        for expert in l.static_range(self.kNumExpertsPerRank):
            tokens = amdgcn_shuffle(state.expert_metadata_[expert // 64], expert % 64)
            prev = next
            next += (tokens + self.kSortedTokenBlock - 1) // self.kSortedTokenBlock
            if prev <= work_id and work_id < next:
                return (
                    True,
                    l.full((), expert, l.uint32),
                    l.minimum(
                        self.kSortedTokenBlock,
                        tokens - (work_id - prev) * self.kSortedTokenBlock,
                    ),
                )
        return False, l.full((), 0, l.uint32), l.full((), 0, l.uint32)


class MegaMoETwoStageScheduler(MegaMoEScheduler):
    def __init__(self, Config):
        super().__init__(Config)
        self.kLinear1Tiles = Config.kInterDim // Config.kStage1GroupN
        self.kLinear2Tiles = Config.kComputeHiddenSize // Config.kGroupN
        self.kStage1M = Config.kGroupM
        self.kTokenCountBits, self.kTokenCountMask = 17, (1 << 17) - 1
        self.kExpertLaneMask = (1 << min(64, self.kNumExpertsPerRank)) - 1

    @device_method
    def FetchRecvSumPerExpert(self, state, wtid):
        preceding_blocks = l.full((), 0, l.uint32)
        expert_metadata = ()
        for i in l.static_range(self.kNumExpertsPerLane):
            value = (l.full((), 0, l.uint32), l.full((), 0, l.uint32))
            expert = i * 64 + wtid
            if expert < self.kNumExpertsPerRank:
                pending = l.full((), True, l.int1)
                while pending:
                    value = BufferResource.LoadU64(
                        state.workspace_.br_,
                        expert * 8,
                        self.Workspace.RecvSumCounterOffset(0),
                        BufferResource.kSC1Bit,
                    )
                    pending = value[1] != self.kNumSMs * self.kNumRanks
            blocks = (value[0] + self.kSortedTokenBlock - 1) // self.kSortedTokenBlock
            inclusive = amdgcn_wave_inclusive_add(blocks, wtid)
            block_base = preceding_blocks + inclusive - blocks
            preceding_blocks += amdgcn_shuffle(inclusive, 63)
            expert_metadata += (value[0] | (block_base << self.kTokenCountBits),)
        return SchedulerState(state.workspace_, expert_metadata)

    @device_method
    def EmptyWork(self):
        zero = l.full((), 0, l.uint32)
        return Work(zero, zero, zero, zero, zero, zero)

    @device_method
    def GetWork(self, state, wtid, logical_id):
        last_metadata = amdgcn_shuffle(
            state.expert_metadata_[(self.kNumExpertsPerRank - 1) // 64],
            (self.kNumExpertsPerRank - 1) % 64,
        )
        last_tokens = last_metadata & self.kTokenCountMask
        total_blocks = (last_metadata >> self.kTokenCountBits) + (
            last_tokens + self.kSortedTokenBlock - 1
        ) // self.kSortedTokenBlock
        linear1_work = total_blocks * self.kLinear1Tiles
        if logical_id < linear1_work:
            phase = l.full((), 0, l.uint32)
            phase_id = logical_id
            tiles_per_block = l.full((), self.kLinear1Tiles, l.uint32)
            target_block = phase_id // tiles_per_block
        else:
            phase = l.full((), 1, l.uint32)
            phase_id = logical_id - linear1_work
            tiles_per_block = l.full((), self.kLinear2Tiles, l.uint32)
            target_block = phase_id // tiles_per_block
            if phase_id >= total_blocks * tiles_per_block:
                return False, self.EmptyWork()
        for i in l.static_range(self.kNumExpertsPerLane):
            lane_metadata = state.expert_metadata_[i]
            lane_tokens = lane_metadata & self.kTokenCountMask
            lane_base = lane_metadata >> self.kTokenCountBits
            lane_blocks = (
                lane_tokens + self.kSortedTokenBlock - 1
            ) // self.kSortedTokenBlock
            matches = (
                amdgcn_ballot(
                    (i * 64 + wtid < self.kNumExpertsPerRank)
                    & (lane_base <= target_block)
                    & (target_block < lane_base + lane_blocks)
                )
                & self.kExpertLaneMask
            )
            if matches != 0:
                expert = amdgcn_ctz64(matches)
                metadata = amdgcn_shuffle(lane_metadata, expert)
                tokens = metadata & self.kTokenCountMask
                block_base = metadata >> self.kTokenCountBits
                block_in_expert = target_block - block_base
                work = Work(
                    phase,
                    i * 64 + expert,
                    target_block,
                    target_block * self.kSortedTokenBlock,
                    l.minimum(
                        self.kSortedTokenBlock,
                        tokens - block_in_expert * self.kSortedTokenBlock,
                    ),
                    phase_id - target_block * tiles_per_block,
                )
                return True, work
        return False, self.EmptyWork()

    @device_method
    def GetStage1Work(self, state, wtid, logical_id):
        l.static_assert(self.kStage1M == 32 or self.kStage1M == 64)
        target_stage1_block = logical_id // self.kLinear1Tiles
        tile = logical_id % self.kLinear1Tiles
        if self.kStage1M == self.kSortedTokenBlock:
            for i in l.static_range(self.kNumExpertsPerLane):
                lane_metadata = state.expert_metadata_[i]
                lane_tokens = lane_metadata & self.kTokenCountMask
                lane_base = lane_metadata >> self.kTokenCountBits
                lane_blocks = (
                    lane_tokens + self.kSortedTokenBlock - 1
                ) // self.kSortedTokenBlock
                matches = (
                    amdgcn_ballot(
                        (i * 64 + wtid < self.kNumExpertsPerRank)
                        & (lane_base <= target_stage1_block)
                        & (target_stage1_block < lane_base + lane_blocks)
                    )
                    & self.kExpertLaneMask
                )
                if matches != 0:
                    expert = amdgcn_ctz64(matches)
                    metadata = amdgcn_shuffle(lane_metadata, expert)
                    tokens = metadata & self.kTokenCountMask
                    block_base = metadata >> self.kTokenCountBits
                    block_in_expert = target_stage1_block - block_base
                    return True, Work(
                        l.full((), 0, l.uint32),
                        i * 64 + expert,
                        target_stage1_block,
                        target_stage1_block * self.kSortedTokenBlock,
                        l.minimum(
                            self.kStage1M, tokens - block_in_expert * self.kStage1M
                        ),
                        tile,
                    )
            return False, self.EmptyWork()
        stage1_block_base = l.full((), 0, l.uint32)
        physical_block_base = l.full((), 0, l.uint32)
        expert = l.full((), 0, l.uint32)
        found = l.full((), False, l.int1)
        work = self.EmptyWork()
        while expert < self.kNumExpertsPerRank and not found:
            lane_metadata = state.expert_metadata_[0]
            if self.kNumExpertsPerLane == 2:
                lane_metadata = l.where(
                    expert < 64, state.expert_metadata_[0], state.expert_metadata_[1]
                )
            metadata = amdgcn_shuffle(lane_metadata, expert % 64)
            tokens = metadata & self.kTokenCountMask
            stage1_blocks = (tokens + self.kStage1M - 1) // self.kStage1M
            physical_blocks = (
                tokens + self.kSortedTokenBlock - 1
            ) // self.kSortedTokenBlock
            if (
                stage1_block_base <= target_stage1_block
                and target_stage1_block < stage1_block_base + stage1_blocks
            ):
                block_in_expert = target_stage1_block - stage1_block_base
                pool_block = physical_block_base + block_in_expert * (
                    self.kStage1M // self.kSortedTokenBlock
                )
                found = l.full((), True, l.int1)
                work = Work(
                    l.full((), 0, l.uint32),
                    expert.to(l.uint32),
                    pool_block,
                    pool_block * self.kSortedTokenBlock,
                    l.minimum(self.kStage1M, tokens - block_in_expert * self.kStage1M),
                    tile,
                )
            stage1_block_base += stage1_blocks
            physical_block_base += physical_blocks
            expert += 1
        return found, work

    @device_method
    def GetStage2Work(self, state, wtid, stage2_id):
        last_metadata = amdgcn_shuffle(
            state.expert_metadata_[(self.kNumExpertsPerRank - 1) // 64],
            (self.kNumExpertsPerRank - 1) % 64,
        )
        last_tokens = last_metadata & self.kTokenCountMask
        total_blocks = (last_metadata >> self.kTokenCountBits) + (
            last_tokens + self.kSortedTokenBlock - 1
        ) // self.kSortedTokenBlock
        return self.GetWork(state, wtid, total_blocks * self.kLinear1Tiles + stage2_id)
