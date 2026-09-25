"""Native compact planner and source-push dispatch with fixed CTA roles."""

from typing import NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    GetConditionShmPtr,
    _native_call,
    _native_load_uint2,
    _native_store_uint2,
    amdgcn_readfirstlane,
    amdgcn_shuffle,
    amdgcn_wave_inclusive_add,
)
from lib.moe.rocm.comm.barrier import (
    compiler_memory_barrier,
    complete_scoped_vmem,
    store_xgpu_epoch_relaxed,
    system_fence_acquire,
    wait_xgpu_epoch_relaxed,
    wait_xgpu_signal_relaxed,
)
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.moe.rocm.memory_ops import MakeBufferResource
from lib.moe.rocm.ops.mega_moe.token_shuffle_common import TokenShuffleCommon
from lib.moe.rocm.profiler import Profiler
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class Shm(NamedTuple):
    expert_count: object
    source_count: object
    generation: object
    payload_plan: object


class Profile(NamedTuple):
    owner_admission: object
    owner_count: object
    owner_plan: object
    producer_admission_wait: object
    producer_plan_wait: object
    producer_payload: object
    producer_rows: object
    producer_row_copy: object
    producer_publish: object
    producer_fragments: object
    producer_publish_blocks: object


@g.jit
def _profile_set(profile, index: l.constexpr, value):
    p = profile[:index] + (value,) + profile[index + 1 :]
    return Profile(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10])


class DirectPushState(NamedTuple):
    num_tokens_: object
    workspace_: object
    shm_: object
    current_epoch_: object
    input_tokens_: object
    input_topk_ids_: object
    input_topk_weights_: object


class DirectPushTokenShuffle(DeviceTemplate):
    def __init__(self, Config, kExternalInputs=False, kProfile=False):
        self._key = (Config.cache_key, kExternalInputs, kProfile)
        self.Workspace, self.Common = MegaMoEWorkspace(Config), TokenShuffleCommon(
            Config
        )
        self.kExternalInputs, self.kProfile = kExternalInputs, kProfile
        self.Profiler = Profiler(kProfile)
        self.kNumSMs, self.kThreads, self.kTopK = (
            Config.kNumSMs,
            Config.kThreads,
            Config.kTopK,
        )
        self.kNumExperts, self.kNumRanks = Config.kNumExperts, Config.kNumRanks
        self.kExpertsPerRank = self.kNumExperts // self.kNumRanks
        self.kExpertsPerLane = (self.kExpertsPerRank + 63) // 64
        self.kMaxTokens, self.kSortedTokenBlock = Config.kMaxTokensPerRank, 32
        self.kInputTokenBytes = Config.kInputTokenBytes
        self.kRowVecs, self.kWorkShards, self.kProducerBlocks = (
            self.kInputTokenBytes // 16,
            8,
            Config.kProducerBlocks,
        )
        self.kPeerCoherent = self.kDeviceStore = BufferResource.kSC1Bit
        self.kSystemStore = BufferResource.kSC0Bit | BufferResource.kSC1Bit
        self.kShmWords = self.kNumExperts * 2 + 4
        assert 0 < self.kTopK <= 64 and self.kNumRanks in (2, 4, 8)
        assert self.kNumExperts % self.kNumRanks == 0 and self.kThreads % 64 == 0
        assert (
            self.kProducerBlocks % self.kNumRanks == 0
            and self.kInputTokenBytes % 16 == 0
        )
        assert self.kNumSMs > self.kProducerBlocks

    @device_method
    def Construct(
        self,
        num_tokens,
        workspace,
        shm,
        input_tokens,
        input_topk_ids,
        input_topk_weights,
    ):
        null = l.full((), 0, l.uint64).to(l.pointer_type(l.uint8))
        tokens = MakeBufferResource(null, 0)
        ids = MakeBufferResource(null, 0)
        weights = MakeBufferResource(null, 0)
        if self.kExternalInputs:
            tokens = MakeBufferResource(
                input_tokens, num_tokens * self.kInputTokenBytes
            )
            ids = MakeBufferResource(input_topk_ids, num_tokens * self.kTopK * 4)
            weights = MakeBufferResource(
                input_topk_weights, num_tokens * self.kTopK * 4
            )
        return DirectPushState(
            num_tokens,
            workspace,
            Shm(
                shm,
                shm + self.kNumExperts,
                shm + 2 * self.kNumExperts,
                shm + 2 * self.kNumExperts + 2,
            ),
            l.full((), 0, l.uint32),
            tokens,
            ids,
            weights,
        )

    @device_method
    def EmptyProfile(self):
        z = l.full((), 0, l.uint64)
        return Profile(z, z, z, z, z, z, z, z, z, z, z)

    @device_method
    def Run(self, state, block, tid, wid, wtid):
        profile = self.EmptyProfile()
        if tid == 0:
            generation = BufferResource.AtomicAddI32(
                state.workspace_.br_,
                self.Workspace.DirectPushEntryCountOffset(
                    state.workspace_.rank_id_, block
                ),
                0,
                1,
                BufferResource.kAtomicScopeAgent,
            )
            l.store(state.shm_.generation, generation)
        l.barrier()
        epoch = l.load(state.shm_.generation) + 1
        state = DirectPushState(
            state.num_tokens_,
            state.workspace_,
            state.shm_,
            epoch,
            state.input_tokens_,
            state.input_topk_ids_,
            state.input_topk_weights_,
        )
        parity, expected = epoch & 1, self.Expected(epoch)
        owner = block == 0
        if owner:
            phase_profiler = self.Profiler.Start()
            self.AdmitLaunch(state, tid, epoch)
            if self.kProfile:
                profile = _profile_set(profile, 0, self.Profiler.End(phase_profiler))
            phase_profiler = self.Profiler.Start()
            self.PopulateSendCounters(state, tid, parity, expected)
            if self.kProfile:
                profile = _profile_set(profile, 1, self.Profiler.End(phase_profiler))
            phase_profiler = self.Profiler.Start()
            self.BuildDestinationPlan(state, tid, wid, wtid, parity, expected)
            if self.kProfile:
                profile = _profile_set(profile, 2, self.Profiler.End(phase_profiler))
        elif block <= self.kProducerBlocks:
            admission_profiler = self.Profiler.Start()
            self.WaitForOwnerAdmission(state, epoch, tid)
            if self.kProfile:
                profile = _profile_set(
                    profile, 3, self.Profiler.End(admission_profiler)
                )
            profile = self.PushPayload(
                state,
                block - 1,
                tid,
                wid,
                wtid,
                parity,
                expected,
                profile,
                self.kProducerBlocks,
            )
        return state, epoch, profile

    @device_method
    def LoadLocalNumTokens(self, state, lane):
        routes = l.full((), 0, l.uint32)
        parity = state.current_epoch_ & 1
        for expert in range(lane, self.kNumExperts, 64):
            routes += BufferResource.LoadU32(
                state.workspace_.br_,
                self.Workspace.SendCounterOffset(expert) + parity * 4,
                0,
                BufferResource.kNone,
            )
        # HIP __reduce_add_sync on a full wave uses the same shuffle reduction.
        for step in l.static_range(6):
            routes += amdgcn_shuffle(routes, lane ^ (1 << step))
        return routes // self.kTopK

    @device_method
    def ResetRoutingCounters(self, state, sm_id, tid):
        self.Common.ResetRoutingCounters(state.workspace_, sm_id, tid)

    @device_method
    def WaitForExpertPayload(self, state, local_expert, epoch, tid):
        if tid == 0:
            pool_block = l.full((), 0, l.uint32)
            for expert in range(local_expert + 1):
                rows = BufferResource.LoadU32(
                    state.workspace_.br_,
                    self.Workspace.RecvSumCounterOffset(expert),
                    0,
                    BufferResource.kSC1Bit,
                )
                if expert == local_expert:
                    self.WaitForPayloadBlocks(state, pool_block, rows)
                pool_block += (
                    rows + self.kSortedTokenBlock - 1
                ) // self.kSortedTokenBlock
        l.barrier()
        system_fence_acquire()
        l.barrier()

    @device_method
    def WaitForLocalPlan(self, state, epoch, tid):
        if tid == 0:
            wait_xgpu_signal_relaxed(
                state.workspace_,
                self.Workspace.DirectPushPlanReadyOffset(
                    state.workspace_.rank_id_, epoch & 1, state.workspace_.rank_id_
                ),
                self.Expected(epoch).to(l.int32),
            )
        l.barrier()

    @device_method
    def WaitForAllPayloads(self, state, epoch, tid):
        if tid == 0:
            pool_block = l.full((), 0, l.uint32)
            for expert in range(self.kExpertsPerRank):
                rows = BufferResource.LoadU32(
                    state.workspace_.br_,
                    self.Workspace.RecvSumCounterOffset(expert),
                    0,
                    BufferResource.kSC1Bit,
                )
                self.WaitForPayloadBlocks(state, pool_block, rows)
                pool_block += (
                    rows + self.kSortedTokenBlock - 1
                ) // self.kSortedTokenBlock
        l.barrier()
        system_fence_acquire()

    @device_method
    def WaitForPayloadBlocks(self, state, pool_block, rows):
        blocks = (rows + self.kSortedTokenBlock - 1) // self.kSortedTokenBlock
        for block in range(blocks):
            block_rows = l.minimum(
                self.kSortedTokenBlock, rows - block * self.kSortedTokenBlock
            )
            ready_mask = l.where(
                block_rows == 32, 0xFFFFFFFF, (1 << block_rows) - 1
            ).to(l.uint32)
            observed = l.full((), 0, l.uint32)
            pending = l.full((), True, l.int1)
            while pending:
                observed = BufferResource.LoadU32(
                    state.workspace_.br_,
                    self.Workspace.L1PayloadArrivalMaskOffset(
                        state.workspace_.rank_id_, pool_block + block
                    ),
                    0,
                    BufferResource.kSC0Bit | BufferResource.kSC1Bit,
                )
                pending = (observed & ready_mask) != ready_mask
                if pending:
                    _native_call("s.sleep.1", "void", (), (), False)

    @device_method
    def Expected(self, epoch):
        return ((epoch + 1) // 2) * self.kNumRanks

    @device_method
    def StoreParityValue(self, state, offset, parity, value, kStoreScope: l.constexpr):
        BufferResource.StoreU32(
            state.workspace_.br_, offset, parity * 4, value, kStoreScope
        )

    @device_method
    def LoadParityValue(self, state, offset, parity):
        return BufferResource.LoadU32(
            state.workspace_.br_, offset, parity * 4, self.kPeerCoherent
        )

    @device_method
    def AdmitLaunch(self, state, tid, epoch):
        if tid < self.kNumRanks:
            peer = (state.workspace_.rank_id_ + tid) % self.kNumRanks
            store_xgpu_epoch_relaxed(
                state.workspace_,
                self.Workspace.DirectPushLaunchReadyOffset(
                    peer, state.workspace_.rank_id_
                ),
                epoch,
            )
            wait_xgpu_epoch_relaxed(
                state.workspace_,
                self.Workspace.DirectPushLaunchReadyOffset(
                    state.workspace_.rank_id_, peer
                ),
                epoch,
            )
        l.barrier()

    @device_method
    def WaitForOwnerAdmission(self, state, epoch, tid):
        if tid == 0:
            wait_xgpu_signal_relaxed(
                state.workspace_,
                self.Workspace.DirectPushEpochGateOffset(state.workspace_.rank_id_),
                epoch.to(l.int32),
            )
        l.barrier()

    @device_method
    def CopyPayloadRow(
        self, state, destination, pool_index, route, vec_lane, vec_stride, header_owner
    ):
        source_token = route // self.kTopK
        source_row = (
            self.Workspace.InputTokensOffset() + source_token * self.kInputTokenBytes
        )
        destination_row = self.Workspace.L1TokenBufferOffset(destination, pool_index)
        for vec in range(vec_lane, self.kRowVecs, vec_stride):
            source_offset = source_token * self.kInputTokenBytes + vec * 16
            if self.kExternalInputs:
                value = BufferResource.Load(
                    state.input_tokens_, source_offset, 0, BufferResource.kNone
                )
            else:
                value = BufferResource.Load(
                    state.workspace_.br_, source_row + vec * 16, 0, BufferResource.kNone
                )
            BufferResource.Store(
                state.workspace_.br_,
                destination_row + vec * 16,
                0,
                value,
                self.kSystemStore,
            )
        if header_owner:
            if self.kExternalInputs:
                weight = BufferResource.LoadU32(
                    state.input_topk_weights_, route * 4, 0, BufferResource.kNone
                )
            else:
                weight = BufferResource.LoadU32(
                    state.workspace_.br_,
                    self.Workspace.InputTokenTopKExpertWeightOffset() + route * 4,
                    0,
                    BufferResource.kNone,
                )
            BufferResource.StoreU32(
                state.workspace_.br_,
                self.Workspace.L1TokenWeightsOffset(destination, pool_index),
                0,
                weight,
                self.kSystemStore,
            )
            BufferResource.StoreU64(
                state.workspace_.br_,
                self.Workspace.TokenMetadataOffset(destination, pool_index),
                0,
                (route, state.workspace_.rank_id_),
                self.kSystemStore,
            )

    @device_method
    def PopulateSendCounters(self, state, tid, parity, expected):
        self.Common.ClearExpertCounts(state.shm_.expert_count, tid)
        route_count = state.num_tokens_ * self.kTopK
        for route in range(tid, route_count, self.kThreads):
            expert = self.LoadInputExpert(state, route)
            l.atomic_add(
                GetConditionShmPtr(
                    state.shm_.expert_count + expert, expert < self.kNumExperts
                ),
                1,
                sem="relaxed",
            )
        l.barrier()
        kIterations: l.constexpr = (
            self.kNumExperts + self.kThreads - 1
        ) // self.kThreads
        for i in l.static_range(kIterations):
            expert = tid + i * self.kThreads
            if expert < self.kNumExperts:
                count = l.load(state.shm_.expert_count + expert)
                self.StoreParityValue(
                    state,
                    self.Workspace.SendCounterOffset(expert),
                    parity,
                    count,
                    self.kDeviceStore,
                )
                self.StoreParityValue(
                    state,
                    self.Workspace.RecvCounterOffset(
                        expert // self.kExpertsPerRank,
                        state.workspace_.rank_id_,
                        expert % self.kExpertsPerRank,
                    ),
                    parity,
                    count,
                    self.kSystemStore,
                )
                l.store(state.shm_.expert_count + expert, 0)
        complete_scoped_vmem()
        l.barrier()
        if tid < self.kNumRanks:
            destination = (state.workspace_.rank_id_ + tid) % self.kNumRanks
            store_xgpu_epoch_relaxed(
                state.workspace_,
                self.Workspace.DirectPushCountDoneOffset(
                    destination, parity, state.workspace_.rank_id_
                ),
                expected,
            )

    @device_method
    def BuildDestinationPlan(self, state, tid, wid, wtid, parity, expected):
        work_head_base = self.Workspace.DirectPushWorkHeadOffset(0, 0)
        work_head_stride = (
            self.Workspace.DirectPushWorkHeadOffset(1, 0) - work_head_base
        )
        work_heads = BufferResource.WithOffset(state.workspace_.br_, work_head_base)
        work_heads = BufferResource.WithRange(
            work_heads, 2 * self.kWorkShards * work_head_stride
        )
        BufferResource.StoreU32(
            work_heads, tid * work_head_stride, 0, 0, self.kDeviceStore
        )
        if wid == 0:
            if wtid < self.kNumRanks:
                wait_xgpu_signal_relaxed(
                    state.workspace_,
                    self.Workspace.DirectPushCountDoneOffset(
                        state.workspace_.rank_id_, parity, wtid
                    ),
                    expected.to(l.int32),
                )
            compiler_memory_barrier()
            pool_rows = l.full((), 0, l.uint32)
            for i in l.static_range(self.kExpertsPerLane):
                expert = i * 64 + wtid
                valid_expert = expert < self.kExpertsPerRank
                total = l.full((), 0, l.uint32)
                if valid_expert:
                    for source in range(self.kNumRanks):
                        count = self.LoadParityValue(
                            state,
                            self.Workspace.RecvCounterOffset(
                                state.workspace_.rank_id_, source, expert
                            ),
                            parity,
                        )
                        l.store(
                            state.shm_.source_count
                            + source * self.kExpertsPerRank
                            + expert,
                            count,
                        )
                        total += count
                padded = l.where(valid_expert, (total + 31) // 32 * 32, 0)
                inclusive = amdgcn_wave_inclusive_add(padded, wtid)
                pool_base = pool_rows + inclusive - padded
                if valid_expert:
                    BufferResource.StoreU64(
                        state.workspace_.br_,
                        self.Workspace.RecvSumCounterOffset(expert),
                        0,
                        (total, self.kNumSMs * self.kNumRanks),
                        self.kDeviceStore,
                    )
                    source_prefix = l.full((), 0, l.uint32)
                    for source in range(self.kNumRanks):
                        self.StoreParityValue(
                            state,
                            self.Workspace.DirectPushPlanBaseOffset(
                                source, state.workspace_.rank_id_, expert
                            ),
                            parity,
                            pool_base + source_prefix,
                            self.kSystemStore,
                        )
                        source_prefix += l.load(
                            state.shm_.source_count
                            + source * self.kExpertsPerRank
                            + expert
                        )
                pool_rows += amdgcn_shuffle(inclusive, 63)
            for block in range(wtid, pool_rows // self.kSortedTokenBlock, 64):
                BufferResource.StoreU32(
                    state.workspace_.br_,
                    self.Workspace.L1PayloadArrivalMaskOffset(
                        state.workspace_.rank_id_, block
                    ),
                    0,
                    0,
                    self.kDeviceStore,
                )
                BufferResource.StoreU32(
                    state.workspace_.br_,
                    self.Workspace.L2ArrivalMaskOffset(block),
                    0,
                    0,
                    self.kDeviceStore,
                )
        else:
            group_tid = (wid - 1) * 64 + wtid
            group_threads = (self.kThreads // 64 - 1) * 64
            route_count = state.num_tokens_ * self.kTopK
            for route in range(group_tid, route_count, group_threads):
                expert = self.LoadInputExpert(state, route)
                if expert < self.kNumExperts:
                    ordinal = l.atomic_add(
                        state.shm_.expert_count + expert, 1, sem="relaxed"
                    )
                    BufferResource.StoreU32(
                        state.workspace_.br_,
                        self.Workspace.RouteIndexOffset(
                            expert // self.kExpertsPerRank,
                            expert % self.kExpertsPerRank,
                            ordinal,
                        ),
                        0,
                        route,
                        self.kDeviceStore,
                    )
        complete_scoped_vmem()
        l.barrier()
        if tid < self.kNumRanks:
            store_xgpu_epoch_relaxed(
                state.workspace_,
                self.Workspace.DirectPushPlanReadyOffset(
                    tid, parity, state.workspace_.rank_id_
                ),
                expected,
            )
        if tid == 0:
            store_xgpu_epoch_relaxed(
                state.workspace_,
                self.Workspace.DirectPushEpochGateOffset(state.workspace_.rank_id_),
                state.current_epoch_,
            )
        l.barrier()

    @device_method
    def PushPayload(
        self,
        state,
        producer_slot,
        tid,
        wid,
        wtid,
        parity,
        expected,
        profile,
        kDispatchBlocks: l.constexpr,
    ):
        destination = producer_slot % self.kNumRanks
        phase_profiler = self.Profiler.Start()
        if tid == 0:
            wait_xgpu_signal_relaxed(
                state.workspace_,
                self.Workspace.DirectPushPlanReadyOffset(
                    state.workspace_.rank_id_, parity, destination
                ),
                expected.to(l.int32),
            )
        l.barrier()
        if self.kProfile:
            profile = _profile_set(profile, 4, self.Profiler.End(phase_profiler))
        phase_profiler = self.Profiler.Start()
        if kDispatchBlocks == 56 and kDispatchBlocks >= self.kNumRanks:
            kProducersPerDestination: l.constexpr = kDispatchBlocks // self.kNumRanks
            destination_producer = producer_slot // self.kNumRanks
            if wid == 0:
                for i in l.static_range(self.kExpertsPerLane):
                    local_expert = i * 64 + wtid
                    if local_expert < self.kExpertsPerRank:
                        expert = destination * self.kExpertsPerRank + local_expert
                        l.store(
                            state.shm_.expert_count + local_expert,
                            self.LoadParityValue(
                                state, self.Workspace.SendCounterOffset(expert), parity
                            ),
                        )
                        l.store(
                            state.shm_.source_count + local_expert,
                            self.LoadParityValue(
                                state,
                                self.Workspace.DirectPushPlanBaseOffset(
                                    state.workspace_.rank_id_, destination, local_expert
                                ),
                                parity,
                            ),
                        )
            l.barrier()
            for local_expert in range(self.kExpertsPerRank):
                plan = (
                    l.load(state.shm_.expert_count + local_expert),
                    l.load(state.shm_.source_count + local_expert),
                )
                workers = l.where(plan[0] >= 64, kProducersPerDestination, 1).to(
                    l.uint32
                )
                primary = local_expert % kProducersPerDestination
                active = l.where(
                    workers == 1,
                    destination_producer == primary,
                    destination_producer < workers,
                )
                if active:
                    worker = l.where(workers == 1, 0, destination_producer)
                    begin, end = (
                        plan[0] * worker // workers,
                        plan[0] * (worker + 1) // workers,
                    )
                    if self.kProfile:
                        profile = _profile_set(
                            profile, 6, profile.producer_rows + end - begin
                        )
                    profile = self.CopyPayloadRows(
                        state,
                        destination,
                        local_expert,
                        plan[1],
                        begin,
                        end,
                        tid,
                        wid,
                        wtid,
                        profile,
                    )
            if self.kProfile:
                profile = _profile_set(profile, 5, self.Profiler.End(phase_profiler))
            return profile

        for task_index in range(producer_slot, self.kNumExperts, kDispatchBlocks):
            fallback_local_expert = task_index // self.kNumRanks
            plan = self.LoadPayloadPlan(
                state, destination, fallback_local_expert, parity, tid
            )
            if self.kProfile:
                profile = _profile_set(profile, 6, profile.producer_rows + plan[0])
            profile = self.CopyPayloadRows(
                state,
                destination,
                fallback_local_expert,
                plan[1],
                0,
                plan[0],
                tid,
                wid,
                wtid,
                profile,
            )
        if self.kProfile:
            profile = _profile_set(profile, 5, self.Profiler.End(phase_profiler))
        return profile

    @device_method
    def LoadPayloadPlan(self, state, destination, local_expert, parity, tid):
        expert = destination * self.kExpertsPerRank + local_expert
        if tid == 0:
            plan0 = self.LoadParityValue(
                state, self.Workspace.SendCounterOffset(expert), parity
            )
            plan1 = self.LoadParityValue(
                state,
                self.Workspace.DirectPushPlanBaseOffset(
                    state.workspace_.rank_id_, destination, local_expert
                ),
                parity,
            )
            _native_store_uint2(state.shm_.payload_plan, (plan0, plan1))
        l.barrier()
        return _native_load_uint2(state.shm_.payload_plan)

    @device_method
    def CopyPayloadRows(
        self,
        state,
        destination,
        local_expert,
        pool_base,
        ordinal_begin,
        ordinal_end,
        tid,
        wid,
        wtid,
        profile,
    ):
        rows = ordinal_end - ordinal_begin
        if self.kProfile:
            profile = _profile_set(profile, 9, profile.producer_fragments + 1)
        phase_profiler = self.Profiler.Start()
        if rows >= self.kThreads // 64 * 2:
            for ordinal in range(ordinal_begin + wid, ordinal_end, self.kThreads // 64):
                route = l.full((), 0, l.uint32)
                if wtid == 0:
                    route = BufferResource.LoadU32(
                        state.workspace_.br_,
                        self.Workspace.RouteIndexOffset(
                            destination, local_expert, ordinal
                        ),
                        0,
                        self.kPeerCoherent,
                    )
                route = amdgcn_readfirstlane(route)
                self.CopyPayloadRow(
                    state, destination, pool_base + ordinal, route, wtid, 64, wtid == 0
                )
        else:
            for ordinal in range(ordinal_begin, ordinal_end):
                route = BufferResource.LoadU32(
                    state.workspace_.br_,
                    self.Workspace.RouteIndexOffset(destination, local_expert, ordinal),
                    0,
                    self.kPeerCoherent,
                )
                self.CopyPayloadRow(
                    state,
                    destination,
                    pool_base + ordinal,
                    route,
                    tid,
                    self.kThreads,
                    tid == 0,
                )
        if self.kProfile:
            profile = _profile_set(
                profile,
                7,
                profile.producer_row_copy + self.Profiler.End(phase_profiler),
            )
        phase_profiler = self.Profiler.Start()
        complete_scoped_vmem()
        l.barrier()
        if tid == 0:
            first, end = pool_base + ordinal_begin, pool_base + ordinal_end
            pool_block = first // self.kSortedTokenBlock
            while pool_block * self.kSortedTokenBlock < end:
                if self.kProfile:
                    profile = _profile_set(
                        profile, 10, profile.producer_publish_blocks + 1
                    )
                block_first = pool_block * self.kSortedTokenBlock
                lo = l.where(first > block_first, first - block_first, 0)
                block_end = block_first + self.kSortedTokenBlock
                hi = l.where(end < block_end, end - block_first, self.kSortedTokenBlock)
                high_mask = l.where(hi == 32, 0xFFFFFFFF, (1 << hi) - 1).to(l.uint32)
                low_mask = l.where(lo == 0, 0, (1 << lo) - 1).to(l.uint32)
                BufferResource.AtomicOrU32(
                    state.workspace_.br_,
                    self.Workspace.L1PayloadArrivalMaskOffset(destination, pool_block),
                    0,
                    high_mask & ~low_mask,
                    BufferResource.kAtomicScopeSystem,
                )
                pool_block += 1
        l.barrier()
        if self.kProfile:
            profile = _profile_set(
                profile, 8, profile.producer_publish + self.Profiler.End(phase_profiler)
            )
        return profile

    @device_method
    def LoadInputExpert(self, state, route):
        if self.kExternalInputs:
            return BufferResource.LoadU32(
                state.input_topk_ids_, route * 4, 0, BufferResource.kNone
            )
        else:
            return BufferResource.LoadU32(
                state.workspace_.br_,
                self.Workspace.InputTokenTopKExpertIDOffset() + route * 4,
                0,
                BufferResource.kNone,
            )
