"""Native compact planner and source-push dispatch with fixed CTA roles."""

from typing import NamedTuple

import triton.language as tl
from lib.gemm.rocm.intrinsics import (
    BufferResource,
    _native_call,
    amdgcn_shuffle,
    amdgcn_wave_inclusive_add,
)
from lib.moe.rocm.comm.barrier import (
    compiler_memory_barrier,
    complete_scoped_vmem,
    system_fence_acquire,
    wait_tensor_signal,
    wait_xgpu_signal_relaxed,
)
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.moe.rocm.memory_ops import MakeBufferResource
from lib.moe.rocm.ops.mega_moe.token_shuffle_common import TokenShuffleCommon
from lib.tal.device import DeviceTemplate, device_method
from lib.tal.tensor_ops import (
    atomic_add,
    atomic_or,
    first,
    load_words,
    prevent_loop_unroll,
    store_words,
    wave_id,
)
from triton.experimental.gluon import language as l


class Shm(NamedTuple):
    expert_count: object
    source_count: object
    generation: object
    payload_plan: object


class DirectPushState(NamedTuple):
    num_tokens_: object
    workspace_: object
    shm_: object
    current_epoch_: object
    input_tokens_: object
    input_topk_ids_: object
    input_topk_weights_: object


class DirectPushTokenShuffle(DeviceTemplate):
    def __init__(self, Config, kExternalInputs):
        self._key = (Config.cache_key, kExternalInputs)
        self.Workspace, self.Common = MegaMoEWorkspace(Config), TokenShuffleCommon(
            Config
        )
        self.kExternalInputs = kExternalInputs
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
    def Run(self, state, block, tid, wid, wtid):
        generation = atomic_add(
            state.workspace_.br_,
            self.Workspace.DirectPushEntryCountOffset(state.workspace_.rank_id_, block),
            0,
            1,
            BufferResource.kAtomicScopeAgent,
            tid == 0,
        )
        l.store(
            state.shm_.generation + l.full(tid.shape, 0, l.uint32, tid.type.layout),
            generation,
            mask=tid == 0,
        )
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
            self.AdmitLaunch(state, tid, epoch)
            self.PopulateSendCounters(state, tid, parity, expected)
            self.BuildDestinationPlan(state, tid, wid, wtid, parity, expected)
        elif block <= self.kProducerBlocks:
            self.WaitForOwnerAdmission(state, epoch, tid)
            self.PushPayload(
                state,
                block - 1,
                tid,
                wid,
                wtid,
                parity,
                expected,
                self.kProducerBlocks,
            )
        return state, epoch

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
        peer = (state.workspace_.rank_id_ + tid) % self.kNumRanks
        store_words(
            state.workspace_.br_,
            self.Workspace.DirectPushLaunchReadyOffset(peer, state.workspace_.rank_id_),
            0,
            epoch,
            1,
            self.kSystemStore,
            tid < self.kNumRanks,
        )
        if wave_id() == 0:
            lane = l.arange(
                0, 64, layout=l.BlockedLayout([1], [64], [l.num_warps()], [0])
            ).to(l.uint32)
            poll_peer = (state.workspace_.rank_id_ + lane) % self.kNumRanks
            wait_tensor_signal(
                state.workspace_,
                self.Workspace.DirectPushLaunchReadyOffset(
                    state.workspace_.rank_id_, poll_peer
                ),
                epoch,
                lane < self.kNumRanks,
                True,
            )
        l.barrier()

    @device_method
    def WaitForOwnerAdmission(self, state, epoch, tid):
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
        for vec_base in range(0, self.kRowVecs, vec_stride):
            vec = vec_base + vec_lane
            valid = vec < self.kRowVecs
            if self.kExternalInputs:
                value = load_words(
                    state.input_tokens_,
                    source_token * self.kInputTokenBytes + vec * 16,
                    0,
                    4,
                    BufferResource.kNone,
                    valid,
                )
            else:
                value = load_words(
                    state.workspace_.br_,
                    source_row + vec * 16,
                    0,
                    4,
                    BufferResource.kNone,
                    valid,
                )
            store_words(
                state.workspace_.br_,
                destination_row + vec * 16,
                0,
                value,
                4,
                self.kSystemStore,
                valid,
            )
        if self.kExternalInputs:
            weight = load_words(
                state.input_topk_weights_,
                route * 4,
                0,
                1,
                BufferResource.kNone,
                header_owner,
            )
        else:
            weight = load_words(
                state.workspace_.br_,
                self.Workspace.InputTokenTopKExpertWeightOffset() + route * 4,
                0,
                1,
                BufferResource.kNone,
                header_owner,
            )
        store_words(
            state.workspace_.br_,
            self.Workspace.L1TokenWeightsOffset(destination, pool_index),
            0,
            weight,
            1,
            self.kSystemStore,
            header_owner,
        )
        store_words(
            state.workspace_.br_,
            self.Workspace.TokenMetadataOffset(destination, pool_index),
            0,
            (route, state.workspace_.rank_id_),
            2,
            self.kSystemStore,
            header_owner,
        )

    @device_method
    def PopulateSendCounters(self, state, tid, parity, expected):
        self.Common.ClearExpertCounts(state.shm_.expert_count, tid)
        route_count = state.num_tokens_ * self.kTopK
        for route_base in range(0, route_count, self.kThreads):
            route = route_base + tid
            expert = self.LoadInputExpert(state, route)
            _native_call(
                "when:memory.atomic.add.i32",
                "i32",
                ("p3", "i32", "i1"),
                (
                    state.shm_.expert_count + expert,
                    1,
                    (route < route_count) & (expert < self.kNumExperts),
                ),
                False,
            )
        l.barrier()
        for i in l.static_range(
            (self.kNumExperts + self.kThreads - 1) // self.kThreads
        ):
            expert = tid + i * self.kThreads
            valid = expert < self.kNumExperts
            count = l.load(state.shm_.expert_count + expert, mask=valid, other=0)
            store_words(
                state.workspace_.br_,
                self.Workspace.SendCounterOffset(expert),
                parity * 4,
                count,
                1,
                self.kDeviceStore,
                valid,
            )
            store_words(
                state.workspace_.br_,
                self.Workspace.RecvCounterOffset(
                    expert // self.kExpertsPerRank,
                    state.workspace_.rank_id_,
                    expert % self.kExpertsPerRank,
                ),
                parity * 4,
                count,
                1,
                self.kSystemStore,
                valid,
            )
            l.store(state.shm_.expert_count + expert, 0, mask=valid)
        complete_scoped_vmem()
        l.barrier()
        destination = (state.workspace_.rank_id_ + tid) % self.kNumRanks
        store_words(
            state.workspace_.br_,
            self.Workspace.DirectPushCountDoneOffset(
                destination, parity, state.workspace_.rank_id_
            ),
            0,
            expected,
            1,
            self.kSystemStore,
            tid < self.kNumRanks,
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
        if wave_id() == 0:
            plan_lane = l.arange(
                0, 64, layout=l.BlockedLayout([1], [64], [l.num_warps()], [0])
            ).to(l.uint32)
            wait_tensor_signal(
                state.workspace_,
                self.Workspace.DirectPushCountDoneOffset(
                    state.workspace_.rank_id_, parity, plan_lane
                ),
                expected.to(l.int32),
                plan_lane < self.kNumRanks,
                False,
            )
            compiler_memory_barrier()
            pool_rows = l.full((), 0, l.uint32)
            for i in l.static_range(self.kExpertsPerLane):
                plan_expert = i * 64 + plan_lane
                plan_valid = plan_expert < self.kExpertsPerRank
                total = l.full(plan_lane.shape, 0, l.uint32, plan_lane.type.layout)
                for source in range(self.kNumRanks):
                    count = load_words(
                        state.workspace_.br_,
                        self.Workspace.RecvCounterOffset(
                            state.workspace_.rank_id_, source, plan_expert
                        ),
                        parity * 4,
                        1,
                        self.kPeerCoherent,
                        plan_valid,
                    )
                    l.store(
                        state.shm_.source_count
                        + source * self.kExpertsPerRank
                        + plan_expert,
                        count,
                        mask=plan_valid,
                    )
                    total += count
                padded = l.where(plan_valid, (total + 31) // 32 * 32, 0)
                inclusive = amdgcn_wave_inclusive_add(padded, plan_lane)
                pool_base = pool_rows + inclusive - padded
                store_words(
                    state.workspace_.br_,
                    self.Workspace.RecvSumCounterOffset(plan_expert),
                    0,
                    (total, self.kNumSMs * self.kNumRanks),
                    2,
                    self.kDeviceStore,
                    plan_valid,
                )
                source_prefix = l.full(
                    plan_lane.shape, 0, l.uint32, plan_lane.type.layout
                )
                for source in range(self.kNumRanks):
                    store_words(
                        state.workspace_.br_,
                        self.Workspace.DirectPushPlanBaseOffset(
                            source, state.workspace_.rank_id_, plan_expert
                        ),
                        parity * 4,
                        pool_base + source_prefix,
                        1,
                        self.kSystemStore,
                        plan_valid,
                    )
                    source_prefix += l.load(
                        state.shm_.source_count
                        + source * self.kExpertsPerRank
                        + plan_expert,
                        mask=plan_valid,
                        other=0,
                    )
                pool_rows += amdgcn_shuffle(inclusive, 63)
            pool_rows = first(pool_rows)
            for block_base in range(0, pool_rows // self.kSortedTokenBlock, 64):
                block = block_base + plan_lane
                plan_valid = block < pool_rows // self.kSortedTokenBlock
                store_words(
                    state.workspace_.br_,
                    self.Workspace.L1PayloadArrivalMaskOffset(
                        state.workspace_.rank_id_, block
                    ),
                    0,
                    0,
                    1,
                    self.kDeviceStore,
                    plan_valid,
                )
                store_words(
                    state.workspace_.br_,
                    self.Workspace.L2ArrivalMaskOffset(block),
                    0,
                    0,
                    1,
                    self.kDeviceStore,
                    plan_valid,
                )
        else:
            group_tid = (wid - 1) * 64 + wtid
            group_threads = (self.kThreads // 64 - 1) * 64
            route_count = state.num_tokens_ * self.kTopK
            for route_base in range(0, route_count, group_threads):
                route = route_base + group_tid
                expert = self.LoadInputExpert(state, route)
                valid = (route < route_count) & (expert < self.kNumExperts)
                ordinal = _native_call(
                    "when:memory.atomic.add.i32",
                    "i32",
                    ("p3", "i32", "i1"),
                    (state.shm_.expert_count + expert, 1, valid),
                    False,
                )
                store_words(
                    state.workspace_.br_,
                    self.Workspace.RouteIndexOffset(
                        expert // self.kExpertsPerRank,
                        expert % self.kExpertsPerRank,
                        ordinal,
                    ),
                    0,
                    route,
                    1,
                    self.kDeviceStore,
                    valid,
                )
        complete_scoped_vmem()
        l.barrier()
        store_words(
            state.workspace_.br_,
            self.Workspace.DirectPushPlanReadyOffset(
                tid, parity, state.workspace_.rank_id_
            ),
            0,
            expected,
            1,
            self.kSystemStore,
            tid < self.kNumRanks,
        )
        store_words(
            state.workspace_.br_,
            self.Workspace.DirectPushEpochGateOffset(state.workspace_.rank_id_),
            0,
            state.current_epoch_,
            1,
            self.kSystemStore,
            tid == 0,
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
        kDispatchBlocks: l.constexpr,
    ):
        destination = producer_slot % self.kNumRanks
        wait_xgpu_signal_relaxed(
            state.workspace_,
            self.Workspace.DirectPushPlanReadyOffset(
                state.workspace_.rank_id_, parity, destination
            ),
            expected.to(l.int32),
        )
        l.barrier()
        if kDispatchBlocks == 56 and kDispatchBlocks >= self.kNumRanks:
            kProducersPerDestination: l.constexpr = kDispatchBlocks // self.kNumRanks
            destination_producer = producer_slot // self.kNumRanks
            if wave_id() == 0:
                for i in l.static_range(self.kExpertsPerLane):
                    local_expert = i * 64 + wtid
                    valid = local_expert < self.kExpertsPerRank
                    expert = destination * self.kExpertsPerRank + local_expert
                    count = load_words(
                        state.workspace_.br_,
                        self.Workspace.SendCounterOffset(expert),
                        parity * 4,
                        1,
                        self.kPeerCoherent,
                        valid,
                    )
                    base = load_words(
                        state.workspace_.br_,
                        self.Workspace.DirectPushPlanBaseOffset(
                            state.workspace_.rank_id_, destination, local_expert
                        ),
                        parity * 4,
                        1,
                        self.kPeerCoherent,
                        valid,
                    )
                    l.store(state.shm_.expert_count + local_expert, count, mask=valid)
                    l.store(state.shm_.source_count + local_expert, base, mask=valid)
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
                    self.CopyPayloadRows(
                        state,
                        destination,
                        local_expert,
                        plan[1],
                        begin,
                        end,
                        tid,
                        wid,
                        wtid,
                    )
            return

        for task_index in range(producer_slot, self.kNumExperts, kDispatchBlocks):
            fallback_local_expert = task_index // self.kNumRanks
            plan = self.LoadPayloadPlan(
                state, destination, fallback_local_expert, parity, tid
            )
            self.CopyPayloadRows(
                state,
                destination,
                fallback_local_expert,
                plan[1],
                0,
                plan[0],
                tid,
                wid,
                wtid,
            )

    @device_method
    def LoadPayloadPlan(self, state, destination, local_expert, parity, tid):
        expert = destination * self.kExpertsPerRank + local_expert
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
        return plan0, plan1

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
    ):
        rows = ordinal_end - ordinal_begin
        if rows >= self.kThreads // 64 * 2:
            for ordinal in range(
                ordinal_begin + wave_id(), ordinal_end, self.kThreads // 64
            ):
                route = BufferResource.LoadU32(
                    state.workspace_.br_,
                    self.Workspace.RouteIndexOffset(destination, local_expert, ordinal),
                    0,
                    self.kPeerCoherent,
                )
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
        complete_scoped_vmem()
        l.barrier()
        first, end = pool_base + ordinal_begin, pool_base + ordinal_end
        for pool_block in tl.range(
            first // self.kSortedTokenBlock,
            (end + self.kSortedTokenBlock - 1) // self.kSortedTokenBlock,
            loop_unroll_factor=1,
        ):
            prevent_loop_unroll()
            block_first = pool_block * self.kSortedTokenBlock
            lo = l.where(first > block_first, first - block_first, 0)
            block_end = block_first + self.kSortedTokenBlock
            hi = l.where(end < block_end, end - block_first, self.kSortedTokenBlock)
            high_mask = l.where(hi == 32, 0xFFFFFFFF, (1 << hi) - 1).to(l.uint32)
            low_mask = l.where(lo == 0, 0, (1 << lo) - 1).to(l.uint32)
            atomic_or(
                state.workspace_.br_,
                self.Workspace.L1PayloadArrivalMaskOffset(destination, pool_block),
                0,
                high_mask & ~low_mask,
                BufferResource.kAtomicScopeSystem,
                tid == 0,
            )
        l.barrier()

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
