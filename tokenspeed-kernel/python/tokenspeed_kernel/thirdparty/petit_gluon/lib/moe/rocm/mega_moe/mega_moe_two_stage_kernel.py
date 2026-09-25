"""Native three-launch MegaMoE: dispatch/stage one, stage two, and combine."""

from typing import NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    _native_call,
    amdgcn_readfirstlane,
    amdgcn_s_waitcnt,
)
from lib.moe.rocm.comm.barrier import (
    compiler_memory_barrier,
    complete_scoped_vmem,
    grid_sync,
    store_xgpu_epoch_release,
    system_fence_acquire,
    wait_xgpu_epoch_relaxed,
    wave_barrier,
)
from lib.moe.rocm.fused_moe import ClearMat
from lib.moe.rocm.mega_moe.scheduler import MegaMoETwoStageScheduler, Work
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.moe.rocm.ops.mega_moe.route_output import (
    Context,
    MegaMoETwoStage2Epilogue,
    SourceRouteReducer,
)
from lib.moe.rocm.ops.mega_moe.token_shuffle_direct_push import DirectPushTokenShuffle
from lib.moe.rocm.ops.mxfp4_activation import MxFp4ActivationQuantizer, MxFp4Stage2Input
from lib.moe.rocm.profiler import Profiler, RecordProfile
from lib.tal.device import DeviceTemplate, device_method
from lib.tal.thread_jit import thread_jit
from triton.experimental.gluon import language as l

kMegaMoEProfileCtas = 3072


class Stage1Cycles(NamedTuple):
    payload_wait: object
    prepare: object
    matmul: object
    quantize_store: object
    publish: object


class MegaMoETwoStageCommComputeKernel(DeviceTemplate):
    def __init__(self, Config, kExternalInputs=False, kProfile=False):
        self._key = (Config.cache_key, kExternalInputs, kProfile)
        self.Config, self.kProfile, self.Profiler = Config, kProfile, Profiler(kProfile)
        for field in (
            "Input",
            "W13Weights",
            "W2Weights",
            "Bias",
            "Stage2Bias",
            "Stage1Tiles",
            "Stage1Op",
            "Stage2Tiles",
        ):
            setattr(self, field, getattr(Config, field))
        self.Stage2Epilogue = MegaMoETwoStage2Epilogue(self.Stage2Tiles)
        self.ActivationQuantizer = MxFp4ActivationQuantizer(Config)
        self.Stage2Input = MxFp4Stage2Input(Config)
        self.Workspace, self.Scheduler = MegaMoEWorkspace(
            Config
        ), MegaMoETwoStageScheduler(Config)
        self.TokenDispatch = DirectPushTokenShuffle(Config, kExternalInputs, kProfile)
        self.XGpuSync = Config.XGpuSync
        for field in (
            "kNumWarps",
            "kThreads",
            "kNumSMs",
            "kTokenBatch",
            "kSortedTokenBlock",
            "kGroupDim",
            "kInterDim",
            "kComputeHiddenSize",
        ):
            setattr(self, field, getattr(Config, field))
        self.kRoutesPerBlock = Config.kGroupM
        self.kK256Tiles = self.kInterDim // self.kGroupDim
        self.kStage1TileCount = self.kInterDim // Config.kStage1GroupN
        self.kStage2GridBlocks, self.kWorkShards = self.kNumSMs * 5, 8
        self.kOverlapStage1WorkId = getattr(Config, "kOverlapStage1WorkId", False)
        self.kComputeWords = max(
            self.Stage1Op.kShmWords,
            self.ActivationQuantizer.kQuantizeShmWords,
            self.Stage2Epilogue.kShmWords,
            self.Stage2Input.kInputShmWords,
        )
        self.kUnionWords = max(self.kComputeWords, self.TokenDispatch.kShmWords, 1)
        self.kWorkIdWord = 0 if self.kOverlapStage1WorkId else self.kUnionWords
        self.kShmWords = self.kUnionWords + (0 if self.kOverlapStage1WorkId else 1)
        assert Config.kGroupN == 256 and Config.kStage1GroupN in (128, 256)
        assert self.kInterDim % 512 == 0 and self.kK256Tiles >= 2
        assert self.kNumSMs % self.kWorkShards == 0
        assert (
            self.Stage1Tiles.kAccumFragments == self.ActivationQuantizer.kInputFragments
        )

    @device_method
    def WorkId(self, shm):
        return shm + self.kWorkIdWord

    @device_method
    def NextDynamicWork(self, workspace, shm, sm_id, tid, logical_id, set=0):
        shard = sm_id & (self.kWorkShards - 1)
        if tid == 0:
            local_work = BufferResource.AtomicAddI32(
                workspace.br_,
                self.Workspace.DirectPushWorkHeadOffset(shard, set),
                0,
                1,
                BufferResource.kAtomicScopeAgent,
            ).to(l.uint32)
            l.store(self.WorkId(shm), shard + local_work * self.kWorkShards)
        l.barrier()
        return l.load(self.WorkId(shm))

    @device_method
    def WaitForPayloadBlocks(self, workspace, work, tid, payload_wait_cycles):
        start = self.Profiler.Start()
        if tid == 0:
            subblocks = (
                work.work_m + self.kSortedTokenBlock - 1
            ) // self.kSortedTokenBlock
            for subblock in range(subblocks):
                rows = l.minimum(
                    self.kSortedTokenBlock,
                    work.work_m - subblock * self.kSortedTokenBlock,
                )
                ready_mask = l.where(rows == 32, 0xFFFFFFFF, (1 << rows) - 1).to(
                    l.uint32
                )
                observed = l.full((), 0, l.uint32)
                pending = l.full((), True, l.int1)
                while pending:
                    observed = BufferResource.LoadU32(
                        workspace.br_,
                        self.Workspace.L1PayloadArrivalMaskOffset(
                            workspace.rank_id_, work.pool_block + subblock
                        ),
                        0,
                        BufferResource.kSC0Bit | BufferResource.kSC1Bit,
                    )
                    pending = (observed & ready_mask) != ready_mask
                    if pending:
                        _native_call("s.sleep.1", "void", (), (), False)
        l.barrier()
        compiler_memory_barrier()
        l.barrier()
        return payload_wait_cycles + self.Profiler.End(start)

    @device_method
    def RunStage1(
        self,
        workspace,
        shm,
        dispatch,
        dispatch_epoch,
        w13,
        scales_w13,
        w13_bias,
        work,
        tid,
        wid,
        wtid,
        cycles,
    ):
        payload_wait = self.WaitForPayloadBlocks(
            workspace, work, tid, cycles.payload_wait
        )
        start = self.Profiler.Start()
        pool_base = work.pool_row
        input_state = self.Input.Initialize(
            workspace.br_, workspace.rank_id_, pool_base, work.work_m, self.Workspace
        )
        self.Input.PrepareScales(input_state, shm, wid, wtid, self.Stage1Op.kStage)
        weights = self.W13Weights.Initialize(
            w13,
            scales_w13,
            work.expert_idx,
            work.tile,
            self.Config.kDim,
            self.Config.kInterDim,
        )
        tiles = self.Stage1Tiles.Construct(input_state, weights.w1_, ())
        tiles = self.Stage1Tiles.InitializeBias(
            tiles, w13_bias, work.expert_idx, work.tile
        )
        tokens = ()
        for i in l.static_range(self.kTokenBatch):
            tokens += (wid * self.kTokenBatch + i,)
        prepare = cycles.prepare + self.Profiler.End(start)
        start = self.Profiler.Start()
        tiles, hidden = self.Stage1Op.Run(
            shm, tiles, tid, wid, wtid, tokens, work.work_m
        )
        matmul = cycles.matmul + self.Profiler.End(start)
        start = self.Profiler.Start()
        l.barrier()
        quant_shm = shm.to(l.pointer_type(l.float32, 3))
        self.ActivationQuantizer.StoreAccumulator(quant_shm, hidden, wid, wtid)
        l.barrier()
        route_in_slice, col_lane = tid // 32, tid % 32
        for route_slice in l.static_range(self.Config.kGroupM // 8):
            route = route_slice * 8 + route_in_slice
            if route < work.work_m:
                for col_segment in l.static_range(self.Config.kStage1GroupN // 128):
                    quant_col = col_segment * 32 + col_lane
                    quantized = self.ActivationQuantizer.Quantize(
                        quant_shm, route, quant_col
                    )
                    self.ActivationQuantizer.Store(
                        workspace.br_,
                        self.Workspace.L2TokenBufferOffset(0),
                        self.Workspace.L2ScaleBufferOffset(),
                        pool_base + route,
                        pool_base + route,
                        work.tile,
                        quant_col,
                        self.kInterDim,
                        self.Workspace.kL2ScaleCols,
                        quantized,
                        BufferResource.kSC1Bit,
                    )
        complete_scoped_vmem()
        quantize_store = cycles.quantize_store + self.Profiler.End(start)
        start = self.Profiler.Start()
        l.barrier()
        if tid == 0:
            for subblock in range((work.work_m + 31) // 32):
                BufferResource.AtomicOrU32(
                    workspace.br_,
                    self.Workspace.L2ArrivalMaskOffset(work.pool_block + subblock),
                    0,
                    1 << work.tile,
                    BufferResource.kAtomicScopeAgent,
                )
        l.barrier()
        publish = cycles.publish + self.Profiler.End(start)
        return Stage1Cycles(payload_wait, prepare, matmul, quantize_store, publish)

    @device_method
    def WaitL2Block(self, workspace, pool_block, tid):
        kReadyMask: l.constexpr = (1 << self.kStage1TileCount) - 1
        if tid == 0:
            observed = l.full((), 0, l.uint32)
            pending = l.full((), True, l.int1)
            while pending:
                observed = BufferResource.LoadU32(
                    workspace.br_,
                    self.Workspace.L2ArrivalMaskOffset(pool_block),
                    0,
                    BufferResource.kSC0Bit | BufferResource.kSC1Bit,
                )
                pending = (observed & kReadyMask) != kReadyMask
                if pending:
                    _native_call("s.sleep.1", "void", (), (), False)
        l.barrier()

    @device_method
    def RunStage2(
        self,
        workspace,
        shm,
        w2,
        scales_w2,
        w2_bias,
        work,
        tid,
        wid,
        wtid,
        ready_wait_cycles,
    ):
        pool_base = work.pool_block * self.kRoutesPerBlock
        weights = self.W2Weights.Initialize(
            w2, scales_w2, work.expert_idx, work.tile, 0
        )
        tiles = self.Stage2Tiles.Construct(weights.w2_, tid)
        tiles = self.Stage2Tiles.InitializeBias(
            tiles, w2_bias, work.expert_idx, work.tile
        )
        accum = ClearMat(tid, self.Stage2Tiles.kAccumFragments)
        start = self.Profiler.Start()
        self.WaitL2Block(workspace, work.pool_block, tid)
        ready_wait_cycles += self.Profiler.End(start)
        row, vector = (
            tid // self.Stage2Input.kVectorsPerRow,
            tid % self.Stage2Input.kVectorsPerRow,
        )
        value_voffset = (pool_base + row) * (self.kInterDim // 2) + vector * 16
        value_soffset = self.Workspace.L2TokenBufferOffset(0)
        scale_voffset = pool_base * self.Stage2Input.kScaleCols
        scale_soffset = self.Workspace.L2ScaleBufferOffset()
        tile_col = work.tile * self.Config.kGroupN
        context = Context(workspace, pool_base, work.work_m)
        route_weights = self.Stage2Epilogue.LoadRouteWeights(context, wtid)
        bias = self.Stage2Epilogue.PrefetchBias(tiles, tile_col, tid)
        for tile_k in l.static_range(self.kK256Tiles):
            stage: l.constexpr
            stage = tile_k & 1
            prefetched = self.Stage2Input.LoadTile(
                workspace.br_,
                value_voffset,
                value_soffset,
                scale_voffset,
                scale_soffset,
                tile_k,
                True,
                wtid,
                BufferResource.kSC1Bit,
            )
            self.Stage2Input.StoreLds(shm, prefetched.value, stage, tid)
            tiles = self.Stage2Tiles.LoadKStage(tiles, stage, tid, wid, wtid)
            l.barrier()
            input_regs = self.Stage2Input.ReadLds(shm, stage, prefetched.scale, wtid)
            accum = self.Stage2Tiles.Matmul(tiles, accum, input_regs, stage, wtid)
        l.barrier()
        accum = self.Stage2Epilogue.Apply(accum, bias, route_weights)
        self.Stage2Epilogue.WriteShm(shm, accum, wid, wtid)
        l.barrier()
        self.Stage2Epilogue.WriteBack(context, shm, tile_col, wid, wtid)
        return ready_wait_cycles

    @device_method
    def ComputeStage1Only(
        self,
        workspace,
        scheduler,
        shm,
        dispatch,
        dispatch_epoch,
        w13,
        scales_w13,
        w13_bias,
        sm_id,
        tid,
        wid,
        wtid,
        profile,
    ):
        zero = l.full((), 0, l.uint64)
        scheduler_cycles, work_cycles, work_count = zero, zero, zero
        cycles = Stage1Cycles(zero, zero, zero, zero, zero)
        loop_start = self.Profiler.Start()
        valid = l.full((), True, l.int1)
        while valid:
            item_start = self.Profiler.Start()
            logical_id = self.NextDynamicWork(workspace, shm, sm_id, tid, 0)
            valid, work = self.Scheduler.GetStage1Work(scheduler, wtid, logical_id)
            scheduler_cycles += self.Profiler.End(item_start)
            if valid:
                phase = amdgcn_readfirstlane(work.phase)
                valid = phase == 0
                if valid:
                    work = Work(
                        phase,
                        amdgcn_readfirstlane(work.expert_idx),
                        amdgcn_readfirstlane(work.pool_block),
                        amdgcn_readfirstlane(work.pool_row),
                        amdgcn_readfirstlane(work.work_m),
                        amdgcn_readfirstlane(work.tile),
                    )
                    item_start = self.Profiler.Start()
                    cycles = self.RunStage1(
                        workspace,
                        shm,
                        dispatch,
                        dispatch_epoch,
                        w13,
                        scales_w13,
                        w13_bias,
                        work,
                        tid,
                        wid,
                        wtid,
                        cycles,
                    )
                    work_cycles += self.Profiler.End(item_start)
                    work_count += 1
        RecordProfile(
            profile, 3, self.Profiler.End(loop_start), tid, sm_id, self.kProfile, 3072
        )
        RecordProfile(profile, 4, scheduler_cycles, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 5, cycles.payload_wait, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 6, work_cycles, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 7, work_count, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 20, cycles.prepare, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 21, cycles.matmul, tid, sm_id, self.kProfile, 3072)
        RecordProfile(
            profile, 22, cycles.quantize_store, tid, sm_id, self.kProfile, 3072
        )
        RecordProfile(profile, 23, cycles.publish, tid, sm_id, self.kProfile, 3072)

    @device_method
    def ComputeStage2Only(
        self,
        workspace,
        scheduler,
        shm,
        w2,
        scales_w2,
        w2_bias,
        sm_id,
        tid,
        wid,
        wtid,
        profile,
    ):
        zero = l.full((), 0, l.uint64)
        scheduler_cycles, ready_wait_cycles, work_cycles, work_count = (
            zero,
            zero,
            zero,
            zero,
        )
        loop_start = self.Profiler.Start()
        stage2_id = sm_id
        valid = l.full((), True, l.int1)
        while valid:
            item_start = self.Profiler.Start()
            valid, work = self.Scheduler.GetStage2Work(scheduler, wtid, stage2_id)
            scheduler_cycles += self.Profiler.End(item_start)
            if valid:
                work = Work(
                    work.phase,
                    amdgcn_readfirstlane(work.expert_idx),
                    amdgcn_readfirstlane(work.pool_block),
                    amdgcn_readfirstlane(work.pool_row),
                    amdgcn_readfirstlane(work.work_m),
                    amdgcn_readfirstlane(work.tile),
                )
                item_start = self.Profiler.Start()
                ready_wait_cycles = self.RunStage2(
                    workspace,
                    shm,
                    w2,
                    scales_w2,
                    w2_bias,
                    work,
                    tid,
                    wid,
                    wtid,
                    ready_wait_cycles,
                )
                work_cycles += self.Profiler.End(item_start)
                work_count += 1
                stage2_id += self.kStage2GridBlocks
        RecordProfile(
            profile, 8, self.Profiler.End(loop_start), tid, sm_id, self.kProfile, 3072
        )
        RecordProfile(profile, 9, scheduler_cycles, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 10, ready_wait_cycles, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 11, work_cycles, tid, sm_id, self.kProfile, 3072)
        RecordProfile(profile, 12, work_count, tid, sm_id, self.kProfile, 3072)

    @device_method
    def RunStage1Kernel(
        self,
        w13,
        scales_w13,
        num_tokens,
        w13_bias,
        base,
        rank,
        input_tokens,
        input_topk_ids,
        input_topk_weights,
        profile,
        shm,
        sm_id,
        tid,
    ):
        wid, wtid = amdgcn_readfirstlane(tid // 64), tid % 64
        workspace = self.Workspace.Initialize(base, rank)
        total_start = self.Profiler.Start()
        phase_start = self.Profiler.Start()
        dispatch = self.TokenDispatch.Construct(
            num_tokens, workspace, shm, input_tokens, input_topk_ids, input_topk_weights
        )
        dispatch, dispatch_epoch, dispatch_profile = self.TokenDispatch.Run(
            dispatch, sm_id, tid, wid, wtid
        )
        RecordProfile(
            profile, 0, self.Profiler.End(phase_start), tid, sm_id, self.kProfile, 3072
        )
        for i in l.static_range(11):
            RecordProfile(
                profile, 24 + i, dispatch_profile[i], tid, sm_id, self.kProfile, 3072
            )
        phase_start = self.Profiler.Start()
        self.TokenDispatch.WaitForLocalPlan(dispatch, dispatch_epoch, tid)
        RecordProfile(
            profile, 1, self.Profiler.End(phase_start), tid, sm_id, self.kProfile, 3072
        )
        phase_start = self.Profiler.Start()
        scheduler = self.Scheduler.Construct(workspace)
        scheduler = self.Scheduler.FetchRecvSumPerExpert(scheduler, wtid)
        RecordProfile(
            profile, 2, self.Profiler.End(phase_start), tid, sm_id, self.kProfile, 3072
        )
        self.ComputeStage1Only(
            workspace,
            scheduler,
            shm,
            dispatch,
            dispatch_epoch,
            w13,
            scales_w13,
            w13_bias,
            sm_id,
            tid,
            wid,
            wtid,
            profile,
        )
        RecordProfile(
            profile, 18, self.Profiler.End(total_start), tid, sm_id, self.kProfile, 3072
        )

    @device_method
    def RunStage2Kernel(
        self,
        out,
        w2,
        scales_w2,
        num_tokens,
        w2_bias,
        base,
        rank,
        profile,
        shm,
        sm_id,
        tid,
    ):
        wid, wtid = amdgcn_readfirstlane(tid // 64), tid % 64
        workspace = self.Workspace.Initialize(base, rank)
        total_start = self.Profiler.Start()
        scheduler = self.Scheduler.Construct(workspace)
        scheduler = self.Scheduler.FetchRecvSumPerExpert(scheduler, wtid)
        self.ComputeStage2Only(
            workspace,
            scheduler,
            shm,
            w2,
            scales_w2,
            w2_bias,
            sm_id,
            tid,
            wid,
            wtid,
            profile,
        )
        RecordProfile(
            profile, 19, self.Profiler.End(total_start), tid, sm_id, self.kProfile, 3072
        )


class MegaMoECombineKernel(DeviceTemplate):
    kNumSMs, kNumWarps, kThreads, kOutputHandoffGridSyncIndex = 128, 8, 512, 4

    def __init__(self, Config, kProfile=False):
        self._key = (Config.cache_key, kProfile)
        self.Config, self.Workspace = Config, MegaMoEWorkspace(Config)
        self.kProfile, self.Profiler = kProfile, Profiler(kProfile)
        self.Reducer = SourceRouteReducer(Config, self.kNumSMs, self.kThreads)

    @device_method
    def Run(self, out, num_tokens, output_row_stride, base, rank, profile, sm_id, tid):
        wid, wtid = tid // 64, tid % 64
        workspace = self.Workspace.Initialize(base, rank)
        total_start = self.Profiler.Start()
        phase_start = self.Profiler.Start()
        dispatch_epoch = BufferResource.LoadU32(
            workspace.br_,
            self.Workspace.DirectPushEpochGateOffset(workspace.rank_id_),
            0,
            BufferResource.kSC0Bit | BufferResource.kSC1Bit,
        )
        RecordProfile(
            profile, 13, self.Profiler.End(phase_start), tid, sm_id, self.kProfile, 3072
        )
        phase_start = self.Profiler.Start()
        grid_sync(
            self.Workspace,
            workspace,
            sm_id,
            tid,
            l.barrier,
            self.kNumSMs,
            self.kOutputHandoffGridSyncIndex,
            False,
            True,
        )
        RecordProfile(
            profile, 14, self.Profiler.End(phase_start), tid, sm_id, self.kProfile, 3072
        )
        phase_start = self.Profiler.Start()
        if tid < 64:
            if sm_id == 0:
                system_fence_acquire()
                if tid < self.Config.kNumRanks:
                    store_xgpu_epoch_release(
                        workspace,
                        self.Workspace.XGpuEpochSignalOffset(tid, workspace.rank_id_),
                        dispatch_epoch,
                    )
            wave_barrier()
            if sm_id == 0:
                amdgcn_s_waitcnt(0, -1, 0)
            if tid < self.Config.kNumRanks:
                wait_xgpu_epoch_relaxed(
                    workspace,
                    self.Workspace.XGpuEpochSignalOffset(workspace.rank_id_, tid),
                    dispatch_epoch,
                )
            wave_barrier()
        l.barrier()
        RecordProfile(
            profile, 15, self.Profiler.End(phase_start), tid, sm_id, self.kProfile, 3072
        )
        phase_start = self.Profiler.Start()
        self.Reducer.Run(
            workspace, out, num_tokens, output_row_stride, sm_id, wid, wtid
        )
        RecordProfile(
            profile, 16, self.Profiler.End(phase_start), tid, sm_id, self.kProfile, 3072
        )
        RecordProfile(
            profile, 17, self.Profiler.End(total_start), tid, sm_id, self.kProfile, 3072
        )


@thread_jit
def _stage1_thread(
    Kernel: l.constexpr,
    w13,
    scales_w13,
    num_tokens,
    w13_bias,
    base,
    rank,
    input_tokens,
    input_topk_ids,
    input_topk_weights,
    profile,
    shm,
    sm_id,
    tid,
):
    Kernel.RunStage1Kernel(
        w13,
        scales_w13,
        num_tokens,
        w13_bias,
        base,
        rank,
        input_tokens,
        input_topk_ids,
        input_topk_weights,
        profile,
        shm,
        sm_id,
        tid,
    )


@thread_jit
def _stage2_thread(
    Kernel: l.constexpr, w2, scales_w2, w2_bias, base, rank, profile, shm, sm_id, tid
):
    Kernel.RunStage2Kernel(
        None, w2, scales_w2, 0, w2_bias, base, rank, profile, shm, sm_id, tid
    )


@thread_jit
def _combine_thread(
    Kernel: l.constexpr,
    out,
    num_tokens,
    output_row_stride,
    base,
    rank,
    profile,
    sm_id,
    tid,
):
    Kernel.Run(out, num_tokens, output_row_stride, base, rank, profile, sm_id, tid)


@g.jit
def MegaMoEStage1(
    w13,
    scales_w13,
    num_tokens,
    w13_bias,
    base,
    rank,
    input_tokens,
    input_topk_ids,
    input_topk_weights,
    Kernel: l.constexpr,
    profile=None,
):
    tid = l.arange(
        0, Kernel.kThreads, layout=l.BlockedLayout([1], [64], [Kernel.kNumWarps], [0])
    ).to(l.uint32)
    storage = l.allocate_shared_memory(
        l.uint32,
        [((Kernel.kShmWords + 3) // 4) * 4],
        l.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    shm = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    _stage1_thread(
        Kernel,
        w13,
        scales_w13,
        num_tokens,
        w13_bias,
        base,
        rank,
        input_tokens,
        input_topk_ids,
        input_topk_weights,
        profile,
        shm,
        l.program_id(0).to(l.uint32),
        tid,
    )
    storage._keep_alive()


@g.jit
def MegaMoEStage2(
    w2, scales_w2, w2_bias, base, rank, Kernel: l.constexpr, profile=None
):
    tid = l.arange(
        0, Kernel.kThreads, layout=l.BlockedLayout([1], [64], [Kernel.kNumWarps], [0])
    ).to(l.uint32)
    storage = l.allocate_shared_memory(
        l.uint32,
        [((Kernel.kShmWords + 3) // 4) * 4],
        l.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    shm = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    _stage2_thread(
        Kernel,
        w2,
        scales_w2,
        w2_bias,
        base,
        rank,
        profile,
        shm,
        l.program_id(0).to(l.uint32),
        tid,
    )
    storage._keep_alive()


@g.jit
def MegaMoECombine(
    out, num_tokens, output_row_stride, base, rank, Kernel: l.constexpr, profile=None
):
    tid = l.arange(
        0, Kernel.kThreads, layout=l.BlockedLayout([1], [64], [Kernel.kNumWarps], [0])
    ).to(l.uint32)
    _combine_thread(
        Kernel,
        out,
        num_tokens,
        output_row_stride,
        base,
        rank,
        profile,
        l.program_id(0).to(l.uint32),
        tid,
    )


MegaMoEStage1Profile = MegaMoEStage1
MegaMoEStage2Profile = MegaMoEStage2
MegaMoECombineProfile = MegaMoECombine
