"""Native three-launch MegaMoE: dispatch/stage one, stage two, and combine."""

import triton.experimental.gluon as g
import triton.language as tl
from lib.gemm.rocm.intrinsics import (
    BufferResource,
    _native_call,
    amdgcn_readfirstlane,
    amdgcn_s_waitcnt,
)
from lib.moe.rocm.comm.barrier import (
    compiler_memory_barrier,
    complete_scoped_vmem,
    system_fence_acquire,
    system_fence_release,
    tensor_grid_sync,
    wait_tensor_signal,
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
from lib.tal.device import DeviceTemplate, device_method
from lib.tal.tensor_ops import (
    atomic_add,
    atomic_or,
    is_first_thread,
    prevent_loop_unroll,
    store_words,
    wave_id,
)
from triton.experimental.gluon import language as l


class MegaMoETwoStageCommComputeKernel(DeviceTemplate):
    """Dispatch tokens and compute both expert projections in separate launches.

    Config defines expert layouts, compute tiles, and workspace geometry.
    kExternalInputs selects caller-provided packed tokens and routing tensors;
    otherwise dispatch reads inputs from the symmetric workspace.
    Stage one publishes activated MXFP4 intermediates; stage two writes weighted
    contributions to source ranks. Launch both stages, then MegaMoECombine, on
    the same stream and workspace on every rank, including ranks with no tokens.
    """

    def __init__(self, Config, kExternalInputs):
        self._key = (Config.cache_key, kExternalInputs)
        self.Config = Config
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
        self.TokenDispatch = DirectPushTokenShuffle(Config, kExternalInputs)
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
        # Recompute the ticket offset here so it does not occupy an SGPR
        # throughout the matrix loop. The tied operand emits no instruction.
        sm_id = l.inline_asm_elementwise(
            "", "=s,0", [sm_id], l.uint32, is_pure=False, pack=1
        )
        shard = sm_id & (self.kWorkShards - 1)
        leader = is_first_thread(tid)
        local_work = atomic_add(
            workspace.br_,
            self.Workspace.DirectPushWorkHeadOffset(shard, set),
            0,
            1,
            BufferResource.kAtomicScopeAgent,
            leader,
        ).to(l.uint32)
        l.store(
            self.WorkId(shm) + l.full(tid.shape, 0, l.uint32, tid.type.layout),
            shard + local_work * self.kWorkShards,
            mask=leader,
        )
        l.barrier()
        return l.load(self.WorkId(shm))

    @device_method
    def WaitForPayloadBlocks(self, workspace, work, tid):
        subblocks = (work.work_m + self.kSortedTokenBlock - 1) // self.kSortedTokenBlock
        for subblock in range(subblocks):
            rows = l.minimum(
                self.kSortedTokenBlock,
                work.work_m - subblock * self.kSortedTokenBlock,
            )
            ready_mask = l.where(rows == 32, 0xFFFFFFFF, (1 << rows) - 1).to(l.uint32)
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
    ):
        self.WaitForPayloadBlocks(workspace, work, tid)
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
        tiles, hidden = self.Stage1Op.Run(
            shm, tiles, tid, wid, wtid, tokens, work.work_m
        )
        l.barrier()
        quant_shm = shm.to(l.pointer_type(l.float32, 3))
        self.ActivationQuantizer.StoreAccumulator(quant_shm, hidden, wid, wtid)
        l.barrier()
        route_in_slice, col_lane = tid // 32, tid % 32
        # Skip slices containing no routed rows, as the original per-thread
        # predicate did. Keep the existing lane ownership and store masks.
        for route_slice in tl.range((work.work_m + 7) // 8, loop_unroll_factor=1):
            prevent_loop_unroll()
            route = route_slice * 8 + route_in_slice
            for col_segment in l.static_range(self.Config.kStage1GroupN // 128):
                quant_col = col_segment * 32 + col_lane
                quantized = self.ActivationQuantizer.Quantize(
                    quant_shm, l.minimum(route, self.Config.kGroupM - 1), quant_col
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
                    route < work.work_m,
                )
        complete_scoped_vmem()
        l.barrier()
        for subblock in tl.range((work.work_m + 31) // 32, loop_unroll_factor=1):
            prevent_loop_unroll()
            atomic_or(
                workspace.br_,
                self.Workspace.L2ArrivalMaskOffset(work.pool_block + subblock),
                0,
                1 << work.tile,
                BufferResource.kAtomicScopeAgent,
                is_first_thread(tid),
            )
        l.barrier()

    @device_method
    def WaitL2Block(self, workspace, pool_block, tid):
        kReadyMask: l.constexpr = (1 << self.kStage1TileCount) - 1
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
        self.WaitL2Block(workspace, work.pool_block, tid)
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
    ):
        valid = l.full((), True, l.int1)
        while valid:
            logical_id = self.NextDynamicWork(workspace, shm, sm_id, tid, 0)
            valid, work = self.Scheduler.GetStage1Work(scheduler, wtid, logical_id)
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
                    self.RunStage1(
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
                    )

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
    ):
        stage2_id = sm_id
        valid = l.full((), True, l.int1)
        while valid:
            valid, work = self.Scheduler.GetStage2Work(scheduler, wtid, stage2_id)
            if valid:
                work = Work(
                    work.phase,
                    amdgcn_readfirstlane(work.expert_idx),
                    amdgcn_readfirstlane(work.pool_block),
                    amdgcn_readfirstlane(work.pool_row),
                    amdgcn_readfirstlane(work.work_m),
                    amdgcn_readfirstlane(work.tile),
                )
                self.RunStage2(
                    workspace,
                    shm,
                    w2,
                    scales_w2,
                    w2_bias,
                    work,
                    tid,
                    wid,
                    wtid,
                )
                stage2_id += self.kStage2GridBlocks

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
        shm,
        sm_id,
        tid,
    ):
        """Dispatch this rank's inputs and execute local experts' first projection.

        All CTAs participate in dispatch admission, including empty ranks.
        Compute waits for routed payloads, then publishes activated MXFP4
        intermediates and readiness flags in the workspace for stage two.
        """
        wid, wtid = tid // 64, tid % 64
        workspace = self.Workspace.Initialize(base, rank)
        dispatch = self.TokenDispatch.Construct(
            num_tokens, workspace, shm, input_tokens, input_topk_ids, input_topk_weights
        )
        dispatch, dispatch_epoch = self.TokenDispatch.Run(
            dispatch, sm_id, tid, wid, wtid
        )
        self.TokenDispatch.WaitForLocalPlan(dispatch, dispatch_epoch, tid)
        scheduler = self.Scheduler.Construct(workspace)
        scheduler = self.Scheduler.FetchRecvSumPerExpert(scheduler, wtid)
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
        shm,
        sm_id,
        tid,
    ):
        """Consume ready intermediates and publish weighted expert contributions.

        Stage one must precede this launch on the same stream. Contributions
        are written to source-rank workspace slots; combine produces the output.
        """
        wid, wtid = tid // 64, tid % 64
        workspace = self.Workspace.Initialize(base, rank)
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
        )


class MegaMoECombineKernel(DeviceTemplate):
    kNumSMs, kNumWarps, kThreads, kOutputHandoffGridSyncIndex = 128, 8, 512, 4

    def __init__(self, Config):
        self._key = (Config.cache_key,)
        self.Config, self.Workspace = Config, MegaMoEWorkspace(Config)
        self.Reducer = SourceRouteReducer(Config, self.kNumSMs, self.kThreads)

    @device_method
    def Run(self, out, num_tokens, output_row_stride, base, rank, sm_id, tid):
        """Synchronize stage-two writes across ranks and reduce local token routes.

        Every rank participates in the epoch handoff, even with zero tokens.
        The reducer writes BF16 rows to out, using output_row_stride elements.
        """
        wid, wtid = tid // 64, tid % 64
        workspace = self.Workspace.Initialize(base, rank)
        dispatch_epoch = BufferResource.LoadU32(
            workspace.br_,
            self.Workspace.DirectPushEpochGateOffset(workspace.rank_id_),
            0,
            BufferResource.kSC0Bit | BufferResource.kSC1Bit,
        )
        tensor_grid_sync(
            self.Workspace,
            workspace,
            sm_id,
            self.kNumSMs,
            self.kOutputHandoffGridSyncIndex,
            False,
            True,
        )
        if wave_id() == 0:
            if sm_id == 0:
                system_fence_acquire()
                system_fence_release()
                store_words(
                    workspace.br_,
                    self.Workspace.XGpuEpochSignalOffset(tid, workspace.rank_id_),
                    0,
                    dispatch_epoch,
                    1,
                    BufferResource.kSC0Bit | BufferResource.kSC1Bit,
                    tid < self.Config.kNumRanks,
                )
            wave_barrier()
            if sm_id == 0:
                amdgcn_s_waitcnt(0, -1, 0)
            peer = l.arange(
                0, 64, layout=l.BlockedLayout([1], [64], [l.num_warps()], [0])
            ).to(l.uint32)
            wait_tensor_signal(
                workspace,
                self.Workspace.XGpuEpochSignalOffset(workspace.rank_id_, peer),
                dispatch_epoch,
                peer < self.Config.kNumRanks,
                True,
            )
            wave_barrier()
        l.barrier()
        self.Reducer.Run(
            workspace, out, num_tokens, output_row_stride, sm_id, wid, wtid
        )


@g.jit
def _stage1(
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
        shm,
        sm_id,
        tid,
    )


@g.jit
def _stage2(Kernel: l.constexpr, w2, scales_w2, w2_bias, base, rank, shm, sm_id, tid):
    Kernel.RunStage2Kernel(None, w2, scales_w2, 0, w2_bias, base, rank, shm, sm_id, tid)


@g.jit
def _combine(
    Kernel: l.constexpr,
    out,
    num_tokens,
    output_row_stride,
    base,
    rank,
    sm_id,
    tid,
):
    Kernel.Run(out, num_tokens, output_row_stride, base, rank, sm_id, tid)


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
):
    """Dispatch inputs, apply W13 and activation, and publish MXFP4 intermediates.

    Weights, scales and optional bias must match Kernel's packed expert layout.
    The three input pointers supply packed local tokens, expert IDs and routing
    weights when external inputs are enabled; otherwise inputs reside in base.
    The caller supplies initialized symmetric workspace in base and launches
    Kernel.kNumSMs CTAs on every rank, including ranks with num_tokens == 0.
    Results remain in the workspace; stage two must follow on the same stream.
    """
    tid = l.arange(
        0, Kernel.kThreads, layout=l.BlockedLayout([1], [64], [Kernel.kNumWarps], [0])
    ).to(l.uint32)
    storage = l.allocate_shared_memory(
        l.uint32,
        [((Kernel.kShmWords + 3) // 4) * 4],
        l.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    shm = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    _stage1(
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
        shm,
        l.program_id(0).to(l.uint32),
        tid,
    )
    storage._keep_alive()


@g.jit
def MegaMoEStage2(w2, scales_w2, w2_bias, base, rank, Kernel: l.constexpr):
    """Project stage-one intermediates through W2 and return weighted routes.

    Weights, scales and optional bias must match Kernel's packed expert layout.
    Launch Kernel.kStage2GridBlocks CTAs after stage one on the same stream,
    with the same symmetric workspace and rank. Every rank participates,
    including empty ranks. Results go to source-rank workspace slots for combine.
    """
    tid = l.arange(
        0, Kernel.kThreads, layout=l.BlockedLayout([1], [64], [Kernel.kNumWarps], [0])
    ).to(l.uint32)
    storage = l.allocate_shared_memory(
        l.uint32,
        [((Kernel.kShmWords + 3) // 4) * 4],
        l.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    shm = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    _stage2(
        Kernel,
        w2,
        scales_w2,
        w2_bias,
        base,
        rank,
        shm,
        l.program_id(0).to(l.uint32),
        tid,
    )
    storage._keep_alive()


@g.jit
def MegaMoECombine(out, num_tokens, output_row_stride, base, rank, Kernel: l.constexpr):
    """Sum peer contributions in FP32 into num_tokens rank-local BF16 rows.

    Launch Kernel.kNumSMs CTAs after stage two on the same stream and workspace.
    All ranks join the completion handshake, including ranks with no tokens.
    Writes out in place with output_row_stride measured in BF16 elements.
    Empty ranks write no output. There is no returned value.
    """
    tid = l.arange(
        0, Kernel.kThreads, layout=l.BlockedLayout([1], [64], [Kernel.kNumWarps], [0])
    ).to(l.uint32)
    _combine(
        Kernel,
        out,
        num_tokens,
        output_row_stride,
        base,
        rank,
        l.program_id(0).to(l.uint32),
        tid,
    )
