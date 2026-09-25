"""Current native shared stage-one kernel and route metadata cache."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    _native_load_scalar,
    _native_shared_memory,
    amdgcn_readfirstlane,
)
from lib.moe.rocm.memory_ops import MakeBufferResource
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class RouteWeightsLayout:
    __triton_builtin__ = True
    kRoutesPerGroup = 32

    @g.jit
    def Load(sorted_weights, route_base, tid):
        weights = MakeBufferResource(sorted_weights + route_base, 32 * 4)
        x = BufferResource.LoadU32(weights, (tid % 16) * 4, 0, BufferResource.kNone)
        y = BufferResource.LoadU32(
            weights, (tid % 16 + 16) * 4, 0, BufferResource.kNone
        )
        return x.to(l.float32, bitcast=True), y.to(l.float32, bitcast=True)


class FusedMoEStage1(DeviceTemplate):
    def __init__(self, Config, Epilogue):
        self._key = (Config.cache_key, Epilogue.cache_key)
        self.Config, self.Epilogue = Config, Epilogue
        self.kNumWarps, self.kThreads = Config.kNumWarps, Config.kNumWarps * 64
        self.kScaleBlockSize, self.kTokenBatch = 128, Config.kTokenBatch
        self.kRoutesPerBlock = self.kTokenBatch * self.kNumWarps
        self.Input, self.W13Weights, self.Bias = (
            Config.Input,
            Config.W13Weights,
            Config.Bias,
        )
        self.Stage1Tiles, self.Stage1Op = Config.Stage1Tiles, Config.Stage1Op
        self.kDataWords = max(self.Stage1Op.kShmWords, Epilogue.kShmWords)
        self.kShmWords = self.kDataWords + self.kThreads
        assert self.kRoutesPerBlock in (32, 64)

    @device_method
    def PrefetchTokenMetadata(
        self, row_metadata, sorted_token_ids, route_base, wid, wtid
    ):
        row = (
            (wtid % self.kTokenBatch)
            + wid * self.kTokenBatch
            + (wtid // self.kTokenBatch) * self.kRoutesPerBlock
        )
        metadata = MakeBufferResource(
            sorted_token_ids + route_base, self.kRoutesPerBlock * 4
        )
        l.store(
            row_metadata + row,
            BufferResource.LoadU32(metadata, row * 4, 0, BufferResource.kNone),
        )

    @device_method
    def ReadTokens(self, row_metadata, wid):
        metadata = row_metadata.to(l.pointer_type(l.uint32, 3))
        tokens = ()
        for i in l.static_range(self.kTokenBatch):
            # Native ShmBuf alignment and the scalar element's constant displacement.
            tokens += (
                _native_load_scalar(
                    metadata + wid * self.kTokenBatch + i, (16, 4, 8, 4)[i % 4]
                )
                & 0x00FFFFFF,
            )
        return tokens

    @device_method
    def Stage1(
        self,
        input_state,
        w13_weights,
        stage1_shm,
        wid,
        wtid,
        tid,
        tokens,
        m,
        expert_id,
        tile_k,
        w13_bias,
    ):
        tiles = self.Stage1Tiles.Construct(input_state, w13_weights.w1_, ())
        tiles = self.Stage1Tiles.InitializeBias(tiles, w13_bias, expert_id, tile_k)
        tiles, h = self.Stage1Op.Run(stage1_shm, tiles, tid, wid, wtid, tokens, m)
        return h

    @device_method
    def RunStage1Route(
        self,
        shm,
        epilogue,
        act,
        w13,
        sorted_token_ids,
        scales_act,
        scales_w13,
        route_group,
        route_group_limit,
        expert_id,
        tile_k,
        m,
        num_experts,
        tid,
        wid,
        wtid,
        w13_bias,
    ):
        valid_expert = True
        if self.Config.kValidateExpertIds:
            valid_expert = expert_id < num_experts
        if valid_expert:
            route_base = route_group * self.kRoutesPerBlock
            row_metadata = shm + self.kDataWords
            self.PrefetchTokenMetadata(
                row_metadata, sorted_token_ids, route_base, wid, wtid
            )
            input_state = self.Input.Initialize(
                act,
                scales_act,
                wid,
                m,
                self.Config.kDim // self.kScaleBlockSize,
                route_group,
                route_group_limit,
            )
            w13_weights = self.W13Weights.Initialize(
                w13,
                scales_w13,
                expert_id,
                tile_k,
                self.Config.kDim,
                self.Config.kInterDim,
            )
            tokens = self.ReadTokens(row_metadata, wid)
            hidden = self.Stage1(
                input_state,
                w13_weights,
                shm,
                wid,
                wtid,
                tid,
                tokens,
                m,
                expert_id,
                tile_k,
                w13_bias,
            )
            l.barrier()
            self.Epilogue.Run(
                epilogue,
                shm,
                hidden,
                tokens,
                row_metadata,
                route_base,
                route_group,
                expert_id,
                tile_k,
                tid,
                wid,
                wtid,
            )

    @device_method
    def Compute(
        self,
        act,
        w13,
        sorted_token_ids,
        sorted_expert_ids,
        scales_act,
        scales_w13,
        num_valid_ids_ptr,
        m,
        num_experts,
        persistent_route_step,
        w13_bias,
        epilogue,
    ):
        tid = l.arange(
            0, self.kThreads, layout=l.BlockedLayout([1], [64], [self.kNumWarps], [0])
        ).to(l.uint32)
        tile_k = l.program_id(0)
        wid = amdgcn_readfirstlane(tid // 64)
        wtid = tid % 64
        shm = _native_shared_memory(self.kShmWords)
        num_valid_ids = l.load(num_valid_ids_ptr)
        route_group_limit = (
            num_valid_ids + self.kRoutesPerBlock - 1
        ) // self.kRoutesPerBlock
        route_group_begin = l.program_id(1)
        route_group_step = persistent_route_step
        route_group_end = l.where(
            route_group_step != 0, route_group_limit, route_group_begin + 1
        )
        if route_group_begin < route_group_limit:
            for route_group in range(
                route_group_begin,
                route_group_end,
                l.where(route_group_step != 0, route_group_step, 1),
            ):
                expert_id = l.load(sorted_expert_ids + route_group).to(l.uint32)
                self.RunStage1Route(
                    shm,
                    epilogue,
                    act,
                    w13,
                    sorted_token_ids,
                    scales_act,
                    scales_w13,
                    route_group,
                    route_group_limit,
                    expert_id,
                    tile_k,
                    m,
                    num_experts,
                    tid,
                    wid,
                    wtid,
                    w13_bias,
                )
                l.barrier()
