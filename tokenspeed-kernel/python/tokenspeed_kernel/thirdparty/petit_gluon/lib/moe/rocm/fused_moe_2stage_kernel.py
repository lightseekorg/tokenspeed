"""Native local two-stage workspace and kernel implementation."""

from lib.moe.rocm.ops.mxfp4_activation import MxFp4ActivationLayout
from lib.tal.device import DeviceTemplate
from lib.tal.host_device import host_device


class TwoStageFusedMoEWorkspace(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.kRoutesPerBlock = Config.kGroupM
        self.Layout = MxFp4ActivationLayout

    @host_device
    def ValueBytes(self, route_groups, inter_dim):
        return self.Layout.ValueBytes(route_groups * self.kRoutesPerBlock, inter_dim)

    @host_device
    def ScaleRows(self, route_groups):
        return self.Layout.PaddedScaleRows(route_groups * self.kRoutesPerBlock)

    @host_device
    def ScaleCols(self, inter_dim):
        return self.Layout.ScaleCols(inter_dim)

    @host_device
    def ScaleOffset(self, route_groups, inter_dim):
        return self.ValueBytes(route_groups, inter_dim)

    @host_device
    def Bytes(self, route_groups, inter_dim):
        return self.ScaleOffset(route_groups, inter_dim) + self.ScaleRows(
            route_groups
        ) * self.ScaleCols(inter_dim)


from typing import NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    _native_shared_memory,
    amdgcn_readfirstlane,
    amdgcn_s_setprio,
)
from lib.moe.rocm.fused_moe import ClearMat
from lib.moe.rocm.fused_moe_blockscale_fp8_kernel import FusedMoEStage1
from lib.moe.rocm.memory_ops import MakeBufferResource
from lib.moe.rocm.ops.mxfp4_activation import MxFp4ActivationQuantizer, MxFp4Stage2Input
from lib.moe.rocm.ops.op_stages import TwoStageStage2Epilogue
from lib.tal.device import device_method
from lib.tal.thread_jit import thread_device_method
from triton.experimental.gluon import language as l


class Stage1EpilogueContext(NamedTuple):
    workspace: object
    scale_base: object
    num_valid_ids: object
    m: object


class MxFp4Stage1WorkspaceEpilogue(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.Config, self.Quantizer = Config, MxFp4ActivationQuantizer(Config)
        self.kInputFragments = self.Quantizer.kInputFragments
        self.kShmWords = self.Quantizer.kQuantizeShmWords
        self.Workspace = TwoStageFusedMoEWorkspace(Config)

    @device_method
    def Run(
        self,
        context,
        shm,
        h,
        tokens,
        row_metadata,
        route_base,
        route_group,
        expert_id,
        tile_n,
        tid,
        wid,
        wtid,
    ):
        quant_shm = shm.to(l.pointer_type(l.float32, 3))
        self.Quantizer.StoreAccumulator(quant_shm, h, wid, wtid)
        l.barrier()
        route_in_slice, col_lane = tid // 32, tid % 32
        metadata = row_metadata
        for route_slice in l.static_range(self.Config.kGroupM // 8):
            row = route_slice * 8 + route_in_slice
            sorted_row = route_group * self.Config.kGroupM + row
            self.StoreRoute(
                context,
                quant_shm,
                l.load(metadata + row),
                row,
                sorted_row,
                tile_n,
                col_lane,
            )

    @device_method
    def StoreRoute(self, context, shm, fused, row, sorted_row, tile_n, col_lane):
        token, slot = fused & 0x00FFFFFF, fused >> 24
        valid = (
            (sorted_row < context.num_valid_ids)
            & (token < context.m)
            & (slot < self.Config.kTopK)
        )
        value_row = token * self.Config.kTopK + slot
        for col_segment in l.static_range(self.Config.kStage1GroupN // 128):
            quant_col_lane = col_segment * 32 + col_lane
            quantized = self.Quantizer.Quantize(shm, row, quant_col_lane)
            self.Quantizer.Store(
                context.workspace,
                0,
                context.scale_base,
                value_row,
                sorted_row,
                tile_n,
                quant_col_lane,
                self.Config.kInterDim,
                self.Workspace.ScaleCols(self.Config.kInterDim),
                quantized,
                valid=valid,
            )


@g.jit
def TwoStageFusedMoEStage1Compute(
    workspace,
    act,
    w13,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    scales_act,
    scales_w13,
    m,
    num_experts,
    max_num_m_blocks,
    w13_bias,
    Kernel: l.constexpr,
    Workspace: l.constexpr,
    kPersistent: l.constexpr,
):
    value_bytes = Workspace.ValueBytes(max_num_m_blocks, Kernel.Config.kInterDim)
    workspace_bytes = Workspace.Bytes(max_num_m_blocks, Kernel.Config.kInterDim)
    epilogue = Stage1EpilogueContext(
        MakeBufferResource(workspace, workspace_bytes),
        value_bytes,
        l.load(num_valid_ids),
        m,
    )
    persistent_route_step = l.num_programs(1) if kPersistent else 0
    Kernel.Compute(
        act,
        w13,
        sorted_token_ids,
        sorted_expert_ids,
        scales_act,
        scales_w13,
        num_valid_ids,
        m,
        num_experts,
        persistent_route_step,
        w13_bias,
        epilogue,
    )


class TwoStageFusedMoEStage2(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.ConfigType = self.Config = Config
        self.kGroupDim, self.kNumWarps, self.kThreads = (
            Config.kGroupDim,
            Config.kNumWarps,
            Config.kNumWarps * 64,
        )
        self.kTokenBatch = Config.kStage2TokenBatch
        self.kRoutesPerBlock = self.kTokenBatch * self.kNumWarps
        self.W2Weights, self.Bias, self.Stage2Tiles = (
            Config.W2Weights,
            Config.Stage2Bias,
            Config.Stage2Tiles,
        )
        self.Stage2Input = MxFp4Stage2Input(Config)
        self.Stage2EpilogueOp = TwoStageStage2Epilogue(self.Stage2Tiles)
        self.Workspace = TwoStageFusedMoEWorkspace(Config)
        self.kPersistentWorkers, self.kInterDim = 256, Config.kInterDim
        self.kK256Tiles = self.kInterDim // self.kGroupDim
        self.kShmWords = self.Stage2EpilogueOp.kShmWords
        assert self.Stage2Tiles.kActivationFragments == 4
        assert self.kInterDim % self.kGroupDim == 0 and self.kK256Tiles > 0
        assert self.Stage2Input.kInputShmWords == self.Stage2EpilogueOp.kOutputWords

    @thread_device_method
    def Stage2(
        self,
        w2_weights,
        sorted_token_br,
        out,
        intermediate_ptr,
        route_group,
        max_route_groups,
        topk,
        num_tokens,
        sorted_weights,
        route_base,
        tile_n,
        tid,
        wid,
        wtid,
        expert_id,
        w2_bias,
        shm,
    ):
        tiles = self.Stage2Tiles.Construct(w2_weights.w2_, tid)
        tiles = self.Stage2Tiles.InitializeBias(tiles, w2_bias, expert_id, tile_n)
        accum = ClearMat(tid, self.Stage2Tiles.kAccumFragments)
        values_bytes = self.Workspace.ValueBytes(max_route_groups, self.kInterDim)
        workspace_size = self.Workspace.Bytes(max_route_groups, self.kInterDim)
        workspace = MakeBufferResource(intermediate_ptr, workspace_size)
        row = tid // self.Stage2Input.kVectorsPerRow
        vector = tid % self.Stage2Input.kVectorsPerRow
        packed_id = BufferResource.LoadU32(
            sorted_token_br, row * 4, 0, BufferResource.kNone
        )
        token, slot = packed_id & 0x00FFFFFF, packed_id >> 24
        valid = (token < num_tokens) & (slot < topk)
        value_voffset = l.where(
            valid, (token * topk + slot) * (self.kInterDim // 2) + vector * 16, 0
        )
        scale_voffset = route_group * self.kRoutesPerBlock * self.Stage2Input.kScaleCols
        output_bytes = num_tokens * self.Config.kDim * 2
        route_weights = MakeBufferResource(
            sorted_weights + route_base, self.kRoutesPerBlock * 4
        )
        if tid < self.kRoutesPerBlock:
            packed_token = BufferResource.LoadU32(
                sorted_token_br, 0, tid * 4, BufferResource.kNone
            )
            token, slot = packed_token & 0x00FFFFFF, packed_token >> 24
            output_row_offset = l.where(
                (token < num_tokens) & (slot < topk),
                token * self.Config.kDim * 2,
                output_bytes,
            )
            route_weight = BufferResource.LoadU32(
                route_weights, tid * 4, 0, BufferResource.kNone
            )
            self.Stage2EpilogueOp.StoreOutputRowOffset(shm, tid, output_row_offset)
            self.Stage2EpilogueOp.StoreRouteWeight(shm, tid, route_weight)
        l.barrier()
        tile_col = tile_n * self.kGroupDim
        for tile_k in l.static_range(self.kK256Tiles):
            stage: l.constexpr
            stage = tile_k & 1
            input_global = self.Stage2Input.LoadTile(
                workspace,
                value_voffset,
                0,
                scale_voffset,
                values_bytes,
                tile_k,
                valid,
                wtid,
            )
            self.Stage2Input.StoreLds(shm, input_global.value, stage, tid)
            tiles = self.Stage2Tiles.LoadKStage(tiles, stage, tid, wid, wtid)
            l.barrier()
            input_regs = self.Stage2Input.ReadLds(shm, stage, input_global.scale, wtid)
            accum = self.Stage2Tiles.Matmul(tiles, accum, input_regs, stage, wtid)
            if tile_k + 1 == self.kK256Tiles:
                epilogue_route_weights = self.Stage2EpilogueOp.LoadRouteWeights(
                    shm, wtid
                )
                epilogue_bias = self.Stage2EpilogueOp.PrefetchBias(tiles, tile_col, tid)
            else:
                l.barrier()
        l.barrier()
        amdgcn_s_setprio(0)
        accum = self.Stage2EpilogueOp.Apply(
            accum, epilogue_bias, epilogue_route_weights
        )
        self.Stage2EpilogueOp.WriteShm(shm, accum, wid, wtid)
        l.barrier()
        output = MakeBufferResource(out, output_bytes)
        self.Stage2EpilogueOp.WriteBack(output, shm, tile_col, tid)

    @device_method
    def Compute(
        self,
        out,
        intermediate_ptr,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        scales_w2,
        num_valid_ids_ptr,
        topk,
        num_experts,
        max_route_groups,
        w2_bias,
    ):
        tid = l.arange(
            0, self.kThreads, layout=l.BlockedLayout([1], [64], [self.kNumWarps], [0])
        ).to(l.uint32)
        tile_n, persistent_worker = l.program_id(0), l.program_id(1)
        wid, wtid = amdgcn_readfirstlane(tid // 64), tid % 64
        shm = _native_shared_memory(self.kShmWords)
        num_valid_ids, num_tokens = l.load(num_valid_ids_ptr), l.load(
            num_valid_ids_ptr + 1
        )
        route_group_limit = (
            num_valid_ids + self.kRoutesPerBlock - 1
        ) // self.kRoutesPerBlock
        route_groups_per_worker = route_group_limit // self.kPersistentWorkers
        route_group_remainder = route_group_limit % self.kPersistentWorkers
        worker_route_groups = route_groups_per_worker + (
            persistent_worker < route_group_remainder
        ).to(l.uint32)
        route_group_begin = persistent_worker * route_groups_per_worker + l.minimum(
            persistent_worker, route_group_remainder
        )
        for worker_tile in range(worker_route_groups):
            route_group = route_group_begin + worker_tile
            route_base = route_group * self.kRoutesPerBlock
            if (route_group < route_group_limit) & (route_base < num_valid_ids):
                expert_id = l.load(
                    sorted_expert_ids
                    + route_group // self.Config.kStage1ToStage2GroupRatio
                ).to(l.uint32)
                valid_expert = (num_experts == 0) | (expert_id < num_experts)
                if valid_expert:
                    w2_weights = self.W2Weights.Initialize(
                        w2, scales_w2, expert_id, tile_n, 0
                    )
                    sorted_token_br = MakeBufferResource(
                        sorted_token_ids + route_base, self.kRoutesPerBlock * 4
                    )
                    self.Stage2(
                        w2_weights,
                        sorted_token_br,
                        out,
                        intermediate_ptr,
                        route_group,
                        max_route_groups,
                        topk,
                        num_tokens,
                        sorted_weights,
                        route_base,
                        tile_n,
                        tid,
                        wid,
                        wtid,
                        expert_id,
                        w2_bias,
                        shm,
                    )
            l.barrier()


@g.jit
def TwoStageFusedMoEStage2Compute(
    out,
    intermediate,
    w2,
    sorted_token_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    scales_w2,
    num_experts,
    max_route_groups,
    w2_bias,
    Kernel: l.constexpr,
):
    Kernel.Compute(
        out,
        intermediate,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        scales_w2,
        num_valid_ids,
        topk,
        num_experts,
        max_route_groups,
        w2_bias,
    )
