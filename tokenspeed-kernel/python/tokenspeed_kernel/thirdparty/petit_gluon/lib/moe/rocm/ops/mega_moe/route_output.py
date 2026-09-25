"""Native destination route writeback and source-rank FP32 combine."""

from typing import NamedTuple

from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    amdgcn_cvt_pk_bf16_f32,
    amdgcn_pk_add_f32,
    amdgcn_readfirstlane,
)
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.moe.rocm.memory_ops import _store_vector4
from lib.moe.rocm.ops.op_stages import TwoStageStage2Epilogue
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class Context(NamedTuple):
    workspace: object
    pool_base: object
    work_m: object


class MegaMoETwoStage2Epilogue(TwoStageStage2Epilogue):
    def __init__(self, TileSchedule):
        super().__init__(TileSchedule)
        self.Workspace = MegaMoEWorkspace(self.Config)
        self.kPayloadLoadAux = (
            BufferResource.kSC1Bit
            if self.Config.kNumRanks > 1
            else BufferResource.kNone
        )

    @device_method
    def LoadRouteWeights(self, context, wtid):
        offset = self.Workspace.L1TokenWeightsOffset(
            context.workspace.rank_id_, context.pool_base
        )
        lane = wtid % 16
        packed = (
            BufferResource.LoadU32(
                context.workspace.br_, lane * 4, offset, self.kPayloadLoadAux
            ),
            BufferResource.LoadU32(
                context.workspace.br_, (lane + 16) * 4, offset, self.kPayloadLoadAux
            ),
        )
        return packed[0].to(l.float32, bitcast=True), packed[1].to(
            l.float32, bitcast=True
        )

    @device_method
    def WriteBack(self, context, shm, tile_col, wid, wtid):
        output = shm.to(l.pointer_type(l.uint32, 3))
        row_bytes: l.constexpr = self.Config.kHiddenSize * 2
        for row_group in l.static_range(self.kTileRows // self.kNumWarps):
            row = wid + row_group * self.kNumWarps
            valid = row < context.work_m
            metadata = (l.full((), 0, l.uint32), l.full((), 0, l.uint32))
            if valid:
                metadata = BufferResource.LoadU64(
                    context.workspace.br_,
                    0,
                    self.Workspace.TokenMetadataOffset(
                        context.workspace.rank_id_, context.pool_base + row
                    ),
                    self.kPayloadLoadAux,
                )
            src_rank = amdgcn_readfirstlane(metadata[1])
            token_topk_idx = amdgcn_readfirstlane(metadata[0])
            output_row = BufferResource.WithOffset(
                context.workspace.br_,
                self.Workspace.RouteOutputBufferOffset(src_rank)
                + token_topk_idx * row_bytes,
            )
            output_row = BufferResource.WithRange(
                output_row, l.where(valid, row_bytes, 0)
            )
            for col_half in l.static_range(2):
                pair_col = col_half * 64 + wtid
                value = l.load(output + row * (self.kTileCols // 2) + pair_col)
                col = tile_col + pair_col * 2
                BufferResource.StoreU32(
                    output_row, col * 2, 0, value, BufferResource.kNTBit
                )


class SourceRouteReducer(DeviceTemplate):
    def __init__(self, Config, kGridBlocks=None, kBlockThreads=None):
        self._key = (Config.cache_key, kGridBlocks, kBlockThreads)
        self.Workspace = MegaMoEWorkspace(Config)
        self.kNumRanks = Config.kNumRanks
        self.kNumSMs = Config.kNumSMs if kGridBlocks is None else kGridBlocks
        self.kThreads = Config.kThreads if kBlockThreads is None else kBlockThreads
        self.kTopK, self.kHiddenSize, self.kComputeHiddenSize = (
            Config.kTopK,
            Config.kHiddenSize,
            Config.kComputeHiddenSize,
        )
        self.kElementsPerVec, self.kVecCols = 8, self.kHiddenSize // 8
        assert (
            self.kThreads % 64 == 0
            and self.kHiddenSize % 8 == 0
            and self.kComputeHiddenSize >= self.kHiddenSize
        )

    @device_method
    def Run(self, workspace, output, num_tokens, output_row_stride, sm_id, wid, wtid):
        num_tokens = (l.full((), 0, l.uint32) + num_tokens).to(l.uint32)
        output_row_stride = (l.full((), 0, l.uint32) + output_row_stride).to(l.uint32)
        kWarpsPerBlock: l.constexpr = self.kThreads // 64
        kTotalWaves: l.constexpr = self.kNumSMs * kWarpsPerBlock
        kPackedElements: l.constexpr = self.kElementsPerVec // 2
        if num_tokens == 0:
            return
        if self.kNumRanks > 1:
            global_wave = sm_id * kWarpsPerBlock + wid
            waves_per_token = (kTotalWaves + num_tokens - 1) // num_tokens
            vecs_per_wave = (self.kVecCols + waves_per_token - 1) // waves_per_token
        else:
            global_wave = l.where(
                num_tokens == 8,
                wid * self.kNumSMs + sm_id,
                sm_id * kWarpsPerBlock + wid,
            )
            waves_per_token = l.full((), (self.kVecCols + 63) // 64, l.uint32)
            vecs_per_wave = l.full((), 64, l.uint32)
        total_wave_tasks = num_tokens * waves_per_token
        owner = self.Workspace.RouteOutputBufferOffset(workspace.rank_id_)
        for wave_task in range(global_wave, total_wave_tasks, kTotalWaves):
            token = wave_task // waves_per_token
            wave_in_token = wave_task % waves_per_token
            vec_in_wave = wtid
            while (
                vec_in_wave < vecs_per_wave
                and wave_in_token * vecs_per_wave + vec_in_wave < self.kVecCols
            ):
                vec_col = wave_in_token * vecs_per_wave + vec_in_wave
                col_offset = vec_col * 16
                route_row_offset = owner + token * self.kTopK * self.kHiddenSize * 2
                route_values = ()
                for topk in l.static_range(self.kTopK):
                    row_offset = amdgcn_readfirstlane(
                        route_row_offset + topk * self.kHiddenSize * 2
                    )
                    route_values += (
                        BufferResource.Load(
                            workspace.br_,
                            col_offset,
                            row_offset,
                            BufferResource.kSC1Bit | BufferResource.kNTBit,
                        ),
                    )
                zero = l.full((), 0.0, l.float32)
                accum = ((zero, zero),) * kPackedElements
                for topk in l.static_range(self.kTopK):
                    for pair in l.static_range(kPackedElements):
                        word = route_values[topk][pair]
                        bf16 = (
                            (word << 16).to(l.float32, bitcast=True),
                            (word & 0xFFFF0000).to(l.float32, bitcast=True),
                        )
                        value = amdgcn_pk_add_f32(accum[pair], bf16)
                        accum = accum[:pair] + (value,) + accum[pair + 1 :]
                packed_output = ()
                for pair in l.static_range(kPackedElements):
                    packed_output += (
                        amdgcn_cvt_pk_bf16_f32(accum[pair][0], accum[pair][1]),
                    )
                _store_vector4(
                    output.to(l.pointer_type(l.uint32))
                    + (token * (output_row_stride // self.kElementsPerVec) + vec_col)
                    * 4,
                    packed_output,
                )
                vec_in_wave += 64
