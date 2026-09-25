"""Native routed MXFP4 input and XOR shared-memory layout."""

from typing import NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import BufferResource, amdgcn_thread_id
from lib.moe.rocm.memory_ops import (
    MakeBufferResource,
    _load_vector4,
    _register_array_get,
)
from lib.tal.device import DeviceTemplate, device_method
from lib.tal.host_device import host_device
from lib.tal.tensor.layout import Layout, Shape, Stride, make_coord
from triton.experimental.gluon import language as l


class InputState(NamedTuple):
    values_: object
    scales_: object
    route_group_: object
    values_offset_vec_: object
    scales_offset_vec_: object


class MxFp4Input(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.kDim, self.kTokenBatch, self.kNumWarps = (
            Config.kDim,
            Config.kTokenBatch,
            Config.kNumWarps,
        )
        self.kWarpsM, self.kWarpsN = Config.kStage1WarpsM, Config.kStage1WarpsN
        self.kGroupDim, self.kThreads = Config.kGroupDim, Config.kNumWarps * 64
        self.kGroupK = self.kGroupDim
        self.kK128Tiles, self.kK256Tiles, self.kK32PerTile = (
            self.kGroupK // 128,
            self.kGroupK // 256,
            4,
        )
        self.kRowVecsPerTile, self.kMmaRows = self.kK128Tiles * 4, 16
        self.kAsyncVecsPerWarp = self.kTokenBatch * self.kRowVecsPerTile
        self.kLoadIterations = (self.kAsyncVecsPerWarp + 63) // 64
        self.kActivationFragments, self.kScaleFragments = (
            2 * self.kK128Tiles,
            self.kK256Tiles,
        )
        self.kScaleBlockSize = 32
        self.kRowVecs = self.kDim // self.kScaleBlockSize
        self.kScaleBlocksPerRouteGroup, self.kScaleWordsPerM32 = self.kDim // 256, 64
        self.kShmActWords = self.kLoadIterations * self.kThreads * 4
        self.kShmScaleWords = (
            self.kScaleFragments * self.kScaleWordsPerM32 * self.kWarpsM
        )
        self.RowVecLayout = Layout(
            Shape(self.kK128Tiles, self.kK32PerTile), Stride(self.kK32PerTile, 1)
        )
        assert self.kGroupK % 256 == 0 and self.kNumWarps == 4
        assert self.kLoadIterations in (1, 2) and self.kScaleFragments == 1
        assert self.kWarpsM * self.kWarpsN == self.kNumWarps
        assert self.kTokenBatch * self.kNumWarps == 32 * self.kWarpsM

    @device_method
    def MakeRowVecLayout(self):
        return self.RowVecLayout

    @host_device
    def SwizzledVector(self, row, vector):
        return vector ^ (row & (self.kRowVecsPerTile - 1))

    @device_method
    def Initialize(
        self, value_ptr, scale_ptr, dim, m, n, route_group, route_group_limit
    ):
        return InputState(
            MakeBufferResource(value_ptr, m * self.kDim // 2),
            MakeBufferResource(scale_ptr, route_group_limit * self.kWarpsM * self.kDim),
            route_group,
            l.full((), 0, l.uint32),
            l.full((), 0, l.uint32),
        )

    @device_method
    def FetchAsync(self, state, shm_x, wid, wtid, tokens):
        row_vec_layout: l.constexpr = self.RowVecLayout
        offset = state.values_offset_vec_
        values_offset_vec = offset + self.kGroupK // self.kScaleBlockSize
        for load in l.static_range(self.kLoadIterations):
            linear = load * 64 + wtid
            token_idx = linear // self.kRowVecsPerTile
            row_vec = linear - token_idx * self.kRowVecsPerTile
            source_row_vec = self.SwizzledVector(token_idx, row_vec)
            k128 = source_row_vec // self.kK32PerTile
            k32 = source_row_vec - k128 * self.kK32PerTile
            dst_idx = wid * self.kAsyncVecsPerWarp + load * 64
            lds_ptr = shm_x + dst_idx * 4
            byte_offset = (
                _register_array_get(tokens, token_idx) * self.kRowVecs
                + offset
                + row_vec_layout(make_coord(k128, k32))
            ) * 16
            # Both supported geometries have complete wave loads.
            l.static_assert(self.kAsyncVecsPerWarp == self.kLoadIterations * 64)
            BufferResource.LoadLds(
                state.values_, lds_ptr, byte_offset, 0, BufferResource.kNone, 16, 0
            )
        return InputState(
            state.values_,
            state.scales_,
            state.route_group_,
            values_offset_vec,
            state.scales_offset_vec_,
        )

    @device_method
    def FetchScaleAsync(self, state, shm_scale, wid, wtid, tokens, m):
        offset = state.scales_offset_vec_
        m32_group = state.route_group_ * self.kWarpsM + wid
        src_word = (m32_group * self.kScaleBlocksPerRouteGroup + offset) * 64 + wtid
        lds_ptr = shm_scale + wid * self.kScaleWordsPerM32
        BufferResource.LoadLds(
            state.scales_,
            lds_ptr,
            src_word * 4,
            0,
            BufferResource.kNone,
            4,
            0,
            predicate=wid < self.kWarpsM,
        )

    @device_method
    def FetchToRegs(self, state, shm_x, wtid):
        wid = amdgcn_thread_id(wtid) // 64
        wave_m = wid // self.kWarpsN
        row = wtid & (self.kMmaRows - 1)
        vector = wtid // self.kMmaRows
        row_base = (wave_m * 32 + row) * self.kRowVecsPerTile
        k0 = shm_x + (row_base + self.SwizzledVector(row, vector)) * 4
        k1 = (
            shm_x + (row_base + self.SwizzledVector(row, vector + self.kK32PerTile)) * 4
        )
        kNextRow: l.constexpr = self.kMmaRows * self.kRowVecsPerTile
        return (
            _load_vector4(k0),
            _load_vector4(k1),
            _load_vector4(k0 + kNextRow * 4),
            _load_vector4(k1 + kNextRow * 4),
        )

    @device_method
    def FetchScaleToReg(self, state, shm_scale, wtid):
        wid = amdgcn_thread_id(wtid) // 64
        wave_m = wid // self.kWarpsN
        return l.load(shm_scale + wave_m * self.kScaleWordsPerM32 + wtid)

    @device_method
    def AdvanceScaleStep(self, state):
        return InputState(
            state.values_,
            state.scales_,
            state.route_group_,
            state.values_offset_vec_,
            state.scales_offset_vec_ + self.kScaleFragments,
        )
