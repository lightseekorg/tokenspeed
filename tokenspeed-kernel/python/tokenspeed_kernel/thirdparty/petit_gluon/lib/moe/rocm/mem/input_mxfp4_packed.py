"""Native token-shuffle rows: E2M1 values followed by row-major E8M0 scales."""

from typing import NamedTuple

from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    amdgcn_mov_dpp,
    amdgcn_perm_b32,
    amdgcn_s_waitcnt_barrier,
    amdgcn_thread_id,
)
from lib.moe.rocm.memory_ops import _load_vector4, _register_array_get
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class PackedInputState(NamedTuple):
    workspace_: object
    activation_offset_: object
    m_: object
    values_offset_vec_: object
    scale_tile_: object


class MxFp4InputPacked(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.kDim, self.kHiddenSize = Config.kDim, Config.kHiddenSize
        self.kTokenBatch, self.kNumWarps = Config.kTokenBatch, Config.kNumWarps
        self.kWarpsM, self.kWarpsN = Config.kStage1WarpsM, Config.kStage1WarpsN
        self.kGroupM, self.kGroupDim = (
            self.kTokenBatch * self.kNumWarps,
            Config.kGroupDim,
        )
        self.kThreads, self.kGroupK = self.kNumWarps * 64, self.kGroupDim
        self.kK128Tiles, self.kK32PerTile = self.kGroupK // 128, 4
        self.kRowVecsPerTile = self.kK128Tiles * self.kK32PerTile
        self.kAsyncVecsPerWarp = self.kTokenBatch * self.kRowVecsPerTile
        self.kLoadIterations = (self.kAsyncVecsPerWarp + 63) // 64
        self.kWaveM64 = getattr(Config, "kStage1WaveM64", False)
        self.kMRepeats = 4 if self.kWaveM64 else 2
        self.kActivationFragments, self.kScaleBlockSize = (
            self.kMRepeats * self.kK128Tiles,
            32,
        )
        self.kValueBytes, self.kScaleBytes = (
            self.kHiddenSize // 2,
            self.kHiddenSize // 32,
        )
        self.kRowStride = (self.kValueBytes + self.kScaleBytes + 15) // 16 * 16
        self.kPaddedScaleBytes = self.kRowStride - self.kValueBytes
        self.kScaleVectorsPerRow, self.kScaleWordsPerRow = (
            self.kPaddedScaleBytes // 16,
            self.kPaddedScaleBytes // 4,
        )
        self.kScaleTiles = Config.kComputeHiddenSize // self.kGroupDim
        self.kScaleWords, self.kScaleStages = self.kScaleTiles * 64 * self.kWarpsM, 2
        self.kScaleWordsPerStage = self.kScaleWords // self.kScaleStages
        self.kScaleVectors = self.kScaleWords // 4
        self.kScaleVectorsPerStage = self.kScaleVectors // self.kScaleStages
        self.kWarpsPerScaleStage = self.kNumWarps // self.kScaleStages
        self.kPayloadLoadAux = (
            BufferResource.kSC1Bit if Config.kNumRanks > 1 else BufferResource.kNone
        )
        self.kShmActWords, self.kShmScaleWords = (
            self.kGroupM * self.kRowVecsPerTile * 4,
            self.kScaleWordsPerStage,
        )
        self.kShmStageWords = self.kShmActWords + self.kShmScaleWords
        assert (
            self.kNumWarps == 4 or (self.kNumWarps == 8 and self.kWaveM64)
        ) and self.kTokenBatch in (8, 16)
        assert self.kGroupDim == 256 and self.kLoadIterations in (1, 2)
        assert self.kWaveM64 or self.kWarpsM * self.kWarpsN == self.kNumWarps
        assert (
            self.kGroupM == 32 * self.kWarpsM
            and Config.kComputeHiddenSize % self.kGroupDim == 0
        )
        assert (
            self.kScaleTiles % self.kScaleStages == 0
            and self.kNumWarps % self.kScaleStages == 0
        )
        assert self.kGroupM * self.kPaddedScaleBytes == self.kScaleWords * 4
        assert self.kPaddedScaleBytes == self.kScaleTiles * 8

    @device_method
    def Initialize(self, workspace, rank, pool_row, m, Workspace: l.constexpr):
        return PackedInputState(
            workspace,
            Workspace.L1TokenBufferOffset(rank, pool_row),
            m,
            l.full((), 0, l.uint32),
            l.full((), 0, l.uint32),
        )

    @device_method
    def FetchAsync(self, state, shm_x, wid, wtid, tokens):
        for load in l.static_range(self.kLoadIterations):
            linear = load * 64 + wtid
            token_idx = linear // self.kRowVecsPerTile
            row_vec = linear - token_idx * self.kRowVecsPerTile
            source_row_vec = row_vec ^ (token_idx & (self.kRowVecsPerTile - 1))
            dst_idx = wid * self.kAsyncVecsPerWarp + load * 64
            row = _register_array_get(tokens, token_idx)
            first_element = (
                state.values_offset_vec_ + source_row_vec
            ) * self.kScaleBlockSize
            actual = (
                state.activation_offset_
                + row * self.kRowStride
                + (state.values_offset_vec_ + source_row_vec) * 16
            )
            offset = l.where(
                (row < state.m_) & (first_element < self.kHiddenSize),
                actual,
                0xFFFFFFFF,
            )
            BufferResource.LoadLds(
                state.workspace_,
                shm_x + dst_idx * 4,
                offset,
                0,
                self.kPayloadLoadAux,
                16,
                0,
                predicate=linear < self.kAsyncVecsPerWarp,
            )
        return PackedInputState(
            state.workspace_,
            state.activation_offset_,
            state.m_,
            state.values_offset_vec_ + self.kGroupDim // self.kScaleBlockSize,
            state.scale_tile_,
        )

    @device_method
    def FetchScaleAsync(self, state, shm_scale, wid, wtid, tokens, m):
        pass

    @device_method
    def PrepareScales(self, state, shm, wid, wtid, kStages: l.constexpr):
        l.static_assert(kStages == self.kScaleStages)
        self.LoadScalesAsync(state, shm, wid, wtid)
        amdgcn_s_waitcnt_barrier(0)
        self.RepackScales(shm, wid * 64 + wtid)

    @device_method
    def FetchToRegs(self, state, shm_x, wtid):
        wid = amdgcn_thread_id(wtid) // 64
        row, vector = wtid & 15, wtid // 16
        wave_m = 0 if self.kWaveM64 else wid // self.kWarpsN
        row_base = (wave_m * 32 + row) * self.kRowVecsPerTile
        regs = ()
        for m16 in l.static_range(self.kMRepeats):
            row16 = shm_x + (row_base + m16 * 16 * self.kRowVecsPerTile) * 4
            regs += (
                _load_vector4(
                    row16 + (vector ^ (row & (self.kRowVecsPerTile - 1))) * 4
                ),
                _load_vector4(
                    row16
                    + ((vector + self.kK32PerTile) ^ (row & (self.kRowVecsPerTile - 1)))
                    * 4
                ),
            )
        return regs

    @device_method
    def FetchScaleToReg(self, state, shm_scale, wtid, wave_m=None):
        if wave_m is None:
            wid = amdgcn_thread_id(wtid) // 64
            wave_m = wid // self.kWarpsN
        return l.load(
            shm_scale
            + (state.scale_tile_ // self.kScaleStages * self.kWarpsM + wave_m) * 64
            + wtid
        )

    @device_method
    def AdvanceScaleStep(self, state):
        return PackedInputState(
            state.workspace_,
            state.activation_offset_,
            state.m_,
            state.values_offset_vec_,
            state.scale_tile_ + 1,
        )

    @device_method
    def LoadScalesAsync(self, state, shm, wid, wtid):
        stage, wave_in_stage = (
            wid // self.kWarpsPerScaleStage,
            wid % self.kWarpsPerScaleStage,
        )
        kVectorsPerIteration: l.constexpr = self.kWarpsPerScaleStage * 64
        kLoadIterations: l.constexpr = (
            self.kScaleVectorsPerStage + kVectorsPerIteration - 1
        ) // kVectorsPerIteration
        scales = BufferResource.WithRange(
            state.workspace_, state.activation_offset_ + self.kGroupM * self.kRowStride
        )
        for load in l.static_range(kLoadIterations):
            local_vector = wave_in_stage * 64 + wtid + load * kVectorsPerIteration
            stage_word = l.where(
                local_vector < self.kScaleVectorsPerStage, local_vector * 4, 0
            )
            lds = shm + stage * self.kShmStageWords + self.kShmActWords + stage_word
            vector = stage * self.kScaleVectorsPerStage + local_vector
            row = vector // self.kScaleVectorsPerRow
            row_vector = vector - row * self.kScaleVectorsPerRow
            src = l.where(
                local_vector < self.kScaleVectorsPerStage,
                state.activation_offset_
                + self.kValueBytes
                + row * self.kRowStride
                + row_vector * 16,
                0xFFFFFFFF,
            )
            BufferResource.LoadLds(scales, lds, src, 0, self.kPayloadLoadAux, 16, 0)

    @device_method
    def RepackScales(self, shm, tid):
        kScaleTasks: l.constexpr = self.kScaleWords // 4
        kNumQuads: l.constexpr = self.kThreads // 4
        kTasksPerThread: l.constexpr = (kScaleTasks + kNumQuads - 1) // kNumQuads
        quad, quad_lane = tid // 4, tid % 4
        packed = ()
        for i in l.static_range(kTasksPerThread):
            task = (quad + i * kNumQuads) % kScaleTasks
            block = task // 16
            tile, wave_m, row16 = block // self.kWarpsM, block % self.kWarpsM, task % 16
            row = wave_m * 32 + row16 + 16 * (quad_lane & 1)
            half = quad_lane >> 1
            raw_word = row * self.kScaleWordsPerRow + tile * 2 + half
            raw_stage = raw_word // self.kScaleWordsPerStage
            raw = l.load(
                shm
                + raw_stage * self.kShmStageWords
                + self.kShmActWords
                + raw_word
                - raw_stage * self.kScaleWordsPerStage
            )
            v0 = amdgcn_mov_dpp(raw, 0x00, 0xF, 0xF, False)
            v1 = amdgcn_mov_dpp(raw, 0x55, 0xF, 0xF, False)
            v2 = amdgcn_mov_dpp(raw, 0xAA, 0xF, 0xF, False)
            v3 = amdgcn_mov_dpp(raw, 0xFF, 0xF, 0xF, False)
            select_pair = 0x0C0C0400 + quad_lane * 0x00000101
            pair01 = amdgcn_perm_b32(v1, v0, select_pair)
            pair23 = amdgcn_perm_b32(v3, v2, select_pair)
            packed += (amdgcn_perm_b32(pair23, pair01, 0x05040100),)
        l.barrier()
        for i in l.static_range(kTasksPerThread):
            task = (quad + i * kNumQuads) % kScaleTasks
            block = task // 16
            tile, wave_m, row16 = block // self.kWarpsM, block % self.kWarpsM, task % 16
            stage = tile % self.kScaleStages
            stage_tile = tile // self.kScaleStages * self.kWarpsM + wave_m
            l.store(
                shm
                + stage * self.kShmStageWords
                + self.kShmActWords
                + stage_tile * 64
                + quad_lane * 16
                + row16,
                packed[i],
            )
        l.barrier()
