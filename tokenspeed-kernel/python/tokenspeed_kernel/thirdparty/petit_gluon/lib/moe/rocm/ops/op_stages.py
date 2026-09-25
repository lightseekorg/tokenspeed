"""Native W13/W2 stage schedules and accumulator epilogues."""

from typing import NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    _native_call,
    _native_store_ushort_component,
    _resource_content,
    _uninitialized_like,
    amdgcn_cvt_pk_bf16_f32,
    amdgcn_pk_mul_f32,
    amdgcn_s_waitcnt_barrier,
)
from lib.moe.rocm.fused_moe import ClearMat, HotLoopScheduler
from lib.moe.rocm.ops.schedule_tiles import InputShm, MxFp4Tile
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class W13State(NamedTuple):
    input: object
    w1: object
    w13_bias: object
    w1_tile: object
    w3_tile: object


class W13TileSchedule(DeviceTemplate):
    def __init__(self, TileOps):
        self._key = TileOps.cache_key
        self.TileOps, self.Config = TileOps, TileOps.Config
        self.Input, self.Weight, self.Bias = (
            self.Config.Input,
            self.Config.W13,
            self.Config.Bias,
        )
        self.kNumWarps, self.kAccumFragments = (
            TileOps.kNumWarps,
            TileOps.kAccumFragments,
        )
        self.kActivationFragments = TileOps.kActivationFragments

    @device_method
    def Construct(self, input_state, w1, bias):
        return W13State(input_state, w1, bias, (), ())

    @device_method
    def InitializeBias(self, state, bias_ptr, expert_id, tile_k):
        projection_stride = self.Bias.PackedStride(self.Config.kInterDim)
        expert_stride = 2 * projection_stride
        bias = self.Bias.Initialize(
            bias_ptr, expert_id, self.Config.kInterDim, tile_k, expert_stride
        )
        return W13State(state.input, state.w1, bias, state.w1_tile, state.w3_tile)

    @device_method
    def PrefetchInput(self, state, shm, wid, wtid, tokens, m):
        inp = self.TileOps.PrefetchInput(state.input, shm, wid, wtid, tokens, m)
        return W13State(inp, state.w1, state.w13_bias, state.w1_tile, state.w3_tile)

    @device_method
    def ReadInput(self, state, shm, wtid):
        inp, regs = self.TileOps.ReadInput(state.input, shm, wtid)
        return (
            W13State(inp, state.w1, state.w13_bias, state.w1_tile, state.w3_tile),
            regs,
        )

    @device_method
    def LoadInitial(self, state, tid, wid, wtid):
        w1_tile = self.TileOps.Load(state.w1, tid, wid, wtid)
        w3_tile = self.LoadW3Tile(state, tid, wid, wtid)
        w1 = self.Weight.AdvanceStep(state.w1, 0, self.TileOps.kKStages)
        return W13State(state.input, w1, state.w13_bias, w1_tile, w3_tile)

    @device_method
    def Matmul(self, state, gate, up, input_regs, tid, wid, wtid):
        gate = self.TileOps.Matmul(gate, state.w1_tile, input_regs, wtid)
        up = self.TileOps.Matmul(up, state.w3_tile, input_regs, wtid)
        w1_tile = self.TileOps.Load(state.w1, tid, wid, wtid)
        w3_tile = self.LoadW3Tile(state, tid, wid, wtid)
        w1 = self.Weight.AdvanceStep(state.w1, 0, self.TileOps.kKStages)
        return W13State(state.input, w1, state.w13_bias, w1_tile, w3_tile), gate, up

    @device_method
    def AddBias(self, state, gate, up, tid):
        gate = self.Bias.AddToAccumulator(state.w13_bias, gate, 0, tid)
        up = self.Bias.AddToAccumulator(
            state.w13_bias, up, self.Bias.PackedStride(self.Config.kInterDim), tid
        )
        return gate, up

    @device_method
    def LoadW3Tile(self, state, tid, wid, wtid):
        kValueOffset: l.constexpr = self.Weight.ValueProjectionOffsetBytes(
            self.Config.kDim, self.Config.kInterDim
        )
        kScaleOffset: l.constexpr = self.Weight.ScaleProjectionOffsetBytes(
            self.Config.kDim, self.Config.kInterDim
        )
        return self.TileOps.LoadProjection(
            state.w1, tid, wid, wtid, kValueOffset, kScaleOffset
        )


class W2State(NamedTuple):
    weight: object
    bias: object
    stages: object


class W2TileSchedule(DeviceTemplate):
    def __init__(self, TileOps):
        self._key = TileOps.cache_key
        self.TileOps, self.Config = TileOps, TileOps.Config
        self.Weight, self.Bias = self.Config.W2, self.Config.Stage2Bias
        self.kNumWarps, self.kAccumFragments = (
            TileOps.kNumWarps,
            TileOps.kAccumFragments,
        )
        self.kActivationFragments = TileOps.kActivationFragments
        self.kOutputPacksPerToken = TileOps.kOutputPacksPerToken

    @device_method
    def Construct(self, weight, tid):
        word = _uninitialized_like(tid, l.uint32)
        values = ((word, word, word, word),) * self.Weight.kLoadGlobal
        tile = MxFp4Tile((values, values), (word,) * (self.Weight.kLoadGlobal // 2))
        return W2State(weight, (), (tile, tile))

    @device_method
    def InitializeBias(self, state, bias_ptr, expert_id, tile_k):
        expert_stride = self.Bias.PackedStride(self.Config.kDim)
        bias_tile = tile_k if self.TileOps.kStage2BiasUsesTileK else 0
        bias = self.Bias.Initialize(
            bias_ptr, expert_id, self.Config.kDim, bias_tile, expert_stride
        )
        return W2State(state.weight, bias, state.stages)

    @device_method
    def LoadStage(self, state, stage: l.constexpr, tid, wid, wtid):
        value = self.TileOps.Load(state.weight, tid, wid, wtid)
        stages = state.stages[:stage] + (value,) + state.stages[stage + 1 :]
        weight = self.Weight.AdvanceStep(state.weight, 1, 0)
        return W2State(weight, state.bias, stages)

    @device_method
    def LoadKStage(self, state, stage: l.constexpr, tid, wid, wtid):
        value = self.TileOps.Load(state.weight, tid, wid, wtid)
        stages = state.stages[:stage] + (value,) + state.stages[stage + 1 :]
        weight = self.Weight.AdvanceStep(state.weight, 0, self.TileOps.kKStages)
        return W2State(weight, state.bias, stages)

    @device_method
    def Matmul(self, state, t, input_regs, stage: l.constexpr, wtid):
        return self.TileOps.Matmul(t, state.stages[stage], input_regs, wtid)


class OnestageFusedMoEStage1DoubleBufferOp(DeviceTemplate):
    def __init__(self, TileSchedule):
        self._key = TileSchedule.cache_key
        self.Tiles, self.Config = TileSchedule, TileSchedule.Config
        self.ActivationOp = self.Config.ActivationOp
        self.kStage, self.kGroupDim, self.kTokenBatch = (
            2,
            self.Config.kGroupDim,
            self.Config.kTokenBatch,
        )
        self.Input, self.kAccumFragments = (
            TileSchedule.Input,
            TileSchedule.kAccumFragments,
        )
        self.kShmStageWords = self.Input.kShmActWords + self.Input.kShmScaleWords
        self.kShmWords = self.kStage * self.kShmStageWords

    @device_method
    def Run(self, shm, state, tid, wid, wtid, tokens, m):
        t_gate = ClearMat(tid, self.kAccumFragments)
        t_up = ClearMat(tid, self.kAccumFragments)
        x = ()
        curr = 0
        s = InputShm(shm, shm + self.Input.kShmActWords)
        state = self.Tiles.PrefetchInput(state, s, wid, wtid, tokens, m)
        state = self.Tiles.LoadInitial(state, tid, wid, wtid)
        amdgcn_s_waitcnt_barrier(0)
        state, regs = self.Tiles.ReadInput(state, s, wtid)
        x = (regs, regs)
        for d in l.static_range(0, self.Config.kDim, 2 * self.kGroupDim):
            for curr in l.static_range(2):
                next: l.constexpr
                next = 1 - curr
                if d + curr * self.kGroupDim < self.Config.kDim:
                    HotLoopScheduler(128, 6, 0, 0, 2)
                    has_next: l.constexpr
                    has_next = d + (curr + 1) * self.kGroupDim < self.Config.kDim
                    s = InputShm(
                        shm + next * self.kShmStageWords,
                        shm + next * self.kShmStageWords + self.Input.kShmActWords,
                    )
                    if has_next:
                        state = self.Tiles.PrefetchInput(state, s, wid, wtid, tokens, m)
                    state, t_gate, t_up = self.Tiles.Matmul(
                        state, t_gate, t_up, x[curr], tid, wid, wtid
                    )
                    if has_next:
                        amdgcn_s_waitcnt_barrier(0)
                        state, regs = self.Tiles.ReadInput(state, s, wtid)
                        x = x[:next] + (regs,) + x[next + 1 :]
        t_gate, t_up = self.Tiles.AddBias(state, t_gate, t_up, tid)
        h = ()
        for i in l.static_range(self.kAccumFragments):
            h += (self.ActivationOp.Apply(t_gate[i], t_up[i]),)
        return state, h


class OnestageFusedMoEStage1SingleBufferOp(OnestageFusedMoEStage1DoubleBufferOp):
    def __init__(self, TileSchedule):
        super().__init__(TileSchedule)
        self.kStage, self.kShmWords = 1, self.kShmStageWords

    @device_method
    def Run(self, shm, state, tid, wid, wtid, tokens, m):
        t_gate = ClearMat(tid, self.kAccumFragments)
        t_up = ClearMat(tid, self.kAccumFragments)
        state = self.Tiles.LoadInitial(state, tid, wid, wtid)
        s = InputShm(shm, shm + self.Input.kShmActWords)
        for d in l.static_range(0, self.Config.kDim, self.kGroupDim):
            state = self.Tiles.PrefetchInput(state, s, wid, wtid, tokens, m)
            amdgcn_s_waitcnt_barrier(0)
            state, x = self.Tiles.ReadInput(state, s, wtid)
            state, t_gate, t_up = self.Tiles.Matmul(
                state, t_gate, t_up, x, tid, wid, wtid
            )
        t_gate, t_up = self.Tiles.AddBias(state, t_gate, t_up, tid)
        h = ()
        for i in l.static_range(self.kAccumFragments):
            h += (self.ActivationOp.Apply(t_gate[i], t_up[i]),)
        return state, h


@g.jit
def BufferAtomicWriteBf16x2(base, vo, value):
    _native_call(
        "buffer.atomic.pk.add.bf16",
        "void",
        ("v4i32", "i32", "i32"),
        (_resource_content(base), vo, value),
        False,
    )


@g.jit
def MultRouteWeights(t, rw2):
    l.static_assert(len(t) % 2 == 0)
    for i in l.static_range(2):
        rw = rw2[i]
        rw_pk = (rw, rw)
        for j in l.static_range(len(t) // 2):
            f = t[j * 2 + i]
            xy = amdgcn_pk_mul_f32(f[:2], rw_pk)
            zw = amdgcn_pk_mul_f32(f[2:], rw_pk)
            t = t[: j * 2 + i] + (xy + zw,) + t[j * 2 + i + 1 :]
    return t


@g.jit
def ToBf16Rn(m):
    return (amdgcn_cvt_pk_bf16_f32(m[0], m[1]), amdgcn_cvt_pk_bf16_f32(m[2], m[3]))


class W2AccumulatorEpilogue(DeviceTemplate):
    def __init__(self, TileSchedule):
        self._key = TileSchedule.cache_key
        self.Tiles, self.Bias = TileSchedule, TileSchedule.Bias
        self.kAccumFragments = TileSchedule.kAccumFragments

    @device_method
    def PrefetchBias(self, tiles_state, tile_col, tid):
        return self.Bias.PrefetchFragments(tiles_state.bias, tile_col, tid)

    @device_method
    def Apply(self, accum, bias, route_weights, SelectedBias: l.constexpr = None):
        if SelectedBias is None:
            accum = self.Bias.Apply(accum, bias)
        else:
            accum = SelectedBias.Apply(accum, bias)
        return MultRouteWeights(accum, route_weights)


class TwoStageStage2Epilogue(DeviceTemplate):
    def __init__(self, TileSchedule):
        self._key = TileSchedule.cache_key
        self.Config = TileSchedule.Config
        self.AccumulatorEpilogue = W2AccumulatorEpilogue(TileSchedule)
        self.kAccumFragments, self.kNumWarps = TileSchedule.kAccumFragments, 4
        self.kTokenBatch = self.Config.kStage2TokenBatch
        self.kTileRows, self.kTileCols = self.kTokenBatch * 4, self.Config.kGroupN
        self.kOutputWords = self.kTileRows * self.kTileCols // 2
        self.kShmWords = self.kOutputWords + 2 * self.kTileRows
        assert self.kTileRows == self.Config.kStage2GroupM == 32
        assert self.kTileCols == 256 and self.kAccumFragments == 8

    @device_method
    def PrefetchBias(self, tiles_state, tile_col, tid):
        return self.AccumulatorEpilogue.PrefetchBias(tiles_state, tile_col, tid)

    @device_method
    def Apply(self, accum, bias, route_weights):
        return self.AccumulatorEpilogue.Apply(accum, bias, route_weights)

    @device_method
    def StoreOutputRowOffset(self, shm, row, offset):
        if row < self.kTileRows:
            l.store(shm + self.kOutputWords + row, offset)

    @device_method
    def StoreRouteWeight(self, shm, row, weight):
        if row < self.kTileRows:
            route_weights = (shm + self.kOutputWords + self.kTileRows).to(
                l.pointer_type(l.float32, 3)
            )
            l.store(route_weights + row, weight.to(l.float32, bitcast=True))

    @device_method
    def LoadRouteWeights(self, shm, wtid):
        row = wtid % 16
        weights = (shm + self.kOutputWords + self.kTileRows).to(
            l.pointer_type(l.float32, 3)
        )
        return l.load(weights + row), l.load(weights + row + 16)

    @device_method
    def WriteShm(self, shm, accum, wid, wtid):
        q, r = wtid // 16, wtid % 16
        output = shm.to(l.pointer_type(l.uint16, 3))
        for mi in l.static_range(2):
            for ni in l.static_range(self.kAccumFragments // 2):
                fragment: l.constexpr
                fragment = 2 * ni + mi
                for component in l.static_range(4):
                    value = accum[fragment][component]
                    packed = amdgcn_cvt_pk_bf16_f32(value, value)
                    row = mi * 16 + r
                    col = wid * 64 + ni * 16 + q * 4 + component
                    _native_store_ushort_component(
                        output,
                        row * self.kTileCols + col,
                        packed.to(l.uint16),
                        component,
                    )

    @device_method
    def WriteBack(self, out, shm, tile_col, tid):
        m_lane, n_lane = tid // 32, tid % 32
        output = shm
        base = m_lane * (self.kTileCols // 2) + n_lane
        for mr in l.static_range(4):
            output_row_offset = l.load(shm + self.kOutputWords + m_lane + mr * 8)
            for nr in l.static_range(4):
                value = l.load(output + base + mr * 8 * (self.kTileCols // 2) + nr * 32)
                col = tile_col + nr * 64 + n_lane * 2
                vo = output_row_offset + col * 2
                BufferAtomicWriteBf16x2(out, vo, value)
