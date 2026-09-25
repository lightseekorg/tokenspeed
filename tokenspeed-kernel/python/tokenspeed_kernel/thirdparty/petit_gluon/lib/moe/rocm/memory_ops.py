# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native MoE memory layouts. Mutable C++ state is returned as SSA tuples."""

from enum import IntEnum
from typing import NamedTuple

import triton
import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    BufferResourceFields,
    _native_load_vector4,
    _native_store_vector4,
    _resource_content,
    _uninitialized_like,
    kWarpSize,
    llvm_amdgcn_raw_buffer_load_v4i32,
)
from lib.tal.device import DeviceTemplate, device_method
from lib.tal.host_device import host_device
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language._core import builtin


class MoeArchitecture(IntEnum):
    kCdna3 = 0
    kCdna4 = 1


class WeightLoadPolicySelector:
    def __init__(self, architecture):
        self.kAux = (
            BufferResource.kNTBit
            if architecture == MoeArchitecture.kCdna4
            else BufferResource.kNone
        )


TargetWeightLoadPolicy = WeightLoadPolicySelector(MoeArchitecture.kCdna4)


@g.jit
def _uninitialized_uint4_array(tid, size: l.constexpr):
    word = _uninitialized_like(tid, l.uint32)
    result = ()
    for i in l.static_range(size):
        result += ((word, word, word, word),)
    return result


@g.jit
def _uint4_words(v):
    even, odd = l.split(v.reshape([256, 2, 2]))
    x, z = l.split(even)
    y, w = l.split(odd)
    L: l.constexpr = l.BlockedLayout([1], [64], [4], [0])
    return (
        l.convert_layout(x, L),
        l.convert_layout(y, L),
        l.convert_layout(z, L),
        l.convert_layout(w, L),
    )


@g.jit
def _load_uint2(ptr):
    # A native LDS uint2 access: load two adjacent uint32 register words.
    V: l.constexpr = l.BlockedLayout([1, 2], [64, 1], [4, 1], [1, 0])
    word = l.arange(0, 2, layout=l.SliceLayout(0, V))
    ptr = l.convert_layout(ptr, l.SliceLayout(1, V))
    ptr = ptr.to(l.pointer_type(l.uint32, 3))
    x, y = l.split(l.load(ptr[:, None] + word[None, :]))
    L: l.constexpr = l.BlockedLayout([1], [64], [4], [0])
    return l.convert_layout(x, L), l.convert_layout(y, L)


@g.jit
def _load_uint4(ptr, index):
    # The scalar-word pointer is an ABI view; index counts native uint4 elements.
    ptr = ptr + index * 4
    # Native uint4 activation reads lower to two i64 halves consumed by MFMA.
    V: l.constexpr = l.BlockedLayout([1, 2], [64, 1], [4, 1], [1, 0])
    half = l.arange(0, 2, layout=l.SliceLayout(0, V))
    ptr = l.convert_layout(ptr, l.SliceLayout(1, V))
    ptr = ptr.to(l.pointer_type(l.uint64, 3))
    xy, zw = l.split(l.load(ptr[:, None] + half[None, :]))
    L: l.constexpr = l.BlockedLayout([1], [64], [4], [0])
    xy, zw = l.convert_layout(xy, L), l.convert_layout(zw, L)
    return (
        xy.to(l.uint32),
        (xy >> 32).to(l.uint32),
        zw.to(l.uint32),
        (zw >> 32).to(l.uint32),
    )


class _MatrixState(NamedTuple):
    v_: object
    scales_: object
    stride_n_: object


class _W13State(NamedTuple):
    v_: object
    scales_: object
    stride_: object
    v_offset_: object
    s_offset_: object


class _W2State(NamedTuple):
    v_: object
    scales_: object
    stride_n_: object
    v_offset_: object


class _MxFp4State(NamedTuple):
    v_: object
    scales_: object
    stride_n_: object
    v_offset_: object
    s_offset_: object


class _InputState(NamedTuple):
    values_: object
    scales_: object
    dim_: object
    values_offset_bytes_: object
    scales_offset_bytes_: object


class MatrixLayout:
    __triton_builtin__ = True
    Scalar = l.float8e4nv
    kVecSize = 16 // (Scalar.primitive_bitwidth // 8)
    kGroupM = 256
    kGroupN = 256
    kNumWarps = 4
    kThreads = kNumWarps * kWarpSize.value
    kRefBufferRange = 0xFFFFFFF0
    kScaleBlockSize = 128
    kLoadGlobal = triton.cdiv(
        kGroupM * kGroupN * (Scalar.primitive_bitwidth // 8) // 16, kThreads
    )

    State = _MatrixState

    @g.jit
    def Initialize(value_ptr, value_range, scale_ptr, scale_range, stride_n):
        v_ = MakeBufferResource(value_ptr, value_range)
        scales_ = MakeBufferResource(scale_ptr, scale_range)
        return _MatrixState(v_, scales_, stride_n.to(l.uint32))

    @g.jit
    def FetchScale(state, tid):
        off = ((tid & 1) * state.stride_n_ // MatrixLayout.kScaleBlockSize) * 4 + (
            tid & 2
        ) * 2
        return BufferResource.LoadU32(state.scales_, off, l.full((), 0, l.int32), 0).to(
            l.float32, bitcast=True
        )


class W13Layout:
    __triton_builtin__ = True
    kGroupMPerWarp = 64
    kNumWarps = 4
    kGroupM = kNumWarps * kGroupMPerWarp
    kGroupN = 256
    kScaleBlockSize = 128
    kTileLoads = kGroupN // 32
    assert kGroupN % 32 == 0

    State = _W13State

    @staticmethod
    @triton.constexpr_function
    def ValueProjectionOffsetBytes(dim, inter_dim):
        return dim * inter_dim

    @staticmethod
    @triton.constexpr_function
    def ScaleProjectionOffsetBytes(dim, inter_dim):
        return (
            dim
            // W13Layout.kScaleBlockSize
            * (inter_dim // W13Layout.kScaleBlockSize)
            * 4
        )

    @g.jit
    def Initialize(value_ptr, value_range, scale_ptr, scale_range, dim):
        v_ = MakeBufferResource(value_ptr, value_range)
        scales_ = MakeBufferResource(scale_ptr, scale_range)
        return _W13State(
            v_,
            scales_,
            dim.to(l.uint32),
            l.full((), 0, l.uint32),
            l.full((), 0, l.uint32),
        )

    @g.jit
    def LoadTile(
        state,
        stage: l.constexpr,
        wid,
        wtid,
        value_offset=0,
        kAux: l.constexpr = TargetWeightLoadPolicy.kAux,
    ):
        v_, scales_, stride_, v_offset_, s_offset_ = state
        voffset = wid * 16 * stride_ + wtid * 16
        row_stride_bytes = kWarpSize * stride_
        reg = ()
        for i in l.static_range(W13Layout.kTileLoads // 2):
            for j in l.static_range(2):
                reg += (
                    _uint4_words(
                        BufferResource.Load(
                            v_,
                            voffset
                            + i * row_stride_bytes
                            + (stage * 2 + j) * kWarpSize * 16
                            + value_offset,
                            v_offset_,
                            kAux,
                        )
                    ),
                )
        return reg, state

    @g.jit
    def LoadScale(state, tid, scale_offset=0):
        v_, scales_, stride_, v_offset_, s_offset_ = state
        off = (
            (tid & 1) * stride_ // W13Layout.kScaleBlockSize * 4
            + (tid & 2) * 2
            + scale_offset
        )
        value = BufferResource.LoadU32(
            scales_, off, s_offset_, BufferResource.kNone
        ).to(l.float32, bitcast=True)
        return value, state

    @g.jit
    def AdvanceStep(state, kTileN: l.constexpr, kTileK: l.constexpr):
        l.static_assert(
            kTileN == 0 and kTileK == 2,
            "W13 advances one K256 tile",
        )
        return _W13State(
            state.v_,
            state.scales_,
            state.stride_,
            state.v_offset_ + (W13Layout.kTileLoads // 2) * kWarpSize * 16,
            state.s_offset_ + 8,
        )


class W2Layout(MatrixLayout):
    Base = MatrixLayout
    kTileLoads = Base.kLoadGlobal // 2
    assert Base.kLoadGlobal % 2 == 0

    State = _W2State

    @g.jit
    def Initialize(value_ptr, value_range, scale_ptr, scale_range, dim, value_offset):
        # Inherited native Initialize preserves the separately constructed offset.
        base = MatrixLayout.Initialize(
            value_ptr, value_range, scale_ptr, scale_range, dim
        )
        return _W2State(
            base.v_, base.scales_, base.stride_n_, value_offset.to(l.uint32)
        )

    @g.jit
    def LoadTile(
        state,
        stage: l.constexpr,
        wid,
        wtid,
        kAux: l.constexpr = TargetWeightLoadPolicy.kAux,
    ):
        v_, scales_, stride_n_, v_offset_ = state
        kInnerStep: l.constexpr = kWarpSize * 16
        voffset = wid * 16 * stride_n_ + wtid * 16
        # Array updates preserve native i-then-j load ordering.
        reg = _uninitialized_uint4_array(wtid, W2Layout.kTileLoads)
        for i in l.static_range(2):
            for j in l.static_range(W2Layout.kTileLoads // 2):
                r = _uint4_words(
                    BufferResource.Load(
                        v_,
                        voffset + 64 * j * stride_n_,
                        v_offset_ + (stage * 2 + i) * kInnerStep,
                        kAux,
                    )
                )
                reg = reg[: j * 2 + i] + (r,) + reg[j * 2 + i + 1 :]
        return reg, state

    @g.jit
    def LoadScale(state, tid):
        _, scales_, stride_n_, _ = state
        off = (tid & 1) * stride_n_ // W2Layout.kScaleBlockSize * 4 + (tid & 2) * 2
        value = BufferResource.LoadU32(
            scales_, off, l.full((), 0, l.int32), BufferResource.kNone
        ).to(l.float32, bitcast=True)
        return value, state

    @g.jit
    def AdvanceStep(state, kTileN: l.constexpr, kTileK: l.constexpr):
        l.static_assert(
            kTileN == 1 and kTileK == 0,
            "W2 advances one N256 tile",
        )
        return _W2State(
            state.v_,
            BufferResource.WithOffset(
                state.scales_,
                state.stride_n_ // W2Layout.kScaleBlockSize * 4 * 2,
            ),
            state.stride_n_,
            state.v_offset_ + state.stride_n_ * W2Layout.kGroupN,
        )


class MxFp4WeightLayout:
    def __new__(cls, kNumWarps=4, kLayout=0):
        return _MxFp4WeightLayoutSpecialization(kNumWarps, kLayout)

    __triton_builtin__ = True
    kGroupM = 128
    kGroupN = 256
    kNumWarps = 4
    kThreads = kNumWarps * kWarpSize.value
    kRowGroupSize = 32
    kRefBufferRange = 0xFFFFFFF0
    kScaleBlockSize = 128
    kLoadGlobal = triton.cdiv(kGroupM * kGroupN // 16 // 2, kThreads)
    assert kGroupN % 64 == 0

    State = _MxFp4State

    @g.jit
    def Initialize(value_ptr, value_range, scale_ptr, scale_range, stride_n):
        v_ = MakeBufferResource(value_ptr, value_range)
        scales_ = MakeBufferResource(scale_ptr, scale_range)
        return _MxFp4State(
            v_,
            scales_,
            stride_n.to(l.uint32),
            l.full((), 0, l.int32),
            l.full((), 0, l.int32),
        )

    @g.jit
    def AdvanceStep(state, TILE_N: l.constexpr, TILE_K: l.constexpr):
        kTileNStep: l.constexpr = 128
        kTileKStep: l.constexpr = 4 * 16
        kValueTileKStepBytes: l.constexpr = kTileKStep * 16
        kScaleTileKStepBytes: l.constexpr = MxFp4WeightLayout.kThreads * 4
        v_, scales_, stride_n_, v_offset_, s_offset_ = state
        return _MxFp4State(
            v_,
            scales_,
            stride_n_,
            v_offset_
            + (TILE_N * kTileNStep * stride_n_ // 2).to(l.int32)
            + TILE_K * kValueTileKStepBytes,
            s_offset_
            + (TILE_N * kTileNStep * stride_n_ // MxFp4WeightLayout.kRowGroupSize).to(
                l.int32
            )
            + TILE_K * kScaleTileKStepBytes,
        )

    @g.jit
    def LoadTile(state, stage: l.constexpr, wid, wtid):
        v_, scales_, stride_n_, v_offset_, s_offset_ = state
        kLoadGlobal: l.constexpr = MxFp4WeightLayout.kLoadGlobal
        voffset = wid * 16 * stride_n_ // 2 + wtid * 16
        reg = ()
        for i in l.static_range(kLoadGlobal):
            reg += (
                _uint4_words(
                    BufferResource.Load(
                        v_,
                        voffset + 64 * i * stride_n_ // 2,
                        v_offset_,
                        0,
                    )
                ),
            )
        return reg, state

    @g.jit
    def LoadScale(state, tid):
        off = tid * 4
        return BufferResource.LoadU32(state.scales_, off, state.s_offset_, 0), state


class InputLayout:
    __triton_builtin__ = True
    kTokenBatch = 8
    kNumWarps = 4
    kGroupDim = 256
    kThreads = kNumWarps * kWarpSize.value
    kGroupK = kGroupDim
    kSubGroupSize = 16
    kActivationFragments = kGroupDim // 32
    kScaleBlockSize = 128
    kRefBufferRange = 0xFFFFFFF0
    kShmInputPaddingBytes = 32 * kNumWarps
    kShmInputElements = kTokenBatch * kThreads + kShmInputPaddingBytes // 4
    kShmInputElementsPerWarp = kShmInputElements // kNumWarps
    kShmInputVec4PerWarp = kShmInputElementsPerWarp // (16 // 4)
    assert kShmInputElements % kNumWarps == 0
    assert kShmInputElementsPerWarp % (16 // 4) == 0

    State = _InputState

    @g.jit
    def Initialize(value_ptr, value_range, scale_ptr, scale_range, dim):
        values_ = MakeBufferResource(value_ptr, value_range)
        scales_ = MakeBufferResource(scale_ptr, scale_range)
        return _InputState(
            values_,
            scales_,
            dim.to(l.uint32),
            l.full((), 0, l.uint32),
            l.full((), 0, l.uint32),
        )

    @g.jit
    def FetchAsync(state, shm_x, wid, wtid, tokens):
        values_, scales_, dim_, values_offset_bytes_, scales_offset_bytes_ = state
        kTokenBatch: l.constexpr = InputLayout.kTokenBatch
        lds_ptr = shm_x + wid * InputLayout.kShmInputElementsPerWarp
        for i in l.static_range(kTokenBatch):
            src_off = tokens[i] * dim_ + values_offset_bytes_ + wtid * 4
            BufferResource.LoadLds(
                values_,
                lds_ptr,
                src_off,
                l.full((), 0, l.int32),
                0,
                4,
                0,
            )
            lds_ptr += kWarpSize
        values_offset_bytes_ += InputLayout.kGroupK
        return _InputState(
            values_, scales_, dim_, values_offset_bytes_, scales_offset_bytes_
        )

    @g.jit
    def FetchToRegs(shm_x, wtid):
        # Offset from reinterpret_cast<const uint4 *>(shm_x), in uint4 units.
        x = (
            InputLayout.kShmInputVec4PerWarp * (wtid & 3)
            + ((wtid >> 2) & 3) * (InputLayout.kGroupK // 16)
            + wtid // InputLayout.kSubGroupSize
        )
        kFragmentsPerRow: l.constexpr = InputLayout.kActivationFragments // 2
        regs = ()
        for i in l.static_range(2):
            for j in l.static_range(kFragmentsPerRow):
                regs += (_load_uint4(shm_x, x + i * kWarpSize + j * 4),)
        return regs

    @g.jit
    def FetchToRegsFP4(shm_x, wtid):
        kActivationFragments: l.constexpr = InputLayout.kActivationFragments
        x = shm_x + 2 * (
            InputLayout.kShmInputVec4PerWarp * 2 * (wtid & 3)
            + ((wtid >> 2) & 3) * (InputLayout.kGroupK // 8)
            + wtid // InputLayout.kSubGroupSize
        )
        words = ()
        for i in l.static_range(2):
            for j in l.static_range(kActivationFragments):
                p = x + 2 * (i * kWarpSize * 2 + j * 4)
                packed = l.load(p.to(l.pointer_type(l.uint64, 3)))
                words += (packed.to(l.uint32), (packed >> 32).to(l.uint32))
        regs = ()
        for i in l.static_range(kActivationFragments):
            regs += (words[i * 4 : i * 4 + 4],)
        return regs

    @g.jit
    def FetchScaleAsync(state, shm_scale_x, wid, wtid, token_select, m):
        values_, scales_, dim_, values_offset_bytes_, scales_offset_bytes_ = state
        lds_ptr = shm_scale_x + wid * kWarpSize
        token_off = l.where((wid & 1) != 0, token_select[1], token_select[0]) * 4
        BufferResource.LoadLds(
            scales_, lds_ptr, token_off, scales_offset_bytes_, 0, 4, 0
        )
        scales_offset_bytes_ += (
            m * (InputLayout.kGroupDim // InputLayout.kScaleBlockSize) * 4
        )
        return _InputState(
            values_, scales_, dim_, values_offset_bytes_, scales_offset_bytes_
        )

    @g.jit
    def FetchScaleToReg(shm_scale_x, tid):
        r = ()
        for i in l.static_range(4):
            r += (
                l.load(shm_scale_x + tid + i * kWarpSize).to(l.float32, bitcast=True),
            )
        return r


@g.jit
def MakeBufferResource(ptr, byte_range):
    fields = BufferResourceFields(
        ptr,
        (l.full((), 0, l.uint32) + byte_range).to(l.uint32),
        BufferResource.kDataFormatU32Config,
    )
    return BufferResource(fields, _resource_content(fields))


@builtin
def _load_vector4(ptr, _semantic):
    """Dereference a native aligned uint4/float4 through its scalar ABI view."""
    return _native_load_vector4(ptr, _semantic=_semantic)


@builtin
def _store_vector4(ptr, value, _semantic):
    """Assign a native aligned uint4/float4 through its scalar ABI view."""
    return _native_store_vector4(ptr, value, _semantic=_semantic)


class MxFp4TileShape(IntEnum):
    kN256 = 0
    kN128 = 1
    kM64N256 = 2
    kM64N256W8 = 3


class MxFp4WeightLayoutSelector(DeviceTemplate):
    def __init__(self, kLayout, kNumWarps):
        self._key = (int(kLayout), kNumWarps)
        self.kLayout = int(kLayout)
        self.kTileM = 64 if kLayout >= 2 else 32
        self.kWaveTileM = 64 if kLayout == 3 else 32
        self.kGroupN = 128 if kLayout == 1 else 256
        self.kTileK = 256
        self.kLoadGlobal = (4, 2, 8, 2)[kLayout]
        self.kWaveTileN = (64, 32, 128, 32)[kLayout]
        self.kWarpsN = 2
        if kLayout == 2:
            assert kNumWarps == 4
        if kLayout == 3:
            assert kNumWarps == 8

    @host_device
    def ValueOffsetBytes(self, wid, fragment, stride_n):
        if self.kLayout == 2:
            wave_n = wid % self.kWarpsN
            return (wave_n * self.kWaveTileN + fragment * 16) * stride_n // 2
        return (wid * self.kWaveTileN + fragment * 16) * stride_n // 2

    @host_device
    def K128OffsetBytes(self, stage):
        return stage * 64 * 16

    @host_device
    def N32Offset(self, wid, n32_pair):
        if self.kLayout == 2:
            return 4 * (wid % self.kWarpsN) + n32_pair
        if self.kLayout == 3:
            return wid
        return 2 * wid + n32_pair


class _MxFp4WeightLayoutSpecialization(DeviceTemplate):
    State = _MxFp4State

    def __init__(self, kNumWarps, kLayout):
        self._key = (kNumWarps, int(kLayout))
        self.kLayout = int(kLayout)
        self.LayoutSelector = MxFp4WeightLayoutSelector(kLayout, kNumWarps)
        self.kNumWarps, self.kThreads = kNumWarps, kNumWarps * 64
        self.kGroupM, self.kRowGroupSize = 128, 32
        self.kRefBufferRange, self.kScaleBlockSize = 0xFFFFFFF0, 128
        for field in ("kGroupN", "kTileM", "kWaveTileM", "kWaveTileN", "kLoadGlobal"):
            setattr(self, field, getattr(self.LayoutSelector, field))
        assert self.kGroupN % 64 == 0

    @host_device
    def ValueProjectionOffsetBytes(self, dim, inter_dim):
        return dim * inter_dim // 2

    @host_device
    def ScaleProjectionOffsetBytes(self, dim, inter_dim):
        return dim * inter_dim // self.kRowGroupSize

    @device_method
    def Initialize(self, value_ptr, value_range, scale_ptr, scale_range, stride_n):
        zero = l.full((), 0, l.uint32)
        return _MxFp4State(
            MakeBufferResource(value_ptr, value_range),
            MakeBufferResource(scale_ptr, scale_range),
            (zero + stride_n).to(l.uint32),
            zero,
            zero,
        )

    @device_method
    def AdvanceStep(self, state, kTileN: l.constexpr, kTileK: l.constexpr):
        l.static_assert(kTileK % 2 == 0, "block scales advance in K256 units")
        v_offset = (
            state.v_offset_ + kTileN * 256 * state.stride_n_ // 2 + kTileK * 64 * 16
        )
        s_offset = (
            state.s_offset_ + kTileN * 256 * state.stride_n_ // self.kRowGroupSize
        )
        s_offset += (kTileK // 2) * 64 * 4
        return _MxFp4State(state.v_, state.scales_, state.stride_n_, v_offset, s_offset)

    @device_method
    def LoadTile(
        self,
        state,
        stage,
        wid,
        wtid,
        value_offset=0,
        kAux: l.constexpr = TargetWeightLoadPolicy.kAux,
    ):
        lane_k_offset = wtid * 16
        k_offset = self.LayoutSelector.K128OffsetBytes(stage)
        regs = ()
        for fragment in l.static_range(self.kLoadGlobal):
            voffset = (
                self.LayoutSelector.ValueOffsetBytes(wid, fragment, state.stride_n_)
                + lane_k_offset
                + k_offset
                + value_offset
            )
            value = llvm_amdgcn_raw_buffer_load_v4i32(
                _resource_content(state.v_), voffset, state.v_offset_, kAux
            )
            regs += (value,)
        return regs

    @device_method
    def LoadScale(self, state, wid, wtid, n32_pair, scale_offset=0):
        if self.kLayout == 1:
            off = wid * state.stride_n_ + wtid * 4 + scale_offset
            return BufferResource.LoadU32(
                state.scales_, off, state.s_offset_, BufferResource.kNone
            )
        else:
            k256_blocks = state.stride_n_ // 256
            n32 = self.LayoutSelector.N32Offset(wid, n32_pair)
            word = n32 * k256_blocks * 64 + wtid
            return BufferResource.LoadU32(
                state.scales_,
                word * 4 + scale_offset,
                state.s_offset_,
                BufferResource.kNone,
            )


@g.jit
def _register_array_get(values, index):
    """Dynamic indexing of a native thread-private register array."""
    result = values[0]
    for i in l.static_range(1, len(values)):
        result = l.where(index == i, values[i], result)
    return result
