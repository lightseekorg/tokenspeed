"""Native TAL bias layouts and optional BF16 projection biases."""

from typing import NamedTuple

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import BufferResource
from lib.moe.rocm.memory_ops import MakeBufferResource
from lib.tal.device import DeviceTemplate, device_method
from lib.tal.host_device import host_device
from lib.tal.tensor.layout import Layout, Shape, Stride, make_coord
from triton.experimental.gluon import language as l


def MxFp4BiasLayout(kGroupM, kGroupN):
    if (kGroupM, kGroupN) == (32, 128):
        return Layout(Shape(2, 4, Shape(2, 2)), Stride(4, 16, Stride(8, 64)))
    if (kGroupM, kGroupN) == (32, 256):
        return Layout(Shape(4, 4, 4), Stride(4, 16, 64))
    if (kGroupM, kGroupN) == (64, 256):
        return Layout(
            Shape(Shape(4, 2), 4, Shape(2, 2)),
            Stride(Stride(4, 64), 16, Stride(128, 0)),
        )
    raise ValueError("unsupported native bias layout")


MxFp4BiasLayoutM64N256 = MxFp4BiasLayout(64, 256)
MxFp4BiasLayoutM64N256W8 = Layout(
    Shape(2, 4, Shape(2, 4)), Stride(4, 16, Stride(8, 64))
)


class Bf16BiasAccess(DeviceTemplate):
    def __init__(self, MemoryLayout):
        self.Layout = MemoryLayout
        self._key = MemoryLayout.cache_key

    @device_method
    def LoadFragment(self, state, fragment, tile_col, tid):
        wid = tid // 64
        q = (tid % 64) // 16
        layout: l.constexpr = self.Layout
        col = tile_col + layout(make_coord(fragment, q, wid))
        return BufferResource.LoadU64(state.v_, col * 2, 0, BufferResource.kNone)

    @device_method
    def LoadFragments(self, state, tile_col, tid, kLoadGlobal: l.constexpr):
        fragments = ()
        for fragment in l.static_range(kLoadGlobal):
            fragments += (self.LoadFragment(state, fragment, tile_col, tid),)
        return fragments


@g.jit
def Bf16BiasToFloat(packed):
    return (
        (packed[0] << 16).to(l.float32, bitcast=True),
        (packed[0] & 0xFFFF0000).to(l.float32, bitcast=True),
        (packed[1] << 16).to(l.float32, bitcast=True),
        (packed[1] & 0xFFFF0000).to(l.float32, bitcast=True),
    )


class BiasState(NamedTuple):
    v_: object


class BiasPrefetch(NamedTuple):
    fragments: object


class Bf16BiasLayout(DeviceTemplate):
    def __init__(self, kNumWarps, kGroupN, MemoryLayout, kLoadGlobal=None, kMRepeats=2):
        self._key = (kNumWarps, kGroupN, MemoryLayout.cache_key, kLoadGlobal, kMRepeats)
        self.kNumWarps, self.kGroupN, self.kThreads = kNumWarps, kGroupN, kNumWarps * 64
        self.kLoadGlobal = kGroupN // 64 if kLoadGlobal is None else kLoadGlobal
        self.kMRepeats, self.kElementBytes, self.kPackedTileElements = kMRepeats, 2, 256
        self.Access = Bf16BiasAccess(MemoryLayout)
        assert kGroupN in (128, 256) and kNumWarps in (4, 8)

    @host_device
    def PackedStride(self, dim):
        return (
            (dim + self.kPackedTileElements - 1) // self.kPackedTileElements
        ) * self.kPackedTileElements

    @device_method
    def Initialize(self, value_ptr, expert_id, dim, tile_k, expert_stride):
        tile_col = tile_k * self.kGroupN
        value_offset = expert_id * expert_stride + tile_col
        if value_ptr is None:
            address = l.full((), 0, l.uint64)
        else:
            address = value_ptr.to(l.uint64)
        byte_range = l.where(
            address == 0, 0, (expert_stride - tile_col) * self.kElementBytes
        )
        ptr = (address + value_offset * self.kElementBytes).to(l.pointer_type(l.uint8))
        return BiasState(MakeBufferResource(ptr, byte_range))

    @device_method
    def AddToAccumulator(self, state, t, tile_col, tid):
        prefetch = self.PrefetchFragments(state, tile_col, tid)
        return self.Apply(t, prefetch)

    @device_method
    def PrefetchFragments(self, state, tile_col, tid):
        return BiasPrefetch(
            self.Access.LoadFragments(state, tile_col, tid, self.kLoadGlobal)
        )

    @device_method
    def Apply(self, t, prefetch):
        for fragment in l.static_range(self.kLoadGlobal):
            value = Bf16BiasToFloat(prefetch.fragments[fragment])
            for m16 in l.static_range(self.kMRepeats):
                idx: l.constexpr
                idx = self.kMRepeats * fragment + m16
                result = (
                    t[idx][0] + value[0],
                    t[idx][1] + value[1],
                    t[idx][2] + value[2],
                    t[idx][3] + value[3],
                )
                t = t[:idx] + (result,) + t[idx + 1 :]
        return t


class NoopBiasLayout(DeviceTemplate):
    def __init__(self, kNumWarps, kGroupN):
        self._key = (kNumWarps, kGroupN)
        self.kNumWarps, self.kGroupN = kNumWarps, kGroupN
        self.kElementBytes = self.kPackedTileElements = 1

    @host_device
    def PackedStride(self, dim):
        return dim

    @device_method
    def Initialize(self, value_ptr, expert_id, dim, tile_k, expert_stride):
        return ()

    @device_method
    def AddToAccumulator(self, state, t, tile_col, tid):
        return t

    @device_method
    def PrefetchFragments(self, state, tile_col, tid):
        return ()

    @device_method
    def Apply(self, t, prefetch):
        return t
