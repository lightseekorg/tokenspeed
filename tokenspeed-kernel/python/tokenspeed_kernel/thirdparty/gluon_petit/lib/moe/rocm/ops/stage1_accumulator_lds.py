"""Native accumulator-to-LDS layouts; pointers index scalar float words."""

import triton.experimental.gluon as g
from lib.moe.rocm.memory_ops import _store_vector4
from triton.experimental.gluon import language as l


@g.jit
def StoreStage1AccumulatorLds(
    shm, h, wid, wtid, kNumWarps: l.constexpr, kGroupN: l.constexpr
):
    l.static_assert(kGroupN % kNumWarps == 0, "invalid accumulator layout")
    kInputFragments: l.constexpr = (32 * kGroupN) // (kNumWarps * 64) // 4
    kColsPerWarp: l.constexpr = kGroupN // kNumWarps
    row_lane = wtid % 16
    col_quadrant = wtid // 16
    for fragment in l.static_range(kInputFragments):
        row = (fragment & 1) * 16 + row_lane
        col = wid * kColsPerWarp + (fragment // 2) * 16 + col_quadrant * 4
        _store_vector4(shm + row * kGroupN + col, h[fragment])


@g.jit
def StoreStage1AccumulatorLds2D(
    shm,
    h,
    wid,
    wtid,
    kGroupM: l.constexpr,
    kGroupN: l.constexpr,
    kWarpsM: l.constexpr,
    kWarpsN: l.constexpr,
):
    l.static_assert(kGroupM == 32 * kWarpsM, "each M wave covers M32")
    l.static_assert(kGroupN % kWarpsN == 0, "invalid N wave partition")
    kWaveN: l.constexpr = kGroupN // kWarpsN
    kInputFragments: l.constexpr = kWaveN // 8
    wave_m = wid // kWarpsN
    wave_n = wid % kWarpsN
    row_lane = wtid % 16
    col_quadrant = wtid // 16
    for fragment in l.static_range(kInputFragments):
        row = wave_m * 32 + (fragment & 1) * 16 + row_lane
        col = wave_n * kWaveN + (fragment // 2) * 16 + col_quadrant * 4
        _store_vector4(shm + row * kGroupN + col, h[fragment])


@g.jit
def StoreStage1AccumulatorLdsM64(
    shm, h, wid, wtid, kGroupN: l.constexpr, kNumWarps: l.constexpr
):
    l.static_assert(kGroupN == 32 * kNumWarps, "each wave owns one N32 slice")
    row_lane = wtid % 16
    col_quadrant = wtid // 16
    for n16 in l.static_range(2):
        for m16 in l.static_range(4):
            row = m16 * 16 + row_lane
            col = wid * 32 + n16 * 16 + col_quadrant * 4
            _store_vector4(shm + row * kGroupN + col, h[n16 * 4 + m16])
