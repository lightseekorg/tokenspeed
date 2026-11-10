# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Experimental three-stage software-pipelined dense16 Gluon GEMM for gfx950.

Computes ``C = A @ W.T`` for ``A [M, K]`` and ``W [N, K]`` (BF16/FP16, FP32
accumulate), aimed at skinny verify-step projections (M = 64..512).

Pipeline: a ring of ``NUM_STAGES`` LDS buffers per operand. ``NUM_STAGES``
K tiles are issued up front with direct-to-LDS buffer loads; each iteration
runs the MFMAs of tile ``i`` (already in registers), waits for tile ``i + 1``,
reads it from LDS into registers and refills the slot tile ``i`` came from
with tile ``i + NUM_STAGES``. So ``NUM_STAGES - 1`` tile loads are in flight
behind every MFMA block.

Global-load and padded LDS layouts follow the 8-wave a16w16 tutorial kernel
(largem.py): each warp's direct-to-LDS copy writes 512 contiguous elements,
followed by 16 elements of padding, and the row permutation spreads MFMA
operand reads across banks.

Optional split-K: each program reduces ``K / SPLIT_K``, stores its FP32
partial and bumps a per-tile counter; the last arriving program sums the
partials in split order, writes C and resets the counter (same protocol as
gluon_mm_a16w16_medium_gfx950).

Tiles that are not a power of two: ``BLOCK_M`` and ``BLOCK_N`` may each be
the sum of two powers of two (``lo`` the largest power of two, e.g. 192 =
128 + 64, 144 = 128 + 16). As in aiter's split head dim (96 = 64 + 32), each
part gets its own LDS ring and the tile runs as up to four MFMA chains
(lo/hi rows x lo/hi columns), one accumulator each; every operand slice is
read from LDS once per K tile and feeds both chains it belongs to. This lets
the tile count match the CU count more closely than power-of-two tiles do.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

LDS_BYTES = 160 * 1024


def _launch_metadata(grid, kernel, args):
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops16": 2 * m * n * k,
        "bytes": m * k * args["a_ptr"].element_size()
        + n * k * args["b_ptr"].element_size()
        + m * n * args["c_ptr"].element_size(),
    }


@gluon.constexpr_function
def _log2(x: int) -> int:
    return int(math.log2(x))


def split_pow2(x: int) -> tuple[int, int]:
    """Split a block size into (lo, hi): lo the largest power of two <= x, hi = x - lo.

    The kernels accept the size when hi is 0 or a power of two.
    """
    lo = 1 << (int(x).bit_length() - 1)
    return lo, int(x) - lo


@gluon.constexpr_function
def _pow2_lo(x: int) -> int:
    """Largest power of two <= x (the lo part of split_pow2)."""
    return 1 << (int(x).bit_length() - 1)


@gluon.constexpr_function
def _operand_bases(rows: int, block_k: int, num_warps: int):
    """Row/K bit assignment for one [rows, block_k] K-contiguous operand tile.

    Returns (reg_bases, lane_bases, warp_bases, shared_offset_bases) in
    [row, k] coordinates. 8 contiguous K elements per lane (16 bytes), the
    rest of the 64 lanes walk K and then rows; warps take the lowest row bits,
    registers the middle ones, lanes the highest ones. LDS order: all K bits,
    then lane row bits (so one warp instruction is 512 contiguous elements),
    then the remaining row bits ascending.
    """
    k_bits = _log2(block_k)
    row_bits = _log2(rows)
    warp_bits = _log2(num_warps)
    lane_k_bits = k_bits - 3
    lane_row_bits = 6 - lane_k_bits
    reg_row_bits = row_bits - warp_bits - lane_row_bits
    assert lane_k_bits >= 0 and lane_row_bits >= 0, (rows, block_k)
    assert reg_row_bits >= 0, f"tile {rows}x{block_k} too small for {num_warps} warps"
    warp_rows = [1 << b for b in range(warp_bits)]
    reg_rows = [1 << b for b in range(warp_bits, warp_bits + reg_row_bits)]
    lane_rows = [1 << b for b in range(warp_bits + reg_row_bits, row_bits)]
    reg = [[0, 1], [0, 2], [0, 4]] + [[r, 0] for r in reg_rows]
    lane = [[0, 1 << b] for b in range(3, k_bits)] + [[r, 0] for r in lane_rows]
    warp = [[r, 0] for r in warp_rows]
    shared = (
        [[0, 1 << b] for b in range(k_bits)]
        + [[r, 0] for r in lane_rows]
        + [[r, 0] for r in sorted(warp_rows + reg_rows)]
    )
    return reg, lane, warp, shared


@gluon.constexpr_function
def _swap(bases):
    return [[b[1], b[0]] for b in bases]


@gluon.constexpr_function
def _gload_layout(rows: int, block_k: int, num_warps: int, transposed: bool):
    reg, lane, warp, _ = _operand_bases(rows, block_k, num_warps)
    shape = [rows, block_k]
    if transposed:
        reg, lane, warp, shape = _swap(reg), _swap(lane), _swap(warp), [block_k, rows]
    return gl.DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp, block_bases=[], shape=shape
    )


@gluon.constexpr_function
def _shared_layout(rows: int, block_k: int, num_warps: int, transposed: bool):
    _, _, _, shared = _operand_bases(rows, block_k, num_warps)
    shape = [rows, block_k]
    if transposed:
        shared, shape = _swap(shared), [block_k, rows]
    return gl.PaddedSharedLayout([[512, 16]], shared, [], shape)


@gluon.constexpr_function
def _store_layout(block_m: int, block_n: int, num_warps: int):
    lanes_n = min(block_n // 8, 64)
    return gl.BlockedLayout([1, 8], [64 // lanes_n, lanes_n], [num_warps, 1], [1, 0])


@gluon.jit
def _pids(
    pid, num_pid_m, num_pid_n, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr
):
    num_tiles = num_pid_m * num_pid_n
    if NUM_XCDS > 1:
        # Hardware dispatches program i to XCD i % NUM_XCDS; give each XCD a
        # contiguous range of tiles so neighbours share its L2.
        per_xcd = (num_tiles + NUM_XCDS - 1) // NUM_XCDS
        tall = num_tiles % NUM_XCDS
        tall = NUM_XCDS if tall == 0 else tall
        xcd = pid % NUM_XCDS
        local = pid // NUM_XCDS
        if xcd < tall:
            pid = xcd * per_xcd + local
        else:
            pid = tall * per_xcd + (xcd - tall) * (per_xcd - 1) + local
    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        group_size = GROUP_SIZE_M * num_pid_n
        first_m = (pid // group_size) * GROUP_SIZE_M
        size_m = min(num_pid_m - first_m, GROUP_SIZE_M)
        pid_m = first_m + (pid % group_size) % size_m
        pid_n = (pid % group_size) // size_m
    return pid_m, pid_n


@gluon.jit
def _issue_tile(
    smemA,
    smemAh,
    smemB,
    smemBh,
    slot,
    a_base,
    b_base,
    a_offs,
    ah_offs,
    b_offs,
    bh_offs,
    kofs,
    SPLIT_M: gl.constexpr,
    SPLIT_N: gl.constexpr,
):
    """Direct-to-LDS loads of one K tile (every operand part) as one commit group."""
    async_copy.buffer_load_to_shared(smemA.index(slot), a_base, a_offs + kofs)
    if SPLIT_M:
        async_copy.buffer_load_to_shared(smemAh.index(slot), a_base, ah_offs + kofs)
    async_copy.buffer_load_to_shared(smemB.index(slot), b_base, b_offs + kofs)
    if SPLIT_N:
        async_copy.buffer_load_to_shared(smemBh.index(slot), b_base, bh_offs + kofs)
    async_copy.commit_group()


@gluon.jit
def _read_tile(
    smemA,
    smemAh,
    smemB,
    smemBh,
    slot,
    dotA: gl.constexpr,
    dotB: gl.constexpr,
    SPLIT_M: gl.constexpr,
    SPLIT_N: gl.constexpr,
):
    """LDS -> registers for one K tile. Unsplit hi parts alias the lo part."""
    a = smemA.index(slot).load(dotA)
    b = smemB.index(slot).load(dotB)
    ah = a
    bh = b
    if SPLIT_M:
        ah = smemAh.index(slot).load(dotA)
    if SPLIT_N:
        bh = smemBh.index(slot).load(dotB)
    return a, ah, b, bh


@gluon.jit
def _mfma_tile(
    a,
    ah,
    b,
    bh,
    acc,
    acc_lh,
    acc_hl,
    acc_hh,
    CD_REGCLASS: gl.constexpr,
    SPLIT_M: gl.constexpr,
    SPLIT_N: gl.constexpr,
):
    """One K tile of the up to four MFMA chains (row part x column part)."""
    acc = cdna4.mfma(a, b, acc, cd_regclass=CD_REGCLASS)
    if SPLIT_N:
        acc_lh = cdna4.mfma(a, bh, acc_lh, cd_regclass=CD_REGCLASS)
    if SPLIT_M:
        acc_hl = cdna4.mfma(ah, b, acc_hl, cd_regclass=CD_REGCLASS)
        if SPLIT_N:
            acc_hh = cdna4.mfma(ah, bh, acc_hh, cd_regclass=CD_REGCLASS)
    return acc, acc_lh, acc_hl, acc_hh


@gluon.jit
def _store_part(
    acc,
    c_ptr,
    partial_ptr,
    part_base,
    M,
    N,
    m0,
    n0,
    stride_cm,
    BM_PART: gl.constexpr,
    BN_PART: gl.constexpr,
    M_OFF: gl.constexpr,
    N_OFF: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    TO_PARTIAL: gl.constexpr,
):
    """Store the [BM_PART, BN_PART] part at (M_OFF, N_OFF) of the tile at (m0, n0).

    TO_PARTIAL: FP32 into the row-major [BLOCK_M, BLOCK_N] split-K partial at
    ``part_base``; otherwise cast and store into C with the M/N mask.
    """
    storeL: gl.constexpr = _store_layout(BM_PART, BN_PART, NUM_WARPS)
    offs_sm = gl.arange(0, BM_PART, gl.SliceLayout(1, storeL))
    offs_sn = gl.arange(0, BN_PART, gl.SliceLayout(0, storeL))
    if TO_PARTIAL:
        cdna4.buffer_store(
            ptr=partial_ptr + part_base,
            offsets=(M_OFF + offs_sm)[:, None] * BLOCK_N + (N_OFF + offs_sn)[None, :],
            stored_value=gl.convert_layout(acc, storeL),
            cache=".wt",
        )
    else:
        sm = m0 + M_OFF + offs_sm
        sn = n0 + N_OFF + offs_sn
        cdna4.buffer_store(
            ptr=c_ptr + (m0 + M_OFF).to(gl.int64) * stride_cm + n0 + N_OFF,
            offsets=offs_sm[:, None] * stride_cm + offs_sn[None, :],
            # Cast before the layout conversion: half the bytes through LDS.
            stored_value=gl.convert_layout(acc.to(c_ptr.dtype.element_ty), storeL),
            mask=(sm[:, None] < M) & (sn[None, :] < N),
        )


@gluon.jit
def _reduce_part(
    c_ptr,
    partial_ptr,
    tile,
    num_tiles,
    M,
    N,
    m0,
    n0,
    stride_cm,
    BM_PART: gl.constexpr,
    BN_PART: gl.constexpr,
    M_OFF: gl.constexpr,
    N_OFF: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SPLIT_K: gl.constexpr,
):
    """Sum one part of a tile's SPLIT_K partials in split order and store it into C."""
    storeL: gl.constexpr = _store_layout(BM_PART, BN_PART, NUM_WARPS)
    TILE_ELEMS: gl.constexpr = BLOCK_M * BLOCK_N
    offs_sm = gl.arange(0, BM_PART, gl.SliceLayout(1, storeL))
    offs_sn = gl.arange(0, BN_PART, gl.SliceLayout(0, storeL))
    p_offs = (M_OFF + offs_sm)[:, None] * BLOCK_N + (N_OFF + offs_sn)[None, :]
    out = cdna4.buffer_load(
        ptr=partial_ptr + tile.to(gl.int64) * TILE_ELEMS, offsets=p_offs, cache=".cv"
    )
    for s in gl.static_range(1, SPLIT_K):
        out += cdna4.buffer_load(
            ptr=partial_ptr + (s * num_tiles + tile).to(gl.int64) * TILE_ELEMS,
            offsets=p_offs,
            cache=".cv",
        )
    _store_part(
        out,
        c_ptr,
        partial_ptr,
        0,
        M,
        N,
        m0,
        n0,
        stride_cm,
        BM_PART,
        BN_PART,
        M_OFF,
        N_OFF,
        BLOCK_N,
        NUM_WARPS,
        False,
    )


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_mm_a16w16_3stage_gfx950(
    a_ptr,
    b_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_am,
    stride_bn: gl.constexpr,
    stride_cm,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    WARP_PIPE: gl.constexpr,
    CD_REGCLASS: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    K_SPLIT: gl.constexpr = K // SPLIT_K
    K_TILES: gl.constexpr = K_SPLIT // BLOCK_K
    # Row and column parts: BLOCK = LO + HI, LO a power of two, HI 0 or one.
    BM_LO: gl.constexpr = _pow2_lo(BLOCK_M)
    BM_HI: gl.constexpr = BLOCK_M - BM_LO
    BN_LO: gl.constexpr = _pow2_lo(BLOCK_N)
    BN_HI: gl.constexpr = BLOCK_N - BN_LO
    SPLIT_M: gl.constexpr = BM_HI > 0
    SPLIT_N: gl.constexpr = BN_HI > 0
    gl.static_assert(K % (SPLIT_K * BLOCK_K) == 0, "K must split into whole K tiles")
    gl.static_assert(K_TILES >= NUM_STAGES, "need at least NUM_STAGES K tiles")
    gl.static_assert(NUM_STAGES >= 2, "pipeline needs two or more stages")

    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    if SPLIT_K > 1:
        tile = pid % num_tiles
        split = pid // num_tiles
    else:
        tile = pid
        split = 0
    pid_m, pid_n = _pids(tile, num_pid_m, num_pid_n, NUM_XCDS, GROUP_SIZE_M)

    gLoadA: gl.constexpr = _gload_layout(BM_LO, BLOCK_K, NUM_WARPS, False)
    gLoadB: gl.constexpr = _gload_layout(BN_LO, BLOCK_K, NUM_WARPS, True)
    sharedA: gl.constexpr = _shared_layout(BM_LO, BLOCK_K, NUM_WARPS, False)
    sharedB: gl.constexpr = _shared_layout(BN_LO, BLOCK_K, NUM_WARPS, True)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dotB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)

    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_STAGES, BM_LO, BLOCK_K], sharedA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_K, BN_LO], sharedB
    )

    offs_am = gl.arange(0, BM_LO, gl.SliceLayout(1, gLoadA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadA))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadB))
    offs_bn = gl.arange(0, BN_LO, gl.SliceLayout(0, gLoadB))

    # Per-program bases in int64 on the scalar pointer; tile offsets stay int32.
    # Rows past M and weight rows past N clamp onto the last valid one (the
    # store mask drops them), so no load needs a mask.
    k0 = split * K_SPLIT
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    a_base = a_ptr + m0.to(gl.int64) * stride_am + k0
    b_base = b_ptr + n0.to(gl.int64) * stride_bn + k0
    rows = gl.minimum(offs_am, M - 1 - m0)
    cols = gl.minimum(offs_bn, N - 1 - n0)
    a_offs = rows[:, None] * stride_am + offs_ak[None, :]
    b_offs = offs_bk[:, None] + cols[None, :] * stride_bn

    # Hi parts: own LDS ring and offsets; when unsplit they alias the lo part
    # and are never touched.
    smemAh = smemA
    smemBh = smemB
    ah_offs = a_offs
    bh_offs = b_offs
    if SPLIT_M:
        gLoadAh: gl.constexpr = _gload_layout(BM_HI, BLOCK_K, NUM_WARPS, False)
        sharedAh: gl.constexpr = _shared_layout(BM_HI, BLOCK_K, NUM_WARPS, False)
        smemAh = gl.allocate_shared_memory(
            a_ptr.dtype.element_ty, [NUM_STAGES, BM_HI, BLOCK_K], sharedAh
        )
        offs_ahm = gl.arange(0, BM_HI, gl.SliceLayout(1, gLoadAh))
        offs_ahk = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadAh))
        rows_h = gl.minimum(BM_LO + offs_ahm, M - 1 - m0)
        ah_offs = rows_h[:, None] * stride_am + offs_ahk[None, :]
    if SPLIT_N:
        gLoadBh: gl.constexpr = _gload_layout(BN_HI, BLOCK_K, NUM_WARPS, True)
        sharedBh: gl.constexpr = _shared_layout(BN_HI, BLOCK_K, NUM_WARPS, True)
        smemBh = gl.allocate_shared_memory(
            b_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_K, BN_HI], sharedBh
        )
        offs_bhk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadBh))
        offs_bhn = gl.arange(0, BN_HI, gl.SliceLayout(0, gLoadBh))
        cols_h = gl.minimum(BN_LO + offs_bhn, N - 1 - n0)
        bh_offs = offs_bhk[:, None] + cols_h[None, :] * stride_bn

    # Prologue: NUM_STAGES tiles in flight, one commit group per tile.
    for s in gl.static_range(NUM_STAGES):
        _issue_tile(
            smemA, smemAh, smemB, smemBh, s, a_base, b_base,
            a_offs, ah_offs, b_offs, bh_offs, s * BLOCK_K, SPLIT_M, SPLIT_N,
        )  # fmt: skip

    async_copy.wait_group(NUM_STAGES - 1)
    a, ah, b, bh = _read_tile(
        smemA, smemAh, smemB, smemBh, 0, dotA, dotB, SPLIT_M, SPLIT_N
    )
    acc = gl.zeros((BM_LO, BN_LO), gl.float32, mfma)
    acc_lh = acc
    acc_hl = acc
    acc_hh = acc
    if SPLIT_N:
        acc_lh = gl.zeros((BM_LO, BN_HI), gl.float32, mfma)
    if SPLIT_M:
        acc_hl = gl.zeros((BM_HI, BN_LO), gl.float32, mfma)
        if SPLIT_N:
            acc_hh = gl.zeros((BM_HI, BN_HI), gl.float32, mfma)

    for i in range(0, K_TILES - NUM_STAGES):
        cur = i % NUM_STAGES
        nxt = (i + 1) % NUM_STAGES
        kofs = (i + NUM_STAGES) * BLOCK_K
        if WARP_PIPE:
            # As in largem.py: the wait stays outside the stages so the
            # barrier that publishes other waves' LDS writes precedes the reads.
            async_copy.wait_group(NUM_STAGES - 2)
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc, acc_lh, acc_hl, acc_hh = _mfma_tile(
                    a, ah, b, bh, acc, acc_lh, acc_hl, acc_hh,
                    CD_REGCLASS, SPLIT_M, SPLIT_N,
                )  # fmt: skip
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a, ah, b, bh = _read_tile(
                    smemA, smemAh, smemB, smemBh, nxt, dotA, dotB, SPLIT_M, SPLIT_N
                )
                _issue_tile(
                    smemA, smemAh, smemB, smemBh, cur, a_base, b_base,
                    a_offs, ah_offs, b_offs, bh_offs, kofs, SPLIT_M, SPLIT_N,
                )  # fmt: skip
        else:
            acc, acc_lh, acc_hl, acc_hh = _mfma_tile(
                a, ah, b, bh, acc, acc_lh, acc_hl, acc_hh,
                CD_REGCLASS, SPLIT_M, SPLIT_N,
            )  # fmt: skip
            async_copy.wait_group(NUM_STAGES - 2)
            a, ah, b, bh = _read_tile(
                smemA, smemAh, smemB, smemBh, nxt, dotA, dotB, SPLIT_M, SPLIT_N
            )
            _issue_tile(
                smemA, smemAh, smemB, smemBh, cur, a_base, b_base,
                a_offs, ah_offs, b_offs, bh_offs, kofs, SPLIT_M, SPLIT_N,
            )  # fmt: skip

    # Drain: the last NUM_STAGES tiles; tile K_TILES - NUM_STAGES is in registers.
    for j in gl.static_range(1, NUM_STAGES):
        acc, acc_lh, acc_hl, acc_hh = _mfma_tile(
            a, ah, b, bh, acc, acc_lh, acc_hl, acc_hh, CD_REGCLASS, SPLIT_M, SPLIT_N
        )
        async_copy.wait_group(NUM_STAGES - 1 - j)
        slot = (K_TILES - NUM_STAGES + j) % NUM_STAGES
        a, ah, b, bh = _read_tile(
            smemA, smemAh, smemB, smemBh, slot, dotA, dotB, SPLIT_M, SPLIT_N
        )
    acc, acc_lh, acc_hl, acc_hh = _mfma_tile(
        a, ah, b, bh, acc, acc_lh, acc_hl, acc_hh, CD_REGCLASS, SPLIT_M, SPLIT_N
    )

    TO_PARTIAL: gl.constexpr = SPLIT_K > 1
    part_base = (split * num_tiles + tile).to(gl.int64) * (BLOCK_M * BLOCK_N)
    _store_part(
        acc, c_ptr, partial_ptr, part_base, M, N, m0, n0, stride_cm,
        BM_LO, BN_LO, 0, 0, BLOCK_N, NUM_WARPS, TO_PARTIAL,
    )  # fmt: skip
    if SPLIT_N:
        _store_part(
            acc_lh, c_ptr, partial_ptr, part_base, M, N, m0, n0, stride_cm,
            BM_LO, BN_HI, 0, BN_LO, BLOCK_N, NUM_WARPS, TO_PARTIAL,
        )  # fmt: skip
    if SPLIT_M:
        _store_part(
            acc_hl, c_ptr, partial_ptr, part_base, M, N, m0, n0, stride_cm,
            BM_HI, BN_LO, BM_LO, 0, BLOCK_N, NUM_WARPS, TO_PARTIAL,
        )  # fmt: skip
        if SPLIT_N:
            _store_part(
                acc_hh, c_ptr, partial_ptr, part_base, M, N, m0, n0, stride_cm,
                BM_HI, BN_HI, BM_LO, BN_LO, BLOCK_N, NUM_WARPS, TO_PARTIAL,
            )  # fmt: skip
    if SPLIT_K > 1:
        # Publish the partial: wait for this wave's stores, then order all
        # waves' stores before the counter bump.
        gl.inline_asm_elementwise(
            asm="s_waitcnt vmcnt(0)",
            constraints="=v,0",
            args=[offs_am],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.barrier()
        old = gl.atomic_add(counter_ptr + tile, 1, sem="relaxed", scope="gpu")
        if old == SPLIT_K - 1:
            # Last arriver: sum the partials in split order, store C, reset.
            _reduce_part(
                c_ptr, partial_ptr, tile, num_tiles, M, N, m0, n0, stride_cm,
                BM_LO, BN_LO, 0, 0, BLOCK_M, BLOCK_N, NUM_WARPS, SPLIT_K,
            )  # fmt: skip
            if SPLIT_N:
                _reduce_part(
                    c_ptr, partial_ptr, tile, num_tiles, M, N, m0, n0, stride_cm,
                    BM_LO, BN_HI, 0, BN_LO, BLOCK_M, BLOCK_N, NUM_WARPS, SPLIT_K,
                )  # fmt: skip
            if SPLIT_M:
                _reduce_part(
                    c_ptr, partial_ptr, tile, num_tiles, M, N, m0, n0, stride_cm,
                    BM_HI, BN_LO, BM_LO, 0, BLOCK_M, BLOCK_N, NUM_WARPS, SPLIT_K,
                )  # fmt: skip
                if SPLIT_N:
                    _reduce_part(
                        c_ptr, partial_ptr, tile, num_tiles, M, N, m0, n0,
                        stride_cm, BM_HI, BN_HI, BM_LO, BN_LO, BLOCK_M, BLOCK_N,
                        NUM_WARPS, SPLIT_K,
                    )  # fmt: skip
            gl.store(counter_ptr + tile, 0)


def tile_parts_fit(block: int, warps: int, block_k: int, num_warps: int) -> bool:
    """Whether a BLOCK_M or BLOCK_N splits into parts the operand layouts accept.

    Each part (lo, and hi when nonzero) must be a power of two, a multiple of
    16 rows per warp along that dimension, and tall enough that one warp's
    direct-to-LDS copy covers 64 lanes x 8 K elements.
    """
    lo, hi = split_pow2(block)
    lane_row_bits = 6 - (_log2(block_k) - 3)
    for rows in (lo, hi) if hi else (lo,):
        if rows & (rows - 1) or rows % (16 * warps):
            return False
        if _log2(rows) < _log2(num_warps) + lane_row_bits:
            return False
    return True


def config_fits(cfg: tuple, M: int, N: int, K: int) -> bool:
    """Whether a (BM, BN, BK, WM, WN, NS, SK, XCD) config is legal for M x N x K.

    BM and BN may be a power of two or the sum of two (see ``split_pow2``).
    """
    bm, bn, bk, wm, wn, ns, sk, _ = cfg
    nw = wm * wn
    if nw not in (4, 8):
        return False
    if not (tile_parts_fit(bm, wm, bk, nw) and tile_parts_fit(bn, wn, bk, nw)):
        return False
    if K % (sk * bk) or K // (sk * bk) < ns:
        return False
    # 16 elements of padding per 512 (PaddedSharedLayout [[512, 16]]).
    if ns * (bm + bn) * bk * 2 * 33 // 32 > LDS_BYTES:
        return False
    return True


_scratch: dict = {}


def launch_gluon_mm_a16w16_3stage_gfx950(
    A: torch.Tensor,
    W: torch.Tensor,
    C: torch.Tensor,
    cfg: tuple,
    warp_pipe: bool = False,
    group_size_m: int = 4,
    schedule_hint: str = "",
    pin_acc: bool = False,
) -> torch.Tensor:
    """C[M, N] = A[M, K] @ W[N, K].T with config (BM, BN, BK, WM, WN, NS, SK, XCD)."""
    bm, bn, bk, wm, wn, ns, sk, xcd = cfg
    M, K = A.shape
    N = W.shape[0]
    assert A.stride(1) == 1 and W.stride(1) == 1 and C.stride(1) == 1
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    if sk > 1:
        key = (A.device, tiles, sk, bm * bn)
        if key not in _scratch:
            _scratch[key] = (
                torch.empty(tiles * sk * bm * bn, device=A.device, dtype=torch.float32),
                torch.zeros(tiles, device=A.device, dtype=torch.int32),
            )
        partial, counters = _scratch[key]
    else:
        partial = counters = C
    gluon_mm_a16w16_3stage_gfx950[(tiles * sk,)](
        A,
        W,
        C,
        partial,
        counters,
        M,
        N,
        K,
        A.stride(0),
        W.stride(0),
        C.stride(0),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        WARPS_M=wm,
        WARPS_N=wn,
        NUM_STAGES=ns,
        SPLIT_K=sk,
        NUM_XCDS=xcd,
        GROUP_SIZE_M=group_size_m,
        WARP_PIPE=warp_pipe,
        CD_REGCLASS="a" if pin_acc else None,
        num_warps=wm * wn,
        # pin_acc keeps the accumulators in AGPRs (tutorial cd_regclass="a");
        # otherwise forbid AGPRs so nothing is copied between register files.
        llvm_fn_attrs=() if pin_acc else (("amdgpu-agpr-alloc", "0,0"),),
        schedule_hint=schedule_hint,
    )
    return C
