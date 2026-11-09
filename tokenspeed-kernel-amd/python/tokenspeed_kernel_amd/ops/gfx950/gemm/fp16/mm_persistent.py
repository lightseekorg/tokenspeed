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

"""Experimental persistent "full rounds + split-K tail" dense16 Gluon GEMM.

``C = A @ W.T`` for skinny ``A [M, K]`` (M = 64..512) against a large
``W [N, K]`` on gfx950. These GEMMs are bound by the weight stream and the
bandwidth scales with the number of busy CUs, so a grid of T tiles that is not
a multiple of the CU count leaves CUs idle in its last round.

One launch of ``GRID`` programs (one per CU) runs three steps:

- Tail: the ``tail = T % GRID`` tiles left over after the full rounds are
  split along K into ``PIECES`` balanced ranges (host-chosen,
  ``tail * PIECES <= GRID``), one range per program. Each program does its
  piece first, stores the FP32 partial and arrives on a per-tile counter.
- Data-parallel rounds: then ``T // GRID`` rounds, in each of which every
  program computes one whole tile with the 3-stage pipeline of mm_3stage.py.
- Tail reduction: at the end the pieces of a tail tile wait until all of them
  have arrived (by then they have, so the wait does not stall) and reduce the
  tile together: piece ``j`` sums 1-D blocks ``j, j + PIECES, ...`` of the
  flattened tile over all partials in fixed piece order and stores them as C.
  The last piece to finish resets the counter.

The wait is a spin on the counter, so it needs every piece of a tail tile to
be resident at the same time: ``GRID`` must not exceed the number of
workgroups that fit on the GPU at once. The launcher caps it at the CU count.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_3stage import (
    LDS_BYTES,
    _gload_layout,
    _launch_metadata,
    _log2,
    _pids,
    _shared_layout,
)

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M", "PIECES"))
def gluon_mm_a16w16_persistent_gfx950(
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
    PIECES,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    GRID: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    CD_REGCLASS: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    KT: gl.constexpr = K // BLOCK_K
    TILE_ELEMS: gl.constexpr = BLOCK_M * BLOCK_N
    gl.static_assert(K % BLOCK_K == 0, "K must be a multiple of BLOCK_K")
    gl.static_assert(KT >= NUM_STAGES, "need at least NUM_STAGES K tiles")
    gl.static_assert(NUM_STAGES >= 2, "pipeline needs two or more stages")
    gl.static_assert(GRID % NUM_XCDS == 0, "GRID must be a multiple of NUM_XCDS")

    pid = gl.program_id(0)
    if NUM_XCDS > 1:
        # Program i runs on XCD i % NUM_XCDS: give each XCD a contiguous block
        # of the round's tiles so neighbours share its L2.
        pid = (pid % NUM_XCDS) * (GRID // NUM_XCDS) + pid // NUM_XCDS
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    rounds = num_tiles // GRID
    tail = num_tiles - rounds * GRID
    n_items = rounds + (pid < tail * PIECES).to(gl.int32)

    gLoadA: gl.constexpr = _gload_layout(BLOCK_M, BLOCK_K, NUM_WARPS, False)
    gLoadB: gl.constexpr = _gload_layout(BLOCK_N, BLOCK_K, NUM_WARPS, True)
    sharedA: gl.constexpr = _shared_layout(BLOCK_M, BLOCK_K, NUM_WARPS, False)
    sharedB: gl.constexpr = _shared_layout(BLOCK_N, BLOCK_K, NUM_WARPS, True)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dotB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    # Tail reduction: 1-D blocks of 4 contiguous FP32 per lane.
    FIX_BLOCK: gl.constexpr = 4 * 64 * NUM_WARPS
    FIX_BLOCKS: gl.constexpr = (TILE_ELEMS + FIX_BLOCK - 1) // FIX_BLOCK
    fixL: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
    FIX_UNROLL: gl.constexpr = 16

    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_M, BLOCK_K], sharedA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_K, BLOCK_N], sharedB
    )
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadA))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadB))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadB))
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    c_offs = offs_cm[:, None] * stride_cm + offs_cn[None, :]
    p_offs = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]
    offs_f = gl.arange(0, FIX_BLOCK, fixL)

    # The tail piece (if any) runs first and only publishes its partial; the
    # data-parallel rounds follow, and the tail tile is reduced at the end. By
    # then every piece has long arrived, so the wait does not stall, and the
    # reduction of the partials is the only extra latency.
    has_tail = pid < tail * PIECES
    tt = pid // PIECES
    piece = pid - tt * PIECES
    for w in range(0, n_items):
        if has_tail & (w == 0):
            tile = rounds * GRID + tt
            k_lo = piece * KT // PIECES
            n_k = (piece + 1) * KT // PIECES - k_lo
        else:
            tile = (w - has_tail.to(gl.int32)) * GRID + pid
            k_lo = 0
            n_k = KT
        pid_m, pid_n = _pids(tile, num_pid_m, num_pid_n, 1, GROUP_SIZE_M)

        # Rows past M and weight rows past N clamp onto the last valid one
        # (the store mask drops them), so no load needs a mask.
        k0 = k_lo * BLOCK_K
        a_base = a_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_am + k0
        b_base = b_ptr + (pid_n * BLOCK_N).to(gl.int64) * stride_bn + k0
        rows = gl.minimum(offs_am, M - 1 - pid_m * BLOCK_M)
        a_offs = rows[:, None] * stride_am + offs_ak[None, :]
        if N % BLOCK_N == 0:
            b_offs = offs_bk[:, None] + offs_bn[None, :] * stride_bn
        else:
            cols = gl.minimum(offs_bn, N - 1 - pid_n * BLOCK_N)
            b_offs = offs_bk[:, None] + cols[None, :] * stride_bn

        # Every wave must be done reading the previous item's LDS slots
        # before this item's prologue overwrites them.
        gl.barrier()
        for s in gl.static_range(NUM_STAGES):
            async_copy.buffer_load_to_shared(
                smemA.index(s), a_base, a_offs + s * BLOCK_K
            )
            async_copy.buffer_load_to_shared(
                smemB.index(s), b_base, b_offs + s * BLOCK_K
            )
            async_copy.commit_group()
        async_copy.wait_group(NUM_STAGES - 1)
        a = smemA.index(0).load(dotA)
        b = smemB.index(0).load(dotB)
        acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
        for i in range(0, n_k - NUM_STAGES):
            cur = i % NUM_STAGES
            nxt = (i + 1) % NUM_STAGES
            kofs = (i + NUM_STAGES) * BLOCK_K
            acc = cdna4.mfma(a, b, acc, cd_regclass=CD_REGCLASS)
            async_copy.wait_group(NUM_STAGES - 2)
            a = smemA.index(nxt).load(dotA)
            b = smemB.index(nxt).load(dotB)
            async_copy.buffer_load_to_shared(smemA.index(cur), a_base, a_offs + kofs)
            async_copy.buffer_load_to_shared(smemB.index(cur), b_base, b_offs + kofs)
            async_copy.commit_group()
        for j in gl.static_range(1, NUM_STAGES):
            acc = cdna4.mfma(a, b, acc, cd_regclass=CD_REGCLASS)
            async_copy.wait_group(NUM_STAGES - 1 - j)
            slot = (n_k - NUM_STAGES + j) % NUM_STAGES
            a = smemA.index(slot).load(dotA)
            b = smemB.index(slot).load(dotB)
        acc = cdna4.mfma(a, b, acc, cd_regclass=CD_REGCLASS)

        if has_tail & (w == 0) & (PIECES > 1):
            cdna4.buffer_store(
                stored_value=acc,
                ptr=partial_ptr + (tt * PIECES + piece).to(gl.int64) * TILE_ELEMS,
                offsets=p_offs,
                cache=".wt",
            )
            # Publish the partial: wait for this wave's stores, then order all
            # waves' stores before the arrival.
            gl.inline_asm_elementwise(
                asm="s_waitcnt vmcnt(0)",
                constraints="=v,0",
                args=[offs_cm],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            gl.barrier()
            gl.atomic_add(counter_ptr + tt, 1, sem="relaxed", scope="gpu")
        else:
            sm = pid_m * BLOCK_M + offs_cm
            sn = pid_n * BLOCK_N + offs_cn
            c_base = (
                c_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_cm + pid_n * BLOCK_N
            )
            cdna4.buffer_store(
                stored_value=acc.to(c_ptr.dtype.element_ty),
                ptr=c_base,
                offsets=c_offs,
                mask=(sm[:, None] < M) & (sn[None, :] < N),
            )

    if has_tail & (PIECES > 1):
        arrived = gl.atomic_add(counter_ptr + tt, 0, sem="acquire", scope="gpu")
        while arrived < PIECES:
            arrived = gl.atomic_add(counter_ptr + tt, 0, sem="acquire", scope="gpu")
        pid_m, pid_n = _pids(
            rounds * GRID + tt, num_pid_m, num_pid_n, 1, GROUP_SIZE_M
        )
        c_base = c_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_cm + pid_n * BLOCK_N
        p_tile = partial_ptr + (tt * PIECES).to(gl.int64) * TILE_ELEMS
        for blk in range(piece, FIX_BLOCKS, PIECES):
            e = blk * FIX_BLOCK + offs_f
            out = gl.zeros((FIX_BLOCK,), gl.float32, fixL)
            # FIX_UNROLL independent loads in flight; sources summed in order.
            for s0 in range(0, PIECES, FIX_UNROLL):
                for u in gl.static_range(FIX_UNROLL):
                    out += cdna4.buffer_load(
                        ptr=p_tile,
                        offsets=e + (s0 + u) * TILE_ELEMS,
                        mask=(e < TILE_ELEMS) & (s0 + u < PIECES),
                        other=0.0,
                        cache=".cv",
                    )
            er = e // BLOCK_N
            ec = e % BLOCK_N
            cdna4.buffer_store(
                stored_value=out.to(c_ptr.dtype.element_ty),
                ptr=c_base,
                offsets=er * stride_cm + ec,
                mask=(pid_m * BLOCK_M + er < M)
                & (pid_n * BLOCK_N + ec < N)
                & (e < TILE_ELEMS),
            )
        # Arrival 2 * PIECES - 1: every piece has left the wait above.
        done = gl.atomic_add(counter_ptr + tt, 1, sem="relaxed", scope="gpu")
        if done == 2 * PIECES - 1:
            gl.store(counter_ptr + tt, 0)


def config_fits(cfg: tuple, M: int, N: int, K: int) -> bool:
    """Whether a (BM, BN, BK, WM, WN, NS, GRID, XCD) config is legal."""
    bm, bn, bk, wm, wn, ns, grid, xcd = cfg[:8]
    nw = wm * wn
    if nw not in (4, 8) or bm % (16 * wm) or bn % (16 * wn):
        return False
    if K % bk or K // bk < ns or grid % xcd:
        return False
    if ns * (bm + bn) * bk * 2 * 33 // 32 > LDS_BYTES:
        return False
    lane_row_bits = 6 - (_log2(bk) - 3)
    for rows in (bm, bn):
        if _log2(rows) < _log2(nw) + lane_row_bits:
            return False
    return True


def tail_pieces(cfg: tuple, M: int, N: int, K: int, max_pieces: int = 32) -> int:
    """K pieces per tail tile: fill the grid, keep NUM_STAGES K tiles per piece."""
    bm, bn, bk, _, _, ns, grid, _ = cfg[:8]
    tail = triton.cdiv(M, bm) * triton.cdiv(N, bn) % grid
    if tail == 0:
        return 1
    return max(1, min(grid // tail, (K // bk) // ns, max_pieces))


_scratch: dict = {}
_cus: dict = {}


def _num_cus(device: torch.device) -> int:
    if device not in _cus:
        _cus[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _cus[device]


def launch_gluon_mm_a16w16_persistent_gfx950(
    A: torch.Tensor,
    W: torch.Tensor,
    C: torch.Tensor,
    cfg: tuple,
    group_size_m: int = 4,
    max_pieces: int = 32,
    schedule_hint: str = "",
    pin_acc: bool = False,
) -> torch.Tensor:
    """C[M, N] = A[M, K] @ W[N, K].T with config (BM, BN, BK, WM, WN, NS, GRID, XCD)."""
    bm, bn, bk, wm, wn, ns, grid, xcd = cfg[:8]
    M, K = A.shape
    N = W.shape[0]
    assert A.stride(1) == 1 and W.stride(1) == 1 and C.stride(1) == 1
    # The tail reduction spins until all pieces of a tile have arrived, so
    # every program must be resident at once.
    assert grid <= _num_cus(A.device), f"GRID {grid} exceeds the CU count"
    pieces = tail_pieces(cfg, M, N, K, max_pieces)
    key = (A.device, grid, bm * bn)
    if key not in _scratch:
        # At most GRID tail pieces, at most GRID tail tiles.
        _scratch[key] = (
            torch.empty(grid * bm * bn, device=A.device, dtype=torch.float32),
            torch.zeros(grid, device=A.device, dtype=torch.int32),
        )
    partial, counters = _scratch[key]
    gluon_mm_a16w16_persistent_gfx950[(grid,)](
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
        pieces,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        WARPS_M=wm,
        WARPS_N=wn,
        NUM_STAGES=ns,
        GRID=grid,
        NUM_XCDS=xcd,
        GROUP_SIZE_M=group_size_m,
        CD_REGCLASS="a" if pin_acc else None,
        num_warps=wm * wn,
        # pin_acc keeps the accumulators in AGPRs (tutorial cd_regclass="a");
        # otherwise forbid AGPRs so nothing is copied between register files.
        llvm_fn_attrs=() if pin_acc else (("amdgpu-agpr-alloc", "0,0"),),
        schedule_hint=schedule_hint,
    )
    return C
