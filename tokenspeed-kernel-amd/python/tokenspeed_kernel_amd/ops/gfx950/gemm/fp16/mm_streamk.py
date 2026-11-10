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

"""Experimental Stream-K, multi-stage dense16 Gluon GEMM for gfx950.

``C = A @ W.T`` for skinny ``A [M, K]`` (M = 64..512) against a large
``W [N, K]``. These GEMMs are bound by the weight stream, and the gfx950
bandwidth model (gfx950-gluon-tutorials, docs/memory_bandwidth_model.md) caps
in-flight loads per CU, so bandwidth scales with the number of busy CUs. A
one-tile-per-program grid of 129..272 tiles leaves CUs idle or a ragged tail.

Here a persistent grid of ``GRID`` programs splits the flattened
``(tile, k_tile)`` iteration space (tiles m-fastest, so programs that share a
weight block run together) into equal contiguous ranges (Stream-K). Each
program runs one continuous ``NUM_STAGES``-deep pipeline across its range:
loads for the next tile are already in flight while the current one finishes,
so tile boundaries cost no pipeline refill. A tile computed by one program is
stored directly. A tile shared by several programs is reduced by the last
arriver from FP32 partials, ordered by a per-tile counter that it resets.
The reduction runs after the main loop, never inside it: inside, it keeps a
second FP32 tile live next to the accumulator and the operand registers, and
large tiles spilled to scratch inside the MFMA loop.

LDS layout, global-load layout and MFMA operand layouts are those of
mm_3stage.py (tutorial padding [[512, 16]]).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_3stage import (
    LDS_BYTES,
    _gload_layout,
    _launch_metadata,
    _log2,
    _shared_layout,
)

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy


@gluon.jit
def _addr(
    g,
    num_pid_m,
    M,
    a_ptr,
    b_ptr,
    stride_am,
    stride_bn: gl.constexpr,
    offs_am,
    offs_ak,
    offs_bk,
    offs_bn,
    N: gl.constexpr,
    KT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    tile = g // KT
    kk = g % KT
    pid_m = tile % num_pid_m
    pid_n = tile // num_pid_m
    a_base = a_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_am + kk * BLOCK_K
    b_base = b_ptr + (pid_n * BLOCK_N).to(gl.int64) * stride_bn + kk * BLOCK_K
    rows = gl.minimum(offs_am, M - 1 - pid_m * BLOCK_M)
    a_offs = rows[:, None] * stride_am + offs_ak[None, :]
    if N % BLOCK_N == 0:
        b_offs = offs_bk[:, None] + offs_bn[None, :] * stride_bn
    else:
        cols = gl.minimum(offs_bn, N - 1 - pid_n * BLOCK_N)
        b_offs = offs_bk[:, None] + cols[None, :] * stride_bn
    return a_base, a_offs, b_base, b_offs


@gluon.jit
def _reduce(
    tile,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    N: gl.constexpr,
    stride_cm,
    num_pid_m,
    per,
    KT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mfma: gl.constexpr,
):
    """Sum the FP32 partials of a tile into C and reset its counter."""
    TILE_ELEMS: gl.constexpr = BLOCK_M * BLOCK_N
    pid_m = tile % num_pid_m
    pid_n = tile // num_pid_m
    tile_start = tile * KT
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    p_offs = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]
    out = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    for w in range(tile_start // per, (tile_start + KT - 1) // per + 1):
        # Slot 0 holds a program's first segment, slot 1 its last.
        ws = (w * per < tile_start).to(gl.int32)
        out += cdna4.buffer_load(
            ptr=partial_ptr + ((w * 2 + ws) * TILE_ELEMS).to(gl.int64),
            offsets=p_offs,
            cache=".cv",
        )
    gl.store(counter_ptr + tile, 0)
    sm = pid_m * BLOCK_M + offs_cm
    sn = pid_n * BLOCK_N + offs_cn
    cdna4.buffer_store(
        stored_value=out.to(c_ptr.dtype.element_ty),
        ptr=c_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_cm + pid_n * BLOCK_N,
        offsets=offs_cm[:, None] * stride_cm + offs_cn[None, :],
        mask=(sm[:, None] < M) & (sn[None, :] < N),
    )


@gluon.jit
def _is_last_arrival(tile, counter_ptr, per, KT: gl.constexpr, dep):
    """Count one partial of a tile; true for the last of its programs."""
    gl.inline_asm_elementwise(
        asm="s_waitcnt vmcnt(0)",
        constraints="=v,0",
        args=[dep],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()
    tile_start = tile * KT
    w_lo = tile_start // per
    w_hi = (tile_start + KT - 1) // per
    old = gl.atomic_add(counter_ptr + tile, 1, sem="relaxed", scope="gpu")
    return old == w_hi - w_lo


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_mm_a16w16_streamk_gfx950(
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
    NUM_XCDS: gl.constexpr,
    CD_REGCLASS: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    KT: gl.constexpr = K // BLOCK_K
    gl.static_assert(K % BLOCK_K == 0, "K must be a multiple of BLOCK_K")
    gl.static_assert(NUM_STAGES >= 2, "pipeline needs two or more stages")
    TILE_ELEMS: gl.constexpr = BLOCK_M * BLOCK_N

    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    if NUM_XCDS > 1:
        # Program i runs on XCD i % NUM_XCDS: hand each XCD a contiguous block
        # of work ranges so neighbouring tiles share that XCD's L2.
        pid = (pid % NUM_XCDS) * (grid // NUM_XCDS) + pid // NUM_XCDS
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    total = num_pid_m * num_pid_n * KT
    per = (total + grid - 1) // grid
    start = pid * per
    end = gl.minimum(start + per, total)
    n = end - start
    if n <= 0:
        return

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
    p_offs = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]

    # Prologue. Iterations past the range reload the last one (clamped) so
    # every iteration commits exactly one group; only NUM_STAGES - 1 such
    # redundant tiles per program, served from L2.
    for s in gl.static_range(NUM_STAGES):
        ab, ao, bb, bo = _addr(
            gl.minimum(start + s, end - 1),
            num_pid_m,
            M,
            a_ptr,
            b_ptr,
            stride_am,
            stride_bn,
            offs_am,
            offs_ak,
            offs_bk,
            offs_bn,
            N,
            KT,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
        async_copy.buffer_load_to_shared(smemA.index(s), ab, ao)
        async_copy.buffer_load_to_shared(smemB.index(s), bb, bo)
        async_copy.commit_group()
    async_copy.wait_group(NUM_STAGES - 1)
    a = smemA.index(0).load(dotA)
    b = smemB.index(0).load(dotB)
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)

    # Load cursor: addresses of the last issued load. Within a tile only the
    # K offset advances, so the tile index divisions and the 2-D offset
    # tensors are recomputed only when the cursor crosses a tile boundary.
    lg = gl.minimum(start + NUM_STAGES - 1, end - 1)
    ab, ao, bb, bo = _addr(
        lg,
        num_pid_m,
        M,
        a_ptr,
        b_ptr,
        stride_am,
        stride_bn,
        offs_am,
        offs_ak,
        offs_bk,
        offs_bn,
        N,
        KT,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    lkk = lg % KT
    ckk = start % KT

    for i in range(0, n):
        acc = cdna4.mfma(a, b, acc, cd_regclass=CD_REGCLASS)
        async_copy.wait_group(NUM_STAGES - 2)
        a = smemA.index((i + 1) % NUM_STAGES).load(dotA)
        b = smemB.index((i + 1) % NUM_STAGES).load(dotB)
        gn = start + i + NUM_STAGES
        if gn < end:
            if lkk == KT - 1:
                ab, ao, bb, bo = _addr(
                    gn,
                    num_pid_m,
                    M,
                    a_ptr,
                    b_ptr,
                    stride_am,
                    stride_bn,
                    offs_am,
                    offs_ak,
                    offs_bk,
                    offs_bn,
                    N,
                    KT,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                )
                lkk = lkk - (KT - 1)
            else:
                ab = ab + BLOCK_K
                bb = bb + BLOCK_K
                lkk = lkk + 1
        # Past the range the last load is reissued (L2 hit) so every
        # iteration commits exactly one group.
        async_copy.buffer_load_to_shared(smemA.index(i % NUM_STAGES), ab, ao)
        async_copy.buffer_load_to_shared(smemB.index(i % NUM_STAGES), bb, bo)
        async_copy.commit_group()

        g = start + i
        tile_end = ckk == KT - 1
        ckk = (ckk + 1) % KT
        if tile_end | (i == n - 1):
            tile = g // KT
            pid_m = tile % num_pid_m
            pid_n = tile // num_pid_m
            tile_start = tile * KT
            sm = pid_m * BLOCK_M + offs_cm
            sn = pid_n * BLOCK_N + offs_cn
            c_mask = (sm[:, None] < M) & (sn[None, :] < N)
            c_base = (
                c_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_cm + pid_n * BLOCK_N
            )
            c_offs = offs_cm[:, None] * stride_cm + offs_cn[None, :]
            if (tile_start >= start) & (g == tile_start + KT - 1):
                cdna4.buffer_store(
                    stored_value=acc.to(c_ptr.dtype.element_ty),
                    ptr=c_base,
                    offsets=c_offs,
                    mask=c_mask,
                )
            else:
                # Slot 0 holds a program's first segment (the tile holding
                # its start), slot 1 its last; a middle segment is a full tile.
                slot = (tile_start > start).to(gl.int32)
                cdna4.buffer_store(
                    stored_value=acc,
                    ptr=partial_ptr + ((pid * 2 + slot) * TILE_ELEMS).to(gl.int64),
                    offsets=p_offs,
                    cache=".wt",
                )
                # A program's first segment arrives now, so the program before
                # it, which finishes that tile at the end of its range, is
                # usually the last arriver. In the rare case this program is,
                # it marks the counter with -(pid + 1) and reduces the tile
                # after the loop, so no extra state is carried by the loop.
                if slot == 0:
                    if _is_last_arrival(tile, counter_ptr, per, KT, offs_cm):
                        gl.atomic_xchg(
                            counter_ptr + tile, -(pid + 1), sem="relaxed", scope="gpu"
                        )
            acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)

    # Reduce the first tile if this program marked it in the loop.
    first = start // KT
    first_partial = (start > first * KT) | (end < (first + 1) * KT)
    if first_partial:
        mark = gl.atomic_add(counter_ptr + first, 0, sem="relaxed", scope="gpu")
    else:
        mark = 0
    if mark == -(pid + 1):
        _reduce(
            first,
            c_ptr,
            partial_ptr,
            counter_ptr,
            M,
            N,
            stride_cm,
            num_pid_m,
            per,
            KT,
            BLOCK_M,
            BLOCK_N,
            mfma,
        )
    # The last segment is partial when the range ends inside a tile that
    # started before it; it arrives here, after the loop.
    last = (end - 1) // KT
    if (last != first) & (end < (last + 1) * KT):
        if _is_last_arrival(last, counter_ptr, per, KT, offs_cm):
            _reduce(
                last,
                c_ptr,
                partial_ptr,
                counter_ptr,
                M,
                N,
                stride_cm,
                num_pid_m,
                per,
                KT,
                BLOCK_M,
                BLOCK_N,
                mfma,
            )


def config_fits(cfg: tuple, M: int, N: int, K: int) -> bool:
    """Whether a (BM, BN, BK, WM, WN, NS, GRID, XCD) config is legal."""
    bm, bn, bk, wm, wn, ns, grid, xcd = cfg
    nw = wm * wn
    if nw not in (4, 8) or bm % (16 * wm) or bn % (16 * wn):
        return False
    if K % bk or grid % xcd:
        return False
    if ns * (bm + bn) * bk * 2 * 33 // 32 > LDS_BYTES:
        return False
    lane_row_bits = 6 - (_log2(bk) - 3)
    for rows in (bm, bn):
        if _log2(rows) < _log2(nw) + lane_row_bits:
            return False
    return True


_scratch: dict = {}


def launch_gluon_mm_a16w16_streamk_gfx950(
    A: torch.Tensor,
    W: torch.Tensor,
    C: torch.Tensor,
    cfg: tuple,
    schedule_hint: str = "",
    pin_acc: bool = False,
) -> torch.Tensor:
    """C[M, N] = A[M, K] @ W[N, K].T with config (BM, BN, BK, WM, WN, NS, GRID, XCD)."""
    bm, bn, bk, wm, wn, ns, grid, xcd = cfg
    M, K = A.shape
    N = W.shape[0]
    assert A.stride(1) == 1 and W.stride(1) == 1 and C.stride(1) == 1
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    key = (A.device, grid, bm * bn, tiles)
    if key not in _scratch:
        _scratch[key] = (
            torch.empty(grid * 2 * bm * bn, device=A.device, dtype=torch.float32),
            torch.zeros(tiles, device=A.device, dtype=torch.int32),
        )
    partial, counters = _scratch[key]
    gluon_mm_a16w16_streamk_gfx950[(grid,)](
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
        NUM_XCDS=xcd,
        CD_REGCLASS="a" if pin_acc else None,
        num_warps=wm * wn,
        # pin_acc keeps the accumulators in AGPRs (tutorial cd_regclass="a");
        # otherwise forbid AGPRs so nothing is copied between register files.
        llvm_fn_attrs=() if pin_acc else (("amdgpu-agpr-alloc", "0,0"),),
        schedule_hint=schedule_hint,
    )
    return C
