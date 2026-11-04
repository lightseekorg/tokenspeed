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

"""K-split BF16 decode GEMV task bodies for gfx950 (up to 16 rows).

A task owns ``ROWS_PER_TASK`` weight rows (a multiple of 16) and one of
``PARTS`` slices of K; task ``t`` is row group ``t % GROUPS`` and K slice
``t // GROUPS``. Each of the 8 waves owns a contiguous ``K / PARTS / 8`` range.
Weights load straight into the 16x16x32 MFMA A-operand layout and the task's
x slice into the B operand, so no operand passes through LDS. A program reads
``rows x K / PARTS`` activation elements instead of all of x, which is what
keeps a full-K decode GEMV flat in N at 8-16 rows.

The finish sums the waves' accumulators through LDS behind one barrier and
writes one FP32 partial per (part, token, output row) as a 64-bit word: the
value's bits low, a caller tag high, written by one store. A reader polls the
tag half, then loads the word, so it never sees a half-written pair. Tags must
differ between consecutive uses of the buffer: readers compare equality.

The parts are summed in part order, so results are deterministic, but they are
not bit-equal to a full-K GEMV.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (
    MAX_VMCNT_WAIT,
    poll_expired,
    poll_stop_site,
    release_drain,
    vote,
    wait_vmcnt,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent.timing import realtime_ticks

cdna4 = gl.amd.cdna4

NUM_WARPS = 8
_MAX_VMCNT_WAIT = gl.constexpr(MAX_VMCNT_WAIT)
# Token tile of the MFMA B operand; rows past the call's rows reread its last row.
_BLOCK_T = gl.constexpr(16)


@gluon.constexpr_function
def _mfma(waves: int):
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[waves, 1, 1],
    )


@gluon.jit
def _pace_wait(OUTSTANDING: gl.constexpr):
    """Wait until at most ``OUTSTANDING`` of this wave's vector memory
    operations are in flight (a window of one: until none are)."""
    if OUTSTANDING == 0:
        release_drain()
    else:
        wait_vmcnt(OUTSTANDING)


@gluon.jit
def gemv_a16w16_splitk_weights_gfx950(
    w_ptr,
    task,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
):
    """Issue task ``task``'s weight tiles; returns them (one 16-row tile each).

    Each lane holds 8 consecutive K values of one row per tile and K step:
    ``ROWS_PER_TASK / 16 * K / PARTS / 8 / 32 * 4`` VGPRs (72 for 144 rows and
    K = 8192 in 16 parts). Loads are ``.cg`` (streamed once).
    """
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    base, group = _weight_tile_base(
        task, K, ROWS_PER_TASK, PARTS, GROUPS, K // PARTS // gl.num_warps()
    )
    tiles = ()
    for t in gl.static_range(TILES):
        tiles += (
            cdna4.buffer_load(w_ptr, base + (group * TILES + t) * 16 * K, cache=".cg"),
        )
    return tiles


@gluon.jit
def gemv_a16w16_splitk_weights_paced_gfx950(
    w_ptr,
    task,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    WINDOW: gl.constexpr,
):
    """:func:`gemv_a16w16_splitk_weights_gfx950` with at most ``WINDOW``
    tiles' loads in flight per wave: before issuing tile ``t >= WINDOW`` the
    wave waits until ``(WINDOW - 1)`` tiles' loads remain (a counted
    ``s_waitcnt vmcnt``, never ``vmcnt(0)``). A stream that overlaps a
    latency-bound chain stays below the fabric's knee this way; the rate is
    about ``waves * WINDOW * tile bytes`` per load latency per CU. Loads the
    wave issued earlier count against the window too.
    """
    WAVES: gl.constexpr = gl.num_warps()
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    # dwordx4 loads per wave per tile: 16 rows x K / PARTS / WAVES BF16 over 64 lanes.
    LOADS: gl.constexpr = K // PARTS // WAVES // 32
    # WINDOW >= TILES issues every tile at once (no wait).
    gl.static_assert(
        1 <= WINDOW and (WINDOW >= TILES or (WINDOW - 1) * LOADS <= _MAX_VMCNT_WAIT)
    )
    base, group = _weight_tile_base(
        task, K, ROWS_PER_TASK, PARTS, GROUPS, K // PARTS // gl.num_warps()
    )
    tiles = ()
    for t in gl.static_range(TILES):
        if t >= WINDOW:
            _pace_wait((WINDOW - 1) * LOADS)
        tiles += (
            cdna4.buffer_load(w_ptr, base + (group * TILES + t) * 16 * K, cache=".cg"),
        )
    return tiles


@gluon.jit
def _weight_tile_base(
    task,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    KSPAN: gl.constexpr,
):
    """Element offsets of a task's weight rows in the MFMA A layout, the
    first ``KSPAN`` of each wave's K range, without the row group, and the
    group: tile ``t`` is ``(group * TILES + t) * 16 * K`` past them."""
    WAVES: gl.constexpr = gl.num_warps()
    KP: gl.constexpr = K // PARTS
    KW: gl.constexpr = KP // WAVES
    gl.static_assert(ROWS_PER_TASK % 16 == 0 and KW % 32 == 0)
    a_l: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=_mfma(WAVES), k_width=8
    )
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    row = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    k = gl.arange(0, KSPAN, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    group = task % GROUPS
    part = task // GROUPS
    base = (
        row[None, :, None] * K
        + (part * KP + wave * KW)[:, None, None]
        + k[None, None, :]
    )
    return base, group


@gluon.jit
def gemv_a16w16_splitk_x_gfx950(
    x_ptr,
    x_stride,
    task,
    M,
    K: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    X_CACHE: gl.constexpr,
):
    """Load task ``task``'s slice of x as the MFMA B operand.

    Returns ``[waves, K / PARTS / waves, 16]`` BF16: wave w's contiguous K
    range of the task's part, for 16 token slots; slots past ``M`` reread
    token ``M - 1``. ``X_CACHE`` as in :func:`gemv_a16w16_splitk_finish_gfx950`.
    """
    WAVES: gl.constexpr = gl.num_warps()
    KP: gl.constexpr = K // PARTS
    KW: gl.constexpr = KP // WAVES
    b_l: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=_mfma(WAVES), k_width=8
    )
    part = task // GROUPS
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, b_l)))
    k = gl.arange(0, KW, gl.SliceLayout(0, gl.SliceLayout(2, b_l)))
    token = gl.minimum(
        gl.arange(0, _BLOCK_T, gl.SliceLayout(0, gl.SliceLayout(1, b_l))), M - 1
    )
    return cdna4.buffer_load(
        x_ptr,
        token[None, None, :] * x_stride
        + (part * KP + wave * KW)[:, None, None]
        + k[None, :, None],
        cache=X_CACHE,
    )


@gluon.jit
def gemv_a16w16_splitk_finish_gfx950(
    tiles,
    x_ptr,
    x_stride,
    part_ptr,
    tag,
    task,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    X_CACHE: gl.constexpr,
    PART_CACHE: gl.constexpr,
):
    """Multiply task ``task``'s tiles by its x slice and publish its partials.

    ``part_ptr`` is an int64 buffer laid out ``[PARTS, M, N]``; each word holds
    the FP32 partial's bits low and ``tag`` (int32, never equal to a stale
    word's) high. ``X_CACHE``/``PART_CACHE`` are the x load and partial store
    cache modifiers: "" in a standalone launch, ".cv"/".wt" when x was written
    and the partials are read within the same launch on other XCDs.
    """
    x = gemv_a16w16_splitk_x_gfx950(x_ptr, x_stride, task, M, K, PARTS, GROUPS, X_CACHE)
    gemv_a16w16_splitk_finish_x_gfx950(
        tiles, x, part_ptr, tag, task, M, N, ROWS_PER_TASK, GROUPS, PART_CACHE
    )


@gluon.jit
def _store_partials(
    bufs, part_ptr, tag, group, part, M, N: gl.constexpr, PART_CACHE: gl.constexpr
):
    """After one barrier, sum each tile's ``[waves, 16, 16]`` accumulators in
    ``bufs`` over the waves and store them as the finish's tagged words."""
    WAVES: gl.constexpr = gl.num_warps()
    TILES: gl.constexpr = len(bufs)
    gl.barrier()
    sum_l: gl.constexpr = gl.BlockedLayout(
        [WAVES, 1, 1], [1, 4, 16], [1, WAVES // 2, 2], [2, 1, 0]
    )
    out_l: gl.constexpr = gl.SliceLayout(0, sum_l)
    r = gl.arange(0, 16, gl.SliceLayout(1, out_l))
    c = gl.arange(0, _BLOCK_T, gl.SliceLayout(0, out_l))
    for t in gl.static_range(TILES):
        value = gl.sum(bufs[t].load(sum_l), axis=0)
        rows = (group * TILES + t) * 16 + r
        word = (tag.to(gl.int64) << 32) | (
            value.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF
        )
        cdna4.buffer_store(
            word,
            part_ptr,
            (part * M + c[None, :]) * N + rows[:, None],
            mask=(c < M)[None, :] & (rows < N)[:, None],
            cache=PART_CACHE,
        )


@gluon.jit
def gemv_a16w16_splitk_finish_x_gfx950(
    tiles,
    x,
    part_ptr,
    tag,
    task,
    M,
    N: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    GROUPS: gl.constexpr,
    PART_CACHE: gl.constexpr,
):
    """:func:`gemv_a16w16_splitk_finish_gfx950` on a B operand ``x`` the
    caller built (the layout of :func:`gemv_a16w16_splitk_x_gfx950`)."""
    WAVES: gl.constexpr = gl.num_warps()
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    mfma: gl.constexpr = _mfma(WAVES)
    group = task % GROUPS
    part = task // GROUPS
    # Every wave's accumulator of every tile to LDS, one barrier, then each
    # thread sums its (row, token) outputs over the waves.
    bufs = ()
    for t in gl.static_range(TILES):
        buf = gl.allocate_shared_memory(
            gl.float32,
            [WAVES, 16, _BLOCK_T],
            gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]),
        )
        buf.store(
            cdna4.mfma(tiles[t], x, gl.zeros((WAVES, 16, _BLOCK_T), gl.float32, mfma))
        )
        bufs += (buf,)
    _store_partials(bufs, part_ptr, tag, group, part, M, N, PART_CACHE)


@gluon.jit
def gemv_a16w16_splitk_exchange_gfx950(accs, OUT_LAYOUT: gl.constexpr):
    """Sum the waves' MFMA accumulators of a task's tiles.

    ``accs`` holds one ``[waves, 16 rows, 16 tokens]`` FP32 accumulator per
    16-row tile (a power-of-two count), each wave's over its own K range.
    They go to one LDS buffer ``[waves, 16 tokens, rows]`` with the rows
    fastest, so after one barrier a thread of ``OUT_LAYOUT`` reads its
    contiguous rows of every wave with wide LDS reads and adds the waves in
    wave order in its own registers.

    ``OUT_LAYOUT`` is a rank-3 ``[waves, 16, rows]`` blocked layout holding
    every wave in-thread (``size_per_thread[0] == waves``). Returns the FP32
    sums ``[16 tokens, rows]`` in ``gl.SliceLayout(0, OUT_LAYOUT)``.
    """
    WAVES: gl.constexpr = gl.num_warps()
    TILES: gl.constexpr = len(accs)
    partials = gl.allocate_shared_memory(
        gl.float32,
        [WAVES, _BLOCK_T, 16 * TILES],
        gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]),
    )
    for t in gl.static_range(TILES):
        partials.slice(16 * t, 16, 2).store(gl.permute(accs[t], [0, 2, 1]))
    gl.barrier()
    return gl.sum(partials.load(OUT_LAYOUT), axis=0)


@gluon.jit
def gemv_a16w16_splitk_wave_sums_gfx950(tiles, x, OUT_LAYOUT: gl.constexpr):
    """Multiply resident ``tiles`` (:func:`gemv_a16w16_splitk_weights_gfx950`)
    by their B operand ``x`` and sum the waves
    (:func:`gemv_a16w16_splitk_exchange_gfx950`): ``[16 tokens, rows]`` FP32
    in ``gl.SliceLayout(0, OUT_LAYOUT)``."""
    mfma: gl.constexpr = _mfma(gl.num_warps())
    accs = ()
    for t in gl.static_range(len(tiles)):
        accs += (
            cdna4.mfma(
                tiles[t], x, gl.zeros((gl.num_warps(), 16, _BLOCK_T), gl.float32, mfma)
            ),
        )
    return gemv_a16w16_splitk_exchange_gfx950(accs, OUT_LAYOUT)


@gluon.jit
def gemv_a16w16_splitk_hi_lo_sums_gfx950(hi, lo, x, OUT_LAYOUT: gl.constexpr):
    """As :func:`gemv_a16w16_splitk_wave_sums_gfx950` for an FP32 weight
    stored as BF16 ``hi`` + ``lo`` tiles and an FP32 B operand ``x``: each
    tile accumulates ``hi xh + lo xh + hi xl`` on BF16 MFMAs, with ``xh =
    bf16(x)`` and ``xl = bf16(x - xh)`` (the ``lo xl`` term, ~2^-16 of the
    product, is dropped)."""
    mfma: gl.constexpr = _mfma(gl.num_warps())
    xh = x.to(gl.bfloat16)
    xl = (x - xh.to(gl.float32)).to(gl.bfloat16)
    accs = ()
    for t in gl.static_range(len(hi)):
        acc = cdna4.mfma(
            hi[t], xh, gl.zeros((gl.num_warps(), 16, _BLOCK_T), gl.float32, mfma)
        )
        acc = cdna4.mfma(lo[t], xh, acc)
        accs += (cdna4.mfma(hi[t], xl, acc),)
    return gemv_a16w16_splitk_exchange_gfx950(accs, OUT_LAYOUT)


# A 32-deep K step: one 16x16x32 MFMA per tile.
_STEP_K = gl.constexpr(32)


@gluon.constexpr_function
def gemv_a16w16_splitk_step_layout(waves: int):
    """The MFMA B operand layout of one 32-deep K step, ``[waves, 32, 16]``."""
    return gl.DotOperandLayout(operand_index=1, parent=_mfma(waves), k_width=8)


@gluon.jit
def gemv_a16w16_splitk_weight_steps_gfx950(
    w_ptr,
    task,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    WINDOW: gl.constexpr,
):
    """:func:`gemv_a16w16_splitk_weights_paced_gfx950` by K step:
    ``tiles[t][s]`` is tile ``t``'s ``[waves, 16, 32]`` BF16 A operand of the
    wave's K step ``s``. The same loads in the same order (one 16-byte load
    per lane per tile and step) and the same pacing (``WINDOW`` at least the
    tile count: no wait), for a caller that builds x one step at a time."""
    return gemv_a16w16_splitk_weight_step_tiles_gfx950(
        w_ptr, task, K, ROWS_PER_TASK, PARTS, GROUPS, WINDOW, 0, ROWS_PER_TASK // 16
    )


@gluon.jit
def gemv_a16w16_splitk_weight_step_tiles_gfx950(
    w_ptr,
    task,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    WINDOW: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
):
    """Tiles ``[FIRST, FIRST + COUNT)`` of
    :func:`gemv_a16w16_splitk_weight_steps_gfx950`, paced as there (a wait
    before each tile ``t >= WINDOW``), so a caller can issue a task's tiles
    in two parts and join the tuples (``first + rest``) into the same
    tiles."""
    STEPS: gl.constexpr = K // PARTS // gl.num_warps() // _STEP_K
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    gl.static_assert(
        1 <= WINDOW and (WINDOW >= TILES or (WINDOW - 1) * STEPS <= _MAX_VMCNT_WAIT)
    )
    gl.static_assert(FIRST + COUNT <= TILES)
    base, group = _weight_tile_base(task, K, ROWS_PER_TASK, PARTS, GROUPS, _STEP_K)
    tiles = ()
    for t in gl.static_range(FIRST, FIRST + COUNT):
        if t >= WINDOW:
            _pace_wait((WINDOW - 1) * STEPS)
        steps = ()
        for step in gl.static_range(STEPS):
            steps += (
                cdna4.buffer_load(
                    w_ptr,
                    base + (group * TILES + t) * 16 * K + step * _STEP_K,
                    cache=".cg",
                ),
            )
        tiles += (steps,)
    return tiles


@gluon.jit
def gemv_a16w16_splitk_weight_steps_windowed_gfx950(
    w_ptr,
    task,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    LOADS: gl.constexpr,
    ISSUED: gl.constexpr,
):
    """:func:`gemv_a16w16_splitk_weight_steps_gfx950` paced by K step: at
    most ``LOADS`` of the stream's step loads (one 16-byte load per lane,
    ``16 * 32 * 2 * waves`` bytes per workgroup) in flight, for streams
    finer than one tile per window (a 16-row router task is one tile).
    ``ISSUED`` is how many of the same stream's loads the wave issued just
    before (a previous call), so two calls pace as one stream."""
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    STEPS: gl.constexpr = K // PARTS // gl.num_warps() // _STEP_K
    gl.static_assert(1 <= LOADS and LOADS - 1 <= _MAX_VMCNT_WAIT)
    base, group = _weight_tile_base(task, K, ROWS_PER_TASK, PARTS, GROUPS, _STEP_K)
    tiles = ()
    for t in gl.static_range(TILES):
        steps = ()
        for step in gl.static_range(STEPS):
            if ISSUED + t * STEPS + step >= LOADS:
                _pace_wait(LOADS - 1)
            steps += (
                cdna4.buffer_load(
                    w_ptr,
                    base + (group * TILES + t) * 16 * K + step * _STEP_K,
                    cache=".cg",
                ),
            )
        tiles += (steps,)
    return tiles


@gluon.jit
def gemv_a16w16_splitk_x_steps_gfx950(
    x_ptr,
    x_stride,
    task,
    M,
    K: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
    X_CACHE: gl.constexpr,
):
    """:func:`gemv_a16w16_splitk_x_gfx950` by K step: a tuple of the wave's
    ``[waves, 32, 16]`` B operands, step ``s`` its K ``[32 s, 32 s + 32)``."""
    WAVES: gl.constexpr = gl.num_warps()
    KP: gl.constexpr = K // PARTS
    KW: gl.constexpr = KP // WAVES
    b_l: gl.constexpr = gemv_a16w16_splitk_step_layout(WAVES)
    part = task // GROUPS
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, b_l)))
    k = gl.arange(0, _STEP_K, gl.SliceLayout(0, gl.SliceLayout(2, b_l)))
    token = gl.minimum(
        gl.arange(0, _BLOCK_T, gl.SliceLayout(0, gl.SliceLayout(1, b_l))), M - 1
    )
    base = (
        token[None, None, :] * x_stride
        + (part * KP + wave * KW)[:, None, None]
        + k[None, :, None]
    )
    steps = ()
    for step in gl.static_range(KW // _STEP_K):
        steps += (cdna4.buffer_load(x_ptr, base + step * _STEP_K, cache=X_CACHE),)
    return steps


@gluon.constexpr_function
def _k_vector_layout(waves: int, kw: int, bits: int):
    # 16 bytes per lane, a wave's 64 lanes over consecutive waves' K ranges, so
    # each wave's LDS-direct copy writes one contiguous run; waves past the
    # rows repeat a copy, which the compiler leaves to one of them.
    vector = 128 // bits
    lanes_k = kw // vector
    return gl.BlockedLayout([1, vector], [64 // lanes_k, lanes_k], [waves, 1], [1, 0])


@gluon.jit
def gemv_a16w16_splitk_k_vector_to_shared_gfx950(
    smem, v_ptr, task, K: gl.constexpr, PARTS: gl.constexpr, GROUPS: gl.constexpr
):
    """Copy task ``task``'s K slice of the per-K vector ``v_ptr`` (FP32 or
    BF16) into ``smem`` ``[waves, K / PARTS / waves]`` (unswizzled), wave
    w's K range in row w, with LDS-direct loads (no registers). The caller
    commits the group and waits for it (``async_copy``) before reading."""
    WAVES: gl.constexpr = gl.num_warps()
    KP: gl.constexpr = K // PARTS
    KW: gl.constexpr = KP // WAVES
    layout: gl.constexpr = _k_vector_layout(
        WAVES, KW, v_ptr.dtype.element_ty.primitive_bitwidth
    )
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, layout))
    k = gl.arange(0, KW, gl.SliceLayout(0, layout))
    cdna4.async_copy.buffer_load_to_shared(
        smem, v_ptr, (task // GROUPS * KP + wave * KW)[:, None] + k[None, :]
    )


@gluon.jit
def gemv_a16w16_splitk_k_vector_step_gfx950(smem, step):
    """K step ``step`` (constexpr) of a vector copied by
    :func:`gemv_a16w16_splitk_k_vector_to_shared_gfx950`, as ``[waves, 32, 1]``
    in the step's B operand layout (broadcast over the tokens)."""
    b_l: gl.constexpr = gemv_a16w16_splitk_step_layout(gl.num_warps())
    chunk = smem.slice(step * _STEP_K, _STEP_K, 1)
    return chunk.load(gl.SliceLayout(2, b_l))[:, :, None]


@gluon.jit
def gemv_a16w16_splitk_b_step_gfx950(smem, step):
    """K step ``step`` (constexpr) of a B operand stored in ``smem``
    ``[waves, K / PARTS / waves, 16]``: ``[waves, 32, 16]`` in the step's
    layout."""
    b_l: gl.constexpr = gemv_a16w16_splitk_step_layout(gl.num_warps())
    return smem.slice(step * _STEP_K, _STEP_K, 1).load(b_l)


@gluon.jit
def gemv_a16w16_splitk_wave_sums_steps_gfx950(
    tiles, X_STEP: gl.constexpr, x_args, OUT_LAYOUT: gl.constexpr
):
    """:func:`gemv_a16w16_splitk_wave_sums_gfx950` with x built one K step at
    a time: ``X_STEP(x_args, s)`` (constexpr ``s``) returns step ``s``'s
    ``[waves, 32, 16]`` BF16 B operand, used by every tile's MFMA before the
    next step's is built, so one step of x and its inputs is live.
    ``tiles`` from :func:`gemv_a16w16_splitk_weight_steps_gfx950`. Each
    tile's MFMAs run in K order on one accumulator, as the full-K dot does,
    so the sums are bitwise the same."""
    mfma: gl.constexpr = _mfma(gl.num_warps())
    accs = ()
    for t in gl.static_range(len(tiles)):
        accs += (gl.zeros((gl.num_warps(), 16, _BLOCK_T), gl.float32, mfma),)
    for step in gl.static_range(len(tiles[0])):
        x = X_STEP(x_args, step)
        next_accs = ()
        for t in gl.static_range(len(tiles)):
            next_accs += (cdna4.mfma(tiles[t][step], x, accs[t]),)
        accs = next_accs
    return gemv_a16w16_splitk_exchange_gfx950(accs, OUT_LAYOUT)


@gluon.jit
def gemv_a16w16_splitk_hi_lo_sums_steps_gfx950(
    hi, lo, X_STEP: gl.constexpr, x_args, OUT_LAYOUT: gl.constexpr
):
    """:func:`gemv_a16w16_splitk_hi_lo_sums_gfx950` with x built one K step
    at a time (``X_STEP`` returns FP32, as in
    :func:`gemv_a16w16_splitk_wave_sums_steps_gfx950`). Only the steps' BF16
    ``xh`` and ``xl`` stay live; each tile still runs ``hi xh`` over all K,
    then ``lo xh``, then ``hi xl`` on one accumulator, so the sums are
    bitwise the same."""
    mfma: gl.constexpr = _mfma(gl.num_warps())
    STEPS: gl.constexpr = len(hi[0])
    accs = ()
    for t in gl.static_range(len(hi)):
        accs += (gl.zeros((gl.num_warps(), 16, _BLOCK_T), gl.float32, mfma),)
    xhs = ()
    xls = ()
    for step in gl.static_range(STEPS):
        x = X_STEP(x_args, step)
        xh = x.to(gl.bfloat16)
        xhs += (xh,)
        xls += ((x - xh.to(gl.float32)).to(gl.bfloat16),)
        next_accs = ()
        for t in gl.static_range(len(hi)):
            next_accs += (cdna4.mfma(hi[t][step], xh, accs[t]),)
        accs = next_accs
    for step in gl.static_range(STEPS):
        next_accs = ()
        for t in gl.static_range(len(hi)):
            next_accs += (cdna4.mfma(lo[t][step], xhs[step], accs[t]),)
        accs = next_accs
    for step in gl.static_range(STEPS):
        next_accs = ()
        for t in gl.static_range(len(hi)):
            next_accs += (cdna4.mfma(hi[t][step], xls[step], accs[t]),)
        accs = next_accs
    return gemv_a16w16_splitk_exchange_gfx950(accs, OUT_LAYOUT)


@gluon.jit
def gemv_a16w16_splitk_finish_steps_gfx950(
    tiles,
    X_STEP: gl.constexpr,
    x_args,
    part_ptr,
    tag,
    task,
    M,
    N: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    GROUPS: gl.constexpr,
    PART_CACHE: gl.constexpr,
):
    """:func:`gemv_a16w16_splitk_finish_x_gfx950` with x built one K step at
    a time, as in :func:`gemv_a16w16_splitk_wave_sums_steps_gfx950`: the
    same MFMA order per tile, so the partial words are bitwise the same."""
    WAVES: gl.constexpr = gl.num_warps()
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    mfma: gl.constexpr = _mfma(WAVES)
    group = task % GROUPS
    part = task // GROUPS
    accs = ()
    for t in gl.static_range(TILES):
        accs += (gl.zeros((WAVES, 16, _BLOCK_T), gl.float32, mfma),)
    for step in gl.static_range(len(tiles[0])):
        x = X_STEP(x_args, step)
        next_accs = ()
        for t in gl.static_range(TILES):
            next_accs += (cdna4.mfma(tiles[t][step], x, accs[t]),)
        accs = next_accs
    bufs = ()
    for t in gl.static_range(TILES):
        buf = gl.allocate_shared_memory(
            gl.float32,
            [WAVES, 16, _BLOCK_T],
            gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]),
        )
        buf.store(accs[t])
        bufs += (buf,)
    _store_partials(bufs, part_ptr, tag, group, part, M, N, PART_CACHE)


@gluon.jit
def gemv_a16w16_splitk_publish_gfx950(
    sums,
    part_ptr,
    tag,
    row0,
    part,
    M,
    N: gl.constexpr,
    PART_CACHE: gl.constexpr,
):
    """Publish ``sums`` (``[16 tokens, R]`` FP32, rows ``row0 ..``) as K part
    ``part``'s partials: the 64-bit words of
    :func:`gemv_a16w16_splitk_finish_gfx950` (value low, ``tag`` high, one
    store each) at ``[part, token, row]`` of ``part_ptr`` ``[PARTS, M, N]``;
    tokens past ``M`` are dropped. :func:`gemv_a16w16_splitk_sum_gfx950`
    reads them."""
    layout: gl.constexpr = sums.type.layout
    c = gl.arange(0, sums.shape[0], gl.SliceLayout(1, layout))
    rows = row0 + gl.arange(0, sums.shape[1], gl.SliceLayout(0, layout))
    word = (tag.to(gl.int64) << 32) | (
        sums.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF
    )
    cdna4.buffer_store(
        word,
        part_ptr,
        (part * M + c[:, None]) * N + rows[None, :],
        mask=(c < M)[:, None] & (rows < N)[None, :],
        cache=PART_CACHE,
    )


@gluon.jit
def _missing(words, tag, live, PARTS: gl.constexpr):
    """Per lane: whether any of its words under ``live`` lacks ``tag`` (axis 1 is in-thread)."""
    missing = gl.zeros(live.shape, gl.int32, live.type.layout)
    for p in gl.static_range(PARTS):
        missing |= (((words[p] >> 32) & 0xFFFFFFFF) != tag).to(gl.int32)
    return gl.max(missing & live, 1)


@gluon.jit
def gemv_a16w16_splitk_sum_gfx950(
    part_ptr,
    tag,
    offsets,
    mask,
    M,
    N: gl.constexpr,
    PARTS: gl.constexpr,
    launch,
    on_expiry: gl.constexpr,
    SITE: gl.constexpr,
):
    """Wait for and sum, in part order, the partials at ``offsets`` (``token * N + row``).

    ``offsets``/``mask`` are 1-D with a multiple of 64 x num_warps elements.
    Each thread reloads its 64-bit words (``.cv``) until every word under
    ``mask`` carries ``tag`` in its high half: a word is one store, so its
    value arrived with its tag. Each wave votes once per retry on what is
    missing and stops at ``launch``'s deadline or abort word, calling
    ``on_expiry(launch, site, 0)``. The loop holds no barrier or LDS.

    Returns:
        FP32 sums with ``offsets``' shape and layout.
    """
    THREADS: gl.constexpr = 64 * gl.num_warps()
    E: gl.constexpr = offsets.shape[0] // THREADS
    gl.static_assert(E * THREADS == offsets.shape[0])
    lanes: gl.constexpr = gl.BlockedLayout([1, E], [64, 1], [gl.num_warps(), 1], [1, 0])
    safe = gl.convert_layout(
        gl.reshape(gl.where(mask, offsets, 0), [THREADS, E]), lanes
    )
    live = gl.convert_layout(gl.reshape(mask.to(gl.int32), [THREADS, E]), lanes)
    words = ()
    for p in gl.static_range(PARTS):
        words += (cdna4.buffer_load(part_ptr, p * M * N + safe, cache=".cv"),)
    missing = _missing(words, tag, live, PARTS)
    start = realtime_ticks()
    retries = 0
    stop = retries < 0
    while vote(missing) & (not stop):
        retries += 1
        words = ()
        for p in gl.static_range(PARTS):
            words += (cdna4.buffer_load(part_ptr, p * M * N + safe, cache=".cv"),)
        missing = _missing(words, tag, live, PARTS)
        stop = poll_expired(launch, start, retries)
    if stop:
        on_expiry(launch, poll_stop_site(launch, SITE), 0)
    total = gl.zeros([THREADS, E], gl.float32, lanes)
    for p in gl.static_range(PARTS):
        total += (words[p] & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True)
    return gl.convert_layout(gl.reshape(total, [THREADS * E]), offsets.type.layout)


@gluon.jit(do_not_specialize=("M", "tag"))
def gluon_gemv_a16w16_splitk_gfx950(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    tag,
    x_stride,
    N: gl.constexpr,
    K: gl.constexpr,
    ROWS_PER_TASK: gl.constexpr,
    PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
):
    """Standalone K-split GEMV: one task per program, partials to ``part_ptr``."""
    task = gl.program_id(0)
    tiles = gemv_a16w16_splitk_weights_gfx950(
        w_ptr, task, K, ROWS_PER_TASK, PARTS, GROUPS
    )
    gemv_a16w16_splitk_finish_gfx950(
        tiles, x_ptr, x_stride, part_ptr, tag, task, M, N, K, ROWS_PER_TASK, PARTS,
        GROUPS, "", ""
    )  # fmt: skip


@gluon.jit(do_not_specialize=("M",))
def gluon_gemv_a16w16_splitk_reduce_gfx950(
    part_ptr, y_ptr, M, N: gl.constexpr, PARTS: gl.constexpr, BLOCK: gl.constexpr
):
    """Sum, in part order, the partials of ``BLOCK`` outputs of one token into BF16 ``y``.

    The launch boundary orders the partials, so this reads them without
    polling; in-launch readers use :func:`gemv_a16w16_splitk_sum_gfx950`.
    """
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK // (64 * gl.num_warps())], [64], [gl.num_warps()], [0]
    )
    token = gl.program_id(0)
    rows = gl.program_id(1) * BLOCK + gl.arange(0, BLOCK, layout)
    mask = rows < N
    total = gl.zeros([BLOCK], gl.float32, layout)
    for p in gl.static_range(PARTS):
        word = gl.load(part_ptr + (p * M + token) * N + rows, mask=mask, other=0)
        total += (word & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True)
    gl.store(y_ptr + token * N + rows, total.to(gl.bfloat16), mask=mask)


def supports_gluon_gemv_a16w16_splitk_gfx950(m: int, n: int, k: int) -> bool:
    """16 tasks of n / 16 rows by 16 K slices, 8 waves of 64-multiple K each."""
    return 1 <= m <= _BLOCK_T.value and n % 256 == 0 and k % (16 * NUM_WARPS * 32) == 0


def launch_gluon_gemv_a16w16_splitk_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    partials: torch.Tensor,
    tag: int,
) -> torch.Tensor:
    """Standalone ``out = bf16(x @ weight.T)`` through K-split partials.

    The two launches the layer fuses: 256 task programs, then a reduce. Used
    by tests and as the reference composition of the task bodies.

    Args:
        x: ``[M, K]`` BF16 rows, unit inner stride, ``M <= 16``.
        weight: Contiguous BF16 ``[N, K]``, ``N`` a multiple of 256.
        out: ``[M, N]`` BF16 destination, contiguous.
        partials: int64 ``[16 * M * N]`` scratch.
        tag: Nonzero int32 tag, different from the scratch's previous use.

    Returns:
        ``out``.
    """
    m, k = x.shape
    n = weight.shape[0]
    if not supports_gluon_gemv_a16w16_splitk_gfx950(m, n, k):
        raise ValueError(f"K-split decode GEMV does not support M={m}, N={n}, K={k}")
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or x.stride(1) != 1
        or not weight.is_contiguous()
        or not out.is_contiguous()
        or tuple(out.shape) != (m, n)
        or partials.dtype != torch.int64
        or partials.numel() < 16 * m * n
        or tag == 0
    ):
        raise ValueError(
            "K-split decode GEMV needs BF16 x/weight/out, int64 partials and a nonzero tag"
        )
    gluon_gemv_a16w16_splitk_gfx950[(256,)](
        x, weight, partials, m, tag, x.stride(0), N=n, K=k, ROWS_PER_TASK=n // 16,
        PARTS=16, GROUPS=16, num_warps=NUM_WARPS
    )  # fmt: skip
    gluon_gemv_a16w16_splitk_reduce_gfx950[(m, triton.cdiv(n, 1024))](
        partials, out, m, N=n, PARTS=16, BLOCK=1024, num_warps=NUM_WARPS
    )
    return out


__all__ = [
    "gemv_a16w16_splitk_exchange_gfx950",
    "gemv_a16w16_splitk_finish_gfx950",
    "gemv_a16w16_splitk_finish_x_gfx950",
    "gemv_a16w16_splitk_hi_lo_sums_gfx950",
    "gemv_a16w16_splitk_publish_gfx950",
    "gemv_a16w16_splitk_sum_gfx950",
    "gemv_a16w16_splitk_wave_sums_gfx950",
    "gemv_a16w16_splitk_weights_gfx950",
    "gemv_a16w16_splitk_weights_paced_gfx950",
    "gemv_a16w16_splitk_x_gfx950",
    "gluon_gemv_a16w16_splitk_gfx950",
    "gluon_gemv_a16w16_splitk_reduce_gfx950",
    "launch_gluon_gemv_a16w16_splitk_gfx950",
    "supports_gluon_gemv_a16w16_splitk_gfx950",
]
