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
from tokenspeed_kernel_amd.ops.gfx950.persistent import (
    poll_expired,
    poll_stop_site,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent.timing import realtime_ticks

cdna4 = gl.amd.cdna4

NUM_WARPS = 8
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
    WAVES: gl.constexpr = gl.num_warps()
    KP: gl.constexpr = K // PARTS
    KW: gl.constexpr = KP // WAVES
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    gl.static_assert(ROWS_PER_TASK % 16 == 0 and KW % 32 == 0)
    a_l: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=_mfma(WAVES), k_width=8
    )
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    row = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    k = gl.arange(0, KW, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    group = task % GROUPS
    part = task // GROUPS
    base = (
        row[None, :, None] * K
        + (part * KP + wave * KW)[:, None, None]
        + k[None, None, :]
    )
    tiles = ()
    for t in gl.static_range(TILES):
        tiles += (
            cdna4.buffer_load(w_ptr, base + (group * TILES + t) * 16 * K, cache=".cg"),
        )
    return tiles


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
    WAVES: gl.constexpr = gl.num_warps()
    KP: gl.constexpr = K // PARTS
    KW: gl.constexpr = KP // WAVES
    TILES: gl.constexpr = ROWS_PER_TASK // 16
    mfma: gl.constexpr = _mfma(WAVES)
    b_l: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    group = task % GROUPS
    part = task // GROUPS
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, b_l)))
    k = gl.arange(0, KW, gl.SliceLayout(0, gl.SliceLayout(2, b_l)))
    token = gl.minimum(
        gl.arange(0, _BLOCK_T, gl.SliceLayout(0, gl.SliceLayout(1, b_l))), M - 1
    )
    x = cdna4.buffer_load(
        x_ptr,
        token[None, None, :] * x_stride
        + (part * KP + wave * KW)[:, None, None]
        + k[None, :, None],
        cache=X_CACHE,
    )
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
def _wave_any(flags):
    """Whether any lane of this wave holds a nonzero int32 (one lane-mask vote)."""
    mask = gl.inline_asm(
        "v_cmp_ne_u32_e64 $0, 0, $1", "=s,v,~{memory}", [flags], gl.int64, is_pure=False
    )
    return mask != 0


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
    while _wave_any(missing) & (not stop):
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
    "gemv_a16w16_splitk_finish_gfx950",
    "gemv_a16w16_splitk_sum_gfx950",
    "gemv_a16w16_splitk_weights_gfx950",
    "gluon_gemv_a16w16_splitk_gfx950",
    "gluon_gemv_a16w16_splitk_reduce_gfx950",
    "launch_gluon_gemv_a16w16_splitk_gfx950",
    "supports_gluon_gemv_a16w16_splitk_gfx950",
]
