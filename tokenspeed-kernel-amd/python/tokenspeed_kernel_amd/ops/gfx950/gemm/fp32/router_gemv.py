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

"""Cold-weight FP32 router GEMV and split-K GEMM for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._extern import extern_compile_options, wait_vmem
from tokenspeed_kernel_amd._triton import compile_ahead, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
    DECODE_M_BUCKETS,
    GROUP_SIZE_M,
    _choose_decode_config,
    _get_partial_scratch,
    gluon_mm_a16w16_medium_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.splitk import check_splitk_counters

buffer = gl.amd.cdna4.buffer
# Non-temporal (NT) loads: the data is not expected to be reused.
_STREAM = gl.constexpr(gl.amd.cdna4.CachePolicy(nontemporal=True))

# Most rows the GEMV takes; more go to the split-K GEMM or torch.addmm.
MAX_ROWS = 16
# A program stages its whole weight row in LDS; longer rows go to torch.matmul.
MAX_K = 16384
_NUM_WARPS = 4
# K values a lane reads per load (16 bytes).
_VECTOR = 4


def _launch_metadata(grid, kernel, args):
    """Report weight, activation and output traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops32": 2 * m * n * k,
        "bytes": (n * k + m * k + m * n) * 4,
    }


@gluon.jit
def _load_x_step(x_ptr, x_offs, row_mask, k, step, K: gl.constexpr, STEP: gl.constexpr):
    """Load K step ``step`` of every row's activations, zero past ``K``."""
    # No ``other``: a masked lane's buffer load gets an out-of-range offset
    # and reads zero (CDNA4 ISA 9.1.5.1), while ``other`` would add a
    # v_cndmask per loaded register.
    return gl.amd.cdna4.buffer_load(
        x_ptr,
        x_offs + step * STEP,
        mask=row_mask & (k + step * STEP < K)[None, :],
    )


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_fp32_router_gemv_gfx950(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    K: gl.constexpr,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    VECTOR: gl.constexpr,
    MULTIPLIER: gl.constexpr,
):
    output = gl.program_id(0)
    # [rows, K step]: a lane holds VECTOR contiguous K values of every row.
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_M, VECTOR], [1, 64], [1, gl.num_warps()], [1, 0]
    )
    STEP: gl.constexpr = VECTOR * 64 * gl.num_warps()
    STEPS: gl.constexpr = (K + STEP - 1) // STEP
    k = gl.arange(0, STEP, gl.SliceLayout(0, layout))
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    row_mask = (rows < M)[:, None]
    x_offs = rows[:, None] * K + k[None, :]
    w_offs = output * K + k

    # The whole weight row goes to LDS at once, one commit group per step, so
    # all of it is in flight; each lane later reads back only what it copied.
    smem = gl.allocate_shared_memory(
        weight_ptr.dtype.element_ty,
        [STEPS, STEP],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    for step in gl.static_range(STEPS):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem.index(step),
            weight_ptr,
            w_offs + step * STEP,
            mask=k + step * STEP < K,
            cache_modifier=".cg",
        )
        gl.amd.cdna4.async_copy.commit_group()
    x = _load_x_step(x_ptr, x_offs, row_mask, k, 0, K, STEP)
    acc = gl.zeros((BLOCK_M, STEP), gl.float32, layout)
    # Unrolled, with the next step's activations in flight during this step's
    # FMAs; a loop-carried copy of a pending load would wait for it.
    for step in gl.static_range(STEPS):
        if step + 1 < STEPS:
            x_next = _load_x_step(x_ptr, x_offs, row_mask, k, step + 1, K, STEP)
        gl.amd.cdna4.async_copy.wait_group(STEPS - 1 - step)
        w = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem.index(step), gl.SliceLayout(0, layout)
        )
        acc = gl.fma(x, w[None, :], acc)
        # Pin this step's FMAs here: left free, the compiler sinks them below
        # the last load and keeps every step's activations live at once.
        acc = gl.inline_asm_elementwise(
            asm="",
            constraints="=v,0",
            args=[acc],
            dtype=gl.float32,
            is_pure=False,
            pack=1,
        )
        if step + 1 < STEPS:
            x = x_next
    result = gl.sum(acc, axis=1) * MULTIPLIER
    gl.store(out_ptr + rows * N + output, result, mask=rows < M)


# Measured split-K tiles per (N, K), keyed by the largest row count each
# serves, as (BLOCK_M, BLOCK_N, SPLIT_K, waves, chunks); None keeps the GEMV
# (or torch.addmm above MAX_ROWS). Swept on MI355X in HIP-graph replays with
# cold weights (rotating over 64 weight copies): each entry beat the path it
# replaces by at least 4% at the row counts it serves; above the last bucket
# hipBLASLt is as fast or faster.
_PREFERRED_SPLITK_CONFIGS: dict[
    tuple[int, int], dict[int, tuple[int, int, int, int, int] | None]
] = {
    # 8192-wide router over 256 experts (Muse Spark): the split-K GEMM replaces
    # the GEMV at 9-16 rows and hipBLASLt at 17-96 rows.
    (256, 8192): {
        8: None,
        64: (16, 16, 8, 4, 1),
        96: (32, 16, 16, 4, 2),
    },
}


def _choose_splitk_config(M: int, N: int, K: int) -> tuple[int, ...] | None:
    buckets = _PREFERRED_SPLITK_CONFIGS.get((N, K), {})
    bucket = min((bound for bound in buckets if M <= bound), default=None)
    return None if bucket is None else buckets[bucket]


def gluon_fp32_router_gemv_num_counters_gfx950(M: int, N: int, K: int) -> int:
    """Split-K arrival counters a router launch needs; 0 when it does not split K."""
    config = _choose_splitk_config(M, N, K)
    if config is None:
        return 0
    return triton.cdiv(M, config[0]) * (N // config[1])


def _splitk_launch_metadata(grid, kernel, args):
    """Report MFMA work, operand reads and partial traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    partials = grid[0] * args["BLOCK_M"] * args["BLOCK_N"]
    return {
        "name": kernel.name,
        "flops32": 2 * m * n * k,
        "bytes": (n * k + m * k + m * n + 2 * partials) * 4,
    }


@gluon.constexpr_function
def _tile_layout(block_m: int, block_n: int, waves: int):
    # Up to four contiguous outputs per lane, rows across lanes then waves.
    vec = max(1, min(4, block_m * block_n // (64 * waves)))
    lanes_n = min(64, block_n // vec)
    return gl.BlockedLayout([1, vec], [64 // lanes_n, lanes_n], [waves, 1], [1, 0])


@gluon.constexpr_function
def _x_load_layout(waves: int, k_chunk: int):
    # [wave, row, K]: lanes run along K, four values (16 bytes) per lane.
    lanes_k = min(64, k_chunk // 4)
    return gl.BlockedLayout(
        [1, 1, 4], [1, 64 // lanes_k, lanes_k], [waves, 1, 1], [2, 1, 0]
    )


@gluon.constexpr_function
def _w_load_layout(waves: int, k_chunk: int):
    # [wave, K, output]: lanes run along K, four values (16 bytes) per lane.
    lanes_k = min(64, k_chunk // 4)
    return gl.BlockedLayout(
        [1, 4, 1], [1, lanes_k, 64 // lanes_k], [waves, 1, 1], [1, 2, 0]
    )


@gluon.jit(launch_metadata=_splitk_launch_metadata, do_not_specialize=("M",))
def gluon_fp32_router_gemv_splitk_gfx950(
    x_ptr,
    weight_ptr,
    out_ptr,
    partial_ptr,
    counter_ptr,
    M,
    K: gl.constexpr,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SPLIT_K: gl.constexpr,
    CHUNKS: gl.constexpr,
    MULTIPLIER: gl.constexpr,
):
    """FP32 MFMA GEMM over ``SPLIT_K`` programs per output tile.

    Every wave of a program reduces its own contiguous K slice into a full
    ``BLOCK_M x BLOCK_N`` tile; the waves' tiles are then summed. Each program
    stores its FP32 partial and bumps the tile's arrival counter; the last
    program to arrive sums all partials in split order, scales and writes the
    tile, and resets the counter to zero.
    """
    WAVES: gl.constexpr = gl.num_warps()
    K_WAVE: gl.constexpr = K // (SPLIT_K * WAVES)
    gl.static_assert(K_WAVE * SPLIT_K * WAVES == K, "K must split evenly")
    gl.static_assert(CHUNKS == 1 or CHUNKS == 2, "1 or 2 chunks")
    K_CHUNK: gl.constexpr = K_WAVE // CHUNKS
    gl.static_assert(K_CHUNK % 16 == 0, "each chunk spans whole 16-deep K blocks")
    gl.static_assert(N % BLOCK_N == 0, "N must tile evenly")
    pid = gl.program_id(0)
    num_n: gl.constexpr = N // BLOCK_N
    num_tiles = gl.cdiv(M, BLOCK_M) * num_n
    tile = pid % num_tiles
    split = pid // num_tiles
    m0 = (tile // num_n) * BLOCK_M
    n0 = (tile % num_n) * BLOCK_N

    # Waves are the batch dimension: wave w owns K slice w of the program.
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[WAVES, 1, 1],
    )
    # Four consecutive K values per lane feed four MFMAs. The MFMA sums them in
    # a different K order than a plain dot, over the same FP32 products.
    dot_x: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=4)
    dot_w: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=4)

    # Loads run along K so a wave reads whole contiguous rows; the MFMA
    # operand layouts would scatter them over many short reads.
    x_layout: gl.constexpr = _x_load_layout(WAVES, K_CHUNK)
    w_layout: gl.constexpr = _w_load_layout(WAVES, K_CHUNK)
    # The scalar offset (soffset) of the x loads below starts them at this
    # tile's first row, so their per-lane offsets stay within a tile.
    x_rows = buffer.make_buffer_descriptor(x_ptr, M * K * 4)
    w_rows = buffer.make_buffer_descriptor(weight_ptr, N * K * 4)
    # Rows past M read row M - 1 instead of masking (a masked load selects
    # after it lands, which waits for it); the store drops those rows.
    xm = gl.minimum(
        gl.arange(0, BLOCK_M, gl.SliceLayout(0, gl.SliceLayout(2, x_layout))),
        M - 1 - m0,
    )
    xw = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, x_layout)))
    xk = gl.arange(0, K_CHUNK, gl.SliceLayout(0, gl.SliceLayout(1, x_layout)))
    x_offs = (
        xm[None, :, None] * K
        + (split * WAVES + xw[:, None, None]) * K_WAVE
        + xk[None, None, :]
    )
    ww = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, w_layout)))
    wk = gl.arange(0, K_CHUNK, gl.SliceLayout(0, gl.SliceLayout(2, w_layout)))
    wn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, w_layout)))
    w_offs = (
        (n0 + wn[None, None, :]) * K
        + (split * WAVES + ww[:, None, None]) * K_WAVE
        + wk[None, :, None]
    )
    # Issue every load before any MFMA, chunk by chunk, so the first MFMAs
    # wait only for the first chunk; loads return in issue order. Each weight
    # element is read once per call, so its loads are non-temporal (NT).
    x0 = buffer.load(x_rows, x_offs, soffset=m0 * K)
    w0 = buffer.load(w_rows, w_offs, cache_policy=_STREAM)
    if CHUNKS > 1:
        x1 = buffer.load(x_rows, x_offs + K_CHUNK, soffset=m0 * K)
        w1 = buffer.load(w_rows, w_offs + K_CHUNK, cache_policy=_STREAM)
    # Keep the loads above the MFMAs: left free, the scheduler sinks them
    # between MFMAs and waits on them one at a time.
    gl.amd.hint.sched_barrier()
    acc = gl.zeros((WAVES, BLOCK_M, BLOCK_N), gl.float32, mfma)
    acc = gl.amd.cdna4.mfma(
        gl.convert_layout(x0, dot_x), gl.convert_layout(w0, dot_w), acc
    )
    if CHUNKS > 1:
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(x1, dot_x), gl.convert_layout(w1, dot_w), acc
        )
    # The sum over waves leaves every wave holding the whole tile; spread it
    # over all lanes so the partial traffic below is not repeated per wave.
    out_layout: gl.constexpr = _tile_layout(BLOCK_M, BLOCK_N, WAVES)
    tile_acc = gl.convert_layout(gl.sum(acc, axis=0), out_layout)

    om = gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))
    on = gl.arange(0, BLOCK_N, gl.SliceLayout(0, out_layout))
    tile_offs = om[:, None] * BLOCK_N + on[None, :]
    # One descriptor covers every partial; the scalar offset picks one.
    TILE: gl.constexpr = BLOCK_M * BLOCK_N
    partials = buffer.make_buffer_descriptor(
        partial_ptr, SPLIT_K * num_tiles * TILE * 4
    )
    # The programs of a tile can run on different XCDs, each with its own L2,
    # so the partials are stored and read at system scope (SC0 SC1), which
    # bypasses L2.
    buffer.store(
        tile_acc,
        partials,
        tile_offs,
        soffset=(split * num_tiles + tile) * TILE,
        scope="sys",
    )
    # Publish the partial as the split-K decode GEMM does (gemm/fp16/mm.py):
    # each wave waits for its stores to complete, and the barrier orders every
    # wave's stores before the counter bump. A release atomic would instead
    # write back this XCD's whole L2 (BUFFER_WBL2).
    wait_vmem()
    gl.barrier()
    arrived = gl.atomic_add(counter_ptr + tile, 1, sem="relaxed", scope="gpu")
    if arrived == SPLIT_K - 1:
        # Sum in split order, not arrival order, so results are deterministic.
        tile_acc = buffer.load(partials, tile_offs, soffset=tile * TILE, scope="sys")
        for s in gl.static_range(1, SPLIT_K):
            tile_acc += buffer.load(
                partials, tile_offs, soffset=(s * num_tiles + tile) * TILE, scope="sys"
            )
        gl.store(counter_ptr + tile, 0)
        out_rows = buffer.make_buffer_descriptor(out_ptr + m0 * N + n0, BLOCK_M * N * 4)
        buffer.store(
            tile_acc * MULTIPLIER,
            out_rows,
            om[:, None] * N + on[None, :],
            mask=(m0 + om < M)[:, None],
        )


def transpose_fp32_router_weight_gfx950(weight: torch.Tensor) -> torch.Tensor:
    """Transpose a canonical ``[K, N]`` router weight for row-coalesced GEMV."""
    if weight.ndim != 2 or weight.dtype != torch.float32 or not weight.is_cuda:
        raise ValueError("router weight must be a two-dimensional FP32 GPU tensor")
    return weight.T.contiguous()


def _gemv_launch(x, weight, out, rows: int, multiplier: float):
    """The GEMV's arguments and compile-time options for ``rows`` rows."""
    k, n = x.shape[1], weight.shape[0]
    return (x, weight, out, rows), dict(
        K=k,
        N=n,
        # Bucketed so at most five variants compile per (K, N).
        BLOCK_M=triton.next_power_of_2(rows),
        # Scalar loads keep K values that are not 16-byte aligned addressable.
        VECTOR=_VECTOR if k % _VECTOR == 0 else 1,
        MULTIPLIER=float(multiplier),
        num_warps=_NUM_WARPS,
    )


def _splitk_launch(
    x, weight, out, partial, counters, rows: int, multiplier: float, config
):
    """The split-K GEMM's arguments and compile-time options for ``config``."""
    block_m, block_n, split_k, waves, chunks = config
    return (x, weight, out, partial, counters, rows), dict(
        K=x.shape[1],
        N=weight.shape[0],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        CHUNKS=chunks,
        MULTIPLIER=float(multiplier),
        num_warps=waves,
        **extern_compile_options(),
    )


def _router_selections(n: int, k: int) -> list[tuple[int, tuple[int, ...] | None]]:
    """Row counts probing each distinct kernel choice of an ``[N, K]`` router.

    Each comes with the split-K config it selects (None: the GEMV): each GEMV
    BLOCK_M that a row count up to MAX_ROWS takes, then each measured split-K
    bucket.
    """
    selections = {}
    for rows in range(1, MAX_ROWS + 1):
        if _choose_splitk_config(rows, n, k) is None and k <= MAX_K:
            # The probe row count is the block itself, so it selects BLOCK_M.
            block_m = triton.next_power_of_2(rows)
            selections.setdefault(("gemv", block_m), (block_m, None))
    for bound, config in sorted(_PREFERRED_SPLITK_CONFIGS.get((n, k), {}).items()):
        if config is not None:
            selections.setdefault(("splitk", config), (bound, config))
    return list(selections.values())


def _compile_router_variants(x, weight, multiplier: float) -> None:
    """Compile, without launching, every GEMV and split-K variant of this router.

    These are the variants any other row count selects. The row count is a
    runtime argument of both kernels, but the GEMV's BLOCK_M (its power of two;
    a block shape, so it cannot be left unspecialized) and the split-K tile
    (its bucket) are compile-time, so a batch in a bucket startup never reached
    would compile while serving. Stand-in tensors have the dtypes and shapes
    serving allocates.
    """
    k, n = x.shape[1], weight.shape[0]
    for rows, config in _router_selections(n, k):
        rows_in = x.new_empty((rows, k))
        out = x.new_empty((rows, n))
        if config is None:
            args, options = _gemv_launch(rows_in, weight, out, rows, multiplier)
            compile_ahead(gluon_fp32_router_gemv_gfx950, *args, **options)
            continue
        block_m, block_n, split_k = config[:3]
        tiles = triton.cdiv(rows, block_m) * (n // block_n)
        partial = x.new_empty((split_k * tiles * block_m * block_n,))
        counters = torch.zeros(tiles, dtype=torch.int32, device=x.device)
        args, options = _splitk_launch(
            rows_in, weight, out, partial, counters, rows, multiplier, config
        )
        compile_ahead(gluon_fp32_router_gemv_splitk_gfx950, *args, **options)


def launch_gluon_fp32_router_gemv_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    enable_pdl: bool,
    *,
    warm_variants: bool,
) -> torch.Tensor:
    """Project FP32 rows through a transposed router weight.

    Shapes and row counts listed in ``_PREFERRED_SPLITK_CONFIGS`` run the
    split-K MFMA GEMM. Other batches of up to ``MAX_ROWS`` rows run the GEMV;
    the rest, and rows longer than ``MAX_K``, take BLAS with ``multiplier`` as
    alpha.

    Args:
        x: Contiguous FP32 ``[M, K]`` rows.
        weight: Contiguous FP32 ``[N, K]`` weight from
            :func:`transpose_fp32_router_weight_gfx950`.
        multiplier: Scale applied to every logit.
        counters: Zeroed int32 split-K arrival counters, at least
            ``gluon_fp32_router_gemv_num_counters_gfx950(M, N, K)`` of them;
            the launch leaves them zeroed.
        enable_pdl: Accepted for the registry signature; unused.
        warm_variants: Also compile, without launching, every variant another
            row count of this shape picks (startup warmup).

    Returns:
        FP32 ``[M, N]`` logits.
    """
    del enable_pdl
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous FP32 GPU rows")
    if (
        weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or weight.dtype != torch.float32
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous transposed FP32 [N, K] beside x")
    rows, k = x.shape
    n = weight.shape[0]
    if warm_variants:
        _compile_router_variants(x, weight, multiplier)
    if rows == 0:
        return x.new_empty((0, n))
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    config = _choose_splitk_config(rows, n, k)
    if config is None and rows <= MAX_ROWS and k <= MAX_K:
        args, options = _gemv_launch(x, weight, out, rows, multiplier)
        gluon_fp32_router_gemv_gfx950[(n,)](*args, **options)
        return out
    if config is None:
        # hipBLASLt applies alpha once to the finished FP32 dot product, which
        # matches a separate multiply bit for bit without its extra launch.
        return torch.addmm(out, x, weight.T, beta=0, alpha=multiplier)
    block_m, block_n, split_k = config[:3]
    num_tiles = triton.cdiv(rows, block_m) * (n // block_n)
    check_splitk_counters(
        counters, num_tiles, x.device, "gfx950 FP32 router split-K GEMM"
    )
    partial = torch.empty(
        (split_k * num_tiles * block_m * block_n,),
        dtype=torch.float32,
        device=x.device,
    )
    args, options = _splitk_launch(
        x, weight, out, partial, counters, rows, multiplier, config
    )
    gluon_fp32_router_gemv_splitk_gfx950[(num_tiles * split_k,)](*args, **options)
    return out


# Router tiles (BLOCK_M, BLOCK_N, SPLIT_K, waves, chunks) when the router
# shares a launch with a decode GEMM: 128 programs per 16-row band at 256
# outputs. Its programs read 32 KB of activations each where a one-output
# GEMV program reads all 256 KB. Next to a 512 x 8192 decode
# GEMM's 128 programs, this tile beat the GEMV (two outputs per program) at
# 8 rows in graph replay on MI355X.
_WITH_MM_ROUTER_CONFIG = (16, 16, 8, 4, 1)
# Up to this many rows the shared launch serves every router shape it fits.
_WITH_MM_ANY_ROUTER_MAX_ROWS = 16
# Past that, it serves only row counts at which the router's standalone plan
# (_PREFERRED_SPLITK_CONFIGS) takes this same tile, so the logits keep the
# standalone router's bits; the largest such count bounds the warm-up.
_WITH_MM_MAX_ROWS = max(
    [_WITH_MM_ANY_ROUTER_MAX_ROWS]
    + [
        bound
        for buckets in _PREFERRED_SPLITK_CONFIGS.values()
        for bound, config in buckets.items()
        if config == _WITH_MM_ROUTER_CONFIG
    ]
)
_MM_GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)


@gluon.jit(do_not_specialize=("M",))
def gluon_fp32_router_gemv_mm_a16w16_gfx950(
    x_ptr,
    weight_ptr,
    out_ptr,
    router_partial_ptr,
    router_counter_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    MM_N,
    stride_am,
    stride_bn,
    stride_cm,
    K: gl.constexpr,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SPLIT_K: gl.constexpr,
    CHUNKS: gl.constexpr,
    MULTIPLIER: gl.constexpr,
    ROUTER_PROGRAMS: gl.constexpr,
    MM_K: gl.constexpr,
    MM_BLOCK_M: gl.constexpr,
    MM_BLOCK_N: gl.constexpr,
    MM_BLOCK_K: gl.constexpr,
    MM_WARPS_M: gl.constexpr,
    MM_WARPS_N: gl.constexpr,
    MM_NUM_BUFFERS: gl.constexpr,
    MM_SPLIT_K: gl.constexpr,
    MM_NUM_XCDS: gl.constexpr,
):
    """Run the split-K router GEMM and an independent decode GEMM in one launch.

    Programs ``[0, ROUTER_PROGRAMS)`` are :func:`gluon_fp32_router_gemv_splitk_gfx950`
    on ``x``; the rest are ``gluon_mm_a16w16_medium_gfx950`` computing
    ``c = a @ b.T``. Neither waits on the other, so the launch saves one kernel
    boundary and both weight streams share the CUs.
    """
    pid = gl.program_id(0)
    if pid < ROUTER_PROGRAMS:
        # ROUTER_PROGRAMS covers the decode row bucket's largest row count, so
        # it is fixed per compiled bucket; programs past this batch's router
        # tiles have no work. The kernel maps the rest to (tile, split) from M.
        if pid < gl.cdiv(M, BLOCK_M) * (N // BLOCK_N) * SPLIT_K:
            gluon_fp32_router_gemv_splitk_gfx950(
                x_ptr,
                weight_ptr,
                out_ptr,
                router_partial_ptr,
                router_counter_ptr,
                M,
                K,
                N,
                BLOCK_M,
                BLOCK_N,
                SPLIT_K,
                CHUNKS,
                MULTIPLIER,
            )
    else:
        gluon_mm_a16w16_medium_gfx950(
            a_ptr, b_ptr, c_ptr, c_ptr, c_ptr, partial_ptr, counter_ptr, M, MM_N,
            MM_K, stride_am, 1, 1, stride_bn, stride_cm, 1, stride_cm, 1,
            stride_cm, 1, MM_BLOCK_M, MM_BLOCK_N, MM_BLOCK_K, MM_WARPS_M,
            MM_WARPS_N, MM_NUM_BUFFERS, _MM_GROUP_SIZE_M, False, MM_SPLIT_K,
            MM_NUM_XCDS, ROUTER_PROGRAMS
        )  # fmt: skip


def supports_gluon_fp32_router_gemv_mm_a16w16_gfx950(
    rows: int, n: int, k: int, mm_n: int, mm_k: int
) -> bool:
    """Whether ``rows`` router rows and the ``[mm_n, mm_k]`` decode GEMM share a launch."""
    block_m, block_n, split_k, waves, _ = _WITH_MM_ROUTER_CONFIG
    mm_config = _choose_decode_config(rows, mm_n, mm_k)
    return (
        0 < rows
        and (
            rows <= _WITH_MM_ANY_ROUTER_MAX_ROWS
            or _choose_splitk_config(rows, n, k) == _WITH_MM_ROUTER_CONFIG
        )
        and n % block_n == 0
        and k % (split_k * waves * 16) == 0
        and mm_config is not None
        # Both halves run the same wave count, the launch's num_warps.
        and mm_config[3] * mm_config[4] == waves
    )


def _router_programs_with_mm(rows: int, n: int) -> int:
    """Router programs of a router + decode GEMM launch: sized for the largest
    row count of ``rows``' decode bucket, a compile-time value fixed per bucket
    (the kernel idles the programs past the batch's tiles)."""
    bucket = next(bound for bound in DECODE_M_BUCKETS if rows <= bound)
    block_m, block_n, split_k = _WITH_MM_ROUTER_CONFIG[:3]
    return triton.cdiv(bucket, block_m) * (n // block_n) * split_k


def gluon_fp32_router_gemv_mm_a16w16_num_counters_gfx950(rows: int, n: int) -> int:
    """Router arrival counters a router + decode GEMM launch needs."""
    block_m, block_n = _WITH_MM_ROUTER_CONFIG[:2]
    return triton.cdiv(rows, block_m) * (n // block_n)


def _mm_launch(
    x,
    weight,
    out,
    router_partial,
    counters,
    mm_a,
    mm_b,
    mm_out,
    partial,
    mm_counters,
    multiplier,
):
    """Grid size, arguments and options of the router + decode GEMM launch."""
    rows, k = x.shape
    n = weight.shape[0]
    mm_n, mm_k = mm_b.shape
    block_m, block_n, split_k, waves, chunks = _WITH_MM_ROUTER_CONFIG
    router_programs = _router_programs_with_mm(rows, n)
    mm_block_m, mm_block_n, mm_block_k, warps_m, warps_n, buffers, mm_split, xcds = (
        _choose_decode_config(rows, mm_n, mm_k)
    )
    mm_tiles = triton.cdiv(rows, mm_block_m) * triton.cdiv(mm_n, mm_block_n)
    args = (
        x,
        weight,
        out,
        router_partial,
        counters,
        mm_a,
        mm_b,
        mm_out,
        partial,
        mm_counters,
        rows,
        mm_n,
        mm_a.stride(0),
        mm_b.stride(0),
        mm_out.stride(0),
    )
    options = dict(
        K=k,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        CHUNKS=chunks,
        MULTIPLIER=float(multiplier),
        ROUTER_PROGRAMS=router_programs,
        MM_K=mm_k,
        MM_BLOCK_M=mm_block_m,
        MM_BLOCK_N=mm_block_n,
        MM_BLOCK_K=mm_block_k,
        MM_WARPS_M=warps_m,
        MM_WARPS_N=warps_n,
        MM_NUM_BUFFERS=buffers,
        MM_SPLIT_K=mm_split,
        MM_NUM_XCDS=xcds,
        num_warps=waves,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
        **extern_compile_options(),
    )
    return router_programs + mm_tiles * mm_split, args, options  # fmt: skip


def _compile_router_mm_variants(x, weight, multiplier: float, mm_b) -> None:
    """Compile, without launching, the router + decode GEMM launch per bucket.

    One binary per decode GEMM bucket that a row count up to _WITH_MM_MAX_ROWS
    selects: the MM_* tile and the router program count are compile-time and
    follow the bucket. Stand-in tensors have the dtypes and shapes serving
    allocates.
    """
    k, n = x.shape[1], weight.shape[0]
    mm_n, mm_k = mm_b.shape
    seen = set()
    for rows in range(1, _WITH_MM_MAX_ROWS + 1):
        if not supports_gluon_fp32_router_gemv_mm_a16w16_gfx950(rows, n, k, mm_n, mm_k):
            continue
        config = _choose_decode_config(rows, mm_n, mm_k)
        # The binary keys on the decode tile and the bucket-sized router
        # program count, which buckets sharing a tile can still differ in.
        key = (config, _router_programs_with_mm(rows, n))
        if key in seen:
            continue
        seen.add(key)
        block_m, block_n, split_k = _WITH_MM_ROUTER_CONFIG[:3]
        router_tiles = gluon_fp32_router_gemv_mm_a16w16_num_counters_gfx950(rows, n)
        mm_out = mm_b.new_empty((rows, mm_n))
        mm_split = config[6]
        if mm_split > 1:
            mm_tiles = triton.cdiv(rows, config[0]) * triton.cdiv(mm_n, config[1])
            # The launch's own scratch, as serving passes it.
            partial = _get_partial_scratch(
                x.device, mm_tiles, mm_split, config[1], config[0]
            )
            mm_counters = torch.zeros(mm_tiles, dtype=torch.int32, device=x.device)
        else:
            partial = mm_counters = mm_out
        _, args, options = _mm_launch(
            x.new_empty((rows, k)),
            weight,
            x.new_empty((rows, n)),
            x.new_empty((split_k * router_tiles * block_m * block_n,)),
            torch.zeros(router_tiles, dtype=torch.int32, device=x.device),
            mm_b.new_empty((rows, mm_k)),
            mm_b,
            mm_out,
            partial,
            mm_counters,
            multiplier,
        )
        compile_ahead(gluon_fp32_router_gemv_mm_a16w16_gfx950, *args, **options)


def launch_gluon_fp32_router_gemv_mm_a16w16_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    mm_a: torch.Tensor,
    mm_b: torch.Tensor,
    mm_out: torch.Tensor,
    mm_counters: torch.Tensor,
    *,
    warm_variants: bool,
) -> torch.Tensor:
    """Project FP32 router rows and compute ``mm_out = mm_a @ mm_b.T`` in one launch.

    The router runs the split-K GEMM for every supported row count, so its
    logits round as that GEMM's do wherever the router's standalone plan
    takes the same tile; ``mm_out`` is bitwise what the decode GEMM alone
    writes.

    Args:
        x: Contiguous FP32 router rows ``[M, K]``.
        weight: Contiguous FP32 ``[N, K]`` weight from
            :func:`transpose_fp32_router_weight_gfx950`.
        multiplier: Scale applied to every logit.
        counters: Zeroed int32 arrival counters for the router's tiles, at
            least ``gluon_fp32_router_gemv_mm_a16w16_num_counters_gfx950(M, N)``
            of them; the launch leaves them zeroed.
        mm_a: Contiguous BF16 ``[M, mm_k]`` rows.
        mm_b: Contiguous BF16 ``[mm_n, mm_k]`` weight.
        mm_out: ``[M, mm_n]`` BF16 destination with unit inner stride.
        mm_counters: Zeroed split-K counters of the decode GEMM, as for
            ``launch_gluon_mm_a16w16_decode_gfx950``.
        warm_variants: Also compile, without launching, the launch for every
            decode GEMM bucket another supported row count picks (startup
            warmup).

    Returns:
        FP32 router logits ``[M, N]``.
    """
    rows, k = x.shape
    n = weight.shape[0]
    mm_n, mm_k = mm_b.shape
    if not supports_gluon_fp32_router_gemv_mm_a16w16_gfx950(rows, n, k, mm_n, mm_k):
        raise ValueError(
            f"router + decode GEMM launch does not support {rows} rows of a "
            f"[{n}, {k}] router with a [{mm_n}, {mm_k}] GEMM"
        )
    if (
        x.dtype != torch.float32
        or weight.dtype != torch.float32
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape[1] != k
        or mm_a.dtype != torch.bfloat16
        or mm_b.dtype != torch.bfloat16
        or mm_out.dtype != torch.bfloat16
        or not mm_a.is_contiguous()
        or not mm_b.is_contiguous()
        or tuple(mm_a.shape) != (rows, mm_k)
        or tuple(mm_out.shape) != (rows, mm_n)
        or mm_out.stride(1) != 1
    ):
        raise ValueError(
            "router + decode GEMM launch needs contiguous FP32 router rows and "
            "weight, contiguous BF16 GEMM operands and a row-major BF16 output"
        )
    if warm_variants:
        _compile_router_mm_variants(x, weight, multiplier, mm_b)
    block_m, block_n, split_k = _WITH_MM_ROUTER_CONFIG[:3]
    router_tiles = gluon_fp32_router_gemv_mm_a16w16_num_counters_gfx950(rows, n)
    check_splitk_counters(
        counters, router_tiles, x.device, "gfx950 router + decode GEMM launch"
    )
    out = torch.empty((rows, n), dtype=torch.float32, device=x.device)
    router_partial = torch.empty(
        (split_k * router_tiles * block_m * block_n,),
        dtype=torch.float32,
        device=x.device,
    )
    mm_config = _choose_decode_config(rows, mm_n, mm_k)
    mm_block_m, mm_block_n = mm_config[:2]
    mm_split = mm_config[6]
    mm_tiles = triton.cdiv(rows, mm_block_m) * triton.cdiv(mm_n, mm_block_n)
    if mm_split > 1:
        check_splitk_counters(
            mm_counters, mm_tiles, x.device, "gfx950 router + decode GEMM launch"
        )
        partial = _get_partial_scratch(
            x.device, mm_tiles, mm_split, mm_block_n, mm_block_m
        )
    else:
        partial = mm_counters = mm_out
    programs, args, options = _mm_launch(
        x, weight, out, router_partial, counters, mm_a, mm_b, mm_out, partial,
        mm_counters, multiplier,
    )  # fmt: skip
    gluon_fp32_router_gemv_mm_a16w16_gfx950[(programs,)](*args, **options)
    return out


__all__ = [
    "gluon_fp32_router_gemv_mm_a16w16_gfx950",
    "gluon_fp32_router_gemv_mm_a16w16_num_counters_gfx950",
    "gluon_fp32_router_gemv_num_counters_gfx950",
    "launch_gluon_fp32_router_gemv_mm_a16w16_gfx950",
    "supports_gluon_fp32_router_gemv_mm_a16w16_gfx950",
    "launch_gluon_fp32_router_gemv_gfx950",
    "transpose_fp32_router_weight_gfx950",
]
