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
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.splitk import check_splitk_counters

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
    return gl.amd.cdna4.buffer_load(
        x_ptr,
        x_offs + step * STEP,
        mask=row_mask & (k + step * STEP < K)[None, :],
        other=0.0,
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
# (or torch.addmm above MAX_ROWS). From cold-weight CUDA-graph sweeps on
# MI355X (64 rotating weight copies, capture_session): each entry beat the
# path it replaces by at least 4% at the row counts it serves; above the last
# bucket hipBLASLt is as fast or faster.
_PREFERRED_SPLITK_CONFIGS: dict[
    tuple[int, int], dict[int, tuple[int, int, int, int, int] | None]
] = {
    # 8192-wide router over 256 experts (Muse Spark), us per call:
    # 9-16 rows 8.5-8.8 (GEMV) -> 7.3; 17-64 rows 12.1 (hipBLASLt) -> 7.4-8.8;
    # 80-96 rows 12.6-14.3 -> 10.5.
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
    # Rows past M read row M - 1 instead of masking (a masked load selects
    # after it lands, which waits for it); the store drops those rows.
    xm = gl.minimum(
        m0 + gl.arange(0, BLOCK_M, gl.SliceLayout(0, gl.SliceLayout(2, x_layout))),
        M - 1,
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
    # wait only for the first chunk; loads return in issue order.
    x0 = gl.amd.cdna4.buffer_load(x_ptr, x_offs)
    w0 = gl.amd.cdna4.buffer_load(weight_ptr, w_offs, cache=".cg")
    if CHUNKS > 1:
        x1 = gl.amd.cdna4.buffer_load(x_ptr, x_offs + K_CHUNK)
        w1 = gl.amd.cdna4.buffer_load(weight_ptr, w_offs + K_CHUNK, cache=".cg")
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
    gl.amd.cdna4.buffer_store(
        tile_acc,
        partial_ptr + (split * num_tiles + tile) * (BLOCK_M * BLOCK_N),
        tile_offs,
        cache=".wt",
    )
    # Publish the partial as the split-K decode GEMM does (gemm/fp16/mm.py):
    # ".wt" writes through to memory, each wave waits for its stores to
    # complete, and the barrier orders every wave's stores before the counter
    # bump; a release atomic would also write back and invalidate the whole L2.
    gl.inline_asm_elementwise(
        asm="s_waitcnt vmcnt(0)",
        constraints="=v,0",
        args=[om],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()
    arrived = gl.atomic_add(counter_ptr + tile, 1, sem="relaxed", scope="gpu")
    if arrived == SPLIT_K - 1:
        # Sum in split order, not arrival order, so results are deterministic.
        # ".cv" reads bypass the caches, so partials from other XCDs are fresh.
        tile_acc = gl.amd.cdna4.buffer_load(
            partial_ptr + tile * (BLOCK_M * BLOCK_N), tile_offs, cache=".cv"
        )
        for s in gl.static_range(1, SPLIT_K):
            tile_acc += gl.amd.cdna4.buffer_load(
                partial_ptr + (s * num_tiles + tile) * (BLOCK_M * BLOCK_N),
                tile_offs,
                cache=".cv",
            )
        gl.store(counter_ptr + tile, 0)
        gl.amd.cdna4.buffer_store(
            tile_acc * MULTIPLIER,
            out_ptr + m0 * N + n0,
            om[:, None] * N + on[None, :],
            mask=(m0 + om < M)[:, None],
        )


def transpose_fp32_router_weight_gfx950(weight: torch.Tensor) -> torch.Tensor:
    """Transpose a canonical ``[K, N]`` router weight for row-coalesced GEMV."""
    if weight.ndim != 2 or weight.dtype != torch.float32 or not weight.is_cuda:
        raise ValueError("router weight must be a two-dimensional FP32 GPU tensor")
    return weight.T.contiguous()


def launch_gluon_fp32_router_gemv_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    enable_pdl: bool,
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
    if rows == 0:
        return x.new_empty((0, n))
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    config = _choose_splitk_config(rows, n, k)
    if config is None and rows <= MAX_ROWS and k <= MAX_K:
        gluon_fp32_router_gemv_gfx950[(n,)](
            x,
            weight,
            out,
            rows,
            K=k,
            N=n,
            # Bucketed so at most five variants compile per (K, N).
            BLOCK_M=triton.next_power_of_2(rows),
            # Scalar loads keep K values that are not 16-byte aligned addressable.
            VECTOR=_VECTOR if k % _VECTOR == 0 else 1,
            MULTIPLIER=float(multiplier),
            num_warps=_NUM_WARPS,
        )
        return out
    if config is None:
        # hipBLASLt applies alpha once to the finished FP32 dot product, which
        # matches a separate multiply bit for bit without its extra launch.
        return torch.addmm(out, x, weight.T, beta=0, alpha=multiplier)
    block_m, block_n, split_k, waves, chunks = config
    num_tiles = triton.cdiv(rows, block_m) * (n // block_n)
    check_splitk_counters(
        counters, num_tiles, x.device, "gfx950 FP32 router split-K GEMM"
    )
    partial = torch.empty(
        (split_k * num_tiles * block_m * block_n,),
        dtype=torch.float32,
        device=x.device,
    )
    gluon_fp32_router_gemv_splitk_gfx950[(num_tiles * split_k,)](
        x,
        weight,
        out,
        partial,
        counters,
        rows,
        K=k,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        CHUNKS=chunks,
        MULTIPLIER=float(multiplier),
        num_warps=waves,
    )
    return out


__all__ = [
    "gluon_fp32_router_gemv_num_counters_gfx950",
    "launch_gluon_fp32_router_gemv_gfx950",
    "transpose_fp32_router_weight_gfx950",
]
