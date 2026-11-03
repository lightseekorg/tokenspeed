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

"""Dense16 decode GEMV ``y = x @ w.T`` (M <= 16) for gfx950.

One program computes ``rows`` consecutive output columns over the full K, so
a launch has one program per CU (``rows = cdiv(N, CUs)``) and every program
issues its whole weight share without split-K or cross-CU reductions:

* M = 1, ``BLOCK_M == 1``: FMA tile. Waves split the rows, lanes split each
  K chunk with 16-byte loads (one load instruction reads a 1 KB contiguous
  row segment); per-lane FP32 partial sums, one cross-lane reduction. With
  few rows per wave (``STAGES > 0``) the share streams its 512-wide chunks
  through a ring of ``STAGES`` LDS buffers instead.
* M = 2..16, ``BLOCK_M == 16``: MFMA tile. The waves split every K
  iteration (a batched MFMA whose batch dimension is the wave) and reduce
  through LDS once. Weights load in a K-contiguous layout and reach the MFMA
  operand layout through LDS.

The summation order depends only on the constexprs, never on the row
blocking, so any ``rows`` gives the same bits. Weights load with ``.cg``
(``sc0 nt``: streamed, read once). The first K chunk of the weights is issued
before the activations are loaded.

The K loops are unrolled: a loop-carried register copy of a pending load
waits for that load, which serialized the chunks (2x slower).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4

__all__ = [
    "NUM_WARPS",
    "gluon_gemv_a16w16_gfx950",
    "launch_gluon_gemv_a16w16_gfx950",
    "prefers_gluon_gemv_a16w16_gfx950",
    "supports_gluon_gemv_a16w16_gfx950",
]

NUM_WARPS = 4
MAX_M = 16
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_WAVE = 64
# FMA tile: a lane loads 16 bytes (8 elements) of every 512-element chunk.
_FMA_MIN_CHUNK = 512
_STAGED_CHUNK = gl.constexpr(_FMA_MIN_CHUNK)
# Weight bytes per lane of one in-flight K chunk (two chunks are in flight).
_CHUNK_BYTES_PER_LANE = 128
# LDS and chunk count the staged M = 1 tile's weight ring may take; deeper
# rings measured no faster on MI350X, and two-chunk rings slower than the
# register tile.
_STAGED_LDS_BYTES = 64 * 1024
_STAGED_MIN_STAGES = 4
_STAGED_MAX_STAGES = 8
# The staged tile beat the register tile at up to this many rows per wave and
# lost by 1-8% above it in MI350X sweeps (N = 512-16384, K = 1024-32768).
_STAGED_MAX_ROWS_PER_WAVE = 4
# LDS per workgroup on gfx950, shared by the weight ring and the activation row.
_LDS_BYTES = 160 * 1024
# MFMA tile: weight rows per tile are a multiple of the 16-wide instruction.
_MFMA_BLOCK_M = 16
_MFMA_MIN_BLOCK_N = 16
_MFMA_LONG_K = 8192
_MFMA_LONG_K_WARPS = 8


def _gemv_launch_metadata(grid, kernel, args):
    """Report weight, activation and output bytes to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    elem = args["w_ptr"].element_size()
    return {
        "name": kernel.name,
        "bytes": (n * k + m * k + m * n) * elem,
        "flops": 2 * m * n * k,
    }


# ---------------------------------------------------------------------------
# FMA tile (M = 1)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _fma_layout(block_n: int, chunk: int, num_warps: int):
    # [rows, chunk / 8, 8]: lane l holds 8 * (chunk // 512) consecutive
    # elements of each row; waves own interleaved rows.
    return gl.BlockedLayout(
        [1, chunk // _FMA_MIN_CHUNK, 8], [1, _WAVE, 1], [num_warps, 1, 1], [2, 1, 0]
    )


@gluon.jit
def _fma_weight_offsets(
    tile,
    N,
    rows,
    K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    KC: gl.constexpr,
):
    """Offsets and mask of chunk 0 of the tile's weight rows."""
    L3: gl.constexpr = _fma_layout(BLOCK_N, KC, gl.num_warps())
    LW: gl.constexpr = gl.SliceLayout(2, L3)
    rn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, LW))
    rc = gl.arange(0, KC // 8, gl.SliceLayout(0, LW))
    re = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    col = tile * rows + rn
    row_ok = (rn < rows) & (col < N)
    wrow = gl.minimum(col, N - 1)
    k = rc[None, :, None] * 8 + re[None, None, :]
    offs = wrow[:, None, None] * K + k
    mask = row_ok[:, None, None] & (k < K)
    return offs, mask, k


@gluon.jit
def _fma_load_chunk(w_ptr, offs, mask, k, chunk, K: gl.constexpr, KC: gl.constexpr):
    return cdna4.buffer_load(
        w_ptr,
        offs + chunk * KC,
        mask=mask & (k + chunk * KC < K),
        other=0.0,
        cache=".cg",
    )


@gluon.jit
def _fma_tile(
    tile,
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    rows,
    K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    KC: gl.constexpr,
):
    L3: gl.constexpr = _fma_layout(BLOCK_N, KC, gl.num_warps())
    LW: gl.constexpr = gl.SliceLayout(2, L3)
    LX: gl.constexpr = gl.SliceLayout(0, L3)
    CHUNKS: gl.constexpr = (K + KC - 1) // KC
    offs, mask, k = _fma_weight_offsets(tile, N, rows, K, BLOCK_N, KC)
    w = _fma_load_chunk(w_ptr, offs, mask, k, 0, K, KC)
    xk = (
        gl.arange(0, KC // 8, gl.SliceLayout(1, LX))[:, None] * 8
        + gl.arange(0, 8, gl.SliceLayout(0, LX))[None, :]
    )
    x = cdna4.buffer_load(x_ptr, xk, mask=xk < K, other=0.0)
    acc = gl.zeros((BLOCK_N, KC // 8), gl.float32, LW)
    # Unrolled, so values flow between chunks without loop-carried copies
    # (copying a pending load waits for it). Chunk c + 1 is in flight while
    # chunk c is summed; loads return in order, so it is issued first.
    for c in gl.static_range(CHUNKS):
        if c + 1 < CHUNKS:
            next_w = _fma_load_chunk(w_ptr, offs, mask, k, c + 1, K, KC)
            next_x = cdna4.buffer_load(
                x_ptr, xk + (c + 1) * KC, mask=xk + (c + 1) * KC < K, other=0.0
            )
        # Keep the scheduler from sinking those loads below the FMAs.
        gl.amd.hint.sched_barrier()
        acc += gl.sum(w.to(gl.float32) * x.to(gl.float32)[None, :, :], axis=2)
        if c + 1 < CHUNKS:
            w = next_w
            x = next_x
    out = gl.sum(acc, axis=1)
    rn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, LW))
    col = tile * rows + rn
    cdna4.buffer_store(
        out.to(y_ptr.dtype.element_ty), y_ptr, col, mask=(rn < rows) & (col < N)
    )


@gluon.constexpr_function
def _next_power_of_2(n: int):
    return triton.next_power_of_2(n)


@gluon.constexpr_function
def _younger_refills(chunk: int, chunks: int, stages: int):
    # Refills of chunks after ``chunk`` issued by the time it is read.
    return min(chunk - 1, chunks - stages - 1) - (chunk - stages)


@gluon.constexpr_function
def _staged_shared_layout(rank: int):
    # Unswizzled: each wave's 64 x 16-byte copy fills one contiguous 1 KB row
    # segment, and every lane later reads back exactly the bytes it copied.
    return gl.SwizzledSharedLayout(1, 1, 1, list(range(rank - 1, -1, -1)))


@gluon.jit
def _staged_tile(
    tile,
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    rows,
    K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    STAGES: gl.constexpr,
):
    """M = 1 FMA tile whose weights stream through an LDS ring of ``STAGES``
    512-wide chunks filled by direct-to-LDS copies, which keeps more of a
    long share in flight than registers can. Same summation order as the
    register FMA tile at chunk 512."""
    CHUNK: gl.constexpr = _STAGED_CHUNK
    CHUNKS: gl.constexpr = (K + CHUNK - 1) // CHUNK
    XCHUNKS: gl.constexpr = _next_power_of_2(CHUNKS)
    L3: gl.constexpr = _fma_layout(BLOCK_N, CHUNK, gl.num_warps())
    LW: gl.constexpr = gl.SliceLayout(2, L3)
    LX: gl.constexpr = gl.SliceLayout(0, L3)
    # The first STAGES chunks of the weight rows, one commit group each.
    # Masked lanes read out of bounds, which the hardware fills with zeros;
    # an explicit `other` branched around every copy and kept each chunk's
    # registers live until the end of the unrolled loop, which spilled.
    smem = gl.allocate_shared_memory(
        w_ptr.dtype.element_ty,
        [STAGES, BLOCK_N, CHUNK // 8, 8],
        _staged_shared_layout(3),
    )
    offs, mask, k = _fma_weight_offsets(tile, N, rows, K, BLOCK_N, CHUNK)
    for s in gl.static_range(STAGES):
        cdna4.async_copy.buffer_load_to_shared(
            smem.index(s),
            w_ptr,
            offs + s * CHUNK,
            mask=mask & (k + s * CHUNK < K),
            cache_modifier=".cg",
        )
        cdna4.async_copy.commit_group()
    # The activation row as one more group. Every wave copies every chunk
    # and later reads back what it copied, so no workgroup barrier is needed.
    xsmem = gl.allocate_shared_memory(
        x_ptr.dtype.element_ty, [XCHUNKS, CHUNK // 8, 8], _staged_shared_layout(2)
    )
    xk = (
        gl.arange(0, CHUNK // 8, gl.SliceLayout(1, LX))[:, None] * 8
        + gl.arange(0, 8, gl.SliceLayout(0, LX))[None, :]
    )
    for c in gl.static_range(CHUNKS):
        cdna4.async_copy.buffer_load_to_shared(
            xsmem.index(c), x_ptr, xk + c * CHUNK, mask=xk + c * CHUNK < K
        )
    cdna4.async_copy.commit_group()
    acc = gl.zeros((BLOCK_N, CHUNK // 8), gl.float32, LW)
    cdna4.async_copy.wait_group(0)
    for c in gl.static_range(CHUNKS):
        if c >= STAGES:
            # Chunk c was the refill issued at iteration c - STAGES; refills
            # issued after it may stay in flight.
            cdna4.async_copy.wait_group(_younger_refills(c, CHUNKS, STAGES))
        w = cdna4.async_copy.load_shared_relaxed(smem.index(c % STAGES), L3)
        x = xsmem.index(c).load(LX)
        acc += gl.sum(w.to(gl.float32) * x.to(gl.float32)[None, :, :], axis=2)
        if c + STAGES < CHUNKS:
            # Each lane overwrites only the bytes it just read, after its
            # FMAs consumed them.
            gl.amd.hint.sched_barrier()
            cdna4.async_copy.buffer_load_to_shared(
                smem.index(c % STAGES),
                w_ptr,
                offs + (c + STAGES) * CHUNK,
                mask=mask & (k + (c + STAGES) * CHUNK < K),
                cache_modifier=".cg",
            )
            cdna4.async_copy.commit_group()
    out = gl.sum(acc, axis=1)
    rn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, LW))
    col = tile * rows + rn
    cdna4.buffer_store(
        out.to(y_ptr.dtype.element_ty), y_ptr, col, mask=(rn < rows) & (col < N)
    )


# ---------------------------------------------------------------------------
# MFMA tile (M = 2..16)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _mfma_layout(num_warps: int):
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1, 1],
    )


@gluon.constexpr_function
def _mfma_weight_layout(block_k: int, num_warps: int):
    # [waves, BLOCK_K, BLOCK_N] with 8 lanes x 16 B covering 128 contiguous
    # bytes of one weight row per instruction.
    tk = min(block_k // 8, _WAVE)
    return gl.BlockedLayout(
        [1, 8, 1], [1, tk, _WAVE // tk], [num_warps, 1, 1], [1, 2, 0]
    )


@gluon.jit
def _mfma_weight_offsets(
    tile,
    N,
    rows,
    K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Offsets and mask of iteration 0 of the tile's weight rows; wave ``s``
    takes ``BLOCK_K`` elements at ``s * BLOCK_K`` of an iteration."""
    WL: gl.constexpr = _mfma_weight_layout(BLOCK_K, gl.num_warps())
    s = gl.arange(0, gl.num_warps(), gl.SliceLayout(1, gl.SliceLayout(2, WL)))
    kk = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gl.SliceLayout(2, WL)))
    rn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, WL)))
    col = tile * rows + rn
    row_ok = (rn < rows) & (col < N)
    wrow = gl.minimum(col, N - 1)
    k = s[:, None, None] * BLOCK_K + kk[None, :, None]
    offs = k + wrow[None, None, :] * K
    mask = row_ok[None, None, :] & (k < K)
    return offs, mask, k


@gluon.jit
def _mfma_load_iter(w_ptr, offs, mask, k, it, K: gl.constexpr, KITER: gl.constexpr):
    return cdna4.buffer_load(
        w_ptr,
        offs + it * KITER,
        mask=mask & (k + it * KITER < K),
        other=0.0,
        cache=".cg",
    )


@gluon.jit
def _mfma_tile(
    tile,
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    rows,
    stride_xm,
    stride_ym,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    KS: gl.constexpr = gl.num_warps()
    KITER: gl.constexpr = KS * BLOCK_K
    ITERS: gl.constexpr = (K + KITER - 1) // KITER
    mfma: gl.constexpr = _mfma_layout(KS)
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    b_l: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    w_offs, w_mask, k = _mfma_weight_offsets(tile, N, rows, K, BLOCK_N, BLOCK_K)
    w = _mfma_load_iter(w_ptr, w_offs, w_mask, k, 0, K, KITER)
    # Activations [waves, BLOCK_M, BLOCK_K], loaded in the operand layout.
    a_s = gl.arange(0, KS, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    a_m = gl.arange(0, BLOCK_M, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    a_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    ak = a_s[:, None, None] * BLOCK_K + a_k[None, None, :]
    a_offs = a_m[None, :, None] * stride_xm + ak
    a_mask = (a_m < M)[None, :, None] & (ak < K)
    a = cdna4.buffer_load(x_ptr, a_offs, mask=a_mask, other=0.0)
    acc = gl.zeros((KS, BLOCK_M, BLOCK_N), gl.float32, mfma)
    # Unrolled, one K iteration in flight ahead of the MFMAs (see _fma_tile).
    for it in gl.static_range(ITERS):
        if it + 1 < ITERS:
            next_w = _mfma_load_iter(w_ptr, w_offs, w_mask, k, it + 1, K, KITER)
            next_a = cdna4.buffer_load(
                x_ptr,
                a_offs + (it + 1) * KITER,
                mask=a_mask & (ak + (it + 1) * KITER < K),
                other=0.0,
            )
        acc = cdna4.mfma(gl.convert_layout(a, a_l), gl.convert_layout(w, b_l), acc)
        if it + 1 < ITERS:
            w = next_w
            a = next_a
    # The waves' K partials, summed through LDS.
    out = gl.sum(acc, axis=0)
    out_l: gl.constexpr = gl.SliceLayout(0, mfma)
    o_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_l))
    o_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, out_l))
    col = tile * rows + o_n
    cdna4.buffer_store(
        out.to(y_ptr.dtype.element_ty),
        y_ptr,
        o_m[:, None] * stride_ym + col[None, :],
        mask=(o_m < M)[:, None] & ((o_n < rows) & (col < N))[None, :],
    )


# ---------------------------------------------------------------------------
# Kernel and launcher
# ---------------------------------------------------------------------------


@gluon.jit(launch_metadata=_gemv_launch_metadata, do_not_specialize=("M",))
def gluon_gemv_a16w16_gfx950(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    rows,
    stride_xm,
    stride_ym,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    STAGES: gl.constexpr,
):
    """Output columns ``[pid * rows, (pid + 1) * rows)`` of every row.

    ``x`` is ``[M, K]`` (row stride ``stride_xm``), ``w`` is ``[N, K]`` and
    ``y`` is ``[M, N]`` (row stride ``stride_ym``). ``BLOCK_N >= rows`` is
    the padded row block; ``BLOCK_K`` is the FMA chunk (``BLOCK_M == 1``) or
    the per-wave MFMA K step; ``STAGES > 0`` selects the LDS-staged M = 1
    tile.
    """
    tile = gl.program_id(0)
    if BLOCK_M == 1 and STAGES > 0:
        _staged_tile(tile, x_ptr, w_ptr, y_ptr, N, rows, K, BLOCK_N, STAGES)
    elif BLOCK_M == 1:
        _fma_tile(tile, x_ptr, w_ptr, y_ptr, N, rows, K, BLOCK_N, BLOCK_K)
    else:
        _mfma_tile(
            tile,
            x_ptr,
            w_ptr,
            y_ptr,
            M,
            N,
            rows,
            stride_xm,
            stride_ym,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )


def _config(m: int, n: int, k: int, cus: int) -> tuple[int, int, int, int, int, int]:
    """``(rows, BLOCK_M, BLOCK_N, BLOCK_K, STAGES, num_warps)`` of a GEMV
    with ``n`` outputs on ``cus`` CUs."""
    rows = triton.cdiv(n, cus)
    if m == 1:
        block_n = max(triton.next_power_of_2(rows), NUM_WARPS)
        rows_per_wave = block_n // NUM_WARPS
        # Largest chunk of at most _CHUNK_BYTES_PER_LANE weight bytes per lane.
        chunk = _CHUNK_BYTES_PER_LANE * _WAVE // (2 * rows_per_wave)
        chunk = max(min(triton.next_power_of_2(k), chunk), _FMA_MIN_CHUNK)
        chunks = triton.cdiv(k, _FMA_MIN_CHUNK)
        stage_bytes = block_n * _FMA_MIN_CHUNK * 2
        # The staged tile also holds the whole activation row in LDS, padded
        # to a power-of-2 chunk count; the ring gets what that leaves.
        x_bytes = triton.next_power_of_2(chunks) * _FMA_MIN_CHUNK * 2
        ring_bytes = min(_STAGED_LDS_BYTES, max(_LDS_BYTES - x_bytes, 0))
        stages = min(chunks, ring_bytes // stage_bytes, _STAGED_MAX_STAGES)
        stages = 1 << (stages.bit_length() - 1) if stages > 0 else 0  # power of 2
        staged = rows_per_wave <= _STAGED_MAX_ROWS_PER_WAVE
        if staged and stages >= _STAGED_MIN_STAGES:
            # Stream the share through an LDS ring of `stages` chunks; this
            # also beats one register chunk that covers all of K.
            return rows, 1, block_n, _FMA_MIN_CHUNK, stages, NUM_WARPS
        return rows, 1, block_n, chunk, 0, NUM_WARPS
    # Long K splits over eight waves, doubling the weight bytes in flight per CU.
    num_warps = _MFMA_LONG_K_WARPS if k >= _MFMA_LONG_K else NUM_WARPS
    block_k = 128 if k >= 128 * num_warps else 64
    block_n = max(triton.next_power_of_2(rows), _MFMA_MIN_BLOCK_N)
    return rows, _MFMA_BLOCK_M, block_n, block_k, 0, num_warps


def supports_gluon_gemv_a16w16_gfx950(m: int, n: int, k: int) -> bool:
    """Shapes the GEMV handles: ``1 <= M <= 16`` and K a multiple of 16."""
    return 1 <= m <= MAX_M and n >= 1 and k % 16 == 0 and k >= 64


# Decode shapes (N, K) and the M buckets (1, 2-4, 5-16) where this GEMV beat
# the default selection of ``gemm.mm`` (Torch's skinny GEMM at one row, the
# decode GEMM or hipBLASLt above) in cold-cache standalone sweeps on MI350X.
_PREFERRED_BUCKETS: dict[tuple[int, int], frozenset[int]] = {
    # Qwen3.8-Flash-Next, TP2 unless noted.
    (1280, 2560): frozenset({1, 4, 16}),  # shared-expert gate-up (TP1)
    (2560, 640): frozenset({1, 4, 16}),  # shared-expert down (TP1)
    (2560, 2560): frozenset({4, 16}),
    (2560, 3072): frozenset({4, 16}),  # GDN out_proj, attention o_proj
    (6144, 2560): frozenset({4}),
    (6656, 2560): frozenset({4, 16}),  # attention QKV with the output gate
    (512, 2560): frozenset({1, 4}),  # MoE router gate
    (640, 2560): frozenset({1, 4}),  # shared-expert gate-up, QSA indexer
    # Kimi K3 TP8.
    (1536, 1536): frozenset({4, 16}),
    (2304, 1536): frozenset({4, 16}),
    (3648, 7168): frozenset({4}),
    (6288, 7168): frozenset({4, 16}),
    (7168, 768): frozenset({4, 16}),
    (7168, 1536): frozenset({4, 16}),
    (7168, 3584): frozenset({4, 16}),
    (7168, 4224): frozenset({4, 16}),
    # GLM 5.3 Flash attention TP4.
    (160, 4096): frozenset({4}),
    (288, 4096): frozenset({4, 16}),  # MoE router gate
    (4096, 1536): frozenset({4, 16}),
    (4096, 2048): frozenset({16}),
    # DeepSeek V4.1 Flash TP4.
    (32, 5120): frozenset({1, 4, 16}),
    # Attention QKV with the output gate, 8192-wide hidden.
    (2304, 8192): frozenset({4, 16}),
}


def prefers_gluon_gemv_a16w16_gfx950(m: int, n: int, k: int) -> bool:
    """Whether this GEMV measured faster than the default selection here."""
    bucket = 1 if m == 1 else 4 if m <= 4 else 16 if m <= MAX_M else None
    return bucket in _PREFERRED_BUCKETS.get((n, k), frozenset())


def launch_gluon_gemv_a16w16_gfx950(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Run the decode GEMV ``out = x @ weight.T``.

    Args:
        x: ``[M, K]`` fp16/bf16 activations with unit inner stride,
            ``1 <= M <= 16`` and ``K`` a multiple of 16.
        weight: Contiguous ``[N, K]`` weight in ``x``'s dtype.
        out: ``[M, N]`` destination in ``x``'s dtype with unit inner stride.

    Returns:
        ``out``.
    """
    if x.ndim != 2 or weight.ndim != 2 or out.ndim != 2:
        raise ValueError("decode GEMV expects 2D x, weight and out")
    if x.dtype not in _SUPPORTED_DTYPES or not x.is_cuda:
        raise ValueError("decode GEMV expects fp16/bf16 GPU activations")
    for name, tensor in (("weight", weight), ("out", out)):
        if tensor.dtype != x.dtype or tensor.device != x.device:
            raise ValueError(f"{name} must match x's dtype and device")
    m, k = x.shape
    n, k_w = weight.shape
    if k_w != k or not weight.is_contiguous():
        raise ValueError(f"weight must be a contiguous [N, {k}] matrix")
    if tuple(out.shape) != (m, n) or out.stride(1) != 1 or x.stride(1) != 1:
        raise ValueError(f"out must be [{m}, {n}] and x and out row-major")
    if not supports_gluon_gemv_a16w16_gfx950(m, n, k):
        raise ValueError(f"decode GEMV does not support M={m}, N={n}, K={k}")
    cus = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows, block_m, block_n, block_k, stages, num_warps = _config(m, n, k, cus)
    gluon_gemv_a16w16_gfx950[(triton.cdiv(n, rows),)](
        x,
        weight,
        out,
        m,
        n,
        rows,
        x.stride(0),
        out.stride(0),
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        STAGES=stages,
        num_warps=num_warps,
    )
    return out
