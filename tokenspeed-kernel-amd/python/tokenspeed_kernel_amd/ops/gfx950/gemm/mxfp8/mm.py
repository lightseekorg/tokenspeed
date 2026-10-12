# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

"""Large-prefill MXFP8 GEMM for gfx950.

This is adapted from the ROCm/gfx950-gluon-tutorials 8-wave ``inter_wave/a8w8``
and ``inter_wave/a4w4/v0_sliceMN`` kernels. Eight waves form two phase-shifted
waves per SIMD: one issues scaled MFMAs while the other advances LDS reads and
asynchronous global-to-LDS copies. The value and row-major E8M0 scale tiles use
separately allocated LDS storage. B scales combine both N quadrants and two K
steps per async copy. Strided scale views retain a direct-load fallback.

The kernel computes ``A @ B.T`` for E4M3 values with one uint8 E8M0 scale per
32 values and FP32 accumulation. Ragged M and N shift their last tile back
inside the matrix, a K tail shifts the K walk so only the prologue tiles are
masked, and under-filled long-K launches split K with an FP32 partial reduce.
See the ops README for the contract and routing.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy

MXFP8_BLOCK_M = 256
MXFP8_BLOCK_N = 256
MXFP8_BLOCK_K = 128
MXFP8_SCALE_GROUP = 32
MXFP8_NUM_WARPS = 8
MXFP8_WAVES_PER_EU = 2
MXFP8_WARPS_M = 2
MXFP8_WARPS_N = 4
MXFP8_NUM_XCDS = 8
MXFP8_GROUP_SIZE_M = 4
MXFP8_N_ALIGN = 16
# Scale staging: direct register loads, dword async copies alongside the
# values, or a one-time preload of the whole (short) K walk into LDS.
MXFP8_SCALES_DIRECT = gl.constexpr(0)
MXFP8_SCALES_ASYNC = gl.constexpr(1)
MXFP8_SCALES_PRELOAD = gl.constexpr(2)
MXFP8_PRELOAD_TILES = 8
MXFP8_NUM_CUS = 256
MXFP8_MIN_SPLIT_PAIRS = 2
MXFP8_MAX_SPLITS = 8
# Split-K cost model (microseconds), measured on MI355X: one 256x256x128 tile
# step of a workgroup, the FP32 partial write plus reduce read per byte, and
# the reduce launch.
MXFP8_TILE_US = 1.3
MXFP8_PARTIAL_US_PER_BYTE = 1 / 1.6e6
MXFP8_REDUCE_US = 5.0
MXFP8_REDUCE_BLOCK_M = 16
MXFP8_REDUCE_BLOCK_N = 256
MXFP8_K_UNROLL = 2 * MXFP8_BLOCK_K

_SUPPORTED_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}


def _mxfp8_launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    scale_values = (m + n) * (k // MXFP8_SCALE_GROUP)
    return {
        "name": kernel.name,
        "flops8": 2 * m * n * k,
        # Split-K writes one FP32 partial copy of the output per split.
        "bytes": m * k
        + n * k
        + scale_values
        + m * n * args["SPLITS"] * args["c_ptr"].element_size(),
    }


@gluon.jit
def _mxfp8_get_pids(
    pid,
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """Distribute adjacent grouped tiles across gfx950's eight XCDs."""
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        pids_per_xcd = gl.cdiv(GRID_MN, NUM_XCDS)
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def _load_scale_pair(
    smem_scale,
    combined_layout: gl.constexpr,
    half_layout: gl.constexpr,
    half_rows: gl.constexpr,
    scale_groups: gl.constexpr,
):
    """Load two K tiles of B scales and split their N and K halves."""
    combined = smem_scale.load(combined_layout)
    left_pair, right_pair = gl.split(
        gl.permute(combined.reshape([2, half_rows, 2 * scale_groups]), [1, 2, 0])
    )
    left, left_next = gl.split(
        gl.permute(left_pair.reshape([half_rows, 2, scale_groups]), [0, 2, 1])
    )
    right, right_next = gl.split(
        gl.permute(right_pair.reshape([half_rows, 2, scale_groups]), [0, 2, 1])
    )
    return (
        gl.convert_layout(left, half_layout),
        gl.convert_layout(right, half_layout),
        gl.convert_layout(left_next, half_layout),
        gl.convert_layout(right_next, half_layout),
    )


@gluon.jit
def _copy_k_tile(
    dest,
    base,
    offsets,
    offs_k,
    K_AXIS: gl.constexpr,
    k_start,
    shift,
    EVEN_K: gl.constexpr,
):
    """Copy one value tile, zero-filling elements below the shifted K origin."""
    if EVEN_K:
        async_copy.buffer_load_to_shared(dest, base, offsets)
    else:
        in_k = offs_k + k_start >= shift
        if K_AXIS == 0:
            mask = in_k[:, None]
        else:
            mask = in_k[None, :]
        async_copy.buffer_load_to_shared(dest, base, offsets, mask=mask)


@gluon.jit
def _load_scales(base, offsets, offs_k, k_group, shift_groups, EVEN_K: gl.constexpr):
    """Directly load a scale fragment, using 2^0 below the shifted K origin.

    The matching values are zero-filled; the fill only keeps a stray E8M0 NaN
    (0xFF) from turning those zero products into NaN.
    """
    if EVEN_K:
        return gl.load(base + offsets)
    else:
        mask = (offs_k + k_group >= shift_groups)[None, :]
        return gl.load(base + offsets, mask=mask, other=127)


@gluon.jit
def _scale_tile(
    ring, ring_index, full, tile, layout: gl.constexpr, SCALE_MODE: gl.constexpr
):
    """Read one K tile of scales from the async ring or the preloaded K walk."""
    if SCALE_MODE == MXFP8_SCALES_PRELOAD:
        return full.index(tile).load(layout)
    else:
        return ring.index(ring_index).load(layout)


@gluon.jit
def _scale_pair_buffer(ring, full, pair, SCALE_MODE: gl.constexpr):
    """Return the LDS tile holding two K tiles of B scales."""
    if SCALE_MODE == MXFP8_SCALES_PRELOAD:
        return full.index(pair)
    else:
        return ring


@gluon.jit
def _preload_scales(
    dest,
    base,
    row_start,
    row_last,
    stride_row,
    stride_k,
    groups,
    shift_groups,
    TILES: gl.constexpr,
    ROWS: gl.constexpr,
    TILE_GROUPS: gl.constexpr,
    layout: gl.constexpr,
):
    """Stage the scales of a whole shifted K walk as ``TILES x [ROWS, TILE_GROUPS]``.

    Rows are clamped to ``row_last``; groups outside ``[0, groups)`` of the
    unshifted K range read 2^0 and meet zero-filled values.
    """
    offs_r = gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    offs_g = gl.arange(0, TILE_GROUPS, gl.SliceLayout(0, layout))
    rows = gl.minimum(offs_r + row_start, row_last)
    for t in gl.static_range(TILES):
        group = offs_g + (t * TILE_GROUPS - shift_groups)
        offsets = rows[:, None] * stride_row + group[None, :] * stride_k
        mask = ((group >= 0) & (group < groups))[None, :]
        dest.index(t).store(gl.load(base + offsets, mask=mask, other=127))


# Row-count-derived scalars skip Triton's divisibility specialization so every
# batch shares one binary per scale/output class. The split stride stays
# specialized: it is 0 or M * N with N % 16 == 0, and its alignment keeps the
# output stores vectorized.
@gluon.jit(
    launch_metadata=_mxfp8_launch_metadata,
    do_not_specialize=("M", "GRID_MN", "SPLITS"),
)
def gluon_mm_mxfp8_gfx950(
    a_ptr,
    b_ptr,
    a_scales_ptr,
    b_scales_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    # Follows M (the token count); runtime so every batch shape shares one binary.
    GRID_MN,
    # Split-K partitions; follows M through the tile count, so it is runtime.
    SPLITS,
    stride_cs,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    SCALE_MODE: gl.constexpr,
    PRELOAD_TILES: gl.constexpr,
    EVEN_K: gl.constexpr,
    SMALL_M: gl.constexpr,
    SMALL_N: gl.constexpr,
):
    """Eight-wave, double-buffered E4M3xE4M3 scaled-MFMA GEMM."""
    SCALE_GROUP: gl.constexpr = 32
    ASYNC_SCALES: gl.constexpr = SCALE_MODE == MXFP8_SCALES_ASYNC
    PRELOAD_SCALES: gl.constexpr = SCALE_MODE == MXFP8_SCALES_PRELOAD
    LDS_SCALES: gl.constexpr = ASYNC_SCALES or PRELOAD_SCALES
    # Split-major launch order: every split covers all output tiles.
    pid = gl.program_id(axis=0)
    split = pid // GRID_MN
    pid_m, pid_n = _mxfp8_get_pids(
        pid - split * GRID_MN, M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M
    )

    # Every active value-loading lane transfers 16 aligned bytes. Promoting one
    # register basis to a warp basis expands the tutorial's four-wave layout to
    # eight waves without changing the physical transaction geometry.
    gload_a: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [16, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0]],
        warp_bases=[[2, 0], [4, 0], [8, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K],
    )
    gload_b: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        lane_bases=[[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2], [0, 4]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

    # Direct-to-LDS loads on gfx950 have a 32-bit minimum transaction width.
    # A scales are row-major, so each active lane owns the four adjacent
    # K-group bytes in one row. Only two waves are needed for a [128, 4] tile;
    # redundant waves are predicated by the lowering.
    gload_scale: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0])
    # Pairing two K steps makes all eight scale bytes for one row contiguous.
    # Adjacent lane pairs own the two four-byte K halves of one row, so each
    # wave's 32 rows form one contiguous 256-byte LDS destination run.
    gload_scale_pair: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [8, 1], [1, 0])

    # Padding breaks the stride aliases that otherwise make MFMA operand LDS
    # reads collide. The layouts are independent of the number of waves.
    shared_a: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [32, 0],
            [64, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
        ],
        [],
        [BLOCK_M // 2, BLOCK_K],
    )
    shared_b: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
        ],
        [],
        [BLOCK_K, BLOCK_N // 2],
    )
    shared_scale: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma, 16)
    scale_a: gl.constexpr = cdna4.get_mfma_scale_layout(
        dot_a, [BLOCK_M // 2, BLOCK_K // SCALE_GROUP]
    )
    scale_b: gl.constexpr = cdna4.get_mfma_scale_layout(
        dot_b, [BLOCK_N // 2, BLOCK_K // SCALE_GROUP]
    )
    scale_b_combined: gl.constexpr = cdna4.get_mfma_scale_layout(
        dot_b, [BLOCK_N, 2 * BLOCK_K // SCALE_GROUP]
    )

    buffers: gl.constexpr = 2
    smem_a_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [buffers, BLOCK_M // 2, BLOCK_K], shared_a
    )
    smem_a_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [buffers, BLOCK_M // 2, BLOCK_K], shared_a
    )
    smem_b_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [buffers, BLOCK_K, BLOCK_N // 2], shared_b
    )
    smem_b_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [buffers, BLOCK_K, BLOCK_N // 2], shared_b
    )
    if PRELOAD_SCALES:
        smem_as_top_all = gl.allocate_shared_memory(
            a_scales_ptr.dtype.element_ty,
            [PRELOAD_TILES, BLOCK_M // 2, BLOCK_K // SCALE_GROUP],
            shared_scale,
        )
        smem_as_bot_all = gl.allocate_shared_memory(
            a_scales_ptr.dtype.element_ty,
            [PRELOAD_TILES, BLOCK_M // 2, BLOCK_K // SCALE_GROUP],
            shared_scale,
        )
        smem_bs_all = gl.allocate_shared_memory(
            b_scales_ptr.dtype.element_ty,
            [PRELOAD_TILES // 2, BLOCK_N, 2 * BLOCK_K // SCALE_GROUP],
            shared_scale,
        )
        # The ring buffers are unused; alias them to keep one code path.
        smem_as_top = smem_as_top_all
        smem_as_bot = smem_as_bot_all
        smem_bs = smem_bs_all
    else:
        smem_as_top = gl.allocate_shared_memory(
            a_scales_ptr.dtype.element_ty,
            [buffers, BLOCK_M // 2, BLOCK_K // SCALE_GROUP],
            shared_scale,
        )
        smem_as_bot = gl.allocate_shared_memory(
            a_scales_ptr.dtype.element_ty,
            [buffers, BLOCK_M // 2, BLOCK_K // SCALE_GROUP],
            shared_scale,
        )
        smem_bs = gl.allocate_shared_memory(
            b_scales_ptr.dtype.element_ty,
            [BLOCK_N, 2 * BLOCK_K // SCALE_GROUP],
            shared_scale,
        )
        smem_as_top_all = smem_as_top
        smem_as_bot_all = smem_as_bot
        smem_bs_all = smem_bs

    # A ragged last tile is shifted back to end at row M (column N), so it
    # overlaps its neighbor and both write identical values; the hot loop and
    # store need no masks and no extra VGPRs. Below one tile, SMALL_M/SMALL_N
    # instead clamp each half's load rows to the last valid row and mask the
    # store. A K that is not a multiple of two K tiles shifts the K walk down by
    # `shift` elements, so only the two prologue tiles see a K boundary.
    if SMALL_M:
        m_start = 0
    else:
        m_start = gl.minimum(pid_m * BLOCK_M, M - BLOCK_M)
    if SMALL_N:
        n_start = 0
    else:
        n_start = gl.minimum(pid_n * BLOCK_N, N - BLOCK_N)
    m_rem = M - m_start
    n_rem = N - n_start
    # Each split walks a contiguous run of K-tile pairs.
    k_pairs = gl.cdiv(K, 2 * BLOCK_K)
    pair_lo = split * k_pairs // SPLITS
    pair_hi = (split + 1) * k_pairs // SPLITS
    iterations = 2 * (pair_hi - pair_lo)
    k_tile0 = 2 * pair_lo
    if EVEN_K:
        # Keeps the base pointers' alignment provable for dword scale copies.
        shift = 0
    else:
        shift = 2 * k_pairs * BLOCK_K - K
    shift_groups = shift // SCALE_GROUP

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gload_a))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gload_a))
    offs_as_k = gl.arange(0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, scale_a))
    offs_scale_m = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, scale_a))
    offs_copy_scale_k = gl.arange(
        0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, gload_scale)
    )
    offs_copy_scale_row = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gload_scale))
    if SMALL_M:
        a_rows_top = gl.minimum(offs_am, m_rem - 1)
        a_rows_bot = gl.minimum(offs_am + BLOCK_M // 2, m_rem - 1)
        as_rows_top = gl.minimum(offs_scale_m, m_rem - 1)
        as_rows_bot = gl.minimum(offs_scale_m + BLOCK_M // 2, m_rem - 1)
        as_copy_rows_top = gl.minimum(offs_copy_scale_row, m_rem - 1)
        as_copy_rows_bot = gl.minimum(offs_copy_scale_row + BLOCK_M // 2, m_rem - 1)
        a_bot_delta = 0
        as_bot_delta = 0
    else:
        a_rows_top = offs_am
        a_rows_bot = offs_am
        as_rows_top = offs_scale_m
        as_rows_bot = offs_scale_m
        as_copy_rows_top = offs_copy_scale_row
        as_copy_rows_bot = offs_copy_scale_row
        a_bot_delta = (BLOCK_M // 2) * stride_am
        as_bot_delta = (BLOCK_M // 2) * stride_asm
    a_off_top = a_rows_top[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_off_bot = a_rows_bot[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_base = a_ptr + m_start * stride_am + (k_tile0 * BLOCK_K - shift) * stride_ak
    as_off_top = as_rows_top[:, None] * stride_asm + offs_as_k[None, :] * stride_ask
    as_off_bot = as_rows_bot[:, None] * stride_asm + offs_as_k[None, :] * stride_ask
    as_row_start = m_start * stride_asm
    if ASYNC_SCALES:
        # The launcher checks dword-aligned scale rows for this path.
        as_row_start = gl.multiple_of(as_row_start, 4)
    as_base = (
        a_scales_ptr
        + as_row_start
        + (k_tile0 * (BLOCK_K // SCALE_GROUP) - shift_groups) * stride_ask
    )
    as_copy_top = (
        as_copy_rows_top[:, None] * stride_asm + offs_copy_scale_k[None, :] * stride_ask
    )
    as_copy_top = gl.multiple_of(as_copy_top, [1, 4])
    as_copy_bot = (
        as_copy_rows_bot[:, None] * stride_asm + offs_copy_scale_k[None, :] * stride_ask
    )
    as_copy_bot = gl.multiple_of(as_copy_bot, [1, 4])

    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gload_b))
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gload_b))
    offs_bs_k = gl.arange(0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, scale_b))
    offs_scale_n = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, scale_b))
    offs_copy_scale_pair_k = gl.arange(
        0, 2 * BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, gload_scale_pair)
    )
    offs_copy_scale_pair_row = gl.arange(
        0, BLOCK_N, gl.SliceLayout(1, gload_scale_pair)
    )
    if SMALL_N:
        b_cols_left = gl.minimum(offs_bn, n_rem - 1)
        b_cols_right = gl.minimum(offs_bn + BLOCK_N // 2, n_rem - 1)
        bs_rows_left = gl.minimum(offs_scale_n, n_rem - 1)
        bs_rows_right = gl.minimum(offs_scale_n + BLOCK_N // 2, n_rem - 1)
        bs_pair_rows = gl.minimum(offs_copy_scale_pair_row, n_rem - 1)
        b_right_delta = 0
        bs_right_delta = 0
    else:
        b_cols_left = offs_bn
        b_cols_right = offs_bn
        bs_rows_left = offs_scale_n
        bs_rows_right = offs_scale_n
        bs_pair_rows = offs_copy_scale_pair_row
        b_right_delta = (BLOCK_N // 2) * stride_bn
        bs_right_delta = (BLOCK_N // 2) * stride_bsn
    b_off_left = offs_bk[:, None] * stride_bk + b_cols_left[None, :] * stride_bn
    b_off_right = offs_bk[:, None] * stride_bk + b_cols_right[None, :] * stride_bn
    b_base = b_ptr + n_start * stride_bn + (k_tile0 * BLOCK_K - shift) * stride_bk
    bs_off_left = bs_rows_left[:, None] * stride_bsn + offs_bs_k[None, :] * stride_bsk
    bs_off_right = bs_rows_right[:, None] * stride_bsn + offs_bs_k[None, :] * stride_bsk
    bs_row_start = n_start * stride_bsn
    if ASYNC_SCALES:
        bs_row_start = gl.multiple_of(bs_row_start, 4)
    bs_base = (
        b_scales_ptr
        + bs_row_start
        + (k_tile0 * (BLOCK_K // SCALE_GROUP) - shift_groups) * stride_bsk
    )
    bs_pair_copy_offsets = (
        bs_pair_rows[:, None] * stride_bsn
        + offs_copy_scale_pair_k[None, :] * stride_bsk
    )
    bs_pair_copy_offsets = gl.multiple_of(bs_pair_copy_offsets, [1, 4])
    k_group = k_tile0 * (BLOCK_K // SCALE_GROUP)

    if PRELOAD_SCALES:
        # Scale rows that are not dword aligned cannot use async copies; for a
        # short K walk, stage all of its scales once so the hot loop still
        # reads them from LDS and issues no register loads.
        preload_a: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [2, 4], [1, 0])
        preload_b: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [4, 2], [1, 0])
        k_groups = K // SCALE_GROUP
        as_rows = a_scales_ptr + m_start * stride_asm
        bs_rows = b_scales_ptr + n_start * stride_bsn
        _preload_scales(
            smem_as_top_all,
            as_rows,
            0,
            m_rem - 1,
            stride_asm,
            stride_ask,
            k_groups,
            shift_groups,
            PRELOAD_TILES,
            BLOCK_M // 2,
            BLOCK_K // SCALE_GROUP,
            preload_a,
        )
        _preload_scales(
            smem_as_bot_all,
            as_rows,
            BLOCK_M // 2,
            m_rem - 1,
            stride_asm,
            stride_ask,
            k_groups,
            shift_groups,
            PRELOAD_TILES,
            BLOCK_M // 2,
            BLOCK_K // SCALE_GROUP,
            preload_a,
        )
        _preload_scales(
            smem_bs_all,
            bs_rows,
            0,
            n_rem - 1,
            stride_bsn,
            stride_bsk,
            k_groups,
            shift_groups,
            PRELOAD_TILES // 2,
            BLOCK_N,
            2 * BLOCK_K // SCALE_GROUP,
            preload_b,
        )

    a_next_k = BLOCK_K * stride_ak
    b_next_k = BLOCK_K * stride_bk
    as_next_k = (BLOCK_K // SCALE_GROUP) * stride_ask
    bs_next_k = (BLOCK_K // SCALE_GROUP) * stride_bsk

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)

    # A value tile and its scale tile share a commit group, so the eight-group
    # ping-pong schedule and wait distances are unchanged. Only these first two
    # K tiles can start below the shifted K origin; asynchronous scale copies
    # are limited to EVEN_K, where no shift exists.
    _copy_k_tile(
        smem_b_left.index(0),
        b_base,
        b_off_left,
        offs_bk,
        0,
        k_tile0 * BLOCK_K,
        shift,
        EVEN_K,
    )
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(smem_bs, bs_base, bs_pair_copy_offsets)
    async_copy.commit_group()
    _copy_k_tile(
        smem_a_top.index(0),
        a_base,
        a_off_top,
        offs_ak,
        1,
        k_tile0 * BLOCK_K,
        shift,
        EVEN_K,
    )
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(smem_as_top.index(0), as_base, as_copy_top)
    async_copy.commit_group()
    _copy_k_tile(
        smem_a_bot.index(0),
        a_base + a_bot_delta,
        a_off_bot,
        offs_ak,
        1,
        k_tile0 * BLOCK_K,
        shift,
        EVEN_K,
    )
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(
            smem_as_bot.index(0), as_base + as_bot_delta, as_copy_bot
        )
    async_copy.commit_group()
    _copy_k_tile(
        smem_b_right.index(0),
        b_base + b_right_delta,
        b_off_right,
        offs_bk,
        0,
        k_tile0 * BLOCK_K,
        shift,
        EVEN_K,
    )
    async_copy.commit_group()

    _copy_k_tile(
        smem_b_left.index(1),
        b_base + b_next_k,
        b_off_left,
        offs_bk,
        0,
        k_tile0 * BLOCK_K + BLOCK_K,
        shift,
        EVEN_K,
    )
    async_copy.commit_group()
    _copy_k_tile(
        smem_a_top.index(1),
        a_base + a_next_k,
        a_off_top,
        offs_ak,
        1,
        k_tile0 * BLOCK_K + BLOCK_K,
        shift,
        EVEN_K,
    )
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(
            smem_as_top.index(1), as_base + as_next_k, as_copy_top
        )
    async_copy.commit_group()
    _copy_k_tile(
        smem_a_bot.index(1),
        a_base + a_bot_delta + a_next_k,
        a_off_bot,
        offs_ak,
        1,
        k_tile0 * BLOCK_K + BLOCK_K,
        shift,
        EVEN_K,
    )
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(
            smem_as_bot.index(1), as_base + as_bot_delta + as_next_k, as_copy_bot
        )
    async_copy.commit_group()
    _copy_k_tile(
        smem_b_right.index(1),
        b_base + b_right_delta + b_next_k,
        b_off_right,
        offs_bk,
        0,
        k_tile0 * BLOCK_K + BLOCK_K,
        shift,
        EVEN_K,
    )
    async_copy.commit_group()

    a_base += 2 * a_next_k
    b_base += 2 * b_next_k

    async_copy.wait_group(6)
    b_left = smem_b_left.index(0).load(dot_b)
    if LDS_SCALES:
        bs_left, bs_right, bs_left_next, bs_right_next = _load_scale_pair(
            _scale_pair_buffer(smem_bs, smem_bs_all, k_group // 8, SCALE_MODE),
            scale_b_combined,
            scale_b,
            BLOCK_N // 2,
            BLOCK_K // SCALE_GROUP,
        )
    else:
        bs_left = _load_scales(
            bs_base, bs_off_left, offs_bs_k, k_group, shift_groups, EVEN_K
        )
    a_top = smem_a_top.index(0).load(dot_a)
    if LDS_SCALES:
        as_top = _scale_tile(
            smem_as_top, 0, smem_as_top_all, k_group // 4, scale_a, SCALE_MODE
        )
    else:
        as_top = _load_scales(
            as_base, as_off_top, offs_as_k, k_group, shift_groups, EVEN_K
        )
    gl.assume(iterations > 3)

    for _ in tl.range(0, iterations - 2, 2):
        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smem_a_bot.index(0).load(dot_a)
            if LDS_SCALES:
                as_bot = _scale_tile(
                    smem_as_bot, 0, smem_as_bot_all, k_group // 4, scale_a, SCALE_MODE
                )
            else:
                as_bot = _load_scales(
                    as_base + as_bot_delta,
                    as_off_bot,
                    offs_as_k,
                    k_group,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(smem_b_left.index(0), b_base, b_off_left)
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_bs,
                    bs_base + 2 * bs_next_k,
                    bs_pair_copy_offsets,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma_scaled(
                a_bot, as_bot, "e4m3", b_left, bs_left, "e4m3", acc_bl
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smem_b_right.index(0).load(dot_b)
            if not LDS_SCALES:
                bs_right = _load_scales(
                    bs_base + bs_right_delta,
                    bs_off_right,
                    offs_bs_k,
                    k_group,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(smem_a_top.index(0), a_base, a_off_top)
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_top.index(0),
                    as_base + 2 * as_next_k,
                    as_copy_top,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smem_b_left.index(1).load(dot_b)
            if not LDS_SCALES:
                bs_left = _load_scales(
                    bs_base + 1 * bs_next_k,
                    bs_off_left,
                    offs_bs_k,
                    k_group + 4,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(
                smem_a_bot.index(0), a_base + a_bot_delta, a_off_bot
            )
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_bot.index(0),
                    as_base + as_bot_delta + 2 * as_next_k,
                    as_copy_bot,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma_scaled(
                a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smem_a_top.index(1).load(dot_a)
            if LDS_SCALES:
                as_top = _scale_tile(
                    smem_as_top,
                    1,
                    smem_as_top_all,
                    k_group // 4 + 1,
                    scale_a,
                    SCALE_MODE,
                )
                bs_left = bs_left_next
                bs_right = bs_right_next
            else:
                as_top = _load_scales(
                    as_base + 1 * as_next_k,
                    as_off_top,
                    offs_as_k,
                    k_group + 4,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(
                smem_b_right.index(0), b_base + b_right_delta, b_off_right
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smem_a_bot.index(1).load(dot_a)
            if LDS_SCALES:
                as_bot = _scale_tile(
                    smem_as_bot,
                    1,
                    smem_as_bot_all,
                    k_group // 4 + 1,
                    scale_a,
                    SCALE_MODE,
                )
            else:
                as_bot = _load_scales(
                    as_base + as_bot_delta + 1 * as_next_k,
                    as_off_bot,
                    offs_as_k,
                    k_group + 4,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(
                smem_b_left.index(1), b_base + b_next_k, b_off_left
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma_scaled(
                a_bot, as_bot, "e4m3", b_left, bs_left, "e4m3", acc_bl
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smem_b_right.index(1).load(dot_b)
            if not LDS_SCALES:
                bs_right = _load_scales(
                    bs_base + bs_right_delta + 1 * bs_next_k,
                    bs_off_right,
                    offs_bs_k,
                    k_group + 4,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(
                smem_a_top.index(1), a_base + a_next_k, a_off_top
            )
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_top.index(1),
                    as_base + 3 * as_next_k,
                    as_copy_top,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smem_b_left.index(0).load(dot_b)
            if not LDS_SCALES:
                bs_left = _load_scales(
                    bs_base + 2 * bs_next_k,
                    bs_off_left,
                    offs_bs_k,
                    k_group + 8,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(
                smem_a_bot.index(1), a_base + a_bot_delta + a_next_k, a_off_bot
            )
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_bot.index(1),
                    as_base + as_bot_delta + 3 * as_next_k,
                    as_copy_bot,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma_scaled(
                a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smem_a_top.index(0).load(dot_a)
            if LDS_SCALES:
                as_top = _scale_tile(
                    smem_as_top,
                    0,
                    smem_as_top_all,
                    k_group // 4 + 2,
                    scale_a,
                    SCALE_MODE,
                )
                bs_left, bs_right, bs_left_next, bs_right_next = _load_scale_pair(
                    _scale_pair_buffer(
                        smem_bs, smem_bs_all, k_group // 8 + 1, SCALE_MODE
                    ),
                    scale_b_combined,
                    scale_b,
                    BLOCK_N // 2,
                    BLOCK_K // SCALE_GROUP,
                )
            else:
                as_top = _load_scales(
                    as_base + 2 * as_next_k,
                    as_off_top,
                    offs_as_k,
                    k_group + 8,
                    shift_groups,
                    EVEN_K,
                )
            async_copy.buffer_load_to_shared(
                smem_b_right.index(1), b_base + b_right_delta + b_next_k, b_off_right
            )
            async_copy.commit_group()
            a_base += 2 * a_next_k
            b_base += 2 * b_next_k
            as_base += 2 * as_next_k
            bs_base += 2 * bs_next_k
            k_group += 2 * (BLOCK_K // SCALE_GROUP)

    store_layout: gl.constexpr = gl.BlockedLayout(
        [4, 8], [4, 16], [WARPS_M, WARPS_N], [1, 0]
    )
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, store_layout))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, store_layout))
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tl_base = c_ptr + split * stride_cs + m_start * stride_cm + n_start * stride_cn
    c_bl_base = c_tl_base + (BLOCK_M // 2) * stride_cm
    c_tr_base = c_tl_base + (BLOCK_N // 2) * stride_cn
    c_br_base = c_bl_base + (BLOCK_N // 2) * stride_cn
    # Drain the final two K tiles. Dot/scale fragments retire before conversion
    # and the four stores, keeping the hot loop within the VGPR budget.
    acc_tl = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl)
    async_copy.wait_group(5)
    local_index = (iterations - 2) % 2
    a_bot = smem_a_bot.index(local_index).load(dot_a)
    if LDS_SCALES:
        as_bot = _scale_tile(
            smem_as_bot, local_index, smem_as_bot_all, k_group // 4, scale_a, SCALE_MODE
        )
    else:
        as_bot = _load_scales(
            as_base + as_bot_delta, as_off_bot, offs_as_k, k_group, shift_groups, EVEN_K
        )

    acc_bl = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_left, bs_left, "e4m3", acc_bl)
    async_copy.wait_group(4)
    b_right = smem_b_right.index(local_index).load(dot_b)
    if not LDS_SCALES:
        bs_right = _load_scales(
            bs_base + bs_right_delta,
            bs_off_right,
            offs_bs_k,
            k_group,
            shift_groups,
            EVEN_K,
        )

    acc_tr = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr)
    async_copy.wait_group(3)
    global_index = 1 - local_index
    b_left = smem_b_left.index(global_index).load(dot_b)
    if not LDS_SCALES:
        bs_left = _load_scales(
            bs_base + 1 * bs_next_k,
            bs_off_left,
            offs_bs_k,
            k_group + 4,
            shift_groups,
            EVEN_K,
        )

    acc_br = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br)
    if LDS_SCALES:
        bs_left = bs_left_next
        bs_right = bs_right_next
    async_copy.wait_group(2)
    a_top = smem_a_top.index(global_index).load(dot_a)
    if LDS_SCALES:
        as_top = _scale_tile(
            smem_as_top,
            global_index,
            smem_as_top_all,
            k_group // 4 + 1,
            scale_a,
            SCALE_MODE,
        )
    else:
        as_top = _load_scales(
            as_base + 1 * as_next_k,
            as_off_top,
            offs_as_k,
            k_group + 4,
            shift_groups,
            EVEN_K,
        )

    acc_tl = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl)
    async_copy.wait_group(1)
    a_bot = smem_a_bot.index(global_index).load(dot_a)
    if LDS_SCALES:
        as_bot = _scale_tile(
            smem_as_bot,
            global_index,
            smem_as_bot_all,
            k_group // 4 + 1,
            scale_a,
            SCALE_MODE,
        )
    else:
        as_bot = _load_scales(
            as_base + as_bot_delta + 1 * as_next_k,
            as_off_bot,
            offs_as_k,
            k_group + 4,
            shift_groups,
            EVEN_K,
        )

    acc_bl = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_left, bs_left, "e4m3", acc_bl)
    async_copy.wait_group(0)
    b_right = smem_b_right.index(global_index).load(dot_b)
    if not LDS_SCALES:
        bs_right = _load_scales(
            bs_base + bs_right_delta + 1 * bs_next_k,
            bs_off_right,
            offs_bs_k,
            k_group + 4,
            shift_groups,
            EVEN_K,
        )

    acc_tr = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr)
    acc_br = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br)

    if SMALL_M or SMALL_N:
        top_mask = offs_cm[:, None] < m_rem
        bot_mask = offs_cm[:, None] + BLOCK_M // 2 < m_rem
        left_mask = offs_cn[None, :] < n_rem
        right_mask = offs_cn[None, :] + BLOCK_N // 2 < n_rem
        c_tl = gl.convert_layout(acc_tl.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_tl, c_tl_base, c_offsets, mask=top_mask & left_mask)
        c_bl = gl.convert_layout(acc_bl.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_bl, c_bl_base, c_offsets, mask=bot_mask & left_mask)
        c_tr = gl.convert_layout(acc_tr.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_tr, c_tr_base, c_offsets, mask=top_mask & right_mask)
        c_br = gl.convert_layout(acc_br.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_br, c_br_base, c_offsets, mask=bot_mask & right_mask)
    else:
        c_tl = gl.convert_layout(acc_tl.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_tl, c_tl_base, c_offsets)
        c_bl = gl.convert_layout(acc_bl.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_bl, c_bl_base, c_offsets)
        c_tr = gl.convert_layout(acc_tr.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_tr, c_tr_base, c_offsets)
        c_br = gl.convert_layout(acc_br.to(c_ptr.dtype.element_ty), store_layout)
        cdna4.buffer_store(c_br, c_br_base, c_offsets)


def _mxfp8_reduce_launch_metadata(grid, kernel, args):
    """Expose split-K partial traffic to Proton."""
    m, n, splits = args["M"], args["N"], args["SPLITS"]
    return {
        "name": kernel.name,
        "bytes": m * n * (4 * splits + args["c_ptr"].element_size()),
    }


# The partial stride (M * N with N % 16 == 0) stays specialized for vector loads.
@gluon.jit(
    launch_metadata=_mxfp8_reduce_launch_metadata,
    do_not_specialize=("M", "SPLITS"),
)
def gluon_mm_mxfp8_reduce_gfx950(
    partials_ptr,
    c_ptr,
    M,
    N,
    SPLITS,
    stride_ps,
    stride_pm,
    stride_cm,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Sum FP32 split-K partials in split order and write the output dtype."""
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    pid_n = gl.program_id(axis=0)
    pid_m = gl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    p_offsets = offs_m[:, None] * stride_pm + offs_n[None, :]
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, layout)
    for split in range(SPLITS):
        acc += cdna4.buffer_load(partials_ptr + split * stride_ps, p_offsets, mask=mask)
    c_offsets = offs_m[:, None] * stride_cm + offs_n[None, :]
    cdna4.buffer_store(acc.to(c_ptr.dtype.element_ty), c_ptr, c_offsets, mask=mask)


def supports_mxfp8_gemm_shape(m: int, n: int, k: int) -> bool:
    """Return whether the kernel covers the shape.

    M is arbitrary. N must keep 16-byte output rows aligned, and K must hold
    whole scale groups and at least three K tiles (the K walk is rounded up to
    an even number of at least four tiles).
    """
    return (
        m >= 1
        and n >= 1
        and n % MXFP8_N_ALIGN == 0
        and k > MXFP8_K_UNROLL
        and k % MXFP8_SCALE_GROUP == 0
    )


def _mxfp8_num_splits(m: int, n: int, k: int) -> int:
    """Pick the split-K partition count for a shape.

    Splitting shortens each workgroup's K walk but adds an FP32 partial round
    trip through memory. The measured cost model below picks the cheapest
    count that keeps every split within one wave of workgroups and at least
    MXFP8_MIN_SPLIT_PAIRS K-tile pairs long.
    """
    tiles = triton.cdiv(m, MXFP8_BLOCK_M) * triton.cdiv(n, MXFP8_BLOCK_N)
    pairs = triton.cdiv(k, MXFP8_K_UNROLL)
    best, best_us = 1, 2 * pairs * MXFP8_TILE_US
    for splits in range(2, MXFP8_MAX_SPLITS + 1):
        if tiles * splits > MXFP8_NUM_CUS or pairs < splits * MXFP8_MIN_SPLIT_PAIRS:
            break
        estimate_us = (
            2 * triton.cdiv(pairs, splits) * MXFP8_TILE_US
            + splits * m * n * 4 * MXFP8_PARTIAL_US_PER_BYTE
            + MXFP8_REDUCE_US
        )
        if estimate_us < best_us:
            best, best_us = splits, estimate_us
    return best


def _validate_scale(name: str, scale: torch.Tensor, rows: int, k: int) -> None:
    expected = (rows, k // MXFP8_SCALE_GROUP)
    if scale.dtype != torch.uint8 or tuple(scale.shape) != expected:
        raise ValueError(
            f"{name} must be row-major uint8 E8M0 with shape {expected}, "
            f"got dtype={scale.dtype}, shape={tuple(scale.shape)}"
        )


def launch_gluon_mm_mxfp8_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None,
    block_size: list[int],
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute an MXFP8 ``A @ B.T`` projection on gfx950.

    Args:
        A: K-contiguous E4M3 activation matrix shaped ``[M, K]``.
        B: K-contiguous E4M3 weight matrix shaped ``[N, K]``.
        A_scales: Strided uint8 E8M0 scales shaped ``[M, K/32]``.
        B_scales: Strided uint8 E8M0 scales shaped ``[N, K/32]``.
        out_dtype: FP16 or BF16 output element type.
        alpha: Optional output multiplier applied after GEMM.
        block_size: Explicit logical scale block, required to be ``[1, 32]``.
        out: Optional row-contiguous output buffer shaped ``[M, N]``.

    Returns:
        The supplied ``out`` tensor or a newly allocated ``[M, N]`` tensor.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("gfx950 MXFP8 GEMM requires rank-2 A and B")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise TypeError("gfx950 MXFP8 GEMM requires E4M3 A and B")
    if not A.is_cuda or not B.is_cuda or A.device != B.device:
        raise ValueError("gfx950 MXFP8 GEMM requires colocated GPU operands")
    if A.stride(-1) != 1 or B.stride(-1) != 1:
        raise ValueError("gfx950 MXFP8 GEMM requires K-contiguous A and B")
    if block_size != [1, MXFP8_SCALE_GROUP]:
        raise ValueError(
            f"gfx950 MXFP8 GEMM requires block_size=[1, {MXFP8_SCALE_GROUP}]"
        )
    if out_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise TypeError(f"gfx950 MXFP8 GEMM does not support output {out_dtype}")

    m, k = A.shape
    n, b_k = B.shape
    if b_k != k:
        raise ValueError(f"gfx950 MXFP8 GEMM K mismatch: A={k}, B={b_k}")
    if not supports_mxfp8_gemm_shape(m, n, k):
        raise ValueError(
            "gfx950 MXFP8 GEMM requires N divisible by 16 and K > 256 "
            "divisible by 32"
        )
    if A_scales.device != A.device or B_scales.device != A.device:
        raise ValueError("gfx950 MXFP8 scales must share the operand device")
    _validate_scale("A_scales", A_scales, m, k)
    _validate_scale("B_scales", B_scales, n, k)

    if out is None:
        output = torch.empty((m, n), device=A.device, dtype=out_dtype)
    else:
        output = out
        if (
            tuple(output.shape) != (m, n)
            or output.dtype != out_dtype
            or output.device != A.device
            or output.stride(1) != 1
        ):
            raise ValueError(
                "gfx950 MXFP8 out must have matching shape/device/dtype and "
                "unit inner stride"
            )

    grid_mn = triton.cdiv(m, MXFP8_BLOCK_M) * triton.cdiv(n, MXFP8_BLOCK_N)
    even_k = k % MXFP8_K_UNROLL == 0
    # Dword scale copies need 4-byte aligned rows and cannot mask the partial
    # scale words of a shifted K walk.
    async_scales = even_k and all(
        scale.stride(1) == 1 and scale.stride(0) % 4 == 0 and scale.data_ptr() % 4 == 0
        for scale in (A_scales, B_scales)
    )
    if async_scales:
        scale_mode = MXFP8_SCALES_ASYNC
    elif triton.cdiv(k, MXFP8_K_UNROLL) * 2 <= MXFP8_PRELOAD_TILES:
        scale_mode = MXFP8_SCALES_PRELOAD
    else:
        scale_mode = MXFP8_SCALES_DIRECT
    splits = _mxfp8_num_splits(m, n, k)
    if splits > 1:
        target = torch.empty((splits, m, n), device=A.device, dtype=torch.float32)
    else:
        target = output
    gluon_mm_mxfp8_gfx950[(grid_mn * splits,)](
        A,
        B,
        A_scales,
        B_scales,
        target,
        m,
        n,
        k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        A_scales.stride(0),
        A_scales.stride(1),
        B_scales.stride(0),
        B_scales.stride(1),
        target.stride(-2),
        target.stride(-1),
        BLOCK_M=MXFP8_BLOCK_M,
        BLOCK_N=MXFP8_BLOCK_N,
        BLOCK_K=MXFP8_BLOCK_K,
        WARPS_M=MXFP8_WARPS_M,
        WARPS_N=MXFP8_WARPS_N,
        GRID_MN=grid_mn,
        SPLITS=splits,
        stride_cs=target.stride(0) if splits > 1 else 0,
        NUM_XCDS=MXFP8_NUM_XCDS,
        GROUP_SIZE_M=MXFP8_GROUP_SIZE_M,
        SCALE_MODE=scale_mode,
        PRELOAD_TILES=MXFP8_PRELOAD_TILES,
        EVEN_K=even_k,
        # Two classes per dimension: below one tile, or tiles shifted to fit.
        SMALL_M=m < MXFP8_BLOCK_M,
        SMALL_N=n < MXFP8_BLOCK_N,
        num_warps=MXFP8_NUM_WARPS,
        waves_per_eu=MXFP8_WAVES_PER_EU,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    if splits > 1:
        gluon_mm_mxfp8_reduce_gfx950[
            (
                triton.cdiv(n, MXFP8_REDUCE_BLOCK_N),
                triton.cdiv(m, MXFP8_REDUCE_BLOCK_M),
            )
        ](
            target,
            output,
            m,
            n,
            splits,
            target.stride(0),
            target.stride(1),
            output.stride(0),
            BLOCK_M=MXFP8_REDUCE_BLOCK_M,
            BLOCK_N=MXFP8_REDUCE_BLOCK_N,
            num_warps=4,
        )
    if alpha is not None:
        output.mul_(alpha.to(device=output.device, dtype=output.dtype))
    return output


__all__ = [
    "launch_gluon_mm_mxfp8_gfx950",
    "supports_mxfp8_gemm_shape",
]
