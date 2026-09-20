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
32 values and FP32 accumulation. Automatic selection uses this kernel for
aligned, large-prefill projection shapes; other shapes use the portable
implementation.
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
MXFP8_MIN_K = 4 * MXFP8_BLOCK_K
MXFP8_K_UNROLL = 2 * MXFP8_BLOCK_K

_SUPPORTED_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}


def _mxfp8_launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    scale_values = (m + n) * (k // MXFP8_SCALE_GROUP)
    return {
        "name": kernel.name,
        "flops8": 2 * m * n * k,
        "bytes": m * k + n * k + scale_values + m * n * args["c_ptr"].element_size(),
    }


@gluon.jit
def _mxfp8_get_pids(
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """Distribute adjacent grouped tiles across gfx950's eight XCDs."""
    pid = gl.program_id(axis=0)
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


@gluon.jit(launch_metadata=_mxfp8_launch_metadata)
def _mxfp8_mm_kernel(
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
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ASYNC_SCALES: gl.constexpr,
):
    """Eight-wave, double-buffered E4M3xE4M3 scaled-MFMA GEMM."""
    SCALE_GROUP: gl.constexpr = 32
    pid_m, pid_n = _mxfp8_get_pids(
        M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M
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

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gload_a))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gload_a))
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gload_b))
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gload_b))
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    offs_as_k = gl.arange(0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, scale_a))
    offs_scale_m = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, scale_a))
    as_offsets = offs_scale_m[:, None] * stride_asm + offs_as_k[None, :] * stride_ask
    as_base = a_scales_ptr + pid_m * BLOCK_M * stride_asm

    offs_bs_k = gl.arange(0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, scale_b))
    offs_scale_n = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, scale_b))
    bs_offsets = offs_scale_n[:, None] * stride_bsn + offs_bs_k[None, :] * stride_bsk
    bs_base = b_scales_ptr + pid_n * BLOCK_N * stride_bsn

    offs_copy_scale_k = gl.arange(
        0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, gload_scale)
    )
    offs_copy_scale_row = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gload_scale))
    as_copy_offsets = (
        offs_copy_scale_row[:, None] * stride_asm
        + offs_copy_scale_k[None, :] * stride_ask
    )
    as_copy_offsets = gl.multiple_of(as_copy_offsets, [1, 4])
    offs_copy_scale_pair_k = gl.arange(
        0, 2 * BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, gload_scale_pair)
    )
    offs_copy_scale_pair_row = gl.arange(
        0, BLOCK_N, gl.SliceLayout(1, gload_scale_pair)
    )
    bs_pair_copy_offsets = (
        offs_copy_scale_pair_row[:, None] * stride_bsn
        + offs_copy_scale_pair_k[None, :] * stride_bsk
    )
    bs_pair_copy_offsets = gl.multiple_of(bs_pair_copy_offsets, [1, 4])

    a_half_m = (BLOCK_M // 2) * stride_am
    b_half_n = (BLOCK_N // 2) * stride_bn
    a_next_k = BLOCK_K * stride_ak
    b_next_k = BLOCK_K * stride_bk
    as_half_m = (BLOCK_M // 2) * stride_asm
    bs_half_n = (BLOCK_N // 2) * stride_bsn
    as_next_k = (BLOCK_K // SCALE_GROUP) * stride_ask
    bs_next_k = (BLOCK_K // SCALE_GROUP) * stride_bsk

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma)
    iterations = gl.cdiv(K, BLOCK_K)

    # A value tile and its scale tile share a commit group, so the eight-group
    # ping-pong schedule and wait distances are unchanged.
    async_copy.buffer_load_to_shared(smem_b_left.index(0), b_base, b_offsets)
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(smem_bs, bs_base, bs_pair_copy_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smem_a_top.index(0), a_base, a_offsets)
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(smem_as_top.index(0), as_base, as_copy_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smem_a_bot.index(0), a_base + a_half_m, a_offsets)
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(
            smem_as_bot.index(0), as_base + as_half_m, as_copy_offsets
        )
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        smem_b_right.index(0), b_base + b_half_n, b_offsets
    )
    async_copy.commit_group()

    async_copy.buffer_load_to_shared(smem_b_left.index(1), b_base + b_next_k, b_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smem_a_top.index(1), a_base + a_next_k, a_offsets)
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(
            smem_as_top.index(1), as_base + as_next_k, as_copy_offsets
        )
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        smem_a_bot.index(1), a_base + a_half_m + a_next_k, a_offsets
    )
    if ASYNC_SCALES:
        async_copy.buffer_load_to_shared(
            smem_as_bot.index(1),
            as_base + as_half_m + as_next_k,
            as_copy_offsets,
        )
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        smem_b_right.index(1), b_base + b_half_n + b_next_k, b_offsets
    )
    async_copy.commit_group()

    a_base += 2 * a_next_k
    b_base += 2 * b_next_k

    async_copy.wait_group(6)
    b_left = smem_b_left.index(0).load(dot_b)
    if ASYNC_SCALES:
        bs_left, bs_right, bs_left_next, bs_right_next = _load_scale_pair(
            smem_bs,
            scale_b_combined,
            scale_b,
            BLOCK_N // 2,
            BLOCK_K // SCALE_GROUP,
        )
    else:
        bs_left = gl.load(bs_base + bs_offsets)
    a_top = smem_a_top.index(0).load(dot_a)
    if ASYNC_SCALES:
        as_top = smem_as_top.index(0).load(scale_a)
    else:
        as_top = gl.load(as_base + as_offsets)
    gl.assume(iterations > 3)

    # Two K iterations expose eight alternating compute/memory clusters. The
    # phase-shifted wave groups keep one MFMA stream runnable on every SIMD.
    for _ in tl.range(0, iterations - 2, 2):
        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smem_a_bot.index(0).load(dot_a)
            if ASYNC_SCALES:
                as_bot = smem_as_bot.index(0).load(scale_a)
            else:
                as_bot = gl.load(as_base + as_half_m + as_offsets)
            async_copy.buffer_load_to_shared(smem_b_left.index(0), b_base, b_offsets)
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
            if not ASYNC_SCALES:
                bs_right = gl.load(bs_base + bs_half_n + bs_offsets)
            async_copy.buffer_load_to_shared(smem_a_top.index(0), a_base, a_offsets)
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_top.index(0),
                    as_base + 2 * as_next_k,
                    as_copy_offsets,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smem_b_left.index(1).load(dot_b)
            if not ASYNC_SCALES:
                bs_left = gl.load(bs_base + bs_next_k + bs_offsets)
            async_copy.buffer_load_to_shared(
                smem_a_bot.index(0), a_base + a_half_m, a_offsets
            )
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_bot.index(0),
                    as_base + as_half_m + 2 * as_next_k,
                    as_copy_offsets,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma_scaled(
                a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smem_a_top.index(1).load(dot_a)
            if ASYNC_SCALES:
                as_top = smem_as_top.index(1).load(scale_a)
                bs_left = bs_left_next
                bs_right = bs_right_next
            else:
                as_top = gl.load(as_base + as_next_k + as_offsets)
            async_copy.buffer_load_to_shared(
                smem_b_right.index(0), b_base + b_half_n, b_offsets
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smem_a_bot.index(1).load(dot_a)
            if ASYNC_SCALES:
                as_bot = smem_as_bot.index(1).load(scale_a)
            else:
                as_bot = gl.load(as_base + as_half_m + as_next_k + as_offsets)
            async_copy.buffer_load_to_shared(
                smem_b_left.index(1), b_base + b_next_k, b_offsets
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma_scaled(
                a_bot, as_bot, "e4m3", b_left, bs_left, "e4m3", acc_bl
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smem_b_right.index(1).load(dot_b)
            if not ASYNC_SCALES:
                bs_right = gl.load(bs_base + bs_half_n + bs_next_k + bs_offsets)
            async_copy.buffer_load_to_shared(
                smem_a_top.index(1), a_base + a_next_k, a_offsets
            )
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_top.index(1),
                    as_base + 3 * as_next_k,
                    as_copy_offsets,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma_scaled(
                a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smem_b_left.index(0).load(dot_b)
            if not ASYNC_SCALES:
                bs_left = gl.load(bs_base + 2 * bs_next_k + bs_offsets)
            async_copy.buffer_load_to_shared(
                smem_a_bot.index(1), a_base + a_half_m + a_next_k, a_offsets
            )
            if ASYNC_SCALES:
                async_copy.buffer_load_to_shared(
                    smem_as_bot.index(1),
                    as_base + as_half_m + 3 * as_next_k,
                    as_copy_offsets,
                )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma_scaled(
                a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br
            )
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smem_a_top.index(0).load(dot_a)
            if ASYNC_SCALES:
                as_top = smem_as_top.index(0).load(scale_a)
                bs_left, bs_right, bs_left_next, bs_right_next = _load_scale_pair(
                    smem_bs,
                    scale_b_combined,
                    scale_b,
                    BLOCK_N // 2,
                    BLOCK_K // SCALE_GROUP,
                )
            else:
                as_top = gl.load(as_base + 2 * as_next_k + as_offsets)
            async_copy.buffer_load_to_shared(
                smem_b_right.index(1), b_base + b_half_n + b_next_k, b_offsets
            )
            async_copy.commit_group()
            a_base += 2 * a_next_k
            b_base += 2 * b_next_k
            as_base += 2 * as_next_k
            bs_base += 2 * bs_next_k

    store_layout: gl.constexpr = gl.BlockedLayout(
        [4, 8], [4, 16], [WARPS_M, WARPS_N], [1, 0]
    )
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, store_layout))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, store_layout))
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tl_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_bl_base = c_tl_base + (BLOCK_M // 2) * stride_cm
    c_tr_base = c_tl_base + (BLOCK_N // 2) * stride_cn
    c_br_base = c_bl_base + (BLOCK_N // 2) * stride_cn

    # Drain the final two K tiles. Dot/scale fragments retire before conversion
    # and the four stores, keeping the hot loop within the VGPR budget.
    acc_tl = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl)
    async_copy.wait_group(5)
    local_index = (iterations - 2) % 2
    a_bot = smem_a_bot.index(local_index).load(dot_a)
    if ASYNC_SCALES:
        as_bot = smem_as_bot.index(local_index).load(scale_a)
    else:
        as_bot = gl.load(as_base + as_half_m + as_offsets)

    acc_bl = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_left, bs_left, "e4m3", acc_bl)
    async_copy.wait_group(4)
    b_right = smem_b_right.index(local_index).load(dot_b)
    if not ASYNC_SCALES:
        bs_right = gl.load(bs_base + bs_half_n + bs_offsets)

    acc_tr = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr)
    async_copy.wait_group(3)
    global_index = 1 - local_index
    b_left = smem_b_left.index(global_index).load(dot_b)
    if not ASYNC_SCALES:
        bs_left = gl.load(bs_base + bs_next_k + bs_offsets)

    acc_br = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br)
    if ASYNC_SCALES:
        bs_left = bs_left_next
        bs_right = bs_right_next
    async_copy.wait_group(2)
    a_top = smem_a_top.index(global_index).load(dot_a)
    if ASYNC_SCALES:
        as_top = smem_as_top.index(global_index).load(scale_a)
    else:
        as_top = gl.load(as_base + as_next_k + as_offsets)

    acc_tl = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_left, bs_left, "e4m3", acc_tl)
    async_copy.wait_group(1)
    a_bot = smem_a_bot.index(global_index).load(dot_a)
    if ASYNC_SCALES:
        as_bot = smem_as_bot.index(global_index).load(scale_a)
    else:
        as_bot = gl.load(as_base + as_half_m + as_next_k + as_offsets)

    acc_bl = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_left, bs_left, "e4m3", acc_bl)
    async_copy.wait_group(0)
    b_right = smem_b_right.index(global_index).load(dot_b)
    if not ASYNC_SCALES:
        bs_right = gl.load(bs_base + bs_half_n + bs_next_k + bs_offsets)

    acc_tr = cdna4.mfma_scaled(a_top, as_top, "e4m3", b_right, bs_right, "e4m3", acc_tr)
    acc_br = cdna4.mfma_scaled(a_bot, as_bot, "e4m3", b_right, bs_right, "e4m3", acc_br)

    c_tl = gl.convert_layout(acc_tl.to(c_ptr.dtype.element_ty), store_layout)
    cdna4.buffer_store(ptr=c_tl_base, offsets=c_offsets, stored_value=c_tl)
    c_bl = gl.convert_layout(acc_bl.to(c_ptr.dtype.element_ty), store_layout)
    cdna4.buffer_store(ptr=c_bl_base, offsets=c_offsets, stored_value=c_bl)
    c_tr = gl.convert_layout(acc_tr.to(c_ptr.dtype.element_ty), store_layout)
    cdna4.buffer_store(ptr=c_tr_base, offsets=c_offsets, stored_value=c_tr)
    c_br = gl.convert_layout(acc_br.to(c_ptr.dtype.element_ty), store_layout)
    cdna4.buffer_store(ptr=c_br_base, offsets=c_offsets, stored_value=c_br)


def supports_mxfp8_gemm_shape(m: int, n: int, k: int) -> bool:
    """Return whether the first large-prefill configuration covers the shape."""
    return (
        m >= MXFP8_BLOCK_M
        and m % MXFP8_BLOCK_M == 0
        and n % MXFP8_BLOCK_N == 0
        and k >= MXFP8_MIN_K
        and k % MXFP8_K_UNROLL == 0
    )


def _validate_scale(name: str, scale: torch.Tensor, rows: int, k: int) -> None:
    expected = (rows, k // MXFP8_SCALE_GROUP)
    if scale.dtype != torch.uint8 or tuple(scale.shape) != expected:
        raise ValueError(
            f"{name} must be row-major uint8 E8M0 with shape {expected}, "
            f"got dtype={scale.dtype}, shape={tuple(scale.shape)}"
        )


def gluon_mm_mxfp8_gfx950(
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
    """Compute an aligned MXFP8 ``A @ B.T`` projection on gfx950.

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
            "gfx950 MXFP8 GEMM requires M and N divisible by 256 and "
            "K >= 512 divisible by 256"
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
    async_scales = all(
        scale.stride(1) == 1 and scale.stride(0) % 4 == 0 and scale.data_ptr() % 4 == 0
        for scale in (A_scales, B_scales)
    )
    _mxfp8_mm_kernel[(grid_mn,)](
        A,
        B,
        A_scales,
        B_scales,
        output,
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
        output.stride(0),
        output.stride(1),
        BLOCK_M=MXFP8_BLOCK_M,
        BLOCK_N=MXFP8_BLOCK_N,
        BLOCK_K=MXFP8_BLOCK_K,
        WARPS_M=MXFP8_WARPS_M,
        WARPS_N=MXFP8_WARPS_N,
        GRID_MN=grid_mn,
        NUM_XCDS=MXFP8_NUM_XCDS,
        GROUP_SIZE_M=MXFP8_GROUP_SIZE_M,
        ASYNC_SCALES=async_scales,
        num_warps=MXFP8_NUM_WARPS,
        waves_per_eu=MXFP8_WAVES_PER_EU,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    if alpha is not None:
        output.mul_(alpha.to(device=output.device, dtype=output.dtype))
    return output


__all__ = [
    "gluon_mm_mxfp8_gfx950",
    "supports_mxfp8_gemm_shape",
]
