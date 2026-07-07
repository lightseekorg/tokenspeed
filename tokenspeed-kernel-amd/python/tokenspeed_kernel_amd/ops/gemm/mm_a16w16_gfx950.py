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

"""Dense16 Gluon GEMM kernels for gfx950.

The public entry point computes ``A @ B.T`` for supported 16-bit floating-point
dense inputs and returns ``None`` when the generic caller should use its torch
fallback.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gemm.mm_a16w16_largem_gfx950 import (
    _supports_largem_shape,
    gluon_mm_a16w16_largem_gfx950,
)

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

DENSE16_BLOCK_M = 16
DENSE16_BLOCK_N = 128
DENSE16_BLOCK_K = 64
WARP_REDUCE_OUTPUTS = 1
WARP_REDUCE_BLOCK_K = 1024
MFMA_LDS_BLOCK_N = 256
MFMA_LDS_BLOCK_K = 64
MFMA_LDS_NUM_WARPS = 4
MFMA_LDS_REDUCE_M = 4
MFMA_LDS_SPLIT_K = 8
GROUP_SIZE_M = 2
MAX_M = 256
LARGEM_DISPATCH_MIN_M = 2048
KERNEL_NAME = "gluon_mm_a16w16_gfx950"

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_partial_cache: dict[tuple[int, int, int, int, int], torch.Tensor] = {}


@gluon.jit
def _tile_to_pid(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_SIZE_M: gl.constexpr,
):
    if GROUP_SIZE_M == 1:
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.constexpr_function
def _mfma_lds_gload_layout_a(block_m: int, block_k: int, _num_warps: int):
    if _num_warps == 8 and block_m == 128 and block_k == 64:
        return gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]],
            block_bases=[],
            shape=[block_m, block_k],
        )
    if block_m <= 16:
        if block_k >= 512:
            return gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
        if block_k >= 256:
            return gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
        if block_k >= 128:
            return gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        return gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    if block_m <= 32:
        if block_k >= 512:
            return gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
        if block_k >= 256:
            return gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
        if block_k >= 128:
            return gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        return gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    if block_m <= 64:
        return gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
    if block_k >= 1024:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 64],
                [0, 128],
                [0, 256],
                [0, 512],
                [4, 0],
                [8, 0],
            ],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[],
            shape=[block_m, block_k],
        )
    if block_k >= 512:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 64],
                [0, 128],
                [0, 256],
                [4, 0],
                [8, 0],
            ],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[],
            shape=[block_m, block_k],
        )
    if block_k >= 256:
        return gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 64], [0, 128], [4, 0], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[],
            shape=[block_m, block_k],
        )
    if block_k >= 128:
        return gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 64], [4, 0], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[],
            shape=[block_m, block_k],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[block_m, block_k],
    )


@gluon.constexpr_function
def _mfma_lds_gload_layout_b(block_n: int, block_k: int, _num_warps: int):
    if _num_warps == 8 and block_n == 128 and block_k == 64:
        return gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2], [0, 4]],
            block_bases=[],
            shape=[block_k, block_n],
        )
    if block_n <= 16:
        if block_k >= 512:
            return gl.BlockedLayout([8, 1], [64, 1], [1, 4], [0, 1])
        if block_k >= 256:
            return gl.BlockedLayout([8, 1], [32, 2], [1, 4], [0, 1])
        if block_k >= 128:
            return gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
        return gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    if block_n <= 32:
        if block_k >= 512:
            return gl.BlockedLayout([8, 1], [64, 1], [1, 4], [0, 1])
        if block_k >= 256:
            return gl.BlockedLayout([8, 1], [32, 2], [1, 4], [0, 1])
        if block_k >= 128:
            return gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
        return gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    if block_n <= 64:
        return gl.BlockedLayout([8, 1], [4, 16], [1, 4], [0, 1])
    if block_k >= 1024:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [1, 0],
                [2, 0],
                [4, 0],
                [64, 0],
                [128, 0],
                [256, 0],
                [512, 0],
                [0, 4],
                [0, 8],
            ],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2]],
            block_bases=[],
            shape=[block_k, block_n],
        )
    if block_k >= 512:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [1, 0],
                [2, 0],
                [4, 0],
                [64, 0],
                [128, 0],
                [256, 0],
                [0, 4],
                [0, 8],
            ],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2]],
            block_bases=[],
            shape=[block_k, block_n],
        )
    if block_k >= 256:
        return gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [0, 4], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2]],
            block_bases=[],
            shape=[block_k, block_n],
        )
    if block_k >= 128:
        return gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [64, 0], [0, 4], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[[0, 1], [0, 2]],
            block_bases=[],
            shape=[block_k, block_n],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[block_k, block_n],
    )


@gluon.constexpr_function
def _mfma_lds_manual_shared_offsets_a(block_m: int, block_k: int):
    bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]]
    if block_k >= 128:
        bases += [[0, 64]]
    if block_k >= 256:
        bases += [[0, 128]]
    if block_k >= 512:
        bases += [[0, 256]]
    if block_k >= 1024:
        bases += [[0, 512]]
    bases += [[1, 0], [2, 0], [4, 0], [8, 0]]
    if block_m > 16:
        bases += [[16, 0]]
    if block_m > 32:
        bases += [[32, 0]]
    if block_m > 64:
        bases += [[64, 0]]
    return bases


@gluon.constexpr_function
def _mfma_lds_manual_shared_offsets_b(block_n: int, block_k: int):
    bases = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]]
    if block_k >= 128:
        bases += [[64, 0]]
    if block_k >= 256:
        bases += [[128, 0]]
    if block_k >= 512:
        bases += [[256, 0]]
    if block_k >= 1024:
        bases += [[512, 0]]
    bases += [[0, 1], [0, 2], [0, 4], [0, 8]]
    if block_n > 16:
        bases += [[0, 16]]
    if block_n > 32:
        bases += [[0, 32]]
    if block_n > 64:
        bases += [[0, 64]]
    return bases


@gluon.constexpr_function
def _mfma_lds_manual_shared_layout_a(block_m: int, block_k: int):
    return gl.PaddedSharedLayout(
        [[512, 16]],
        _mfma_lds_manual_shared_offsets_a(block_m, block_k),
        [],
        [block_m, block_k],
    )


@gluon.constexpr_function
def _mfma_lds_manual_shared_layout_b(block_n: int, block_k: int):
    return gl.PaddedSharedLayout(
        [[512, 16]],
        _mfma_lds_manual_shared_offsets_b(block_n, block_k),
        [],
        [block_k, block_n],
    )


@gluon.constexpr_function
def _mfma_lds_shared_layout_a(dot_layout_a, block_m: int, block_k: int, dtype):
    if block_m <= 64:
        return _mfma_lds_manual_shared_layout_a(block_m, block_k)
    layout = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_layout_a,
        [block_m, block_k],
        dtype,
        is_k_contig=True,
    )
    if layout is not None:
        return layout
    return _mfma_lds_manual_shared_layout_a(block_m, block_k)


@gluon.constexpr_function
def _mfma_lds_shared_layout_b(dot_layout_b, block_n: int, block_k: int, dtype):
    if block_n <= 64:
        return _mfma_lds_manual_shared_layout_b(block_n, block_k)
    layout = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_layout_b,
        [block_k, block_n],
        dtype,
        is_k_contig=True,
    )
    if layout is not None:
        return layout
    return _mfma_lds_manual_shared_layout_b(block_n, block_k)


@gluon.jit
def _warp_reduce_smallm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_OUT: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Warp-reduce dense16 GEMM for very small-M low/medium-K decode tiles."""
    pid = gl.program_id(axis=0)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [1, 64],
        [BLOCK_OUT, 1],
        [1, 0],
    )
    out_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)

    lane_out = gl.arange(0, BLOCK_OUT, out_layout)
    offs_k = gl.arange(0, BLOCK_K, k_layout)
    out_idx = pid * BLOCK_OUT + lane_out
    n = out_idx // M
    m = out_idx - n * M
    valid_out = out_idx < (M * N)

    acc = gl.zeros((BLOCK_OUT,), gl.float32, out_layout)
    for kt in range(0, gl.cdiv(K, BLOCK_K)):
        k = kt * BLOCK_K + offs_k
        a_offsets = (m[:, None] * stride_am + k[None, :] * stride_ak).to(gl.int32)
        b_offsets = (n[:, None] * stride_bn + k[None, :] * stride_bk).to(gl.int32)
        k_valid = k < K
        a = cdna4.buffer_load(
            ptr=a_ptr,
            offsets=a_offsets,
            mask=valid_out[:, None] & k_valid[None, :],
            other=0.0,
        ).to(gl.float32)
        b = cdna4.buffer_load(
            ptr=b_ptr,
            offsets=b_offsets,
            mask=valid_out[:, None] & k_valid[None, :],
            other=0.0,
        ).to(gl.float32)
        acc += gl.sum(a * b, axis=1)

    c_offsets = (m * stride_cm + n * stride_cn).to(gl.int32)
    cdna4.buffer_store(
        ptr=c_ptr,
        offsets=c_offsets,
        stored_value=acc.to(c_ptr.dtype.element_ty),
        mask=valid_out,
    )


@gluon.jit
def _mfma_lds_smallm_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    partial_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    PARTIAL_M: gl.constexpr,
    K_TILES_PER_SPLIT: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
):
    """Split-K producer for high-K M in {1, 2, 4} dense16 decode GEMM."""
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    total_work = num_pid_n * SPLIT_K

    gLoadLayoutBbig: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_K],
    )
    gLoadLayoutAsmall: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [16, 4],
        [4, 1],
        [1, 0],
    )
    sharedLayoutBbig: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [128, 0],
        ],
        [],
        [BLOCK_N, BLOCK_K],
    )
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    dotOpLayoutBbig: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutAsmall: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    storeLayout: gl.constexpr = mfmaLayout

    nBuffers: gl.constexpr = 2
    smem_bbig = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_N, BLOCK_K], sharedLayoutBbig
    )

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, gLoadLayoutBbig))
    offs_bk_big = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutBbig))
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutAsmall))
    offs_ak_small = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutAsmall))

    offs_store_n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, storeLayout))
    offs_store_m = gl.arange(0, BLOCK_M, gl.SliceLayout(0, storeLayout))
    partial_offsets = offs_store_m[None, :] * BLOCK_N + offs_store_n[:, None]

    gl.static_assert(K_TILES_PER_SPLIT >= 2, "MFMA LDS GEMM needs at least two K tiles")

    work_id = pid
    while work_id < total_work:
        tile_n = work_id // SPLIT_K
        split_id = work_id - tile_n * SPLIT_K
        k_split_start = split_id * K_TILES_PER_SPLIT * BLOCK_K
        n_base = tile_n * BLOCK_N
        n_load = n_base + offs_bn
        n_store = n_base + offs_store_n

        bbig_base_offsets = (
            n_load[:, None] * stride_bn + offs_bk_big[None, :] * stride_bk
        ).to(gl.int32)
        asmall_base_offsets = (
            offs_am[:, None] * stride_am + offs_ak_small[None, :] * stride_ak
        ).to(gl.int32)
        asmall_mask = offs_am[:, None] < M

        bbig_offsets_0 = (bbig_base_offsets + k_split_start * stride_bk).to(gl.int32)
        asmall_offsets_0 = (asmall_base_offsets + k_split_start * stride_ak).to(
            gl.int32
        )
        async_copy.buffer_load_to_shared(smem_bbig.index(0), b_ptr, bbig_offsets_0)
        async_copy.commit_group()

        bbig_offsets_1 = (bbig_base_offsets + (k_split_start + BLOCK_K) * stride_bk).to(
            gl.int32
        )
        async_copy.buffer_load_to_shared(smem_bbig.index(1), b_ptr, bbig_offsets_1)
        async_copy.commit_group()

        async_copy.wait_group(1)
        bbig = smem_bbig.index(0).load(dotOpLayoutBbig)
        asmall = cdna4.buffer_load(
            ptr=a_ptr,
            offsets=asmall_offsets_0,
            mask=asmall_mask,
            other=0.0,
        )
        asmall = gl.convert_layout(asmall.trans(1, 0), layout=dotOpLayoutAsmall)
        acc = gl.zeros((BLOCK_N, BLOCK_M), gl.float32, mfmaLayout)

        for kt in gl.static_range(0, K_TILES_PER_SPLIT - 2):
            acc = cdna4.mfma(bbig, asmall, acc)

            next_tile = kt + 2
            g_idx = kt % nBuffers
            l_idx = (kt + 1) % nBuffers
            prefetch_k_advance = k_split_start + next_tile * BLOCK_K
            bbig_offsets = (bbig_base_offsets + prefetch_k_advance * stride_bk).to(
                gl.int32
            )
            async_copy.buffer_load_to_shared(
                smem_bbig.index(g_idx), b_ptr, bbig_offsets
            )
            async_copy.commit_group()
            async_copy.wait_group(1)

            bbig = smem_bbig.index(l_idx).load(dotOpLayoutBbig)
            load_k_advance = k_split_start + (kt + 1) * BLOCK_K
            asmall_offsets = (asmall_base_offsets + load_k_advance * stride_ak).to(
                gl.int32
            )
            asmall = cdna4.buffer_load(
                ptr=a_ptr,
                offsets=asmall_offsets,
                mask=asmall_mask,
                other=0.0,
            )
            asmall = gl.convert_layout(asmall.trans(1, 0), layout=dotOpLayoutAsmall)

        acc = cdna4.mfma(bbig, asmall, acc)
        async_copy.wait_group(0)

        last_idx: gl.constexpr = (K_TILES_PER_SPLIT - 1) % nBuffers
        bbig = smem_bbig.index(last_idx).load(dotOpLayoutBbig)
        last_k_advance = k_split_start + (K_TILES_PER_SPLIT - 1) * BLOCK_K
        asmall_offsets = (asmall_base_offsets + last_k_advance * stride_ak).to(gl.int32)
        asmall = cdna4.buffer_load(
            ptr=a_ptr,
            offsets=asmall_offsets,
            mask=asmall_mask,
            other=0.0,
        )
        asmall = gl.convert_layout(asmall.trans(1, 0), layout=dotOpLayoutAsmall)
        acc = cdna4.mfma(bbig, asmall, acc)

        acc_store = gl.convert_layout(acc, layout=storeLayout)
        c_base = c_ptr + n_base * stride_cn
        c_offsets = (
            offs_store_n[:, None] * stride_cn + offs_store_m[None, :] * stride_cm
        )
        c_mask = (n_store[:, None] < N) & (offs_store_m[None, :] < M)
        partial_mask = c_mask & (offs_store_m[None, :] < PARTIAL_M)

        if SPLIT_K == 1:
            cdna4.buffer_store(
                ptr=c_base,
                offsets=c_offsets,
                stored_value=acc_store.to(c_ptr.dtype.element_ty),
                mask=c_mask,
            )
        else:
            tile_elems: gl.constexpr = BLOCK_N * PARTIAL_M
            split_partial_offset = (tile_n * SPLIT_K + split_id) * tile_elems
            cdna4.buffer_store(
                ptr=partial_ptr,
                offsets=(split_partial_offset + partial_offsets).to(gl.int32),
                stored_value=acc_store,
                mask=partial_mask,
            )

        work_id += NUM_PROGRAMS


@gluon.jit
def _mfma_lds_smallm_reduce_kernel(
    partial_ptr,
    c_ptr,
    stride_cm,
    stride_cn,
    BLOCK_N: gl.constexpr,
    REDUCE_M: gl.constexpr,
    SPLIT_K: gl.constexpr,
):
    """Partial-sum reducer for small-M split-K dense16 partials."""
    tile_n = gl.program_id(axis=0)
    n_base = tile_n * BLOCK_N

    reduceLayout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 64],
        [4, 1],
        [1, 0],
    )
    offs_m = gl.arange(0, REDUCE_M, gl.SliceLayout(1, reduceLayout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, reduceLayout))
    partial_offsets = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    c_offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    tile_elems: gl.constexpr = BLOCK_N * REDUCE_M
    tile_partial_offset = tile_n * SPLIT_K * tile_elems
    reduced = gl.zeros((REDUCE_M, BLOCK_N), gl.float32, reduceLayout)
    for split in gl.static_range(SPLIT_K):
        split_offset = tile_partial_offset + split * tile_elems
        reduced += cdna4.buffer_load(
            ptr=partial_ptr,
            offsets=(split_offset + partial_offsets).to(gl.int32),
        ).to(gl.float32)

    c_base = c_ptr + n_base * stride_cn
    cdna4.buffer_store(
        ptr=c_base,
        offsets=c_offsets,
        stored_value=reduced.to(c_ptr.dtype.element_ty),
    )


@gluon.jit
def _mfma_lds_mediumm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """MFMA/LDS dense16 GEMM for selected 8 <= M <= 128 decode tiles."""
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _tile_to_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    gl.static_assert(
        BLOCK_M % 16 == 0, "tiled MFMA GEMM requires BLOCK_M multiple of 16"
    )
    gl.static_assert(
        BLOCK_N % 16 == 0, "tiled MFMA GEMM requires BLOCK_N multiple of 16"
    )
    gl.static_assert(
        BLOCK_K == 64
        or BLOCK_K == 128
        or BLOCK_K == 256
        or BLOCK_K == 512
        or BLOCK_K == 1024,
        "tiled MFMA GEMM expects BLOCK_K in {64, 128, 256, 512, 1024}",
    )
    gl.static_assert(
        NUM_BUFFERS == 1 or NUM_BUFFERS == 2 or NUM_BUFFERS == 3,
        "tiled MFMA GEMM uses a one-, two-, or three-buffer LDS pipeline",
    )

    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    gl.static_assert(
        NUM_WARPS == 4 or NUM_WARPS == 8,
        "tiled MFMA GEMM currently uses four- or eight-wave load layouts",
    )
    if NUM_WARPS == 8:
        gl.static_assert(
            BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64,
            "eight-wave tiled MFMA GEMM supports only 128x128x64 tiles",
        )
    if BLOCK_K == 1024:
        gl.static_assert(
            BLOCK_M == 16 and BLOCK_N == 16 and NUM_BUFFERS == 2,
            "1024-depth tiled MFMA GEMM fits LDS only as 16x16x1024 with two buffers",
        )
    gLoadLayoutA: gl.constexpr = _mfma_lds_gload_layout_a(BLOCK_M, BLOCK_K, NUM_WARPS)
    gLoadLayoutB: gl.constexpr = _mfma_lds_gload_layout_b(BLOCK_N, BLOCK_K, NUM_WARPS)

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    sharedLayoutA: gl.constexpr = _mfma_lds_shared_layout_a(
        dotOpLayoutA, BLOCK_M, BLOCK_K, a_ptr.dtype.element_ty
    )
    sharedLayoutB: gl.constexpr = _mfma_lds_shared_layout_b(
        dotOpLayoutB, BLOCK_N, BLOCK_K, b_ptr.dtype.element_ty
    )

    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_K, BLOCK_N], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_tile_base = pid_m * BLOCK_M * stride_am
    b_tile_base = pid_n * BLOCK_N * stride_bn

    a_offs0 = (
        a_tile_base + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    ).to(gl.int32)
    a_offs1 = (a_offs0 + BLOCK_K * stride_ak).to(gl.int32)
    a_offs2 = (a_offs0 + 2 * BLOCK_K * stride_ak).to(gl.int32)
    b_offs0 = (
        b_tile_base + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    ).to(gl.int32)
    b_offs1 = (b_offs0 + BLOCK_K * stride_bk).to(gl.int32)
    b_offs2 = (b_offs0 + 2 * BLOCK_K * stride_bk).to(gl.int32)
    a_kstep = NUM_BUFFERS * BLOCK_K * stride_ak
    b_kstep = NUM_BUFFERS * BLOCK_K * stride_bk

    m = pid_m * BLOCK_M + offs_am
    a_mask = m[:, None] < M

    k_tiles: gl.constexpr = K // BLOCK_K
    if NUM_BUFFERS == 1:
        gl.static_assert(
            k_tiles == 1, "one-buffer tiled MFMA GEMM supports exactly one K tile"
        )
    elif NUM_BUFFERS == 2:
        gl.static_assert(
            k_tiles >= 2, "two-buffer tiled MFMA GEMM needs at least two K tiles"
        )
    else:
        gl.static_assert(
            k_tiles >= 3, "three-buffer tiled MFMA GEMM needs at least three K tiles"
        )

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    if NUM_BUFFERS == 1:
        async_copy.buffer_load_to_shared(smemA.index(0), a_ptr, a_offs0, mask=a_mask)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemB.index(0), b_ptr, b_offs0)
        async_copy.commit_group()
        async_copy.wait_group(0)
        a = async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
        b = async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)
        acc = cdna4.mfma(a, b, acc)

    elif NUM_BUFFERS == 2:
        async_copy.buffer_load_to_shared(smemA.index(0), a_ptr, a_offs0, mask=a_mask)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemB.index(0), b_ptr, b_offs0)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemA.index(1), a_ptr, a_offs1, mask=a_mask)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemB.index(1), b_ptr, b_offs1)
        async_copy.commit_group()
        a_offs0 = (a_offs0 + a_kstep).to(gl.int32)
        a_offs1 = (a_offs1 + a_kstep).to(gl.int32)
        b_offs0 = (b_offs0 + b_kstep).to(gl.int32)
        b_offs1 = (b_offs1 + b_kstep).to(gl.int32)

        async_copy.wait_group(2)
        a = async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
        b = async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)

        main_loop_pairs: gl.constexpr = (k_tiles - NUM_BUFFERS) // NUM_BUFFERS
        for _ in tl.range(0, main_loop_pairs):
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc = cdna4.mfma(a, b, acc)
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                async_copy.wait_group(0)
                a = async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
                b = async_copy.load_shared_relaxed(smemB.index(1), dotOpLayoutB)
                async_copy.buffer_load_to_shared(
                    smemA.index(0), a_ptr, a_offs0, mask=a_mask
                )
                async_copy.commit_group()
                async_copy.buffer_load_to_shared(smemB.index(0), b_ptr, b_offs0)
                async_copy.commit_group()

            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc = cdna4.mfma(a, b, acc)
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                async_copy.wait_group(0)
                a = async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
                b = async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)
                async_copy.buffer_load_to_shared(
                    smemA.index(1), a_ptr, a_offs1, mask=a_mask
                )
                async_copy.commit_group()
                async_copy.buffer_load_to_shared(smemB.index(1), b_ptr, b_offs1)
                async_copy.commit_group()
                a_offs0 = (a_offs0 + a_kstep).to(gl.int32)
                a_offs1 = (a_offs1 + a_kstep).to(gl.int32)
                b_offs0 = (b_offs0 + b_kstep).to(gl.int32)
                b_offs1 = (b_offs1 + b_kstep).to(gl.int32)

        tiles_processed: gl.constexpr = main_loop_pairs * NUM_BUFFERS
        tiles_remaining: gl.constexpr = k_tiles - tiles_processed

        acc = cdna4.mfma(a, b, acc)
        async_copy.wait_group(0)

        a = async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
        b = async_copy.load_shared_relaxed(smemB.index(1), dotOpLayoutB)
        acc = cdna4.mfma(a, b, acc)

        if tiles_remaining > 2:
            async_copy.buffer_load_to_shared(
                smemA.index(0), a_ptr, a_offs0, mask=a_mask
            )
            async_copy.commit_group()
            async_copy.buffer_load_to_shared(smemB.index(0), b_ptr, b_offs0)
            async_copy.commit_group()
            async_copy.wait_group(0)
            a = async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
            b = async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)
            acc = cdna4.mfma(a, b, acc)

    else:
        async_copy.buffer_load_to_shared(smemA.index(0), a_ptr, a_offs0, mask=a_mask)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemB.index(0), b_ptr, b_offs0)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemA.index(1), a_ptr, a_offs1, mask=a_mask)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemB.index(1), b_ptr, b_offs1)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemA.index(2), a_ptr, a_offs2, mask=a_mask)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smemB.index(2), b_ptr, b_offs2)
        async_copy.commit_group()
        a_offs0 = (a_offs0 + a_kstep).to(gl.int32)
        a_offs1 = (a_offs1 + a_kstep).to(gl.int32)
        a_offs2 = (a_offs2 + a_kstep).to(gl.int32)
        b_offs0 = (b_offs0 + b_kstep).to(gl.int32)
        b_offs1 = (b_offs1 + b_kstep).to(gl.int32)
        b_offs2 = (b_offs2 + b_kstep).to(gl.int32)

        async_copy.wait_group(4)
        a = async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
        b = async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)

        main_loop_triples: gl.constexpr = (k_tiles - NUM_BUFFERS) // NUM_BUFFERS
        for _ in tl.range(0, main_loop_triples):
            async_copy.wait_group(2)
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc = cdna4.mfma(a, b, acc)
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a = async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
                b = async_copy.load_shared_relaxed(smemB.index(1), dotOpLayoutB)
                async_copy.buffer_load_to_shared(
                    smemA.index(0), a_ptr, a_offs0, mask=a_mask
                )
                async_copy.commit_group()
                async_copy.buffer_load_to_shared(smemB.index(0), b_ptr, b_offs0)
                async_copy.commit_group()

            async_copy.wait_group(2)
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc = cdna4.mfma(a, b, acc)
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a = async_copy.load_shared_relaxed(smemA.index(2), dotOpLayoutA)
                b = async_copy.load_shared_relaxed(smemB.index(2), dotOpLayoutB)
                async_copy.buffer_load_to_shared(
                    smemA.index(1), a_ptr, a_offs1, mask=a_mask
                )
                async_copy.commit_group()
                async_copy.buffer_load_to_shared(smemB.index(1), b_ptr, b_offs1)
                async_copy.commit_group()

            async_copy.wait_group(2)
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc = cdna4.mfma(a, b, acc)
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a = async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
                b = async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)
                async_copy.buffer_load_to_shared(
                    smemA.index(2), a_ptr, a_offs2, mask=a_mask
                )
                async_copy.commit_group()
                async_copy.buffer_load_to_shared(smemB.index(2), b_ptr, b_offs2)
                async_copy.commit_group()
                a_offs0 = (a_offs0 + a_kstep).to(gl.int32)
                a_offs1 = (a_offs1 + a_kstep).to(gl.int32)
                a_offs2 = (a_offs2 + a_kstep).to(gl.int32)
                b_offs0 = (b_offs0 + b_kstep).to(gl.int32)
                b_offs1 = (b_offs1 + b_kstep).to(gl.int32)
                b_offs2 = (b_offs2 + b_kstep).to(gl.int32)

        tiles_processed: gl.constexpr = main_loop_triples * NUM_BUFFERS
        tiles_remaining: gl.constexpr = k_tiles - tiles_processed

        acc = cdna4.mfma(a, b, acc)
        async_copy.wait_group(0)

        a = async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
        b = async_copy.load_shared_relaxed(smemB.index(1), dotOpLayoutB)
        acc = cdna4.mfma(a, b, acc)

        a = async_copy.load_shared_relaxed(smemA.index(2), dotOpLayoutA)
        b = async_copy.load_shared_relaxed(smemB.index(2), dotOpLayoutB)
        acc = cdna4.mfma(a, b, acc)

        if tiles_remaining > 3:
            async_copy.buffer_load_to_shared(
                smemA.index(0), a_ptr, a_offs0, mask=a_mask
            )
            async_copy.commit_group()
            async_copy.buffer_load_to_shared(smemB.index(0), b_ptr, b_offs0)
            async_copy.commit_group()
            async_copy.wait_group(0)
            a = async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
            b = async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)
            acc = cdna4.mfma(a, b, acc)

        if tiles_remaining > 4:
            async_copy.buffer_load_to_shared(
                smemA.index(1), a_ptr, a_offs1, mask=a_mask
            )
            async_copy.commit_group()
            async_copy.buffer_load_to_shared(smemB.index(1), b_ptr, b_offs1)
            async_copy.commit_group()
            async_copy.wait_group(0)
            a = async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
            b = async_copy.load_shared_relaxed(smemB.index(1), dotOpLayoutB)
            acc = cdna4.mfma(a, b, acc)

    acc_store = gl.convert_layout(acc, layout=mfmaLayout)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfmaLayout))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfmaLayout))
    cm = pid_m * BLOCK_M + offs_cm
    cn = pid_n * BLOCK_N + offs_cn
    c_offsets = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (cm[:, None] < M) & (cn[None, :] < N)
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    cdna4.buffer_store(
        ptr=c_base,
        offsets=c_offsets,
        stored_value=acc_store.to(c_ptr.dtype.element_ty),
        mask=c_mask,
    )


def _allocate_partial_scratch(
    device: torch.device,
    total_tiles: int,
    split_k: int,
    block_n: int,
    block_m: int = DENSE16_BLOCK_M,
) -> torch.Tensor:
    return torch.empty(
        (total_tiles * split_k * block_m * block_n,),
        device=device,
        dtype=torch.float32,
    )


def _get_partial_scratch(
    device: torch.device,
    total_tiles: int,
    split_k: int,
    block_n: int,
    block_m: int = DENSE16_BLOCK_M,
) -> torch.Tensor:
    """Return stream-local split-K partial sums.

    Same-stream calls are ordered, so they can safely reuse scratch. Different
    streams must not share the buffer because the producer and reducer kernels run
    asynchronously and can otherwise overwrite another invocation's partial sums.
    During CUDA/HIP graph capture, allocate from PyTorch's graph-aware allocator
    without storing graph-private memory in the eager cache.
    """
    if torch.cuda.is_current_stream_capturing():
        return _allocate_partial_scratch(device, total_tiles, split_k, block_n, block_m)

    device_index = torch.cuda.current_device() if device.index is None else device.index
    stream_id = torch.cuda.current_stream(device_index).cuda_stream
    key = (device_index, stream_id, total_tiles, split_k, block_m * block_n)
    cached = _partial_cache.get(key)
    if cached is not None:
        return cached

    partial = _allocate_partial_scratch(device, total_tiles, split_k, block_n, block_m)
    _partial_cache[key] = partial
    return partial


def _check_supported_dense16_shape(M: int, N: int, K: int) -> None:
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(
            "dense16 Gluon GEMM requires positive M/N/K, got " f"M={M}, N={N}, K={K}"
        )
    if M > MAX_M:
        raise ValueError(f"dense16 Gluon GEMM requires M <= {MAX_M}; got M={M}")
    if N % DENSE16_BLOCK_N != 0:
        raise ValueError(
            f"dense16 Gluon GEMM requires N to be a multiple of {DENSE16_BLOCK_N}; "
            f"got N={N}"
        )
    if K % DENSE16_BLOCK_K != 0:
        raise ValueError(
            f"dense16 Gluon GEMM requires K to be a multiple of {DENSE16_BLOCK_K}; "
            f"got K={K}"
        )


def _use_warp_reduce_smallm(M: int, _N: int, K: int) -> bool:
    return (M in (1, 2) and K <= 2048) or (M == 4 and K <= 1024)


def _choose_mfma_lds_split_k(K: int) -> int:
    k_tiles = K // MFMA_LDS_BLOCK_K
    for split_k in (MFMA_LDS_SPLIT_K, 7, 6, 5, 4, 3, 2):
        if k_tiles % split_k == 0 and k_tiles // split_k >= 2:
            return split_k
    return 0


def _supports_mfma_lds_smallm(M: int, N: int, K: int) -> bool:
    return (
        M in (1, 2, MFMA_LDS_REDUCE_M)
        and K >= MFMA_LDS_BLOCK_K * 2 * MFMA_LDS_SPLIT_K
        and N % MFMA_LDS_BLOCK_N == 0
        and K % MFMA_LDS_BLOCK_K == 0
        and _choose_mfma_lds_split_k(K) > 1
    )


def _use_mfma_lds_smallm(_M: int, _N: int, _K: int) -> bool:
    # Keep the split-K kernels available for direct experiments, but do not
    # route them by default until they are consistently faster than torch.mm.
    return False


def _choose_mfma_lds_mediumm_config(
    M: int, N: int, K: int
) -> tuple[int, int, int, int, int, int] | None:
    if M < 8 or K % DENSE16_BLOCK_K != 0 or N % DENSE16_BLOCK_N != 0:
        return None
    if M <= 32:
        block_m = 16 if M <= 16 else 32
        if K == DENSE16_BLOCK_K:
            return block_m, 32, DENSE16_BLOCK_K, 2, 2, 1
        if K < 1024:
            if K % 256 == 0 and K // 256 >= 2:
                num_buffers = 2 if K // 256 == 2 else 3
                return block_m, 32, 256, 2, 2, num_buffers
            if K % 128 == 0 and K // 128 >= 2:
                num_buffers = 2 if K // 128 == 2 else 3
                return block_m, 32, 128, 2, 2, num_buffers
            num_buffers = 2 if K // DENSE16_BLOCK_K == 2 else 3
            return block_m, 32, DENSE16_BLOCK_K, 2, 2, num_buffers
        if block_m == 16:
            if K == 1024:
                return block_m, 32, 512, 2, 2, 2
            return None
        if K == 2048:
            return 16, 16, 512, 2, 2, 2
        return None
    if M <= 128:
        if K == 1024:
            return 32, 32, 512, 2, 2, 2
        if K == 2048:
            if M <= 64:
                if N == 1280:
                    return 32, 32, 512, 2, 2, 2
                if N == 2560:
                    return 32, 32, 128, 2, 2, 3
                return None
            return 32, 32, DENSE16_BLOCK_K, 2, 2, 3
        if 2048 < K < 4096 and K % 128 != 0:
            return 32, 32, DENSE16_BLOCK_K, 2, 2, 3
        if M == 128 and N == 4096 and K == 4096:
            return 16, 128, DENSE16_BLOCK_K, 1, 4, 3
    return None


def _use_mfma_lds_mediumm(M: int, N: int, K: int) -> bool:
    return _choose_mfma_lds_mediumm_config(M, N, K) is not None


def _use_mfma_lds_largem(M: int, N: int, K: int) -> bool:
    return M >= LARGEM_DISPATCH_MIN_M and _supports_largem_shape(M, N, K)


def gluon_mm_a16w16_warp_reduce_smallm_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute small-M dense ``A @ B.T`` with one warp per output.

    Intended for very small decode tiles: ``M`` in ``{1, 2}`` with ``K <= 2048``
    or ``M == 4`` with ``K <= 1024``.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"small-M dense16 warp-reduce GEMM expects 2D inputs, got {A.ndim=} {B.ndim=}"
        )
    if A.dtype not in _SUPPORTED_DTYPES or B.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "small-M dense16 warp-reduce GEMM expects fp16/bf16 inputs, got "
            f"{A.dtype=} {B.dtype=}"
        )
    if A.dtype != B.dtype:
        raise TypeError(
            "small-M dense16 warp-reduce GEMM expects matching input dtypes, got "
            f"{A.dtype=} {B.dtype=}"
        )
    if out_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"small-M dense16 warp-reduce GEMM expects fp16/bf16 output, got {out_dtype=}"
        )
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("small-M dense16 warp-reduce GEMM requires CUDA/HIP tensors")

    M, K = A.shape
    N, K_b = B.shape
    if M > 4:
        raise ValueError(f"small-M warp-reduce GEMM requires M <= 4; got M={M}")
    if K_b != K:
        raise ValueError(
            "small-M dense16 warp-reduce GEMM K mismatch: "
            f"A={tuple(A.shape)} B={tuple(B.shape)}"
        )
    _check_supported_dense16_shape(M, N, K)

    C = torch.empty((M, N), device=A.device, dtype=out_dtype)
    total_outputs = M * N
    grid = (triton.cdiv(total_outputs, WARP_REDUCE_OUTPUTS),)
    _warp_reduce_smallm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
        BLOCK_OUT=WARP_REDUCE_OUTPUTS,
        BLOCK_K=WARP_REDUCE_BLOCK_K,
        num_warps=WARP_REDUCE_OUTPUTS,
    )

    if alpha is not None:
        C = C * alpha.to(dtype=C.dtype)
    return C


def gluon_mm_a16w16_mfma_lds_smallm_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute high-K small-M dense ``A @ B.T`` with MFMA and partial-sum reduction.

    Intended for ``M`` in ``{1, 2, 4}``, large ``K``, and ``N`` aligned to the
    split-K MFMA/LDS tile.

    This decode path maps the wide N dimension onto the MFMA rows, streams the
    ``[N, K]`` weight tile through double-buffered LDS async copies, splits K
    into eight partial tiles, and sums those partial results in a second Gluon kernel.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            "small-M dense16 MFMA LDS GEMM expects 2D inputs, got "
            f"{A.ndim=} {B.ndim=}"
        )
    if A.dtype not in _SUPPORTED_DTYPES or B.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "small-M dense16 MFMA LDS GEMM expects fp16/bf16 inputs, got "
            f"{A.dtype=} {B.dtype=}"
        )
    if A.dtype != B.dtype:
        raise TypeError(
            "small-M dense16 MFMA LDS GEMM expects matching input dtypes, got "
            f"{A.dtype=} {B.dtype=}"
        )
    if out_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"small-M dense16 MFMA LDS GEMM expects fp16/bf16 output, got {out_dtype=}"
        )
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("small-M dense16 MFMA LDS GEMM requires CUDA/HIP tensors")

    M, K = A.shape
    N, K_b = B.shape
    if K_b != K:
        raise ValueError(
            "small-M dense16 MFMA LDS GEMM K mismatch: "
            f"A={tuple(A.shape)} B={tuple(B.shape)}"
        )
    _check_supported_dense16_shape(M, N, K)
    if not _supports_mfma_lds_smallm(M, N, K):
        raise ValueError(
            "small-M dense16 MFMA LDS GEMM requires M in {1, 2, 4}, "
            f"N divisible by {MFMA_LDS_BLOCK_N}, K >= "
            f"{MFMA_LDS_BLOCK_K * 2 * MFMA_LDS_SPLIT_K}, K divisible by "
            f"{MFMA_LDS_BLOCK_K}, and K evenly splittable by 2..8 CTAs; "
            f"got M={M}, N={N}, K={K}"
        )

    C_full = torch.empty((MFMA_LDS_REDUCE_M, N), device=A.device, dtype=out_dtype)
    C = C_full[:M, :]
    split_k = _choose_mfma_lds_split_k(K)
    num_n_tiles = triton.cdiv(N, MFMA_LDS_BLOCK_N)
    total_work = num_n_tiles * split_k
    sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = min(total_work, sms)
    partial = _get_partial_scratch(
        A.device, num_n_tiles, split_k, MFMA_LDS_BLOCK_N, MFMA_LDS_REDUCE_M
    )
    k_tiles_per_split = (K // MFMA_LDS_BLOCK_K) // split_k

    _mfma_lds_smallm_splitk_kernel[(grid,)](
        A,
        B,
        C,
        partial,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
        BLOCK_M=DENSE16_BLOCK_M,
        BLOCK_N=MFMA_LDS_BLOCK_N,
        BLOCK_K=MFMA_LDS_BLOCK_K,
        SPLIT_K=split_k,
        PARTIAL_M=MFMA_LDS_REDUCE_M,
        K_TILES_PER_SPLIT=k_tiles_per_split,
        NUM_PROGRAMS=grid,
        num_warps=MFMA_LDS_NUM_WARPS,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    _mfma_lds_smallm_reduce_kernel[(num_n_tiles,)](
        partial,
        C,
        C.stride(0),
        C.stride(1),
        BLOCK_N=MFMA_LDS_BLOCK_N,
        REDUCE_M=MFMA_LDS_REDUCE_M,
        SPLIT_K=split_k,
        num_warps=MFMA_LDS_NUM_WARPS,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )

    if alpha is not None:
        C = C * alpha.to(dtype=C.dtype)
    return C


def gluon_mm_a16w16_mfma_lds_mediumm_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute tuned medium-M dense ``A @ B.T`` with MFMA/LDS tiling.

    Intended for selected ``8 <= M <= 128`` dense16 decode/prefill tiles where
    the Gluon route is competitive with torch.mm.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"medium-M dense16 MFMA LDS GEMM expects 2D inputs, got {A.ndim=} {B.ndim=}"
        )
    if A.dtype not in _SUPPORTED_DTYPES or B.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "medium-M dense16 MFMA LDS GEMM expects fp16/bf16 inputs, got "
            f"{A.dtype=} {B.dtype=}"
        )
    if A.dtype != B.dtype:
        raise TypeError(
            "medium-M dense16 MFMA LDS GEMM expects matching input dtypes, got "
            f"{A.dtype=} {B.dtype=}"
        )
    if out_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"medium-M dense16 MFMA LDS GEMM expects fp16/bf16 output, got {out_dtype=}"
        )
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("medium-M dense16 MFMA LDS GEMM requires CUDA/HIP tensors")

    M, K = A.shape
    N, K_b = B.shape
    if K_b != K:
        raise ValueError(
            "medium-M dense16 MFMA LDS GEMM K mismatch: "
            f"A={tuple(A.shape)} B={tuple(B.shape)}"
        )
    _check_supported_dense16_shape(M, N, K)
    config = _choose_mfma_lds_mediumm_config(M, N, K)
    if config is None:
        raise ValueError(
            "medium-M dense16 MFMA LDS GEMM requires 8 <= M <= 128, "
            f"N divisible by {DENSE16_BLOCK_N}, and K divisible by {DENSE16_BLOCK_K}; "
            f"got M={M}, N={N}, K={K}"
        )

    block_m, block_n, block_k, warps_m, warps_n, num_buffers = config
    C = torch.empty((M, N), device=A.device, dtype=out_dtype)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _mfma_lds_mediumm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        NUM_BUFFERS=num_buffers,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=warps_m * warps_n,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )

    if alpha is not None:
        C = C * alpha.to(dtype=C.dtype)
    return C


def gluon_mm_a16w16_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute dense16 GEMM ``A @ B.T`` on gfx950 when supported.

    Dispatches among warp-reduce small-M, tuned medium-M, and large-M paths.
    Returns ``None`` for unsupported or disabled dense16 shapes so the generic
    GEMM caller can use its torch fallback.

    Args:
        A: Activation matrix with shape ``[M, K]``.
        B: Weight matrix with shape ``[N, K]``.
        out_dtype: Output dtype, either fp16 or bf16.
        alpha: Optional post-GEMM scale.

    Returns:
        Tensor with shape ``[M, N]`` when a Gluon path is selected; otherwise
        ``None``.
    """
    if A.ndim != 2 or B.ndim != 2:
        return None
    if A.dtype not in _SUPPORTED_DTYPES or B.dtype not in _SUPPORTED_DTYPES:
        return None
    if A.dtype != B.dtype:
        return None
    if out_dtype not in _SUPPORTED_DTYPES:
        return None
    if not A.is_cuda or not B.is_cuda:
        return None

    M, K = A.shape
    N, K_b = B.shape
    if K_b != K:
        return None
    if M <= 0 or N <= 0 or K <= 0:
        return None
    if N % DENSE16_BLOCK_N != 0 or K % DENSE16_BLOCK_K != 0:
        return None

    if _use_mfma_lds_largem(M, N, K):
        return gluon_mm_a16w16_largem_gfx950(
            A,
            B,
            out_dtype,
            alpha=alpha,
        )
    if M > MAX_M:
        return None

    if _use_warp_reduce_smallm(M, N, K):
        return gluon_mm_a16w16_warp_reduce_smallm_gfx950(
            A,
            B,
            out_dtype,
            alpha=alpha,
        )
    if _use_mfma_lds_smallm(M, N, K):
        return gluon_mm_a16w16_mfma_lds_smallm_gfx950(
            A,
            B,
            out_dtype,
            alpha=alpha,
        )
    if _use_mfma_lds_mediumm(M, N, K):
        return gluon_mm_a16w16_mfma_lds_mediumm_gfx950(
            A,
            B,
            out_dtype,
            alpha=alpha,
        )
    return None
