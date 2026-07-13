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

"""MLA decode Gluon kernels for AMD GFX950 (bf16 Q + bf16 KV).

Two regimes share one kernel, dispatched by ``num_q_heads``:

* ``bh16bn64`` -- BLOCK_H=16, BLOCK_N=64, ``num_q_heads <= 16``, 2-D
  (batch, split) grid.
* ``bh64`` -- BLOCK_H=64, BLOCK_N=64, ``num_q_heads in {64, 128}``, 3-D
  XCD-aware grid, ``batch_size`` divisible by 64.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.attention.gluon.utils import _INV_LN2


@gluon.jit
def _mla_decode_gluon(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,  # noqa: E741
    sm_scale,
    kv_scale,
    stride_q_nope_bs,
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    Mid_lse,  # split>1: per-split fp32 lse [B, H, NUM_KV_SPLITS] (else None)
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    Final_lse,  # RETURN_LSE only: merged fp32 lse [B, H] (else None)
    stride_final_lse_b,
    stride_final_lse_h,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    HEAD_DIM_KPE: gl.constexpr,
    KV_PE_OFFSET: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NHEAD: gl.constexpr,
    REGIME: gl.constexpr,
    RETURN_LSE: gl.constexpr,
):
    # Grid mapping: bh64 uses a 3-D XCD-aware multi-batch grid
    # (NUM_XCDS, head_block, (batch // NUM_XCDS) * NUM_KV_SPLITS); bh16bn64 uses
    # a 2-D (batch, split) grid (for batch_size=1 this is (1, NUM_KV_SPLITS)).
    if REGIME == "bh64":
        cur_batch = gl.program_id(0) + (gl.program_id(2) // NUM_KV_SPLITS) * NUM_XCDS
        cur_head_id = gl.program_id(1)
        split_kv_id = gl.program_id(2) % NUM_KV_SPLITS
    else:
        cur_batch = gl.program_id(0)
        cur_head_id = 0
        split_kv_id = gl.program_id(1)

    # Paged 2-D view: Req_to_tokens = block_table[batch, max_pages],
    # B_seq_len = cache_seqlens[batch].
    batch_page_start = stride_req_to_tokens_bs * cur_batch
    cur_batch_seq_len = gl.load(B_seq_len + cur_batch)

    # Q-side layouts + mfma_layout: switch by BLOCK_H.
    # bh64 has BLOCK_H=64 (warps tile M); bh16bn64 has BLOCK_H=16 (warps tile K).
    if BLOCK_H == 64:
        # bh64: Q is [64, 512] / [64, 64]; warps tile M.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
            ],
            cga_layout=[],
            shape=[64, 512],
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
            lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
            warp_bases=((1, 0), (2, 0)),
            block_bases=[],
            shape=[64, 64],
        )
        shared_q_pe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [4, 0],
                [8, 0],
                [16, 0],
                [1, 0],
                [2, 0],
                [32, 0],
            ],
            cga_layout=[],
            shape=[64, 64],
        )
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[4, 1],
        )
    else:
        # bh16bn64: Q is [16, 512] / [16, 64]; warps tile K.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ],
            cga_layout=[],
            shape=[16, 512],
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4)),
            lane_bases=((0, 8), (0, 16), (0, 32), (1, 0), (2, 0), (4, 0)),
            warp_bases=((8, 0), (0, 0)),
            block_bases=[],
            shape=[16, 64],
        )
        shared_q_pe: gl.constexpr = gl.SwizzledSharedLayout(
            vec=8, per_phase=2, max_phase=8, order=[1, 0]
        )
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[1, 4],
        )

    # KV-side layouts (BLOCK_N=64, bf16 KV): K is [512, 64], KPE is [64, 64].
    # Shared by both regimes.
    blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16), (0, 32)),
        lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[512, 64],
    )
    shared_kv: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
            [128, 0],
            [256, 0],
            [0, 1],
            [0, 2],
            [0, 8],
            [0, 4],
            [0, 16],
            [0, 32],
        ],
        cga_layout=[],
        shape=[512, 64],
    )
    blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 32)),
        lane_bases=((8, 0), (16, 0), (32, 0), (0, 4), (0, 8), (0, 16)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[64, 64],
    )
    shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 1],
            [0, 2],
            [0, 32],
        ],
        cga_layout=[],
        shape=[64, 64],
    )
    blocked_page: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,),),
        lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
        warp_bases=((0,), (0,)),
        block_bases=[],
        shape=[64],
    )
    blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
        lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[512, 32],
    )

    # V is the latent slice of K, read back transposed for the PV dot.
    # bh64 tiles M across warps (degenerate warp_bases + extra reg bases);
    # bh16bn64 tiles the 64-wide K across warps.
    if REGIME == "bh64":
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=(
                (0, 1),
                (0, 2),
                (0, 4),
                (0, 32),
                (16, 0),
                (32, 0),
                (64, 0),
                (128, 0),
                (256, 0),
            ),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((0, 0), (0, 0)),
            block_bases=[],
            shape=[512, 64],
        )
    else:
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((16, 0), (32, 0)),
            block_bases=[],
            shape=[512, 64],
        )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    dtype = Q_nope.type.element_ty
    kvtype = Kv_c_cache.type.element_ty

    buf_q_nope = gl.allocate_shared_memory(
        dtype, shape=[BLOCK_H, HEAD_DIM_CKV], layout=shared_q_nope
    )
    buf_q_pe = gl.allocate_shared_memory(
        dtype, shape=[BLOCK_H, HEAD_DIM_KPE], layout=shared_q_pe
    )

    # load q_nope
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q_nope))
    cur_head = cur_head_id * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope)
    )
    offs_q_nope = (
        cur_batch * stride_q_nope_bs
        + cur_head[:, None] * stride_q_nope_h
        + offs_d_ckv[None, :]
    )
    # For nhead < BLOCK_H, mask OOB heads to zero on Q load and skip OOB O stores; wasted MFMA lanes are free (memory-bound).
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_q_nope,
        Q_nope,
        offs_q_nope,
        mask=(cur_head < NHEAD)[:, None] if NHEAD < BLOCK_H else None,
    )
    gl.amd.cdna4.async_copy.commit_group()

    # load q_pe
    offs_d_kpe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(0, blocked_q_pe))
    cur_head_qpe = cur_head_id * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_pe)
    )
    offs_q_pe = (
        cur_batch * stride_q_pe_bs
        + cur_head_qpe[:, None] * stride_q_pe_h
        + offs_d_kpe[None, :]
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_q_pe,
        Q_pe,
        offs_q_pe,
        mask=(cur_head_qpe < NHEAD)[:, None] if NHEAD < BLOCK_H else None,
    )
    gl.amd.cdna4.async_copy.commit_group()

    e_max = gl.zeros(
        [BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)
    ) - float("inf")
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)

    num_pages = gl.cdiv(cur_batch_seq_len, PAGE_SIZE)
    pages_per_split = gl.cdiv(num_pages, NUM_KV_SPLITS)
    split_start_page = split_kv_id * pages_per_split
    split_end_page = gl.minimum(split_start_page + pages_per_split, num_pages)
    split_kv_start = split_start_page * PAGE_SIZE
    split_kv_end = gl.minimum(split_end_page * PAGE_SIZE, cur_batch_seq_len)
    # Clamp so empty (trailing) splits have start == end -> num_iter == 0.
    split_kv_end = gl.maximum(split_kv_end, split_kv_start)
    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    # Fold KV dequant scale into the QK temperature.
    # bf16 KV: the wrapper passes kv_scale=1.0, so this is a no-op.
    qk_scale = sm_scale * kv_scale

    # bufs of page_number
    shared_page: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    bufs_page = gl.allocate_shared_memory(
        gl.int32, shape=[2, BLOCK_N], layout=shared_page
    )

    offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)

    # prologue
    # global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_page.index(0), Req_to_tokens, offs_page, offs_n_page < split_kv_end
    )
    gl.amd.cdna4.async_copy.commit_group()

    start_n += BLOCK_N
    # global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_page.index(1), Req_to_tokens, offs_page, offs_n_page < split_kv_end
    )
    gl.amd.cdna4.async_copy.commit_group()

    # local load Q
    gl.amd.cdna4.async_copy.wait_group(2)
    q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_nope, mfma_layout_a)
    q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_pe, mfma_layout_a)

    # move here to work around allocate_shared_memory bug
    bufs_kv = gl.allocate_shared_memory(
        kvtype, shape=[2, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv
    )
    bufs_kpe = gl.allocate_shared_memory(
        kvtype, shape=[2, HEAD_DIM_KPE, BLOCK_N], layout=shared_kpe
    )

    # global load K
    # local load page number
    gl.amd.cdna4.async_copy.wait_group(1)
    kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page.index(0), gl.SliceLayout(0, blocked_kpe)
    )
    # paged KV: physical row = page * PAGE_SIZE + (token % PAGE_SIZE)
    offs_n_pe0 = split_kv_start + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe)
    )
    kv_loc_pe = kv_page_number_pe * PAGE_SIZE + (offs_n_pe0 % PAGE_SIZE)

    # local load page number for slice 0
    bufs_page_0 = bufs_page.index(0).slice(0, BLOCK_N // 2, 0)
    kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page_0, gl.SliceLayout(0, blocked_kv_slice)
    )
    offs_n_nope0 = split_kv_start + gl.arange(
        0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice)
    )
    kv_loc0 = kv_page_number_0 * PAGE_SIZE + (offs_n_nope0 % PAGE_SIZE)

    # global load K_nope slice 0
    offs_d_ckv_10 = gl.arange(
        0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice)
    )
    offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    bufs_kv0 = bufs_kv.index(0).slice(0, BLOCK_N // 2, 1)
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0)
    gl.amd.cdna4.async_copy.commit_group()

    # global load K_pe
    offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
    offs_k_pe = (
        kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
    )
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_kpe.index(0),
            K_pe_cache,
            offs_k_pe,
            mask=offs_n_pe0[None, :] < split_kv_end,
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(
            bufs_kpe.index(0), K_pe_cache + offs_k_pe
        )
    gl.amd.cdna4.async_copy.commit_group()

    # local load page number for slice 1
    bufs_page_1 = bufs_page.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
    kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page_1, gl.SliceLayout(0, blocked_kv_slice)
    )
    offs_n_nope1 = offs_n_nope0 + BLOCK_N // 2
    kv_loc1 = kv_page_number_1 * PAGE_SIZE + (offs_n_nope1 % PAGE_SIZE)

    # global load K_nope slice 1
    bufs_kv1 = bufs_kv.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
    offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n_nope1[None, :] < split_kv_end
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1)
    gl.amd.cdna4.async_copy.commit_group()

    buf_idx = 0
    # main loop
    for i in range(num_iter - 2):
        async_idx = (buf_idx + 1) % 2

        gl.amd.cdna4.async_copy.wait_group(0)
        # global load page number
        offs_n_page = start_n + BLOCK_N + offs_page_raw
        offs_page = batch_page_start + offs_n_page // PAGE_SIZE
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_page.index(buf_idx),
            Req_to_tokens,
            offs_page,
            offs_n_page < split_kv_end,
        )
        gl.amd.cdna4.async_copy.commit_group()

        # global load K
        bufs_kv0 = bufs_kv.index(async_idx).slice(0, BLOCK_N // 2, 1)
        bufs_kv1 = bufs_kv.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
        # local load page number for slice 0
        bufs_page_0 = bufs_page.index(async_idx).slice(0, BLOCK_N // 2, 0)
        kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page_0, gl.SliceLayout(0, blocked_kv_slice)
        )
        offs_n_nope0 = start_n + gl.arange(
            0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice)
        )
        kv_loc0 = kv_page_number_0 * PAGE_SIZE + (offs_n_nope0 % PAGE_SIZE)
        # global load K_nope slice 0
        offs_d_ckv_10 = gl.arange(
            0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice)
        )
        offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_kv0,
                Kv_c_cache,
                offs_k_c0,
                mask=offs_n_nope0[None, :] < split_kv_end,
            )
        else:
            # No mask needed on global_load path in the loop body: all
            # iterations are guaranteed in-bounds by num_iter arithmetic.
            # Only the epilogue uses mask + other=0 for the last
            # potentially-partial block.
            gl.amd.cdna4.async_copy.global_load_to_shared(
                bufs_kv0, Kv_c_cache + offs_k_c0
            )
        gl.amd.cdna4.async_copy.commit_group()

        # local load page_number_pe
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe)
        )
        offs_n_pe = start_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe)
        )
        kv_loc_pe = kv_page_number_pe * PAGE_SIZE + (offs_n_pe % PAGE_SIZE)
        # global load K_pe
        offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
        offs_k_pe = (
            kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
        )
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_kpe.index(async_idx),
                K_pe_cache,
                offs_k_pe,
                mask=offs_n_pe[None, :] < split_kv_end,
            )
        else:
            # No mask needed: loop iterations are in-bounds (see KV slice 0 comment).
            gl.amd.cdna4.async_copy.global_load_to_shared(
                bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe
            )
        gl.amd.cdna4.async_copy.commit_group()

        # dot, softmax, dot (part0)
        k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_kv.index(buf_idx), mfma_layout_b
        )
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)
        k_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_kpe.index(buf_idx), mfma_layout_b
        )
        qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)

        # local load page number for slice 1
        bufs_page_1 = bufs_page.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
        kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page_1, gl.SliceLayout(0, blocked_kv_slice)
        )
        offs_n1 = offs_n_nope0 + BLOCK_N // 2
        kv_loc1 = kv_page_number_1 * PAGE_SIZE + (offs_n1 % PAGE_SIZE)
        # global load K_nope slice 1
        offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n1[None, :] < split_kv_end
            )
        else:
            # No mask needed: loop iterations are in-bounds (see KV slice 0 comment).
            gl.amd.cdna4.async_copy.global_load_to_shared(
                bufs_kv1, Kv_c_cache + offs_k_c1
            )
        gl.amd.cdna4.async_copy.commit_group()

        # dot, softmax, dot (part1)
        qk *= qk_scale
        offs_n_qk = (
            split_kv_start
            + i * BLOCK_N
            + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        )
        qk = gl.where(offs_n_qk[None, :] < split_kv_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp2((e_max - n_e_max) * _INV_LN2)
        p = gl.exp2((qk - n_e_max[:, None]) * _INV_LN2)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_kv.index(buf_idx), linear_v
        )
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    # epilogue 1
    # Runtime guard: a split can cover fewer than 2 KV blocks (short sequences).
    if num_iter >= 2:
        async_idx = (buf_idx + 1) % 2

        # global load K
        # local load page number
        gl.amd.cdna4.async_copy.wait_group(3)
        kv_page_number = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kv)
        )
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe)
        )
        offs_n_nope = start_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv)
        )
        offs_n_pe = start_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe)
        )
        kv_loc = kv_page_number * PAGE_SIZE + (offs_n_nope % PAGE_SIZE)
        kv_loc_pe = kv_page_number_pe * PAGE_SIZE + (offs_n_pe % PAGE_SIZE)
        # global load K_nope
        offs_d_ckv_1 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
        offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv_1[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_kv.index(async_idx),
                Kv_c_cache,
                offs_k_c,
                mask=offs_n_nope[None, :] < split_kv_end,
            )
        else:
            # No mask needed: out-of-range positions are discarded by the qk score mask
            gl.amd.cdna4.async_copy.global_load_to_shared(
                bufs_kv.index(async_idx), Kv_c_cache + offs_k_c
            )
        gl.amd.cdna4.async_copy.commit_group()
        # global load K_pe
        offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
        offs_k_pe = (
            kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
        )
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_kpe.index(async_idx),
                K_pe_cache,
                offs_k_pe,
                mask=offs_n_pe[None, :] < split_kv_end,
            )
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe
            )
        gl.amd.cdna4.async_copy.commit_group()

        # dot, softmax, dot
        gl.amd.cdna4.async_copy.wait_group(2)
        k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

        k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
        qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
        qk *= qk_scale
        offs_n_qk = (
            split_kv_start
            + (num_iter - 2) * BLOCK_N
            + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        )
        qk = gl.where(offs_n_qk[None, :] < split_kv_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp2((e_max - n_e_max) * _INV_LN2)
        p = gl.exp2((qk - n_e_max[:, None]) * _INV_LN2)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    # epilogue 2
    # dot, softmax, dot
    gl.amd.cdna4.async_copy.wait_group(0)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

    k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
    qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
    qk *= qk_scale
    offs_n_qk = (
        split_kv_start
        + (num_iter - 1) * BLOCK_N
        + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    )
    qk = gl.where(offs_n_qk[None, :] < split_kv_end, qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
    re_scale = gl.exp2((e_max - n_e_max) * _INV_LN2)
    p = gl.exp2((qk - n_e_max[:, None]) * _INV_LN2)
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    e_max = n_e_max
    p = p.to(dtype)
    p = gl.convert_layout(p, mfma_layout_a)
    acc *= re_scale[:, None]
    v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
    v_c = v_c.to(dtype)
    v_c = gl.permute(v_c, [1, 0])
    v_c = gl.convert_layout(v_c, mfma_layout_b)
    acc = gl.amd.cdna4.mfma(p, v_c, acc)

    cur_head_o = cur_head_id * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_d_ckv_o = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, mfma_layout))
    offs_o = (
        cur_batch * stride_o_b
        + cur_head_o[:, None] * stride_o_h
        + split_kv_id * stride_o_s
        + offs_d_ckv_o[None, :]
    )
    acc *= kv_scale
    rcp = 1.0 / e_sum
    stored_value = (acc * rcp[:, None]).to(dtype)
    if NHEAD < BLOCK_H:
        gl.amd.cdna4.buffer_store(
            stored_value, ptr=O, offsets=offs_o, mask=(cur_head_o < NHEAD)[:, None]
        )
    else:
        gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o)

    # store lse
    blocked_lse: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0]
    )
    cur_head_lse = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=blocked_lse)
    if RETURN_LSE and NUM_KV_SPLITS == 1:
        # split==1: single split is the whole sequence, so its lse is the final lse.
        offs_final_lse = (
            cur_batch * stride_final_lse_b + cur_head_lse * stride_final_lse_h
        )
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD < BLOCK_H:
            gl.amd.cdna4.buffer_store(
                lse, ptr=Final_lse, offsets=offs_final_lse, mask=(cur_head_lse < NHEAD)
            )
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse)
    elif NUM_KV_SPLITS > 1:
        # per-split lse for stage-2 reduce.
        offs_mid_lse = (
            cur_batch * stride_mid_lse_b
            + cur_head_lse * stride_mid_lse_h
            + split_kv_id * stride_mid_lse_s
        )
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD < BLOCK_H:
            gl.amd.cdna4.buffer_store(
                lse, ptr=Mid_lse, offsets=offs_mid_lse, mask=(cur_head_lse < NHEAD)
            )
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse)


@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    Mid_lse,
    O,  # noqa: E741
    Final_lse,
    B_seq_len,
    stride_l_b,
    stride_l_h,
    stride_l_s,
    stride_ml_b,
    stride_ml_h,
    stride_ml_s,
    stride_o_b,
    stride_o_h,
    stride_fl_b,
    stride_fl_h,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HAS_FINAL_LSE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_l = cur_batch * stride_l_b + cur_head * stride_l_h + offs_d_ckv
    offs_ml = cur_batch * stride_ml_b + cur_head * stride_ml_h

    cur_batch_seq_len = tl.load(B_seq_len + cur_batch)
    num_pages = tl.cdiv(cur_batch_seq_len, PAGE_SIZE)
    pages_per_split = tl.cdiv(num_pages, NUM_KV_SPLITS)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    for split_kv_id in range(0, NUM_KV_SPLITS):
        split_valid = split_kv_id * pages_per_split < num_pages
        logits = tl.load(
            Logits + offs_l + split_kv_id * stride_l_s,
            mask=split_valid,
            other=0.0,
        )
        logits_1 = tl.load(
            Mid_lse + offs_ml + split_kv_id * stride_ml_s,
            mask=split_valid,
            other=-float("inf"),
        )

        n_e_max = tl.maximum(logits_1, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(logits_1 - n_e_max)
        acc += exp_logic * logits

        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    tl.store(
        O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d_ckv,
        acc / e_sum,
    )
    if HAS_FINAL_LSE:
        tl.store(
            Final_lse + cur_batch * stride_fl_b + cur_head * stride_fl_h,
            e_max + tl.log(e_sum),
        )


_WAVE_WORKGROUPS = 256

_NUM_XCDS = 8


def _select_num_kv_splits_bh16bn64(
    *, batch: int, max_seqlen_k: int, block_n: int
) -> int:
    occupancy_cap = _WAVE_WORKGROUPS // batch
    blocks = (max_seqlen_k + block_n - 1) // block_n
    return max(1, min(occupancy_cap, blocks))


def _select_num_kv_splits_bh64(
    *, batch: int, nhead: int, num_xcds: int, block_h: int
) -> int:
    base_grid = num_xcds * triton.cdiv(nhead, block_h) * (batch // num_xcds)
    return max(1, triton.next_power_of_2(triton.cdiv(_WAVE_WORKGROUPS, base_grid)))


def gluon_mla_decode_bf16_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Absorbed MLA decode over a paged compressed KV cache (gfx950, bf16).

    ``q`` is ``[batch, 1, num_q_heads, kv_lora_rank + qk_rope_head_dim]`` and
    ``kv_cache`` is ``[num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]``
    (first ``kv_lora_rank`` latent, last ``qk_rope_head_dim`` RoPE). Output is
    ``[batch, 1, num_q_heads, kv_lora_rank]``. ``num_q_heads`` selects the
    regime: ``bh16bn64`` (``<= 16``) or ``bh64`` (``{64, 128}``, ``batch_size``
    divisible by 64).
    """
    if logit_cap != 0.0:
        raise NotImplementedError(
            "gluon_mla_decode_bf16_gfx950 does not support logit_cap"
        )
    if q.dim() != 4 or q.shape[1] != 1:
        raise ValueError(
            f"q must be [batch, 1, num_q_heads, R + rope], got {tuple(q.shape)}"
        )
    qk_dim = kv_lora_rank + qk_rope_head_dim
    if q.shape[-1] != qk_dim:
        raise ValueError(f"q head dim must be {qk_dim}, got {q.shape[-1]}")
    if kv_lora_rank != 512 or qk_rope_head_dim != 64:
        raise NotImplementedError(
            "gluon MLA decode requires kv_lora_rank=512, qk_rope_head_dim=64, "
            f"got {kv_lora_rank}/{qk_rope_head_dim}"
        )
    batch_size, _, nhead, _ = q.shape
    if nhead in (64, 128):
        regime = "bh64"
        block_h = 64
        num_xcds = _NUM_XCDS
        if batch_size % 64 != 0:
            raise NotImplementedError(
                "gluon MLA decode (bh64) is large-batch only and requires "
                f"batch_size divisible by 64, got {batch_size}"
            )
    elif 1 <= nhead <= 16:
        regime = "bh16bn64"
        block_h = 16
        num_xcds = 1  # unused by the 2-D (batch, split) grid
    else:
        raise NotImplementedError(
            "gluon MLA decode supports num_q_heads in [1, 16] (bh16bn64) or "
            f"{{64, 128}} (bh64), got {nhead}"
        )
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(f"gluon MLA decode requires bf16 q, got {q.dtype}")

    if kv_cache.dim() == 4:
        if kv_cache.shape[2] != 1 or kv_cache.shape[3] != qk_dim:
            raise ValueError(
                f"kv_cache must be [num_pages, page_size, 1, {qk_dim}], "
                f"got {tuple(kv_cache.shape)}"
            )
        page_size = kv_cache.shape[1]
    else:
        raise ValueError(f"kv_cache must be 4D, got {kv_cache.dim()}D")
    if kv_cache.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"gluon MLA decode requires bf16 kv_cache, got {kv_cache.dtype}"
        )
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous")
    if cache_seqlens.dtype != torch.int32:
        raise ValueError(f"cache_seqlens must be int32, got {cache_seqlens.dtype}")
    if page_table.dtype != torch.int32:
        raise ValueError(f"page_table must be int32, got {page_table.dtype}")

    q_nope = q[:, 0, :, :kv_lora_rank]
    q_pe = q[:, 0, :, kv_lora_rank:]
    kv_c = kv_cache.reshape(-1, qk_dim)

    if out is None:
        out = torch.empty(
            (batch_size, 1, nhead, kv_lora_rank), dtype=q.dtype, device=q.device
        )
    o = out.view(batch_size, nhead, kv_lora_rank)

    if return_lse:
        final_lse = torch.empty(
            (batch_size, nhead), dtype=torch.float32, device=q.device
        )
        stride_final_lse_b, stride_final_lse_h = final_lse.stride()
    else:
        final_lse = None
        stride_final_lse_b, stride_final_lse_h = 0, 0

    # buffer_load uses a scalar base + 32-bit offsets; KV pools > 2 GB fall back
    # to global_load (64-bit pointers).
    max_kv_bytes = kv_c.shape[0] * kv_c.stride(0) * kv_c.element_size()
    within_2gb = max_kv_bytes <= 0x80000000

    if regime == "bh64":
        num_kv_splits = _select_num_kv_splits_bh64(
            batch=batch_size, nhead=nhead, num_xcds=num_xcds, block_h=block_h
        )
    else:
        num_kv_splits = _select_num_kv_splits_bh16bn64(
            batch=batch_size, max_seqlen_k=max_seqlen_k, block_n=64
        )

    def _grid(splits: int) -> tuple[int, ...]:
        if regime == "bh64":
            # 3-D XCD-aware: (NUM_XCDS, head_block, (batch // NUM_XCDS) * splits).
            return (
                num_xcds,
                (nhead + block_h - 1) // block_h,
                (batch_size // num_xcds) * splits,
            )
        return (batch_size, splits)

    common_kwargs = dict(
        BLOCK_H=block_h,
        BLOCK_N=64,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        HEAD_DIM_CKV=kv_lora_rank,
        HEAD_DIM_KPE=qk_rope_head_dim,
        KV_PE_OFFSET=kv_lora_rank,
        WITHIN_2GB=within_2gb,
        NUM_XCDS=num_xcds,
        NHEAD=nhead,
        REGIME=regime,
        RETURN_LSE=return_lse,
        num_warps=4,
    )

    if num_kv_splits == 1:
        # Fast path: the single split spans the whole sequence, so stage-1
        # writes the final output (and lse) directly -- no stage-2 reduce.
        logits_buf = o.view(batch_size, nhead, 1, kv_lora_rank)
        grid = _grid(1)
        _mla_decode_gluon[grid](
            q_nope,
            q_pe,
            kv_c,
            kv_c,  # k_pe shares the compressed cache (shared latent+rope layout)
            page_table,
            cache_seqlens,
            logits_buf,
            softmax_scale,
            1.0,  # kv_scale (bf16 -> no dequant)
            q_nope.stride(0),
            q_nope.stride(1),
            q_pe.stride(0),
            q_pe.stride(1),
            kv_c.stride(-2),
            kv_c.stride(-2),
            page_table.stride(0),
            logits_buf.stride(0),
            logits_buf.stride(1),
            logits_buf.stride(2),
            None,
            0,
            0,
            0,
            final_lse,
            stride_final_lse_b,
            stride_final_lse_h,
            **common_kwargs,
        )
    else:
        # Split-K: stage-1 writes per-split partials + lse into scratch; the
        # stage-2 reduce merges them, masking the trailing empty splits that a
        # short sequence leaves behind.
        logits = torch.empty(
            (batch_size, nhead, num_kv_splits, kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        mid_lse = torch.empty(
            (batch_size, nhead, num_kv_splits),
            dtype=torch.float32,
            device=q.device,
        )
        grid = _grid(num_kv_splits)
        _mla_decode_gluon[grid](
            q_nope,
            q_pe,
            kv_c,
            kv_c,  # k_pe shares the compressed cache (shared latent+rope layout)
            page_table,
            cache_seqlens,
            logits,
            softmax_scale,
            1.0,  # kv_scale (bf16 -> no dequant)
            q_nope.stride(0),
            q_nope.stride(1),
            q_pe.stride(0),
            q_pe.stride(1),
            kv_c.stride(-2),
            kv_c.stride(-2),
            page_table.stride(0),
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            mid_lse,
            mid_lse.stride(0),
            mid_lse.stride(1),
            mid_lse.stride(2),
            None,  # Final_lse: written by the stage-2 reduce, not stage-1
            0,
            0,
            **common_kwargs,
        )

        reduce_grid = (batch_size, nhead)
        _mla_softmax_reducev_kernel[reduce_grid](
            logits,
            mid_lse,
            o,
            final_lse,
            cache_seqlens,
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            mid_lse.stride(0),
            mid_lse.stride(1),
            mid_lse.stride(2),
            o.stride(0),
            o.stride(1),
            stride_final_lse_b,
            stride_final_lse_h,
            NUM_KV_SPLITS=num_kv_splits,
            PAGE_SIZE=page_size,
            HEAD_DIM_CKV=kv_lora_rank,
            HAS_FINAL_LSE=return_lse,
        )

    if return_lse:
        return out, final_lse.view(batch_size, 1, nhead)
    return out
