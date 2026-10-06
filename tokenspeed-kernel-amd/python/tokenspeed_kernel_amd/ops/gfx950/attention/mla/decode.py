# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""MLA decode Gluon kernels for AMD GFX950.

The FP8 layout and ``bh16bn128`` regime are adapted from ROCm/AITER's
MIT-licensed ``aiter/ops/triton/gluon/mla_gluon.py`` at commit
``ae0bae8954110b12655e3232f68262dd63cd694e``. The implementation here retains
TokenSpeed's paged-cache interface (page IDs plus an arbitrary page size).

The shared kernel supports five fixed regimes:

* ``bh16bn128`` -- BF16/FP8 Q + FP8 KV, BLOCK_H=16, BLOCK_N=128 and
  ``num_q_heads <= 16`` and arbitrary batch sizes.
* ``bh16bn64`` -- BF16 Q + BF16 KV, BLOCK_H=16, BLOCK_N=64,
  ``num_q_heads <= 16``.
* ``bh64`` -- BLOCK_H=64, BLOCK_N=64, ``num_q_heads in {64, 128}``, 3-D
  XCD-aware grid, ``batch_size`` divisible by 64.
* ``bh16-multiblock`` -- BF16 Q/KV, BLOCK_H=16,
  ``num_q_heads == 64``, ``batch_size == 1``, 3-D grid.
* ``bh64-small`` -- BF16 Q/KV, BLOCK_H=64,
  ``num_q_heads == 64``, ``batch_size in {2, 4}``, 3-D grid.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention._common import (
    _INV_LN2,
    _mfma_unscaled_fp8,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mla.reduce_project_value import (
    gluon_mla_reduce_project_value_gfx950,
)

# Scale before the FP8 cast to retain small probabilities; cancel at normalization.
_FP8_PROBABILITY_SCALE = gl.constexpr(256.0)

# ===-----------------------------------------------------------------------===#
# Kernel Config
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionConfig:
    BLOCK_H: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_KV_SPLITS: gl.constexpr
    PAGE_SIZE: gl.constexpr
    HEAD_DIM_CKV: gl.constexpr
    HEAD_DIM_KPE: gl.constexpr
    Q_PE_BLOCK_H: gl.constexpr
    KV_PE_OFFSET: gl.constexpr
    WITHIN_2GB: gl.constexpr
    NUM_XCDS: gl.constexpr
    NHEAD: gl.constexpr
    REGIME: gl.constexpr
    IS_FP8_Q: gl.constexpr
    RETURN_LSE: gl.constexpr
    stride_q_nope_bs: gl.constexpr
    stride_q_nope_h: gl.constexpr
    stride_q_pe_bs: gl.constexpr
    stride_q_pe_h: gl.constexpr
    stride_kv_c_bs: gl.constexpr
    stride_k_pe_bs: gl.constexpr
    stride_req_to_tokens_bs: gl.constexpr
    stride_o_b: gl.constexpr
    stride_o_h: gl.constexpr
    stride_o_s: gl.constexpr
    stride_mid_lse_b: gl.constexpr
    stride_mid_lse_h: gl.constexpr
    stride_mid_lse_s: gl.constexpr
    stride_final_lse_b: gl.constexpr
    stride_final_lse_h: gl.constexpr
    blocked_q_nope: gl.constexpr
    shared_q_nope: gl.constexpr
    blocked_q_pe: gl.constexpr
    shared_q_pe: gl.constexpr
    mfma_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    blocked_kv: gl.constexpr
    shared_kv: gl.constexpr
    blocked_kpe: gl.constexpr
    shared_kpe: gl.constexpr
    blocked_page: gl.constexpr
    blocked_kv_slice: gl.constexpr
    linear_v: gl.constexpr
    shared_page: gl.constexpr
    blocked_lse: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        BLOCK_H,
        BLOCK_N,
        NUM_KV_SPLITS,
        PAGE_SIZE,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        KV_PE_OFFSET,
        WITHIN_2GB,
        NUM_XCDS,
        NHEAD,
        REGIME,
        IS_FP8_Q,
        RETURN_LSE,
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
        stride_mid_lse_b,
        stride_mid_lse_h,
        stride_mid_lse_s,
        stride_final_lse_b,
        stride_final_lse_h,
    ):
        # Q-side layouts + mfma_layout: switch by BLOCK_H.
        # bh64 has BLOCK_H=64 (warps tile M); bh16bn64 has BLOCK_H=16 (warps tile K).
        if BLOCK_H == 64:
            # bh64: Q is [64, 512] / [64, 64]; warps tile M.
            blocked_q_nope = gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[1, 64],
                warps_per_cta=[4, 1],
                order=[1, 0],
            )
            shared_q_nope = gl.PaddedSharedLayout(
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
            blocked_q_pe = gl.DistributedLinearLayout(
                reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
                lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
                warp_bases=((1, 0), (2, 0)),
                block_bases=[],
                shape=[64, 64],
            )
            shared_q_pe = gl.PaddedSharedLayout(
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
            mfma_layout = gl.amd.AMDMFMALayout(
                version=4,
                instr_shape=[16, 16, 32],
                transposed=True,
                warps_per_cta=[4, 1],
            )
        elif IS_FP8_Q:
            # Stage FP8 Q as [K, H] so K is the contiguous 16-byte DMA
            # dimension, then permute the descriptor for operand-A loads.
            blocked_q_nope = gl.DistributedLinearLayout(
                reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 4)),
                lane_bases=(
                    (16, 0),
                    (32, 0),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                    (0, 8),
                ),
                warp_bases=((0, 1), (0, 2)),
                block_bases=[],
                shape=[512, 16],
            )
            shared_q_nope = gl.PaddedSharedLayout(
                interval_padding_pairs=[[1024, 32], [8192, 16]],
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
                    [0, 8],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                ],
                cga_layout=[],
                shape=[512, 16],
            )
            # Pad the small Q-PE head tile to 64 so FP8 DMA remains legal.
            blocked_q_pe = gl.DistributedLinearLayout(
                reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 2)),
                lane_bases=(
                    (16, 0),
                    (32, 0),
                    (0, 4),
                    (0, 8),
                    (0, 16),
                    (0, 32),
                ),
                warp_bases=((0, 0), (0, 1)),
                block_bases=[],
                shape=[64, 64],
            )
            shared_q_pe = gl.PaddedSharedLayout(
                interval_padding_pairs=[[2048, 16]],
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
                    [0, 32],
                    [0, 1],
                    [0, 2],
                ],
                cga_layout=[],
                shape=[64, 64],
            )
            mfma_layout = gl.amd.AMDMFMALayout(
                version=4,
                instr_shape=[16, 16, 32],
                transposed=True,
                warps_per_cta=[1, 4],
            )
        else:
            # bh16bn64: Q is [16, 512] / [16, 64]; warps tile K.
            blocked_q_nope = gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[1, 64],
                warps_per_cta=[4, 1],
                order=[1, 0],
            )
            shared_q_nope = gl.PaddedSharedLayout(
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
            blocked_q_pe = gl.DistributedLinearLayout(
                reg_bases=((0, 1), (0, 2), (0, 4)),
                lane_bases=((0, 8), (0, 16), (0, 32), (1, 0), (2, 0), (4, 0)),
                warp_bases=((8, 0), (0, 0)),
                block_bases=[],
                shape=[16, 64],
            )
            shared_q_pe = gl.SwizzledSharedLayout(
                vec=8, per_phase=2, max_phase=8, order=[1, 0]
            )
            mfma_layout = gl.amd.AMDMFMALayout(
                version=4,
                instr_shape=[16, 16, 32],
                transposed=True,
                warps_per_cta=[1, 4],
            )

        # FP8 KV uses a 128-token tile; BF16 KV uses 64. These layouts are
        # adapted from AITER's bh16bn128/bh16bn64 implementation.
        if BLOCK_N == 128:
            blocked_kv = gl.DistributedLinearLayout(
                reg_bases=(
                    (1, 0),
                    (2, 0),
                    (4, 0),
                    (8, 0),
                    (0, 8),
                    (0, 4),
                    (0, 32),
                    (0, 64),
                ),
                lane_bases=(
                    (16, 0),
                    (32, 0),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                    (0, 16),
                ),
                warp_bases=((0, 1), (0, 2)),
                block_bases=[],
                shape=[512, 128],
            )
            shared_kv = gl.PaddedSharedLayout(
                interval_padding_pairs=[[1024, 32], [8192, 16]],
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
                    [0, 16],
                    [0, 1],
                    [0, 2],
                    [0, 8],
                    [0, 4],
                    [0, 32],
                    [0, 64],
                ],
                cga_layout=[],
                shape=[512, 128],
            )
            blocked_kpe = gl.DistributedLinearLayout(
                reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 2)),
                lane_bases=((16, 0), (32, 0), (0, 4), (0, 8), (0, 16), (0, 32)),
                warp_bases=((0, 64), (0, 1)),
                block_bases=[],
                shape=[64, 128],
            )
            shared_kpe = gl.PaddedSharedLayout(
                interval_padding_pairs=[[2048, 16]],
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
                    [0, 32],
                    [0, 64],
                    [0, 1],
                    [0, 2],
                ],
                cga_layout=[],
                shape=[64, 128],
            )
            blocked_page = gl.DistributedLinearLayout(
                reg_bases=((0,),),
                lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
                warp_bases=((64,), (0,)),
                block_bases=[],
                shape=[128],
            )
            blocked_kv_slice = gl.DistributedLinearLayout(
                reg_bases=(
                    (1, 0),
                    (2, 0),
                    (4, 0),
                    (8, 0),
                    (0, 8),
                    (0, 4),
                    (0, 32),
                ),
                lane_bases=(
                    (16, 0),
                    (32, 0),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                    (0, 16),
                ),
                warp_bases=((0, 1), (0, 2)),
                block_bases=[],
                shape=[512, 64],
            )
        else:
            blocked_kv = gl.DistributedLinearLayout(
                reg_bases=(
                    (1, 0),
                    (2, 0),
                    (4, 0),
                    (0, 8),
                    (0, 4),
                    (0, 16),
                    (0, 32),
                ),
                lane_bases=(
                    (8, 0),
                    (16, 0),
                    (32, 0),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                ),
                warp_bases=((0, 1), (0, 2)),
                block_bases=[],
                shape=[512, 64],
            )
            shared_kv = gl.PaddedSharedLayout(
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
            blocked_kpe = gl.DistributedLinearLayout(
                reg_bases=((1, 0), (2, 0), (4, 0), (0, 32)),
                lane_bases=((8, 0), (16, 0), (32, 0), (0, 4), (0, 8), (0, 16)),
                warp_bases=((0, 1), (0, 2)),
                block_bases=[],
                shape=[64, 64],
            )
            shared_kpe = gl.PaddedSharedLayout(
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
            blocked_page = gl.DistributedLinearLayout(
                reg_bases=((0,),),
                lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
                warp_bases=((0,), (0,)),
                block_bases=[],
                shape=[64],
            )
            blocked_kv_slice = gl.DistributedLinearLayout(
                reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
                lane_bases=(
                    (8, 0),
                    (16, 0),
                    (32, 0),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                ),
                warp_bases=((0, 1), (0, 2)),
                block_bases=[],
                shape=[512, 32],
            )

        # V is the latent slice of K, read back transposed for the PV dot.
        # bh64 tiles M across warps (degenerate warp_bases + extra reg bases);
        # bh16bn64 tiles the 64-wide K across warps.
        if BLOCK_H == 64:
            linear_v = gl.DistributedLinearLayout(
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
        elif REGIME == "bh16bn128":
            linear_v = gl.DistributedLinearLayout(
                reg_bases=(
                    (0, 1),
                    (0, 2),
                    (0, 4),
                    (0, 32),
                    (0, 64),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                ),
                lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
                warp_bases=((16, 0), (32, 0)),
                block_bases=[],
                shape=[512, 128],
            )
        else:
            linear_v = gl.DistributedLinearLayout(
                reg_bases=(
                    (0, 1),
                    (0, 2),
                    (0, 4),
                    (0, 32),
                    (64, 0),
                    (128, 0),
                    (256, 0),
                ),
                lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
                warp_bases=((16, 0), (32, 0)),
                block_bases=[],
                shape=[512, 64],
            )

        qk_k_width = 16 if IS_FP8_Q else 8
        pv_k_width = 8
        q_layout = gl.DotOperandLayout(
            operand_index=0, parent=mfma_layout, k_width=qk_k_width
        )
        k_layout = gl.DotOperandLayout(
            operand_index=1, parent=mfma_layout, k_width=qk_k_width
        )
        p_layout = gl.DotOperandLayout(
            operand_index=0, parent=mfma_layout, k_width=pv_k_width
        )
        v_layout = gl.DotOperandLayout(
            operand_index=1, parent=mfma_layout, k_width=pv_k_width
        )
        # Page-number scratch + lse store layouts (regime-independent).
        shared_page = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[0]
        )
        blocked_lse = gl.BlockedLayout(
            size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0]
        )

        self.BLOCK_H = gl.constexpr(BLOCK_H)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_KV_SPLITS = gl.constexpr(NUM_KV_SPLITS)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.HEAD_DIM_CKV = gl.constexpr(HEAD_DIM_CKV)
        self.HEAD_DIM_KPE = gl.constexpr(HEAD_DIM_KPE)
        self.Q_PE_BLOCK_H = gl.constexpr(64 if IS_FP8_Q else BLOCK_H)
        self.KV_PE_OFFSET = gl.constexpr(KV_PE_OFFSET)
        self.WITHIN_2GB = gl.constexpr(WITHIN_2GB)
        self.NUM_XCDS = gl.constexpr(NUM_XCDS)
        self.NHEAD = gl.constexpr(NHEAD)
        self.REGIME = gl.constexpr(REGIME)
        self.IS_FP8_Q = gl.constexpr(IS_FP8_Q)
        self.RETURN_LSE = gl.constexpr(RETURN_LSE)
        self.stride_q_nope_bs = gl.constexpr(stride_q_nope_bs)
        self.stride_q_nope_h = gl.constexpr(stride_q_nope_h)
        self.stride_q_pe_bs = gl.constexpr(stride_q_pe_bs)
        self.stride_q_pe_h = gl.constexpr(stride_q_pe_h)
        self.stride_kv_c_bs = gl.constexpr(stride_kv_c_bs)
        self.stride_k_pe_bs = gl.constexpr(stride_k_pe_bs)
        self.stride_req_to_tokens_bs = gl.constexpr(stride_req_to_tokens_bs)
        self.stride_o_b = gl.constexpr(stride_o_b)
        self.stride_o_h = gl.constexpr(stride_o_h)
        self.stride_o_s = gl.constexpr(stride_o_s)
        self.stride_mid_lse_b = gl.constexpr(stride_mid_lse_b)
        self.stride_mid_lse_h = gl.constexpr(stride_mid_lse_h)
        self.stride_mid_lse_s = gl.constexpr(stride_mid_lse_s)
        self.stride_final_lse_b = gl.constexpr(stride_final_lse_b)
        self.stride_final_lse_h = gl.constexpr(stride_final_lse_h)
        self.blocked_q_nope = gl.constexpr(blocked_q_nope)
        self.shared_q_nope = gl.constexpr(shared_q_nope)
        self.blocked_q_pe = gl.constexpr(blocked_q_pe)
        self.shared_q_pe = gl.constexpr(shared_q_pe)
        self.mfma_layout = gl.constexpr(mfma_layout)
        self.q_layout = gl.constexpr(q_layout)
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(p_layout)
        self.v_layout = gl.constexpr(v_layout)
        self.blocked_kv = gl.constexpr(blocked_kv)
        self.shared_kv = gl.constexpr(shared_kv)
        self.blocked_kpe = gl.constexpr(blocked_kpe)
        self.shared_kpe = gl.constexpr(shared_kpe)
        self.blocked_page = gl.constexpr(blocked_page)
        self.blocked_kv_slice = gl.constexpr(blocked_kv_slice)
        self.linear_v = gl.constexpr(linear_v)
        self.shared_page = gl.constexpr(shared_page)
        self.blocked_lse = gl.constexpr(blocked_lse)


# ===-----------------------------------------------------------------------===#
# Kernel Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionProgram:
    cfg: gl.constexpr
    Q_nope: gl.tensor
    Q_pe: gl.tensor
    Kv_c_cache: gl.tensor
    K_pe_cache: gl.tensor
    Req_to_tokens: gl.tensor
    Out: gl.tensor
    kv_scale: gl.tensor
    qk_scale: gl.tensor
    cur_batch: gl.tensor
    cur_head_id: gl.tensor
    split_kv_id: gl.tensor
    batch_page_start: gl.tensor
    split_kv_start: gl.tensor
    split_kv_end: gl.tensor
    num_iter: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        Q_nope,
        Q_pe,
        Kv_c_cache,
        K_pe_cache,
        Req_to_tokens,
        Out,
        kv_scale,
        qk_scale,
        cur_batch,
        cur_head_id,
        split_kv_id,
        batch_page_start,
        split_kv_start,
        split_kv_end,
        num_iter,
    ):
        self.cfg = gl.constexpr(cfg)
        self.Q_nope = Q_nope
        self.Q_pe = Q_pe
        self.Kv_c_cache = Kv_c_cache
        self.K_pe_cache = K_pe_cache
        self.Req_to_tokens = Req_to_tokens
        self.Out = Out
        self.kv_scale = kv_scale
        self.qk_scale = qk_scale
        self.cur_batch = cur_batch
        self.cur_head_id = cur_head_id
        self.split_kv_id = split_kv_id
        self.batch_page_start = batch_page_start
        self.split_kv_start = split_kv_start
        self.split_kv_end = split_kv_end
        self.num_iter = num_iter

    @gluon.jit
    def create(
        cfg,
        Q_nope,
        Q_pe,
        Kv_c_cache,
        K_pe_cache,
        Req_to_tokens,
        B_seq_len,
        Out,
        sm_scale,
        kv_scale,
    ):
        if cfg.REGIME == "bh64":
            cur_batch = (
                gl.program_id(0)
                + (gl.program_id(2) // cfg.NUM_KV_SPLITS) * cfg.NUM_XCDS
            )
            cur_head_id = gl.program_id(1)
            split_kv_id = gl.program_id(2) % cfg.NUM_KV_SPLITS
        elif cfg.REGIME == "bh16bn64" or cfg.REGIME == "bh16bn128":
            cur_batch = gl.program_id(0)
            split_kv_id = gl.program_id(1)
            # Head-block 0; use a runtime zero (aggregate fields hold tensors).
            cur_head_id = split_kv_id - split_kv_id
        elif cfg.REGIME == "bh16-multiblock" or cfg.REGIME == "bh64-small":
            cur_batch = gl.program_id(0)
            cur_head_id = gl.program_id(1)
            split_kv_id = gl.program_id(2)
        else:
            gl.static_assert(False, "unsupported MLA decode regime")

        # Paged 2-D view: Req_to_tokens = block_table[batch, max_pages],
        # B_seq_len = cache_seqlens[batch].
        batch_page_start = cfg.stride_req_to_tokens_bs * cur_batch
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch)

        num_pages = gl.cdiv(cur_batch_seq_len, cfg.PAGE_SIZE)
        pages_per_split = gl.cdiv(num_pages, cfg.NUM_KV_SPLITS)
        split_start_page = split_kv_id * pages_per_split
        split_end_page = gl.minimum(split_start_page + pages_per_split, num_pages)
        split_kv_start = split_start_page * cfg.PAGE_SIZE
        split_kv_end = gl.minimum(split_end_page * cfg.PAGE_SIZE, cur_batch_seq_len)
        # Clamp so empty (trailing) splits have start == end -> num_iter == 0.
        split_kv_end = gl.maximum(split_kv_end, split_kv_start)
        num_iter = gl.cdiv(split_kv_end - split_kv_start, cfg.BLOCK_N)

        # Fold KV dequant scale into the QK temperature.
        # bf16 KV: the wrapper passes kv_scale=1.0, so this is a no-op.
        qk_scale = sm_scale * kv_scale

        return AttentionProgram(
            gl.constexpr(cfg),
            Q_nope,
            Q_pe,
            Kv_c_cache,
            K_pe_cache,
            Req_to_tokens,
            Out,
            kv_scale,
            qk_scale,
            cur_batch,
            cur_head_id,
            split_kv_id,
            batch_page_start,
            split_kv_start,
            split_kv_end,
            num_iter,
        )

    @gluon.jit
    def issue_load_q_nope(self, buf):
        cfg = self.cfg
        if cfg.IS_FP8_Q:
            offs_d_ckv = gl.arange(
                0, cfg.HEAD_DIM_CKV, layout=gl.SliceLayout(1, cfg.blocked_q_nope)
            )
            cur_head = self.cur_head_id * cfg.BLOCK_H + gl.arange(
                0, cfg.BLOCK_H, layout=gl.SliceLayout(0, cfg.blocked_q_nope)
            )
            offs_q_nope = (
                self.cur_batch * cfg.stride_q_nope_bs
                + cur_head[None, :] * cfg.stride_q_nope_h
                + offs_d_ckv[:, None]
            )
            mask = (cur_head < cfg.NHEAD)[None, :]
        else:
            offs_d_ckv = gl.arange(
                0, cfg.HEAD_DIM_CKV, layout=gl.SliceLayout(0, cfg.blocked_q_nope)
            )
            cur_head = self.cur_head_id * cfg.BLOCK_H + gl.arange(
                0, cfg.BLOCK_H, layout=gl.SliceLayout(1, cfg.blocked_q_nope)
            )
            offs_q_nope = (
                self.cur_batch * cfg.stride_q_nope_bs
                + cur_head[:, None] * cfg.stride_q_nope_h
                + offs_d_ckv[None, :]
            )
            mask = (cur_head < cfg.NHEAD)[:, None] if cfg.NHEAD < cfg.BLOCK_H else None
        # For nhead < BLOCK_H, mask OOB heads to zero on Q load and skip OOB O
        # stores; wasted MFMA lanes are free (memory-bound).
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            buf,
            self.Q_nope,
            offs_q_nope,
            mask=mask,
        )
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def issue_load_q_pe(self, buf):
        cfg = self.cfg
        if cfg.IS_FP8_Q:
            offs_d_kpe = gl.arange(
                0, cfg.HEAD_DIM_KPE, layout=gl.SliceLayout(1, cfg.blocked_q_pe)
            )
            cur_head_qpe = self.cur_head_id * cfg.BLOCK_H + gl.arange(
                0, cfg.Q_PE_BLOCK_H, layout=gl.SliceLayout(0, cfg.blocked_q_pe)
            )
            offs_q_pe = (
                self.cur_batch * cfg.stride_q_pe_bs
                + cur_head_qpe[None, :] * cfg.stride_q_pe_h
                + offs_d_kpe[:, None]
            )
            mask = (cur_head_qpe < cfg.NHEAD)[None, :]
        else:
            offs_d_kpe = gl.arange(
                0, cfg.HEAD_DIM_KPE, layout=gl.SliceLayout(0, cfg.blocked_q_pe)
            )
            cur_head_qpe = self.cur_head_id * cfg.BLOCK_H + gl.arange(
                0, cfg.BLOCK_H, layout=gl.SliceLayout(1, cfg.blocked_q_pe)
            )
            offs_q_pe = (
                self.cur_batch * cfg.stride_q_pe_bs
                + cur_head_qpe[:, None] * cfg.stride_q_pe_h
                + offs_d_kpe[None, :]
            )
            mask = (
                (cur_head_qpe < cfg.NHEAD)[:, None] if cfg.NHEAD < cfg.BLOCK_H else None
            )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            buf,
            self.Q_pe,
            offs_q_pe,
            mask=mask,
        )
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def local_load_q(self, buf_q_nope, buf_q_pe):
        cfg = self.cfg
        if cfg.IS_FP8_Q:
            q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(
                buf_q_nope.permute([1, 0]), cfg.q_layout
            )
            q_pe_buffer = buf_q_pe.permute([1, 0])
            if cfg.Q_PE_BLOCK_H != cfg.BLOCK_H:
                q_pe_buffer = q_pe_buffer.slice(0, cfg.BLOCK_H, 0)
            q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
                q_pe_buffer, cfg.q_layout
            )
        else:
            q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(
                buf_q_nope, cfg.q_layout
            )
            q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_pe, cfg.q_layout)
        return q_nope, q_pe

    @gluon.jit
    def issue_page_load(self, buf, start_n):
        cfg = self.cfg
        offs_n_page = start_n + gl.arange(0, cfg.BLOCK_N, layout=cfg.blocked_page)
        offs_page = self.batch_page_start + offs_n_page // cfg.PAGE_SIZE
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            buf, self.Req_to_tokens, offs_page, offs_n_page < self.split_kv_end
        )
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def physical_token_location(
        self, page_number, token_offset, WITHIN_2GB: gl.constexpr
    ):
        cfg = self.cfg
        page_offset = token_offset % cfg.PAGE_SIZE
        if WITHIN_2GB:
            # Keep the buffer-load specialization in 32-bit arithmetic; its
            # cache size guarantees the final byte offset remains below 2 GiB.
            location = page_number * cfg.PAGE_SIZE + page_offset
        else:
            # Widen before multiplying so neither this row nor its later
            # stride can wrap for a large-cache global load.
            location = page_number.to(gl.int64) * cfg.PAGE_SIZE + page_offset.to(
                gl.int64
            )
        return location

    @gluon.jit
    def issue_kv_load(self, smem, ptr, offsets, mask):
        # Buffer loads use 32-bit offsets. Large pools use 64-bit global loads;
        # initialized page scratch keeps addresses for tail lanes in bounds.
        if self.cfg.WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smem, ptr, offsets, mask=mask)
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptr + offsets)
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def compute_qk(self, q_nope, q_pe, kv_buf, kpe_buf, RELAXED: gl.constexpr):
        cfg = self.cfg
        dtype = self.Q_nope.type.element_ty
        if RELAXED:
            k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_buf, cfg.k_layout)
        else:
            k_c = kv_buf.load(layout=cfg.k_layout)
        zeros = gl.zeros(
            [cfg.BLOCK_H, cfg.BLOCK_N], dtype=gl.float32, layout=cfg.mfma_layout
        )
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)
        if RELAXED:
            k_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(kpe_buf, cfg.k_layout)
        else:
            k_pe = kpe_buf.load(layout=cfg.k_layout)
        qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
        return qk

    @gluon.jit
    def softmax(self, qk, offs_base, e_max, e_sum, acc):
        cfg = self.cfg
        dtype = self.Q_nope.type.element_ty
        qk *= self.qk_scale
        offs_n_qk = (
            self.split_kv_start
            + offs_base
            + gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.mfma_layout))
        )
        qk = gl.where(offs_n_qk[None, :] < self.split_kv_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp2((e_max - n_e_max) * _INV_LN2)
        p = gl.exp2((qk - n_e_max[:, None]) * _INV_LN2)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        if cfg.IS_FP8_Q:
            p *= _FP8_PROBABILITY_SCALE
        p = p.to(dtype)
        p = gl.convert_layout(p, cfg.p_layout)
        acc *= re_scale[:, None]
        return p, e_max, e_sum, acc

    @gluon.jit
    def compute_pv(self, p, acc, kv_buf, RELAXED: gl.constexpr):
        cfg = self.cfg
        dtype = self.Q_nope.type.element_ty
        if RELAXED:
            v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_buf, cfg.linear_v)
        else:
            v_c = kv_buf.load(layout=cfg.linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, cfg.v_layout)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)
        return acc

    @gluon.jit
    def store_output(self, acc, e_sum):
        cfg = self.cfg
        out_dtype = self.Out.type.element_ty
        cur_head_o = self.cur_head_id * cfg.BLOCK_H + gl.arange(
            0, cfg.BLOCK_H, layout=gl.SliceLayout(1, cfg.mfma_layout)
        )
        offs_d_ckv_o = gl.arange(
            0, cfg.HEAD_DIM_CKV, layout=gl.SliceLayout(0, cfg.mfma_layout)
        )
        offs_o = (
            self.cur_batch * cfg.stride_o_b
            + cur_head_o[:, None] * cfg.stride_o_h
            + self.split_kv_id * cfg.stride_o_s
            + offs_d_ckv_o[None, :]
        )
        acc *= self.kv_scale
        rcp = 1.0 / e_sum
        if cfg.IS_FP8_Q:
            rcp /= _FP8_PROBABILITY_SCALE
        stored_value = (acc * rcp[:, None]).to(out_dtype)
        if cfg.NHEAD < cfg.BLOCK_H:
            gl.amd.cdna4.buffer_store(
                stored_value,
                ptr=self.Out,
                offsets=offs_o,
                mask=(cur_head_o < cfg.NHEAD)[:, None],
            )
        else:
            gl.amd.cdna4.buffer_store(stored_value, ptr=self.Out, offsets=offs_o)

    @gluon.jit
    def store_lse(self, e_max, e_sum, Mid_lse, Final_lse):
        # Mid_lse / Final_lse can be None (they're passed straight from the
        # kernel args), so they stay method params rather than aggregate fields.
        cfg = self.cfg
        cur_head_lse = self.cur_head_id * cfg.BLOCK_H + gl.arange(
            0, cfg.BLOCK_H, layout=cfg.blocked_lse
        )
        if cfg.RETURN_LSE and cfg.NUM_KV_SPLITS == 1:
            # split==1: single split is the whole sequence, so its lse is final.
            offs_final_lse = (
                self.cur_batch * cfg.stride_final_lse_b
                + cur_head_lse * cfg.stride_final_lse_h
            )
            lse = e_max + gl.log(e_sum)
            lse = gl.convert_layout(lse, cfg.blocked_lse)
            if cfg.NHEAD < cfg.BLOCK_H:
                gl.amd.cdna4.buffer_store(
                    lse,
                    ptr=Final_lse,
                    offsets=offs_final_lse,
                    mask=(cur_head_lse < cfg.NHEAD),
                )
            else:
                gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse)
        elif cfg.NUM_KV_SPLITS > 1:
            # per-split lse for stage-2 reduce.
            offs_mid_lse = (
                self.cur_batch * cfg.stride_mid_lse_b
                + cur_head_lse * cfg.stride_mid_lse_h
                + self.split_kv_id * cfg.stride_mid_lse_s
            )
            lse = e_max + gl.log(e_sum)
            lse = gl.convert_layout(lse, cfg.blocked_lse)
            if cfg.NHEAD < cfg.BLOCK_H:
                gl.amd.cdna4.buffer_store(
                    lse,
                    ptr=Mid_lse,
                    offsets=offs_mid_lse,
                    mask=(cur_head_lse < cfg.NHEAD),
                )
            else:
                gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse)


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _mla_decode_gluon(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    sm_scale,
    kv_scale,
    stride_q_nope_bs: gl.constexpr,
    stride_q_nope_h: gl.constexpr,
    stride_q_pe_bs: gl.constexpr,
    stride_q_pe_h: gl.constexpr,
    stride_kv_c_bs: gl.constexpr,
    stride_k_pe_bs: gl.constexpr,
    stride_req_to_tokens_bs: gl.constexpr,
    stride_o_b: gl.constexpr,
    stride_o_h: gl.constexpr,
    stride_o_s: gl.constexpr,
    Mid_lse,  # split>1: per-split fp32 lse [B, H, NUM_KV_SPLITS] (else None)
    stride_mid_lse_b: gl.constexpr,
    stride_mid_lse_h: gl.constexpr,
    stride_mid_lse_s: gl.constexpr,
    Final_lse,  # RETURN_LSE only: merged fp32 lse [B, H] (else None)
    stride_final_lse_b: gl.constexpr,
    stride_final_lse_h: gl.constexpr,
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
    IS_FP8_Q: gl.constexpr,
    RETURN_LSE: gl.constexpr,
):
    cfg = AttentionConfig(
        BLOCK_H,
        BLOCK_N,
        NUM_KV_SPLITS,
        PAGE_SIZE,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        KV_PE_OFFSET,
        WITHIN_2GB,
        NUM_XCDS,
        NHEAD,
        REGIME,
        IS_FP8_Q,
        RETURN_LSE,
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
        stride_mid_lse_b,
        stride_mid_lse_h,
        stride_mid_lse_s,
        stride_final_lse_b,
        stride_final_lse_h,
    )
    program = AttentionProgram.create(
        cfg,
        Q_nope,
        Q_pe,
        Kv_c_cache,
        K_pe_cache,
        Req_to_tokens,
        B_seq_len,
        O,
        sm_scale,
        kv_scale,
    )

    _mla_decode_program(program, Mid_lse, Final_lse)


@gluon.jit
def _mla_decode_program(program, Mid_lse, Final_lse):
    if program.split_kv_start >= program.split_kv_end:
        return

    cfg = program.cfg
    dtype = program.Q_nope.type.element_ty
    kvtype = program.Kv_c_cache.type.element_ty

    if cfg.IS_FP8_Q:
        buf_q_nope = gl.allocate_shared_memory(
            dtype, shape=[cfg.HEAD_DIM_CKV, cfg.BLOCK_H], layout=cfg.shared_q_nope
        )
        buf_q_pe = gl.allocate_shared_memory(
            dtype, shape=[cfg.HEAD_DIM_KPE, cfg.Q_PE_BLOCK_H], layout=cfg.shared_q_pe
        )
    else:
        buf_q_nope = gl.allocate_shared_memory(
            dtype, shape=[cfg.BLOCK_H, cfg.HEAD_DIM_CKV], layout=cfg.shared_q_nope
        )
        buf_q_pe = gl.allocate_shared_memory(
            dtype, shape=[cfg.BLOCK_H, cfg.HEAD_DIM_KPE], layout=cfg.shared_q_pe
        )

    # load q_nope / q_pe
    program.issue_load_q_nope(buf_q_nope)
    program.issue_load_q_pe(buf_q_pe)

    e_max = gl.zeros(
        [cfg.BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, cfg.mfma_layout)
    ) - float("inf")
    e_sum = gl.zeros(
        [cfg.BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, cfg.mfma_layout)
    )
    acc = gl.zeros(
        [cfg.BLOCK_H, cfg.HEAD_DIM_CKV], dtype=gl.float32, layout=cfg.mfma_layout
    )

    num_iter = program.num_iter
    split_kv_start = program.split_kv_start
    start_n = split_kv_start

    # bufs of page_number
    bufs_page = gl.allocate_shared_memory(
        gl.int32, shape=[2, cfg.BLOCK_N], layout=cfg.shared_page
    )
    if not cfg.WITHIN_2GB:
        # Large-cache global loads are unmasked, so every page-ID lane must be
        # valid. Initialize each ping-pong buffer once; masked page loads then
        # retain either zero or a valid ID written by an earlier tile.
        page_zeros = gl.zeros([cfg.BLOCK_N], dtype=gl.int32, layout=cfg.blocked_page)
        bufs_page.index(0).store(page_zeros)
        bufs_page.index(1).store(page_zeros)

    # prologue: global load page numbers for the first two tiles
    program.issue_page_load(bufs_page.index(0), start_n)
    start_n += cfg.BLOCK_N
    program.issue_page_load(bufs_page.index(1), start_n)

    # local load Q
    gl.amd.cdna4.async_copy.wait_group(2)
    q_nope, q_pe = program.local_load_q(buf_q_nope, buf_q_pe)

    # move here to work around allocate_shared_memory bug
    bufs_kv = gl.allocate_shared_memory(
        kvtype, shape=[2, cfg.HEAD_DIM_CKV, cfg.BLOCK_N], layout=cfg.shared_kv
    )
    bufs_kpe = gl.allocate_shared_memory(
        kvtype, shape=[2, cfg.HEAD_DIM_KPE, cfg.BLOCK_N], layout=cfg.shared_kpe
    )

    # global load K (first tile)
    # local load page number (pe view)
    gl.amd.cdna4.async_copy.wait_group(1)
    kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page.index(0), gl.SliceLayout(0, cfg.blocked_kpe)
    )
    # paged KV: physical row = page * PAGE_SIZE + (token % PAGE_SIZE)
    offs_n_pe0 = split_kv_start + gl.arange(
        0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.blocked_kpe)
    )
    kv_loc_pe = program.physical_token_location(
        kv_page_number_pe, offs_n_pe0, cfg.WITHIN_2GB
    )

    # local load page number for slice 0
    bufs_page_0 = bufs_page.index(0).slice(0, cfg.BLOCK_N // 2, 0)
    kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page_0, gl.SliceLayout(0, cfg.blocked_kv_slice)
    )
    offs_n_nope0 = split_kv_start + gl.arange(
        0, cfg.BLOCK_N // 2, layout=gl.SliceLayout(0, cfg.blocked_kv_slice)
    )
    kv_loc0 = program.physical_token_location(
        kv_page_number_0, offs_n_nope0, cfg.WITHIN_2GB
    )

    # global load K_nope slice 0
    offs_d_ckv_10 = gl.arange(
        0, cfg.HEAD_DIM_CKV, layout=gl.SliceLayout(1, cfg.blocked_kv_slice)
    )
    offs_k_c0 = kv_loc0[None, :] * cfg.stride_kv_c_bs + offs_d_ckv_10[:, None]
    bufs_kv0 = bufs_kv.index(0).slice(0, cfg.BLOCK_N // 2, 1)
    program.issue_kv_load(
        bufs_kv0,
        program.Kv_c_cache,
        offs_k_c0,
        offs_n_nope0[None, :] < program.split_kv_end,
    )

    # global load K_pe
    offs_d_kpe_1 = gl.arange(
        0, cfg.HEAD_DIM_KPE, layout=gl.SliceLayout(1, cfg.blocked_kpe)
    )
    offs_k_pe = (
        kv_loc_pe[None, :] * cfg.stride_k_pe_bs
        + offs_d_kpe_1[:, None]
        + cfg.KV_PE_OFFSET
    )
    program.issue_kv_load(
        bufs_kpe.index(0),
        program.K_pe_cache,
        offs_k_pe,
        offs_n_pe0[None, :] < program.split_kv_end,
    )

    # local load page number for slice 1
    bufs_page_1 = bufs_page.index(0).slice(cfg.BLOCK_N // 2, cfg.BLOCK_N // 2, 0)
    kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_page_1, gl.SliceLayout(0, cfg.blocked_kv_slice)
    )
    offs_n_nope1 = offs_n_nope0 + cfg.BLOCK_N // 2
    kv_loc1 = program.physical_token_location(
        kv_page_number_1, offs_n_nope1, cfg.WITHIN_2GB
    )

    # global load K_nope slice 1
    bufs_kv1 = bufs_kv.index(0).slice(cfg.BLOCK_N // 2, cfg.BLOCK_N // 2, 1)
    offs_k_c1 = kv_loc1[None, :] * cfg.stride_kv_c_bs + offs_d_ckv_10[:, None]
    program.issue_kv_load(
        bufs_kv1,
        program.Kv_c_cache,
        offs_k_c1,
        offs_n_nope1[None, :] < program.split_kv_end,
    )

    buf_idx = 0
    # main loop
    for i in range(num_iter - 2):
        async_idx = (buf_idx + 1) % 2

        gl.amd.cdna4.async_copy.wait_group(0)
        # global load page number (prefetch tile i+2)
        program.issue_page_load(bufs_page.index(buf_idx), start_n + cfg.BLOCK_N)

        # global load K slice 0
        bufs_kv0 = bufs_kv.index(async_idx).slice(0, cfg.BLOCK_N // 2, 1)
        bufs_kv1 = bufs_kv.index(async_idx).slice(cfg.BLOCK_N // 2, cfg.BLOCK_N // 2, 1)
        # local load page number for slice 0
        bufs_page_0 = bufs_page.index(async_idx).slice(0, cfg.BLOCK_N // 2, 0)
        kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page_0, gl.SliceLayout(0, cfg.blocked_kv_slice)
        )
        offs_n_nope0 = start_n + gl.arange(
            0, cfg.BLOCK_N // 2, layout=gl.SliceLayout(0, cfg.blocked_kv_slice)
        )
        kv_loc0 = program.physical_token_location(
            kv_page_number_0, offs_n_nope0, cfg.WITHIN_2GB
        )
        # global load K_nope slice 0
        offs_d_ckv_10 = gl.arange(
            0, cfg.HEAD_DIM_CKV, layout=gl.SliceLayout(1, cfg.blocked_kv_slice)
        )
        offs_k_c0 = kv_loc0[None, :] * cfg.stride_kv_c_bs + offs_d_ckv_10[:, None]
        program.issue_kv_load(
            bufs_kv0,
            program.Kv_c_cache,
            offs_k_c0,
            offs_n_nope0[None, :] < program.split_kv_end,
        )

        # local load page_number_pe + global load K_pe
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page.index(async_idx), gl.SliceLayout(0, cfg.blocked_kpe)
        )
        offs_n_pe = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.blocked_kpe)
        )
        kv_loc_pe = program.physical_token_location(
            kv_page_number_pe, offs_n_pe, cfg.WITHIN_2GB
        )
        offs_d_kpe_1 = gl.arange(
            0, cfg.HEAD_DIM_KPE, layout=gl.SliceLayout(1, cfg.blocked_kpe)
        )
        offs_k_pe = (
            kv_loc_pe[None, :] * cfg.stride_k_pe_bs
            + offs_d_kpe_1[:, None]
            + cfg.KV_PE_OFFSET
        )
        program.issue_kv_load(
            bufs_kpe.index(async_idx),
            program.K_pe_cache,
            offs_k_pe,
            offs_n_pe[None, :] < program.split_kv_end,
        )

        # dot (part0)
        qk = program.compute_qk(
            q_nope, q_pe, bufs_kv.index(buf_idx), bufs_kpe.index(buf_idx), True
        )

        # local load page number for slice 1 + global load K_nope slice 1
        bufs_page_1 = bufs_page.index(async_idx).slice(
            cfg.BLOCK_N // 2, cfg.BLOCK_N // 2, 0
        )
        kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page_1, gl.SliceLayout(0, cfg.blocked_kv_slice)
        )
        offs_n1 = offs_n_nope0 + cfg.BLOCK_N // 2
        kv_loc1 = program.physical_token_location(
            kv_page_number_1, offs_n1, cfg.WITHIN_2GB
        )
        offs_k_c1 = kv_loc1[None, :] * cfg.stride_kv_c_bs + offs_d_ckv_10[:, None]
        program.issue_kv_load(
            bufs_kv1,
            program.Kv_c_cache,
            offs_k_c1,
            offs_n1[None, :] < program.split_kv_end,
        )

        # softmax + dot (part1)
        p, e_max, e_sum, acc = program.softmax(qk, i * cfg.BLOCK_N, e_max, e_sum, acc)
        acc = program.compute_pv(p, acc, bufs_kv.index(buf_idx), True)

        start_n += cfg.BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    # epilogue 1
    # Runtime guard: a split can cover fewer than 2 KV blocks (short sequences).
    if num_iter >= 2:
        async_idx = (buf_idx + 1) % 2

        # global load K (full tile)
        gl.amd.cdna4.async_copy.wait_group(3)
        kv_page_number = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page.index(async_idx), gl.SliceLayout(0, cfg.blocked_kv)
        )
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_page.index(async_idx), gl.SliceLayout(0, cfg.blocked_kpe)
        )
        offs_n_nope = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.blocked_kv)
        )
        offs_n_pe = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.blocked_kpe)
        )
        kv_loc = program.physical_token_location(
            kv_page_number, offs_n_nope, cfg.WITHIN_2GB
        )
        kv_loc_pe = program.physical_token_location(
            kv_page_number_pe, offs_n_pe, cfg.WITHIN_2GB
        )
        # global load K_nope
        offs_d_ckv_1 = gl.arange(
            0, cfg.HEAD_DIM_CKV, layout=gl.SliceLayout(1, cfg.blocked_kv)
        )
        offs_k_c = kv_loc[None, :] * cfg.stride_kv_c_bs + offs_d_ckv_1[:, None]
        program.issue_kv_load(
            bufs_kv.index(async_idx),
            program.Kv_c_cache,
            offs_k_c,
            offs_n_nope[None, :] < program.split_kv_end,
        )
        # global load K_pe
        offs_d_kpe_1 = gl.arange(
            0, cfg.HEAD_DIM_KPE, layout=gl.SliceLayout(1, cfg.blocked_kpe)
        )
        offs_k_pe = (
            kv_loc_pe[None, :] * cfg.stride_k_pe_bs
            + offs_d_kpe_1[:, None]
            + cfg.KV_PE_OFFSET
        )
        program.issue_kv_load(
            bufs_kpe.index(async_idx),
            program.K_pe_cache,
            offs_k_pe,
            offs_n_pe[None, :] < program.split_kv_end,
        )

        # dot, softmax, dot
        gl.amd.cdna4.async_copy.wait_group(2)
        qk = program.compute_qk(
            q_nope, q_pe, bufs_kv.index(buf_idx), bufs_kpe.index(buf_idx), False
        )
        p, e_max, e_sum, acc = program.softmax(
            qk, (num_iter - 2) * cfg.BLOCK_N, e_max, e_sum, acc
        )
        acc = program.compute_pv(p, acc, bufs_kv.index(buf_idx), False)

        start_n += cfg.BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    # epilogue 2
    # dot, softmax, dot
    gl.amd.cdna4.async_copy.wait_group(0)
    qk = program.compute_qk(
        q_nope, q_pe, bufs_kv.index(buf_idx), bufs_kpe.index(buf_idx), False
    )
    p, e_max, e_sum, acc = program.softmax(
        qk, (num_iter - 1) * cfg.BLOCK_N, e_max, e_sum, acc
    )
    acc = program.compute_pv(p, acc, bufs_kv.index(buf_idx), False)

    program.store_output(acc, e_sum)
    program.store_lse(e_max, e_sum, Mid_lse, Final_lse)


@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    Mid_lse,
    O,
    Final_lse,
    B_seq_len,
    stride_l_b: tl.constexpr,
    stride_l_h: tl.constexpr,
    stride_l_s: tl.constexpr,
    stride_ml_b: tl.constexpr,
    stride_ml_h: tl.constexpr,
    stride_ml_s: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_fl_b: tl.constexpr,
    stride_fl_h: tl.constexpr,
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

    for split_kv_id in range(NUM_KV_SPLITS):
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


@gluon.jit
def _load_page(
    Pages,
    request,
    start,
    end,
    stride_page,
    PAGE: gl.constexpr,
):
    # Keep the page ID in a VGPR until the next copy needs a buffer descriptor.
    # Moving it to an SGPR here would wait for this load and the preceding
    # direct-to-LDS copy, since these loads complete in order.
    return gl.load(Pages + request * stride_page + start // PAGE, start < end, 0)


@gluon.jit
def _make_kv_offsets(
    STRIDE_KV: gl.constexpr,
    N: gl.constexpr,
    kv_load: gl.constexpr,
    pe_load: gl.constexpr,
    shared: gl.constexpr,
    shared_pe: gl.constexpr,
):
    # Direct-to-LDS copies write 16 bytes per lane to linear LDS addresses.
    # Precompute the source-column permutation that produces the swizzled
    # tile through these linear writes, avoiding per-copy lane shuffles of
    # offsets and masks.
    gl.static_assert(shared.order[0] == 0 and shared_pe.order[0] == 0)
    n = gl.arange(0, N, layout=gl.SliceLayout(0, kv_load))
    d = gl.arange(0, 512, layout=gl.SliceLayout(1, kv_load))
    phase = (n[None, :] // shared.per_phase) % shared.max_phase
    column = ((d[:, None] // shared.vec) ^ phase) * shared.vec + d[:, None] % shared.vec
    n_pe = gl.arange(0, N, layout=gl.SliceLayout(0, pe_load))
    d_pe = gl.arange(0, 64, layout=gl.SliceLayout(1, pe_load))
    phase_pe = (n_pe[None, :] // shared_pe.per_phase) % shared_pe.max_phase
    column_pe = ((d_pe[:, None] // shared_pe.vec) ^ phase_pe) * shared_pe.vec + d_pe[
        :, None
    ] % shared_pe.vec
    return n[None, :] * STRIDE_KV + column, n_pe[None, :] * STRIDE_KV + 512 + column_pe


@gluon.jit
def _issue_load_kv(
    buf,
    pebuf,
    KV,
    page,
    start,
    end,
    offsets,
    pe_offsets,
    STRIDE_KV: gl.constexpr,
    PAGE: gl.constexpr,
    N: gl.constexpr,
    kv_load: gl.constexpr,
    pe_load: gl.constexpr,
    linear: gl.constexpr,
):
    # Base each buffer at the tile's first token using 64-bit arithmetic.
    # The 32-bit buffer-load offsets then stay below N * STRIDE_KV even for
    # a KV pool larger than 2 GiB.
    gl.static_assert(PAGE >= N and PAGE % N == 0)
    # Buffer descriptors must be in SGPRs; the page ID is uniform.
    page = gl.inline_asm(
        "v_readfirstlane_b32 $0, $1", ("=s", "v"), [page], gl.int32, is_pure=True
    )
    base = KV + (page.to(gl.int64) * PAGE + start % PAGE) * STRIDE_KV
    rows = end - start
    n = gl.arange(0, N, layout=gl.SliceLayout(0, kv_load))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf.reinterpret(buf.dtype, [512, N], linear),
        base,
        offsets,
        mask=(n < rows)[None, :],
    )
    n_pe = gl.arange(0, N, layout=gl.SliceLayout(0, pe_load))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        pebuf.reinterpret(pebuf.dtype, [64, N], linear),
        base,
        pe_offsets,
        mask=(n_pe < rows)[None, :],
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.constexpr_function
def _transpose_operand_layout(layout):
    return gl.DistributedLinearLayout(
        [list(reversed(b)) for b in layout.reg_bases],
        [list(reversed(b)) for b in layout.lane_bases],
        [list(reversed(b)) for b in layout.warp_bases],
        [list(reversed(b)) for b in layout.block_bases],
        list(reversed(layout.shape)),
    )


@gluon.constexpr_function
def _replicated_pair_layout(row_layout):
    return gl.DistributedLinearLayout(
        [[x[0], 0] for x in row_layout.reg_bases] + [[0, 1]],
        [[x[0], 0] for x in row_layout.lane_bases],
        [[x[0], 0] for x in row_layout.warp_bases],
        [],
        [row_layout.shape[0], 2],
    )


_KV_REUSE_MIN_HISTORY = gl.constexpr(4096)
_QUERY_ROW_TILE = gl.constexpr(64)
_QUERY_HEAD_TILE = gl.constexpr(16)
_QUERY_VALUE_TILE = gl.constexpr(128)


@gluon.constexpr_function
def _supports_wide_query_block(heads, queries, page):
    return page == 64 and _QUERY_ROW_TILE.value < queries * heads <= 128


@gluon.constexpr_function
def _single_query_split_bucket(heads, queries, split_bucket):
    groups = triton.cdiv(queries * heads, _QUERY_ROW_TILE.value)
    return triton.next_power_of_2(groups * split_bucket // queries)


@gluon.jit
def _use_wide_query_block(length, splits, H: gl.constexpr, QLEN: gl.constexpr):
    tiles = gl.cdiv(gl.cdiv(length, 64), splits)
    return tiles >= gl.cdiv(QLEN * H, 4) + 8


@gluon.jit
def _compute_pv(
    buffer,
    probability,
    acc,
    alpha,
    chunk: gl.constexpr,
    value_layout: gl.constexpr,
    v_layout: gl.constexpr,
):
    value = gl.amd.cdna4.async_copy.load_shared_relaxed(
        buffer.slice(chunk * _QUERY_VALUE_TILE, _QUERY_VALUE_TILE, 0), value_layout
    )
    value = gl.convert_layout(gl.permute(value, [1, 0]), v_layout, assert_trivial=True)
    return _mfma_unscaled_fp8(probability, value, acc * alpha[:, None])


@gluon.jit
def _store_partial(Output, acc, reciprocal, valid, chunk: gl.constexpr):
    columns = gl.arange(0, _QUERY_VALUE_TILE, layout=gl.SliceLayout(0, acc.type.layout))
    gl.store(
        Output + chunk * _QUERY_VALUE_TILE + columns[None, :],
        acc * reciprocal[:, None],
        valid[:, None],
    )


def _launch_metadata(grid, kernel, args):
    return {"name": kernel.name}


@gluon.jit
def _decode_single_query(
    Q,
    KV,
    Pages,
    Partials,
    LSE,
    length,
    query_group,
    split,
    STRIDE_Q_B: gl.constexpr,
    STRIDE_Q_S: gl.constexpr,
    STRIDE_Q_H: gl.constexpr,
    STRIDE_KV: gl.constexpr,
    stride_page,
    H: gl.constexpr,
    QLEN: gl.constexpr,
    PAGE: gl.constexpr,
    splits,
    SPLIT_BUCKET: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    sm_scale,
):
    request = gl.program_id(0)
    groups: gl.constexpr = gl.cdiv(QLEN * H, _QUERY_ROW_TILE)
    work = query_group * splits + split
    query = work % QLEN
    short_split = work // QLEN
    short_splits = groups * splits // QLEN
    if short_split >= short_splits:
        return
    visible = gl.maximum(length - QLEN + query + 1, 0)
    pages_per_split = gl.cdiv(gl.cdiv(visible, PAGE), short_splits)
    first = short_split * pages_per_split * PAGE
    end = gl.minimum(first + pages_per_split * PAGE, visible)
    q_offset = request * STRIDE_Q_B + query * STRIDE_Q_S
    row = (request * QLEN + query) * H
    if first >= end:
        head = gl.arange(0, 16, layout=gl.BlockedLayout([1], [64], [4], [0]))
        gl.store(
            LSE + (row + head) * SPLIT_BUCKET + short_split, -float("inf"), head < H
        )
        return
    cfg = AttentionConfig(
        BLOCK_H=16,
        BLOCK_N=128,
        NUM_KV_SPLITS=SPLIT_BUCKET,
        PAGE_SIZE=PAGE,
        HEAD_DIM_CKV=512,
        HEAD_DIM_KPE=64,
        KV_PE_OFFSET=512,
        WITHIN_2GB=WITHIN_2GB,
        NUM_XCDS=1,
        NHEAD=H,
        REGIME="bh16bn128",
        IS_FP8_Q=True,
        RETURN_LSE=False,
        stride_q_nope_bs=0,
        stride_q_nope_h=STRIDE_Q_H,
        stride_q_pe_bs=0,
        stride_q_pe_h=STRIDE_Q_H,
        stride_kv_c_bs=STRIDE_KV,
        stride_k_pe_bs=STRIDE_KV,
        stride_req_to_tokens_bs=0,
        stride_o_b=0,
        stride_o_h=SPLIT_BUCKET * 512,
        stride_o_s=512,
        stride_mid_lse_b=0,
        stride_mid_lse_h=SPLIT_BUCKET,
        stride_mid_lse_s=1,
        stride_final_lse_b=0,
        stride_final_lse_h=0,
    )
    zero = gl.cast(0, gl.int32)
    program = AttentionProgram(
        cfg=cfg,
        Q_nope=Q + q_offset,
        Q_pe=Q + q_offset + cfg.HEAD_DIM_CKV,
        Kv_c_cache=KV,
        K_pe_cache=KV,
        Req_to_tokens=Pages,
        Out=Partials + row * SPLIT_BUCKET * cfg.HEAD_DIM_CKV,
        kv_scale=gl.cast(1.0, gl.float32),
        qk_scale=sm_scale,
        cur_batch=zero,
        cur_head_id=zero,
        split_kv_id=short_split,
        batch_page_start=request * stride_page,
        split_kv_start=first,
        split_kv_end=end,
        num_iter=gl.cdiv(end - first, cfg.BLOCK_N),
    )
    _mla_decode_program(program, LSE + row * SPLIT_BUCKET, None)


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=["stride_page", "splits", "wide_splits", "reuse_min_history"],
)
def gluon_mla_decode_fp8_query_blocks_gfx950(
    Q,
    KV,
    Pages,
    Lengths,
    Partials,
    LSE,
    STRIDE_Q_B: gl.constexpr,
    STRIDE_Q_S: gl.constexpr,
    STRIDE_Q_H: gl.constexpr,
    STRIDE_KV: gl.constexpr,
    stride_page,
    H: gl.constexpr,
    QLEN: gl.constexpr,
    PAGE: gl.constexpr,
    splits,
    wide_splits,
    reuse_min_history,
    SPLIT_BUCKET: gl.constexpr,
    WIDE_SPLIT_BUCKET: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    sm_scale,
):
    request = gl.program_id(0)
    group = gl.program_id(1)
    split = gl.program_id(2)
    length = gl.load(Lengths + request)
    if _supports_wide_query_block(H, QLEN, PAGE):
        if _use_wide_query_block(length, wide_splits, H, QLEN):
            if split < wide_splits:
                _decode_query_block(
                    Q,
                    KV,
                    Pages,
                    Partials,
                    LSE,
                    length,
                    0,
                    split,
                    STRIDE_Q_B,
                    STRIDE_Q_S,
                    STRIDE_Q_H,
                    STRIDE_KV,
                    stride_page,
                    H,
                    QLEN,
                    PAGE,
                    wide_splits,
                    WIDE_SPLIT_BUCKET,
                    sm_scale,
                    128,
                )
            return
        group = split % 2
        split = split // 2
        if split >= splits:
            return
        # Keep narrow rows contiguous within each request's reserved storage.
        offset = request * QLEN * H * (WIDE_SPLIT_BUCKET - SPLIT_BUCKET)
        Partials += offset * 512
        LSE += offset
    groups: gl.constexpr = gl.cdiv(QLEN * H, _QUERY_ROW_TILE)
    if H <= _QUERY_HEAD_TILE and PAGE == 64 and groups * SPLIT_BUCKET >= QLEN:
        if length < reuse_min_history and groups * splits >= QLEN:
            _decode_single_query(
                Q,
                KV,
                Pages,
                Partials,
                LSE,
                length,
                group,
                split,
                STRIDE_Q_B,
                STRIDE_Q_S,
                STRIDE_Q_H,
                STRIDE_KV,
                stride_page,
                H,
                QLEN,
                PAGE,
                splits,
                SPLIT_BUCKET,
                WITHIN_2GB,
                sm_scale,
            )
            return
    _decode_query_block(
        Q,
        KV,
        Pages,
        Partials,
        LSE,
        length,
        group,
        split,
        STRIDE_Q_B,
        STRIDE_Q_S,
        STRIDE_Q_H,
        STRIDE_KV,
        stride_page,
        H,
        QLEN,
        PAGE,
        splits,
        SPLIT_BUCKET,
        sm_scale,
        64,
    )


@gluon.jit
def _decode_query_block(
    Q,
    KV,
    Pages,
    Partials,
    LSE,
    length,
    group,
    split,
    STRIDE_Q_B: gl.constexpr,
    STRIDE_Q_S: gl.constexpr,
    STRIDE_Q_H: gl.constexpr,
    STRIDE_KV: gl.constexpr,
    stride_page,
    H: gl.constexpr,
    QLEN: gl.constexpr,
    PAGE: gl.constexpr,
    splits,
    SPLIT_BUCKET: gl.constexpr,
    sm_scale,
    M: gl.constexpr,
):
    N: gl.constexpr = 64
    request = gl.program_id(0)
    pages_per_split = gl.cdiv(gl.cdiv(length, PAGE), splits)
    first = split * pages_per_split * PAGE
    end = gl.minimum((split + 1) * pages_per_split * PAGE, length)

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[4, 1] if M == 128 else [2, 2],
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mfma, 16)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mfma, 16)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, mfma, 16)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, mfma, 16)
    # K and V share [latent, token] storage. Loading with v_layout's axes
    # swapped lets V be viewed as [token, latent] without moving data.
    value_linear: gl.constexpr = gl.to_linear_layout(v_layout, [N, _QUERY_VALUE_TILE])
    value_layout: gl.constexpr = _transpose_operand_layout(value_linear)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    m = gl.arange(0, M, layout=gl.SliceLayout(1, load_layout))
    query = (group * M + m) // H
    head = (group * M + m) % H
    d = gl.arange(0, 512, layout=gl.SliceLayout(0, load_layout))
    dpe = gl.arange(0, 64, layout=gl.SliceLayout(0, load_layout))
    q_offset = request * STRIDE_Q_B + query * STRIDE_Q_S + head * STRIDE_Q_H
    q_mask = (head < H) & (query < QLEN)
    q = gl.load(Q + q_offset[:, None] + d[None, :], q_mask[:, None], 0.0)
    qpe = gl.load(Q + q_offset[:, None] + 512 + dpe[None, :], q_mask[:, None], 0.0)
    q = gl.convert_layout(q, a_layout)
    qpe = gl.convert_layout(qpe, a_layout)

    m_acc = gl.arange(0, M, layout=gl.SliceLayout(1, mfma))
    query_acc = (group * M + m_acc) // H
    head_acc = (group * M + m_acc) % H
    visible = length - QLEN + query_acc + 1
    maximum = gl.full([M], -float("inf"), gl.float32, gl.SliceLayout(1, mfma))
    if M == 128:
        denominator = gl.zeros([M], gl.float32, gl.SliceLayout(1, mfma))
    else:
        # Each 32-key half of a row belongs to a different wave. Keep two sums
        # per row to avoid exchanging partial sums on every iteration. Both use
        # the same maximum and alpha, so they can be added after the loop.
        denominator = gl.sum(
            gl.reshape(gl.zeros([M, N], gl.float32, mfma), [M, 2, N // 2]), 2
        )
        denominator_layout: gl.constexpr = denominator.type.layout
    acc0 = gl.zeros([M, _QUERY_VALUE_TILE], gl.float32, mfma)
    acc1 = gl.zeros([M, _QUERY_VALUE_TILE], gl.float32, mfma)
    acc2 = gl.zeros([M, _QUERY_VALUE_TILE], gl.float32, mfma)
    acc3 = gl.zeros([M, _QUERY_VALUE_TILE], gl.float32, mfma)
    token = gl.arange(0, N, layout=gl.SliceLayout(0, mfma))

    shared: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=8, order=[0, 1]
    )
    shared_pe: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=4, order=[0, 1]
    )
    kv_load: gl.constexpr = gl.BlockedLayout([16, 1], [32, 2], [1, 4], [0, 1])
    pe_load: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, 4], [0, 1])
    buffers = gl.allocate_shared_memory(Q.type.element_ty, [2, 512, N], shared)
    pe_buffers = gl.allocate_shared_memory(Q.type.element_ty, [2, 64, N], shared_pe)
    if M == 64:
        maximum_buffer = gl.allocate_shared_memory(
            gl.float32, [M, 2], gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
        )
        maximum_layout: gl.constexpr = _replicated_pair_layout(
            gl.to_linear_layout(gl.SliceLayout(1, mfma), [M])
        )
    # DMA writes in lane order; offsets already account for the LDS swizzle.
    linear: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    offsets, pe_offsets = _make_kv_offsets(
        STRIDE_KV, N, kv_load, pe_load, shared, shared_pe
    )
    page = _load_page(Pages, request, first, end, stride_page, PAGE)
    _issue_load_kv(
        buffers.index(0),
        pe_buffers.index(0),
        KV,
        page,
        first,
        end,
        offsets,
        pe_offsets,
        STRIDE_KV,
        PAGE,
        N,
        kv_load,
        pe_load,
        linear,
    )
    # Prefetch the next page ID while the current KV copy is in flight.
    if M == 64:
        page = _load_page(Pages, request, first + N, end, stride_page, PAGE)
    current = 0
    for start in range(first, end, N):
        gl.amd.cdna4.async_copy.wait_group(0)
        other = 1 - current
        if start + N < end:
            if M == 128:
                page = _load_page(Pages, request, start + N, end, stride_page, PAGE)
            _issue_load_kv(
                buffers.index(other),
                pe_buffers.index(other),
                KV,
                page,
                start + N,
                end,
                offsets,
                pe_offsets,
                STRIDE_KV,
                PAGE,
                N,
                kv_load,
                pe_load,
                linear,
            )
            if M == 64:
                page = _load_page(Pages, request, start + 2 * N, end, stride_page, PAGE)
        key = gl.amd.cdna4.async_copy.load_shared_relaxed(
            buffers.index(current), b_layout
        )
        key_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(
            pe_buffers.index(current), b_layout
        )
        logits = _mfma_unscaled_fp8(q, key, gl.zeros([M, N], gl.float32, mfma))
        logits = _mfma_unscaled_fp8(qpe, key_pe, logits) * sm_scale
        allowed = (start + token[None, :] < end) & (
            start + token[None, :] < visible[:, None]
        )
        logits = gl.where(allowed, logits, -float("inf"))
        if M == 128:
            row_maximum = gl.max(logits, 1)
        else:
            local_maximum = gl.max(gl.reshape(logits, [M, 2, N // 2]), 2)
            maximum_buffer.store(local_maximum)
            row_maximum = gl.convert_layout(
                gl.max(maximum_buffer.load(maximum_layout), 1), gl.SliceLayout(1, mfma)
            )
        next_max = gl.maximum(maximum, row_maximum)
        safe_max = gl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = gl.exp2((maximum - safe_max) * _INV_LN2)
        probability = gl.exp2((logits - safe_max[:, None]) * _INV_LN2)
        if M == 128:
            denominator = denominator * alpha + gl.sum(probability, 1)
        else:
            partial_probability = gl.reshape(probability, [M, 2, N // 2])
            denominator = denominator * gl.convert_layout(
                alpha, gl.SliceLayout(1, denominator_layout)
            )[:, None] + gl.sum(partial_probability, 2)
        maximum = next_max
        probability = gl.convert_layout(
            (probability * _FP8_PROBABILITY_SCALE).to(Q.type.element_ty), p_layout
        )
        acc0 = _compute_pv(
            buffers.index(current), probability, acc0, alpha, 0, value_layout, v_layout
        )
        acc1 = _compute_pv(
            buffers.index(current), probability, acc1, alpha, 1, value_layout, v_layout
        )
        acc2 = _compute_pv(
            buffers.index(current), probability, acc2, alpha, 2, value_layout, v_layout
        )
        acc3 = _compute_pv(
            buffers.index(current), probability, acc3, alpha, 3, value_layout, v_layout
        )
        current = other

    row = (request * QLEN + query_acc) * H + head_acc
    valid = (head_acc < H) & (query_acc < QLEN)
    partial_row = row * SPLIT_BUCKET + split
    if M == 64:
        denominator = gl.convert_layout(gl.sum(denominator, 1), gl.SliceLayout(1, mfma))
    reciprocal = gl.where(
        denominator > 0, 1.0 / (denominator * _FP8_PROBABILITY_SCALE), 0.0
    )
    partial_out = Partials + partial_row[:, None] * 512
    _store_partial(partial_out, acc0, reciprocal, valid, 0)
    _store_partial(partial_out, acc1, reciprocal, valid, 1)
    _store_partial(partial_out, acc2, reciprocal, valid, 2)
    _store_partial(partial_out, acc3, reciprocal, valid, 3)
    gl.store(LSE + partial_row, maximum + gl.log(denominator), valid)


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=["splits", "wide_splits", "reuse_min_history"],
)
def gluon_mla_decode_fp8_query_blocks_reduce_gfx950(
    Partials,
    LSE,
    Lengths,
    Output,
    FinalLSE,
    H: gl.constexpr,
    QLEN: gl.constexpr,
    PAGE: gl.constexpr,
    splits,
    wide_splits,
    reuse_min_history,
    SPLIT_BUCKET: gl.constexpr,
    WIDE_SPLIT_BUCKET: gl.constexpr,
    RETURN_LSE: gl.constexpr,
):
    row = gl.program_id(0)
    column = gl.program_id(1)
    length = gl.load(Lengths + row // (QLEN * H))
    if _supports_wide_query_block(H, QLEN, PAGE):
        if _use_wide_query_block(length, wide_splits, H, QLEN):
            _reduce_splits(
                Partials,
                LSE,
                Output,
                FinalLSE,
                row,
                column,
                splits=wide_splits,
                SPLIT_STRIDE=WIDE_SPLIT_BUCKET,
                REDUCE_SPLITS=WIDE_SPLIT_BUCKET,
                VALUE_TILE=_QUERY_VALUE_TILE,
                RETURN_LSE=RETURN_LSE,
            )
            return
        request = row // (QLEN * H)
        offset = request * QLEN * H * (WIDE_SPLIT_BUCKET - SPLIT_BUCKET)
        Partials += offset * 512
        LSE += offset
    groups: gl.constexpr = gl.cdiv(QLEN * H, _QUERY_ROW_TILE)
    if H <= _QUERY_HEAD_TILE and PAGE == 64 and groups * SPLIT_BUCKET >= QLEN:
        if length < reuse_min_history and groups * splits >= QLEN:
            if column == 0:
                _reduce_splits(
                    Partials,
                    LSE,
                    Output,
                    FinalLSE,
                    row,
                    column,
                    splits=groups * splits // QLEN,
                    SPLIT_STRIDE=SPLIT_BUCKET,
                    REDUCE_SPLITS=_single_query_split_bucket(H, QLEN, SPLIT_BUCKET),
                    VALUE_TILE=512,
                    RETURN_LSE=RETURN_LSE,
                )
            return
    _reduce_splits(
        Partials,
        LSE,
        Output,
        FinalLSE,
        row,
        column,
        splits=splits,
        SPLIT_STRIDE=SPLIT_BUCKET,
        REDUCE_SPLITS=SPLIT_BUCKET,
        VALUE_TILE=_QUERY_VALUE_TILE,
        RETURN_LSE=RETURN_LSE,
    )


@gluon.jit
def _reduce_splits(
    Partials,
    LSE,
    Output,
    FinalLSE,
    row,
    column,
    splits,
    SPLIT_STRIDE: gl.constexpr,
    REDUCE_SPLITS: gl.constexpr,
    VALUE_TILE: gl.constexpr,
    RETURN_LSE: gl.constexpr,
):
    if VALUE_TILE == 512 and REDUCE_SPLITS <= 16:
        # Each thread merges its own columns without exchanging partials.
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 4], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    s = gl.arange(0, REDUCE_SPLITS, layout=gl.SliceLayout(1, layout))
    d = column * VALUE_TILE + gl.arange(0, VALUE_TILE, layout=gl.SliceLayout(0, layout))
    lse = gl.load(LSE + row * SPLIT_STRIDE + s, s < splits, -float("inf"))
    valid = lse != -float("inf")
    maximum = gl.max(lse, 0)
    safe_max = gl.where(maximum == -float("inf"), 0.0, maximum)
    weights = gl.exp2((lse - safe_max) * _INV_LN2)
    denom = gl.sum(weights, 0)
    values = gl.load(
        Partials + (row * SPLIT_STRIDE + s[:, None]) * 512 + d[None, :],
        valid[:, None],
        0.0,
    ).to(gl.float32)
    result = gl.sum(values * weights[:, None], 0)
    gl.store(Output + row * 512 + d, gl.where(denom > 0, result / denom, 0.0))

    if RETURN_LSE:
        if column == 0:
            gl.store(FinalLSE + row, maximum + gl.log(denom))


_WAVE_WORKGROUPS = 256

_NUM_XCDS = 8

_DEFAULT_SMALL_BATCH_TARGET_WORKGROUPS = {
    1: 256,
    2: 128,
    4: 256,
}

_SMALL_BATCH_REGIMES = frozenset({"bh16-multiblock", "bh64-small"})

_MLA_DECODE_REGIMES = frozenset(
    {"bh16bn128", "bh16bn64", "bh64", "bh16-multiblock", "bh64-small"}
)


def _select_num_kv_splits_bh16bn64(
    *, batch: int, max_seqlen_k: int, block_n: int
) -> int:
    occupancy_cap = _WAVE_WORKGROUPS // batch
    blocks = (max_seqlen_k + block_n - 1) // block_n
    return max(1, min(occupancy_cap, blocks))


def _select_num_kv_splits_bh16bn128(
    *, batch: int, max_seqlen_k: int, block_n: int
) -> int:
    occupancy_cap = (_WAVE_WORKGROUPS // 2) // batch
    blocks = (max_seqlen_k + block_n - 1) // block_n
    splits = max(1, min(occupancy_cap, blocks))
    # K3 TP8 decode has one request and 12 local heads. For a server capped at
    # 16K context, 128 split partials make the 512-wide FP32 reducer dominate
    # the actual 4K attention scan. Sixteen splits keep 64 stage-1 waves in
    # flight and halve the full graph-replayed MLA kernel pair. Configurations
    # admitting longer contexts retain the original occupancy-oriented policy.
    if batch == 1 and max_seqlen_k <= 16_384:
        splits = min(splits, 16)
    return splits


def _select_num_kv_splits_bh16bn128_fp8(
    *, batch: int, max_seqlen_k: int, block_n: int
) -> int:
    blocks = (max_seqlen_k + block_n - 1) // block_n
    # For short contexts, the split-reduction launch costs more than the
    # additional stage-1 parallelism saves.
    if blocks <= 8:
        return 1

    # Keep at most four KV blocks in each split until the stage-1 grid reaches
    # the device-wide occupancy target. Large batches need fewer splits because
    # the batch dimension already contributes parallel workgroups.
    work_cap = (blocks + 3) // 4
    occupancy_cap = max(1, _WAVE_WORKGROUPS // batch)
    return max(1, min(occupancy_cap, work_cap))


def _select_projected_value_num_kv_splits(
    *, batch: int, max_seqlen_k: int, block_n: int
) -> int:
    if batch == 1:
        return 16
    blocks = (max_seqlen_k + block_n - 1) // block_n
    work_cap = max(4, (blocks + 7) // 8)
    occupancy_cap = max(1, (_WAVE_WORKGROUPS // 2) // batch)
    return max(1, min(64, occupancy_cap, work_cap))


def _select_num_kv_splits_bh64(
    *, batch: int, nhead: int, num_xcds: int, block_h: int
) -> int:
    base_grid = num_xcds * triton.cdiv(nhead, block_h) * (batch // num_xcds)
    return max(1, triton.next_power_of_2(triton.cdiv(_WAVE_WORKGROUPS, base_grid)))


def _select_num_kv_splits_small_batch(
    *,
    batch: int,
    nhead: int,
    block_h: int,
    max_seqlen_k: int,
    block_n: int,
    target_workgroups: int,
) -> int:
    head_blocks = (nhead + block_h - 1) // block_h
    base_grid = batch * head_blocks
    occupancy_splits = (target_workgroups + base_grid - 1) // base_grid
    blocks = (max_seqlen_k + block_n - 1) // block_n
    return max(1, min(occupancy_splits, blocks))


def _gluon_mla_decode_gfx950(
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
    regime: str,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    value_weight: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    projected_out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch one fixed absorbed MLA decode regime."""
    if regime not in _MLA_DECODE_REGIMES:
        raise ValueError(
            f"unsupported MLA decode regime {regime!r}; "
            f"expected one of {sorted(_MLA_DECODE_REGIMES)}"
        )
    if logit_cap != 0.0:
        raise NotImplementedError("gluon_mla_decode_gfx950 does not support logit_cap")
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
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    is_fp8_q = q.dtype in fp8_dtypes
    valid_bf16_q = q.dtype == torch.bfloat16 and kv_cache.dtype in (
        torch.bfloat16,
        *fp8_dtypes,
    )
    valid_fp8_q = is_fp8_q and kv_cache.dtype == q.dtype
    if not (valid_bf16_q or valid_fp8_q):
        raise NotImplementedError(
            "gluon MLA decode requires bf16 q with bf16/fp8 kv_cache or "
            "matching fp8 q and kv_cache, got "
            f"{q.dtype}/{kv_cache.dtype}"
        )
    is_fp8_kv = kv_cache.dtype != torch.bfloat16

    batch_size, _, nhead, _ = q.shape
    small_batch_h64 = regime in _SMALL_BATCH_REGIMES
    if regime == "bh16bn128":
        if not is_fp8_kv:
            raise NotImplementedError(
                "gluon MLA decode (bh16bn128) requires an FP8 kv_cache"
            )
        if not 1 <= nhead <= 16:
            raise NotImplementedError(
                "gluon MLA decode (bh16bn128) requires num_q_heads in [1, 16], "
                f"got {nhead}"
            )
        block_h = 16
        block_n = 128
        num_xcds = 1
    elif regime == "bh16bn64":
        if q.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
            raise NotImplementedError(
                "gluon MLA decode (bh16bn64) requires bf16 q and kv_cache"
            )
        if not 1 <= nhead <= 16:
            raise NotImplementedError(
                "gluon MLA decode (bh16bn64) requires num_q_heads in [1, 16], "
                f"got {nhead}"
            )
        block_h = 16
        block_n = 64
        num_xcds = 1
    elif regime == "bh64":
        if q.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
            raise NotImplementedError("gluon MLA bh64 requires bf16 q and kv_cache")
        if nhead not in (64, 128):
            raise NotImplementedError(
                "gluon MLA decode (bh64) requires num_q_heads in {64, 128}, "
                f"got {nhead}"
            )
        block_h = 64
        block_n = 64
        num_xcds = _NUM_XCDS
        if batch_size % 64 != 0:
            raise NotImplementedError(
                "gluon MLA decode (bh64) is large-batch only and requires "
                f"batch_size divisible by 64, got {batch_size}"
            )
    else:
        if q.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"gluon MLA decode ({regime}) requires bf16 q and kv_cache"
            )
        if nhead != 64 or batch_size not in _DEFAULT_SMALL_BATCH_TARGET_WORKGROUPS:
            raise NotImplementedError(
                f"gluon MLA decode ({regime}) requires num_q_heads=64 and "
                f"batch_size in {sorted(_DEFAULT_SMALL_BATCH_TARGET_WORKGROUPS)}, "
                f"got H={nhead}, B={batch_size}"
            )
        block_h = 16 if regime == "bh16-multiblock" else 64
        block_n = 64
        num_xcds = 1
    if kv_cache.dim() == 4:
        if kv_cache.shape[2] != 1 or kv_cache.shape[3] != qk_dim:
            raise ValueError(
                f"kv_cache must be [num_pages, page_size, 1, {qk_dim}], "
                f"got {tuple(kv_cache.shape)}"
            )
        page_size = kv_cache.shape[1]
    else:
        raise ValueError(f"kv_cache must be 4D, got {kv_cache.dim()}D")
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
            (batch_size, 1, nhead, kv_lora_rank),
            dtype=torch.bfloat16,
            device=q.device,
        )
    elif out.dtype != torch.bfloat16:
        raise ValueError(f"gluon MLA decode requires bf16 out, got {out.dtype}")
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

    if small_batch_h64:
        num_kv_splits = _select_num_kv_splits_small_batch(
            batch=batch_size,
            nhead=nhead,
            block_h=block_h,
            max_seqlen_k=max_seqlen_k,
            block_n=64,
            target_workgroups=_DEFAULT_SMALL_BATCH_TARGET_WORKGROUPS[batch_size],
        )
    elif regime == "bh64":
        num_kv_splits = _select_num_kv_splits_bh64(
            batch=batch_size, nhead=nhead, num_xcds=num_xcds, block_h=block_h
        )
    elif regime == "bh16bn128":
        if value_weight is not None:
            num_kv_splits = _select_projected_value_num_kv_splits(
                batch=batch_size,
                max_seqlen_k=max_seqlen_k,
                block_n=block_n,
            )
        else:
            split_selector = (
                _select_num_kv_splits_bh16bn128_fp8
                if is_fp8_q
                else _select_num_kv_splits_bh16bn128
            )
            num_kv_splits = split_selector(
                batch=batch_size,
                max_seqlen_k=max_seqlen_k,
                block_n=block_n,
            )
    else:
        num_kv_splits = _select_num_kv_splits_bh16bn64(
            batch=batch_size, max_seqlen_k=max_seqlen_k, block_n=block_n
        )

    def _grid(splits: int) -> tuple[int, ...]:
        if regime == "bh64":
            # 3-D XCD-aware: (NUM_XCDS, head_block, (batch // NUM_XCDS) * splits).
            return (
                num_xcds,
                (nhead + block_h - 1) // block_h,
                (batch_size // num_xcds) * splits,
            )
        if regime == "bh16bn64" or regime == "bh16bn128":
            return (batch_size, splits)
        return (batch_size, (nhead + block_h - 1) // block_h, splits)

    common_kwargs = {
        "BLOCK_H": block_h,
        "BLOCK_N": block_n,
        "NUM_KV_SPLITS": num_kv_splits,
        "PAGE_SIZE": page_size,
        "HEAD_DIM_CKV": kv_lora_rank,
        "HEAD_DIM_KPE": qk_rope_head_dim,
        "KV_PE_OFFSET": kv_lora_rank,
        "WITHIN_2GB": within_2gb,
        "NUM_XCDS": num_xcds,
        "NHEAD": nhead,
        "REGIME": regime,
        "IS_FP8_Q": is_fp8_q,
        "RETURN_LSE": return_lse,
        "num_warps": 4,
    }

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
            dtype=out.dtype,
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
        if value_weight is not None:
            if return_lse or projected_out is None:
                raise ValueError("MLA projected-value decode requires out")
            # Fuse softmax reduction, value projection, and optional gating.
            gluon_mla_reduce_project_value_gfx950(
                logits,
                mid_lse,
                cache_seqlens,
                value_weight,
                gate=gate,
                page_size=page_size,
                out=projected_out,
            )
        else:
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

    if value_weight is not None:
        return projected_out
    if return_lse:
        return out, final_lse.view(batch_size, 1, nhead)
    return out


def launch_gluon_mla_decode_bf16xbf16_gfx950_bh16bn64(*args, **kwargs):
    """Run the fixed BLOCK_H=16, (batch, split) BF16 MLA decode regime."""
    return _gluon_mla_decode_gfx950(
        *args,
        regime="bh16bn64",
        **kwargs,
    )


def launch_gluon_mla_decode_bf16xbf16_gfx950_bh64(*args, **kwargs):
    """Run the fixed BLOCK_H=64, XCD-aware BF16 MLA decode regime."""
    return _gluon_mla_decode_gfx950(
        *args,
        regime="bh64",
        **kwargs,
    )


def launch_gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock(*args, **kwargs):
    """Run the fixed BLOCK_H=16 small-batch BF16 MLA decode regime."""
    return _gluon_mla_decode_gfx950(
        *args,
        regime="bh16-multiblock",
        **kwargs,
    )


def launch_gluon_mla_decode_bf16xbf16_gfx950_bh64_small(*args, **kwargs):
    """Run the fixed BLOCK_H=64 small-batch BF16 MLA decode regime."""
    return _gluon_mla_decode_gfx950(
        *args,
        regime="bh64-small",
        **kwargs,
    )


def gluon_mla_decode_bf16xbf16_gfx950(
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
    """Dispatch BF16-Q/BF16-KV MLA decode to a fixed GFX950 regime."""
    if q.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        raise NotImplementedError(
            "gluon_mla_decode_bf16xbf16_gfx950 requires bf16 q and kv_cache"
        )
    if q.dim() != 4 or q.shape[1] != 1:
        raise ValueError(
            f"q must be [batch, 1, num_q_heads, R + rope], got {tuple(q.shape)}"
        )
    batch_size, _, nhead, _ = q.shape
    if 1 <= nhead <= 16:
        impl = launch_gluon_mla_decode_bf16xbf16_gfx950_bh16bn64
    elif nhead == 64 and batch_size == 1:
        impl = launch_gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock
    elif nhead == 64 and batch_size in (2, 4):
        impl = launch_gluon_mla_decode_bf16xbf16_gfx950_bh64_small
    elif nhead in (64, 128) and batch_size % 64 == 0:
        impl = launch_gluon_mla_decode_bf16xbf16_gfx950_bh64
    else:
        raise NotImplementedError(
            "gluon MLA decode supports H in [1, 16], H=64 with B in {1, 2, 4}, "
            "or H in {64, 128} with B divisible by 64; "
            f"got H={nhead}, B={batch_size}"
        )
    return impl(
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        max_seqlen_k,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        softmax_scale,
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
    )


def gluon_mla_decode_bf16xfp8_gfx950(
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
    """Run BF16-Q/FP8-KV MLA decode with the GFX950 ``bh16bn128`` regime."""
    if q.dtype != torch.bfloat16 or kv_cache.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise NotImplementedError(
            "gluon_mla_decode_bf16xfp8_gfx950 requires bf16 q and fp8 kv_cache"
        )
    return _gluon_mla_decode_gfx950(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        regime="bh16bn128",
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
    )


def gluon_mla_decode_fp8xfp8_gfx950(
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
    """Run native FP8-Q/FP8-KV MLA decode with BF16 output.

    Query and cache values use unscaled E4M3 semantics.
    """
    if q.dtype != torch.float8_e4m3fn or kv_cache.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "gluon_mla_decode_fp8xfp8_gfx950 requires float8_e4m3fn q and kv_cache"
        )
    if out is not None and out.dtype != torch.bfloat16:
        raise ValueError(f"native FP8 MLA requires bf16 out, got {out.dtype}")
    return _gluon_mla_decode_gfx950(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        regime="bh16bn128",
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
    )


def launch_gluon_mla_decode_projected_value_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    value_weight: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    out: torch.Tensor,
    logit_cap: float = 0.0,
) -> torch.Tensor:
    """Run batched native FP8 MLA with a projected-value epilogue."""
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if q.dtype not in fp8_dtypes or kv_cache.dtype != q.dtype:
        raise NotImplementedError("projected-value MLA requires matching fp8 q and kv")
    if (qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim) != (128, 512, 64):
        raise NotImplementedError(
            "projected-value MLA requires qk_nope/kv_lora/qk_rope dimensions "
            "(128, 512, 64)"
        )
    if value_weight.ndim != 3:
        raise ValueError("value_weight must be rank-3")
    batch, heads = q.shape[0], q.shape[2]
    value = value_weight.shape[2]
    expected_weight = (heads, kv_lora_rank, value)
    if tuple(value_weight.shape) != expected_weight:
        raise ValueError(f"value_weight must have shape {expected_weight}")
    expected_out = (batch, heads * value)
    if tuple(out.shape) != expected_out:
        raise ValueError(f"out must have shape {expected_out}")
    if gate is not None and gate.shape != out.shape:
        raise ValueError("gate and out must have matching shapes")
    if (
        out.dtype != torch.bfloat16
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != q.device
    ):
        raise ValueError("projected-value MLA requires contiguous colocated bf16 out")
    return _gluon_mla_decode_gfx950(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        regime="bh16bn128",
        logit_cap=logit_cap,
        value_weight=value_weight,
        gate=gate,
        projected_out=out,
    )


def launch_gluon_mla_decode_fp8_query_blocks_gfx950(
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
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Decode causal FP8 query blocks with one page-table row per request.

    Args:
        q: E4M3 queries shaped ``[requests, queries, heads, 576]`` with a
            contiguous last dimension. Supports 2 to 16 queries per request.
        kv_cache: Contiguous E4M3 cache shaped ``[pages, page_size, 1, 576]``;
            supported page sizes are 64, 128 and 256.
        page_table: Int32 physical page indices shaped ``[requests, pages]``
            with a contiguous last dimension.
        cache_seqlens: Contiguous int32 final KV lengths shaped ``[requests]``.
            Query ``j`` sees ``cache_seqlens[i] - queries + j + 1`` tokens.
        max_seqlen_k: Host-side KV-length upper bound used to choose the split
            count and scratch allocation. Actual attention limits come from
            ``cache_seqlens``.
        qk_nope_head_dim: Original non-RoPE query width. Unused because ``q``
            has already been projected into the latent space.
        kv_lora_rank: Latent width; must be 512.
        qk_rope_head_dim: Positional width; must be 64.
        softmax_scale: Scale applied to QK logits before softmax.
        logit_cap: Soft cap on attention logits; only ``0.0`` is supported.
        return_lse: Whether to also return natural-log log-sum-exp values.
        out: Optional contiguous BF16 destination shaped
            ``[requests, queries, heads, 512]`` on the input device.

    Returns:
        BF16 latent output shaped ``[requests, queries, heads, 512]``, or
        ``(output, lse)`` when ``return_lse`` is true. LSE is FP32 shaped
        ``[requests, queries, heads]``. Rows with no visible history have
        zero output and negative-infinity LSE.
    """
    if q.ndim != 4 or q.shape[2] <= 0 or q.shape[3] != 576:
        raise ValueError(
            "query-block MLA requires q [requests,queries,heads,576], "
            f"got {tuple(q.shape)}"
        )
    if not 2 <= q.shape[1] <= 16:
        raise NotImplementedError(
            f"query-block MLA supports 2..16 queries per request, got {q.shape[1]}"
        )
    if q.dtype != torch.float8_e4m3fn or kv_cache.dtype != q.dtype:
        raise NotImplementedError(
            "query-block MLA requires float8_e4m3fn q and KV, "
            f"got {q.dtype} and {kv_cache.dtype}"
        )
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        raise NotImplementedError(
            "query-block MLA requires latent/positional dimensions 512/64, "
            f"got {kv_lora_rank}/{qk_rope_head_dim}"
        )
    if logit_cap != 0.0:
        raise NotImplementedError(
            f"query-block MLA requires logit_cap=0.0, got {logit_cap}"
        )
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (1, 576):
        raise ValueError(
            "query-block MLA requires KV [pages,page_size,1,576], "
            f"got {tuple(kv_cache.shape)}"
        )
    if kv_cache.shape[1] not in (64, 128, 256):
        raise NotImplementedError(
            "query-block MLA supports page sizes 64, 128 and 256, "
            f"got {kv_cache.shape[1]}"
        )
    if not kv_cache.is_contiguous() or q.stride(-1) != 1:
        raise ValueError(
            "query-block MLA requires contiguous KV and query head dimensions, "
            f"got KV strides {kv_cache.stride()} and q strides {q.stride()}"
        )
    batch, width, heads, _ = q.shape
    if (
        page_table.ndim != 2
        or page_table.shape[0] != batch
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            "query-block MLA requires one unit-stride page-table row per request, "
            f"got shape {tuple(page_table.shape)} and strides {page_table.stride()}"
        )
    if cache_seqlens.shape != (batch,) or not cache_seqlens.is_contiguous():
        raise ValueError(
            "query-block MLA requires one contiguous cache length per request, "
            f"got shape {tuple(cache_seqlens.shape)} and strides {cache_seqlens.stride()}"
        )
    if page_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise ValueError(
            "query-block MLA metadata must be int32, "
            f"got {page_table.dtype} and {cache_seqlens.dtype}"
        )
    if not q.is_cuda or any(
        tensor.device != q.device for tensor in (kv_cache, page_table, cache_seqlens)
    ):
        raise ValueError(
            "query-block MLA inputs must share one GPU, "
            f"got q={q.device}, KV={kv_cache.device}, "
            f"page_table={page_table.device}, cache_seqlens={cache_seqlens.device}"
        )
    shape = (batch, width, heads, kv_lora_rank)
    if out is None:
        out = torch.empty(shape, device=q.device, dtype=torch.bfloat16)
    elif (
        out.shape != shape
        or out.dtype != torch.bfloat16
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"query-block MLA out must be contiguous BF16 {shape} on {q.device}, "
            f"got shape {tuple(out.shape)}, dtype {out.dtype}, device {out.device} "
            f"and strides {out.stride()}"
        )
    final_lse = (
        torch.empty((batch, width, heads), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    if not batch:
        return (out, final_lse) if return_lse else out

    page_size = kv_cache.shape[1]
    reuse_min_history = (
        (12288, 8193, 5377, 10241, 4609, 4096, 4096, 5121)[batch - 1]
        if batch <= 8
        else _KV_REUSE_MIN_HISTORY.value
    )
    query_groups = triton.cdiv(width * heads, _QUERY_ROW_TILE.value)
    splits = _select_num_kv_splits_bh16bn128_fp8(
        batch=batch * query_groups, max_seqlen_k=max_seqlen_k, block_n=64
    )
    wide_splits = splits
    grid = (batch, query_groups, splits)
    if _supports_wide_query_block(heads, width, page_size):
        wide_splits = _select_num_kv_splits_bh16bn128_fp8(
            batch=batch, max_seqlen_k=max_seqlen_k, block_n=64
        )
        grid = (batch, 1, max(query_groups * splits, wide_splits))
    split_bucket = triton.next_power_of_2(splits)
    wide_split_bucket = triton.next_power_of_2(wide_splits)
    partial_shape = (batch * width * heads, wide_split_bucket, kv_lora_rank)
    partials = torch.empty(partial_shape, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(partial_shape[:-1], device=q.device, dtype=torch.float32)
    gluon_mla_decode_fp8_query_blocks_gfx950[grid](
        q,
        kv_cache,
        page_table,
        cache_seqlens,
        partials,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(1),
        page_table.stride(0),
        heads,
        width,
        page_size,
        splits,
        wide_splits,
        reuse_min_history,
        split_bucket,
        wide_split_bucket,
        kv_cache.numel() * kv_cache.element_size() < 0x80000000,
        softmax_scale,
        num_warps=4,
    )
    gluon_mla_decode_fp8_query_blocks_reduce_gfx950[
        (batch * width * heads, kv_lora_rank // _QUERY_VALUE_TILE.value)
    ](
        partials,
        lse,
        cache_seqlens,
        out,
        final_lse,
        heads,
        width,
        page_size,
        splits,
        wide_splits,
        reuse_min_history,
        split_bucket,
        wide_split_bucket,
        return_lse,
        num_warps=4,
    )
    return (out, final_lse) if return_lse else out
