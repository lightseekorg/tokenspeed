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

"""Selected-slot DSA Gluon kernels for AMD GFX950."""

from __future__ import annotations

import functools
import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention._common import select_kv_splits

__all__ = [
    "gluon_dsa_decode_dcp_records_supported_gfx950",
    "gluon_dsa_decode_dcp_records_workspace_bytes_gfx950",
    "launch_gluon_dsa_decode_dcp_records_gfx950",
    "launch_gluon_dsa_decode_gfx950",
    "launch_gluon_dsa_prefill_gfx950",
]

_REGISTERED_TOPK_WIDTHS = (512, 1024, 2048, 2049, 2050, 2051)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_GFX950_COMPUTE_UNITS = 256
_GLM52_SINGLE_ROW_DECODE_SPLITS = 16


@gluon.constexpr_function
def _value_layout(
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    return gl.BlockedLayout([1, 1], [1, 64], [NUM_WARPS, 1], [1, 0])


@gluon.constexpr_function
def _value_output_layout(BLOCK_H: gl.constexpr):
    """Assign each lane two adjacent 16-column groups in ordinary registers."""

    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 16]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
        warp_bases=[[0, 32], [0, 64]],
        block_bases=[],
        shape=[BLOCK_H, 128],
    )


@gluon.jit
def _dense_score(
    q,
    kv,
    slots,
    valid,
    q_base,
    kv_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    layout: gl.constexpr,
):
    score = gl.full(
        [BLOCK_TOPK],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, layout),
    )
    for dim in gl.static_range(0, kv_lora_rank):
        q_val = gl.load(q + q_base + dim).to(gl.float32)
        k_val = gl.load(
            kv + slots * kv_dim + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * q_val
    for dim in gl.static_range(0, qk_rope_head_dim):
        q_val = gl.load(q + q_base + kv_lora_rank + dim).to(gl.float32)
        k_val = gl.load(
            kv + slots * kv_dim + kv_lora_rank + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * q_val
    return score


@gluon.jit
def _packed_score(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    slots,
    valid,
    q_base,
    row_bytes: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    layout: gl.constexpr,
):
    score = gl.full(
        [BLOCK_TOPK],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, layout),
    )
    for dim in gl.static_range(0, kv_lora_rank):
        q_val = gl.load(q + q_base + dim).to(gl.float32)
        k_val = gl.load(
            kv_fp8 + slots * row_bytes + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        k_scale = gl.load(
            kv_scale + (slots * row_bytes + kv_lora_rank + (dim // 128) * 4) // 4,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * k_scale * q_val
    rope_base = (slots * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4) // 2
    for dim in gl.static_range(0, qk_rope_head_dim):
        q_val = gl.load(q + q_base + kv_lora_rank + dim).to(gl.float32)
        k_val = gl.load(
            kv_rope + rope_base + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * q_val
    return score


@gluon.jit
def _dsa_score_4x128(
    q_smem: gl.shared_memory_descriptor,
    kv_smem: gl.shared_memory_descriptor,
    q_layout: gl.constexpr,
    k_layout: gl.constexpr,
    score_layout: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
):
    """Compute the 512-wide score product as four phase-local products."""

    phase: gl.constexpr = 128
    q0 = q_smem.slice(0, phase, 1).load(q_layout)
    k0 = kv_smem.slice(0, phase, 0).load(k_layout)
    scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=score_layout)
    scores = gl.amd.cdna4.mfma(q0, k0, scores)

    for phase_idx in gl.static_range(1, 4):
        q_phase = q_smem.slice(phase_idx * phase, phase, 1).load(q_layout)
        k_phase = kv_smem.slice(phase_idx * phase, phase, 0).load(k_layout)
        score_tile = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=score_layout)
        score_tile = gl.amd.cdna4.mfma(q_phase, k_phase, score_tile)
        scores += score_tile
    return scores


@gluon.jit
def _dsa_score_4x128_regs(
    q0,
    q1,
    q2,
    q3,
    kv_smem: gl.shared_memory_descriptor,
    k_layout: gl.constexpr,
    score_layout: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
):
    """The 512-wide score product with the query phases already in registers."""

    phase: gl.constexpr = 128
    scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=score_layout)
    scores = gl.amd.cdna4.mfma(q0, kv_smem.slice(0, phase, 0).load(k_layout), scores)
    score_tile = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=score_layout)
    score_tile = gl.amd.cdna4.mfma(
        q1, kv_smem.slice(phase, phase, 0).load(k_layout), score_tile
    )
    scores += score_tile
    score_tile = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=score_layout)
    score_tile = gl.amd.cdna4.mfma(
        q2, kv_smem.slice(2 * phase, phase, 0).load(k_layout), score_tile
    )
    scores += score_tile
    score_tile = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=score_layout)
    score_tile = gl.amd.cdna4.mfma(
        q3, kv_smem.slice(3 * phase, phase, 0).load(k_layout), score_tile
    )
    scores += score_tile
    return scores


@gluon.jit
def _dsa_value_4x128(
    probs_smem: gl.shared_memory_descriptor,
    kv_smem: gl.shared_memory_descriptor,
    acc0,
    acc1,
    acc2,
    acc3,
    alpha,
    probs,
    p_layout: gl.constexpr,
    v_layout: gl.constexpr,
    acc_layout: gl.constexpr,
    output_layout: gl.constexpr,
    BLOCK_H: gl.constexpr,
    smem_dtype: gl.constexpr,
):
    """Rescale the running outputs and add four phase-local value products."""

    phase: gl.constexpr = 128
    fp8_max: gl.constexpr = 448.0 if smem_dtype == gl.float8e4nv else 1.0
    alpha_out = gl.convert_layout(alpha, gl.SliceLayout(1, output_layout))
    acc0 *= alpha_out[:, None]
    acc1 *= alpha_out[:, None]
    acc2 *= alpha_out[:, None]
    acc3 *= alpha_out[:, None]
    probs_smem.store((probs * fp8_max).to(smem_dtype))
    probs_dot = probs_smem.load(p_layout)

    v0 = kv_smem.slice(0, phase, 0).permute([1, 0]).load(v_layout)
    value_tile = gl.zeros([BLOCK_H, phase], dtype=gl.float32, layout=acc_layout)
    value_tile = gl.amd.cdna4.mfma(probs_dot, v0, value_tile)
    value_tile = gl.convert_layout(value_tile, output_layout, assert_trivial=True)
    acc0 += value_tile * (1.0 / fp8_max)

    v1 = kv_smem.slice(phase, phase, 0).permute([1, 0]).load(v_layout)
    value_tile = gl.zeros([BLOCK_H, phase], dtype=gl.float32, layout=acc_layout)
    value_tile = gl.amd.cdna4.mfma(probs_dot, v1, value_tile)
    value_tile = gl.convert_layout(value_tile, output_layout, assert_trivial=True)
    acc1 += value_tile * (1.0 / fp8_max)

    v2 = kv_smem.slice(2 * phase, phase, 0).permute([1, 0]).load(v_layout)
    value_tile = gl.zeros([BLOCK_H, phase], dtype=gl.float32, layout=acc_layout)
    value_tile = gl.amd.cdna4.mfma(probs_dot, v2, value_tile)
    value_tile = gl.convert_layout(value_tile, output_layout, assert_trivial=True)
    acc2 += value_tile * (1.0 / fp8_max)

    v3 = kv_smem.slice(3 * phase, phase, 0).permute([1, 0]).load(v_layout)
    value_tile = gl.zeros([BLOCK_H, phase], dtype=gl.float32, layout=acc_layout)
    value_tile = gl.amd.cdna4.mfma(probs_dot, v3, value_tile)
    value_tile = gl.convert_layout(value_tile, output_layout, assert_trivial=True)
    acc3 += value_tile * (1.0 / fp8_max)
    return acc0, acc1, acc2, acc3


@gluon.jit
def _dsa_online_softmax(scores, valid, mask_h, m_i, l_i, scale):
    """Update the shared online softmax, including empty selected rows."""

    scores = gl.where(valid[None, :] & mask_h[:, None], scores * scale, -float("inf"))
    m_new = gl.maximum(m_i, gl.max(scores, axis=1))
    safe_m_new = gl.where(m_new > -float("inf"), m_new, 0.0)
    alpha = gl.exp(m_i - safe_m_new)
    probs = gl.exp(scores - safe_m_new[:, None])
    l_new = alpha * l_i + gl.sum(probs, axis=1)
    return probs, alpha, m_new, l_new


@gluon.jit
def _dsa_store_output(
    out,
    acc0,
    acc1,
    acc2,
    acc3,
    l_i,
    o_base,
    stride_o_h,
    hg_offset,
    num_heads,
    output_layout: gl.constexpr,
    BLOCK_H: gl.constexpr,
    stride_o_r,
    LOCAL_HEADS: gl.constexpr,
):
    """Normalize and store four output phases, keeping empty rows exactly zero.
    With ``LOCAL_HEADS`` head ``r * LOCAL_HEADS + h`` lands at rank ``r``'s
    local head ``h``, as DCP records hold them."""

    l_i_out = gl.convert_layout(l_i, gl.SliceLayout(1, output_layout))
    safe_l_i = gl.where(l_i_out > 0.0, l_i_out, 1.0)
    valid = l_i_out[:, None] > 0.0
    offs_h = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, output_layout))
    offs_v = gl.arange(0, 128, layout=gl.SliceLayout(0, output_layout))
    if LOCAL_HEADS:
        o_row = (
            o_base
            + (offs_h // LOCAL_HEADS)[:, None].to(tl.int64) * stride_o_r
            + (offs_h % LOCAL_HEADS)[:, None].to(tl.int64) * stride_o_h
        )
    else:
        o_row = o_base + offs_h[:, None].to(tl.int64) * stride_o_h
    mask = (offs_h < num_heads)[:, None]
    acc0 = gl.where(valid, acc0 / safe_l_i[:, None], 0.0)
    acc1 = gl.where(valid, acc1 / safe_l_i[:, None], 0.0)
    acc2 = gl.where(valid, acc2 / safe_l_i[:, None], 0.0)
    acc3 = gl.where(valid, acc3 / safe_l_i[:, None], 0.0)
    gl.amd.cdna4.buffer_store(
        stored_value=acc0.to(out.dtype.element_ty),
        ptr=out,
        offsets=(o_row + offs_v[None, :].to(tl.int64)).to(tl.int32),
        mask=mask,
    )
    gl.amd.cdna4.buffer_store(
        stored_value=acc1.to(out.dtype.element_ty),
        ptr=out,
        offsets=(o_row + (128 + offs_v[None, :]).to(tl.int64)).to(tl.int32),
        mask=mask,
    )
    gl.amd.cdna4.buffer_store(
        stored_value=acc2.to(out.dtype.element_ty),
        ptr=out,
        offsets=(o_row + (256 + offs_v[None, :]).to(tl.int64)).to(tl.int32),
        mask=mask,
    )
    gl.amd.cdna4.buffer_store(
        stored_value=acc3.to(out.dtype.element_ty),
        ptr=out,
        offsets=(o_row + (384 + offs_v[None, :]).to(tl.int64)).to(tl.int32),
        mask=mask,
    )


# The split count and LSE strides follow the batch; output strides stay 16-aligned.
@gluon.jit(
    do_not_specialize=(
        "stride_lse_t",
        "stride_lse_h",
        "stride_lse_s",
        "num_kv_splits",
        "stride_o_r",
        "stride_lse_r",
        "split_tiles",
    )
)
def _dsa_dense_mfma_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    partial_lse,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    # Head r * Q_LOCAL_HEADS + h of a rank-major query is at q[r, token, h];
    # a token-major query has Q_LOCAL_HEADS 0 and no rank stride. The query
    # strides stay specialized: the direct-to-LDS copy needs their 16-element
    # alignment, which every token count shares.
    stride_q_r: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_o_s: tl.int64,
    stride_lse_t: tl.int64,
    stride_lse_h: tl.int64,
    stride_lse_s: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    num_kv_splits: tl.int32,
    # A rank-major query writes DCP records: every split's output and LSE
    # land rank-major (these rank strides) in a record-shaped workspace whose
    # split 0 is the records, and a row splits only into as many parts as
    # its live tiles fill split_tiles each (at most num_kv_splits). A row of
    # one split stores its record itself; the reduction merges the rest.
    stride_o_r: tl.int64,
    stride_lse_r: tl.int64,
    split_tiles: tl.int32,
    TOPK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    D_V: gl.constexpr,
    D_ROPE: gl.constexpr,
    HAS_ROPE: gl.constexpr,
    CONVERT_INPUTS: gl.constexpr,
    NATIVE_FP8: gl.constexpr,
    NATIVE_E5M2: gl.constexpr,
    SPLIT_KV: gl.constexpr,
    STORE_LSE: gl.constexpr,
    Q_LOCAL_HEADS: gl.constexpr = 0,
):
    # CDNA4 exposes the FP8 MFMA at K=32. Keeping FP8 operands native halves
    # the score/value instruction count and avoids per-tile BF16 expansion.
    instr_k: gl.constexpr = 32 if NATIVE_FP8 else 16
    QK_K_WIDTH: gl.constexpr = 16 if NATIVE_FP8 else 8
    PV_K_WIDTH: gl.constexpr = 8 if NATIVE_FP8 else 4
    INPUT_VEC: gl.constexpr = 16 if NATIVE_FP8 else 8
    ROPE_LOAD_VEC: gl.constexpr = 4 if NATIVE_FP8 else 2
    SHARED_INTERVAL: gl.constexpr = 1024 if NATIVE_FP8 else 512
    SHARED_PADDING: gl.constexpr = 32 if NATIVE_FP8 else 16
    KV_STAGES: gl.constexpr = 1 if CONVERT_INPUTS else 2
    # Split selected keys across the four waves and retain that ownership
    # through softmax and the probability store.
    mfma_qk: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, instr_k],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    mfma_s: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, instr_k],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    softmax_layout: gl.constexpr = mfma_qk
    softmax_scalar_layout: gl.constexpr = gl.SliceLayout(1, softmax_layout)
    mfma_acc: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, instr_k],
        transposed=True,
        warps_per_cta=[1, 4],
        # Each wave owns two adjacent 16-column value-product tiles.
        tiles_per_warp=[1, 2],
    )

    _qlora_tpw_k: gl.constexpr = min(64, D_V // INPUT_VEC)
    _qlora_tpw_m: gl.constexpr = 64 // _qlora_tpw_k
    blk_qlora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, INPUT_VEC],
        threads_per_warp=[_qlora_tpw_m, _qlora_tpw_k],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    # This ordinary-register layout is linearly identical to the [1, 4]
    # value-product MFMA layout above. Lane[0:4] select the 16 heads,
    # lane[4:6] select four-column fragments, and each wave owns one disjoint
    # 32-column partition. Keeping this as a non-MFMA encoding makes the four
    # persistent output fragments ordinary VGPR values across the key loop.
    blk_out_phase: gl.constexpr = _value_output_layout(BLOCK_H)
    _qrope_tpw_k: gl.constexpr = min(64, D_ROPE // INPUT_VEC)
    _qrope_tpw_m: gl.constexpr = 64 // _qrope_tpw_k
    blk_qrope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, INPUT_VEC],
        threads_per_warp=[_qrope_tpw_m, _qrope_tpw_k],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    _klora_tpw_m: gl.constexpr = min(64, D_V // INPUT_VEC)
    _klora_tpw_n: gl.constexpr = 64 // _klora_tpw_m
    blk_klora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[INPUT_VEC, 1],
        threads_per_warp=[_klora_tpw_m, _klora_tpw_n],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    _krope_tpw_m: gl.constexpr = min(64, D_ROPE // ROPE_LOAD_VEC)
    _krope_tpw_n: gl.constexpr = 64 // _krope_tpw_m
    blk_krope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ROPE_LOAD_VEC, 1],
        threads_per_warp=[_krope_tpw_m, _krope_tpw_n],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    sh_qlora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[SHARED_INTERVAL, SHARED_PADDING]],
        [BLOCK_H, D_V],
        [1, 0],
    )
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=INPUT_VEC,
        per_phase=2,
        max_phase=8,
        order=[1, 0],
    )
    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[SHARED_INTERVAL, SHARED_PADDING]],
        [D_V, TILE_K],
        [0, 1],
    )
    sh_probs: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[SHARED_INTERVAL, SHARED_PADDING]],
        [BLOCK_H, TILE_K],
        [1, 0],
    )
    sh_topk: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=INPUT_VEC,
        per_phase=2,
        max_phase=8,
        order=[0, 1],
    )

    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_qk,
        k_width=QK_K_WIDTH,
    )
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_qk,
        k_width=QK_K_WIDTH,
    )
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_qk,
        k_width=QK_K_WIDTH,
    )
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_qk,
        k_width=QK_K_WIDTH,
    )
    dot_p_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_acc,
        k_width=PV_K_WIDTH,
    )
    dot_v_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_acc,
        k_width=PV_K_WIDTH,
    )

    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    if SPLIT_KV:
        split_idx = gl.program_id(axis=2)
    else:
        split_idx: gl.constexpr = 0
    hg_offset = hg_idx * BLOCK_H
    valid_len = gl.load(topk_lens + token_idx).to(tl.int32)
    total_tiles = gl.cdiv(
        gl.minimum(gl.maximum(valid_len, 0), TOPK),
        TILE_K,
    )
    if SPLIT_KV:
        if Q_LOCAL_HEADS:
            row_splits = gl.minimum(
                num_kv_splits, gl.maximum(gl.cdiv(total_tiles, split_tiles), 1)
            )
            # Before the query's copy to LDS: nothing may be in flight.
            if split_idx >= row_splits:
                return
        else:
            row_splits = num_kv_splits
        tile_start = (split_idx * total_tiles) // row_splits
        tile_end = ((split_idx + 1) * total_tiles) // row_splits
        split_has_tiles = tile_start < tile_end
        num_tiles = gl.maximum(tile_end - tile_start, 1)
    else:
        tile_start: gl.constexpr = 0
        split_has_tiles: gl.constexpr = True
        num_tiles = gl.maximum(total_tiles, 1)

    offs_h_qlora = hg_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, blk_qlora),
    )
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads
    q_base = token_idx.to(tl.int64) * stride_q_t
    if Q_LOCAL_HEADS:
        q_rows_lora = (offs_h_qlora // Q_LOCAL_HEADS).to(tl.int64) * stride_q_r + (
            offs_h_qlora % Q_LOCAL_HEADS
        ).to(tl.int64) * stride_q_h
        q_offs_lora = q_base + q_rows_lora[:, None] + offs_v_qlora[None, :].to(tl.int64)
    else:
        q_offs_lora = (
            q_base
            + offs_h_qlora[:, None].to(tl.int64) * stride_q_h
            + offs_v_qlora[None, :].to(tl.int64)
        )

    offs_h_qrope = hg_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, blk_qrope),
    )
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_qrope = (offs_h_qrope < num_heads) & HAS_ROPE
    if Q_LOCAL_HEADS:
        q_rows_rope = (offs_h_qrope // Q_LOCAL_HEADS).to(tl.int64) * stride_q_r + (
            offs_h_qrope % Q_LOCAL_HEADS
        ).to(tl.int64) * stride_q_h
        q_offs_rope = (
            q_base + q_rows_rope[:, None] + (D_V + offs_r_qrope[None, :]).to(tl.int64)
        )
    else:
        q_offs_rope = (
            q_base
            + offs_h_qrope[:, None].to(tl.int64) * stride_q_h
            + (D_V + offs_r_qrope[None, :]).to(tl.int64)
        )

    smem_dtype: gl.constexpr = (
        gl.float8e5 if NATIVE_E5M2 else gl.float8e4nv if NATIVE_FP8 else gl.bfloat16
    )
    smem_qlora = gl.allocate_shared_memory(
        smem_dtype,
        [BLOCK_H, D_V],
        layout=sh_qlora,
    )
    if HAS_ROPE:
        smem_qrope = gl.allocate_shared_memory(
            smem_dtype,
            [BLOCK_H, D_ROPE],
            layout=sh_qrope,
        )
    if CONVERT_INPUTS:
        q_lora = gl.load(
            q + q_offs_lora,
            mask=mask_h_qlora[:, None],
            other=0.0,
        ).to(smem_dtype)
        smem_qlora.store(q_lora)
        if HAS_ROPE:
            q_rope = gl.load(
                q + q_offs_rope,
                mask=mask_h_qrope[:, None],
                other=0.0,
            ).to(smem_dtype)
            smem_qrope.store(q_rope)
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_qlora,
            ptr=q,
            offsets=q_offs_lora.to(tl.int32),
            mask=mask_h_qlora[:, None],
        )
        if HAS_ROPE:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                dest=smem_qrope,
                ptr=q,
                offsets=q_offs_rope.to(tl.int32),
                mask=mask_h_qrope[:, None],
            )
        gl.amd.cdna4.async_copy.commit_group()

    topk_base = token_idx.to(tl.int64) * stride_topk_t

    first_tile_base = tile_start * TILE_K
    local_offs_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    local_offs_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    local_offs_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, softmax_layout))
    offs_tile_klora = first_tile_base + local_offs_klora
    offs_tile_krope = first_tile_base + local_offs_krope
    offs_tile_mma = first_tile_base + local_offs_mma

    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    topk_pos_krope = gl.amd.cdna4.buffer_load(
        ptr=topk_indices,
        offsets=topk_base.to(tl.int32) + offs_tile_krope,
        mask=split_has_tiles & (offs_tile_krope < TOPK),
        other=-1,
    )
    topk_pos_mma = gl.amd.cdna4.buffer_load(
        ptr=topk_indices,
        offsets=topk_base.to(tl.int32) + offs_tile_mma,
        mask=split_has_tiles & (offs_tile_mma < TOPK),
        other=-1,
    )
    valid_mma = split_has_tiles & (offs_tile_mma < valid_len) & (topk_pos_mma >= 0)
    # Reuse the score-column index load for the KV gather.
    smem_topk = gl.allocate_shared_memory(gl.int32, [TILE_K], layout=sh_topk)
    # Keep one-tile rows free of an initial shared-index handoff. Longer
    # rows reuse the staged score-column indices inside the tile loop.
    topk_pos_klora = gl.amd.cdna4.buffer_load(
        ptr=topk_indices,
        offsets=topk_base.to(tl.int32) + offs_tile_klora,
        mask=split_has_tiles & (offs_tile_klora < TOPK),
        other=-1,
    )
    valid_klora = (
        split_has_tiles & (offs_tile_klora < valid_len) & (topk_pos_klora >= 0)
    )
    valid_krope = (
        split_has_tiles
        & (offs_tile_krope < valid_len)
        & (topk_pos_krope >= 0)
        & HAS_ROPE
    )
    safe_klora = gl.where(valid_klora, topk_pos_klora, 0)
    safe_krope = gl.where(valid_krope, topk_pos_krope, 0)

    if HAS_ROPE:
        smem_krope = gl.allocate_shared_memory(
            smem_dtype,
            [KV_STAGES, D_ROPE, TILE_K],
            layout=sh_krope,
        )
    smem_klora = gl.allocate_shared_memory(
        smem_dtype,
        [KV_STAGES, D_V, TILE_K],
        layout=sh_klora,
    )
    smem_probs = gl.allocate_shared_memory(
        smem_dtype,
        [BLOCK_H, TILE_K],
        layout=sh_probs,
    )

    klora_offs = safe_klora[None, :].to(tl.int64) * stride_kv_t + offs_v_klora[
        :, None
    ].to(tl.int64)
    krope_offs = safe_krope[None, :].to(tl.int64) * stride_kv_t + (
        D_V + offs_r_krope[:, None]
    ).to(tl.int64)
    if CONVERT_INPUTS:
        k_lora = gl.load(
            kv + klora_offs,
            mask=valid_klora[None, :],
            other=0.0,
        ).to(smem_dtype)
        smem_klora.index(0).store(k_lora)
        if HAS_ROPE:
            k_rope = gl.load(
                kv + krope_offs,
                mask=valid_krope[None, :],
                other=0.0,
            ).to(smem_dtype)
            smem_krope.index(0).store(k_rope)
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_klora.index(0),
            ptr=kv,
            offsets=klora_offs.to(tl.int32),
            mask=valid_klora[None, :],
        )
        if HAS_ROPE:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                dest=smem_krope.index(0),
                ptr=kv,
                offsets=krope_offs.to(tl.int32),
                mask=valid_krope[None, :],
            )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(1)
    if HAS_ROPE:
        q_rope_dot = smem_qrope.load(dot_qrope_a)

    m_i = gl.full(
        [BLOCK_H],
        float("-inf"),
        dtype=gl.float32,
        layout=softmax_scalar_layout,
    )
    l_i = gl.full(
        [BLOCK_H],
        0.0,
        dtype=gl.float32,
        layout=softmax_scalar_layout,
    )
    # Keep persistent outputs in ordinary VGPRs. Only the short-lived PV
    # scratch in _dsa_value_4x128 uses the MFMA accumulator layout.
    acc0 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    acc1 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    acc2 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    acc3 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    cur_buf = 0
    for tile_idx in range(num_tiles - 1):
        if not CONVERT_INPUTS:
            next_base = (tile_start + tile_idx + 1) * TILE_K
            next_offs_klora = next_base + local_offs_klora
            next_offs_krope = next_base + local_offs_krope
            next_offs_mma = next_base + local_offs_mma
            next_buf = 1 - cur_buf
            topk_pos_krope_next = gl.amd.cdna4.buffer_load(
                ptr=topk_indices,
                offsets=topk_base.to(tl.int32) + next_offs_krope,
                mask=next_offs_krope < TOPK,
                other=-1,
            )
            topk_pos_mma_next = gl.amd.cdna4.buffer_load(
                ptr=topk_indices,
                offsets=topk_base.to(tl.int32) + next_offs_mma,
                mask=next_offs_mma < TOPK,
                other=-1,
            )
            smem_topk.store(topk_pos_mma_next)
            topk_pos_klora_next = smem_topk.load(gl.SliceLayout(0, blk_klora))
            valid_klora_next = (next_offs_klora < valid_len) & (
                topk_pos_klora_next >= 0
            )
            valid_krope_next = (
                (next_offs_krope < valid_len) & (topk_pos_krope_next >= 0) & HAS_ROPE
            )
            valid_mma_next = (next_offs_mma < valid_len) & (topk_pos_mma_next >= 0)
            safe_klora_next = gl.where(valid_klora_next, topk_pos_klora_next, 0)
            safe_krope_next = gl.where(valid_krope_next, topk_pos_krope_next, 0)
            klora_offs_next = safe_klora_next[None, :].to(
                tl.int64
            ) * stride_kv_t + offs_v_klora[:, None].to(tl.int64)
            krope_offs_next = safe_krope_next[None, :].to(tl.int64) * stride_kv_t + (
                D_V + offs_r_krope[:, None]
            ).to(tl.int64)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                dest=smem_klora.index(next_buf),
                ptr=kv,
                offsets=klora_offs_next.to(tl.int32),
                mask=valid_klora_next[None, :],
            )
            if HAS_ROPE:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    dest=smem_krope.index(next_buf),
                    ptr=kv,
                    offsets=krope_offs_next.to(tl.int32),
                    mask=valid_krope_next[None, :],
                )
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.wait_group(1)

        klora_smem_cur = smem_klora.index(cur_buf)
        if HAS_ROPE:
            k_rope_t_dot = smem_krope.index(cur_buf).load(dot_krope_b)

        scores = _dsa_score_4x128(
            smem_qlora,
            klora_smem_cur,
            dot_qlora_a,
            dot_klora_b,
            mfma_qk,
            BLOCK_H,
            TILE_K,
        )
        if HAS_ROPE:
            scores = gl.amd.cdna4.mfma(q_rope_dot, k_rope_t_dot, scores)
        offs_h_mma = hg_offset + gl.arange(
            0,
            BLOCK_H,
            layout=softmax_scalar_layout,
        )
        mask_h_mma = offs_h_mma < num_heads
        probs, alpha, m_new, l_new = _dsa_online_softmax(
            scores, valid_mma, mask_h_mma, m_i, l_i, scale
        )

        acc0, acc1, acc2, acc3 = _dsa_value_4x128(
            smem_probs,
            klora_smem_cur,
            acc0,
            acc1,
            acc2,
            acc3,
            alpha,
            probs,
            dot_p_a,
            dot_v_b,
            mfma_acc,
            blk_out_phase,
            BLOCK_H,
            smem_dtype,
        )
        m_i = m_new
        l_i = l_new
        if CONVERT_INPUTS:
            # Mixed-type conversion is blocking, so loading into an alternate
            # buffer before this tile's matrix work cannot overlap anything.
            # Refill the consumed tile instead, keeping load operands out of
            # the live range of the score/value accumulators.
            next_base = (tile_start + tile_idx + 1) * TILE_K
            next_offs_klora = next_base + local_offs_klora
            next_offs_krope = next_base + local_offs_krope
            next_offs_mma = next_base + local_offs_mma
            topk_pos_krope_next = gl.amd.cdna4.buffer_load(
                ptr=topk_indices,
                offsets=topk_base.to(tl.int32) + next_offs_krope,
                mask=next_offs_krope < TOPK,
                other=-1,
            )
            topk_pos_mma_next = gl.amd.cdna4.buffer_load(
                ptr=topk_indices,
                offsets=topk_base.to(tl.int32) + next_offs_mma,
                mask=next_offs_mma < TOPK,
                other=-1,
            )
            valid_mma_next = (next_offs_mma < valid_len) & (topk_pos_mma_next >= 0)
            smem_topk.store(topk_pos_mma_next)
            topk_pos_klora_next = smem_topk.load(gl.SliceLayout(0, blk_klora))
            valid_klora_next = (next_offs_klora < valid_len) & (
                topk_pos_klora_next >= 0
            )
            valid_krope_next = (
                (next_offs_krope < valid_len) & (topk_pos_krope_next >= 0) & HAS_ROPE
            )
            safe_klora_next = gl.where(valid_klora_next, topk_pos_klora_next, 0)
            safe_krope_next = gl.where(valid_krope_next, topk_pos_krope_next, 0)
            klora_offs_next = safe_klora_next[None, :].to(
                tl.int64
            ) * stride_kv_t + offs_v_klora[:, None].to(tl.int64)
            krope_offs_next = safe_krope_next[None, :].to(tl.int64) * stride_kv_t + (
                D_V + offs_r_krope[:, None]
            ).to(tl.int64)

            k_lora_next = gl.load(
                kv + klora_offs_next,
                mask=valid_klora_next[None, :],
                other=0.0,
            ).to(smem_dtype)
            smem_klora.index(0).store(k_lora_next)
            if HAS_ROPE:
                k_rope_next = gl.load(
                    kv + krope_offs_next,
                    mask=valid_krope_next[None, :],
                    other=0.0,
                ).to(smem_dtype)
                smem_krope.index(0).store(k_rope_next)
        if not CONVERT_INPUTS:
            cur_buf = next_buf
        valid_mma = valid_mma_next

    if not CONVERT_INPUTS:
        gl.amd.cdna4.async_copy.wait_group(0)

    klora_smem_cur = smem_klora.index(cur_buf)
    if HAS_ROPE:
        k_rope_t_dot = smem_krope.index(cur_buf).load(dot_krope_b)

    scores = _dsa_score_4x128(
        smem_qlora,
        klora_smem_cur,
        dot_qlora_a,
        dot_klora_b,
        mfma_qk,
        BLOCK_H,
        TILE_K,
    )
    if HAS_ROPE:
        scores = gl.amd.cdna4.mfma(q_rope_dot, k_rope_t_dot, scores)
    offs_h_mma = hg_offset + gl.arange(
        0,
        BLOCK_H,
        layout=softmax_scalar_layout,
    )
    mask_h_mma = offs_h_mma < num_heads
    probs, alpha, m_new, l_new = _dsa_online_softmax(
        scores, valid_mma, mask_h_mma, m_i, l_i, scale
    )

    acc0, acc1, acc2, acc3 = _dsa_value_4x128(
        smem_probs,
        klora_smem_cur,
        acc0,
        acc1,
        acc2,
        acc3,
        alpha,
        probs,
        dot_p_a,
        dot_v_b,
        mfma_acc,
        blk_out_phase,
        BLOCK_H,
        smem_dtype,
    )
    if SPLIT_KV:
        o_base = (
            token_idx.to(tl.int64) * stride_o_t + split_idx.to(tl.int64) * stride_o_s
        )
    else:
        o_base = token_idx.to(tl.int64) * stride_o_t
    _dsa_store_output(
        out,
        acc0,
        acc1,
        acc2,
        acc3,
        l_new,
        o_base,
        stride_o_h,
        hg_offset,
        num_heads,
        blk_out_phase,
        BLOCK_H,
        stride_o_r,
        Q_LOCAL_HEADS,
    )
    if SPLIT_KV or STORE_LSE:
        lse = gl.where(l_new > 0.0, m_new + gl.log(l_new), -float("inf"))
        lse = gl.convert_layout(lse, gl.SliceLayout(1, mfma_s))
        offs_h_lse = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
        mask_h_lse = offs_h_lse < num_heads
        if Q_LOCAL_HEADS:
            lse_offsets = (
                token_idx.to(tl.int64) * stride_lse_t
                + (offs_h_lse // Q_LOCAL_HEADS).to(tl.int64) * stride_lse_r
                + (offs_h_lse % Q_LOCAL_HEADS).to(tl.int64) * stride_lse_h
            )
        else:
            lse_offsets = (
                token_idx.to(tl.int64) * stride_lse_t
                + offs_h_lse.to(tl.int64) * stride_lse_h
            )
        if SPLIT_KV:
            lse_offsets += split_idx.to(tl.int64) * stride_lse_s
        gl.amd.cdna4.buffer_store(
            stored_value=lse,
            ptr=partial_lse,
            offsets=lse_offsets.to(tl.int32),
            mask=mask_h_lse,
        )


@gluon.jit
def _load_selected(topk_indices, idx_row, offsets, TOPK: gl.constexpr):
    """A row's selected KV slots at ``offsets``, -1 past ``TOPK``."""
    return gl.amd.cdna4.buffer_load(
        ptr=topk_indices, offsets=idx_row + offsets, mask=offsets < TOPK, other=-1
    )


# Same contract as _dsa_dense_mfma_kv_kernel for native FP8 with 64 RoPE dims.
@gluon.jit(
    do_not_specialize=(
        "stride_lse_t",
        "stride_lse_h",
        "stride_lse_s",
        "num_kv_splits",
    )
)
def _dsa_sparse_prefill_fp8_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    partial_lse,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_o_s: tl.int64,
    stride_lse_t: tl.int64,
    stride_lse_h: tl.int64,
    stride_lse_s: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    num_kv_splits: tl.int32,
    TOPK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    D_V: gl.constexpr,
    D_ROPE: gl.constexpr,
    NATIVE_E5M2: gl.constexpr,
    SPLIT_KV: gl.constexpr,
    STORE_LSE: gl.constexpr,
):
    """Native-FP8 sparse MLA for prefill rows, one key tile at a time.

    The per-tile work is a short dependent chain (gather, score, cross-wave
    softmax, value product), so throughput comes from CTAs per CU rather than
    from prefetch depth: the query sits in registers, each tile loads its
    selected indices from global memory (an LDS copy of the row's selection
    would cost a CTA per CU), and one key tile buffer keeps four CTAs per CU
    at the launch's four waves per EU.
    """
    instr_k: gl.constexpr = 32
    QK_K_WIDTH: gl.constexpr = 16
    PV_K_WIDTH: gl.constexpr = 8
    INPUT_VEC: gl.constexpr = 16
    ROPE_LOAD_VEC: gl.constexpr = 4
    SHARED_INTERVAL: gl.constexpr = 1024
    SHARED_PADDING: gl.constexpr = 32
    LAST_TILE: gl.constexpr = (TOPK + TILE_K - 1) // TILE_K - 1
    mfma_qk: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, instr_k], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_s: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, instr_k], transposed=True, warps_per_cta=[4, 1]
    )
    softmax_layout: gl.constexpr = mfma_qk
    softmax_scalar_layout: gl.constexpr = gl.SliceLayout(1, softmax_layout)
    mfma_acc: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, instr_k],
        transposed=True,
        warps_per_cta=[1, 4],
        tiles_per_warp=[1, 2],
    )
    blk_out_phase: gl.constexpr = _value_output_layout(BLOCK_H)
    _qrope_tpw_k: gl.constexpr = min(64, D_ROPE // INPUT_VEC)
    blk_qrope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, INPUT_VEC],
        threads_per_warp=[64 // _qrope_tpw_k, _qrope_tpw_k],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    _klora_tpw_m: gl.constexpr = min(64, D_V // INPUT_VEC)
    blk_klora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[INPUT_VEC, 1],
        threads_per_warp=[_klora_tpw_m, 64 // _klora_tpw_m],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    _krope_tpw_m: gl.constexpr = min(64, D_ROPE // ROPE_LOAD_VEC)
    blk_krope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ROPE_LOAD_VEC, 1],
        threads_per_warp=[_krope_tpw_m, 64 // _krope_tpw_m],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=INPUT_VEC, per_phase=2, max_phase=8, order=[1, 0]
    )
    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[SHARED_INTERVAL, SHARED_PADDING]], [D_V, TILE_K], [0, 1]
    )
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=INPUT_VEC, per_phase=2, max_phase=8, order=[0, 1]
    )
    sh_probs: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[SHARED_INTERVAL, SHARED_PADDING]], [BLOCK_H, TILE_K], [1, 0]
    )
    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_qk, k_width=QK_K_WIDTH
    )
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_qk, k_width=QK_K_WIDTH
    )
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_qk, k_width=QK_K_WIDTH
    )
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_qk, k_width=QK_K_WIDTH
    )
    dot_p_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_acc, k_width=PV_K_WIDTH
    )
    dot_v_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_acc, k_width=PV_K_WIDTH
    )
    smem_dtype: gl.constexpr = gl.float8e5 if NATIVE_E5M2 else gl.float8e4nv

    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    if SPLIT_KV:
        split_idx = gl.program_id(axis=2)
    else:
        split_idx: gl.constexpr = 0
    hg_offset = hg_idx * BLOCK_H
    valid_len = gl.load(topk_lens + token_idx).to(tl.int32)
    q_base = token_idx.to(tl.int64) * stride_q_t

    offs_h_qrope = hg_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope)
    )
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    q_offs_rope = (
        q_base
        + offs_h_qrope[:, None].to(tl.int64) * stride_q_h
        + (D_V + offs_r_qrope[None, :]).to(tl.int64)
    )
    smem_qrope = gl.allocate_shared_memory(
        smem_dtype, [BLOCK_H, D_ROPE], layout=sh_qrope
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_qrope,
        ptr=q,
        offsets=q_offs_rope.to(tl.int32),
        mask=(offs_h_qrope < num_heads)[:, None],
    )
    gl.amd.cdna4.async_copy.commit_group()

    offs_h_qdot = hg_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, dot_qlora_a)
    )
    offs_v_qdot = gl.arange(0, 128, layout=gl.SliceLayout(0, dot_qlora_a))
    q_offs_dot = (
        q_base
        + offs_h_qdot[:, None].to(tl.int64) * stride_q_h
        + offs_v_qdot[None, :].to(tl.int64)
    ).to(tl.int32)
    mask_h_qdot = (offs_h_qdot < num_heads)[:, None]
    q_dot0 = gl.amd.cdna4.buffer_load(
        ptr=q, offsets=q_offs_dot, mask=mask_h_qdot, other=0.0
    )
    q_dot1 = gl.amd.cdna4.buffer_load(
        ptr=q, offsets=q_offs_dot + 128, mask=mask_h_qdot, other=0.0
    )
    q_dot2 = gl.amd.cdna4.buffer_load(
        ptr=q, offsets=q_offs_dot + 256, mask=mask_h_qdot, other=0.0
    )
    q_dot3 = gl.amd.cdna4.buffer_load(
        ptr=q, offsets=q_offs_dot + 384, mask=mask_h_qdot, other=0.0
    )

    idx_row = (token_idx.to(tl.int64) * stride_topk_t).to(tl.int32)

    total_tiles = gl.cdiv(gl.minimum(gl.maximum(valid_len, 0), TOPK), TILE_K)
    if SPLIT_KV:
        tile_start = (split_idx * total_tiles) // num_kv_splits
        tile_end = ((split_idx + 1) * total_tiles) // num_kv_splits
        num_tiles = gl.maximum(tile_end - tile_start, 1)
    else:
        tile_start: gl.constexpr = 0
        tile_end = total_tiles
        num_tiles = gl.maximum(total_tiles, 1)

    local_offs_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    local_offs_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    local_offs_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, softmax_layout))
    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))
    smem_krope = gl.allocate_shared_memory(
        smem_dtype, [D_ROPE, TILE_K], layout=sh_krope
    )
    smem_klora = gl.allocate_shared_memory(smem_dtype, [D_V, TILE_K], layout=sh_klora)
    smem_probs = gl.allocate_shared_memory(
        smem_dtype, [BLOCK_H, TILE_K], layout=sh_probs
    )

    gl.amd.cdna4.async_copy.wait_group(0)
    q_rope_dot = smem_qrope.load(dot_qrope_a)
    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=softmax_scalar_layout)
    mask_h_mma = offs_h_mma < num_heads
    m_i = gl.full(
        [BLOCK_H], float("-inf"), dtype=gl.float32, layout=softmax_scalar_layout
    )
    l_i = gl.full([BLOCK_H], 0.0, dtype=gl.float32, layout=softmax_scalar_layout)
    acc0 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    acc1 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    acc2 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    acc3 = gl.zeros([BLOCK_H, 128], dtype=gl.float32, layout=blk_out_phase)
    for i in range(num_tiles):
        tile = tile_start + i
        live = tile < tile_end
        # An empty split still runs one fully masked tile.
        idx_base = gl.minimum(tile, LAST_TILE) * TILE_K
        idx_k = _load_selected(topk_indices, idx_row, idx_base + local_offs_klora, TOPK)
        idx_r = _load_selected(topk_indices, idx_row, idx_base + local_offs_krope, TOPK)
        valid_k = live & ((tile * TILE_K + local_offs_klora) < valid_len) & (idx_k >= 0)
        valid_r = live & ((tile * TILE_K + local_offs_krope) < valid_len) & (idx_r >= 0)
        klora_offs = gl.where(valid_k, idx_k, 0)[None, :].to(
            tl.int64
        ) * stride_kv_t + offs_v_klora[:, None].to(tl.int64)
        krope_offs = gl.where(valid_r, idx_r, 0)[None, :].to(tl.int64) * stride_kv_t + (
            D_V + offs_r_krope[:, None]
        ).to(tl.int64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_klora,
            ptr=kv,
            offsets=klora_offs.to(tl.int32),
            mask=valid_k[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_krope,
            ptr=kv,
            offsets=krope_offs.to(tl.int32),
            mask=valid_r[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()
        idx_m = _load_selected(topk_indices, idx_row, idx_base + local_offs_mma, TOPK)
        valid_mma = live & ((tile * TILE_K + local_offs_mma) < valid_len) & (idx_m >= 0)
        gl.amd.cdna4.async_copy.wait_group(0)

        scores = _dsa_score_4x128_regs(
            q_dot0,
            q_dot1,
            q_dot2,
            q_dot3,
            smem_klora,
            dot_klora_b,
            mfma_qk,
            BLOCK_H,
            TILE_K,
        )
        scores = gl.amd.cdna4.mfma(q_rope_dot, smem_krope.load(dot_krope_b), scores)
        probs, alpha, m_i, l_i = _dsa_online_softmax(
            scores, valid_mma, mask_h_mma, m_i, l_i, scale
        )
        acc0, acc1, acc2, acc3 = _dsa_value_4x128(
            smem_probs,
            smem_klora,
            acc0,
            acc1,
            acc2,
            acc3,
            alpha,
            probs,
            dot_p_a,
            dot_v_b,
            mfma_acc,
            blk_out_phase,
            BLOCK_H,
            smem_dtype,
        )

    if SPLIT_KV:
        o_base = (
            token_idx.to(tl.int64) * stride_o_t + split_idx.to(tl.int64) * stride_o_s
        )
    else:
        o_base = token_idx.to(tl.int64) * stride_o_t
    _dsa_store_output(
        out,
        acc0,
        acc1,
        acc2,
        acc3,
        l_i,
        o_base,
        stride_o_h,
        hg_offset,
        num_heads,
        blk_out_phase,
        BLOCK_H,
        0,
        0,
    )
    if SPLIT_KV or STORE_LSE:
        lse = gl.where(l_i > 0.0, m_i + gl.log(l_i), -float("inf"))
        lse = gl.convert_layout(lse, gl.SliceLayout(1, mfma_s))
        offs_h_lse = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
        lse_offsets = (
            token_idx.to(tl.int64) * stride_lse_t
            + offs_h_lse.to(tl.int64) * stride_lse_h
        )
        if SPLIT_KV:
            lse_offsets += split_idx.to(tl.int64) * stride_lse_s
        gl.amd.cdna4.buffer_store(
            stored_value=lse,
            ptr=partial_lse,
            offsets=lse_offsets.to(tl.int32),
            mask=offs_h_lse < num_heads,
        )


@triton.jit(
    do_not_specialize=(
        "stride_lse_t",
        "stride_lse_h",
        "stride_lo_t",
        "stride_lo_h",
        "num_kv_splits",
    )
)
def _dsa_dense_mfma_reduce_kernel(
    partial_out,
    partial_lse,
    out,
    lse_out,
    stride_po_t,
    stride_po_h,
    stride_po_s,
    stride_lse_t,
    stride_lse_h,
    stride_lse_s,
    stride_o_t,
    stride_o_h,
    stride_lo_t,
    stride_lo_h,
    num_kv_splits,
    D_V: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_v = tl.arange(0, D_V)
    lse_base = token_idx * stride_lse_t + head_idx * stride_lse_h
    partial_base = token_idx * stride_po_t + head_idx * stride_po_h

    max_lse = -float("inf")
    for split_idx in tl.static_range(MAX_SPLITS):
        split_lse = tl.load(
            partial_lse + lse_base + split_idx * stride_lse_s,
            mask=split_idx < num_kv_splits,
            other=-float("inf"),
        )
        max_lse = tl.maximum(max_lse, split_lse)

    denom = 0.0
    acc = tl.zeros([D_V], dtype=tl.float32)
    safe_max_lse = tl.where(max_lse > -float("inf"), max_lse, 0.0)
    for split_idx in tl.static_range(MAX_SPLITS):
        live = split_idx < num_kv_splits
        split_lse = tl.load(
            partial_lse + lse_base + split_idx * stride_lse_s,
            mask=live,
            other=-float("inf"),
        )
        split_valid = split_lse > -float("inf")
        safe_split_lse = tl.where(split_valid, split_lse, 0.0)
        weight = tl.where(split_valid, tl.exp(safe_split_lse - safe_max_lse), 0.0)
        partial = tl.load(
            partial_out + partial_base + split_idx * D_V + offs_v,
            mask=live,
            other=0.0,
        ).to(tl.float32)
        acc += weight * partial
        denom += weight

    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    result = tl.where(denom > 0.0, acc / safe_denom, 0.0)
    tl.store(out + token_idx * stride_o_t + head_idx * stride_o_h + offs_v, result)
    if STORE_LSE:
        tl.store(
            lse_out + token_idx * stride_lo_t + head_idx * stride_lo_h,
            tl.where(denom > 0.0, safe_max_lse + tl.log(safe_denom), -float("inf")),
        )


@triton.jit(
    do_not_specialize=(
        "stride_s",
        "stride_r",
        "stride_t",
        "stride_h",
        "stride_lse_s",
        "stride_lse_r",
        "stride_lse_t",
        "stride_lse_h",
        "num_kv_splits",
        "topk_width",
        "tile_k",
        "split_tiles",
    )
)
def _dsa_dense_mfma_records_reduce_kernel(
    records,
    records_lse,
    topk_lens,
    # Split s of head r * LOCAL_HEADS + h sits at records[s, r, token, h]
    # (its LSE likewise); split 0 is the record, which the merge overwrites.
    stride_s,
    stride_r,
    stride_t,
    stride_h,
    stride_lse_s,
    stride_lse_r,
    stride_lse_t,
    stride_lse_h,
    num_heads,
    # A row ran as many splits as its live tiles fill split_tiles each, at
    # most num_kv_splits; the core stored a one-split row's record itself.
    num_kv_splits,
    topk_width,
    tile_k,
    split_tiles,
    LOCAL_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_V: tl.constexpr,
):
    token_idx = tl.program_id(0)
    valid_len = tl.load(topk_lens + token_idx).to(tl.int32)
    total_tiles = tl.cdiv(tl.minimum(tl.maximum(valid_len, 0), topk_width), tile_k)
    row_splits = tl.minimum(
        num_kv_splits, tl.maximum(tl.cdiv(total_tiles, split_tiles), 1)
    )
    if row_splits == 1:
        return
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    rank = offs_h // LOCAL_HEADS
    head = offs_h % LOCAL_HEADS
    base = rank * stride_r + token_idx * stride_t + head * stride_h
    lse_base = rank * stride_lse_r + token_idx * stride_lse_t + head * stride_lse_h
    offs_v = tl.arange(0, D_V)

    max_lse = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    for split_idx in range(row_splits):
        split_lse = tl.load(
            records_lse + lse_base + split_idx * stride_lse_s,
            mask=mask_h,
            other=-float("inf"),
        )
        max_lse = tl.maximum(max_lse, split_lse)

    denom = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)
    safe_max_lse = tl.where(max_lse > -float("inf"), max_lse, 0.0)
    for split_idx in range(row_splits):
        split_lse = tl.load(
            records_lse + lse_base + split_idx * stride_lse_s,
            mask=mask_h,
            other=-float("inf"),
        )
        split_valid = split_lse > -float("inf")
        safe_split_lse = tl.where(split_valid, split_lse, 0.0)
        weight = tl.where(split_valid, tl.exp(safe_split_lse - safe_max_lse), 0.0)
        partial = tl.load(
            records + base[:, None] + split_idx * stride_s + offs_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += weight[:, None] * partial
        denom += weight

    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    result = tl.where(denom[:, None] > 0.0, acc / safe_denom[:, None], 0.0)
    # The merge overwrites split 0 in place, and every warp reads its LSE: no
    # warp may store before all have loaded.
    tl.debug_barrier()
    tl.store(records + base[:, None] + offs_v[None, :], result, mask=mask_h[:, None])
    tl.store(
        records_lse + lse_base,
        tl.where(denom > 0.0, safe_max_lse + tl.log(safe_denom), -float("inf")),
        mask=mask_h,
    )


@gluon.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    lse,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    kv_dim: gl.constexpr,
    topk: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
    STORE_LSE: gl.constexpr,
):
    token = gl.program_id(0)
    head = gl.program_id(1)
    v_block = gl.program_id(2)
    layout: gl.constexpr = _value_layout(BLOCK_TOPK, BLOCK_V, gl.num_warps())
    topk_offsets = gl.arange(0, BLOCK_TOPK, layout=gl.SliceLayout(1, layout))
    v_offsets = v_block * BLOCK_V + gl.arange(
        0, BLOCK_V, layout=gl.SliceLayout(0, layout)
    )
    q_base = (token * num_heads + head) * head_dim
    valid_len = gl.load(topk_lens + token).to(gl.int32)
    max_score = gl.full((), -float("inf"), gl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _dense_score(
            q,
            kv,
            slots,
            valid,
            q_base,
            kv_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        max_score = gl.maximum(max_score, gl.max(score, axis=0))

    denom = gl.full((), 0.0, gl.float32)
    acc = gl.full(
        [BLOCK_V],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(0, layout),
    )
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _dense_score(
            q,
            kv,
            slots,
            valid,
            q_base,
            kv_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        probs = gl.exp(score - max_score)
        probs = gl.where(valid, probs, 0.0)
        denom += gl.sum(probs, axis=0)
        v_vals = gl.load(
            kv + slots[:, None] * kv_dim + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(gl.float32)
        acc += gl.sum(probs[:, None] * v_vals, axis=0)

    result = acc / denom
    result = gl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    gl.store(out + out_base + v_offsets, result, mask=v_mask)
    if STORE_LSE:
        if v_block == 0:
            row_lse = gl.where(denom > 0.0, max_score + gl.log(denom), -float("inf"))
            gl.store(lse + token * num_heads + head, row_lse)


@gluon.jit
def _dsa_packed_kv_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    lse,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    row_bytes: gl.constexpr,
    topk: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
    STORE_LSE: gl.constexpr,
):
    token = gl.program_id(0)
    head = gl.program_id(1)
    v_block = gl.program_id(2)
    layout: gl.constexpr = _value_layout(BLOCK_TOPK, BLOCK_V, gl.num_warps())
    topk_offsets = gl.arange(0, BLOCK_TOPK, layout=gl.SliceLayout(1, layout))
    v_offsets = v_block * BLOCK_V + gl.arange(
        0, BLOCK_V, layout=gl.SliceLayout(0, layout)
    )
    q_base = (token * num_heads + head) * head_dim
    valid_len = gl.load(topk_lens + token).to(gl.int32)
    max_score = gl.full((), -float("inf"), gl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _packed_score(
            q,
            kv_fp8,
            kv_scale,
            kv_rope,
            slots,
            valid,
            q_base,
            row_bytes,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        max_score = gl.maximum(max_score, gl.max(score, axis=0))

    denom = gl.full((), 0.0, gl.float32)
    acc = gl.full(
        [BLOCK_V],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(0, layout),
    )
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _packed_score(
            q,
            kv_fp8,
            kv_scale,
            kv_rope,
            slots,
            valid,
            q_base,
            row_bytes,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        probs = gl.exp(score - max_score)
        probs = gl.where(valid, probs, 0.0)
        denom += gl.sum(probs, axis=0)
        v_vals = gl.load(
            kv_fp8 + slots[:, None] * row_bytes + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(gl.float32)
        v_scale = gl.load(
            kv_scale
            + (
                slots[:, None] * row_bytes
                + kv_lora_rank
                + (v_offsets[None, :] // 128) * 4
            )
            // 4,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(gl.float32)
        acc += gl.sum(probs[:, None] * v_vals * v_scale, axis=0)

    result = acc / denom
    result = gl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    gl.store(out + out_base + v_offsets, result, mask=v_mask)
    if STORE_LSE:
        if v_block == 0:
            row_lse = gl.where(denom > 0.0, max_score + gl.log(denom), -float("inf"))
            gl.store(lse + token * num_heads + head, row_lse)


def _flatten_packed_kv_cache(packed_kv_cache: torch.Tensor) -> torch.Tensor:
    if packed_kv_cache.dim() == 2:
        return packed_kv_cache
    return packed_kv_cache.reshape(-1, packed_kv_cache.shape[-1])


def _flatten_dense_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache
    if kv_cache.dim() == 3:
        return kv_cache.squeeze(1)
    if kv_cache.shape[1] == 1:
        kv_cache = kv_cache.permute(0, 2, 1, 3)
    return kv_cache.reshape(-1, kv_cache.shape[-1])


def _flatten_query(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 3:
        return q
    return q.reshape(-1, q.shape[-2], q.shape[-1])


def _check_inputs(
    q: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
) -> None:
    if q.dtype != torch.bfloat16 and q.dtype not in _FP8_DTYPES:
        raise TypeError(f"Gluon DSA supports BF16/FP8 q, got {q.dtype}")
    if page_size != 64:
        raise ValueError(f"Gluon DSA supports page_size=64, got {page_size}")
    if qk_nope_head_dim not in (128, 192, 256):
        raise ValueError(
            "Gluon DSA supports qk_nope_head_dim in {128, 192, 256}, got "
            f"{qk_nope_head_dim}"
        )
    if kv_lora_rank not in (128, 512):
        raise ValueError(
            f"Gluon DSA supports kv_lora_rank in {{128, 512}}, got {kv_lora_rank}"
        )
    if qk_rope_head_dim not in (0, 64):
        raise ValueError(
            f"Gluon DSA supports qk_rope_head_dim in {{0, 64}}, got {qk_rope_head_dim}"
        )
    expected_head_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    if q.shape[-1] != expected_head_dim:
        raise ValueError(
            "q head dim must equal kv_lora_rank + qk_rope_head_dim, got "
            f"q={q.shape[-1]}, expected={expected_head_dim}"
        )
    if topk_slots.dtype != torch.int32 or topk_slots.dim() != 2:
        raise ValueError("topk_slots must be int32 with shape [tokens, topk]")
    if topk_lens is None:
        raise ValueError("Gluon DSA requires topk_lens for this milestone")
    if topk_lens.dtype != torch.int32 or topk_lens.shape != (topk_slots.shape[0],):
        raise ValueError("topk_lens must be int32 with shape [tokens]")


def _output_dtype(q: torch.Tensor) -> torch.dtype:
    return torch.bfloat16 if q.dtype in _FP8_DTYPES else q.dtype


def _shares_storage(tensor: torch.Tensor, *others: torch.Tensor | None) -> bool:
    storage_ptr = tensor.untyped_storage().data_ptr()
    return any(
        other is not None and other.untyped_storage().data_ptr() == storage_ptr
        for other in others
    )


def _select_num_kv_splits(
    *,
    num_tokens: int,
    num_heads: int,
    topk_width: int,
    max_seqlen_k: int,
    is_fp8: bool,
    native_fp8: bool = False,
    qk_rope_head_dim: int = 64,
    is_prefill: bool = False,
) -> int:
    work_tiles = max(1, (min(int(topk_width), int(max_seqlen_k)) + 31) // 32)
    if work_tiles <= 8:
        return 1
    if not is_fp8:
        # This tuning comes from GLM-5.3-Flash. Split sufficiently large selected-KV
        # rows across CTAs so the launch provides enough parallel work to fill the
        # GPU.
        # Pooled selection can append up to three tail slots to the configured
        # 2,048-token budget, so graph captures use widths from 2,048 through 2,051.
        if (
            int(qk_rope_head_dim) == 0
            and int(num_heads) == 16
            and 2048 <= int(topk_width) <= 2051
            and work_tiles >= 64
        ):
            if int(num_tokens) == 16:
                return 16
            if int(num_tokens) == 64:
                return 4
        return 1
    if (
        int(num_tokens) == 1
        and int(num_heads) == 16
        and int(topk_width) == 2048
        and work_tiles == 64
    ):
        return _GLM52_SINGLE_ROW_DECODE_SPLITS
    base_ctas = max(1, int(num_tokens) * triton.cdiv(int(num_heads), 16))
    if native_fp8 and int(num_heads) == 16:
        # Large prefill grids already fill the two-wave occupancy target. Splitting
        # them would only multiply the full-rank reduction workspace.
        if is_prefill and base_ctas >= _GFX950_COMPUTE_UNITS * 2:
            return 1
        # Native FP8 DSA CTAs carry enough work that the generic two-wave
        # occupancy target oversplits medium/large graph batches. These
        # thresholds are expressed only in kernel shape terms and keep short
        # selected contexts on the generic schedule.
        if work_tiles >= 64:
            if int(num_tokens) >= 64:
                return 4
            if int(num_tokens) >= 8:
                return 8
        elif work_tiles >= 32 and (int(num_tokens) == 1 or int(num_tokens) >= 8):
            return 4
    return select_kv_splits(
        base_ctas=base_ctas,
        num_pages=work_tiles,
        sm_count=_GFX950_COMPUTE_UNITS,
    )


# Each split of a KVP decode's records row runs at least this many of its live
# key tiles. A rank owns about an eighth of each row's top-k, which the
# launch's split count, sized for full rows, would spread a tile or less per
# split.
_RECORDS_MIN_SPLIT_TILES = 2
# Heads one program of the records reduction merges: few enough that a small
# launch's split rows spread over the CUs, enough that a large launch of
# one-split rows exits in few programs.
_RECORDS_REDUCE_HEADS = 2


def _records_split_tiles(
    *, tokens: int, heads: int, ranks: int, topk: int, tile_k: int, num_kv_splits: int
) -> int:
    """Live key tiles per split of a KVP decode's records rows.

    A row of a rank's typical share (``topk / ranks`` keys) splits only while
    the launch's rows and head groups fill under three quarters of the CUs,
    into as many parts as reach that; longer rows split proportionally more,
    and a full-width row into all ``num_kv_splits``. Splitting a launch that
    already fills the CUs only adds the reduction: on MI350X, 32 to 64 rows
    of 96 heads and 256 keys take 20-25 us unsplit against 29-37 us in two
    splits.
    """
    typical_tiles = triton.cdiv(triton.cdiv(topk, ranks), tile_k)
    base_ctas = tokens * triton.cdiv(heads, 16)
    splits = max(1, _GFX950_COMPUTE_UNITS * 3 // 4 // base_ctas)
    full_row = triton.cdiv(topk, tile_k) // num_kv_splits
    return max(
        _RECORDS_MIN_SPLIT_TILES,
        min(triton.cdiv(typical_tiles, splits), full_row),
    )


def _trim_topk_slots_for_context(
    topk_slots: torch.Tensor,
    max_seqlen_k: int,
) -> torch.Tensor:
    topk = int(topk_slots.shape[1])
    effective_topk = topk
    # Keep launch shapes on the widths registered by tokenspeed-kernel.
    for supported_topk in _REGISTERED_TOPK_WIDTHS:
        if max_seqlen_k <= supported_topk <= topk:
            effective_topk = supported_topk
            break
    if effective_topk < int(topk_slots.shape[1]):
        return topk_slots[:, :effective_topk].contiguous()
    return topk_slots


def _use_dense_mfma_core(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> bool:
    """Select phase-local products for the supported dense matrix inputs."""

    return (
        q_dtype in (torch.bfloat16, *_FP8_DTYPES)
        and kv_dtype in (torch.bfloat16, *_FP8_DTYPES)
        and int(kv_lora_rank) == 512
        and int(qk_rope_head_dim) in (0, 64)
    )


def _run_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_prefill: bool,
    out: torch.Tensor | None = None,
    warm_split_variants: bool = False,
    lse: torch.Tensor | None = None,
    records: bool = False,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """``q`` is ``[tokens, heads, dim]``; with ``records`` it is a KVP
    gather's rank-major ``[ranks, tokens, local_heads, dim]``, and each head's
    output and LSE land in its DCP record (int32 ``[ranks, tokens,
    local_heads, kv_lora_rank / 2 + 1]``, returned) instead of ``out`` and
    ``lse``. The records then lead ``workspace`` (flat int32) when given."""
    if not records:
        tokens, heads = q.shape[0], q.shape[1]
    else:
        ranks, tokens, q_local_heads = q.shape[:3]
        heads = ranks * q_local_heads
    if out is None and not records:
        out = torch.empty(
            (tokens, heads, kv_lora_rank),
            dtype=_output_dtype(q),
            device=q.device,
        )
    native_fp8 = q.dtype == kv_cache.dtype and q.dtype in _FP8_DTYPES
    # Use the configured selected-key capacity for scheduling.
    # Per-row live/causal lengths stay in topk_lens and are loaded by the
    # kernel. This keeps graph replay on one compiled object when those
    # runtime lengths change.
    schedule = dict(
        num_heads=heads,
        topk_width=topk_slots.shape[1],
        max_seqlen_k=int(topk_slots.shape[1]),
        is_fp8=q.dtype in _FP8_DTYPES,
        native_fp8=native_fp8,
        qk_rope_head_dim=qk_rope_head_dim,
        is_prefill=is_prefill,
    )
    launch = functools.partial(
        _launch_dense_kv,
        softmax_scale=softmax_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_prefill=is_prefill,
    )
    if warm_split_variants and not records and tokens > 0:
        # One row per reduce bucket compiles every split variant serving can reach.
        for num_kv_splits in _warmup_split_counts(tokens, **schedule):
            launch(
                q[:1],
                kv_cache,
                topk_slots[:1],
                topk_lens[:1],
                out[:1],
                num_kv_splits=num_kv_splits,
                lse=None if lse is None else lse[:1],
            )
    return launch(
        q,
        kv_cache,
        topk_slots,
        topk_lens,
        out,
        num_kv_splits=_select_num_kv_splits(num_tokens=tokens, **schedule),
        lse=lse,
        records=records,
        workspace=workspace,
    )


@functools.cache
def _warmup_split_counts(max_tokens: int, **schedule) -> tuple[int, ...]:
    """One split count per reduce bucket that up to ``max_tokens`` rows select."""
    by_bucket: dict[int, int] = {}
    for num_tokens in range(1, max_tokens + 1):
        num_kv_splits = _select_num_kv_splits(num_tokens=num_tokens, **schedule)
        if num_kv_splits > 1:
            by_bucket.setdefault(triton.next_power_of_2(num_kv_splits), num_kv_splits)
    return tuple(by_bucket.values())


def _launch_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    out: torch.Tensor | None,
    *,
    num_kv_splits: int,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_prefill: bool,
    lse: torch.Tensor | None = None,
    records: bool = False,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    if not records:
        tokens, heads = q.shape[0], q.shape[1]
        stride_q_r, stride_q_t, stride_q_h = 0, q.stride(0), q.stride(1)
    else:
        ranks, tokens, q_local_heads = q.shape[:3]
        heads = ranks * q_local_heads
        stride_q_r, stride_q_t, stride_q_h = q.stride()[:3]
    # Direct-to-shared copies preserve the source bits. Mixed inputs must
    # convert to the common BF16 compute format before their shared store.
    convert_inputs = q.dtype != kv_cache.dtype
    native_fp8 = q.dtype == kv_cache.dtype and q.dtype in _FP8_DTYPES
    tile_k = 64 if qk_rope_head_dim == 0 else 32
    split_tiles = 1
    stride_o_r = stride_lse_r = 0
    if records:
        split_tiles = _records_split_tiles(
            tokens=tokens,
            heads=heads,
            ranks=ranks,
            topk=topk_slots.shape[1],
            tile_k=tile_k,
            num_kv_splits=num_kv_splits,
        )
        # One record-shaped plane per split; split 0 is the records, which a
        # one-split row writes directly and the reduction overwrites.
        planes = (num_kv_splits, ranks, tokens, q_local_heads, kv_lora_rank // 2 + 1)
        if workspace is None:
            workspace = torch.empty(planes, dtype=torch.int32, device=q.device)
        else:
            workspace = workspace[: math.prod(planes)].view(planes)
        # A record is the head's output bits, then its LSE word.
        stage_out = workspace.view(_output_dtype(q))
        partial_lse = workspace.view(torch.float32)[..., -1]
        stride_o_s, stride_o_r, stride_o_t, stride_o_h = stage_out.stride()[:4]
        stride_lse_s, stride_lse_r, stride_lse_t, stride_lse_h = partial_lse.stride()
    elif num_kv_splits == 1:
        stage_out = out
        stride_o_t, stride_o_h = out.stride(0), out.stride(1)
        stride_o_s = 0
        if lse is None:
            partial_lse = out
            stride_lse_t = stride_lse_h = stride_lse_s = 0
        else:
            partial_lse = lse
            stride_lse_t, stride_lse_h = lse.stride()
            stride_lse_s = 0
    else:
        stage_out = torch.empty(
            (tokens, heads, num_kv_splits, kv_lora_rank),
            dtype=_output_dtype(q),
            device=q.device,
        )
        partial_lse = torch.empty(
            (tokens, heads, num_kv_splits),
            dtype=torch.float32,
            device=q.device,
        )
        stride_o_t, stride_o_h, stride_o_s = stage_out.stride()[:3]
        stride_lse_t, stride_lse_h, stride_lse_s = partial_lse.stride()

    def grid(meta):
        return (
            tokens,
            triton.cdiv(heads, meta["BLOCK_H"]),
            num_kv_splits,
        )

    buffers = (q, kv_cache, topk_slots, topk_lens, stage_out, partial_lse)
    scalars = (
        kv_cache.stride(0),
        stride_o_t,
        stride_o_h,
        stride_o_s,
        stride_lse_t,
        stride_lse_h,
        stride_lse_s,
        topk_slots.stride(0),
        float(softmax_scale),
        heads,
        num_kv_splits,
    )
    store_lse = (lse is not None or records) and num_kv_splits == 1
    if is_prefill and native_fp8 and qk_rope_head_dim == 64 and not records:
        _dsa_sparse_prefill_fp8_kernel[grid](
            *buffers,
            stride_q_t,
            stride_q_h,
            *scalars,
            TOPK=topk_slots.shape[1],
            BLOCK_H=16,
            TILE_K=64,
            D_V=kv_lora_rank,
            D_ROPE=qk_rope_head_dim,
            NATIVE_E5M2=q.dtype == torch.float8_e5m2,
            SPLIT_KV=num_kv_splits > 1,
            STORE_LSE=store_lse,
            num_warps=4,
            # Four CTAs per CU: their LDS fits, and 128 VGPRs hold the loop.
            waves_per_eu=4,
        )
    else:
        _dsa_dense_mfma_kv_kernel[grid](
            *buffers,
            stride_q_t,
            stride_q_h,
            stride_q_r,
            *scalars,
            stride_o_r,
            stride_lse_r,
            split_tiles,
            TOPK=topk_slots.shape[1],
            D_V=kv_lora_rank,
            # MFMA-legal shape for no-RoPE models; HAS_ROPE=False removes all RoPE work.
            D_ROPE=max(qk_rope_head_dim, 64),
            HAS_ROPE=qk_rope_head_dim > 0,
            CONVERT_INPUTS=convert_inputs,
            NATIVE_FP8=native_fp8,
            NATIVE_E5M2=(
                q.dtype == torch.float8_e5m2 and kv_cache.dtype == torch.float8_e5m2
            ),
            SPLIT_KV=num_kv_splits > 1,
            STORE_LSE=store_lse,
            # A model's heads per rank: one more binary for the records.
            Q_LOCAL_HEADS=q_local_heads if records else 0,
            BLOCK_H=16,
            TILE_K=tile_k,
            num_warps=4,
            # Unbounded SLP splits the in-loop KV prefetch around an LDS wait.
            llvm_cl_opts="slp-schedule-budget=100",
        )
    if num_kv_splits > 1 and records:
        # The splits merge into their split 0, the records.
        _dsa_dense_mfma_records_reduce_kernel[
            (tokens, triton.cdiv(heads, _RECORDS_REDUCE_HEADS))
        ](
            stage_out,
            partial_lse,
            topk_lens,
            stride_o_s,
            stride_o_r,
            stride_o_t,
            stride_o_h,
            stride_lse_s,
            stride_lse_r,
            stride_lse_t,
            stride_lse_h,
            heads,
            num_kv_splits,
            topk_slots.shape[1],
            tile_k,
            split_tiles,
            LOCAL_HEADS=q_local_heads,
            BLOCK_H=_RECORDS_REDUCE_HEADS,
            D_V=kv_lora_rank,
            num_warps=4,
        )
    elif num_kv_splits > 1:
        _dsa_dense_mfma_reduce_kernel[(tokens, heads)](
            stage_out,
            partial_lse,
            out,
            out if lse is None else lse,
            stage_out.stride(0),
            stage_out.stride(1),
            stage_out.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            out.stride(0),
            out.stride(1),
            0 if lse is None else lse.stride(0),
            0 if lse is None else lse.stride(1),
            num_kv_splits,
            D_V=kv_lora_rank,
            MAX_SPLITS=triton.next_power_of_2(num_kv_splits),
            STORE_LSE=lse is not None,
            num_warps=4,
        )
    return workspace[0] if records else out


def _run_dense_kv_scalar(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    kv_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=_output_dtype(q),
        device=q.device,
    )
    _dsa_dense_kv_kernel[(q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))](
        q,
        kv_cache,
        topk_slots,
        topk_lens,
        out,
        out if lse is None else lse,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        kv_dim,
        topk_slots.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_V=64,
        STORE_LSE=lse is not None,
        num_warps=4,
    )
    return out


def _run_packed_kv(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    row_bytes = int(packed_kv.shape[1])
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=_output_dtype(q),
        device=q.device,
    )
    _dsa_packed_kv_kernel[(q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))](
        q,
        packed_kv.view(torch.float8_e4m3fn),
        packed_kv.view(torch.float32),
        packed_kv.view(torch.bfloat16),
        topk_slots,
        topk_lens,
        out,
        out if lse is None else lse,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        row_bytes,
        topk_slots.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_V=64,
        STORE_LSE=lse is not None,
        num_warps=4,
    )
    return out


def _run_dsa(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    k_scale: float,
    out: torch.Tensor | None,
    max_seqlen_k: int,
    is_prefill: bool,
    warm_split_variants: bool,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    _check_inputs(
        q,
        topk_slots,
        topk_lens,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
    )
    q = _flatten_query(q).contiguous()
    topk_slots = topk_slots.contiguous()
    topk_lens = topk_lens.contiguous()
    length_stable_dense = (
        kv_cache is not None
        and sparse_kv_cache is None
        and _use_dense_mfma_core(
            q_dtype=q.dtype,
            kv_dtype=kv_cache.dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    )
    if not length_stable_dense:
        topk_slots = _trim_topk_slots_for_context(topk_slots, max_seqlen_k)
    softmax_scale = float(softmax_scale) * float(k_scale)
    out_view = None
    if (
        out is not None
        and out.is_contiguous()
        and out.device == q.device
        and not _shares_storage(
            out,
            q,
            kv_cache,
            sparse_kv_cache,
            topk_slots,
            topk_lens,
        )
    ):
        expected_numel = q.shape[0] * q.shape[1] * kv_lora_rank
        if out.numel() == expected_numel and out.dtype == _output_dtype(q):
            out_view = out.view(q.shape[0], q.shape[1], kv_lora_rank)
    # Natural-log LSE of the scaled scores per (token, head); -inf for rows
    # without a valid key, as the DCP partial merge expects.
    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    # The tiled kernel maps to dense BF16 or FP8 KV. TokenSpeed's packed sparse
    # FP8 rows use a different physical layout, so they use the packed sparse
    # implementation instead of the dense tiled implementation.
    if sparse_kv_cache is not None:
        result = _run_packed_kv(
            q,
            _flatten_packed_kv_cache(sparse_kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            lse=lse,
        )
    elif kv_cache is not None:
        dense_kv = _flatten_dense_kv_cache(kv_cache).contiguous()
        mfma_dtypes = {torch.bfloat16, *_FP8_DTYPES}
        use_tiled_kernel = (
            q.dtype in mfma_dtypes
            and dense_kv.dtype in mfma_dtypes
            and int(kv_lora_rank) == 512
        )
        if use_tiled_kernel:
            result = _run_dense_kv(
                q,
                dense_kv,
                topk_slots,
                topk_lens,
                softmax_scale=softmax_scale,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                is_prefill=is_prefill,
                out=out_view,
                warm_split_variants=warm_split_variants,
                lse=lse,
            )
        else:
            result = _run_dense_kv_scalar(
                q,
                dense_kv,
                topk_slots,
                topk_lens,
                softmax_scale=softmax_scale,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                lse=lse,
            )
    else:
        raise ValueError("Gluon DSA requires kv_cache or sparse_kv_cache")
    if out is not None:
        if result is not out_view:
            out.copy_(result.reshape_as(out))
        result = out
    return (result, lse) if return_lse else result


def launch_gluon_dsa_decode_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    warm_split_variants: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del q_len_per_req
    if logit_cap != 0.0:
        raise ValueError("Gluon DSA does not support logit_cap")
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=page_size,
        k_scale=k_scale,
        out=out,
        max_seqlen_k=max_seqlen_k,
        is_prefill=False,
        warm_split_variants=warm_split_variants,
        return_lse=return_lse,
    )


def gluon_dsa_decode_dcp_records_supported_gfx950(
    *,
    tokens: int,
    heads: int,
    topk: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> bool:
    """Whether :func:`launch_gluon_dsa_decode_dcp_records_gfx950` serves a
    decode of this size: the dense matrix core, whose one-split rows store
    their records and whose reduction merges the others'. Host values
    only."""
    if tokens <= 0 or not _use_dense_mfma_core(
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    ):
        return False
    # The core's buffer loads and stores and the reduction keep int32 offsets
    # into the query and the split workspace; a decode's are tens of MB.
    query_bytes = tokens * heads * (kv_lora_rank + qk_rope_head_dim) * q_dtype.itemsize
    return query_bytes < 2**31 and (
        gluon_dsa_decode_dcp_records_workspace_bytes_gfx950(
            tokens=tokens,
            heads=heads,
            topk=topk,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        < 2**31
    )


def gluon_dsa_decode_dcp_records_workspace_bytes_gfx950(
    *,
    tokens: int,
    heads: int,
    topk: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> int:
    """Bytes of the int32 workspace a
    :func:`launch_gluon_dsa_decode_dcp_records_gfx950` decode of this size
    writes: one record-shaped plane per key split, the records first. Host
    values only."""
    splits = _select_num_kv_splits(
        num_tokens=tokens,
        num_heads=heads,
        topk_width=topk,
        max_seqlen_k=topk,
        is_fp8=q_dtype in _FP8_DTYPES,
        native_fp8=q_dtype == kv_dtype and q_dtype in _FP8_DTYPES,
        qk_rope_head_dim=qk_rope_head_dim,
        is_prefill=False,
    )
    return splits * tokens * heads * (kv_lora_rank // 2 + 1) * 4


def launch_gluon_dsa_decode_dcp_records_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`launch_gluon_dsa_decode_gfx950` with ``return_lse`` over a
    KVP gather's rank-major query ``[ranks, tokens, local_heads, dim]``,
    returning the DCP records ``[ranks, tokens, local_heads, kv_lora_rank /
    2 + 1]`` (int32) that ``dispatch_dcp_attention_partials`` packs from its
    output and LSE. A row runs at most as many key splits as its live tiles
    fill with :func:`_records_split_tiles` each: a short one, as one of about
    its rank's share of the selection, runs fewer splits than that decode
    (and may round differently), one filling them all the same splits: the
    same LSE, and outputs within one BF16 rounding. The ranks
    may lie further apart than their contiguous rows, as when the query
    rides in a larger gathered chunk; other layouts, and rank strides or
    bases the query's copy to LDS cannot take, are copied first. A flat
    contiguous int32 ``workspace`` of at least
    :func:`gluon_dsa_decode_dcp_records_workspace_bytes_gfx950` bytes holds
    the splits, and the records returned are its leading view. Raises
    ``ValueError`` for a size
    :func:`gluon_dsa_decode_dcp_records_supported_gfx950` refuses."""
    if logit_cap != 0.0:
        raise ValueError("Gluon DSA does not support logit_cap")
    if q.dim() != 4:
        raise ValueError(
            "DSA records take a rank-major query [ranks, tokens, local_heads, "
            f"dim], got {tuple(q.shape)}"
        )
    _check_inputs(
        q,
        topk_slots,
        topk_lens,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
    )
    ranks, tokens, local_heads, _ = q.shape
    if topk_slots.shape[0] != tokens:
        raise ValueError(
            f"DSA records selection {tuple(topk_slots.shape)} does not match "
            f"{tokens} query rows"
        )
    dense_kv = _flatten_dense_kv_cache(kv_cache).contiguous()
    if not gluon_dsa_decode_dcp_records_supported_gfx950(
        tokens=tokens,
        heads=ranks * local_heads,
        topk=topk_slots.shape[1],
        q_dtype=q.dtype,
        kv_dtype=dense_kv.dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    ):
        raise ValueError(
            "DSA decode records need the dense matrix core "
            "(check gluon_dsa_decode_dcp_records_supported_gfx950 first)"
        )
    if workspace is not None:
        needed = gluon_dsa_decode_dcp_records_workspace_bytes_gfx950(
            tokens=tokens,
            heads=ranks * local_heads,
            topk=topk_slots.shape[1],
            q_dtype=q.dtype,
            kv_dtype=dense_kv.dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        if (
            workspace.dtype != torch.int32
            or workspace.dim() != 1
            or not workspace.is_contiguous()
            or workspace.nbytes < needed
        ):
            raise ValueError(
                f"DSA records need a flat contiguous int32 workspace of {needed} "
                f"bytes, got {workspace.dtype} {tuple(workspace.shape)}"
            )
    dim = q.shape[3]
    # The query's direct-to-LDS copy needs a 16-byte-aligned base and
    # 16-element-aligned row offsets, and its buffer loads a span under 2**31
    # bytes.
    if (
        q.stride()[1:] != (local_heads * dim, dim, 1)
        or q.stride(0) % 16
        or q.data_ptr() % 16
        or ((ranks - 1) * q.stride(0) + tokens * local_heads * dim) * q.element_size()
        >= 2**31
    ):
        q = q.contiguous()
    return _run_dense_kv(
        q,
        dense_kv,
        topk_slots.contiguous(),
        topk_lens.contiguous(),
        softmax_scale=float(softmax_scale) * float(k_scale),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_prefill=False,
        records=True,
        workspace=workspace,
    )


def launch_gluon_dsa_prefill_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    warm_split_variants: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del q_len_per_req
    if logit_cap != 0.0:
        raise ValueError("Gluon DSA does not support logit_cap")
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=page_size,
        k_scale=k_scale,
        out=out,
        max_seqlen_k=max_seqlen_k,
        is_prefill=True,
        warm_split_variants=warm_split_variants,
        return_lse=return_lse,
    )
