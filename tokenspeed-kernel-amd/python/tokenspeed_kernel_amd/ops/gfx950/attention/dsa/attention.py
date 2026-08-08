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

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = [
    "gluon_dsa_decode_gfx950",
    "gluon_dsa_prefill_gfx950",
]

_REGISTERED_TOPK_WIDTHS = (512, 1024, 2048)


@gluon.constexpr_function
def _value_layout(
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    return gl.BlockedLayout([1, 1], [1, 64], [NUM_WARPS, 1], [1, 0])


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
def _dsa_dense_mfma_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    TOPK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    D_V: gl.constexpr,
    D_ROPE: gl.constexpr,
):
    mfma_s: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mfma_acc: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[4, 1],
    )

    _qlora_tpw_k: gl.constexpr = min(64, D_V // 8)
    _qlora_tpw_m: gl.constexpr = 64 // _qlora_tpw_k
    blk_qlora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[_qlora_tpw_m, _qlora_tpw_k],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    blk_qrope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    _klora_tpw_m: gl.constexpr = min(64, D_V // 8)
    _klora_tpw_n: gl.constexpr = 64 // _klora_tpw_m
    blk_klora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[_klora_tpw_m, _klora_tpw_n],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    blk_krope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2, 1],
        threads_per_warp=[32, 2],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    blk_topk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )

    sh_qlora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [BLOCK_H, D_V],
        [1, 0],
    )
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8,
        per_phase=2,
        max_phase=8,
        order=[1, 0],
    )
    sh_klora: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]],
        [D_V, TILE_K],
        [0, 1],
    )
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8,
        per_phase=2,
        max_phase=8,
        order=[0, 1],
    )

    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_s,
        k_width=8,
    )
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_s,
        k_width=8,
    )
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_s,
        k_width=8,
    )
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_s,
        k_width=8,
    )
    dot_p_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_acc,
        k_width=4,
    )
    dot_v_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_acc,
        k_width=4,
    )

    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H
    valid_len = gl.load(topk_lens + token_idx).to(tl.int32)

    offs_h_qlora = hg_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, blk_qlora),
    )
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads
    q_base = token_idx.to(tl.int64) * stride_q_t
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
    mask_h_qrope = offs_h_qrope < num_heads
    q_offs_rope = (
        q_base
        + offs_h_qrope[:, None].to(tl.int64) * stride_q_h
        + (D_V + offs_r_qrope[None, :]).to(tl.int64)
    )

    smem_qlora = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [BLOCK_H, D_V],
        layout=sh_qlora,
    )
    smem_qrope = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [BLOCK_H, D_ROPE],
        layout=sh_qrope,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_qlora,
        ptr=q,
        offsets=q_offs_lora.to(tl.int32),
        mask=mask_h_qlora[:, None],
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_qrope,
        ptr=q,
        offsets=q_offs_rope.to(tl.int32),
        mask=mask_h_qrope[:, None],
    )
    gl.amd.cdna4.async_copy.commit_group()

    NUM_TILES: gl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx.to(tl.int64) * stride_topk_t

    offs_tile_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    offs_tile_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma_s))

    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    topk_pos_klora = gl.amd.cdna4.buffer_load(
        ptr=topk_indices,
        offsets=topk_base.to(tl.int32) + offs_tile_klora,
        mask=offs_tile_klora < TOPK,
        other=-1,
    )
    topk_pos_krope = gl.amd.cdna4.buffer_load(
        ptr=topk_indices,
        offsets=topk_base.to(tl.int32) + offs_tile_krope,
        mask=offs_tile_krope < TOPK,
        other=-1,
    )
    topk_pos_mma = gl.amd.cdna4.buffer_load(
        ptr=topk_indices,
        offsets=topk_base.to(tl.int32) + offs_tile_mma,
        mask=offs_tile_mma < TOPK,
        other=-1,
    )

    valid_klora = (offs_tile_klora < valid_len) & (topk_pos_klora != -1)
    valid_krope = (offs_tile_krope < valid_len) & (topk_pos_krope != -1)
    valid_mma = (offs_tile_mma < valid_len) & (topk_pos_mma != -1)
    safe_klora = gl.where(valid_klora, topk_pos_klora, 0)
    safe_krope = gl.where(valid_krope, topk_pos_krope, 0)

    smem_krope = gl.allocate_shared_memory(
        kv.dtype.element_ty,
        [2, D_ROPE, TILE_K],
        layout=sh_krope,
    )
    smem_klora = gl.allocate_shared_memory(
        kv.dtype.element_ty,
        [2, D_V, TILE_K],
        layout=sh_klora,
    )

    klora_offs = safe_klora[None, :].to(tl.int64) * stride_kv_t + offs_v_klora[
        :, None
    ].to(tl.int64)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_klora.index(0),
        ptr=kv,
        offsets=klora_offs.to(tl.int32),
        mask=valid_klora[None, :],
    )
    krope_offs = safe_krope[None, :].to(tl.int64) * stride_kv_t + (
        D_V + offs_r_krope[:, None]
    ).to(tl.int64)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_krope.index(0),
        ptr=kv,
        offsets=krope_offs.to(tl.int32),
        mask=valid_krope[None, :],
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(1)
    q_lora_dot = smem_qlora.load(dot_qlora_a)
    q_rope_dot = smem_qrope.load(dot_qrope_a)

    m_i = gl.full(
        [BLOCK_H],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, mfma_s),
    )
    l_i = gl.full(
        [BLOCK_H],
        0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, mfma_s),
    )
    acc = gl.zeros([BLOCK_H, D_V], dtype=gl.float32, layout=mfma_acc)

    cur_buf = 0
    for tile_idx in range(NUM_TILES - 1):
        next_base = (tile_idx + 1) * TILE_K
        next_offs_klora = next_base + offs_tile_klora
        next_offs_krope = next_base + offs_tile_krope
        next_offs_mma = next_base + offs_tile_mma

        topk_pos_klora_next = gl.amd.cdna4.buffer_load(
            ptr=topk_indices,
            offsets=topk_base.to(tl.int32) + next_offs_klora,
            mask=next_offs_klora < TOPK,
            other=-1,
        )
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

        valid_klora_next = (next_offs_klora < valid_len) & (topk_pos_klora_next != -1)
        valid_krope_next = (next_offs_krope < valid_len) & (topk_pos_krope_next != -1)
        valid_mma_next = (next_offs_mma < valid_len) & (topk_pos_mma_next != -1)
        safe_klora_next = gl.where(valid_klora_next, topk_pos_klora_next, 0)
        safe_krope_next = gl.where(valid_krope_next, topk_pos_krope_next, 0)
        next_buf = 1 - cur_buf

        klora_offs_next = safe_klora_next[None, :].to(
            tl.int64
        ) * stride_kv_t + offs_v_klora[:, None].to(tl.int64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_klora.index(next_buf),
            ptr=kv,
            offsets=klora_offs_next.to(tl.int32),
            mask=valid_klora_next[None, :],
        )
        krope_offs_next = safe_krope_next[None, :].to(tl.int64) * stride_kv_t + (
            D_V + offs_r_krope[:, None]
        ).to(tl.int64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_krope.index(next_buf),
            ptr=kv,
            offsets=krope_offs_next.to(tl.int32),
            mask=valid_krope_next[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(1)

        klora_smem_cur = smem_klora.index(cur_buf)
        k_lora_t_dot = klora_smem_cur.load(dot_klora_b)
        v_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)
        k_rope_t_dot = smem_krope.index(cur_buf).load(dot_krope_b)

        scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s)
        scores = gl.amd.cdna4.mfma(q_lora_dot, k_lora_t_dot, scores)
        scores = gl.amd.cdna4.mfma(q_rope_dot, k_rope_t_dot, scores)
        scores = scores * scale

        offs_h_mma = hg_offset + gl.arange(
            0,
            BLOCK_H,
            layout=gl.SliceLayout(1, mfma_s),
        )
        mask_h_mma = offs_h_mma < num_heads
        scores = gl.where(
            valid_mma[None, :] & mask_h_mma[:, None], scores, -float("inf")
        )

        m_j = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_j)
        m_new = gl.where(m_new > -float("inf"), m_new, 0.0)
        alpha = gl.exp(m_i - m_new)
        probs = gl.exp(scores - m_new[:, None])
        l_new = alpha * l_i + gl.sum(probs, axis=1)

        alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, mfma_acc))
        acc = acc * alpha_acc[:, None]
        probs_dot = gl.convert_layout(probs.to(q.dtype.element_ty), dot_p_a)
        acc = gl.amd.cdna4.mfma(probs_dot, v_lora_dot, acc)

        m_i = m_new
        l_i = l_new
        cur_buf = next_buf
        valid_mma = valid_mma_next

    gl.amd.cdna4.async_copy.wait_group(0)

    klora_smem_cur = smem_klora.index(cur_buf)
    k_lora_t_dot = klora_smem_cur.load(dot_klora_b)
    v_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)
    k_rope_t_dot = smem_krope.index(cur_buf).load(dot_krope_b)

    scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s)
    scores = gl.amd.cdna4.mfma(q_lora_dot, k_lora_t_dot, scores)
    scores = gl.amd.cdna4.mfma(q_rope_dot, k_rope_t_dot, scores)
    scores = scores * scale

    offs_h_mma = hg_offset + gl.arange(
        0,
        BLOCK_H,
        layout=gl.SliceLayout(1, mfma_s),
    )
    mask_h_mma = offs_h_mma < num_heads
    scores = gl.where(valid_mma[None, :] & mask_h_mma[:, None], scores, -float("inf"))

    m_j = gl.max(scores, axis=1)
    m_new = gl.maximum(m_i, m_j)
    m_new = gl.where(m_new > -float("inf"), m_new, 0.0)
    alpha = gl.exp(m_i - m_new)
    probs = gl.exp(scores - m_new[:, None])
    l_new = alpha * l_i + gl.sum(probs, axis=1)

    alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, mfma_acc))
    acc = acc * alpha_acc[:, None]
    probs_dot = gl.convert_layout(probs.to(q.dtype.element_ty), dot_p_a)
    acc = gl.amd.cdna4.mfma(probs_dot, v_lora_dot, acc)

    l_i_acc = gl.convert_layout(l_new, gl.SliceLayout(1, mfma_acc))
    safe_l_i = gl.where(l_i_acc > 0.0, l_i_acc, 1.0)
    acc = acc / safe_l_i[:, None]
    acc = gl.where(l_i_acc[:, None] > 0.0, acc, 0.0)

    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_o = offs_h_o < num_heads
    o_base = token_idx.to(tl.int64) * stride_o_t
    o_offs = (
        o_base
        + offs_h_o[:, None].to(tl.int64) * stride_o_h
        + offs_v_o[None, :].to(tl.int64)
    )
    out_vals = gl.convert_layout(acc.to(out.dtype.element_ty), blk_qlora)
    gl.amd.cdna4.buffer_store(
        stored_value=out_vals,
        ptr=out,
        offsets=o_offs.to(tl.int32),
        mask=mask_h_o[:, None],
    )


@gluon.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    kv_dim: gl.constexpr,
    topk: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
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


@gluon.jit
def _dsa_packed_kv_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    row_bytes: gl.constexpr,
    topk: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
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
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(f"Gluon DSA supports BF16/FP8 q, got {q.dtype}")
    if page_size != 64:
        raise ValueError(f"Gluon DSA supports page_size=64, got {page_size}")
    if qk_nope_head_dim not in (128, 192):
        raise ValueError(
            "Gluon DSA supports qk_nope_head_dim in {128, 192}, got "
            f"{qk_nope_head_dim}"
        )
    if kv_lora_rank not in (128, 512):
        raise ValueError(
            f"Gluon DSA supports kv_lora_rank in {{128, 512}}, got {kv_lora_rank}"
        )
    if qk_rope_head_dim != 64:
        raise ValueError(
            f"Gluon DSA supports qk_rope_head_dim=64, got {qk_rope_head_dim}"
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
    return torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype


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


def _run_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=_output_dtype(q),
        device=q.device,
    )
    grid = lambda meta: (q.shape[0], triton.cdiv(q.shape[1], meta["BLOCK_H"]))
    _dsa_dense_mfma_kv_kernel[grid](
        q,
        kv_cache,
        topk_slots,
        topk_lens,
        out,
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        out.stride(0),
        out.stride(1),
        topk_slots.stride(0),
        float(softmax_scale),
        q.shape[1],
        topk_slots.shape[1],
        D_V=kv_lora_rank,
        D_ROPE=qk_rope_head_dim,
        BLOCK_H=16,
        TILE_K=32,
        num_warps=4,
    )
    return out


def _run_dense_kv_scalar(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
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
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        kv_dim,
        topk_slots.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_V=64,
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
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        row_bytes,
        topk_slots.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_V=64,
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
) -> torch.Tensor:
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
    topk_slots = _trim_topk_slots_for_context(topk_slots, max_seqlen_k)
    softmax_scale = float(softmax_scale) * float(k_scale)
    # The AITER-style tiled kernel maps to dense BF16 KV. TokenSpeed's packed
    # sparse FP8 rows use a different physical layout and stay on the scalar path.
    if sparse_kv_cache is not None:
        result = _run_packed_kv(
            q,
            _flatten_packed_kv_cache(sparse_kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    elif kv_cache is not None:
        dense_kv = _flatten_dense_kv_cache(kv_cache).contiguous()
        if (
            q.dtype == torch.bfloat16
            and dense_kv.dtype == torch.bfloat16
            and int(kv_lora_rank) == 512
        ):
            result = _run_dense_kv(
                q,
                dense_kv,
                topk_slots,
                topk_lens,
                softmax_scale=softmax_scale,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
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
            )
    else:
        raise ValueError("Gluon DSA requires kv_cache or sparse_kv_cache")
    if out is None:
        return result
    out_view = out.reshape_as(result)
    out_view.copy_(result)
    return out


def gluon_dsa_decode_gfx950(
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
) -> torch.Tensor:
    del q_len_per_req
    if logit_cap != 0.0 or return_lse:
        raise ValueError("Gluon DSA does not support logit_cap or return_lse")
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
    )


def gluon_dsa_prefill_gfx950(
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
) -> torch.Tensor:
    del q_len_per_req
    if logit_cap != 0.0 or return_lse:
        raise ValueError("Gluon DSA does not support logit_cap or return_lse")
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
    )
