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

"""Sparse MLA prefill over MXFP4 latent cache rows on GFX950.

The rows are ``tokenspeed_kernel.ops.kvcache.mla_mxfp4``'s: 256 bytes of
nibble-packed E2M1 codes (even element in the low nibble), 16 UE8M0
exponents (one per 32 latent values), then the 64-wide BF16 RoPE key. The
kernel follows ``_dsa_sparse_prefill_fp8_kernel``: one CTA per (token, 16
heads, split), one 64-key tile at a time, the selected rows copied
asynchronously into LDS, an online softmax across the tile. Per tile:

- latent ``q . k``: the MXFP8 query (E4M3 with a UE8M0 scale per 32
  channels, quantized once per launch exactly as the Triton kernel does)
  times the cache's codes and exponents on the block-scaled
  ``v_mfma_scale_f32_16x16x128_f8f6f4``, straight from the cache bytes;
- RoPE ``q . k``: a BF16 MFMA;
- ``p . v``: per 128-channel phase the codes are decoded to BF16 by the
  hardware ``v_cvt_scalef32_pk_bf16_fp4`` (exact: an E2M1 value times its
  group's power of two), staged in LDS, and multiplied with the BF16
  probabilities by a BF16 MFMA -- the latent's scales run along the key axis
  there, which a block-scaled MFMA cannot apply.

This is the Triton kernel's arithmetic (``mxfp4_sparse_mla_attention`` with
``qk_format="mxfp8"``) on a gfx950 schedule.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
    _GFX950_COMPUTE_UNITS,
    _dsa_dense_mfma_reduce_kernel,
    _dsa_online_softmax,
    _dsa_store_output,
    _load_selected,
    _value_output_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.attention._common import select_kv_splits

__all__ = ["launch_gluon_mxfp4_sparse_mla_prefill_gfx950"]

_KV_LORA_RANK = 512
_ROPE_DIM = 64
_CODE_BYTES = _KV_LORA_RANK // 2
_GROUPS = _KV_LORA_RANK // 32
_ROW_BYTES = _CODE_BYTES + _GROUPS + 2 * _ROPE_DIM
_BLOCK_H = 16
_TILE_K = 64


@triton.jit
def _mxfp8_query_kernel(
    q,
    q_fp8,
    q_scale,
    stride_q_t,
    stride_q_h,
    num_heads,
    kv_lora_rank: tl.constexpr,
):
    """One head's latent query as MXFP8: per 32 channels the smallest power
    of two that brings the amax within E4M3's 448 (floored so the scale stays
    normal); the Triton MXFP4 kernel's in-kernel quantization."""
    row = tl.program_id(0)
    token = (row // num_heads).to(tl.int64)
    head = (row % num_heads).to(tl.int64)
    groups: tl.constexpr = kv_lora_rank // 32
    values = tl.load(
        q + token * stride_q_t + head * stride_q_h + tl.arange(0, kv_lora_rank)
    ).to(tl.float32)
    values = tl.reshape(values, (groups, 32))
    amax = tl.max(tl.abs(values), 1)
    scaled_max = tl.div_rn(tl.maximum(amax, 448.0 * 1.1754943508222875e-38), 448.0)
    bits = scaled_max.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(exponent, 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    codes = tl.reshape(tl.div_rn(values, scale[:, None]), (kv_lora_rank,))
    tl.store(
        q_fp8 + row.to(tl.int64) * kv_lora_rank + tl.arange(0, kv_lora_rank),
        codes.to(tl.float8e4nv),
    )
    tl.store(
        q_scale + row.to(tl.int64) * groups + tl.arange(0, groups),
        exponent.to(tl.uint8),
    )


@gluon.jit
def _row_offset(slots, page_size, page_stride, ROW_BYTES: gl.constexpr):
    """Byte offset of cache slot ``slots`` within the cache view; 16-byte
    aligned (the launcher checks the page stride), which the 16-byte async
    copies need."""
    offset = (slots // page_size).to(tl.int64) * page_stride + (
        slots % page_size
    ).to(tl.int64) * ROW_BYTES
    return gl.multiple_of(offset, [16])


@gluon.jit(
    do_not_specialize=(
        "stride_lse_t",
        "stride_lse_h",
        "stride_lse_s",
        "num_kv_splits",
    )
)
def _mxfp4_sparse_prefill_kernel(
    q,  # bf16 [tokens, heads, 576]; only the RoPE part is read
    q_fp8,  # e4m3 [tokens * heads, 512]
    q_scale,  # uint8 [tokens * heads, 16]
    kv,  # uint8 rows
    kv_rope,  # bf16 view of the same bytes
    topk_indices,
    topk_lens,
    out,
    partial_lse,
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_o_s: tl.int64,
    stride_lse_t: tl.int64,
    stride_lse_h: tl.int64,
    stride_lse_s: tl.int64,
    stride_topk_t: tl.int64,
    page_size: tl.int32,
    page_stride: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    num_kv_splits: tl.int32,
    TOPK: gl.constexpr,
    ROW_BYTES: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    SPLIT_KV: gl.constexpr,
    STORE_LSE: gl.constexpr,
):
    D_V: gl.constexpr = 512
    D_ROPE: gl.constexpr = 64
    CODE_BYTES: gl.constexpr = D_V // 2
    GROUPS: gl.constexpr = D_V // 32
    PHASE: gl.constexpr = 128
    PHASE_BYTES: gl.constexpr = PHASE // 2
    PHASE_GROUPS: gl.constexpr = PHASE // 32
    # The RoPE key's BF16 element offset within a row.
    ROPE_START: gl.constexpr = (CODE_BYTES + GROUPS) // 2
    LAST_TILE: gl.constexpr = (TOPK + TILE_K - 1) // TILE_K - 1

    # Scores: block-scaled 16x16x128 for the latent, BF16 16x16x32 for RoPE;
    # both accumulate in the same 16x16 register tile.
    mfma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_rope: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_s: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 1]
    )
    mfma_acc: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
        tiles_per_warp=[1, 2],
    )
    softmax_layout: gl.constexpr = mfma_qk
    softmax_scalar_layout: gl.constexpr = gl.SliceLayout(1, softmax_layout)
    blk_out_phase: gl.constexpr = _value_output_layout(BLOCK_H)

    dot_q_a: gl.constexpr = gl.DotOperandLayout(0, mfma_qk, 16)
    dot_codes_b: gl.constexpr = gl.DotOperandLayout(1, mfma_qk, 16)
    q_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_q_a, [BLOCK_H, GROUPS]
    )
    k_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_codes_b, [TILE_K, GROUPS]
    )
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(0, mfma_rope, 8)
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(1, mfma_rope, 8)
    dot_p_a: gl.constexpr = gl.DotOperandLayout(0, mfma_acc, 8)
    dot_v_b: gl.constexpr = gl.DotOperandLayout(1, mfma_acc, 8)

    # Async row copies: 16 code bytes, 4 exponent bytes, 8 RoPE values a lane.
    blk_codes: gl.constexpr = gl.BlockedLayout([16, 1], [16, 4], [1, 4], [0, 1])
    blk_scales: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, 4], [0, 1])
    blk_krope: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    blk_qrope: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    # The decode: a lane upcasts one 32-value group (16 code bytes) of a key.
    blk_packed: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    blk_decoded: gl.constexpr = gl.BlockedLayout([1, 32], [16, 4], [4, 1], [1, 0])
    blk_group_scale: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [4, 1], [1, 0])

    # The async copies write each warp's rows contiguously; the padding
    # between those chunks rotates the reads through the banks.
    sh_codes: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[4 * CODE_BYTES, 16]], [CODE_BYTES, TILE_K], [0, 1]
    )
    sh_scales: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[16 * GROUPS, 4]], [GROUPS, TILE_K], [0, 1]
    )
    sh_krope: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[8 * D_ROPE, 8]], [D_ROPE, TILE_K], [0, 1]
    )
    sh_value: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[PHASE, 8]], [PHASE, TILE_K], [0, 1]
    )
    sh_probs: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[TILE_K, 8]], [BLOCK_H, TILE_K], [1, 0]
    )

    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    if SPLIT_KV:
        split_idx = gl.program_id(axis=2)
    else:
        split_idx: gl.constexpr = 0
    hg_offset = hg_idx * BLOCK_H
    valid_len = gl.load(topk_lens + token_idx).to(tl.int32)
    q_base = token_idx.to(tl.int64) * stride_q_t

    # The latent query and its exponents, in registers for the whole row.
    offs_h_qa = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, dot_q_a))
    offs_c_qa = gl.arange(0, D_V, layout=gl.SliceLayout(0, dot_q_a))
    q_rows_a = (token_idx * num_heads + offs_h_qa).to(tl.int64)
    q_lat = gl.amd.cdna4.buffer_load(
        ptr=q_fp8,
        offsets=(q_rows_a[:, None] * D_V + offs_c_qa[None, :]).to(tl.int32),
        mask=(offs_h_qa < num_heads)[:, None],
        other=0.0,
    )
    offs_h_qs = hg_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, q_scale_layout)
    )
    offs_g_qs = gl.arange(0, GROUPS, layout=gl.SliceLayout(0, q_scale_layout))
    q_rows_s = (token_idx * num_heads + offs_h_qs).to(tl.int64)
    q_exp = gl.amd.cdna4.buffer_load(
        ptr=q_scale,
        offsets=(q_rows_s[:, None] * GROUPS + offs_g_qs[None, :]).to(tl.int32),
        mask=(offs_h_qs < num_heads)[:, None],
        other=127,
    )
    offs_h_qr = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_qr = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    q_rope = gl.amd.cdna4.buffer_load(
        ptr=q,
        offsets=(
            q_base
            + offs_h_qr[:, None].to(tl.int64) * stride_q_h
            + (D_V + offs_r_qr[None, :]).to(tl.int64)
        ).to(tl.int32),
        mask=(offs_h_qr < num_heads)[:, None],
        other=0.0,
    )
    q_rope_dot = gl.convert_layout(q_rope.to(gl.bfloat16), dot_qrope_a)

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

    local_offs_codes = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_codes))
    local_offs_scales = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_scales))
    local_offs_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    local_offs_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, softmax_layout))
    offs_b_codes = gl.arange(0, CODE_BYTES, layout=gl.SliceLayout(1, blk_codes))
    offs_g_scales = gl.arange(0, GROUPS, layout=gl.SliceLayout(1, blk_scales))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    smem_codes = gl.allocate_shared_memory(
        gl.uint8, [CODE_BYTES, TILE_K], layout=sh_codes
    )
    smem_scales = gl.allocate_shared_memory(gl.uint8, [GROUPS, TILE_K], layout=sh_scales)
    smem_krope = gl.allocate_shared_memory(
        gl.bfloat16, [D_ROPE, TILE_K], layout=sh_krope
    )
    smem_value = gl.allocate_shared_memory(
        gl.bfloat16, [PHASE, TILE_K], layout=sh_value
    )
    smem_probs = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_H, TILE_K], layout=sh_probs
    )

    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=softmax_scalar_layout)
    mask_h_mma = offs_h_mma < num_heads
    m_i = gl.full(
        [BLOCK_H], float("-inf"), dtype=gl.float32, layout=softmax_scalar_layout
    )
    l_i = gl.full([BLOCK_H], 0.0, dtype=gl.float32, layout=softmax_scalar_layout)
    acc0 = gl.zeros([BLOCK_H, PHASE], dtype=gl.float32, layout=blk_out_phase)
    acc1 = gl.zeros([BLOCK_H, PHASE], dtype=gl.float32, layout=blk_out_phase)
    acc2 = gl.zeros([BLOCK_H, PHASE], dtype=gl.float32, layout=blk_out_phase)
    acc3 = gl.zeros([BLOCK_H, PHASE], dtype=gl.float32, layout=blk_out_phase)
    for i in range(num_tiles):
        tile = tile_start + i
        live = tile < tile_end
        # An empty split still runs one fully masked tile.
        idx_base = gl.minimum(tile, LAST_TILE) * TILE_K
        idx_c = _load_selected(topk_indices, idx_row, idx_base + local_offs_codes, TOPK)
        idx_s = _load_selected(
            topk_indices, idx_row, idx_base + local_offs_scales, TOPK
        )
        idx_r = _load_selected(topk_indices, idx_row, idx_base + local_offs_krope, TOPK)
        valid_c = live & ((tile * TILE_K + local_offs_codes) < valid_len) & (idx_c >= 0)
        valid_s = (
            live & ((tile * TILE_K + local_offs_scales) < valid_len) & (idx_s >= 0)
        )
        valid_r = live & ((tile * TILE_K + local_offs_krope) < valid_len) & (idx_r >= 0)
        # Masked keys copy in as zero bytes (buffer loads' out-of-bounds
        # value): zero scores the softmax drops and zero values.
        row_c = _row_offset(gl.where(valid_c, idx_c, 0), page_size, page_stride, ROW_BYTES)
        row_s = _row_offset(gl.where(valid_s, idx_s, 0), page_size, page_stride, ROW_BYTES)
        # The BF16 view's element offset; page strides are even.
        row_r = (
            _row_offset(gl.where(valid_r, idx_r, 0), page_size, page_stride, ROW_BYTES)
            // 2
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_codes,
            ptr=kv,
            offsets=(row_c[None, :] + offs_b_codes[:, None].to(tl.int64)).to(tl.int32),
            mask=valid_c[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_scales,
            ptr=kv,
            offsets=(
                row_s[None, :] + (CODE_BYTES + offs_g_scales[:, None]).to(tl.int64)
            ).to(tl.int32),
            mask=valid_s[None, :],
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem_krope,
            ptr=kv_rope,
            offsets=(
                row_r[None, :] + (ROPE_START + offs_r_krope[:, None]).to(tl.int64)
            ).to(tl.int32),
            mask=valid_r[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()
        idx_m = _load_selected(topk_indices, idx_row, idx_base + local_offs_mma, TOPK)
        valid_mma = live & ((tile * TILE_K + local_offs_mma) < valid_len) & (idx_m >= 0)
        gl.amd.cdna4.async_copy.wait_group(0)

        scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_qk)
        scores = gl.amd.cdna4.mfma_scaled(
            q_lat,
            q_exp,
            "e4m3",
            smem_codes.load(dot_codes_b),
            smem_scales.permute([1, 0]).load(k_scale_layout),
            "e2m1",
            scores,
        )
        rope_scores = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_rope)
        rope_scores = gl.amd.cdna4.mfma(
            q_rope_dot, smem_krope.load(dot_krope_b), rope_scores
        )
        scores += gl.convert_layout(rope_scores, mfma_qk, assert_trivial=True)
        probs, alpha, m_i, l_i = _dsa_online_softmax(
            scores, valid_mma, mask_h_mma, m_i, l_i, scale
        )

        alpha_out = gl.convert_layout(alpha, gl.SliceLayout(1, blk_out_phase))
        smem_probs.store(probs.to(gl.bfloat16))
        probs_dot = smem_probs.load(dot_p_a)
        value_t = smem_value.permute([1, 0])
        for phase_idx in gl.static_range(4):
            packed = (
                smem_codes.slice(phase_idx * PHASE_BYTES, PHASE_BYTES, 0)
                .permute([1, 0])
                .load(blk_packed)
            )
            group_exp = (
                smem_scales.slice(phase_idx * PHASE_GROUPS, PHASE_GROUPS, 0)
                .permute([1, 0])
                .load(blk_group_scale)
            )
            values = gl.amd.cdna4.scaled_upcast(
                packed, group_exp, gl.bfloat16, axis=1
            )
            value_t.store(gl.convert_layout(values, blk_decoded, assert_trivial=True))
            value_tile = gl.zeros([BLOCK_H, PHASE], dtype=gl.float32, layout=mfma_acc)
            value_tile = gl.amd.cdna4.mfma(
                probs_dot, value_t.load(dot_v_b), value_tile
            )
            value_tile = gl.convert_layout(
                value_tile, blk_out_phase, assert_trivial=True
            )
            if phase_idx == 0:
                acc0 = acc0 * alpha_out[:, None] + value_tile
            elif phase_idx == 1:
                acc1 = acc1 * alpha_out[:, None] + value_tile
            elif phase_idx == 2:
                acc2 = acc2 * alpha_out[:, None] + value_tile
            else:
                acc3 = acc3 * alpha_out[:, None] + value_tile

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


def _num_kv_splits(tokens: int, heads: int, topk: int) -> int:
    base_ctas = tokens * triton.cdiv(heads, _BLOCK_H)
    work_tiles = triton.cdiv(topk, 32)
    # Prefill grids that already fill the GPU twice over keep one split and
    # skip the merge, as the FP8 prefill kernel does.
    if work_tiles <= 8 or base_ctas >= 2 * _GFX950_COMPUTE_UNITS:
        return 1
    return select_kv_splits(
        base_ctas=base_ctas, num_pages=work_tiles, sm_count=_GFX950_COMPUTE_UNITS
    )


def launch_gluon_mxfp4_sparse_mla_prefill_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    page_size: int,
    page_stride: int,
    row_stride: int,
    softmax_scale: float,
    out: torch.Tensor,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse MLA over MXFP4 latent rows for prefill rows on gfx950.

    Args:
        q: ``[tokens, heads, 576]`` BF16 absorbed query (512 latent + 64 RoPE
            channels), unit channel stride.
        kv_cache: the uint8 cache view, ``(page_size, page_stride,
            row_stride)`` its ``mla_latent_mxfp4_geometry``; a page's rows
            must be dense (``row_stride == 400``).
        topk_slots: ``[tokens, topk]`` int32 cache slots; -1 is no key.
        topk_lens: ``[tokens]`` int32 live leading entries per row.
        softmax_scale: scale of ``q . k`` before the softmax.
        out: dense ``[tokens, heads, 512]`` output in ``q``'s dtype.
        lse: optional ``[tokens, heads]`` FP32 natural-log LSE output.

    Returns:
        ``out``.
    """
    tokens, heads, head_dim = q.shape
    if q.dtype != torch.bfloat16 or head_dim != _KV_LORA_RANK + _ROPE_DIM:
        raise ValueError(
            f"q must be BF16 [tokens, heads, {_KV_LORA_RANK + _ROPE_DIM}], got "
            f"{tuple(q.shape)} {q.dtype}"
        )
    if q.stride(-1) != 1:
        raise ValueError("q needs unit channel stride")
    if (
        row_stride != _ROW_BYTES
        or page_stride < page_size * row_stride
        or page_stride % 16
        or kv_cache.data_ptr() % 16
    ):
        raise ValueError(
            f"the gfx950 MXFP4 prefill kernel reads 16-byte aligned pages of dense "
            f"{_ROW_BYTES}-byte rows, got page_size={page_size} page_stride={page_stride} "
            f"row_stride={row_stride}"
        )
    # Pages of a [pages, page_size, 1, row] view, slots of a [slots, 1, row] one.
    span = (kv_cache.shape[0] - 1) * page_stride + page_size * row_stride
    if span >= 2**31:
        raise ValueError(
            f"the MXFP4 cache view ({span} bytes) exceeds the kernel's 32-bit "
            "buffer offsets"
        )
    if topk_slots.dtype != torch.int32 or topk_slots.dim() != 2:
        raise ValueError("topk_slots must be int32 [tokens, topk]")
    if topk_slots.shape[0] != tokens or topk_slots.stride(1) != 1:
        raise ValueError(
            f"topk_slots {tuple(topk_slots.shape)} is not a dense [{tokens}, topk]"
        )
    if topk_lens.dtype != torch.int32 or topk_lens.shape != (tokens,):
        raise ValueError(f"topk_lens must be int32 [{tokens}]")
    if (
        out.shape != (tokens, heads, _KV_LORA_RANK)
        or out.dtype != q.dtype
        or out.stride(-1) != 1
    ):
        raise ValueError(
            f"out {tuple(out.shape)} {out.dtype} is not [{tokens}, {heads}, "
            f"{_KV_LORA_RANK}] {q.dtype} with dense channels"
        )
    if lse is not None and (lse.shape != (tokens, heads) or lse.dtype != torch.float32):
        raise ValueError(f"lse must be FP32 [{tokens}, {heads}]")
    if tokens == 0:
        return out
    if tokens * heads * (_KV_LORA_RANK + _ROPE_DIM) >= 2**31:
        raise ValueError("the query exceeds the kernel's 32-bit buffer offsets")

    q_fp8 = torch.empty(
        (tokens * heads, _KV_LORA_RANK), dtype=torch.float8_e4m3fn, device=q.device
    )
    q_scale = torch.empty((tokens * heads, _GROUPS), dtype=torch.uint8, device=q.device)
    _mxfp8_query_kernel[(tokens * heads,)](
        q,
        q_fp8,
        q_scale,
        q.stride(0),
        q.stride(1),
        heads,
        kv_lora_rank=_KV_LORA_RANK,
        num_warps=1,
    )

    topk = topk_slots.shape[1]
    num_kv_splits = _num_kv_splits(tokens, heads, topk)
    if num_kv_splits == 1:
        stage_out = out
        stride_o_t, stride_o_h, stride_o_s = out.stride(0), out.stride(1), 0
        partial_lse = out if lse is None else lse
        stride_lse_t, stride_lse_h = (0, 0) if lse is None else lse.stride()
        stride_lse_s = 0
    else:
        stage_out = torch.empty(
            (tokens, heads, num_kv_splits, _KV_LORA_RANK),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (tokens, heads, num_kv_splits), dtype=torch.float32, device=q.device
        )
        stride_o_t, stride_o_h, stride_o_s = stage_out.stride()[:3]
        stride_lse_t, stride_lse_h, stride_lse_s = partial_lse.stride()

    _mxfp4_sparse_prefill_kernel[
        (tokens, triton.cdiv(heads, _BLOCK_H), num_kv_splits)
    ](
        q,
        q_fp8,
        q_scale,
        kv_cache,
        kv_cache.view(torch.bfloat16),
        topk_slots,
        topk_lens,
        stage_out,
        partial_lse,
        q.stride(0),
        q.stride(1),
        stride_o_t,
        stride_o_h,
        stride_o_s,
        stride_lse_t,
        stride_lse_h,
        stride_lse_s,
        topk_slots.stride(0),
        page_size,
        page_stride,
        float(softmax_scale),
        heads,
        num_kv_splits,
        TOPK=topk,
        ROW_BYTES=_ROW_BYTES,
        BLOCK_H=_BLOCK_H,
        TILE_K=_TILE_K,
        SPLIT_KV=num_kv_splits > 1,
        STORE_LSE=lse is not None and num_kv_splits == 1,
        num_warps=4,
        # Three CTAs per CU fit the LDS: 1.04 ms at 4096 Kimi TP8 rows on MI350X,
        # against 1.21 ms at two and 1.11 ms at four.
        waves_per_eu=3,
    )
    if num_kv_splits > 1:
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
            D_V=_KV_LORA_RANK,
            MAX_SPLITS=triton.next_power_of_2(num_kv_splits),
            STORE_LSE=lse is not None,
            num_warps=4,
        )
    return out
