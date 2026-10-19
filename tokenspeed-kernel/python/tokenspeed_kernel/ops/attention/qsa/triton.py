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

"""Triton cache-management and selection kernels for Qwen4-Exp QSA."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.dsa.triton import (
    triton_topk_from_logits,
)
from tokenspeed_kernel.platform import (
    CapabilityRequirement,
    current_platform,
    max_active_ctas,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_is_nvidia = current_platform().is_nvidia
_SCORE_MAX_WAVES = 3
LOG2E = math.log2(math.e)


@triton.jit
def _qwen4_exp_qsa_prepare_metadata_kernel(
    seq_lens,
    query_lengths,
    qsa_page_table,
    recent_page_table,
    positions,
    requests,
    qsa_locs,
    recent_locs,
    complete_blocks,
    draft_logical_positions,
    uniform_len,
    qsa_page_size,
    recent_page_size,
    stride_seq_b,
    stride_len_b,
    stride_qsa_pt_b,
    stride_recent_pt_b,
    stride_dlp_r,
    stride_dlp_s,
    HAS_UNIFORM: tl.constexpr,
    RESET_DRAFT_TAGS: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    RESET_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    request = tl.program_id(0)
    if HAS_UNIFORM:
        length = tl.full((), uniform_len, tl.int64)
        start = request.to(tl.int64) * uniform_len
    else:
        length = tl.load(query_lengths + request * stride_len_b).to(tl.int64)
        start = tl.zeros((), dtype=tl.int64)
        for other in range(0, request):
            start += tl.load(query_lengths + other * stride_len_b).to(tl.int64)

    if RESET_DRAFT_TAGS:
        reset_offsets = tl.arange(0, RESET_BLOCK)
        tl.store(
            draft_logical_positions
            + request * stride_dlp_r
            + reset_offsets * stride_dlp_s,
            tl.full((RESET_BLOCK,), -9223372036854775808, tl.int64),
            mask=reset_offsets < COMPRESS_RATIO,
        )

    base = tl.load(seq_lens + request * stride_seq_b).to(tl.int64) - length
    for offset in range(0, length, BLOCK):
        row_offsets = offset + tl.arange(0, BLOCK)
        mask = row_offsets < length
        rows = start + row_offsets
        logical = base + row_offsets
        valid = logical >= 0
        safe_logical = tl.maximum(logical, 0)

        tl.store(positions + rows, logical, mask=mask)
        tl.store(
            requests + rows,
            tl.full((BLOCK,), request, tl.int64),
            mask=mask,
        )
        tl.store(
            complete_blocks + rows,
            ((logical + 1) // COMPRESS_RATIO).to(tl.int32),
            mask=mask,
        )

        qsa_columns = safe_logical // qsa_page_size
        qsa_pages = tl.load(
            qsa_page_table + request * stride_qsa_pt_b + qsa_columns,
            mask=mask & valid,
            other=0,
        ).to(tl.int64)
        qsa_values = qsa_pages * qsa_page_size + safe_logical % qsa_page_size
        qsa_valid = valid & (qsa_pages > 0)
        tl.store(
            qsa_locs + rows,
            tl.where(qsa_valid, qsa_values, 0).to(tl.int32),
            mask=mask,
        )

        recent_columns = safe_logical // recent_page_size
        recent_pages = tl.load(
            recent_page_table + request * stride_recent_pt_b + recent_columns,
            mask=mask & valid,
            other=0,
        ).to(tl.int64)
        recent_values = (
            recent_pages * recent_page_size + safe_logical % recent_page_size
        )
        recent_valid = valid & (recent_pages > 0)
        tl.store(
            recent_locs + rows,
            tl.where(recent_valid, recent_values, 0).to(tl.int32),
            mask=mask,
        )


def qwen4_exp_qsa_prepare_metadata(
    seq_lens: torch.Tensor,
    query_lengths: torch.Tensor | int,
    total_tokens: int,
    qsa_page_table: torch.Tensor,
    qsa_page_size: int,
    recent_page_table: torch.Tensor,
    recent_page_size: int,
    compress_ratio: int,
    *,
    enable_pdl: bool,
    draft_logical_positions: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build all per-forward QSA row metadata in one launch.

    The kernel expands request lengths, maps both QSA cache groups, derives
    complete compressed-block counts, and optionally invalidates draft scratch
    tags in the same request CTA so the first MTP step needs no separate fill.

    Args:
        seq_lens: Sequence length per request.
        query_lengths: Per-request row counts or one uniform Python integer.
        total_tokens: Total flattened query rows.
        qsa_page_table: Raw compressed-cache block ids, with holes and padding
            normalized to zero; one entry per logical page.
        qsa_page_size: Logical tokens covered by a compressed page.
        recent_page_table: Raw recent-cache block ids in the same format.
        recent_page_size: Logical tokens covered by a recent page.
        compress_ratio: Raw tokens represented by one compressed key.
        enable_pdl: Enable NVIDIA programmatic dependent launch.
        draft_logical_positions: Optional request-local draft tags to reset.

    Returns:
        Logical positions, request ids, compressed-cache locations,
        recent-cache locations and complete-block counts, all with one entry
        per flattened query row.
    """

    batch = seq_lens.shape[0]
    device = seq_lens.device
    outputs = (
        torch.empty((total_tokens,), dtype=torch.int64, device=device),
        torch.empty((total_tokens,), dtype=torch.int64, device=device),
        torch.empty((total_tokens,), dtype=torch.int32, device=device),
        torch.empty((total_tokens,), dtype=torch.int32, device=device),
        torch.empty((total_tokens,), dtype=torch.int32, device=device),
    )
    if not total_tokens or not batch:
        return outputs
    if isinstance(query_lengths, int):
        uniform_len = int(query_lengths)
        lengths_arg = seq_lens
        stride_len_b = 0
        has_uniform = True
        if uniform_len * batch != total_tokens:
            raise ValueError("uniform QSA query length does not match total tokens")
    else:
        if query_lengths.shape[0] < batch:
            raise ValueError("QSA query lengths need one entry per request")
        uniform_len = 0
        lengths_arg = query_lengths
        stride_len_b = query_lengths.stride(0)
        has_uniform = False
    if draft_logical_positions is None:
        draft_arg = outputs[0]
        stride_dlp_r = stride_dlp_s = 0
    else:
        if draft_logical_positions.shape != (batch, compress_ratio):
            raise ValueError("QSA draft logical tags have invalid shape")
        draft_arg = draft_logical_positions
        stride_dlp_r = draft_logical_positions.stride(0)
        stride_dlp_s = draft_logical_positions.stride(1)
    _qwen4_exp_qsa_prepare_metadata_kernel[(batch,)](
        seq_lens,
        lengths_arg,
        qsa_page_table,
        recent_page_table,
        *outputs,
        draft_arg,
        uniform_len,
        qsa_page_size,
        recent_page_size,
        seq_lens.stride(0),
        stride_len_b,
        qsa_page_table.stride(0),
        recent_page_table.stride(0),
        stride_dlp_r,
        stride_dlp_s,
        HAS_UNIFORM=has_uniform,
        RESET_DRAFT_TAGS=draft_logical_positions is not None,
        COMPRESS_RATIO=compress_ratio,
        RESET_BLOCK=triton.next_power_of_2(compress_ratio),
        BLOCK=128,
        ENABLE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return outputs


@triton.jit
def _qwen4_exp_qsa_compress_and_store_kernel(
    token_k,
    query,
    query_norm_weight,
    query_output,
    logical_positions,
    request_indices,
    recent_locs,
    raw_cache,
    position_values,
    position_cache,
    norm_weight,
    cos_sin_cache,
    qsa_locs,
    compressed_cache,
    write_mask,
    draft_raw_cache,
    draft_logical_positions,
    draft_position_cache,
    staged_k,
    staged_positions,
    staged_logical,
    staged_recent,
    head_dim,
    rotary_dim,
    norm_epsilon,
    query_norm_epsilon,
    stage_verify_start,
    recent_page_size,
    compress_ratio,
    compressed_token_page_size,
    compressed_rows_per_page,
    section0,
    section1,
    section2,
    stride_k_n,
    stride_k_d,
    stride_q_n,
    stride_q_d,
    stride_qo_n,
    stride_qo_h,
    stride_raw_p,
    stride_raw_s,
    stride_raw_d,
    stride_pv_n,
    stride_pv_a,
    stride_pc_p,
    stride_cs_n,
    stride_cc_row,
    stride_cc_d,
    stride_drc_r,
    stride_drc_s,
    stride_drc_d,
    stride_dlp_r,
    stride_dlp_s,
    stride_dpc_r,
    stride_sk_r,
    stride_sk_s,
    stride_sk_d,
    stride_sp_r,
    stride_sl_r,
    stride_sl_s,
    COMPRESS_RATIO: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    HAS_QUERY: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    HAS_DRAFT: tl.constexpr,
    STAGE_VERIFY: tl.constexpr,
    STAGE_DRAFT: tl.constexpr,
    HAS_SECTIONS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Successors may prepare now; their waits still protect these outputs.
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    half = rotary_dim // 2
    half_offsets = tl.arange(0, BLOCK_HALF)
    half_mask = half_offsets < half
    second_offsets = half + half_offsets
    dim_offsets = tl.arange(0, BLOCK_D)
    pass_mask = (dim_offsets >= rotary_dim) & (dim_offsets < head_dim)
    position = tl.load(logical_positions + row).to(tl.int64)
    request = tl.load(request_indices + row).to(tl.int64)
    recent_loc = tl.load(recent_locs + row).to(tl.int64)
    page = tl.maximum(recent_loc, 0) // recent_page_size

    if HAS_QUERY:
        query_axes = tl.arange(0, 4)
        query_axis_mask = query_axes < 3
        query_positions = tl.load(
            position_values + row * stride_pv_n + query_axes * stride_pv_a,
            mask=query_axis_mask,
            other=0,
        )
        query_p0 = tl.sum(tl.where(query_axes == 0, query_positions, 0))
        query_p1 = tl.sum(tl.where(query_axes == 1, query_positions, 0))
        query_p2 = tl.sum(tl.where(query_axes == 2, query_positions, 0))
        if HAS_SECTIONS:
            if INTERLEAVED:
                query_axis = tl.where(
                    (half_offsets % 3 == 1) & (half_offsets < 3 * section1),
                    1,
                    tl.where(
                        (half_offsets % 3 == 2) & (half_offsets < 3 * section2),
                        2,
                        0,
                    ),
                )
            else:
                query_axis = tl.where(
                    half_offsets < section0,
                    0,
                    tl.where(half_offsets < section0 + section1, 1, 2),
                )
            query_selected_positions = tl.where(
                query_axis == 0,
                query_p0,
                tl.where(query_axis == 1, query_p1, query_p2),
            )
        else:
            query_selected_positions = tl.where(half_mask, query_p0, 0)
        query_cos = tl.load(
            cos_sin_cache + query_selected_positions * stride_cs_n + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        query_sin = tl.load(
            cos_sin_cache
            + query_selected_positions * stride_cs_n
            + half
            + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        query_weight_first = tl.load(
            query_norm_weight + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        query_weight_second = tl.load(
            query_norm_weight + second_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        query_weight_pass = tl.load(
            query_norm_weight + dim_offsets,
            mask=pass_mask,
            other=0.0,
        ).to(tl.float32)
        for head in tl.static_range(NUM_QUERY_HEADS):
            query_head = query + row * stride_q_n + head * head_dim * stride_q_d
            query_first = tl.load(
                query_head + half_offsets * stride_q_d,
                mask=half_mask,
                other=0.0,
            ).to(tl.float32)
            query_second = tl.load(
                query_head + second_offsets * stride_q_d,
                mask=half_mask,
                other=0.0,
            ).to(tl.float32)
            query_pass = tl.load(
                query_head + dim_offsets * stride_q_d,
                mask=pass_mask,
                other=0.0,
            ).to(tl.float32)
            query_squares = (
                tl.sum(query_first * query_first, axis=0)
                + tl.sum(query_second * query_second, axis=0)
                + tl.sum(query_pass * query_pass, axis=0)
            )
            query_scale = 1.0 / tl.sqrt(query_squares / head_dim + query_norm_epsilon)
            query_norm_first = query_first * query_scale * query_weight_first
            query_norm_second = query_second * query_scale * query_weight_second
            query_rotated_first = (
                query_norm_first * query_cos - query_norm_second * query_sin
            )
            query_rotated_second = (
                query_norm_second * query_cos + query_norm_first * query_sin
            )
            query_out = query_output + row * stride_qo_n + head * stride_qo_h
            query_out_dtype = query_output.dtype.element_ty
            tl.store(
                query_out + half_offsets,
                query_rotated_first.to(query_out_dtype),
                mask=half_mask,
            )
            tl.store(
                query_out + second_offsets,
                query_rotated_second.to(query_out_dtype),
                mask=half_mask,
            )
            tl.store(
                query_out + dim_offsets,
                (query_pass * query_scale * query_weight_pass).to(query_out_dtype),
                mask=pass_mask,
            )

    # Gather the compression group and average its raw keys. Members still
    # present in the current batch come from the input keys; older members
    # fall back to the raw cache page. The fused path must finish these
    # reads before any kernel writes new raw keys.
    acc_first = tl.zeros((BLOCK_HALF,), dtype=tl.float32)
    acc_second = tl.zeros((BLOCK_HALF,), dtype=tl.float32)
    acc_pass = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for member in tl.static_range(COMPRESS_RATIO):
        offset = COMPRESS_RATIO - 1 - member
        source = row - offset
        expected = position - offset
        from_current = source >= 0
        safe_source = tl.maximum(source, 0)
        if from_current:
            from_current = (
                tl.load(request_indices + source).to(tl.int64) == request
            ) & (tl.load(logical_positions + source).to(tl.int64) == expected)
        slot = (expected % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
        from_draft = False
        if HAS_DRAFT:
            from_draft = (
                (expected >= 0)
                & (not from_current)
                & (
                    tl.load(
                        draft_logical_positions
                        + request * stride_dlp_r
                        + slot * stride_dlp_s
                    ).to(tl.int64)
                    == expected
                )
            )
        current_base = token_k + safe_source * stride_k_n
        draft_base = draft_raw_cache + request * stride_drc_r + slot * stride_drc_s
        cached_base = raw_cache + page * stride_raw_p + slot * stride_raw_s
        current_mask = half_mask & from_current
        draft_mask = half_mask & from_draft
        cached_mask = half_mask & (not from_current) & (not from_draft)
        acc_first += tl.load(
            current_base + half_offsets * stride_k_d, mask=current_mask, other=0.0
        ).to(tl.float32)
        acc_first += tl.load(
            draft_base + half_offsets * stride_drc_d, mask=draft_mask, other=0.0
        ).to(tl.float32)
        acc_first += tl.load(
            cached_base + half_offsets * stride_raw_d, mask=cached_mask, other=0.0
        ).to(tl.float32)
        acc_second += tl.load(
            current_base + second_offsets * stride_k_d, mask=current_mask, other=0.0
        ).to(tl.float32)
        acc_second += tl.load(
            draft_base + second_offsets * stride_drc_d, mask=draft_mask, other=0.0
        ).to(tl.float32)
        acc_second += tl.load(
            cached_base + second_offsets * stride_raw_d, mask=cached_mask, other=0.0
        ).to(tl.float32)
        acc_pass += tl.load(
            current_base + dim_offsets * stride_k_d,
            mask=pass_mask & from_current,
            other=0.0,
        ).to(tl.float32)
        acc_pass += tl.load(
            draft_base + dim_offsets * stride_drc_d,
            mask=pass_mask & from_draft,
            other=0.0,
        ).to(tl.float32)
        acc_pass += tl.load(
            cached_base + dim_offsets * stride_raw_d,
            mask=pass_mask & (not from_current) & (not from_draft),
            other=0.0,
        ).to(tl.float32)
    pooled_first = acc_first / COMPRESS_RATIO
    pooled_second = acc_second / COMPRESS_RATIO
    pooled_pass = acc_pass / COMPRESS_RATIO

    # Gemma RMSNorm on the pooled key (weight already holds 1 + gamma).
    squares = (
        tl.sum(pooled_first * pooled_first, axis=0)
        + tl.sum(pooled_second * pooled_second, axis=0)
        + tl.sum(pooled_pass * pooled_pass, axis=0)
    )
    scale = 1.0 / tl.sqrt(squares / head_dim + norm_epsilon)
    weight_first = tl.load(norm_weight + half_offsets, mask=half_mask, other=0.0).to(
        tl.float32
    )
    weight_second = tl.load(norm_weight + second_offsets, mask=half_mask, other=0.0).to(
        tl.float32
    )
    weight_pass = tl.load(norm_weight + dim_offsets, mask=pass_mask, other=0.0).to(
        tl.float32
    )
    norm_first = pooled_first * scale * weight_first
    norm_second = pooled_second * scale * weight_second
    norm_pass = pooled_pass * scale * weight_pass

    # Group-start RoPE positions: current batch row or cached page header.
    head_source = row - (COMPRESS_RATIO - 1)
    head_expected = position - (COMPRESS_RATIO - 1)
    from_current_head = head_source >= 0
    safe_head_source = tl.maximum(head_source, 0)
    if from_current_head:
        from_current_head = (
            tl.load(request_indices + head_source).to(tl.int64) == request
        ) & (tl.load(logical_positions + head_source).to(tl.int64) == head_expected)
    head_slot = (head_expected % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
    from_draft_head = False
    if HAS_DRAFT:
        from_draft_head = (
            (head_expected >= 0)
            & (not from_current_head)
            & (
                tl.load(
                    draft_logical_positions
                    + request * stride_dlp_r
                    + head_slot * stride_dlp_s
                ).to(tl.int64)
                == head_expected
            )
        )
    axes = tl.arange(0, 4)
    axis_mask = axes < 3
    current_positions = tl.load(
        position_values + safe_head_source * stride_pv_n + axes * stride_pv_a,
        mask=axis_mask & from_current_head,
        other=0,
    )
    draft_positions = tl.load(
        draft_position_cache + request * stride_dpc_r + axes,
        mask=axis_mask & from_draft_head,
        other=0,
    )
    cached_positions = tl.load(
        position_cache + page * stride_pc_p + axes,
        mask=axis_mask & (not from_current_head) & (not from_draft_head),
        other=0,
    )
    first_positions = current_positions + draft_positions + cached_positions
    p0 = tl.sum(tl.where(axes == 0, first_positions, 0))
    p1 = tl.sum(tl.where(axes == 1, first_positions, 0))
    p2 = tl.sum(tl.where(axes == 2, first_positions, 0))

    # Neox-style rotation with optional multimodal section selection.
    if HAS_SECTIONS:
        if INTERLEAVED:
            axis = tl.where(
                (half_offsets % 3 == 1) & (half_offsets < 3 * section1),
                1,
                tl.where((half_offsets % 3 == 2) & (half_offsets < 3 * section2), 2, 0),
            )
        else:
            axis = tl.where(
                half_offsets < section0,
                0,
                tl.where(half_offsets < section0 + section1, 1, 2),
            )
        selected_positions = tl.where(axis == 0, p0, tl.where(axis == 1, p1, p2))
    else:
        selected_positions = tl.where(half_mask, p0, 0)
    cos = tl.load(
        cos_sin_cache + selected_positions * stride_cs_n + half_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + selected_positions * stride_cs_n + half + half_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    rotated_first = norm_first * cos - norm_second * sin
    rotated_second = norm_second * cos + norm_first * sin

    # Scatter at compression-group boundaries only.
    qsa_loc = tl.load(qsa_locs + row).to(tl.int64)
    boundary = ((position + 1) % compress_ratio == 0) & (qsa_loc > 0) & (recent_loc > 0)
    if HAS_WRITE_MASK:
        boundary &= tl.load(write_mask + row) != 0
    if boundary:
        compressed_page = qsa_loc // compressed_token_page_size
        within = qsa_loc % compressed_token_page_size
        target = (
            compressed_page * compressed_rows_per_page + within // compress_ratio
        ) * stride_cc_row
        out_dtype = compressed_cache.dtype.element_ty
        tl.store(
            compressed_cache + target + half_offsets * stride_cc_d,
            rotated_first.to(out_dtype),
            mask=half_mask,
        )
        tl.store(
            compressed_cache + target + second_offsets * stride_cc_d,
            rotated_second.to(out_dtype),
            mask=half_mask,
        )
        tl.store(
            compressed_cache + target + dim_offsets * stride_cc_d,
            norm_pass.to(out_dtype),
            mask=pass_mask,
        )
    if STAGE_VERIFY or STAGE_DRAFT:
        # Compression has consumed the old raw/draft ring before any staged
        # value can alias it. The PDL-dependent score kernel needs neither
        # staging destination; its wait still covers these producer stores.
        if STAGE_DRAFT:
            tl.debug_barrier()
        staged_values = tl.load(
            token_k + row * stride_k_n + dim_offsets * stride_k_d,
            mask=dim_offsets < head_dim,
            other=0.0,
        )
        staged_position_values = tl.load(
            position_values + row * stride_pv_n + axes * stride_pv_a,
            mask=axis_mask,
            other=0,
        )
        if STAGE_VERIFY:
            if row >= stage_verify_start:
                staged_row = row - stage_verify_start
                tl.store(
                    staged_k + staged_row * head_dim + dim_offsets,
                    staged_values.to(staged_k.dtype.element_ty),
                    mask=dim_offsets < head_dim,
                )
                tl.store(
                    staged_positions + staged_row * 3 + axes,
                    staged_position_values,
                    mask=axis_mask,
                )
                tl.store(staged_logical + staged_row, position)
                tl.store(staged_recent + staged_row, recent_loc.to(tl.int32))
        if STAGE_DRAFT and recent_loc > 0:
            staged_slot = (position % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
            tl.store(
                staged_k
                + request * stride_sk_r
                + staged_slot * stride_sk_s
                + dim_offsets * stride_sk_d,
                staged_values.to(staged_k.dtype.element_ty),
                mask=dim_offsets < head_dim,
            )
            tl.store(
                staged_logical + request * stride_sl_r + staged_slot * stride_sl_s,
                position,
            )
            if staged_slot == 0:
                tl.store(
                    staged_positions + request * stride_sp_r + axes,
                    staged_position_values,
                    mask=axis_mask,
                )


def qwen4_exp_qsa_compress_and_store(
    token_k: torch.Tensor,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    recent_locs: torch.Tensor,
    raw_cache: torch.Tensor,
    position_values: torch.Tensor,
    position_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_epsilon: float,
    cos_sin_cache: torch.Tensor,
    qsa_locs: torch.Tensor,
    compressed_cache: torch.Tensor,
    recent_page_size: int,
    compress_ratio: int,
    compressed_token_page_size: int,
    *,
    sections: tuple[int, ...] | None = None,
    interleaved: bool = False,
    write_mask: torch.Tensor | None = None,
    draft_raw_cache: torch.Tensor | None = None,
    draft_logical_positions: torch.Tensor | None = None,
    draft_position_cache: torch.Tensor | None = None,
    enable_pdl: bool = False,
    query: torch.Tensor | None = None,
    query_norm_weight: torch.Tensor | None = None,
    query_norm_epsilon: float | None = None,
    num_query_heads: int | None = None,
    stage_verify_buffers: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    ),
    stage_draft: bool,
) -> torch.Tensor | None:
    """Pool, normalize, rotate, and scatter compressed QSA keys in one kernel.

    Each row gathers its compression group from current-batch keys, optional
    speculative draft keys, or the committed raw-cache fallback. It averages
    the members, applies Gemma RMSNorm and neox RoPE at the group-start
    position, then scatters the result into the compressed cache when the row
    ends a compression group. The kernel only reads raw-key stores; raw-key
    writes must happen afterwards in a separate launch.

    Args:
        token_k: Raw indexer keys shaped ``[rows, 1, head_dim]``.
        logical_positions: Absolute logical position per row.
        request_indices: Owning request id per row.
        recent_locs: Recent-cache slots per row; non-positive means invalid.
        raw_cache: Raw key cache shaped ``[pages, compress_ratio, 1, head_dim]``.
        position_values: RoPE positions shaped ``[rows, 3]``.
        position_cache: Group-start RoPE positions shaped ``[pages, 3]``.
        norm_weight: Gemma RMSNorm scale shaped ``[head_dim]``, pre-offset
            by one.
        norm_epsilon: RMSNorm variance epsilon.
        cos_sin_cache: Fused cos/sin table shaped
            ``[max_positions, rotary_dim]``.
        qsa_locs: Compressed-cache slots per row; non-positive means invalid.
        compressed_cache: Compressed key cache shaped
            ``[pages, rows_per_page, 1, head_dim]``.
        recent_page_size: Rows covered by one raw-cache page.
        compress_ratio: Tokens grouped into one compressed block.
        compressed_token_page_size: Logical tokens covered by one
            compressed page.
        sections: Optional multimodal section sizes selecting the RoPE
            position axis per rotary half-dimension.
        interleaved: Whether multimodal sections are interleaved by axis.
        write_mask: Optional authoritative per-row compression write mask.
        draft_raw_cache: Optional request-local speculative raw keys shaped
            ``[requests, compress_ratio, 1, head_dim]``.
        draft_logical_positions: Exact logical position tags for
            ``draft_raw_cache``, shaped ``[requests, compress_ratio]``.
        draft_position_cache: Group-start RoPE positions for speculative
            draft groups, shaped ``[requests, 3]``.
        enable_pdl: Wait for input producers and allow the follow-up kernel to launch early
            on NVIDIA GPUs; the dependent kernel still waits for this grid.
        query: Optional raw projected queries shaped
            ``[rows, num_query_heads * head_dim]``. When present, query
            normalization and RoPE share this compression launch.
        query_norm_weight: Gemma RMSNorm weight for ``query``.
        query_norm_epsilon: RMSNorm epsilon for ``query``.
        num_query_heads: Query heads packed into each projected row.
        stage_verify_buffers: Contiguous K, position, logical-position,
            and recent-location destinations for target verification. Pass
            None explicitly when target verification staging is not needed.
        stage_draft: Store each row into the supplied request-local draft
            scratch after compression has consumed its previous contents.
            Pass False explicitly when draft staging is not needed.

    Returns:
        Normalized and rotated query rows when ``query`` is provided;
        otherwise ``None``.
    """

    rows = logical_positions.shape[0]
    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim % 2:
        raise ValueError("Qwen4-Exp QSA compression needs an even rotary dimension")
    head_dim = token_k.shape[-1]
    if rotary_dim > head_dim:
        raise ValueError("Qwen4-Exp QSA index head is narrower than the rotary dim")
    query_args = (
        query,
        query_norm_weight,
        query_norm_epsilon,
        num_query_heads,
    )
    has_query = any(value is not None for value in query_args)
    if has_query and not all(value is not None for value in query_args):
        raise ValueError("Qwen4-Exp QSA fused query preparation needs all arguments")
    if has_query:
        if (
            query.ndim != 2
            or query.shape != (rows, int(num_query_heads) * head_dim)
            or query_norm_weight.shape != (head_dim,)
        ):
            raise ValueError("Qwen4-Exp QSA fused query has invalid shapes")
        query_output = torch.empty(
            (rows, int(num_query_heads), head_dim),
            dtype=query.dtype,
            device=query.device,
        )
    else:
        query = token_k
        query_norm_weight = norm_weight
        query_norm_epsilon = 0.0
        num_query_heads = 0
        query_output = token_k
    if not rows:
        return query_output if has_query else None
    draft_buffers = (
        draft_raw_cache,
        draft_logical_positions,
        draft_position_cache,
    )
    has_draft = any(value is not None for value in draft_buffers)
    if has_draft and not all(value is not None for value in draft_buffers):
        raise ValueError("Qwen4-Exp QSA draft compression needs all scratch buffers")
    if has_draft:
        if (
            draft_raw_cache.ndim != 4
            or draft_raw_cache.shape[1:] != (compress_ratio, 1, head_dim)
            or draft_logical_positions.shape != draft_raw_cache.shape[:2]
            or draft_position_cache.shape != (draft_raw_cache.shape[0], 3)
        ):
            raise ValueError("Qwen4-Exp QSA draft scratch buffers have invalid shapes")
    else:
        draft_raw_cache = raw_cache
        draft_logical_positions = logical_positions
        draft_position_cache = position_cache
    if stage_verify_buffers is not None and stage_draft:
        raise ValueError("QSA cannot stage target verification and draft rows together")
    if stage_draft and not has_draft:
        raise ValueError("QSA draft staging requires all draft scratch buffers")
    if stage_verify_buffers is not None:
        staged_k, staged_positions, staged_logical, staged_recent = stage_verify_buffers
        staged_rows = staged_logical.numel()
        expected_numels = (
            staged_rows * head_dim,
            staged_rows * 3,
            staged_rows,
            staged_rows,
        )
        if staged_rows > rows:
            raise ValueError("QSA verify staging has more rows than its sources")
        for staged, expected_numel in zip(
            stage_verify_buffers, expected_numels, strict=True
        ):
            if staged.numel() != expected_numel or not staged.is_contiguous():
                raise ValueError("QSA verify staging buffers have invalid shapes")
        if (
            staged_k.dtype != token_k.dtype
            or staged_positions.dtype != position_values.dtype
            or staged_logical.dtype != logical_positions.dtype
            or staged_recent.dtype != recent_locs.dtype
        ):
            raise ValueError("QSA verify staging buffers have invalid dtypes")
    elif stage_draft:
        staged_rows = 0
        staged_k = draft_raw_cache
        staged_positions = draft_position_cache
        staged_logical = draft_logical_positions
        staged_recent = recent_locs
    else:
        staged_rows = 0
        staged_k = token_k
        staged_positions = position_values
        staged_logical = logical_positions
        staged_recent = recent_locs
    if write_mask is None:
        write_mask = recent_locs
    # ``token_k`` is usually a strided view of the QK projection output; the
    # kernel reads it through its strides directly, and index tensors are
    # cast in-kernel, so no host-side copy/cast kernel is launched.
    use_pdl = _is_nvidia and enable_pdl
    _qwen4_exp_qsa_compress_and_store_kernel[(rows,)](
        token_k,
        query,
        query_norm_weight,
        query_output,
        logical_positions,
        request_indices,
        recent_locs,
        raw_cache,
        position_values,
        position_cache,
        norm_weight,
        cos_sin_cache,
        qsa_locs,
        compressed_cache,
        write_mask,
        draft_raw_cache,
        draft_logical_positions,
        draft_position_cache,
        staged_k,
        staged_positions,
        staged_logical,
        staged_recent,
        head_dim,
        rotary_dim,
        float(norm_epsilon),
        float(query_norm_epsilon),
        rows - staged_rows,
        recent_page_size,
        compress_ratio,
        compressed_token_page_size,
        compressed_cache.shape[1],
        0 if sections is None else int(sections[0]),
        0 if sections is None else int(sections[1]),
        0 if sections is None else int(sections[2]),
        token_k.stride(0),
        token_k.stride(-1),
        query.stride(0),
        query.stride(-1),
        query_output.stride(0),
        query_output.stride(1),
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(-1),
        position_values.stride(0),
        position_values.stride(1),
        position_cache.stride(0),
        cos_sin_cache.stride(0),
        compressed_cache.stride(1),
        compressed_cache.stride(-1),
        draft_raw_cache.stride(0),
        draft_raw_cache.stride(1),
        draft_raw_cache.stride(-1),
        draft_logical_positions.stride(0),
        draft_logical_positions.stride(1) if has_draft else 0,
        draft_position_cache.stride(0),
        staged_k.stride(0),
        staged_k.stride(1),
        staged_k.stride(-1),
        staged_positions.stride(0),
        staged_logical.stride(0),
        staged_logical.stride(1) if staged_logical.ndim > 1 else 0,
        COMPRESS_RATIO=compress_ratio,
        NUM_QUERY_HEADS=int(num_query_heads),
        HAS_QUERY=has_query,
        HAS_WRITE_MASK=write_mask is not recent_locs,
        HAS_DRAFT=has_draft,
        STAGE_VERIFY=stage_verify_buffers is not None,
        STAGE_DRAFT=stage_draft,
        HAS_SECTIONS=sections is not None,
        INTERLEAVED=interleaved,
        BLOCK_HALF=triton.next_power_of_2(max(rotary_dim // 2, 1)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        ENABLE_PDL=use_pdl,
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return query_output if has_query else None


@triton.jit
def _qwen4_exp_qsa_recent_write_kernel(
    token_k,
    logical_positions,
    request_indices,
    recent_locs,
    position_values,
    raw_cache,
    position_cache,
    write_mask,
    rows,
    head_dim,
    recent_page_size,
    request_limit,
    stride_k_n,
    stride_k_d,
    stride_raw_p,
    stride_raw_s,
    stride_raw_d,
    stride_pv_n,
    stride_pv_a,
    stride_pc_p,
    COMPRESS_RATIO: tl.constexpr,
    HAS_EXTRA_MASK: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    loc = tl.load(recent_locs + row).to(tl.int64)
    if HAS_EXTRA_MASK:
        write = tl.load(write_mask + row) != 0
    else:
        write = loc > 0
    if write:
        # Keep only the last writer of every ring slot. The shadow check is
        # valid whenever rows are grouped by request in logical order; in
        # other shapes no later row can shadow the current one.
        future = row + COMPRESS_RATIO
        if future < rows:
            if HAS_EXTRA_MASK:
                has_future = tl.load(write_mask + future) != 0
            else:
                has_future = tl.load(recent_locs + future).to(tl.int64) > 0
            if has_future:
                has_future &= tl.load(request_indices + future).to(tl.int64) == tl.load(
                    request_indices + row
                ).to(tl.int64)
            if has_future:
                has_future &= (
                    tl.load(logical_positions + future).to(tl.int64)
                    == tl.load(logical_positions + row).to(tl.int64) + COMPRESS_RATIO
                )
            write &= not has_future
    if write and HAS_LIMIT:
        write &= tl.load(request_indices + row).to(tl.int64) < request_limit
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim
    values = tl.load(
        token_k + row * stride_k_n + dim_offsets * stride_k_d,
        mask=dim_mask & write,
        other=0.0,
    )
    if ENABLE_PDL:
        # Compression's incoming wait makes the preloaded inputs visible.
        # Wait for its old-ring reads before overwriting cache slots. Even
        # non-writing CTAs must order compression before releasing scoring.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if write:
        position = tl.load(logical_positions + row).to(tl.int64)
        slot = (position % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
        page = loc // recent_page_size
        tl.store(
            raw_cache
            + page * stride_raw_p
            + slot * stride_raw_s
            + dim_offsets * stride_raw_d,
            values.to(raw_cache.dtype.element_ty),
            mask=dim_mask,
        )
        if slot == 0:
            axes = tl.arange(0, 4)
            axis_mask = axes < 3
            rope_positions = tl.load(
                position_values + row * stride_pv_n + axes * stride_pv_a,
                mask=axis_mask,
                other=0,
            )
            tl.store(
                position_cache + page * stride_pc_p + axes,
                rope_positions,
                mask=axis_mask,
            )


def qwen4_exp_qsa_recent_write(
    token_k: torch.Tensor,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    recent_locs: torch.Tensor,
    position_values: torch.Tensor,
    raw_cache: torch.Tensor,
    position_cache: torch.Tensor,
    recent_page_size: int,
    compress_ratio: int,
    *,
    write_mask: torch.Tensor | None = None,
    request_limit: int | None = None,
    enable_pdl: bool = False,
) -> None:
    """Write the latest raw keys and group-start RoPE positions.

    Without ``write_mask`` the kernel keeps only the last writer of every
    ring slot and can additionally skip rows whose request id is not smaller
    than ``request_limit``. With ``write_mask`` the boolean mask selects the
    candidate rows; later masked writers of the same ring slot still shadow
    earlier ones so every physical slot receives exactly one deterministic
    write.

    Args:
        token_k: Raw indexer keys shaped ``[rows, 1, head_dim]``.
        logical_positions: Absolute logical position per row.
        request_indices: Owning request id per row.
        recent_locs: Recent-cache slots per row; non-positive means invalid.
        position_values: RoPE positions shaped ``[rows, 3]``.
        raw_cache: Raw key cache shaped ``[pages, compress_ratio, 1, head_dim]``.
        position_cache: Group-start RoPE positions shaped ``[pages, 3]``.
        recent_page_size: Rows covered by one raw-cache page.
        compress_ratio: Tokens grouped into one compressed block.
        write_mask: Optional authoritative per-row write mask.
        request_limit: Optional exclusive request-id write bound.
        enable_pdl: Wait on the compression kernel via programmatic dependent
            launch on NVIDIA GPUs. Compression must have waited for the producer
            of token keys and metadata before triggering this writer, allowing
            those inputs to preload while compression finishes its ring reads.

    Returns:
        None.
    """

    rows = logical_positions.shape[0]
    if not rows:
        return
    # Strided ``token_k`` reads and in-kernel index casts avoid host-side
    # copy/cast kernels before the launch.
    head_dim = token_k.shape[-1]
    if write_mask is None:
        write_mask = recent_locs
    use_pdl = _is_nvidia and enable_pdl
    _qwen4_exp_qsa_recent_write_kernel[(rows,)](
        token_k,
        logical_positions,
        request_indices,
        recent_locs,
        position_values,
        raw_cache,
        position_cache,
        write_mask,
        rows,
        head_dim,
        recent_page_size,
        -1 if request_limit is None else request_limit,
        token_k.stride(0),
        token_k.stride(-1),
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(-1),
        position_values.stride(0),
        position_values.stride(1),
        position_cache.stride(0),
        COMPRESS_RATIO=compress_ratio,
        HAS_EXTRA_MASK=write_mask is not recent_locs,
        HAS_LIMIT=request_limit is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        ENABLE_PDL=use_pdl,
        **({"launch_pdl": True} if use_pdl else {}),
    )


@triton.jit
def _qwen4_exp_qsa_verify_row_mask(
    logical_positions,
    accepted_lengths,
    row,
    verify_width,
    compress_ratio,
):
    """Return whether one staged row belongs to the accepted trailing window."""

    request = row // verify_width
    step = row % verify_width
    accepted = tl.load(accepted_lengths + request).to(tl.int64)
    accepted = tl.minimum(tl.maximum(accepted, 0), verify_width)
    write = step < accepted
    if write:
        last_index = request * verify_width + accepted - 1
        last_position = tl.load(logical_positions + last_index).to(tl.int64)
        position = tl.load(logical_positions + row).to(tl.int64)
        write = position > last_position - compress_ratio
    return write


@triton.jit
def _qwen4_exp_qsa_commit_verify_layers_kernel(
    raw_addresses,
    position_addresses,
    staged_k,
    logical_positions,
    recent_locs,
    position_values,
    accepted_lengths,
    head_dim,
    recent_page_size,
    verify_width,
    stride_k_l,
    stride_pv_n,
    stride_pv_a,
    stride_raw_p,
    stride_raw_s,
    stride_raw_d,
    stride_pc_p,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    # The accepted trailing window has at most one writer per ring slot.
    write = _qwen4_exp_qsa_verify_row_mask(
        logical_positions, accepted_lengths, row, verify_width, COMPRESS_RATIO
    )
    loc = tl.load(recent_locs + row).to(tl.int64)
    write &= loc > 0
    if write:
        position = tl.load(logical_positions + row).to(tl.int64)
        slot = (position % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
        page = loc // recent_page_size
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < head_dim
        values = tl.load(
            staged_k + layer.to(tl.int64) * stride_k_l + row * head_dim + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        raw_ptr = tl.cast(tl.load(raw_addresses + layer), tl.pointer_type(tl.bfloat16))
        tl.store(
            raw_ptr
            + page * stride_raw_p
            + slot * stride_raw_s
            + dim_offsets * stride_raw_d,
            values.to(tl.bfloat16),
            mask=dim_mask,
        )
        if slot == 0:
            axes = tl.arange(0, 4)
            axis_mask = axes < 3
            rope_positions = tl.load(
                position_values + row * stride_pv_n + axes * stride_pv_a,
                mask=axis_mask,
                other=0,
            )
            position_ptr = tl.cast(
                tl.load(position_addresses + layer), tl.pointer_type(tl.int64)
            )
            tl.store(
                position_ptr + page * stride_pc_p + axes,
                rope_positions,
                mask=axis_mask,
            )


def qwen4_exp_qsa_commit_verify_layers(
    raw_addresses: torch.Tensor,
    position_addresses: torch.Tensor,
    staged_k: torch.Tensor,
    logical_positions: torch.Tensor,
    recent_locs: torch.Tensor,
    position_values: torch.Tensor,
    accepted_lengths: torch.Tensor,
    raw_cache: torch.Tensor,
    position_cache: torch.Tensor,
    recent_page_size: int,
    compress_ratio: int,
    *,
    verify_width: int,
) -> None:
    """Commit accepted target-verify raw keys into every QSA layer at once.

    The layer-dependent keys arrive as one contiguous layer-major staging
    tensor. Logical positions, recent-cache locations, and RoPE positions are
    shared by every layer; destination fields are supplied as address tables.

    Args:
        raw_addresses: CUDA uint64 base address for each raw-key cache field.
        position_addresses: CUDA uint64 base address for each position field.
        staged_k: Keys shaped ``[layers, capacity, width, 1, head_dim]`` in
            the model's floating-point dtype. Loads use that dtype and stores
            convert to the raw-key cache's fixed bfloat16 format.
        logical_positions: Consecutive logical positions within each request,
            in request-major order.
        recent_locs: Recent-cache locations for live rows.
        position_values: RoPE positions shaped ``[rows, 3]``.
        accepted_lengths: Accepted width for each request.
        raw_cache: One raw-key field used as the stride and dtype donor.
        position_cache: One position field used as the stride and dtype donor.
        recent_page_size: Rows covered by one recent-cache page.
        compress_ratio: Tokens grouped into one compressed entry.
        verify_width: Candidate width per request.

    Returns:
        None.
    """

    num_layers = raw_addresses.numel()
    rows = logical_positions.shape[0]
    if num_layers == 0 or not rows:
        return
    if position_addresses.numel() != num_layers:
        raise ValueError(
            "Qwen4-Exp QSA batched verify commit needs one address per layer "
            "in both destination tables"
        )
    if raw_addresses.dtype != torch.uint64 or position_addresses.dtype != torch.uint64:
        raise ValueError("QSA cache field address tables must have dtype torch.uint64")
    if staged_k.shape[0] != num_layers:
        raise ValueError(
            "QSA staged keys must hold exactly one layer block per address"
        )
    if verify_width < 1 or rows % verify_width:
        raise ValueError("QSA verify rows must be a positive multiple of verify_width")
    if accepted_lengths.shape[0] != rows // verify_width:
        raise ValueError(
            "QSA batched verify commits need one accepted length per request"
        )
    head_dim = staged_k.shape[-1]
    if staged_k.ndim != 5 or tuple(staged_k.shape[2:]) != (
        verify_width,
        1,
        head_dim,
    ):
        raise ValueError(
            "QSA staged keys must be shaped "
            "[num_layers, bucket_rows, verify_width, 1, head_dim]"
        )
    if not staged_k.is_contiguous():
        raise ValueError(
            "QSA staged keys must be contiguous so one layer stride addresses "
            "every row"
        )
    if rows > staged_k.shape[1] * verify_width:
        raise ValueError(
            "QSA staged keys bucket covers fewer rows than this verify commit "
            f"needs: {staged_k.shape[1]} x {verify_width} for {rows} rows"
        )
    if raw_cache.dtype != torch.bfloat16:
        raise ValueError("QSA raw-key cache fields must be bfloat16")
    if position_cache.dtype != torch.int64:
        raise ValueError("QSA RoPE position cache fields must be int64")
    if raw_cache.shape[-1] != head_dim or raw_cache.shape[1] != compress_ratio:
        raise ValueError(
            "QSA raw-key cache geometry disagrees with the staged keys or the "
            "compression ratio"
        )
    if position_values.shape[0] != rows or recent_locs.shape[0] != rows:
        raise ValueError(
            "QSA batched verify commit needs one shared row per staged row"
        )

    _qwen4_exp_qsa_commit_verify_layers_kernel[(rows, num_layers)](
        raw_addresses,
        position_addresses,
        staged_k,
        logical_positions,
        recent_locs,
        position_values,
        accepted_lengths,
        head_dim,
        recent_page_size,
        verify_width,
        staged_k.stride(0),
        position_values.stride(0),
        position_values.stride(1),
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(-1),
        position_cache.stride(0),
        COMPRESS_RATIO=compress_ratio,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


@triton.jit
def _qwen4_exp_qsa_compare_keys(
    keys,
    QUERY_GROUP_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    STAGE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    # Split comparison pairs instead of XOR-reducing a sorting hypercube.
    # This avoids redundant layout conversions and shared-memory barriers.
    pairs = tl.reshape(keys, (QUERY_GROUP_SIZE, WIDTH // (2 * STRIDE), 2, STRIDE))
    left, right = tl.split(tl.permute(pairs, (0, 1, 3, 2)))
    low = tl.minimum(left, right)
    high = tl.maximum(left, right)
    if STAGE == WIDTH.bit_length() - 1:
        descending = DESCENDING
    else:
        descending = (
            (tl.arange(0, WIDTH // (2 * STRIDE)) * (2 * STRIDE)) >> STAGE
        ) & 1 != 0
        descending = descending[None, :, None]
    pairs = tl.join(tl.where(descending, high, low), tl.where(descending, low, high))
    return tl.reshape(tl.permute(pairs, (0, 1, 3, 2)), (QUERY_GROUP_SIZE, WIDTH))


@triton.jit
def _qwen4_exp_qsa_sort_keys(
    keys,
    QUERY_GROUP_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL_SORT: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    first_stage: tl.constexpr = 1 if FULL_SORT else WIDTH.bit_length() - 1
    for stage in tl.static_range(first_stage, WIDTH.bit_length()):
        for step in tl.static_range(stage - 1, -1, -1):
            keys = _qwen4_exp_qsa_compare_keys(
                keys, QUERY_GROUP_SIZE, WIDTH, 1 << step, stage, DESCENDING
            )
    return keys


@triton.jit
def _qwen4_exp_qsa_pad_keys_to_topk(
    keys,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
):
    """Append zeros to a sorted tile while preserving its descending order."""
    for stage in tl.static_range((BLOCK_TOPK // BLOCK_N).bit_length() - 1):
        keys = tl.cat(
            keys,
            tl.full((QUERY_GROUP_SIZE, BLOCK_N << stage), 0, tl.int64),
            dim=1,
        )
    return keys


@triton.jit
def _qwen4_exp_qsa_stream_block_topk_kernel(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    selected_blocks,
    num_heads,
    head_dim,
    num_blocks,
    page_size,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    stride_o_n,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0) * QUERY_GROUP_SIZE
    query_offsets = tl.arange(0, QUERY_GROUP_SIZE)
    complete = tl.maximum(
        0, tl.minimum(num_blocks, tl.load(complete_blocks + row + query_offsets))
    )
    block_end = tl.max(complete, axis=0)
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    output = (
        selected_blocks
        + (row + query_offsets[:, None]) * stride_o_n
        + topk_offsets[None, :]
    )
    if block_end <= BLOCK_TOPK:
        # All complete blocks qualify; neither scoring nor sorting is needed.
        tl.store(
            output,
            tl.where(
                topk_offsets[None, :] < complete[:, None], topk_offsets[None, :], -1
            ),
        )
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return
    # Explicit uniform request runs ensure every query in the group shares K.
    request = tl.load(request_indices + row).to(tl.int64)
    head_offsets = tl.arange(0, BLOCK_H * QUERY_GROUP_SIZE)
    head_mask = head_offsets % BLOCK_H < num_heads
    acc = tl.zeros((QUERY_GROUP_SIZE, BLOCK_TOPK), dtype=tl.int64)
    cutoff = tl.zeros((QUERY_GROUP_SIZE,), dtype=tl.int64)
    for tile_start in range(0, block_end, BLOCK_N):
        block_ids = tile_start + tl.arange(0, BLOCK_N)
        tile_mask = block_ids < block_end
        pages = tl.load(
            page_table + request * stride_pt_b + block_ids // page_size,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)
        slots = pages * page_size + block_ids % page_size
        scores = tl.zeros((BLOCK_H * QUERY_GROUP_SIZE, BLOCK_N), tl.float32)
        # A narrow reduction tile bounds K shared memory independently of the
        # model's head dimension, without a full-key or query copy.
        for dim_start in range(0, head_dim, BLOCK_D):
            dim_offsets = dim_start + tl.arange(0, BLOCK_D)
            dim_mask = dim_offsets < head_dim
            q = tl.load(
                query
                + (row + head_offsets[:, None] // BLOCK_H) * stride_q_n
                + (head_offsets[:, None] % BLOCK_H) * stride_q_h
                + dim_offsets[None, :] * stride_q_d,
                mask=head_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            keys = tl.load(
                key_cache
                + slots[None, :] * stride_k_n
                + dim_offsets[:, None] * stride_k_d,
                mask=tile_mask[None, :] & dim_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(
                q, keys, acc=scores, out_dtype=tl.float32, input_precision="ieee"
            )
        scores = tl.sum(
            tl.reshape(
                tl.where(head_mask[:, None], tl.maximum(scores, 0.0), 0.0),
                (QUERY_GROUP_SIZE, BLOCK_H, BLOCK_N),
            ),
            axis=1,
        )
        # Non-negative FP32 scores and full logical ids form exact ordered
        # keys. Adding one distinguishes valid zero scores from padding.
        score_bits = scores.to(tl.int32, bitcast=True)
        packed = ((score_bits.to(tl.int64) + 1) << 32) | block_ids[None, :].to(tl.int64)
        packed = tl.where(block_ids[None, :] < complete[:, None], packed, 0)
        if tl.sum((tl.max(packed, axis=1) > cutoff).to(tl.int32), axis=0) > 0:
            acc = _qwen4_exp_qsa_sort_keys(
                acc, QUERY_GROUP_SIZE, BLOCK_TOPK, FULL_SORT=False, DESCENDING=False
            )
            packed = _qwen4_exp_qsa_sort_keys(
                packed, QUERY_GROUP_SIZE, BLOCK_N, FULL_SORT=True, DESCENDING=True
            )
            padded = _qwen4_exp_qsa_pad_keys_to_topk(
                packed, BLOCK_TOPK, BLOCK_N, QUERY_GROUP_SIZE
            )
            # Ascending old keys and descending new keys retain the exact top-k
            # under an elementwise maximum. Their result is again bitonic.
            acc = tl.maximum(acc, padded)
            cutoff = tl.min(acc, axis=1)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    block_ids = tl.where(acc != 0, (acc & 0xFFFFFFFF).to(tl.int32), -1)
    tl.store(output, block_ids)


@triton.jit(do_not_specialize=["num_query_groups"])
def _qwen4_exp_qsa_score_blocks_kernel(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    logits,
    num_heads,
    head_dim,
    num_blocks,
    num_query_groups,
    page_size,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    stride_l_n,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    LOOP_QUERY_GROUPS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Successors may prepare now; their waits still protect these outputs.
        tl.extra.cuda.gdc_launch_dependents()
    group_start = tl.program_id(0)
    group_step = tl.num_programs(0)
    if LOOP_QUERY_GROUPS:
        group_iterations = tl.cdiv(num_query_groups - group_start, group_step)
    else:
        group_iterations = 1
    for iteration in range(group_iterations):
        group = group_start + iteration * group_step
        row = group * QUERY_GROUP_SIZE
        tile = tl.program_id(1)
        # Every group stays within one request, but each query keeps its own
        # complete-block frontier and output row.
        request = tl.load(request_indices + row).to(tl.int64)
        query_offsets = tl.arange(0, QUERY_GROUP_SIZE)
        complete = tl.load(complete_blocks + row + query_offsets).to(tl.int64)
        max_complete = tl.minimum(tl.max(complete, axis=0), num_blocks)
        # The grid stays bounded during graph replay. Each CTA walks only its
        # valid tiles, reusing Q when a request outgrows the launch width.
        if tile * BLOCK_N < max_complete:
            head_offsets = tl.arange(0, BLOCK_H * QUERY_GROUP_SIZE)
            dim_offsets = tl.arange(0, BLOCK_D)
            head_mask = head_offsets % BLOCK_H < num_heads
            dim_mask = dim_offsets < head_dim
            q = tl.load(
                query
                + (row + head_offsets[:, None] // BLOCK_H) * stride_q_n
                + (head_offsets[:, None] % BLOCK_H) * stride_q_h
                + dim_offsets[None, :] * stride_q_d,
                mask=head_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            for start in range(
                tile * BLOCK_N, max_complete, tl.num_programs(1) * BLOCK_N
            ):
                block_ids = start + tl.arange(0, BLOCK_N)
                valid = block_ids < max_complete
                columns = block_ids // page_size
                offsets = block_ids % page_size
                pages = tl.load(
                    page_table + request * stride_pt_b + columns,
                    mask=valid,
                    other=0,
                ).to(tl.int64)
                slots = pages * page_size + offsets
                keys = tl.load(
                    key_cache
                    + slots[None, :] * stride_k_n
                    + dim_offsets[:, None] * stride_k_d,
                    mask=valid[None, :] & dim_mask[:, None],
                    other=0.0,
                )
                scores = tl.dot(q, keys, out_dtype=tl.float32)
                scores = tl.maximum(scores, 0.0)
                scores = tl.sum(
                    tl.reshape(
                        tl.where(head_mask[:, None], scores, 0.0),
                        (QUERY_GROUP_SIZE, BLOCK_H, BLOCK_N),
                    ),
                    axis=1,
                )
                tl.store(
                    logits
                    + (row + query_offsets[:, None]) * stride_l_n
                    + block_ids[None, :],
                    scores,
                    mask=valid[None, :] & (block_ids[None, :] < complete[:, None]),
                )


def triton_qsa_block_topk_stream(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    queries_per_request: int | None,
    max_partial_bytes: int,
    enable_pdl: bool,
) -> torch.Tensor:
    """Fused streaming path: score tiles and fold them into a running top-k.

    Streams each row's page-table blocks tile by tile, scores them against
    the row's query, and maintains a packed-key running top-k, so neither
    the expanded slot matrix nor the full score matrix is materialized.
    One program shares K across a group within an explicit uniform request
    run, retains each row's frontier, and writes final ids directly.
    Only the output is allocated; the running set stays inside the program.

    Args:
        query: Queries shaped ``[rows, heads, head_dim]``.
        key_cache: Compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Physical pages with contiguous columns.
        request_indices: Contiguous owning request id per row.
        complete_blocks: Contiguous complete-block count per row.
        page_size: Compressed blocks per cache page.
        block_topk: Power-of-two selection width, at least 64.
        queries_per_request: Uniform request-run width or ``None``. The
            portable kernel groups a divisor of this width to share K.
            Ragged runs use one row per program.
        max_partial_bytes: Non-negative global scratch budget. This
            implementation needs no scratch, including with a zero budget.
        enable_pdl: Explicit programmatic dependent launch ordering.

    Returns:
        Int32 block ids shaped ``[rows, block_topk]``, padded with ``-1``.
    """

    if max_partial_bytes < 0:
        raise ValueError("QSA streaming scratch budget must be non-negative")
    rows = query.shape[0]
    if queries_per_request is not None and (
        queries_per_request < 1 or rows % queries_per_request
    ):
        raise ValueError("QSA queries_per_request must be positive and divide rows")
    num_blocks = page_table.shape[1] * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )
    selected = torch.empty((rows, block_topk), dtype=torch.int32, device=query.device)
    # Four-row groups reuse K while keeping selection registers bounded.
    query_group = math.gcd(queries_per_request or 1, 4)
    block_n = min(512, block_topk)
    num_warps = 4 if query_group == 1 else 8
    # PDL overlaps successor setup with scoring; each consumer waits before
    # accessing the preceding kernel's results.
    use_pdl = _is_nvidia and enable_pdl
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    # Strided query reads keep the GEMM-view input copy-free.
    _qwen4_exp_qsa_stream_block_topk_kernel[(rows // query_group,)](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        selected,
        query.shape[1],
        query.shape[2],
        num_blocks,
        page_size,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        selected.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_N=block_n,
        BLOCK_H=triton.next_power_of_2(query.shape[1]),
        BLOCK_D=32,
        QUERY_GROUP_SIZE=query_group,
        ENABLE_PDL=use_pdl,
        num_warps=num_warps,
        num_stages=1,
        **pdl_kwargs,
    )
    return selected


def _score_grid(
    kernel, query_groups: int, num_tiles: int, sm_count: int
) -> tuple[int, int, int]:
    # Use the loaded specialization's occupancy so resource changes preserve
    # the wave limit. Round down to keep the whole grid within the budget.
    max_ctas = _SCORE_MAX_WAVES * sm_count * max_active_ctas(kernel)
    group_programs = min(query_groups, max_ctas)
    tile_programs = min(num_tiles, max_ctas // group_programs)
    return group_programs, tile_programs, 1


def triton_qsa_block_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    queries_per_request: int | None,
    enable_pdl: bool,
) -> torch.Tensor:
    """Portable QSA block scoring; see ``qsa_block_scores``."""

    rows = query.shape[0]
    num_blocks = page_table.shape[1] * page_size
    use_pdl = _is_nvidia and enable_pdl
    logits = torch.empty((rows, num_blocks), dtype=torch.float32, device=query.device)
    if rows == 0 or num_blocks == 0:
        return logits
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    block_h = triton.next_power_of_2(query.shape[1])
    # Long uniform runs reuse K across 32 queries. With four index heads this
    # exposes a 128-row dot to Blackwell tensor cores instead of padding a
    # four-row dot to mma.sync's 16-row instruction. Small/ragged batches keep
    # their latency-oriented grouping; every group stays within one request.
    max_query_group = max(1, min(4, 16 // block_h))
    if _is_nvidia and rows >= 1024 and block_h == 4 and query.shape[2] == 128:
        max_query_group = 32
    query_group_size = math.gcd(queries_per_request or 1, max_query_group)
    # The old 512-block tile used 129 KiB shared memory for 4x128 BF16 heads,
    # limiting a B300 SM to one CTA. 128 blocks leave room for concurrent CTAs.
    block_n = 128
    query_groups = rows // query_group_size
    sm_count = current_platform().sm_count
    args = (
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        logits,
        query.shape[1],
        query.shape[2],
        num_blocks,
        query_groups,
        page_size,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        logits.stride(0),
    )
    constants = {
        "BLOCK_N": block_n,
        "BLOCK_H": block_h,
        "BLOCK_D": triton.next_power_of_2(query.shape[2]),
        "QUERY_GROUP_SIZE": query_group_size,
        # Two buckets bound specialization. The small bucket fits even at
        # one resident CTA/SM and folds away the outer loop during decode.
        "LOOP_QUERY_GROUPS": query_groups > _SCORE_MAX_WAVES * sm_count,
        "ENABLE_PDL": use_pdl,
    }
    # Warmup performs the usual JIT cache lookup without launching. Launch the
    # same compiled kernel after sizing the grid; no device lengths are read.
    kernel = _qwen4_exp_qsa_score_blocks_kernel.warmup(
        *args,
        **constants,
        grid=(1,),
        num_warps=4,
        num_stages=2,
        **pdl_kwargs,
    )
    grid = _score_grid(
        kernel,
        query_groups,
        triton.cdiv(num_blocks, block_n),
        sm_count,
    )
    kernel[grid](*args, *constants.values())
    return logits


def triton_qsa_block_topk_select(
    logits: torch.Tensor,
    topk: int,
    *,
    lengths: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Length-aware Triton selection; see ``qsa_block_topk_select``."""
    return triton_topk_from_logits(
        logits, topk, seq_lens=lengths, enable_pdl=_is_nvidia and enable_pdl
    )


_FP32_SCORES = format_signature(logits=dense_tensor_format(torch.float32))

register_kernel(
    "attention",
    "qsa_block_scores",
    name="triton_qsa_block_scores",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                query=dense_tensor_format(dtype),
                key_cache=dense_tensor_format(dtype),
            )
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
        }
    ),
    priority=Priority.PORTABLE,
    features=frozenset({"logits"}),
)(triton_qsa_block_scores)

register_kernel(
    "attention",
    "qsa_block_topk_stream",
    name="triton_qsa_block_topk_stream",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                query=dense_tensor_format(dtype),
                key_cache=dense_tensor_format(dtype),
            )
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
        }
    ),
    priority=Priority.PORTABLE,
    features=frozenset({"stream"}),
)(triton_qsa_block_topk_stream)

register_kernel(
    "attention",
    "qsa_block_topk_select",
    name="triton_qsa_block_topk_select",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({_FP32_SCORES}),
    priority=Priority.PORTABLE,
    features=frozenset({"logits"}),
)(triton_qsa_block_topk_select)


@triton.jit
def _qwen4_exp_qsa_selected_slots_kernel(
    selected_blocks,
    complete_blocks,
    logical_positions,
    request_indices,
    page_table,
    selected_slots,
    stride_b_n,
    stride_pt_b,
    stride_o_n,
    TOKEN_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < WIDTH
    complete = tl.load(complete_blocks + row).to(tl.int64)
    is_token = columns < TOKEN_TOPK
    block_ids = tl.load(
        selected_blocks + row * stride_b_n + columns // COMPRESS_RATIO,
        mask=mask & is_token,
        other=-1,
    ).to(tl.int64)
    token_ok = (block_ids >= 0) & (block_ids < complete)
    token_values = block_ids * COMPRESS_RATIO + columns % COMPRESS_RATIO
    position = tl.load(logical_positions + row).to(tl.int64)
    suffix_values = complete * COMPRESS_RATIO + (columns - TOKEN_TOPK)
    suffix_ok = suffix_values <= position
    logical_tokens = tl.where(
        is_token,
        tl.where(token_ok, token_values, -1),
        tl.where(suffix_ok, suffix_values, -1),
    )
    # Keep the final causal guard local to the physical mapping. Besides
    # protecting externally supplied block lists, this prevents a stale
    # complete-block count from exposing later rows during MTP compaction.
    valid = (logical_tokens >= 0) & (logical_tokens <= position)
    safe_tokens = tl.maximum(logical_tokens, 0)
    page_columns = safe_tokens // PAGE_SIZE
    request = tl.load(request_indices + row).to(tl.int64)
    pages = tl.load(
        page_table + request * stride_pt_b + page_columns,
        mask=mask & valid,
        other=0,
    ).to(tl.int64)
    slots = pages * PAGE_SIZE + safe_tokens % PAGE_SIZE
    tl.store(
        selected_slots + row * stride_o_n + columns,
        tl.where(valid & (pages > 0), slots, -1).to(tl.int32),
        mask=mask,
    )


def qwen4_exp_qsa_selected_slots(
    selected_blocks: torch.Tensor,
    complete_blocks: torch.Tensor,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    compress_ratio: int,
    token_topk: int,
    *,
    enable_pdl: bool,
) -> torch.Tensor:
    """Expand selected blocks directly into physical full-attention slots.

    Args:
        selected_blocks: Top-k block ids shaped ``[rows, block_topk]``;
            negative entries are invalid.
        complete_blocks: Fully compressed block counts shaped ``[rows]``.
        logical_positions: Absolute logical position per row.
        request_indices: Owning request id per row.
        page_table: Full-attention page table shaped
            ``[requests, max_pages]``.
        page_size: Tokens covered by one full-attention page-table entry.
        compress_ratio: Tokens grouped into one compressed block.
        token_topk: Number of block-derived tokens kept per row.
        enable_pdl: Enable NVIDIA programmatic dependent launch.

    Returns:
        Int32 physical cache slots shaped
        ``[rows, token_topk + compress_ratio - 1]``; invalid entries are
        ``-1`` and the trailing columns cover the in-progress compression
        group.
    """

    if selected_blocks.ndim != 2:
        raise ValueError("Qwen4-Exp QSA selected blocks must be rank two")
    rows = selected_blocks.shape[0]
    if complete_blocks.shape != (rows,):
        raise ValueError("complete_blocks must have one entry per selected row")
    if logical_positions.shape != (rows,):
        raise ValueError("logical_positions must have one entry per selected row")
    if request_indices.shape != (rows,):
        raise ValueError("request_indices must have one entry per selected row")
    if page_table.ndim != 2:
        raise ValueError("Qwen4-Exp QSA full page table must be rank two")
    if page_size <= 0:
        raise ValueError("Qwen4-Exp QSA full page size must be positive")
    width = token_topk + compress_ratio - 1
    # The kernel stores every column including the -1 invalid entries, so
    # the buffer starts uninitialized.
    output = torch.empty(
        (rows, width), dtype=torch.int32, device=selected_blocks.device
    )
    if rows:
        _qwen4_exp_qsa_selected_slots_kernel[(rows, triton.cdiv(width, 256))](
            selected_blocks,
            complete_blocks,
            logical_positions,
            request_indices,
            page_table,
            output,
            selected_blocks.stride(0),
            page_table.stride(0),
            output.stride(0),
            TOKEN_TOPK=token_topk,
            COMPRESS_RATIO=compress_ratio,
            PAGE_SIZE=page_size,
            WIDTH=width,
            BLOCK=256,
            ENABLE_PDL=enable_pdl,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return output


# ===-----------------------------------------------------------------------===#
# Sparse attention over selected slots
# ===-----------------------------------------------------------------------===#

# Selected slots attended per inner-loop step.
_SPARSE_BLOCK_N = 32
# Target number of (row, kv_head, split) programs. Small decode batches split
# each row's selected slots across programs and merge the partials afterwards.
_SPARSE_TARGET_PROGRAMS = 512
_SPARSE_MAX_SPLITS = 16


@triton.jit
def _triton_qsa_sparse_attention_kernel(
    q,
    k_cache,
    v_cache,
    selected_slots,
    output,
    partial_out,
    partial_lse,
    scale_log2,
    k_scale,
    v_scale,
    width,
    split_len,
    num_splits,
    stride_q_n,
    stride_q_h,
    stride_k_n,
    stride_k_h,
    stride_v_n,
    stride_v_h,
    stride_s_n,
    stride_o_n,
    stride_o_h,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    heads = tl.arange(0, GROUP_PAD)
    head_mask = heads < GROUP
    q_heads = kv_head * GROUP + heads
    d = tl.arange(0, HEAD_DIM)
    dv = tl.arange(0, VALUE_DIM)
    q_tile = tl.load(
        q + row * stride_q_n + q_heads[:, None] * stride_q_h + d[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    m_i = tl.full([GROUP_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([GROUP_PAD], tl.float32)
    acc = tl.zeros([GROUP_PAD, VALUE_DIM], tl.float32)
    start = split * split_len
    end = tl.minimum(start + split_len, width)
    for block_start in range(start, end, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        slots = tl.load(
            selected_slots + row * stride_s_n + cols, mask=cols < end, other=-1
        ).to(tl.int64)
        valid = slots > 0
        # Slot 0 may hold NaN padding; slot 1 is the finite masked-read target.
        slots = tl.where(valid, slots, 1)
        keys = tl.load(
            k_cache + slots[:, None] * stride_k_n + kv_head * stride_k_h + d[None, :]
        ).to(q_tile.dtype)
        scores = tl.dot(q_tile, tl.trans(keys)) * (scale_log2 * k_scale)
        scores = tl.where(valid[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        # Rows with no valid slot so far keep m = -inf; avoid -inf - -inf.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        values = tl.load(
            v_cache + slots[:, None] * stride_v_n + kv_head * stride_v_h + dv[None, :]
        ).to(q_tile.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(q_tile.dtype), values)
        m_i = m_new

    # Rows without any valid slot produce zeros.
    out = acc * (v_scale / tl.where(l_i > 0, l_i, 1.0))[:, None]
    if SPLIT:
        part = (row * GROUP * tl.num_programs(1) + q_heads) * num_splits + split
        tl.store(
            partial_out + part[:, None] * VALUE_DIM + dv[None, :],
            out,
            mask=head_mask[:, None],
        )
        # Base-2 log-sum-exp; -inf marks a split without valid slots.
        lse = tl.where(l_i > 0, m_i + tl.log2(l_i), float("-inf"))
        tl.store(partial_lse + part, lse, mask=head_mask)
    else:
        tl.store(
            output + row * stride_o_n + q_heads[:, None] * stride_o_h + dv[None, :],
            out.to(output.dtype.element_ty),
            mask=head_mask[:, None],
        )


@triton.jit
def _triton_qsa_sparse_attention_merge_kernel(
    partial_out,
    partial_lse,
    output,
    num_splits,
    stride_o_n,
    stride_o_h,
    NUM_HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dv = tl.arange(0, VALUE_DIM)
    base = (row * NUM_HEADS + head) * num_splits
    lse_max = float("-inf")
    for split in range(num_splits):
        lse_max = tl.maximum(lse_max, tl.load(partial_lse + base + split))
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    total = 0.0
    acc = tl.zeros([VALUE_DIM], tl.float32)
    for split in range(num_splits):
        # Splits without valid slots carry lse = -inf and weigh zero.
        weight = tl.exp2(tl.load(partial_lse + base + split) - lse_max)
        total += weight
        acc += weight * tl.load(partial_out + (base + split) * VALUE_DIM + dv)
    acc = acc / tl.where(total > 0, total, 1.0)
    tl.store(
        output + row * stride_o_n + head * stride_o_h + dv,
        acc.to(output.dtype.element_ty),
    )


def _scalar_descale(scale: float | torch.Tensor | None, name: str) -> float:
    if scale is None:
        return 1.0
    if isinstance(scale, torch.Tensor):
        raise TypeError(f"Triton QSA sparse attention expects a float {name}")
    return float(scale)


def triton_qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int,
    metadata_capacity_rows: int | None,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
) -> torch.Tensor:
    """Portable QSA sparse attention; see ``qsa_sparse_attention``.

    Every query row attends independently to its own selected slots, so
    ``max_seqlen_q`` does not change the computation, and this kernel keeps no
    metadata workspace sized by ``metadata_capacity_rows``.
    """

    del max_seqlen_q, metadata_capacity_rows
    rows, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    value_dim = v_cache.shape[-1]
    width = selected_slots.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("QSA query heads must be a multiple of KV heads")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("QSA Q/K/V head dimensions must be contiguous")
    if selected_slots.stride(-1) != 1:
        raise ValueError("QSA selected slots must be contiguous per row")
    group = num_heads // num_kv_heads
    output = q.new_empty((rows, num_heads, value_dim))

    programs = rows * num_kv_heads
    max_splits = min(_SPARSE_MAX_SPLITS, triton.cdiv(width, _SPARSE_BLOCK_N))
    num_splits = max(1, min(max_splits, triton.cdiv(_SPARSE_TARGET_PROGRAMS, programs)))
    split_len = triton.cdiv(triton.cdiv(width, num_splits), _SPARSE_BLOCK_N)
    split_len *= _SPARSE_BLOCK_N
    num_splits = triton.cdiv(width, split_len)
    use_split = num_splits > 1
    if use_split:
        partial_out = torch.empty(
            (rows, num_heads, num_splits, value_dim),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (rows, num_heads, num_splits), dtype=torch.float32, device=q.device
        )
    else:
        partial_out = partial_lse = output
    _triton_qsa_sparse_attention_kernel[(rows, num_kv_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        selected_slots,
        output,
        partial_out,
        partial_lse,
        scale * LOG2E,
        _scalar_descale(k_scale, "k_scale"),
        _scalar_descale(v_scale, "v_scale"),
        width,
        split_len,
        num_splits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        selected_slots.stride(0),
        output.stride(0),
        output.stride(1),
        GROUP=group,
        GROUP_PAD=max(16, triton.next_power_of_2(group)),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        BLOCK_N=_SPARSE_BLOCK_N,
        SPLIT=use_split,
        num_warps=4,
        num_stages=1,
    )
    if use_split:
        _triton_qsa_sparse_attention_merge_kernel[(rows, num_heads)](
            partial_out,
            partial_lse,
            output,
            num_splits,
            output.stride(0),
            output.stride(1),
            NUM_HEADS=num_heads,
            VALUE_DIM=value_dim,
            num_warps=1,
        )
    return output


register_kernel(
    "attention",
    "qsa_sparse_attention",
    name="triton_qsa_sparse_attention",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                k_cache=dense_tensor_format(cache_dtype),
                v_cache=dense_tensor_format(cache_dtype),
            )
            for cache_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        }
    ),
    traits={
        "head_dim": frozenset({64, 128, 256}),
        "value_head_dim": frozenset({64, 128, 256}),
    },
    priority=Priority.PORTABLE,
)(triton_qsa_sparse_attention)


__all__ = [
    "qwen4_exp_qsa_compress_and_store",
    "qwen4_exp_qsa_prepare_metadata",
    "qwen4_exp_qsa_recent_write",
    "qwen4_exp_qsa_selected_slots",
    "triton_qsa_block_scores",
    "triton_qsa_block_topk_select",
    "triton_qsa_block_topk_stream",
    "triton_qsa_sparse_attention",
]
