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

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.cuda.dsa_topk import (
    has_ragged_decode_topk,
    ragged_decode_topk,
)
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    triton_topk_from_logits,
)
from tokenspeed_kernel.platform import current_platform

_is_nvidia = current_platform().is_nvidia
_PERSISTENT_TOPK_WORKSPACE_BYTES = 1024 * 1024


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
    qsa_expansion,
    qsa_page_size,
    recent_expansion,
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
):
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
            qsa_page_table + request * stride_qsa_pt_b + qsa_columns * qsa_expansion,
            mask=mask & valid,
            other=0,
        ).to(tl.int64)
        qsa_pages = qsa_pages // qsa_expansion
        qsa_values = qsa_pages * qsa_page_size + safe_logical % qsa_page_size
        qsa_valid = valid & (qsa_pages > 0)
        tl.store(
            qsa_locs + rows,
            tl.where(qsa_valid, qsa_values, 0).to(tl.int32),
            mask=mask,
        )

        recent_columns = safe_logical // recent_page_size
        recent_pages = tl.load(
            recent_page_table
            + request * stride_recent_pt_b
            + recent_columns * recent_expansion,
            mask=mask & valid,
            other=0,
        ).to(tl.int64)
        recent_pages = recent_pages // recent_expansion
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
    qsa_expansion: int,
    qsa_page_size: int,
    recent_page_table: torch.Tensor,
    recent_expansion: int,
    recent_page_size: int,
    compress_ratio: int,
    *,
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
        qsa_page_table: Compressed-cache page table at consumer granularity.
        qsa_expansion: Consumer pages per compressed logical page.
        qsa_page_size: Logical tokens covered by a compressed page.
        recent_page_table: Recent-cache page table at consumer granularity.
        recent_expansion: Consumer pages per recent logical page.
        recent_page_size: Logical tokens covered by a recent page.
        compress_ratio: Raw tokens represented by one compressed key.
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
        qsa_expansion,
        qsa_page_size,
        recent_expansion,
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
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if STAGE_VERIFY or STAGE_DRAFT:
        # Compression has consumed the old raw/draft ring before any staged
        # value can alias it. The PDL-dependent score kernel needs neither
        # staging destination, so these stores remain in the producer tail.
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
    ) = None,
    stage_draft: bool = False,
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
        enable_pdl: Allow the follow-up raw-key write kernel to launch early
            on NVIDIA GPUs; the dependent kernel still waits for this grid.
        query: Optional raw projected queries shaped
            ``[rows, num_query_heads * head_dim]``. When present, query
            normalization and RoPE share this compression launch.
        query_norm_weight: Gemma RMSNorm weight for ``query``.
        query_norm_epsilon: RMSNorm epsilon for ``query``.
        num_query_heads: Query heads packed into each projected row.
        stage_verify_buffers: Optional contiguous K, position, logical-position,
            and recent-location destinations for target verification.
        stage_draft: Store each row into the supplied request-local draft
            scratch after compression has consumed its previous contents.

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
    if write:
        position = tl.load(logical_positions + row).to(tl.int64)
        slot = (position % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
        page = loc // recent_page_size
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < head_dim
        values = tl.load(
            token_k + row * stride_k_n + dim_offsets * stride_k_d,
            mask=dim_mask,
            other=0.0,
        )
        if ENABLE_PDL:
            # Only the raw-cache and position-header writes must observe the
            # compression kernel's reads of the old ring slots; everything
            # above already overlapped that kernel's tail.
            tl.extra.cuda.gdc_wait()
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
            launch on NVIDIA GPUs, hiding the launch gap between the two.

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
def _qwen4_exp_qsa_pad_keys_to_topk(
    keys, BLOCK_TOPK: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Zero-pad a packed-key tile up to the top-k width via constexpr joins."""

    if BLOCK_TOPK >= 2 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((BLOCK_N,), 0, tl.int64)), (2 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 4 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((2 * BLOCK_N,), 0, tl.int64)), (4 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 8 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((4 * BLOCK_N,), 0, tl.int64)), (8 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 16 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((8 * BLOCK_N,), 0, tl.int64)), (16 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 32 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((16 * BLOCK_N,), 0, tl.int64)), (32 * BLOCK_N,)
        )
    return keys


@triton.jit
def _qwen4_exp_qsa_stream_block_topk_kernel(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    partial_keys,
    num_heads,
    head_dim,
    num_blocks,
    page_size,
    page_expansion,
    blocks_per_split,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    stride_pk_n,
    stride_pk_s,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    block_start = split * blocks_per_split
    complete = tl.load(complete_blocks + row).to(tl.int64)
    block_end = tl.minimum(block_start + blocks_per_split, num_blocks)
    block_end = tl.minimum(block_end, complete)
    if block_start >= block_end:
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return
    request = tl.load(request_indices + row).to(tl.int64)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        query
        + row * stride_q_n
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    acc = tl.zeros((BLOCK_TOPK,), dtype=tl.int64)
    for tile_start in range(block_start, block_end, BLOCK_N):
        block_ids = tile_start + tl.arange(0, BLOCK_N)
        tile_mask = block_ids < block_end
        columns = block_ids // page_size
        offsets = block_ids % page_size
        # Page tables live at consumer granularity; entry ``col * expansion``
        # maps back to one logical compressed page.
        pages = tl.load(
            page_table + request * stride_pt_b + columns * page_expansion,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)
        pages = pages // page_expansion
        slots = pages * page_size + offsets
        keys = tl.load(
            key_cache + slots[None, :] * stride_k_n + dim_offsets[:, None] * stride_k_d,
            mask=tile_mask[None, :] & dim_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(q, keys, out_dtype=tl.float32)
        scores = tl.maximum(scores, 0.0)
        scores = tl.sum(tl.where(head_mask[:, None], scores, 0.0), axis=0)
        # Pack (score, block id) into one monotonic int64 key: shifted
        # score bits in the high word, block id in the low word, zero
        # reserved as the invalid sentinel.
        score_bits = scores.to(tl.int32, bitcast=True)
        packed = ((score_bits.to(tl.int64) + 1) << 32) | block_ids.to(tl.int64)
        packed = tl.where(tile_mask, packed, 0)
        # Streaming merge in the style of the DSA top-k kernel: rotate
        # the running set into a bitonic layout every tile, then fold in
        # the tile's sorted top-k with an elementwise maximum instead of
        # re-selecting over a joined 2k-wide buffer. The prune still
        # skips the tile sort when no packed key can enter the set.
        acc = tl.bitonic_merge(acc)
        if tl.max(packed, axis=0) > tl.min(acc, axis=0):
            padded = _qwen4_exp_qsa_pad_keys_to_topk(packed, BLOCK_TOPK, BLOCK_N)
            acc = tl.maximum(acc, tl.sort(padded, descending=True))
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    tl.store(partial_keys + row * stride_pk_n + split * stride_pk_s + topk_offsets, acc)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _qwen4_exp_qsa_merge_pairs(cur, ROWS: tl.constexpr, BLOCK_TOPK: tl.constexpr):
    """Merge adjacent bitonic rows and retain each pair's largest K keys."""

    # Every partial row is bitonic: the streaming producer stores
    # maximum(ascending, descending), while a previous tree level stores the
    # same layout. Negating odd rows lets one batched ascending bitonic merge
    # arrange every pair as (ascending, descending); the pairwise maximum is
    # then exactly the largest K keys from their union and remains bitonic.
    odd = (tl.arange(0, ROWS) % 2 == 1)[:, None]
    ordered = tl.where(odd, -cur, cur)
    ordered = tl.bitonic_merge(ordered)
    ordered = tl.where(odd, -ordered, ordered)
    return tl.max(tl.reshape(ordered, (ROWS // 2, 2, BLOCK_TOPK)), axis=1)


@triton.jit
def _qwen4_exp_qsa_merge_tree(cur, ROWS: tl.constexpr, BLOCK_TOPK: tl.constexpr):
    """Reduce power-of-two bitonic rows to one descending top-k row."""

    if ROWS >= 256:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 256, BLOCK_TOPK)
    if ROWS >= 128:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 128, BLOCK_TOPK)
    if ROWS >= 64:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 64, BLOCK_TOPK)
    if ROWS >= 32:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 32, BLOCK_TOPK)
    if ROWS >= 16:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 16, BLOCK_TOPK)
    if ROWS >= 8:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 8, BLOCK_TOPK)
    if ROWS >= 4:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 4, BLOCK_TOPK)
    if ROWS >= 2:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 2, BLOCK_TOPK)
    return tl.reshape(tl.bitonic_merge(cur, descending=True), (BLOCK_TOPK,))


@triton.jit
def _qwen4_exp_qsa_merge_block_topk_kernel(
    partial_keys,
    selected_blocks,
    complete_blocks,
    blocks_per_split,
    stride_pk_n,
    stride_pk_s,
    stride_o_n,
    BLOCK_TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    POW2_SPLITS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    complete = tl.load(complete_blocks + row).to(tl.int64)
    split_ids = tl.arange(0, POW2_SPLITS)[:, None]
    entries = tl.arange(0, BLOCK_TOPK)[None, :]
    # Mask only on SPLITS so the partials load issues in parallel with the
    # complete scalar load; splits past the complete frontier were skipped by
    # the streaming kernel and carry garbage, so they are zeroed afterwards
    # instead of being masked off before the load.
    partials = tl.load(
        partial_keys + row * stride_pk_n + split_ids * stride_pk_s + entries,
        mask=split_ids < SPLITS,
        other=0,
    )
    valid_splits = (split_ids < SPLITS) & (split_ids * blocks_per_split < complete)
    partials = tl.where(valid_splits, partials, 0)
    acc = _qwen4_exp_qsa_merge_tree(partials, POW2_SPLITS, BLOCK_TOPK)
    block_ids = (acc & 0xFFFFFFFF).to(tl.int32)
    block_ids = tl.where(acc != 0, block_ids, -1)
    tl.store(selected_blocks + row * stride_o_n + tl.arange(0, BLOCK_TOPK), block_ids)


@triton.jit
def _qwen4_exp_qsa_merge_chunk_kernel(
    partial_keys,
    merged_keys,
    complete_blocks,
    blocks_per_split,
    stride_pk_n,
    stride_pk_s,
    stride_mk_n,
    BLOCK_TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    complete = tl.load(complete_blocks + row).to(tl.int64)
    split_ids = chunk * CHUNK + tl.arange(0, CHUNK)[:, None]
    entries = tl.arange(0, BLOCK_TOPK)[None, :]
    # Same late-zeroing contract as the single-level merge, and no early-out
    # for fully invalid chunks: keeping every chunk CTA on the same path
    # avoids an uneven grid, which measurably delays the dependent launch in
    # latency-bound single-request cases.
    partials = tl.load(
        partial_keys + row * stride_pk_n + split_ids * stride_pk_s + entries,
        mask=split_ids < SPLITS,
        other=0,
    )
    valid_splits = (split_ids < SPLITS) & (split_ids * blocks_per_split < complete)
    partials = tl.where(valid_splits, partials, 0)
    acc = _qwen4_exp_qsa_merge_tree(partials, CHUNK, BLOCK_TOPK)
    tl.store(
        merged_keys + row * stride_mk_n + chunk * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK),
        acc,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
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
    page_size,
    page_expansion,
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
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    request = tl.load(request_indices + row).to(tl.int64)
    complete = tl.load(complete_blocks + row).to(tl.int64)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        query
        + row * stride_q_n
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    block_ids = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    tile_mask = block_ids < num_blocks
    columns = block_ids // page_size
    offsets = block_ids % page_size
    # Page tables live at consumer granularity; entry ``col * expansion``
    # maps back to one logical compressed page.
    pages = tl.load(
        page_table + request * stride_pt_b + columns * page_expansion,
        mask=tile_mask,
        other=0,
    ).to(tl.int64)
    pages = pages // page_expansion
    slots = pages * page_size + offsets
    valid = tile_mask & (block_ids < complete)
    keys = tl.load(
        key_cache + slots[None, :] * stride_k_n + dim_offsets[:, None] * stride_k_d,
        mask=valid[None, :] & dim_mask[:, None],
        other=0.0,
    )
    scores = tl.dot(q, keys, out_dtype=tl.float32)
    scores = tl.maximum(scores, 0.0)
    scores = tl.sum(tl.where(head_mask[:, None], scores, 0.0), axis=0)
    # Invalid blocks become -inf so the downstream selection drops them;
    # the PDL trigger lets selection launch while the tail tiles drain.
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(logits + row * stride_l_n + block_ids, scores, mask=tile_mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _qwen4_exp_qsa_block_topk_stream(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    page_expansion: int,
    max_partial_bytes: int,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Fused streaming path: score tiles and fold them into a running top-k.

    Streams each row's page-table blocks tile by tile, scores them against
    the row's query, and maintains a packed-key running top-k, so neither
    the expanded slot matrix nor the full score matrix is materialized.
    Block ranges are split across programs to keep small-row batches busy;
    the partial top-k buffers are bounded by ``max_partial_bytes``.
    """

    rows = query.shape[0]
    num_blocks = triton.cdiv(page_table.shape[1], page_expansion) * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )
    # The merge kernel unconditionally rewrites every row, including the
    # ``-1`` sentinels for empty slots, so skip the fill and leave the
    # buffer uninitialized.
    selected = torch.empty((rows, block_topk), dtype=torch.int32, device=query.device)
    # Wide tiles amortize the running top-k merge for large block_topk;
    # small topk keeps the original 64-block tiles.
    block_n = min(512, block_topk)
    splits = max(1, max_partial_bytes // max(rows * block_topk * 8, 1))
    # Target roughly 256 streaming programs so single-row decode batches
    # still fill the GPU while large batches do not oversubscribe it; the
    # partial buffers stay within ``max_partial_bytes``.
    split_cap = min(256, max(8, 256 // rows))
    # The merge selects at most ``merge_slots`` x ``block_topk`` packed keys
    # per program; two-stage merging needs the same bound on both levels.
    merge_slots = max(1, 8192 // block_topk)
    split_cap = min(split_cap, merge_slots * merge_slots)
    splits = min(splits, split_cap, triton.cdiv(num_blocks, block_n))
    blocks_per_split = triton.cdiv(num_blocks, splits)
    while splits > 1 and blocks_per_split * (splits - 1) >= num_blocks:
        splits -= 1
        blocks_per_split = triton.cdiv(num_blocks, splits)
    partial = torch.empty(
        (rows, splits, block_topk), dtype=torch.int64, device=query.device
    )
    # PDL overlaps the merge launch with the tail of the streaming grid on
    # NVIDIA GPUs; the merge's gdc_wait still enforces the full dependency.
    use_pdl = _is_nvidia and enable_pdl
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    # Strided query reads keep the GEMM-view input copy-free.
    _qwen4_exp_qsa_stream_block_topk_kernel[(rows, splits)](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        partial,
        query.shape[1],
        query.shape[2],
        num_blocks,
        page_size,
        page_expansion,
        blocks_per_split,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        partial.stride(0),
        partial.stride(1),
        BLOCK_TOPK=block_topk,
        BLOCK_N=block_n,
        BLOCK_H=triton.next_power_of_2(query.shape[1]),
        BLOCK_D=triton.next_power_of_2(query.shape[2]),
        ENABLE_PDL=use_pdl,
        num_warps=8,
        num_stages=2,
        **pdl_kwargs,
    )
    if splits * block_topk <= 8192:
        _qwen4_exp_qsa_merge_block_topk_kernel[(rows,)](
            partial,
            selected,
            complete_blocks,
            blocks_per_split,
            partial.stride(0),
            partial.stride(1),
            selected.stride(0),
            BLOCK_TOPK=block_topk,
            SPLITS=splits,
            POW2_SPLITS=triton.next_power_of_2(splits),
            ENABLE_PDL=use_pdl,
            num_warps=8,
            **pdl_kwargs,
        )
    else:
        # Two-stage merge: chunked programs reduce the split partials first,
        # one final program selects the global top-k of the chunk results.
        chunks = triton.cdiv(splits, merge_slots)
        scratch = torch.empty(
            (rows, chunks, block_topk), dtype=torch.int64, device=query.device
        )
        _qwen4_exp_qsa_merge_chunk_kernel[(rows, chunks)](
            partial,
            scratch,
            complete_blocks,
            blocks_per_split,
            partial.stride(0),
            partial.stride(1),
            scratch.stride(0),
            BLOCK_TOPK=block_topk,
            SPLITS=splits,
            CHUNK=merge_slots,
            ENABLE_PDL=use_pdl,
            num_warps=8,
            **pdl_kwargs,
        )
        _qwen4_exp_qsa_merge_block_topk_kernel[(rows,)](
            scratch,
            selected,
            complete_blocks,
            blocks_per_split * merge_slots,
            scratch.stride(0),
            scratch.stride(1),
            selected.stride(0),
            BLOCK_TOPK=block_topk,
            SPLITS=chunks,
            POW2_SPLITS=triton.next_power_of_2(chunks),
            ENABLE_PDL=use_pdl,
            num_warps=8,
            **pdl_kwargs,
        )
    return selected


def _qwen4_exp_qsa_block_topk_logits(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    page_expansion: int,
    persistent_topk_workspace: torch.Tensor | None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Materialized path: score every block to FP32, then radix-select.

    Writes the ``[rows, num_blocks]`` score matrix (blocks at or beyond
    ``complete_blocks`` are ``-inf``). On NVIDIA, a valid caller-owned
    workspace routes selection through the length-aware persistent CUDA radix
    kernel; ``complete_blocks`` prevents the selector from rescanning
    graph-padded columns. Other platforms and direct callers without a
    workspace retain the portable Triton top-k fallback. Callers bound the
    score matrix via the routing heuristic in ``qwen4_exp_qsa_block_topk``.
    """

    rows = query.shape[0]
    num_blocks = triton.cdiv(page_table.shape[1], page_expansion) * page_size
    logits = torch.empty((rows, num_blocks), dtype=torch.float32, device=query.device)
    use_pdl = _is_nvidia and enable_pdl
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _qwen4_exp_qsa_score_blocks_kernel[(rows, triton.cdiv(num_blocks, 512))](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        logits,
        query.shape[1],
        query.shape[2],
        num_blocks,
        page_size,
        page_expansion,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        logits.stride(0),
        BLOCK_N=512,
        BLOCK_H=triton.next_power_of_2(query.shape[1]),
        BLOCK_D=triton.next_power_of_2(query.shape[2]),
        ENABLE_PDL=use_pdl,
        num_warps=8,
        num_stages=2,
        **pdl_kwargs,
    )
    if (
        _is_nvidia
        and persistent_topk_workspace is not None
        and block_topk in (512, 1024, 2048)
        and persistent_topk_workspace.is_cuda
        and persistent_topk_workspace.device == logits.device
        and persistent_topk_workspace.dtype == torch.uint8
        and persistent_topk_workspace.numel() >= _PERSISTENT_TOPK_WORKSPACE_BYTES
        and has_ragged_decode_topk()
    ):
        selected = torch.empty(
            (rows, block_topk), dtype=torch.int32, device=query.device
        )
        ragged_decode_topk(
            logits,
            selected,
            block_topk,
            lengths=complete_blocks,
            workspace=persistent_topk_workspace,
            max_seq_len=num_blocks,
        )
        return selected
    return triton_topk_from_logits(logits, block_topk, enable_pdl=use_pdl)


def qwen4_exp_qsa_block_topk(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    page_expansion: int = 1,
    max_partial_bytes: int = 32 * 1024 * 1024,
    solution: str = "stream",
    persistent_topk_workspace: torch.Tensor | None = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Select the highest-scoring compressed blocks for each query row.

    Args:
        query: Query tensor shaped ``[rows, heads, head_dim]``.
        key_cache: Flattened compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Page table shaped ``[requests, max_pages]`` stored at
            consumer granularity.
        request_indices: Owning request id per query row.
        complete_blocks: Fully compressed block counts shaped ``[rows]``.
        page_size: Compressed-cache rows covered by one logical page.
        block_topk: Blocks selected per row; must be a power of two and at
            least 64.
        page_expansion: Consumer page-table entries covered by one logical
            page.
        max_partial_bytes: Memory budget for the partial top-k buffers
            (``stream`` solution only).
        solution: ``"stream"`` fuses scoring and selection without
            materializing scores (default); ``"logits"`` materializes the
            ``[rows, num_blocks]`` FP32 scores and radix-selects them.
        persistent_topk_workspace: Optional caller-owned CUDA uint8 workspace
            of at least 1 MiB. The ``"logits"`` solution uses it for the
            length-aware persistent radix top-k when that kernel is available;
            otherwise selection falls back to portable Triton.
        enable_pdl: Allow programmatic dependent launch between the
            producer and selection kernels on NVIDIA GPUs.

    Returns:
        Int32 block ids shaped ``[rows, block_topk]``; invalid entries are
        ``-1``.
    """

    if query.ndim != 3 or key_cache.ndim != 3:
        raise ValueError("Qwen4-Exp QSA block top-k expects rank-three tensors")
    if block_topk < 64 or (block_topk & (block_topk - 1)):
        raise ValueError("Qwen4-Exp QSA block top-k needs a power-of-two topk >= 64")
    page_expansion = int(page_expansion)
    if page_expansion < 1:
        raise ValueError("Qwen4-Exp QSA block top-k needs a positive expansion")
    if solution not in ("stream", "logits"):
        raise ValueError(
            "Qwen4-Exp QSA block top-k solution must be 'stream' or 'logits', "
            f"got {solution!r}"
        )
    rows = query.shape[0]
    num_blocks = triton.cdiv(page_table.shape[1], page_expansion) * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )
    if solution == "logits":
        return _qwen4_exp_qsa_block_topk_logits(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            page_expansion=page_expansion,
            persistent_topk_workspace=persistent_topk_workspace,
            enable_pdl=enable_pdl,
        )
    return _qwen4_exp_qsa_block_topk_stream(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        page_expansion=page_expansion,
        max_partial_bytes=max_partial_bytes,
        enable_pdl=enable_pdl,
    )


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
):
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
        )
    return output


__all__ = [
    "qwen4_exp_qsa_block_topk",
    "qwen4_exp_qsa_compress_and_store",
    "qwen4_exp_qsa_prepare_metadata",
    "qwen4_exp_qsa_recent_write",
    "qwen4_exp_qsa_selected_slots",
]
