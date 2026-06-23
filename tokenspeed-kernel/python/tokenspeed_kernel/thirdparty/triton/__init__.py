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

from __future__ import annotations

from typing import Optional

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = [
    "deepseek_v4_softplus_topk",
    "minimax_biased_grouped_topk",
    "moe_routing_from_topk_medium_m",
    "moe_routing_from_topk_small_m",
    "stage_deepseek_v4_mega_moe_inputs",
]


_DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE = 128


@triton.jit
def _deepseek_v4_stage_mega_moe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def stage_deepseek_v4_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            f"a multiple of {_DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE}."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    block_k = _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(topk_ids.shape[1])
    _deepseek_v4_stage_mega_moe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        topk_ids.shape[1],
        BLOCK_K=block_k,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )


@triton.jit
def _deepseek_v4_softplus_topk_kernel(
    gating_output_ptr,
    correction_bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    num_experts: tl.constexpr,
    renormalize: tl.constexpr,
    has_correction_bias: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    expert_mask = offs_e < num_experts

    logits = tl.load(
        gating_output_ptr + token_id * stride_gm + offs_e * stride_ge,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)
    softplus = tl.log(1.0 + tl.exp(-tl.abs(logits))) + tl.maximum(logits, 0.0)
    scores = tl.sqrt(softplus)
    choice_scores = scores
    if has_correction_bias:
        bias = tl.load(
            correction_bias_ptr + offs_e,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        choice_scores = choice_scores + bias
    choice_scores = tl.where(expert_mask, choice_scores, -float("inf"))

    weights_sum = 0.0
    for k in tl.static_range(0, TOPK):
        best_choice_score = tl.max(choice_scores, axis=0)
        best_expert = tl.min(
            tl.where(choice_scores == best_choice_score, offs_e, BLOCK_E), axis=0
        )
        best_weight = tl.max(tl.where(offs_e == best_expert, scores, 0.0), axis=0)
        weights_sum += best_weight

        tl.store(
            topk_ids_ptr + token_id * stride_im + k * stride_ik,
            best_expert.to(tl.int32),
        )
        tl.store(
            topk_weights_ptr + token_id * stride_wm + k * stride_wk,
            best_weight,
        )
        choice_scores = tl.where(offs_e == best_expert, -float("inf"), choice_scores)

    if renormalize:
        denom = tl.where(weights_sum != 0.0, weights_sum, 1.0)
        for k in tl.static_range(0, TOPK):
            weight = tl.load(topk_weights_ptr + token_id * stride_wm + k * stride_wk)
            tl.store(
                topk_weights_ptr + token_id * stride_wm + k * stride_wk,
                weight / denom,
            )


def _deepseek_v4_softplus_topk_reference(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
):
    scores = torch.nn.functional.softplus(gating_output.float()).sqrt()
    choice_scores = scores
    if correction_bias is not None:
        choice_scores = choice_scores + correction_bias.to(
            device=scores.device,
            dtype=scores.dtype,
        ).unsqueeze(0)
    topk_ids = torch.topk(choice_scores, k=topk, dim=-1, sorted=True)[1]
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_weights.dtype).tiny
        )
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def deepseek_v4_softplus_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
):
    if (
        gating_output.ndim != 2
        or topk <= 0
        or topk > 32
        or gating_output.shape[1] > 1024
        or (correction_bias is not None and correction_bias.ndim != 1)
        or (
            correction_bias is not None
            and correction_bias.shape[0] != gating_output.shape[1]
        )
        or not gating_output.is_cuda
    ):
        return _deepseek_v4_softplus_topk_reference(
            gating_output,
            correction_bias,
            topk=topk,
            renormalize=renormalize,
        )

    num_tokens, num_experts = gating_output.shape
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    correction_bias_arg = (
        correction_bias
        if correction_bias is not None
        else gating_output.reshape(-1)[:num_experts]
    )
    _deepseek_v4_softplus_topk_kernel[(num_tokens,)](
        gating_output,
        correction_bias_arg,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        num_experts=num_experts,
        renormalize=renormalize,
        has_correction_bias=correction_bias is not None,
        BLOCK_E=triton.next_power_of_2(num_experts),
        TOPK=topk,
        num_warps=1,
    )
    return topk_weights, topk_ids


@triton.jit
def _moe_topk_routing_small_m_kernel(
    topk_weights_ptr,
    topk_ids_ptr,
    slice_sizes_ptr,
    slice_offs_ptr,
    block_offs_ptr,
    block_schedule_ptr,
    gather_indx_ptr,
    scatter_indx_ptr,
    gate_scal_ptr,
    weights_stride_m: tl.constexpr,
    weights_stride_k: tl.constexpr,
    ids_stride_m: tl.constexpr,
    ids_stride_k: tl.constexpr,
    block_offs_stride_b: tl.constexpr,
    block_offs_stride_e: tl.constexpr,
    block_schedule_stride_b: tl.constexpr,
    block_schedule_stride_j: tl.constexpr,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    num_block_sizes: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < num_experts

    flat = tl.arange(0, BLOCK_G)
    valid = flat < num_tokens * topk
    token_idx = flat // topk
    slot_idx = flat - token_idx * topk

    expert_ids = tl.load(
        topk_ids_ptr + token_idx * ids_stride_m + slot_idx * ids_stride_k,
        mask=valid,
        other=-1,
    ).to(tl.int32)
    expert_valid = valid & (expert_ids >= 0) & (expert_ids < num_experts)

    hist = tl.zeros([BLOCK_E], dtype=tl.int32)
    for flat_idx in tl.static_range(0, BLOCK_G):
        token = flat_idx // topk
        slot = flat_idx - token * topk
        valid_flat = flat_idx < num_tokens * topk
        expert = tl.load(
            topk_ids_ptr + token * ids_stride_m + slot * ids_stride_k,
            mask=valid_flat,
            other=-1,
        ).to(tl.int32)
        hist += tl.where((experts == expert) & expert_mask & valid_flat, 1, 0)

    inclusive = tl.cumsum(hist, 0)
    slice_offs = inclusive - hist
    total_rows = tl.sum(hist, 0)
    tl.store(slice_sizes_ptr + experts, hist, mask=expert_mask)
    tl.store(slice_offs_ptr + experts, slice_offs, mask=expert_mask)
    tl.store(slice_offs_ptr + num_experts, total_rows)

    has_block = hist > 0
    block_count = has_block.to(tl.int32)
    block_inclusive = tl.cumsum(block_count, 0)
    block_offs = block_inclusive - block_count
    total_blocks = tl.sum(block_count, 0)

    schedule_pos = tl.arange(0, BLOCK_B)
    schedule_mask = schedule_pos < max_num_blocks
    fill_mask = schedule_mask & (schedule_pos >= total_blocks)
    for block_idx in tl.static_range(0, num_block_sizes):
        block_offs_base = block_offs_ptr + block_idx * block_offs_stride_b
        tl.store(
            block_offs_base + experts * block_offs_stride_e,
            block_offs,
            mask=expert_mask,
        )
        tl.store(block_offs_base + num_experts * block_offs_stride_e, total_blocks)

        schedule_base = block_schedule_ptr + block_idx * block_schedule_stride_b
        tl.store(
            schedule_base + schedule_pos * block_schedule_stride_j,
            -1,
            mask=fill_mask,
        )
        tl.store(
            schedule_base + block_offs * block_schedule_stride_j,
            experts.to(tl.int32),
            mask=expert_mask & has_block,
        )

    row_ids = tl.expand_dims(expert_ids, 1)
    col_ids = tl.expand_dims(expert_ids, 0)
    row_flat = tl.expand_dims(flat, 1)
    col_flat = tl.expand_dims(flat, 0)
    row_valid = tl.expand_dims(expert_valid, 1)
    col_valid = tl.expand_dims(expert_valid, 0)
    sorted_pos = tl.sum(
        tl.where(
            row_valid
            & col_valid
            & (
                (col_ids < row_ids)
                | ((col_ids == row_ids) & (col_flat < row_flat))
            ),
            1,
            0,
        ),
        axis=1,
    )
    weights = tl.load(
        topk_weights_ptr + token_idx * weights_stride_m + slot_idx * weights_stride_k,
        mask=valid,
        other=0.0,
    )

    tl.store(gather_indx_ptr + sorted_pos, token_idx.to(tl.int32), mask=expert_valid)
    tl.store(scatter_indx_ptr + sorted_pos, flat.to(tl.int32), mask=expert_valid)
    tl.store(gate_scal_ptr + sorted_pos, weights, mask=expert_valid)


def moe_routing_from_topk_small_m(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    gate_dtype: torch.dtype,
    num_block_sizes: int,
    max_num_blocks: int,
):
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}")

    num_tokens, topk = topk_ids.shape
    total_rows = num_tokens * topk
    if num_tokens > 16 or total_rows > 128:
        raise ValueError(
            "small-M top-k routing metadata supports at most 16 tokens and "
            f"128 routed rows, got {num_tokens} tokens and {total_rows} rows"
        )
    if num_experts <= 0 or num_experts > 1024:
        raise ValueError(f"num_experts must be in [1, 1024], got {num_experts}")
    if num_block_sizes <= 0 or max_num_blocks < 0:
        raise ValueError(
            "num_block_sizes must be positive and max_num_blocks must be nonnegative"
        )

    device = topk_ids.device

    slice_sizes = torch.empty((num_experts,), dtype=torch.int32, device=device)
    slice_offs = torch.empty((num_experts + 1,), dtype=torch.int32, device=device)
    block_offs_data = torch.empty(
        (num_block_sizes, num_experts + 1), dtype=torch.int32, device=device
    )
    block_schedule_data = torch.empty(
        (num_block_sizes, max_num_blocks), dtype=torch.int32, device=device
    )
    gather_indx = torch.empty((total_rows,), dtype=torch.int32, device=device)
    scatter_indx = torch.empty((total_rows,), dtype=torch.int32, device=device)
    gate_scal = torch.empty((total_rows,), dtype=gate_dtype, device=device)

    if total_rows == 0:
        slice_sizes.zero_()
        slice_offs.zero_()
        block_offs_data.zero_()
        return (
            slice_sizes,
            slice_offs,
            block_offs_data,
            block_schedule_data,
            gather_indx,
            scatter_indx,
            gate_scal,
        )

    block_g = triton.next_power_of_2(total_rows)
    block_e = triton.next_power_of_2(num_experts)
    block_b = triton.next_power_of_2(max(1, max_num_blocks))
    _moe_topk_routing_small_m_kernel[(1,)](
        topk_weights,
        topk_ids,
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        block_offs_data.stride(0),
        block_offs_data.stride(1),
        block_schedule_data.stride(0),
        block_schedule_data.stride(1),
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        num_block_sizes=num_block_sizes,
        max_num_blocks=max_num_blocks,
        BLOCK_E=block_e,
        BLOCK_G=block_g,
        BLOCK_B=block_b,
        num_warps=4,
    )
    return (
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
    )


@triton.jit
def _moe_topk_routing_medium_hist_kernel(
    topk_ids_ptr,
    slice_sizes_ptr,
    num_rows: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_G)
    row_mask = row_offsets < num_rows
    expert_ids = tl.load(topk_ids_ptr + row_offsets, mask=row_mask, other=0).to(
        tl.int32
    )
    valid = row_mask & (expert_ids >= 0) & (expert_ids < num_experts)
    expert_ids = tl.where(valid, expert_ids, 0)

    hist = tl.histogram(expert_ids, BLOCK_E, mask=row_mask)
    expert_offsets = tl.arange(0, BLOCK_E)
    tl.store(
        slice_sizes_ptr + expert_offsets,
        hist,
        mask=expert_offsets < num_experts,
    )


@triton.jit
def _moe_topk_routing_medium_offsets_kernel(
    slice_sizes_ptr,
    slice_offs_ptr,
    block_offs_ptr,
    block_schedule_ptr,
    offsets_ptr,
    block_offs_stride_b: tl.constexpr,
    block_offs_stride_e: tl.constexpr,
    block_schedule_stride_b: tl.constexpr,
    block_schedule_stride_j: tl.constexpr,
    num_experts: tl.constexpr,
    num_block_sizes: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    counts = tl.load(slice_sizes_ptr + expert_offsets, mask=expert_mask, other=0)
    inclusive = tl.cumsum(counts, axis=0)
    exclusive = inclusive - counts
    total_rows = tl.sum(counts, axis=0)
    tl.store(slice_offs_ptr + expert_offsets, exclusive, mask=expert_mask)
    tl.store(slice_offs_ptr + num_experts, total_rows)
    tl.store(offsets_ptr + expert_offsets, exclusive, mask=expert_mask)

    schedule_offsets = tl.arange(0, BLOCK_B)
    schedule_mask = schedule_offsets < max_num_blocks
    for block_idx in tl.static_range(0, num_block_sizes):
        block_size = 16 << block_idx
        block_counts = tl.cdiv(counts, block_size)
        block_inclusive = tl.cumsum(block_counts, axis=0)
        block_exclusive = block_inclusive - block_counts
        total_blocks = tl.sum(block_counts, axis=0)
        block_offs_base = block_offs_ptr + block_idx * block_offs_stride_b
        tl.store(
            block_offs_base + expert_offsets * block_offs_stride_e,
            block_exclusive,
            mask=expert_mask,
        )
        tl.store(
            block_offs_base + num_experts * block_offs_stride_e,
            total_blocks,
        )

        schedule_base = block_schedule_ptr + block_idx * block_schedule_stride_b
        tl.store(
            schedule_base + schedule_offsets * block_schedule_stride_j,
            -1,
            mask=schedule_mask,
        )


@triton.jit
def _moe_topk_routing_medium_schedule_kernel(
    slice_sizes_ptr,
    block_offs_ptr,
    block_schedule_ptr,
    block_offs_stride_b: tl.constexpr,
    block_offs_stride_e: tl.constexpr,
    block_schedule_stride_b: tl.constexpr,
    block_schedule_stride_j: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    expert_id = tl.program_id(0)
    block_idx = tl.program_id(1)
    block_size = 16 << block_idx

    count = tl.load(slice_sizes_ptr + expert_id)
    n_blocks = tl.cdiv(count, block_size)
    block_offs_base = block_offs_ptr + block_idx * block_offs_stride_b
    block_start = tl.load(block_offs_base + expert_id * block_offs_stride_e)

    local_blocks = tl.arange(0, BLOCK_L)
    schedule_offsets = block_start + local_blocks
    schedule_base = block_schedule_ptr + block_idx * block_schedule_stride_b
    packed = (local_blocks.to(tl.int32) << 16) + expert_id
    tl.store(
        schedule_base + schedule_offsets * block_schedule_stride_j,
        packed,
        mask=local_blocks < n_blocks,
    )


@triton.jit
def _moe_topk_routing_medium_place_kernel(
    topk_weights_ptr,
    topk_ids_ptr,
    offsets_ptr,
    gather_indx_ptr,
    scatter_indx_ptr,
    gate_scal_ptr,
    num_rows: tl.constexpr,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row_offsets = tl.arange(0, BLOCK_G)
    row_mask = row_offsets < num_rows
    expert_ids = tl.load(topk_ids_ptr + row_offsets, mask=row_mask, other=0).to(
        tl.int32
    )
    weights = tl.load(topk_weights_ptr + row_offsets, mask=row_mask, other=0.0)
    valid = row_mask & (expert_ids >= 0) & (expert_ids < num_experts)
    expert_ids = tl.where(valid, expert_ids, 0)
    weights = tl.where(valid, weights, 0.0)

    sorted_pos = tl.atomic_add(offsets_ptr + expert_ids, 1, mask=row_mask)
    sorted_pos = tl.where(row_mask, sorted_pos, 0)

    tl.store(
        gather_indx_ptr + sorted_pos,
        (row_offsets // topk).to(tl.int32),
        mask=row_mask,
    )
    tl.store(scatter_indx_ptr + sorted_pos, row_offsets.to(tl.int32), mask=row_mask)
    tl.store(gate_scal_ptr + sorted_pos, weights, mask=row_mask)


def moe_routing_from_topk_medium_m(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    gate_dtype: torch.dtype,
    num_block_sizes: int,
    max_num_blocks: int,
):
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("topk_weights and topk_ids must be contiguous")

    num_tokens, topk = topk_ids.shape
    total_rows = num_tokens * topk
    if total_rows > 4096:
        raise ValueError(
            "medium-M top-k routing metadata supports at most 4096 routed rows, "
            f"got {total_rows}"
        )
    if num_experts <= 0 or num_experts > 1024:
        raise ValueError(f"num_experts must be in [1, 1024], got {num_experts}")
    if num_block_sizes <= 0 or max_num_blocks < 0:
        raise ValueError(
            "num_block_sizes must be positive and max_num_blocks must be nonnegative"
        )

    device = topk_ids.device
    slice_sizes = torch.empty((num_experts,), dtype=torch.int32, device=device)
    slice_offs = torch.empty((num_experts + 1,), dtype=torch.int32, device=device)
    block_offs_data = torch.empty(
        (num_block_sizes, num_experts + 1), dtype=torch.int32, device=device
    )
    block_schedule_data = torch.empty(
        (num_block_sizes, max_num_blocks), dtype=torch.int32, device=device
    )
    offsets = torch.empty((num_experts,), dtype=torch.int32, device=device)
    gather_indx = torch.empty((total_rows,), dtype=torch.int32, device=device)
    scatter_indx = torch.empty((total_rows,), dtype=torch.int32, device=device)
    gate_scal = torch.empty((total_rows,), dtype=gate_dtype, device=device)

    if total_rows == 0:
        slice_sizes.zero_()
        slice_offs.zero_()
        block_offs_data.zero_()
        block_schedule_data.fill_(-1)
        return (
            slice_sizes,
            slice_offs,
            block_offs_data,
            block_schedule_data,
            gather_indx,
            scatter_indx,
            gate_scal,
        )

    block_g = max(32, triton.next_power_of_2(total_rows))
    block_e = max(32, triton.next_power_of_2(num_experts))
    block_b = max(1, triton.next_power_of_2(max_num_blocks))
    block_l = max(1, triton.next_power_of_2(triton.cdiv(total_rows, 16)))
    topk_ids_flat = topk_ids.reshape(-1)
    topk_weights_flat = topk_weights.reshape(-1)

    _moe_topk_routing_medium_hist_kernel[(1,)](
        topk_ids_flat,
        slice_sizes,
        num_rows=total_rows,
        num_experts=num_experts,
        BLOCK_G=block_g,
        BLOCK_E=block_e,
        num_warps=1,
    )
    _moe_topk_routing_medium_offsets_kernel[(1,)](
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        offsets,
        block_offs_data.stride(0),
        block_offs_data.stride(1),
        block_schedule_data.stride(0),
        block_schedule_data.stride(1),
        num_experts=num_experts,
        num_block_sizes=num_block_sizes,
        max_num_blocks=max_num_blocks,
        BLOCK_E=block_e,
        BLOCK_B=block_b,
        num_warps=1,
    )
    _moe_topk_routing_medium_schedule_kernel[(num_experts, num_block_sizes)](
        slice_sizes,
        block_offs_data,
        block_schedule_data,
        block_offs_data.stride(0),
        block_offs_data.stride(1),
        block_schedule_data.stride(0),
        block_schedule_data.stride(1),
        num_experts=num_experts,
        BLOCK_L=block_l,
        num_warps=1,
    )
    _moe_topk_routing_medium_place_kernel[(1,)](
        topk_weights_flat,
        topk_ids_flat,
        offsets,
        gather_indx,
        scatter_indx,
        gate_scal,
        num_rows=total_rows,
        topk=topk,
        num_experts=num_experts,
        BLOCK_G=block_g,
        num_warps=1,
    )
    return (
        slice_sizes,
        slice_offs,
        block_offs_data,
        block_schedule_data,
        gather_indx,
        scatter_indx,
        gate_scal,
    )


@triton.jit
def _minimax_biased_grouped_topk_kernel(
    gating_output_ptr,
    correction_bias_ptr,
    static_logical_to_physical_map_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    num_experts: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    renormalize: tl.constexpr,
    has_static_expert_map: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    expert_mask = offs_e < num_experts

    logits = tl.load(
        gating_output_ptr + token_id * stride_gm + offs_e * stride_ge,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)
    bias = tl.load(
        correction_bias_ptr + offs_e,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.sigmoid(logits)
    choice_scores = tl.where(expert_mask, scores + bias, -float("inf"))

    weights_sum = 0.0
    for k in tl.static_range(0, TOPK):
        best_choice_score = tl.max(choice_scores, axis=0)
        best_expert = tl.min(
            tl.where(choice_scores == best_choice_score, offs_e, BLOCK_E), axis=0
        )
        best_weight = tl.max(tl.where(offs_e == best_expert, scores, 0.0), axis=0)
        stored_expert = best_expert
        if has_static_expert_map:
            stored_expert = tl.load(static_logical_to_physical_map_ptr + best_expert)
        weights_sum += best_weight

        tl.store(
            topk_ids_ptr + token_id * stride_im + k * stride_ik,
            stored_expert.to(tl.int32),
        )
        tl.store(
            topk_weights_ptr + token_id * stride_wm + k * stride_wk,
            best_weight,
        )
        choice_scores = tl.where(offs_e == best_expert, -float("inf"), choice_scores)

    if renormalize:
        denom = tl.where(weights_sum != 0.0, weights_sum, 1.0)
        for k in tl.static_range(0, TOPK):
            weight = tl.load(topk_weights_ptr + token_id * stride_wm + k * stride_wk)
            weight = weight / denom
            weight = weight * routed_scaling_factor
            tl.store(topk_weights_ptr + token_id * stride_wm + k * stride_wk, weight)


def _biased_grouped_topk_reference(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = 1.0,
    num_token_non_padded: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert (
        routed_scaling_factor is not None
    ), "routed_scaling_factor is required for biased_grouped_topk"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        topk_weights *= routed_scaling_factor

    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    if logical_to_physical_map is not None:
        topk_ids = logical_to_physical_map[topk_ids]
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def minimax_biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = 1.0,
    num_token_non_padded: Optional[torch.Tensor] = None,
    logical_to_physical_map: Optional[torch.Tensor] = None,
):
    if (
        gating_output.ndim != 2
        or correction_bias.ndim != 1
        or hidden_states.shape[0] != gating_output.shape[0]
        or gating_output.shape[1] != correction_bias.shape[0]
        or gating_output.shape[1] > 256
        or topk != 8
        or num_expert_group != 1
        or topk_group != 1
        or num_fused_shared_experts != 0
        or routed_scaling_factor is None
        or num_token_non_padded is not None
    ):
        return _biased_grouped_topk_reference(
            hidden_states,
            gating_output,
            correction_bias,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            num_token_non_padded=num_token_non_padded,
            logical_to_physical_map=logical_to_physical_map,
        )

    num_tokens, num_experts = gating_output.shape
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    block_e = triton.next_power_of_2(num_experts)
    static_map = (
        logical_to_physical_map
        if logical_to_physical_map is not None
        else correction_bias
    )
    _minimax_biased_grouped_topk_kernel[(num_tokens,)](
        gating_output,
        correction_bias,
        static_map,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        num_experts=num_experts,
        routed_scaling_factor=float(routed_scaling_factor),
        renormalize=renormalize,
        has_static_expert_map=logical_to_physical_map is not None,
        BLOCK_E=block_e,
        TOPK=topk,
        num_warps=1,
    )
    return topk_weights, topk_ids
