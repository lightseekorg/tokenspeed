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
    "minimax_biased_grouped_topk",
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
