"""Triton MoE routing kernels."""

from __future__ import annotations

from typing import Optional

import torch
from tokenspeed_kernel._triton import tl, triton


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
    choice_scores = scores + bias
    choice_scores = tl.where(
        expert_mask & (choice_scores == choice_scores),
        choice_scores,
        -float("inf"),
    )

    weights_sum = 0.0
    kept_weights = ()
    for k in tl.static_range(0, TOPK):
        best_choice_score = tl.max(choice_scores, axis=0)
        best_expert = tl.min(
            tl.where(choice_scores == best_choice_score, offs_e, BLOCK_E), axis=0
        )
        best_expert = tl.minimum(best_expert, num_experts - 1)
        best_weight = tl.max(tl.where(offs_e == best_expert, scores, 0.0), axis=0)
        stored_expert = best_expert
        if has_static_expert_map:
            stored_expert = tl.load(static_logical_to_physical_map_ptr + best_expert)
        weights_sum += best_weight
        kept_weights = kept_weights + (best_weight,)

        tl.store(
            topk_ids_ptr + token_id * stride_im + k * stride_ik,
            stored_expert.to(tl.int32),
        )
        choice_scores = tl.where(offs_e == best_expert, -float("inf"), choice_scores)

    if renormalize:
        denom = tl.where(weights_sum != 0.0, weights_sum, 1.0)
        factor = routed_scaling_factor
    else:
        denom = 1.0
        factor = 1.0
    for k in tl.static_range(0, TOPK):
        weight = kept_weights[k] / denom * factor
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
    weights_dtype: torch.dtype = torch.float32,
):
    if (
        gating_output.ndim != 2
        or correction_bias.ndim != 1
        or hidden_states.shape[0] != gating_output.shape[0]
        or gating_output.shape[1] != correction_bias.shape[0]
        or gating_output.shape[1] > 1024
        or not (1 <= topk <= 16)
        or num_expert_group != 1
        or topk_group != 1
        or num_fused_shared_experts != 0
        or routed_scaling_factor is None
        or num_token_non_padded is not None
    ):
        ref_w, ref_i = _biased_grouped_topk_reference(
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
        return ref_w.to(weights_dtype), ref_i

    num_tokens, num_experts = gating_output.shape
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=weights_dtype, device=gating_output.device
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


__all__ = ["minimax_biased_grouped_topk"]
