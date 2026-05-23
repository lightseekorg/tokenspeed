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
import torch.nn.functional as F
from tokenspeed_kernel.numerics.moe import (
    canonicalize_align_block_size,
    compute_align_block_size_buffer_dims,
)
from tokenspeed_kernel.ops.moe.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.torch_compile import get_compiler_backend

# ---------------------------------------------------------------------------
# Fused MoE (reference)
# ---------------------------------------------------------------------------


@register_kernel(
    "moe",
    "fused",
    name="reference_moe_fused",
    features={"pre_routed"},
    solution="reference",
    dtypes={torch.float16, torch.bfloat16, torch.float32},
    priority=Priority.REFERENCE,
    traits={
        "weight_dtype": frozenset({"bf16", "fp16", "fp32"}),
        "tp": frozenset({False}),
        "ep": frozenset({False}),
    },
    tags={"portability", "determinism"},
)
def fused_moe_forward_native(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    w13_weights = w13_weight[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = w2_weight[topk_ids]
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
    if activation == "silu":
        x1 = F.silu(x1)
    elif activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))


# ---------------------------------------------------------------------------
# Routing kernels
# ---------------------------------------------------------------------------


@register_kernel(
    "moe",
    "route",
    name="torch_compile_fused_topk_bias",
    solution="reference",
    dtypes={torch.float16, torch.bfloat16, torch.float32},
    traits={
        "output_type": frozenset({"topk"}),
        "biased": frozenset({True}),
        "grouped": frozenset({False}),
        "ep": frozenset({True, False}),
    },
    priority=Priority.PORTABLE + 3,
    tags={"portability", "determinism"},
)
@torch.compile(dynamic=True, backend=get_compiler_backend())
def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    n_routed_experts = gating_output.shape[-1]
    scores = gating_output.softmax(dim=-1)
    scores_for_choice = scores.view(-1, n_routed_experts) + correction_bias.unsqueeze(0)
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_indices = topk_ids_logical_to_physical(
        topk_indices, expert_location_dispatch_info
    )

    return topk_weights.to(torch.float32).to(hidden_states.device), topk_indices.to(
        torch.int32
    ).to(hidden_states.device)


@register_kernel(
    "moe",
    "route",
    name="torch_native_fused_topk",
    solution="reference",
    dtypes={torch.float16, torch.bfloat16, torch.float32},
    traits={
        "output_type": frozenset({"topk"}),
        "biased": frozenset({True, False}),
        "grouped": frozenset({False}),
        "ep": frozenset({False}),
    },
    priority=Priority.PORTABLE + 1,
    tags={"portability", "determinism"},
)
def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
):
    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = gating_output.softmax(dim=-1)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = F.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return
    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


# This is used by the Deepseek V2/V3/R1 series models
@register_kernel(
    "moe",
    "route",
    name="torch_compile_grouped_topk",
    solution="reference",
    dtypes={torch.float16, torch.bfloat16, torch.float32},
    traits={
        "output_type": frozenset({"topk"}),
        "biased": frozenset({False}),
        "grouped": frozenset({True}),
        "ep": frozenset({True, False}),
    },
    priority=Priority.PORTABLE + 3,
    tags={"portability", "determinism"},
)
@torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        factor = routed_scaling_factor or 1.0
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output and routed_scaling_factor is not None:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


@register_kernel(
    "moe",
    "route",
    name="torch_compile_biased_grouped_topk",
    solution="reference",
    dtypes={torch.float16, torch.bfloat16, torch.float32},
    traits={
        "output_type": frozenset({"topk"}),
        "biased": frozenset({True}),
        "grouped": frozenset({True}),
        "ep": frozenset({True, False}),
    },
    priority=Priority.PORTABLE + 3,
    tags={"portability", "determinism"},
)
@torch.compile(dynamic=True, backend=get_compiler_backend())
def biased_grouped_topk_gpu(
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
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
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
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
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
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# Align block size (reference)
# ---------------------------------------------------------------------------


@register_kernel(
    "moe",
    "align_block_size",
    name="torch_moe_align_block_size",
    solution="reference",
    dtypes={torch.int32},
    traits={},
    priority=10,
    tags={"determinism", "portability"},
)
def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> torch.Tensor:
    """Pure-torch reference for moe_align_block_size.

    Returns the canonical packed tensor (see ``canonicalize_align_block_size``).
    """
    assert topk_ids.dtype == torch.int32
    total_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    pad_id = total_tokens * top_k

    max_num_m_blocks, sorted_ids_size = compute_align_block_size_buffer_dims(
        pad_id, num_experts, block_size
    )

    # For each (token, k) slot, assign a flat-id 0..pad_id-1; group by expert.
    flat_token_ids = torch.arange(pad_id, device=device, dtype=torch.int32)
    flat_expert = topk_ids.flatten()

    # Gather slot ids per expert (e in [0, num_experts) is a real expert; e =
    # num_experts marks the filtered-out / "no expert" slot in EP setups).
    sorted_ids = torch.full(
        (sorted_ids_size,), pad_id, device=device, dtype=torch.int32
    )
    expert_ids = torch.zeros((max_num_m_blocks,), device=device, dtype=torch.int32)

    write_pos = 0
    block_idx = 0
    for e in range(num_experts + 1):
        slot_ids = flat_token_ids[flat_expert == e]
        n_slots = slot_ids.numel()
        # Pad to multiple of block_size.
        n_blocks = (n_slots + block_size - 1) // block_size
        for b in range(n_blocks):
            block_start = write_pos + b * block_size
            block_slot_count = min(block_size, n_slots - b * block_size)
            sorted_ids[block_start : block_start + block_slot_count] = slot_ids[
                b * block_size : b * block_size + block_slot_count
            ]
            expert_ids[block_idx + b] = e
        write_pos += n_blocks * block_size
        block_idx += n_blocks

    num_tokens_post_pad = torch.tensor([write_pos], device=device, dtype=torch.int32)
    return canonicalize_align_block_size(
        sorted_ids, expert_ids, num_tokens_post_pad, block_size
    )
