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

"""Portable DeepSeek V4 sqrt-softplus expert routing."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _select_experts_kernel(
    logits_ptr,
    bias_ptr,
    hash_ptr,
    ids_ptr,
    weights_ptr,
    experts_ptr,
    scores_ptr,
    logits_stride_m,
    logits_stride_e,
    bias_stride,
    hash_stride_m,
    hash_stride_k,
    EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HASH: tl.constexpr,
    BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    NEED_SCORES: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK_E)
    logits = tl.load(
        logits_ptr + token * logits_stride_m + expert * logits_stride_e,
        mask=expert < EXPERTS,
        other=0,
    ).to(tl.float32)
    softplus = tl.maximum(logits, 0.0) + libdevice.log1p(tl.exp(-tl.abs(logits)))
    scores = tl.sqrt(tl.where(logits > 20.0, logits, softplus))
    if NEED_SCORES:
        tl.store(scores_ptr + token * EXPERTS + expert, scores, mask=expert < EXPERTS)
    lanes = tl.arange(0, BLOCK_K)
    chosen_ids = tl.full((BLOCK_K,), -1, tl.int32)
    chosen_weights = tl.zeros((BLOCK_K,), tl.float32)
    choice = scores
    if BIAS:
        choice += tl.load(
            bias_ptr + expert * bias_stride, mask=expert < EXPERTS, other=0
        )
    choice = tl.where(expert < EXPERTS, choice, -float("inf"))
    if HASH:
        input_id = tl.load(ids_ptr + token).to(tl.int64)
    for rank in tl.static_range(TOPK):
        if HASH:
            selected = tl.load(
                hash_ptr + input_id * hash_stride_m + rank * hash_stride_k
            ).to(tl.int32)
        else:
            selected = tl.min(
                tl.where(
                    (choice == tl.max(choice, 0)) & (expert < EXPERTS), expert, EXPERTS
                ),
                0,
            )
            choice = tl.where(expert == selected, -float("inf"), choice)
        weight = tl.sum(tl.where(expert == selected, scores, 0.0), 0)
        chosen_ids = tl.where(lanes == rank, selected, chosen_ids)
        chosen_weights = tl.where(lanes == rank, weight, chosen_weights)
    if RENORMALIZE:
        chosen_weights /= tl.maximum(tl.sum(chosen_weights, 0), 1.1754943508222875e-38)
    tl.store(weights_ptr + token * TOPK + lanes, chosen_weights, mask=lanes < TOPK)
    tl.store(experts_ptr + token * TOPK + lanes, chosen_ids, mask=lanes < TOPK)


@register_kernel(
    "moe",
    "dsv4_select_experts",
    name="triton_dsv4_select_experts",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    traits={"routing_kind": frozenset({"plain", "bias", "hash"})},
    priority=Priority.PORTABLE,
    tags={"portability", "routing"},
)
def triton_dsv4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select experts using the public dsv4_select_experts contract.

    Args:
        router_logits: Floating-point logits shaped [tokens, experts].
        top_k: Number of routes per token.
        renormalize: Normalize selected unbiased scores to sum to one.
        correction_bias: Optional selection-only bias, one value per expert.
        hash_indices_table: Optional [vocabulary, top_k] expert lookup table.
        input_ids: Token ids indexing the hash table.
        need_scores: Materialize all sqrt-softplus scores when true.

    Returns:
        FP32 route weights, INT32 expert ids, and FP32 scores (or the unused
        input logits when need_scores is false).
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if not router_logits.is_cuda:
        raise RuntimeError("Triton DSV4 routing requires GPU tensors")
    tokens, experts = router_logits.shape
    if not 0 < top_k <= experts:
        raise ValueError("top_k must be between one and the number of experts")
    bias_routing = correction_bias is not None and hash_indices_table is None
    if bias_routing and correction_bias.device != router_logits.device:
        raise ValueError("correction_bias must share the router_logits device")
    weights = torch.empty(
        (tokens, top_k), dtype=torch.float32, device=router_logits.device
    )
    ids = torch.empty((tokens, top_k), dtype=torch.int32, device=router_logits.device)
    scores = (
        torch.empty((tokens, experts), dtype=torch.float32, device=router_logits.device)
        if need_scores
        else router_logits
    )
    if tokens:
        _select_experts_kernel[(tokens,)](
            router_logits,
            correction_bias,
            hash_indices_table,
            None if input_ids is None else input_ids.contiguous(),
            weights,
            ids,
            scores,
            router_logits.stride(0),
            router_logits.stride(1),
            correction_bias.stride(0) if bias_routing else 0,
            hash_indices_table.stride(0) if hash_indices_table is not None else 0,
            hash_indices_table.stride(1) if hash_indices_table is not None else 0,
            EXPERTS=experts,
            TOPK=top_k,
            BLOCK_E=triton.next_power_of_2(experts),
            BLOCK_K=triton.next_power_of_2(top_k),
            HASH=hash_indices_table is not None,
            BIAS=bias_routing,
            RENORMALIZE=renormalize,
            NEED_SCORES=need_scores,
            num_warps=4,
        )
    return weights, ids, scores
