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

"""PyTorch reference MoE kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

# ===-----------------------------------------------------------------------===#
# Routing: softmax top-k
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "softmax_topk",
    name="torch_softmax_topk",
    solution="torch",
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def torch_softmax_topk(
    *,
    router_logits: torch.Tensor,
    topk: int,
    topk_indices_dtype: torch.dtype,
    renormalize: bool,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for ordinary softmax top-k routing."""
    del enable_pdl
    logits = router_logits.float()
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1, sorted=True)
    if renormalize:
        topk_weights = torch.softmax(topk_logits, dim=-1)
    else:
        topk_weights = torch.softmax(logits, dim=-1).gather(-1, topk_ids)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(topk_indices_dtype)


# ===-----------------------------------------------------------------------===#
# Routing: sigmoid-bias top-k
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="torch_sigmoid_bias_topk",
    solution="torch",
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def torch_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation matching Kimi's existing routing path."""
    scores = router_logits.sigmoid()
    topk_ids = torch.topk(
        scores + correction_bias.unsqueeze(0), topk, dim=-1, sorted=False
    ).indices
    topk_weights = scores.gather(1, topk_ids)
    if normalize_topk_weights:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(weights_dtype), topk_ids.to(torch.int32)


# ===-----------------------------------------------------------------------===#
# Routing: sqrt-softplus top-k
# ===-----------------------------------------------------------------------===#


@register_kernel(
    "moe",
    "topk",
    name="torch_sqrt_softplus_topk",
    solution="torch",
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    traits={
        "routing_kind": frozenset({"plain", "bias", "hash"}),
        "score_function": frozenset({"sqrt_softplus"}),
    },
    priority=Priority.REFERENCE,
)
def torch_sqrt_softplus_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.sqrt(F.softplus(router_logits.float()))
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash routing requires input_ids")
        table = hash_indices_table.to(device=scores.device, dtype=torch.int64)
        ids = input_ids.reshape(-1).to(device=scores.device, dtype=torch.int64)
        topk_ids = table[ids]
    else:
        selection_scores = scores
        if correction_bias is not None:
            bias = correction_bias.to(device=scores.device, dtype=scores.dtype)
            if bias.ndim == 1:
                bias = bias.unsqueeze(0)
            selection_scores = selection_scores + bias
        topk_ids = torch.topk(
            selection_scores,
            k=top_k,
            dim=-1,
            sorted=True,
        ).indices
    topk_weights = scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(topk_weights.dtype).tiny)
    output_scores = scores if need_scores else router_logits
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(weights_dtype), topk_ids.to(torch.int32), output_scores


@register_kernel(
    "moe",
    "softmax_bias_topk",
    name="torch_softmax_bias_topk",
    solution="torch",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
    ),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    traits={"experts_max": frozenset({2048}), "tie_order": frozenset({"torch"})},
    features={"batch_invariant", "cuda_graph"},
    priority=Priority.REFERENCE,
)
def torch_softmax_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    num_real_experts: int,
    routed_scaling_factor: float,
    indices_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correction-bias routing in the trainer's torch order (``--router-topk torch``).

    fp32 ``torch.softmax`` keeps every probability bit; ``torch.topk`` on the
    biased probabilities with ``sorted=True`` gives PyTorch's tie order; the
    weights are the UNBIASED probabilities times ``routed_scaling_factor``.
    Zero experts (ids past the real experts) become ``-1`` and keep their
    weight so the model can apply its identity residual.

    Args:
        router_logits: ``[tokens, num_candidates]`` router logits (any float
            dtype); candidates are the real experts followed by zero experts.
        correction_bias: ``[num_candidates]`` fp32 selection bias.
        top_k: Experts selected per token.
        num_real_experts: Candidates below this id are real experts.
        routed_scaling_factor: Multiplier on the selected probabilities.
        indices_dtype: dtype of the returned ids.

    Returns:
        ``(weights, ids)`` with shapes ``[tokens, top_k]``; weights fp32, ids
        in ``indices_dtype`` with ``-1`` for zero experts.
    """
    probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    _, ids = torch.topk(probs + correction_bias, k=top_k, dim=-1, sorted=True)
    weights = probs.gather(1, ids) * routed_scaling_factor
    ids = ids.masked_fill(ids >= num_real_experts, -1).to(indices_dtype)
    return weights, ids
