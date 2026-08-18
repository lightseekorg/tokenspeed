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

"""DeepSeek V4 expert selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cuda.routing import (
    hash_softplus_sqrt_topk_flash,
    softplus_sqrt_topk_flash,
)


def _routing_kind(
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> str:
    if hash_indices_table is not None:
        return "hash"
    if correction_bias is not None:
        return "bias"
    return "plain"


def deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    need_scores: bool = True,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select DeepSeek V4 experts from sqrt-softplus router scores.

    Correction bias affects selection only; returned weights are gathered from
    the unbiased scores. Hash routing uses the checkpoint table for expert ids.

    Args:
        router_logits: Router logits shaped [tokens, experts].
        top_k: Number of experts selected for each token.
        renormalize: Normalize selected weights to sum to one when true.
        correction_bias: Optional selection-only bias shaped [experts].
        hash_indices_table: Optional token-id to expert-id table.
        input_ids: Token ids used with hash_indices_table.
        need_scores: Whether callers consume the full score tensor. Specialized
            kernels avoid materializing it when false.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        FP32 weights, INT32 expert ids, and a tensor shaped [tokens, experts].
        The first two tensors have shape [tokens, top_k]. When need_scores is
        false, a specialized kernel may return router_logits as the ignored
        third value instead of materializing scores.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    if not router_logits.is_floating_point():
        raise ValueError("router_logits must be a floating-point tensor")
    tokens, experts = router_logits.shape
    if not 0 < top_k <= experts:
        raise ValueError(f"top_k must be in [1, {experts}], got {top_k}")
    if correction_bias is not None and correction_bias.shape != (experts,):
        raise ValueError(f"correction_bias must have shape [{experts}]")
    if hash_indices_table is not None and input_ids is None:
        raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")

    routing_kind = _routing_kind(correction_bias, hash_indices_table)
    traits = {
        "tokens": int(tokens),
        "experts": experts,
        "top_k": int(top_k),
        "renormalize": bool(renormalize),
        "routing_kind": routing_kind,
    }
    signature = format_signature(router_logits=dense_tensor_format(router_logits.dtype))
    kernel = select_kernel(
        "moe",
        "deepseek_v4_select_experts",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    shape_params = {
        "tokens": int(tokens),
        "experts": int(experts),
        "top_k": int(top_k),
        "renormalize": bool(renormalize),
        "routing_kind": routing_kind,
        "need_scores": bool(need_scores),
    }
    ShapeCapture.get().record(
        "moe",
        "deepseek_v4_select_experts",
        kernel.name,
        router_logits.dtype,
        shape_params,
    )
    try:
        with kernel_scope(
            "moe",
            "deepseek_v4_select_experts",
            router_logits.dtype,
            kernel_name=kernel.name,
            **shape_params,
        ):
            return kernel(
                router_logits,
                top_k,
                renormalize,
                correction_bias,
                hash_indices_table,
                input_ids,
                need_scores,
            )
    except (AttributeError, RuntimeError):
        spec = KernelRegistry.get().get_by_name(kernel.name)
        if override is not None or solution is not None or spec is None:
            raise
        if spec.solution == "torch":
            raise
        fallback = select_kernel(
            "moe",
            "deepseek_v4_select_experts",
            signature,
            traits=traits,
            solution="torch",
        )
        return fallback(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            need_scores,
        )


@register_kernel(
    "moe",
    "deepseek_v4_select_experts",
    name="cuda_deepseek_v4_select_experts",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    traits={
        "experts": frozenset({256, 384}),
        "top_k": frozenset({6}),
        "renormalize": frozenset({True}),
        "routing_kind": frozenset({"bias", "hash"}),
    },
    priority=Priority.SPECIALIZED,
    tags={"nvidia", "routing", "latency"},
)
def cuda_deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused CUDA sqrt-softplus expert selector."""
    logits_f32 = router_logits.float().contiguous()
    topk_weights = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=torch.int32,
        device=router_logits.device,
    )
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        hash_softplus_sqrt_topk_flash(
            logits_f32,
            input_ids.reshape(-1).to(device=router_logits.device).contiguous(),
            hash_indices_table.to(
                device=router_logits.device,
                dtype=torch.int32,
            ).contiguous(),
            topk_ids,
            topk_weights,
            1.0,
            renormalize,
        )
    elif correction_bias is not None:
        softplus_sqrt_topk_flash(
            logits_f32,
            correction_bias.to(
                device=router_logits.device,
                dtype=torch.float32,
            ).contiguous(),
            topk_ids,
            topk_weights,
            1.0,
            renormalize,
        )
    else:
        raise ValueError("fused DeepSeek V4 selection requires bias or hash routing")
    scores = (
        torch.sqrt(F.softplus(router_logits.float())) if need_scores else router_logits
    )
    return topk_weights, topk_ids, scores


@register_kernel(
    "moe",
    "deepseek_v4_select_experts",
    name="torch_deepseek_v4_select_experts",
    solution="torch",
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    priority=Priority.PORTABLE,
    tags={"portability", "reference", "routing"},
)
def torch_deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run portable sqrt-softplus expert selection with PyTorch."""
    scores = torch.sqrt(F.softplus(router_logits.float()))
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        table = hash_indices_table.to(device=scores.device, dtype=torch.int64)
        ids = input_ids.reshape(-1).to(device=scores.device, dtype=torch.int64)
        topk_ids = table[ids]
    else:
        scores_for_choice = scores
        if correction_bias is not None:
            scores_for_choice = scores_for_choice + correction_bias.to(
                device=scores.device,
                dtype=scores.dtype,
            ).unsqueeze(0)
        topk_ids = torch.topk(
            scores_for_choice,
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
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32), scores


__all__ = ["deepseek_v4_select_experts"]
