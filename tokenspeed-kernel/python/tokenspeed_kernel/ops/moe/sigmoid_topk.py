# Copyright (c) 2026 LightSeek Foundation

"""Biased sigmoid top-k routing entry point."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.moe.triton.kimi3_sigmoid_topk import (
    kimi3_sigmoid_bias_topk,
)
from tokenspeed_kernel.platform import CapabilityRequirement, Platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


def _gluon_eligible(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
) -> bool:
    tokens, experts = router_logits.shape
    return (
        router_logits.is_cuda
        and router_logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and tokens > 0
        and 0 < topk <= 16
        and topk <= experts <= 1024
        and router_logits.stride(1) == 1
        and correction_bias.is_contiguous()
    )


def moe_sigmoid_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
    logical_to_physical_map: torch.Tensor | None = None,
    weights_dtype: torch.dtype = torch.float32,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using ``sigmoid(logits) + correction_bias``.

    Args:
        router_logits: Router logits with shape ``[tokens, experts]``.
        correction_bias: Expert-selection bias with shape ``[experts]``.
        topk: Number of experts selected per token.
        routed_scaling_factor: Scale applied after optional normalization.
        normalize_topk_weights: Normalize selected sigmoid scores when true.
        logical_to_physical_map: Optional ``[experts]`` integer map applying
            static expert-location dispatch to the selected ids (fused into
            the specialized decode kernel; a gather elsewhere).
        weights_dtype: Output dtype for the route weights (selection always
            runs in FP32).
        solution: Optional implementation override.

    Returns:
        ``(topk_weights, topk_ids)`` with shapes ``[tokens, topk]``. Weights
        are ``weights_dtype`` and ids are INT32 (physical ids when a dispatch
        map is given).
    """
    if router_logits.dim() != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    tokens, experts = router_logits.shape
    if correction_bias.shape != (experts,):
        raise ValueError("correction_bias must have shape [experts]")
    if correction_bias.device != router_logits.device:
        raise ValueError("correction_bias and router_logits must share a device")
    if logical_to_physical_map is not None and (
        logical_to_physical_map.shape != (experts,)
        or logical_to_physical_map.dtype not in (torch.int32, torch.int64)
        or logical_to_physical_map.device != router_logits.device
        or not logical_to_physical_map.is_contiguous()
    ):
        raise ValueError(
            "logical_to_physical_map must be contiguous colocated INT32/INT64 "
            "[experts]"
        )
    if not 0 < topk <= experts:
        raise ValueError(f"topk must be in [1, {experts}], got {topk}")
    if tokens == 0:
        shape = (0, topk)
        return (
            torch.empty(shape, device=router_logits.device, dtype=weights_dtype),
            torch.empty(shape, device=router_logits.device, dtype=torch.int32),
        )
    platform = Platform.get()
    if (
        solution is None
        and (platform.is_nvidia or (platform.is_cdna4 and tokens == 1))
        and experts == 896
        and router_logits.dtype == torch.float32
        and correction_bias.dtype == torch.float32
        and router_logits.is_cuda
        and router_logits.is_contiguous()
        and correction_bias.is_contiguous()
        and topk == 16
    ):
        # CDNA4 retains M=1; NVIDIA runs one packed-key CTA per token.
        topk_weights, topk_ids = kimi3_sigmoid_bias_topk(
            router_logits,
            correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
            logical_to_physical_map=(
                logical_to_physical_map
                if logical_to_physical_map is not None
                and logical_to_physical_map.dtype == torch.int32
                else None
            ),
            weights_dtype=weights_dtype,
        )
        if (
            logical_to_physical_map is not None
            and logical_to_physical_map.dtype == torch.int64
        ):
            topk_ids = logical_to_physical_map[topk_ids.long()].to(torch.int32)
        return topk_weights, topk_ids
    if solution is None and not _gluon_eligible(router_logits, correction_bias, topk):
        solution = "torch"

    signature = format_signature(router_logits=dense_tensor_format(router_logits.dtype))
    traits = {"tokens": tokens, "experts": experts, "topk": topk}
    if (
        logical_to_physical_map is not None
        and logical_to_physical_map.dtype == torch.int32
        and solution is None
    ):
        try:
            mapped_kernel = select_kernel(
                "moe",
                "sigmoid_bias_topk_mapped",
                signature,
                traits=traits,
            )
        except NoKernelFoundError:
            mapped_kernel = None
        if mapped_kernel is not None:
            topk_weights, topk_ids = mapped_kernel(
                router_logits=router_logits,
                correction_bias=correction_bias,
                topk=topk,
                routed_scaling_factor=routed_scaling_factor,
                normalize_topk_weights=normalize_topk_weights,
                logical_to_physical_map=logical_to_physical_map,
            )
            if topk_weights.dtype != weights_dtype:
                topk_weights = topk_weights.to(weights_dtype)
            return topk_weights, topk_ids

    kernel = select_kernel(
        "moe",
        "sigmoid_bias_topk",
        signature,
        traits=traits,
        solution=solution,
    )
    topk_weights, topk_ids = kernel(
        router_logits=router_logits,
        correction_bias=correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
    )
    if logical_to_physical_map is not None:
        topk_ids = logical_to_physical_map[topk_ids.long()].to(torch.int32)
    if topk_weights.dtype != weights_dtype:
        topk_weights = topk_weights.to(weights_dtype)
    return topk_weights, topk_ids


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="triton_minimax_sigmoid_bias_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PERFORMANT,
    tags={"nvidia", "cuda_graph"},
)
def triton_minimax_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused single-kernel routing via the minimax biased-topk Triton kernel.

    The ungrouped biased-sigmoid case is exactly minimax routing with one
    expert group; the multi-launch torch reference costs ~3x on NVIDIA
    (40us vs 13us at [1, 896] topk=16). ``hidden_states`` is only shape-
    validated by the kernel wrapper, so the logits stand in for it.
    """
    from tokenspeed_kernel.thirdparty.triton import minimax_biased_grouped_topk

    topk_weights, topk_ids = minimax_biased_grouped_topk(
        router_logits,
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=normalize_topk_weights,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        weights_dtype=torch.float32,
    )
    if not normalize_topk_weights:
        topk_weights.mul_(routed_scaling_factor)
    return topk_weights, topk_ids.to(torch.int32)


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="torch_sigmoid_bias_topk",
    solution="torch",
    signatures=format_signatures(
        "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PORTABLE,
    tags={"portability", "reference"},
)
def torch_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
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
    return topk_weights.float(), topk_ids.to(torch.int32)


import tokenspeed_kernel.ops.moe.gluon.sigmoid_topk  # noqa: E402,F401
import tokenspeed_kernel.ops.moe.triton.decode_sigmoid_topk  # noqa: E402,F401

__all__ = ["moe_sigmoid_bias_topk"]
