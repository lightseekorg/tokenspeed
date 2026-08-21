# Copyright (c) 2026 LightSeek Foundation

"""Packed-key biased-sigmoid top-k routing for one decode token."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _float32_to_ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    full = tl.full(bits.shape, 0xFFFFFFFF, tl.uint32)
    return bits ^ tl.where((bits & sign) != 0, full, sign)


@triton.jit
def _decode_sigmoid_bias_topk_kernel(
    logits,
    correction_bias,
    logical_to_physical_map,
    topk_ids,
    topk_weights,
    NUM_EXPERTS: tl.constexpr,
    PADDED_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
):
    expert = tl.arange(0, PADDED_EXPERTS)
    valid = expert < NUM_EXPERTS

    all_logits = tl.load(
        logits + expert,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.sigmoid(all_logits)
    bias = tl.load(
        correction_bias + expert,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    choice = tl.where(valid, scores + bias, -float("inf"))

    # Pack the ordered FP32 selection score and inverse expert id. A single
    # bitonic top-k then implements the reference's descending score order and
    # lower-id tie break without sixteen full-row reductions.
    packed = (_float32_to_ordered_key(choice).to(tl.uint64) << 32) | (
        PADDED_EXPERTS - expert
    ).to(tl.uint64)
    selected = tl.topk(packed, TOPK, dim=0)
    selected_ids = (PADDED_EXPERTS - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)

    # Selection uses score+bias, while route weights use the unbiased sigmoid.
    selected_logits = tl.load(logits + selected_ids).to(tl.float32)
    selected_weights = tl.sigmoid(selected_logits)
    if NORMALIZE_TOPK_WEIGHTS:
        denominator = tl.sum(selected_weights, axis=0)
        denominator = tl.where(denominator != 0.0, denominator, 1.0)
        selected_weights /= denominator
    selected_weights *= ROUTED_SCALING_FACTOR

    offset = tl.arange(0, TOPK)
    output_ids = selected_ids
    if HAS_EXPERT_MAP:
        output_ids = tl.load(logical_to_physical_map + selected_ids)
    tl.store(topk_ids + offset, output_ids)
    tl.store(topk_weights + offset, selected_weights)


def _decode_sigmoid_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    logical_to_physical_map: torch.Tensor | None = None,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route one decode token using biased sigmoid scores.

    Args:
        router_logits: Contiguous FP32 logits shaped ``[1, experts]``.
        correction_bias: Contiguous FP32 selection bias shaped ``[experts]``.
        topk: Number of selected experts, at most 16.
        routed_scaling_factor: Scale applied to selected route weights.
        normalize_topk_weights: Normalize selected sigmoid scores when true.
        logical_to_physical_map: Optional contiguous INT32 mapping from logical
            router expert ids to physical expert ids.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[1, topk]`` with ``weights_dtype``
        and INT32 ids.
    """
    experts = router_logits.shape[1] if router_logits.ndim == 2 else 0
    if (
        router_logits.shape != (1, experts)
        or router_logits.dtype != torch.float32
        or not router_logits.is_cuda
        or not router_logits.is_contiguous()
        or not 0 < topk <= min(experts, 16)
        or experts > 1024
    ):
        raise ValueError(
            "decode sigmoid top-k requires contiguous GPU FP32 logits "
            "[1, experts], experts <= 1024, and topk <= 16"
        )
    if (
        correction_bias.shape != (experts,)
        or correction_bias.dtype != torch.float32
        or correction_bias.device != router_logits.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError(
            "decode sigmoid top-k requires contiguous colocated FP32 bias [experts]"
        )
    if logical_to_physical_map is not None and (
        logical_to_physical_map.shape != (experts,)
        or logical_to_physical_map.dtype != torch.int32
        or logical_to_physical_map.device != router_logits.device
        or not logical_to_physical_map.is_contiguous()
    ):
        raise ValueError(
            "decode sigmoid top-k expert map must be contiguous colocated INT32"
        )

    topk_ids = torch.empty(
        (1, topk),
        dtype=torch.int32,
        device=router_logits.device,
    )
    topk_weights = torch.empty(
        (1, topk),
        dtype=weights_dtype,
        device=router_logits.device,
    )
    padded_experts = triton.next_power_of_2(experts)
    _decode_sigmoid_bias_topk_kernel[(1,)](
        router_logits,
        correction_bias,
        topk_ids if logical_to_physical_map is None else logical_to_physical_map,
        topk_ids,
        topk_weights,
        NUM_EXPERTS=experts,
        PADDED_EXPERTS=padded_experts,
        TOPK=topk,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        HAS_EXPERT_MAP=logical_to_physical_map is not None,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )
    return topk_weights, topk_ids


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="triton_decode_sigmoid_bias_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures("router_logits", "dense", {torch.float32}),
    priority=Priority.SPECIALIZED + 1,
    traits={
        "tokens": frozenset({1}),
        "experts": frozenset(range(1, 1025)),
        "topk": frozenset(range(1, 17)),
    },
    tags={"decode", "routing", "cuda_graph"},
)
def triton_decode_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _decode_sigmoid_bias_topk(
        router_logits,
        correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        weights_dtype=weights_dtype,
    )


@register_kernel(
    "moe",
    "sigmoid_bias_topk_mapped",
    name="triton_decode_sigmoid_bias_topk_mapped",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures("router_logits", "dense", {torch.float32}),
    priority=Priority.SPECIALIZED + 1,
    traits={
        "tokens": frozenset({1}),
        "experts": frozenset(range(1, 1025)),
        "topk": frozenset(range(1, 17)),
    },
    tags={"decode", "routing", "expert_map", "cuda_graph"},
)
def triton_decode_sigmoid_bias_topk_mapped(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    logical_to_physical_map: torch.Tensor,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _decode_sigmoid_bias_topk(
        router_logits,
        correction_bias,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
        logical_to_physical_map=logical_to_physical_map,
        weights_dtype=weights_dtype,
    )


__all__ = [
    "triton_decode_sigmoid_bias_topk",
    "triton_decode_sigmoid_bias_topk_mapped",
]
