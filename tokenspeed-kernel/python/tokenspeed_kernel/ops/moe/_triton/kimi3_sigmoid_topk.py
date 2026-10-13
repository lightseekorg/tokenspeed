# Copyright (c) 2026 LightSeek Foundation

"""Packed-key sigmoid top-k routing for Kimi K3 decode."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import Platform, current_platform


@triton.jit
def _float32_to_ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    full = tl.full(bits.shape, 0xFFFFFFFF, tl.uint32)
    return bits ^ tl.where((bits & sign) != 0, full, sign)


@triton.jit
def _kimi3_sigmoid_bias_topk_kernel(
    logits,
    correction_bias,
    topk_ids,
    topk_weights,
    logical_to_physical,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: tl.constexpr,
    HAS_DISPATCH_MAP: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    num_experts: tl.constexpr = 896
    padded_experts: tl.constexpr = 1024
    topk: tl.constexpr = 16
    # One program per row; every reduction below is row-local.
    row = tl.program_id(0)
    logits += row * num_experts
    topk_ids += row * topk
    topk_weights += row * topk
    expert = tl.arange(0, padded_experts)
    valid = expert < num_experts

    if ENABLE_PDL:
        # ``logits`` is the router GEMV output written by the predecessor
        # kernel; correction_bias / logical_to_physical are static, but the
        # very first load below consumes logits, so fence up front.
        tl.extra.cuda.gdc_wait()

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
    choice = scores + bias
    # Squash NaN like the grouped kernel: the packed key sorts NaN above finite.
    choice = tl.where(valid & (choice == choice), choice, -float("inf"))

    # Pack the ordered FP32 selection score and inverse expert id. A single
    # bitonic top-k then implements the reference's descending score order and
    # lower-id tie break without sixteen full-row reductions.
    packed = (_float32_to_ordered_key(choice).to(tl.uint64) << 32) | (
        padded_experts - expert
    ).to(tl.uint64)
    selected = tl.topk(packed, topk, dim=0)
    selected_ids = (padded_experts - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)

    # Selection uses score+bias, while route weights use the unbiased sigmoid.
    selected_logits = tl.load(logits + selected_ids).to(tl.float32)
    selected_weights = tl.sigmoid(selected_logits)
    if NORMALIZE_TOPK_WEIGHTS:
        denominator = tl.sum(selected_weights, axis=0)
        denominator = tl.where(denominator != 0.0, denominator, 1.0)
        selected_weights /= denominator
    selected_weights *= ROUTED_SCALING_FACTOR

    if HAS_DISPATCH_MAP:
        # Static expert-location dispatch: route each selected logical expert
        # to this rank's physical replica inside the same launch.
        selected_ids = tl.load(logical_to_physical + selected_ids).to(tl.int32)

    offset = tl.arange(0, topk)
    tl.store(topk_ids + offset, selected_ids)
    tl.store(topk_weights + offset, selected_weights)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def kimi3_sigmoid_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    logical_to_physical_map: torch.Tensor | None = None,
    weights_dtype: torch.dtype = torch.float32,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route Kimi K3 tokens to their top 16 of 896 experts, one row per CTA.

    Args:
        router_logits: Contiguous FP32 logits shaped ``[tokens, 896]``.
        correction_bias: Contiguous FP32 selection bias shaped ``[896]``.
        routed_scaling_factor: Scale applied to selected route weights.
        normalize_topk_weights: Normalize selected sigmoid scores when true.
        logical_to_physical_map: Optional contiguous integer map shaped
            ``[896]``; when given, selected logical expert ids are translated
            to physical ids inside the kernel (static EP dispatch).
        weights_dtype: Output dtype for the route weights (selection and
            normalization always run in FP32; the store casts).
        enable_pdl: Launch with programmatic dependent launch and fence the
            router-logits read with ``gdc_wait`` / ``gdc_launch_dependents``
            (NVIDIA only; ignored elsewhere). Safe only when the router GEMV
            that writes ``router_logits`` is chained on the same stream/graph.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[tokens, 16]`` with ``weights_dtype``
        weights and INT32 ids (physical ids when a dispatch map is given).
    """
    if (
        router_logits.dim() != 2
        or router_logits.shape[1] != 896
        # The kernel offsets its row base with an INT32 program id.
        or not 0 < router_logits.shape[0] <= (2**31 - 1) // 896
        or router_logits.dtype != torch.float32
        or not router_logits.is_cuda
        or not router_logits.is_contiguous()
    ):
        raise ValueError(
            "Kimi K3 top-k requires contiguous GPU FP32 logits [tokens, 896]"
        )
    tokens = router_logits.shape[0]
    if (
        correction_bias.shape != (896,)
        or correction_bias.dtype != torch.float32
        or correction_bias.device != router_logits.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("Kimi K3 top-k requires contiguous colocated FP32 bias [896]")
    if logical_to_physical_map is not None and (
        logical_to_physical_map.shape != (896,)
        or logical_to_physical_map.dtype not in (torch.int32, torch.int64)
        or logical_to_physical_map.device != router_logits.device
        or not logical_to_physical_map.is_contiguous()
    ):
        raise ValueError(
            "Kimi K3 top-k dispatch map must be a contiguous colocated "
            "int32/int64 [896] tensor"
        )

    topk_ids = torch.empty(
        (tokens, 16),
        dtype=torch.int32,
        device=router_logits.device,
    )
    topk_weights = torch.empty(
        (tokens, 16),
        dtype=weights_dtype,
        device=router_logits.device,
    )
    # waves_per_eu is a CDNA-only occupancy hint; NVIDIA Triton rejects it.
    amd_kwargs = {"waves_per_eu": 1} if Platform.get().is_amd else {}
    pdl_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _kimi3_sigmoid_bias_topk_kernel[(tokens,)](
        router_logits,
        correction_bias,
        topk_ids,
        topk_weights,
        logical_to_physical_map,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        HAS_DISPATCH_MAP=logical_to_physical_map is not None,
        ENABLE_PDL=enable_pdl,
        num_warps=8,
        num_stages=1,
        **amd_kwargs,
        **pdl_kwargs,
    )
    return topk_weights, topk_ids


__all__ = ["kimi3_sigmoid_bias_topk"]
