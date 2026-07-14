# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Fail-closed Triton Top-4 routing for MiniMax-M3."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton

_TOPK = 4
_MAX_EXPERTS = 256
_ROUTING_BLOCK = 1024


@triton.jit
def _minimax_m3_topk_kernel(
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
    routed_scaling_factor: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
):
    token_id = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts

    logits = tl.load(
        gating_output_ptr
        + token_id * stride_gm
        + expert_offsets * stride_ge,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)
    correction_bias = tl.load(
        correction_bias_ptr + expert_offsets,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.sigmoid(logits)
    choice_scores = tl.where(
        expert_mask,
        scores + correction_bias,
        -float("inf"),
    )

    weights_sum = 0.0
    for topk_offset in tl.static_range(0, TOPK):
        best_choice_score = tl.max(choice_scores, axis=0)
        best_expert = tl.min(
            tl.where(
                choice_scores == best_choice_score,
                expert_offsets,
                BLOCK_E,
            ),
            axis=0,
        )
        best_weight = tl.max(
            tl.where(expert_offsets == best_expert, scores, 0.0),
            axis=0,
        )
        weights_sum += best_weight

        tl.store(
            topk_ids_ptr
            + token_id * stride_im
            + topk_offset * stride_ik,
            best_expert.to(tl.int32),
        )
        tl.store(
            topk_weights_ptr
            + token_id * stride_wm
            + topk_offset * stride_wk,
            best_weight,
        )
        choice_scores = tl.where(
            expert_offsets == best_expert,
            -float("inf"),
            choice_scores,
        )

    denominator = tl.where(weights_sum != 0.0, weights_sum, 1.0)
    for topk_offset in tl.static_range(0, TOPK):
        weight = tl.load(
            topk_weights_ptr
            + token_id * stride_wm
            + topk_offset * stride_wk
        )
        weight = weight / denominator * routed_scaling_factor
        tl.store(
            topk_weights_ptr
            + token_id * stride_wm
            + topk_offset * stride_wk,
            weight,
        )


@triton.jit
def _minimax_m3_route_counts_kernel(
    topk_ids_ptr,
    col_sum_ptr,
    num_routes,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    count = 0
    for route_start in tl.range(0, num_routes, BLOCK):
        route_offsets = route_start + tl.arange(0, BLOCK)
        route_mask = route_offsets < num_routes
        route_ids = tl.load(
            topk_ids_ptr + route_offsets,
            mask=route_mask,
            other=-1,
        )
        valid = (route_ids >= 0) & (route_ids < num_experts)
        safe_ids = tl.where(valid, route_ids, 0)
        selected = ((safe_ids == expert) & route_mask).to(tl.int32)
        count += tl.sum(selected, axis=0)
    tl.store(col_sum_ptr + expert, count)


@triton.jit
def _minimax_m3_route_order_kernel(
    topk_weights_ptr,
    topk_ids_ptr,
    slice_offs_ptr,
    gather_indx_ptr,
    scatter_indx_ptr,
    gate_scal_ptr,
    num_routes,
    num_experts: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    output_start = tl.load(slice_offs_ptr + expert)
    routes_seen = 0
    for route_start in tl.range(0, num_routes, BLOCK):
        route_offsets = route_start + tl.arange(0, BLOCK)
        route_mask = route_offsets < num_routes
        route_ids = tl.load(
            topk_ids_ptr + route_offsets,
            mask=route_mask,
            other=-1,
        )
        valid = (route_ids >= 0) & (route_ids < num_experts)
        safe_ids = tl.where(valid, route_ids, 0)
        selected = (safe_ids == expert) & route_mask
        selected_int = selected.to(tl.int32)
        local_ranks = tl.cumsum(selected_int, axis=0) - 1
        output_offsets = output_start + routes_seen + local_ranks
        weights = tl.load(
            topk_weights_ptr + route_offsets,
            mask=route_mask,
            other=0.0,
        )
        weights = tl.where(valid, weights, 0.0)
        tl.store(gather_indx_ptr + output_offsets, route_offsets // TOPK, mask=selected)
        tl.store(scatter_indx_ptr + output_offsets, route_offsets, mask=selected)
        tl.store(gate_scal_ptr + output_offsets, weights, mask=selected)
        routes_seen += tl.sum(selected_int, axis=0)


@triton.jit
def _minimax_m3_reduce_topk_kernel(
    expert_output_ptr,
    output_ptr,
    hidden_size,
    stride_output_token,
    stride_output_topk,
    stride_output_hidden,
    stride_result_token,
    stride_result_hidden,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden_size
    accumulated = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for topk_offset in tl.static_range(0, TOPK):
        values = tl.load(
            expert_output_ptr
            + token * stride_output_token
            + topk_offset * stride_output_topk
            + hidden_offsets * stride_output_hidden,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        accumulated += values
    tl.store(
        output_ptr
        + token * stride_result_token
        + hidden_offsets * stride_result_hidden,
        accumulated,
        mask=hidden_mask,
    )


def _validate_route_inputs(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> int:
    if topk_ids.ndim != 2 or topk_ids.shape[1] != _TOPK:
        raise ValueError(
            "MiniMax-M3 route metadata requires INT32 ids shaped [tokens, 4], "
            f"got {tuple(topk_ids.shape)}."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MiniMax-M3 route weights and ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}."
        )
    if topk_ids.dtype != torch.int32:
        raise TypeError(
            f"MiniMax-M3 route ids must use INT32, got {topk_ids.dtype}."
        )
    if topk_weights.dtype != torch.float32:
        raise TypeError(
            f"MiniMax-M3 route weights must use FP32, got {topk_weights.dtype}."
        )
    if not topk_ids.is_cuda or not topk_weights.is_cuda:
        raise RuntimeError("MiniMax-M3 route metadata requires GPU tensors.")
    if topk_ids.device != topk_weights.device:
        raise RuntimeError("MiniMax-M3 route weights and ids must share one device.")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("MiniMax-M3 route weights and ids must be contiguous.")
    if not _TOPK <= num_experts <= _MAX_EXPERTS:
        raise ValueError(
            "MiniMax-M3 route metadata requires between 4 and 256 experts, got "
            f"{num_experts}."
        )
    return topk_ids.numel()


def minimax_m3_route_counts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count stable Top-4 routes per expert with a native Triton kernel."""
    num_routes = _validate_route_inputs(topk_weights, topk_ids, num_experts)
    col_sum = torch.empty(
        (num_experts,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    _minimax_m3_route_counts_kernel[(num_experts,)](
        topk_ids,
        col_sum,
        num_routes,
        num_experts=num_experts,
        BLOCK=_ROUTING_BLOCK,
        num_warps=4,
    )
    return col_sum


def minimax_m3_route_order(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    slice_offs: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize stable gather/scatter/gamma arrays for routed Top-4 GEMMs."""
    num_routes = _validate_route_inputs(topk_weights, topk_ids, num_experts)
    if (
        slice_offs.shape != (num_experts + 1,)
        or slice_offs.dtype != torch.int32
        or slice_offs.device != topk_ids.device
    ):
        raise ValueError(
            "MiniMax-M3 route slice offsets must be INT32 on the routing device "
            f"with shape {(num_experts + 1,)}, got shape={tuple(slice_offs.shape)}, "
            f"dtype={slice_offs.dtype}, device={slice_offs.device}."
        )
    gather_indx = torch.empty(
        (num_routes,), dtype=torch.int32, device=topk_ids.device
    )
    scatter_indx = torch.empty_like(gather_indx)
    gate_scal = torch.empty_like(topk_weights).reshape(-1)
    if num_routes:
        _minimax_m3_route_order_kernel[(num_experts,)](
            topk_weights,
            topk_ids,
            slice_offs,
            gather_indx,
            scatter_indx,
            gate_scal,
            num_routes,
            num_experts=num_experts,
            TOPK=_TOPK,
            BLOCK=_ROUTING_BLOCK,
            num_warps=4,
        )
    return gather_indx, scatter_indx, gate_scal


def minimax_m3_reduce_topk(expert_output: torch.Tensor) -> torch.Tensor:
    """Reduce MiniMax-M3's four routed expert outputs with Triton."""
    if expert_output.ndim != 3 or expert_output.shape[1] != _TOPK:
        raise ValueError(
            "MiniMax-M3 expert reduction requires [tokens, 4, hidden], got "
            f"{tuple(expert_output.shape)}."
        )
    if expert_output.dtype != torch.bfloat16:
        raise TypeError(
            f"MiniMax-M3 expert reduction requires BF16, got {expert_output.dtype}."
        )
    if not expert_output.is_cuda:
        raise RuntimeError("MiniMax-M3 expert reduction requires a GPU tensor.")
    if expert_output.stride(-1) != 1:
        raise ValueError("MiniMax-M3 expert output must be contiguous in hidden size.")
    num_tokens, _, hidden_size = expert_output.shape
    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=expert_output.dtype,
        device=expert_output.device,
    )
    if num_tokens:
        block_h = 1024
        _minimax_m3_reduce_topk_kernel[
            (num_tokens, triton.cdiv(hidden_size, block_h))
        ](
            expert_output,
            output,
            hidden_size,
            expert_output.stride(0),
            expert_output.stride(1),
            expert_output.stride(2),
            output.stride(0),
            output.stride(1),
            TOPK=_TOPK,
            BLOCK_H=block_h,
            num_warps=4,
        )
    return output


def minimax_m3_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route MiniMax-M3 tokens with its native Triton Top-4 algorithm.

    Args:
        hidden_states: BF16 token states shaped ``[tokens, hidden_size]``. The
            values are not read, but the token dimension is validated against
            ``gating_output`` to preserve the model routing contract.
        gating_output: FP32 router logits shaped ``[tokens, experts]``.
        correction_bias: FP32 expert-selection bias shaped ``[experts]``.
        topk: Number of selected experts. MiniMax-M3 requires exactly four.
        renormalize: Whether selected sigmoid scores are normalized. MiniMax-M3
            requires this to be true.
        routed_scaling_factor: Positive finite multiplier applied after
            normalization.

    Returns:
        A pair ``(topk_weights, topk_ids)`` with shapes ``[tokens, 4]`` and
        dtypes FP32 and INT32 respectively.

    Raises:
        ValueError: If a shape or MiniMax-M3 routing option is unsupported.
        TypeError: If an input dtype is unsupported.
        RuntimeError: If inputs are not colocated on one CUDA/ROCm device.

    This function is intentionally fail-closed. It never dispatches to a Torch
    reference or another kernel implementation.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            "MiniMax-M3 Top-4 routing requires 2D hidden states, got "
            f"shape={tuple(hidden_states.shape)}."
        )
    if gating_output.ndim != 2:
        raise ValueError(
            "MiniMax-M3 Top-4 routing requires 2D router logits, got "
            f"shape={tuple(gating_output.shape)}."
        )
    if correction_bias.ndim != 1:
        raise ValueError(
            "MiniMax-M3 Top-4 routing requires a 1D correction bias, got "
            f"shape={tuple(correction_bias.shape)}."
        )
    if hidden_states.shape[0] != gating_output.shape[0]:
        raise ValueError(
            "MiniMax-M3 Top-4 routing token count mismatch: "
            f"hidden_states={hidden_states.shape[0]}, "
            f"router_logits={gating_output.shape[0]}."
        )

    num_experts = gating_output.shape[1]
    if correction_bias.shape[0] != num_experts:
        raise ValueError(
            "MiniMax-M3 Top-4 routing expert count mismatch: "
            f"router_logits={num_experts}, correction_bias={correction_bias.shape[0]}."
        )
    if not _TOPK <= num_experts <= _MAX_EXPERTS:
        raise ValueError(
            "MiniMax-M3 Top-4 routing requires between 4 and 256 experts, got "
            f"{num_experts}."
        )
    if topk != _TOPK:
        raise ValueError(f"MiniMax-M3 routing requires topk=4, got {topk}.")
    if renormalize is not True:
        raise ValueError("MiniMax-M3 routing requires renormalize=True.")

    try:
        scale = float(routed_scaling_factor)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "MiniMax-M3 routed_scaling_factor must be a real number."
        ) from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "MiniMax-M3 routed_scaling_factor must be positive and finite, got "
            f"{routed_scaling_factor!r}."
        )

    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "MiniMax-M3 Top-4 routing requires BF16 hidden states, got "
            f"{hidden_states.dtype}."
        )
    if gating_output.dtype != torch.float32:
        raise TypeError(
            "MiniMax-M3 Top-4 routing requires FP32 router logits, got "
            f"{gating_output.dtype}."
        )
    if correction_bias.dtype != torch.float32:
        raise TypeError(
            "MiniMax-M3 Top-4 routing requires FP32 correction bias, got "
            f"{correction_bias.dtype}."
        )
    if correction_bias.stride(0) != 1:
        raise ValueError("MiniMax-M3 correction bias must be contiguous.")

    device = gating_output.device
    if (
        not hidden_states.is_cuda
        or not gating_output.is_cuda
        or not correction_bias.is_cuda
    ):
        raise RuntimeError("MiniMax-M3 Top-4 routing requires GPU tensors.")
    if hidden_states.device != device or correction_bias.device != device:
        raise RuntimeError(
            "MiniMax-M3 Top-4 routing inputs must be on the same device."
        )

    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(
        (num_tokens, _TOPK),
        dtype=torch.float32,
        device=device,
    )
    topk_ids = torch.empty(
        (num_tokens, _TOPK),
        dtype=torch.int32,
        device=device,
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    block_experts = triton.next_power_of_2(num_experts)
    _minimax_m3_topk_kernel[(num_tokens,)](
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        num_experts=num_experts,
        routed_scaling_factor=scale,
        BLOCK_E=block_experts,
        TOPK=_TOPK,
        num_warps=1,
    )
    return topk_weights, topk_ids


__all__ = [
    "minimax_m3_reduce_topk",
    "minimax_m3_route_counts",
    "minimax_m3_route_order",
    "minimax_m3_topk",
]
