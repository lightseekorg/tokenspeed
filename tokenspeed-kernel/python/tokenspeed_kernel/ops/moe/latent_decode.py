# Copyright (c) 2026 LightSeek Foundation

"""Optional joint routed/shared decode for latent-space MoE models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _selection(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_weight: torch.Tensor,
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
):
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        w13_weight=dense_tensor_format(w13_weight.dtype),
        w13_scale=dense_tensor_format(w13_scale.dtype),
        w2_weight=dense_tensor_format(w2_weight.dtype),
        w2_scale=dense_tensor_format(w2_scale.dtype),
        topk_weights=dense_tensor_format(topk_weights.dtype),
        topk_ids=dense_tensor_format(topk_ids.dtype),
        shared_input=dense_tensor_format(shared_input.dtype),
        shared_weight=dense_tensor_format(shared_weight.dtype),
        routed_out=dense_tensor_format(routed_out.dtype),
        shared_out=dense_tensor_format(shared_out.dtype),
    )
    traits = {
        "tokens": hidden_states.shape[0],
        "latent_size": hidden_states.shape[1],
        "topk": topk_ids.shape[1],
        "num_local_experts": w13_weight.shape[0],
        "intermediate_size": w2_weight.shape[-1] * 2,
        "shared_size": shared_input.shape[1],
        "output_size": shared_weight.shape[0],
        "linear_weights": True,
        "inputs_contiguous": all(
            tensor.is_contiguous()
            for tensor in (
                hidden_states,
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                topk_weights,
                topk_ids,
                shared_input,
                shared_weight,
                routed_out,
                shared_out,
            )
        ),
    }
    return signature, traits


def latent_moe_decode_pipeline_available(
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    expert_plan: Mapping[str, Any],
    *,
    joint_reduce: bool,
) -> bool:
    """Return whether joint routed/shared one-token decode is available.

    This construction-time probe owns backend, tensor-format, expert-plan, and
    exact-shape constraints needed by the orchestration-level optimization.
    """
    return (
        joint_reduce
        and expert_plan.get("apply_kernel_name")
        == "gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply"
        and router_weight.dtype
        == routed_weight.dtype
        == shared_gate_up_weight.dtype
        == shared_down_weight.dtype
        == torch.bfloat16
        and all(
            weight.is_cuda and weight.is_contiguous()
            for weight in (
                router_weight,
                routed_weight,
                shared_gate_up_weight,
                shared_down_weight,
            )
        )
    )


def latent_moe_expert_shared(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_weight: torch.Tensor,
    *,
    activation_clamp: float,
    linear_clamp: float | None,
    expert_start: int,
    w13_interleaved: bool,
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run routed experts and an independent shared down-projection jointly.

    This operation is optional and has no opaque fallback: callers use
    :func:`latent_moe_decode_pipeline_available` before choosing the joint
    orchestration. Ordinary model execution remains the fallback.
    """
    signature, traits = _selection(
        hidden_states,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        topk_weights,
        topk_ids,
        shared_input,
        shared_weight,
        routed_out,
        shared_out,
    )
    kernel = select_kernel(
        "moe",
        "latent_expert_shared",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    result = kernel(
        hidden_states=hidden_states,
        w13_weight=w13_weight,
        w13_scale=w13_scale,
        w2_weight=w2_weight,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_input=shared_input,
        shared_weight=shared_weight,
        activation_clamp=activation_clamp,
        linear_clamp=linear_clamp,
        expert_start=expert_start,
        w13_interleaved=w13_interleaved,
        routed_out=routed_out,
        shared_out=shared_out,
    )
    if not isinstance(result, tuple):
        raise RuntimeError("joint latent MoE kernel did not return both outputs")
    return result


__all__ = [
    "latent_moe_decode_pipeline_available",
    "latent_moe_expert_shared",
]
