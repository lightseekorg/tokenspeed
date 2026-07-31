# Copyright (c) 2026 LightSeek Foundation

"""Independent input projections for latent-space MoE decode."""

from __future__ import annotations

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def moe_input_projections(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    *,
    gate_clamp: float,
    up_clamp: float | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the three independent consumers of a latent-MoE input.

    The operation computes FP32 router logits, a materialized routed latent,
    and a materialized gated shared-expert input. A backend may fuse the three
    independent projections; unsupported shapes execute the same composition
    with PyTorch.

    Args:
        hidden_states: Input shaped ``[tokens, hidden_size]``.
        router_weight: Router weight shaped ``[experts, hidden_size]``.
        routed_weight: Latent projection shaped ``[latent_size, hidden_size]``.
        shared_gate_up_weight: Stacked shared gate/up projection shaped
            ``[2 * shared_size, hidden_size]``.
        gate_clamp: Positive tanh clamp for the shared gate branch.
        up_clamp: Optional positive tanh clamp for the shared up branch.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        ``(router_logits, routed_input, shared_input)``.
    """
    if hidden_states.ndim != 2 or hidden_states.shape[0] < 1:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    tokens, hidden_size = hidden_states.shape
    weights = (router_weight, routed_weight, shared_gate_up_weight)
    if any(
        weight.ndim != 2
        or weight.shape[1] != hidden_size
        or weight.dtype != hidden_states.dtype
        or weight.device != hidden_states.device
        for weight in weights
    ):
        raise ValueError(
            "projection weights must match the input width, dtype, and device"
        )
    if shared_gate_up_weight.shape[0] % 2:
        raise ValueError("shared gate/up output width must be even")
    if gate_clamp <= 0.0 or (up_clamp is not None and up_clamp <= 0.0):
        raise ValueError("activation clamp values must be positive")

    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        router_weight=dense_tensor_format(router_weight.dtype),
        routed_weight=dense_tensor_format(routed_weight.dtype),
        shared_gate_up_weight=dense_tensor_format(shared_gate_up_weight.dtype),
    )
    traits = {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "num_experts": router_weight.shape[0],
        "latent_size": routed_weight.shape[0],
        "shared_size": shared_gate_up_weight.shape[0] // 2,
        "inputs_contiguous": all(
            tensor.is_contiguous() for tensor in (hidden_states, *weights)
        ),
    }
    try:
        kernel = select_kernel(
            "gemm",
            "moe_input_projections",
            signature,
            traits=traits,
            override=override,
            solution=solution,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        kernel = None

    if kernel is not None:
        ShapeCapture.get().record(
            "gemm",
            "moe_input_projections",
            kernel.name,
            hidden_states.dtype,
            traits,
        )
        with kernel_scope(
            "gemm",
            "moe_input_projections",
            hidden_states.dtype,
            kernel_name=kernel.name,
            **traits,
        ):
            return kernel(
                hidden_states=hidden_states,
                router_weight=router_weight,
                routed_weight=routed_weight,
                shared_gate_up_weight=shared_gate_up_weight,
                gate_clamp=gate_clamp,
                up_clamp=up_clamp,
            )

    router_logits = torch.nn.functional.linear(
        hidden_states.float(), router_weight.float()
    )
    routed_input = torch.nn.functional.linear(hidden_states, routed_weight)
    shared_raw = torch.nn.functional.linear(hidden_states, shared_gate_up_weight)
    gate, up = shared_raw.chunk(2, dim=-1)
    gate_fp32 = gate.float()
    up_fp32 = up.float()
    gate_fp32 = gate_clamp * torch.tanh(gate_fp32 / gate_clamp)
    gate_fp32 *= torch.sigmoid(gate.float())
    if up_clamp is not None:
        up_fp32 = up_clamp * torch.tanh(up_fp32 / up_clamp)
    shared_input = (gate_fp32 * up_fp32).to(hidden_states.dtype)
    return router_logits, routed_input, shared_input


__all__ = ["moe_input_projections"]
