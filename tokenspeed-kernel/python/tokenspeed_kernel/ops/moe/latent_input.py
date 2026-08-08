# Copyright (c) 2026 LightSeek Foundation

"""Input projections for latent-space MoE layers."""

from __future__ import annotations

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# Column tiles of a packed projection GEMM must not straddle a projection
# boundary, so every projection width has to be a multiple of the tile width.
REGION_ALIGNMENT = 128


def packed_projection_weight_view(
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
) -> torch.Tensor | None:
    """Return the single tensor backing three consecutive projection weights.

    Returns ``None`` unless the three weights are contiguous rows of one
    allocation in router/routed/shared order, which is what lets one GEMM read
    them as a single operand.
    """
    parts = (router_weight, routed_weight, shared_gate_up_weight)
    if not all(part.ndim == 2 and part.is_contiguous() for part in parts):
        return None
    hidden_size = router_weight.shape[1]
    if any(part.shape[1] != hidden_size for part in parts):
        return None
    # Adjacent addresses alone are not enough: separate allocations can land
    # back to back, and reading across them would run off the end of the first.
    storage = router_weight.untyped_storage()
    if any(part.untyped_storage().data_ptr() != storage.data_ptr() for part in parts):
        return None
    row_bytes = hidden_size * router_weight.element_size()
    address = router_weight.data_ptr()
    for part in parts:
        if part.data_ptr() != address:
            return None
        address += part.shape[0] * row_bytes
    if address > storage.data_ptr() + storage.nbytes():
        return None
    total_rows = sum(part.shape[0] for part in parts)
    return router_weight.as_strided((total_rows, hidden_size), (hidden_size, 1))


def _weights_packed(weights: tuple[torch.Tensor, ...]) -> bool:
    """Return whether one GEMM can read the three weights as a single operand.

    Beyond sharing an allocation, every projection width must be a multiple of
    the column-tile width so that no tile straddles two projections and the
    epilogue stays uniform across a program.
    """
    if packed_projection_weight_view(*weights) is None:
        return False
    return all(weight.shape[0] % REGION_ALIGNMENT == 0 for weight in weights)


def latent_moe_input_projections(
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
        "weights_packed": _weights_packed(weights),
        "hidden_size_multiple_64": hidden_size % 64 == 0,
    }
    try:
        kernel = select_kernel(
            "moe",
            "latent_input",
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
            "moe",
            "latent_input",
            kernel.name,
            hidden_states.dtype,
            traits,
        )
        with kernel_scope(
            "moe",
            "latent_input",
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


__all__ = ["latent_moe_input_projections", "packed_projection_weight_view"]
