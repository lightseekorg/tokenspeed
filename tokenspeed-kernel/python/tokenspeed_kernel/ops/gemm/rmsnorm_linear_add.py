# Copyright (c) 2026 LightSeek Foundation

"""RMSNorm, linear projection, and residual additions."""

from __future__ import annotations

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def rmsnorm_linear_add(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    *addends: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Normalize and project hidden states, then accumulate residual rows."""

    if hidden_states.ndim != 2 or hidden_states.shape[0] < 1:
        raise ValueError("hidden_states must have shape [tokens, input_size]")
    tokens, input_size = hidden_states.shape
    if (
        norm_weight.shape != (input_size,)
        or linear_weight.ndim != 2
        or linear_weight.shape[1] != input_size
    ):
        raise ValueError("normalization and projection weights have invalid shapes")
    output_shape = (tokens, linear_weight.shape[0])
    tensors = (norm_weight, linear_weight, *addends)
    if any(
        tensor.dtype != hidden_states.dtype or tensor.device != hidden_states.device
        for tensor in tensors
    ):
        raise ValueError("weights and addends must match the input dtype and device")
    if any(tuple(addend.shape) != output_shape for addend in addends):
        raise ValueError(f"addends must have shape {output_shape}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if out is None:
        out = hidden_states.new_empty(output_shape)
    elif (
        tuple(out.shape) != output_shape
        or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous with shape {output_shape}")
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        norm_weight=dense_tensor_format(norm_weight.dtype),
        linear_weight=dense_tensor_format(linear_weight.dtype),
        out=dense_tensor_format(out.dtype),
    )
    traits = {
        "tokens": tokens,
        "input_size": input_size,
        "output_size": linear_weight.shape[0],
        "num_addends": len(addends),
        "inputs_contiguous": all(
            tensor.is_contiguous()
            for tensor in (hidden_states, norm_weight, linear_weight, *addends, out)
        ),
    }
    try:
        kernel = select_kernel(
            "gemm",
            "rmsnorm_linear_add",
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
            "rmsnorm_linear_add",
            kernel.name,
            hidden_states.dtype,
            traits,
        )
        with kernel_scope(
            "gemm",
            "rmsnorm_linear_add",
            hidden_states.dtype,
            kernel_name=kernel.name,
            **traits,
        ):
            return kernel(
                hidden_states=hidden_states,
                norm_weight=norm_weight,
                linear_weight=linear_weight,
                addends=addends,
                eps=eps,
                out=out,
            )

    normalized = hidden_states.float()
    inverse_rms = torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (normalized * inverse_rms * norm_weight.float()).to(
        hidden_states.dtype
    )
    projected = torch.nn.functional.linear(normalized, linear_weight)
    accumulated = projected.float()
    for addend in addends:
        accumulated = accumulated + addend.float()
    out.copy_(accumulated)
    return out


__all__ = ["rmsnorm_linear_add"]
