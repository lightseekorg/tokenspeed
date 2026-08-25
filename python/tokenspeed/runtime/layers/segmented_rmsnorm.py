"""Engine-local segmented RMSNorm used by the K3 DSpark draft."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _segmented_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_segments: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    segment = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < hidden_size

    row_width = n_segments * hidden_size
    input_offsets = row * row_width + segment * hidden_size + offsets
    weight_offsets = segment * hidden_size + offsets

    x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + input_offsets, x * tl.rsqrt(variance + eps) * weight, mask=mask)


def segmented_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply independent RMSNorm to the final ``[segments, hidden]`` axes."""
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("segmented_rmsnorm requires CUDA tensors")
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")
    if weight.ndim != 2 or tuple(x.shape[-2:]) != tuple(weight.shape):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match "
            f"input segments {tuple(x.shape[-2:])}"
        )
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")

    out = torch.empty_like(x) if out is None else out
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("out must match x shape, dtype, and device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if x.numel() == 0:
        return out

    num_segments, hidden_size = weight.shape
    n_rows = x.numel() // (num_segments * hidden_size)
    block = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block >= 4096 else 4
    _segmented_rmsnorm_kernel[(n_rows, num_segments)](
        x,
        weight,
        out,
        num_segments,
        hidden_size,
        eps,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out
