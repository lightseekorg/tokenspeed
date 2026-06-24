"""RMSNorm kernels for Ascend NPU via torch_npu."""

from __future__ import annotations

import torch
import torch_npu

__all__ = ["rmsnorm", "rmsnorm_fused_parallel"]


def torch_npu_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm via torch_npu native operators.

    Args:
        x: Input tensor with shape [..., hidden_size].
        weight: Norm weight with shape [hidden_size].
        eps: Epsilon for numerical stability.
        residual: Optional residual tensor for fused add+norm.
        out: Optional output tensor (ignored when residual is provided).
    """
    if residual is not None:
        if x.dtype != residual.dtype:
            x = x.to(residual.dtype)
        result = torch_npu.npu_add_rms_norm(x, residual, weight, eps)
        return result[0], result[2]

    result = torch_npu.npu_rms_norm(x, weight, eps)
    normed = result[0]
    if out is not None:
        out.copy_(normed)
        return out
    return normed


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    return torch_npu_rmsnorm(x, weight, eps, residual=residual, out=out)


def rmsnorm_fused_parallel(
    input1: torch.Tensor,
    weight1: torch.Tensor,
    output1: torch.Tensor,
    input2: torch.Tensor,
    weight2: torch.Tensor,
    output2: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> None:
    del enable_pdl
    if input1.shape[0] == 0:
        return
    if not input1.is_contiguous():
        input1 = input1.contiguous()
    if not input2.is_contiguous():
        input2 = input2.contiguous()
    torch_npu_rmsnorm(input1, weight1, eps, out=output1)
    torch_npu_rmsnorm(input2, weight2, eps, out=output2)
