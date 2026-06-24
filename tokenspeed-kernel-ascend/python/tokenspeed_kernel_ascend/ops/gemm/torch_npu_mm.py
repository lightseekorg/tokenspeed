"""GEMM kernel for Ascend NPU."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def torch_npu_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor | None = None,
    b_scales: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense matrix multiply on Ascend NPU via the vllm-ascend linear path.

    TokenSpeed unquantized weights are stored as ``[out_features, in_features]``.
    ``F.linear`` consumes that layout directly and dispatches through the NPU
    PrivateUse1 backend, matching vllm-ascend's unquantized GEMM wrapper.

    Args:
        a: Left operand with shape [M, K].
        b: Weight operand with shape [N, K].
        a_scales: Unused dense-GEMM compatibility argument.
        b_scales: Unused dense-GEMM compatibility argument.
        out_dtype: Optional output dtype.
        alpha: Optional post-GEMM multiplier.
        block_size: Unused dense-GEMM compatibility argument.
        bias: Optional bias vector with shape [N].
        out: Optional output tensor.
    """
    if a_scales is not None or b_scales is not None:
        raise ValueError("Ascend dense torch_npu_mm does not accept scales")
    if block_size is not None:
        raise ValueError("Ascend dense torch_npu_mm does not accept block_size")

    out_dtype = out_dtype or a.dtype
    if alpha is None:
        result = F.linear(a, b, bias)
    else:
        result = F.linear(a, b)
        result = result * alpha.to(dtype=result.dtype)
        if bias is not None:
            result = result + bias.to(dtype=result.dtype)
    result = result.to(out_dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result
