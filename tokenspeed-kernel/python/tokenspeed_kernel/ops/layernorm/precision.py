# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""RMSNorm output precision and per-head L2 clipping contracts."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.layernorm._triton.precision import (
    triton_add_rmsnorm_precision,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    pre_weight: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> None:
    if (
        x.ndim < 1
        or x.shape[-1] < 1
        or x.dtype not in _FLOAT_DTYPES
        or output_dtype not in _FLOAT_DTYPES
    ):
        raise ValueError("RMSNorm needs floating rows and a floating output dtype")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    for parameter in (weight, pre_weight):
        if parameter is not None and (
            parameter.shape != (x.shape[-1],)
            or parameter.device != x.device
            or parameter.dtype not in _FLOAT_DTYPES
            or not parameter.is_contiguous()
        ):
            raise ValueError(
                "RMSNorm weights must be contiguous floating vectors matching "
                "the input width and device"
            )


def rmsnorm_precision(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    pre_weight: torch.Tensor | None,
    output_dtype: torch.dtype,
    *,
    mean_by_reciprocal: bool,
) -> torch.Tensor:
    """Normalize final-dimension rows in FP32 and store the requested dtype.

    Args:
        x: Floating input shaped [..., width]. A ``[tokens, groups, width]``
            view whose groups' rows are adjacent (per-head rows of a column
            slice, say) is read without a copy.
        weight: Optional learned gain [width], applied after normalization.
        eps: Finite positive epsilon for the mean square.
        pre_weight: Optional scale [width], applied in FP32 before statistics.
        output_dtype: Explicit floating output dtype, including BF16 for FP32 x.
        mean_by_reciprocal: Take the mean square as a multiply by FP32
            ``1 / width``, not a correctly rounded divide.

    Returns:
        A contiguous tensor with x's shape in output_dtype.
    """
    _check_rmsnorm(x, weight, eps, pre_weight, output_dtype)
    if x.is_cuda:
        kernel = select_kernel(
            "layernorm",
            "rmsnorm_precision",
            format_signature(x=dense_tensor_format(x.dtype)),
        )
        return kernel(
            x,
            weight,
            eps,
            pre_weight,
            output_dtype,
            mean_by_reciprocal=mean_by_reciprocal,
        )
    values = x.float()
    if pre_weight is not None:
        values = values * pre_weight.float()
    squares = values.square().sum(dim=-1, keepdim=True)
    # An elementwise tensor divisor divides; a scalar one may become a reciprocal.
    mean = (
        squares * (1.0 / values.shape[-1])
        if mean_by_reciprocal
        else squares / torch.full_like(squares, values.shape[-1])
    )
    normalized = values * torch.rsqrt(mean + eps)
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.to(output_dtype)


def add_rmsnorm_precision(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    pre_weight: torch.Tensor | None,
    output_dtype: torch.dtype,
    sum_dtype: torch.dtype,
    *,
    mean_by_reciprocal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rmsnorm_precision`` of ``x.float() + residual.float()``, with that
    FP32 sum also returned in ``sum_dtype``, in one pass on CUDA.

    Args:
        x: Floating rows shaped [..., width].
        residual: Floating rows shaped like ``x``.
        weight: Optional learned gain [width], applied after normalization.
        eps: Finite positive epsilon for the mean square.
        pre_weight: Optional scale [width], applied in FP32 before statistics.
        output_dtype: Floating dtype of the normalized rows.
        sum_dtype: Floating dtype the sum is stored in.
        mean_by_reciprocal: As in ``rmsnorm_precision``.

    Returns:
        Contiguous ``(normalized, summed)``, both shaped like ``x``.
    """
    _check_rmsnorm(x, weight, eps, pre_weight, output_dtype)
    if (
        residual.shape != x.shape
        or residual.device != x.device
        or residual.dtype not in _FLOAT_DTYPES
        or sum_dtype not in _FLOAT_DTYPES
    ):
        raise ValueError("The residual must be floating rows matching the input")
    if x.is_cuda:
        return triton_add_rmsnorm_precision(
            x.contiguous(),
            residual.contiguous(),
            weight,
            eps,
            pre_weight,
            output_dtype,
            sum_dtype,
            mean_by_reciprocal=mean_by_reciprocal,
        )
    summed = x.float() + residual.float()
    return (
        rmsnorm_precision(
            summed,
            weight,
            eps,
            pre_weight,
            output_dtype,
            mean_by_reciprocal=mean_by_reciprocal,
        ),
        summed.to(sum_dtype),
    )


def l2_clip(
    x: torch.Tensor,
    threshold: float,
    eps: float,
    scale: torch.Tensor | None,
    output_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Clip vectors per head and materialize before optional token scaling.

    Args:
        x: BF16, FP16 or FP32 [tokens, heads, channels], with unit channel stride.
        threshold: Finite positive L2 limit after optional learned scaling.
        eps: Finite positive epsilon added to each vector's L2 norm.
        scale: Optional learned [heads, channels] floating multiplier.
        output_scale: Optional FP32 [tokens] multiplier applied after clipping.

    Returns:
        Contiguous output matching x. Clipped values round through x.dtype
        before output_scale multiplies them in FP32 and rounds again.
    """
    if (
        x.ndim != 3
        or min(x.shape[1:]) < 1
        or x.stride(-1) != 1
        or x.dtype not in _FLOAT_DTYPES
    ):
        raise ValueError(
            "L2 clipping needs floating [tokens, heads, channels] rows with unit channel stride"
        )
    if (
        not math.isfinite(threshold)
        or threshold <= 0
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError(
            "L2 clipping threshold and epsilon must be finite and positive"
        )
    if scale is not None and (
        scale.shape != x.shape[1:]
        or scale.device != x.device
        or scale.dtype not in _FLOAT_DTYPES
        or not scale.is_contiguous()
    ):
        raise ValueError(
            "L2 clipping scale must be a colocated contiguous [heads, channels] floating tensor"
        )
    if output_scale is not None and (
        output_scale.shape != (x.shape[0],)
        or output_scale.device != x.device
        or output_scale.dtype != torch.float32
        or not output_scale.is_contiguous()
    ):
        raise ValueError(
            "L2 output scale must be a colocated contiguous FP32 [tokens] vector"
        )
    values = x.float()
    if scale is not None:
        values = values * scale.float()
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    output = (values * (threshold / (norm + eps)).clamp(max=1.0)).to(x.dtype)
    if output_scale is not None:
        output = (output.float() * output_scale[:, None, None]).to(x.dtype)
    return output
