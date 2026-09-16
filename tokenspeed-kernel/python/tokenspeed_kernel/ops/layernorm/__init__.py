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

"""Layernorm kernel entry points."""

from __future__ import annotations

import tokenspeed_kernel.ops.layernorm.flashinfer  # noqa: F401
import torch
from tokenspeed_kernel.ops.layernorm.triton import (
    grouped_gemma_rmsnorm as _grouped_gemma_rmsnorm,
)
from tokenspeed_kernel.ops.layernorm.triton import grouped_rmsnorm as _grouped_rmsnorm
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import register_kernel_api
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_platform = current_platform()

if _platform.is_npu:
    from tokenspeed_kernel.ops.layernorm.ascend import qk_rmsnorm as _qk_rmsnorm
    from tokenspeed_kernel.ops.layernorm.ascend import rmsnorm as _rmsnorm
elif _platform.is_amd:
    from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm as _qk_rmsnorm
else:
    from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm as _qk_rmsnorm


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
    solution: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm with one platform-independent call contract."""
    if residual is not None and out is not None:
        if _platform.is_nvidia:
            raise ValueError("fused_add_rmsnorm does not support out")
        if _platform.is_amd:
            raise ValueError("fused add rmsnorm does not support out")
        raise ValueError("rmsnorm does not support residual and out together")
    if _platform.is_npu:
        if residual is not None:
            return _rmsnorm(x, weight, eps, residual=residual)
        return _rmsnorm(x, weight, eps, out=out)
    kernel = select_kernel(
        "layernorm",
        "rmsnorm",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={
            "has_residual": residual is not None,
            "has_out": out is not None,
        },
        solution=solution,
    )
    return kernel(
        x=x,
        weight=weight,
        eps=eps,
        residual=residual,
        out=out,
        enable_pdl=enable_pdl,
    )


register_kernel_api(
    family="layernorm",
    mode="rmsnorm",
    public_api=rmsnorm,
    warmup_config_type=None,
)


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    kernel = select_kernel(
        "layernorm",
        "gemma_rmsnorm",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={},
        solution=solution,
    )
    return kernel(
        x=x,
        weight=weight,
        eps=eps,
        out=out,
        enable_pdl=enable_pdl,
    )


register_kernel_api(
    family="layernorm",
    mode="gemma_rmsnorm",
    public_api=gemma_rmsnorm,
    warmup_config_type=None,
)


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = select_kernel(
        "layernorm",
        "gemma_fused_add_rmsnorm",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={},
        solution=solution,
    )
    return kernel(
        x=x,
        residual=residual,
        weight=weight,
        eps=eps,
        enable_pdl=enable_pdl,
    )


register_kernel_api(
    family="layernorm",
    mode="gemma_fused_add_rmsnorm",
    public_api=gemma_fused_add_rmsnorm,
    warmup_config_type=None,
)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    solution: str | None = None,
) -> torch.Tensor:
    kernel = select_kernel(
        "layernorm",
        "layernorm",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={},
        solution=solution,
    )
    return kernel(x=x, weight=weight, bias=bias, eps=eps)


register_kernel_api(
    family="layernorm",
    mode="layernorm",
    public_api=layernorm,
    warmup_config_type=None,
)


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the platform per-head Q/K RMSNorm implementation."""
    return _qk_rmsnorm(
        q,
        k,
        q_weight,
        k_weight,
        eps,
    )


def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int | None,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Gemma RMSNorm independently to last-dimension groups.

    Args:
        x: GPU input shaped ``[..., width]``.
        weight: Gemma checkpoint weight offset shaped ``[width]``; the
            effective multiplier is ``1 + weight``.
        group_size: Elements sharing one variance statistic. ``None`` means
            the full last dimension.
        eps: Epsilon added before reciprocal square root.
        out: Optional contiguous output matching ``x``.

    Returns:
        Normalized tensor matching ``x`` shape and dtype.
    """
    if not x.is_cuda:
        raise ValueError("grouped_gemma_rmsnorm requires GPU tensors")
    width = int(x.shape[-1])
    effective_group_size = width if group_size is None else int(group_size)
    return _grouped_gemma_rmsnorm(x, weight, effective_group_size, eps, out=out)


def grouped_rmsnorm(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Apply weight-free RMSNorm to contiguous groups of the last dimension.

    Args:
        x: GPU input shaped ``[..., width]``.
        group_size: Number of contiguous values sharing one RMS statistic.
        eps: Epsilon added before reciprocal square root.
        out: Optional contiguous output matching ``x``; may alias ``x``.

    Returns:
        Normalized tensor matching ``x`` shape and dtype.
    """
    if not x.is_cuda:
        raise ValueError("grouped_rmsnorm requires GPU tensors")
    return _grouped_rmsnorm(x, int(group_size), eps, out=out)


__all__ = [
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "grouped_gemma_rmsnorm",
    "grouped_rmsnorm",
    "layernorm",
    "qk_rmsnorm",
    "rmsnorm",
]
