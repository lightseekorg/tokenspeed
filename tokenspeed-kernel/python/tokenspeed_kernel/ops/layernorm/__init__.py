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

import torch
from tokenspeed_kernel.ops.layernorm.triton import (
    grouped_gemma_rmsnorm as _grouped_gemma_rmsnorm,
)
from tokenspeed_kernel.ops.layernorm.triton import grouped_rmsnorm as _grouped_rmsnorm
from tokenspeed_kernel.platform import current_platform

_platform = current_platform()

if _platform.is_npu:
    from tokenspeed_kernel.ops.layernorm.ascend import qk_rmsnorm as _qk_rmsnorm
    from tokenspeed_kernel.ops.layernorm.ascend import rmsnorm as _rmsnorm
elif _platform.is_amd:
    from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm as _qk_rmsnorm
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm
else:
    from tokenspeed_kernel.ops.layernorm.flashinfer import (
        fused_add_rmsnorm as _fused_add_rmsnorm,
    )
    from tokenspeed_kernel.ops.layernorm.flashinfer import rmsnorm as _rmsnorm
    from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm as _qk_rmsnorm


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm with one platform-independent call contract."""
    if _platform.is_amd:
        if residual is not None:
            if out is not None:
                raise ValueError("fused add rmsnorm does not support out")
            return triton_rmsnorm(
                x,
                weight,
                eps,
                residual=residual,
            )
        return triton_rmsnorm(
            x,
            weight,
            eps,
            out=out,
        )
    if _platform.is_nvidia and residual is not None:
        if out is not None:
            raise ValueError("fused_add_rmsnorm does not support out")
        _fused_add_rmsnorm(
            x,
            residual,
            weight,
            eps,
        )
        return x, residual
    if _platform.is_nvidia:
        return _rmsnorm(x, weight, eps, out=out)
    if residual is not None:
        if out is not None:
            raise ValueError("rmsnorm does not support residual and out together")
        return _rmsnorm(x, weight, eps, residual=residual)
    return _rmsnorm(x, weight, eps, out=out)


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


__all__ = ["grouped_gemma_rmsnorm", "grouped_rmsnorm", "qk_rmsnorm", "rmsnorm"]
