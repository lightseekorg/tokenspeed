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
        weight: Gemma checkpoint weight offset shaped ``[width]`` or
            ``[group_size]`` for shared group weights; the effective multiplier
            is ``1 + weight``.
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
    return _grouped_gemma_rmsnorm(
        x,
        weight,
        effective_group_size,
        eps,
        out=out,
        block_output=None,
        inject_logits=None,
        residual_out=None,
        preload_residual=False,
    )


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


def gated_residual_combine_norm(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_logits: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float,
    *,
    preload_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject a sublayer output and normalize each updated residual branch.

    One CTA computes both outputs for one token and branch. The updated
    residual is rounded to the input dtype before computing RMS statistics,
    matching a standalone combine followed by grouped Gemma RMSNorm.

    Args:
        block_output: GPU sublayer output shaped ``[..., hidden_size]``.
        residual: GPU residual streams shaped ``[..., hc_count * hidden_size]``.
        inject_logits: Per-branch gate logits shaped ``[..., hc_count]``.
            The injection multiplier is ``2 * sigmoid(inject_logits)``.
        weight: Gemma weight offsets shaped ``[hc_count * hidden_size]`` or
            ``[hidden_size]`` for weights shared across branches. The effective
            norm multiplier is ``1 + weight``.
        hc_count: Number of residual branches, greater than one.
        hidden_size: Width of one branch and the RMS reduction size.
        eps: Epsilon added to the per-branch mean square.
        preload_residual: Whether residual and weight are already visible before
            the current PDL producer starts. If true, load both before the PDL
            wait; block output and injection logits are always read after it.

    Returns:
        Updated residual and normalized residual, both matching the residual
        shape and dtype. Inputs are left unchanged.
    """
    if not residual.is_cuda:
        raise ValueError("gated_residual_combine_norm requires GPU tensors")
    if hc_count <= 1 or hidden_size <= 0:
        raise ValueError("hc_count must exceed one and hidden_size must be positive")
    if residual.ndim < 1 or residual.shape[-1] != hc_count * hidden_size:
        raise ValueError("residual last dimension must equal hc_count * hidden_size")
    combined = torch.empty_like(residual, memory_format=torch.contiguous_format)
    normalized = _grouped_gemma_rmsnorm(
        residual,
        weight,
        hidden_size,
        eps,
        out=None,
        block_output=block_output,
        inject_logits=inject_logits,
        residual_out=combined,
        preload_residual=preload_residual,
    )
    return combined, normalized


__all__ = [
    "gated_residual_combine_norm",
    "grouped_gemma_rmsnorm",
    "grouped_rmsnorm",
    "qk_rmsnorm",
    "rmsnorm",
]
