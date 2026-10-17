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

"""Weight-free and centered-affine RMSNorm entry points."""

import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_format_signature


def weightless_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    eps: float,
    q_scale: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q/K independently per head and scale normalized queries.

    Args:
        q: Query rows shaped ``[tokens, query_heads * head_dim]``.
        k: Key rows shaped ``[tokens, kv_heads * head_dim]``.
        head_dim: Width of each attention head.
        eps: Epsilon added to each per-head variance.
        q_scale: Multiplier applied only to normalized queries.
        enable_pdl: Enable NVIDIA programmatic dependent launch.

    Returns:
        Fresh contiguous normalized query and key tensors.
    """
    kernel = select_kernel(
        "layernorm",
        "weightless_qk_rmsnorm",
        dense_format_signature(("q", "k"), (q.dtype, k.dtype)),
    )
    return kernel(q, k, head_dim, eps, q_scale, enable_pdl)


def weightless_rmsnorm(
    x: torch.Tensor, eps: float, out_dtype: torch.dtype
) -> torch.Tensor:
    """Apply RMSNorm without an affine weight.

    Args:
        x: Input whose final dimension is normalized.
        eps: Epsilon added to the variance.
        out_dtype: Output dtype; the math runs in FP32 and rounds once on store.

    Returns:
        A contiguous normalized tensor with ``x``'s shape in ``out_dtype``.
    """
    kernel = select_kernel(
        "layernorm",
        "weightless_rmsnorm",
        dense_format_signature(("x",), (x.dtype,)),
    )
    return kernel(x, eps, out_dtype)


def affine_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply RMSNorm, then multiply by a centered affine weight.

    Args:
        x: Input whose final dimension is normalized.
        weight: Learned scale matching ``x``'s final dimension.
        eps: Epsilon added to the variance.
        weight_offset: Center added to ``weight`` before scaling.
        out_dtype: Output dtype; the math runs in FP32 and rounds once on store.

    Returns:
        A contiguous normalized tensor with ``x``'s shape in ``out_dtype``.
    """
    kernel = select_kernel(
        "layernorm", "affine_rmsnorm", dense_format_signature(("x",), (x.dtype,))
    )
    return kernel(x, weight, eps, weight_offset, out_dtype)


def weightless_rmsnorm_gated_sigmoid(
    x: torch.Tensor,
    gate: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
    enable_pdl: bool,
) -> torch.Tensor:
    """Fuse per-head weightless RMSNorm with an elementwise sigmoid gate.

    Args:
        x: Contiguous attention output shaped ``[tokens, heads * head_dim]``.
        gate: Gate logits with the same shape as ``x``.
        num_heads: Number of local attention heads.
        head_dim: Width of each attention head.
        eps: Epsilon added to each per-head variance.
        enable_pdl: Enable NVIDIA programmatic dependent launch.

    Returns:
        Gated, normalized attention output with ``x``'s shape and dtype.
    """
    kernel = select_kernel(
        "layernorm",
        "weightless_rmsnorm_gated_sigmoid",
        dense_format_signature(("x", "gate"), (x.dtype, gate.dtype)),
    )
    return kernel(x, gate, num_heads, head_dim, eps, enable_pdl)


__all__ = [
    "affine_rmsnorm",
    "weightless_qk_rmsnorm",
    "weightless_rmsnorm",
    "weightless_rmsnorm_gated_sigmoid",
]
