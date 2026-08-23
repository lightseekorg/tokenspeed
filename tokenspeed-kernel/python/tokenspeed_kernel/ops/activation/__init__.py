# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Activation kernel entry points."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.activation.triton import (
    add3,
)
from tokenspeed_kernel.ops.activation.triton import silu_and_mul as triton_silu_and_mul
from tokenspeed_kernel.ops.activation.triton import (
    situ_and_mul,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
    limit: float | None = None,
) -> torch.Tensor:
    """Apply SwiGLU through the platform implementation.

    Positive ``limit`` values use the portable Triton implementation because
    the CUDA implementation does not expose the checkpoint's clamp semantics.
    """
    if (
        limit is not None
        or current_platform().is_amd
        or flashinfer_silu_and_mul is error_fn
    ):
        return triton_silu_and_mul(x, out, enable_pdl=enable_pdl, limit=limit)
    return flashinfer_silu_and_mul(x, out, enable_pdl=enable_pdl)


def prepare_fp8_linear_activation(
    plan: object,
    x: torch.Tensor,
    *,
    activation: str,
    limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Prepare an activation for a compatible block-FP8 linear plan.

    The prepared linear implementation decides whether it can fuse activation
    and quantization. ``None`` means the caller must evaluate the activation
    normally and invoke the linear operation through its ordinary path.

    Args:
        plan: Opaque plan returned by the GEMM layer's ``prepare_fp8_linear``.
        x: Input to the activation.
        activation: Semantic activation name, currently ``"swiglu"``.
        limit: Optional activation clamp limit.
        alpha: Sigmoid multiplier for SwiGLU.
        beta: Value added to SwiGLU's up branch.
        enable_pdl: Join a Programmatic Dependent Launch chain when supported.

    Returns:
        Prepared FP8 values and scales, or ``None`` when no fused contract is
        available.
    """
    from tokenspeed_kernel.ops.gemm.fp8_linear import _fp8_linear_activation

    return _fp8_linear_activation(
        plan,
        x,
        activation=activation,
        limit=limit,
        alpha=alpha,
        beta=beta,
        enable_pdl=enable_pdl,
    )


def prepared_fp8_linear_supports_activation(plan: object, activation: str) -> bool:
    """Return whether a prepared FP8 linear accepts an activation boundary.

    Args:
        plan: Opaque plan returned by the GEMM layer's ``prepare_fp8_linear``.
        activation: Semantic activation name, such as ``"swiglu"``.

    Returns:
        Whether the prepared implementation supports fused activation input.
    """
    from tokenspeed_kernel.ops.gemm.fp8_linear import (
        _fp8_linear_supports_activation,
    )

    return _fp8_linear_supports_activation(plan, activation)


__all__ = [
    "add3",
    "prepare_fp8_linear_activation",
    "prepared_fp8_linear_supports_activation",
    "silu_and_mul",
    "situ_and_mul",
]
