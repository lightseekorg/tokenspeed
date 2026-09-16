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

import tokenspeed_kernel.ops.activation.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.activation.triton  # noqa: F401
import torch
from tokenspeed_kernel.ops.activation.triton import add3
from tokenspeed_kernel.ops.activation.triton import situ_and_mul as triton_situ_and_mul
from tokenspeed_kernel.ops.gemm import _fp8_linear_activation
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.registry import register_kernel_api
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Apply SwiGLU through a registered implementation."""
    kernel = select_kernel(
        "activation",
        "silu_and_mul",
        format_signature(x=dense_tensor_format(x.dtype)),
        traits={"has_limit": limit is not None},
        solution=solution,
    )
    return kernel(
        x=x,
        out=out,
        enable_pdl=pdl_enabled(),
        limit=limit,
    )


register_kernel_api(
    family="activation",
    mode="silu_and_mul",
    public_api=silu_and_mul,
    warmup_config_type=None,
)


def prepare_fp8_linear_activation(
    plan: object,
    x: torch.Tensor,
    *,
    activation: str,
    limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
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

    Returns:
        Prepared FP8 values and scales, or ``None`` when no fused contract is
        available.
    """
    return _fp8_linear_activation(
        plan,
        x,
        activation=activation,
        limit=limit,
        alpha=alpha,
        beta=beta,
        enable_pdl=pdl_enabled(),
    )


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """Apply SiTU through the portable Triton implementation."""

    return triton_situ_and_mul(
        x,
        out,
        beta=beta,
        linear_beta=linear_beta,
        enable_pdl=pdl_enabled(),
    )


__all__ = [
    "add3",
    "prepare_fp8_linear_activation",
    "silu_and_mul",
    "situ_and_mul",
]
