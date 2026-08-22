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

"""DeepSeek V4 FP32-output linear projection."""

from __future__ import annotations

from math import prod

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def deepseek_v4_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    enable_pdl: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Project DeepSeek V4 hidden states and return FP32 output.

    Args:
        hidden_states: Floating-point activations with trailing dimension K.
        weight: Floating-point row-major weight shaped [N, K].
        enable_pdl: Request Programmatic Dependent Launch when supported.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        FP32 projected activations with trailing dimension N.
    """
    if hidden_states.ndim == 0:
        raise ValueError("hidden_states must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N, K], got {tuple(weight.shape)}")
    if hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError(
            "DeepSeek V4 linear K mismatch: "
            f"hidden_states K={hidden_states.shape[-1]}, weight K={weight.shape[1]}"
        )
    if not hidden_states.is_floating_point() or not weight.is_floating_point():
        raise ValueError("hidden_states and weight must be floating-point tensors")

    traits = {
        "hidden_rank": hidden_states.ndim,
        "weight_rank": weight.ndim,
        "has_tokens": hidden_states.numel() > 0,
        "k_match": True,
    }
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        weight=dense_tensor_format(weight.dtype),
    )
    kernel = select_kernel(
        "gemm",
        "deepseek_v4_linear_fp32",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    k = int(weight.shape[1])
    shape_params = {
        "M": int(prod(hidden_states.shape[:-1])),
        "N": int(weight.shape[0]),
        "K": k,
        "enable_pdl": bool(enable_pdl),
    }
    ShapeCapture.get().record(
        "gemm",
        "deepseek_v4_linear_fp32",
        kernel.name,
        hidden_states.dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "deepseek_v4_linear_fp32",
        hidden_states.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(hidden_states, weight, enable_pdl=enable_pdl)


__all__ = ["deepseek_v4_linear_fp32"]
