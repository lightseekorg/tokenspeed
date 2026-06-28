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

from __future__ import annotations

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.transform.faster_hadamard_transform  # noqa: F401
import tokenspeed_kernel.ops.transform.triton  # noqa: F401
import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["hadamard_transform"]


# ===-----------------------------------------------------------------------===#
# Hadamard Transform Kernels
# ===-----------------------------------------------------------------------===#


def hadamard_transform(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Apply a Hadamard transform along the last dimension.

    Args:
        x: Input tensor. Registered implementations currently accept dense CUDA
            tensors. The Triton implementation supports last dimension 128.
        scale: Multiplicative output scale applied after the transform.
        override: Optional exact kernel override name.
        solution: Optional registered solution to force through normal selection.

    Returns:
        Tensor with the same shape and dtype as x.

    """
    if x.dim() == 0:
        raise ValueError("hadamard_transform requires at least one dimension")

    traits = {
        "last_dim": x.shape[-1],
    }
    signature = format_signature(x=dense_tensor_format(x.dtype))
    kernel = select_kernel(
        "transform",
        "hadamard_transform",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "shape": tuple(x.shape),
        "last_dim": x.shape[-1],
    }
    ShapeCapture.get().record(
        "transform",
        "hadamard_transform",
        kernel.name,
        x.dtype,
        shape_params,
    )

    with kernel_scope(
        "transform",
        "hadamard_transform",
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(x, scale=scale)
