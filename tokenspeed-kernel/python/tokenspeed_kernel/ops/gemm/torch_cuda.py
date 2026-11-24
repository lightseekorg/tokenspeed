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

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "gemm",
    "grouped_bf16_projection",
    name="torch_grouped_bf16_projection_cuda",
    solution="torch",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    traits={"is_cuda": frozenset({True})},
    priority=Priority.PORTABLE + 1,
)
def torch_grouped_bf16_projection_cuda(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Write the cuBLAS grouped BF16 projection directly into token-major rows.

    Args:
        x: BF16 [tokens, groups, input_dim] activations.
        weight: BF16 [groups, output_dim, input_dim] weights.
        out: Optional contiguous [tokens, groups, output_dim] destination.

    Returns:
        The contiguous BF16 result, reusing out when supplied.
    """
    if out is None:
        out = x.new_empty((x.shape[0], weight.shape[0], weight.shape[1]))
    # cuBLAS supports this interleaved output; keep its measured copy-free path
    # separate from the portable reference's contiguous batched output.
    with torch.no_grad():
        torch.bmm(x.transpose(0, 1), weight.transpose(1, 2), out=out.transpose(0, 1))
    return out
