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

"""PyTorch GEMM implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@register_kernel(
    "gemm",
    "deepseek_v4_linear_fp32",
    name="torch_deepseek_v4_linear_fp32",
    solution="torch",
    signatures=frozenset(
        format_signature(
            hidden_states=dense_tensor_format(hidden_dtype),
            weight=dense_tensor_format(weight_dtype),
        )
        for hidden_dtype in (torch.float16, torch.bfloat16, torch.float32)
        for weight_dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    priority=Priority.PORTABLE,
    tags={"portability", "reference"},
)
def torch_deepseek_v4_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run the projection with FP32 operands and return FP32."""
    del enable_pdl
    return F.linear(hidden_states.float(), weight.float())


__all__ = ["torch_deepseek_v4_linear_fp32"]
