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

"""PyTorch references for ``tokenspeed_kernel.ops.transform``."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def sylvester_hadamard(n: int, device: torch.device | str) -> torch.Tensor:
    """The ``n x n`` Sylvester Hadamard matrix in FP32, ``n`` a power of two:
    entry ``(i, j)`` is ``(-1) ** popcount(i & j)``."""
    idx = torch.arange(n, device=device)
    bits = idx[:, None] & idx[None, :]
    parity = torch.zeros_like(bits)
    for shift in range(max(n.bit_length() - 1, 0)):
        parity ^= (bits >> shift) & 1
    return torch.where(parity == 0, 1.0, -1.0)


@register_kernel(
    "transform",
    "hadamard_transform",
    name="torch_hadamard_transform",
    solution="torch",
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_hadamard_transform(x: torch.Tensor, *, scale: float) -> torch.Tensor:
    """``(x @ H) * scale`` along the last dimension, in ``x``'s dtype (math in
    FP32, rounded once)."""
    hadamard = sylvester_hadamard(x.shape[-1], x.device)
    return ((x.float() @ hadamard) * scale).to(x.dtype)
