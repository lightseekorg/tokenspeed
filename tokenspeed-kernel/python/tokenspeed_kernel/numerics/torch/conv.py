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

"""PyTorch references for ``tokenspeed_kernel.ops.conv``."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FLOAT_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


@register_kernel(
    "conv",
    "short_conv",
    name="torch_short_conv",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    use_residual: bool,
) -> torch.Tensor:
    """Causal depthwise FIR of one sequence over ``[prefix ++ x]``.

    Args:
        x: ``[T, channels]``.
        weight: ``[channels, width]`` taps, oldest first.
        prefix: ``[width - 1, channels]`` history before ``x``.
        use_residual: Add ``x`` to the output.

    Returns:
        ``[T, channels]`` in ``x``'s dtype (math in FP32).
    """
    xp = torch.cat([prefix, x]).float()
    y = sum(xp[w : w + len(x)] * weight[:, w].float() for w in range(weight.shape[1]))
    return (y + x.float() if use_residual else y).to(x.dtype)


@register_kernel(
    "conv",
    "block_grouped_conv",
    name="torch_block_grouped_conv",
    solution="torch",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.REFERENCE,
)
def torch_block_grouped_conv(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Block-local causal conv with per-row, per-group coefficients.

    Row ``r`` sums ``(base[tap] + delta[r, tap]) * x[r - tap]`` over taps that
    stay inside its ``block_size``-row block; coefficients apply per group of
    ``group_size`` channels.

    Args:
        x: ``[rows, groups * group_size]``.
        delta: ``[rows, taps, groups]``.
        base: ``[taps, groups * group_size]``.

    Returns:
        ``[rows, groups * group_size]`` in ``x``'s dtype (math in FP32).
    """
    taps, num_groups = base.shape[0], delta.shape[2]
    blocks = x.float().unflatten(-1, (num_groups, group_size))
    coefficients = base.float().view(
        1, taps, num_groups, group_size
    ) + delta.float().unsqueeze(-1)
    y = torch.zeros_like(blocks)
    for row in range(x.shape[0]):
        for tap in range(min(row % block_size + 1, taps)):
            y[row] += coefficients[row, tap] * blocks[row - tap]
    return y.flatten(-2).to(x.dtype)
