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

"""Portable torch leaves for the sampling family."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "sampling",
    "block_sumexp",
    name="torch_block_sumexp",
    solution="torch",
    features={"batch_invariant"},
    signatures=format_signatures("shifted", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def torch_block_sumexp(shifted: torch.Tensor, *, block_size: int) -> torch.Tensor:
    """Per-block ``sum(exp(x))`` with one fixed binary tree per block.

    Every block is reduced by halving: element ``i`` is added to element
    ``i + half`` at each level, so the association order depends only on
    ``block_size`` and every row of every batch shape follows the same tree
    (the ``batch_invariant`` feature). Vendor leaves (e.g. the fixed-order
    operator kit's ``sumexp``) register the same op with their own in-block
    order; this leaf is the portable one.

    Args:
        shifted: ``[rows, vocab]`` fp32 logits already shifted by the row max;
            ``vocab`` is a multiple of ``block_size``.
        block_size: Width of each block; a power of two.

    Returns:
        ``[rows, vocab // block_size]`` fp32 block sums.
    """
    rows, vocab = shifted.shape
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two, got {block_size}")
    if vocab % block_size:
        raise ValueError(f"vocab {vocab} is not a multiple of block_size {block_size}")
    # ``exp`` allocates the one buffer the fold runs in: ``reshape`` takes a
    # non-contiguous ``shifted`` (``view`` would not), and each level adds the
    # upper half onto the lower half in place, so no level allocates.
    values = torch.exp(shifted).reshape(rows, vocab // block_size, block_size)
    width = block_size
    while width > 1:
        half = width // 2
        values[..., :half] += values[..., half:width]
        width = half
    return values[..., 0]
