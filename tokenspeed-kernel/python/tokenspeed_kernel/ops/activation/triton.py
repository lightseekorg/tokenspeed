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

"""Triton activation helper kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["sigmoid_mul"]


@triton.jit
def _sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    n_elements,
    hidden_dim: tl.constexpr,
    head_dim: tl.constexpr,
    gate_row_stride: tl.constexpr,
    gate_head_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row = offsets // hidden_dim
    col = offsets % hidden_dim
    head = col // head_dim
    d = col % head_dim
    gate_addrs = gate_ptr + row * gate_row_stride + head * gate_head_stride + d

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(gate_addrs, mask=mask).to(tl.float32)
    out = x * tl.sigmoid(g)
    tl.store(x_ptr + offsets, out, mask=mask)


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """In-place ``x *= sigmoid(gate)``.

    ``x`` must be contiguous 2D ``[num_tokens, hidden_dim]`` and is mutated.
    ``gate`` may be either

    - 2D contiguous ``[num_tokens, hidden_dim]``, or
    - 3D ``[num_tokens, num_heads, head_dim]`` with ``stride(-1) == 1`` —
      the strided view that ``torch.chunk(q_gate, 2, dim=-1)`` produces from
      a packed ``[num_tokens, num_heads, 2 * head_dim]`` tensor.

    The strided form lets callers skip the ``.reshape(-1)`` copy after the
    chunk; both layouts share the same kernel via the explicit gate strides.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.ndim}D")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if gate.stride(-1) != 1:
        raise ValueError(f"gate must have stride(-1) == 1, got {gate.stride()}")
    if x.dtype != gate.dtype:
        raise ValueError(f"dtype mismatch: x={x.dtype} gate={gate.dtype}")

    num_tokens, hidden_dim = x.shape

    if gate.ndim == 2:
        if gate.shape != x.shape:
            raise ValueError(f"shape mismatch: x={x.shape} gate={gate.shape}")
        head_dim = hidden_dim
        gate_row_stride = gate.stride(0)
        gate_head_stride = hidden_dim
    elif gate.ndim == 3:
        gate_tokens, num_heads, head_dim = gate.shape
        if gate_tokens != num_tokens:
            raise ValueError(f"num_tokens mismatch: x={num_tokens} gate={gate_tokens}")
        if num_heads * head_dim != hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: x={hidden_dim} gate={num_heads}*{head_dim}"
            )
        gate_row_stride = gate.stride(0)
        gate_head_stride = gate.stride(1)
    else:
        raise ValueError(f"gate must be 2D or 3D, got {gate.ndim}D")

    n = x.numel()
    if n == 0:
        return x

    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _sigmoid_mul_kernel[grid](
        x,
        gate,
        n,
        hidden_dim=hidden_dim,
        head_dim=head_dim,
        gate_row_stride=gate_row_stride,
        gate_head_stride=gate_head_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return x
