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

"""Ascend (NPU) activation implementations.

``silu_and_mul`` is served as a torch-API composition (``gate * sigmoid(gate) *
up``) because the torch_npu fused ``npu_swiglu`` expects its gate/up pairs in
the *interleaved* ``[..., 2i, 2i+1]`` layout while the kernel package's
``silu_and_mul`` contract uses the *half-half* layout (gate in the first half
of the last dimension, up in the second half).
"""

from __future__ import annotations

import torch

__all__ = ["silu_and_mul"]


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Fused ``SiLU(x[..., :D]) * x[..., D:]`` on Ascend NPU.

    ``x`` is interpreted as ``[..., 2 * D]`` with gate values in the first half
    and up values in the second half. The output has shape ``[..., D]``.
    Positive ``limit`` clamps the gate to ``min(gate, limit)`` and the up
    branch to ``[-limit, limit]``, matching the portable Triton contract.
    """
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {x.shape[-1]}")
    if x.stride(-1) != 1:
        x = x.contiguous()

    hidden_dim = x.shape[-1] // 2
    output_shape = (*x.shape[:-1], hidden_dim)
    if out is None:
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    elif tuple(out.shape) != output_shape:
        raise ValueError(f"out shape must be {output_shape}, got {tuple(out.shape)}")
    if out.stride(-1) != 1:
        raise ValueError("out must have stride(-1) == 1")

    gate, up = x.chunk(2, dim=-1)
    gate_f = gate.float()
    up_f = up.float()
    if limit is not None:
        gate_f = torch.minimum(gate_f, torch.tensor(limit, device=x.device))
        up_f = torch.clamp(up_f, -limit, limit)
    result = (gate_f * torch.sigmoid(gate_f) * up_f).to(x.dtype)
    out.copy_(result)
    return out
