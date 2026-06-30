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
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def _hadamard_128_kernel(
    x,
    out,
    n_rows: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    out_block = tl.program_id(1)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    in_offsets = tl.arange(0, 128)

    vals = tl.load(x + row * 128 + in_offsets).to(tl.float32)
    bits = out_offsets[:, None] & in_offsets[None, :]
    parity = bits ^ (bits >> 1)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 4)
    parity = parity & 1
    signs = tl.where(parity == 0, 1.0, -1.0)
    acc = tl.sum(signs * vals[None, :], axis=1) * scale

    tl.store(
        out + row * 128 + out_offsets,
        acc,
        mask=(row < n_rows) & (out_offsets < 128),
    )


@register_kernel(
    "transform",
    "hadamard_transform",
    name="triton_hadamard_transform_128",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.bfloat16, torch.float16, torch.float32},
    ),
    traits={
        "last_dim": frozenset({128}),
    },
    priority=Priority.PORTABLE,
)
def triton_hadamard_transform_128(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a length-128 Sylvester Hadamard transform along the last dim."""
    if x.shape[-1] != 128:
        raise ValueError(
            f"triton_hadamard_transform_128 requires last dim 128, got {x.shape[-1]}"
        )
    if not x.is_cuda:
        raise RuntimeError("triton_hadamard_transform_128 requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(
            f"triton_hadamard_transform_128 does not support dtype {x.dtype}"
        )

    shape = x.shape
    x_2d = x.reshape(-1, 128).contiguous()
    out = torch.empty_like(x_2d)
    if x_2d.shape[0] == 0:
        return out.reshape(shape)
    _hadamard_128_kernel[(x_2d.shape[0], 8)](
        x_2d,
        out,
        n_rows=x_2d.shape[0],
        scale=float(scale),
        BLOCK_OUT=16,
        num_warps=8,
        num_stages=1,
    )
    return out.reshape(shape)


__all__ = ["triton_hadamard_transform_128"]
