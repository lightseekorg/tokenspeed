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

"""Cold-weight FP32 router GEMV for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_BLOCK_K = 256
_NUM_WARPS = 4
_SUBGROUP_SIZE = 64


@gluon.jit
def _fp32_router_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: gl.constexpr,
    N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    MULTIPLIER: gl.constexpr,
):
    output = gl.program_id(0)
    row = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_K // (SUBGROUP_SIZE * NUM_WARPS)],
        [SUBGROUP_SIZE],
        [NUM_WARPS],
        [0],
    )
    offsets = gl.arange(0, BLOCK_K, layout=layout)
    accumulator = gl.zeros((BLOCK_K,), gl.float32, layout=layout)
    for start in range(0, K, BLOCK_K):
        columns = start + offsets
        valid = columns < K
        x = gl.amd.cdna4.buffer_load(
            x_ptr,
            row * K + columns.to(gl.int32),
            mask=valid,
            other=0.0,
        )
        weight = gl.amd.cdna4.buffer_load(
            weight_ptr,
            (output * K + columns).to(gl.int32),
            mask=valid,
            other=0.0,
            cache=".cg",
        )
        accumulator += x * weight
    result = gl.sum(accumulator, axis=0) * MULTIPLIER
    gl.store(out_ptr + row * N + output, result)


def transpose_fp32_router_weight_gfx950(weight: torch.Tensor) -> torch.Tensor:
    """Transpose a canonical ``[K, N]`` router weight for row-coalesced GEMV."""
    if weight.ndim != 2 or weight.dtype != torch.float32 or not weight.is_cuda:
        raise ValueError("router weight must be a two-dimensional FP32 GPU tensor")
    return weight.T.contiguous()


def gluon_fp32_router_gemv_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Project up to two FP32 decode rows through a transposed router weight."""
    del enable_pdl
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous FP32 GPU rows")
    if (
        weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or weight.dtype != torch.float32
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous transposed FP32 [N, K] beside x")
    if (
        counters.dtype != torch.int32
        or counters.device != x.device
        or counters.numel() != (weight.shape[0] + 31) // 32
    ):
        raise ValueError("counters must match the router output width")
    rows = x.shape[0]
    if rows == 0:
        return x.new_empty((0, weight.shape[0]))
    if rows > 2:
        return torch.matmul(x, weight.T) * multiplier
    out = torch.empty((rows, weight.shape[0]), dtype=x.dtype, device=x.device)
    _fp32_router_gemv_kernel[(weight.shape[0], rows)](
        x,
        weight,
        out,
        K=x.shape[1],
        N=weight.shape[0],
        BLOCK_K=_BLOCK_K,
        NUM_WARPS=_NUM_WARPS,
        SUBGROUP_SIZE=_SUBGROUP_SIZE,
        MULTIPLIER=float(multiplier),
        num_warps=_NUM_WARPS,
    )
    return out


__all__ = [
    "gluon_fp32_router_gemv_gfx950",
    "transpose_fp32_router_weight_gfx950",
]
