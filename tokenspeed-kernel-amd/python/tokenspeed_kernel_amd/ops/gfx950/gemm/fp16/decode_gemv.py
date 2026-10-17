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

"""Cold-weight, single-row BF16 GEMV for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_SUBGROUP_SIZE = 64


def _rowcta_config(n: int, k: int) -> tuple[int, int]:
    """Choose the cold-weight reduction geometry for a one-row projection."""
    if k == 8192:
        return (256, 2) if n <= 1024 else (1024, 2)
    if k == 2048:
        return 1024, 2
    return 512, 4


@gluon.jit
def _rowcta_bf16_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    output = gl.program_id(0)
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
            columns.to(gl.int32),
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        weight = gl.amd.cdna4.buffer_load(
            weight_ptr,
            (output * K + columns).to(gl.int32),
            mask=valid,
            other=0.0,
            cache=".cg",
        ).to(gl.float32)
        accumulator += x * weight
    gl.store(out_ptr + output, gl.sum(accumulator, axis=0))


def gluon_rowcta_bf16_gemv_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` for one contiguous BF16 row.

    Args:
        x: Contiguous BF16 input shaped ``[1, K]``.
        weight: Contiguous BF16 weight shaped ``[N, K]``.
        out: Contiguous BF16 destination shaped ``[1, N]``.

    Returns:
        ``out`` after the projection completes.
    """
    if (
        x.ndim != 2
        or x.shape[0] != 1
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be one contiguous BF16 GPU row")
    if (
        weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous BF16 [N, K] beside x")
    if (
        out.shape != (1, weight.shape[0])
        or out.dtype != torch.bfloat16
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 [1, N] beside x")
    k = x.shape[1]
    if k <= 0:
        raise ValueError("K must be positive")
    block_k, num_warps = _rowcta_config(weight.shape[0], k)
    _rowcta_bf16_gemv_kernel[(weight.shape[0],)](
        x,
        weight,
        out,
        K=k,
        BLOCK_K=block_k,
        NUM_WARPS=num_warps,
        SUBGROUP_SIZE=_SUBGROUP_SIZE,
        num_warps=num_warps,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_rowcta_bf16_gemv_gfx950"]
