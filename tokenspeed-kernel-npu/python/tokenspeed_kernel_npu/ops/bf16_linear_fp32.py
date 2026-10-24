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

"""BF16 operands with FP32 accumulation for V4 projections."""

from math import prod

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["M", "N"])
def _bf16_linear_fp32_kernel(
    X,
    W,
    O,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, K, BK):
        k = start + inner
        x = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        O + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def bf16_linear_fp32(hidden_states, weight):
    """Return FP32 ``hidden_states @ weight.T`` without expanding BF16 operands.

    Both operands retain their original BF16 values. Products and accumulation
    are FP32; only summation order can differ from the FP32 CANN reference.
    The live weight is read on every call/replay, including in-place updates.
    """
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BF16 operands are required; this kernel never quantizes")
    if weight.ndim != 2 or hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError("Expected hidden [...,K] and weight [N,K]")
    if hidden_states.device != weight.device or hidden_states.device.type != "npu":
        raise ValueError("Operands must be on the same NPU")
    k = weight.shape[1]
    flat = hidden_states.reshape(prod(hidden_states.shape[:-1]), k).contiguous()
    m, n = flat.shape[0], weight.shape[0]
    output = torch.empty((m, n), device=flat.device, dtype=torch.float32)
    if not weight.is_contiguous():
        from tokenspeed_kernel_npu.ops.grouped_bf16_projection import (
            strided_bf16_linear_fp32,
        )

        strided_bf16_linear_fp32(flat, weight, output)
        return output.reshape(*hidden_states.shape[:-1], n)
    if m and n:
        # M/N remain runtime arguments; the tiling uses bounded row buckets.
        bm = 16 if m <= 16 or (n <= 2048 and m <= 64) else 32
        # Narrow projections need smaller column tiles to occupy the Cube cores.
        # Grow the tile with the row grid so wider batches retain weight reuse.
        row_blocks = triton.cdiv(m, bm)
        bn = min(128, max(16, triton.next_power_of_2(triton.cdiv(n * row_blocks, 24))))
        bk = 1024 if bm == 16 and bn <= 32 else 512
        _bf16_linear_fp32_kernel[(row_blocks, triton.cdiv(n, bn))](
            flat,
            weight,
            output,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=True,
        )
    return output.reshape(*hidden_states.shape[:-1], n)


def bf16_short_projection(x, weight, out):
    """Compute a short-K projection with unchanged BF16 inputs and output.

    The shared Cube kernel accumulates in FP32, then rounds once at the store.
    Row tiling grows with the batch to reuse weights; the common dispatcher
    limits selection to measured geometry. Strided output retains its owner.
    """
    bm = min(128, max(16, triton.next_power_of_2(x.shape[0])))
    return _bf16_projection(x, weight, out, bm, 256, 256)


def bf16_wide_projection(x, weight, out):
    """Project a column shard of Engram with BF16 operands and one BF16 store."""
    m = x.shape[0]
    bm = min(64, max(16, triton.next_power_of_2(m)))
    bn, bk = (32, 1024) if m <= 16 else (128, 512)
    return _bf16_projection(x, weight, out, bm, bn, bk)


def _bf16_projection(x, weight, out, bm, bn, bk):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Expected input [M,K] and weight [N,K]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BF16 operands are required; this kernel never quantizes")
    if x.device.type != "npu" or x.device != weight.device:
        raise ValueError("Operands must be on the same NPU")
    m, k = x.shape
    n = weight.shape[0]
    if out is not None and (
        out.shape != (m, n) or out.dtype != x.dtype or out.device != x.device
    ):
        raise ValueError("Output must match the projection shape, dtype and device")
    result = (
        out
        if out is not None and out.is_contiguous()
        else torch.empty((m, n), device=x.device, dtype=x.dtype)
    )
    if m and n:
        _bf16_linear_fp32_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            x.contiguous(),
            weight.contiguous(),
            result,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=True,
        )
    if out is not None and result is not out:
        out.copy_(result)
        return out
    return result
