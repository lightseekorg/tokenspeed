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

"""BF16 grouped projection directly from live canonical strided operands."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=["M", "N"])
def _grouped_projection(
    X,
    W,
    Y,
    M,
    N,
    K: tl.constexpr,
    G: tl.constexpr,
    XS0,
    XS1,
    XS2,
    WS0,
    WS1,
    WS2,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    g = tl.program_id(2)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, K, BK):
        q = start + k
        x = tl.load(
            X + r[:, None] * XS0 + g * XS1 + q[None, :] * XS2,
            (r[:, None] < M) & (q[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + g * WS0 + c[None, :] * WS1 + q[:, None] * WS2,
            (c[None, :] < N) & (q[:, None] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        Y + r[:, None] * (G * N) + g * N + c[None, :],
        acc,
        (r[:, None] < M) & (c[None, :] < N),
    )


def strided_bf16_linear_fp32(x, weight, out):
    """Read a live [N,K] weight view and write FP32 partial projections."""
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, bk = (16, 64, 1024) if m <= 16 else (32, 128, 512)
    if m > 128:
        bm = 64
    if m and n:
        grouped_x, grouped_weight = x.unsqueeze(1), weight.unsqueeze(0)
        _grouped_projection[(triton.cdiv(m, bm), triton.cdiv(n, bn), 1)](
            grouped_x,
            grouped_weight,
            out,
            m,
            n,
            k,
            1,
            *grouped_x.stride(),
            *grouped_weight.stride(),
            bm,
            bn,
            bk,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=True,
        )
    return out


def grouped_bf16_projection(x, weight, out):
    """Compute [M,G,K] x [G,N,K] -> [M,G,N], without transpose buffers.

    BF16 operands are unchanged; Cube accumulates in FP32 before the final
    BF16 store. Every invocation reads the live weight, including graph replay.
    The common dispatcher restricts this implementation to measured geometry.
    """
    m, groups, k = x.shape
    n = weight.shape[1]
    if out is None:
        out = torch.empty((m, groups, n), device=x.device, dtype=torch.bfloat16)
    if m:
        _grouped_projection[(triton.cdiv(m, 16), triton.cdiv(n, 256), groups)](
            x,
            weight,
            out,
            m,
            n,
            k,
            groups,
            *x.stride(),
            *weight.stride(),
            16,
            256,
            256,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=True,
        )
    return out
