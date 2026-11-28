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

"""BF16 GEMM for a few rows on RDNA4: ``out = a @ w.T`` streams ``w`` once."""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

# Rows of ``w`` per CTA, warps, and K per step (eight BF16 values per lane).
_BLOCK_N = 16
_NUM_WARPS = 8
BLOCK_K = 256
MAX_ROWS = 4


def _launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops16": 2 * m * n * k,
        "bytes": 2 * (m * k + n * k + m * n),
    }


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_mm_bf16_decode_gfx120x(
    A,
    W,
    Out,
    M,
    N,
    K: gl.constexpr,
    MP: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """One CTA computes ``BLOCK_N`` columns of all ``M`` (at most ``MP``)
    rows, issuing the next K step's weight loads before this step's math,
    and sums in FP32."""
    L: gl.constexpr = gl.BlockedLayout(
        [MP, 1, BLOCK_K // 32], [1, 1, 32], [1, gl.num_warps(), 1], [2, 1, 0]
    )
    WL: gl.constexpr = gl.SliceLayout(0, L)
    AL: gl.constexpr = gl.SliceLayout(1, L)
    OL: gl.constexpr = gl.SliceLayout(2, L)
    n0 = gl.program_id(0) * BLOCK_N
    rn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, WL))
    rk = gl.arange(0, BLOCK_K, gl.SliceLayout(0, WL))
    rm = gl.arange(0, MP, gl.SliceLayout(1, AL))
    ka = gl.arange(0, BLOCK_K, gl.SliceLayout(0, AL))
    rows = gl.minimum(n0 + rn, N - 1).to(gl.int64)
    w_ptr = W + rows[:, None] * K + rk[None, :]
    a_ptr = A + gl.minimum(rm, M - 1)[:, None] * K + ka[None, :]
    acc = gl.zeros([MP, BLOCK_N, BLOCK_K], gl.float32, L)
    w = gl.load(w_ptr)
    for k in range(0, K, BLOCK_K):
        w_next = gl.load(w_ptr + gl.minimum(k + BLOCK_K, K - BLOCK_K))
        a = gl.load(a_ptr + k).to(gl.float32)
        acc += a[:, None, :] * w.to(gl.float32)[None, :, :]
        w = w_next
    y = gl.sum(acc, 2).to(gl.bfloat16)
    om = gl.arange(0, MP, gl.SliceLayout(1, OL))
    on = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(0, OL))
    mask = (om[:, None] < M) & (on[None, :] < N)
    gl.store(Out + om[:, None] * N + on[None, :], y, mask=mask)


def launch_gluon_mm_bf16_decode_gfx120x(
    a: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``a @ w.T`` for BF16 ``a`` ``[M, K]`` (``M <= MAX_ROWS``) and ``w``
    ``[N, K]``, both contiguous, ``K`` a multiple of ``BLOCK_K``.

    Returns:
        BF16 ``[M, N]`` (``out`` when given, contiguous).
    """
    m, k = a.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    gluon_mm_bf16_decode_gfx120x[(triton.cdiv(n, _BLOCK_N),)](
        a,
        w,
        out,
        m,
        n,
        k,
        1 if m == 1 else MAX_ROWS,
        _BLOCK_N,
        BLOCK_K,
        num_warps=_NUM_WARPS,
    )
    return out
