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

"""Split-K MoE router GEMV for AMD GFX1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = ["launch_gluon_router_sqrt_softplus_topk_gemv_gfx1250"]

cdna5 = gl.amd.cdna5


def _gemv_launch_metadata(grid, kernel, args):
    """Report useful FLOPs and modeled traffic to Proton.

    The weight is read once, the activation once per expert tile, and each
    K split writes FP32 partials.
    """
    m = args["M"]
    n = grid[0] * args["BLOCK_N"]
    k = grid[1] * args["BLOCK_K"]
    return {
        "name": kernel.name,
        "flops16": 2 * m * n * k,
        "bytes": n * k * args["w"].element_size()
        + grid[0] * m * k * args["x"].element_size()
        + grid[1] * m * n * args["partials"].element_size(),
    }


@gluon.jit(launch_metadata=_gemv_launch_metadata, do_not_specialize=["M"])
def gluon_router_sqrt_softplus_topk_gemv_gfx1250(
    x,
    w,
    partials,
    M,
    stride_xm,
    stride_wn,
    stride_ps,
    stride_pm,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Store one K split of ``x @ w.T`` for a BLOCK_N-expert tile, in FP32.

    Program ``(n, k)`` loads its ``[BLOCK_K, BLOCK_N]`` weight tile once and
    walks every 16-row tile of the M rows of ``x`` against it, writing
    ``partials[k, :M, n * BLOCK_N:(n + 1) * BLOCK_N]``. The row-tile loop is
    bounded at runtime, so one binary serves every batch size.
    """
    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    k0 = pid_k * BLOCK_K
    bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, dot_b))[:, None]
    bn = (
        pid_n * BLOCK_N
        + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b))[None, :]
    )
    # The weight tile is read once; every 16-row tile reuses it.
    b = cdna5.buffer_load(w, (bn * stride_wn + k0 + bk).to(gl.int32))
    am = gl.arange(0, 16, layout=gl.SliceLayout(1, dot_a))[:, None]
    ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, dot_a))[None, :]
    cm = gl.arange(0, 16, layout=gl.SliceLayout(1, wmma))[:, None]
    cn = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, wmma))[None, :]
    )
    for t in range(gl.cdiv(M, 16)):
        rows = t * 16 + am
        a = cdna5.buffer_load(
            x,
            (rows * stride_xm + k0 + ak).to(gl.int32),
            mask=(rows < M) & (ak < BLOCK_K),
            other=0.0,
        )
        acc = cdna5.wmma(a, b, gl.zeros([16, BLOCK_N], gl.float32, wmma))
        out_rows = t * 16 + cm
        cdna5.buffer_store(
            acc,
            partials,
            (pid_k * stride_ps + out_rows * stride_pm + cn).to(gl.int32),
            mask=(out_rows < M) & (cn < (pid_n + 1) * BLOCK_N),
        )


def launch_gluon_router_sqrt_softplus_topk_gemv_gfx1250(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_k: int = 512,
) -> torch.Tensor:
    """Return FP32 split-K partials ``[K // block_k, M, N]`` of ``x @ weight.T``.

    Summing the partials over dim 0 in order yields the router logits.

    Args:
        x: BF16 activations ``[M, K]`` with unit stride along K.
        weight: Contiguous BF16 router weight ``[N, K]``, N a multiple of 16.
        block_k: Width of each K split; K must be a multiple of it.
    """
    tokens, hidden = x.shape
    experts = weight.shape[0]
    block_n = 16
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("router GEMV requires BF16 inputs")
    if weight.shape[1] != hidden or x.stride(1) != 1 or not weight.is_contiguous():
        raise ValueError("router GEMV requires row-major x and a contiguous weight")
    if hidden % block_k or experts % block_n:
        raise ValueError("router GEMV requires K % block_k == 0 and N % 16 == 0")
    split = hidden // block_k
    partials = torch.empty(
        (split, tokens, experts), dtype=torch.float32, device=x.device
    )
    if tokens == 0:
        return partials
    gluon_router_sqrt_softplus_topk_gemv_gfx1250[(experts // block_n, split)](
        x,
        weight,
        partials,
        tokens,
        x.stride(0),
        weight.stride(0),
        partials.stride(0),
        partials.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=1,
    )
    return partials
