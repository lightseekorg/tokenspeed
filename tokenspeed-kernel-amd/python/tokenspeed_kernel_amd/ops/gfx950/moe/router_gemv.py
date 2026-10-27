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

"""Split-K MoE router GEMV for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = ["launch_gluon_router_sqrt_softplus_topk_gemv_gfx950"]

cdna4 = gl.amd.cdna4


@gluon.jit
def gluon_router_sqrt_softplus_topk_gemv_gfx950(
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
    M_TILES: gl.constexpr,
):
    pid_n = gl.program_id(0)
    pid_k = gl.program_id(1)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, gl.num_warps()],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    k0 = pid_k * BLOCK_K
    bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, dot_b))[:, None]
    bn = (
        pid_n * BLOCK_N
        + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b))[None, :]
    )
    # The weight tile is read once; every 16-row tile reuses it.
    b = cdna4.buffer_load(w, (bn * stride_wn + k0 + bk).to(gl.int32))
    am = gl.arange(0, 16, layout=gl.SliceLayout(1, dot_a))[:, None]
    ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, dot_a))[None, :]
    cm = gl.arange(0, 16, layout=gl.SliceLayout(1, mfma))[:, None]
    cn = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma))[None, :]
    )
    for t in gl.static_range(M_TILES):
        rows = t * 16 + am
        a = cdna4.buffer_load(
            x,
            (rows * stride_xm + k0 + ak).to(gl.int32),
            mask=(rows < M) & (ak < BLOCK_K),
            other=0.0,
        )
        acc = cdna4.mfma(a, b, gl.zeros([16, BLOCK_N], gl.float32, mfma))
        out_rows = t * 16 + cm
        cdna4.buffer_store(
            acc,
            partials,
            (pid_k * stride_ps + out_rows * stride_pm + cn).to(gl.int32),
            mask=(out_rows < M) & (cn < (pid_n + 1) * BLOCK_N),
        )


def launch_gluon_router_sqrt_softplus_topk_gemv_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_k: int = 512,
    block_n: int = 16,
) -> torch.Tensor:
    """Return FP32 split-K partials ``[K // block_k, M, N]`` of ``x @ weight.T``.

    Summing the partials over dim 0 in order yields the router logits.
    """
    tokens, hidden = x.shape
    experts = weight.shape[0]
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("router GEMV requires BF16 inputs")
    if weight.shape[1] != hidden or x.stride(1) != 1 or not weight.is_contiguous():
        raise ValueError("router GEMV requires row-major x and a contiguous weight")
    if hidden % block_k or experts % block_n:
        raise ValueError("router GEMV requires K % block_k == 0 and N % block_n == 0")
    split = hidden // block_k
    partials = torch.empty(
        (split, tokens, experts), dtype=torch.float32, device=x.device
    )
    if tokens == 0:
        return partials
    gluon_router_sqrt_softplus_topk_gemv_gfx950[(experts // block_n, split)](
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
        M_TILES=triton.cdiv(tokens, 16),
        num_warps=block_n // 16,
    )
    return partials
