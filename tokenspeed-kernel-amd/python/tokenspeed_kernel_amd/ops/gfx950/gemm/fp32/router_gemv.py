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
from tokenspeed_kernel_amd._triton import gl, gluon, triton

# Rows above this go to torch.matmul.
MAX_ROWS = 16
# A program stages its whole weight row in LDS; longer rows go to torch.matmul.
MAX_K = 16384
_NUM_WARPS = 4
# K values a lane reads per load (16 bytes).
_VECTOR = 4


def _launch_metadata(grid, kernel, args):
    """Report weight, activation and output traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops32": 2 * m * n * k,
        "bytes": (n * k + m * k + m * n) * 4,
    }


@gluon.jit
def _load_x_step(x_ptr, x_offs, row_mask, k, step, K: gl.constexpr, STEP: gl.constexpr):
    """Load K step ``step`` of every row's activations, zero past ``K``."""
    return gl.amd.cdna4.buffer_load(
        x_ptr,
        x_offs + step * STEP,
        mask=row_mask & (k + step * STEP < K)[None, :],
        other=0.0,
    )


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_fp32_router_gemv_gfx950(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    K: gl.constexpr,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    VECTOR: gl.constexpr,
    MULTIPLIER: gl.constexpr,
):
    output = gl.program_id(0)
    # [rows, K step]: a lane holds VECTOR contiguous K values of every row.
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_M, VECTOR], [1, 64], [1, gl.num_warps()], [1, 0]
    )
    STEP: gl.constexpr = VECTOR * 64 * gl.num_warps()
    STEPS: gl.constexpr = (K + STEP - 1) // STEP
    k = gl.arange(0, STEP, gl.SliceLayout(0, layout))
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    row_mask = (rows < M)[:, None]
    x_offs = rows[:, None] * K + k[None, :]
    w_offs = output * K + k

    # The whole weight row goes to LDS at once, one commit group per step, so
    # all of it is in flight; each lane later reads back only what it copied.
    smem = gl.allocate_shared_memory(
        weight_ptr.dtype.element_ty,
        [STEPS, STEP],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    for step in gl.static_range(STEPS):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem.index(step),
            weight_ptr,
            w_offs + step * STEP,
            mask=k + step * STEP < K,
            cache_modifier=".cg",
        )
        gl.amd.cdna4.async_copy.commit_group()
    x = _load_x_step(x_ptr, x_offs, row_mask, k, 0, K, STEP)
    acc = gl.zeros((BLOCK_M, STEP), gl.float32, layout)
    # Unrolled, with the next step's activations in flight during this step's
    # FMAs; a loop-carried copy of a pending load would wait for it.
    for step in gl.static_range(STEPS):
        if step + 1 < STEPS:
            x_next = _load_x_step(x_ptr, x_offs, row_mask, k, step + 1, K, STEP)
        gl.amd.cdna4.async_copy.wait_group(STEPS - 1 - step)
        w = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem.index(step), gl.SliceLayout(0, layout)
        )
        acc = gl.fma(x, w[None, :], acc)
        # Pin this step's FMAs here: left free, the compiler sinks them below
        # the last load and keeps every step's activations live at once.
        acc = gl.inline_asm_elementwise(
            asm="",
            constraints="=v,0",
            args=[acc],
            dtype=gl.float32,
            is_pure=False,
            pack=1,
        )
        if step + 1 < STEPS:
            x = x_next
    result = gl.sum(acc, axis=1) * MULTIPLIER
    gl.store(out_ptr + rows * N + output, result, mask=rows < M)


def transpose_fp32_router_weight_gfx950(weight: torch.Tensor) -> torch.Tensor:
    """Transpose a canonical ``[K, N]`` router weight for row-coalesced GEMV."""
    if weight.ndim != 2 or weight.dtype != torch.float32 or not weight.is_cuda:
        raise ValueError("router weight must be a two-dimensional FP32 GPU tensor")
    return weight.T.contiguous()


def launch_gluon_fp32_router_gemv_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    enable_pdl: bool,
) -> torch.Tensor:
    """Project up to ``MAX_ROWS`` FP32 decode rows through a transposed router weight.

    More rows, or rows longer than ``MAX_K``, take BLAS with ``multiplier`` as alpha.

    Args:
        x: Contiguous FP32 ``[M, K]`` rows.
        weight: Contiguous FP32 ``[N, K]`` weight from
            :func:`transpose_fp32_router_weight_gfx950`.
        multiplier: Scale applied to every logit.
        enable_pdl: Accepted for the registry signature; unused.

    Returns:
        FP32 ``[M, N]`` logits.
    """
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
    rows, k = x.shape
    if rows == 0:
        return x.new_empty((0, weight.shape[0]))
    if rows > MAX_ROWS or k > MAX_K:
        # hipBLASLt applies alpha once to the finished FP32 dot product, which
        # matches a separate multiply bit for bit without its extra launch.
        return torch.addmm(
            x.new_empty((rows, weight.shape[0])), x, weight.T, beta=0, alpha=multiplier
        )
    out = torch.empty((rows, weight.shape[0]), dtype=x.dtype, device=x.device)
    gluon_fp32_router_gemv_gfx950[(weight.shape[0],)](
        x,
        weight,
        out,
        rows,
        K=k,
        N=weight.shape[0],
        # Bucketed so at most five variants compile per (K, N).
        BLOCK_M=triton.next_power_of_2(rows),
        # Scalar loads keep K values that are not 16-byte aligned addressable.
        VECTOR=_VECTOR if k % _VECTOR == 0 else 1,
        MULTIPLIER=float(multiplier),
        num_warps=_NUM_WARPS,
    )
    return out


__all__ = [
    "launch_gluon_fp32_router_gemv_gfx950",
    "transpose_fp32_router_weight_gfx950",
]
