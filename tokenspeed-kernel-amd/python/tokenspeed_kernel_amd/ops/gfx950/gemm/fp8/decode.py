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

"""Packed block-FP8 projections for GLM-5.3 decode and short prefill."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8.largem import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    GLM53_BLOCK_FP8_PROJECTION_SHAPES,
    GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    GROUP_SIZE_M,
    NUM_WARPS,
    WARPS_M,
    WARPS_N,
    gluon_mm_fp8_blockscale_largem_gfx950,
)


def _select_split_k(m: int, k: int) -> int:
    if k == 512:
        return 1
    if k == 1536:
        return 3 if m <= 64 else 1
    if k == 3072:
        return 4
    if k == 4096:
        return 8 if m <= 16 else 4
    raise ValueError(f"unsupported block-FP8 K={k}")


def _reduce_launch_metadata(grid, kernel, args):
    m, n, split_k = args["M"], args["N"], args["SPLIT_K"]
    return {
        "name": kernel.name,
        "flops32": m * n * (split_k - 1),
        "bytes": m * n * (4 * split_k + 2),
    }


@gluon.jit(launch_metadata=_reduce_launch_metadata, do_not_specialize=["M"])
def gluon_mm_fp8_blockscale_decode_reduce_gfx950(
    partial_ptr,
    c_ptr,
    M,
    N: gl.constexpr,
    stride_cm,
    stride_cn,
    SPLIT_K: gl.constexpr,
    BLOCK: gl.constexpr,
):
    """Combine FP32 K partitions before a single BF16 rounding."""
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    output_index = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout)
    total = M * N
    result = gl.zeros((BLOCK,), gl.float32, layout)
    for split in range(SPLIT_K):
        result += gl.load(
            partial_ptr + split * total + output_index,
            mask=output_index < total,
            other=0.0,
        )
    row = output_index // N
    column = output_index % N
    gl.store(
        c_ptr + row * stride_cm + column * stride_cn,
        result.to(c_ptr.dtype.element_ty),
        mask=output_index < total,
    )


def launch_gluon_mm_fp8_blockscale_decode_gfx950(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    block_size: list[int],
    weight_layout: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project E4M3 [M,K] with packed [N,K] and FP32 block scales to BF16."""
    if weight_layout != GLUON_BLOCK_FP8_WEIGHT_LAYOUT:
        raise ValueError("block-FP8 weight requires the N64 packed layout")
    m, k = activation.shape
    n, weight_k = packed_weight.shape
    if (
        block_size != [128, 128]
        or weight_k != k
        or not 1 <= m <= 128
        or (n, k) not in GLM53_BLOCK_FP8_PROJECTION_SHAPES
    ):
        raise ValueError(f"unsupported block-FP8 shape: M={m}, N={n}, K={k}")
    if (
        activation.dtype != torch.float8_e4m3fn
        or packed_weight.dtype != torch.float8_e4m3fn
        or activation_scales.dtype != torch.float32
        or weight_scales.dtype != torch.float32
        or out_dtype != torch.bfloat16
    ):
        raise TypeError("block-FP8 projection requires E4M3, FP32 scales, and BF16")
    if not packed_weight.is_contiguous():
        raise ValueError("block-FP8 packed weight must be contiguous")

    output = out
    if output is None:
        output = torch.empty((m, n), device=activation.device, dtype=out_dtype)
    split_k = _select_split_k(m, k)
    partials = (
        torch.empty((split_k, m, n), device=activation.device, dtype=torch.float32)
        if split_k > 1
        else output
    )
    gluon_mm_fp8_blockscale_largem_gfx950[
        (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N), split_k)
    ](
        activation.view(torch.uint8),
        packed_weight.view(torch.uint8),
        activation_scales,
        weight_scales,
        partials,
        m,
        n,
        k,
        activation.stride(0),
        activation.stride(1),
        activation_scales.stride(0),
        activation_scales.stride(1),
        weight_scales.stride(0),
        weight_scales.stride(1),
        partials.stride(-2),
        partials.stride(-1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=split_k,
        ONE_M_TILE=m <= 64,
        num_warps=NUM_WARPS,
        num_stages=1,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    if split_k > 1:
        gluon_mm_fp8_blockscale_decode_reduce_gfx950[(triton.cdiv(m * n, 256),)](
            partials,
            output,
            m,
            n,
            output.stride(0),
            output.stride(1),
            SPLIT_K=split_k,
            BLOCK=256,
            num_warps=1,
        )
    return output


__all__ = ["launch_gluon_mm_fp8_blockscale_decode_gfx950"]
