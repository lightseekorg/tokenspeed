# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""AITER-derived gfx950 preshuffled block-FP8 GEMM."""

import torch
import triton
import triton.language as tl


def preshuffle_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    """Convert an ``[N, K]`` FP8 weight to AITER's 16x16 MFMA layout."""
    n, k = weight.shape
    if n % 16 or k % 32:
        raise ValueError(f"FP8 preshuffle requires N%16=0 and K%32=0, got {(n, k)}")
    return (
        weight.view(n // 16, 16, k // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(n, k)
    )


@triton.jit
def _preshuffled_fp8_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    pid_unified = tl.program_id(0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if pid_k * SPLITK_BLOCK_SIZE < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_shuffle_arr = tl.arange(0, BLOCK_SIZE_K * 16)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_k_shuffle = pid_k * SPLITK_BLOCK_SIZE * 16 + offs_k_shuffle_arr
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % (
            N // 16
        )
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
        )
        offs_k_scale = pid_k * SPLITK_BLOCK_SIZE // BLOCK_SIZE_K
        a_scale_ptrs = (
            a_scale_ptr + offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + (pid_n * BLOCK_SIZE_N // 128) * stride_bscale_n
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
        for _ in range(0, num_k_iter):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=".cg")
            b = (
                b.reshape(
                    1,
                    BLOCK_SIZE_N // 16,
                    BLOCK_SIZE_K // 32,
                    2,
                    16,
                    16,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .trans(1, 0)
            )
            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)
            accumulator += tl.dot(a, b) * (a_scale * b_scale)[:, None]
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * 16 * stride_bk
            a_scale_ptrs += stride_ascale_k
            b_scale_ptrs += stride_bscale_k

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + pid_k * stride_ck
            + offs_cm[:, None] * stride_cm
            + offs_cn[None, :] * stride_cn
        )
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=mask)


@triton.jit
def _reduce_splitk(
    partial_ptr,
    out_ptr,
    M,
    N,
    stride_pk,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, NUM_KSPLIT)
    ptrs = (
        partial_ptr
        + offs_k[:, None, None] * stride_pk
        + offs_m[None, :, None] * stride_pm
        + offs_n[None, None, :] * stride_pn
    )
    partials = tl.load(
        ptrs,
        mask=(offs_m[None, :, None] < M) & (offs_n[None, None, :] < N),
        other=0.0,
    )
    out = tl.sum(partials, axis=0)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def aiter_preshuffled_fp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the DSV4 TP8 N2048/K7168 preshuffled GEMM."""
    m, k = x.shape
    n = weight.shape[0]
    if (n, k) != (2048, 7168):
        raise ValueError(f"unsupported preshuffled FP8 shape {(m, n, k)}")
    output = (
        out
        if out is not None
        else torch.empty((m, n), dtype=output_dtype, device=x.device)
    )
    if m <= 32:
        block_m, block_n, num_warps = 16, 16, 1
    else:
        block_m, block_n, num_warps = 32, 64, 4
    num_ksplit = 8
    splitk_block_size = k // num_ksplit
    partials = torch.empty((num_ksplit, m, n), dtype=torch.float32, device=x.device)
    shuffled = weight.view(n // 16, k * 16)
    scales = weight_scale.transpose(0, 1)
    grid_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    _preshuffled_fp8_gemm[(num_ksplit * grid_mn,)](
        x,
        shuffled,
        partials,
        x_scale,
        scales,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        shuffled.stride(0),
        shuffled.stride(1),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=128,
        NUM_KSPLIT=num_ksplit,
        SPLITK_BLOCK_SIZE=splitk_block_size,
        GRID_MN=grid_mn,
        num_warps=num_warps,
        num_stages=2,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
    )
    _reduce_splitk[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        partials,
        output,
        m,
        n,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        NUM_KSPLIT=num_ksplit,
    )
    return output


__all__ = ["aiter_preshuffled_fp8_gemm", "preshuffle_fp8_weight"]
