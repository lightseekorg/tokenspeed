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

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform


@triton.jit
def _mxfp4_mm_kernel(
    A,
    B,
    A_scales,
    B_scales,
    C,
    # Activation-side geometry follows the batch and stays runtime; the
    # weight-side geometry is fixed for the process lifetime.
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_asm,
    stride_asg,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        packed_k = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        scale_k = k_start // 32 + tl.arange(0, BLOCK_K // 32)
        if A_BF16:
            offs_k = k_start + tl.arange(0, BLOCK_K)
            a = tl.load(
                A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0,
            )
            a_scale = None
            a_format: tl.constexpr = "bf16"
        else:
            a = tl.load(
                A + offs_m[:, None] * stride_am + packed_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (packed_k[None, :] < K // 2),
                other=0,
            )
            a_scale = tl.load(
                A_scales + offs_m[:, None] * stride_asm + scale_k[None, :] * stride_asg,
                mask=(offs_m[:, None] < M) & (scale_k[None, :] < K // 32),
                other=127,
            )
            a_format: tl.constexpr = "e2m1"

        b = tl.load(
            B + offs_n[:, None] * stride_bn + packed_k[None, :] * stride_bk,
            mask=(offs_n[:, None] < N) & (packed_k[None, :] < K // 2),
            other=0,
        )
        b_scale = tl.load(
            B_scales + offs_n[:, None] * stride_bsn + scale_k[None, :] * stride_bsg,
            mask=(offs_n[:, None] < N) & (scale_k[None, :] < K // 32),
            other=127,
        )

        acc = tl.dot_scaled(
            a,
            a_scale,
            a_format,
            b.trans(),
            b_scale,
            "e2m1",
            acc=acc,
            fast_math=True,
        )

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _e2m1_value(nibble):
    exp = (nibble >> 1) & 3
    mant = (nibble & 1).to(tl.float32)
    magnitude = tl.where(
        exp == 0, mant * 0.5, (1.0 + mant * 0.5) * tl.exp2((exp - 1).to(tl.float32))
    )
    return tl.where((nibble & 8) != 0, -magnitude, magnitude)


@triton.jit
def _mxfp4_dequantize_kernel(
    A,
    A_scales,
    Out,
    K: tl.constexpr,
    stride_am,
    stride_asm,
    stride_om,
    BLOCK_K: tl.constexpr,
):
    # Exact: every MXFP4 value is representable in BF16.
    row = tl.program_id(0).to(tl.int64)
    groups = tl.program_id(1) * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    packed = tl.load(
        A + row * stride_am + groups[:, None] * 16 + tl.arange(0, 16)[None, :]
    )
    scale = tl.exp2(
        tl.load(A_scales + row * stride_asm + groups).to(tl.float32) - 127.0
    )
    values = tl.join(_e2m1_value(packed & 0xF), _e2m1_value(packed >> 4))
    values = tl.reshape(values, (BLOCK_K // 32, 32)) * scale[:, None]
    offs = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(Out + row * stride_om + offs, values.to(tl.bfloat16))


def triton_mm_mxfp4(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    del alpha, block_size
    if A.dtype != torch.uint8 or B.dtype != torch.uint8:
        raise TypeError("triton_mm_mxfp4 expects packed uint8 inputs")
    if A_scales is None or B_scales is None:
        raise ValueError("A_scales and B_scales are required for MXFP4 GEMM")
    if A.shape[-1] != B.shape[1]:
        raise ValueError(f"MXFP4 GEMM K mismatch: {A.shape=} {B.shape=}")
    M = A.shape[0]
    K = A.shape[1] * 2
    N = B.shape[0]
    if K % 32 != 0:
        raise ValueError("MXFP4 GEMM requires K divisible by 32")
    rdna4 = current_platform().is_rdna4
    block_m, block_n, block_k = (128, 128, 64) if rdna4 else (16, 32, 32)
    if rdna4:
        # Expanding the activations to BF16 once measured faster than
        # upcasting them in every tile.
        A_bf16 = torch.empty(M, K, device=A.device, dtype=torch.bfloat16)
        dequant_k = math.gcd(K, 1024)
        _mxfp4_dequantize_kernel[(M, K // dequant_k)](
            A,
            A_scales,
            A_bf16,
            K,
            A.stride(0),
            A_scales.stride(0),
            A_bf16.stride(0),
            BLOCK_K=dequant_k,
        )
        A = A_bf16
    C = out if out is not None else torch.empty(M, N, device=A.device, dtype=out_dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _mxfp4_mm_kernel[grid](
        A,
        B,
        A_scales,
        B_scales,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        A_scales.stride(0),
        A_scales.stride(1),
        B_scales.stride(0),
        B_scales.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        A_BF16=rdna4,
        num_warps=4,
    )
    return C
