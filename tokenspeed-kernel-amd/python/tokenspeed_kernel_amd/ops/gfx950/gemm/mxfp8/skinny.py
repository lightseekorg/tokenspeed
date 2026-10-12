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

import torch
from tokenspeed_kernel_amd._triton import tl, triton


@triton.jit
def _e8m0_to_f32(s):
    return (s.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["M"])
def triton_mm_mxfp8_skinny_gfx950(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program owns a BLOCK_N strip of B and all rows of A.
    #
    # gfx950 FP8 MFMAs (v_mfma_f32_16x16x32_fp8_fp8 and the f8f6f4 forms behind
    # tl.dot_scaled) sum products at reduced internal precision, which misses
    # the FP8 GEMM contract (error at the output-cast floor). Upcast exactly
    # instead: e4m3 times a power-of-two E8M0 scale is exact in bf16, and bf16
    # MFMAs accumulate in fp32. Each 32-value scale group is one batch slice of
    # the dot, so the BLOCK_K // 32 slices spread over the warps.
    S: tl.constexpr = BLOCK_K // 32
    offs_s = tl.arange(0, S)
    offs_j = tl.arange(0, 32)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((S, BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        gg = k // 32 + offs_s
        g_mask = gg < K // 32
        kk = gg[:, None] * 32 + offs_j[None, :]
        a = tl.load(
            a_ptr + offs_m[None, :, None] * stride_am + kk[:, None, :],
            mask=m_mask[None, :, None] & g_mask[:, None, None],
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[None, :, None] * stride_bn + kk[:, None, :],
            mask=n_mask[None, :, None] & g_mask[:, None, None],
            other=0.0,
        )
        a_s = tl.load(
            as_ptr + offs_m[None, :] * stride_asm + gg[:, None] * stride_ask,
            mask=m_mask[None, :] & g_mask[:, None],
            other=127,
        )
        b_s = tl.load(
            bs_ptr + offs_n[None, :] * stride_bsn + gg[:, None] * stride_bsk,
            mask=n_mask[None, :] & g_mask[:, None],
            other=127,
        )
        a = (a.to(tl.float32) * _e8m0_to_f32(a_s)[:, :, None]).to(tl.bfloat16)
        b = (b.to(tl.float32) * _e8m0_to_f32(b_s)[:, :, None]).to(tl.bfloat16)
        acc = tl.dot(a, tl.permute(b, (0, 2, 1)), acc)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
        tl.sum(acc, axis=0).to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def supports_mxfp8_skinny_shape(m: int, n: int, k: int) -> bool:
    # Past 64 rows the exact bf16-MFMA upcast loses to the portable kernel.
    return 0 < m <= 64 and n > 0 and k > 0 and k % 32 == 0


def launch_triton_mm_mxfp8_skinny_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    alpha: torch.Tensor | None,
    block_size: list[int],
    out: torch.Tensor | None,
) -> torch.Tensor:
    if alpha is not None:
        raise ValueError("gfx950 skinny MXFP8 GEMM does not support alpha")
    if list(block_size) != [1, 32]:
        raise ValueError("gfx950 skinny MXFP8 GEMM requires block_size=[1, 32]")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise TypeError("gfx950 skinny MXFP8 GEMM requires E4M3 A and B")
    if A_scales.dtype != torch.uint8 or B_scales.dtype != torch.uint8:
        raise TypeError("gfx950 skinny MXFP8 GEMM requires uint8 E8M0 scales")
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"gfx950 skinny MXFP8 GEMM does not support {out_dtype}")
    if A.ndim != 2 or B.ndim != 2 or A.stride(-1) != 1 or B.stride(-1) != 1:
        raise ValueError("gfx950 skinny MXFP8 GEMM requires K-contiguous 2D A and B")
    m, k = A.shape
    n = B.shape[0]
    if B.shape[1] != k or not supports_mxfp8_skinny_shape(m, n, k):
        raise ValueError(f"gfx950 skinny MXFP8 GEMM does not cover M={m} N={n} K={k}")
    if A_scales.shape != (m, k // 32) or B_scales.shape != (n, k // 32):
        raise ValueError("gfx950 skinny MXFP8 GEMM scale shapes must be [rows, K/32]")
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=A.device)
    elif out.shape != (m, n) or out.dtype != out_dtype or out.stride(-1) != 1:
        raise ValueError(
            "gfx950 skinny MXFP8 GEMM out must be [M, N] with unit inner stride"
        )
    triton_mm_mxfp8_skinny_gfx950[(triton.cdiv(n, 16),)](
        A,
        B,
        out,
        A_scales,
        B_scales,
        m,
        n,
        k,
        A.stride(0),
        B.stride(0),
        out.stride(0),
        A_scales.stride(0),
        A_scales.stride(1),
        B_scales.stride(0),
        B_scales.stride(1),
        # One compile per power-of-two row bucket: 16, 32, 64.
        BLOCK_M=max(16, triton.next_power_of_2(m)),
        BLOCK_N=16,
        BLOCK_K=256,
        num_warps=8,
        # Staging the upcast tiles through a software pipeline measured slower.
        num_stages=1,
    )
    return out
