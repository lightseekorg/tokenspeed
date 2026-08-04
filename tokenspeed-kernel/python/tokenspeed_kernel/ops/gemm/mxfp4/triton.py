"""Triton MXFP4 GEMM implementation."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signatures

_MXFP4_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"),
    "mxfp4",
    {torch.uint8},
    scale=ScaleFormat(
        storage_dtype=torch.uint8,
        granularity="block",
        block_shape=(32,),
    ),
)


@triton.jit
def _mxfp4_mm_kernel(
    A,
    B,
    A_scales,
    B_scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_asg: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        packed_k = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        scale_k = k_start // 32 + tl.arange(0, BLOCK_K // 32)
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
            "e2m1",
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


@register_kernel(
    "gemm",
    "mm",
    name="triton_mm_mxfp4",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=_MXFP4_FORMAT_SIGNATURES,
    traits={"quant": frozenset({"mxfp4"})},
    priority=Priority.PORTABLE,
)
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
    C = out if out is not None else torch.empty(M, N, device=A.device, dtype=out_dtype)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 32))
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
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return C
