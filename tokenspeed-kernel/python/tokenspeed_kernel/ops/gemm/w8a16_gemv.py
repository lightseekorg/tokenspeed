"""W8A16 decode GEMV for pre-Hopper NVIDIA (SM80/SM86/SM88).

Serving FP8 checkpoints on hardware without FP8 MMA: the tiled block-scaled
kernel cannot stream weights at bandwidth for tiny M (every configuration
~10x slower than the cuBLAS bf16 GEMV floor), so decode uses a split-K GEMV:

- grid (N/BN, KSPLITS): each program converts and reduces a K-strip,
- partials land in a [KSPLITS, BM, N] fp32 buffer (deterministic store, no
  atomics -- replay-stable under CUDA graph capture),
- a small reduce kernel sums the splits and applies bias.

e4m3 -> fp32 conversion uses an fp16 bitcast trick: place the 7 low e4m3
bits (exp:4 mant:3) into the low fp16 bits and bitcast; fp16 value =
2^(e-15)(1+m/8), x256 = 2^(e-7)(1+m/8) = the e4m3 value exactly (normals
and subnormals); only the e4m3fn NaN pattern maps to a finite value (480),
which never occurs in weights. ~4 ALU ops/element, no data-dependent
selects. Weights arrive as uint8 bit patterns: SM80 Triton cannot load
fp8e4nv at all.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_gemv_split_kernel(
    A, W, Bs, P,
    M, N, K,
    GN: tl.constexpr,
    GK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    KSPLITS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_off = pid_n * BN
    offs_n = n_off + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_m = tl.arange(0, BM)
    m_mask = offs_m[:, None] < M
    scale_row = n_off // GN
    scale_col = K // GK
    k_per_split = K // KSPLITS
    k_base = pid_k * k_per_split

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_base, k_base + k_per_split, BK):
        u8 = tl.load(W + offs_n[:, None] * K + (k0 + offs_k)[None, :])
        u16 = u8.to(tl.uint16)
        h = ((u16 & 0x7F) << 7) | ((u16 & 0x80) << 8)
        wb = h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
        s = tl.load(Bs + scale_row * scale_col + k0 // GK)
        wb = wb * s
        for m in tl.static_range(BM):
            a = tl.load(A + m * K + k0 + offs_k, mask=m < M, other=0.0)
            contrib = tl.sum(a[None, :].to(tl.float32) * wb, axis=1)
            acc += tl.where(tl.arange(0, BM)[:, None] == m, contrib[None, :], 0.0)

    dst = P + pid_k * BM * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(dst, acc)


@triton.jit
def _w8a16_gemv_reduce(
    P, Out, Bias,
    M, N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    KSPLITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    m_mask = offs_m[:, None] < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for s in tl.static_range(KSPLITS):
        acc += tl.load(P + s * BM * N + offs_m[:, None] * N + offs_n[None, :], mask=m_mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(Bias + offs_n).to(tl.float32)[None, :]
    tl.store(Out + offs_m[:, None] * N + offs_n[None, :], acc.to(Out.dtype.element_ty), mask=m_mask)


def w8a16_decode_gemv(
    A: torch.Tensor,
    W: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
    splits: int = 8,
) -> torch.Tensor:
    """``A @ W.T`` for tiny M with block-scaled FP8 weights on SM80.

    Args:
        A: ``[M, K]`` contiguous bf16 activations, ``M <= 8``.
        W: ``[N, K]`` contiguous FP8 (e4m3) weight; read as uint8 bit
            patterns and widened in-kernel.
        B_scales: ``[N // 128, K // 128]`` contiguous fp32 weight block
            scales.
        out_dtype: Output element dtype (bf16).
        bias: Optional ``[N]`` bias added to the output.
        splits: K-split factor; Out partials are deterministic per replay.

    Returns:
        ``[M, N]`` output in ``out_dtype``.
    """
    M, K = A.shape
    N = W.shape[0]
    assert out_dtype == torch.bfloat16, "w8a16_decode_gemv emits bf16"
    assert N % 64 == 0 and K % 128 == 0, "N%64 and K%128 required"
    assert A.is_contiguous() and W.is_contiguous() and B_scales.is_contiguous()
    assert M <= 8, "decode GEMV is for M <= 8"
    BM = max(1, triton.next_power_of_2(M))
    splits = min(splits, K // 128)
    partials = torch.empty(splits, BM, N, device=A.device, dtype=torch.float32)
    _w8a16_gemv_split_kernel[(N // 64, splits)](
        A, W.view(torch.uint8), B_scales, partials,
        M, N, K,
        GN=128, GK=128,
        BM=BM, BN=64, BK=128,
        KSPLITS=splits,
        num_warps=4, num_stages=3,
    )
    out = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    _w8a16_gemv_reduce[(N // 64,)](
        partials, out, bias,
        M, N,
        BM=BM, BN=64, KSPLITS=splits,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return out
