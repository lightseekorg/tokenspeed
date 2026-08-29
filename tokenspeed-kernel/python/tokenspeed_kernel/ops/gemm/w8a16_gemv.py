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

Two split kernels:

- ``_w8a16_gemv_split_kernel`` (M == 1): scalar fp32 reduction per weight
  row; wins at M=1 because it only converts/accumulates what the single
  activation row needs,
- ``_w8a16_gemv_split_dot_kernel`` (2 <= M <= 8): the widened fp16 weights
  feed one tensor-core ``tl.dot`` per K-block (M padded to 16). CUDA-core
  fp32 math per byte limits the scalar kernel as M grows; the dot keeps
  throughput flat across M. bf16 activations widen to fp16 with no
  representable-value loss except magnitudes < 2^-24 (subnormal fp16),
  which cannot move a K>=512 reduction at the 3e-3 L2 gate.

N and K are constexpr: decode sees a handful of fixed shapes and the
specialization lets Triton prove 16B alignment of the K-contiguous weight
rows (measured ~2.4x on the 16384x5120 GEMV alone).
"""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _e4m3_u16_to_f16_q8(u16: tl.tensor) -> tl.tensor:
    """Widen uint8 e4m3 bit patterns (in uint16 lanes) to fp16 x 2^-8."""
    h = ((u16 & 0x7F) << 7) | ((u16 & 0x80) << 8)
    return h.to(tl.float16, bitcast=True)


@triton.jit
def _w8a16_gemv_split_kernel(
    A,
    W,
    Bs,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
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
    scale_col = K // GK
    k_per_split = K // KSPLITS
    k_base = pid_k * k_per_split
    w_ptr = W + offs_n[:, None] * K + (k_base + offs_k)[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_base, k_base + k_per_split, BK):
        u8 = tl.load(w_ptr)
        wb = _e4m3_u16_to_f16_q8(u8.to(tl.uint16)).to(tl.float32) * 256.0
        s = tl.load(Bs + (n_off // GN) * scale_col + k0 // GK)
        wb = wb * s[:, None]
        for m in tl.static_range(BM):
            a = tl.load(A + m * K + k0 + offs_k, mask=m < M, other=0.0)
            contrib = tl.sum(a[None, :].to(tl.float32) * wb, axis=1)
            acc += tl.where(tl.arange(0, BM)[:, None] == m, contrib[None, :], 0.0)
        w_ptr += BK

    dst = P + pid_k * BM * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(dst, acc)


@triton.jit
def _w8a16_gemv_split_dot_kernel(
    A,
    W,
    Bs,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GN: tl.constexpr,
    GK: tl.constexpr,
    BM: tl.constexpr,  # padded M (>= 16 for tl.dot)
    BN: tl.constexpr,
    BK: tl.constexpr,  # must equal GK: one scale group per dot
    KSPLITS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_off = pid_n * BN
    offs_n = n_off + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_m = tl.arange(0, BM)
    scale_col = K // GK
    k_per_split = K // KSPLITS
    k_base = pid_k * k_per_split
    w_ptr = W + offs_n[:, None] * K + (k_base + offs_k)[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_base, k_base + k_per_split, BK):
        w = _e4m3_u16_to_f16_q8(tl.load(w_ptr).to(tl.uint16))
        a = tl.load(
            A + offs_m[:, None] * K + k0 + offs_k[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        ).to(tl.float16)
        d = tl.dot(a, tl.trans(w))  # fp32; weights carry the 2^-8 factor
        s = tl.load(Bs + (n_off // GN) * scale_col + k0 // GK)
        acc += d * (s * 256.0)[None, :]
        w_ptr += BK

    dst = P + pid_k * BM * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(dst, acc)


@triton.jit
def _w8a16_gemv_reduce(
    P,
    Out,
    Bias,
    M,
    N,
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
        acc += tl.load(
            P + s * BM * N + offs_m[:, None] * N + offs_n[None, :],
            mask=m_mask,
            other=0.0,
        )
    if HAS_BIAS:
        acc += tl.load(Bias + offs_n).to(tl.float32)[None, :]
    tl.store(
        Out + offs_m[:, None] * N + offs_n[None, :],
        acc.to(Out.dtype.element_ty),
        mask=m_mask,
    )


def _largest_divisor_at_most(k_blocks: int, limit: int) -> int:
    s = min(limit, k_blocks)
    while k_blocks % s:
        s -= 1
    return s


def _pick_config(M: int, N: int, K: int) -> tuple[bool, int, int, int, int]:
    """Return (use_dot, splits, BN, num_warps, num_stages) for a GEMV shape.

    Tuned on GA100 (SM80, 140 SMs) at the Qwen3 27B decode shapes:
    - M == 1: scalar kernel, splits=8/BN=64 (K=5120), splits=17/BN=32
      (K=17408); ~910-990 GB/s effective,
    - 2 <= M <= 8: tensor-core dot kernel, splits=4/BN=128 (splits=17 for
      K=17408); flat ~870-980 GB/s across M.
    """
    k_blocks = K // 128
    if M == 1:
        splits = 17 if K % 17408 == 0 and k_blocks % 17 == 0 else 8
        BN = 32 if splits == 17 else 64
        return False, _largest_divisor_at_most(k_blocks, splits), BN, 4, 3
    splits = 17 if K % 17408 == 0 and k_blocks % 17 == 0 else 4
    return True, _largest_divisor_at_most(k_blocks, splits), 128, 4, 3


def w8a16_decode_gemv(
    A: torch.Tensor,
    W: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
    splits: int | None = None,
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
        splits: Optional K-split factor override; Out partials are
            deterministic per replay.

    Returns:
        ``[M, N]`` output in ``out_dtype``.
    """
    M, K = A.shape
    N = W.shape[0]
    assert out_dtype == torch.bfloat16, "w8a16_decode_gemv emits bf16"
    assert N % 128 == 0 and K % 128 == 0, "N%128 and K%128 required"
    assert A.is_contiguous() and W.is_contiguous() and B_scales.is_contiguous()
    assert M <= 8, "decode GEMV is for M <= 8"
    use_dot, tuned_splits, BN, warps, stages = _pick_config(M, N, K)
    if splits is not None:
        tuned_splits = _largest_divisor_at_most(K // 128, splits)
    if N % BN:
        BN = 64 if N % 64 == 0 else 32
    BM = 16 if use_dot else max(1, triton.next_power_of_2(M))
    partials = torch.empty(tuned_splits, BM, N, device=A.device, dtype=torch.float32)
    w8 = W.view(torch.uint8)
    grid = (N // BN, tuned_splits)
    kernel = _w8a16_gemv_split_dot_kernel if use_dot else _w8a16_gemv_split_kernel
    kernel[grid](
        A,
        w8,
        B_scales,
        partials,
        M,
        N=N,
        K=K,
        GN=128,
        GK=128,
        BM=BM,
        BN=BN,
        BK=128,
        KSPLITS=tuned_splits,
        num_warps=warps,
        num_stages=stages,
    )
    out = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    _w8a16_gemv_reduce[(N // 64,)](
        partials,
        out,
        bias,
        M,
        N,
        BM=BM,
        BN=64,
        KSPLITS=tuned_splits,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return out
