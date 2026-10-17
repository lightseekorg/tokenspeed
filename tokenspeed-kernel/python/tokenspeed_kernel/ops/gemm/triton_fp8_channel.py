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

"""Small-M FP8 GEMM with per-token and per-channel scales in its epilogue.

Decode rows of an FP8 W8A8 linear (compressed-tensors ``FP8_DYNAMIC``: FP8
weights with one FP32 scale per output channel, activations quantized per
token at run time) read the whole FP8 weight for a few rows, so the weight
bytes and the launches set the time. This kernel reads the ``[N, K]`` FP8
weight once, ``BLOCK_N`` rows per program, multiplies it with the activation
rows on tensor cores (``W @ A^T``, so the weight rows fill the MMA's M side
and the few tokens its N side), and applies the token and channel scales to
the FP32 accumulator before the one output rounding.

The activations arrive quantized (FP8 codes and their scales);
``tokenspeed_kernel.ops.gemm.mm`` quantizes 16-bit rows first. The tile of
each launch comes from a rule fitted to a GB200 sweep of decode-sized
projections, or from ``_SMALL_M_TILES`` where a shape has a measured entry.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signatures

__all__ = [
    "TRITON_FP8_CHANNEL_SMALL_M_MAX_M",
    "triton_mm_fp8_channel_small_m",
]

_FP8 = torch.float8_e4m3fn
# The rows this kernel takes: the decode rows below PyTorch's row-wise CUTLASS
# GEMM (gemm.torch.TORCH_FP8_SCALED_MIN_M).
TRITON_FP8_CHANNEL_SMALL_M_MAX_M = 64
# K is consumed in whole tiles of at least this many elements.
_K_ALIGN = 128


@triton.jit(do_not_specialize=["M"])
def _fp8_channel_small_m_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    A_SCALE_PER_ROW: tl.constexpr,
    B_SCALE_PER_COL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    m_mask = offs_m < M
    b_ptrs = b_ptr + offs_n[:, None].to(tl.int64) * stride_bn + offs_k[None, :]
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    if A_SCALE_PER_ROW:
        a_scale = tl.load(a_scale_ptr + offs_m, mask=m_mask, other=1.0)
    else:
        a_scale = tl.load(a_scale_ptr + offs_m * 0)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        w = tl.load(b_ptrs + k, mask=n_mask[:, None], other=0.0)
        xq = tl.load(a_ptrs + k, mask=m_mask[:, None], other=0.0)
        acc = tl.dot(w, tl.trans(xq), acc)

    if B_SCALE_PER_COL:
        b_scale = tl.load(b_scale_ptr + offs_n, mask=n_mask, other=1.0)
    else:
        b_scale = tl.load(b_scale_ptr + offs_n * 0)
    # Token scale first, then channel scale, as triton_scaled_mm applies them.
    out = acc * a_scale[None, :]
    out = out * b_scale[:, None]
    c_ptrs = c_ptr + offs_m[None, :].to(tl.int64) * stride_cm + offs_n[:, None]
    tl.store(
        c_ptrs,
        out.to(c_ptr.dtype.element_ty),
        mask=m_mask[None, :] & n_mask[:, None],
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


# (N, K, BLOCK_M) -> (BLOCK_N, BLOCK_K, num_warps, num_stages): measured
# tiles that take precedence over the rule of ``_small_m_config`` for their
# shape. A sweep of a model's weight shapes on its GPU (each BLOCK_M bucket
# averaged over its row counts, 1-16 | 32 | 64) fills it. The entries are the
# fastest tiles of a GB200 sweep for Beam's TP8 rank shards: Q/K/V + output
# gate, output projection, dense gate/up and down. BLOCK_K must divide K: the
# kernel's K loop reads whole tiles, masked along M and N only.
_SMALL_M_TILES: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (1792, 5120, 16): (16, 512, 4, 3),
    (1792, 5120, 32): (16, 256, 4, 4),
    (1792, 5120, 64): (64, 256, 8, 3),
    (5120, 768, 16): (64, 256, 4, 3),
    (5120, 768, 32): (64, 256, 8, 3),
    (5120, 768, 64): (64, 256, 4, 3),
    (5120, 5120, 16): (64, 512, 4, 4),
    (5120, 5120, 32): (64, 512, 4, 3),
    (5120, 5120, 64): (64, 256, 8, 3),
    (5120, 2560, 16): (64, 512, 4, 4),
    (5120, 2560, 32): (64, 512, 4, 3),
    (5120, 2560, 64): (64, 256, 8, 3),
}


def _small_m_config(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`` for ``m`` rows of
    an ``[N, K]`` weight.

    BLOCK_M is ``m`` rounded up to a power of two (at least 16, the smallest
    MMA side Triton takes): three compiled variants cover 1 to 64 rows. The
    shapes of ``_SMALL_M_TILES`` take their entry, whose ``BLOCK_K`` must
    divide ``k`` (a ``ValueError`` otherwise); any other shape takes the
    rule fitted to a GB200 sweep: 16 weight rows per program for narrow weights (at most
    2,048 rows, so the launch still spreads over the SMs) and 64 above, K
    tiles of 512 where K is long, and 64 x 256 tiles on eight warps for the
    64-row bucket.
    """
    block_m = max(16, triton.next_power_of_2(m))
    tile = _SMALL_M_TILES.get((n, k, block_m))
    if tile is not None:
        block_n, block_k, num_warps, num_stages = tile
        if k % block_k:
            raise ValueError(
                f"_SMALL_M_TILES[{(n, k, block_m)}] has BLOCK_K {block_k}, which "
                f"does not divide K {k}: the kernel's K loop reads whole tiles"
            )
        return block_m, block_n, block_k, num_warps, num_stages
    block_k = 256 if k % 256 == 0 else _K_ALIGN
    if block_m == 64:
        return block_m, 64, block_k, 8, 3
    if k % 512 == 0 and k >= 2048:
        block_k = 512
    block_n = 16 if n <= 2048 else 64
    num_stages = 4 if block_m == 16 else 3
    return block_m, block_n, block_k, 4, num_stages


def launch_fp8_channel_small_m(
    a: torch.Tensor,
    a_scales: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Run the kernel with an explicit tile configuration (no checks).

    ``a`` is ``[M, K]`` FP8 with ``a_scales`` (``[M]``/``[M, 1]`` or one
    value); ``weight`` is the ``[N, K]`` FP8 weight with contiguous rows and
    ``weight_scales`` its ``[N]``/``[N, 1]`` or one-value FP32 scales. Writes
    and returns ``out`` (``[M, N]``, contiguous last axis).
    """
    m, k = a.shape
    n = weight.shape[0]
    a_scale_arg = a_scales.reshape(-1)
    b_scales = weight_scales.reshape(-1)
    enable_pdl = pdl_enabled() and current_platform().is_nvidia
    _fp8_channel_small_m_kernel[(triton.cdiv(n, block_n),)](
        a,
        a_scale_arg,
        weight,
        b_scales,
        out,
        m,
        n,
        k,
        a.stride(0),
        weight.stride(0),
        out.stride(0),
        A_SCALE_PER_ROW=a_scale_arg.numel() > 1,
        B_SCALE_PER_COL=b_scales.numel() > 1,
        BLOCK_M=max(16, triton.next_power_of_2(m)),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out


_FP8_CHANNEL_SCALE = ScaleFormat(storage_dtype=torch.float32, granularity="channel")


@register_kernel(
    "gemm",
    "mm",
    name="triton_mm_fp8_channel_small_m",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(
        ("a", "b"), "scaled-fp8", {_FP8}, scale=_FP8_CHANNEL_SCALE
    ),
    # Row-major A and a column-major B (a transposed [N, K] weight with
    # contiguous rows); K in whole 128-element tiles.
    traits={
        "m_max": frozenset({TRITON_FP8_CHANNEL_SMALL_M_MAX_M}),
        "k_align": frozenset({_K_ALIGN}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({False}),
        "b_layout": frozenset({"KN"}),
        "out_dtype": frozenset({torch.bfloat16, torch.float16}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_mm_fp8_channel_small_m(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 GEMM of up to 64 rows with token and channel scales in the epilogue.

    Args:
        A: ``[M, K]`` row-major FP8 E4M3 activations.
        B: ``[K, N]`` column-major FP8 weights (a transposed ``[N, K]``).
        A_scales: FP32 ``[M, 1]`` / ``[M]`` per-token or one-element
            per-tensor scales of ``A``.
        B_scales: FP32 ``[N, 1]`` per-channel or one-element per-tensor scales.
        out_dtype: BF16 or FP16 output dtype.
        alpha: Must be None; the scales carry the dequant.
        block_size: Must be None; the scales are per token or channel.
        out: Optional ``[M, N]`` output buffer with a contiguous last axis.

    Returns:
        ``[M, N]`` tensor ``(A * A_scales) @ (B * B_scales^T)``, ``out`` when
        given.
    """
    if alpha is not None or block_size is not None:
        raise ValueError("row-wise scaled FP8 GEMM takes no alpha or block_size")
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"small-M FP8 GEMM writes BF16 or FP16, not {out_dtype}")
    m, k = A.shape
    if B.dtype != _FP8 or B.shape[0] != k or B.stride(0) != 1:
        raise ValueError(
            "B must be a column-major [K, N] FP8 weight (a transposed [N, K]), "
            f"got {B.dtype} {tuple(B.shape)} stride {B.stride()}"
        )
    if A.stride(-1) != 1 or k % _K_ALIGN:
        raise ValueError(
            f"A must have a contiguous last axis and K a multiple of {_K_ALIGN}, "
            f"got stride {A.stride()} K={k}"
        )
    if A.dtype != _FP8:
        raise ValueError(
            f"A must be FP8 E4M3 (mm quantizes 16-bit rows), got {A.dtype}"
        )
    if A_scales is None:
        raise ValueError("FP8 A needs A_scales")
    if m > TRITON_FP8_CHANNEL_SMALL_M_MAX_M:
        raise ValueError(
            f"small-M FP8 GEMM takes up to {TRITON_FP8_CHANNEL_SMALL_M_MAX_M} "
            f"rows, got {m}"
        )
    weight = B.t()
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=A.device)
    if m == 0:
        return out
    _, block_n, block_k, num_warps, num_stages = _small_m_config(m, n, k)
    return launch_fp8_channel_small_m(
        A,
        A_scales,
        weight,
        B_scales,
        out,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
