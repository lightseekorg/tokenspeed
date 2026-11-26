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

"""Hopper SM90 small-M W8A8 block-128x128 FP8 GEMM via vectorized Split-K + PDL."""

from __future__ import annotations

from functools import cache

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import Platform

HOPPER_SMALL_M_FP8_MAX_ROWS = 32
_BLOCK_N = 64
_BLOCK_K = 128
_COUNTERS: dict[tuple[torch.device, int], torch.Tensor] = {}


@cache
def _select_splits(tiles: int, k_blks: int) -> int:
    """Choose K-splits in (1, 2, 3, 4, 8) for Hopper small-M bandwidth saturation.

    Short-K shapes (<= 8 K-blocks) with >= 64 N-tiles run single-split (zero
    partials/atomics). Larger shapes target 192-512 CTAs and <= 8 K-blocks per CTA.
    """
    if k_blks <= 8 and tiles >= 64:
        return 1
    best = 1
    for sp in (1, 2, 3, 4, 8):
        if k_blks % sp != 0:
            continue
        ctas = tiles * sp
        if sp > 1 and ctas > 512:
            break
        best = sp
        if ctas >= 192 and (k_blks // sp) <= 8:
            return sp
    return best


@triton.jit
def _prefetch_weight_rows_fp8(
    W,
    n0,
    k0,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BYTES: tl.constexpr,
):
    """Issue bulk L2 prefetch for the leading weight bytes of a tile."""
    rows = n0 + tl.arange(0, BN)
    address = (W + tl.minimum(rows, N - 1).to(tl.int64) * K + k0).to(
        tl.pointer_type(tl.int8)
    )
    tl.inline_asm_elementwise(
        """
        {
        cp.async.bulk.prefetch.L2.global [$1], $2;
        mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        [address, tl.full((BN,), BYTES, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["M", "stride_xsm", "stride_xsk"])
def _hopper_small_m_fp8_kernel(
    Xq,
    Xs,
    W,
    Ws,
    Y,
    Partials,
    Counters,
    M,
    stride_xsm,
    stride_xsk,
    stride_ym,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    w_tile = pid // SPLITS
    split = pid % SPLITS
    K_BLKS: tl.constexpr = K // 128
    BLKS_PER_SPLIT: tl.constexpr = K_BLKS // SPLITS
    k_blk_start = split * BLKS_PER_SPLIT
    n0 = w_tile * BN
    scale_row = n0 // 128

    if ENABLE_PDL:
        PREFETCH_BYTES: tl.constexpr = 256 if BLKS_PER_SPLIT >= 2 else 128
        _prefetch_weight_rows_fp8(W, n0, k_blk_start * 128, N, K, BN, PREFETCH_BYTES)
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    offs_n = n0 + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    mask_m = offs_m < M

    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for kb in tl.range(k_blk_start, k_blk_start + BLKS_PER_SPLIT):
        offs_k = kb * 128 + tl.arange(0, 128)
        w_fp8 = tl.load(
            W + offs_n[:, None] * K + offs_k[None, :],
            eviction_policy="evict_first",
        )
        ws = tl.load(Ws + scale_row * K_BLKS + kb).to(tl.float32)
        x_fp8 = tl.load(
            Xq + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xs = tl.load(
            Xs + offs_m * stride_xsm + kb * stride_xsk,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        dot_val = tl.dot(w_fp8, tl.trans(x_fp8), out_dtype=tl.float32)
        acc += dot_val * (ws * xs[None, :])

    out_mask = mask_m[None, :] & (offs_n < N)[:, None]
    if SPLITS == 1:
        tl.store(
            Y + offs_m[None, :] * stride_ym + offs_n[:, None],
            acc.to(Y.dtype.element_ty),
            mask=out_mask,
        )
    else:
        off_p = tl.arange(0, BN)[:, None] * BM + tl.arange(0, BM)[None, :]
        slot = (w_tile * SPLITS + split).to(tl.int64) * (BN * BM)
        tl.store(Partials + slot + off_p, acc, cache_modifier=".cg")
        if (
            tl.atomic_add(Counters + w_tile, 1, sem="acq_rel", scope="gpu")
            == SPLITS - 1
        ):
            base = (w_tile * SPLITS).to(tl.int64) * (BN * BM)
            total = tl.load(Partials + base + off_p, cache_modifier=".cg")
            for s in tl.static_range(1, SPLITS):
                total += tl.load(
                    Partials + base + s * (BN * BM) + off_p,
                    cache_modifier=".cg",
                )
            tl.store(Counters + w_tile, 0)
            tl.store(
                Y + offs_m[None, :] * stride_ym + offs_n[:, None],
                total.to(Y.dtype.element_ty),
                mask=out_mask,
            )


def hopper_small_m_fp8_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> bool:
    """Return True when operands match Hopper small-M block-FP8 GEMM."""
    if not Platform.get().is_hopper or out_dtype != torch.bfloat16:
        return False
    if a.ndim != 2 or b.ndim != 2:
        return False
    m, k = a.shape
    n, kb = b.shape
    # Rotating-weight SM90 measurements show M in (16, 32] stays faster than
    # DeepGEMM only for larger weights (>= 9M elements); smaller shapes cross over at M=16.
    max_m = HOPPER_SMALL_M_FP8_MAX_ROWS if n * k >= 3072 * 3072 else 16
    return (
        0 < m <= max_m
        and k == kb
        and n >= _BLOCK_N
        and n % _BLOCK_N == 0
        and k >= _BLOCK_K
        and k % _BLOCK_K == 0
        and a.is_cuda
        and b.device == a.device
        and a.dtype == torch.float8_e4m3fn
        and b.dtype == torch.float8_e4m3fn
        and a_scales.dtype == torch.float32
        and b_scales.dtype == torch.float32
        and a.is_contiguous()
        and b.is_contiguous()
        and b_scales.is_contiguous()
        and b.data_ptr() % 16 == 0
    )


def hopper_small_m_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Hopper SM90 small-M (M <= 32) W8A8 block-128x128 FP8 GEMM -> BF16."""
    m, k = a.shape
    n = b.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    if m == 0:
        return out

    bm = 16 if m <= 16 else 32
    tiles = n // _BLOCK_N
    k_blks = k // _BLOCK_K
    splits = _select_splits(tiles, k_blks)
    blks_per_split = k_blks // splits
    num_stages = 3 if blks_per_split >= 3 else 2
    num_warps = 8 if (splits == 1 and tiles <= 64) else 4

    if splits > 1:
        partials = torch.empty(
            (tiles * splits * _BLOCK_N * bm,),
            dtype=torch.float32,
            device=a.device,
        )
        key = (a.device, tiles)
        counters = _COUNTERS.get(key)
        if counters is None:
            counters = _COUNTERS[key] = torch.zeros(
                (tiles,), dtype=torch.int32, device=a.device
            )
    else:
        partials = out
        counters = out

    _hopper_small_m_fp8_kernel[(tiles * splits,)](
        a,
        a_scales,
        b,
        b_scales,
        out,
        partials,
        counters,
        m,
        a_scales.stride(0),
        a_scales.stride(1),
        out.stride(0),
        N=n,
        K=k,
        SPLITS=splits,
        BM=bm,
        BN=_BLOCK_N,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out


__all__ = [
    "HOPPER_SMALL_M_FP8_MAX_ROWS",
    "hopper_small_m_fp8_mm",
    "hopper_small_m_fp8_supported",
]
