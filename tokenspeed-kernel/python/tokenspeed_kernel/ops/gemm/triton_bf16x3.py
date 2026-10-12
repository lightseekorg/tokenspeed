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

"""FP32-weight GEMM on BF16 tensor cores through an exact weight split.

``triton_bf16x3_gemm_fp32`` computes ``x @ w.T`` in FP32 for BF16 activations
``x`` and an FP32 weight ``w``. The weight is split once, when it is loaded,
into three BF16 pieces whose sum is ``w`` (:func:`split_fp32_weight_bf16x3`).
A BF16 times BF16 product is exact in FP32, so ``x . w`` becomes three BF16
tensor-core MMAs per K block with no rounding of either operand.

The tensor-core accumulator does not round each addition to nearest, and its
error grows with the number of MMA steps it absorbs. The kernel restarts it
every ``DRAIN`` elements of K and adds each block result into an FP32 register
accumulator with a round-to-nearest add. Split-K partials go to a workspace
and a second kernel adds them in split order: no atomics, so the result does
not depend on scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import TensorDescriptor, tl, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "BF16X3_MIN_M",
    "split_fp32_weight_bf16x3",
    "triton_bf16x3_gemm_fp32",
]

# Up to 16 rows the fixed cost of this kernel (two launches and a split-K
# reduction over 1.5x the FP32 weight bytes) loses to an FP32 kernel that
# streams each weight row once.
BF16X3_MIN_M = 17

# K block of one load and the MMA accumulator span: every BK block is drained.
_BLOCK_K = 64
_DRAIN = 64
_BF16_MAX = torch.finfo(torch.bfloat16).max


def split_fp32_weight_bf16x3(weight: torch.Tensor) -> torch.Tensor:
    """Split an FP32 weight into three BF16 pieces that sum to it.

    ``w1 = bf16(w)``, ``w2 = bf16(w - w1)`` and ``w3 = bf16(w - w1 - w2)``,
    each rounded to nearest even. Both residuals are exact in FP32, and BF16
    keeps FP32's exponent range, so ``w1 + w2 + w3 == w`` for every finite
    ``w`` with ``|w| >= 2**-110``. Smaller weights lose the bits below BF16's
    subnormal spacing: the sum is within ``2**-134`` of ``w``. A finite
    weight past BF16's largest value keeps that value as ``w1`` instead of
    rounding to infinity. Infinite and NaN weights are kept in ``w1`` with
    zero ``w2`` and ``w3``, so they reach the output as they would in an FP32
    GEMM.

    Args:
        weight: ``[N, K]`` FP32 weight.

    Returns:
        Contiguous ``[3, N, K]`` BF16 pieces ``(w1, w2, w3)``.
    """
    if weight.ndim != 2 or weight.dtype != torch.float32:
        raise ValueError(
            f"expected an [N, K] float32 weight, got {weight.dtype} "
            f"{tuple(weight.shape)}"
        )
    finite = torch.isfinite(weight)
    lead = torch.where(finite, weight.clamp(-_BF16_MAX, _BF16_MAX), weight)
    w1 = lead.to(torch.bfloat16)
    r1 = torch.where(finite, weight - w1.float(), 0.0)
    w2 = r1.to(torch.bfloat16)
    w3 = (r1 - w2.float()).to(torch.bfloat16)
    return torch.stack((w1, w2, w3)).contiguous()


@triton.jit
def _bf16x3_block(x_desc, w_desc, a0, b0, k, part, N: tl.constexpr, SWAP: tl.constexpr):
    """``part += x . w3 + x . w2 + x . w1`` over one K block, small pieces
    first. ``w_desc`` views the pieces as ``[3 N, K]``."""
    if SWAP:  # a0: weight row offset, b0: activation row offset
        xt = x_desc.load([b0, k])
        w3t = w_desc.load([2 * N + a0, k])
        w2t = w_desc.load([N + a0, k])
        w1t = w_desc.load([a0, k])
        xT = xt.T
        part = tl.dot(w3t, xT, part)
        part = tl.dot(w2t, xT, part)
        part = tl.dot(w1t, xT, part)
    else:  # a0: activation row offset, b0: weight row offset
        xt = x_desc.load([a0, k])
        w3t = w_desc.load([2 * N + b0, k])
        w2t = w_desc.load([N + b0, k])
        w1t = w_desc.load([b0, k])
        part = tl.dot(xt, w3t.T, part)
        part = tl.dot(xt, w2t.T, part)
        part = tl.dot(xt, w1t.T, part)
    return part


@triton.jit
def _bf16x3_store_tile(
    base_ptr,
    acc,
    a0,
    b0,
    M,
    N: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    SWAP: tl.constexpr,
):
    if SWAP:  # acc is [weight rows, activation rows]
        cols = a0 + tl.arange(0, BA)
        rows = b0 + tl.arange(0, BB)
        tl.store(
            base_ptr + rows.to(tl.int64)[None, :] * N + cols[:, None],
            acc,
            mask=(rows < M)[None, :],
        )
    else:  # acc is [activation rows, weight rows]
        rows = a0 + tl.arange(0, BA)
        cols = b0 + tl.arange(0, BB)
        tl.store(
            base_ptr + rows.to(tl.int64)[:, None] * N + cols[None, :],
            acc,
            mask=(rows < M)[:, None],
        )


@triton.jit
def _bf16x3_gemm_kernel(
    x_desc,
    w_desc,
    ws_ptr,
    out_ptr,
    M,
    KS,
    N: tl.constexpr,
    SWAP: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    DRAIN: tl.constexpr,
    USE_WS: tl.constexpr,
):
    """One ``[BA, BB]`` output tile over the ``KS`` elements of K of split
    ``program_id(2)``. ``SWAP`` puts the weight rows on the MMA's M side for
    few activation rows; TMA zero-fills activation rows past ``M``.
    """
    pid_k = tl.program_id(2)
    a0 = tl.program_id(0) * BA
    b0 = tl.program_id(1) * BB
    k_begin = pid_k * KS
    acc = tl.zeros([BA, BB], tl.float32)
    part = tl.zeros([BA, BB], tl.float32)
    # One flat loop so the loads pipeline over num_stages. The drain is a
    # branch on the loop-carried ``part`` rather than ``acc + tl.dot(.., 0)``,
    # which Triton would fold into one MMA accumulator.
    for i in range(0, KS // BK):
        part = _bf16x3_block(x_desc, w_desc, a0, b0, k_begin + i * BK, part, N, SWAP)
        if (i + 1) % (DRAIN // BK) == 0:
            acc = acc + part
            part = tl.zeros([BA, BB], tl.float32)
    if USE_WS:
        _bf16x3_store_tile(
            ws_ptr + pid_k.to(tl.int64) * M * N, acc, a0, b0, M, N, BA, BB, SWAP
        )
    else:
        _bf16x3_store_tile(out_ptr, acc, a0, b0, M, N, BA, BB, SWAP)


@triton.jit
def _bf16x3_splitk_reduce_kernel(
    ws_ptr, out_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr
):
    """``out = ((p0 + p1) + p2) + ...`` over the split partials."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    s = tl.load(ws_ptr + offs, mask=mask, other=0.0)
    for i in tl.static_range(1, SPLIT):
        s = s + tl.load(ws_ptr + i * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, s, mask=mask)


@dataclass(frozen=True)
class _Bf16x3Config:
    swap: bool  # weight rows on the MMA's M side (block_a), activations on N
    block_a: int
    block_b: int
    num_stages: int
    split: int


def _bf16x3_config(m: int, n: int, k: int) -> _Bf16x3Config:
    """Tile and split-K choice, measured at 256 x K weights on GB200.

    Up to 64 rows the weight rows take the MMA's M side and the rows pad to a
    power of two of at least 16 on its N side. More rows use row-major
    tiles: 128 x 64 from 129 to 512 rows, 64 x 128 otherwise. The split is
    the smallest divisor of ``k / 64`` that brings the grid to 120 CTAs, or
    240 above 512 rows, where a second wave of split-K CTAs paid off.
    Stages fill about 200 KiB of shared memory.
    """
    if m <= 64:
        swap, block_a, block_b, num_stages = (
            True,
            64,
            max(16, triton.next_power_of_2(m)),
            4,
        )
        tiles = (n // block_a) * triton.cdiv(m, block_b)
    else:
        swap = False
        if 128 < m <= 512:
            block_a, block_b, num_stages = 128, 64, 4
        else:
            block_a, block_b, num_stages = 64, 128, 3
        tiles = triton.cdiv(m, block_a) * (n // block_b)
    blocks = k // _BLOCK_K
    target = 120 if m <= 512 else 240
    split = next(
        (s for s in range(1, blocks + 1) if blocks % s == 0 and tiles * s >= target),
        blocks,
    )
    return _Bf16x3Config(swap, block_a, block_b, num_stages, split)


@register_kernel(
    "gemm",
    "decode_gemv",
    name="triton_bf16x3_gemm_fp32",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.float32),
            )
        }
    ),
    traits={
        "m_min": frozenset({BF16X3_MIN_M}),
        "n_align": frozenset({128}),
        # In a GB200 sweep (N 128 to 1024, K 2048 to 8192, 17 to 2048 rows)
        # this kernel was up to 1.6x slower than the FP32 Torch product for
        # 128-row weights at most row counts up to 128, and at least 1.05x
        # faster at every point for weights of 256 rows and more. 128-row
        # weights stay on Torch.
        "n_min": frozenset({256}),
        "n_max": frozenset({1024}),
        "k_align": frozenset({_BLOCK_K}),
        # K stays within that sweep's range. Up to 128 rows the smallest margin
        # for 256-row weights fell from 1.52x at K 6144 to 1.05x at K 2048.
        "k_min": frozenset({2048}),
        "k_max": frozenset({8192}),
        "weight_split": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
    weight_preprocessor=split_fp32_weight_bf16x3,
)
def triton_bf16x3_gemm_fp32(
    x: torch.Tensor, weight_split: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ w.T`` in FP32 from BF16 rows and the split of an FP32 weight.

    The error against an exact product is a few FP32 units of the largest
    output of the row, below that of an FP32 SIMT GEMM over a long K. The
    registry selects it, through ``decode_gemv(..., weight_split=...)``, from
    17 rows for weights of 256 to 1024 rows that are a multiple of 128, K a
    multiple of 64 from 2048 to 8192. On GB200 it was faster than the FP32
    Torch product at every point of a sweep over weights of 256, 512, 768 and
    1024 rows with K from 2048 to 8192, from 17 to 2048 rows; the rest of the
    registered range was not swept. Other shapes that meet the layout below
    compute correctly but are not registered; 128-row weights measured slower
    than the FP32 Torch product at most row counts up to 128.

    Non-finite weights reach the output as in an FP32 GEMM. A non-finite
    activation leaves its output row non-finite too, but where an FP32 GEMM
    gives +-inf this kernel can give NaN: an infinite activation meets pieces
    of opposite signs, or a zero piece.

    Args:
        x: ``[M, K]`` contiguous BF16 activations.
        weight_split: ``[3, N, K]`` pieces from :func:`split_fp32_weight_bf16x3`;
            ``N`` a multiple of 128 and ``K`` of 64.
        out: optional contiguous ``[M, N]`` FP32 destination.

    Returns:
        ``[M, N]`` FP32 output.
    """
    m, k = x.shape
    n = weight_split.shape[1]
    assert x.dtype == torch.bfloat16 and weight_split.dtype == torch.bfloat16
    assert weight_split.shape == (3, n, k) and weight_split.is_contiguous()
    assert n % 128 == 0 and k % _BLOCK_K == 0 and x.is_contiguous()
    if out is None:
        out = torch.empty(m, n, dtype=torch.float32, device=x.device)
    if m == 0:
        return out
    if x.data_ptr() % 16:
        x = x.clone()  # TMA needs a 16-byte aligned base
    cfg = _bf16x3_config(m, n, k)
    x_rows, w_rows = (
        (cfg.block_b, cfg.block_a) if cfg.swap else (cfg.block_a, cfg.block_b)
    )
    x_desc = TensorDescriptor.from_tensor(x, [x_rows, _BLOCK_K])
    w_desc = TensorDescriptor.from_tensor(
        weight_split.view(3 * n, k), [w_rows, _BLOCK_K]
    )
    if cfg.swap:
        grid = (n // cfg.block_a, triton.cdiv(m, cfg.block_b), cfg.split)
    else:
        grid = (triton.cdiv(m, cfg.block_a), n // cfg.block_b, cfg.split)
    # The split divides K / 64, a fixed set per weight, so the reduction
    # compiles at most once per divisor whatever the row count.
    use_ws = cfg.split > 1
    ws = (
        torch.empty(cfg.split * m * n, dtype=torch.float32, device=x.device)
        if use_ws
        else out
    )
    _bf16x3_gemm_kernel[grid](
        x_desc,
        w_desc,
        ws,
        out,
        m,
        k // cfg.split,
        N=n,
        SWAP=cfg.swap,
        BA=cfg.block_a,
        BB=cfg.block_b,
        BK=_BLOCK_K,
        DRAIN=_DRAIN,
        USE_WS=use_ws,
        num_warps=4,
        num_stages=cfg.num_stages,
        enable_fp_fusion=False,
    )
    if use_ws:
        _bf16x3_splitk_reduce_kernel[(triton.cdiv(m * n, 1024),)](
            ws,
            out,
            m * n,
            SPLIT=cfg.split,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
