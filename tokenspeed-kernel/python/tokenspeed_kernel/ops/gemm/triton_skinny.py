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

"""Decode-size BF16 GEMM ``x @ w.T`` streaming the weight once (NVIDIA).

Weight-major split-K: each program owns 64 weight rows (16 for a small
weight) and one K range, the last program of a weight tile folds its splits
in order, and one store writes the FP32 accumulator (or its BF16 rounding,
plus an optional bias). A token row's result therefore does not depend on
how many rows ride along.
Launched with PDL, a program first asks L2 for the head of its weight slice,
which does not depend on the predecessor, and only then waits for the input;
after that wait it releases its dependents.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import TensorDescriptor, tl, triton
from tokenspeed_kernel.platform import current_platform, pdl_enabled

SKINNY_MM_MAX_ROWS = 32
_BLOCK_W = 64
# A small weight's tiles; K splits stay those of 64-row tiles, so the bits match.
_NARROW_BLOCK_W = 16
# Every call tiles 32 rows, so a row's arithmetic never depends on M.
_BLOCK_T = 32
_BLOCK_K = 256
# Per (device, N, K): calls of one shape run in stream order.
_COUNTERS: dict[tuple[torch.device, int, int], torch.Tensor] = {}
_SM_COUNTS: dict[torch.device, int] = {}


@triton.jit
def _prefetch_weight_rows(W, n0, k0, N, stride_w, BW: tl.constexpr, BYTES):
    """One bulk L2 prefetch per weight row of the slice."""
    rows = n0 + tl.arange(0, BW)
    address = (W + tl.minimum(rows, N - 1).to(tl.int64) * stride_w + k0).to(
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
        [address, tl.full((BW,), BYTES, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["rows"])
def _skinny_mm_kernel(
    X,
    W,
    WDesc,
    Bias,
    Y,
    Partials,
    Counters,
    rows,
    stride_w,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    w_tile = pid // SPLITS
    split = pid % SPLITS
    KS: tl.constexpr = K // SPLITS
    n0 = w_tile * BW
    if ENABLE_PDL:
        # Two K blocks only: a whole slice delays first loads (B300: 11.0 -> 9.7 us).
        PREFETCH_K: tl.constexpr = 2 * BK if 2 * BK < KS else KS
        _prefetch_weight_rows(W, n0, split * KS, N, stride_w, BW, PREFETCH_K * 2)
        tl.extra.cuda.gdc_wait()
        # Inputs are ready: successors may launch while this projection streams.
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.arange(0, BT)
    acc = tl.zeros((BW, BT), tl.float32)
    for k0 in tl.range(split * KS, split * KS + KS, BK):
        w = WDesc.load([n0, k0])
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + t[:, None] * K + k[None, :], (t < rows)[:, None], other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    n = n0 + tl.arange(0, BW)
    mask = (n < N)[:, None] & (t < rows)[None, :]
    if SPLITS > 1:
        # The tile's last program folds every split in split order.
        offs = tl.arange(0, BW)[:, None] * BT + tl.arange(0, BT)[None, :]
        tl.store(
            Partials + (w_tile * SPLITS + split).to(tl.int64) * BW * BT + offs, acc
        )
        tl.debug_barrier()
        if tl.atomic_add(Counters + w_tile, 1, sem="acq_rel") == SPLITS - 1:
            first = Partials + (w_tile * SPLITS).to(tl.int64) * BW * BT
            acc = tl.load(first + offs, cache_modifier=".cg")
            for s in tl.static_range(1, SPLITS):
                acc += tl.load(first + s * BW * BT + offs, cache_modifier=".cg")
            tl.store(Counters + w_tile, 0)
            if HAS_BIAS:
                acc += tl.load(Bias + n, n < N, other=0.0).to(tl.float32)[:, None]
            tl.store(Y + t[None, :] * N + n[:, None], acc.to(Y.dtype.element_ty), mask)
    else:
        if HAS_BIAS:
            acc += tl.load(Bias + n, n < N, other=0.0).to(tl.float32)[:, None]
        tl.store(Y + t[None, :] * N + n[:, None], acc.to(Y.dtype.element_ty), mask)


def _splits(tiles: int, k: int) -> int:
    """The fewest K splits (dividing K's blocks) that give each program at
    most 320 KB of weight and the launch at least 32 programs: more splits
    cost more in the fold than they gain (B300, decode shapes)."""
    blocks = k // _BLOCK_K
    for splits in range(1, blocks + 1):
        if blocks % splits:
            continue
        if _BLOCK_W * (k // splits) * 2 <= 320 * 1024 and tiles * splits >= 32:
            return splits
    return blocks


def _weight_rows_per_program(programs: int, device: torch.device) -> int:
    """16 weight rows per program where four times the 64-row launch still
    fits one wave (a router's 384 rows: 36 -> 144 programs, 6.0 -> 3.9 us on
    B300), else 64; bigger launches gain nothing from narrower tiles."""
    sms = _SM_COUNTS.get(device)
    if sms is None:
        sms = _SM_COUNTS[device] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    narrow = _BLOCK_W // _NARROW_BLOCK_W
    return _NARROW_BLOCK_W if programs * narrow <= sms else _BLOCK_W


def skinny_mm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``skinny_mm`` serves these operands."""
    return (
        current_platform().is_nvidia
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.ndim == 2
        and weight.ndim == 2
        and 0 < x.shape[0] <= SKINNY_MM_MAX_ROWS
        and x.shape[1] == weight.shape[1]
        and weight.shape[1] % _BLOCK_K == 0
        and x.is_contiguous()
        and weight.stride(1) == 1
        and weight.stride(0) % 8 == 0
        and weight.data_ptr() % 16 == 0
    )


def skinny_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``x @ weight.T (+ bias)`` with FP32 accumulation and one rounding.

    Args:
        x: BF16 ``[M, K]`` rows, ``M <= SKINNY_MM_MAX_ROWS``.
        weight: BF16 ``[N, K]`` with dense columns and 16-byte aligned rows
            (a column slice of a wider weight works), ``K`` a multiple of 256.
        bias: Optional ``[N]`` added in FP32 before the store.
        out_dtype: ``torch.float32`` or ``torch.bfloat16``.

    Returns:
        ``[M, N]`` in ``out_dtype``; each row is independent of ``M``.
    """
    if not skinny_mm_supported(x, weight) or out_dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("skinny_mm needs decode-size BF16 rows and weights")
    rows, k = x.shape
    n = weight.shape[0]
    splits = _splits(triton.cdiv(n, _BLOCK_W), k)
    block_w = _weight_rows_per_program(triton.cdiv(n, _BLOCK_W) * splits, x.device)
    tiles = triton.cdiv(n, block_w)
    out = torch.empty((rows, n), dtype=out_dtype, device=x.device)
    partials = (
        torch.empty(
            (tiles * splits * block_w * _BLOCK_T,),
            dtype=torch.float32,
            device=x.device,
        )
        if splits > 1
        else out
    )
    key = (x.device, n, k)
    counters = _COUNTERS.get(key)
    if counters is None:
        counters = _COUNTERS[key] = torch.zeros(
            tiles, dtype=torch.int32, device=x.device
        )
    enable_pdl = pdl_enabled()
    _skinny_mm_kernel[(tiles * splits,)](
        x,
        weight,
        TensorDescriptor.from_tensor(weight, [block_w, _BLOCK_K]),
        bias,
        out,
        partials,
        counters,
        rows,
        weight.stride(0),
        N=n,
        K=k,
        SPLITS=splits,
        BW=block_w,
        BT=_BLOCK_T,
        BK=_BLOCK_K,
        HAS_BIAS=bias is not None,
        ENABLE_PDL=enable_pdl,
        num_warps=4 if block_w == _BLOCK_W else 2,
        num_stages=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out
