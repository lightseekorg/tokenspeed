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

"""Split-K FP8 GEMM for decode rows with per-row and per-channel scales.

The GEMM, its reduction and SGLang's tuning entries are vendored from SGLang
in ``tokenspeed_kernel.thirdparty.sglang``; the 256-row tiles and every
[1024, 8192] entry are local.

``W [N, K]`` is FP8 E4M3 with one FP32 scale per output channel and the
activation rows arrive FP8 E4M3 with one FP32 scale per row. ``SK`` programs
along K each accumulate their K slice of a ``BM x BN`` tile in FP32 into a
partial buffer; a second kernel sums the ``SK`` partials and applies both
scales before the one BF16 rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.signature import ScaleFormat, format_signatures
from tokenspeed_kernel.thirdparty.sglang.fp8_small_m_gemm import (
    FP8W_SMALL_M_CONFIGS,
    _fp8_rowmajor_partial,
    _sum_scaled,
)

__all__ = [
    "PARTIAL_SUM_SPLITS",
    "SplitKPartials",
    "fp8_channel_splitk_parts",
    "triton_mm_fp8_channel_splitk",
    "triton_mm_fp8_channel_splitk_partials",
]

_FP8 = torch.float8_e4m3fn
_MAX_M = 256
# (N, K, BM) -> (BN, BK, SK, num_warps, num_stages): SGLang's tuning
# (FP8W_SMALL_M_CONFIGS) plus entries for shapes or row tiles it lacks. BM is
# the row tile rounded up to 32: rows 1-16 take the 32-row entry with a 16-row
# tile, rows 129-256 the 256-row entry.
_CONFIGS = {
    **FP8W_SMALL_M_CONFIGS,
    (4608, 8192, 256): (128, 128, 4, 8, 3),
    (2304, 8192, 256): (64, 128, 4, 8, 3),
    (1024, 8192, 32): (32, 256, 8, 4, 3),
    (1024, 8192, 64): (32, 256, 8, 4, 3),
    (1024, 8192, 128): (32, 256, 8, 4, 3),
    (1024, 8192, 256): (32, 128, 8, 4, 3),
}
# The GEMM and reduction kernels index the [split_k, rows, n] partials and the
# [rows, n] output in int32: every served problem has at most _MAX_M rows and
# a _CONFIGS width and split count, which keeps those offsets far below 2^31.
if max(n * sk for (n, _, _), (_, _, sk, _, _) in _CONFIGS.items()) * _MAX_M >= 2**31:
    raise RuntimeError("a split-K FP8 GEMM entry's partials overflow int32 offsets")


def _row_tile(m: int) -> int:
    return max(16, 1 << (m - 1).bit_length())


def _config(m: int, n: int, k: int) -> tuple[int, int, int, int, int] | None:
    return _CONFIGS.get((n, k, max(32, _row_tile(m))))


def _supports_problem(m: int, n: int, k: int) -> bool:
    return 0 < m <= _MAX_M and _config(m, n, k) is not None


# Split counts whose _sum_scaled order a consumer of the partials reproduces.
# _sum_scaled lays its [SK, 512] tile on warps [2, 2] (sizePerThread [1, 4]):
# a thread holds rows r, r + 2, r + 4, ... of one parity, sums them in-thread
# as a balanced tree, and the two parities' warps are added last. So the sum
# is p0, p0 + p1, (p0 + p2) + (p1 + p3), and at 8
# ((p0 + p2) + (p4 + p6)) + ((p1 + p3) + (p5 + p7)) -- not neighbour pairs.
# Other splits stay with _sum_scaled.
PARTIAL_SUM_SPLITS = (1, 2, 4, 8)


def _supports_partials_problem(m: int, n: int, k: int) -> bool:
    return fp8_channel_splitk_parts(m, n, k) in PARTIAL_SUM_SPLITS


@dataclass(frozen=True)
class SplitKPartials:
    """A split-K FP8 GEMM stopped before its reduction.

    The BF16 result is ``bf16(sum(partials) * row_scales[:, None] *
    col_scales)`` with the sum over the split axis in ``_sum_scaled``'s order
    (``PARTIAL_SUM_SPLITS``); the consumer finishes it.

    Attributes:
        partials: FP32 ``[split_k, rows, n]`` partial products.
        row_scales: FP32 ``[rows]`` activation scales.
        col_scales: FP32 ``[n]`` weight channel scales.
    """

    partials: torch.Tensor
    row_scales: torch.Tensor
    col_scales: torch.Tensor


_FP8_CHANNEL_SCALE = ScaleFormat(storage_dtype=torch.float32, granularity="channel")


@register_kernel(
    "gemm",
    "mm",
    name="triton_mm_fp8_channel_splitk",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=format_signatures(
        ("a", "b"), "scaled-fp8", {_FP8}, scale=_FP8_CHANNEL_SCALE
    ),
    traits={
        "mnk_problem_filter": frozenset({_supports_problem}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({False}),
        "b_layout": frozenset({"KN"}),
        "out_dtype": frozenset({torch.bfloat16}),
    },
    # Below triton_mm_fp8_scaled (PERFORMANT + 2): its split-K count follows
    # the row count, so it is not batch-invariant, and mm's ranked selection
    # must not move other FP8 models onto it. Callers that need it name it
    # (override=).
    priority=Priority.PERFORMANT,
)
def triton_mm_fp8_channel_splitk(
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
    """Split-K FP8 GEMM of up to 256 rows with row and channel scales.

    Args:
        A: ``[M, K]`` FP8 E4M3 activation rows, contiguous.
        B: ``[K, N]`` FP8 E4M3 weight, a transposed contiguous ``[N, K]``.
        A_scales: FP32 per-row scales, ``[M]`` or ``[M, 1]``.
        B_scales: FP32 per-output-channel scales, ``[N]`` or ``[N, 1]``.
        out_dtype: BF16.
        alpha: Must be None; the scales carry the dequantization.
        block_size: Must be None.
        out: Optional contiguous ``[M, N]`` BF16 output.

    Returns:
        ``[M, N]`` ``(A * A_scales) @ (B * B_scales^T)`` in BF16. The codes
        and scales must be finite.
    """
    if alpha is not None or block_size is not None:
        raise ValueError("row-wise scaled FP8 GEMM takes no alpha or block_size")
    m, k = A.shape
    weight = B.t()
    n = weight.shape[0]
    config = _config(m, n, k)
    if (
        out_dtype != torch.bfloat16
        or A.dtype != _FP8
        or not A.is_contiguous()
        or weight.dtype != _FP8
        or not weight.is_contiguous()
        or A_scales is None
        or B_scales is None
        or A_scales.dtype != torch.float32
        or B_scales.dtype != torch.float32
        or A_scales.numel() != m
        or B_scales.numel() != n
        or not A_scales.is_contiguous()
        or not B_scales.is_contiguous()
    ):
        raise ValueError("unsupported split-K FP8 GEMM operands")
    if config is None or m > _MAX_M:
        # Only a by-name selection reaches here; ranked selection's problem
        # filter has already declined.
        raise NoKernelFoundError(
            f"no tuned split-K FP8 GEMM entry for {m} rows of [{n}, {k}]"
        )
    if out is not None and (
        out.shape != (m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError("split-K FP8 GEMM output must be contiguous [M, N] BF16")
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=A.device)
    if m == 0:
        return out
    partials = _fp8_channel_splitk_partials(A, weight)
    _sum_scaled[(triton.cdiv(m * n, 512),)](
        partials,
        A_scales.reshape(-1),
        B_scales.reshape(-1),
        out,
        m,
        n,
        partials.shape[0],
        512,
        num_warps=4,
    )
    return out


def fp8_channel_splitk_parts(m: int, n: int, k: int) -> int | None:
    """Split-K slices of ``triton_mm_fp8_channel_splitk`` at this problem, or
    None where it does not run (no tuned config, or more than 256 rows)."""
    config = _config(m, n, k)
    return config[2] if config is not None and 0 < m <= _MAX_M else None


def _fp8_channel_splitk_partials(A: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The split-K FP32 partial sums of ``A @ weight^T``, before ``_sum_scaled``.

    Args:
        A: ``[M, K]`` FP8 E4M3 activation rows, contiguous, ``0 < M <= 256``.
        weight: ``[N, K]`` FP8 E4M3 weight, contiguous; a tuned (N, K).

    Returns:
        ``[split_k, M, N]`` FP32: each plane one slice of K's unscaled sum.
    """
    m, k = A.shape
    n = weight.shape[0]
    config = _config(m, n, k)
    if (
        A.dtype != _FP8
        or weight.dtype != _FP8
        or not A.is_contiguous()
        or not weight.is_contiguous()
        or config is None
        or not 0 < m <= _MAX_M
    ):
        raise ValueError("unsupported split-K FP8 GEMM operands")
    block_n, block_k, split_k, num_warps, num_stages = config
    # Transient, freed after the reduction: at most 256 rows x 4608 columns
    # x 4 splits of FP32 (18 MiB, the most any tuned entry takes), which the caching
    # allocator recycles between steps.
    partials = torch.empty((split_k, m, n), device=A.device, dtype=torch.float32)
    _fp8_rowmajor_partial[(n // block_n, split_k)](
        A,
        weight,
        partials,
        m,
        n,
        k,
        _row_tile(m),
        block_n,
        block_k,
        split_k,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=16,
    )
    return partials


@register_kernel(
    "gemm",
    "mm_split_k_partials",
    name="triton_mm_fp8_channel_splitk_partials",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=format_signatures(
        ("a", "b"), "scaled-fp8", {_FP8}, scale=_FP8_CHANNEL_SCALE
    ),
    traits={
        "mnk_problem_filter": frozenset({_supports_partials_problem}),
        "a_inner_stride_one": frozenset({True}),
        "b_inner_stride_one": frozenset({False}),
        "b_layout": frozenset({"KN"}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_mm_fp8_channel_splitk_partials(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
) -> SplitKPartials:
    """``triton_mm_fp8_channel_splitk`` without its reduction launch.

    Arguments match ``triton_mm_fp8_channel_splitk``. Serves only problems
    whose split count is in ``PARTIAL_SUM_SPLITS``, the splits whose
    reduction order a consumer reproduces bit for bit.

    Returns:
        The unreduced :class:`SplitKPartials` of ``(A * A_scales) @ (B *
        B_scales^T)``.
    """
    m, k = A.shape
    weight = B.t()
    if fp8_channel_splitk_parts(m, weight.shape[0], k) not in PARTIAL_SUM_SPLITS:
        raise ValueError("unsupported split-K FP8 partials problem")
    return SplitKPartials(
        _fp8_channel_splitk_partials(A, weight),
        A_scales.reshape(-1),
        B_scales.reshape(-1),
    )
