# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention input projection of a decode step, one engine per prepared row count.

GB300, TP8 shard of the K3.1 KDA (6288 x 7168) and DSA (3808 x 7168) projections,
back-to-back launches over weights larger than L2, us:

  rows   KDA  split-K / warp     DSA  split-K / warp     (SOL KDA / DSA)
  1           14.54 / 13.41           10.93 /  8.60      11.02 / 6.68
  5           14.86 / 13.82           11.13 /  8.51      11.04 / 6.69
  10          15.30 / 14.64            9.82 /  8.91      11.05 / 6.70
  32          15.46 / 17.61           10.11 / 10.14      11.13 / 6.76

so up to 16 rows the warp split-K (many small co-resident CTAs keep the weight stream
at HBM speed) and past it the cluster split-K. The cluster split-K is prepared for
every row count: it also takes the operands the warp split-K cannot (a weight that is
not packed BF16 or not 32 B aligned, strided or misaligned activations or outputs).
Both accumulate BF16 products in FP32 and round once to BF16; only the order in which
the K partials are summed differs."""

import torch

from .projection import PreparedProjection
from .warp_splitk import (
    _CTA_RESERVED_SMEM,
    _SM_SMEM_BYTES,
    PreparedWarpSplitK,
    WarpSplitKTactic,
    _smem_bytes,
    check_tactic,
)

WARP_ROWS = 16  # warp split-K up to here; cluster split-K past it


def warp_tactic(m, n, k, sms):
    """Warp split-K tactic of ``m`` rows of an [n, k] weight: 32-row output tiles
    when they fill at most two waves, the smallest token tile holding ``m``, 256-wide
    K tiles and the deepest ring that keeps every wave's CTAs resident on an SM."""
    output = 32 if n % 32 == 0 and n // 32 <= 2 * sms else 16
    token = 8 if m <= 8 else 16 if m <= 16 else 32
    k_tile = 256 if k % 256 == 0 and k // 256 != 2 else 128
    ctas = (n // output) * -(-m // token)
    resident = -(-ctas // sms)
    best = None
    for stages in range(2, min(16, k // k_tile) + 1):
        t = WarpSplitKTactic(output, token, k_tile, stages, 2)
        if resident * (_smem_bytes(t) + _CTA_RESERVED_SMEM) <= _SM_SMEM_BYTES:
            best = t
    if best is None:
        best = WarpSplitKTactic(output, token, k_tile, 2, 2)
    check_tactic(best, n, k)
    return best


def warp_eligible(n, k):
    # ``warp_tactic`` rings at least two K tiles; K = 128 or 256 is a single tile.
    return n % 16 == 0 and k % 128 == 0 and 3 <= k // 128 <= 64


def warp_weight(weight):
    """Whether the warp split-K takes ``weight``.

    Args:
        weight: The [N, K] projection weight.

    Returns:
        True for a packed BF16 weight, 32 B aligned, whose N and K the tiles divide.
    """
    n, k = weight.shape
    return (
        weight.dtype == torch.bfloat16
        and weight.stride() == (k, 1)
        and weight.data_ptr() % 32 == 0
        and warp_eligible(n, k)
    )


def warp_operands(x, out):
    """Whether the warp split-K takes ``x`` and ``out`` (its code has static row
    strides).

    Args:
        x: The [M, K] activations.
        out: The [M, N] output.

    Returns:
        True when both are packed row-major and 32 B aligned.
    """
    k, n = x.shape[1], out.shape[1]
    return (
        x.stride() == (k, 1)
        and out.stride() == (n, 1)
        and x.data_ptr() % 32 == 0
        and out.data_ptr() % 32 == 0
    )


class DecodeProjection:
    """``out[M, N] = x[M, K] @ weight[N, K]^T`` for each prepared M: the warp split-K
    up to ``WARP_ROWS`` rows when it takes the weight, else the cluster split-K."""

    def __init__(self, weight, batches, policies):
        """Compile every engine the prepared row counts can take.

        Args:
            weight: The BF16 [N, K] weight (row pitch kept); it must keep its
                address while the projection is used.
            batches: The row counts ``run`` serves.
            policies: The cluster split-K tactic (M tile, N tile, split, stages)
                of every row count in ``batches``, which the warp split-K's rows
                fall back to for operands it cannot take.
        """
        missing = sorted(set(batches) - set(policies))
        if missing:
            raise ValueError(f"no cluster split-K tactic for batches {missing}")
        n, k = weight.shape
        self.weight = weight
        warp = {}
        if warp_weight(weight):
            sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
            warp = {b: warp_tactic(b, n, k, sms) for b in batches if b <= WARP_ROWS}
        self.warp = PreparedWarpSplitK(weight, warp) if warp else None
        split = {b: policies[b] for b in batches}
        self.split = PreparedProjection(weight, split).prepare()
        self.kinds = {b: "warp" if b in warp else "split_k" for b in batches}

    def engine(self, x, out):
        """The engine ``run(x, out)`` takes.

        Args:
            x: The BF16 [M, K] activations, M one of the prepared row counts.
            out: The BF16 [M, N] output.

        Returns:
            ``"warp"`` or ``"split_k"``.
        """
        rows, (n, k) = x.shape[0], self.weight.shape
        shapes = x.shape[1] == k and out.shape == (rows, n)
        # Both engines compile BF16 activations and outputs.
        dtypes = x.dtype == out.dtype == torch.bfloat16
        if rows not in self.kinds or not shapes or not dtypes:
            raise ValueError(
                f"projection {tuple(x.shape)} {x.dtype} -> {tuple(out.shape)} "
                f"{out.dtype}: prepared BF16 batches {sorted(self.kinds)}, weight "
                f"{tuple(self.weight.shape)}"
            )
        if self.kinds[rows] == "warp" and not warp_operands(x, out):
            return "split_k"
        return self.kinds[rows]

    def run(self, x, out):
        """Write ``x @ weight^T`` into ``out`` with the engine ``engine`` names.

        Args:
            x: The BF16 [M, K] activations, M one of the prepared row counts.
            out: The BF16 [M, N] output.

        Returns:
            ``out``.
        """
        if self.engine(x, out) == "warp":
            return self.warp.run(x, out)
        return self.split.run(x, out)
