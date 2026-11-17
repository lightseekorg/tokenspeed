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

"""Top-k over a vocabulary-parallel head, selected around one all-gather.

``triton_vocab_shard_topk`` takes one rank's ``[rows, shard]`` logits to its
``k`` best ``(value, global id)`` pairs, packed for the all-gather: chunks of
the shard select their best in parallel, then one program per row merges the
chunks. ``triton_vocab_topk_merge`` takes every rank's pairs to the row's
sorted ``k`` best. A row's candidates are ordered by value, then by position
(shard column, or rank-major slot after the gather), lowest first.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import (
    vocab_shard_topk_supports,
    vocab_topk_merge_supports,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "triton_vocab_shard_topk",
    "triton_vocab_topk_merge",
]

# Columns one program selects from. A whole vocab shard in one program leaves
# most CUs idle at a few rows; chunks select in parallel, then one program per
# row merges their candidates.
_SHARD_CHUNK_COLUMNS = 2048


@triton.jit
def _ordered_key16(x):
    # A 16-bit float's bits as an integer that increases with its value (NaN
    # largest, as torch.topk orders it).
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    return tl.where(x != x, 0xFFFF, key)


@triton.jit
def _ordered_key32(x):
    # An fp32 value's bits as an int64 that increases with its value.
    bits = x.to(tl.uint32, bitcast=True).to(tl.int64)
    key = tl.where((bits & 0x80000000) != 0, bits ^ 0xFFFFFFFF, bits | 0x80000000)
    return tl.where(x != x, 0xFFFFFFFF, key)


@triton.jit(
    do_not_specialize=[
        "n_cols",
        "chunk_cols",
        "logit_scale",
        "softcap",
        "vocab_start",
        "stride_mask",
    ]
)
def _vocab_shard_topk_kernel(
    logits_ptr,
    out_ptr,
    mask_ptr,
    stride_l,
    stride_o,
    stride_mask,
    n_cols,
    chunk_cols,
    logit_scale,
    softcap,
    vocab_start,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    MASK: tl.constexpr,
):
    # Program (row, chunk) writes the chunk's best K pairs at pair block
    # ``chunk`` of the row.
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    first = chunk * chunk_cols
    local = tl.arange(0, BLOCK)
    valid = (local < chunk_cols) & (first + local < n_cols)
    x = tl.load(logits_ptr + row * stride_l + first + local, mask=valid, other=0.0)
    if SCALE:
        # Rounded to the logits' dtype, as an in-place multiply would.
        x = (x.to(tl.float32) * logit_scale).to(logits_ptr.dtype.element_ty)
    if SOFTCAP:
        # The logits processor's softcap step for step, rounded to the logits'
        # dtype as it stores them.
        capped = x.to(tl.float32) / softcap
        capped = 2 * tl.sigmoid(2 * capped) - 1
        x = (capped * softcap).to(logits_ptr.dtype.element_ty)
    if MASK:
        # Token bitmask in the xgrammar layout: bit c % 32 of word c // 32 of
        # the row is set when global token c is allowed.
        token = vocab_start + first + local
        word = tl.load(mask_ptr + row * stride_mask + token // 32, mask=valid, other=0)
        allowed = ((word >> (token % 32)) & 1) != 0
        x = tl.where(allowed, x, float("-inf"))
    # Value key above, reversed column below: the largest selection keys are
    # the largest values, lowest column first among equal ones.
    key = (_ordered_key16(x) << 15) | (BLOCK - 1 - local)
    key = tl.where(valid, key, -1)
    # The bitonic selection keeps a power of two; the first K are the best K.
    best = tl.topk(key, K_PAD)
    col = first + BLOCK - 1 - (best & (BLOCK - 1))
    value_key = best >> 15
    bits = tl.where(value_key >= 0x8000, value_key ^ 0x8000, value_key ^ 0xFFFF)
    value = bits.to(tl.uint16).to(logits_ptr.dtype.element_ty, bitcast=True)
    slots = tl.arange(0, K_PAD)
    kept = slots < K
    out_row = out_ptr + row * stride_o + chunk * 2 * K
    tl.store(out_row + slots, value.to(tl.float32), mask=kept)
    # Global ids in fp32: exact below 2**24, which the caller checks.
    tl.store(out_row + K + slots, (col + vocab_start).to(tl.float32), mask=kept)


@register_kernel(
    "sampling",
    "vocab_shard_topk",
    name="triton_vocab_shard_topk",
    solution="triton",
    signatures=format_signatures("logits", "dense", {torch.bfloat16, torch.float16}),
    priority=Priority.PORTABLE,
)
def triton_vocab_shard_topk(
    logits: torch.Tensor,
    k: int,
    logit_scale: float | None,
    vocab_start: int,
    out: torch.Tensor,
    *,
    softcap: float | None,
    vocab_mask: torch.Tensor | None,
) -> None:
    """Write each row's ``k`` best shard logits and their global ids.

    Args:
        logits: ``[M, N]`` 16-bit logits of one vocab shard, unit column
            stride; ``vocab_shard_topk_supports`` must hold.
        k: Candidates kept per row.
        logit_scale: Multiplier applied to every logit (rounded to the logits'
            dtype) before the selection, or None.
        vocab_start: Global vocab id of shard column 0.
        out: ``[M, 2 * k]`` fp32, unit column stride: the values, then the
            global ids as exact fp32 integers, in descending value order
            (lowest column first among equal values).
        softcap: Final logit softcapping ``c`` in ``c * tanh(x / c)``, applied
            after the scale and rounded to the logits' dtype before the
            selection, or None.
        vocab_mask: Contiguous int32 ``[M, ceil(V / 32)]`` token bitmask over
            the whole vocabulary (bit ``t % 32`` of word ``t // 32`` set when
            token ``t`` is allowed); disallowed logits become ``-inf`` before
            the selection. None for no mask.
    """
    rows, n_cols = logits.shape
    if not vocab_shard_topk_supports(logits.dtype, n_cols, k):
        raise ValueError(
            f"vocab shard top-k does not take {logits.dtype} [{rows}, {n_cols}], k={k}"
        )
    if logits.stride(1) != 1 or out.stride(1) != 1 or out.dtype != torch.float32:
        raise ValueError("vocab shard top-k takes unit column strides, fp32 out")
    if tuple(out.shape) != (rows, 2 * k):
        raise ValueError(f"out must be [{rows}, {2 * k}], got {tuple(out.shape)}")
    if vocab_mask is not None and (
        vocab_mask.dtype != torch.int32
        or vocab_mask.ndim != 2
        or vocab_mask.shape[0] < rows
        or vocab_mask.shape[1] * 32 < vocab_start + n_cols
        or vocab_mask.stride(1) != 1
    ):
        raise ValueError(
            f"vocab_mask must be an int32 [>= {rows}, >= "
            f"{triton.cdiv(vocab_start + n_cols, 32)}] bitmask with unit column "
            f"stride, got {vocab_mask.dtype} {tuple(vocab_mask.shape)}"
        )
    if rows == 0:
        return
    # The shard width is fixed per server, so the chunking compiles once.
    chunks = triton.cdiv(n_cols, _SHARD_CHUNK_COLUMNS)
    chunk_cols = triton.cdiv(n_cols, chunks)
    if chunk_cols < k or not vocab_topk_merge_supports(chunks, k):
        chunks, chunk_cols = 1, n_cols
    # One chunk writes straight to out; more go through a pair block each.
    picks = out if chunks == 1 else out.new_empty((rows, chunks * 2 * k))
    _vocab_shard_topk_kernel[(rows, chunks)](
        logits,
        picks,
        vocab_mask,
        logits.stride(0),
        picks.stride(0),
        0 if vocab_mask is None else vocab_mask.stride(0),
        n_cols,
        chunk_cols,
        1.0 if logit_scale is None else float(logit_scale),
        1.0 if softcap is None else float(softcap),
        int(vocab_start),
        K=k,
        K_PAD=triton.next_power_of_2(k),
        BLOCK=triton.next_power_of_2(chunk_cols),
        SCALE=logit_scale is not None,
        SOFTCAP=softcap is not None,
        MASK=vocab_mask is not None,
        num_warps=4,
    )
    if chunks > 1:
        # Chunk-major slots keep the lowest column first among equal values.
        _launch_merge(
            picks.view(rows, chunks, 2 * k).transpose(0, 1),
            k,
            None,
            out[:, k:],
            out[:, :k],
        )


@triton.jit(do_not_specialize=["softcap"])
def _vocab_topk_merge_kernel(
    pairs_ptr,
    ids_ptr,
    values_ptr,
    stride_rank,
    stride_row,
    stride_ids,
    stride_values,
    softcap,
    RANKS: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, BLOCK)
    valid = slot < RANKS * K
    rank = slot // K
    lane = slot - rank * K
    pair = pairs_ptr + row * stride_row + rank * stride_rank + lane
    value = tl.load(pair, mask=valid, other=0.0)
    # Value key above, reversed rank-major slot below: lowest slot first
    # among equal values.
    key = (_ordered_key32(value) << 16) | (BLOCK - 1 - slot)
    key = tl.where(valid, key, -1)
    best = tl.topk(key, K_PAD)
    source = BLOCK - 1 - (best & (BLOCK - 1))
    source_rank = source // K
    source_lane = source - source_rank * K
    picked = pairs_ptr + row * stride_row + source_rank * stride_rank + source_lane
    out = tl.arange(0, K_PAD)
    kept = out < K
    value = tl.load(picked, mask=kept, other=0.0)
    token = tl.load(picked + K, mask=kept, other=0.0)
    if SOFTCAP:
        # The same steps as the logits processor's fused softcap.
        value = value / softcap
        value = 2 * tl.sigmoid(2 * value) - 1
        value = value * softcap
    tl.store(
        ids_ptr + row * stride_ids + out,
        token.to(ids_ptr.dtype.element_ty),
        mask=kept,
    )
    tl.store(values_ptr + row * stride_values + out, value, mask=kept)


@register_kernel(
    "sampling",
    "vocab_topk_merge",
    name="triton_vocab_topk_merge",
    solution="triton",
    signatures=format_signatures("pairs", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def triton_vocab_topk_merge(
    pairs: torch.Tensor,
    k: int,
    softcap: float | None,
    ids_out: torch.Tensor,
    values_out: torch.Tensor,
) -> None:
    """Merge every rank's ``triton_vocab_shard_topk`` pairs into each row's
    sorted ``k`` best.

    Args:
        pairs: ``[R, M, 2 * k]`` fp32 view of the gathered rank blocks, any
            rank and row strides, unit column stride.
        k: Candidates kept per row, as the shard selection kept.
        softcap: Final logit softcapping ``c`` in ``c * tanh(x / c)`` applied
            to the kept values, or None. Monotone, so it never changes which
            candidates survive.
        ids_out: Contiguous ``[M, k]`` int64 or int32 global ids, descending
            value order, lowest rank-major slot first among equal values.
        values_out: Contiguous ``[M, k]`` fp32 values, in the same order.
    """
    ranks, rows, width = pairs.shape
    if width != 2 * k or pairs.stride(2) != 1:
        raise ValueError(f"pairs must be [R, M, {2 * k}] with unit column stride")
    if not vocab_topk_merge_supports(ranks, k):
        raise ValueError(f"vocab top-k merge does not take {ranks} ranks x {k}")
    for name, tensor, dtypes in (
        ("ids_out", ids_out, (torch.int64, torch.int32)),
        ("values_out", values_out, (torch.float32,)),
    ):
        if (
            tuple(tensor.shape) != (rows, k)
            or tensor.dtype not in dtypes
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous [{rows}, {k}] {dtypes}")
    if rows == 0:
        return
    _launch_merge(pairs, k, softcap, ids_out, values_out)


def _launch_merge(
    pairs: torch.Tensor,
    k: int,
    softcap: float | None,
    ids_out: torch.Tensor,
    values_out: torch.Tensor,
) -> None:
    """One program per row over ``[B, M, 2 * k]`` pair blocks; ids are stored
    in ``ids_out``'s dtype (int64, or fp32 to pack them again)."""
    blocks, rows = pairs.shape[:2]
    # B and k are fixed per server, so the merge compiles once.
    _vocab_topk_merge_kernel[(rows,)](
        pairs,
        ids_out,
        values_out,
        pairs.stride(0),
        pairs.stride(1),
        ids_out.stride(0),
        values_out.stride(0),
        1.0 if softcap is None else float(softcap),
        RANKS=blocks,
        K=k,
        K_PAD=triton.next_power_of_2(k),
        BLOCK=max(triton.next_power_of_2(blocks * k), triton.next_power_of_2(k)),
        SOFTCAP=softcap is not None,
        num_warps=1 if blocks * k <= 256 else 4,
    )
