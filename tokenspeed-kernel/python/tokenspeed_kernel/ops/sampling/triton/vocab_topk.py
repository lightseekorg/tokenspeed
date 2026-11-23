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
(shard column, or rank-major slot after the gather), lowest first. For
output logprobs the shard selection also writes the shard's final logits and
the leading columns the previous shard's logprob blocks read
(``vocab_logprobs.py``).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.activation.triton import (
    tanh_softcap_scalars,
    tanh_softcap_value,
)
from tokenspeed_kernel.ops.sampling import (
    vocab_shard_topk_chunks,
    vocab_shard_topk_supports,
    vocab_topk_merge_supports,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "triton_vocab_shard_topk",
    "triton_vocab_topk_merge",
]

# ``softcap_form`` of ``vocab_shard_topk`` as the kernel's SOFTCAP.
_SOFTCAP_FORMS = {None: 0, "sigmoid": 1, "tanh": 2}


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
        "inv_softcap",
        "vocab_start",
        "stride_mask",
        "stride_v",
        "stride_h",
        "halo_cols",
    ]
)
def _vocab_shard_topk_kernel(
    logits_ptr,
    out_ptr,
    mask_ptr,
    values_ptr,
    halo_ptr,
    stride_l,
    stride_o,
    stride_mask,
    stride_v,
    stride_h,
    n_cols,
    chunk_cols,
    logit_scale,
    softcap,
    inv_softcap,
    vocab_start,
    halo_cols,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    MASK: tl.constexpr,
    VALUES: tl.constexpr,
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
    if SOFTCAP == 1:
        # The logits processor's fused softcap step for step, rounded to the
        # logits' dtype as it stores them.
        capped = x.to(tl.float32) / softcap
        capped = 2 * tl.sigmoid(2 * capped) - 1
        x = (capped * softcap).to(logits_ptr.dtype.element_ty)
    elif SOFTCAP == 2:
        # ``tanh_softcap``'s steps (``(x / cap).tanh() * cap``), likewise.
        capped = tanh_softcap_value(x.to(tl.float32), inv_softcap, softcap)
        x = capped.to(logits_ptr.dtype.element_ty)
    if MASK:
        # Token bitmask in the xgrammar layout: bit c % 32 of word c // 32 of
        # the row is set when global token c is allowed.
        token = vocab_start + first + local
        word = tl.load(mask_ptr + row * stride_mask + token // 32, mask=valid, other=0)
        allowed = ((word >> (token % 32)) & 1) != 0
        x = tl.where(allowed, x, float("-inf"))
    if VALUES:
        # The final logits as the whole rows hold them: at their global
        # columns, and the shard's first ``halo_cols`` for the gather.
        col = first + local
        tl.store(values_ptr + row * stride_v + vocab_start + col, x, mask=valid)
        tl.store(
            halo_ptr + row * stride_h + col,
            x.to(tl.float32),
            mask=valid & (col < halo_cols),
        )
    # Value key above, reversed column below: the largest selection keys are
    # the largest values, lowest column first among equal ones.
    if logits_ptr.dtype.element_ty.primitive_bitwidth == 16:
        key = (_ordered_key16(x) << 15) | (BLOCK - 1 - local)
    else:
        key = (_ordered_key32(x) << 15) | (BLOCK - 1 - local)
    key = tl.where(valid, key, -1)
    # The bitonic selection keeps a power of two; the first K are the best K.
    best = tl.topk(key, K_PAD)
    col = first + BLOCK - 1 - (best & (BLOCK - 1))
    value_key = best >> 15
    if logits_ptr.dtype.element_ty.primitive_bitwidth == 16:
        bits = tl.where(value_key >= 0x8000, value_key ^ 0x8000, value_key ^ 0xFFFF)
        value = bits.to(tl.uint16).to(logits_ptr.dtype.element_ty, bitcast=True)
    else:
        bits = tl.where(
            value_key >= 0x80000000, value_key ^ 0x80000000, value_key ^ 0xFFFFFFFF
        )
        value = bits.to(tl.uint32).to(tl.float32, bitcast=True)
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
    signatures=format_signatures(
        "logits", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
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
    softcap_form: str | None,
    vocab_mask: torch.Tensor | None,
    values_out: torch.Tensor | None,
    halo_out: torch.Tensor | None,
) -> None:
    """Write each row's ``k`` best shard logits and their global ids.

    Args:
        logits: ``[M, N]`` 16-bit or fp32 logits of one vocab shard, unit
            column stride; ``vocab_shard_topk_supports`` must hold.
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
        softcap_form: How the softcap computes, as the logits processor
            does: ``"sigmoid"`` (``c * (2 * sigmoid(2 x / c) - 1)``, the
            fused softcap) or ``"tanh"`` (``tanh_softcap``); None without a
            softcap.
        vocab_mask: Contiguous int32 ``[M, ceil(V / 32)]`` token bitmask over
            the whole vocabulary (bit ``t % 32`` of word ``t // 32`` set when
            token ``t`` is allowed); disallowed logits become ``-inf`` before
            the selection. None for no mask.
        values_out: ``[M, >= vocab_start + N]`` unit-column-stride tensor of
            the logits' dtype receiving the final (scaled, capped, masked)
            logits at their global columns, or None.
        halo_out: fp32 ``[M, H]`` unit-column-stride tensor receiving the
            final logits of the shard's first ``H`` columns (those past ``N``
            left as they are), or None.
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
    if (softcap is None) != (softcap_form is None) or (
        softcap_form not in _SOFTCAP_FORMS
    ):
        raise ValueError(f"softcap {softcap} has no form {softcap_form!r}")
    halo_cols = 0 if halo_out is None else halo_out.shape[1]
    if values_out is not None and (
        values_out.dtype != logits.dtype
        or values_out.shape[0] != rows
        or values_out.shape[1] < vocab_start + n_cols
        or values_out.stride(1) != 1
    ):
        raise ValueError(
            f"values_out must be {logits.dtype} [{rows}, >= {vocab_start + n_cols}]"
        )
    if halo_out is not None and (
        halo_out.dtype != torch.float32
        or halo_out.shape[0] != rows
        or halo_out.stride(1) != 1
    ):
        raise ValueError(f"halo_out must be fp32 [{rows}, H]")
    if (values_out is None) != (halo_out is None):
        raise ValueError("values_out and halo_out come together")
    if rows == 0:
        return
    inv_softcap, softcap32 = (
        tanh_softcap_scalars(softcap) if softcap_form == "tanh" else (1.0, softcap)
    )
    # The shard width is fixed per server, so the chunking compiles once.
    chunks = vocab_shard_topk_chunks(n_cols, k)
    chunk_cols = triton.cdiv(n_cols, chunks)
    # One chunk writes straight to out; more go through a pair block each.
    picks = out if chunks == 1 else out.new_empty((rows, chunks * 2 * k))
    _vocab_shard_topk_kernel[(rows, chunks)](
        logits,
        picks,
        vocab_mask,
        values_out,
        halo_out,
        logits.stride(0),
        picks.stride(0),
        0 if vocab_mask is None else vocab_mask.stride(0),
        0 if values_out is None else values_out.stride(0),
        0 if halo_out is None else halo_out.stride(0),
        n_cols,
        chunk_cols,
        1.0 if logit_scale is None else float(logit_scale),
        1.0 if softcap32 is None else float(softcap32),
        float(inv_softcap),
        int(vocab_start),
        halo_cols,
        K=k,
        K_PAD=triton.next_power_of_2(k),
        BLOCK=triton.next_power_of_2(chunk_cols),
        SCALE=logit_scale is not None,
        SOFTCAP=_SOFTCAP_FORMS[softcap_form],
        MASK=vocab_mask is not None,
        VALUES=values_out is not None,
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
