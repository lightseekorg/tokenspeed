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

"""Draft proposals sampled from a vocabulary-parallel head, around one small gather.

``triton_vocab_shard_draft_stats`` reduces each block of one rank's shard
logits to partials: the maximum and exp-sum of ``logits / T``, the best
Gumbel score and the largest logit with their global ids, and a count of
values that are not finite. The ranks gather the partials instead of the
logits. ``triton_vocab_shard_draft_record`` combines every rank's partials
into the row's normalizer and token and writes ``q = softmax(logits / T)``
over the shard's columns. The Gumbel noise is keyed by global token id as
``gumbel_sample_from_pools`` keys it, so the token is the one that kernel
draws from the whole row.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import (
    VOCAB_SHARD_DRAFT_BLOCK,
    VOCAB_SHARD_DRAFT_FIELDS,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "triton_vocab_shard_draft_record",
    "triton_vocab_shard_draft_stats",
]

# An id above every token: the losing id of an empty selection.
_NO_TOKEN = tl.constexpr(2147483647)


@triton.jit
def _head_logits(x, logit_scale, softcap, SCALE: tl.constexpr, SOFTCAP: tl.constexpr):
    # The logits processor's scale and softcap, each rounded to the logits' dtype as it stores them.
    if SCALE:
        x = (x.to(tl.float32) * logit_scale).to(x.dtype)
    if SOFTCAP:
        capped = x.to(tl.float32) / softcap
        capped = 2 * tl.sigmoid(2 * capped) - 1
        x = (capped * softcap).to(x.dtype)
    return x.to(tl.float32)


@triton.jit(
    do_not_specialize=["n_cols", "vocab_start", "logit_scale", "softcap", "stride_l"]
)
def _vocab_shard_draft_stats_kernel(
    logits_ptr,
    slots_ptr,
    temperature_ptr,
    seed_ptr,
    offsets_ptr,
    out_ptr,
    stride_l,
    stride_o: tl.constexpr,
    n_cols,
    vocab_start,
    logit_scale,
    softcap,
    FIELDS: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    # Program (row, block) writes the block's partials at field block ``block`` of the row.
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    local = block * BLOCK + tl.arange(0, BLOCK)
    valid = local < n_cols
    x = tl.load(logits_ptr + row * stride_l + local, mask=valid, other=0.0)
    x = _head_logits(x, logit_scale, softcap, SCALE, SOFTCAP)
    slot = tl.load(slots_ptr + row)
    temperature = tl.maximum(tl.load(temperature_ptr + slot).to(tl.float32), 1.0e-20)
    seed = tl.load(seed_ptr + slot).to(tl.int64)
    offset = tl.load(offsets_ptr + slot).to(tl.int64)
    tokens = vocab_start + local

    # A NaN or +inf of logits / T leaves the row without a finite softmax.
    z = x / temperature
    bad = valid & ((z != z) | (z == float("inf")))
    z = tl.where(valid & ~bad, z, float("-inf"))
    z_max = tl.max(z, axis=0)
    z_sum = tl.sum(
        tl.where(z > float("-inf"), tl.exp(z - z_max), 0.0),
        axis=0,
    )

    # gumbel_sample_from_pools' noise and score, step for step.
    uniform = tl.maximum(tl.rand(tl.randint(seed, offset), tokens), 1.0e-7)
    score = tl.where(z > float("-inf"), z - tl.log(-tl.log(uniform)), float("-inf"))
    best = tl.max(score, axis=0)
    best_id = tl.min(tl.where(valid & (score == best), tokens, _NO_TOKEN), axis=0)

    # The lowest-id maximum of the logits themselves, for greedy rows.
    top = tl.where(valid & (x == x), x, float("-inf"))
    top_max = tl.max(top, axis=0)
    top_id = tl.min(tl.where(valid & (top == top_max), tokens, _NO_TOKEN), axis=0)

    field = tl.arange(0, FIELDS)
    partials = tl.where(field == 0, z_max, 0.0)
    partials = tl.where(field == 1, z_sum, partials)
    partials = tl.where(field == 2, best, partials)
    partials = tl.where(field == 3, best_id.to(tl.float32), partials)
    partials = tl.where(field == 4, top_max, partials)
    partials = tl.where(field == 5, top_id.to(tl.float32), partials)
    partials = tl.where(field == 6, tl.sum(bad.to(tl.float32), axis=0), partials)
    tl.store(out_ptr + row * stride_o + block * FIELDS + field, partials)


@triton.jit(
    do_not_specialize=[
        "n_cols",
        "vocab_start",
        "vocab_size",
        "logit_scale",
        "softcap",
        "sentinel",
        "stride_l",
        "stride_sr",
        "stride_sb",
    ]
)
def _vocab_shard_draft_record_kernel(
    logits_ptr,
    stats_ptr,
    slots_ptr,
    write_slots_ptr,
    temperature_ptr,
    top_k_ptr,
    q_ptr,
    tokens_ptr,
    stride_l,
    stride_sr,
    stride_sb,
    stride_q: tl.constexpr,
    n_cols,
    vocab_start,
    vocab_size,
    logit_scale,
    softcap,
    sentinel,
    FIELDS: tl.constexpr,
    BLOCKS: tl.constexpr,
    PARTIALS: tl.constexpr,
    PARTIALS_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    # Program (row, block) combines every rank's partials of the row, then writes q over the block.
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    partial = tl.arange(0, PARTIALS_PAD)
    present = partial < PARTIALS
    base = (
        stats_ptr
        + (partial // BLOCKS) * stride_sr
        + row * stride_sb
        + (partial % BLOCKS) * FIELDS
    )
    z_max = tl.load(base, mask=present, other=float("-inf"))
    z_sum = tl.load(base + 1, mask=present, other=0.0)
    best = tl.load(base + 2, mask=present, other=float("-inf"))
    best_id = tl.load(base + 3, mask=present, other=_NO_TOKEN).to(tl.int64)
    top = tl.load(base + 4, mask=present, other=float("-inf"))
    top_id = tl.load(base + 5, mask=present, other=_NO_TOKEN).to(tl.int64)
    bad = tl.load(base + 6, mask=present, other=0.0)

    # The row's log-normalizer over every rank, its Gumbel token and its lowest-id maximum.
    row_max = tl.max(z_max, axis=0)
    row_sum = tl.sum(
        tl.where(z_max > float("-inf"), z_sum * tl.exp(z_max - row_max), 0.0),
        axis=0,
    )
    best_max = tl.max(best, axis=0)
    sampled = tl.min(tl.where(best == best_max, best_id, _NO_TOKEN), axis=0)
    top_max = tl.max(top, axis=0)
    canonical = tl.min(tl.where(top == top_max, top_id, _NO_TOKEN), axis=0)
    no_proposal = (tl.sum(bad, axis=0) > 0) | (row_max == float("-inf"))

    slot = tl.load(slots_ptr + row)
    greedy = tl.load(top_k_ptr + slot) == 1
    temperature = tl.maximum(tl.load(temperature_ptr + slot).to(tl.float32), 1.0e-20)
    if block == 0:
        # An empty or NaN row picks no real id; any in-range one does, its q rejects it.
        token = tl.where(greedy, canonical, sampled)
        token = tl.minimum(tl.maximum(token, 0), vocab_size - 1)
        tl.store(tokens_ptr + row, token.to(tl.int32))

    local = block * BLOCK + tl.arange(0, BLOCK)
    valid = local < n_cols
    write_slot = tl.load(write_slots_ptr + row).to(tl.int64)
    if no_proposal:
        q = tl.zeros((BLOCK,), tl.float32) + sentinel
    elif greedy:
        q = tl.where(vocab_start + local == canonical, 1.0, 0.0)
    else:
        x = tl.load(logits_ptr + row * stride_l + local, mask=valid, other=0.0)
        z = _head_logits(x, logit_scale, softcap, SCALE, SOFTCAP) / temperature
        q = tl.where(z > float("-inf"), tl.exp(z - row_max) / row_sum, 0.0)
    tl.store(q_ptr + write_slot * stride_q + local, q, mask=valid)


def _check_rows(name: str, tensor: torch.Tensor, rows: int, dtypes) -> None:
    if tensor.ndim != 1 or tensor.shape[0] < rows or tensor.dtype not in dtypes:
        raise ValueError(
            f"{name} must be a [>= {rows}] {' or '.join(map(str, dtypes))} vector, "
            f"got {tensor.dtype} {tuple(tensor.shape)}"
        )


@register_kernel(
    "sampling",
    "vocab_shard_draft_stats",
    name="triton_vocab_shard_draft_stats",
    solution="triton",
    signatures=format_signatures(
        "logits", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def triton_vocab_shard_draft_stats(
    logits: torch.Tensor,
    vocab_start: int,
    slots: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    out: torch.Tensor,
    *,
    logit_scale: float | None,
    softcap: float | None,
) -> None:
    """Reduce each block of one rank's shard logits to the partials ``vocab_shard_draft_record`` combines.

    Args:
        logits: ``[M, N]`` raw head logits of one vocabulary shard, unit
            column stride.
        vocab_start: Global token id of shard column 0.
        slots: ``[>= M]`` int32 or int64 pool slot of each row.
        temperature_pool: ``[P]`` temperature by slot (floored at 1e-20).
        seed_pool: ``[P]`` int64 Philox seed by slot.
        offsets_pool: ``[P]`` int64 Philox offset by slot.
        out: ``[>= M, vocab_shard_draft_stats_width(N)]`` fp32, unit column
            stride: per block of ``VOCAB_SHARD_DRAFT_BLOCK`` columns,
            ``VOCAB_SHARD_DRAFT_FIELDS`` partials.
        logit_scale: The logits processor's multiplier (rounded to the
            logits' dtype), or None.
        softcap: The processor's final logit softcap, applied after the
            scale and rounded to the logits' dtype, or None.
    """
    rows, n_cols = logits.shape
    blocks = triton.cdiv(n_cols, VOCAB_SHARD_DRAFT_BLOCK)
    if logits.stride(1) != 1 or out.stride(1) != 1 or out.dtype != torch.float32:
        raise ValueError("vocab shard draft stats take unit column strides, fp32 out")
    if (
        out.ndim != 2
        or out.shape[0] < rows
        or out.shape[1] < blocks * VOCAB_SHARD_DRAFT_FIELDS
    ):
        raise ValueError(
            f"out must be [>= {rows}, >= {blocks * VOCAB_SHARD_DRAFT_FIELDS}], "
            f"got {tuple(out.shape)}"
        )
    _check_rows("slots", slots, rows, (torch.int32, torch.int64))
    if seed_pool.dtype != torch.int64 or offsets_pool.dtype != torch.int64:
        raise ValueError("seed_pool and offsets_pool must be int64")
    if rows == 0:
        return
    _vocab_shard_draft_stats_kernel[(rows, blocks)](
        logits,
        slots,
        temperature_pool,
        seed_pool,
        offsets_pool,
        out,
        logits.stride(0),
        out.stride(0),
        n_cols,
        int(vocab_start),
        1.0 if logit_scale is None else float(logit_scale),
        1.0 if softcap is None else float(softcap),
        FIELDS=VOCAB_SHARD_DRAFT_FIELDS,
        BLOCK=VOCAB_SHARD_DRAFT_BLOCK,
        SCALE=logit_scale is not None,
        SOFTCAP=softcap is not None,
        num_warps=8,
    )


@register_kernel(
    "sampling",
    "vocab_shard_draft_record",
    name="triton_vocab_shard_draft_record",
    solution="triton",
    signatures=format_signatures(
        "logits", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def triton_vocab_shard_draft_record(
    logits: torch.Tensor,
    stats: torch.Tensor,
    vocab_start: int,
    slots: torch.Tensor,
    write_slots: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    q_out: torch.Tensor,
    tokens_out: torch.Tensor,
    *,
    logit_scale: float | None,
    softcap: float | None,
    sentinel: float,
) -> None:
    """Combine every rank's partials into each row's token and write its ``q`` over this shard.

    Args:
        logits: ``[M, N]`` raw head logits of this rank's shard, the ones
            ``vocab_shard_draft_stats`` reduced.
        stats: ``[R, M, >= width]`` fp32 view of every rank's gathered
            partials, rank-major, any rank and row strides, unit column
            stride; ``R * N`` is the vocabulary.
        vocab_start: Global token id of shard column 0.
        slots: ``[>= M]`` int32 or int64 pool slot of each row.
        write_slots: ``[>= M]`` int32 or int64 row of ``q_out`` each row writes.
        temperature_pool: ``[P]`` temperature by slot.
        top_k_pool: ``[P]`` int32 top-k by slot; 1 marks a greedy request.
        q_out: ``[P', N]`` fp32, unit column stride, receives each row's
            ``q``: ``softmax(logits / T)`` over this shard's columns, one-hot
            at the lowest-id maximum logit for greedy rows, and ``sentinel``
            everywhere for a row whose softmax is not finite.
        tokens_out: ``[>= M]`` int32 global token ids, the same on every
            rank: the Gumbel-max draw of ``logits / T`` (lowest id among
            equal scores), or the lowest-id maximum for greedy rows.
        logit_scale: As for ``vocab_shard_draft_stats``.
        softcap: As for ``vocab_shard_draft_stats``.
        sentinel: The recorded probability that marks no proposal.
    """
    rows, n_cols = logits.shape
    blocks = triton.cdiv(n_cols, VOCAB_SHARD_DRAFT_BLOCK)
    if stats.ndim != 3 or stats.dtype != torch.float32 or stats.stride(2) != 1:
        raise ValueError(
            f"stats must be an fp32 [R, M, width] view with unit column stride, got "
            f"{stats.dtype} {tuple(stats.shape)} strides {stats.stride()}"
        )
    ranks = stats.shape[0]
    if stats.shape[1] < rows or stats.shape[2] < blocks * VOCAB_SHARD_DRAFT_FIELDS:
        raise ValueError(
            f"stats must hold [{ranks}, >= {rows}, >= "
            f"{blocks * VOCAB_SHARD_DRAFT_FIELDS}], got {tuple(stats.shape)}"
        )
    if q_out.ndim != 2 or q_out.shape[1] != n_cols or q_out.stride(1) != 1:
        raise ValueError(
            f"q_out must be [P, {n_cols}] with unit column stride, got "
            f"{tuple(q_out.shape)} strides {q_out.stride()}"
        )
    if q_out.dtype != torch.float32:
        raise ValueError(f"q_out must be fp32, got {q_out.dtype}")
    _check_rows("slots", slots, rows, (torch.int32, torch.int64))
    _check_rows("write_slots", write_slots, rows, (torch.int32, torch.int64))
    _check_rows("tokens_out", tokens_out, rows, (torch.int32,))
    if rows == 0:
        return
    partials = ranks * blocks
    _vocab_shard_draft_record_kernel[(rows, blocks)](
        logits,
        stats,
        slots,
        write_slots,
        temperature_pool,
        top_k_pool,
        q_out,
        tokens_out,
        logits.stride(0),
        stats.stride(0),
        stats.stride(1),
        q_out.stride(0),
        n_cols,
        int(vocab_start),
        ranks * n_cols,
        1.0 if logit_scale is None else float(logit_scale),
        1.0 if softcap is None else float(softcap),
        float(sentinel),
        FIELDS=VOCAB_SHARD_DRAFT_FIELDS,
        BLOCKS=blocks,
        PARTIALS=partials,
        PARTIALS_PAD=triton.next_power_of_2(partials),
        BLOCK=VOCAB_SHARD_DRAFT_BLOCK,
        SCALE=logit_scale is not None,
        SOFTCAP=softcap is not None,
        num_warps=8,
    )
