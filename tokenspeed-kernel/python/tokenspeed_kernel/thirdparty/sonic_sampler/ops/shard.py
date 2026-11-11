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

"""Vocabulary-sharded sampling: each rank's top-k candidates, and the draw from their merge.

A vocabulary-parallel head leaves each rank the logits of its own vocabulary
columns. Each rank reduces its shard with ``bitpacked_reduction_kernel`` (at
its ``shard_offset``) and ``shard_candidates_kernel`` writes its best
``max_k`` (value, global token id) pairs; the ranks gather the pairs, and
``cumulative_selection_kernel`` and ``cumulative_unpack_kernel`` merge them
into each row's sorted best ``max_k`` over the whole vocabulary
(``merged_candidates``, ``merged_first``) in place of their own unpack. A
verify whose draft probabilities each rank records over its own shard has
the pairs carry them as a third block (``merged_candidates_with_drafts``).
A row drawing from its whole vocabulary (``whole_block``) gathers its rank's
best perturbed value instead, with a verify's normalizer and draft
(``merged_whole``).
"""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton  # noqa.

jit = triton.jit

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import fast_expf, gdc_wait
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.prologue import (
    exact_candidates,
    whole_normalizer,
    whole_row,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.unpack import unpack_top_k

__all__ = [
    "merged_candidates",
    "merged_candidates_with_drafts",
    "merged_first",
    "merged_whole",
    "shard_candidates_kernel",
]


@jit
def first_maximum(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    exact: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_s: tl.constexpr,
    block_n: tl.constexpr,
    block_v: tl.constexpr,
):
    """A greedy row's maximum over this rank's blocks, as fp32, and the first column holding it."""

    span = tl.arange(0, block_v)
    span_mask = span < vocab_blocks
    offsets = (row_id * stride_s) + span

    if enable_pdl:

        gdc_wait()

    if exact:

        # Each block's fp32 maximum bits, then its first column (``exact_maximum``).
        maxima = tl.load(s_ptr + offsets, mask=span_mask, other=0xFF800000)
        columns = tl.load(s_ptr + offsets + vocab_blocks, mask=span_mask, other=0)

        value, block = tl.max(
            maxima.to(tl.float32, bitcast=True), axis=0, return_indices=True
        )
        column = tl.sum(tl.where(span == block, columns, 0), axis=0).to(tl.int32)

    else:

        # Each block's key: its encoded bf16 maximum above its first column, inverted (``greedy_reduction``).
        keys = tl.load(s_ptr + offsets, mask=span_mask, other=0)
        encoded = keys >> 16
        top = tl.max(encoded, axis=0)

        # The first block holding the maximum holds the first maximal column.
        block = tl.min(tl.where(encoded == top, span, block_v), axis=0)
        key = tl.sum(tl.where(span == block, keys, 0), axis=0)
        column = (block * block_n + ((block_n - 1) ^ (key & 0xFFFF))).to(tl.int32)

        # Undo the order-preserving encoding of the bf16 bits.
        bits = tl.where((top & 0x8000) != 0, top ^ 0x8000, top ^ 0xFFFF)
        value = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)

    return value, column


@jit
def shard_candidates_kernel(
    # Input Data Pointer(s).
    s_ptr,  # Scratch-Pad -> [ L, Z_v • M ] of this rank's vocab blocks.
    z_ptr,  # Indicators -> [ B ].
    f_ptr,  # Slot Mapping -> [ B ].
    # Output Data Pointer(s).
    o_ptr,  # Candidate Pairs -> [ L, 2 • M ] fp32: values, then global token ids.
    # Model Specific(s).
    vocab_blocks: tl.constexpr,  # This Rank's Vocab Blocks -> Z_v.
    lookahead: tl.constexpr,  # Drafted Tokens per Request -> γ (0 for one row per request).
    max_k: tl.constexpr,  # Maximum Top-K Value -> M.
    total_cols: tl.constexpr,  # Total Bit-Packed Columns -> Z_v' • M.
    # Conditional Flag(s).
    is_alternating: tl.constexpr,  # Alternating Ascending-Descending Tile(s) Flag.
    enable_pdl: tl.constexpr,  # PDL Enablement Flag.
    # Stride(s).
    stride_s: tl.constexpr,  # Batch Stride of Scratch-Pad.
    stride_o: tl.constexpr,  # Batch Stride of Candidate Pairs.
    # Kernel Specific(s).
    block_n: tl.constexpr,  # Block Size of Vocab Block(s).
    block_v: tl.constexpr,  # Next Power-of-2 Vocab Block(s) -> Z_v'.
    # Exact fp32 Re-Read of the Candidates (exact only).
    x_ptr,  # Shard Logits -> [ L, V_s ].
    gr_ptr,  # Grammar Bit-Mask -> [ B • (γ' + 1), V_t / 32 ].
    cr_ptr,  # Context Counts -> [ B, V_t ].
    dr_ptr,  # Decode Counts -> [ B, V_t ].
    tr_ptr,  # Drafted Tokens -> [ L ].
    rr_ptr,  # Repetition Penalties -> [ B ].
    fr_ptr,  # Frequency Penalties -> [ B ].
    pr_ptr,  # Presence Penalties -> [ B ].
    br_ptr,  # Logit Bias -> [ B, V_t ].
    er_ptr,  # Temperature -> [ B ].
    vocab_size: tl.constexpr,  # This Rank's Vocab Columns -> V_s.
    shard_offset: tl.constexpr,  # Global Token Id of Shard Column 0.
    stride_x: tl.constexpr,  # Batch Stride of Shard Logits.
    stride_gr: tl.constexpr,  # Batch Stride of Grammar Bit-Mask.
    stride_cr: tl.constexpr,  # Batch Stride of Counts and Bias.
    block_t: tl.constexpr,  # Block Size of Drafted Tokens.
    exact: tl.constexpr,  # Exact fp32 Logits Flag.
    logit_scale,  # Logit Scale of the Raw Head Logits, or None.
    softcap,  # Final Logit Softcap of the Raw Head Logits, or None.
    # This Rank's Recorded Draft Probabilities (a verify's third pairs block).
    d_ptr=None,  # Draft Probabilities -> [ P, γ, V_s ] over this rank's columns.
    stride_d=None,  # Slot Stride of Draft Probabilities.
    stride_dg: tl.constexpr = None,  # Step Stride of Draft Probabilities.
    whole: tl.constexpr = False,  # Whole-Vocabulary Rows Flag (``whole_block``).
) -> None:
    """
    Writes each row's best ``max_k`` candidates over this rank's vocab shard, as
    ``vocab_topk_merge`` reads them: the values in descending order (the lower
    column first among equal values), then their global token ids as exact fp32
    integers; with ``d_ptr``, then the draft probability the slot recorded at
    each candidate for the row's draft (zero for greedy and bonus rows).

    Note(s):

        • Sampled rows keep the top-k their bf16 keys select, with their bf16
          values; under ``exact`` their fp32 values are re-read and re-sorted
          (``exact_candidates``), so the merge selects the fp32 top-k.

        • Greedy rows keep only their first maximum; the other slots are -inf.

        • Whole-vocabulary rows keep their best perturbed value and its global
          token id in slot 0; a verify's draft rows add their maximum value
          and exp-sum in slot 1 and their draft's value and perturbed value
          in slot 2, as raw fp32 in both blocks (``merged_whole``).

        • Rows are request-major, ``lookahead + 1`` per request.

    """
    timesteps: tl.constexpr = lookahead + 1

    row_id = tl.program_id(axis=0)
    request = row_id // timesteps
    step_id = row_id % timesteps
    margin = request * timesteps

    batch_id = tl.load(f_ptr + request)
    indicator = tl.load(z_ptr + batch_id)

    span = tl.arange(0, max_k)
    pairs = (row_id * stride_o) + span

    if whole_row(indicator, whole, lookahead == 0):

        value, column = first_maximum(
            s_ptr, row_id, vocab_blocks, True, enable_pdl, stride_s, block_n, block_v
        )

        values = tl.where(span == 0, value, -float("inf"))
        indices = tl.where(span == 0, column, 0)
        ids = (indices + shard_offset).to(tl.float32)

        if lookahead > 0:

            maximum, total, draft_value, draft_score = whole_normalizer(
                s_ptr, row_id, vocab_blocks, stride_s, block_v
            )

            values = tl.where(span == 1, maximum, values)
            values = tl.where(span == 2, draft_value, values)
            ids = tl.where(span == 1, total, ids)
            ids = tl.where(span == 2, draft_score, ids)

    elif (indicator & 0x40) > 0:

        value, column = first_maximum(
            s_ptr, row_id, vocab_blocks, exact, enable_pdl, stride_s, block_n, block_v
        )

        values = tl.where(span == 0, value, -float("inf"))
        indices = tl.where(span == 0, column, 0)
        ids = (indices + shard_offset).to(tl.float32)

    else:

        values, indices = unpack_top_k(
            s_ptr,
            row_id,
            vocab_blocks,
            1,
            vocab_size,
            max_k,
            None,
            block_v,
            total_cols,
            False,
            False,
            is_alternating,
            enable_pdl,
            None,
            None,
            stride_s,
            block_n,
            block_v,
        )

        if exact:

            # Re-read the candidates' exact values: bf16 keys only chose them.

            values, indices = exact_candidates(
                x_ptr,
                row_id,
                indices,
                indicator,
                None,
                gr_ptr,
                cr_ptr,
                dr_ptr,
                tr_ptr,
                rr_ptr,
                fr_ptr,
                pr_ptr,
                br_ptr,
                er_ptr,
                batch_id,
                step_id,
                margin,
                timesteps,
                vocab_size,
                shard_offset,
                lookahead,
                False,
                lookahead > 0,
                lookahead == 0,
                stride_x,
                stride_gr,
                stride_cr,
                None,
                block_t,
                max_k,
                logit_scale,
                softcap,
            )

        # The greedy branch's types: fp32 values, int32 columns.
        values = values.to(tl.float32)
        indices = indices.to(tl.int32)
        ids = (indices + shard_offset).to(tl.float32)

    tl.store(o_ptr + pairs, values)
    tl.store(o_ptr + pairs + max_k, ids)

    if d_ptr is not None:

        # Only this rank holds q over its columns, so its candidates carry q to the merge.
        drafts = tl.zeros((max_k,), dtype=tl.float32)

        if ((indicator & 0x40) == 0) & (step_id < lookahead):

            positions = (batch_id * stride_d) + (step_id * stride_dg) + indices
            drafts = tl.load(
                d_ptr + positions, mask=(indices >= 0) & (indices < vocab_size), other=0
            )

        tl.store(o_ptr + pairs + 2 * max_k, drafts)


@jit
def merged_keys(
    cand_ptr,
    row_id,
    ranks: tl.constexpr,
    stride_mr,
    stride_mb: tl.constexpr,
    enable_pdl: tl.constexpr,
    max_k: tl.constexpr,
):
    """Every rank's candidates of a row as sortable keys (the value's order-preserving bits above the inverted rank-major slot), and their token ids."""

    span = tl.arange(0, ranks * max_k)
    pairs = cand_ptr + (span // max_k) * stride_mr + row_id * stride_mb + span % max_k

    if enable_pdl:

        gdc_wait()

    bits = tl.load(pairs).to(tl.uint32, bitcast=True)
    ids = tl.load(pairs + max_k).to(tl.int32)

    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
    keys = (ordered.to(tl.uint64) << 32) | (~span.to(tl.uint32)).to(tl.uint64)

    return keys, ids


@jit
def unpack_key(keys):
    """The values and rank-major slots ``merged_keys`` packed."""

    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where((ordered & 0x80000000) != 0, ordered & 0x7FFFFFFF, ~ordered)
    slots = (~(keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)

    return bits.to(tl.float32, bitcast=True), slots


@jit
def merged_selection(
    cand_ptr,
    row_id,
    ranks: tl.constexpr,
    stride_mr,
    stride_mb: tl.constexpr,
    enable_pdl: tl.constexpr,
    max_k: tl.constexpr,
):
    """A row's best ``max_k`` gathered values in descending order (fp32), their rank-major slots, and every gathered token id."""

    keys, ids = merged_keys(
        cand_ptr, row_id, ranks, stride_mr, stride_mb, enable_pdl, max_k
    )

    best = tl.gather(tl.sort(keys, descending=True), tl.arange(0, max_k), axis=0)
    values, slots = unpack_key(best)

    return values, slots, ids


@jit
def merged_candidates(
    cand_ptr,
    row_id,
    ranks: tl.constexpr,
    stride_mr,
    stride_mb: tl.constexpr,
    exact: tl.constexpr,
    enable_pdl: tl.constexpr,
    max_k: tl.constexpr,
):
    """
    A row's best ``max_k`` candidates over the whole vocabulary, merged from
    every rank's gathered pairs (``[ranks, rows, 2 • max_k]`` at ``cand_ptr``):
    the values in descending order, the lowest rank-major slot (so the lowest
    token id) first among equal values (bf16 unless ``exact``, as the unpack
    keeps them), and their token ids.
    """

    values, slots, ids = merged_selection(
        cand_ptr, row_id, ranks, stride_mr, stride_mb, enable_pdl, max_k
    )
    indices = tl.gather(ids, slots, axis=0)

    if not exact:

        values = values.to(tl.bfloat16)

    return values, indices


@jit
def merged_candidates_with_drafts(
    cand_ptr,
    row_id,
    ranks: tl.constexpr,
    stride_mr,
    stride_mb: tl.constexpr,
    exact: tl.constexpr,
    enable_pdl: tl.constexpr,
    max_k: tl.constexpr,
):
    """``merged_candidates`` from pairs with a third block (``[ranks, rows, 3 • max_k]``), and the draft probabilities it holds for the merged candidates."""

    values, slots, ids = merged_selection(
        cand_ptr, row_id, ranks, stride_mr, stride_mb, enable_pdl, max_k
    )

    span = tl.arange(0, ranks * max_k)
    pairs = cand_ptr + (span // max_k) * stride_mr + row_id * stride_mb + span % max_k
    drafts = tl.load(pairs + 2 * max_k)

    indices = tl.gather(ids, slots, axis=0)
    proposals = tl.gather(drafts, slots, axis=0)

    if not exact:

        values = values.to(tl.bfloat16)

    return values, indices, proposals


@jit
def merged_first(
    cand_ptr,
    row_id,
    ranks: tl.constexpr,
    stride_mr,
    stride_mb: tl.constexpr,
    enable_pdl: tl.constexpr,
    max_k: tl.constexpr,
):
    """A greedy row's first maximum over the whole vocabulary, from every rank's gathered pairs."""

    keys, ids = merged_keys(
        cand_ptr, row_id, ranks, stride_mr, stride_mb, enable_pdl, max_k
    )

    _, slot = unpack_key(tl.max(keys, axis=0))

    return tl.sum(tl.where(tl.arange(0, ranks * max_k) == slot, ids, 0), axis=0)


@jit
def merged_whole(
    cand_ptr,
    row_id,
    ranks: tl.constexpr,
    stride_mr,
    stride_mb: tl.constexpr,
    enable_pdl: tl.constexpr,
    max_k: tl.constexpr,
):
    """
    A whole-vocabulary row over every rank's gathered pairs: the best
    perturbed value and its token id (the lowest rank's among equal values),
    then for a verify's draft row its maximum value, its exp-sum below that
    maximum, and its draft's value and perturbed value.
    """
    minima: tl.constexpr = -float("inf")

    span = tl.arange(0, ranks)
    pairs = cand_ptr + span * stride_mr + row_id * stride_mb

    if enable_pdl:

        gdc_wait()

    scores = tl.load(pairs)
    ids = tl.load(pairs + max_k)
    maxima = tl.load(pairs + 1)
    totals = tl.load(pairs + max_k + 1)

    score, rank = tl.max(scores, axis=0, return_indices=True)
    token = tl.sum(tl.where(span == rank, ids, 0.0), axis=0).to(tl.int32)

    maximum = tl.max(maxima, axis=0)
    total = tl.sum(
        tl.where(maxima > minima, fast_expf(maxima - maximum) * totals, 0.0), axis=0
    )

    return (
        score,
        token,
        maximum,
        total,
        tl.max(tl.load(pairs + 2), axis=0),
        tl.max(tl.load(pairs + max_k + 2), axis=0),
    )
