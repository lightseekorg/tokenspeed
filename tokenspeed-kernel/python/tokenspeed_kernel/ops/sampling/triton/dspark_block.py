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

"""Greedy DSpark block sampling over a vocabulary-sharded head.

Each block step adds the Markov bigram bias of the previous token to that
step's base logits and takes the argmax over the whole vocabulary. The head is
sharded over tensor-parallel ranks, so a step is one kernel per rank plus one
all-gather: the kernel scores this rank's vocabulary tiles and emits one packed
``(logit, token)`` candidate per tile; every rank gathers every rank's
candidates, and the next step's kernel (or the final resolve) reduces them to
the winning token. Packing the ordered logit bits above the negated token id
makes a plain integer max pick the largest logit and, on ties, the lowest
token id -- ``torch.argmax`` semantics over the concatenated shards.

The bias is a tensor-core dot of the previous token's BF16 bigram row against
the BF16 projection shard: BF16 products are exact in FP32, so this matches
the reference's FP32 GEMM over BF16-valued weights up to accumulation order.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = [
    "dspark_block_candidate_tiles",
    "dspark_block_greedy_resolve",
    "dspark_block_greedy_step",
]

_INT64_MIN = tl.constexpr(-(1 << 63))
_LOW32 = tl.constexpr(0xFFFFFFFF)
# Rows per program, also the M of the bias dot; the vocabulary entries one
# program scores into one candidate; the N of one dot inside that tile.
_ROWS = 16
_BLOCK_V = 256
_SUB_V = 128


def dspark_block_candidate_tiles(local_vocab: int) -> int:
    """Return how many candidates one rank emits per row for ``local_vocab``."""
    if local_vocab <= 0:
        raise ValueError(f"local_vocab must be positive, got {local_vocab}")
    return triton.cdiv(local_vocab, _BLOCK_V)


@triton.jit
def _orderable_key(values):
    """Map FP32 to int32 so integer order equals floating-point order."""
    bits = values.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _resolve_rows(
    candidates_ptr,
    rows,
    row_offsets,
    row_mask,
    TP: tl.constexpr,
    TP_PAD: tl.constexpr,
    N_TILES: tl.constexpr,
    N_TILES_PAD: tl.constexpr,
):
    """Reduce ``[TP, rows, N_TILES]`` packed candidates to token ids."""
    tp_offsets = tl.arange(0, TP_PAD)
    tile_offsets = tl.arange(0, N_TILES_PAD)
    offsets = (
        tp_offsets[:, None, None].to(tl.int64) * rows + row_offsets[None, :, None]
    ) * N_TILES + tile_offsets[None, None, :]
    mask = (
        (tp_offsets < TP)[:, None, None]
        & row_mask[None, :, None]
        & (tile_offsets < N_TILES)[None, None, :]
    )
    packed = tl.load(candidates_ptr + offsets, mask=mask, other=_INT64_MIN)
    best = tl.max(tl.max(packed, axis=2), axis=0)
    return (_LOW32 - (best & _LOW32)).to(tl.int32)


@triton.jit
def _block_step_kernel(
    base_ptr,
    base_row_stride,
    anchor_ptr,
    anchor_stride,
    candidates_ptr,
    previous_out_ptr,
    out_row_stride,
    embedding_ptr,
    projection_ptr,
    partials_ptr,
    rows,
    num_valid,
    vocab_start,
    vocab_size,
    RANK: tl.constexpr,
    RANK_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SUB_V: tl.constexpr,
    TP: tl.constexpr,
    TP_PAD: tl.constexpr,
    N_TILES: tl.constexpr,
    N_TILES_PAD: tl.constexpr,
    RESOLVE_PREVIOUS: tl.constexpr,
):
    tile = tl.program_id(0)
    row_offsets = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    row_mask = row_offsets < rows
    if RESOLVE_PREVIOUS:
        previous = _resolve_rows(
            candidates_ptr,
            rows,
            row_offsets,
            row_mask,
            TP,
            TP_PAD,
            N_TILES,
            N_TILES_PAD,
        )
        if tile == 0:
            tl.store(
                previous_out_ptr + row_offsets * out_row_stride, previous, mask=row_mask
            )
    else:
        previous = tl.load(
            anchor_ptr + row_offsets * anchor_stride, mask=row_mask, other=0
        ).to(tl.int32)
    # Graph padding rows carry arbitrary anchors; keep their gathers in bounds.
    previous = tl.minimum(tl.maximum(previous, 0), vocab_size - 1)

    rank_offsets = tl.arange(0, RANK_PAD)
    rank_mask = rank_offsets < RANK
    bigram = tl.load(
        embedding_ptr + previous.to(tl.int64)[:, None] * RANK + rank_offsets[None, :],
        mask=row_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    best = tl.full((ROWS,), _INT64_MIN, tl.int64)
    for sub in tl.static_range(0, BLOCK_V, SUB_V):
        vocab_offsets = tile * BLOCK_V + sub + tl.arange(0, SUB_V)
        vocab_mask = vocab_offsets < num_valid
        weight = tl.load(
            projection_ptr
            + vocab_offsets.to(tl.int64)[:, None] * RANK
            + rank_offsets[None, :],
            mask=vocab_mask[:, None] & rank_mask[None, :],
            other=0.0,
        )
        bias = tl.dot(bigram, tl.trans(weight))
        base = tl.load(
            base_ptr + row_offsets[:, None] * base_row_stride + vocab_offsets[None, :],
            mask=row_mask[:, None] & vocab_mask[None, :],
            other=float("-inf"),
        )
        token_rank = _LOW32 - (vocab_start + vocab_offsets).to(tl.int64)
        key = (_orderable_key(base + bias).to(tl.int64) << 32) | token_rank[None, :]
        key = tl.where(vocab_mask[None, :], key, _INT64_MIN)
        best = tl.maximum(best, tl.max(key, axis=1))
    tl.store(partials_ptr + row_offsets * N_TILES + tile, best, mask=row_mask)


@triton.jit
def _block_resolve_kernel(
    candidates_ptr,
    out_ptr,
    out_row_stride,
    rows,
    ROWS: tl.constexpr,
    TP: tl.constexpr,
    TP_PAD: tl.constexpr,
    N_TILES: tl.constexpr,
    N_TILES_PAD: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = row_offsets < rows
    tokens = _resolve_rows(
        candidates_ptr,
        rows,
        row_offsets,
        row_mask,
        TP,
        TP_PAD,
        N_TILES,
        N_TILES_PAD,
    )
    tl.store(out_ptr + row_offsets * out_row_stride, tokens, mask=row_mask)


def _check_candidates(candidates: torch.Tensor, rows: int, n_tiles: int) -> int:
    if candidates.ndim != 3 or candidates.shape[1:] != (rows, n_tiles):
        raise ValueError(
            f"candidates shape {tuple(candidates.shape)} must be "
            f"[tp, {rows}, {n_tiles}]"
        )
    if candidates.dtype != torch.int64 or not candidates.is_contiguous():
        raise ValueError("candidates must be a contiguous int64 tensor")
    return candidates.shape[0]


def _check_output_column(output: torch.Tensor, rows: int, block: int) -> None:
    if output.shape != (rows, block):
        raise ValueError(f"output shape {tuple(output.shape)} must be {(rows, block)}")
    if output.dtype != torch.int32 or output.stride(1) != 1:
        raise ValueError("output must be int32 with contiguous rows")


def dspark_block_greedy_step(
    base_logits: torch.Tensor,
    step: int,
    anchor_ids: torch.Tensor,
    candidates: torch.Tensor,
    embedding: torch.Tensor,
    projection: torch.Tensor,
    vocab_start: int,
    num_valid: int,
    partials: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Score this rank's vocabulary shard for one block step.

    Args:
        base_logits: ``[rows, block, local_vocab]`` FP32 base logits of this
            rank's shard; column ``v`` is token ``vocab_start + v``.
        step: Block step in ``[0, block)`` being scored.
        anchor_ids: ``[rows]`` int32/int64 tokens preceding step 0, in any
            stride (the drafters pass a column of their token table); read
            only when ``step == 0``.
        candidates: ``[tp, rows, n_tiles]`` int64 candidates gathered after
            the previous step; read only when ``step > 0``, and then disjoint
            from ``partials``.
        embedding: ``[vocab, rank]`` BF16 replicated bigram table.
        projection: ``[padded_local_vocab, rank]`` BF16 projection shard whose
            row ``v`` belongs to token ``vocab_start + v``.
        vocab_start: First token id of this rank's shard.
        num_valid: Shard columns that are real tokens; the rest are padding.
        partials: ``[rows, n_tiles]`` int64 destination for this rank's packed
            candidates.
        output: ``[rows, block]`` int32 block tokens; column ``step - 1``
            receives the previous step's winner when ``step > 0``.

    Returns:
        None. ``partials`` and, for ``step > 0``, one ``output`` column are
        written in place.

    Raises:
        ValueError: A tensor is not the shape, dtype or layout the kernel
            indexes with, or ``step`` is outside the block.
    """
    if base_logits.ndim != 3 or base_logits.dtype != torch.float32:
        raise ValueError("base_logits must be a 3D FP32 tensor")
    if base_logits.stride(2) != 1:
        raise ValueError("base_logits vocabulary axis must be contiguous")
    rows, block, local_vocab = base_logits.shape
    if not 0 <= step < block:
        raise ValueError(f"step {step} is outside the block of {block}")
    if base_logits.stride(1) != local_vocab:
        raise ValueError("base_logits steps must be contiguous")
    if embedding.ndim != 2 or projection.ndim != 2:
        raise ValueError("embedding and projection must be 2D")
    rank = embedding.shape[1]
    if projection.shape[1] != rank:
        raise ValueError(
            f"projection rank {projection.shape[1]} must match embedding rank {rank}"
        )
    if embedding.dtype != torch.bfloat16 or projection.dtype != torch.bfloat16:
        raise ValueError("embedding and projection must be BF16")
    if not (embedding.is_contiguous() and projection.is_contiguous()):
        raise ValueError("embedding and projection must be contiguous")
    if not 0 < num_valid <= local_vocab or num_valid > projection.shape[0]:
        raise ValueError(
            f"num_valid {num_valid} must lie in (0, {local_vocab}] and fit the "
            f"projection shard of {projection.shape[0]} rows"
        )
    if anchor_ids.shape != (rows,) or anchor_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"anchor_ids must be [{rows}] int32/int64")
    n_tiles = dspark_block_candidate_tiles(local_vocab)
    tp = _check_candidates(candidates, rows, n_tiles)
    if partials.shape != (rows, n_tiles) or partials.dtype != torch.int64:
        raise ValueError(f"partials must be [{rows}, {n_tiles}] int64")
    if not partials.is_contiguous():
        raise ValueError("partials must be contiguous")
    # Programs read every previous candidate while others already write their
    # tile; the two buffers must not share memory.
    if step > 0 and (
        candidates.data_ptr() < partials.data_ptr() + partials.numel() * 8
        and partials.data_ptr() < candidates.data_ptr() + candidates.numel() * 8
    ):
        raise ValueError("candidates and partials must not overlap")
    _check_output_column(output, rows, block)
    if rows == 0:
        return

    previous_out = output[:, step - 1] if step > 0 else output[:, 0]
    grid = (n_tiles, triton.cdiv(rows, _ROWS))
    _block_step_kernel[grid](
        base_logits[:, step],
        base_logits.stride(0),
        anchor_ids,
        anchor_ids.stride(0),
        candidates,
        previous_out,
        output.stride(0),
        embedding,
        projection,
        partials,
        rows,
        num_valid,
        vocab_start,
        embedding.shape[0],
        RANK=rank,
        RANK_PAD=max(16, triton.next_power_of_2(rank)),
        ROWS=_ROWS,
        BLOCK_V=_BLOCK_V,
        SUB_V=_SUB_V,
        TP=tp,
        TP_PAD=triton.next_power_of_2(tp),
        N_TILES=n_tiles,
        N_TILES_PAD=triton.next_power_of_2(n_tiles),
        RESOLVE_PREVIOUS=step > 0,
        num_warps=4,
    )


def dspark_block_greedy_resolve(
    candidates: torch.Tensor,
    output: torch.Tensor,
    step: int,
) -> None:
    """Write the winners of the gathered ``candidates`` into ``output[:, step]``.

    Args:
        candidates: ``[tp, rows, n_tiles]`` int64 candidates gathered after
            step ``step``.
        output: ``[rows, block]`` int32 block tokens.
        step: Column of ``output`` to fill, normally the last block step.

    Returns:
        None. One ``output`` column is written in place.

    Raises:
        ValueError: A tensor is not the shape, dtype or layout the kernel
            indexes with, or ``step`` is outside the block.
    """
    if candidates.ndim != 3:
        raise ValueError("candidates must be [tp, rows, n_tiles]")
    _, rows, n_tiles = candidates.shape
    tp = _check_candidates(candidates, rows, n_tiles)
    if output.ndim != 2 or not 0 <= step < output.shape[1]:
        raise ValueError(f"step {step} is outside the output block")
    _check_output_column(output, rows, output.shape[1])
    if rows == 0:
        return
    _block_resolve_kernel[(triton.cdiv(rows, _ROWS),)](
        candidates,
        output[:, step],
        output.stride(0),
        rows,
        ROWS=_ROWS,
        TP=tp,
        TP_PAD=triton.next_power_of_2(tp),
        N_TILES=n_tiles,
        N_TILES_PAD=triton.next_power_of_2(n_tiles),
        num_warps=4,
    )
