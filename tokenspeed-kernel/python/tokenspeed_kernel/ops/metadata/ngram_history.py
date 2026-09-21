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

"""Assemble every input row's n-gram history for one forward in two launches.

An n-gram model reads, for each input token, the ``context`` tokens before it.
For token ``j`` of a request at local row ``l`` and distance ``d``, that is
the request's own input ``j - d`` when ``l >= d`` and otherwise entry
``d - l - 1`` of the request's accepted prefix -- the last ``context`` tokens
committed before this forward. The prefix itself is either carried over from
the previous forward (the per-slot tail) or, when a request is (re)seeded,
taken from the host snapshot of its history, aligned by how far the accepted
frontier has moved past the snapshot position.

The seed kernel settles every request's prefix into the tail; the history
kernel then reads the tails and the packed input ids to write the per-row
history and validity mask, and blanks the rows past the batch so graph
padding rows see no history. Recorded eagerly this was about fifty launches.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["fill_ngram_history"]

_MAX_BATCH = 4096


@triton.jit
def _ngram_seed_kernel(
    tokens_ptr,
    positions_ptr,
    reset_ptr,
    slots_ptr,
    valid_cache_lengths_ptr,
    tail_ptr,
    needs_seed_ptr,
    batch_size,
    vocab_size,
    BLOCK: tl.constexpr,
    CONTEXT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    row_mask = rows < batch_size
    slots = tl.load(slots_ptr + rows, mask=row_mask, other=0)
    seed = (tl.load(reset_ptr + rows, mask=row_mask, other=0) != 0) | (
        tl.load(needs_seed_ptr + slots, mask=row_mask, other=0) != 0
    )
    positions = tl.load(positions_ptr + rows, mask=row_mask, other=0)
    valid = tl.load(valid_cache_lengths_ptr + slots, mask=row_mask, other=0).to(
        tl.int64
    )
    # Only lifecycle seeds need host coverage; a steady-state decode may
    # trail the snapshot by more than the whole window and never reads it.
    delta = valid - positions
    tl.device_assert(
        (~row_mask) | (~seed) | ((delta >= 0) & (delta <= 1)),
        "Engram seed snapshot does not cover the accepted input frontier",
    )
    for column in tl.static_range(CONTEXT):
        distance = column + 1
        snapshot_column = tl.minimum(tl.maximum(distance - delta, 0), CONTEXT)
        seeded = tl.load(
            tokens_ptr + rows * (CONTEXT + 1) + snapshot_column, mask=row_mask, other=-1
        )
        carried = tl.load(tail_ptr + slots * CONTEXT + column, mask=row_mask, other=-1)
        prefix = tl.where(seed, seeded, carried)
        prefix = tl.where((prefix < 0) | (prefix >= vocab_size), -1, prefix)
        tl.store(tail_ptr + slots * CONTEXT + column, prefix, mask=row_mask)
    tl.store(needs_seed_ptr + slots, tl.zeros((BLOCK,), tl.int8), mask=row_mask)


@triton.jit
def _ngram_history_kernel(
    input_ids_ptr,
    input_lengths_ptr,
    slots_ptr,
    tail_ptr,
    previous_ptr,
    mask_ptr,
    batch_size,
    total_tokens,
    capacity,
    vocab_size,
    BLOCK_T: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    CONTEXT: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    in_buffer = rows < capacity
    live = rows < total_tokens
    requests = tl.arange(0, BLOCK_BS)
    lengths = tl.load(
        input_lengths_ptr + requests, mask=requests < batch_size, other=0
    ).to(tl.int64)
    ends = tl.cumsum(lengths, axis=0)
    # A request lies entirely before row j when its end is <= j; counting
    # them is the row's request index and summing them is its start row.
    before = ends[None, :] <= rows[:, None]
    request = tl.sum(before.to(tl.int32), axis=1)
    start = tl.sum(tl.where(before, lengths[None, :], 0), axis=1)
    local = rows - start
    request_ok = live & (request < batch_size)
    slots = tl.load(slots_ptr + request, mask=request_ok, other=0)
    ids = tl.load(input_ids_ptr + rows, mask=live, other=-1).to(tl.int64)
    for column in tl.static_range(CONTEXT):
        distance = column + 1
        from_input = local >= distance
        own = tl.load(
            input_ids_ptr + rows - distance, mask=live & from_input, other=-1
        ).to(tl.int64)
        prefix_column = tl.minimum(tl.maximum(distance - local - 1, 0), CONTEXT - 1)
        carried = tl.load(
            tail_ptr + slots * CONTEXT + prefix_column,
            mask=request_ok & (~from_input),
            other=-1,
        )
        previous = tl.where(from_input, own, carried)
        previous = tl.where((previous < 0) | (previous >= vocab_size), -1, previous)
        tl.store(
            previous_ptr + rows * CONTEXT + column,
            tl.where(live, previous, -1),
            mask=in_buffer,
        )
    valid = live & (ids >= 0) & (ids < vocab_size)
    tl.store(mask_ptr + rows, valid.to(tl.int8), mask=in_buffer)


def fill_ngram_history(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    reset: torch.Tensor,
    slots: torch.Tensor,
    input_lengths: torch.Tensor,
    input_ids: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    tail: torch.Tensor,
    needs_seed: torch.Tensor,
    previous_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    total_tokens: int,
    vocab_size: int,
) -> None:
    """Seed the per-slot prefixes and write every input row's history.

    Args:
        tokens: ``[batch, context + 1]`` int64 host snapshot per request: the
            token at ``positions[i]`` followed by its predecessors, newest
            first; ``-1`` marks tokens that do not exist.
        positions: ``[batch]`` int64 position each snapshot row describes.
        reset: ``[batch]`` int64 flags forcing a reseed from the snapshot.
        slots: ``[batch]`` int64 state slot of every request; distinct.
        input_lengths: ``[batch]`` integer input rows of every request, in
            packed order.
        input_ids: ``[capacity]`` integer packed input tokens; rows past
            ``total_tokens`` are ignored.
        valid_cache_lengths: ``[pool]`` integer accepted length per slot.
        tail: ``[pool, context]`` int64 accepted prefix per slot, newest
            first; updated in place for every request of the batch.
        needs_seed: ``[pool]`` bool slots whose tail must be reseeded; cleared
            in place for every request of the batch.
        previous_tokens: ``[capacity, context]`` int64 destination; row ``j``
            receives the tokens at distances ``1..context`` before input
            ``j``, ``-1`` where none exists, and rows past ``total_tokens``
            are blanked to ``-1``.
        token_mask: ``[capacity]`` bool destination, true where the input
            token is in vocabulary; rows past ``total_tokens`` are cleared.
        total_tokens: Packed input rows of this forward.
        vocab_size: Tokens ``>= vocab_size`` or ``< 0`` are barriers.

    Returns:
        None. ``tail``, ``needs_seed``, ``previous_tokens`` and ``token_mask``
        are written in place. A seeded request whose accepted frontier is not
        within one token of its snapshot position trips a device assertion.

    Raises:
        ValueError: A tensor is not the shape, dtype or layout the kernels
            index with.
    """
    batch_size = slots.shape[0]
    capacity, context = previous_tokens.shape
    if context < 1:
        raise ValueError("n-gram history needs at least one token of context")
    if tokens.shape != (batch_size, context + 1) or tokens.dtype != torch.int64:
        raise ValueError(f"tokens must be [{batch_size}, {context + 1}] int64")
    for name, flags in (("positions", positions), ("reset", reset), ("slots", slots)):
        if flags.shape != (batch_size,) or flags.dtype != torch.int64:
            raise ValueError(f"{name} must be [{batch_size}] int64")
    if input_lengths.shape != (batch_size,) or input_lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"input_lengths must be [{batch_size}] int32/int64")
    if input_ids.shape != (capacity,) or input_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"input_ids must be [{capacity}] int32/int64")
    if valid_cache_lengths.ndim != 1 or valid_cache_lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("valid_cache_lengths must be a 1D integer tensor")
    pool = valid_cache_lengths.shape[0]
    if tail.shape != (pool, context) or tail.dtype != torch.int64:
        raise ValueError(f"tail must be [{pool}, {context}] int64")
    if needs_seed.shape != (pool,) or needs_seed.dtype != torch.bool:
        raise ValueError(f"needs_seed must be [{pool}] bool")
    if previous_tokens.dtype != torch.int64:
        raise ValueError("previous_tokens must be int64")
    if token_mask.shape != (capacity,) or token_mask.dtype != torch.bool:
        raise ValueError(f"token_mask must be [{capacity}] bool")
    if not 0 <= total_tokens <= capacity:
        raise ValueError(f"total_tokens {total_tokens} exceeds capacity {capacity}")
    if batch_size > _MAX_BATCH:
        raise ValueError(f"batch of {batch_size} exceeds {_MAX_BATCH} requests")
    tensors = (
        tokens,
        positions,
        reset,
        slots,
        input_lengths,
        input_ids,
        valid_cache_lengths,
        tail,
        needs_seed,
        previous_tokens,
        token_mask,
    )
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("fill_ngram_history needs contiguous tensors")
    if not all(t.is_cuda for t in tensors):
        raise ValueError("fill_ngram_history needs CUDA tensors")

    if batch_size:
        _ngram_seed_kernel[(1,)](
            tokens,
            positions,
            reset,
            slots,
            valid_cache_lengths,
            tail,
            needs_seed.view(torch.int8),
            batch_size,
            vocab_size,
            BLOCK=triton.next_power_of_2(batch_size),
            CONTEXT=context,
            num_warps=4,
            debug=True,
        )
    if capacity == 0:
        return
    # The row-to-request search is a [BLOCK_T, BLOCK_BS] compare; keep it
    # register-sized for wide batches.
    block_bs = triton.next_power_of_2(max(batch_size, 1))
    block_t = max(16, min(256, 65536 // block_bs))
    _ngram_history_kernel[(triton.cdiv(capacity, block_t),)](
        input_ids,
        input_lengths,
        slots,
        tail,
        previous_tokens,
        token_mask.view(torch.int8),
        batch_size,
        total_tokens,
        capacity,
        vocab_size,
        BLOCK_T=block_t,
        BLOCK_BS=block_bs,
        CONTEXT=context,
        num_warps=4,
    )
