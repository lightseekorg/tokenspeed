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

"""Advance every request's accepted frontier after a forward in one launch.

A step accepts ``input_lengths[i]`` tokens of an extend request and
``accept_lengths[i]`` of a decode request. That many tokens join the
request's valid cache, and, when the model keeps an n-gram history, the
request's history tail becomes the last accepted input row's token followed
by that row's own history. Recorded eagerly this is a cumsum, a handful of
gathers and two scatters -- about twenty launches on the critical path
between the forward and the results copy. On CUDA (and ROCm) devices one
Triton launch does it; other devices run the same arithmetic as tensor ops.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["advance_accepted_frontier"]

_BLOCK = 1024


@triton.jit
def _advance_accepted_frontier_kernel(
    req_pool_indices_ptr,
    input_lengths_ptr,
    accept_lengths_ptr,
    valid_cache_lengths_ptr,
    tail_ptr,
    previous_tokens_ptr,
    token_mask_ptr,
    input_ids_ptr,
    batch_size,
    num_extends,
    padding_index,
    max_num_tokens,
    BLOCK: tl.constexpr,
    CONTEXT: tl.constexpr,
    CONTEXT_PAD: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    first = tl.program_id(0) * BLOCK
    rows = first + tl.arange(0, BLOCK)
    row_mask = rows < batch_size
    slots = tl.load(req_pool_indices_ptr + rows, mask=row_mask, other=padding_index)
    input_lengths = tl.load(input_lengths_ptr + rows, mask=row_mask, other=0).to(
        tl.int32
    )
    accept_lengths = tl.load(accept_lengths_ptr + rows, mask=row_mask, other=0).to(
        tl.int32
    )
    live = row_mask & (slots != padding_index)
    deltas = tl.where(rows < num_extends, input_lengths, accept_lengths)
    deltas = tl.where(live, deltas, 0)

    if HAS_TAIL:
        # Row start = inputs of every earlier request: the full blocks before
        # this one, then the exclusive scan inside it.
        base = tl.zeros((), tl.int32)
        for chunk in range(0, first, BLOCK):
            base += tl.sum(
                tl.load(input_lengths_ptr + chunk + tl.arange(0, BLOCK)).to(tl.int32),
                axis=0,
            )
        starts = base + tl.cumsum(input_lengths, axis=0) - input_lengths
        # Accepted inputs end at row start + delta - 1, not at the sampled
        # bonus or the end of the proposed window.
        last_rows = tl.minimum(
            tl.maximum(starts + deltas - 1, 0), max_num_tokens - 1
        ).to(tl.int64)
        write = live & (deltas > 0)
        current_mask = tl.load(token_mask_ptr + last_rows, mask=write, other=0)
        current = tl.load(input_ids_ptr + last_rows, mask=write, other=0).to(tl.int64)
        current = tl.where(current_mask != 0, current, -1)
        columns = tl.arange(0, CONTEXT_PAD)
        column_mask = columns < CONTEXT
        shifted = tl.load(
            previous_tokens_ptr + last_rows[:, None] * CONTEXT + (columns[None, :] - 1),
            mask=write[:, None] & (columns[None, :] > 0) & column_mask[None, :],
            other=-1,
        )
        history = tl.where(columns[None, :] == 0, current[:, None], shifted)
        tl.store(
            tail_ptr + slots[:, None] * CONTEXT + columns[None, :],
            history,
            mask=write[:, None] & column_mask[None, :],
        )

    valid = tl.load(valid_cache_lengths_ptr + slots, mask=live, other=0)
    tl.store(valid_cache_lengths_ptr + slots, valid + deltas, mask=live)


def _advance_accepted_frontier_torch(
    req_pool_indices: torch.Tensor,
    input_lengths: torch.Tensor,
    accept_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    num_extends: int,
    padding_index: int,
    ngram_tail: torch.Tensor | None,
    ngram_previous_tokens: torch.Tensor | None,
    ngram_token_mask: torch.Tensor | None,
    input_ids: torch.Tensor | None,
) -> None:
    deltas = torch.cat([input_lengths[:num_extends], accept_lengths[num_extends:]])
    deltas = torch.where(req_pool_indices != padding_index, deltas, 0).to(torch.int32)
    if ngram_tail is not None:
        assert ngram_previous_tokens is not None
        assert ngram_token_mask is not None
        assert input_ids is not None
        last_rows = (input_lengths.cumsum(0) - input_lengths + deltas - 1).clamp(
            0, input_ids.shape[0] - 1
        )
        current = torch.where(
            ngram_token_mask[last_rows], input_ids[last_rows].to(torch.int64), -1
        )
        history = torch.cat(
            [current[:, None], ngram_previous_tokens[last_rows, :-1]], dim=1
        )
        ngram_tail[req_pool_indices] = torch.where(
            (deltas > 0)[:, None], history, ngram_tail[req_pool_indices]
        )
    valid_cache_lengths.index_add_(0, req_pool_indices, deltas)


def advance_accepted_frontier(
    req_pool_indices: torch.Tensor,
    input_lengths: torch.Tensor,
    accept_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    num_extends: int,
    padding_index: int,
    *,
    ngram_tail: torch.Tensor | None,
    ngram_previous_tokens: torch.Tensor | None,
    ngram_token_mask: torch.Tensor | None,
    input_ids: torch.Tensor | None,
) -> None:
    """Add each request's accepted tokens to its cache length and history.

    Args:
        req_pool_indices: ``[batch]`` int64 state slot of every request;
            ``padding_index`` marks graph padding rows, which change nothing.
            Live slots must be distinct.
        input_lengths: ``[batch]`` integer input rows of every request.
        accept_lengths: ``[batch]`` integer accepted tokens of every decode
            request; extend requests accept all of their input rows.
        valid_cache_lengths: ``[pool]`` int32 cache lengths, advanced in place.
        num_extends: Leading requests that are extends.
        padding_index: Slot value of a padding row.
        ngram_tail: ``[pool, context]`` int64 per-slot n-gram history, or None
            when the model keeps none; then the other n-gram arguments must be
            None as well.
        ngram_previous_tokens: ``[tokens, context]`` int64 history of every
            input row of this forward.
        ngram_token_mask: ``[tokens]`` bool marking input rows whose token is
            in vocabulary. ``input_ids`` was clamped for the embedding lookup,
            so the mask, not the id, carries an out-of-vocabulary barrier.
        input_ids: ``[tokens]`` int32 input tokens of this forward.

    Returns:
        None. ``valid_cache_lengths`` and ``ngram_tail`` are updated in place.

    Raises:
        ValueError: A tensor is not the shape, dtype or layout the update
            indexes with.
    """
    batch_size = req_pool_indices.shape[0]
    if req_pool_indices.ndim != 1 or req_pool_indices.dtype != torch.int64:
        raise ValueError("req_pool_indices must be a 1D int64 tensor")
    for name, lengths in (
        ("input_lengths", input_lengths),
        ("accept_lengths", accept_lengths),
    ):
        if lengths.shape != (batch_size,) or lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{name} must be [{batch_size}] int32/int64")
    if valid_cache_lengths.ndim != 1 or valid_cache_lengths.dtype != torch.int32:
        raise ValueError("valid_cache_lengths must be a 1D int32 tensor")
    if not 0 <= num_extends <= batch_size:
        raise ValueError(f"num_extends {num_extends} exceeds the batch of {batch_size}")
    tensors = [req_pool_indices, input_lengths, accept_lengths, valid_cache_lengths]

    given = [
        t is not None
        for t in (ngram_tail, ngram_previous_tokens, ngram_token_mask, input_ids)
    ]
    if any(given) and not all(given):
        raise ValueError("n-gram tail arguments must all be given or all be None")
    has_tail = ngram_tail is not None
    context = 0
    max_num_tokens = 0
    if has_tail:
        assert ngram_previous_tokens is not None
        assert ngram_token_mask is not None
        assert input_ids is not None
        if ngram_tail.ndim != 2 or ngram_tail.dtype != torch.int64:
            raise ValueError("ngram_tail must be a 2D int64 tensor")
        context = ngram_tail.shape[1]
        max_num_tokens = input_ids.shape[0]
        if ngram_tail.shape[0] != valid_cache_lengths.shape[0]:
            raise ValueError("ngram_tail and valid_cache_lengths cover different pools")
        if ngram_previous_tokens.shape != (max_num_tokens, context) or (
            ngram_previous_tokens.dtype != torch.int64
        ):
            raise ValueError(
                f"ngram_previous_tokens must be [{max_num_tokens}, {context}] int64"
            )
        if ngram_token_mask.shape != (max_num_tokens,) or (
            ngram_token_mask.dtype != torch.bool
        ):
            raise ValueError(f"ngram_token_mask must be [{max_num_tokens}] bool")
        if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a 1D int32/int64 tensor")
        if context == 0:
            raise ValueError("ngram_tail must keep at least one token")
        tensors += [ngram_tail, ngram_previous_tokens, ngram_token_mask, input_ids]
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("advance_accepted_frontier needs contiguous tensors")
    if any(t.device != req_pool_indices.device for t in tensors):
        raise ValueError("advance_accepted_frontier needs colocated tensors")
    if batch_size == 0:
        return
    if not req_pool_indices.is_cuda:
        _advance_accepted_frontier_torch(
            req_pool_indices,
            input_lengths,
            accept_lengths,
            valid_cache_lengths,
            num_extends,
            padding_index,
            ngram_tail,
            ngram_previous_tokens,
            ngram_token_mask,
            input_ids,
        )
        return

    block = min(_BLOCK, triton.next_power_of_2(batch_size))
    _advance_accepted_frontier_kernel[(triton.cdiv(batch_size, block),)](
        req_pool_indices,
        input_lengths,
        accept_lengths,
        valid_cache_lengths,
        ngram_tail if has_tail else valid_cache_lengths,
        ngram_previous_tokens if has_tail else valid_cache_lengths,
        ngram_token_mask if has_tail else valid_cache_lengths,
        input_ids if has_tail else valid_cache_lengths,
        batch_size,
        num_extends,
        padding_index,
        max_num_tokens,
        BLOCK=block,
        CONTEXT=context,
        CONTEXT_PAD=max(2, triton.next_power_of_2(context)),
        HAS_TAIL=has_tail,
        num_warps=4,
    )
