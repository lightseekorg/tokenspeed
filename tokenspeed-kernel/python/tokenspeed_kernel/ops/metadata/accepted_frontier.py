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
request's valid cache. On CUDA (and ROCm) devices one Triton launch does it;
other devices run the same arithmetic as tensor ops. Models that keep an
n-gram history commit it with the cache length in ``ngram.commit_ngram_inputs``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["advance_accepted_frontier"]

_BLOCK = 128


@triton.jit(do_not_specialize=["batch_size", "num_extends"])
def _advance_accepted_frontier_kernel(
    req_pool_indices_ptr,
    input_lengths_ptr,
    accept_lengths_ptr,
    valid_cache_lengths_ptr,
    batch_size,
    num_extends,
    padding_index,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    valid = tl.load(valid_cache_lengths_ptr + slots, mask=live, other=0)
    tl.store(valid_cache_lengths_ptr + slots, valid + deltas, mask=live)


def _advance_accepted_frontier_torch(
    req_pool_indices: torch.Tensor,
    input_lengths: torch.Tensor,
    accept_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    num_extends: int,
    padding_index: int,
) -> None:
    deltas = torch.cat([input_lengths[:num_extends], accept_lengths[num_extends:]])
    deltas = torch.where(req_pool_indices != padding_index, deltas, 0).to(torch.int32)
    valid_cache_lengths.index_add_(0, req_pool_indices, deltas)


def advance_accepted_frontier(
    req_pool_indices: torch.Tensor,
    input_lengths: torch.Tensor,
    accept_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    num_extends: int,
    padding_index: int,
) -> None:
    """Add each request's accepted tokens to its cache length.

    Models with n-gram history commit it together with the cache length
    through ``ngram.commit_ngram_inputs`` instead.

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

    Returns:
        None. ``valid_cache_lengths`` is updated in place.

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
        )
        return

    _advance_accepted_frontier_kernel[(triton.cdiv(batch_size, _BLOCK),)](
        req_pool_indices,
        input_lengths,
        accept_lengths,
        valid_cache_lengths,
        batch_size,
        num_extends,
        padding_index,
        BLOCK=_BLOCK,
        num_warps=4,
    )
