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

"""The inputs of an MTP / EAGLE chain draft in one launch.

After a target verify of decode requests, a chain drafter needs, for every
request: the last verified token (the first column of its draft window, and
the seed of every other column), the row of the verify output it continues
from, its accepted frontier (the cache length plus the accepted tokens), and
the first multi-step position and length. As separate tensor operations
these are about ten small kernels in every decode graph.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["eagle_chain_prologue", "eagle_chain_step"]

_BLOCK = 128


@triton.jit(do_not_specialize=["batch_size"])
def _eagle_chain_prologue_kernel(
    tokens_ptr,
    accept_lengths_ptr,
    gather_offsets_ptr,
    valid_cache_lengths_ptr,
    req_pool_indices_ptr,
    next_tokens_ptr,
    next_tokens_stride,
    gather_ids_ptr,
    frontier_ptr,
    seq_lens_ptr,
    positions_ptr,
    batch_size,
    WINDOW: tl.constexpr,
    WINDOW_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = rows < batch_size
    accepted = tl.load(accept_lengths_ptr + rows, mask=live, other=0)
    gather = tl.load(gather_offsets_ptr + rows, mask=live, other=0) + accepted
    tl.store(gather_ids_ptr + rows, gather, mask=live)
    token = tl.load(tokens_ptr + gather, mask=live, other=0).to(tl.int32)
    cols = tl.arange(0, WINDOW_P)
    tl.store(
        next_tokens_ptr + rows[:, None] * next_tokens_stride + cols[None, :],
        tl.broadcast_to(token[:, None], [BLOCK, WINDOW_P]),
        mask=live[:, None] & (cols < WINDOW)[None, :],
    )
    slot = tl.load(req_pool_indices_ptr + rows, mask=live, other=0)
    frontier = tl.load(
        valid_cache_lengths_ptr + slot, mask=live, other=0
    ) + accepted.to(tl.int32)
    tl.store(frontier_ptr + rows, frontier, mask=live)
    tl.store(seq_lens_ptr + rows, frontier + 1, mask=live)
    tl.store(positions_ptr + rows, frontier.to(tl.int64), mask=live)


def eagle_chain_prologue(
    tokens: torch.Tensor,
    accept_lengths: torch.Tensor,
    gather_offsets: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    *,
    next_tokens: torch.Tensor,
    gather_ids: torch.Tensor,
    frontier: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Every decode request's chain-draft inputs after a target verify.

    Args:
        tokens: The verify's output tokens, ``[batch * window]`` (any integer
            type).
        accept_lengths: ``[batch]`` int32 accepted tokens of each request.
        gather_offsets: ``[batch]`` int64 ``request * window - 1``.
        valid_cache_lengths: ``[pool]`` int32 cache length of each slot
            before this round's commit.
        req_pool_indices: ``[batch]`` int64 slot of each request.
        next_tokens: ``[batch, window]`` int32 output: every column is the
            request's last verified token, ``tokens[gather_ids]``.
        gather_ids: ``[batch]`` int64 output: ``gather_offsets +
            accept_lengths``, the verify row each draft continues from.
        frontier: ``[batch]`` int32 output: ``valid_cache_length +
            accept_length``.
        seq_lens: ``[batch]`` int32 output: ``frontier + 1``, the first
            multi-step length.
        positions: ``[batch]`` int64 output: ``frontier``, the first
            multi-step position (int64, as the models' positions).

    Returns:
        None. The outputs are written in place.
    """
    batch_size = accept_lengths.shape[0]
    if (
        accept_lengths.dtype != torch.int32
        or gather_offsets.dtype != torch.int64
        or valid_cache_lengths.dtype != torch.int32
        or req_pool_indices.dtype != torch.int64
        or next_tokens.dtype != torch.int32
        or gather_ids.dtype != torch.int64
        or any(t.dtype != torch.int32 for t in (frontier, seq_lens))
        or positions.dtype != torch.int64
    ):
        raise ValueError("eagle_chain_prologue: unexpected operand dtypes")
    if next_tokens.shape[0] != batch_size or next_tokens.stride(1) != 1:
        raise ValueError("next_tokens must be [batch, window] with unit columns")
    window = next_tokens.shape[1]
    _eagle_chain_prologue_kernel[(triton.cdiv(batch_size, _BLOCK),)](
        tokens,
        accept_lengths,
        gather_offsets,
        valid_cache_lengths,
        req_pool_indices,
        next_tokens,
        next_tokens.stride(0),
        gather_ids,
        frontier,
        seq_lens,
        positions,
        batch_size,
        WINDOW=window,
        WINDOW_P=triton.next_power_of_2(window),
        BLOCK=_BLOCK,
    )


@triton.jit(do_not_specialize=["column", "batch_size"])
def _eagle_chain_step_kernel(
    ids_ptr,
    next_tokens_ptr,
    next_tokens_stride,
    column,
    positions_ptr,
    seq_lens_ptr,
    batch_size,
    ADVANCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = rows < batch_size
    token = tl.load(ids_ptr + rows, mask=live, other=0).to(tl.int32)
    tl.store(next_tokens_ptr + rows * next_tokens_stride + column, token, mask=live)
    if ADVANCE:
        position = tl.load(positions_ptr + rows, mask=live, other=0)
        tl.store(positions_ptr + rows, position + 1, mask=live)
        length = tl.load(seq_lens_ptr + rows, mask=live, other=0)
        tl.store(seq_lens_ptr + rows, length + 1, mask=live)


def eagle_chain_step(
    ids: torch.Tensor,
    *,
    next_tokens: torch.Tensor,
    column: int,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    advance: bool,
) -> None:
    """The end of a chain draft step, in one launch: the step's tokens go to
    their column of the draft window, and (``advance``, a step that another
    step follows) the positions and lengths move on one token.

    Args:
        ids: ``[batch]`` the step's tokens (any integer type).
        next_tokens: ``[batch, window]`` int32 draft window, unit columns.
        column: The window column of this step's tokens.
        positions: ``[batch]`` int64 positions, incremented in place.
        seq_lens: ``[batch]`` int32 lengths, incremented in place.
        advance: Whether to increment ``positions`` and ``seq_lens``.

    Returns:
        None. The outputs are written in place.
    """
    batch_size = ids.shape[0]
    if (
        next_tokens.dtype != torch.int32
        or next_tokens.stride(1) != 1
        or positions.dtype != torch.int64
        or seq_lens.dtype != torch.int32
    ):
        raise ValueError("eagle_chain_step: unexpected operand dtypes or layout")
    if (
        next_tokens.shape[0] < batch_size
        or not 0 <= column < next_tokens.shape[1]
        or positions.shape[0] < batch_size
        or seq_lens.shape[0] < batch_size
    ):
        raise ValueError("eagle_chain_step: operands smaller than the batch")
    if batch_size == 0:
        return
    _eagle_chain_step_kernel[(triton.cdiv(batch_size, _BLOCK),)](
        ids,
        next_tokens,
        next_tokens.stride(0),
        column,
        positions,
        seq_lens,
        batch_size,
        ADVANCE=advance,
        BLOCK=_BLOCK,
    )
