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

"""Selected-token logprobs of vocab-sharded logits, bit for bit
``selected_token_logprobs`` on the gathered rows.

``selected_token_logprobs`` sums ``exp(x - max)`` over 1024-column blocks
from column 0 and adds the block sums in order. Its shard form keeps that
sum: each rank sums the blocks that start in its shard
(``triton_vocab_shard_logprob_blocks``; a block crossing into the next
shard reads that shard's first columns, which the ranks gathered with the
top-k candidates), and after a gather of the block sums every rank adds
them in block order (``triton_vocab_shard_logprobs``). A block sum is the
reference's own reduction: the same 1024-lane load of rows of the same
stride, so the same layout and reduction tree.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import (
    SELECTED_LOGPROBS_BLOCK,
    _logprob_owned_blocks,
    vocab_shard_logprob_layout,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "triton_vocab_shard_logprob_blocks",
    "triton_vocab_shard_logprobs",
]


@triton.jit(
    do_not_specialize=["vocab_start", "shard_columns", "first_block", "n_blocks"]
)
def _vocab_shard_logprob_blocks_kernel(
    values_ptr,
    rows_ptr,
    tokens_ptr,
    row_max_ptr,
    out_ptr,
    stride_row_max,
    stride_out,
    vocab_start,
    shard_columns,
    first_block,
    n_blocks,
    vocab_size: tl.constexpr,
    logits_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program (i, j): output row i's sum over the j-th block this shard owns,
    # in slot 1 + j; program (i, 0) also writes the selected token's value in
    # slot 0 when this shard holds it.
    i = tl.program_id(0)
    j = tl.program_id(1)
    src = tl.load(rows_ptr + i).to(tl.int64)
    # The reference's row pointer and block load: rows of the same constant
    # stride and blocks at multiples of BLOCK_SIZE give the load, and so the
    # reduction, the same layout.
    row_ptr = values_ptr + src * logits_row_stride
    row_max = tl.load(row_max_ptr + src * stride_row_max)
    out_row = out_ptr + i.to(tl.int64) * stride_out
    if j < n_blocks:
        start = (first_block + j) * BLOCK_SIZE
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        vals = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        weights = tl.exp(vals - row_max)
        tl.store(out_row + 1 + j, tl.sum(tl.where(mask, weights, 0.0), axis=0))
    if j == 0:
        token = tl.load(tokens_ptr + i).to(tl.int64)
        if (token >= vocab_start) & (token < vocab_start + shard_columns):
            tl.store(out_row, tl.load(row_ptr + token).to(tl.float32))


@register_kernel(
    "sampling",
    "vocab_shard_logprob_blocks",
    name="triton_vocab_shard_logprob_blocks",
    solution="triton",
    signatures=format_signatures(
        "values", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def triton_vocab_shard_logprob_blocks(
    values: torch.Tensor,
    rows: torch.Tensor,
    tokens: torch.Tensor,
    row_max: torch.Tensor,
    vocab_start: int,
    shard_columns: int,
    out: torch.Tensor,
) -> None:
    """This shard's part of each output row's selected-token logprob.

    Args:
        values: Contiguous fp32 ``[S, V]`` whole-vocabulary rows holding the
            final (scaled, capped, masked) logits of this shard's columns
            and of the next shard's first ``halo`` columns
            (``vocab_shard_logprob_layout``); other columns are not read.
        rows: ``[M]`` int32 or int64 source row of each output row.
        tokens: ``[M]`` int32 or int64 selected token of each output row.
        row_max: fp32 ``[S]`` maximum of each source row, any stride.
        vocab_start: Global id of this shard's column 0.
        shard_columns: Columns per shard (every shard but the last starts
            ``shard_columns`` after the previous one).
        out: fp32 ``[M, width]``, unit column stride: slot 0 the selected
            token's value when this shard holds it (else untouched), slots
            ``1 ..`` the sums of the blocks starting in this shard.
    """
    sources, vocab_size = values.shape
    if not values.is_contiguous():
        raise ValueError("values must be contiguous [S, V] rows")
    width = vocab_shard_logprob_layout(shard_columns, vocab_size)[1]
    first, end = _logprob_owned_blocks(vocab_start, shard_columns, vocab_size)
    outputs = rows.shape[0]
    if tuple(out.shape) != (outputs, width) or out.stride(1) != 1:
        raise ValueError(f"out must be [{outputs}, {width}] with unit column stride")
    if tokens.shape[0] != outputs:
        raise ValueError("tokens must hold one token per output row")
    if outputs == 0:
        return
    _vocab_shard_logprob_blocks_kernel[(outputs, max(end - first, 1))](
        values,
        rows,
        tokens,
        row_max,
        out,
        row_max.stride(0),
        out.stride(0),
        vocab_start,
        shard_columns,
        first,
        end - first,
        vocab_size=vocab_size,
        logits_row_stride=vocab_size,
        BLOCK_SIZE=SELECTED_LOGPROBS_BLOCK,
        num_warps=4,
    )


@triton.jit(do_not_specialize=["shard_columns", "n_blocks"])
def _vocab_shard_logprobs_kernel(
    partials_ptr,
    rows_ptr,
    tokens_ptr,
    row_max_ptr,
    out_ptr,
    stride_rank,
    stride_partial,
    stride_row_max,
    shard_columns,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    i = tl.program_id(0)
    row = partials_ptr + i.to(tl.int64) * stride_partial
    # Block b sits in the shard holding its first column, after that shard's
    # earlier blocks.
    b = tl.arange(0, BLOCKS)
    owner = (b * BLOCK_SIZE) // shard_columns
    local = b - tl.cdiv(owner * shard_columns, BLOCK_SIZE)
    sums = tl.load(
        row + owner.to(tl.int64) * stride_rank + 1 + local, mask=b < n_blocks, other=0.0
    )
    # The reference's running sum, block by block: a block's sum taken out
    # of the vector alone is exact (x + 0.0 == x).
    denom = tl.full((), 0.0, tl.float32)
    for block in range(0, n_blocks):
        denom += tl.sum(tl.where(b == block, sums, 0.0), axis=0)
    token = tl.load(tokens_ptr + i).to(tl.int64)
    selected = tl.load(row + (token // shard_columns) * stride_rank)
    src = tl.load(rows_ptr + i).to(tl.int64)
    row_max = tl.load(row_max_ptr + src * stride_row_max)
    tl.store(out_ptr + i, selected - row_max - tl.log(tl.maximum(denom, 1.0e-20)))


@register_kernel(
    "sampling",
    "vocab_shard_logprobs",
    name="triton_vocab_shard_logprobs",
    solution="triton",
    signatures=format_signatures("partials", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def triton_vocab_shard_logprobs(
    partials: torch.Tensor,
    rows: torch.Tensor,
    tokens: torch.Tensor,
    row_max: torch.Tensor,
    shard_columns: int,
    vocab_size: int,
    out: torch.Tensor,
) -> None:
    """Each output row's selected-token logprob from every shard's parts.

    Args:
        partials: ``[R, M, width]`` fp32 view of the gathered
            ``triton_vocab_shard_logprob_blocks`` outputs, rank-major, any
            rank and row strides, unit column stride.
        rows: ``[M]`` source row of each output row.
        tokens: ``[M]`` selected token of each output row.
        row_max: fp32 ``[S]`` maximum of each source row, any stride.
        shard_columns: Columns per shard.
        vocab_size: Real vocabulary size ``V``.
        out: Contiguous fp32 ``[M]`` logprobs.
    """
    ranks, outputs, width = partials.shape
    if partials.stride(2) != 1:
        raise ValueError("partials must have a unit column stride")
    if vocab_shard_logprob_layout(shard_columns, vocab_size)[1] != width:
        raise ValueError(f"partials width {width} is not the shards' layout")
    if out.shape[0] < outputs or out.dtype != torch.float32:
        raise ValueError(f"out must be fp32 with at least {outputs} rows")
    if outputs == 0:
        return
    n_blocks = triton.cdiv(vocab_size, SELECTED_LOGPROBS_BLOCK)
    _vocab_shard_logprobs_kernel[(outputs,)](
        partials,
        rows,
        tokens,
        row_max,
        out,
        partials.stride(0),
        partials.stride(1),
        row_max.stride(0),
        shard_columns,
        n_blocks,
        BLOCK_SIZE=SELECTED_LOGPROBS_BLOCK,
        # The vocabulary is fixed per server, so this compiles once.
        BLOCKS=triton.next_power_of_2(n_blocks),
        num_warps=1,
    )
