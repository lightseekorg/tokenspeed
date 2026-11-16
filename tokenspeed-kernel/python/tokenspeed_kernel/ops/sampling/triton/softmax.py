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

"""Row-wise ``softmax(logits / temperature)`` in fp32."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Columns per program. Rows are cut into chunks of this width so that a few
# rows of a large vocabulary still spread over many compute units.
_MAX_BLOCK = 4096
_NUM_WARPS = 4


@triton.jit
def _chunk_stats_kernel(
    logits_ptr,
    temperature_ptr,
    chunk_max_ptr,
    chunk_sum_ptr,
    vocab,
    logits_row_stride,
    chunks,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(
        logits_ptr + row * logits_row_stride + cols,
        mask=cols < vocab,
        other=float("-inf"),
    )
    x = x.to(tl.float32) / tl.load(temperature_ptr + row)
    # The shift is 0 for a chunk with no value above -inf, so its sum is
    # exp(-inf) = 0 rather than NaN. A NaN or +inf makes the sum NaN.
    x_max = tl.max(x, axis=0)
    shift = tl.where(x_max == float("-inf"), 0.0, x_max)
    tl.store(chunk_max_ptr + row * chunks + chunk, x_max)
    tl.store(chunk_sum_ptr + row * chunks + chunk, tl.sum(tl.exp(x - shift), axis=0))


@triton.jit
def _normalize_kernel(
    logits_ptr,
    temperature_ptr,
    chunk_max_ptr,
    chunk_sum_ptr,
    out_ptr,
    vocab,
    logits_row_stride,
    chunks,
    BLOCK: tl.constexpr,
    CHUNKS_POW2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)

    # Every program of a row combines all of the row's chunks. A chunk with no
    # value above -inf adds 0 * exp(-inf) = 0, a NaN chunk sum leaves the row
    # NaN even when tl.max skipped the NaN, and a row with no value above -inf
    # divides 0 by 0, as torch.softmax does.
    offs = tl.arange(0, CHUNKS_POW2)
    mask = offs < chunks
    chunk_max = tl.load(
        chunk_max_ptr + row * chunks + offs, mask=mask, other=float("-inf")
    )
    chunk_sum = tl.load(chunk_sum_ptr + row * chunks + offs, mask=mask, other=0.0)
    row_max = tl.max(chunk_max, axis=0)
    shift = tl.where(row_max == float("-inf"), 0.0, row_max)
    row_sum = tl.sum(chunk_sum * tl.exp(chunk_max - shift), axis=0)

    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < vocab
    x = tl.load(
        logits_ptr + row * logits_row_stride + cols,
        mask=col_mask,
        other=float("-inf"),
    )
    x = x.to(tl.float32) / tl.load(temperature_ptr + row)
    tl.store(out_ptr + row * vocab + cols, tl.exp(x - shift) / row_sum, mask=col_mask)


@register_kernel(
    "sampling",
    "softmax",
    name="triton_softmax",
    solution="triton",
    signatures=format_signatures(
        "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def triton_softmax(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """``softmax(logits / temperature)`` of each row in fp32.

    Rows are cut into chunks of up to ``_MAX_BLOCK`` columns. The first launch
    writes each chunk's maximum and exp-sum; the second combines a row's chunks
    and writes the chunk's probabilities, reading the logits again.

    Args:
        logits: ``[M, N]`` logits with unit column stride.
        temperature: ``[M]`` contiguous fp32 temperature of each row.

    Returns:
        ``[M, N]`` contiguous fp32 probabilities.
    """
    rows, vocab = logits.shape
    block = min(_MAX_BLOCK, triton.next_power_of_2(vocab))
    chunks = triton.cdiv(vocab, block)
    stats = torch.empty((2, rows, chunks), dtype=torch.float32, device=logits.device)
    out = torch.empty((rows, vocab), dtype=torch.float32, device=logits.device)
    _chunk_stats_kernel[(rows, chunks)](
        logits,
        temperature,
        stats[0],
        stats[1],
        vocab,
        logits.stride(0),
        chunks,
        BLOCK=block,
        num_warps=_NUM_WARPS,
    )
    _normalize_kernel[(rows, chunks)](
        logits,
        temperature,
        stats[0],
        stats[1],
        out,
        vocab,
        logits.stride(0),
        chunks,
        BLOCK=block,
        CHUNKS_POW2=triton.next_power_of_2(chunks),
        num_warps=_NUM_WARPS,
    )
    return out
