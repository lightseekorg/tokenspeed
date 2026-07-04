# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

# Computes selected-token logprobs without full-vocabulary materialization.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _selected_token_logprobs_kernel(
    logits_ptr,
    tokens_ptr,
    out_ptr,
    vocab_size: tl.constexpr,
    logits_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_ptr = logits_ptr + row * logits_row_stride

    row_max = tl.full((), float("-inf"), tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offsets
        mask = cols < vocab_size
        vals = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        row_max = tl.maximum(
            row_max, tl.max(tl.where(mask, vals, float("-inf")), axis=0)
        )

    denom = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offsets
        mask = cols < vocab_size
        vals = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        weights = tl.exp(vals - row_max)
        denom += tl.sum(tl.where(mask, weights, 0.0), axis=0)

    token = tl.load(tokens_ptr + row).to(tl.int64)
    selected = tl.load(row_ptr + token).to(tl.float32)
    tl.store(out_ptr + row, selected - row_max - tl.log(tl.maximum(denom, 1.0e-20)))


def selected_token_logprobs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``log_softmax(logits)[row, tokens[row]]`` without materializing it."""
    if logits.ndim != 2:
        raise ValueError(f"selected_token_logprobs expects 2D logits")
    if logits.device.type != "cuda":
        raise ValueError("selected_token_logprobs requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            "selected_token_logprobs requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    rows, vocab_size = logits.shape
    if tokens.numel() != rows:
        raise ValueError(
            f"tokens length must match rows, got {tokens.numel()} and {rows}"
        )
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"tokens must be int32 or int64, got {tokens.dtype}")
    if out is None:
        out = torch.empty((rows,), dtype=torch.float32, device=logits.device)
    if out.dtype != torch.float32:
        raise ValueError(f"out must be float32, got {out.dtype}")
    if out.shape[0] < rows:
        raise ValueError(f"out is too small: {out.shape[0]} < {rows}")
    if rows == 0:
        return out[:0]

    _selected_token_logprobs_kernel[(rows,)](
        logits,
        tokens.reshape(-1),
        out,
        vocab_size=vocab_size,
        logits_row_stride=logits.stride(0),
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=3,
    )
    return out[:rows]
