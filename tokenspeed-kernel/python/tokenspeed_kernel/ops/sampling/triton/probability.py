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

# Probability-route compatibility helper for the existing FlashInfer full backend.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _min_p_renorm_prob_kernel(
    probs_ptr,
    min_p_ptr,
    out_ptr,
    vocab_size: tl.constexpr,
    probs_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    probs_row = probs_ptr + row * probs_row_stride
    out_row = out_ptr + row * out_row_stride

    max_prob = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        max_prob = tl.maximum(max_prob, tl.max(tl.where(mask, vals, 0.0), axis=0))

    threshold = max_prob * tl.load(min_p_ptr + row).to(tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        keep = mask & (vals >= threshold)
        denom += tl.sum(tl.where(keep, vals, 0.0), axis=0)

    inv_denom = 1.0 / tl.maximum(denom, 1.0e-20)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        keep = mask & (vals >= threshold)
        out = tl.where(keep, vals * inv_denom, 0.0)
        tl.store(out_row + cols, out, mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def min_p_renorm_prob(
    probs: torch.Tensor,
    min_p: torch.Tensor,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Renormalize probabilities after applying a per-row min-p cutoff.

    For each row, this computes ``threshold = min_p[row] * max(probs[row])``,
    zeros probabilities below the threshold, and renormalizes the surviving
    probabilities so the row sums to one.
    """
    if probs.ndim != 2:
        raise ValueError(f"min_p_renorm_prob expects 2D probs, got {probs.ndim}D")
    if min_p.ndim != 1:
        raise ValueError(f"min_p_renorm_prob expects 1D min_p, got {min_p.ndim}D")
    if min_p.shape[0] != probs.shape[0]:
        raise ValueError(
            "min_p length must match probs rows, "
            f"got {min_p.shape[0]} and {probs.shape[0]}"
        )
    if probs.device.type != "cuda" or min_p.device.type != "cuda":
        raise ValueError("min_p_renorm_prob requires CUDA tensors")
    if probs.stride(-1) != 1:
        raise ValueError(
            f"min_p_renorm_prob requires stride-1 vocab dimension, got stride={probs.stride()}"
        )
    if not min_p.is_contiguous():
        min_p = min_p.contiguous()

    out = torch.empty_like(probs)
    rows, vocab_size = probs.shape
    if rows == 0:
        return out

    block_size = min(4096, triton.next_power_of_2(vocab_size))
    num_warps = 4 if block_size <= 1024 else 8
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _min_p_renorm_prob_kernel[(rows,)](
        probs,
        min_p,
        out,
        vocab_size=vocab_size,
        probs_row_stride=probs.stride(0),
        out_row_stride=out.stride(0),
        BLOCK_SIZE=block_size,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=3,
        **extra_kwargs,
    )
    return out
