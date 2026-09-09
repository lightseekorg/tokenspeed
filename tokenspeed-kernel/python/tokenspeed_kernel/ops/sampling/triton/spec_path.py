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

"""Path selection over a block drafter's candidate lattice.

``scores[b, l, p, c]`` is the edge weight of taking candidate ``c`` at step
``l`` after candidate ``p`` at step ``l - 1``; step 0's predecessor is the
anchor, so all of its ``p`` rows are equal. One Triton program walks one
request, breaking ties toward the lowest index.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["dflash2_greedy_path"]


@triton.jit
def _greedy_path_kernel(
    candidate_ids_ptr,
    scores_ptr,
    anchor_ptr,
    out_ptr,
    out_row_stride,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    K_PAD: tl.constexpr,
):
    req = tl.program_id(0)
    out_base = req * out_row_stride
    tl.store(out_ptr + out_base, tl.load(anchor_ptr + req).to(tl.int32))

    lanes = tl.arange(0, K_PAD)
    lane_ok = lanes < TOP_K
    previous = 0
    for step in tl.static_range(NUM_STEPS):
        step_base = (req * NUM_STEPS + step) * TOP_K
        row = tl.load(
            scores_ptr + (step_base + previous) * TOP_K + lanes,
            mask=lane_ok,
            other=float("-inf"),
        ).to(tl.float32)
        previous = tl.min(tl.where(row == tl.max(row, axis=0), lanes, TOP_K), axis=0)
        token = tl.load(candidate_ids_ptr + step_base + previous).to(tl.int32)
        tl.store(out_ptr + out_base + step + 1, token)


def dflash2_greedy_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Walk the lattice one locally best edge at a time.

    Args:
        candidate_ids: ``[batch, num_steps, top_k]`` candidate token ids.
        scores: ``[batch, num_steps, top_k, top_k]`` predecessor-conditioned
            edge weights.
        anchor_token_ids: ``[batch]`` token each request's walk starts from.
        out: ``[batch, num_steps + 1]`` int32 destination; column 0 receives
            the anchor and column ``l + 1`` the step-``l`` token.

    Returns:
        ``out``, filled in place.

    Raises:
        ValueError: The lattice is not the shape, dtype or layout the kernel
            indexes with.
    """
    if candidate_ids.ndim != 3:
        raise ValueError(f"candidate_ids must be 3D, got {candidate_ids.ndim}D")
    batch_size, num_steps, top_k = candidate_ids.shape
    if scores.shape != (batch_size, num_steps, top_k, top_k):
        raise ValueError(
            f"scores shape {tuple(scores.shape)} must be "
            f"{(batch_size, num_steps, top_k, top_k)}"
        )
    if anchor_token_ids.shape != (batch_size,):
        raise ValueError(
            f"anchor_token_ids shape {tuple(anchor_token_ids.shape)} must be "
            f"{(batch_size,)}"
        )
    if out.shape != (batch_size, num_steps + 1):
        raise ValueError(
            f"out shape {tuple(out.shape)} must be {(batch_size, num_steps + 1)}"
        )
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    if not (candidate_ids.is_contiguous() and scores.is_contiguous()):
        raise ValueError("candidate_ids and scores must be contiguous")
    if out.stride(1) != 1:
        raise ValueError("out rows must be contiguous")
    if not all(
        tensor.device.type == "cuda"
        for tensor in (candidate_ids, scores, anchor_token_ids, out)
    ):
        raise ValueError("dflash2_greedy_path requires CUDA tensors")
    if batch_size == 0 or num_steps == 0:
        return out

    _greedy_path_kernel[(batch_size,)](
        candidate_ids,
        scores,
        anchor_token_ids,
        out,
        out.stride(0),
        NUM_STEPS=num_steps,
        TOP_K=top_k,
        K_PAD=triton.next_power_of_2(top_k),
    )
    return out
