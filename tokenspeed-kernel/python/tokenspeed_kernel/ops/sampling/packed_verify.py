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

"""Position-keyed probability sampling and packed chain verification."""

import math

import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
)

DRAFT_POS_HASH_MUL = 10000019
RNG_TAG_TARGET = 0
RNG_TAG_DRAFT = 1
RNG_TAG_ACCEPT = 2
RNG_TAG_REJECT = 3
RNG_TAG_BONUS = 4

__all__ = [
    "DRAFT_POS_HASH_MUL",
    "RNG_TAG_ACCEPT",
    "RNG_TAG_BONUS",
    "RNG_TAG_DRAFT",
    "RNG_TAG_REJECT",
    "RNG_TAG_TARGET",
    "sample_probs",
    "verify_packed_chain",
]


def _check_tensor(tensor, shape, dtype, device, name):
    if tensor.shape != shape or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _check_probs(probs):
    if probs.ndim != 2 or probs.shape[1] < 1 or probs.dtype != torch.float32:
        raise ValueError("probabilities must be float32 [rows, vocabulary]")
    if not probs.is_contiguous():
        raise ValueError("probability rows must be contiguous")
    if probs.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("packed probability sampling requires NVIDIA CUDA")


def sample_probs(*, probs, seeds, positions, output, rng_tag):
    """Draw one token per normalized probability row without changing inputs.

    Args:
        probs: Contiguous float32 [R,V] normalized, nonnegative probabilities.
        seeds: Int64 [R] request seeds (their bit patterns are unsigned keys).
        positions: Int64 [R] encoded sample positions.
        output: Int32 [R] output token ids.
        rng_tag: Explicit uint32 Philox namespace; use the exported RNG tags.

    Returns:
        None; writes output in place. A zero-mass row yields its lowest argmax.
    """
    _check_probs(probs)
    rows = probs.shape[0]
    for tensor, dtype, name in (
        (seeds, torch.int64, "seeds"),
        (positions, torch.int64, "positions"),
        (output, torch.int32, "output"),
    ):
        _check_tensor(tensor, (rows,), dtype, probs.device, name)
    if not isinstance(rng_tag, int) or not 0 <= rng_tag < 2**32:
        raise ValueError("rng_tag must be a uint32 integer")
    if rows:
        import tokenspeed_kernel.ops.sampling._cute_dsl.packed_verify  # noqa: F401

        kernel = select_kernel(
            "sampling",
            "sample_probs",
            format_signature(probs=dense_tensor_format(torch.float32)),
        )
        kernel(probs, seeds, positions, output, rng_tag)


def verify_packed_chain(
    *,
    predicts,
    accept_index,
    accept_length,
    candidates,
    target_probs,
    draft_probs,
    query_start_loc,
    seeds,
    positions,
    reject_draft_prob_threshold,
    draft_pool_indices=None,
):
    """Verify packed target rows against recorded draft probabilities.

    Args:
        predicts: Contiguous int32 [B*W] output; unused tail is zeroed.
        accept_index: Int32 [B,W] flat indices, -1 beyond accepted output.
        accept_length: Int32 [B] output length including residual/bonus.
        candidates: Int32 [B,W] anchor followed by draft token ids.
        target_probs: Contiguous float32 [Q,V] normalized target laws.
        draft_probs: Contiguous float32 [slots,W,V] recorded proposal laws.
            q(candidate)>reject_draft_prob_threshold marks an unavailable
            proposal; its full target law is sampled. The bonus row is never read.
        query_start_loc: Int64 [B+1] nonnegative CSR boundaries, with row counts
            in [0,W] and final boundary <= Q. Rows outside the intervals are ignored.
        seeds: Int64 [B] request keys.
        positions: Int64 [B] starting sample positions; local step i uses
            positions[b] + i * DRAFT_POS_HASH_MUL modulo 2**64.
            The request's position span must stay below DRAFT_POS_HASH_MUL
            to keep different local-depth streams disjoint across rounds.
        reject_draft_prob_threshold: Finite cutoff >=1 for unavailable proposals.
        draft_pool_indices: Optional int64 [B] mapping into draft_probs.
            Omission means its first B rows are already in batch order.

    Returns:
        None. Writes the three output buffers; inputs remain unchanged.
        Empty CSR requests produce zero outputs and length zero.
    """
    _check_probs(target_probs)
    if (
        type(reject_draft_prob_threshold) not in (int, float)
        or not math.isfinite(reject_draft_prob_threshold)
        or reject_draft_prob_threshold < 1
    ):
        raise ValueError("reject_draft_prob_threshold must be finite and at least 1")
    if candidates.ndim != 2 or candidates.shape[1] < 1:
        raise ValueError("candidates must have shape [B,W] with W>=1")
    batch, width = candidates.shape
    vocab = target_probs.shape[1]
    if batch * width >= 2**31:
        raise ValueError("flat output indices must fit int32")
    device = target_probs.device
    if draft_probs.ndim != 3 or draft_probs.shape[1:] != (width, vocab):
        raise ValueError("draft_probs must have shape [slots,W,V]")
    if draft_pool_indices is None:
        if draft_probs.shape[0] < batch:
            raise ValueError("draft_probs does not cover the batch")
        draft_pool_indices = torch.arange(batch, device=device, dtype=torch.int64)
    for tensor, shape, dtype, name in (
        (predicts, (batch * width,), torch.int32, "predicts"),
        (accept_index, (batch, width), torch.int32, "accept_index"),
        (accept_length, (batch,), torch.int32, "accept_length"),
        (candidates, (batch, width), torch.int32, "candidates"),
        (draft_probs, draft_probs.shape, torch.float32, "draft_probs"),
        (draft_pool_indices, (batch,), torch.int64, "draft_pool_indices"),
        (query_start_loc, (batch + 1,), torch.int64, "query_start_loc"),
        (seeds, (batch,), torch.int64, "seeds"),
        (positions, (batch,), torch.int64, "positions"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    if batch:
        import tokenspeed_kernel.ops.sampling._cute_dsl.packed_verify  # noqa: F401

        kernel = select_kernel(
            "sampling",
            "verify_packed_chain",
            format_signature(target_probs=dense_tensor_format(torch.float32)),
        )
        kernel(
            predicts,
            accept_index,
            accept_length,
            candidates,
            target_probs,
            draft_probs,
            query_start_loc,
            seeds,
            positions,
            draft_pool_indices,
            reject_draft_prob_threshold,
        )
