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

# Adapted from vLLM MRV2 speculative decode flow:
#   https://vllm.ai/blog/2026-03-24-mrv2
#   https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/rejection_sample.py
# TokenSpeed samples target rows first, then verifies by token-id comparison.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _verify_chain_target_sampled_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    target_sampled_ptr,
    NUM_DRAFT_TOKENS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row = tl.program_id(0)
    base = row * NUM_DRAFT_TOKENS

    tl.store(accept_index_ptr + base, base)

    active = tl.full((), 1, tl.int32)
    num_accepted = tl.full((), 0, tl.int32)
    for i in tl.range(1, NUM_DRAFT_TOKENS):
        target_id = tl.load(target_sampled_ptr + base + i - 1)
        draft_id = tl.load(candidates_ptr + base + i)
        match = (active != 0) & (draft_id == target_id)

        tl.store(predicts_ptr + base + i - 1, target_id, mask=match)
        tl.store(accept_index_ptr + base + i, base + i, mask=match)

        num_accepted = tl.where(match, i, num_accepted)
        active = tl.where(match, 1, 0)

    final_id = tl.load(target_sampled_ptr + base + num_accepted)
    tl.store(accept_token_num_ptr + row, num_accepted)
    tl.store(predicts_ptr + base + num_accepted, final_id)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def verify_chain_target_sampled(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    target_sampled: torch.Tensor,
    *,
    enable_pdl: bool = False,
) -> None:
    """Verify a speculative chain against already-sampled target tokens."""
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got {candidates.ndim}D")
    if accept_index.shape != candidates.shape:
        raise ValueError(
            f"accept_index shape {accept_index.shape} must match candidates {candidates.shape}"
        )
    bs, num_draft_tokens = candidates.shape
    total = bs * num_draft_tokens
    if predicts.shape[0] < total:
        raise ValueError(f"predicts is too small: {predicts.shape[0]} < {total}")
    if accept_token_num.shape[0] < bs:
        raise ValueError(
            f"accept_token_num is too small: {accept_token_num.shape[0]} < {bs}"
        )
    if target_sampled.numel() < total:
        raise ValueError(
            f"target_sampled is too small: {target_sampled.numel()} < {total}"
        )
    if candidates.dtype != torch.int32:
        raise ValueError(f"candidates must be int32, got {candidates.dtype}")
    if predicts.dtype != torch.int32:
        raise ValueError(f"predicts must be int32, got {predicts.dtype}")
    if accept_index.dtype != torch.int32:
        raise ValueError(f"accept_index must be int32, got {accept_index.dtype}")
    if accept_token_num.dtype != torch.int32:
        raise ValueError(
            f"accept_token_num must be int32, got {accept_token_num.dtype}"
        )
    if target_sampled.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"target_sampled must be int32 or int64, got {target_sampled.dtype}"
        )
    if candidates.device.type != "cuda":
        raise ValueError("verify_chain_target_sampled requires CUDA tensors")
    if bs == 0:
        return

    target_sampled = target_sampled.reshape(-1)
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _verify_chain_target_sampled_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        target_sampled,
        NUM_DRAFT_TOKENS=num_draft_tokens,
        ENABLE_PDL=enable_pdl,
        num_warps=1,
        **extra_kwargs,
    )
