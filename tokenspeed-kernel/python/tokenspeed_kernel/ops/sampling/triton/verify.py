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

# TokenSpeed samples target rows first, then verifies by token-id comparison.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import pdl_enabled


@triton.jit(
    do_not_specialize=[
        "predicts_stride",
        "accept_index_row_stride",
        "accept_index_col_stride",
        "accept_token_num_stride",
        "candidates_row_stride",
        "candidates_col_stride",
        "target_sampled_stride",
    ],
)
def _verify_chain_target_sampled_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    target_sampled_ptr,
    predicts_stride,
    accept_index_row_stride,
    accept_index_col_stride,
    accept_token_num_stride,
    candidates_row_stride,
    candidates_col_stride,
    target_sampled_stride,
    NUM_DRAFT_TOKENS: tl.constexpr,
    WRITE_REJECTED_TARGETS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row = tl.program_id(0).to(tl.int64)
    base = row * NUM_DRAFT_TOKENS

    tl.store(accept_index_ptr + row * accept_index_row_stride, base)

    active = tl.full((), 1, tl.int32)
    num_accepted = tl.full((), 0, tl.int32)
    for i in tl.range(1, NUM_DRAFT_TOKENS):
        target_id = tl.load(target_sampled_ptr + (base + i - 1) * target_sampled_stride)
        if WRITE_REJECTED_TARGETS:
            # The Torch greedy reference converts target ids before comparing.
            target_id = target_id.to(tl.int32)
        draft_id = tl.load(
            candidates_ptr + row * candidates_row_stride + i * candidates_col_stride
        )
        match = (active != 0) & (draft_id == target_id)

        if WRITE_REJECTED_TARGETS:
            tl.store(predicts_ptr + (base + i - 1) * predicts_stride, target_id)
            tl.store(
                accept_index_ptr
                + row * accept_index_row_stride
                + i * accept_index_col_stride,
                tl.where(match, base + i, -1),
            )
        else:
            tl.store(
                predicts_ptr + (base + i - 1) * predicts_stride, target_id, mask=match
            )
            tl.store(
                accept_index_ptr
                + row * accept_index_row_stride
                + i * accept_index_col_stride,
                base + i,
                mask=match,
            )

        num_accepted = tl.where(match, i, num_accepted)
        active = tl.where(match, 1, 0)

    if WRITE_REJECTED_TARGETS:
        last = base + NUM_DRAFT_TOKENS - 1
        tl.store(
            predicts_ptr + last * predicts_stride,
            tl.load(target_sampled_ptr + last * target_sampled_stride),
        )
    final_id = tl.load(
        target_sampled_ptr + (base + num_accepted) * target_sampled_stride
    )
    tl.store(accept_token_num_ptr + row * accept_token_num_stride, num_accepted)
    tl.store(predicts_ptr + (base + num_accepted) * predicts_stride, final_id)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def verify_chain_target_sampled(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    target_sampled: torch.Tensor,
    *,
    write_rejected_targets: bool,
    enable_pdl: bool | None = None,
) -> None:
    """Verify a chain against target token ids, with an explicit output policy.

    Args:
        predicts: Writable int32 [>= batch * width] target-token destination.
        accept_index: Writable int32 [batch,width] flattened accepted row ids.
        accept_token_num: Writable int32 [>= batch] accepted draft counts,
            excluding the bonus token (callers add one).
        candidates: Int32 [batch,width] draft chain; column zero is its anchor.
        target_sampled: Int32/int64 target ids with at least batch * width entries.
        write_rejected_targets: True writes every target id as int32, compares
            after the same int32 conversion, and fills rejected indices with -1,
            matching the Torch greedy reference. False preserves rejected token
            and index tails, so accept_index must be -1-filled on entry.
        enable_pdl: Override the platform PDL setting when supplied.

    Returns:
        None. Supplied outputs are updated in place; each chain emits its
        longest matching draft prefix and the target's following bonus token.
    """
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got {candidates.ndim}D")
    if accept_index.shape != candidates.shape:
        raise ValueError(
            f"accept_index shape {accept_index.shape} must match candidates {candidates.shape}"
        )
    bs, num_draft_tokens = candidates.shape
    if num_draft_tokens < 1:
        raise ValueError("A verification chain must contain its anchor token")
    total = bs * num_draft_tokens
    if predicts.ndim != 1 or accept_token_num.ndim != 1:
        raise ValueError(
            "Verification token and length outputs must be one-dimensional"
        )
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
    if any(
        t.device != candidates.device
        for t in (predicts, accept_index, accept_token_num, target_sampled)
    ):
        raise ValueError("Verification inputs and outputs must share a device")
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
        predicts.stride(0),
        *accept_index.stride(),
        accept_token_num.stride(0),
        *candidates.stride(),
        target_sampled.stride(0),
        NUM_DRAFT_TOKENS=num_draft_tokens,
        WRITE_REJECTED_TARGETS=write_rejected_targets,
        ENABLE_PDL=enable_pdl,
        num_warps=1,
        **extra_kwargs,
    )
