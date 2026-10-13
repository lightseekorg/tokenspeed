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

"""Top-``k`` log-probabilities of every row of a logits matrix.

Equivalent to ``torch.topk(torch.log_softmax(logits.float(), -1), k)`` for
small ``k`` without materialising the float log-softmax. Each row's
vocabulary is split across programs: every split streams its slice once,
keeping an online log-sum-exp and a running top-``k``; a second pass merges
the splits. Draft-tree expansion calls this once per drafting step.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["logprob_topk"]

# Vocabulary per split: enough splits to fill the GPU at small row counts.
_SPLIT_VOCAB = 8192


@triton.jit
def _logprob_topk_split_kernel(
    logits_ptr,
    part_v_ptr,  # [rows, SPLITS, K_PAD] float32 split top-k logits (-inf padded)
    part_i_ptr,  # [rows, SPLITS, K_PAD] int64
    part_m_ptr,  # [rows, SPLITS] float32 split max
    part_s_ptr,  # [rows, SPLITS] float32 split sum of exp(x - max)
    stride_row,
    vocab,
    split_len,
    SPLITS: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    slots = tl.arange(0, K_PAD)
    # Padding slots hold +inf so the running minimum never picks them.
    best_v = tl.where(slots < K, float("-inf"), float("inf"))
    best_i = tl.zeros([K_PAD], dtype=tl.int64)
    run_max = float("-inf")
    run_sum = 0.0
    lo = split * split_len
    hi = tl.minimum(lo + split_len, vocab)
    for start in range(lo, hi, BLOCK):
        cols = start + offs
        x = tl.load(
            logits_ptr + row.to(tl.int64) * stride_row + cols,
            mask=cols < hi,
            other=float("-inf"),
        ).to(tl.float32)
        new_max = tl.maximum(run_max, tl.max(x, axis=0))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        run_sum = run_sum * tl.exp(run_max - safe_max) + tl.sum(
            tl.exp(x - safe_max), axis=0
        )
        run_max = new_max
        for _ in tl.static_range(K):
            top = tl.max(x, axis=0)
            kth = tl.min(best_v, axis=0)
            take = top > kth
            col = tl.min(tl.where(x == top, cols, vocab), axis=0)
            slot = tl.min(tl.where(best_v == kth, slots, K_PAD), axis=0)
            hit = (slots == slot) & take
            best_v = tl.where(hit, top, best_v)
            best_i = tl.where(hit, col.to(tl.int64), best_i)
            x = tl.where(cols == col, float("-inf"), x)
    part = row * SPLITS + split
    tl.store(
        part_v_ptr + part * K_PAD + slots, tl.where(slots < K, best_v, float("-inf"))
    )
    tl.store(part_i_ptr + part * K_PAD + slots, best_i)
    tl.store(part_m_ptr + part, run_max)
    tl.store(part_s_ptr + part, run_sum)


@triton.jit
def _logprob_topk_merge_kernel(
    part_v_ptr,
    part_i_ptr,
    part_m_ptr,
    part_s_ptr,
    scores_ptr,
    ids_ptr,
    SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    splits = tl.arange(0, SPLITS_PAD)
    split_ok = splits < SPLITS
    m = tl.load(part_m_ptr + row * SPLITS + splits, mask=split_ok, other=float("-inf"))
    s = tl.load(part_s_ptr + row * SPLITS + splits, mask=split_ok, other=0.0)
    top_m = tl.max(m, axis=0)
    safe_m = tl.where(m == float("-inf"), top_m, m)
    lse = top_m + tl.log(tl.sum(s * tl.exp(safe_m - top_m), axis=0))

    cand = tl.arange(0, SPLITS_PAD * K_PAD)
    cand_ok = (cand // K_PAD) < SPLITS
    v = tl.load(
        part_v_ptr + row * SPLITS * K_PAD + cand, mask=cand_ok, other=float("-inf")
    )
    ids = tl.load(part_i_ptr + row * SPLITS * K_PAD + cand, mask=cand_ok, other=0)
    # Best first; ties go to the lower candidate slot.
    for j in tl.static_range(K):
        top = tl.max(v, axis=0)
        pick = tl.min(tl.where(v == top, cand, SPLITS_PAD * K_PAD), axis=0)
        tl.store(scores_ptr + row * K + j, top - lse)
        tl.store(ids_ptr + row * K + j, tl.sum(tl.where(cand == pick, ids, 0), axis=0))
        v = tl.where(cand == pick, float("-inf"), v)


def logprob_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` log-probabilities per row, best first.

    Args:
        logits: ``[rows, vocab]`` logits with unit stride along the vocabulary.
        k: entries to keep per row, at most 16.

    Returns:
        ``(scores, ids)``: ``[rows, k]`` float32 log-probabilities and int64
        vocabulary ids, sorted by descending score.
    """
    if not 1 <= k <= 16:
        raise ValueError(f"logprob_topk keeps 1..16 entries, got {k}")
    if logits.stride(-1) != 1:
        raise ValueError("logprob_topk needs unit stride along the vocabulary")
    rows, vocab = logits.shape
    scores = torch.empty((rows, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    if rows == 0:
        return scores, ids
    splits = triton.cdiv(vocab, _SPLIT_VOCAB)
    k_pad = max(2, triton.next_power_of_2(k))
    part_v = torch.empty(
        (rows, splits, k_pad), dtype=torch.float32, device=logits.device
    )
    part_i = torch.empty((rows, splits, k_pad), dtype=torch.int64, device=logits.device)
    part_m = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    part_s = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    _logprob_topk_split_kernel[(rows, splits)](
        logits,
        part_v,
        part_i,
        part_m,
        part_s,
        logits.stride(0),
        vocab,
        _SPLIT_VOCAB,
        SPLITS=splits,
        K=k,
        K_PAD=k_pad,
        BLOCK=min(4096, triton.next_power_of_2(vocab)),
        num_warps=8,
    )
    _logprob_topk_merge_kernel[(rows,)](
        part_v,
        part_i,
        part_m,
        part_s,
        scores,
        ids,
        SPLITS=splits,
        SPLITS_PAD=triton.next_power_of_2(splits),
        K=k,
        K_PAD=k_pad,
    )
    return scores, ids
