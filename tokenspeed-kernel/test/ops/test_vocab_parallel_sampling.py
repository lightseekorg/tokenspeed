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

"""Top-k / top-p sampling and verify from candidates selected per vocab shard.

Each shard keeps its best ``top_k_pad`` logits, the shards' lists merge into
the row's sorted best, and the presorted kernels sample and verify from
those: every token and accept decision must equal the whole-row kernels'.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.sampling import vocab_shard_topk, vocab_topk_merge
from tokenspeed_kernel.ops.sampling.triton import (
    gumbel_sample_top_k_top_p_from_candidates,
    gumbel_sample_top_k_top_p_from_pools,
    verify_rows_top_k_top_p_from_candidates,
    verify_rows_top_k_top_p_from_pools,
)

_RANKS = 8
# Wider than one 2048-column chunk, so every shard selects in several chunks
# and merges them, as a 25k-column shard does.
_SHARD = 2600
_VOCAB = _RANKS * _SHARD
_BLOCK = 1024
_PAD = 128
_N = 8


def _logits(rows: int, seed: int, levels: int | None, device: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, _VOCAB, generator=gen) * 3
    if levels is not None:
        # Few distinct values: ties straddle shard and block boundaries.
        logits = torch.round(logits * levels) / levels
    return logits.to(device, torch.bfloat16)


def _shard_candidates(
    logits: torch.Tensor, k: int, vocab_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every shard's best ``k``, gathered rank-major and merged."""
    rows = logits.shape[0]
    pairs = torch.empty(_RANKS, rows, 2 * k, device=logits.device)
    for rank in range(_RANKS):
        shard = logits[:, rank * _SHARD : (rank + 1) * _SHARD]
        vocab_shard_topk(
            shard,
            k,
            None,
            rank * _SHARD,
            pairs[rank],
            softcap=None,
            vocab_mask=vocab_mask,
        )
    ids = torch.empty(rows, k, dtype=torch.int32, device=logits.device)
    values = torch.empty(rows, k, device=logits.device)
    vocab_topk_merge(pairs, k, None, ids, values)
    return ids, values


class _Pools:
    def __init__(self, bs, temperature, top_k, top_p, device):
        pool_rows = bs + 3
        index = torch.arange(1, bs + 1)
        self.req_pool_indices = index.to(device, torch.int32)

        def pool(values, dtype, fill):
            out = torch.full((pool_rows,), fill, dtype=dtype)
            out[index] = torch.tensor(values, dtype=dtype)
            return out.to(device)

        self.temperature = pool(temperature, torch.float32, 1.0)
        self.top_k = pool(top_k, torch.int32, 1)
        self.top_p = pool(top_p, torch.float32, 1.0)
        self.seed = (torch.arange(pool_rows, dtype=torch.int64) * 7919 + 13).to(device)
        self.offsets = (torch.arange(pool_rows, dtype=torch.int64) * 5 + 300).to(device)

    def args(self):
        return (
            self.req_pool_indices,
            self.temperature,
            self.top_k,
            self.top_p,
            self.seed,
            self.offsets,
        )


def _scratch(rows: int, device: str):
    width = triton.cdiv(_VOCAB, _BLOCK) * _PAD
    return (
        torch.empty(rows, width, dtype=torch.int32, device=device),
        torch.empty(rows, width, dtype=torch.float32, device=device),
    )


# top_k past the pad keeps the pad's best, as the whole-row stage 2 does.
_PARAMS = [
    ([1.0, 0.7, 1.0], [64, 64, 64], [1.0, 1.0, 1.0]),
    ([1.0, 0.5, 2.0], [1, 20, 128], [0.9, 0.5, 1.0]),
    ([1.0, 1.0, 1.0], [500, 3, 64], [1.0, 0.95, 0.7]),
]


@pytest.mark.parametrize("levels", [None, 2])
@pytest.mark.parametrize("temperature,top_k,top_p", _PARAMS)
def test_sample_from_shard_candidates_matches_whole_rows(
    levels, temperature, top_k, top_p, device: str
) -> None:
    bs = len(top_k)
    rows = bs * _N
    logits = _logits(rows, 3, levels, device)
    pools = _Pools(bs, temperature, top_k, top_p, device)

    whole = torch.full((rows,), -1, dtype=torch.int32, device=device)
    gumbel_sample_top_k_top_p_from_pools(
        logits,
        *pools.args(),
        *_scratch(rows, device),
        whole,
        block_size=_BLOCK,
        top_k_pad=_PAD,
        num_tokens_per_req=_N,
    )
    ids, values = _shard_candidates(logits, _PAD, None)
    sharded = torch.full((rows,), -2, dtype=torch.int32, device=device)
    gumbel_sample_top_k_top_p_from_candidates(
        ids,
        values,
        *pools.args(),
        sharded,
        vocab_size=_VOCAB,
        num_tokens_per_req=_N,
    )
    assert torch.equal(sharded, whole)


def _draft_q(bs: int, logits: torch.Tensor, device: str):
    """A chain's q over 16 candidates per step, drawn near the target's mode
    so drafts are accepted and rejected alike; request 0 records no q."""
    gen = torch.Generator().manual_seed(5)
    steps, num_candidates = _N - 1, 16
    pool_rows = bs + 3
    index = torch.arange(1, bs + 1)
    best = torch.topk(logits.float(), num_candidates, dim=-1).indices.cpu()
    q_ids = torch.zeros(pool_rows, steps, num_candidates, dtype=torch.int32)
    q = torch.zeros(pool_rows, steps, num_candidates)
    lanes = torch.randint(0, 4, (bs, steps), generator=gen)
    for b in range(bs):
        for step in range(steps):
            # Row ``step`` verifies draft ``step + 1`` against q row ``step``.
            q_ids[1 + b, step] = best[b * _N + step].to(torch.int32)
            weights = torch.rand(num_candidates, generator=gen) ** 4
            q[1 + b, step] = weights / weights.sum()
    slot = torch.full((pool_rows, _N), -1, dtype=torch.int32)
    slot[index[1:]] = torch.arange(-1, steps, dtype=torch.int32)
    pos = torch.zeros(pool_rows, _N, dtype=torch.int32)
    pos[index, 1:] = lanes.to(torch.int32)
    drafts = torch.gather(q_ids[index], 2, lanes[..., None])[..., 0]
    candidates = torch.zeros(bs, _N, dtype=torch.int32)
    candidates[:, 1:] = drafts
    return tuple(t.to(device) for t in (candidates, q, q_ids, slot, pos))


@pytest.mark.parametrize("levels", [None, 2])
@pytest.mark.parametrize("temperature,top_k,top_p", _PARAMS)
def test_verify_from_shard_candidates_matches_whole_rows(
    levels, temperature, top_k, top_p, device: str
) -> None:
    bs = len(top_k)
    rows = bs * _N
    logits = _logits(rows, 9, levels, device)
    pools = _Pools(bs, temperature, top_k, top_p, device)
    draft_q = _draft_q(bs, logits, device)

    def outputs():
        return (
            torch.full((rows,), -7, dtype=torch.int32, device=device),
            torch.full((rows,), -7, dtype=torch.int32, device=device),
        )

    whole = outputs()
    verify_rows_top_k_top_p_from_pools(
        logits,
        *pools.args(),
        *_scratch(rows, device),
        *draft_q,
        *whole,
        block_size=_BLOCK,
        top_k_pad=_PAD,
        num_tokens_per_req=_N,
    )
    ids, values = _shard_candidates(logits, _PAD, None)
    sharded = outputs()
    verify_rows_top_k_top_p_from_candidates(
        ids,
        values,
        *pools.args(),
        *draft_q,
        *sharded,
        vocab_size=_VOCAB,
        num_tokens_per_req=_N,
    )
    assert torch.equal(sharded[0], whole[0])
    assert torch.equal(sharded[1], whole[1])
    # Both rules ran: some drafts were accepted and some rejected.
    accepted = whole[0].view(bs, _N)[1:, :-1]
    assert 0 < int(accepted.sum()) < accepted.numel()


def test_shard_selection_applies_the_token_bitmask(device: str) -> None:
    """A disallowed token is never a candidate, as if masked to -inf first;
    the shard's bitmask words straddle its boundaries (2600 % 32 != 0)."""
    bs = 2
    rows = bs * _N
    logits = _logits(rows, 4, None, device)
    gen = torch.Generator().manual_seed(8)
    allowed = torch.rand(rows, _VOCAB, generator=gen) < 0.3
    bits = torch.zeros(rows, triton.cdiv(_VOCAB, 32) * 32, dtype=torch.int64)
    bits[:, :_VOCAB] = allowed.to(torch.int64)
    words = (bits.view(rows, -1, 32) << torch.arange(32)).sum(-1)
    # Two's complement int32 words, as xgrammar stores them.
    mask = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    pools = _Pools(bs, [1.0, 0.8], [64, 128], [1.0, 0.9], device)

    masked = logits.masked_fill(~allowed.to(device), float("-inf"))
    whole = torch.full((rows,), -1, dtype=torch.int32, device=device)
    gumbel_sample_top_k_top_p_from_pools(
        masked,
        *pools.args(),
        *_scratch(rows, device),
        whole,
        block_size=_BLOCK,
        top_k_pad=_PAD,
        num_tokens_per_req=_N,
    )
    ids, values = _shard_candidates(logits, _PAD, mask.to(device))
    sharded = torch.full((rows,), -2, dtype=torch.int32, device=device)
    gumbel_sample_top_k_top_p_from_candidates(
        ids, values, *pools.args(), sharded, vocab_size=_VOCAB, num_tokens_per_req=_N
    )
    assert torch.equal(sharded, whole)
    assert bool(allowed[torch.arange(rows), sharded.long().cpu()].all())
