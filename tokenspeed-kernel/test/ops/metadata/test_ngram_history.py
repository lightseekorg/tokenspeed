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

"""Parity tests: the fused n-gram history assembly must equal the eager chain."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import fill_ngram_history

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _eager_reference(
    tokens,
    positions,
    reset,
    slots,
    input_lengths,
    input_ids,
    valid_cache_lengths,
    tail,
    needs_seed,
    previous_tokens,
    token_mask,
    total_tokens,
    vocab_size,
):
    context = tail.shape[1]
    previous_tokens[total_tokens:].fill_(-1)
    token_mask[total_tokens:].zero_()
    if slots.numel():
        seed = (reset != 0) | needs_seed[slots]
        delta = valid_cache_lengths[slots] - positions
        assert bool((~seed | ((delta >= 0) & (delta <= 1))).all())
        distances = torch.arange(1, context + 1, device=tail.device)
        columns = (distances - delta[:, None]).clamp(0, context)
        prefix = torch.where(seed[:, None], tokens.gather(1, columns), tail[slots])
        prefix.masked_fill_((prefix < 0) | (prefix >= vocab_size), -1)
        tail[slots] = prefix
        needs_seed[slots] = False
    if total_tokens == 0:
        return
    lengths = input_lengths.long()
    ends = lengths.cumsum(0)
    rows = torch.arange(total_tokens, device=tail.device)
    requests = torch.searchsorted(ends, rows, right=True)
    local_rows = rows - (ends - lengths)[requests]
    distances = torch.arange(1, context + 1, device=tail.device)
    columns = (distances - local_rows[:, None] - 1).clamp(0, context - 1)
    ids = input_ids[:total_tokens].long()
    previous = torch.where(
        local_rows[:, None] >= distances,
        ids[(rows[:, None] - distances).clamp_min(0)],
        tail[slots[requests]].gather(1, columns),
    )
    previous.masked_fill_((previous < 0) | (previous >= vocab_size), -1)
    previous_tokens[:total_tokens].copy_(previous)
    token_mask[:total_tokens].copy_((ids >= 0) & (ids < vocab_size))


def _batch(batch, context, pool, capacity, vocab, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    slots = torch.randperm(pool, device=dev)[:batch]
    # Mixed extend chunks and speculative verify windows, packed in order.
    input_lengths = torch.randint(1, 7, (batch,), device=dev, dtype=torch.int32)
    total = int(input_lengths.sum())
    assert total <= capacity
    input_ids = torch.randint(-2, vocab + 3, (capacity,), device=dev, dtype=torch.int32)
    valid = torch.randint(3, 50, (pool,), device=dev, dtype=torch.int32)
    # Seeded rows sit within one token of their snapshot position.
    positions = valid[slots].long() - torch.randint(0, 2, (batch,), device=dev)
    reset = torch.randint(0, 2, (batch,), device=dev)
    needs_seed = torch.rand(pool, device=dev) > 0.5
    tokens = torch.randint(-1, vocab + 2, (batch, context + 1), device=dev)
    tail = torch.randint(-1, vocab + 2, (pool, context), device=dev)
    previous = torch.full((capacity, context), 123, device=dev, dtype=torch.int64)
    mask = torch.ones(capacity, device=dev, dtype=torch.bool)
    return (
        tokens,
        positions,
        reset,
        slots,
        input_lengths,
        input_ids,
        valid,
        tail,
        needs_seed,
        previous,
        mask,
        total,
        vocab,
    )


@pytest.mark.parametrize(
    ("batch", "context", "capacity"),
    ((1, 3, 32), (7, 3, 64), (16, 3, 128), (5, 1, 40), (33, 4, 300), (0, 3, 16)),
)
def test_fused_history_matches_eager_chain(batch, context, capacity):
    args = _batch(batch, context, pool=64, capacity=capacity, vocab=100, seed=batch)
    (
        tokens,
        positions,
        reset,
        slots,
        lengths,
        ids,
        valid,
        tail,
        needs,
        prev,
        mask,
        total,
        vocab,
    ) = args
    ref_tail, ref_needs = tail.clone(), needs.clone()
    ref_prev, ref_mask = prev.clone(), mask.clone()
    _eager_reference(
        tokens,
        positions,
        reset,
        slots,
        lengths,
        ids,
        valid,
        ref_tail,
        ref_needs,
        ref_prev,
        ref_mask,
        total,
        vocab,
    )

    fill_ngram_history(
        tokens,
        positions,
        reset,
        slots,
        lengths,
        ids,
        valid,
        tail,
        needs,
        prev,
        mask,
        total,
        vocab,
    )
    torch.cuda.synchronize()

    assert torch.equal(tail, ref_tail)
    assert torch.equal(needs, ref_needs)
    assert torch.equal(prev, ref_prev)
    assert torch.equal(mask, ref_mask)


def test_fused_history_rejects_misshapen_inputs():
    (
        tokens,
        positions,
        reset,
        slots,
        lengths,
        ids,
        valid,
        tail,
        needs,
        prev,
        mask,
        total,
        vocab,
    ) = _batch(4, 3, pool=16, capacity=32, vocab=50, seed=3)
    with pytest.raises(ValueError, match="tokens must be"):
        fill_ngram_history(
            tokens[:, :-1],
            positions,
            reset,
            slots,
            lengths,
            ids,
            valid,
            tail,
            needs,
            prev,
            mask,
            total,
            vocab,
        )
    with pytest.raises(ValueError, match="needs_seed must be"):
        fill_ngram_history(
            tokens,
            positions,
            reset,
            slots,
            lengths,
            ids,
            valid,
            tail,
            needs.to(torch.int8),
            prev,
            mask,
            total,
            vocab,
        )
    with pytest.raises(ValueError, match="exceeds capacity"):
        fill_ngram_history(
            tokens,
            positions,
            reset,
            slots,
            lengths,
            ids,
            valid,
            tail,
            needs,
            prev,
            mask,
            33,
            vocab,
        )
