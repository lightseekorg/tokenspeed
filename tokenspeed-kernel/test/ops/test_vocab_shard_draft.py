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

"""Draft proposals sampled from vocabulary shards draw and record what whole rows do.

One GPU plays every rank: each rank's partials are stacked as the ranks'
gather would leave them, then every rank records over its own columns.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import (
    vocab_shard_draft_record,
    vocab_shard_draft_stats,
    vocab_shard_draft_stats_width,
)
from tokenspeed_kernel.ops.sampling.triton import (
    gumbel_sample_from_pools,
    gumbel_scratch_shape,
)
from tokenspeed_kernel.ops.sampling.triton.vocab_shard_draft import (
    _head_logits,
    _vocab_shard_draft_record_kernel,
    _vocab_shard_draft_stats_kernel,
)
from utils import assert_no_triton_compile

RANKS = 4
# A shard that ends inside a stats block.
SHARD = 7000
VOCAB = RANKS * SHARD
# Slots 1..8 serve the rows; slot 0 is never read.
SLOTS = 9
GREEDY_SLOT = 4
TEMPERATURES = [1.0, 1.0, 0.7, 1.3, 1.0, 0.05, 2.0, 1.0, 0.9]
SENTINEL = 3.0


def _pools(device: str) -> tuple[torch.Tensor, ...]:
    """Temperature, top-k (slot 4 greedy), seed and offset per slot."""
    temperature = torch.tensor(TEMPERATURES, dtype=torch.float32, device=device)
    top_k = torch.full((SLOTS,), 128, dtype=torch.int32, device=device)
    top_k[GREEDY_SLOT] = 1
    seed = torch.arange(SLOTS, dtype=torch.int64, device=device) * 7919 + 13
    offsets = torch.arange(SLOTS, dtype=torch.int64, device=device) * 41 + 5
    return temperature, top_k, seed, offsets


def _logits(rows: int, seed: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, VOCAB, generator=generator) * 3).to(device, dtype)


def _sharded(logits, slots, write_slots, pools, steps_q, *, scale=None, softcap=None):
    """Every rank's tokens and its columns of q, recorded into ``steps_q`` (one ``[SLOTS, SHARD]`` table per rank)."""
    temperature, top_k, seed, offsets = pools
    rows = logits.shape[0]
    width = vocab_shard_draft_stats_width(SHARD)
    shards = [logits[:, r * SHARD : (r + 1) * SHARD].contiguous() for r in range(RANKS)]
    stats = torch.empty((RANKS, rows, width), device=logits.device)
    for rank, shard in enumerate(shards):
        vocab_shard_draft_stats(
            shard,
            rank * SHARD,
            slots,
            temperature,
            seed,
            offsets,
            stats[rank],
            logit_scale=scale,
            softcap=softcap,
        )
    tokens = []
    for rank, shard in enumerate(shards):
        out = torch.empty((rows,), dtype=torch.int32, device=logits.device)
        vocab_shard_draft_record(
            shard,
            stats,
            rank * SHARD,
            slots,
            write_slots,
            temperature,
            top_k,
            steps_q[rank],
            out,
            logit_scale=scale,
            softcap=softcap,
            sentinel=SENTINEL,
        )
        tokens.append(out)
    return tokens


def _whole_row_draw(logits, slots, pools):
    """``gumbel_sample_from_pools`` over the gathered rows, as the whole-row drafter draws."""
    temperature, _, seed, offsets = pools
    rows = logits.shape[0]
    shape = gumbel_scratch_shape(rows, VOCAB)
    out = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    return gumbel_sample_from_pools(
        logits,
        slots.int(),
        temperature,
        seed,
        offsets,
        torch.empty(shape, dtype=torch.int32, device=logits.device),
        torch.empty(shape, dtype=torch.float32, device=logits.device),
        out,
    ).clone()


def _tables(device: str) -> list[torch.Tensor]:
    return [torch.zeros((SLOTS, SHARD), device=device) for _ in range(RANKS)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shards_draw_the_whole_row_token_and_record_its_softmax(dtype, device: str):
    rows = 8
    slots = torch.arange(1, rows + 1, dtype=torch.int64, device=device)
    # Rows write their slots in reverse, as the padded write map may reorder.
    write_slots = slots.flip(0)
    pools = _pools(device)
    logits = _logits(rows, 1, dtype, device)
    # The greedy row's maximum is tied in shards 3 and 1: the lower id wins.
    greedy_row = GREEDY_SLOT - 1
    logits[greedy_row, 3 * SHARD + 17] = 30.0
    logits[greedy_row, SHARD + 4321] = 30.0
    tables = _tables(device)

    tokens = _sharded(logits, slots, write_slots, pools, tables)

    expected = _whole_row_draw(logits, slots, pools)
    expected[greedy_row] = SHARD + 4321
    for rank_tokens in tokens:
        assert torch.equal(rank_tokens, expected)
    q = torch.cat(tables, dim=1)[write_slots]
    temperature = pools[0][slots]
    reference = torch.softmax(logits.float() / temperature[:, None], dim=1)
    reference[greedy_row] = 0.0
    reference[greedy_row, SHARD + 4321] = 1.0
    torch.testing.assert_close(q, reference, rtol=2e-4, atol=1e-7)
    assert q[greedy_row].sum().item() == 1.0
    # Slot 0 is no row's write slot: untouched.
    assert all(int(table[0].count_nonzero()) == 0 for table in tables)


def test_rows_without_a_finite_softmax_record_the_sentinel(device: str):
    rows = 4
    slots = torch.arange(1, rows + 1, dtype=torch.int64, device=device)
    pools = _pools(device)
    logits = _logits(rows, 2, torch.float32, device)
    # A NaN in shard 2, an overflow in shard 0, every column masked; slot 4 is greedy.
    logits[0, 2 * SHARD + 9] = float("nan")
    logits[1, 5] = float("inf")
    logits[2] = -float("inf")
    logits[3, 3 * SHARD + 1] = float("nan")
    tables = _tables(device)

    tokens = _sharded(logits, slots, slots, pools, tables)

    for rank_tokens in tokens:
        assert torch.equal(rank_tokens, tokens[0])
        assert bool(((rank_tokens >= 0) & (rank_tokens < VOCAB)).all())
    q = torch.cat(tables, dim=1)[slots]
    assert bool((q == SENTINEL).all())


@triton.jit
def _head_logits_kernel(
    x_ptr, out_ptr, logit_scale, softcap, columns, BLOCK: tl.constexpr
):
    """The kernels' scale and softcap over whole rows, as the processor stores them."""
    row = tl.program_id(0)
    span = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = span < columns
    x = tl.load(x_ptr + row * columns + span, mask=mask)
    tl.store(
        out_ptr + row * columns + span,
        _head_logits(x, logit_scale, softcap, True, True),
        mask=mask,
    )


def test_shards_apply_the_logit_scale_and_softcap_once(device: str):
    rows = 8
    slots = torch.arange(1, rows + 1, dtype=torch.int64, device=device)
    pools = _pools(device)
    raw = _logits(rows, 3, torch.bfloat16, device) * 6
    capped = torch.empty(raw.shape, dtype=torch.float32, device=device)
    _head_logits_kernel[(rows, triton.cdiv(VOCAB, 4096))](
        raw, capped, 0.5, 15.0, VOCAB, BLOCK=4096
    )
    tables = _tables(device)

    tokens = _sharded(raw, slots, slots, pools, tables, scale=0.5, softcap=15.0)

    expected = _whole_row_draw(capped, slots, pools)
    expected[GREEDY_SLOT - 1] = capped[GREEDY_SLOT - 1].argmax()
    assert torch.equal(tokens[0], expected)
    q = torch.cat(tables, dim=1)[slots]
    reference = torch.softmax(capped / pools[0][slots][:, None], dim=1)
    sampled = torch.arange(rows, device=device) != GREEDY_SLOT - 1
    torch.testing.assert_close(q[sampled], reference[sampled], rtol=2e-4, atol=1e-7)


def test_every_batch_size_and_step_compiles_nothing_after_warm_up(device: str):
    pools = _pools(device)
    # The runtime's table: every slot, one q row per draft step.
    steps = 3
    tables = [torch.zeros((SLOTS, steps, SHARD), device=device) for _ in range(RANKS)]
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device=device)
    logits = _logits(SLOTS - 1, 4, torch.bfloat16, device)
    _sharded(logits[:1], slots[:1], slots[:1], pools, [t[:, 0] for t in tables])
    with assert_no_triton_compile(
        _vocab_shard_draft_stats_kernel, _vocab_shard_draft_record_kernel
    ):
        for step in range(steps):
            for bs in range(1, SLOTS):
                _sharded(
                    logits[:bs],
                    slots[:bs],
                    slots[:bs],
                    pools,
                    [t[:, step] for t in tables],
                )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
