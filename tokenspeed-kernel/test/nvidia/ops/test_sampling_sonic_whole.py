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

"""sonic rows without top-k, top-p or min-p draw from the whole vocabulary.

Such a row's token is the Gumbel-max over every token with the noise the
candidates' draw adds (keyed by the slot's seed, offset, the row's position
and the token id), so tokens past the top 128 are drawn at their probability,
and a verify accepts a point-draw draft at its probability over the whole
vocabulary and draws a rejected position from the rest of it.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import sonic
from tokenspeed_kernel.ops.sampling.sonic import (
    MAX_K,
    Indicator,
    SlotState,
    VocabShard,
    configs,
    pick,
    scratchpad_columns,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import (
    exponential_log,
    philox_gumbel,
    philox_key,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="sonic's kernels run on NVIDIA GPUs",
)

VOCAB = 32768
ROWS = 64
SLOTS = ROWS + 1
GRAMMAR_SLOT = 3


@triton.jit
def _noise_kernel(
    seeds_ptr, offsets_ptr, slots_ptr, steps_ptr, out_ptr, vocab, BLOCK: tl.constexpr
):
    """The Gumbel noise sonic's draws subtract, at every token of each row's slot and position."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key = philox_key(
        seeds_ptr, offsets_ptr, tl.load(slots_ptr + row), tl.load(steps_ptr + row)
    )
    tl.store(out_ptr + row * vocab + cols, philox_gumbel(key, cols), mask=cols < vocab)


def _noise(state: SlotState, offsets, slots, steps) -> torch.Tensor:
    out = torch.empty((slots.numel(), VOCAB), device="cuda")
    _noise_kernel[(slots.numel(), triton.cdiv(VOCAB, 4096))](
        state.seeds, offsets, slots, steps, out, VOCAB, BLOCK=4096
    )
    return out


def _state(
    timesteps: int,
    dtype: torch.dtype,
    temperature: float = 1.0,
    point_drafts: bool = False,
) -> SlotState:
    """Every slot plain (no truncation) at ``temperature``, verifying point-draw drafts if asked; one slot allows only every third token."""
    state = SlotState.allocate(SLOTS, timesteps, VOCAB, 7, "cuda")
    state.seeds.copy_(torch.arange(SLOTS) * 7919 + 3)
    bits = int(
        Indicator.from_params(temperature=temperature, greedy_draft=point_drafts)
    )
    state.indicators.fill_(bits)
    state.indicators[GRAMMAR_SLOT] = bits | int(Indicator.GRAMMAR)
    allowed = torch.zeros(VOCAB, dtype=torch.bool, device="cuda")
    allowed[::3] = True
    words = allowed.view(-1, 32).int() << torch.arange(32, device="cuda")
    state.grammar[GRAMMAR_SLOT] = words.sum(dim=1, dtype=torch.int64).to(torch.int32)
    scalars = torch.tensor([[temperature, 1.0, 0.0, 1.0, 0.0, 0.0]] * SLOTS).T
    return state.with_scalars(scalars.contiguous().to("cuda", dtype))


def _grammar_mask(rows_per_request: int) -> torch.Tensor:
    """Which tokens each logits row may draw."""
    allowed = torch.ones((ROWS, VOCAB), dtype=torch.bool, device="cuda")
    allowed[GRAMMAR_SLOT - 1] = torch.arange(VOCAB, device="cuda") % 3 == 0
    return allowed.repeat_interleave(rows_per_request, dim=0)


def _flat(
    rows: int, seed: int, dtype: torch.dtype, temperature: float = 1.0
) -> torch.Tensor:
    """Logits whose top 128 tokens hold under half of each row's mass at ``temperature``."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((rows, VOCAB), generator=generator, device="cuda") * 0.5
    head = torch.randint(0, VOCAB, (rows, 64), generator=generator, device="cuda")
    logits.scatter_(1, head, 6.0)
    return (logits * temperature).to(dtype)


def _scratchpad(rows: int, columns: int = VOCAB) -> torch.Tensor:
    return torch.empty(
        (rows, scratchpad_columns(columns)), dtype=torch.uint32, device="cuda"
    )


def _sample(logits, state, slots, offsets, config) -> torch.Tensor:
    tokens = torch.empty((ROWS, 1), dtype=torch.int32, device="cuda")
    sonic.sample(
        logits, state, slots, offsets, tokens, _scratchpad(ROWS), config, False
    )
    return tokens.view(-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_whole_rows_take_the_gumbel_max_over_every_token(dtype, temperature):
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device="cuda")
    offsets = torch.arange(SLOTS, dtype=torch.int64, device="cuda") * 13 + 5
    logits = _flat(ROWS, 1, dtype, temperature)
    state = _state(1, dtype, temperature)
    # Halving is exact in bf16, so the kernels' tempered logits are these values.
    values = logits.float() / temperature
    values = values.masked_fill(~_grammar_mask(1), -float("inf"))
    noise = _noise(state, offsets, slots, torch.zeros_like(slots))
    expected = (values - noise).argmax(dim=1).int()

    for config in configs(VOCAB):
        tokens = _sample(logits, state, slots, offsets, config)
        assert torch.equal(tokens, expected), config

    # Most draws land past the top 128, which the candidates never reach.
    top = values.topk(MAX_K, dim=1).indices
    outside = ~(top == expected[:, None].long()).any(dim=1)
    assert int(outside.sum()) > ROWS // 4
    assert int(expected[GRAMMAR_SLOT - 1]) % 3 == 0


def test_the_tail_past_the_top_128_is_drawn_at_its_probability():
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device="cuda")
    logits = _flat(ROWS, 2, torch.bfloat16)
    values = logits.float().masked_fill(~_grammar_mask(1), -float("inf"))
    probs = torch.softmax(values, dim=1)
    top = values.topk(MAX_K, dim=1).indices
    tail = 1.0 - probs.gather(1, top).sum(dim=1)
    state = _state(1, torch.bfloat16)
    draws, hits = 60, torch.zeros(ROWS, device="cuda")
    for step in range(draws):
        offsets = torch.full((SLOTS,), step, dtype=torch.int64, device="cuda")
        tokens = _sample(logits, state, slots, offsets, pick(VOCAB, ROWS))
        hits += ~(top == tokens[:, None].long()).any(dim=1)
    expected = float(tail.sum()) * draws
    deviation = math.sqrt(float((tail * (1 - tail)).sum()) * draws)
    assert abs(float(hits.sum()) - expected) < 5 * deviation, (
        float(hits.sum()),
        expected,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_shard_starting_off_a_multiple_of_four_draws_the_whole_rows_token(dtype):
    """Two ranks split at an unaligned column: the second takes ``philox_gumbel`` per token, the first ``block_gumbel``, and the merge draws the whole row's token."""
    split = VOCAB // 2 + 2
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device="cuda")
    offsets = torch.arange(SLOTS, dtype=torch.int64, device="cuda") * 3 + 1
    logits = _flat(ROWS, 5, dtype)
    state = _state(1, dtype)
    expected = _sample(logits, state, slots, offsets, pick(VOCAB, ROWS))
    shards = [logits[:, :split].contiguous(), logits[:, split:].contiguous()]
    landing = torch.empty((2, ROWS, 2 * MAX_K), device="cuda")

    def gather(pairs: torch.Tensor) -> torch.Tensor:
        """The second rank's candidates beside the first's, as the ranks' all-gather leaves them."""
        sonic.shard_candidates(
            shards[1],
            state,
            slots,
            offsets,
            None,
            _scratchpad(ROWS, VOCAB - split),
            VocabShard(split, None, None, None, landing[1]),
            pick(VOCAB - split, ROWS),
            False,
        )
        return landing

    tokens = torch.empty((ROWS, 1), dtype=torch.int32, device="cuda")
    sonic.sample(
        shards[0],
        state,
        slots,
        offsets,
        tokens,
        _scratchpad(ROWS, split),
        pick(split, ROWS),
        False,
        shard=VocabShard(0, None, None, gather, landing[0]),
    )
    assert torch.equal(tokens.view(-1), expected)
    # Some draws land in the second rank's columns.
    assert bool((expected >= split).any())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_verify_rows_take_the_gumbel_max_without_their_draft(dtype):
    timesteps = 3
    rows = ROWS * timesteps
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device="cuda")
    offsets = torch.arange(SLOTS, dtype=torch.int64, device="cuda") * 11 + 2
    logits = _flat(rows, 3, dtype)
    # Point-draw drafts verify with the masked residual.
    state = _state(timesteps, dtype, point_drafts=True)
    values = logits.float().masked_fill(~_grammar_mask(timesteps), -float("inf"))
    steps = torch.arange(timesteps, device="cuda").repeat(ROWS)
    noise = _noise(state, offsets, slots.repeat_interleave(timesteps), steps)
    scores = (values - noise).view(ROWS, timesteps, VOCAB)
    # Drafts: each position's own whole-vocabulary draw where the request is even, so they verify, else another token.
    drawn = scores.argmax(dim=2).int()
    drafts = torch.where(
        (torch.arange(ROWS, device="cuda") % 2 == 0)[:, None],
        drawn,
        (drawn + 3) % VOCAB,
    )
    drafted = torch.zeros((ROWS, timesteps), dtype=torch.int32, device="cuda")
    drafted[:, :-1] = drafts[:, :-1]
    original = drafted.clone()
    accepted = torch.empty((ROWS, 1), dtype=torch.int32, device="cuda")
    values_out = torch.empty((rows, MAX_K), device="cuda")
    indices_out = torch.empty((rows, MAX_K), dtype=torch.int32, device="cuda")
    sonic.verify(
        logits,
        state,
        slots,
        offsets,
        drafted,
        accepted,
        _scratchpad(rows),
        values_out,
        indices_out,
        pick(VOCAB, rows),
        False,
    )
    probs = torch.softmax(values.view(ROWS, timesteps, VOCAB), dim=2)
    rows_out = values_out.view(ROWS, timesteps, MAX_K)
    ids_out = indices_out.view(ROWS, timesteps, MAX_K)
    for request in range(ROWS):
        count = int(accepted[request])
        assert torch.equal(drafted[request, :count], original[request, :count])
        position = count
        if position < timesteps - 1:
            # The draw past a rejected draft: the best perturbed value without the draft.
            without = scores[request, position].clone()
            without[original[request, position]] = -float("inf")
            assert int(drafted[request, position]) == int(without.argmax())
        else:
            assert int(drafted[request, position]) == int(drawn[request, position])
        for step in range(timesteps - 1):
            # Each draft row carries p(draft) over the whole vocabulary.
            draft = int(original[request, step])
            torch.testing.assert_close(
                rows_out[request, step, 0],
                probs[request, step, draft],
                rtol=1e-4,
                atol=1e-7,
            )
            assert int(ids_out[request, step, 0]) == draft
            # Its third slot is the draw with the draft included.
            assert int(ids_out[request, step, 2]) == int(drawn[request, step])


def test_verify_accepts_a_point_draft_at_its_whole_vocabulary_probability():
    """A boosted draft at p = 0.25 over the whole vocabulary, though nearly all of the top 128's mass."""
    timesteps = 2
    rows = ROWS * timesteps
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device="cuda")
    logits = torch.zeros((rows, VOCAB), device="cuda")
    draft = 4321
    logits[:, draft] = math.log((VOCAB - 1) / 3)
    logits = logits.to(torch.bfloat16)
    p_draft = float(torch.softmax(logits[0].float(), dim=0)[draft])
    state = _state(timesteps, torch.bfloat16, point_drafts=True)
    # Every slot may draw the draft.
    state.indicators[GRAMMAR_SLOT] = state.indicators[0]
    trials, hits, tail = 40, 0, 0
    for step in range(trials):
        offsets = torch.full(
            (SLOTS,), step * timesteps, dtype=torch.int64, device="cuda"
        )
        drafted = torch.full((ROWS, timesteps), draft, dtype=torch.int32, device="cuda")
        accepted = torch.empty((ROWS, 1), dtype=torch.int32, device="cuda")
        sonic.verify(
            logits,
            state,
            slots,
            offsets,
            drafted,
            accepted,
            _scratchpad(rows),
            torch.empty((rows, MAX_K), device="cuda"),
            torch.empty((rows, MAX_K), dtype=torch.int32, device="cuda"),
            pick(VOCAB, rows),
            False,
        )
        hits += int(accepted.sum())
        rejected = accepted.view(-1) == 0
        assert not bool((drafted[rejected, 0] == draft).any())
        tail += int(rejected.sum())
    total = trials * ROWS
    assert abs(hits / total - p_draft) < 5 * math.sqrt(p_draft * (1 - p_draft) / total)
    assert tail > 0


@triton.jit
def _exponential_log_kernel(bits_ptr, out_ptr, size, BLOCK: tl.constexpr):
    span = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(bits_ptr + span, mask=span < size)
    tl.store(out_ptr + span, exponential_log(bits), mask=span < size)


def test_the_noise_is_accurate_where_gumbel_max_winners_draw():
    """log(-log u) for u = (bits + 0.5) / 2**32, near u = 1 (small -log u) above all."""
    generator = torch.Generator().manual_seed(0)
    high = 2**32 - 1 - torch.randint(0, 2**26, (1 << 16,), generator=generator)
    spread = torch.randint(0, 2**32, (1 << 16,), generator=generator)
    edges = torch.tensor(
        [0, 1, 2**31, 2**32 - 2**24 - 1, 2**32 - 2**24, 2**32 - 2, 2**32 - 1]
    )
    bits = torch.cat([high, spread, edges])
    out = torch.empty(bits.numel(), device="cuda")
    _exponential_log_kernel[(triton.cdiv(bits.numel(), 1024),)](
        bits.to(torch.uint32).cuda(), out, bits.numel(), BLOCK=1024
    )
    rest = (2**32 - 0.5 - bits.double()) / 2**32
    expected = torch.log(-torch.log1p(-rest))
    assert float((out.double().cpu() - expected).abs().max()) < 1e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
