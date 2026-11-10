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

"""sonic over vocabulary-sharded logits draws what it draws over the whole rows.

One GPU plays every rank: rank 0's ``gather`` runs the other ranks' candidate
selection over their column slices into the landing buffer, as the ranks'
all-gather would leave it.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import sonic
from tokenspeed_kernel.ops.sampling.sonic import (
    MAX_K,
    Indicator,
    SlotState,
    VocabShard,
    pick,
    scratchpad_columns,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.prologue import head_logits

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="sonic's kernels run on NVIDIA GPUs",
)

VOCAB = 32768
RANKS = 4
GAMMA = 3
STEPS = 6
THRESHOLD = 2.0

# One request per sampling setting; the last slot is the padding sentinel.
PARAMS = (
    dict(top_k=1),
    dict(temperature=0.7),
    dict(temperature=0.9, top_k=20),
    dict(top_p=0.8),
    dict(min_p=0.1),
    dict(multiplicative=1.3, frequency=0.4, presence=0.2),
    dict(logit_bias=True, temperature=0.8),
    dict(grammar=True, top_p=0.9),
)
REQUESTS = len(PARAMS)
SLOTS = REQUESTS + 1


def _state(timesteps: int, logits: torch.Tensor) -> SlotState:
    """Every request's slot set to its ``PARAMS``, with counts, bias and grammar spread over every shard."""
    state = SlotState.allocate(SLOTS, timesteps, VOCAB, 7, "cuda")
    state.seeds.copy_(torch.arange(SLOTS) * 1000 + 11)
    scalars = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]] * SLOTS).T.contiguous()
    for slot, params in enumerate(PARAMS):
        state.indicators[slot] = int(Indicator.from_params(**params))
        state.top_k[slot] = params.get("top_k", MAX_K)
        scalars[:, slot] = torch.tensor(
            [
                params.get("temperature", 1.0),
                params.get("top_p", 1.0),
                params.get("min_p", 0.0),
                params.get("multiplicative", 1.0),
                params.get("frequency", 0.0),
                params.get("presence", 0.0),
            ]
        )
    # The strongest tokens of each row: penalties and bias then reorder the head.
    head = logits.float().view(-1, timesteps, VOCAB)[:, 0].topk(64, dim=1).indices
    state.decode_counts[5, head[5, ::3]] = 2
    state.context_counts[5, head[5, 1::3]] = 1
    weak = torch.arange(97, VOCAB, VOCAB // 16, device="cuda")
    state.logit_bias[6, weak] = 12.0
    allowed = torch.zeros(VOCAB, dtype=torch.bool, device="cuda")
    allowed[::3] = True
    bits = allowed.view(-1, 32).int() << torch.arange(32, device="cuda")
    state.grammar[7] = bits.sum(dim=1, dtype=torch.int64).to(torch.int32)
    # Scalars match the logits: bf16, or fp32 for exact fp32 sampling.
    return state.with_scalars(scalars.to(logits.dtype).cuda())


def _copy(state: SlotState) -> SlotState:
    """The same slots with their own decode counts, which every draw updates."""
    return dataclasses.replace(state, decode_counts=state.decode_counts.clone())


def _logits(dtype: torch.dtype, rows: int, seed: int) -> torch.Tensor:
    """Random logits whose greedy rows hold one clear maximum (unsharded sonic breaks bf16 ties by block column)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((rows, VOCAB), generator=generator, device="cuda") * 3
    logits.view(REQUESTS, -1, VOCAB)[0, :, 12345] = 20.0
    return logits.to(dtype)


class _Ranks:
    """Rank 0's ``VocabShard`` over ``logits`` split in ``RANKS`` column slices; its gather runs every other rank's selection.

    With ``draft_probs`` each rank holds its own columns of the table, as a drafter recording q per shard leaves them.
    """

    def __init__(
        self,
        logits,
        state,
        slots,
        drafted,
        logit_scale=None,
        softcap=None,
        draft_probs=None,
    ):
        rows = logits.shape[0]
        self.width = VOCAB // RANKS
        self.shards = [
            logits[:, r * self.width : (r + 1) * self.width].contiguous()
            for r in range(RANKS)
        ]
        self.draft_probs = None
        if draft_probs is not None:
            self.draft_probs = [
                draft_probs[:, :, r * self.width : (r + 1) * self.width].contiguous()
                for r in range(RANKS)
            ]
        self.config = pick(self.width, rows)
        pairs = (2 if draft_probs is None else 3) * MAX_K
        self.landing = torch.empty((RANKS, rows, pairs), device="cuda")
        self.scale, self.softcap = logit_scale, softcap
        self.state, self.slots, self.drafted = state, slots, drafted

    def shard(self, rank: int) -> VocabShard:
        return VocabShard(
            rank * self.width,
            self.scale,
            self.softcap,
            self.gather,
            self.landing[rank],
        )

    def gather(self, pairs: torch.Tensor) -> torch.Tensor:
        assert pairs.data_ptr() == self.landing[0].data_ptr()
        for rank in range(1, RANKS):
            scratchpad = _scratchpad(self.shards[rank])
            sonic.shard_candidates(
                self.shards[rank],
                self.state,
                self.slots,
                self.drafted,
                scratchpad,
                self.shard(rank),
                self.config,
                False,
                None if self.draft_probs is None else self.draft_probs[rank],
            )
        return self.landing


def _scratchpad(logits: torch.Tensor) -> torch.Tensor:
    rows, vocab = logits.shape
    return torch.empty(
        (rows, scratchpad_columns(vocab)), dtype=torch.uint32, device="cuda"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sharded_sampling_draws_what_whole_rows_draw(dtype):
    exact = dtype == torch.float32
    slots = torch.arange(REQUESTS, dtype=torch.int64, device="cuda")
    for step in range(STEPS):
        logits = _logits(dtype, REQUESTS, step)
        if exact:
            # Well-separated heads, so the bf16 keys pick the fp32 top-k (sharded ranks re-sort before merging).
            logits = _separated(logits)
        whole = _state(1, logits)
        sharded = _copy(whole)
        offsets = torch.full((SLOTS,), step, dtype=torch.int64, device="cuda")
        expected = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
        sonic.sample(
            logits,
            whole,
            slots,
            offsets,
            expected,
            _scratchpad(logits),
            pick(VOCAB, REQUESTS),
            False,
        )
        ranks = _Ranks(logits, sharded, slots, None)
        tokens = torch.empty_like(expected)
        sonic.sample(
            ranks.shards[0],
            sharded,
            slots,
            offsets,
            tokens,
            _scratchpad(ranks.shards[0]),
            ranks.config,
            False,
            shard=ranks.shard(0),
        )
        assert torch.equal(tokens, expected), (step, tokens.view(-1), expected.view(-1))
        assert torch.equal(sharded.decode_counts, whole.decode_counts)
        # The grammar row draws an allowed token (every third).
        assert int(expected[7, 0]) % 3 == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sharded_verify_accepts_what_whole_rows_accept(dtype):
    exact = dtype == torch.float32
    timesteps = GAMMA + 1
    rows = REQUESTS * timesteps
    slots = torch.arange(REQUESTS, dtype=torch.int64, device="cuda")
    accepted_any = False
    for step in range(STEPS):
        logits = _logits(dtype, rows, 100 + step)
        if exact:
            logits = _separated(logits)
        whole = _state(timesteps, logits)
        sharded = _copy(whole)
        offsets = torch.full((SLOTS,), step, dtype=torch.int64, device="cuda")
        # Drafts: each row's top token where the step is even, a weak token otherwise.
        top = logits.float().view(REQUESTS, timesteps, VOCAB).argmax(dim=2).int()
        drafts = torch.where(
            torch.arange(timesteps, device="cuda") % 2 == step % 2, top, 5
        )
        results = []
        for state, ranks in ((whole, None), (sharded, True)):
            drafted = torch.zeros(
                (REQUESTS, timesteps), dtype=torch.int32, device="cuda"
            )
            drafted[:, :GAMMA] = drafts[:, :GAMMA]
            accepted = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
            values = torch.empty((rows, MAX_K), device="cuda")
            indices = torch.empty((rows, MAX_K), dtype=torch.int32, device="cuda")
            if ranks is None:
                launch, config, shard = logits, pick(VOCAB, rows), None
            else:
                ranks = _Ranks(logits, state, slots, drafted)
                launch, config, shard = ranks.shards[0], ranks.config, ranks.shard(0)
            sonic.verify(
                launch,
                state,
                slots,
                offsets,
                drafted,
                accepted,
                _scratchpad(launch),
                values,
                indices,
                config,
                False,
                shard=shard,
            )
            results.append((drafted, accepted))
        (expected, expected_accepted), (drafted, accepted) = results
        assert torch.equal(accepted, expected_accepted), step
        assert torch.equal(drafted, expected), step
        assert torch.equal(sharded.decode_counts, whole.decode_counts)
        accepted_any |= bool((accepted > 0).any())
    assert accepted_any


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_local_draft_probs_verify_as_the_whole_table_does(dtype):
    """Each rank's candidates carry q over its own columns: verify accepts and draws what whole rows reading the dense table do."""
    exact = dtype == torch.float32
    timesteps = GAMMA + 1
    rows = REQUESTS * timesteps
    slots = torch.arange(REQUESTS, dtype=torch.int64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(5)
    accepted_any = rejected_any = False
    for step in range(STEPS):
        logits = _logits(dtype, rows, 200 + step)
        if exact:
            logits = _separated(logits)
        whole = _state(timesteps, logits)
        sharded = _copy(whole)
        offsets = torch.full((SLOTS,), step, dtype=torch.int64, device="cuda")
        # A drafter near the target at each draft position, drafts sampled from its q.
        target = logits.float().view(REQUESTS, timesteps, VOCAB)[:, :GAMMA]
        noise = torch.randn(target.shape, generator=generator, device="cuda")
        q = torch.softmax(target + noise * 0.5, dim=2)
        drafts = torch.multinomial(q.view(-1, VOCAB), 1, generator=generator)
        draft_probs = torch.zeros((SLOTS, GAMMA, VOCAB), device="cuda")
        draft_probs[:REQUESTS] = q
        # Request 2's slot has no proposal (the runtime's sentinel).
        draft_probs[2] = THRESHOLD + 1
        results = []
        for state, local in ((whole, False), (sharded, True)):
            drafted = torch.zeros(
                (REQUESTS, timesteps), dtype=torch.int32, device="cuda"
            )
            drafted[:, :GAMMA] = drafts.view(REQUESTS, GAMMA)
            accepted = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
            values = torch.empty((rows, MAX_K), device="cuda")
            indices = torch.empty((rows, MAX_K), dtype=torch.int32, device="cuda")
            if local:
                ranks = _Ranks(logits, state, slots, drafted, draft_probs=draft_probs)
                launch, config, shard = ranks.shards[0], ranks.config, ranks.shard(0)
                table = ranks.draft_probs[0]
            else:
                launch, config, shard = logits, pick(VOCAB, rows), None
                table = draft_probs
            sonic.verify(
                launch,
                state,
                slots,
                offsets,
                drafted,
                accepted,
                _scratchpad(launch),
                values,
                indices,
                config,
                False,
                shard=shard,
                draft_probs=table,
                reject_threshold=THRESHOLD,
                candidate_drafts=torch.empty((rows, MAX_K), device="cuda"),
            )
            results.append((drafted, accepted))
        (expected, expected_accepted), (drafted, accepted) = results
        assert torch.equal(accepted, expected_accepted), step
        assert torch.equal(drafted, expected), step
        assert torch.equal(sharded.decode_counts, whole.decode_counts)
        assert int(accepted[2]) == 0
        accepted_any |= bool((accepted[1:] > 0).any())
        rejected_any |= bool((accepted[1:] < GAMMA).any())
    assert accepted_any and rejected_any


def _separated(logits: torch.Tensor) -> torch.Tensor:
    """fp32 logits whose 256 strongest tokens per row sit on a 0.25 grid (wider than a bf16 ulp there), the rest far below."""
    rows = logits.shape[0]
    generator = torch.Generator(device="cuda").manual_seed(rows)
    order = torch.rand((rows, VOCAB), generator=generator, device="cuda").argsort(dim=1)
    values = torch.full((rows, VOCAB), -30.0, device="cuda") + logits.float() * 0.01
    grid = 12.0 - 0.25 * torch.arange(256, device="cuda")
    values.scatter_(1, order[:, :256], grid.expand(rows, -1))
    values.view(REQUESTS, -1, VOCAB)[0, :, 12345] = 20.0
    return values


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sharded_greedy_takes_the_lowest_token_id_among_equal_maxima(dtype):
    slots = torch.arange(REQUESTS, dtype=torch.int64, device="cuda")
    logits = _logits(dtype, REQUESTS, 3)
    # Equal maxima in shards 3 and 1, and at two columns of shard 2's last block.
    logits[0, :] = -5.0
    logits[0, [3 * 8192 + 7, 8192 + 4000, 2 * 8192 + 8000, 2 * 8192 + 8100]] = 9.0
    state = _state(1, logits)
    ranks = _Ranks(logits, state, slots, None)
    tokens = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
    sonic.sample(
        ranks.shards[0],
        state,
        slots,
        torch.zeros((SLOTS,), dtype=torch.int64, device="cuda"),
        tokens,
        _scratchpad(ranks.shards[0]),
        ranks.config,
        False,
        shard=ranks.shard(0),
    )
    assert int(tokens[0]) == 8192 + 4000


@triton.jit
def _head_logits_kernel(x_ptr, logit_scale, softcap, columns, BLOCK: tl.constexpr):
    """The shard path's in-kernel scale and softcap, applied to whole rows in place."""
    row = tl.program_id(0)
    span = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = span < columns
    pointers = x_ptr + row * columns + span
    logits = tl.load(pointers, mask=mask, other=-float("inf"))
    tl.store(pointers, head_logits(logits, mask, logit_scale, softcap), mask=mask)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shards_apply_the_logit_scale_and_softcap_once(dtype):
    exact = dtype == torch.float32
    slots = torch.arange(REQUESTS, dtype=torch.int64, device="cuda")
    raw = _logits(dtype, REQUESTS, 9) * 4
    if exact:
        raw = _separated(raw)
    capped = raw.clone()
    _head_logits_kernel[(REQUESTS, triton.cdiv(VOCAB, 4096))](
        capped, 0.5, 15.0, VOCAB, BLOCK=4096
    )
    whole = _state(1, capped)
    sharded = _copy(whole)
    offsets = torch.zeros((SLOTS,), dtype=torch.int64, device="cuda")
    expected = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
    sonic.sample(
        capped,
        whole,
        slots,
        offsets,
        expected,
        _scratchpad(capped),
        pick(VOCAB, REQUESTS),
        False,
    )
    ranks = _Ranks(raw, sharded, slots, None, logit_scale=0.5, softcap=15.0)
    tokens = torch.empty_like(expected)
    sonic.sample(
        ranks.shards[0],
        sharded,
        slots,
        offsets,
        tokens,
        _scratchpad(ranks.shards[0]),
        ranks.config,
        False,
        shard=ranks.shard(0),
    )
    assert torch.equal(tokens, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
