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

"""Packed CSR (``query_start_loc``) verification tests for ``sonic.verify``."""

from __future__ import annotations

import dataclasses

import pytest
import torch
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

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="sonic's kernels run on NVIDIA GPUs",
)

VOCAB = 32768
RANKS = 4
TIMESTEPS = 4
GAMMA = TIMESTEPS - 1
THRESHOLD = 2.0

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


def _state(
    timesteps: int, logits: torch.Tensor, *, point_drafts: bool = False
) -> SlotState:
    state = SlotState.allocate(SLOTS, timesteps, VOCAB, 7, "cuda")
    state.seeds.copy_(torch.arange(SLOTS, dtype=torch.int64, device="cuda") * 1000 + 11)
    scalars = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]] * SLOTS).T.contiguous()
    for slot, params in enumerate(PARAMS):
        state.indicators[slot] = int(
            Indicator.from_params(greedy_draft=point_drafts, **params)
        )
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
    row0 = logits.float().topk(64, dim=1).indices
    if row0.shape[0] >= SLOTS:
        state.decode_counts[5, row0[5, ::3]] = 2
        state.context_counts[5, row0[5, 1::3]] = 1
    weak = torch.arange(97, VOCAB, VOCAB // 16, device="cuda")
    state.logit_bias[6, weak] = 12.0
    allowed = torch.zeros(VOCAB, dtype=torch.bool, device="cuda")
    allowed[::3] = True
    bits = allowed.view(-1, 32).int() << torch.arange(32, device="cuda")
    state.grammar[7] = bits.sum(dim=1, dtype=torch.int64).to(torch.int32)
    return state.with_scalars(scalars.to(logits.dtype).cuda())


def _copy_state(state: SlotState) -> SlotState:
    return dataclasses.replace(state, decode_counts=state.decode_counts.clone())


def _logits(dtype: torch.dtype, rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((rows, VOCAB), generator=generator, device="cuda") * 3.0
    if rows >= REQUESTS:
        logits[: rows : max(rows // REQUESTS, 1), 12345] = 20.0
    return logits.to(dtype)


def _scratchpad(logits: torch.Tensor) -> torch.Tensor:
    rows, vocab = logits.shape
    return torch.empty(
        (max(rows, 1), scratchpad_columns(vocab)), dtype=torch.uint32, device="cuda"
    )[:rows]


class _Ranks:
    def __init__(
        self,
        logits: torch.Tensor,
        state: SlotState,
        slots: torch.Tensor,
        offsets: torch.Tensor,
        drafted: torch.Tensor,
        *,
        draft_probs: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        config=None,
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
        self.config = config or pick(self.width, max(rows, 1))
        pairs = (2 if draft_probs is None else 3) * MAX_K
        self.landing = torch.empty((RANKS, max(rows, 1), pairs), device="cuda")[
            :, :rows
        ]
        self.state = state
        self.slots = slots
        self.offsets = offsets
        self.drafted = drafted
        self.query_start_loc = query_start_loc

    def shard(self, rank: int) -> VocabShard:
        return VocabShard(
            rank * self.width, None, None, self.gather, self.landing[rank]
        )

    def gather(self, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() > 0:
            assert pairs.data_ptr() == self.landing[0].data_ptr()
        for rank in range(1, RANKS):
            sonic.shard_candidates(
                self.shards[rank],
                self.state,
                self.slots,
                self.offsets,
                self.drafted,
                _scratchpad(self.shards[rank]),
                self.shard(rank),
                self.config,
                False,
                None if self.draft_probs is None else self.draft_probs[rank],
                query_start_loc=self.query_start_loc,
            )
        return self.landing


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("point_drafts", [False, True])
def test_packed_verify_matches_fullwidth_and_skips_padding_rows(dtype, point_drafts):
    """Packed CSR rows match full-width steps while shortened bonus steps and tail padding stay exact."""
    rows_full = REQUESTS * TIMESTEPS
    slots = torch.arange(REQUESTS, dtype=torch.int64, device="cuda")
    offsets = torch.full((SLOTS,), 7, dtype=torch.int64, device="cuda")
    full_logits = _logits(dtype, rows_full, 41)
    # Force strong peaks on row 0..3 so greedy / low-temp requests accept cleanly.
    full_view = full_logits.view(REQUESTS, TIMESTEPS, VOCAB)
    full_view[:, :, 777] = 18.0
    full_view[0, :, 12345] = 22.0

    full_state = _state(TIMESTEPS, full_logits, point_drafts=point_drafts)
    packed_state = _copy_state(full_state)
    eq_state = _copy_state(full_state)

    drafts = full_view[:, :GAMMA].float().argmax(dim=-1).to(torch.int32)
    drafted_full = torch.zeros((REQUESTS, TIMESTEPS), dtype=torch.int32, device="cuda")
    drafted_full[:, :GAMMA] = drafts

    # Full-width CSR matches un-packed verify bit-for-bit.
    full_starts = torch.arange(
        0, rows_full + 1, TIMESTEPS, device="cuda", dtype=torch.int32
    )
    drafted_eq = drafted_full.clone()
    acc_full = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
    acc_eq = torch.empty_like(acc_full)
    config = pick(VOCAB, rows_full)
    sonic.verify(
        full_logits,
        full_state,
        slots,
        offsets,
        drafted_full,
        acc_full,
        _scratchpad(full_logits),
        torch.empty((rows_full, MAX_K), device="cuda"),
        torch.empty((rows_full, MAX_K), dtype=torch.int32, device="cuda"),
        config,
        False,
    )
    sonic.verify(
        full_logits,
        eq_state,
        slots,
        offsets,
        drafted_eq,
        acc_eq,
        _scratchpad(full_logits),
        torch.empty((rows_full, MAX_K), device="cuda"),
        torch.empty((rows_full, MAX_K), dtype=torch.int32, device="cuda"),
        config,
        False,
        query_start_loc=full_starts,
    )
    assert torch.equal(acc_eq, acc_full)
    cols = torch.arange(drafted_eq.shape[1], device=drafted_eq.device).unsqueeze(0)
    assert torch.equal(
        torch.where(cols <= acc_eq, drafted_eq, 0),
        torch.where(cols <= acc_full, drafted_full, 0),
    )
    assert (drafted_eq[cols > acc_eq] == 0).all()
    assert torch.equal(eq_state.decode_counts, full_state.decode_counts)

    # Shortened ragged windows + poisoned tail logits beyond query_start_loc[-1].
    counts = [4, 2, 1, 3, 2, 1, 4, 2]
    starts = [0]
    gather_rows = []
    for req, cnt in enumerate(counts):
        for col in range(cnt):
            gather_rows.append(req * TIMESTEPS + col)
        starts.append(starts[-1] + cnt)
    active_rows = starts[-1]
    padded_rows = active_rows + 5
    packed_logits = torch.full(
        (padded_rows, VOCAB), float("nan"), dtype=dtype, device="cuda"
    )
    packed_logits[:active_rows] = full_logits.index_select(
        0, torch.tensor(gather_rows, device="cuda")
    )
    qsl = torch.tensor(starts, dtype=torch.int64, device="cuda")
    drafted_packed = torch.zeros(
        (REQUESTS, TIMESTEPS), dtype=torch.int32, device="cuda"
    )
    drafted_packed[:, :GAMMA] = drafts
    acc_packed = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
    sonic.verify(
        packed_logits,
        packed_state,
        slots,
        offsets,
        drafted_packed,
        acc_packed,
        _scratchpad(packed_logits),
        torch.empty((padded_rows, MAX_K), device="cuda"),
        torch.empty((padded_rows, MAX_K), dtype=torch.int32, device="cuda"),
        pick(VOCAB, padded_rows),
        False,
        query_start_loc=qsl,
    )
    for req, cnt in enumerate(counts):
        acc = int(acc_packed[req])
        assert 0 <= acc <= cnt - 1
        # Trailing unused columns past req_timesteps are zeroed cleanly.
        assert torch.equal(
            drafted_packed[req, cnt:],
            torch.zeros((TIMESTEPS - cnt,), dtype=torch.int32, device="cuda"),
        )
        # Anchor-only requests never emit a drafted token and cannot output Nan/poison.
        if cnt == 1:
            assert acc == 0
            assert 0 <= int(drafted_packed[req, 0]) < VOCAB
        # Full-length requests match full-width verification identically.
        if cnt == TIMESTEPS:
            assert acc == int(acc_full[req])
            assert torch.equal(drafted_packed[req], drafted_full[req])
    # Greedy shortened request accepts up to cnt - 1 and draws bonus without NaN.
    assert int(acc_packed[0]) == int(acc_full[0])


def test_packed_verify_handles_empty_requests_and_zero_rows():
    """Empty intervals (``c_b == 0``) output ``accepted = -1``, zeroed predictions, and skip counts."""
    counts = [3, 0, 2, 0]
    starts = torch.tensor([0, 3, 3, 5, 5], dtype=torch.int32, device="cuda")
    bs = len(counts)
    slots = torch.arange(bs, dtype=torch.int64, device="cuda")
    offsets = torch.full((SLOTS,), 3, dtype=torch.int64, device="cuda")
    logits = _logits(torch.bfloat16, 7, 19)
    logits[5:] = float("nan")
    state = _state(TIMESTEPS, logits)
    # Seed distinct non-zero decode count on empty row 1 to ensure it stays untouched.
    before_row1 = state.decode_counts[1].clone()
    drafted = torch.randint(0, VOCAB, (bs, TIMESTEPS), dtype=torch.int32, device="cuda")
    accepted = torch.empty((bs, 1), dtype=torch.int32, device="cuda")
    sonic.verify(
        logits,
        state,
        slots,
        offsets,
        drafted,
        accepted,
        _scratchpad(logits),
        torch.empty((7, MAX_K), device="cuda"),
        torch.empty((7, MAX_K), dtype=torch.int32, device="cuda"),
        pick(VOCAB, 7),
        False,
        query_start_loc=starts,
    )
    assert int(accepted[1]) == -1
    assert int(accepted[3]) == -1
    assert torch.equal(
        drafted[1], torch.zeros(TIMESTEPS, dtype=torch.int32, device="cuda")
    )
    assert torch.equal(
        drafted[3], torch.zeros(TIMESTEPS, dtype=torch.int32, device="cuda")
    )
    assert torch.equal(state.decode_counts[1], before_row1)

    # All-empty batch with zero logit rows.
    zero_logits = logits[:0]
    zero_starts = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
    drafted.fill_(99)
    sonic.verify(
        zero_logits,
        state,
        slots,
        offsets,
        drafted,
        accepted,
        _scratchpad(zero_logits),
        torch.empty((0, MAX_K), device="cuda"),
        torch.empty((0, MAX_K), dtype=torch.int32, device="cuda"),
        pick(VOCAB, 1),
        False,
        query_start_loc=zero_starts,
    )
    assert accepted.view(-1).tolist() == [-1] * bs
    assert drafted.count_nonzero().item() == 0

    # Zero batch size.
    sonic.verify(
        zero_logits,
        state,
        slots[:0],
        offsets,
        drafted[:0],
        accepted[:0],
        _scratchpad(zero_logits),
        torch.empty((0, MAX_K), device="cuda"),
        torch.empty((0, MAX_K), dtype=torch.int32, device="cuda"),
        pick(VOCAB, 1),
        False,
        query_start_loc=zero_starts[:1],
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_draft_probs", [False, True])
def test_packed_verify_sharded_matches_whole_rows(dtype, use_draft_probs):
    """Sharded ``sonic.verify(..., query_start_loc=...)`` matches whole-row results bit-for-bit."""
    counts = [4, 2, 0, 1, 3, 2, 4, 1]
    starts = [0]
    for cnt in counts:
        starts.append(starts[-1] + cnt)
    qsl = torch.tensor(starts, dtype=torch.int64, device="cuda")
    rows = starts[-1]
    padded_rows = rows + 3
    slots = torch.arange(REQUESTS, dtype=torch.int64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(23)

    for step in range(3):
        logits = torch.full(
            (padded_rows, VOCAB), float("nan"), dtype=dtype, device="cuda"
        )
        live_logits = _logits(dtype, rows, 100 + step)
        if dtype == torch.float32:
            # Remove tiny near-ties while keeping whole-row argmax separation.
            live_logits = live_logits.round(decimals=2)
        logits[:rows] = live_logits
        whole = _state(TIMESTEPS, logits)
        sharded = _copy_state(whole)
        offsets = torch.full(
            (SLOTS,), step * TIMESTEPS, dtype=torch.int64, device="cuda"
        )
        drafts = torch.randint(
            0,
            VOCAB,
            (REQUESTS, GAMMA),
            generator=generator,
            device="cuda",
            dtype=torch.int32,
        )
        # Seed high-probability drafts on row 0..3 so accepts occur reliably.
        drafts[0, :2] = 12345
        draft_probs = None
        if use_draft_probs:
            draft_probs = torch.softmax(
                torch.randn((SLOTS, GAMMA, VOCAB), generator=generator, device="cuda")
                * 0.25,
                dim=-1,
            )
            # Slot 4 exercises missing-proposal sentinel fallback on shortened window.
            draft_probs[4] = THRESHOLD + 1.0

        results = []
        for state, is_sharded in ((whole, False), (sharded, True)):
            drafted = torch.zeros(
                (REQUESTS, TIMESTEPS), dtype=torch.int32, device="cuda"
            )
            drafted[:, :GAMMA] = drafts
            accepted = torch.empty((REQUESTS, 1), dtype=torch.int32, device="cuda")
            values = torch.empty((padded_rows, MAX_K), device="cuda")
            indices = torch.empty(
                (padded_rows, MAX_K), dtype=torch.int32, device="cuda"
            )
            if is_sharded:
                ranks = _Ranks(
                    logits,
                    state,
                    slots,
                    offsets,
                    drafted,
                    draft_probs=draft_probs,
                    query_start_loc=qsl,
                )
                launch, config, shard = ranks.shards[0], ranks.config, ranks.shard(0)
                table = None if ranks.draft_probs is None else ranks.draft_probs[0]
            else:
                launch, config, shard = logits, pick(VOCAB, padded_rows), None
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
                candidate_drafts=(
                    torch.empty((padded_rows, MAX_K), device="cuda")
                    if is_sharded and use_draft_probs
                    else None
                ),
                query_start_loc=qsl,
            )
            results.append((drafted, accepted))
        (exp_drafted, exp_acc), (got_drafted, got_acc) = results
        assert torch.equal(got_acc, exp_acc), step
        assert torch.equal(got_drafted, exp_drafted), step
        assert torch.equal(sharded.decode_counts, whole.decode_counts), step


def test_packed_verify_runtime_query_start_loc_does_not_compile():
    """Varying CSR layouts and padded row counts compile no new Triton specializations."""
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import (
        multistep,
        prologue,
    )
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import shard as shard_ops
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import verify as verify_ops
    from utils import assert_no_triton_compile

    bs = 4
    max_rows = bs * TIMESTEPS
    slots = torch.arange(bs, dtype=torch.int64, device="cuda")
    offsets = torch.full((SLOTS,), 5, dtype=torch.int64, device="cuda")
    logits = _logits(torch.bfloat16, max_rows, 51)
    state = _state(TIMESTEPS, logits)
    whole_cfg = pick(VOCAB, max_rows)
    drafted = torch.zeros((bs, TIMESTEPS), dtype=torch.int32, device="cuda")
    accepted = torch.empty((bs, 1), dtype=torch.int32, device="cuda")
    values = torch.empty((max_rows, MAX_K), device="cuda")
    indices = torch.empty((max_rows, MAX_K), dtype=torch.int32, device="cuda")
    draft_probs = torch.full((SLOTS, GAMMA, VOCAB), 1.0 / VOCAB, device="cuda")

    def run_whole(starts_list, rows):
        qsl = torch.tensor(starts_list, dtype=torch.int32, device="cuda")
        view = logits[:rows]
        sonic.verify(
            view,
            state,
            slots,
            offsets,
            drafted,
            accepted,
            _scratchpad(view),
            values[:rows],
            indices[:rows],
            whole_cfg,
            False,
            draft_probs=draft_probs,
            reject_threshold=THRESHOLD,
            query_start_loc=qsl,
        )

    def run_sharded(starts_list, rows):
        qsl = torch.tensor(starts_list, dtype=torch.int32, device="cuda")
        view = logits[:rows]
        ranks = _Ranks(
            view,
            state,
            slots,
            offsets,
            drafted,
            draft_probs=draft_probs,
            query_start_loc=qsl,
            config=pick(VOCAB // RANKS, max_rows),
        )
        sonic.verify(
            ranks.shards[0],
            state,
            slots,
            offsets,
            drafted,
            accepted,
            _scratchpad(ranks.shards[0]),
            values[:rows],
            indices[:rows],
            ranks.config,
            False,
            shard=ranks.shard(0),
            draft_probs=ranks.draft_probs[0],
            reject_threshold=THRESHOLD,
            candidate_drafts=torch.empty((max_rows, MAX_K), device="cuda")[:rows],
            query_start_loc=qsl,
        )

    run_whole([0, 4, 8, 12, 16], 16)
    run_sharded([0, 4, 8, 12, 16], 16)

    with assert_no_triton_compile(
        prologue.bitpacked_reduction_kernel,
        prologue.whole_reduction_kernel,
        multistep.cumulative_unpack_kernel,
        shard_ops.shard_candidates_kernel,
        verify_ops.chain_speculative_verification_kernel,
    ):
        for starts_list, rows in (
            ([0, 1, 2, 3, 4], 4),
            ([0, 4, 6, 7, 10], 12),
            ([0, 0, 3, 3, 6], 8),
            ([0, 2, 4, 6, 8], 10),
        ):
            run_whole(starts_list, rows)
            run_sharded(starts_list, rows)
