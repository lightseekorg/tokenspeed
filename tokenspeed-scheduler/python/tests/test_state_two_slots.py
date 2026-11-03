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

"""Fused-role State groups: two slots per request, no growth.

A State group's row is ``[slot0, slot1]``: the checkpoint a forward reads and
the block it writes. A prompt with no checkpoint to resume from starts from a
fresh zeroed block, so a request holds its two blocks from its first admission
on; every later rotation swaps the released input for the fresh live slot, and
a decode step writes slot 1 in place. A pool sized at two blocks per live
request therefore back-pressures at admission and never wedges at the first
decode.

Shape: K3-shaped 4-group config (P = 2 tokens per block, 32 usable pages),
one-block prompts, max_new_tokens = 4: 2 History pages + 3 x 2 state blocks
per admitted request, four requests fill the pool.
"""

from __future__ import annotations

import pytest
from conftest import K3_STATE_GROUPS, _make_k3_config

ts = pytest.importorskip("tokenspeed_scheduler")

PROMPT_TOKENS = 2  # exactly one block at P=2
MAX_NEW_TOKENS = 4
NUM_REQUESTS = 6
MAX_ROUNDS = 200
IDLE_LIMIT = 10


def _fused_config(decode_input_tokens: int) -> "ts.SchedulerConfig":
    cfg = _make_k3_config()
    cfg.decode_input_tokens = decode_input_tokens
    cfg.max_batch_size = 8
    return cfg


def _spec(request_id: str, tokens: list[int], max_new_tokens: int) -> "ts.RequestSpec":
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(tokens)
    spec.max_new_tokens = max_new_tokens
    return spec


def _advance(scheduler, request_id: str, tokens: list[int]) -> None:
    ec = ts.ExecutionEvent()
    ev = ts.ForwardEvent.ExtendResult()
    ev.request_id = request_id
    ev.tokens = list(tokens)
    ec.add_event(ev)
    scheduler.advance(ec)


def _reserve(scheduler, request_id: str, n: int) -> None:
    ec = ts.ExecutionEvent()
    ev = ts.ForwardEvent.UpdateReserveNumTokens()
    ev.request_id = request_id
    ev.reserve_num_tokens_in_next_schedule_event = n
    ec.add_event(ev)
    scheduler.advance(ec)


def _finish(scheduler, request_id: str) -> None:
    ec = ts.ExecutionEvent()
    ev = ts.ForwardEvent.Finish()
    ev.request_id = request_id
    ec.add_event(ev)
    scheduler.advance(ec)


def _dispatched(plan) -> list:
    """Forward batches that actually carry requests (the plan may hold an empty Batch)."""
    return [op for op in plan.forward if list(op.request_ids)]


def _state_rows(batch, request_id: str) -> dict[str, list[int]]:
    """The two-slot row per State group for one request row of a Batch."""
    row = list(batch.request_ids).index(request_id)
    tables = dict(batch.block_tables)
    return {g: list(tables[g][row]) for g in K3_STATE_GROUPS}


def _submit(scheduler, rids: list[str]) -> None:
    scheduler.submit_requests(
        [
            _spec(rid, [10 * i + t for t in range(PROMPT_TOKENS)], MAX_NEW_TOKENS)
            for i, rid in enumerate(rids)
        ]
    )


def _run_closed_loop(scheduler, rids: list[str]) -> dict:
    """Drive the Fused scheduler like the event loop: every dispatched row (prefill
    or decode) yields one token, requests finish at MAX_NEW_TOKENS."""
    generated = {rid: 0 for rid in rids}
    finished: list[str] = []
    idle_rounds = rounds = 0
    while len(finished) < len(rids) and rounds < MAX_ROUNDS:
        rounds += 1
        plan = scheduler.next_execution_plan()
        progressed = False
        for op in _dispatched(plan):
            for rid in op.request_ids:
                generated[rid] += 1
                _advance(scheduler, rid, [200 + generated[rid]])
                if generated[rid] >= MAX_NEW_TOKENS:
                    _finish(scheduler, rid)
                    finished.append(rid)
                elif generated[rid] > 1:
                    # The first token comes from the prefill row and leaves the
                    # request in PrefillDone, where UpdateReserveNumTokens is an
                    # invalid FSM transition (it already carries the decode
                    # reserve); only decode rows update the reserve.
                    _reserve(scheduler, rid, 1)
            progressed = True
        idle_rounds = 0 if progressed else idle_rounds + 1
        if idle_rounds >= IDLE_LIMIT:
            break
    return {
        "rounds": rounds,
        "idle_rounds": idle_rounds,
        "finished": finished,
        "waiting": scheduler.waiting_size(),
        "decoding": scheduler.decoding_size(),
        "available": scheduler.available_lcm_blocks(),
        "active": scheduler.active_lcm_blocks(),
    }


class TestFusedStateTwoSlots:
    def test_prompts_filling_the_pool_never_wedge(self):
        """Six one-block prompts against a pool that holds four: admission
        back-pressures and every request runs to completion."""
        scheduler = ts.Scheduler(_fused_config(decode_input_tokens=1))
        rids = [f"r{i}" for i in range(NUM_REQUESTS)]
        _submit(scheduler, rids)
        result = _run_closed_loop(scheduler, rids)
        assert result["idle_rounds"] < IDLE_LIMIT, f"scheduler went idle: {result}"
        assert sorted(result["finished"]) == sorted(rids), result
        assert result["waiting"] == 0 and result["decoding"] == 0
        assert result["active"] == 0

    def test_two_slots_rotate_into_the_first_decode_and_decode_in_place(self):
        """The prefill row reads a fresh zeroed block (slot 0) and writes slot 1;
        the first decode rotates (the prompt-end state becomes slot 0, a fresh
        slot 1 is written); every later decode releases slot 0 and writes slot 1
        in place."""
        scheduler = ts.Scheduler(_fused_config(decode_input_tokens=1))
        scheduler.submit_requests([_spec("r0", [1, 2, 3, 4], MAX_NEW_TOKENS)])
        plan = scheduler.next_execution_plan()
        [prefill] = _dispatched(plan)
        assert list(prefill.request_ids) == ["r0"]
        zeroed = dict(plan.pages_to_zero)
        prefill_rows = _state_rows(prefill, "r0")
        for group, (slot0, slot1) in prefill_rows.items():
            assert slot0 > 0 and slot1 > 0 and slot0 != slot1, (group, slot0, slot1)
            assert slot0 in zeroed[group] and slot1 in zeroed[group], group

        _advance(scheduler, "r0", [201])
        plan = scheduler.next_execution_plan()
        [decode1] = _dispatched(plan)
        zeroed = dict(plan.pages_to_zero)
        decode1_rows = _state_rows(decode1, "r0")
        for group, (slot0, slot1) in decode1_rows.items():
            assert slot0 == prefill_rows[group][1], "the prompt-end state is the input"
            assert slot1 > 0 and slot1 != slot0
            assert zeroed[group] == [slot1], "only the fresh live slot is new"

        _advance(scheduler, "r0", [202])
        _reserve(scheduler, "r0", 1)
        plan = scheduler.next_execution_plan()
        [decode2] = _dispatched(plan)
        for group, row in _state_rows(decode2, "r0").items():
            assert row == [0, decode1_rows[group][1]], (group, row)
        assert all(not pages for pages in dict(plan.pages_to_zero).values())

    def test_a_repeated_prompt_resumes_from_the_published_prompt_end_state(self):
        """The first decode's admission publishes the prompt-end state; a later
        prompt extending the same tokens claims it as slot 0 and skips the prefix."""
        scheduler = ts.Scheduler(_fused_config(decode_input_tokens=1))
        scheduler.submit_requests([_spec("seed", [1, 2, 3, 4], MAX_NEW_TOKENS)])
        [prefill] = _dispatched(scheduler.next_execution_plan())
        _advance(scheduler, "seed", [201])
        [decode1] = _dispatched(scheduler.next_execution_plan())
        checkpoints = {g: row[0] for g, row in _state_rows(decode1, "seed").items()}
        _advance(scheduler, "seed", [202])
        _finish(scheduler, "seed")
        scheduler.next_execution_plan()

        scheduler.submit_requests([_spec("reuse", [1, 2, 3, 4, 5, 6], MAX_NEW_TOKENS)])
        plan = scheduler.next_execution_plan()
        [reuse] = _dispatched(plan)
        assert list(reuse.extend_prefix_lens) == [4]
        assert list(reuse.input_lengths) == [2]
        for group, (slot0, slot1) in _state_rows(reuse, "reuse").items():
            assert slot0 == checkpoints[group], "the published checkpoint is slot 0"
            assert slot1 > 0 and slot1 != slot0
        assert all(
            not pairs for pairs in dict(plan.pages_to_copy).values()
        ), "an aligned hit copies nothing"


class TestStateCheckpointInForward:
    """``SchedulerConfig.state_checkpoint_in_forward``: the forward that spans a
    long prompt's last page boundary before its final token also writes the
    state there, into a third block of its State rows, with no chunk cut; a
    later prompt that changes only that last page resumes there."""

    @pytest.mark.parametrize("in_forward", [False, True])
    def test_a_changed_last_page_resumes_from_the_in_forward_checkpoint(
        self, in_forward
    ):
        cfg = _fused_config(decode_input_tokens=1)
        cfg.max_scheduled_tokens = 4
        cfg.state_checkpoint_in_forward = in_forward
        scheduler = ts.Scheduler(cfg)
        prompt = [1, 2, 3, 4, 5, 6, 7]
        scheduler.submit_requests([_spec("seed", prompt, MAX_NEW_TOKENS)])
        chunks: list[int] = []
        checkpoint_lens: list[int] = []
        rows: list[dict[str, list[int]]] = []
        while sum(chunks) < len(prompt) and len(chunks) < 8:
            [batch] = _dispatched(scheduler.next_execution_plan())
            chunks.append(list(batch.input_lengths)[0])
            checkpoint_lens.append(list(batch.extend_state_checkpoint_lens)[0])
            rows.append(_state_rows(batch, "seed"))
            _advance(scheduler, "seed", [201] if sum(chunks) == len(prompt) else [])
        # No extra round: the second chunk [4, 7) spans the boundary 6 and,
        # with the option, writes the state after its first 2 tokens into a
        # third block of every State row.
        assert chunks == [4, 3]
        assert checkpoint_lens == ([0, 2] if in_forward else [0, 0])
        assert all(len(row) == 2 for row in rows[0].values())
        assert all(len(row) == (3 if in_forward else 2) for row in rows[1].values())
        _dispatched(scheduler.next_execution_plan())
        _advance(scheduler, "seed", [202])
        _finish(scheduler, "seed")
        scheduler.next_execution_plan()

        scheduler.submit_requests([_spec("second", prompt[:6] + [99], MAX_NEW_TOKENS)])
        [batch] = _dispatched(scheduler.next_execution_plan())
        # By default no state was published at 6 and the request starts over.
        assert list(batch.extend_prefix_lens) == [6 if in_forward else 0]
        if in_forward:
            second = _state_rows(batch, "second")
            assert {g: row[0] for g, row in second.items()} == {
                g: row[2] for g, row in rows[1].items()
            }, "resumes from the in-forward checkpoint's blocks"
