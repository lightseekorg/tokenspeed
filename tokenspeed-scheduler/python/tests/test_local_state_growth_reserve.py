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

"""Local (Fused-role) counterpart of the D-role state-group growth reserve.

An aligned prompt has no tail to split, so its completing admission used to
materialize one state block with zero spare and its first decode step needed a
fresh EMPTY parent per state group -- the D-role wedge, on the local path.
``reserveSnapshotStateGrowth`` banks one growth block on every admission that
finishes shaping a state group, so the first decode step is consumed in place.

Shape: K3-shaped 4-group config (P = 2 tokens per block, 32 usable pages),
one-block aligned prompts, max_new_tokens = 4.  Unfixed: 3 History pages + 3
state blocks = 6 pages per admission, five of six prompts admit (30/32) and
wedge at their first decode.  Fixed: 9 pages per admission, three admit, three
queue, everything finishes.
"""

from __future__ import annotations

import pytest
from conftest import K3_STATE_GROUPS, _make_k3_config

ts = pytest.importorskip("tokenspeed_scheduler")

PROMPT_TOKENS = 2  # exactly one block at P=2 -> aligned, no tail to split
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


def _state_blocks(batch, request_id: str) -> dict[str, list[int]]:
    """Positive block ids per snapshot-state group for one request row of a Batch."""
    row = list(batch.request_ids).index(request_id)
    tables = dict(batch.block_tables)
    return {g: [p for p in tables[g][row] if p > 0] for g in K3_STATE_GROUPS}


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
        "available": scheduler.available_kv_pages(),
        "active": scheduler.active_kv_pages(),
    }


class TestLocalStateGroupGrowthReserve:
    def test_aligned_prompts_filling_the_pool_never_wedge(self):
        """Six aligned one-block prompts.  Unfixed: five admit (30/32 pages) and
        every first decode step needs 3 empty parents while 2 exist -> no forward op
        is ever produced again.  Fixed: admission back-pressures at three and every
        request runs to completion."""
        scheduler = ts.Scheduler(_fused_config(decode_input_tokens=1))
        rids = [f"r{i}" for i in range(NUM_REQUESTS)]
        _submit(scheduler, rids)
        result = _run_closed_loop(scheduler, rids)
        assert result["idle_rounds"] < IDLE_LIMIT, f"scheduler went idle: {result}"
        assert sorted(result["finished"]) == sorted(rids), result
        assert result["waiting"] == 0 and result["decoding"] == 0
        assert result["active"] == 0

    def test_completing_admission_banks_one_growth_block_per_state_group(self):
        """An aligned two-block prompt ends its (single-chunk) prefill owning the
        endpoint block plus one banked growth block in every state group, and the
        first decode step reuses that block instead of acquiring a fresh one."""
        scheduler = ts.Scheduler(_fused_config(decode_input_tokens=1))
        scheduler.submit_requests([_spec("r0", [1, 2, 3, 4], MAX_NEW_TOKENS)])
        prefill = _dispatched(scheduler.next_execution_plan())
        assert len(prefill) == 1 and list(prefill[0].request_ids) == ["r0"]
        banked = _state_blocks(prefill[0], "r0")
        for group, blocks in banked.items():
            assert len(blocks) == 2, (group, blocks)
        # PrefillDone already carries the decode reserve for the first step.
        _advance(scheduler, "r0", [201])
        decode = _dispatched(scheduler.next_execution_plan())
        assert len(decode) == 1 and list(decode[0].request_ids) == ["r0"]
        for group, blocks in _state_blocks(decode[0], "r0").items():
            assert blocks == banked[group], (group, blocks, banked[group])
