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

"""D-role State groups: the landing is slot 1, decode runs in place.

A D-role remote admission hands each State group a two-slot row ``[0, slot1]``:
the peer's prompt-end state lands in slot 1 and nothing is read from slot 0
(the D runs no prefill). The first decode keeps the pair and reads slot 1,
later decodes write it in place, so a request never needs a state block after
its admission and a pool filled by remote admissions before any KV lands
still drains.

Shape: K3-shaped 4-group config (1 History + 3 State groups, P = 2 tokens per
block, 32 usable pages), one-block prompts with max_new_tokens = 4.
"""

from __future__ import annotations

import pytest
from conftest import K3_STATE_GROUPS, _make_k3_config

ts = pytest.importorskip("tokenspeed_scheduler")

PROMPT_TOKENS = 2  # exactly one block at P=2
MAX_NEW_TOKENS = 4
NUM_REQUESTS = 5
MAX_ROUNDS = 200
IDLE_LIMIT = 10


def _d_role_config(decode_input_tokens: int) -> "ts.SchedulerConfig":
    cfg = _make_k3_config()
    cfg.role = ts.SchedulerConfig.Role.D
    cfg.decode_input_tokens = decode_input_tokens
    cfg.max_batch_size = 8
    for group in cfg.cache_groups:
        group.transfer_policy = (
            ts.CacheTransferPolicy.LatestSnapshot
            if group.group_id in K3_STATE_GROUPS
            else ts.CacheTransferPolicy.FullSuffix
        )
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


def _land(scheduler, request_id: str) -> None:
    scheduler.advance(
        ts.ExecutionEvent().add_event(ts.PD.RemotePrefillDoneEvent(request_id, 500))
    )


def _state_rows(batch, request_id: str) -> dict[str, list[int]]:
    """The two-slot row per State group for one request row of a Batch."""
    row = list(batch.request_ids).index(request_id)
    tables = dict(batch.block_tables)
    return {g: list(tables[g][row]) for g in K3_STATE_GROUPS}


def _dispatched(plan) -> list:
    """Forward batches that actually carry requests (the plan may hold an empty Batch)."""
    return [op for op in plan.forward if list(op.request_ids)]


def _submit_and_bootstrap(scheduler, rids: list[str]) -> None:
    scheduler.submit_requests(
        [
            _spec(rid, [10 * i + t for t in range(PROMPT_TOKENS)], MAX_NEW_TOKENS)
            for i, rid in enumerate(rids)
        ]
    )
    ec = ts.ExecutionEvent()
    for rid in rids:
        ec.add_event(ts.PD.BootstrappedEvent(rid))
    scheduler.advance(ec)


def _admit_all_then_land(scheduler, rids: list[str]) -> list[str]:
    """Production shape: P is the bottleneck, so D admits (reserves pages for) every
    Submitted prompt it can before any KV lands; then all of them land at once."""
    admitted: list[str] = []
    for _ in range(len(rids) + 2):
        plan = scheduler.next_execution_plan()
        assert not _dispatched(
            plan
        ), "nothing should be decodable before any KV has landed"
        if plan.remote_prefill is None:
            break
        admitted.extend(plan.remote_prefill.request_ids)
    for rid in admitted:
        _land(scheduler, rid)
    return admitted


def _run_closed_loop(scheduler, rids: list[str]) -> dict:
    """Drive the D-role scheduler like the real event loop: every dispatched decode
    step accepts one token, requests finish at MAX_NEW_TOKENS, later admissions land
    immediately.  Stops when everything finished or the loop went idle."""
    generated = {rid: 0 for rid in rids}
    finished: list[str] = []
    idle_rounds = rounds = 0
    while len(finished) < len(rids) and rounds < MAX_ROUNDS:
        rounds += 1
        plan = scheduler.next_execution_plan()
        progressed = False
        if plan.remote_prefill is not None:
            for rid in plan.remote_prefill.request_ids:
                _land(scheduler, rid)
            progressed = True
        for op in _dispatched(plan):
            for rid in op.request_ids:
                generated[rid] += 1
                _advance(scheduler, rid, [200 + generated[rid]])
                if generated[rid] >= MAX_NEW_TOKENS:
                    _finish(scheduler, rid)
                    finished.append(rid)
                else:
                    _reserve(scheduler, rid, 1)
            progressed = True
        idle_rounds = 0 if progressed else idle_rounds + 1
        if idle_rounds >= IDLE_LIMIT:
            break
    return {
        "rounds": rounds,
        "idle_rounds": idle_rounds,
        "finished": finished,
        "generated": generated,
        "waiting": scheduler.waiting_size(),
        "decoding": scheduler.decoding_size(),
        "available": scheduler.available_lcm_blocks(),
        "active": scheduler.active_lcm_blocks(),
    }


class TestDRoleStateSlots:
    def test_pool_filled_by_remote_admissions_before_landing_never_wedges(self):
        """Every admitted request owns the slot its decodes write; once the KV
        lands, all of them run to completion and release every page."""
        scheduler = ts.Scheduler(_d_role_config(decode_input_tokens=1))
        rids = [f"r{i}" for i in range(NUM_REQUESTS)]
        _submit_and_bootstrap(scheduler, rids)
        admitted = _admit_all_then_land(scheduler, rids)
        assert admitted, "no remote admission happened at all"

        result = _run_closed_loop(scheduler, rids)

        assert len(result["finished"]) == len(rids), (
            "D-role deadlock: landed requests never finished; "
            f"admitted={admitted} finished={result['finished']} generated={result['generated']} "
            f"waiting={result['waiting']} decoding={result['decoding']} "
            f"available={result['available']} active={result['active']} idle_rounds={result['idle_rounds']}"
        )
        assert result["active"] == 0, "finished requests must release every page"

    def test_remote_admission_lands_the_state_in_slot_one(self):
        """The landing row is [0, slot1]: no checkpoint to read, a fresh block
        for the peer's prompt-end state, and no state transfer skipped."""
        scheduler = ts.Scheduler(_d_role_config(decode_input_tokens=1))
        _submit_and_bootstrap(scheduler, ["r0"])
        plan = scheduler.next_execution_plan()
        assert plan.remote_prefill is not None and list(
            plan.remote_prefill.request_ids
        ) == ["r0"]
        assert list(plan.remote_prefill.state_prefix_cached) == [False]
        zeroed = dict(plan.pages_to_zero)
        for group, (slot0, slot1) in _state_rows(plan.remote_prefill, "r0").items():
            assert slot0 == 0, group
            assert slot1 > 0 and slot1 in zeroed[group], group

    def test_landed_request_decodes_in_place_with_an_exhausted_pool(self):
        """With the pool otherwise exhausted, every freshly landed prompt takes
        its first decode step and keeps writing the landed slot in place."""
        scheduler = ts.Scheduler(_d_role_config(decode_input_tokens=1))
        rids = [f"r{i}" for i in range(NUM_REQUESTS)]
        _submit_and_bootstrap(scheduler, rids)
        admitted = _admit_all_then_land(scheduler, rids)
        assert admitted
        plan = scheduler.next_execution_plan()
        dispatched = [rid for op in _dispatched(plan) for rid in op.request_ids]
        assert set(dispatched) == set(admitted), (
            f"first decode step blocked for {sorted(set(admitted) - set(dispatched))}: "
            f"available={scheduler.available_lcm_blocks()} active={scheduler.active_lcm_blocks()}"
        )
        [batch] = _dispatched(plan)
        landed = {rid: _state_rows(batch, rid) for rid in admitted}
        for rid in admitted:
            _advance(scheduler, rid, [201])
            _reserve(scheduler, rid, 1)
        [batch] = _dispatched(scheduler.next_execution_plan())
        for rid in admitted:
            assert _state_rows(batch, rid) == landed[rid], rid
            for row in landed[rid].values():
                assert row[0] == 0 and row[1] > 0


class TestFullRemoteAdmissionQueue:
    def test_full_remote_queue_resumes_after_eager_resident_release(self):
        """A full slot pool must retain queued remote work until release."""
        config = _d_role_config(decode_input_tokens=1)
        config.max_batch_size = 1
        scheduler = ts.Scheduler(config)
        _submit_and_bootstrap(scheduler, ["resident", "queued"])
        first = scheduler.next_execution_plan().remote_prefill
        assert list(first.request_ids) == ["resident"]
        slot = int(first.request_pool_indices[0])
        for _ in range(4):
            plan = scheduler.next_execution_plan()
            assert plan.remote_prefill is None
            assert not plan.aborts
        _land(scheduler, "resident")
        [batch] = _dispatched(scheduler.next_execution_plan())
        assert list(batch.request_ids) == ["resident"]
        _advance(scheduler, "resident", [201])
        _finish(scheduler, "resident")
        plan = scheduler.next_execution_plan()
        assert list(plan.remote_prefill.request_ids) == ["queued"]
        assert list(plan.remote_prefill.request_pool_indices) == [slot]
        _land(scheduler, "queued")
        [batch] = _dispatched(scheduler.next_execution_plan())
        assert list(batch.request_ids) == ["queued"]
