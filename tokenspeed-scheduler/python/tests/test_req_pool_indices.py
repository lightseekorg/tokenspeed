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

"""Tests for request_pool_indices in ExecutionPlan forward ops.

Covers:
  - Prefill assigns a pool index in [0, max_batch_size)
  - Pool index is stable across prefill and decode phases
  - Concurrent requests get distinct pool slots
  - Pool slot is reused after a request finishes
"""

from tokenspeed_scheduler import (
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def make_config(
    max_scheduled_tokens: int = 512,
    max_batch_size: int = 8,
    page_size: int = 16,
    num_device_pages: int = 1024,
) -> SchedulerConfig:
    cfg = SchedulerConfig()
    cfg.block_size = page_size
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = max_batch_size
    cfg.num_device_pages = num_device_pages
    return cfg


def make_spec(request_id: str, tokens: list[int]) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    return spec


def submit(scheduler: Scheduler, request_id: str, tokens: list[int]) -> None:
    scheduler.submit_requests([make_spec(request_id, tokens)])


def advance_forward(
    scheduler: Scheduler, request_id: str, tokens: list[int] = (), finish: bool = False
) -> None:
    ec = ExecutionEvent()
    if tokens:
        ev = ForwardEvent.ExtendResult()
        ev.request_id = request_id
        ev.tokens = list(tokens)
        ec.add_event(ev)
    if finish:
        ev = ForwardEvent.Finish()
        ev.request_id = request_id
        ec.add_event(ev)
    scheduler.advance(ec)


def send_reserve(scheduler: Scheduler, request_id: str, n: int = 0) -> None:
    """Send UpdateReserveNumTokens to a Decoding request to satisfy the assert."""
    ec = ExecutionEvent()
    ev = ForwardEvent.UpdateReserveNumTokens()
    ev.request_id = request_id
    ev.reserve_num_tokens_in_next_schedule_event = n
    ec.add_event(ev)
    scheduler.advance(ec)


def get_to_decoding(
    scheduler: Scheduler,
    request_id: str,
    tokens: list[int],
    decoding_peers: list[str] = (),
) -> None:
    """Submit and advance through prefill to reach Decoding.

    decoding_peers: request_ids already Decoding that need reserve set before next plan.
    """
    submit(scheduler, request_id, tokens)
    scheduler.next_execution_plan()  # → PrefillDone
    for peer in decoding_peers:
        send_reserve(scheduler, peer, 0)
    scheduler.next_execution_plan()  # → Decoding


def get_forward_op(plan):
    ops = plan.forward
    return ops[0] if ops else None


class TestReqPoolIndices:
    def test_prefill_assigns_pool_index_in_range(self):
        """Each prefilled request gets a pool index in [0, max_batch_size)."""
        s = Scheduler(make_config(max_scheduled_tokens=512, max_batch_size=4))
        submit(s, "r0", list(range(4)))

        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.num_extends() > 0
        assert len(op.request_pool_indices) == 1
        assert op.request_pool_indices[0] == 1

    def test_decode_carries_same_pool_index_as_prefill(self):
        """A request keeps the same pool index across prefill and decode phases."""
        s = Scheduler(make_config(max_scheduled_tokens=512, max_batch_size=4))
        submit(s, "r0", list(range(4)))  # → Submitted

        prefill_plan = s.next_execution_plan()  # → PrefillDone
        prefill_idx = prefill_plan.forward[0].request_pool_indices[0]

        decode_plan = s.next_execution_plan()  # PrefillDone → Decoding
        decode_idx = decode_plan.forward[0].request_pool_indices[0]
        assert prefill_idx == decode_idx

        send_reserve(s, "r0", 0)  # set reserve before second decode plan
        decode_plan = s.next_execution_plan()  # Decoding → Decoding
        decode_idx = decode_plan.forward[0].request_pool_indices[0]

        assert prefill_idx == decode_idx

    def test_two_requests_get_distinct_pool_indices(self):
        """Concurrent requests are assigned distinct pool slots."""
        s = Scheduler(make_config(max_scheduled_tokens=512, max_batch_size=4))
        get_to_decoding(s, "r0", list(range(4)))
        get_to_decoding(s, "r1", list(range(4)), decoding_peers=["r0"])

        # Both r0 and r1 are Decoding; set reserve before plan.
        send_reserve(s, "r0", 0)
        send_reserve(s, "r1", 0)
        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.num_extends() == 0
        assert len(op.request_pool_indices) == 2
        assert op.request_pool_indices[0] != op.request_pool_indices[1]
        for idx in op.request_pool_indices:
            assert 1 <= idx <= 4

    def test_pool_index_reused_after_request_finishes(self):
        """After a request finishes, its pool slot can be reused by a new request."""
        s = Scheduler(make_config(max_scheduled_tokens=512, max_batch_size=2))
        get_to_decoding(s, "r0", list(range(4)))

        send_reserve(s, "r0", 0)  # set reserve before first decode plan
        first_plan = s.next_execution_plan()
        r0_idx = first_plan.forward[0].request_pool_indices[0]

        advance_forward(s, "r0", finish=True)
        s.next_execution_plan()  # erases r0 (FinishEvent already applied)

        submit(s, "r1", list(range(4)))
        new_plan = s.next_execution_plan()  # r1: Submitted → PrefillDone
        r1_idx = new_plan.forward[0].request_pool_indices[0]

        submit(s, "r2", list(range(4)))
        new_plan = (
            s.next_execution_plan()
        )  # r2: Submitted → PrefillDone (r1 also schedules)
        r2_idx = new_plan.forward[0].request_pool_indices[0]

        assert r0_idx == 1
        assert r1_idx == 2
        assert r0_idx == r2_idx
