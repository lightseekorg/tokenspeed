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

"""Tests for occupied_pages, begins, and sizes in FlatForwardOp.

Covers:
  - Prefill allocates pages; occupied_pages contains all held page indices,
    begins/sizes describe the newly allocated slice
  - begins is 0 on first schedule (no pages were occupied before)
  - Chunked prefill: begins grows across chunks as more pages are allocated
  - Decode step: no new pages allocated when tail page still has capacity
  - Multiple requests in one batch each get their own page lists
"""

from tokenspeed_scheduler import (
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Submit and advance through prefill to Decoding.

    decoding_peers: requests already Decoding that need reserve set before next plan.
    """
    submit(scheduler, request_id, tokens)
    scheduler.next_execution_plan()  # → PrefillDone
    for peer in decoding_peers:
        send_reserve(scheduler, peer, 0)
    scheduler.next_execution_plan()  # → Decoding


def get_forward_op(plan):
    ops = plan.forward
    return ops[0] if ops else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOccupiedPages:
    def test_occupied_pages_length_per_request(self):
        """occupied_pages has one entry per request in the batch."""
        s = Scheduler(make_config(page_size=16, max_scheduled_tokens=512))
        submit(s, "r0", list(range(16)))

        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.num_extends() > 0
        assert len(op.occupied_pages) == 1

    def test_begins_zero_on_first_schedule(self):
        """On first schedule (Submitted state), no pages were occupied before, so begins=0."""
        s = Scheduler(make_config(page_size=16, max_scheduled_tokens=512))
        submit(s, "r0", list(range(16)))

        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.begins == [0]

    def test_prefill_allocates_nonzero_pages(self):
        """First prefill of a request allocates at least 1 page."""
        s = Scheduler(make_config(page_size=16, max_scheduled_tokens=512))
        submit(s, "r0", list(range(16)))

        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.num_extends() > 0
        assert op.sizes[0] > 0
        assert (
            len(list(op.occupied_pages[0])[op.begins[0] : op.begins[0] + op.sizes[0]])
            > 0
        )

    def test_chunked_prefill_begins_grows(self):
        """After first chunk allocates pages, begins on second chunk reflects that."""
        # page_size=16, budget=16 per step → chunk1 fills 1 page, chunk2 fills another
        s = Scheduler(make_config(page_size=16, max_scheduled_tokens=16))
        submit(s, "r0", list(range(32)))

        plan1 = s.next_execution_plan()
        op1 = plan1.forward[0]
        assert op1.begins == [0]
        assert op1.sizes[0] > 0

        plan2 = s.next_execution_plan()
        op2 = plan2.forward[0]
        # After chunk1, some pages were already occupied → begins > 0
        assert op2.begins[0] > 0

    def test_decode_no_new_pages_when_tail_has_capacity(self):
        """Decode steps don't allocate new pages when the tail page still has space."""
        s = Scheduler(make_config(page_size=16, max_scheduled_tokens=512))
        submit(s, "r0", list(range(8)))

        s.next_execution_plan()  # → PrefillDone
        decode_plan = s.next_execution_plan()  # → Decoding
        op = decode_plan.forward[0]
        assert op.num_extends() == 0
        assert op.sizes[0] == len(
            list(op.occupied_pages[0])[op.begins[0] : op.begins[0] + op.sizes[0]]
        )

    def test_two_requests_independent_page_lists(self):
        """Each request in a batch gets its own occupied_pages entry."""
        s = Scheduler(
            make_config(page_size=16, max_scheduled_tokens=512, max_batch_size=8)
        )
        # Submit both together so they are both Submitted at first plan → both get PrefillDone,
        # no Decoding peer issue.
        submit(s, "r0", list(range(16)))
        submit(s, "r1", list(range(16)))

        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.num_extends() > 0
        assert len(op.occupied_pages) == 2
        assert len(op.begins) == 2
        assert op.begins[0] == 0
        assert op.begins[1] == 0
        # The two requests must not share the same physical pages
        new_pages_r0 = set(
            list(op.occupied_pages[0])[op.begins[0] : op.begins[0] + op.sizes[0]]
        )
        new_pages_r1 = set(
            list(op.occupied_pages[1])[op.begins[1] : op.begins[1] + op.sizes[1]]
        )
        assert new_pages_r0.isdisjoint(new_pages_r1)

    def test_no_new_page_when_tail_enough(self):
        s = Scheduler(
            make_config(page_size=20, max_scheduled_tokens=512, max_batch_size=8)
        )
        submit(s, "r0", list(range(16)))
        _ = s.next_execution_plan()  # Submitted → PrefillDone
        plan = (
            s.next_execution_plan()
        )  # PrefillDone → Decoding (no new page, tail has 4 free slots)
        ops = plan.forward[0]
        assert ops.sizes[0] == 0
        send_reserve(s, "r0", 0)  # set reserve before next decode plan
        plan = s.next_execution_plan()  # Decoding → Decoding (still no new page)
        ops = plan.forward[0]
        assert ops.sizes[0] == 0
