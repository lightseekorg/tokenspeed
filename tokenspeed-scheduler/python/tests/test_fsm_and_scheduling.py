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

"""Tests for FSM state transitions and scheduling logic.

Covers:
  - Submitted → Prefilling → Decoding via ScheduleEvent
  - Decoding → Finished via Forward.FinishEvent
  - Chunked prefill: single request split across multiple plans
  - PrefillFirst: prefilling requests take priority over submitted
  - PrefillFirst: req_pool slot budget limits new submitted requests
  - Decode batch: all decoding requests scheduled together
  - advance with output tokens appended to token container
"""

import pytest
from tokenspeed_scheduler import (
    Cache,
    ExecutionEvent,
    ExecutionPlan,
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
    cfg.page_size = page_size
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


def get_forward_op(plan: ExecutionPlan):
    """Return the single FlatForwardOp from a plan, or None."""
    ops = plan.forward
    return ops[0] if ops else None


# ---------------------------------------------------------------------------
# FSM: Submitted → Prefilling → Decoding → Finished
# ---------------------------------------------------------------------------


class TestFSMTransitions:
    def test_invalid_transition_reports_event_and_state(self):
        s = Scheduler(make_config())
        submit(s, "r0", list(range(4)))

        with pytest.raises(
            RuntimeError,
            match=r"FSM transition invalid: event=tokenspeed::fsm::FinishEvent; state=tokenspeed::fsm::Submitted",
        ):
            ev = ForwardEvent.Finish()
            ev.request_id = "r0"
            ec = ExecutionEvent()
            ec.add_event(ev)
            s.advance(ec)

    def test_submit_creates_waiting_request(self):
        """Newly submitted request is in Submitted (waiting) state."""
        s = Scheduler(make_config())
        submit(s, "r0", list(range(8)))
        assert s.waiting_size() == 1
        assert s.prefilling_size() == 0
        assert s.decoding_size() == 0

    def test_first_plan_moves_submitted_to_prefilling(self):
        """After first next_execution_plan, request leaves Submitted (waiting → 0)."""
        s = Scheduler(make_config())
        submit(s, "r0", list(range(8)))
        plan = s.next_execution_plan()
        assert s.waiting_size() == 0
        assert s.decoding_size() == 0
        assert s.prefilling_size() == 1
        assert len(plan.forward) == 1
        op = plan.forward[0]
        assert op.num_extends() > 0

    def test_prefilling_completes_to_decoding(self):
        """After all prefill tokens consumed, ScheduleEvent transitions to Decoding."""
        s = Scheduler(make_config(max_scheduled_tokens=512))
        submit(s, "r0", list(range(8)))

        prefill_plan = (
            s.next_execution_plan()
        )  # Submitted → PrefillDone (8 tokens, all fit)
        assert s.decoding_size() == 0
        assert s.prefilling_size() == 1
        assert len(prefill_plan.forward) == 1
        assert prefill_plan.forward[0].num_extends() > 0

        decode_plan = s.next_execution_plan()  # PrefillDone → Decoding
        assert s.decoding_size() == 1
        assert s.prefilling_size() == 0
        assert len(decode_plan.forward) == 1
        assert decode_plan.forward[0].num_extends() == 0

    def test_finish_event_erases_request(self):
        """advance(finish=True) triggers FinishEvent: Decoding → Finished, erased on next plan."""
        s = Scheduler(make_config(max_scheduled_tokens=512))
        submit(s, "r0", list(range(4)))

        s.next_execution_plan()  # → PrefillDone
        s.next_execution_plan()  # → Decoding  (reserve consumed, new Decoding state has reserve=-1)
        send_reserve_num_tokens(s, "r0", 0)  # reset reserve before next decode plan
        decode_plan = s.next_execution_plan()  # Decoding → Decoding
        assert s.decoding_size() == 1
        assert decode_plan.forward[0].num_extends() == 0

        advance_forward(s, "r0", finish=True)
        s.next_execution_plan()  # r0 erased (FinishEvent was already applied via advance)

        assert s.decoding_size() == 0
        assert s.get_request_token_size("r0") == -1

    def test_output_tokens_appended_via_advance(self):
        """advance with tokens extends the token container."""
        s = Scheduler(make_config(max_scheduled_tokens=512))
        submit(s, "r0", list(range(4)))  # 4 input tokens

        prefill_plan = s.next_execution_plan()  # → Prefilling
        assert prefill_plan.forward[0].num_extends() > 0
        decode_plan = s.next_execution_plan()  # → Decoding
        assert decode_plan.forward[0].num_extends() == 0

        advance_forward(s, "r0", tokens=[99])
        assert s.get_request_token_size("r0") == 5  # 4 input + 1 output


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------


class TestChunkedPrefill:
    def test_large_request_split_across_plans(self):
        """budget=10, request=30 tokens: takes 3 Prefilling steps to finish prefill."""
        s = Scheduler(make_config(max_scheduled_tokens=10))
        submit(s, "r0", list(range(30)))

        plan1 = s.next_execution_plan()  # chunk 1: Submitted → Prefilling
        assert s.prefilling_size() == 1
        assert s.waiting_size() == 0
        assert s.decoding_size() == 0
        assert plan1.forward[0].num_extends() > 0

        plan2 = s.next_execution_plan()  # chunk 2: Prefilling → Prefilling
        assert s.prefilling_size() == 1
        assert s.decoding_size() == 0
        assert plan2.forward[0].num_extends() > 0

        plan3 = s.next_execution_plan()  # chunk 3: Prefilling → PrefillDone
        assert s.prefilling_size() == 1
        assert s.decoding_size() == 0
        assert plan3.forward[0].num_extends() > 0

        plan4 = s.next_execution_plan()  # PrefillDone → Decoding
        assert s.prefilling_size() == 0
        assert s.decoding_size() == 1
        assert plan4.forward[0].num_extends() == 0

        send_reserve_num_tokens(s, "r0", 0)  # reset before next decode step
        plan5 = s.next_execution_plan()
        assert s.decoding_size() == 1
        assert plan5.forward[0].num_extends() == 0

        send_reserve_num_tokens(s, "r0", 0)  # reset before next decode step
        plan6 = s.next_execution_plan()
        assert s.decoding_size() == 1
        assert plan6.forward[0].num_extends() == 0

    def test_chunked_prefill_token_count_unchanged(self):
        """Token count reflects submitted tokens, not computed chunks."""
        s = Scheduler(make_config(max_scheduled_tokens=5))
        submit(s, "r0", list(range(20)))

        plan = s.next_execution_plan()
        assert plan.forward[0].num_extends() > 0
        assert s.get_request_token_size("r0") == 20


# ---------------------------------------------------------------------------
# PrefillFirst: prefilling requests take priority
# ---------------------------------------------------------------------------


class TestPrefillFirst:
    def test_prefilling_scheduled_before_new_submitted(self):
        """A chunked-prefilling request is continued before new submitted requests."""
        s = Scheduler(make_config(max_scheduled_tokens=5))
        submit(s, "r0", list(range(10)))

        plan1 = s.next_execution_plan()  # r0: Submitted → Prefilling (5 tokens)
        assert s.waiting_size() == 0
        assert s.prefilling_size() == 1
        assert s.decoding_size() == 0
        assert plan1.forward[0].num_extends() > 0
        assert plan1.forward[0].request_ids == ["r0"]

        submit(s, "r1", list(range(5)))
        assert s.waiting_size() == 1
        assert s.prefilling_size() == 1

        plan2 = s.next_execution_plan()  # r0 continued (prefilling priority)
        assert s.prefilling_size() == 1
        assert s.waiting_size() == 1
        assert s.decoding_size() == 0
        assert plan2.forward[0].num_extends() > 0
        assert plan2.forward[0].request_ids == ["r0"]

    def test_decode_batch_only_when_no_prefill_work(self):
        """Decode batch is only scheduled when there are no prefilling/submitted requests."""
        s = Scheduler(make_config(max_scheduled_tokens=512))

        submit(s, "r0", list(range(4)))
        s.next_execution_plan()  # → Prefilling
        s.next_execution_plan()  # → Decoding

        submit(s, "r1", list(range(4)))

        plan = s.next_execution_plan()  # Should schedule r1 prefill, not decode r0
        assert s.waiting_size() == 0  # r1 moved out of Submitted
        assert plan.forward[0].num_extends() > 0
        assert plan.forward[0].request_ids == ["r1"]

    def test_mixed_prefill_decode_can_schedule_decode_with_new_prefill(self):
        cfg = make_config(max_scheduled_tokens=512, max_batch_size=8)
        cfg.enable_mixed_prefill_decode = True
        s = Scheduler(cfg)

        submit(s, "r0", list(range(8)))
        s.next_execution_plan()  # r0 → PrefillDone
        s.next_execution_plan()  # r0 → Decoding
        advance_forward(s, "r0", tokens=[99])

        submit(s, "r1", list(range(8)))
        plan = s.next_execution_plan()
        op = plan.forward[0]

        assert op.request_ids == ["r1", "r0"]
        assert op.num_extends() == 1
        assert len(op.input_ids) == sum(op.input_lengths[: op.num_extends()])
        assert len(op.input_ids) + len(op.decode_input_ids) == sum(op.input_lengths)
        assert op.sizes == [1, 0]

    def test_max_batch_size_limits_scheduled_requests(self):
        """max_batch_size caps the number of requests per plan."""
        s = Scheduler(make_config(max_scheduled_tokens=512, max_batch_size=2))
        for i in range(5):
            submit(s, f"r{i}", list(range(4)))
        assert s.waiting_size() == 5

        plan = s.next_execution_plan()
        assert s.waiting_size() >= 3
        assert plan.forward[0].num_extends() > 0
        assert len(plan.forward[0].request_ids) <= 2


# ---------------------------------------------------------------------------
# Decode batch
# ---------------------------------------------------------------------------


class TestDecodeBatch:
    def _get_to_decoding(
        self,
        scheduler: Scheduler,
        request_id: str,
        tokens: list[int],
        decoding_peers: list[str] = (),
    ) -> None:
        """Helper: submit and advance through prefill to reach Decoding.

        decoding_peers: request_ids already in Decoding that need UpdateReserveNumTokens
        before the second next_execution_plan, because the scheduler asserts the value
        is set before scheduling decode.
        """
        submit(scheduler, request_id, tokens)
        scheduler.next_execution_plan()  # → PrefillDone
        for peer in decoding_peers:
            send_reserve_num_tokens(scheduler, peer, 0)
        scheduler.next_execution_plan()  # → Decoding

    def test_multiple_decoding_requests_scheduled_together(self):
        """All decoding requests are included in the same decode batch."""
        s = Scheduler(make_config(max_scheduled_tokens=512))
        self._get_to_decoding(s, "r0", list(range(4)))
        self._get_to_decoding(s, "r1", list(range(4)), decoding_peers=["r0"])
        assert s.decoding_size() == 2

        # Both r0 and r1 are in Decoding; must set reserve before scheduling them.
        send_reserve_num_tokens(s, "r0", 0)
        send_reserve_num_tokens(s, "r1", 0)
        plan = s.next_execution_plan()
        assert s.decoding_size() == 2  # still decoding (no finish event)
        assert len(plan.forward) == 1
        op = plan.forward[0]
        assert op.num_extends() == 0
        assert len(op.request_ids) == 2

    def test_finish_removes_one_of_multiple_decoding(self):
        """Finishing one decoding request doesn't affect others."""
        s = Scheduler(make_config(max_scheduled_tokens=512))
        self._get_to_decoding(s, "r0", list(range(4)))
        self._get_to_decoding(s, "r1", list(range(4)), decoding_peers=["r0"])

        advance_forward(s, "r0", finish=True)
        # r1 is still Decoding and needs reserve; r0 received FinishEvent so it won't be scheduled.
        send_reserve_num_tokens(s, "r1", 0)
        plan = s.next_execution_plan()

        assert s.get_request_token_size("r0") == -1  # erased
        assert s.decoding_size() == 1  # r1 still decoding
        assert plan.forward[0].num_extends() == 0
        assert plan.forward[0].request_ids == ["r1"]

    def test_decode_step_appends_token(self):
        """Each decode advance appends one output token."""
        s = Scheduler(make_config(max_scheduled_tokens=512))
        self._get_to_decoding(s, "r0", list(range(4)))

        for step in range(3):
            send_reserve_num_tokens(
                s, "r0", 0
            )  # must set reserve before each decode plan
            plan = s.next_execution_plan()
            assert plan.forward[0].num_extends() == 0
            advance_forward(s, "r0", tokens=[100 + step])
            assert s.get_request_token_size("r0") == 4 + step + 1


# ---------------------------------------------------------------------------
# Prefill input_ids content
# ---------------------------------------------------------------------------


class TestPrefillInputIds:
    def test_single_request_full_prefill(self):
        """input_ids contains all submitted tokens when prefill fits in one chunk."""
        tokens = list(range(8))
        s = Scheduler(make_config(max_scheduled_tokens=512))
        submit(s, "r0", tokens)

        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.num_extends() > 0
        assert op.input_lengths == [len(tokens)]
        assert op.input_ids == tokens
        assert op.shifted_input_ids == tokens[1:] + [-1]
        assert op.extend_prefix_lens == [0]

    def test_chunked_prefill_input_ids_split_correctly(self):
        """input_ids for each chunk contains only the tokens for that chunk."""
        tokens = list(range(15))
        s = Scheduler(make_config(max_scheduled_tokens=5))
        submit(s, "r0", tokens)

        # chunk 1: tokens[0:5], extend_prefix_len=0
        plan1 = s.next_execution_plan()
        assert plan1.forward[0].input_lengths == [5]
        assert plan1.forward[0].input_ids == tokens[0:5]
        assert plan1.forward[0].shifted_input_ids == tokens[1:6]
        assert plan1.forward[0].extend_prefix_lens == [0]

        # chunk 2: tokens[5:10], extend_prefix_len=5
        plan2 = s.next_execution_plan()
        assert plan2.forward[0].input_lengths == [5]
        assert plan2.forward[0].input_ids == tokens[5:10]
        assert plan2.forward[0].shifted_input_ids == tokens[6:11]
        assert plan2.forward[0].extend_prefix_lens == [5]

        # chunk 3: tokens[10:15], extend_prefix_len=10
        plan3 = s.next_execution_plan()
        assert plan3.forward[0].input_lengths == [5]
        assert plan3.forward[0].input_ids == tokens[10:15]
        assert plan3.forward[0].shifted_input_ids == tokens[11:15] + [-1]
        assert plan3.forward[0].extend_prefix_lens == [10]

    def test_multi_request_prefill_input_ids_concatenated(self):
        """input_ids for a batch of prefill requests is the concatenation of each request's tokens."""
        tokens_by_id = {"r0": list(range(4)), "r1": list(range(10, 14))}
        s = Scheduler(make_config(max_scheduled_tokens=512, max_batch_size=8))
        for rid, tokens in tokens_by_id.items():
            submit(s, rid, tokens)

        plan = s.next_execution_plan()
        op = plan.forward[0]
        assert op.num_extends() > 0
        assert set(op.request_ids) == {"r0", "r1"}
        assert op.input_lengths == [len(tokens_by_id[rid]) for rid in op.request_ids]
        expected_ids = [t for rid in op.request_ids for t in tokens_by_id[rid]]
        assert op.input_ids == expected_ids
        assert op.extend_prefix_lens == [0, 0]

    def test_extend_prefix_lens_nonzero_on_second_chunk(self):
        """extend_prefix_lens reflects how many tokens were already scheduled in prior chunks."""
        tokens = list(range(20))
        s = Scheduler(make_config(max_scheduled_tokens=8))
        submit(s, "r0", tokens)

        plan1 = s.next_execution_plan()
        assert plan1.forward[0].input_lengths == [8]
        assert plan1.forward[0].extend_prefix_lens == [0]
        assert plan1.forward[0].input_ids == tokens[0:8]

        plan2 = s.next_execution_plan()
        assert plan2.forward[0].input_lengths == [8]
        assert plan2.forward[0].extend_prefix_lens == [8]
        assert plan2.forward[0].input_ids == tokens[8:16]

        plan3 = s.next_execution_plan()
        assert plan3.forward[0].input_lengths == [4]
        assert plan3.forward[0].extend_prefix_lens == [16]
        assert plan3.forward[0].input_ids == tokens[16:20]


# ---------------------------------------------------------------------------
# UpdateReserveNumTokens
# ---------------------------------------------------------------------------


def make_update_reserve_event(
    request_id: str, reserve_num_tokens_in_next_schedule_event: int
) -> ExecutionEvent:
    ec = ExecutionEvent()
    ev = ForwardEvent.UpdateReserveNumTokens()
    ev.request_id = request_id
    ev.reserve_num_tokens_in_next_schedule_event = (
        reserve_num_tokens_in_next_schedule_event
    )
    ec.add_event(ev)
    return ec


def send_reserve_num_tokens(scheduler: Scheduler, request_id: str, n: int = 0) -> None:
    """Set reserve_num_tokens_in_next_schedule_event for a Decoding request."""
    scheduler.advance(make_update_reserve_event(request_id, n))


class TestUpdateReserveNumTokens:
    def test_binding_fields_readable_and_writable(self):
        """UpdateReserveNumTokens event fields can be set and read back."""
        ev = ForwardEvent.UpdateReserveNumTokens()
        ev.request_id = "r0"
        ev.reserve_num_tokens_in_next_schedule_event = 42
        assert ev.request_id == "r0"
        assert ev.reserve_num_tokens_in_next_schedule_event == 42

    def test_invalid_transition_on_submitted_request(self):
        """UpdateReserveNumTokens on a Submitted request raises an exception."""
        s = Scheduler(make_config())
        submit(s, "r0", list(range(4)))
        with pytest.raises(Exception):
            s.advance(make_update_reserve_event("r0", 8))

    def test_reserve_causes_extra_page_allocation_on_decode(self):
        """When reserve_num_tokens_in_next_schedule_event exceeds tail capacity, extra pages are allocated."""
        # page_size=16: after prefilling 8 tokens, tail page has 8 free slots.
        # First decode: PrefillDone→Decoding uses PrefillDone's built-in reserve (decode_input_tokens=1),
        # no extra page. Then set reserve=16 for next decode step, which needs 1 extra page.
        s = Scheduler(make_config(page_size=16, num_device_pages=1024))
        submit(s, "r0", list(range(8)))
        s.next_execution_plan()  # → PrefillDone (1 page, 8 slots used, 8 free)
        s.next_execution_plan()  # PrefillDone → Decoding (reserve consumed, new reserve=-1)
        # Now r0 is Decoding. Set reserve=16 before next decode plan.
        s.advance(make_update_reserve_event("r0", 16))

        decode_plan = (
            s.next_execution_plan()
        )  # Decoding: allocates 1 extra page for reserve=16
        op = decode_plan.forward[0]
        assert op.num_extends() == 0
        assert op.sizes[0] == 1

    def test_reserve_update_overrides_previous_value(self):
        """The last UpdateReserveNumTokens wins; earlier values are discarded."""
        s = Scheduler(make_config(page_size=16, num_device_pages=1024))
        submit(s, "r0", list(range(8)))
        s.next_execution_plan()  # → PrefillDone (tail has 8 free slots)
        s.next_execution_plan()  # PrefillDone → Decoding (reserve consumed, new reserve=-1)
        # Now r0 is in Decoding. Both updates apply to Decoding; last one wins.
        s.advance(make_update_reserve_event("r0", 16))  # would need 1 extra page
        s.advance(make_update_reserve_event("r0", 1))  # fits in tail, no extra page

        decode_plan = s.next_execution_plan()
        assert decode_plan.forward[0].sizes[0] == 0

    def test_reserve_larger_than_one_page_allocates_multiple_pages(self):
        """A reserve requiring more than one extra page allocates all of them."""
        # page_size=8: after prefilling 4 tokens, first decode step uses PrefillDone reserve.
        # Then set reserve=20, which should require multiple extra pages on next decode step.
        s = Scheduler(make_config(page_size=8, num_device_pages=1024))
        submit(s, "r0", list(range(4)))
        s.next_execution_plan()  # → PrefillDone
        s.next_execution_plan()  # PrefillDone → Decoding (reserve consumed)
        # Now r0 is Decoding. Set large reserve for next decode plan.
        s.advance(make_update_reserve_event("r0", 20))

        decode_plan = s.next_execution_plan()  # allocates pages for reserve=20
        op = decode_plan.forward[0]
        assert op.num_extends() == 0
        assert op.sizes[0] > 1  # multiple extra pages allocated

    def test_reserve_auto_resets_after_allocation(self):
        """After decode allocates for reserve, subsequent steps need no extra pages if reserve is small."""
        s = Scheduler(make_config(page_size=16, num_device_pages=1024))
        submit(s, "r0", list(range(8)))
        s.next_execution_plan()  # → PrefillDone
        s.next_execution_plan()  # PrefillDone → Decoding (reserve consumed)
        # Set a large reserve; this will allocate 1 extra page next decode step.
        s.advance(make_update_reserve_event("r0", 16))
        s.next_execution_plan()  # Decoding: allocates 1 extra page for reserve=16
        advance_forward(s, "r0", tokens=[99])

        # Next decode: set reserve=0 (small), tail has capacity → no extra page
        s.advance(make_update_reserve_event("r0", 0))
        next_plan = s.next_execution_plan()
        assert next_plan.forward[0].sizes[0] == 0

    def test_reserve_updated_during_decoding(self):
        """UpdateReserveNumTokens applied while already Decoding takes effect next decode step."""
        s = Scheduler(make_config(page_size=16, num_device_pages=1024))
        submit(s, "r0", list(range(8)))
        s.next_execution_plan()  # → PrefillDone
        s.next_execution_plan()  # PrefillDone → Decoding (reserve consumed, tail has 8 free slots)
        # r0 is now Decoding with reserve=-1. Provide reserve=16 for next decode step.
        s.advance(make_update_reserve_event("r0", 16))

        next_plan = s.next_execution_plan()  # allocates 1 extra page for reserve=16
        assert next_plan.forward[0].sizes[0] == 1

    def test_reserve_on_prefill_done_state(self):
        """UpdateReserveNumTokens can only be applied in Decoding state (not PrefillDone).

        PrefillDone carries its own built-in reserve (decode_input_tokens=1 by default).
        To test extra pages, transition to Decoding first, then set a large reserve.
        """
        s = Scheduler(make_config(page_size=16, num_device_pages=1024))
        submit(s, "r0", list(range(8)))
        prefill_plan = s.next_execution_plan()  # → PrefillDone
        assert prefill_plan.forward[0].num_extends() > 0

        # PrefillDone uses its built-in reserve (decode_input_tokens=1): tail has capacity → sizes[0]==0.
        first_decode = s.next_execution_plan()  # PrefillDone → Decoding
        assert first_decode.forward[0].num_extends() == 0
        assert (
            first_decode.forward[0].sizes[0] == 0
        )  # no extra page: reserve=1 fits in tail

        # Now r0 is Decoding. Set reserve=16 which needs 1 extra page.
        s.advance(make_update_reserve_event("r0", 16))
        decode_plan = s.next_execution_plan()
        op = decode_plan.forward[0]
        assert op.num_extends() == 0
        assert op.sizes[0] == 1

    def test_reserve_independent_per_request(self):
        """Reserve settings on one request do not affect another in the same batch."""
        s = Scheduler(
            make_config(page_size=16, num_device_pages=1024, max_batch_size=8)
        )
        # Submit both together so they are both Submitted at the first plan → both become PrefillDone.
        submit(s, "r0", list(range(8)))
        submit(s, "r1", list(range(8)))
        s.next_execution_plan()  # Both r0 and r1 → PrefillDone (processed together)
        # First decode: both PrefillDone → Decoding using built-in reserve.
        s.next_execution_plan()  # Both → Decoding
        # Now both are Decoding with reserve=-1. Set large reserve on r0 only.
        s.advance(make_update_reserve_event("r0", 16))
        s.advance(make_update_reserve_event("r1", 0))  # r1: just clear reserve

        decode_plan = s.next_execution_plan()
        op = decode_plan.forward[0]
        r0_idx = op.request_ids.index("r0")
        r1_idx = op.request_ids.index("r1")
        assert op.sizes[r0_idx] == 1  # r0: extra page needed for reserve=16
        assert op.sizes[r1_idx] == 0  # r1: unaffected


# ---------------------------------------------------------------------------
# DecodeInputIds
# ---------------------------------------------------------------------------


def _make_retract_config(
    page_size: int = 2, num_device_pages: int = 2, num_host_pages: int = 16
) -> SchedulerConfig:
    """Config with very limited device pages to force a retract quickly."""
    cfg = SchedulerConfig()
    cfg.page_size = page_size
    cfg.num_device_pages = num_device_pages
    cfg.num_host_pages = num_host_pages
    cfg.max_scheduled_tokens = 512
    cfg.max_batch_size = 8
    return cfg


def _send_write_back_done(scheduler: Scheduler, op_id: int) -> None:
    ev = Cache.WriteBackDoneEvent()
    ev.op_id = op_id
    ev.success = True
    ec = ExecutionEvent()
    ec.add_event(ev)
    scheduler.advance(ec)


def _writeback_ops(plan: ExecutionPlan):
    return [
        cache_op for cache_op in plan.cache if isinstance(cache_op, Cache.WriteBackOp)
    ]


class TestWriteBackPriority:
    def test_finish_writeback_is_not_retract_writeback(self):
        s = Scheduler(_make_retract_config(page_size=2, num_device_pages=16))
        submit(s, "r0", [10, 11])

        s.next_execution_plan()
        advance_forward(s, "r0", tokens=[42])
        s.next_execution_plan()
        advance_forward(s, "r0", finish=True)

        plan = s.next_execution_plan()
        writebacks = _writeback_ops(plan)

        assert len(writebacks) == 1
        assert writebacks[0].is_retract == [False]

    def test_retract_writeback_is_marked_as_retract(self):
        s = Scheduler(_make_retract_config(page_size=2, num_device_pages=3))
        submit(s, "r0", [10, 11])

        s.next_execution_plan()
        advance_forward(s, "r0", tokens=[42])
        s.next_execution_plan()
        send_reserve_num_tokens(s, "r0", n=4)

        plan = s.next_execution_plan()
        writebacks = _writeback_ops(plan)

        assert len(writebacks) == 1
        assert writebacks[0].is_retract == [True]


class TestDecodeInputIds:
    """Verify that FlatForwardOperation.decode_input_ids is populated correctly."""

    def test_normal_decode_has_minus_one(self):
        """Normal decode (not recovered from Retract) yields decode_input_ids == [-1]."""
        s = Scheduler(make_config(page_size=16, num_device_pages=1024))
        submit(s, "r0", list(range(8)))
        s.next_execution_plan()  # Submitted → PrefillDone
        decode_plan = s.next_execution_plan()  # PrefillDone → Decoding
        op = decode_plan.forward[0]
        assert len(op.decode_input_ids) == 1
        assert op.decode_input_ids[0] == -1

    def test_retract_recovered_carries_last_prefill_token(self):
        """Retract-recovered request carries the last prefill token as decode_input_id."""
        # page_size=2, 4 device pages allow prefill to enter Decoding, then reserve
        # pressure triggers retract.
        s = Scheduler(_make_retract_config(page_size=2, num_device_pages=4))
        last_token = 77
        submit(s, "r0", [10, last_token])  # exactly 1 page

        s.next_execution_plan()  # Submitted → PrefillDone
        advance_forward(s, "r0", tokens=[last_token])
        s.next_execution_plan()  # PrefillDone → Decoding  (device now full)

        # Force retract: reserve=4 needs 2 extra pages, but device is full.
        send_reserve_num_tokens(s, "r0", n=4)
        retract_plan = s.next_execution_plan()

        # ACK the WriteBack op so request transitions to Retracted.
        wb_op_id = None
        for cache_op in retract_plan.cache:
            if hasattr(cache_op, "op_ids") and cache_op.op_ids:
                wb_op_id = cache_op.op_ids[0]
                break
        assert wb_op_id is not None, "Expected a WriteBack cache op"
        _send_write_back_done(s, wb_op_id)

        # Recovery plan: ScheduleDecodeFromRetractedEvent → decode_input_id = last_token.
        recovery_plan = s.next_execution_plan()
        assert recovery_plan.forward, "Expected forward op in recovery plan"
        op = recovery_plan.forward[0]
        assert len(op.decode_input_ids) > 0
        assert (
            op.decode_input_ids[0] == last_token
        ), f"Expected {last_token}, got {op.decode_input_ids[0]}"

    def test_mixed_batch_decode_input_ids_length(self):
        """decode_input_ids has one entry per decode request; all -1 for normal decodes."""
        cfg = make_config(page_size=16, num_device_pages=1024, max_batch_size=8)
        cfg.enable_mixed_prefill_decode = True
        s = Scheduler(cfg)
        # Bring r0 to Decoding.
        submit(s, "r0", list(range(8)))
        s.next_execution_plan()  # r0 → PrefillDone
        s.next_execution_plan()  # r0 → Decoding
        advance_forward(s, "r0", tokens=[99])

        # Submit r1 so that next plan has one prefill + one decode.
        submit(s, "r1", list(range(8)))
        mixed_plan = s.next_execution_plan()
        op = mixed_plan.forward[0]
        num_decodes = len(op.request_ids) - op.num_extends()
        assert len(op.decode_input_ids) == num_decodes
        assert all(did == -1 for did in op.decode_input_ids)
