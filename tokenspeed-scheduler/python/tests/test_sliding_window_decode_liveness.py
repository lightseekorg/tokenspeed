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

"""Sliding-window decode liveness at KV exhaustion: the sliding-window prepay
and the victim policy's last resort (docs/design/scheduler.md §1 and §2)."""

from __future__ import annotations

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

PAGE = 16  # prefix granularity == block granularity of both groups
STALL_ROUNDS = 50
MAX_ROUNDS = 2000
SLIDING_GROUP = "sliding_attention"
FULL_GROUP = "full_attention"


def _config(
    num_usable_pages: int,
    *,
    window: int,
    prefix_cache: bool,
    overlap_schedule_depth: int,
    max_scheduled_tokens: int,
) -> "ts.SchedulerConfig":
    """Fused role, one sliding-window and one full-history group drawing one
    page each from the same LCM pool."""
    cfg = ts.SchedulerConfig()
    cfg.num_device_pages = num_usable_pages + 1  # page 0 is the null sentinel
    cfg.num_host_pages = 0
    cfg.disable_l2_cache = True
    cfg.enable_l3_storage = False
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = 8
    cfg.prefix_granularity = PAGE
    cfg.decode_input_tokens = 1
    cfg.overlap_schedule_depth = overlap_schedule_depth
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.disable_prefix_cache = not prefix_cache
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=SLIDING_GROUP,
            block_granularity=PAGE,
            total_pages=cfg.num_device_pages,
            retention=ts.CacheRetention.SlidingWindow,
            sliding_window_tokens=window,
        ),
        ts.CacheGroupConfig(
            group_id=FULL_GROUP,
            block_granularity=PAGE,
            total_pages=cfg.num_device_pages,
            retention=ts.CacheRetention.FullHistory,
        ),
    ]
    return cfg


def _d_role(cfg: "ts.SchedulerConfig") -> "ts.SchedulerConfig":
    """The same groups on a decode worker, whose peer sends every group's
    whole suffix."""
    cfg.role = ts.SchedulerConfig.Role.D
    for group in cfg.cache_groups:
        group.transfer_policy = ts.CacheTransferPolicy.FullSuffix
    return cfg


def _spec(request_id: str, tokens: list[int], max_new_tokens: int) -> "ts.RequestSpec":
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(tokens)
    spec.max_new_tokens = max_new_tokens
    return spec


def _run_closed_loop(
    num_usable_pages: int,
    *,
    num_requests: int,
    shared_tokens: int,
    unique_tokens: int,
    max_new_tokens: int,
    window: int,
    prefix_cache: bool,
    decode_worker: bool = False,
) -> dict:
    """Drive the scheduler like the event loop with overlap depth 1 (a plan's
    results land one round later, each row yields one token) until every
    request reaches max_new_tokens, or report a stall after STALL_ROUNDS
    consecutive rounds without a forward batch. Request k's prompt is the
    shared prefix plus unique_tokens of its own. The first request arrives two
    rounds before the rest, so its prompt pages are published by the time the
    others are admitted. On a decode worker every request is bootstrapped on
    arrival and its peer's prefill lands as soon as it is planned."""
    overlap_schedule_depth = 1
    cfg = _config(
        num_usable_pages,
        window=window,
        prefix_cache=prefix_cache,
        overlap_schedule_depth=overlap_schedule_depth,
        max_scheduled_tokens=1024,
    )
    scheduler = ts.Scheduler(_d_role(cfg) if decode_worker else cfg)

    def submit(specs: list) -> None:
        scheduler.submit_requests(specs)
        if decode_worker:
            events = ts.ExecutionEvent()
            for spec in specs:
                events.add_event(ts.PD.BootstrappedEvent(spec.request_id))
            scheduler.advance(events)

    shared = [100 + i for i in range(shared_tokens)]
    prompts = {
        f"r{k}": shared + [5000 + 100 * k + j for j in range(unique_tokens)]
        for k in range(num_requests)
    }
    specs = [_spec(rid, tokens, max_new_tokens) for rid, tokens in prompts.items()]
    submit(specs[:1])
    later = specs[1:]

    generated = {rid: 0 for rid in prompts}
    finished: set[str] = set()
    # Prefill rows of requests that already produced output. Only a retracted
    # request prefills again, once per chunk of every readmission, including
    # one retracted again before its next token.
    reprefill_rows = 0
    in_flight: list = []
    idle_rounds = 0
    for round_index in range(MAX_ROUNDS):
        if round_index == 2 and later:
            submit(later)
            later = []
        plan = scheduler.next_execution_plan()
        if plan.remote_prefill is not None:
            landed = ts.ExecutionEvent()
            for rid in plan.remote_prefill.request_ids:
                landed.add_event(ts.PD.RemotePrefillDoneEvent(rid, 1))
            scheduler.advance(landed)
        batch = next((op for op in plan.forward if list(op.request_ids)), None)
        if batch is not None:
            in_flight.append(batch)
            idle_rounds = 0
            prefill_ids = list(batch.request_ids)[: batch.num_extends()]
            reprefill_rows += sum(generated[rid] > 0 for rid in prefill_ids)
        else:
            idle_rounds += 1

        # Results land overlap_schedule_depth rounds behind; a round without
        # a batch drains everything still out, as the event loop does.
        events = ts.ExecutionEvent()
        num_events = 0
        while len(in_flight) > (overlap_schedule_depth if batch is not None else 0):
            done = in_flight.pop(0)
            num_extends = done.num_extends()
            prefill_lengths = list(done.prefill_lengths)
            for row, rid in enumerate(done.request_ids):
                if rid in finished:
                    continue
                result = ts.ForwardEvent.ExtendResult()
                result.request_id = rid
                if (
                    row < num_extends
                    and done.extend_prefix_lens[row] + done.input_lengths[row]
                    < prefill_lengths[row]
                ):
                    # An intermediate prefill chunk produces no token.
                    result.tokens = []
                    events.add_event(result)
                    num_events += 1
                    continue
                generated[rid] += 1
                result.tokens = [9000 + generated[rid]]
                events.add_event(result)
                num_events += 1
                if generated[rid] >= max_new_tokens:
                    finished.add(rid)
                    finish = ts.ForwardEvent.Finish()
                    finish.request_id = rid
                    events.add_event(finish)
                elif row >= num_extends:
                    # Only decode rows update the reserve: the row that
                    # completes the prompt already carries the decode slot.
                    reserve = ts.ForwardEvent.UpdateReserveNumTokens()
                    reserve.request_id = rid
                    reserve.reserve_num_tokens_in_next_schedule_event = 1
                    events.add_event(reserve)
        if num_events:
            scheduler.advance(events)

        state = {
            "rounds": round_index + 1,
            "reprefill_rows": reprefill_rows,
            "generated": dict(generated),
            "waiting": scheduler.waiting_size(),
            "decoding": scheduler.decoding_size(),
            "active": scheduler.active_lcm_blocks(),
            "empty": scheduler.empty_lcm_blocks(),
        }
        if len(finished) == len(prompts) and not later:
            return {"outcome": "done", **state}
        if idle_rounds >= STALL_ROUNDS:
            return {"outcome": "stall", **state}
    return {"outcome": "max_rounds", **state}


class TestSlidingWindowDecodeLiveness:
    def test_shared_prefix_hits_finish_without_retraction(self):
        """Six requests share a 160-token prefix (10 pages) and hit it at
        admission; the 4-page window lookback they share slides out as they
        decode. Unfixed: admission charges each hit request's sliding-window
        group its tail and the decode slot only, the pool fills and the stall
        hits with 4 requests decoding and 2 waiting. Fixed: the hit requests
        prepay their window growth, fewer run at once, and nobody has to be
        retracted."""
        result = _run_closed_loop(
            40,
            num_requests=6,
            shared_tokens=160,
            unique_tokens=8,
            max_new_tokens=64,
            window=64,
            prefix_cache=True,
        )
        assert result["outcome"] == "done", result
        assert result["reprefill_rows"] == 0, result
        assert result["active"] == 0, result

    def test_request_blocked_beside_running_decodes_is_not_retracted(self):
        """Six requests share a 160-token prefix under a 128-token window. The
        second one computes the shared prefix itself and publishes it after
        its own admission; the later ones hit those pages, so its reserve
        does not cover the growth over them, and it blocks at a window page
        boundary while four others decode. The last resort must not take it:
        the round runs work whose completions free capacity. It waits two
        rounds and nobody is prefilled again."""
        result = _run_closed_loop(
            64,
            num_requests=6,
            shared_tokens=160,
            unique_tokens=40,
            max_new_tokens=32,
            window=128,
            prefix_cache=True,
        )
        assert result["outcome"] == "done", result
        assert result["reprefill_rows"] == 0, result
        assert result["active"] == 0, result

    def test_deduplicated_prompt_pages_finish_without_prefix_cache(self):
        """With the prefix cache disabled nothing hits at admission, but
        publishing the completed prompt pages still dedupes identical ones
        onto a single shared page. Unfixed: eight requests sharing a 96-token
        prefix stall with the pool full, 7 decoding and 1 waiting. Fixed: the
        last resort retracts one exempt request and its pages go to the other
        blocked decodes, not to the waiting prompt, which would need three
        victims at once; the bound on re-prefill rows keeps that from turning
        into retract/readmit churn."""
        result = _run_closed_loop(
            64,
            num_requests=8,
            shared_tokens=96,
            unique_tokens=8,
            max_new_tokens=64,
            window=64,
            prefix_cache=False,
        )
        assert result["outcome"] == "done", result
        assert result["reprefill_rows"] <= 2, result
        assert result["active"] == 0, result

    @pytest.mark.parametrize(
        "decode_worker", [False, True], ids=["fused", "decode_worker"]
    )
    def test_distinct_prompts_shorter_than_window_finish(self, decode_worker: bool):
        """Eight distinct 104-token prompts under a 128-token window share no
        page, but nothing slides out before a sequence outgrows the window,
        so their early decode growth has nothing to recycle and only the
        last resort keeps the scheduler live. Unfixed: stall with the pool
        full. Fixed: the last resort retracts one decoding request and its
        pages go to the other blocked decodes. Handed to a waiting prompt
        instead, they admit it, and more decoding requests are retracted later
        to readmit the first; the bound on re-prefill rows catches that."""
        result = _run_closed_loop(
            64,
            num_requests=8,
            shared_tokens=0,
            unique_tokens=104,
            max_new_tokens=32,
            window=128,
            prefix_cache=True,
            decode_worker=decode_worker,
        )
        assert result["outcome"] == "done", result
        assert result["reprefill_rows"] <= 2, result
        assert result["active"] == 0, result


def _submit_and_step(
    scheduler, request_id: str, tokens: list[int], max_new_tokens: int
):
    scheduler.submit_requests([_spec(request_id, tokens, max_new_tokens)])
    return scheduler.next_execution_plan()


def _row_of(plan, request_id: str) -> tuple:
    batch = next(op for op in plan.forward if request_id in list(op.request_ids))
    row = list(batch.request_ids).index(request_id)
    tables = {
        group: list(rows[row]) for group, rows in dict(batch.block_tables).items()
    }
    return batch, row, tables


def _land(
    scheduler, request_id: str, token: int, *, decode_reserve: int | None
) -> None:
    events = ts.ExecutionEvent()
    result = ts.ForwardEvent.ExtendResult()
    result.request_id = request_id
    result.tokens = [token]
    events.add_event(result)
    if decode_reserve is not None:
        reserve = ts.ForwardEvent.UpdateReserveNumTokens()
        reserve.request_id = request_id
        reserve.reserve_num_tokens_in_next_schedule_event = decode_reserve
        events.add_event(reserve)
    scheduler.advance(events)


class TestSlidingWindowPrepay:
    WINDOW = 64  # 4 pages; the lookback a token reads is 63 tokens

    def test_prefix_hit_prepays_growth_over_shared_lookback(self):
        """A prompt admitted without a hit holds only the decode slot in the
        sliding-window group. A prompt that hits its 160-token prefix while
        the first request still holds it shares the 4-page lookback and
        prepays the growth over it, capped at the 63-token lookback: 8 + 63
        tokens, 5 private pages (unfixed, 8 + 1 tokens in 1 page). The
        full-history group prepays the whole generation either way."""
        scheduler = ts.Scheduler(
            _config(
                64,
                window=self.WINDOW,
                prefix_cache=True,
                overlap_schedule_depth=0,
                max_scheduled_tokens=1024,
            )
        )
        shared = [100 + i for i in range(160)]
        tail = 8

        # First request: no hit, so the sliding-window group reserves only
        # the decode slot (168 + 1 tokens -> 11 pages).
        plan = _submit_and_step(
            scheduler, "a", shared + [7000 + j for j in range(tail)], 200
        )
        _, _, first = _row_of(plan, "a")
        assert len(first[SLIDING_GROUP]) == (160 + tail + 1 + PAGE - 1) // PAGE
        first_pages = {page for page in first[SLIDING_GROUP] if page > 0}
        _land(scheduler, "a", 1, decode_reserve=None)
        # Scheduling its first decode publishes the prompt pages.
        plan = scheduler.next_execution_plan()
        assert [
            list(op.request_ids) for op in plan.forward if list(op.request_ids)
        ] == [["a"]]
        _land(scheduler, "a", 2, decode_reserve=1)

        plan = _submit_and_step(
            scheduler, "b", shared + [8000 + j for j in range(tail)], 200
        )
        batch, row, second = _row_of(plan, "b")
        assert batch.extend_prefix_lens[row] == 160
        assert batch.input_lengths[row] == tail

        sliding = [page for page in second[SLIDING_GROUP] if page > 0]
        shared_lookback = [page for page in sliding if page in first_pages]
        private = [page for page in sliding if page not in first_pages]
        assert len(shared_lookback) == self.WINDOW // PAGE, second[SLIDING_GROUP]
        assert len(private) == 5, second[SLIDING_GROUP]
        assert len(second[SLIDING_GROUP]) == 160 // PAGE + 5
        assert len(second[FULL_GROUP]) == len(first[FULL_GROUP])

    def test_prefix_hit_alone_prepays_nothing(self):
        """With no other request holding pages, a prompt that hits the
        160-token prefix a finished request left cached prepays nothing: its
        sliding-window group holds 8 + 1 tokens in 1 private page, so a
        request alone asks for what max_single_request_tokens counts."""
        scheduler = ts.Scheduler(
            _config(
                64,
                window=self.WINDOW,
                prefix_cache=True,
                overlap_schedule_depth=0,
                max_scheduled_tokens=1024,
            )
        )
        shared = [100 + i for i in range(160)]
        _submit_and_step(scheduler, "a", shared + [7000 + j for j in range(8)], 1)
        _land(scheduler, "a", 1, decode_reserve=None)
        finish = ts.ForwardEvent.Finish()
        finish.request_id = "a"
        scheduler.advance(ts.ExecutionEvent().add_event(finish))

        plan = _submit_and_step(
            scheduler, "b", shared + [8000 + j for j in range(8)], 200
        )
        batch, row, tables = _row_of(plan, "b")
        assert batch.extend_prefix_lens[row] == 160
        assert len(tables[SLIDING_GROUP]) == 160 // PAGE + 1, tables[SLIDING_GROUP]

    def test_first_chunk_prepays_the_lookback_its_next_chunk_reads(self):
        """A 200-token prompt in 64-token chunks beside a resident request
        that holds pages: the second chunk reads the first one's 63-token
        lookback before any page of it can slide out, so the first chunk
        prepays it: 64 + 63 tokens, 8 sliding-window pages (unfixed, the
        chunk's 4)."""
        scheduler = ts.Scheduler(
            _config(
                64,
                window=self.WINDOW,
                prefix_cache=True,
                overlap_schedule_depth=0,
                max_scheduled_tokens=64,
            )
        )
        _submit_and_step(scheduler, "a", [7000 + j for j in range(8)], 40)
        _land(scheduler, "a", 1, decode_reserve=None)

        plan = _submit_and_step(scheduler, "b", [5000 + j for j in range(200)], 40)
        batch, row, tables = _row_of(plan, "b")
        assert batch.extend_prefix_lens[row] == 0
        assert batch.input_lengths[row] == 64
        assert len(tables[SLIDING_GROUP]) == 8, tables[SLIDING_GROUP]
        # 64 + the 136-token rest of the prompt + 40 tokens of headroom.
        assert len(tables[FULL_GROUP]) == 15

    def test_remote_landing_prepays_no_later_chunk_beside_a_resident(self):
        """D role. The first request lands and keeps decoding; the second
        shares only its first 32 tokens, too few for the sliding-window group
        to resume from, so its landing is admitted up to that promotion
        boundary. The peer still fills the whole prompt: there is no later
        local chunk whose lookback the landing would prepay, even beside a
        resident. Its sliding-window group holds the window tail of the
        104-token prompt (tokens 41-103, pages 2-6) and nothing more;
        prepaying a lookback would ask for 4 more pages than the 40-page pool
        has left."""
        scheduler = ts.Scheduler(
            _d_role(
                _config(
                    40,
                    window=self.WINDOW,
                    prefix_cache=True,
                    overlap_schedule_depth=0,
                    max_scheduled_tokens=1024,
                )
            )
        )
        shared = [100 + i for i in range(32)]
        scheduler.submit_requests(
            [_spec("a", shared + [7000 + j for j in range(72)], 200)]
        )
        scheduler.advance(ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent("a")))
        assert scheduler.next_execution_plan().remote_prefill is not None
        scheduler.advance(
            ts.ExecutionEvent().add_event(ts.PD.RemotePrefillDoneEvent("a", 1))
        )
        # Scheduling its first decode publishes the landed prompt pages.
        plan = scheduler.next_execution_plan()
        assert [
            list(op.request_ids) for op in plan.forward if list(op.request_ids)
        ] == [["a"]]
        _land(scheduler, "a", 1, decode_reserve=1)

        scheduler.submit_requests(
            [_spec("b", shared + [8000 + j for j in range(72)], 16)]
        )
        scheduler.advance(ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent("b")))
        landing = scheduler.next_execution_plan().remote_prefill
        assert landing is not None and list(landing.request_ids) == ["b"]
        assert landing.input_lengths[0] == 32
        sliding = list(dict(landing.block_tables)[SLIDING_GROUP][0])
        held_slots = [slot for slot, page in enumerate(sliding) if page > 0]
        assert held_slots == [2, 3, 4, 5, 6], sliding
