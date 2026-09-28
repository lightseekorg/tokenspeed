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

"""Decode liveness with a sliding-window cache group beside a full-history one.

A sliding-window group funds its growth by recycling the pages that slide out
of the window, so admission reserves it little beyond the chunk. Recycling
cannot fund growth when the slid-out pages are shared with another request (a
prefix hit, or publish-time dedup of identical prompt pages), because releasing
a shared page frees nothing, nor while nothing can slide out yet: the sequence
is still shorter than the window, or the next prompt chunk still reads this
one's lookback. Once the pool is full, every step that needs a new page fails
admission, while the victim policy exempts every request whose admission
headroom covers its whole generation. Unfixed, nothing is retracted, every plan
comes back empty and no request can ever finish.

Fixed by two changes. When a round builds nothing and no resident can be
retracted under the exemption, the victim policy falls back to the exempt
requests, and the pages it frees go first to that round's decodes, none of
which got one: a generated token survives any later retraction, while a prompt
given the pages can block again, or the victim itself take them back, round
after round. While another request holds pages, a first-chunk admission also
prepays the sliding-window growth that recycling cannot fund: over a prefix
hit, and over the rest of the prompt when a local chunk leaves some behind,
capped by the window's lookback and the prompt headroom. Alone, a request
prepays nothing and asks for what it asks unfixed, so the prepay never makes a
request that max_single_request_tokens accepts inadmissible on its own.

The liveness tests drive the Fused scheduler like the event loop, by default
with overlap depth 1 (a plan's results land one round later; some tests also
run overlap depth 0): each row yields one token, and requests finish at
max_new_tokens. Unfixed, the ones that reach the
stall stop producing forward batches within about a hundred rounds and report
it after STALL_ROUNDS empty rounds.
"""

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


def _prompts(
    num_requests: int,
    shared_tokens: int,
    unique_tokens: int,
    unique_tokens_step: int = 0,
) -> dict[str, list[int]]:
    shared = [100 + i for i in range(shared_tokens)]
    return {
        f"r{k}": shared
        + [5000 + 100 * k + j for j in range(unique_tokens + unique_tokens_step * k)]
        for k in range(num_requests)
    }


def _run_closed_loop(
    num_usable_pages: int,
    *,
    num_requests: int,
    shared_tokens: int,
    unique_tokens: int,
    max_new_tokens: int,
    window: int,
    prefix_cache: bool,
    max_scheduled_tokens: int,
    unique_tokens_step: int = 0,
    overlap_schedule_depth: int = 1,
    later_arrival_round: int = 2,
) -> dict:
    """Serve every request to completion, or report a stall after STALL_ROUNDS
    consecutive rounds without a forward batch. Request k has unique_tokens +
    k * unique_tokens_step tokens of its own. The first request arrives
    later_arrival_round rounds before the rest (two by default, so its prompt
    pages are published by the time the others are admitted)."""
    scheduler = ts.Scheduler(
        _config(
            num_usable_pages,
            window=window,
            prefix_cache=prefix_cache,
            overlap_schedule_depth=overlap_schedule_depth,
            max_scheduled_tokens=max_scheduled_tokens,
        )
    )
    prompts = _prompts(num_requests, shared_tokens, unique_tokens, unique_tokens_step)
    specs = [_spec(rid, tokens, max_new_tokens) for rid, tokens in prompts.items()]
    scheduler.submit_requests(specs[:1])
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
        if round_index == later_arrival_round and later:
            scheduler.submit_requests(later)
            later = []
        plan = scheduler.next_execution_plan()
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
            max_scheduled_tokens=1024,
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
            max_scheduled_tokens=1024,
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
            max_scheduled_tokens=1024,
        )
        assert result["outcome"] == "done", result
        assert result["reprefill_rows"] <= 2, result
        assert result["active"] == 0, result

    @pytest.mark.parametrize(
        "prefix_cache", [True, False], ids=["prefix_cache", "no_prefix_cache"]
    )
    @pytest.mark.parametrize(
        ("max_scheduled_tokens", "max_reprefill_rows"),
        [(1024, 2), (64, 4)],
        ids=["one_chunk", "two_chunks"],
    )
    def test_distinct_prompts_shorter_than_window_finish(
        self, max_scheduled_tokens: int, max_reprefill_rows: int, prefix_cache: bool
    ):
        """Eight distinct 104-token prompts under a 128-token window share no
        page, but nothing slides out before a sequence outgrows the window,
        so their early decode growth has nothing to recycle and only the
        last-resort victim keeps the scheduler live. Unfixed: stall with the
        pool full, with or without the prefix cache. Fixed: the last resort
        retracts one decoding request and its pages go to the other blocked
        decodes. Handed to a waiting prompt instead, they admit it, and two
        more decoding requests are retracted later to readmit the first. The
        bound on re-prefill rows keeps the fallback's retractions from
        churning."""
        result = _run_closed_loop(
            64,
            num_requests=8,
            shared_tokens=0,
            unique_tokens=104,
            max_new_tokens=32,
            window=128,
            prefix_cache=prefix_cache,
            max_scheduled_tokens=max_scheduled_tokens,
        )
        assert result["outcome"] == "done", result
        assert result["reprefill_rows"] <= max_reprefill_rows, result
        assert result["active"] == 0, result

    @pytest.mark.parametrize(
        "overlap_schedule_depth", [0, 1], ids=["overlap_0", "overlap_1"]
    )
    def test_chunked_prompt_beside_a_decode_is_not_retracted_forever(
        self, overlap_schedule_depth: int
    ):
        """Distinct 72- and 80-token prompts arrive together under a
        128-token window, in 40-token chunks, on a 24-page pool. The second
        one's first chunk joins the first one's last chunk and, since the
        first one holds pages, prepays the lookback its next chunk reads,
        which fills the pool. Both sequences are shorter than the window, so
        nothing recycles: once the first one's decode needs a new page and
        the second one's completing chunk its decode slot, the round builds
        nothing and the last resort retracts the second one. Its pages must
        go to the blocked decode. Given to nobody, the second prompt is
        admitted again ahead of any decode, prepays the same pages and is
        retracted again, forever, with no request producing a token.
        Unfixed, both finish without a retraction."""
        result = _run_closed_loop(
            24,
            num_requests=2,
            shared_tokens=0,
            unique_tokens=72,
            unique_tokens_step=8,
            max_new_tokens=32,
            window=128,
            prefix_cache=True,
            max_scheduled_tokens=40,
            overlap_schedule_depth=overlap_schedule_depth,
            later_arrival_round=0,
        )
        assert result["outcome"] == "done", result
        assert result["reprefill_rows"] == 0, result
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


class TestPrefixHitSlidingWindowReserve:
    SHARED = 160  # 10 pages
    TAIL = 8
    WINDOW = 64  # 4 pages; the lookback a token reads is 63 tokens

    @pytest.mark.parametrize(
        ("max_new_tokens", "private_sliding_pages"),
        [(40, 3), (200, 5)],
        ids=["capped_by_headroom", "capped_by_lookback"],
    )
    def test_prefix_hit_prepays_growth_over_shared_lookback(
        self, max_new_tokens: int, private_sliding_pages: int
    ):
        """A prompt admitted without a hit holds only the decode slot in the
        sliding-window group. A prompt that hits its 160-token prefix while
        the first request still holds it shares the 4-page lookback and
        prepays the growth over it: min(prompt headroom, min(hit, window - 1))
        tokens beyond its tail. With max_new_tokens=40 that is 40 tokens
        (8 + 40 -> 3 private pages); with 200 it is capped at the 63-token
        lookback (8 + 63 -> 5 pages). Unfixed, the hit request holds 1 private
        page (8 + 1 tokens). The full-history group prepays the whole
        generation either way."""
        scheduler = ts.Scheduler(
            _config(
                64,
                window=self.WINDOW,
                prefix_cache=True,
                overlap_schedule_depth=0,
                max_scheduled_tokens=1024,
            )
        )
        shared = [100 + i for i in range(self.SHARED)]

        # First request: no hit, so the sliding-window group reserves only
        # the decode slot (168 + 1 tokens -> 11 pages).
        plan = _submit_and_step(
            scheduler,
            "a",
            shared + [7000 + j for j in range(self.TAIL)],
            max_new_tokens,
        )
        _, _, first = _row_of(plan, "a")
        assert (
            len(first[SLIDING_GROUP])
            == (self.SHARED + self.TAIL + 1 + PAGE - 1) // PAGE
        )
        first_pages = {page for page in first[SLIDING_GROUP] if page > 0}
        _land(scheduler, "a", 1, decode_reserve=None)
        # Scheduling its first decode publishes the prompt pages.
        plan = scheduler.next_execution_plan()
        assert [
            list(op.request_ids) for op in plan.forward if list(op.request_ids)
        ] == [["a"]]
        _land(scheduler, "a", 2, decode_reserve=1)

        plan = _submit_and_step(
            scheduler,
            "b",
            shared + [8000 + j for j in range(self.TAIL)],
            max_new_tokens,
        )
        batch, row, second = _row_of(plan, "b")
        assert batch.extend_prefix_lens[row] == self.SHARED
        assert batch.input_lengths[row] == self.TAIL

        sliding = [page for page in second[SLIDING_GROUP] if page > 0]
        shared_lookback = [page for page in sliding if page in first_pages]
        private = [page for page in sliding if page not in first_pages]
        assert len(shared_lookback) == self.WINDOW // PAGE, second[SLIDING_GROUP]
        assert len(private) == private_sliding_pages, second[SLIDING_GROUP]
        assert len(second[SLIDING_GROUP]) == self.SHARED // PAGE + private_sliding_pages
        assert len(second[FULL_GROUP]) == len(first[FULL_GROUP])


class TestLaterChunkSlidingWindowReserve:
    WINDOW = 64  # 4 pages; the lookback a token reads is 63 tokens

    @pytest.mark.parametrize(
        ("resident", "sliding_pages"),
        [(False, 4), (True, 8)],
        ids=["alone", "beside_a_resident"],
    )
    def test_first_chunk_prepays_the_lookback_its_next_chunk_reads(
        self, resident: bool, sliding_pages: int
    ):
        """A 200-token prompt in 64-token chunks: the second chunk reads the
        first one's 63-token lookback before any page of it can slide out.
        Beside a resident request that holds pages, the first chunk prepays
        that lookback: 64 + 63 tokens, 8 sliding-window pages. Alone it
        prepays nothing and holds only its chunk, 4 pages, as it does
        unfixed either way."""
        scheduler = ts.Scheduler(
            _config(
                64,
                window=self.WINDOW,
                prefix_cache=True,
                overlap_schedule_depth=0,
                max_scheduled_tokens=64,
            )
        )
        if resident:
            _submit_and_step(scheduler, "a", [7000 + j for j in range(8)], 40)
            _land(scheduler, "a", 1, decode_reserve=None)

        plan = _submit_and_step(scheduler, "b", [5000 + j for j in range(200)], 40)
        batch, row, tables = _row_of(plan, "b")
        assert batch.extend_prefix_lens[row] == 0
        assert batch.input_lengths[row] == 64
        assert len(tables[SLIDING_GROUP]) == sliding_pages, tables[SLIDING_GROUP]
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


def _serve_alone(
    scheduler, request_id: str, tokens: list[int], max_new_tokens: int, *, remote: bool
) -> bool:
    """Serve one request on an otherwise idle scheduler with overlap depth 0;
    with `remote` (D role) the peer's prefill lands as soon as it is planned.
    False unless it finishes within MAX_ROUNDS rounds."""
    scheduler.submit_requests([_spec(request_id, tokens, max_new_tokens)])
    if remote:
        scheduler.advance(
            ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent(request_id))
        )
    generated = 0
    for _ in range(MAX_ROUNDS):
        plan = scheduler.next_execution_plan()
        if plan.remote_prefill is not None:
            scheduler.advance(
                ts.ExecutionEvent().add_event(
                    ts.PD.RemotePrefillDoneEvent(request_id, 1)
                )
            )
            continue
        batch = next((op for op in plan.forward if list(op.request_ids)), None)
        if batch is None:
            continue
        prefill = batch.num_extends() > 0
        events = ts.ExecutionEvent()
        result = ts.ForwardEvent.ExtendResult()
        result.request_id = request_id
        result.tokens = []
        if (
            not prefill
            or batch.extend_prefix_lens[0] + batch.input_lengths[0]
            == list(batch.prefill_lengths)[0]
        ):
            generated += 1
            result.tokens = [9000 + generated]
        events.add_event(result)
        if generated >= max_new_tokens:
            finish = ts.ForwardEvent.Finish()
            finish.request_id = request_id
            events.add_event(finish)
        elif not prefill:
            reserve = ts.ForwardEvent.UpdateReserveNumTokens()
            reserve.request_id = request_id
            reserve.reserve_num_tokens_in_next_schedule_event = 1
            events.add_event(reserve)
        scheduler.advance(events)
        if generated >= max_new_tokens:
            scheduler.next_execution_plan()  # reaps the finished request
            return True
    return False


class TestSingleRequestBound:
    """A request that max_single_request_tokens accepts must be admissible
    once it is the only one, as it is unfixed: with no other request holding
    pages the sliding-window prepay does not apply, since the capacity model
    bounds a single request without it. Each pool is the smallest whose bound
    accepts the request."""

    @staticmethod
    def _bound(num_usable_pages: int, *, remote: bool, **config) -> int:
        cfg = _config(num_usable_pages, **config)
        return ts.Scheduler(_d_role(cfg) if remote else cfg).max_single_request_tokens()

    def test_lone_prefix_hit_is_admitted(self):
        """The first request leaves its 128-token prefix cached; the second
        hits it while no other request holds pages. Prepaying the growth over
        the hit (4 more sliding-window pages) would ask for 37 of the 34
        pages. The hit pages are held by the prefix index alone, so they
        recycle like the request's own and need no prepay."""
        config = dict(
            window=64,
            prefix_cache=True,
            overlap_schedule_depth=0,
            max_scheduled_tokens=128,
        )
        shared = [100 + i for i in range(128)]
        tokens = shared + [8000 + j for j in range(136)]
        assert (
            self._bound(33, remote=False, **config)
            < len(tokens) + 64
            <= self._bound(34, remote=False, **config)
        )
        scheduler = ts.Scheduler(_config(34, **config))
        first = shared + [7000 + j for j in range(136)]
        assert _serve_alone(scheduler, "a", first, 64, remote=False)
        assert _serve_alone(scheduler, "b", tokens, 64, remote=False)
        assert scheduler.active_lcm_blocks() == 0

    def test_lone_prefix_hit_with_a_later_chunk_is_admitted(self):
        """The first request leaves its 128-token prefix cached; the second
        hits it with 150 tokens of its own, so its 128-token first chunk
        leaves 22 for a later chunk. That chunk already holds the 4-page
        lookback of hit pages behind it; prepaying the lookback the later
        chunk reads on top would ask for 14 + 18 pages, one more than the
        31-page pool. Alone, the hit pages recycle into the later chunk, so
        nothing is prepaid."""
        config = dict(
            window=64,
            prefix_cache=True,
            overlap_schedule_depth=0,
            max_scheduled_tokens=128,
        )
        shared = [100 + i for i in range(128)]
        tokens = shared + [8000 + j for j in range(150)]
        assert (
            self._bound(30, remote=False, **config)
            < len(tokens) + 1
            <= self._bound(31, remote=False, **config)
        )
        scheduler = ts.Scheduler(_config(31, **config))
        first = shared + [7000 + j for j in range(150)]
        assert _serve_alone(scheduler, "a", first, 1, remote=False)
        assert _serve_alone(scheduler, "b", tokens, 1, remote=False)
        assert scheduler.active_lcm_blocks() == 0

    def test_lone_multi_chunk_prompt_is_admitted(self):
        """A 200-token prompt in 128-token chunks under a 65-token window.
        Alone, its first chunk prepays nothing for the 64-token lookback the
        second chunk reads: the capacity model counts that lookback plus one
        chunk, and the first chunk's slid-out pages fund the second."""
        config = dict(
            window=65,
            prefix_cache=True,
            overlap_schedule_depth=0,
            max_scheduled_tokens=128,
        )
        tokens = [5000 + j for j in range(200)]
        assert (
            self._bound(24, remote=False, **config)
            < len(tokens) + 1
            <= self._bound(25, remote=False, **config)
        )
        scheduler = ts.Scheduler(_config(25, **config))
        assert _serve_alone(scheduler, "a", tokens, 1, remote=False)
        assert scheduler.active_lcm_blocks() == 0

    def test_lone_remote_landing_after_a_local_prefix_is_admitted(self):
        """D role. The first request lands, decodes and leaves its prompt
        cached; the second shares its first 32 tokens, too few for the
        sliding-window group to resume from, so its admission stops at that
        promotion boundary. Alone it prepays nothing, and the peer fills the
        whole prompt, so there is no later local chunk whose lookback the
        landing would have to prepay; prepaying one would ask for more than
        the 16-page pool."""
        config = dict(
            window=64,
            prefix_cache=True,
            overlap_schedule_depth=0,
            max_scheduled_tokens=1024,
        )
        shared = [100 + i for i in range(32)]
        tokens = shared + [8000 + j for j in range(72)]
        assert (
            self._bound(15, remote=True, **config)
            < len(tokens) + 16
            <= self._bound(16, remote=True, **config)
        )
        scheduler = ts.Scheduler(_d_role(_config(16, **config)))
        first = shared + [7000 + j for j in range(72)]
        assert _serve_alone(scheduler, "a", first, 16, remote=True)
        assert _serve_alone(scheduler, "b", tokens, 16, remote=True)
        assert scheduler.active_lcm_blocks() == 0
