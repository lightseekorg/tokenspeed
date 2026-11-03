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

"""Eviction keeps a finished request's resumable prefix over slid-out pages.

A sliding-window group keeps every page that slides out of a request's window
cached (Device-only) so an interior boundary stays resumable. Under memory
pressure those pages, published at an interior chunk or decode step and never
hit, are evicted before any other cached block, so an older finished request's
full-attention pages and prompt-end window survive and a resent prompt hits.
"""

from __future__ import annotations

from tokenspeed_scheduler import (
    AttnKind,
    CacheGroupConfig,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

P = 4
WINDOW = 4
PROMPT = 30  # unaligned: a resend resumes at the last full page, 28
OUTPUT = 16


def _scheduler(pages: int) -> Scheduler:
    cfg = SchedulerConfig()
    cfg.prefix_granularity = P
    cfg.max_scheduled_tokens = 8  # several interior chunks per prompt
    cfg.max_batch_size = 8
    cfg.num_device_pages = pages
    cfg.num_snapshot_pages = 1
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="sliding",
            block_granularity=P,
            total_pages=pages,
            attn_kind=AttnKind.SlidingWindow,
            sliding_window_tokens=WINDOW,
        ),
        CacheGroupConfig(
            group_id="full",
            block_granularity=P,
            total_pages=pages,
            attn_kind=AttnKind.Full,
        ),
    ]
    return Scheduler(cfg)


def _advance(scheduler: Scheduler, kind: str, request_id: str, **fields) -> None:
    event = getattr(ForwardEvent, kind)()
    event.request_id = request_id
    for name, value in fields.items():
        setattr(event, name, value)
    execution_event = ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _serve(scheduler: Scheduler, request_id: str, tokens: list[int]) -> int:
    """Runs one request alone to its finish; returns its prefix-cache hit."""
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    spec.max_new_tokens = OUTPUT
    scheduler.submit_requests([spec])
    hit = None
    generated = 0
    while generated < OUTPUT:
        plan = scheduler.next_execution_plan()
        rows = [
            (op, i)
            for op in plan.forward
            for i, rid in enumerate(op.request_ids)
            if rid == request_id
        ]
        assert rows, f"{request_id} was not scheduled"
        op, i = rows[0]
        decode = i >= op.num_extends()
        if not decode:
            if hit is None:
                hit = op.extend_prefix_lens[i]
            if op.extend_prefix_lens[i] + op.input_lengths[i] < op.prefill_lengths[i]:
                _advance(scheduler, "ExtendResult", request_id, tokens=[])
                continue
        generated += 1
        _advance(scheduler, "ExtendResult", request_id, tokens=[900 + generated])
        if decode and generated < OUTPUT:
            _advance(
                scheduler,
                "UpdateReserveNumTokens",
                request_id,
                reserve_num_tokens_in_next_schedule_event=1,
            )
    assert hit is not None, f"{request_id} never ran as an extend"
    _advance(scheduler, "Finish", request_id)
    scheduler.next_execution_plan()
    return hit


def test_older_request_prefix_outlives_newer_slid_out_pages() -> None:
    # Three 30-token prompts with 16 output tokens each do not all fit with
    # every slid-out page cached. Evicting by request age alone drops the
    # oldest prompt whole; evicting the slid-out pages first keeps it.
    scheduler = _scheduler(pages=66)
    prompts = {
        name: [base + t for t in range(PROMPT)]
        for name, base in (("a", 100), ("b", 200), ("c", 300))
    }
    for name in ("a", "b", "c"):
        assert _serve(scheduler, f"{name}1", prompts[name]) == 0
    assert _serve(scheduler, "a2", prompts["a"]) == 28
    assert _serve(scheduler, "c2", prompts["c"]) == 28
