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

"""Token-level prefix hits, driven against the real C++ scheduler.

No engine: the scheduler is built through ``make_config`` as the event loop
builds it and its plans are read the way ``DeviceHandle.execute`` reads them.
A request that finishes publishes its partial last page as a tail, a later
prompt sharing more than the hash-complete pages hits at the tail's token
count, its plan carries the one page copy the hitter's private block needs,
and its first chunk starts right at the hit.
"""

from __future__ import annotations

import numpy as np
import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    advance_scheduler,
    make_config,
    make_extend_result_event,
)

PAGE = 16
PROMPT = list(range(40))  # 2 hash-complete pages and an 8-token tail


def _scheduler() -> ts.Scheduler:
    groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=PAGE,
            total_pages=64 + 1,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        )
    ]
    return ts.Scheduler(
        make_config(
            num_device_pages=64 + 1,
            max_scheduled_tokens=256,
            max_batch_size=4,
            prefix_granularity=PAGE,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            role="fused",
            num_snapshot_pages=1,
            max_retracted_requests=0,
            l3_prefetch_min_pages=0,
            cache_groups=groups,
        )
    )


def _submit(scheduler: ts.Scheduler, rid: str, tokens: list[int]) -> None:
    spec = ts.RequestSpec()
    spec.request_id = rid
    spec.tokens = list(tokens)
    spec.max_new_tokens = 4
    scheduler.submit_requests([spec])


def _forwards(plan):
    return [op for op in plan.forward if op.request_ids]


def _land(scheduler: ts.Scheduler, forward_op, token: int) -> None:
    (rid,) = forward_op.request_ids
    advance_scheduler(scheduler, [make_extend_result_event(rid, [token])])


def _finish(scheduler: ts.Scheduler, forward_op) -> None:
    (rid,) = forward_op.request_ids
    event = ts.ForwardEvent.Finish()
    event.request_id = rid
    advance_scheduler(scheduler, [event])


def _run_cold(scheduler: ts.Scheduler, rid: str, tokens: list[int]) -> int:
    """Prefill, decode once, finish; returns the computed token count."""
    _submit(scheduler, rid, tokens)
    plan = scheduler.next_execution_plan()
    (prefill,) = _forwards(plan)
    assert list(prefill.extend_prefix_lens) == [0]
    assert list(prefill.input_lengths) == [len(tokens)]
    assert plan.pages_to_copy_arrays()["history"].shape == (0, 2)
    _land(scheduler, prefill, 1000)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    assert decode.num_extends() == 0
    _land(scheduler, decode, 1001)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    _finish(scheduler, decode)
    # Computed tokens at finish: the prompt and the one landed decode token;
    # the second decode's token was never sampled.
    return len(tokens) + 1


def _hit(scheduler: ts.Scheduler, rid: str, tokens: list[int]):
    _submit(scheduler, rid, tokens)
    plan = scheduler.next_execution_plan()
    (prefill,) = _forwards(plan)
    return plan, prefill


def test_prompt_end_and_finish_publish_tails_a_later_prompt_hits() -> None:
    """Natural publication points: the prompt end (the owner moved to a copy
    of its partial page) and the finish (the partial page in place). A prompt
    sharing the tail's tokens hits at the tail's token count, takes one page
    copy for the block holding it, and computes from there."""
    scheduler = _scheduler()
    computed = _run_cold(scheduler, "a", PROMPT)
    assert computed == 41
    assert scheduler.tail_entries() == 2

    # Shares the prompt-end tail (tokens [32, 40)) and diverges after it.
    plan, prefill = _hit(scheduler, "b", PROMPT + [999])
    assert list(prefill.extend_prefix_lens) == [40]
    assert list(prefill.input_lengths) == [1]
    copies = plan.pages_to_copy_arrays()["history"]
    assert copies.shape == (1, 2)
    source, destination = (int(value) for value in copies[0])
    table = prefill.block_tables_arrays()["history"][0]
    # The hitter's table: the two shared pages, then its private copy of page 2.
    assert table[2] == destination
    assert destination not in plan.pages_to_zero
    assert source not in table.tolist()
    _land(scheduler, prefill, 2000)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    _finish(scheduler, decode)

    # Shares the finish tail (the prompt plus the decoded token, [32, 41)).
    plan, prefill = _hit(scheduler, "c", PROMPT + [1000, 5, 6])
    assert list(prefill.extend_prefix_lens) == [41]
    assert list(prefill.input_lengths) == [2]
    assert plan.pages_to_copy_arrays()["history"].shape == (1, 2)


def test_a_junction_nobody_ended_at_resolves_to_the_page() -> None:
    """The tails under page 2 cover tokens [32, 40) and [32, 41); a prompt
    that diverges at token 37 can use neither and hits at the page."""
    scheduler = _scheduler()
    _run_cold(scheduler, "a", PROMPT)
    plan, prefill = _hit(scheduler, "b", PROMPT[:37] + [777, 778, 779])
    assert list(prefill.extend_prefix_lens) == [32]
    assert list(prefill.input_lengths) == [8]
    assert plan.pages_to_copy_arrays()["history"].shape == (0, 2)


def test_an_identical_prompt_keeps_one_token_to_compute() -> None:
    """The tails end at 40 and 41; a 40-token prompt must compute at least
    one token, so neither tail serves it and the hit is the hash-complete
    pages."""
    scheduler = _scheduler()
    _run_cold(scheduler, "a", PROMPT)
    plan, prefill = _hit(scheduler, "b", PROMPT)
    assert list(prefill.extend_prefix_lens) == [32]
    assert list(prefill.input_lengths) == [8]


def test_pages_to_copy_arrays_are_int32_pairs_per_group() -> None:
    scheduler = _scheduler()
    _submit(scheduler, "a", PROMPT)
    plan = scheduler.next_execution_plan()
    copies = plan.pages_to_copy_arrays()
    assert set(copies) == {"history"}
    assert copies["history"].dtype == np.int32
    assert copies["history"].shape == (0, 2)
    assert plan.pages_to_copy == {"history": []}
