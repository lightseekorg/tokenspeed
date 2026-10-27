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

from test.runtime.scheduler_driver import (  # noqa: E402
    advance_scheduler,
    make_extend_result_event,
)

from tokenspeed.runtime.engine.scheduler_utils import make_config  # noqa: E402

PAGE = 16
PROMPT = list(range(40))  # 2 hash-complete pages and an 8-token tail


def _scheduler(*, with_state: bool = False) -> ts.Scheduler:
    groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=PAGE,
            total_pages=64 + 1,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    if with_state:
        groups.append(
            ts.CacheGroupConfig(
                group_id="state",
                block_granularity=0,
                total_pages=64 + 1,
                attn_kind=ts.AttnKind.State,
            )
        )
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
            storage_config=None,
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


def _state_row(forward_op) -> list[int]:
    return [int(value) for value in forward_op.block_tables_arrays()["state"][0]]


def _run_cold(scheduler: ts.Scheduler, rid: str, tokens: list[int]) -> int:
    """Prefill, decode twice, finish as the runtime does under the overlap
    schedule (on the first decode's token, with the second decode already
    dispatched), let it land and settle; returns the published frontier."""
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
    _land(scheduler, decode, 1002)
    # The landing settles the finish: the frontier is the prompt and the two
    # consumed decode tokens (1002 was never consumed).
    return len(tokens) + 2


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
    assert computed == 42
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
    assert destination not in plan.pages_to_zero["history"]
    assert source not in plan.pages_to_zero["history"]
    assert source not in table.tolist()
    _land(scheduler, prefill, 2000)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    _finish(scheduler, decode)

    # Shares the finish tail (the prompt plus the two consumed tokens,
    # [32, 42)); b, finishing with a decode in flight, holds its pages still.
    plan, prefill = _hit(scheduler, "c", PROMPT + [1000, 1001, 5, 6])
    assert list(prefill.extend_prefix_lens) == [computed]
    assert list(prefill.input_lengths) == [2]
    assert plan.pages_to_copy_arrays()["history"].shape == (1, 2)


def test_a_junction_nobody_ended_at_resolves_to_the_page() -> None:
    """The tails under page 2 cover tokens [32, 40) and [32, 42); a prompt
    that diverges at token 37 can use neither and hits at the page."""
    scheduler = _scheduler()
    _run_cold(scheduler, "a", PROMPT)
    plan, prefill = _hit(scheduler, "b", PROMPT[:37] + [777, 778, 779])
    assert list(prefill.extend_prefix_lens) == [32]
    assert list(prefill.input_lengths) == [8]
    assert plan.pages_to_copy_arrays()["history"].shape == (0, 2)


def test_an_identical_prompt_keeps_one_token_to_compute() -> None:
    """The tails end at 40 and 42; a 40-token prompt must compute at least
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


def test_a_state_group_hits_where_its_checkpoint_was_published() -> None:
    """A hybrid's prefix hit needs the state of the same instant. The cold run
    published two: the prompt-end block the first decode rotated out of the
    live slot, and the live block at finish. A prompt sharing the prompt end
    claims the state block into slot 0 (shared, never copied) beside the KV
    page copy; a prompt diverging inside the page finds no state below its
    boundary and starts from the zero state, a fresh zeroed block in slot 0."""
    scheduler = _scheduler(with_state=True)
    _submit(scheduler, "a", PROMPT)
    plan = scheduler.next_execution_plan()
    (prefill,) = _forwards(plan)
    cold_row = _state_row(prefill)
    assert cold_row[0] > 0 and cold_row[1] > 0 and cold_row[0] != cold_row[1]
    assert all(block in plan.pages_to_zero["state"] for block in cold_row)
    _land(scheduler, prefill, 1000)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    prompt_end_block = cold_row[1]
    first_decode_row = _state_row(decode)
    assert first_decode_row[0] == prompt_end_block
    assert first_decode_row[1] > 0 and first_decode_row[1] != prompt_end_block
    assert plan.pages_to_copy_arrays()["state"].shape == (0, 2)
    assert plan.pages_to_copy_arrays()["history"].shape == (1, 2)
    _land(scheduler, decode, 1001)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    live_block = first_decode_row[1]
    assert _state_row(decode) == [0, live_block]
    # The second decode is in flight, writing the live block in place, when
    # the finish is decided: the request waits for it to land, and its landing
    # publishes the frontier it produced -- 42 tokens, the live block as the
    # state at 42.
    _finish(scheduler, decode)
    _land(scheduler, decode, 1002)
    computed = 42

    plan, prefill = _hit(scheduler, "b", PROMPT + [999])
    assert list(prefill.extend_prefix_lens) == [40]
    row = _state_row(prefill)
    assert row[0] == prompt_end_block and row[1] > 0 and row[1] != row[0]
    assert plan.pages_to_copy_arrays()["state"].shape == (0, 2)
    assert plan.pages_to_copy_arrays()["history"].shape == (1, 2)
    _land(scheduler, prefill, 2000)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    _finish(scheduler, decode)

    plan, prefill = _hit(scheduler, "c", PROMPT + [1000, 1001, 5, 6])
    assert list(prefill.extend_prefix_lens) == [computed]
    row = _state_row(prefill)
    assert row[0] == live_block and row[1] > 0 and row[1] != row[0]
    _land(scheduler, prefill, 4000)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    _finish(scheduler, decode)

    # The KV pages reach 32 but no state does: the request starts from the
    # zero state and the junction cut ends its first chunk at the KV hit, so
    # that chunk's state block is published under page 2's hash and the next
    # such prompt hits 32 with every group.
    plan, prefill = _hit(scheduler, "d", PROMPT[:37] + [777, 778, 779])
    assert list(prefill.extend_prefix_lens) == [0]
    assert list(prefill.input_lengths) == [32]
    row = _state_row(prefill)
    assert row[0] > 0 and row[0] in plan.pages_to_zero["state"] and row[1] > 0
    assert plan.pages_to_copy_arrays()["history"].shape == (0, 2)
    junction_block = row[1]
    advance_scheduler(scheduler, [make_extend_result_event("d", [])])
    plan = scheduler.next_execution_plan()
    (prefill,) = _forwards(plan)
    assert list(prefill.extend_prefix_lens) == [32]
    assert list(prefill.input_lengths) == [8]
    assert _state_row(prefill)[0] == junction_block
    _land(scheduler, prefill, 3000)
    plan = scheduler.next_execution_plan()
    (decode,) = _forwards(plan)
    _finish(scheduler, decode)

    plan, prefill = _hit(scheduler, "e", PROMPT[:37] + [888, 889, 890])
    assert list(prefill.extend_prefix_lens) == [32]
    assert list(prefill.input_lengths) == [8]
    assert _state_row(prefill)[0] == junction_block
