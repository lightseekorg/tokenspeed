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

"""``Forward.Batch.block_table_spans()``: the (released, pages) pair the
Python bridge rebuilds full rows from (see
``tokenspeed.runtime.engine.scheduler_utils.packed_block_tables_from_forward_op``).
For every group and row, ``row[:released]`` must be the null-hole prefix,
``row[pages:]`` the -1 padding suffix, and the entry at ``pages - 1`` (the
last held page) non-negative when ``pages > 0``. A Full group never releases
a prefix block, so its ``released`` stays 0; a SlidingWindow group's
``released`` grows as its window slides past older pages."""

from __future__ import annotations

import math

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

PAGE = 4
WINDOW_TOKENS = 16
SWA_GROUP = "sliding_attention"
FULL_GROUP = "full_attention"
MAX_ROUNDS = 2000


def _config(
    num_usable_pages: int, *, max_scheduled_tokens: int
) -> "ts.SchedulerConfig":
    """One replayable SlidingWindow group and one Full group, Fused role,
    prefix caching on, a retraction-capable snapshot pool."""
    cfg = ts.SchedulerConfig()
    cfg.num_device_pages = num_usable_pages + 1  # page 0 is the null sentinel
    cfg.num_host_pages = 0
    cfg.disable_l2_cache = True
    cfg.enable_l3_storage = False
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = 8
    cfg.max_resident_requests = cfg.max_batch_size
    cfg.num_snapshot_pages = cfg.num_device_pages
    cfg.max_retracted_requests = cfg.max_batch_size
    cfg.prefix_granularity = PAGE
    cfg.decode_input_tokens = 1
    cfg.overlap_schedule_depth = 0
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=SWA_GROUP,
            block_granularity=PAGE,
            total_pages=cfg.num_device_pages,
            attn_kind=ts.AttnKind.SlidingWindow,
            sliding_window_tokens=WINDOW_TOKENS,
            replayable=True,
        ),
        ts.CacheGroupConfig(
            group_id=FULL_GROUP,
            block_granularity=PAGE,
            total_pages=cfg.num_device_pages,
            attn_kind=ts.AttnKind.Full,
        ),
    ]
    cfg.host_group_packing = [1, 1]
    return cfg


def _spec(request_id: str, tokens: list[int], max_new_tokens: int) -> "ts.RequestSpec":
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(tokens)
    spec.max_new_tokens = max_new_tokens
    return spec


def _land_cache_ops(scheduler, plan) -> int:
    """Acknowledge every snapshot/restore/write-back as if it finished at
    once. Returns how many requests the plan retracted."""
    events = ts.ExecutionEvent()
    num_events = 0
    retracted = 0
    for op in plan.cache:
        if isinstance(op, ts.Cache.SnapshotOp):
            retracted += len(op.request_ids)
            done_type = ts.Cache.SnapshotDoneEvent
        elif isinstance(op, ts.Cache.RestoreOp):
            done_type = ts.Cache.RestoreDoneEvent
        elif isinstance(op, ts.Cache.WriteBackOp):
            done_type = ts.Cache.WriteBackDoneEvent
        else:
            continue
        for op_id in op.op_ids:
            done = done_type()
            done.op_id = int(op_id)
            events.add_event(done)
            num_events += 1
    if num_events:
        scheduler.advance(events)
    return retracted


def _assert_span_invariants(arrays, spans):
    """The per-row/per-group contract ``block_table_spans`` documents,
    checked on every forward op this test sees."""
    for group_id, arr in arrays.items():
        group_spans = spans[group_id]
        assert group_spans.shape == (arr.shape[0], 2)
        assert group_spans.dtype == arr.dtype  # both int32
        released, pages = group_spans[:, 0], group_spans[:, 1]
        for row in range(arr.shape[0]):
            r, p = int(released[row]), int(pages[row])
            assert 0 <= r <= p <= arr.shape[1], (group_id, row, r, p, arr.shape[1])
            assert (arr[row, :r] == 0).all(), (group_id, row, "released prefix")
            assert (arr[row, p:] == -1).all(), (group_id, row, "padding suffix")
            if p > 0:
                assert arr[row, p - 1] >= 0, (group_id, row, "last held page")
        if group_id == FULL_GROUP:
            assert (released == 0).all(), "a Full group never releases a prefix block"


def test_spans_hold_through_chunking_prefix_hit_and_sliding_holes():
    """Chunked prefill, a prefix-cache hit that replays the sliding-window
    group, and decode steps that slide the window into a null hole -- the
    span contract is checked on every single forward op throughout, not
    only at the end. Pool size and token budget mirror
    test/runtime/test_block_tables_bridge.py's
    ``test_forward_tables_match_through_chunking_prefix_hit_and_sliding_holes``,
    which this test is the C++-scheduler-side counterpart of."""
    max_scheduled_tokens = 32  # >= window(16) + max(window, prefix_granularity)
    cfg = _config(num_usable_pages=80, max_scheduled_tokens=max_scheduled_tokens)
    scheduler = ts.Scheduler(cfg)

    shared = list(range(100, 100 + 40))  # 40-token shared prefix
    r0 = _spec("r0", shared + list(range(1, 9)), max_new_tokens=40)
    siblings = [
        _spec(
            f"r{k}", shared + list(range(900 + 10 * k, 908 + 10 * k)), max_new_tokens=40
        )
        for k in (1, 2)
    ]
    scheduler.submit_requests([r0])

    generated = {req.request_id: 0 for req in [r0, *siblings]}
    finished: set[str] = set()
    saw_chunking = False
    saw_prefix_hit = False
    saw_sliding_hole = False
    siblings_submitted = False
    idle_rounds = 0
    STALL_ROUNDS = 100

    for round_index in range(MAX_ROUNDS):
        plan = scheduler.next_execution_plan()
        _land_cache_ops(scheduler, plan)
        batch = next((op for op in plan.forward if list(op.request_ids)), None)
        if batch is None:
            idle_rounds += 1
            if finished == set(generated):
                break
            if idle_rounds >= STALL_ROUNDS:
                pytest.fail(f"stalled at round {round_index}: generated={generated}")
            continue
        idle_rounds = 0

        arrays = batch.block_tables_arrays()
        spans = batch.block_table_spans()
        _assert_span_invariants(arrays, spans)

        num_extends = batch.num_extends()
        prefill_lengths = list(batch.prefill_lengths)
        extend_prefix_lens = list(batch.extend_prefix_lens)
        input_lengths = list(batch.input_lengths)
        for row in range(num_extends):
            if extend_prefix_lens[row] + input_lengths[row] < prefill_lengths[row]:
                saw_chunking = True
            if extend_prefix_lens[row] > 0:
                saw_prefix_hit = True
        if num_extends == 0:
            swa_released, swa_pages = spans[SWA_GROUP][:, 0], spans[SWA_GROUP][:, 1]
            width = int((swa_pages - swa_released).max())
            assert width <= math.ceil(WINDOW_TOKENS / PAGE) + 2, width
            if (swa_released > 0).any():
                saw_sliding_hole = True

        # Siblings arrive once r0 has produced at least one decode token, so
        # their shared prefix is published and hittable.
        if not siblings_submitted and generated["r0"] >= 1:
            scheduler.submit_requests(siblings)
            siblings_submitted = True

        events = ts.ExecutionEvent()
        num_events = 0
        for row, rid in enumerate(batch.request_ids):
            if rid in finished:
                continue
            if row < num_extends and extend_prefix_lens[row] + input_lengths[row] < (
                prefill_lengths[row]
            ):
                continue  # an intermediate prefill chunk produces no token
            result = ts.ForwardEvent.ExtendResult()
            result.request_id = rid
            generated[rid] += 1
            result.tokens = [9000 + generated[rid]]
            events.add_event(result)
            num_events += 1
            if generated[rid] >= 40:
                finished.add(rid)
                finish = ts.ForwardEvent.Finish()
                finish.request_id = rid
                events.add_event(finish)
            elif row >= num_extends:
                reserve = ts.ForwardEvent.UpdateReserveNumTokens()
                reserve.request_id = rid
                reserve.reserve_num_tokens_in_next_schedule_event = 1
                events.add_event(reserve)
        if num_events:
            scheduler.advance(events)

    assert finished == set(generated), (finished, generated)
    assert siblings_submitted, "the shared-prefix siblings were never admitted"
    assert saw_chunking, "the small token budget never produced an intermediate chunk"
    assert saw_prefix_hit, "no sibling request hit the published shared prefix"
    assert saw_sliding_hole, "decode never slid the sliding-window group's pages"


def test_spans_hold_through_retraction():
    """Same span contract as above, under a device pool small enough
    relative to concurrent demand that the victim policy retracts a request
    and later restores it. Mirrors
    test/runtime/test_block_tables_bridge.py's
    ``test_forward_tables_match_through_retraction``, and
    test_retraction_image.py's proven shape (one request with 3x the pages
    of the other three, decoded together until growth blocks everyone)."""
    cfg = _config(num_usable_pages=20, max_scheduled_tokens=64)
    scheduler = ts.Scheduler(cfg)

    request_ids = ("a", "b", "c", "d")
    specs = [_spec("a", list(range(1, 13)), max_new_tokens=20)] + [
        _spec(
            rid, [100 * i + 1, 100 * i + 2, 100 * i + 3, 100 * i + 4], max_new_tokens=20
        )
        for i, rid in enumerate(request_ids[1:], 1)
    ]
    scheduler.submit_requests(specs)

    generated = {s.request_id: 0 for s in specs}
    finished: set[str] = set()
    retracted = 0
    idle_rounds = 0
    STALL_ROUNDS = 100

    for round_index in range(MAX_ROUNDS):
        plan = scheduler.next_execution_plan()
        retracted += _land_cache_ops(scheduler, plan)
        batch = next((op for op in plan.forward if list(op.request_ids)), None)
        if batch is None:
            idle_rounds += 1
            if finished == set(generated):
                break
            if idle_rounds >= STALL_ROUNDS:
                pytest.fail(
                    f"stalled at round {round_index}: generated={generated}, "
                    f"retracted={retracted}"
                )
            continue
        idle_rounds = 0

        arrays = batch.block_tables_arrays()
        spans = batch.block_table_spans()
        _assert_span_invariants(arrays, spans)

        num_extends = batch.num_extends()
        prefill_lengths = list(batch.prefill_lengths)
        extend_prefix_lens = list(batch.extend_prefix_lens)
        input_lengths = list(batch.input_lengths)

        events = ts.ExecutionEvent()
        num_events = 0
        for row, rid in enumerate(batch.request_ids):
            if rid in finished:
                continue
            if row < num_extends and extend_prefix_lens[row] + input_lengths[row] < (
                prefill_lengths[row]
            ):
                continue
            result = ts.ForwardEvent.ExtendResult()
            result.request_id = rid
            generated[rid] += 1
            result.tokens = [9000 + generated[rid]]
            events.add_event(result)
            num_events += 1
            if generated[rid] >= 20:
                finished.add(rid)
                finish = ts.ForwardEvent.Finish()
                finish.request_id = rid
                events.add_event(finish)
            elif row >= num_extends:
                reserve = ts.ForwardEvent.UpdateReserveNumTokens()
                reserve.request_id = rid
                reserve.reserve_num_tokens_in_next_schedule_event = 1
                events.add_event(reserve)
        if num_events:
            scheduler.advance(events)

    assert finished == set(generated), (finished, generated)
    assert retracted > 0, (
        "this configuration never triggered a retraction; the span "
        "assertions above still ran on every forward op, but the "
        "retraction + restore path specifically is uncovered by this test"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
