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

"""Real C++ scheduler; only device computation/completion is a CPU stand-in."""

import numpy as np
import pytest
import tokenspeed_scheduler as ts


def config(*, snapshot, overlap, host, batch_size=4):
    c = ts.SchedulerConfig()
    c.prefix_granularity = 8
    c.max_scheduled_tokens = 1024
    c.max_batch_size = batch_size
    c.num_device_pages = 128
    c.num_host_pages = 128 if host else 0
    c.disable_l2_cache = not host
    c.num_snapshot_pages = 128 if snapshot else 1
    c.max_retracted_requests = 4 if snapshot else 0
    c.decode_input_tokens = 8
    c.overlap_schedule_depth = overlap
    c.enable_l3_storage = False
    c.disable_prefix_cache = False
    c.cache_groups = [
        ts.CacheGroupConfig(
            group_id="kv",
            block_granularity=8,
            total_pages=128,
            attn_kind=ts.AttnKind.Full,
        ),
        ts.CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=128,
            attn_kind=ts.AttnKind.State,
        ),
    ]
    return c


def scheduler(*, snapshot=False, overlap=0, host=False, deferred=True, batch_size=4):
    c = config(snapshot=snapshot, overlap=overlap, host=host, batch_size=batch_size)
    return ts.Scheduler(c, ["state"]) if deferred else ts.Scheduler(c)


def send(s, *events):
    e = ts.ExecutionEvent()
    for item in events:
        e.add_event(item)
    s.advance(e)


def result(s, rid, tokens):
    e = ts.ForwardEvent.ExtendResult()
    e.request_id, e.tokens = rid, tokens
    send(s, e)


def control(s, rid, kind):
    e = kind()
    e.request_id = rid
    send(s, e)


def submit(s, rid, tokens):
    spec = ts.RequestSpec()
    spec.request_id, spec.tokens, spec.max_new_tokens = rid, tokens, 100
    s.submit_requests([spec])


def forwards(plan):
    return [b for b in plan.forward if b.request_ids]


def row(batch, rid):
    i = list(batch.request_ids).index(rid)
    return list(batch.block_tables["state"][i])


def ack_writebacks(s, plan):
    for op in plan.cache:
        if isinstance(op, ts.Cache.WriteBackOp):
            for op_id in op.op_ids:
                event = ts.Cache.WriteBackDoneEvent()
                event.op_id = int(op_id)
                send(s, event)


def started(s, rid="r"):
    submit(s, rid, list(range(125)))
    p = s.next_execution_plan()
    ack_writebacks(s, p)
    assert len(forwards(p)) == 1
    result(s, rid, [1000])
    p = s.next_execution_plan()
    ack_writebacks(s, p)
    b = forwards(p)[0]
    assert not p.state_prepares
    assert b.num_extends() == 0
    result(s, rid, [1001, 1002, 1003, 1004, 1005])
    return int(b.request_pool_indices[0]), row(b, rid)[1]


def ack(s, op, *, success):
    send(s, ts.Cache.StatePrepareDoneEvent(op.op_id, success))


def test_finish_holds_until_ready_and_publishes_exact_consumed_frontier():
    s = scheduler()
    slot, block = started(s)
    before = s.active_lcm_blocks()
    control(s, "r", ts.ForwardEvent.Finish)
    p = s.next_execution_plan()
    (op,) = p.state_prepares
    ack_writebacks(s, p)
    assert (op.request_id, op.request_pool_index, op.num_computed_tokens) == (
        "r",
        slot,
        130,
    )
    assert dict(op.state_blocks) == {"state": block}
    assert s.active_lcm_blocks() == before
    assert not s.can_clear_cache()
    assert not s.clear_l1_cache()
    for _ in range(4):
        p = s.next_execution_plan()
        assert not forwards(p) and not p.state_prepares and not p.cache
        assert s.active_lcm_blocks() == before
    ack(s, op, success=True)
    assert s.active_lcm_blocks() == 0
    # The last returned token 1005 is not consumed. It must not be hashed as
    # state T=131. This query can hit the complete checkpoint at T=130.
    submit(s, "hit", list(range(125)) + [1000, 1001, 1002, 1003, 1004, 9999])
    b = forwards(s.next_execution_plan())[0]
    assert list(b.extend_prefix_lens) == [130]
    assert row(b, "hit")[0] == block


def test_unready_state_is_not_prefix_visible():
    s = scheduler()
    _, block = started(s)
    control(s, "r", ts.ForwardEvent.Finish)
    (op,) = s.next_execution_plan().state_prepares
    submit(s, "miss", list(range(125)) + [1000, 1001, 1002, 1003, 1004, 9999])
    b = forwards(s.next_execution_plan())[0]
    assert int(b.extend_prefix_lens[0]) < 130
    assert row(b, "miss")[0] != block
    ack(s, op, success=True)


def test_finish_waits_for_all_overlapped_forward_results():
    s = scheduler(overlap=1)
    started(s)
    b = forwards(s.next_execution_plan())[0]
    control(s, "r", ts.ForwardEvent.Finish)
    assert not s.next_execution_plan().state_prepares
    result(s, "r", [1006, 1007, 1008])
    (op,) = s.next_execution_plan().state_prepares
    assert op.num_computed_tokens == 133
    assert op.request_pool_index == int(b.request_pool_indices[0])
    ack(s, op, success=True)


@pytest.mark.parametrize("abort_before_submit", [False, True])
def test_abort_does_not_reuse_slot_until_writer_drains(abort_before_submit):
    s = scheduler(batch_size=1)
    slot, _ = started(s)
    control(s, "r", ts.ForwardEvent.Finish)
    if abort_before_submit:
        control(s, "r", ts.ForwardEvent.Abort)
    p = s.next_execution_plan()
    (op,) = p.state_prepares
    if not abort_before_submit:
        control(s, "r", ts.ForwardEvent.Abort)
    submit(s, "next", [900, 901])
    assert not forwards(s.next_execution_plan())
    assert s.active_lcm_blocks() > 0
    ack(s, op, success=True)
    p = s.next_execution_plan()
    b = forwards(p)[0]
    assert list(b.request_ids) == ["next"]
    assert int(b.request_pool_indices[0]) == slot
    assert int(b.extend_prefix_lens[0]) == 0
    result(s, "next", [902])
    control(s, "next", ts.ForwardEvent.Finish)
    (next_op,) = s.next_execution_plan().state_prepares
    assert next_op.op_id != op.op_id
    ack(s, op, success=True)  # stale ACK cannot release the slot's new owner
    assert s.active_lcm_blocks() > 0
    ack(s, next_op, success=True)
    assert s.active_lcm_blocks() == 0


def test_failed_prepare_never_publishes_and_reports_existing_abort_channel():
    s = scheduler()
    started(s)
    control(s, "r", ts.ForwardEvent.Finish)
    (op,) = s.next_execution_plan().state_prepares
    ack(s, op, success=False)
    p = s.next_execution_plan()
    assert [a.request_id for a in p.aborts] == ["r"]
    assert p.aborts[0].reason == ts.AbortReason.StatePrepareFailed
    assert s.active_lcm_blocks() == 0
    submit(s, "probe", list(range(125)) + [1000, 1001, 1002, 1003, 1004, 9999])
    b = forwards(s.next_execution_plan())[0]
    assert int(b.extend_prefix_lens[0]) < 130
    ack(s, op, success=True)  # duplicate success cannot undo failure


def test_eager_backend_preserves_original_finish_without_prepare():
    s = scheduler(deferred=False)
    started(s)
    control(s, "r", ts.ForwardEvent.Finish)
    assert s.active_lcm_blocks() == 0
    assert not s.next_execution_plan().state_prepares


@pytest.mark.parametrize("abort", [False, True])
def test_finish_or_abort_while_retraction_is_preparing(abort):
    c = config(snapshot=True, overlap=0, host=False)
    c.debug_force_retraction_interval = 1
    s = ts.Scheduler(c, ["state"])
    started(s)
    (op,) = s.next_execution_plan().state_prepares
    control(s, "r", ts.ForwardEvent.Abort if abort else ts.ForwardEvent.Finish)
    assert s.active_lcm_blocks() > 0
    ack(s, op, success=True)
    p = s.next_execution_plan()
    assert not p.state_prepares and not p.cache
    assert s.active_lcm_blocks() == 0
    assert s.retracted_size() == 0


def test_unrelated_completion_cannot_release_preparing_request():
    s = scheduler()
    started(s)
    control(s, "r", ts.ForwardEvent.Finish)
    (op,) = s.next_execution_plan().state_prepares
    before = s.active_lcm_blocks()
    send(s, ts.Cache.StatePrepareDoneEvent(op.op_id + 100, True))
    assert s.active_lcm_blocks() == before
    ack(s, op, success=True)


def test_ordinary_decode_never_prepares_even_across_page_boundaries():
    s = scheduler()
    started(s)
    for i in range(6):
        p = s.next_execution_plan()
        assert not p.state_prepares
        assert forwards(p)[0].num_extends() == 0
        result(s, "r", [2000 + 10 * i + j for j in range(5)])
    control(s, "r", ts.ForwardEvent.Finish)
    (op,) = s.next_execution_plan().state_prepares
    assert op.num_computed_tokens == 160


@pytest.mark.parametrize("host", [False, True])
def test_retraction_waits_then_reuses_existing_snapshot_and_restore(host):
    c = config(snapshot=True, overlap=0, host=host)
    c.debug_force_retraction_interval = 1
    s = ts.Scheduler(c, ["state"])
    started(s)
    p = s.next_execution_plan()
    (op,) = p.state_prepares
    assert not forwards(p) and not any(
        isinstance(x, ts.Cache.SnapshotOp) for x in p.cache
    )
    assert s.retracted_size() == 0 and s.active_lcm_blocks() > 0
    assert not s.next_execution_plan().state_prepares  # exactly once
    ack(s, op, success=True)
    p = s.next_execution_plan()
    snapshots = [x for x in p.cache if isinstance(x, ts.Cache.SnapshotOp)]
    assert len(snapshots) == 1
    assert list(snapshots[0].request_ids) == ["r"]
    assert s.retracted_size() == 1
    # Snapshot must still gate restore, independently of prepare readiness.
    assert not any(
        isinstance(x, ts.Cache.RestoreOp) for x in s.next_execution_plan().cache
    )
    for cache_op in p.cache:
        if isinstance(cache_op, ts.Cache.WriteBackOp):
            for op_id in cache_op.op_ids:
                e = ts.Cache.WriteBackDoneEvent()
                e.op_id = op_id
                send(s, e)
    e = ts.Cache.SnapshotDoneEvent()
    e.op_id = snapshots[0].op_ids[0]
    send(s, e)
    p = s.next_execution_plan()
    assert any(isinstance(x, ts.Cache.RestoreOp) for x in p.cache)


def test_cpu_delta_ring_materializes_accepted_state_before_publication():
    # A small delta-rule oracle checks the protocol's data handoff. This is
    # not a numerical test of the Ascend implementation.
    rng = np.random.default_rng(42)
    s0 = rng.normal(size=(4, 4))
    k = rng.normal(size=(8, 4)) * 0.1
    e = rng.normal(size=(8, 4)) * 0.1
    d = rng.uniform(0.8, 1, size=(8, 4))

    def replay(state, count):
        for i in range(count):
            state = d[i, :, None] * state + np.outer(k[i], e[i])
        return state

    s = scheduler()
    _, block = started(s)
    memory = {block: s0.copy()}
    control(s, "r", ts.ForwardEvent.Finish)
    (op,) = s.next_execution_plan().state_prepares
    assert op.num_computed_tokens - 125 == 5
    np.testing.assert_array_equal(memory[block], s0)
    memory[block] = replay(memory[block], op.num_computed_tokens - 125)
    ack(s, op, success=True)
    submit(s, "hit", list(range(125)) + [1000, 1001, 1002, 1003, 1004, 9999])
    b = forwards(s.next_execution_plan())[0]
    np.testing.assert_allclose(
        memory[row(b, "hit")[0]], replay(s0.copy(), 5), rtol=0, atol=0
    )
    assert not np.allclose(memory[block], replay(s0.copy(), 8))
