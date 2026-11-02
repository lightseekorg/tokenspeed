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

"""Scheduler integration: remote objects become Host data before admission."""

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _finish, _spec, storage_ops
from test_async_l3 import done, event, requests, tick


def config(*, host=32, grain=2, groups=None):
    c = ts.SchedulerConfig()
    c.prefix_granularity = grain
    c.num_snapshot_pages = 1
    c.num_device_pages = 64
    c.num_host_pages = host
    c.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
    c.max_scheduled_tokens = 64
    c.max_batch_size = 8
    c.enable_l3_storage = True
    c.storage_config = ts.StorageConfig(
        8,
        64,
        host - 1,
        16,
        64,
        grain,
        100,
        100,
        ts.StoragePressure(8, 1048576, 1000),
        ts.StoragePressure(4, 524288, 500),
    )
    c.cache_groups = groups or [group("full", grain, ts.AttnKind.Full)]
    c.host_group_packing = [
        entry.cache_blocks_per_lcm_block for entry in c.cache_groups
    ]
    return c


def group(name, grain, family, *, window=None):
    return ts.CacheGroupConfig(
        group_id=name,
        block_granularity=0 if family == ts.AttnKind.State else grain,
        total_pages=64,
        attn_kind=family if window is None else ts.AttnKind.SlidingWindow,
        sliding_window_tokens=window,
    )


def start(c):
    s = ts.Scheduler(c)
    tick(s, 1, forward=True, storage=True)
    return s


def submit(s, rid="r", tokens=None):
    spec = _spec(rid, list(range(8)) if tokens is None else tokens)
    if tokens is None:
        # Keep the general transaction cases aligned; tail cases supply tokens.
        spec.max_cached_prefix_tokens = 6
    s.submit_requests([spec])
    return storage_ops(s.next_execution_plan())[0]


def read(s, query, flags=None):
    done(s, query, [1] * len(query.keys) if flags is None else flags, retired=True)
    return s.next_execution_plan()


def acknowledge_get(s, plan):
    op = next(x for x in storage_ops(plan) if x.kind == ts.StorageKind.Get)
    done(s, op, [1] * len(op.keys), retired=True)
    return s.next_execution_plan()


def test_remote_miss_enters_normal_prefill_without_loadback():
    s = start(config())
    q = submit(s)
    plan = read(s, q, [0] * len(q.keys))
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert not any(isinstance(x, ts.Cache.LoadBackOp) for x in plan.cache)


def test_remote_hint_alone_never_claims_unfilled_host_data():
    s = start(config())
    q = submit(s)
    assert q.kind == ts.StorageKind.Query
    assert s.host_pool_cached_blocks() == 0
    get_plan = read(s, q)
    assert s.host_pool_cached_blocks() == 0
    assert not requests(get_plan)
    plan = acknowledge_get(s, get_plan)
    assert list(plan.forward[0].extend_prefix_lens) == [6]


def test_host_shortage_clamps_complete_prefix_before_device_admission():
    s = start(config(host=3))
    plan = acknowledge_get(s, read(s, submit(s)))
    batch = plan.forward[0]
    assert list(batch.extend_prefix_lens) == [4]
    assert list(batch.input_lengths) == [4]


def test_mixed_grain_shortage_cannot_skip_unrestored_prefix():
    c = config(
        host=2,
        grain=4,
        groups=[
            group("fine", 2, ts.AttnKind.Full),
            group("coarse", 4, ts.AttnKind.Full),
        ],
    )
    s = start(c)
    plan = read(s, submit(s))
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert s.storage_held_blocks() == 0


def test_swa_missing_cannot_be_replaced_by_full_kv_hit():
    s = start(
        config(
            groups=[
                group("full", 2, ts.AttnKind.Full),
                group("swa", 2, ts.AttnKind.Full, window=4),
            ]
        )
    )
    q = submit(s)
    plan = read(s, q, [int(key[0] == 0) for key in q.keys])
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]


@pytest.mark.parametrize("state_groups", [1, 3])
def test_sparse_state_hit_uses_existing_group_matcher(state_groups):
    s = start(
        config(
            groups=[
                group("full", 2, ts.AttnKind.Full),
                *[
                    group(f"state{i}", 2, ts.AttnKind.State)
                    for i in range(state_groups)
                ],
            ]
        )
    )
    q = submit(s)
    checkpoint = s.prefix_hashes_for_tokens(list(range(8)))[-1]
    final_states = {(gid, checkpoint, 0) for gid in range(1, state_groups + 1)}
    flags = [int(key[0] == 0 or tuple(key) in final_states) for key in q.keys]
    get_plan = read(s, q, flags)
    get = storage_ops(get_plan)[0]
    assert sum(key[0] != 0 for key in get.keys) == state_groups
    plan = acknowledge_get(s, get_plan)
    assert list(plan.forward[0].extend_prefix_lens) == [6]


def test_partial_get_uses_only_complete_contiguous_prefix():
    s = start(config())
    get = storage_ops(read(s, submit(s)))[0]
    done(s, get, [1, 0, 1], retired=True)
    plan = s.next_execution_plan()
    assert list(plan.forward[0].extend_prefix_lens) == [2]


def test_l2_warm_new_request_does_not_send_query():
    s = start(config())
    plan = acknowledge_get(s, read(s, submit(s)))
    load = next(x for x in plan.cache if isinstance(x, ts.Cache.LoadBackOp))
    event(s, ts.Cache.LoadBackDoneEvent(load.op_ids[0]))
    s.submit_requests([_spec("warm", list(range(8)))])
    plan = s.next_execution_plan()
    assert "warm" in requests(plan)
    assert not any(x.kind == ts.StorageKind.Query for x in storage_ops(plan))


def test_abort_id_reuse_does_not_publish_old_get():
    s = start(config())
    old_get = storage_ops(read(s, submit(s)))[0]
    abort = ts.ForwardEvent.Abort()
    abort.request_id = "r"
    event(s, abort)
    cancel_plan = s.next_execution_plan()
    assert storage_ops(cancel_plan)[0].kind == ts.StorageKind.Cancel
    query = submit(s, tokens=list(range(10, 18)))
    assert query.op_id != old_get.op_id
    done(s, old_get, [1] * len(old_get.keys), retired=True)
    assert s.host_pool_cached_blocks() == 0
    assert s.storage_pending()


def test_l3_put_failure_preserves_published_host_cache():
    s = start(config())
    q = submit(s)
    read(s, q, [0] * len(q.keys))
    _advance(s, "r", [99])
    plan = s.next_execution_plan()
    wb = next(x for x in plan.cache if isinstance(x, ts.Cache.WriteBackOp))
    ack = ts.Cache.WriteBackDoneEvent()
    ack.op_id = wb.op_ids[0]
    event(s, ack)
    cached = s.host_pool_cached_blocks()
    put = next(
        x for x in storage_ops(s.next_execution_plan()) if x.kind == ts.StorageKind.Put
    )
    assert cached > 0 and s.storage_held_blocks() > 0
    done(s, put, [0] * len(put.keys), retired=True)
    assert s.host_pool_cached_blocks() == cached
    assert s.storage_held_blocks() == 0


def test_flush_rejects_query_without_erasing_its_operation():
    s = start(config())
    q = submit(s)
    assert not s.clear_cache()
    assert s.storage_pending()
    assert s.clear_l1_cache()
    plan = read(s, q, [0] * len(q.keys))
    assert requests(plan) == ["r"]


def test_state_groups_cannot_resume_from_different_remote_checkpoints():
    s = start(
        config(
            groups=[
                group("full", 2, ts.AttnKind.Full),
                group("state0", 0, ts.AttnKind.State),
                group("state1", 0, ts.AttnKind.State),
            ]
        )
    )
    q = submit(s)
    states = {gid: [tuple(k) for k in q.keys if k[0] == gid] for gid in (1, 2)}
    # One recurrent group has only the last checkpoint, the other only the
    # first. Full KV coverage cannot make either mismatched state usable.
    hits = {states[1][-1], states[2][0]}
    plan = read(s, q, [int(k[0] == 0 or tuple(k) in hits) for k in q.keys])
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert not any(op.kind == ts.StorageKind.Get for op in storage_ops(plan))


def test_storage_configuration_cannot_be_silently_ignored():
    cfg = config()
    cfg.enable_l3_storage = False
    with pytest.raises(ValueError, match="StorageConfig requires L3"):
        ts.Scheduler(cfg)


@pytest.mark.parametrize("kinds", [("full",), ("state",), ("full", "swa", "state")])
@pytest.mark.parametrize("suffix_tokens", [1, 4])
def test_cold_reader_restores_actual_published_unaligned_checkpoint(
    kinds, suffix_tokens
):
    groups = [
        group(
            kind,
            2,
            ts.AttnKind.State if kind == "state" else ts.AttnKind.Full,
            window=4 if kind == "swa" else None,
        )
        for kind in kinds
    ]
    writer = start(config(groups=groups))
    query = submit(writer, tokens=list(range(5)))
    plan = read(writer, query, [0] * len(query.keys))
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    _advance(writer, "r", [77])
    _finish(writer, "r")
    plan = writer.next_execution_plan()
    writes = [op for op in plan.cache if isinstance(op, ts.Cache.WriteBackOp)]
    assert writes
    for op in writes:
        for op_id in op.op_ids:
            ack = ts.Cache.WriteBackDoneEvent()
            ack.op_id = op_id
            event(writer, ack)
    puts = [
        op
        for op in storage_ops(writer.next_execution_plan())
        if op.kind == ts.StorageKind.Put
    ]
    remote = {tuple(key) for op in puts for key in op.keys}
    assert remote
    for op in puts:
        done(writer, op, [1] * len(op.keys), retired=True)

    # A separate scheduler has no local TailTable or Host/Device cache state.
    reader = start(config(groups=groups))
    spec = _spec("r", [0, 1, 2, 3, 4] + [88] * suffix_tokens)
    # Also cover an explicit, nonaligned logprob/replay bound inside a longer
    # prompt: nothing beyond that token count may be claimed.
    spec.max_cached_prefix_tokens = 5
    reader.submit_requests([spec])
    query = storage_ops(reader.next_execution_plan())[0]
    plan = read(reader, query, [int(tuple(key) in remote) for key in query.keys])
    get = next(op for op in storage_ops(plan) if op.kind == ts.StorageKind.Get)
    assert all(tuple(key) in remote for key in get.keys)
    assert [sum(key[0] == gid for key in get.keys) for gid in range(len(kinds))] == [
        {"full": 3, "swa": 2, "state": 1}[kind] for kind in kinds
    ]
    assert reader.host_pool_cached_blocks() == 0
    plan = acknowledge_get(reader, plan)
    assert list(plan.forward[0].extend_prefix_lens) == [5]
    assert list(plan.forward[0].input_lengths) == [suffix_tokens]
    assert any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)


def test_l3_preparation_with_no_free_request_slot():
    c = config(
        host=32, grain=2, groups=[group("full", 2, ts.AttnKind.Full, window=None)]
    )
    c.max_batch_size = 1
    c.enable_mixed_prefill_decode = False
    c.disable_prefix_cache = False
    s = start(c)
    s.submit_requests([_spec("resident", [100, 101])])
    plan = s.next_execution_plan()
    for query in storage_ops(plan):
        done(s, query, [0] * len(query.keys), retired=True)
    if not requests(plan):
        plan = s.next_execution_plan()
    assert requests(plan) == ["resident"]
    _advance(s, "resident", [102])

    waiting = _spec("waiting", list(range(8)))
    waiting.max_cached_prefix_tokens = 6
    s.submit_requests([waiting])
    plan = s.next_execution_plan()
    assert requests(plan) == ["resident"]
    _advance(s, "resident", [103])
    query = storage_ops(plan)[0]
    assert query.kind == ts.StorageKind.Query
    plan = read(s, query, [1] * len(query.keys))
    assert requests(plan) == ["resident"]
    _advance(s, "resident", [104])
    plan = acknowledge_get(s, plan)
    assert requests(plan) == ["resident"]
    assert not any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)
    _advance(s, "resident", [105])

    _finish(s, "resident")
    plan = s.next_execution_plan()
    assert requests(plan) == ["waiting"]
    assert list(plan.forward[0].extend_prefix_lens) == [6]
    assert any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)
