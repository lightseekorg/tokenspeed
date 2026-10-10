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

"""Tests for cache-related Python bindings."""

import pytest
import tokenspeed_scheduler as ts
from tokenspeed_scheduler import Cache, ExecutionEvent


def test_removed_storage_cache_api_is_not_exported():
    assert not hasattr(ts, "PrefixCacheAdjunctSpec")

    config = ts.SchedulerConfig()
    assert not hasattr(config, "prefix_cache_adjunct")
    assert not hasattr(config, "prefetch_threshold")

    request = ts.RequestSpec()
    assert not hasattr(request, "rolling_hashes")
    assert not hasattr(request, "storage_hit_pages")

    # Cache.PrefetchOp / Cache.PrefetchDoneEvent exist again, as the
    # pre-admission L3 fill (not the old adjunct API).
    assert not hasattr(Cache, "BackUpOp")
    assert not hasattr(Cache, "CacheKind")
    assert not hasattr(Cache.WriteBackOp, "is_retract")
    assert not hasattr(Cache.WriteBackOp, "src_pages_by_kind")
    assert not hasattr(Cache.LoadBackOp, "src_pages_by_kind")
    assert not hasattr(ts.Forward.Batch, "hist_token_lens")
    assert not hasattr(ts.Scheduler, "get_request_paged_cache_page_ids")
    # Retraction suspends with an image; there is no recompute-from-scratch
    # event of any kind (an L3 object that is gone is found by the prefetch).
    assert not hasattr(ts.ForwardEvent, "Retract")
    assert not hasattr(ts.ForwardEvent, "RecomputeRetract")
    assert not hasattr(ts.Scheduler, "has_recoverable_snapshot")


def test_write_back_op_carries_the_source_guard():
    # The runtime branches on how the scheduler guards the Device sources
    # (pinned until the ACK, or released and stream-ordered), never on why.
    assert hasattr(Cache.WriteBackOp, "source_pinned")


def test_cache_event_fields_are_bound():
    write_back = Cache.WriteBackDoneEvent()
    write_back.op_id = 7
    assert write_back.op_id == 7

    snapshot = Cache.SnapshotDoneEvent()
    snapshot.op_id = 10
    assert snapshot.op_id == 10

    restore = Cache.RestoreDoneEvent()
    restore.op_id = 11
    assert restore.op_id == 11

    load_back = Cache.LoadBackDoneEvent(8)
    assert load_back.op_id == 8
    assert not hasattr(
        load_back, "success"
    ), "a load cannot miss: an L3 miss is a PrefetchDoneEvent"

    prefetch = Cache.PrefetchDoneEvent(12, 3)
    assert prefetch.op_id == 12
    assert prefetch.landed_pages == 3
    with pytest.raises(TypeError):
        Cache.PrefetchDoneEvent()
    with pytest.raises(TypeError):
        Cache.PrefetchDoneEvent(13)


def test_execution_event_accepts_cache_events():
    execution_event = ExecutionEvent()

    write_back = Cache.WriteBackDoneEvent()
    assert execution_event.add_event(write_back) is execution_event

    load_back = Cache.LoadBackDoneEvent(8)
    assert execution_event.add_event(load_back) is execution_event

    assert execution_event.add_event(Cache.PrefetchDoneEvent(9, 0)) is execution_event
    assert execution_event.add_event(Cache.SnapshotDoneEvent()) is execution_event
    assert execution_event.add_event(Cache.RestoreDoneEvent()) is execution_event


def test_snapshot_ops_carry_the_blob_slot_and_source_tier():
    # Both image legs name the request slot whose state blob travels with the
    # pages; the restore additionally says which Host tier each row comes from.
    for op in (Cache.SnapshotOp, Cache.RestoreOp):
        for field in (
            "op_ids",
            "request_ids",
            "request_pool_indices",
            "snapshot_slots",
            "group_ids",
            "src_pages",
            "dst_pages",
        ):
            assert hasattr(op, field), (op, field)
    for field in ("content_hashes", "page_offsets", "source_tiers"):
        assert hasattr(Cache.RestoreOp, field)
        assert not hasattr(Cache.SnapshotOp, field)
    assert int(Cache.HostTier.L2) == 0
    assert int(Cache.HostTier.SnapshotPool) == 1


def test_recompute_retract_is_gone_and_prefetch_op_is_bound():
    # There is no retraction without an image: the L3-miss case is handled
    # before admission by the prefetch op, never by a forward-event retract.
    assert not hasattr(ts.ForwardEvent, "RecomputeRetract")
    assert not hasattr(ts.ForwardEvent, "Retract")
    for field in (
        "op_ids",
        "request_ids",
        "first_pages",
        "num_pages",
        "group_ids",
        "host_pages",
        "content_hashes",
        "page_offsets",
        "page_indices",
    ):
        assert hasattr(Cache.PrefetchOp, field), field
    assert not hasattr(Cache.LoadBackOp, "prefetch_from_storage")
    assert not hasattr(ts.Scheduler, "waiting_prefix_hashes")


def test_scheduler_aborts_are_bound():
    assert int(ts.AbortReason.ImageDoesNotFit) == 0
    for field in ("request_id", "reason", "detail"):
        assert hasattr(ts.SchedulerAbort, field), field
    assert hasattr(ts.ExecutionPlan, "aborts")


def test_snapshot_pool_is_stated_explicitly():
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 64
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full_attention", block_granularity=16, total_pages=64
        )
    ]
    # Unset: the pool is a behavioural choice, not a default. The diagnostics
    # name the server args the operator has to change.
    with pytest.raises(ValueError, match="--retraction-snapshot-host-gb"):
        ts.Scheduler(cfg)
    cfg.num_snapshot_pages = 9
    with pytest.raises(ValueError, match="--retraction-snapshot-max-requests"):
        ts.Scheduler(cfg)
    cfg.max_retracted_requests = 2
    # The debug knob needs somewhere to put the images it forces.
    cfg.num_snapshot_pages = 1
    cfg.max_retracted_requests = 0
    cfg.debug_force_retraction_interval = 3
    with pytest.raises(ValueError, match="--debug-force-retraction-interval"):
        ts.Scheduler(cfg)
    cfg.num_snapshot_pages = 9
    cfg.max_retracted_requests = 2
    cfg.debug_force_retraction_interval = 0
    scheduler = ts.Scheduler(cfg)
    assert scheduler.snapshot_pool_free_blocks() == 8
    assert scheduler.retracted_size() == 0
    assert scheduler.host_pool_pinned_blocks() == 0
    cfg.role = ts.SchedulerConfig.Role.P
    cfg.cache_groups[0].transfer_policy = ts.CacheTransferPolicy.FullSuffix
    with pytest.raises(ValueError, match="P role"):
        ts.Scheduler(cfg)
