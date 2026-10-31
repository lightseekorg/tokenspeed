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
from conftest import (
    K3_GROUP_IDS,
    PAGE,
    _advance,
    _find_forward_op,
    _finish,
    _make_k3_config,
    _spec,
)
from tokenspeed_scheduler import Cache, ExecutionEvent


def test_removed_storage_cache_api_is_not_exported():
    assert not hasattr(ts, "PrefixCacheAdjunctSpec")

    config = ts.SchedulerConfig()
    assert not hasattr(config, "prefix_cache_adjunct")
    assert not hasattr(config, "prefetch_threshold")

    request = ts.RequestSpec()
    assert not hasattr(request, "rolling_hashes")
    assert not hasattr(request, "storage_hit_pages")

    # The pre-admission prefetch protocol is gone with the adjunct API.
    assert not hasattr(Cache, "PrefetchOp")
    assert not hasattr(Cache, "PrefetchDoneEvent")
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


def test_execution_event_accepts_cache_events():
    execution_event = ExecutionEvent()

    write_back = Cache.WriteBackDoneEvent()
    assert execution_event.add_event(write_back) is execution_event

    load_back = Cache.LoadBackDoneEvent(8)
    assert execution_event.add_event(load_back) is execution_event

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
            group_id="full_attention",
            block_granularity=16,
            total_pages=64,
            attn_kind=ts.AttnKind.Full,
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
    cfg.host_group_packing = [1]
    scheduler = ts.Scheduler(cfg)
    assert scheduler.snapshot_pool_free_blocks() == 8
    assert scheduler.retracted_size() == 0
    assert scheduler.host_pool_pinned_blocks() == 0
    # No Host pool: its gauges read 0.
    assert scheduler.host_pool_total_blocks() == 0
    assert scheduler.host_pool_free_blocks() == 0
    cfg.role = ts.SchedulerConfig.Role.P
    cfg.cache_groups[0].transfer_policy = ts.CacheTransferPolicy.FullSuffix
    with pytest.raises(ValueError, match="P role"):
        ts.Scheduler(cfg)


def test_host_write_policy_round_trips():
    policy = ts.SchedulerConfig.HostWritePolicy
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 64
    cfg.num_host_pages = 9
    cfg.num_snapshot_pages = 1
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full_attention",
            block_granularity=16,
            total_pages=64,
            attn_kind=ts.AttnKind.Full,
            # The D role below requires it.
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    cfg.host_group_packing = [1]
    # Unset: what feeds Host L2 is a behavioural choice, not a default.
    assert cfg.host_write_policy == policy.Unspecified
    with pytest.raises(ValueError, match="explicit host_write_policy"):
        ts.Scheduler(cfg)
    cfg.host_write_policy = policy.WriteBack
    assert cfg.host_write_policy == policy.WriteBack
    assert ts.Scheduler(cfg).host_pool_total_blocks() == 8
    cfg.role = ts.SchedulerConfig.Role.D
    with pytest.raises(
        ValueError, match="host_write_policy WriteBack is not available on the D role"
    ):
        ts.Scheduler(cfg)
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.host_write_policy = policy.WriteThrough
    assert cfg.host_write_policy == policy.WriteThrough
    assert ts.Scheduler(cfg).host_pool_total_blocks() == 8


def test_an_eviction_store_is_stream_ordered_and_published_at_its_ack():
    """Under Host write-back (the prefill role) nothing reaches the Host until
    an admission evicts: a handed-off prompt's pages leave the Device for the
    next prompt, in the plan that re-grants them, so their store is
    stream-ordered. Its Host blocks are taken while the copy is in flight,
    and the Host publishes its entries at the ACK."""
    page = 16
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.P
    cfg.prefix_granularity = page
    cfg.max_scheduled_tokens = 256
    cfg.max_batch_size = 4
    # Four usable pages: two 40-token prompts do not fit together.
    cfg.num_device_pages = 4 + 1
    cfg.num_host_pages = 8 + 1
    cfg.num_snapshot_pages = 1  # the P role never retracts
    cfg.disable_l2_cache = False
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=page,
            total_pages=cfg.num_device_pages,
            attn_kind=ts.AttnKind.Full,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    cfg.host_group_packing = [1]
    scheduler = ts.Scheduler(cfg)

    def prefill_and_hand_off(rid, tokens):
        # The prompt's forward, then its hand-off to the decode peer, whose
        # success caches the prompt on the Device; the two plans' cache ops.
        scheduler.submit_requests([_spec(rid, tokens)])
        scheduler.advance(ExecutionEvent().add_event(ts.PD.BootstrappedEvent(rid)))
        prefill = scheduler.next_execution_plan()
        assert list(_find_forward_op(prefill).request_ids) == [rid]
        _advance(scheduler, rid, [1000])
        hand_off = scheduler.next_execution_plan()
        assert list(hand_off.remote_decode.request_ids) == [rid]
        scheduler.advance(ExecutionEvent().add_event(ts.PD.SucceededEvent(rid)))
        return list(prefill.cache), list(hand_off.cache)

    # A 40-token prompt: two full pages and a tail, cached on the Device once
    # the decode peer has it; nothing is stored to the Host meanwhile.
    assert prefill_and_hand_off("a", list(range(40))) == ([], [])
    assert scheduler.host_pool_cached_blocks() == 0
    assert scheduler.host_pool_free_blocks() == 8

    # The second prompt takes the free page and two of the first prompt's,
    # its tail and the page under it: one store, stream-ordered, in the plan
    # that re-grants them. It takes two Host blocks and publishes nothing
    # while it is in flight, the hand-off's plan included.
    prefill_ops, hand_off_ops = prefill_and_hand_off("b", list(range(500, 540)))
    (store,) = prefill_ops
    assert hand_off_ops == []
    assert isinstance(store, Cache.WriteBackOp)
    assert list(store.source_pinned) == [False]
    assert [len(rows) for rows in store.src_pages] == [2]
    assert scheduler.host_pool_cached_blocks() == 0
    assert scheduler.host_pool_free_blocks() == 6

    # The ACK publishes both entries.
    acks = ExecutionEvent()
    for op_id in store.op_ids:
        done = Cache.WriteBackDoneEvent()
        done.op_id = op_id
        acks.add_event(done)
    scheduler.advance(acks)
    assert scheduler.host_pool_cached_blocks() == 2
    assert scheduler.host_pool_free_blocks() == 6


def test_device_evictions_count_a_requests_own_reclaim_apart():
    """K3 shape on twelve Device parents, a block each, a page per chunk. a
    publishes two pages and its checkpoint at 4 (five parents); b resumes
    there, its first chunk takes four parents more. b's second chunk rotates
    a's checkpoint out of slot 0 and needs four parents with three free: its
    own released checkpoint funds the fourth, counted as the request's
    reclaim, not as an admission's victim."""
    cfg = _make_k3_config()
    cfg.num_device_pages = 13  # page 0 = null sentinel, 12 usable parents
    cfg.max_scheduled_tokens = PAGE
    for group in cfg.cache_groups:
        group.total_pages = cfg.num_device_pages
    scheduler = ts.Scheduler(cfg)
    a = list(range(1, 2 * PAGE + 1))
    for rid, tokens in (("a", a), ("b", a + list(range(50, 50 + 2 * PAGE)))):
        spec = _spec(rid, tokens)
        spec.max_new_tokens = 1
        scheduler.submit_requests([spec])
        computed = 0
        while computed < len(tokens):
            batch = _find_forward_op(scheduler.next_execution_plan())
            assert list(batch.request_ids) == [rid]
            computed = batch.extend_prefix_lens[0] + batch.input_lengths[0]
            _advance(scheduler, rid, [101] if computed == len(tokens) else [])
        _finish(scheduler, rid)

    stats = scheduler.host_cache_stats()
    assert stats.prefix_hits.device_hit_tokens == 2 * PAGE  # b's, at a's checkpoint
    evictions = stats.device_evictions
    # Each State group's released checkpoint at 4.
    assert evictions.request_reclaim.blocks == 3
    assert evictions.admission.blocks == 0
    assert evictions.stranded_suffix.blocks == 0


def test_host_cache_stats_count_publication_and_finish_stores():
    cfg = _make_k3_config()
    cfg.num_host_pages = 33  # page 0 = null sentinel, 32 usable parents
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
    cfg.disable_l2_cache = False
    # Two History blocks per parent; a State block still takes a whole
    # parent.
    history = cfg.cache_groups[0]
    history.cache_blocks_per_lcm_block = 2
    history.total_pages = 2 * 32 + 1
    cfg.host_group_packing = [
        group.cache_blocks_per_lcm_block for group in cfg.cache_groups
    ]
    scheduler = ts.Scheduler(cfg)
    num_groups = len(K3_GROUP_IDS)

    # A fresh scheduler: what this test counts reads 0, and every by-group
    # block list has one entry per cache group.
    fresh = scheduler.host_cache_stats()
    assert (
        len(fresh.publication_stores.blocks)
        == len(fresh.prefix_loads.blocks)
        == num_groups
    )
    assert fresh.host_evictions.lru.blocks == fresh.host_evictions.flush.blocks == 0
    assert fresh.device_evictions.admission.blocks == 0
    assert len(fresh.eviction_stores.blocks) == num_groups
    assert (
        fresh.eviction_stores.ops
        == fresh.eviction_stores.recopied
        == fresh.eviction_stores.cancelled
        == fresh.dead_entries_not_written_back
        == 0
    )
    assert (
        fresh.host_evictions.duplicate.blocks
        == fresh.host_evictions.superseded.blocks
        == fresh.host_evictions.dead_entry.blocks
        == fresh.host_evictions.tail_cap.blocks
        == 0
    )
    assert fresh.publication_stores.ops == fresh.finish_stores.ops == 0
    assert list(fresh.publication_stores.blocks) == [0] * num_groups
    assert list(fresh.finish_stores.blocks) == [0] * num_groups
    assert fresh.finish_stores.host_cached == 0
    assert fresh.prefix_loads.ops == 0
    assert fresh.prefix_hits.prompt_tokens == 0
    assert scheduler.host_pool_total_blocks() == 32
    assert scheduler.host_pool_free_blocks() == 32

    # Prefill, then the first decode, ACKing every write-back. The fused role
    # publishes the prompt at its end -- three full pages, the one-token tail
    # and each State group's checkpoint -- and streams it to Host in one
    # pinned write-back.
    scheduler.submit_requests([_spec("r", [1, 2, 3, 4, 5, 6, 7])])
    publication_rows = [0] * num_groups
    for token in (100, 101):
        plan = scheduler.next_execution_plan()
        acks = ExecutionEvent()
        for op in plan.cache:
            if isinstance(op, Cache.WriteBackOp):
                for op_id, groups in zip(op.op_ids, op.group_ids):
                    for group in groups:
                        publication_rows[group] += 1
                    done = Cache.WriteBackDoneEvent()
                    done.op_id = op_id
                    acks.add_event(done)
        scheduler.advance(acks)
        _advance(scheduler, "r", [token])
    stats = scheduler.host_cache_stats()
    assert publication_rows == [4, 1, 1, 1]
    assert list(stats.publication_stores.blocks) == publication_rows
    assert stats.publication_stores.ops == 1
    assert stats.prefix_hits.prompt_tokens == 7

    # The decode token completed a fourth page. Finish stores it and its
    # State checkpoints (the three earlier pages are on Host already), and
    # drains inside the event: the op is counted before the plan that
    # carries it.
    _finish(scheduler, "r")
    stats = scheduler.host_cache_stats()
    assert stats.finish_stores.ops == 1
    assert list(stats.finish_stores.blocks) == [1, 1, 1, 1]
    assert stats.finish_stores.host_cached == 3
    [store] = [
        op
        for op in scheduler.next_execution_plan().cache
        if isinstance(op, Cache.WriteBackOp)
    ]
    [op_id] = store.op_ids
    [groups] = store.group_ids
    assert list(stats.finish_stores.blocks) == [
        list(groups).count(group) for group in range(num_groups)
    ]

    done = Cache.WriteBackDoneEvent()
    done.op_id = op_id
    acks = ExecutionEvent()
    acks.add_event(done)
    scheduler.advance(acks)
    assert scheduler.host_cache_stats().finish_stores.ops == 1
    # Every stored block is now a published Host entry: five History blocks
    # fill three parents, and six State blocks take a parent each.
    assert scheduler.host_pool_cached_blocks() == 5 + 6
    assert scheduler.host_pool_total_blocks() == 32
    assert scheduler.host_pool_free_blocks() == 32 - 3 - 6


def test_host_cache_stats_count_eviction_write_backs():
    """The prefill role under Host write-back: one Full group of 16-token
    pages on four Device and two Host pages. a's 40-token prompt leaves two
    pages and a tail; b's evicts a's tail and second page and writes both
    back, filling the Host. c's 20-token prompt evicts a's first page and b's
    tail, two segments. The Host plan lists its leaves -- a's tail, and b's on
    its way -- and takes a's tail (Host-only, never loaded), after which a's
    second page is a Host leaf; it stays, its parent being on its way to the
    Host. One free page places b's tail, the newer segment; a's first page is
    dropped whole. With it gone from both tiers, the collection takes a's second
    page from the Host: nothing reaches it any more."""
    page = 16
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = page
    cfg.max_scheduled_tokens = 256
    cfg.max_batch_size = 4
    cfg.num_device_pages = 4 + 1
    cfg.num_host_pages = 2 + 1
    cfg.num_snapshot_pages = 1
    cfg.disable_l2_cache = False
    cfg.role = ts.SchedulerConfig.Role.P
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full_attention",
            block_granularity=page,
            total_pages=4 + 1,
            attn_kind=ts.AttnKind.Full,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    cfg.host_group_packing = [1]
    scheduler = ts.Scheduler(cfg)
    stores = []
    for rid, tokens in (
        ("a", list(range(40))),
        ("b", list(range(100, 140))),
        ("c", list(range(200, 220))),
    ):
        scheduler.submit_requests([_spec(rid, tokens)])
        events = ExecutionEvent()
        events.add_event(ts.PD.BootstrappedEvent(rid))
        scheduler.advance(events)
        # The prefill, then the hand-off to the decode peer, which succeeds.
        plan = scheduler.next_execution_plan()
        acks = ExecutionEvent()
        for op in plan.cache:
            stores.append([list(rows) for rows in op.src_pages])
            for op_id in op.op_ids:
                done = Cache.WriteBackDoneEvent()
                done.op_id = op_id
                acks.add_event(done)
        scheduler.advance(acks)
        _advance(scheduler, rid, [1000])
        assert list(scheduler.next_execution_plan().remote_decode.request_ids) == [rid]
        events = ExecutionEvent()
        events.add_event(ts.PD.SucceededEvent(rid))
        scheduler.advance(events)
    assert [[len(rows) for rows in store] for store in stores] == [[2], [1]]

    stats = scheduler.host_cache_stats()
    eviction_stores = stats.eviction_stores
    assert (eviction_stores.ops, list(eviction_stores.blocks)) == (2, [3])
    assert eviction_stores.no_placement == 1
    assert (eviction_stores.host_cached, eviction_stores.in_flight) == (0, 0)
    assert (eviction_stores.recopied, eviction_stores.cancelled) == (0, 0)
    assert stats.dead_entries_not_written_back == 0
    # a's tail, Host-only, which the Host plan took (lru), and a's second
    # page, which the collection took (dead_entry).
    host = stats.host_evictions
    assert (
        host.lru.blocks,
        host.lru.device_resident,
        host.duplicate.blocks,
        host.superseded.blocks,
        host.dead_entry.blocks,
        host.dead_entry.device_resident,
        host.tail_cap.blocks,
    ) == (1, 0, 0, 0, 1, 0, 0)
    # Every block an admission evicted for the cache was stored or dropped:
    # a's tail and second page, then a's first page and b's tail.
    device = stats.device_evictions
    assert (
        device.admission.blocks
        == sum(eviction_stores.blocks) + eviction_stores.no_placement
    )
    assert (device.admission.blocks, device.stranded_suffix.blocks) == (4, 0)
    # b's tail is the Host's one entry.
    assert scheduler.host_pool_cached_blocks() == 1
