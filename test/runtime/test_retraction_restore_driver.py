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

"""The runtime's side of a retraction, driven against the real C++ scheduler.

No engine: the scheduler is built through ``make_config`` exactly as the event
loop builds it, its plans are adapted by ``cache_ops_from_plan`` exactly as
``DeviceHandle.execute`` adapts them, and the ACKs the Host cache executor
would produce are fed back through ``CacheOpHooks`` and ``advance_scheduler``
exactly as the loop feeds them. The forced-retraction knob
(``debug_force_retraction_interval``) stands in for capacity pressure, so a
decoding request is retracted, imaged, restored and resumes ``Decoding`` with
the token count it left with -- the control-plane half of the suspend/resume
contract (``docs/design/scheduler.md`` section 4, ``docs/design/event-loop.md``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.cache.transfer.ops import (  # noqa: E402
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.engine.cache_hooks import CacheOpHooks  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    advance_scheduler,
    cache_ops_from_plan,
    make_abort_event,
    make_config,
    make_extend_result_event,
)

PAGE = 16
PROMPT = list(range(40))  # 2 hash-complete pages and a 8-token tail
POOL_LCM_BLOCKS = 8


def _scheduler(*, l2: bool, force_interval: int) -> ts.Scheduler:
    groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=PAGE,
            total_pages=64 + 1,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        )
    ]
    config = make_config(
        num_device_pages=64 + 1,
        max_scheduled_tokens=256,
        max_batch_size=4,
        prefix_granularity=PAGE,
        num_host_pages=32 + 1 if l2 else 0,
        disable_l2_cache=not l2,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=POOL_LCM_BLOCKS + 1,
        max_retracted_requests=2,
        debug_force_retraction_interval=force_interval,
        cache_groups=groups,
    )
    return ts.Scheduler(config)


class _Acks:
    """The DeviceHandle surface the hooks poll: ACKs queued by the test."""

    def __init__(self) -> None:
        self.results: list = []

    def poll_cache_results(self) -> list:
        results, self.results = self.results, []
        return results

    def consume_l3_backup_poll_failure(self) -> bool:
        return False


def _hooks(device) -> CacheOpHooks:
    return CacheOpHooks(
        device,
        speculative_algorithm=None,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        global_rank=0,
    )


def _ack(kind: str, op_id: int):
    event = getattr(ts.Cache, kind)()
    event.op_id = op_id
    return event


class _Driver:
    """One round: plan, count, 'execute' (record the adapted ops), ACK later."""

    def __init__(self, scheduler: ts.Scheduler) -> None:
        self.scheduler = scheduler
        self.device = _Acks()
        self.hooks = _hooks(self.device)
        self.last_token = PROMPT[-1]
        self.next_token = 1000

    def round(self) -> SimpleNamespace:
        # Head of the round: the completed cache ops reach the scheduler
        # before it plans, as in the event loop.
        advance_scheduler(self.scheduler, self.hooks.poll_ready_events())
        plan = self.scheduler.next_execution_plan()
        self.hooks.count_plan_ops(plan)
        ops = cache_ops_from_plan(plan)
        forwards = [op for op in plan.forward if op.request_ids]
        return SimpleNamespace(plan=plan, ops=ops, forwards=forwards)

    def complete(self, ops) -> None:
        """The executor's ACKs for every adapted op, one per ticket."""
        for op in ops:
            if isinstance(op, ts.Cache.WriteBackOp):
                self.device.results.extend(
                    _ack("WriteBackDoneEvent", op_id) for op_id in op.op_ids
                )
            elif isinstance(op, ts.Cache.LoadBackOp):
                self.device.results.extend(
                    ts.Cache.LoadBackDoneEvent(op_id, True) for op_id in op.op_ids
                )
            elif isinstance(op, SnapshotOp):
                self.device.results.append(_ack("SnapshotDoneEvent", op.op_id))
            elif isinstance(op, RestoreOp):
                self.device.results.append(_ack("RestoreDoneEvent", op.op_id))
            else:
                raise AssertionError(f"unexpected op {op!r}")

    def land(self, forward_op) -> None:
        """One forward's result: a prefill's or decode's one sampled token."""
        (rid,) = forward_op.request_ids
        advance_scheduler(
            self.scheduler, [make_extend_result_event(rid, [self.next_token])]
        )
        self.last_token = self.next_token
        self.next_token += 1

    def submit(self, rid: str) -> None:
        spec = ts.RequestSpec()
        spec.request_id = rid
        spec.tokens = list(PROMPT)
        spec.max_new_tokens = 100
        self.scheduler.submit_requests([spec])


def _prefill_and_decode_once(driver: _Driver) -> int:
    """Admit, prefill and run one decode; returns the request's slot."""
    driver.submit("a")
    rnd = driver.round()
    assert [op.num_extends() for op in rnd.forwards] == [1]
    driver.complete(rnd.ops)
    driver.land(rnd.forwards[0])
    rnd = driver.round()
    (decode,) = rnd.forwards
    assert decode.num_extends() == 0 and list(decode.input_lengths) == [1]
    driver.complete(rnd.ops)
    driver.land(decode)
    return int(decode.request_pool_indices[0])


@pytest.mark.parametrize("l2", [False, True])
def test_forced_retraction_images_restores_and_resumes_decoding(l2: bool) -> None:
    """Retract -> SnapshotOp (+ the L2 leg's pins) -> ACK -> RestoreOp -> ACK ->
    the request decodes again, in a new slot, with the token count it left."""
    scheduler = _scheduler(l2=l2, force_interval=3)
    driver = _Driver(scheduler)
    pool_free = scheduler.snapshot_pool_free_blocks()
    slot = _prefill_and_decode_once(driver)
    tokens_before = scheduler.request_token_size("a")

    # The third plan arms the oldest quiescent Decoding request and retracts
    # it: no forward, one snapshot store of its tail (the whole image without
    # L2), the request suspended.
    rnd = driver.round()
    assert rnd.forwards == []
    (store,) = rnd.ops
    assert isinstance(store, SnapshotOp)
    assert store.request_id == "a" and store.request_pool_index == slot
    assert store.transfers, "the unaligned tail page always goes to the pool"
    pages_in_pool = {t.destination_page for t in store.transfers}
    assert len(pages_in_pool) == len(store.transfers)
    if l2:
        # The two hash-complete pages were published to L2 at the end of the
        # prefill and are pinned there by the suspended request; only the
        # tail is in the pool.
        assert len(store.transfers) == 1
        assert scheduler.host_pool_pinned_blocks() == 2
    else:
        assert len(store.transfers) == 3
        assert scheduler.host_pool_pinned_blocks() == 0
    assert scheduler.retracted_size() == 1 and scheduler.decoding_size() == 0
    assert scheduler.waiting_size() == 1
    assert scheduler.snapshot_pool_free_blocks() < pool_free

    # Until the store's ACK lands the image is not landed: no restore.
    rnd = driver.round()
    assert rnd.ops == [] and rnd.forwards == []
    driver.complete([store])

    # The ACK makes the image eligible; the next plan restores: one op whose
    # rows copy the pool pages back into fresh Device pages, naming the new
    # slot and the same blob slot; the request stays unschedulable until the
    # restore's ACK.
    rnd = driver.round()
    assert rnd.forwards == []
    (restore,) = rnd.ops
    assert isinstance(restore, RestoreOp)
    assert restore.request_id == "a" and restore.snapshot_slot == store.snapshot_slot
    assert restore.request_pool_index != slot
    assert {t.source_page for t in restore.transfers} == pages_in_pool
    assert set(restore.source_tier) == {HostTier.SNAPSHOT_POOL}
    assert all(t.content_hash == "" for t in restore.transfers)
    assert scheduler.retracted_size() == 1 and scheduler.decoding_size() == 0
    rnd = driver.round()
    assert rnd.ops == [] and rnd.forwards == []
    driver.complete([restore])

    # RestoreDone: Decoding again, same token count, nothing left in the pool
    # or pinned in L2, and the next decode runs in the restored slot. Its
    # input is the request's last token either way the scheduler states it:
    # explicitly (no forward of the request is in flight to capture it from)
    # or as -1, deferring to the restored row, whose column 0 is that token;
    # the runtime takes both (RuntimeStates.import_slot_state).
    rnd = driver.round()
    assert scheduler.retracted_size() == 0 and scheduler.decoding_size() == 1
    assert scheduler.request_token_size("a") == tokens_before
    assert scheduler.snapshot_pool_free_blocks() == pool_free
    assert scheduler.host_pool_pinned_blocks() == 0
    (decode,) = rnd.forwards
    assert list(decode.request_ids) == ["a"] and decode.num_extends() == 0
    assert list(decode.request_pool_indices) == [restore.request_pool_index]
    assert list(decode.decode_input_ids) in ([driver.last_token], [-1])
    driver.complete(rnd.ops)
    driver.land(decode)
    assert scheduler.request_token_size("a") == tokens_before + 1
    assert driver.hooks._num_inflight == 0


def test_an_aborted_victims_store_is_still_acknowledged_and_frees_its_image() -> None:
    """The blob slot and pool blocks stay pinned until the store's ACK even
    when the client aborts the suspended request; the runtime ACKs every op
    it received, and the ACK releases the image."""
    scheduler = _scheduler(l2=False, force_interval=3)
    driver = _Driver(scheduler)
    pool_free = scheduler.snapshot_pool_free_blocks()
    _prefill_and_decode_once(driver)
    rnd = driver.round()
    (store,) = rnd.ops
    assert isinstance(store, SnapshotOp)

    advance_scheduler(scheduler, [make_abort_event("a")])
    assert scheduler.retracted_size() == 0 and scheduler.waiting_size() == 0
    driver.complete([store])
    rnd = driver.round()
    assert rnd.ops == [] and rnd.forwards == []
    assert scheduler.snapshot_pool_free_blocks() == pool_free
    assert driver.hooks._num_inflight == 0


def test_the_knob_is_off_by_default_and_a_null_pool_never_retracts() -> None:
    scheduler = _scheduler(l2=False, force_interval=0)
    driver = _Driver(scheduler)
    _prefill_and_decode_once(driver)
    for _ in range(4):
        rnd = driver.round()
        assert rnd.ops == [] and len(rnd.forwards) == 1
        driver.land(rnd.forwards[0])
    assert scheduler.retracted_size() == 0
