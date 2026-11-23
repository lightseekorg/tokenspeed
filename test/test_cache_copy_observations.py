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

"""CPU device-boundary regressions for completed cache-copy observations."""

import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace

import pytest

from tokenspeed.runtime.cache.transfer.observations import (
    CompletedObservations,
    CopyCounts,
    count_owned_rows,
    observe_copy,
)
from tokenspeed.runtime.cache.transfer.ops import (
    CacheTransfer,
    HostTier,
    PrefixLoadOp,
    PrefixLoadTelemetry,
    RestoreOp,
    RestoreTelemetry,
)


def telemetry(admission=17):
    return RestoreTelemetry(admission, 3, 1, 4, 9, 16, 2, 120, 119, 100)


def observation(request_id="request-a", admission=17):
    return observe_copy(
        kind="snapshot_restore",
        op_id=8,
        request_id=request_id,
        request_pool_index=2,
        telemetry=telemetry(admission),
        l2=CopyCounts(2, 24),
        pool=CopyCounts(1, 16),
        slot_state_image_bytes=16,
    )


def test_completed_queue_is_bounded_and_reports_loss_without_truncating_identity():
    queue = CompletedObservations(1)
    queue.extend(observation())
    queue.extend(observation("request-b", 18))
    queue.extend(observation("x" * 257))
    records, dropped = queue.drain()
    assert dropped == 2 and len(records) == 1
    assert records[0]["request_id"] == "request-a"
    assert records[0]["admission_id"] == 17 and records[0]["op_generation"] == 3
    assert queue.drain() == ([], 0)
    queue.extend(observation("界" * 100))
    assert queue.drain() == ([], 1)
    disabled = CompletedObservations(0)
    disabled.extend(observation())
    assert disabled.drain() == ([], 0)
    with pytest.raises(ValueError):
        CompletedObservations(-1)


class Stream:
    """A deferred device queue: submission does not execute pending copies."""

    def __init__(self):
        self.work = deque()
        self.submitted = 0
        self.completed = 0

    def enqueue(self, action, ready=lambda: True):
        self.work.append((ready, action))
        self.submitted += 1

    def wait_stream(self, other):
        boundary = other.submitted
        self.enqueue(lambda: None, lambda: other.completed >= boundary)

    def wait_event(self, event):
        self.enqueue(lambda: None, event.query)

    def advance(self):
        while self.work and self.work[0][0]():
            _, action = self.work.popleft()
            action()
            self.completed += 1


class Event:
    def __init__(self):
        self.stream = None
        self.boundary = 0

    def record(self, stream):
        self.stream, self.boundary = stream, stream.submitted

    def query(self):
        return self.stream is None or self.stream.completed >= self.boundary

    def wait(self, stream):
        stream.wait_event(self)

    def synchronize(self):
        raise AssertionError("completion polling must not synchronize")


class Device:
    Event = Event

    def __init__(self):
        self.current = Stream()
        self.streams = [self.current]

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def synchronize(self):
        # Only the existing shutdown path may call this device boundary.
        for stream in self.streams:
            stream.advance()


@dataclass
class Buffer:
    groups: list
    device: str = "cpu"


@dataclass
class Storage:
    host_buffer: Buffer


@dataclass(frozen=True)
class Geometry:
    fields: tuple = ((0, 0, 4), (0, 4, 8), (1, 0, 16))
    layer_slices: tuple = ((0, 2), (2, 1))
    num_field_rows: int = 3


@dataclass(frozen=True)
class Backend:
    uses_device_tables: bool = False
    layer_ready: bool = False


class Workspace:
    def __init__(self):
        self.rows = ()

    def load_block_transfers(self, rows, *, geometry):
        self.rows = tuple(rows)
        return len(rows), 0

    def prepare_backend(self, buffers, host, *, backend):
        return Backend()


class Lane:
    def __init__(self, failure=False):
        self.failure = failure

    def start_h2d(
        self,
        rows,
        *,
        device_buffers,
        host_buffer,
        geometry,
        stream,
        prerequisite_stream,
        backend
    ):
        if self.failure:
            raise RuntimeError("device copy failed")
        for group, device, host in rows:
            stream.enqueue(
                lambda g=group, d=device, h=host: device_buffers[0]
                .groups[g][d]
                .__setitem__(slice(None), host_buffer.groups[g][h])
            )


class SlotState:
    def __init__(self, failure=False):
        self.values = {}
        self.failure = failure

    def import_(self, slot, blob, stream, *, admission_id):
        if self.failure:
            raise RuntimeError("slot import failed")
        stream.enqueue(
            lambda: self.values.__setitem__((admission_id, slot), bytes(blob))
        )


class MemoryRegions:
    def named_region(self, name):
        return nullcontext()


@pytest.fixture
def executor_factory(monkeypatch):
    # Discovery, streams, copy kernels, and slot-state device owners are the only
    # unavailable boundaries. Ownership translation, collection, ACK polling,
    # completion queue, and telemetry accounting execute their real code.
    from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo

    monkeypatch.setattr(
        Platform,
        "_instance",
        PlatformInfo(
            vendor="nvidia",
            arch_version=ArchVersion(9, 0),
            device_name="CPU fixture",
            device_count=0,
            total_memory=0,
            memory_bandwidth=0,
            sm_count=132,
            max_threads_per_sm=2048,
            max_shared_memory_per_sm=232448,
        ),
    )
    from tokenspeed.runtime.cache.l2 import executor as module
    from tokenspeed.runtime.cache.l2 import layerwise_load
    from tokenspeed.runtime.cache.transfer.lanes import CompletionQueue
    from tokenspeed.runtime.cache.transfer.layout import (
        CacheField,
        CacheGroupLayout,
        CacheTransferLayout,
    )
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation

    device = Device()
    monkeypatch.setattr(module, "device_module", device)
    monkeypatch.setattr(layerwise_load, "device_module", device)

    def copy_fields(
        direction,
        buffers,
        host,
        geometry,
        workspace,
        stream,
        *,
        num_blocks,
        geometry_offset,
        num_geometry_rows,
        backend,
        grid_cap,
        layer_ready_flags
    ):
        assert direction == "h2d"
        for group, target, source in workspace.rows:
            for owner, offset, count in geometry.fields[
                geometry_offset : geometry_offset + num_geometry_rows
            ]:
                if owner == group:
                    stream.enqueue(
                        lambda g=group, d=target, h=source, a=offset, n=count: buffers[
                            0
                        ]
                        .groups[g][d]
                        .__setitem__(slice(a, a + n), host.groups[g][h][a : a + n])
                    )

    monkeypatch.setattr(module, "transfer_cache_blocks", copy_fields)

    class CPUExecutor(module.HostCacheExecutor):
        def __init__(self, *, capacity, rank, copy_failure=False, import_failure=False):
            self._cache_observations = CompletedObservations(capacity)
            self._completions = CompletionQueue()
            self._ack_lock = threading.Lock()
            self._ready_load_acks = []
            self._load_poisoned = False
            self.load_stream = Stream()
            device.streams.append(self.load_stream)
            self.attn_tp_rank = rank
            groups = (
                CacheGroupLayout(
                    "sharded",
                    1,
                    (CacheField("a", 0, 0, 12, 4), CacheField("b", 0, 4, 12, 8)),
                ),
                CacheGroupLayout("replicated", 1, (CacheField("c", 0, 0, 16, 16),)),
            )
            buffers = Buffer([[bytearray(n) for _ in range(12)] for n in (12, 16)])
            self.layout = CacheTransferLayout(
                num_lcm_blocks=8,
                buffers=(buffers,),
                groups=groups,
                consumers=(("a", "b"), ("c",)),
            )

            def host(value):
                return Storage(
                    Buffer(
                        [
                            [bytearray([value + i] * n) for i in range(12)]
                            for n in (12, 16)
                        ]
                    )
                )

            self.host_storage, self.snapshot_storage = host(20), host(40)
            self.block_owners = BlockOwnerTranslation(
                shard_counts=(2, 1),
                device_virtual_counts=(20, 12),
                host_virtual_counts=(20, 12),
                rank=rank,
            )
            self.snapshot_block_owners = BlockOwnerTranslation(
                shard_counts=(2, 1),
                device_virtual_counts=(20, 12),
                host_virtual_counts=(20, 12),
                rank=rank,
            )
            self._restore_l2_lane, self._restore_pool_lane = Lane(copy_failure), Lane()
            self._transfer_geometry = self._snapshot_geometry = Geometry()
            self.blob_bytes = 16
            self.blob_arena = [bytearray([index] * 16) for index in range(5)]
            self.max_retracted_requests = 4
            self._slot_state = SlotState(import_failure)
            self._memory_saver_adapter = MemoryRegions()
            tracker = layerwise_load.LayerwiseLoadTracker(2)
            self._load_trackers = [(tracker, 2)]
            self._load_workspaces = tuple(Workspace() for _ in tracker.event_sets)
            self.transfer_backend = "direct"

    return CPUExecutor, device


def restore(op_id=9, admission=17, *, empty=False):
    if empty:
        transfers = ()
    else:
        transfers = (
            CacheTransfer(0, 1, 5, "key", 0),
            CacheTransfer(0, 2, 6, "key", 0),
            CacheTransfer(1, 2, 3, "", 0),
        )
    tiers = () if empty else (HostTier.L2, HostTier.L2, HostTier.SNAPSHOT_POOL)
    return RestoreOp(
        op_id, "request-a", admission, 2, 1, transfers, tiers, telemetry(admission)
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_restore_observation_follows_owned_copies_and_slot_import(
    executor_factory, rank
):
    factory, device = executor_factory
    executor = factory(capacity=8, rank=rank)
    executor.submit_load_backs([restore()], prerequisite_stream=device.current_stream())
    assert executor.poll_results() == []
    assert executor.poll_cache_observations() == ([], 0)
    assert executor._slot_state.values == {}
    executor.load_stream.advance()
    assert executor._slot_state.values[17, 2] == bytes([1] * 16)
    assert (
        executor.layout.buffers[0].groups[0][3]
        == executor.host_storage.host_buffer.groups[0][1]
    )
    assert (
        executor.layout.buffers[0].groups[1][3]
        == executor.snapshot_storage.host_buffer.groups[1][2]
    )
    acknowledgments = executor.poll_results()
    assert [ack.op_id for ack in acknowledgments] == [9]
    records, dropped = executor.poll_cache_observations()
    assert dropped == 0 and len(records) == 1
    assert {
        k: records[0][k] for k in ("l2_rows", "l2_bytes", "pool_rows", "pool_bytes")
    } == dict(l2_rows=1, l2_bytes=12, pool_rows=1, pool_bytes=16)
    assert (
        records[0]["slot_state_image_bytes"] == 16 and records[0]["admission_id"] == 17
    )
    assert executor.poll_results() == [] and executor.poll_cache_observations() == (
        [],
        0,
    )


@pytest.mark.parametrize("empty_local", [False, True])
def test_prefix_local_completion_counts_include_empty_owner(
    executor_factory, empty_local
):
    factory, device = executor_factory
    executor = factory(capacity=4, rank=1 if empty_local else 0)
    op = PrefixLoadOp(
        4,
        "prefix-a",
        PrefixLoadTelemetry(31, 2, 1, 5),
        (CacheTransfer(0, 1, 5, "prefix", 0),),
    )
    executor.submit_load_backs([op], prerequisite_stream=device.current_stream())
    if not empty_local:
        assert executor.poll_results() == []
        assert executor.poll_cache_observations() == ([], 0)
        executor.load_stream.advance()
        assert (
            executor.layout.buffers[0].groups[0][3]
            == executor.host_storage.host_buffer.groups[0][1]
        )
    assert [ack.op_id for ack in executor.poll_results()] == [4]
    events, dropped = executor.poll_cache_observations()
    assert dropped == 0 and len(events) == 1
    assert events[0]["kind"] == "prefix_load" and events[0]["op_generation"] == 2
    assert events[0]["l2_rows"] == (0 if empty_local else 1)
    assert events[0]["l2_bytes"] == (0 if empty_local else 12)
    assert events[0]["pool_rows"] == events[0]["pool_bytes"] == 0


@pytest.mark.parametrize("failure", ["copy", "import"])
def test_failed_copy_or_import_has_no_success_observation(executor_factory, failure):
    factory, device = executor_factory
    executor = factory(
        capacity=4,
        rank=0,
        copy_failure=failure == "copy",
        import_failure=failure == "import",
    )
    with pytest.raises(RuntimeError, match="failed"):
        executor.submit_load_backs(
            [restore()], prerequisite_stream=device.current_stream()
        )
    executor.load_stream.advance()
    assert executor.poll_results() == [] and executor.poll_cache_observations() == (
        [],
        0,
    )


def test_shutdown_discards_pending_and_completed_unconsumed_records(executor_factory):
    factory, device = executor_factory
    executor = factory(capacity=4, rank=0)
    executor.submit_load_backs([restore()], prerequisite_stream=device.current_stream())
    executor.shutdown()
    assert executor.poll_results() == [] and executor.poll_cache_observations() == (
        [],
        0,
    )
    executor = factory(capacity=4, rank=0)
    executor.submit_load_backs([restore()], prerequisite_stream=device.current_stream())
    executor.load_stream.advance()
    assert len(executor.poll_results()) == 1
    executor.shutdown()
    assert executor.poll_cache_observations() == ([], 0)


def test_observation_overflow_or_absence_never_loses_real_ack(executor_factory):
    factory, device = executor_factory
    executor = factory(capacity=1, rank=0)
    for op in (restore(9), restore(10, 18), replace(restore(11, 19), telemetry=None)):
        executor.submit_load_backs([op], prerequisite_stream=device.current_stream())
        executor.load_stream.advance()
        assert [ack.op_id for ack in executor.poll_results()] == [op.op_id]
    records, dropped = executor.poll_cache_observations()
    assert len(records) == 1 and records[0]["admission_id"] == 17 and dropped == 2


def test_empty_restore_still_waits_for_state_import(executor_factory):
    factory, device = executor_factory
    executor = factory(capacity=4, rank=0)
    executor.submit_load_backs(
        [restore(empty=True)], prerequisite_stream=device.current_stream()
    )
    assert executor.poll_results() == []
    executor.load_stream.advance()
    assert [ack.op_id for ack in executor.poll_results()] == [9]
    events, dropped = executor.poll_cache_observations()
    assert dropped == 0 and events[0]["l2_bytes"] == events[0]["pool_bytes"] == 0
    assert executor._slot_state.values[17, 2] == bytes([1] * 16)


def test_payload_byte_accounting_exceeds_int32_without_allocating_device_storage():
    from tokenspeed.runtime.cache.transfer.layout import CacheField, CacheGroupLayout

    groups = (
        CacheGroupLayout("large", 1, (CacheField("field", 0, 0, 2**32, 2**31 + 16),)),
    )
    assert count_owned_rows(((0, 1, 1), (0, 2, 2)), groups) == CopyCounts(2, 2**32 + 32)
    assert count_owned_rows((), groups) == CopyCounts(0, 0)


@dataclass
class RestoreWire:
    op_ids: list
    request_ids: list
    request_pool_indices: list
    snapshot_slots: list
    group_ids: list
    src_pages: list
    dst_pages: list
    content_hashes: list
    page_offsets: list
    source_tiers: list
    telemetry: list


@dataclass
class PrefixWire:
    op_ids: list
    request_ids: list
    telemetry: list
    group_ids: list
    src_pages: list
    dst_pages: list
    content_hashes: list
    page_offsets: list


def test_adapter_freezes_native_identity_instead_of_recycled_request_lookup(
    executor_factory,
):
    from tokenspeed.runtime.engine.scheduler_utils import (
        prefix_load_ops_from_wire,
        restore_ops_from_wire,
    )

    wire = RestoreWire(
        [9],
        ["reused"],
        [2],
        [1],
        [[0]],
        [[1]],
        [[5]],
        [["key"]],
        [[0]],
        [[0]],
        [telemetry()],
    )
    current_admissions = {"reused": 99}
    operation = restore_ops_from_wire(
        wire, request_admission=current_admissions.__getitem__, record_observations=True
    )[0]
    assert operation.admission_id == 17 and operation.telemetry.op_generation == 3
    wire.request_ids[0] = "later"
    wire.telemetry[0] = telemetry(99)
    wire.src_pages[0][0] = 8
    assert operation.request_id == "reused" and operation.transfers[0].source_page == 1
    assert operation.telemetry.admission_id == 17
    with pytest.raises(ValueError, match="ragged"):
        restore_ops_from_wire(
            replace(wire, telemetry=[]),
            request_admission=current_admissions.__getitem__,
            record_observations=True,
        )
    with pytest.raises(ValueError, match="native admission"):
        restore_ops_from_wire(
            replace(wire, telemetry=[telemetry(0)]),
            request_admission=current_admissions.__getitem__,
            record_observations=True,
        )
    prefix = PrefixWire(
        [10],
        ["prefix"],
        [PrefixLoadTelemetry(23, 7, 3, 4)],
        [[0]],
        [[1]],
        [[5]],
        [["key"]],
        [[0]],
    )
    captured = prefix_load_ops_from_wire(prefix)[0]
    prefix.telemetry[0] = PrefixLoadTelemetry(42, 8, 4, 5)
    assert (
        captured.telemetry.admission_id == 23 and captured.telemetry.op_generation == 7
    )
    with pytest.raises(ValueError, match="ragged"):
        prefix_load_ops_from_wire(replace(prefix, telemetry=[]))


def test_multiple_restore_ops_keep_individual_tier_counts(executor_factory):
    factory, device = executor_factory
    executor = factory(capacity=4, rank=0)
    first = restore()
    second = replace(
        restore(10, 18),
        snapshot_slot=2,
        request_pool_index=3,
        request_id="request-b",
        transfers=(CacheTransfer(1, 3, 4, "", 0),),
        source_tier=(HostTier.SNAPSHOT_POOL,),
    )
    executor.submit_load_backs(
        [first, second], prerequisite_stream=device.current_stream()
    )
    executor.load_stream.advance()
    assert [ack.op_id for ack in executor.poll_results()] == [9, 10]
    records, dropped = executor.poll_cache_observations()
    assert dropped == 0 and len(records) == 2
    assert [(r["admission_id"], r["l2_rows"], r["pool_rows"]) for r in records] == [
        (17, 1, 1),
        (18, 0, 1),
    ]
    assert [(r["l2_bytes"], r["pool_bytes"]) for r in records] == [(12, 16), (0, 16)]


def test_prefix_transfer_failure_never_publishes_partial_copy_as_success(
    executor_factory, monkeypatch
):
    from tokenspeed.runtime.cache.l2 import executor as module

    factory, device = executor_factory
    executor = factory(capacity=4, rank=0)
    original = module.transfer_cache_blocks

    def failing_copy(*args, **kwargs):
        if kwargs["geometry_offset"] > 0:
            raise RuntimeError("later copy layer failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "transfer_cache_blocks", failing_copy)
    operation = PrefixLoadOp(
        5,
        "prefix-a",
        PrefixLoadTelemetry(31, 2, 1, 5),
        (
            CacheTransfer(0, 1, 5, "prefix", 0),
            CacheTransfer(1, 2, 3, "prefix", 0),
        ),
    )
    with pytest.raises(RuntimeError, match="later copy layer failed"):
        executor.submit_load_backs(
            [operation], prerequisite_stream=device.current_stream()
        )
    executor.load_stream.advance()
    assert (
        executor.layout.buffers[0].groups[0][3]
        == executor.host_storage.host_buffer.groups[0][1]
    )
    assert executor.poll_results() == [] and executor.poll_cache_observations() == (
        [],
        0,
    )


def test_disabled_observations_preserve_restore_ack_without_telemetry(executor_factory):
    factory, device = executor_factory
    executor = factory(capacity=0, rank=0)
    executor.submit_load_backs(
        [replace(restore(), telemetry=None)],
        prerequisite_stream=device.current_stream(),
    )
    executor.load_stream.advance()
    assert [ack.op_id for ack in executor.poll_results()] == [9]
    assert executor.poll_cache_observations() == ([], 0)


def test_startup_opt_in_is_explicit_and_capacity_is_bounded():
    from tokenspeed.runtime.cache.transfer.observations import (
        cache_observation_capacity_from_env,
    )

    assert cache_observation_capacity_from_env({}) == 0
    assert cache_observation_capacity_from_env({"TOKENSPEED_CACHE_EVENT_LOG": "0"}) == 0
    assert (
        cache_observation_capacity_from_env({"TOKENSPEED_CACHE_EVENT_LOG": "1"}) == 4096
    )
    with pytest.raises(ValueError, match="must be 0 or 1"):
        cache_observation_capacity_from_env({"TOKENSPEED_CACHE_EVENT_LOG": "maybe"})


def test_host_data_plane_drains_completed_cpu_observations(executor_factory):
    from tokenspeed.runtime.engine.profiler import Profiler
    from tokenspeed.runtime.engine.rust_host import HostDataPlane
    from tokenspeed.runtime.execution.device import DeviceHandle
    from tokenspeed.runtime.utils.server_args import ServerArgs

    factory, device = executor_factory
    executor = factory(capacity=4, rank=0)

    class ModelBoundary:
        def __init__(self):
            self.device = "cpu"

    handle = DeviceHandle(ModelBoundary(), host_cache_executor=executor)
    args = ServerArgs(model="fixture")
    args.mapping.rank = 0
    plane = HostDataPlane(
        handle,
        Profiler(
            server_args=args,
            device=handle,
            attn_mapping=args.mapping.attn,
            sync_size=1,
            sync_group=None,
            host_spans=None,
        ),
        {},
        report_compiles=False,
        max_encoder_tokens=None,
        attn_tp_cpu_group=None,
        epd_publisher=None,
        starting=False,
        record_cache_observations=True,
    )
    executor.submit_load_backs([restore()], prerequisite_stream=device.current_stream())
    assert plane.poll_cache_events() == []
    assert plane.poll_cache_observations() == ([], 0)
    executor.load_stream.advance()
    # The observation poll does not itself query/publish a completed event.
    assert plane.poll_cache_observations() == ([], 0)
    assert [op_id for _, op_id in plane.poll_cache_events()] == [9]
    events, dropped = plane.poll_cache_observations()
    assert dropped == 0 and len(events) == 1 and events[0]["admission_id"] == 17
    assert plane.poll_cache_observations() == ([], 0)
    empty = DeviceHandle(ModelBoundary())
    assert empty.poll_cache_observations() == ([], 0)
