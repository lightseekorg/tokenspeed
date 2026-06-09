from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from tokenspeed.runtime.cache.transfer.types import (
    PAGED_CACHE_KIND,
    CacheKind,
    Location,
    TransferUnit,
)


class FakeEvent:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True

    def wait(self, stream):
        return None

    def query(self):
        return True

    def synchronize(self):
        self.recorded = True


class FakeStream:
    def synchronize(self):
        return None


class FakeDeviceModule:
    Event = FakeEvent
    Stream = FakeStream

    @staticmethod
    def stream(stream):
        return nullcontext()

    @staticmethod
    def current_stream():
        return FakeStream()


class FakeLayerEvent:
    def __init__(self, num_layers: int):
        self.start_event = FakeEvent()
        self.load_events = [FakeEvent() for _ in range(num_layers)]

    def complete(self, layer_idx: int):
        self.load_events[layer_idx].record()

    @property
    def finish_event(self):
        return self.load_events[-1]


class FakeCounter:
    def __init__(self, num_layers: int):
        self.events = [FakeLayerEvent(num_layers) for _ in range(3)]
        self.producer = -1
        self.consumer = None

    def update_producer(self):
        self.producer = (self.producer + 1) % len(self.events)
        return self.producer

    def set_consumer(self, producer_index):
        self.consumer = producer_index

    def reset(self):
        self.producer = -1
        self.consumer = None


class FakePool:
    def __init__(self, kind: CacheKind, page_size: int, num_layers: int):
        self.kind = kind
        self._page_size = page_size
        self._num_layers = num_layers
        self.device = torch.device("cpu")
        self.host_layout = "layer_first"
        self.writebacks: list[tuple[list[int], list[int]]] = []
        self.loadbacks: list[tuple[int, list[int], list[int]]] = []
        self.counter = FakeCounter(num_layers)

    def page_size(self):
        return self._page_size

    def num_layers(self):
        return self._num_layers

    def supports_layerwise_loadback(self):
        return True

    def writeback(self, src_indices, dst_indices):
        self.writebacks.append((src_indices.tolist(), dst_indices.tolist()))

    def loadback(self, src_indices, dst_indices, layer_idx: int):
        self.loadbacks.append((layer_idx, src_indices.tolist(), dst_indices.tolist()))

    def get_layer_done_counter(self):
        return self.counter

    def reset(self):
        self.writebacks.clear()
        self.loadbacks.clear()
        self.counter.reset()


class FakePagedPool:
    kind = PAGED_CACHE_KIND

    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.device = torch.device("cpu")
        self.counter = FakeCounter(num_layers)
        self.writebacks = []
        self.loadbacks = []

    def num_layers(self):
        return self._num_layers

    def get_layer_done_counter(self):
        return self.counter

    def writeback_paged(self, transfers):
        self.writebacks.append(list(transfers))

    def loadback_paged(self, transfers, layer_idx: int):
        self.loadbacks.append((layer_idx, list(transfers)))


def _patch_host_executor_device(monkeypatch):
    import tokenspeed.runtime.cache.executor.host_executor as host_executor

    monkeypatch.setattr(host_executor, "device_module", FakeDeviceModule)
    return host_executor.HostExecutor


def test_transfer_unit_exposes_direction():
    unit = TransferUnit(
        kind=CacheKind.MAMBA,
        src_loc=Location.DEVICE,
        dst_loc=Location.HOST,
        src_indices=torch.tensor([1, 2], dtype=torch.int64),
        dst_indices=torch.tensor([3, 4], dtype=torch.int64),
        op_id=99,
    )

    assert unit.direction == (Location.DEVICE, Location.HOST)


def test_host_executor_keeps_page_indices_on_cpu_until_flush(monkeypatch):
    host_executor = __import__(
        "tokenspeed.runtime.cache.executor.host_executor",
        fromlist=["HostExecutor"],
    )
    monkeypatch.setattr(host_executor, "device_module", FakeDeviceModule)

    seen_devices = []
    real_converter = host_executor.page_ids_to_token_indices

    def spy_page_ids_to_token_indices(page_ids, page_size, device="cpu"):
        seen_devices.append(device)
        return real_converter(page_ids, page_size, device)

    monkeypatch.setattr(
        host_executor, "page_ids_to_token_indices", spy_page_ids_to_token_indices
    )
    executor = host_executor.HostExecutor(
        pools=[FakePool(CacheKind.KV, page_size=4, num_layers=2)], io_backend="kernel"
    )

    executor.enqueue_writeback(1, src_pages=[2], dst_pages=[5], kind=CacheKind.KV)
    executor.enqueue_loadback(2, src_pages=[7], dst_pages=[11], kind=CacheKind.KV)

    assert seen_devices == ["cpu", "cpu", "cpu", "cpu"]
    assert executor.write_queues[CacheKind.KV][0].src_indices.device.type == "cpu"
    assert executor.write_queues[CacheKind.KV][0].dst_indices.device.type == "cpu"
    assert executor.load_queues[CacheKind.KV][0].src_indices.device.type == "cpu"
    assert executor.load_queues[CacheKind.KV][0].dst_indices.device.type == "cpu"


def test_host_executor_batches_writeback_by_cache_kind_and_acks_once(monkeypatch):
    HostExecutor = _patch_host_executor_device(monkeypatch)
    kv_pool = FakePool(CacheKind.KV, page_size=4, num_layers=2)
    mamba_pool = FakePool(CacheKind.MAMBA, page_size=1, num_layers=3)
    executor = HostExecutor(pools=[kv_pool, mamba_pool], io_backend="kernel")

    executor.enqueue_writeback(
        7, src_pages=[2], dst_pages=[5], kind=CacheKind.KV, is_retract=True
    )
    executor.enqueue_writeback(
        7, src_pages=[11], dst_pages=[13], kind=CacheKind.MAMBA, is_retract=True
    )

    executor.flush()

    assert kv_pool.writebacks == [([8, 9, 10, 11], [20, 21, 22, 23])]
    assert mamba_pool.writebacks == [([11], [13])]

    results = executor.drain()
    assert [event.op_id for event in results] == [7]
    assert all(event.success for event in results)


def test_host_executor_rejects_loadback_during_cuda_graph_capture(monkeypatch):
    import tokenspeed.runtime.cache.executor.host_executor as host_executor

    monkeypatch.setattr(host_executor, "device_module", FakeDeviceModule)
    monkeypatch.setattr(host_executor, "get_is_capture_mode", lambda: True)
    executor = host_executor.HostExecutor(
        pools=[FakePool(CacheKind.MAMBA, page_size=1, num_layers=1)],
        io_backend="kernel",
    )

    executor.enqueue_loadback(1, src_pages=[2], dst_pages=[3], kind=CacheKind.MAMBA)

    with pytest.raises(AssertionError, match="eager admission iter"):
        executor.flush()


def test_host_executor_loadback_uses_independent_layer_counters(monkeypatch):
    HostExecutor = _patch_host_executor_device(monkeypatch)
    kv_pool = FakePool(CacheKind.KV, page_size=2, num_layers=2)
    mamba_pool = FakePool(CacheKind.MAMBA, page_size=1, num_layers=3)
    executor = HostExecutor(pools=[kv_pool, mamba_pool], io_backend="kernel")

    executor.enqueue_loadback(10, src_pages=[4], dst_pages=[8], kind=CacheKind.KV)
    executor.enqueue_loadback(20, src_pages=[6], dst_pages=[9], kind=CacheKind.MAMBA)

    executor.flush()

    assert kv_pool.loadbacks == [
        (0, [8, 9], [16, 17]),
        (1, [8, 9], [16, 17]),
    ]
    assert mamba_pool.loadbacks == [
        (0, [6], [9]),
        (1, [6], [9]),
        (2, [6], [9]),
    ]

    assert executor.get_producer_index(CacheKind.KV, 10) == 0
    assert executor.get_producer_index(CacheKind.MAMBA, 20) == 0
    executor.set_consumer(CacheKind.KV, [0])
    executor.set_consumer(CacheKind.MAMBA, [0])
    assert kv_pool.counter.consumer == [0]


def test_host_executor_paged_cache_loadback_marks_layers_and_producer(monkeypatch):
    HostExecutor = _patch_host_executor_device(monkeypatch)
    paged_pool = FakePagedPool(num_layers=3)
    executor = HostExecutor(pools=[], paged_pool=paged_pool, io_backend="kernel")
    transfer = object()

    executor.enqueue_paged_cache_loadback(33, [transfer])
    executor.flush()

    assert paged_pool.loadbacks == [
        (0, [transfer]),
        (1, [transfer]),
        (2, [transfer]),
    ]
    assert all(event.recorded for event in paged_pool.counter.events[0].load_events)
    assert executor.get_producer_index(PAGED_CACHE_KIND, 33) == 0
    executor.set_consumer(PAGED_CACHE_KIND, [0])
    assert paged_pool.counter.consumer == [0]
    assert executor.drain() == []


def test_cache_event_payloads_commit_only_common_ops_and_and_success():
    from tokenspeed.runtime.engine.scheduler_utils import (
        pop_common_cache_event_payloads,
    )

    rank0 = [
        {"kind": "WriteBackDoneEvent", "op_id": 3, "success": True},
        {"kind": "WriteBackDoneEvent", "op_id": 5, "success": True},
    ]
    rank1 = [
        {"kind": "WriteBackDoneEvent", "op_id": 3, "success": False},
    ]

    ready = pop_common_cache_event_payloads([rank0, rank1])

    assert ready == [{"kind": "WriteBackDoneEvent", "op_id": 3, "success": False}]
    assert pop_common_cache_event_payloads([rank0, []]) == []


def test_memory_executor_submit_dispatches_flat_op_by_cache_kind(monkeypatch):
    import tokenspeed.runtime.cache.executor.memory_executor as memory_executor

    class FakeCache:
        class WriteBackOp:
            pass

        class LoadBackOp:
            pass

        class PrefetchOp:
            pass

        class BackUpOp:
            pass

    class FakeHostExec:
        def __init__(self):
            self.pools = {CacheKind.KV: object(), CacheKind.MAMBA: object()}
            self.writebacks = []
            self.loadbacks = []
            self.completed_writebacks = []
            self.order = []

        def enqueue_writeback(
            self, op_id, src_pages, dst_pages, is_retract=False, kind=CacheKind.KV
        ):
            self.order.append(("writeback", kind, op_id))
            self.writebacks.append((kind, op_id, src_pages, dst_pages, is_retract))

        def enqueue_loadback(self, op_id, src_pages, dst_pages, kind=CacheKind.KV):
            self.order.append(("loadback", kind, op_id))
            self.loadbacks.append((kind, op_id, src_pages, dst_pages))

        def flush(self):
            self.order.append(("flush",))

    monkeypatch.setattr(memory_executor, "Cache", FakeCache)
    executor = object.__new__(memory_executor.MemoryExecutor)
    executor.host_exec = FakeHostExec()
    executor.storage_exec = None

    wb = FakeCache.WriteBackOp()
    wb.op_ids = [7]
    wb.src_pages = [[1]]
    wb.dst_pages = [[11]]
    wb.src_pages_by_kind = {"kv": [[1]], "mamba": [[2, 3]]}
    wb.dst_pages_by_kind = {"kv": [[11]], "mamba": [[22, 23]]}
    wb.is_retract = [True]
    executor.submit(wb)

    assert executor.host_exec.writebacks == [
        (CacheKind.KV, 7, [1], [11], True),
        (CacheKind.MAMBA, 7, [2, 3], [22, 23], True),
    ]
    assert executor.host_exec.completed_writebacks == []

    lb = FakeCache.LoadBackOp()
    lb.op_ids = [9]
    lb.src_pages = [[10]]
    lb.dst_pages = [[20]]
    lb.src_pages_by_kind = {"kv": [[10]], "mamba": [[30]]}
    lb.dst_pages_by_kind = {"kv": [[20]], "mamba": [[40]]}
    executor.submit(lb)

    assert executor.host_exec.loadbacks == [
        (CacheKind.KV, 9, [10], [20]),
        (CacheKind.MAMBA, 9, [30], [40]),
    ]


def test_memory_executor_submit_dispatches_paged_cache_transfers(monkeypatch):
    import tokenspeed.runtime.cache.executor.memory_executor as memory_executor

    class FakeCache:
        class WriteBackOp:
            pass

        class LoadBackOp:
            pass

        class PrefetchOp:
            pass

        class BackUpOp:
            pass

    class FakeHostExec:
        def __init__(self):
            self.pools = {}
            self.completed_writebacks = []
            self.paged_writebacks = []
            self.paged_loadbacks = []

        def enqueue_paged_cache_writeback(self, op_id, transfers, is_retract=False):
            self.paged_writebacks.append((op_id, transfers, is_retract))

        def enqueue_paged_cache_loadback(self, op_id, transfers):
            self.paged_loadbacks.append((op_id, transfers))

        def flush(self):
            pass

    monkeypatch.setattr(memory_executor, "Cache", FakeCache)
    executor = object.__new__(memory_executor.MemoryExecutor)
    executor.host_exec = FakeHostExec()
    executor.storage_exec = None

    transfer = object()
    wb = FakeCache.WriteBackOp()
    wb.op_ids = [7]
    wb.src_pages = [[]]
    wb.dst_pages = [[]]
    wb.src_pages_by_kind = {"kv": [[]], "mamba": [[]]}
    wb.dst_pages_by_kind = {"kv": [[]], "mamba": [[]]}
    wb.paged_cache_transfers = [[transfer]]
    wb.is_retract = [True]
    executor.submit(wb)

    assert executor.host_exec.paged_writebacks == [(7, [transfer], True)]
    assert executor.host_exec.completed_writebacks == []

    lb = FakeCache.LoadBackOp()
    lb.op_ids = [9]
    lb.src_pages = [[]]
    lb.dst_pages = [[]]
    lb.src_pages_by_kind = {"kv": [[]], "mamba": [[]]}
    lb.dst_pages_by_kind = {"kv": [[]], "mamba": [[]]}
    lb.paged_cache_transfers = [[transfer]]
    executor.submit(lb)

    assert executor.host_exec.paged_loadbacks == [(9, [transfer])]


def test_memory_executor_submit_plan_keeps_generic_submit_signature(monkeypatch):
    import tokenspeed.runtime.cache.executor.memory_executor as memory_executor

    class FakeCache:
        class WriteBackOp:
            pass

        class LoadBackOp:
            pass

        class PrefetchOp:
            pass

        class BackUpOp:
            pass

    monkeypatch.setattr(memory_executor, "Cache", FakeCache)
    executor = object.__new__(memory_executor.MemoryExecutor)
    executor.seen = []

    wb = FakeCache.WriteBackOp()
    plan = type("Plan", (), {"cache": [wb]})()

    def submit(self, op):
        self.seen.append(op)

    monkeypatch.setattr(memory_executor.MemoryExecutor, "submit", submit)
    executor.host_exec = type("HostExec", (), {"flush": lambda self: None})()

    executor.submit_plan(plan)

    assert executor.seen == [wb]


def test_memory_executor_mamba_layerwise_cow_uses_dedicated_context(monkeypatch):
    import tokenspeed.runtime.cache.executor.memory_executor as memory_executor

    class FakeCache:
        class WriteBackOp:
            pass

        class LoadBackOp:
            pass

        class PrefetchOp:
            pass

        class BackUpOp:
            pass

    class FakeHostExec:
        def __init__(self):
            self.pools = {CacheKind.KV: object(), CacheKind.MAMBA: object()}
            self.completed_writebacks = []
            self.loadbacks = []

        def enqueue_loadback(
            self,
            op_id,
            src_pages,
            dst_pages,
            kind=CacheKind.KV,
            layerwise_cow_dst_pages_by_src=None,
        ):
            self.loadbacks.append(
                (kind, op_id, src_pages, dst_pages, layerwise_cow_dst_pages_by_src)
            )

        def flush(self):
            pass

    monkeypatch.setattr(memory_executor, "Cache", FakeCache)
    executor = object.__new__(memory_executor.MemoryExecutor)
    executor.host_exec = FakeHostExec()
    executor.storage_exec = None
    executor.set_mamba_layerwise_cow({40: [400]})

    lb = FakeCache.LoadBackOp()
    lb.op_ids = [9]
    lb.src_pages = [[10]]
    lb.dst_pages = [[20]]
    lb.src_pages_by_kind = {"kv": [[10]], "mamba": [[30]]}
    lb.dst_pages_by_kind = {"kv": [[20]], "mamba": [[40]]}
    plan = type("Plan", (), {"cache": [lb]})()

    executor.submit_plan(plan)

    assert executor.host_exec.loadbacks == [
        (CacheKind.KV, 9, [10], [20], None),
        (CacheKind.MAMBA, 9, [30], [40], {40: [400]}),
    ]
    assert executor._pending_mamba_layerwise_cow is None
