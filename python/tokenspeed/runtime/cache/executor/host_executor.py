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

"""Host-side executor for cache writeback and loadback operations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, NamedTuple

import torch
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()
CONCURRENT_WRITEBACK_BLOCK_QUOTA = 2


KV_TRANSFER_KIND = int(Cache.TRANSFER_KIND_KV)
MAMBA_TRANSFER_KIND = int(Cache.TRANSFER_KIND_MAMBA)
TransferPairs = tuple[tuple[int, int], ...]
TransferGroups = tuple[tuple[int, TransferPairs], ...]


@dataclass(frozen=True, slots=True)
class CacheTransferBatch:
    op_id: int
    units_by_kind: TransferGroups
    is_retract: bool = False

    @classmethod
    def from_pages(
        cls,
        op_id: int,
        src_pages: Iterable[int],
        dst_pages: Iterable[int],
        *,
        is_retract: bool = False,
    ) -> "CacheTransferBatch":
        pairs = tuple((int(src), int(dst)) for src, dst in zip(src_pages, dst_pages))
        groups = ((KV_TRANSFER_KIND, pairs),) if pairs else ()
        return cls(int(op_id), groups, is_retract)

    @classmethod
    def from_scheduler(
        cls, op_id: int, kinds, src_indices, dst_indices, *, is_retract: bool = False
    ) -> "CacheTransferBatch":
        grouped: dict[int, list[tuple[int, int]]] = {}
        for kind, src, dst in zip(kinds, src_indices, dst_indices):
            grouped.setdefault(int(kind), []).append((int(src), int(dst)))
        return cls(int(op_id), cls._freeze(grouped), is_retract)

    @staticmethod
    def _freeze(groups: dict[int, list[tuple[int, int]]]) -> TransferGroups:
        return tuple((kind, tuple(pairs)) for kind, pairs in groups.items() if pairs)

    def pairs(self, kind: int) -> TransferPairs:
        for group_kind, pairs in self.units_by_kind:
            if group_kind == int(kind):
                return pairs
        return ()

    def mamba_pairs(self) -> TransferPairs:
        return self.pairs(MAMBA_TRANSFER_KIND)


def _cache_stream_priorities() -> tuple[int | None, int | None]:
    priority_range = getattr(device_module.Stream, "priority_range", None)
    if priority_range is None:
        return None, None
    try:
        least_priority, greatest_priority = priority_range()
    except (RuntimeError, TypeError):
        return None, None
    return least_priority, greatest_priority


def _new_cache_stream(priority: int | None = None):
    if priority is None:
        return device_module.Stream()
    try:
        return device_module.Stream(priority=priority)
    except (RuntimeError, TypeError):
        return device_module.Stream()


def page_ids_to_token_indices(
    page_ids: list[int],
    page_size: int,
    device: str = "cpu",
) -> torch.Tensor:
    if len(page_ids) == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    pages = torch.tensor(page_ids, dtype=torch.int64, device=device)
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (pages[:, None] * page_size + offsets[None, :]).reshape(-1)


def _dedupe_page_pairs(
    src_pages: Iterable[int],
    dst_pages: Iterable[int],
) -> tuple[list[int], list[int]]:
    seen = set()
    deduped_src = []
    deduped_dst = []
    for src_page, dst_page in zip(src_pages, dst_pages):
        pair = (int(src_page), int(dst_page))
        if pair in seen:
            continue
        seen.add(pair)
        deduped_src.append(pair[0])
        deduped_dst.append(pair[1])
    return deduped_src, deduped_dst


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()

    def complete(self, layer_index: int) -> None:
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int) -> None:
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_indices: tuple[int, ...] = ()

    def update_producer(self) -> int:
        next_index = (self.producer_index + 1) % self.num_counters
        if not self.events[next_index].finish_event.query():
            self.events[next_index].finish_event.synchronize()
        self.producer_index = next_index
        return self.producer_index

    def set_consumer(self, indices: int | Iterable[int]) -> None:
        if isinstance(indices, int):
            self.consumer_indices = () if indices < 0 else (indices,)
            return
        deduped = []
        for index in indices:
            if index >= 0 and index not in deduped:
                deduped.append(index)
        self.consumer_indices = tuple(deduped)

    def wait_until(self, threshold: int) -> None:
        if not self.consumer_indices:
            return
        for consumer_index in self.consumer_indices:
            self.events[consumer_index].wait(threshold)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_indices = ()


class _TransferOp:

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        is_retract: bool = False,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.is_retract = is_retract

    @staticmethod
    def merge(ops: list["_TransferOp"]) -> "_TransferOp":
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        merged = _TransferOp(host_indices, device_indices, -1)
        merged.node_ids = []
        merged.is_retract = False
        for op in ops:
            merged.node_ids.extend(op.node_ids)
            merged.is_retract = merged.is_retract or op.is_retract
        return merged


class _Ack(NamedTuple):
    finish_event: object  # device_module.Event
    node_ids: list[int]


class _CacheTransferBackend:
    kind: int
    write_order = 0
    load_order = 0

    def make_op(
        self,
        executor,
        op_id: int,
        pairs: TransferPairs,
        is_retract: bool,
        *,
        is_writeback: bool,
    ) -> _TransferOp | None:
        raise NotImplementedError

    def prepare_write(self, executor, op: _TransferOp):
        return None

    def write(self, executor, op: _TransferOp, prepared) -> None:
        raise NotImplementedError

    def prepare_load(self, executor, op: _TransferOp):
        return None

    def load(
        self, executor, op: _TransferOp, prepared, producer_event: LayerLoadingEvent
    ) -> bool:
        raise NotImplementedError


class _KVTransferBackend(_CacheTransferBackend):
    kind = KV_TRANSFER_KIND
    write_order = 0
    load_order = 10

    def make_op(
        self,
        executor,
        op_id: int,
        pairs: TransferPairs,
        is_retract: bool,
        *,
        is_writeback: bool,
    ) -> _TransferOp | None:
        src = [src for src, _ in pairs]
        dst = [dst for _, dst in pairs]
        if is_writeback:
            device_pages, host_pages = _dedupe_page_pairs(src, dst)
        else:
            host_pages, device_pages = _dedupe_page_pairs(src, dst)
        if not host_pages:
            return None
        host_indices = page_ids_to_token_indices(host_pages, executor.page_size, "cpu")
        device_indices = page_ids_to_token_indices(
            device_pages, executor.page_size, str(executor.device)
        )
        return _TransferOp(host_indices, device_indices, op_id, is_retract)

    def prepare_write(self, executor, op: _TransferOp):
        return self._prepare_indices(executor, op)

    def write(self, executor, op: _TransferOp, prepared) -> None:
        host_indices, device_indices, draft_host_indices, draft_device_indices = (
            prepared
        )
        executor.host_pool.backup_from_device_all_layer(
            executor.device_pool,
            host_indices.to(torch.int64),
            device_indices.to(torch.int64),
            executor.io_backend,
            block_quota=executor._writeback_block_quota,
        )
        if executor.draft_host_pool is not None:
            executor.draft_host_pool.backup_from_device_all_layer(
                executor.draft_device_pool,
                draft_host_indices.to(torch.int64),
                draft_device_indices.to(torch.int64),
                executor.io_backend,
                block_quota=executor._writeback_block_quota,
            )
            self._record_stream(draft_host_indices, executor.write_stream)
            self._record_stream(draft_device_indices, executor.write_stream)
        self._record_stream(host_indices, executor.write_stream)
        self._record_stream(device_indices, executor.write_stream)

    def prepare_load(self, executor, op: _TransferOp):
        return self._prepare_indices(executor, op)

    def load(
        self, executor, op: _TransferOp, prepared, producer_event: LayerLoadingEvent
    ) -> bool:
        host_indices, device_indices, draft_host_indices, draft_device_indices = (
            prepared
        )
        for layer_index in range(executor.layer_num):
            executor.host_pool.load_to_device_per_layer(
                executor.device_pool,
                host_indices.to(torch.int64),
                device_indices.to(torch.int64),
                layer_index,
                executor.io_backend,
            )
            producer_event.complete(layer_index)
        if executor.draft_host_pool is not None:
            for layer_index in range(executor.draft_layer_num):
                executor.draft_host_pool.load_to_device_per_layer(
                    executor.draft_device_pool,
                    draft_host_indices.to(torch.int64),
                    draft_device_indices.to(torch.int64),
                    layer_index,
                    executor.io_backend,
                )
            self._record_stream(draft_host_indices, executor.load_stream)
            self._record_stream(draft_device_indices, executor.load_stream)
        self._record_stream(host_indices, executor.load_stream)
        self._record_stream(device_indices, executor.load_stream)
        return True

    @staticmethod
    def _prepare_indices(executor, op: _TransferOp):
        host_indices, device_indices = executor._move_indices(op, executor.host_pool)
        if executor.draft_host_pool is None:
            return host_indices, device_indices, None, None
        draft_host_indices, draft_device_indices = executor._move_indices(
            op, executor.draft_host_pool
        )
        return host_indices, device_indices, draft_host_indices, draft_device_indices

    @staticmethod
    def _record_stream(tensor, stream) -> None:
        if tensor is not None and tensor.is_cuda:
            tensor.record_stream(stream)


class _MambaTransferBackend(_CacheTransferBackend):
    kind = MAMBA_TRANSFER_KIND
    write_order = 10
    load_order = 0

    def make_op(
        self,
        executor,
        op_id: int,
        pairs: TransferPairs,
        is_retract: bool,
        *,
        is_writeback: bool,
    ) -> _TransferOp | None:
        executor._require_mamba_pools()
        src = [src for src, _ in pairs]
        dst = [dst for _, dst in pairs]
        if is_writeback:
            device_slots, host_slots = src, dst
        else:
            host_slots, device_slots = src, dst
        if not host_slots:
            return None
        host_indices = torch.tensor(host_slots, dtype=torch.int64)
        device_indices = torch.tensor(device_slots, dtype=torch.int64)
        return _TransferOp(host_indices, device_indices, op_id, is_retract)

    def write(self, executor, op: _TransferOp, prepared) -> None:
        executor._copy_mamba_slots(
            executor.mamba_pool,
            executor.mamba_host_pool,
            op.device_indices,
            op.host_indices,
        )

    def load(
        self, executor, op: _TransferOp, prepared, producer_event: LayerLoadingEvent
    ) -> bool:
        executor._copy_mamba_slots(
            executor.mamba_host_pool,
            executor.mamba_pool,
            op.host_indices,
            op.device_indices,
        )
        return False


class HostExecutor:

    def __init__(
        self,
        page_size: int,
        device_pool,
        host_pool,
        io_backend: str,
        layer_num: int,
        draft_device_pool=None,
        draft_host_pool=None,
        draft_layer_num: int = 0,
        mamba_pool=None,
        mamba_host_pool=None,
    ):
        self.page_size = page_size
        self.device_pool = device_pool
        self.host_pool = host_pool
        self.io_backend = io_backend
        self.layer_num = layer_num
        self.device = device_pool.device

        # Optional draft model pools (share the same page mapping as base model)
        self.draft_device_pool = draft_device_pool
        self.draft_host_pool = draft_host_pool
        self.draft_layer_num = draft_layer_num
        self.mamba_pool = mamba_pool
        self.mamba_host_pool = mamba_host_pool

        write_priority, load_priority = _cache_stream_priorities()
        self.write_stream = _new_cache_stream(write_priority)
        self.load_stream = _new_cache_stream(load_priority)
        self._writeback_block_quota: int | None = None

        self.transfer_backends = self._build_transfer_backends()
        self.write_queues: dict[int, list[_TransferOp]] = {
            kind: [] for kind in self.transfer_backends
        }
        self.load_queues: dict[int, list[_TransferOp]] = {
            kind: [] for kind in self.transfer_backends
        }

        self.ack_write_queue: list[_Ack] = []
        self.ack_load_queue: list[_Ack] = []
        self.completed_writebacks: list[int] = []

        self.layer_done_counter = LayerDoneCounter(layer_num)
        device_pool.register_layer_transfer_counter(self.layer_done_counter)

        self._producer_map: OrderedDict[int, int] = OrderedDict()
        self._producer_map_limit = 1024

    @staticmethod
    def _build_transfer_backends() -> dict[int, _CacheTransferBackend]:
        backends = (_KVTransferBackend(), _MambaTransferBackend())
        return {backend.kind: backend for backend in backends}

    def enqueue_writeback(self, batch: CacheTransferBatch) -> None:
        self._enqueue_batch(batch, is_writeback=True)

    def enqueue_loadback(self, batch: CacheTransferBatch) -> None:
        self._enqueue_batch(batch, is_writeback=False)

    def _enqueue_batch(
        self,
        batch: CacheTransferBatch,
        *,
        is_writeback: bool,
    ) -> None:
        queues = self.write_queues if is_writeback else self.load_queues
        queued = False
        for kind, pairs in batch.units_by_kind:
            backend = self.transfer_backends.get(kind)
            if backend is None:
                raise ValueError(f"unsupported cache transfer kind={kind}")
            op = backend.make_op(
                self,
                batch.op_id,
                pairs,
                batch.is_retract,
                is_writeback=is_writeback,
            )
            if op is None:
                continue
            queues[kind].append(op)
            queued = True
        if is_writeback and not queued:
            self.completed_writebacks.append(batch.op_id)

    def _require_mamba_pools(self) -> None:
        if self.mamba_pool is None or self.mamba_host_pool is None:
            raise RuntimeError("mamba cache transfer requested without mamba pools")

    @staticmethod
    def _has_queued(queues: dict[int, list[_TransferOp]]) -> bool:
        return any(queues.values())

    @staticmethod
    def _merge_queue(queue: list[_TransferOp]) -> _TransferOp | None:
        return _TransferOp.merge(queue) if queue else None

    @staticmethod
    def _clear_queues(queues: dict[int, list[_TransferOp]]) -> None:
        for queue in queues.values():
            queue.clear()

    def flush(self) -> None:
        has_loadback = self._has_queued(self.load_queues)
        write_queues = [op for queue in self.write_queues.values() for op in queue]
        throttle_writeback = has_loadback and not any(
            getattr(op, "is_retract", False) for op in write_queues
        )
        writeback_block_quota = (
            CONCURRENT_WRITEBACK_BLOCK_QUOTA if throttle_writeback else None
        )
        previous_writeback_block_quota = getattr(self, "_writeback_block_quota", None)
        self._writeback_block_quota = writeback_block_quota
        try:
            self._start_loading()
            self._start_writing()
        finally:
            self._writeback_block_quota = previous_writeback_block_quota

    def _queued_transfers(
        self,
        queues: dict[int, list[_TransferOp]],
        *,
        order_attr: str,
    ) -> list[tuple[int, _CacheTransferBackend, _TransferOp]]:
        transfers = []
        for kind, queue in queues.items():
            op = self._merge_queue(queue)
            if op is None:
                continue
            backend = self.transfer_backends[kind]
            transfers.append((getattr(backend, order_attr), kind, backend, op))
        transfers.sort(key=lambda item: item[0])
        return [(kind, backend, op) for _, kind, backend, op in transfers]

    @staticmethod
    def _node_ids(
        transfers: list[tuple[int, _CacheTransferBackend, _TransferOp]],
    ) -> list[int]:
        node_ids = []
        for _, _, op in transfers:
            node_ids.extend(op.node_ids)
        return node_ids

    def _start_writing(self) -> None:
        transfers = self._queued_transfers(self.write_queues, order_attr="write_order")
        if not transfers:
            return

        prepared = {
            kind: backend.prepare_write(self, op) for kind, backend, op in transfers
        }
        self._clear_queues(self.write_queues)
        node_ids = self._node_ids(transfers)

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            for kind, backend, op in transfers:
                backend.write(self, op, prepared[kind])
            finish_event.record()

        self.ack_write_queue.append(_Ack(finish_event, node_ids))

    def _start_loading(self) -> None:
        transfers = self._queued_transfers(self.load_queues, order_attr="load_order")
        if not transfers:
            return

        producer_id = self.layer_done_counter.update_producer()
        prepared = {
            kind: backend.prepare_load(self, op) for kind, backend, op in transfers
        }
        self._clear_queues(self.load_queues)
        node_ids = self._node_ids(transfers)

        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            completed_layers = False
            for kind, backend, op in transfers:
                completed_layers = (
                    backend.load(self, op, prepared[kind], producer_event)
                    or completed_layers
                )
            if not completed_layers:
                for layer_index in range(self.layer_num):
                    producer_event.complete(layer_index)

        self.ack_load_queue.append(_Ack(producer_event.finish_event, node_ids))
        for op_id in node_ids:
            self._producer_map[op_id] = producer_id
        while len(self._producer_map) > self._producer_map_limit:
            self._producer_map.popitem(last=False)

    def _copy_mamba_slots(self, src_pool, dst_pool, src_indices, dst_indices) -> None:
        for src_idx, dst_idx in zip(src_indices.tolist(), dst_indices.tolist()):
            for src_cache, dst_cache in zip(src_pool.mamba_cache, dst_pool.mamba_cache):
                dst_cache[:, int(dst_idx)].copy_(
                    src_cache[:, int(src_idx)], non_blocking=True
                )

    def _move_indices(self, op: _TransferOp, host_pool):
        host_indices = op.host_indices
        device_indices = op.device_indices
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if host_pool.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
        raise ValueError(f"Unsupported io_backend={self.io_backend}")

    def drain(self) -> list:
        results: list = []
        results.extend(self._poll_write_acks())
        results.extend(self._poll_load_acks())
        return results

    def _poll_write_acks(self) -> list:
        results = []
        completed_writebacks = getattr(self, "completed_writebacks", [])
        for op_id in completed_writebacks:
            evt = Cache.WriteBackDoneEvent()
            evt.op_id = op_id
            evt.success = True
            results.append(evt)
        completed_writebacks.clear()
        remaining = []
        for ack in self.ack_write_queue:
            if ack.finish_event.query():
                for op_id in ack.node_ids:
                    evt = Cache.WriteBackDoneEvent()
                    evt.op_id = op_id
                    evt.success = True
                    results.append(evt)
            else:
                remaining.append(ack)
        self.ack_write_queue[:] = remaining
        return results

    def _poll_load_acks(self) -> list:
        results = []
        remaining = []
        for ack in self.ack_load_queue:
            if not ack.finish_event.query():
                remaining.append(ack)
        self.ack_load_queue[:] = remaining
        return results

    def get_producer_index(self, op_id: int) -> int | None:
        return self._producer_map.pop(op_id, None)

    def set_consumer(self, producer_index: int | Iterable[int]) -> None:
        self.layer_done_counter.set_consumer(producer_index)

    def shutdown(self) -> None:
        self.write_stream.synchronize()
        self.load_stream.synchronize()

    def reset(self) -> None:
        self.write_stream.synchronize()
        self.load_stream.synchronize()
        self._clear_queues(self.write_queues)
        self._clear_queues(self.load_queues)
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        self._producer_map.clear()
        self.layer_done_counter.reset()
