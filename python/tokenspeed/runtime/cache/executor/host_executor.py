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
from typing import Iterable, NamedTuple

import torch
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()
CONCURRENT_WRITEBACK_BLOCK_QUOTA = 2


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


def _run_per_layer_loadback(
    host_pool,
    device_pool,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    io_backend: str,
    layer_num: int,
    producer_event,
) -> None:
    for layer_index in range(layer_num):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_index, io_backend
        )
        if producer_event is not None:
            producer_event.complete(layer_index)


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

        write_priority, load_priority = _cache_stream_priorities()
        self.write_stream = _new_cache_stream(write_priority)
        self.load_stream = _new_cache_stream(load_priority)
        self._writeback_block_quota: int | None = None

        self.write_queue: list[_TransferOp] = []
        self.load_queue: list[_TransferOp] = []

        self.ack_write_queue: list[_Ack] = []
        self.ack_load_queue: list[_Ack] = []
        self.completed_writebacks: list[int] = []

        self.layer_done_counter = LayerDoneCounter(layer_num)
        device_pool.register_layer_transfer_counter(self.layer_done_counter)

        self._producer_map: OrderedDict[int, int] = OrderedDict()
        self._producer_map_limit = 1024

    def enqueue_writeback(
        self, op_id, src_pages, dst_pages, is_retract: bool = False
    ) -> None:
        src_pages, dst_pages = _dedupe_page_pairs(src_pages, dst_pages)
        if not src_pages:
            completed_writebacks = getattr(self, "completed_writebacks", None)
            if completed_writebacks is None:
                completed_writebacks = []
                self.completed_writebacks = completed_writebacks
            completed_writebacks.append(op_id)
            return
        # Both on host; _move_indices async-moves to idx_device.
        device_indices = page_ids_to_token_indices(src_pages, self.page_size, "cpu")
        host_indices = page_ids_to_token_indices(dst_pages, self.page_size, "cpu")
        self.write_queue.append(
            _TransferOp(host_indices, device_indices, op_id, is_retract)
        )

    def enqueue_loadback(self, op_id, src_pages, dst_pages) -> None:
        src_pages, dst_pages = _dedupe_page_pairs(src_pages, dst_pages)
        if not src_pages:
            return
        host_indices = page_ids_to_token_indices(src_pages, self.page_size, "cpu")
        device_indices = page_ids_to_token_indices(dst_pages, self.page_size, "cpu")
        self.load_queue.append(_TransferOp(host_indices, device_indices, op_id))

    def flush(self) -> None:
        throttle_writeback = self.load_queue and not any(
            getattr(op, "is_retract", False) for op in self.write_queue
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

    def _start_writing(self) -> None:
        if not self.write_queue:
            return

        op = _TransferOp.merge(self.write_queue)
        host_indices, device_indices = self._move_indices(op, self.host_pool)
        # Prepare draft indices outside the stream context so non_blocking H2D
        # copies are issued on the default stream and then consumed by write_stream.
        if self.draft_host_pool is not None:
            draft_host_indices, draft_device_indices = self._move_indices(
                op, self.draft_host_pool
            )
        else:
            draft_host_indices = draft_device_indices = None
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.host_pool.backup_from_device_all_layer(
                self.device_pool,
                host_indices.to(torch.int64),
                device_indices.to(torch.int64),
                self.io_backend,
                block_quota=self._writeback_block_quota,
            )
            # Draft model shares the same page mapping; backup its KV cache too.
            if self.draft_host_pool is not None:
                self.draft_host_pool.backup_from_device_all_layer(
                    self.draft_device_pool,
                    draft_host_indices.to(torch.int64),
                    draft_device_indices.to(torch.int64),
                    self.io_backend,
                    block_quota=self._writeback_block_quota,
                )
                if draft_host_indices.is_cuda:
                    draft_host_indices.record_stream(self.write_stream)
                if draft_device_indices.is_cuda:
                    draft_device_indices.record_stream(self.write_stream)
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(_Ack(finish_event, op.node_ids))

    def _start_loading(self) -> None:
        if not self.load_queue:
            return

        producer_id = self.layer_done_counter.update_producer()
        op = _TransferOp.merge(self.load_queue)
        host_indices, device_indices = self._move_indices(op, self.host_pool)
        self.load_queue.clear()

        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        # Prepare draft indices once if draft pool is present.
        if self.draft_host_pool is not None:
            draft_host_indices, draft_device_indices = self._move_indices(
                op, self.draft_host_pool
            )
        else:
            draft_host_indices = draft_device_indices = None

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            _run_per_layer_loadback(
                self.host_pool,
                self.device_pool,
                host_indices.to(torch.int64),
                device_indices.to(torch.int64),
                self.io_backend,
                self.layer_num,
                producer_event,
            )
            # Draft layers follow base layers in the same load stream.
            if self.draft_host_pool is not None:
                _run_per_layer_loadback(
                    self.draft_host_pool,
                    self.draft_device_pool,
                    draft_host_indices.to(torch.int64),
                    draft_device_indices.to(torch.int64),
                    self.io_backend,
                    self.draft_layer_num,
                    producer_event=None,
                )
                if draft_host_indices.is_cuda:
                    draft_host_indices.record_stream(self.load_stream)
                if draft_device_indices.is_cuda:
                    draft_device_indices.record_stream(self.load_stream)
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(_Ack(producer_event.finish_event, op.node_ids))
        for op_id in op.node_ids:
            self._producer_map[op_id] = producer_id
        while len(self._producer_map) > self._producer_map_limit:
            self._producer_map.popitem(last=False)

    def _move_indices(self, op: _TransferOp, host_pool):
        host_indices = op.host_indices
        device_indices = op.device_indices
        if self.io_backend == "kernel":
            idx_device = host_pool.idx_device
            if host_indices.device != idx_device:
                host_indices = host_indices.to(idx_device, non_blocking=True)
            if device_indices.device != idx_device:
                device_indices = device_indices.to(idx_device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if host_pool.layout == "layer_first":
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
        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        self._producer_map.clear()
        self.layer_done_counter.reset()
