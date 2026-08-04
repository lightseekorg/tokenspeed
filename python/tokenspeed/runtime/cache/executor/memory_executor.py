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

"""Descriptor-driven executor for compact Host cache transfers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import psutil
import torch
from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_segments
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.host_storage import HostCacheStorage
from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter
from tokenspeed.runtime.cache.layout import combine_cache_transfer_layouts
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()

_HOST_MEM_HEADROOM_BYTES = 10 * (1024**3)


def _cache_stream_priorities() -> tuple[int | None, int | None]:
    priority_range = getattr(device_module.Stream, "priority_range", None)
    if priority_range is None:
        return None, None
    try:
        return priority_range()
    except (RuntimeError, TypeError):
        return None, None


def _new_cache_stream(priority: int | None = None):
    if priority is None:
        return device_module.Stream()
    try:
        return device_module.Stream(priority=priority)
    except (RuntimeError, TypeError):
        return device_module.Stream()


def _ordered_unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


class _Ack(NamedTuple):
    finish_event: object
    op_ids: list[int]


def _num_host_lcm_blocks(
    *, parent_bytes: int, device_lcm_blocks: int, host_ratio: float, host_size_gb: float
) -> int:
    if host_size_gb > 0:
        count = int(host_size_gb * 1e9 // parent_bytes)
    else:
        count = int(device_lcm_blocks * host_ratio)
    if count <= 0:
        raise ValueError("Host L2 resolved to zero LCM blocks")
    return count


class MemoryExecutor:
    """Execute group-aware D2H/H2D operations against one compact Host pool."""

    emits_loadback_acks = True

    def __init__(
        self,
        device_pool,
        *,
        draft_pool=None,
        host_ratio: float,
        host_size_gb: float,
    ):
        target_layout = device_pool.cache_transfer_layout()
        draft_layout = (
            draft_pool.cache_transfer_layout() if draft_pool is not None else None
        )
        self.layout = combine_cache_transfer_layouts(target_layout, draft_layout)
        self.packed = self.layout.pack()
        host_lcm_blocks = _num_host_lcm_blocks(
            parent_bytes=self.packed.parent_bytes,
            device_lcm_blocks=self.layout.num_lcm_blocks,
            host_ratio=host_ratio,
            host_size_gb=host_size_gb,
        )
        requested_bytes = host_lcm_blocks * self.packed.parent_bytes
        available_bytes = psutil.virtual_memory().available - _HOST_MEM_HEADROOM_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                "Not enough Host memory for L2: requesting "
                f"{requested_bytes / 1e9:.2f} GB, available "
                f"{available_bytes / 1e9:.2f} GB"
            )
        self.storage = HostCacheStorage(self.layout, num_lcm_blocks=host_lcm_blocks)
        # C++ includes logical null parent 0 in allocator page counts.
        self.num_host_pages = host_lcm_blocks + 1
        logger.info(
            "Allocated %.2f GB compact Host L2 (%s parents, %s bytes/parent)",
            requested_bytes / 1e9,
            host_lcm_blocks,
            self.packed.parent_bytes,
        )

        pool_layouts = [(device_pool, target_layout)]
        if draft_pool is not None:
            pool_layouts.append((draft_pool, draft_layout))
        self._consumer_counters = []
        for pool, layout in pool_layouts:
            counter = LayerDoneCounter(len(layout.consumers))
            pool.register_layer_transfer_counter(counter)
            self._consumer_counters.append((counter, len(layout.consumers)))
        write_priority, load_priority = _cache_stream_priorities()
        self.write_stream = _new_cache_stream(write_priority)
        self.load_stream = _new_cache_stream(load_priority)

        self._pending_write_transfers: list[tuple[int, int, int]] = []
        self._pending_write_op_ids: list[int] = []
        self._pending_load_transfers: list[tuple[int, int, int]] = []
        self._pending_load_op_ids: list[int] = []
        self.ack_write_queue: list[_Ack] = []
        self.ack_load_queue: list[_Ack] = []
        self._immediate_write_op_ids: list[int] = []
        self._immediate_load_op_ids: list[int] = []
        self._producer_map: OrderedDict[int, int] = OrderedDict()
        self._producer_map_limit = 1024

    def submit_plan(self, plan) -> None:
        for operation in plan.cache:
            self.submit(operation)
        self.flush()

    def submit(self, operation) -> None:
        if isinstance(operation, Cache.WriteBackOp):
            self._submit(
                operation.op_ids,
                operation.group_ids,
                operation.src_pages,
                operation.dst_pages,
                pending_op_ids=self._pending_write_op_ids,
                pending_transfers=self._pending_write_transfers,
                source_is_device=True,
            )
        elif isinstance(operation, Cache.LoadBackOp):
            self._submit(
                operation.op_ids,
                operation.group_ids,
                operation.src_pages,
                operation.dst_pages,
                pending_op_ids=self._pending_load_op_ids,
                pending_transfers=self._pending_load_transfers,
                source_is_device=False,
            )
        else:
            raise ValueError(f"unsupported cache op {type(operation).__name__}")

    @staticmethod
    def _submit(
        op_ids: Sequence[int],
        group_ids: Sequence[Sequence[int]],
        src_pages: Sequence[Sequence[int]],
        dst_pages: Sequence[Sequence[int]],
        *,
        pending_op_ids: list[int],
        pending_transfers: list[tuple[int, int, int]],
        source_is_device: bool,
    ) -> None:
        if not (len(op_ids) == len(group_ids) == len(src_pages) == len(dst_pages)):
            raise ValueError("ragged cache operation batch")
        for op_id, groups, sources, destinations in zip(
            op_ids, group_ids, src_pages, dst_pages
        ):
            if not (len(groups) == len(sources) == len(destinations)):
                raise ValueError(f"ragged cache operation {op_id}")
            pending_op_ids.append(int(op_id))
            for group, source, destination in zip(groups, sources, destinations):
                device_page, host_page = (
                    (source, destination) if source_is_device else (destination, source)
                )
                pending_transfers.append((int(group), int(device_page), int(host_page)))

    def _descriptors(
        self,
        transfers: Sequence[tuple[int, int, int]],
        segment_ids: set[str] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        descriptors = []
        for group_index, device_page, host_page in transfers:
            group = self.layout.groups[group_index]
            for segment_index, segment in enumerate(group.segments):
                if segment_ids is not None and segment.segment_id not in segment_ids:
                    continue
                descriptors.append(
                    (
                        segment.buffer_index,
                        segment.page_zero_offset
                        + device_page * segment.page_stride_bytes,
                        self.storage.segment_offset(
                            group_index, host_page, segment_index
                        ),
                        segment.payload_bytes,
                    )
                )
        return descriptors

    def flush(self) -> None:
        self._start_loading()
        self._start_writing()

    def _start_writing(self) -> None:
        if not self._pending_write_op_ids:
            return
        op_ids = _ordered_unique(self._pending_write_op_ids)
        transfers = self._pending_write_transfers
        self._pending_write_op_ids = []
        self._pending_write_transfers = []
        if not transfers:
            self._immediate_write_op_ids.extend(op_ids)
            return
        # Retraction is issued only after the scheduler has consumed every
        # outstanding forward result. Still order this stream explicitly after
        # current-stream work before reading the Device snapshot.
        start = torch.cuda.Event()
        start.record()
        start.wait(self.write_stream)
        transfer_cache_segments(
            "d2h",
            self.layout.buffers,
            self.storage.backing,
            self._descriptors(transfers),
            self.write_stream,
        )
        finish = torch.cuda.Event()
        finish.record(self.write_stream)
        self.ack_write_queue.append(_Ack(finish, op_ids))

    def _start_loading(self) -> None:
        if not self._pending_load_op_ids:
            return
        if get_is_capture_mode():
            raise RuntimeError("Host cache load must run outside CUDA Graph capture")
        op_ids = _ordered_unique(self._pending_load_op_ids)
        transfers = self._pending_load_transfers
        self._pending_load_op_ids = []
        self._pending_load_transfers = []
        if not transfers:
            self._immediate_load_op_ids.extend(op_ids)
            return

        # EventLoop zeroes freshly allocated Device pages before submitting the
        # load. Recording the producer event here makes every layer-wise H2D
        # copy wait for that zeroing; the per-layer load events then keep model
        # consumers from reading a partially restored snapshot.
        producer_id = None
        consumer_offset = 0
        finish = None
        for counter, consumer_count in self._consumer_counters:
            current_producer_id = counter.update_producer()
            if producer_id is None:
                producer_id = current_producer_id
            elif current_producer_id != producer_id:
                raise RuntimeError("target and draft Host-load counters diverged")
            producer = counter.events[current_producer_id]
            producer.start_event.record()
            producer.start_event.wait(self.load_stream)
            for layer_index in range(consumer_count):
                consumer = self.layout.consumers[consumer_offset + layer_index]
                transfer_cache_segments(
                    "h2d",
                    self.layout.buffers,
                    self.storage.backing,
                    self._descriptors(transfers, set(consumer)),
                    self.load_stream,
                )
                finish = torch.cuda.Event()
                finish.record(self.load_stream)
                producer.load_events[layer_index] = finish
            consumer_offset += consumer_count
        if producer_id is None or finish is None:
            raise RuntimeError("cache transfer layout has no layer consumers")
        self.ack_load_queue.append(_Ack(finish, op_ids))
        for op_id in op_ids:
            self._producer_map[op_id] = producer_id
        while len(self._producer_map) > self._producer_map_limit:
            self._producer_map.popitem(last=False)

    def poll_results(self) -> list:
        results = [self._write_done(op_id) for op_id in self._immediate_write_op_ids]
        self._immediate_write_op_ids.clear()
        results.extend(self._load_done(op_id) for op_id in self._immediate_load_op_ids)
        self._immediate_load_op_ids.clear()
        self.ack_write_queue[:] = self._drain(
            self.ack_write_queue, self._write_done, results
        )
        self.ack_load_queue[:] = self._drain(
            self.ack_load_queue, self._load_done, results
        )
        return results

    @staticmethod
    def _drain(queue, done, results):
        pending = []
        for ack in queue:
            if ack.finish_event.query():
                results.extend(done(op_id) for op_id in ack.op_ids)
            else:
                pending.append(ack)
        return pending

    @staticmethod
    def _write_done(op_id: int):
        event = Cache.WriteBackDoneEvent()
        event.op_id = op_id
        event.success = True
        return event

    @staticmethod
    def _load_done(op_id: int):
        event = Cache.LoadBackDoneEvent()
        event.op_id = op_id
        event.success = True
        return event

    def get_producer_index(self, op_id: int):
        return self._producer_map.pop(int(op_id), None)

    def set_consumer(self, producer_index) -> None:
        for counter, _ in self._consumer_counters:
            counter.set_consumer(producer_index)

    def shutdown(self) -> None:
        self.write_stream.synchronize()
        self.load_stream.synchronize()

    def reset(self) -> None:
        self.shutdown()
        self._pending_write_transfers.clear()
        self._pending_write_op_ids.clear()
        self._pending_load_transfers.clear()
        self._pending_load_op_ids.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        self._immediate_write_op_ids.clear()
        self._immediate_load_op_ids.clear()
        self._producer_map.clear()
        for counter, _ in self._consumer_counters:
            counter.reset()
