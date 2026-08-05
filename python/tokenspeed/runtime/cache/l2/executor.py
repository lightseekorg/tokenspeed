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
from tokenspeed_kernel.ops.kvcache.host_transfer import transfer_cache_ranges
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.l2.storage import (
    HostCacheStorage,
    compute_host_lcm_block_bytes,
)
from tokenspeed.runtime.cache.transfer.layout import combine_cache_transfer_layouts
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
    *,
    host_lcm_block_bytes: int,
    device_lcm_blocks: int,
    host_ratio: float,
    host_size_gb: float,
) -> int:
    if host_size_gb > 0:
        count = int(host_size_gb * 1e9 // host_lcm_block_bytes)
    else:
        count = int(device_lcm_blocks * host_ratio)
    if count <= 0:
        raise ValueError("Host L2 resolved to zero LCM blocks")
    return count


class L2CacheExecutor:
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
        host_lcm_block_bytes = compute_host_lcm_block_bytes(self.layout)
        host_lcm_blocks = _num_host_lcm_blocks(
            host_lcm_block_bytes=host_lcm_block_bytes,
            device_lcm_blocks=self.layout.num_lcm_blocks,
            host_ratio=host_ratio,
            host_size_gb=host_size_gb,
        )
        requested_host_bytes = host_lcm_blocks * host_lcm_block_bytes
        available_host_bytes = (
            psutil.virtual_memory().available - _HOST_MEM_HEADROOM_BYTES
        )
        if requested_host_bytes > available_host_bytes:
            raise ValueError(
                "Not enough Host memory for L2: requesting "
                f"{requested_host_bytes / 1e9:.2f} GB, available "
                f"{available_host_bytes / 1e9:.2f} GB"
            )
        self.host_storage = HostCacheStorage(
            self.layout,
            num_host_lcm_blocks=host_lcm_blocks,
        )
        # The scheduler wire includes logical null LCMBlock 0 in its count.
        self.num_host_pages = host_lcm_blocks + 1
        logger.info(
            "Allocated %.2f GB compact Host L2 (%s LCM blocks, %s bytes/block)",
            requested_host_bytes / 1e9,
            host_lcm_blocks,
            host_lcm_block_bytes,
        )

        pool_layouts = [(device_pool, target_layout)]
        if draft_pool is not None:
            pool_layouts.append((draft_pool, draft_layout))
        self._load_trackers = []
        for pool, layout in pool_layouts:
            tracker = LayerwiseLoadTracker(len(layout.consumers))
            pool.register_layerwise_load_tracker(tracker)
            self._load_trackers.append((tracker, len(layout.consumers)))
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
        self._load_index_by_op_id: OrderedDict[int, int] = OrderedDict()
        self._load_index_history_limit = 1024

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
            raise TypeError(f"unsupported cache op {type(operation).__name__}")

    @staticmethod
    def _submit(
        op_ids: Sequence[int],
        group_ids: Sequence[Sequence[int]],
        src_blocks: Sequence[Sequence[int]],
        dst_blocks: Sequence[Sequence[int]],
        *,
        pending_op_ids: list[int],
        pending_transfers: list[tuple[int, int, int]],
        source_is_device: bool,
    ) -> None:
        if not (len(op_ids) == len(group_ids) == len(src_blocks) == len(dst_blocks)):
            raise ValueError("ragged cache operation batch")
        for op_id, groups, sources, destinations in zip(
            op_ids, group_ids, src_blocks, dst_blocks
        ):
            if not (len(groups) == len(sources) == len(destinations)):
                raise ValueError(f"ragged cache operation {op_id}")
            pending_op_ids.append(int(op_id))
            for group, source, destination in zip(groups, sources, destinations):
                device_block_id, host_block_id = (
                    (source, destination) if source_is_device else (destination, source)
                )
                pending_transfers.append(
                    (int(group), int(device_block_id), int(host_block_id))
                )

    def _transfer_ranges(
        self,
        transfers: Sequence[tuple[int, int, int]],
        field_ids: set[str] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        ranges = []
        for group_index, device_block_id, host_block_id in transfers:
            group = self.layout.groups[group_index]
            for field_index, field in enumerate(group.fields):
                if field_ids is not None and field.field_id not in field_ids:
                    continue
                ranges.append(
                    (
                        field.device_buffer_index,
                        field.device_block_zero_offset_bytes
                        + device_block_id * field.block_stride_bytes,
                        self.host_storage.host_field_offset(
                            group_index, host_block_id, field_index
                        ),
                        field.payload_bytes,
                    )
                )
        return ranges

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
        transfer_cache_ranges(
            "d2h",
            self.layout.buffers,
            self.host_storage.host_buffer,
            self._transfer_ranges(transfers),
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

        # EventLoop zeroes freshly allocated Device blocks before submitting the
        # load. Recording the start event here makes every layer-wise H2D
        # copy wait for that zeroing; the per-layer load events then keep model
        # consumers from reading a partially restored snapshot.
        load_index = None
        consumer_offset = 0
        finish = None
        for tracker, consumer_count in self._load_trackers:
            current_load_index = tracker.begin_load()
            if load_index is None:
                load_index = current_load_index
            elif current_load_index != load_index:
                raise RuntimeError("target and draft Host-load trackers diverged")
            load_events = tracker.event_sets[current_load_index]
            load_events.start_event.record()
            load_events.start_event.wait(self.load_stream)
            for layer_index in range(consumer_count):
                consumer = self.layout.consumers[consumer_offset + layer_index]
                transfer_cache_ranges(
                    "h2d",
                    self.layout.buffers,
                    self.host_storage.host_buffer,
                    self._transfer_ranges(transfers, set(consumer)),
                    self.load_stream,
                )
                finish = torch.cuda.Event()
                finish.record(self.load_stream)
                load_events.layer_done_events[layer_index] = finish
            consumer_offset += consumer_count
        if load_index is None or finish is None:
            raise RuntimeError("cache transfer layout has no layer consumers")
        self.ack_load_queue.append(_Ack(finish, op_ids))
        for op_id in op_ids:
            self._load_index_by_op_id[op_id] = load_index
        while len(self._load_index_by_op_id) > self._load_index_history_limit:
            self._load_index_by_op_id.popitem(last=False)

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

    def take_load_index(self, op_id: int):
        return self._load_index_by_op_id.pop(int(op_id), None)

    def set_consumers(self, load_indices) -> None:
        for tracker, _ in self._load_trackers:
            tracker.set_consumers(load_indices)

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
        self._load_index_by_op_id.clear()
        for tracker, _ in self._load_trackers:
            tracker.reset()
