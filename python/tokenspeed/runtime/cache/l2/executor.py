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

import threading
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import psutil
import torch
from tokenspeed_kernel.ops.kvcache.host_transfer import (
    HostTransferWorkspace,
    transfer_cache_ranges,
)
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.l2.storage import (
    HostCacheStorage,
    compute_host_lcm_block_bytes,
)
from tokenspeed.runtime.cache.l3.executor import L3HostStore, StoragePage
from tokenspeed.runtime.cache.transfer.layout import combine_cache_transfer_layouts
from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()

_HOST_MEM_HEADROOM_BYTES = 10 * (1024**3)


def _load_stream_priority() -> int | None:
    priority_range = getattr(device_module.Stream, "priority_range", None)
    if priority_range is None:
        return None
    try:
        _, load_priority = priority_range()
    except (RuntimeError, TypeError):
        return None
    return load_priority


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
    backup_pages: list[StoragePage] = []


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

    def __init__(
        self,
        device_pool,
        *,
        draft_pool=None,
        host_ratio: float,
        host_size_gb: float,
        io_backend: str,
        attn_tp_rank: int = 0,
        storage_backend=None,
        storage_key_prefix: str = "",
        storage_rank: int = 0,
    ):
        if io_backend not in ("direct", "kernel"):
            raise ValueError(f"unsupported KVStore IO backend {io_backend!r}")
        self.attn_tp_rank = attn_tp_rank
        self.transfer_backend = "dma" if io_backend == "direct" else "auto"
        target_layout = device_pool.cache_transfer_layout()
        draft_layout = (
            draft_pool.cache_transfer_layout() if draft_pool is not None else None
        )
        scheduler_group_ids = tuple(
            spec.group_id for spec in device_pool.arena.cache_group_specs
        )
        self.layout = combine_cache_transfer_layouts(
            target_layout,
            draft_layout,
            group_ids=scheduler_group_ids or None,
        )
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
        self.l3_store = None
        if storage_backend is not None:
            self.attach_l3_storage(
                storage_backend,
                key_prefix=storage_key_prefix,
                rank=storage_rank,
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
        if draft_pool is not None and self.layout is not target_layout:
            pool_layouts.append((draft_pool, draft_layout))
        self._load_trackers = []
        for pool, layout in pool_layouts:
            tracker = LayerwiseLoadTracker(len(layout.consumers))
            pool.register_layerwise_load_tracker(tracker)
            self._load_trackers.append((tracker, len(layout.consumers)))
        # Write-backs run on the CALLER's current stream -- the forward
        # thread's default stream, the same one that carries the plan's page
        # zeroing and fences its forwards. That single-stream FIFO is the
        # correctness story for retraction: the D2H snapshot copy reads the
        # victim's pages after the victim's last forward wrote them and
        # before anything later in the plan (zeroing, the granted request's
        # work) can touch them. Loads keep their own stream: their consumers
        # are fenced per layer by the tracker events.
        self.load_stream = _new_cache_stream(_load_stream_priority())
        self._write_workspace = HostTransferWorkspace()
        # A tracker waits for an event set's previous final-layer event before
        # reusing its index. Aligning workspaces to those indices keeps each
        # load's pinned and Device range tables immutable until all consumers
        # of that table have completed.
        load_workspace_count = len(self._load_trackers[0][0].event_sets)
        if any(
            len(tracker.event_sets) != load_workspace_count
            for tracker, _ in self._load_trackers
        ):
            raise RuntimeError("target and draft Host-load event sets diverged")
        self._load_workspaces = tuple(
            HostTransferWorkspace() for _ in range(load_workspace_count)
        )

        # Submission runs on the forward thread and polling on the control
        # plane (event queries only), so the completion queues below are the
        # cross-thread handoff; the lock covers every mutation of them.
        self._ack_lock = threading.Lock()
        self._write_acks: list[_Ack] = []
        self._load_acks: list[_Ack] = []
        self._ready_write_op_ids: list[int] = []
        self._ready_load_op_ids: list[int] = []

    def attach_l3_storage(
        self,
        storage_backend,
        *,
        key_prefix: str = "",
        rank: int = 0,
    ) -> None:
        """Bind an L3 backend to the compact Host buffer after allocation.

        Mooncake Store must ``register_buffer`` against the pinned Host L2
        allocation, so the backend is constructed after ``HostCacheStorage``.
        """

        if self.l3_store is not None:
            raise RuntimeError("L3 storage backend is already attached")
        if storage_backend is None:
            raise ValueError("storage_backend is required")
        self.l3_store = L3HostStore(
            storage_backend,
            self.host_storage,
            key_prefix=key_prefix,
            rank=rank,
        )

    def submit_write_backs(self, plan) -> None:
        """Enqueue the plan's D2H snapshot copies on the current stream.

        Must run BEFORE the plan's page zeroing: the scheduler may have
        granted a snapshot's source pages to another request in the same
        plan, and only the stream order keeps the copy reading the old bytes.
        """
        op_ids: list[int] = []
        transfers: list[tuple[int, int, int]] = []
        write_pages: list[StoragePage] = []
        for operation in plan.cache:
            if isinstance(operation, Cache.WriteBackOp):
                self._append_transfers(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    collected_op_ids=op_ids,
                    transfers=transfers,
                    source_is_device=True,
                )
                write_pages.extend(
                    self._storage_pages(operation, host_is_destination=True)
                )
        self._start_writing(op_ids, transfers, write_pages)

    def submit_load_backs(self, plan) -> None:
        """Launch the plan's H2D loads; runs after the plan's page zeroing."""
        op_ids: list[int] = []
        transfers: list[tuple[int, int, int]] = []
        prefetch_pages: list[StoragePage] = []
        for operation in plan.cache:
            if isinstance(operation, Cache.LoadBackOp):
                self._append_transfers(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    collected_op_ids=op_ids,
                    transfers=transfers,
                    source_is_device=False,
                )
                prefetch_pages.extend(
                    self._storage_pages(
                        operation,
                        host_is_destination=False,
                        prefetch_only=True,
                    )
                )
        if prefetch_pages:
            self._prefetch_from_storage(prefetch_pages)
        load_index = self._start_loading(op_ids, transfers)
        for tracker, _ in self._load_trackers:
            tracker.set_consumers(load_index if load_index is not None else -1)

    @staticmethod
    def _append_transfers(
        operation_ids: Sequence[int],
        group_ids: Sequence[Sequence[int]],
        src_blocks: Sequence[Sequence[int]],
        dst_blocks: Sequence[Sequence[int]],
        *,
        collected_op_ids: list[int],
        transfers: list[tuple[int, int, int]],
        source_is_device: bool,
    ) -> None:
        if not (
            len(operation_ids) == len(group_ids) == len(src_blocks) == len(dst_blocks)
        ):
            raise ValueError("ragged cache operation batch")
        for op_id, groups, sources, destinations in zip(
            operation_ids, group_ids, src_blocks, dst_blocks
        ):
            if not (len(groups) == len(sources) == len(destinations)):
                raise ValueError(f"ragged cache operation {op_id}")
            collected_op_ids.append(int(op_id))
            for group, source, destination in zip(groups, sources, destinations):
                device_block_id, host_block_id = (
                    (source, destination) if source_is_device else (destination, source)
                )
                transfers.append((int(group), int(device_block_id), int(host_block_id)))

    @staticmethod
    def _storage_pages(
        operation,
        *,
        host_is_destination: bool,
        prefetch_only: bool = False,
    ) -> list[StoragePage]:
        hashes = getattr(operation, "content_hashes", None)
        offsets = getattr(operation, "page_offsets", None)
        if not hashes or not offsets:
            return []
        host_pages = operation.dst_pages if host_is_destination else operation.src_pages
        prefetch_flags = getattr(operation, "prefetch_from_storage", None)
        pages: list[StoragePage] = []
        for groups, hosts, hash_row, offset_row, flags in zip(
            operation.group_ids,
            host_pages,
            hashes,
            offsets,
            prefetch_flags or [None] * len(hashes),
        ):
            flag_row = flags if flags is not None else [1] * len(groups)
            for group, host_page, content_hash, page_offset, flag in zip(
                groups, hosts, hash_row, offset_row, flag_row
            ):
                if prefetch_only and int(flag) == 0:
                    continue
                if not content_hash:
                    continue
                pages.append(
                    (int(group), int(host_page), str(content_hash), int(page_offset))
                )
        return pages

    def _prefetch_from_storage(self, pages: Sequence[StoragePage]) -> None:
        l3_store = getattr(self, "l3_store", None)
        if l3_store is None:
            raise RuntimeError(
                "LoadBack requested L3 prefetch but no storage backend is configured"
            )
        results = l3_store.prefetch(pages)
        if not all(results):
            missed = [page for page, ok in zip(pages, results) if not ok]
            raise RuntimeError(f"L3 prefetch failed for {len(missed)} Host page(s)")

    def l3_exists(self, pages: Sequence[StoragePage]) -> list[bool] | None:
        l3_store = getattr(self, "l3_store", None)
        if l3_store is None:
            return None
        return l3_store.exists(pages)

    def rotate_l3_namespace(self) -> None:
        l3_store = getattr(self, "l3_store", None)
        if l3_store is not None:
            l3_store.rotate_namespace()

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

    def _fill_workspace_ranges(
        self,
        workspace: HostTransferWorkspace,
        transfers: Sequence[tuple[int, int, int]],
        field_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        ranges = self._transfer_ranges(transfers, field_ids)
        if not ranges:
            return 0, 0
        return workspace.load_ranges(ranges)

    def _start_writing(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        backup_pages: Sequence[StoragePage] | None = None,
    ) -> None:
        if not op_ids:
            return
        op_ids = _ordered_unique(op_ids)
        backup_pages = list(backup_pages or ())
        if not transfers:
            self._backup_to_storage(backup_pages)
            with self._ack_lock:
                self._ready_write_op_ids.extend(op_ids)
            return
        if self.attn_tp_rank == 0:
            logger.info(
                "[L2] writeback started: operations=%d blocks=%d",
                len(op_ids),
                len(transfers),
            )
        # On the caller's (forward thread's default) stream: the scheduler
        # releases -- and may re-grant -- the source pages the moment it
        # emits this op, and the single-stream FIFO is what keeps the copy
        # ahead of the pages' next writer.
        stream = device_module.current_stream()
        num_ranges, max_bytes = self._fill_workspace_ranges(
            self._write_workspace, transfers
        )
        transfer_cache_ranges(
            "d2h",
            self.layout.buffers,
            self.host_storage.host_buffer,
            (),
            stream,
            backend=self.transfer_backend,
            workspace=self._write_workspace,
            num_ranges=num_ranges,
            max_bytes=max_bytes,
        )
        finish = device_module.Event()
        finish.record(stream)
        with self._ack_lock:
            self._write_acks.append(_Ack(finish, op_ids, backup_pages))

    def _start_loading(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
    ) -> int | None:
        if not op_ids:
            return None
        if get_is_capture_mode():
            raise RuntimeError("Host cache load must run outside CUDA Graph capture")
        op_ids = _ordered_unique(op_ids)
        if not transfers:
            with self._ack_lock:
                self._ready_load_op_ids.extend(op_ids)
            return None
        if self.attn_tp_rank == 0:
            logger.info(
                "[L2] load started: operations=%d blocks=%d",
                len(op_ids),
                len(transfers),
            )

        # EventLoop zeroes freshly allocated Device blocks before submitting the
        # load. Recording the start event here makes every layer-wise H2D
        # copy wait for that zeroing; the per-layer load events then keep model
        # consumers from reading partially restored cache state.
        load_index = None
        finish = None
        active_trackers = []
        for tracker, consumer_count in self._load_trackers:
            current_load_index = tracker.begin_load()
            if load_index is None:
                load_index = current_load_index
            elif current_load_index != load_index:
                raise RuntimeError("target and draft Host-load trackers diverged")
            load_events = tracker.event_sets[current_load_index]
            load_events.start_event.record()
            load_events.start_event.wait(self.load_stream)
            active_trackers.append((load_events, consumer_count))
        if load_index is None:
            raise RuntimeError("cache transfer layout has no layer consumers")

        layer_ranges = []
        for layer_index in range(sum(count for _, count in active_trackers)):
            consumer = self.layout.consumers[layer_index]
            layer_ranges.append(self._transfer_ranges(transfers, set(consumer)))

        range_descriptors = None
        workspace = None
        device = None
        if self.transfer_backend != "dma":
            device = self.layout.buffers[0].device
        if device is not None and device.type != "npu":
            workspace = self._load_workspaces[load_index]
            range_descriptors = workspace.load_range_batches(layer_ranges)
            total_ranges = sum(descriptor[1] for descriptor in range_descriptors)
            if total_ranges:
                # All layer kernels below read slices of this one table. The
                # indexed workspace is not refilled until this event set wraps.
                with device_module.stream(self.load_stream):
                    workspace.commit_ranges(
                        total_ranges,
                        device,
                        non_blocking=True,
                    )

        flat_layer_index = 0
        for load_events, consumer_count in active_trackers:
            for layer_index in range(consumer_count):
                ranges = layer_ranges[flat_layer_index]
                if range_descriptors is None:
                    transfer_cache_ranges(
                        "h2d",
                        self.layout.buffers,
                        self.host_storage.host_buffer,
                        ranges,
                        self.load_stream,
                        backend=self.transfer_backend,
                    )
                else:
                    range_offset, num_ranges, max_bytes = range_descriptors[
                        flat_layer_index
                    ]
                    transfer_cache_ranges(
                        "h2d",
                        self.layout.buffers,
                        self.host_storage.host_buffer,
                        (),
                        self.load_stream,
                        backend=self.transfer_backend,
                        workspace=workspace,
                        num_ranges=num_ranges,
                        max_bytes=max_bytes,
                        range_offset=range_offset,
                        ranges_committed=True,
                    )
                finish = device_module.Event()
                finish.record(self.load_stream)
                load_events.layer_done_events[layer_index] = finish
                flat_layer_index += 1
        if finish is None:
            raise RuntimeError("cache transfer layout has no layer consumers")
        with self._ack_lock:
            self._load_acks.append(_Ack(finish, op_ids))
        return load_index

    def poll_results(self) -> list:
        with self._ack_lock:
            results = [self._write_done(op_id) for op_id in self._ready_write_op_ids]
            self._ready_write_op_ids.clear()
            results.extend(self._load_done(op_id) for op_id in self._ready_load_op_ids)
            self._ready_load_op_ids.clear()
            self._write_acks[:] = self._drain_writes(self._write_acks, results)
            self._load_acks[:] = self._drain(self._load_acks, self._load_done, results)
        return results

    def _backup_to_storage(self, pages: Sequence[StoragePage]) -> None:
        l3_store = getattr(self, "l3_store", None)
        if not pages or l3_store is None:
            return
        results = l3_store.backup(pages)
        if len(results) != len(pages) or not all(results):
            ok = sum(1 for flag in results if flag)
            raise RuntimeError(
                f"L3 backup failed for Host page(s): ok={ok}/{len(pages)}"
            )

    def _drain_writes(self, queue, results):
        pending = []
        for ack in queue:
            if ack.finish_event.query():
                self._backup_to_storage(ack.backup_pages)
                results.extend(self._write_done(op_id) for op_id in ack.op_ids)
            else:
                pending.append(ack)
        return pending

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
        return event

    @staticmethod
    def _load_done(op_id: int):
        event = Cache.LoadBackDoneEvent()
        event.op_id = op_id
        return event

    def shutdown(self) -> None:
        # Write-backs ride the default stream (shared per device, so this
        # thread's handle reaches them); only loads have their own stream.
        torch.cuda.current_stream().synchronize()
        self.load_stream.synchronize()
        with self._ack_lock:
            pending_writes = list(self._write_acks)
            self._write_acks.clear()
        # Synchronization above makes every D2H snapshot complete. Persist the
        # final batch before closing L3; otherwise a clean process shutdown can
        # acknowledge work in memory and silently lose the remote object.
        for ack in pending_writes:
            self._backup_to_storage(ack.backup_pages)
        if getattr(self, "l3_store", None) is not None:
            self.l3_store.close()

    def reset(self) -> None:
        self.shutdown()
        self._write_acks.clear()
        self._load_acks.clear()
        self._ready_write_op_ids.clear()
        self._ready_load_op_ids.clear()
        for tracker, _ in self._load_trackers:
            tracker.reset()
