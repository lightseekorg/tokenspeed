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
    build_host_transfer_geometry,
    layer_ready_ptx_supported,
    transfer_cache_blocks,
    wait_layer_ready,
)
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
        device = self.layout.buffers[0].device
        fields_by_id = {}
        for group_index, group in enumerate(self.layout.groups):
            for field_index, field in enumerate(group.fields):
                if field.field_id in fields_by_id:
                    raise ValueError(
                        f"cache transfer field {field.field_id!r} appears twice"
                    )
                fields_by_id[field.field_id] = (
                    group_index,
                    field_index,
                    group,
                    field,
                )

        rows = []
        layer_slices = []
        consumed_fields = set()
        for consumer in self.layout.consumers:
            layer_offset = len(rows)
            for field_id in consumer:
                if field_id in consumed_fields:
                    raise ValueError(
                        f"cache transfer field {field_id!r} has two consumers"
                    )
                try:
                    group_index, field_index, group, field = fields_by_id[field_id]
                except KeyError as exc:
                    raise ValueError(
                        f"cache consumer references unknown field {field_id!r}"
                    ) from exc
                consumed_fields.add(field_id)
                rows.append(
                    (
                        group_index,
                        field.device_buffer_index,
                        field.device_block_zero_offset_bytes,
                        field.block_stride_bytes,
                        self.host_storage.host_cache_block_bytes[group_index],
                        self.host_storage.host_field_offsets[group_index][field_index],
                        group.cache_blocks_per_lcm_block,
                        field.payload_bytes,
                    )
                )
            layer_slices.append((layer_offset, len(rows) - layer_offset))
        missing_fields = set(fields_by_id) - consumed_fields
        if missing_fields:
            raise ValueError(
                f"cache transfer fields have no consumer {sorted(missing_fields)}"
            )

        geometry = build_host_transfer_geometry(
            rows=tuple(rows),
            layer_slices=tuple(layer_slices),
            group_packing=tuple(
                group.cache_blocks_per_lcm_block for group in self.layout.groups
            ),
            host_lcm_block_bytes=self.host_storage.host_lcm_block_bytes,
            num_host_lcm_blocks=self.host_storage.num_host_lcm_blocks,
            num_device_lcm_blocks=self.layout.num_lcm_blocks,
            num_device_buffers=len(self.layout.buffers),
        )
        if io_backend == "kernel" and device.type != "npu":
            # Both the caller stream (D2H) and load stream (H2D) consume this
            # immutable table, so publish it synchronously once at init.
            geometry = geometry.bind(device)
        self._transfer_geometry = geometry
        self._write_workspace = HostTransferWorkspace()
        # A tracker waits for an event set's previous final-layer event before
        # reusing its index. Aligning workspaces to those indices keeps each
        # load's pinned and Device block-ID tables immutable until all
        # consumers of that table have completed.
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
        self._load_poisoned = False

    def submit_write_backs(self, plan) -> None:
        """Enqueue the plan's D2H snapshot copies on the current stream.

        Must run BEFORE the plan's page zeroing: the scheduler may have
        granted a snapshot's source pages to another request in the same
        plan, and only the stream order keeps the copy reading the old bytes.
        """
        op_ids: list[int] = []
        transfers: list[tuple[int, int, int]] = []
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
        self._start_writing(op_ids, transfers)

    def submit_load_backs(self, plan) -> None:
        """Launch the plan's H2D loads; runs after the plan's page zeroing."""
        op_ids: list[int] = []
        transfers: list[tuple[int, int, int]] = []
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

    def _start_writing(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
    ) -> None:
        if not op_ids:
            return
        op_ids = _ordered_unique(op_ids)
        if not transfers:
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
        num_blocks, _ = self._write_workspace.load_block_transfers(
            transfers, geometry=self._transfer_geometry
        )
        if self._transfer_geometry.device_rows is not None:
            # The single write workspace can be refilled by the next plan as
            # soon as this method returns, so finish staging before releasing
            # the caller thread. Device-table reuse remains ordered by stream.
            self._write_workspace.commit_block_transfers(
                num_blocks, self.layout.buffers[0].device
            )
        transfer_cache_blocks(
            "d2h",
            self.layout.buffers,
            self.host_storage.host_buffer,
            self._transfer_geometry,
            self._write_workspace,
            stream,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=self._transfer_geometry.num_field_rows,
            backend=self.transfer_backend,
        )
        finish = device_module.Event()
        finish.record(stream)
        with self._ack_lock:
            self._write_acks.append(_Ack(finish, op_ids))

    def _start_loading(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
    ) -> int | None:
        if getattr(self, "_load_poisoned", False):
            raise RuntimeError(
                "L2 cache executor is poisoned after failed Host-load retirement"
            )
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
        # load. Recording the start event here makes the H2D copy wait for that
        # zeroing; per-layer ready flags (Triton) or events (DMA) then keep
        # model consumers from reading partially restored cache state.
        load_index = None
        finish = None
        flags = None
        active_trackers = []
        try:
            for tracker, consumer_count in self._load_trackers:
                current_load_index = tracker.begin_load()
                load_events = tracker.event_sets[current_load_index]
                # Register the generation immediately after begin_load so an
                # exception in tracker convergence or start-event setup still
                # retires every target/draft event set that advanced.
                active_trackers.append((load_events, consumer_count))
                if load_index is None:
                    load_index = current_load_index
                elif current_load_index != load_index:
                    raise RuntimeError("target and draft Host-load trackers diverged")
                load_events.start_event.record()
                load_events.start_event.wait(self.load_stream)
            if load_index is None:
                raise RuntimeError("cache transfer layout has no layer consumers")

            device = self.layout.buffers[0].device
            workspace = self._load_workspaces[load_index]
            num_blocks, _ = workspace.load_block_transfers(
                transfers, geometry=self._transfer_geometry
            )
            layer_slices = self._transfer_geometry.layer_slices
            num_field_rows = sum(num_rows for _, num_rows in layer_slices)
            use_layer_flags = (
                self._transfer_geometry.device_rows is not None
                and layer_ready_ptx_supported()
            )
            # All layer kernels below read slices of this one table. The
            # indexed workspace is not refilled until this event set wraps.
            # Commit whenever Device geometry exists so AMD can still launch
            # unflagged per-layer Triton copies; only NVIDIA uses PTX flags.
            if self._transfer_geometry.device_rows is not None:
                with device_module.stream(self.load_stream):
                    workspace.commit_block_transfers(
                        num_blocks,
                        device,
                        non_blocking=True,
                    )
                    if use_layer_flags:
                        flags = workspace.prepare_layer_ready(len(layer_slices), device)
                        for load_events, _ in active_trackers:
                            load_events.layer_ready_init_event.record(self.load_stream)
            if use_layer_flags:
                flag_offset = 0
                for load_events, consumer_count in active_trackers:
                    load_events.layer_ready_flags = flags[
                        flag_offset : flag_offset + consumer_count
                    ]
                    load_events.wait_layer_ready = wait_layer_ready
                    flag_offset += consumer_count
                transfer_cache_blocks(
                    "h2d",
                    self.layout.buffers,
                    self.host_storage.host_buffer,
                    self._transfer_geometry,
                    workspace,
                    self.load_stream,
                    num_blocks=num_blocks,
                    geometry_offset=0,
                    num_geometry_rows=num_field_rows,
                    backend=self.transfer_backend,
                    layer_ready_flags=flags,
                )
                finish = device_module.Event()
                finish.record(self.load_stream)
                for load_events, consumer_count in active_trackers:
                    load_events.layer_done_events[:] = [finish] * consumer_count
            else:
                for load_events, _ in active_trackers:
                    load_events.layer_ready_flags = None
                    load_events.wait_layer_ready = None
                flat_layer_index = 0
                for load_events, consumer_count in active_trackers:
                    for layer_index in range(consumer_count):
                        geometry_offset, num_geometry_rows = layer_slices[
                            flat_layer_index
                        ]
                        transfer_cache_blocks(
                            "h2d",
                            self.layout.buffers,
                            self.host_storage.host_buffer,
                            self._transfer_geometry,
                            workspace,
                            self.load_stream,
                            num_blocks=num_blocks,
                            geometry_offset=geometry_offset,
                            num_geometry_rows=num_geometry_rows,
                            backend=self.transfer_backend,
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
        except BaseException as original_error:
            if active_trackers:
                try:
                    if flags is not None:
                        with device_module.stream(self.load_stream):
                            flags.fill_(1)
                    retirement = device_module.Event()
                    retirement.record(self.load_stream)
                    for load_events, _ in active_trackers:
                        load_events.layer_done_events[:] = [retirement] * len(
                            load_events.layer_done_events
                        )
                except BaseException as retirement_error:
                    # If an Event cannot be reliably published, stream
                    # completion is the fallback workspace-retirement fence.
                    try:
                        self.load_stream.synchronize()
                    except BaseException as sync_error:
                        self._load_poisoned = True
                        add_note = getattr(original_error, "add_note", None)
                        if add_note is not None:
                            try:
                                add_note(
                                    "Host-load retirement failed; executor poisoned: "
                                    f"retirement error={retirement_error!r}; "
                                    f"synchronize error={sync_error!r}"
                                )
                            except BaseException:
                                pass
            raise

    def poll_results(self) -> list:
        with self._ack_lock:
            results = [self._write_done(op_id) for op_id in self._ready_write_op_ids]
            self._ready_write_op_ids.clear()
            results.extend(self._load_done(op_id) for op_id in self._ready_load_op_ids)
            self._ready_load_op_ids.clear()
            self._write_acks[:] = self._drain(
                self._write_acks, self._write_done, results
            )
            self._load_acks[:] = self._drain(self._load_acks, self._load_done, results)
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

    def reset(self) -> None:
        self.shutdown()
        self._write_acks.clear()
        self._load_acks.clear()
        self._ready_write_op_ids.clear()
        self._ready_load_op_ids.clear()
        for tracker, _ in self._load_trackers:
            tracker.reset()
