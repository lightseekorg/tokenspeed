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
from tokenspeed_kernel.ops.kvcache.host_transfer import (
    HostTransferWorkspace,
    build_host_transfer_geometry,
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


class _WriteLane:
    """Staging for one kind of write-back submission.

    Each lane owns its transfer workspace and the event guarding that
    workspace's pinned metadata staging, so the two lanes a round may submit
    (stream-ordered, then pinned) never wait on each other's staging.
    """

    __slots__ = ("metadata_done", "workspace")

    def __init__(self) -> None:
        self.workspace = HostTransferWorkspace()
        self.metadata_done = None


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
        # Every copy runs on its own stream, ordered after the prerequisite
        # stream the caller names per submission: for a write-back the one
        # the forwards wrote the source pages on, for a load the one that
        # zeroed the destination pages. What differs per write-back op is who
        # waits on the copy: a stream-ordered op (a retraction's snapshot,
        # whose sources this very plan may re-grant) fences the fence stream
        # the caller names on its completion, so the plan's zeroing,
        # load-backs and forwards stay behind it; a pinned op (an ordinary
        # publication, whose sources the scheduler holds until the ACK) fences
        # nothing and never holds up the round. A load's consumers are fenced
        # per layer by the tracker events.
        self.write_stream = _new_cache_stream(None)
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
            # Both the write stream (D2H) and load stream (H2D) consume this
            # immutable table, so publish it synchronously once at init.
            geometry = geometry.bind(device, non_blocking=False)
        self._transfer_geometry = geometry
        self._ordered_write_lane = _WriteLane()
        self._pinned_write_lane = _WriteLane()
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
        self._load_poisoned = False

    def submit_write_backs(self, plan, *, prerequisite_stream, fence_stream) -> None:
        """Enqueue the plan's D2H copies on the write stream.

        Must run BEFORE the plan's page zeroing. Every copy is ordered behind
        ``prerequisite_stream`` -- here the stream the forwards wrote the
        source pages on -- so it reads their final bytes. The scheduler marks
        each op ``source_pinned``: a pinned op's sources stay cached and
        unevictable until the ACK, so its copy rides the write stream and
        nobody waits on it; an unpinned op's sources may already be granted to
        another request in this very plan, so it goes first and
        ``fence_stream`` waits on its completion -- the plan's zeroing,
        load-backs and forwards are ordered behind that wait.

        Args:
            plan: The round's ExecutionPlan; its ``Cache.WriteBackOp``
                entries are read here.
            prerequisite_stream: The stream whose completed work every copy
                must observe -- the model executor's execution stream, where
                the forwards wrote the source pages.
            fence_stream: The stream a stream-ordered op's completion fences
                -- the one the plan's page zeroing runs on next.
        """
        ordered_op_ids: list[int] = []
        ordered_transfers: list[tuple[int, int, int]] = []
        pinned_op_ids: list[int] = []
        pinned_transfers: list[tuple[int, int, int]] = []
        for operation in plan.cache:
            if isinstance(operation, Cache.WriteBackOp):
                self._append_write_backs(
                    operation,
                    ordered_op_ids=ordered_op_ids,
                    ordered_transfers=ordered_transfers,
                    pinned_op_ids=pinned_op_ids,
                    pinned_transfers=pinned_transfers,
                )
        fence = self._start_writing(
            ordered_op_ids,
            ordered_transfers,
            lane=self._ordered_write_lane,
            prerequisite_stream=prerequisite_stream,
        )
        if fence is not None:
            fence_stream.wait_event(fence)
        self._start_writing(
            pinned_op_ids,
            pinned_transfers,
            lane=self._pinned_write_lane,
            prerequisite_stream=prerequisite_stream,
        )

    def submit_load_backs(self, plan, *, prerequisite_stream) -> None:
        """Launch the plan's H2D loads; runs after the plan's page zeroing.

        Args:
            plan: The round's ExecutionPlan; its ``Cache.LoadBackOp``
                entries are read here.
            prerequisite_stream: The stream whose completed work every copy
                must observe -- the one the plan's page zeroing ran on, so the
                loads land on zeroed destination pages.
        """
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
        load_index = self._start_loading(
            op_ids, transfers, prerequisite_stream=prerequisite_stream
        )
        for tracker, _ in self._load_trackers:
            tracker.set_consumers(load_index if load_index is not None else -1)

    @classmethod
    def _append_write_backs(
        cls,
        operation,
        *,
        ordered_op_ids: list[int],
        ordered_transfers: list[tuple[int, int, int]],
        pinned_op_ids: list[int],
        pinned_transfers: list[tuple[int, int, int]],
    ) -> None:
        source_pinned = operation.source_pinned
        if len(source_pinned) != len(operation.op_ids):
            raise ValueError("ragged cache operation batch")
        for index, pinned in enumerate(source_pinned):
            op_ids, transfers = (
                (pinned_op_ids, pinned_transfers)
                if pinned
                else (ordered_op_ids, ordered_transfers)
            )
            cls._append_transfers(
                operation.op_ids[index : index + 1],
                operation.group_ids[index : index + 1],
                operation.src_pages[index : index + 1],
                operation.dst_pages[index : index + 1],
                collected_op_ids=op_ids,
                transfers=transfers,
                source_is_device=True,
            )

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
            # An op is acknowledged by its copy's completion event; one with
            # nothing to copy could never be acknowledged and the scheduler
            # would hold its tickets forever.
            if not groups:
                raise ValueError(f"cache operation {op_id} carries no transfers")
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
        *,
        lane: _WriteLane,
        prerequisite_stream,
    ):
        """Launch one D2H batch on the write stream; return its completion event.

        Returns None when the lane has no ops this round.
        """
        if not op_ids:
            return None
        op_ids = _ordered_unique(op_ids)
        if self.attn_tp_rank == 0:
            logger.info(
                "[L2] writeback started: operations=%d blocks=%d pinned=%s",
                len(op_ids),
                len(transfers),
                lane is self._pinned_write_lane,
            )
        # Behind the forwards that wrote the source pages: that is what lets
        # the copy read their final bytes.
        self.write_stream.wait_stream(prerequisite_stream)
        # CPU writes are not ordered by stream FIFO. Retire the previous
        # metadata upload before refilling its pinned source, not at submit.
        if lane.metadata_done is not None and not lane.metadata_done.query():
            lane.metadata_done.synchronize()
        num_blocks, _ = lane.workspace.load_block_transfers(
            transfers, geometry=self._transfer_geometry
        )
        # Address-table allocation and the metadata H2D must be enqueued on
        # the write stream itself: the payload kernel below reads those tables
        # from that stream, and a copy issued on the caller's stream would sit
        # behind the wait recorded above with nothing ordering it first.
        with device_module.stream(self.write_stream):
            mode = lane.workspace.prepare_backend(
                self.layout.buffers,
                self.host_storage.host_buffer,
                backend=self.transfer_backend,
            )
            if mode.uses_device_tables:
                if lane.metadata_done is None:
                    lane.metadata_done = device_module.Event()
                try:
                    lane.workspace.commit_block_transfers(
                        num_blocks, self.layout.buffers[0].device, non_blocking=True
                    )
                finally:
                    # Also protect a partially submitted upload if staging
                    # fails. This event excludes the payload transfer; Device
                    # table reuse remains ordered by the write stream's FIFO.
                    lane.metadata_done.record(self.write_stream)
        transfer_cache_blocks(
            "d2h",
            self.layout.buffers,
            self.host_storage.host_buffer,
            self._transfer_geometry,
            lane.workspace,
            self.write_stream,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=self._transfer_geometry.num_field_rows,
            backend=self.transfer_backend,
            grid_cap=None,
            layer_ready_flags=None,
        )
        finish = device_module.Event()
        finish.record(self.write_stream)
        with self._ack_lock:
            self._write_acks.append(_Ack(finish, op_ids))
        return finish

    def _start_loading(
        self,
        op_ids: Sequence[int],
        transfers: Sequence[tuple[int, int, int]],
        *,
        prerequisite_stream,
    ) -> int | None:
        if self._load_poisoned:
            raise RuntimeError(
                "L2 cache executor is poisoned after failed Host-load retirement"
            )
        if not op_ids:
            return None
        if get_is_capture_mode():
            raise RuntimeError("Host cache load must run outside CUDA Graph capture")
        op_ids = _ordered_unique(op_ids)
        if self.attn_tp_rank == 0:
            logger.info(
                "[L2] load started: operations=%d blocks=%d",
                len(op_ids),
                len(transfers),
            )

        # EventLoop zeroes freshly allocated Device blocks on the prerequisite
        # stream before submitting the load. Recording the start event there
        # makes the H2D copy wait for that zeroing; per-layer ready flags
        # (Triton) or events (DMA) then keep model consumers from reading
        # partially restored cache state.
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
                load_events.start_event.record(prerequisite_stream)
                load_events.start_event.wait(self.load_stream)
            if load_index is None:
                raise RuntimeError("cache transfer layout has no layer consumers")

            device = self.layout.buffers[0].device
            workspace = self._load_workspaces[load_index]
            num_blocks, _ = workspace.load_block_transfers(
                transfers, geometry=self._transfer_geometry
            )
            layer_slices = self._transfer_geometry.layer_slices
            # Resolve the transport before choosing the consumer wait protocol.
            with device_module.stream(self.load_stream):
                mode = workspace.prepare_backend(
                    self.layout.buffers,
                    self.host_storage.host_buffer,
                    backend=self.transfer_backend,
                )
                if mode.uses_device_tables:
                    workspace.commit_block_transfers(
                        num_blocks,
                        device,
                        non_blocking=True,
                    )
                if mode.layer_ready:
                    flags = workspace.prepare_layer_ready(len(layer_slices), device)
                    for load_events, _ in active_trackers:
                        load_events.layer_ready_init_event.record(self.load_stream)
            if mode.layer_ready:
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
                    num_geometry_rows=self._transfer_geometry.num_field_rows,
                    backend=self.transfer_backend,
                    layer_ready_flags=flags,
                    grid_cap=None,
                )
                finish = device_module.Event()
                finish.record(self.load_stream)
                for load_events, consumer_count in active_trackers:
                    load_events.set_completion(finish)
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
                            grid_cap=None,
                            layer_ready_flags=None,
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
            self._retire_failed_load(active_trackers, flags, original_error)
            raise

    def _retire_failed_load(self, active_trackers, flags, original_error) -> None:
        """Retire submitted GPU readers without publishing a success ACK."""
        if not active_trackers:
            return
        try:
            if flags is not None:
                with device_module.stream(self.load_stream):
                    flags.fill_(1)
            retirement = device_module.Event()
            retirement.record(self.load_stream)
            for load_events, _ in active_trackers:
                load_events.set_completion(retirement)
            return
        except BaseException as retirement_error:
            # If event publication fails, only stream completion permits reuse.
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

    def poll_results(self) -> list:
        results: list = []
        with self._ack_lock:
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
        # The fences and start events live on streams the callers named per
        # submission; the whole device covers them and the transfer streams.
        device_module.synchronize()

    def reset(self) -> None:
        self.shutdown()
        self._write_acks.clear()
        self._load_acks.clear()
        for tracker, _ in self._load_trackers:
            tracker.reset()
