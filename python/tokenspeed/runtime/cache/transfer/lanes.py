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

"""Host-transfer machinery under the compact Host cache executor.

``HostCacheExecutor`` (``cache/l2/executor.py``) moves whole CacheBlocks
between the Device arena and two compact pinned Host buffers -- the L2 prefix
tier and the retraction snapshot pool -- with ``transfer_cache_blocks``. The
machinery that is indifferent to which buffer a copy targets lives here: the
transfer streams, the static field geometry derived from a transfer layout
and one Host buffer, the staging lane that owns one workspace's pinned
metadata tables and its retirement event, the D2H/H2D launch discipline, and
the completion queue the control plane polls. Tier-specific behaviour -- L3
backups and layerwise load fences for L2, the slot-state image for the pool
-- stays in the executor.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import psutil
from tokenspeed_kernel.ops.kvcache.host_transfer import (
    HostTransferGeometry,
    HostTransferWorkspace,
    build_host_transfer_geometry,
    transfer_cache_blocks,
)

from tokenspeed.runtime.utils import get_device_module

device_module = get_device_module()


def load_stream_priority() -> int | None:
    """The highest stream priority the device offers, or None when unknown.

    H2D loads run at it so a restore is not starved by the forwards.
    """
    priority_range = getattr(device_module.Stream, "priority_range", None)
    if priority_range is None:
        return None
    try:
        _, load_priority = priority_range()
    except (RuntimeError, TypeError):
        return None
    return load_priority


def new_cache_stream(priority: int | None):
    """A dedicated transfer stream, at ``priority`` when the device supports it."""
    if priority is None:
        return device_module.Stream()
    try:
        return device_module.Stream(priority=priority)
    except (RuntimeError, TypeError):
        return device_module.Stream()


def check_host_memory(
    requested_bytes: int, *, headroom_bytes: int, purpose: str
) -> None:
    """Refuse a pinned allocation that would leave less than ``headroom_bytes``.

    Args:
        requested_bytes: The pinned bytes about to be allocated.
        headroom_bytes: Host memory that must stay available after it.
        purpose: Names the allocation in the error.

    Raises:
        ValueError: The allocation does not fit.
    """
    available_host_bytes = psutil.virtual_memory().available - headroom_bytes
    if requested_bytes > available_host_bytes:
        raise ValueError(
            f"Not enough Host memory for {purpose}: requesting "
            f"{requested_bytes / 1e9:.2f} GB, available "
            f"{available_host_bytes / 1e9:.2f} GB"
        )


def build_transfer_geometry(
    layout, host_storage, *, io_backend: str
) -> HostTransferGeometry:
    """The static field geometry one executor's transfers share.

    One row per field in consumer (layer) order, each naming the field's
    Device buffer and block stride on one side and its packed Host offset on
    the other; the layer slices partition the rows by consumer. The ``kernel``
    backend reads the rows from the Device, so they are published there once,
    synchronously, at init -- both the D2H and the H2D stream consume the same
    immutable table.

    Args:
        layout: The executor's ``CacheTransferLayout`` (target and draft
            fields combined, groups in scheduler order).
        host_storage: The ``HostCacheStorage`` the geometry packs into.
        io_backend: ``"direct"`` (DMA ranges) or ``"kernel"`` (mapped-Host
            Triton copies, Device-bound rows).

    Returns:
        The validated geometry.

    Raises:
        ValueError: A field appears twice, has two consumers or none, or a
            consumer names an unknown field.
    """
    device = layout.buffers[0].device
    fields_by_id = {}
    for group_index, group in enumerate(layout.groups):
        for field_index, field in enumerate(group.fields):
            if field.field_id in fields_by_id:
                raise ValueError(
                    f"cache transfer field {field.field_id!r} appears twice"
                )
            fields_by_id[field.field_id] = (group_index, field_index, group, field)

    rows = []
    layer_slices = []
    consumed_fields = set()
    for consumer in layout.consumers:
        layer_offset = len(rows)
        for field_id in consumer:
            if field_id in consumed_fields:
                raise ValueError(f"cache transfer field {field_id!r} has two consumers")
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
                    host_storage.host_cache_block_bytes[group_index],
                    host_storage.host_field_offsets[group_index][field_index],
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
            group.cache_blocks_per_lcm_block for group in layout.groups
        ),
        host_lcm_block_bytes=host_storage.host_lcm_block_bytes,
        num_host_lcm_blocks=host_storage.num_host_lcm_blocks,
        num_device_lcm_blocks=layout.num_lcm_blocks,
        num_device_buffers=len(layout.buffers),
    )
    if io_backend == "kernel" and device.type != "npu":
        geometry = geometry.bind(device, non_blocking=False)
    return geometry


class HostTransferLane:
    """Staging for one kind of transfer submission.

    A lane owns one transfer workspace and the event guarding that
    workspace's pinned metadata staging. Two submissions that must never wait
    on each other's staging (L2's stream-ordered and pinned write-backs; the
    snapshot executor's stores and restores) take separate lanes.
    """

    __slots__ = ("metadata_done", "workspace")

    def __init__(self) -> None:
        self.workspace = HostTransferWorkspace()
        self.metadata_done = None

    def _stage(
        self,
        transfers: Sequence[tuple[int, int, int]],
        *,
        device_buffers,
        host_buffer,
        geometry: HostTransferGeometry,
        stream,
        backend: str,
    ) -> int:
        """Load this submission's block pairs into the lane's tables.

        Returns the number of block rows staged. CPU writes are not ordered by
        stream FIFO, so the previous metadata upload is retired before its
        pinned source is refilled, not at submit. Address-table allocation and
        the metadata H2D are enqueued on ``stream`` itself: the payload kernel
        reads those tables from that stream, and a copy issued on the caller's
        stream would sit behind the wait recorded before this call with
        nothing ordering it first.
        """
        if self.metadata_done is not None and not self.metadata_done.query():
            self.metadata_done.synchronize()
        num_blocks, _ = self.workspace.load_block_transfers(
            transfers, geometry=geometry
        )
        with device_module.stream(stream):
            mode = self.workspace.prepare_backend(
                device_buffers, host_buffer, backend=backend
            )
            if mode.uses_device_tables:
                if self.metadata_done is None:
                    self.metadata_done = device_module.Event()
                try:
                    self.workspace.commit_block_transfers(
                        num_blocks, device_buffers[0].device, non_blocking=True
                    )
                finally:
                    # Also protect a partially submitted upload if staging
                    # fails. This event excludes the payload transfer; Device
                    # table reuse remains ordered by the stream's FIFO.
                    self.metadata_done.record(stream)
        return num_blocks

    def _start(
        self,
        direction: str,
        transfers: Sequence[tuple[int, int, int]],
        *,
        device_buffers,
        host_buffer,
        geometry: HostTransferGeometry,
        stream,
        prerequisite_stream,
        backend: str,
    ):
        """Launch one whole-geometry batch on ``stream``; return its completion event.

        No layer-ready flags in either direction: a D2H's sources are read by
        nobody until the ACK, and an H2D's destinations are read by nothing in
        flight (the snapshot restore's request is not schedulable until the
        ACK). L2's layerwise loads keep their own path.

        Args:
            direction: ``"d2h"`` or ``"h2d"``.
            transfers: ``(group_index, device_block_id, host_block_id)`` rows,
                1-based local block ids. Empty when every row of the op
                belongs to other KVP ranks: the op then completes as an empty
                copy -- nothing is staged and the transport is never asked for
                a zero-row transfer (the kernel backend's table upload refuses
                one) -- and the event alone acknowledges it once ``stream``
                reaches it.
            device_buffers: The layout's Device buffers.
            host_buffer: The compact pinned Host buffer.
            geometry: The executor's static geometry.
            stream: The transfer stream the copy runs on.
            prerequisite_stream: The stream whose completed work the copy must
                observe -- for a D2H the one the forwards wrote the source
                pages on, for an H2D the one that zeroed the destination
                pages -- or None when the caller already ordered ``stream``
                behind it.
            backend: The ``transfer_cache_blocks`` transport.

        Returns:
            The event recorded on ``stream`` after the copy.
        """
        if prerequisite_stream is not None:
            # Behind the forwards that wrote the source pages (D2H) or the
            # zeroing of the destinations (H2D): that is what lets the copy
            # read, or land on, their final bytes.
            stream.wait_stream(prerequisite_stream)
        if transfers:
            num_blocks = self._stage(
                transfers,
                device_buffers=device_buffers,
                host_buffer=host_buffer,
                geometry=geometry,
                stream=stream,
                backend=backend,
            )
            transfer_cache_blocks(
                direction,
                device_buffers,
                host_buffer,
                geometry,
                self.workspace,
                stream,
                num_blocks=num_blocks,
                geometry_offset=0,
                num_geometry_rows=geometry.num_field_rows,
                backend=backend,
                grid_cap=None,
                layer_ready_flags=None,
            )
        finish = device_module.Event()
        finish.record(stream)
        return finish

    def start_d2h(self, transfers: Sequence[tuple[int, int, int]], **launch):
        """Launch one Device-to-Host batch; arguments as for :meth:`_start`."""
        return self._start("d2h", transfers, **launch)

    def start_h2d(self, transfers: Sequence[tuple[int, int, int]], **launch):
        """Launch one Host-to-Device batch; arguments as for :meth:`_start`."""
        return self._start("h2d", transfers, **launch)


class CompletionQueue:
    """In-flight copies whose completion events have not been polled yet.

    Submission runs on the forward thread and polling on the control plane
    (event queries only), so this queue is the cross-thread handoff; the
    lock covers every mutation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[tuple[Any, Any]] = []

    def push(self, finish_event, payload) -> None:
        """Queue ``payload`` to be released once ``finish_event`` has completed."""
        with self._lock:
            self._pending.append((finish_event, payload))

    def pop_ready(self) -> list:
        """Release the payloads whose events completed, oldest first. Never blocks."""
        ready = []
        with self._lock:
            still = []
            for finish_event, payload in self._pending:
                if finish_event.query():
                    ready.append(payload)
                else:
                    still.append((finish_event, payload))
            self._pending = still
        return ready

    def drop_all(self) -> list:
        """Forget every pending entry (shutdown); returns the dropped payloads."""
        with self._lock:
            dropped = [payload for _event, payload in self._pending]
            self._pending = []
        return dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)
