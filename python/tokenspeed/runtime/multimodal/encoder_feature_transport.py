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

"""Materialize multimodal input features on their encoder devices."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from tokenspeed.runtime.distributed.mapping import VisionTowerMapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager,
)
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)
LOG_MM_TIMING = envs.TOKENSPEED_LOG_MM_TIMING.get()

# Small transfers are faster after staging the whole batch; larger transfers
# benefit from overlapping each H2D enqueue with the next SHM-to-pinned copy.
_INTERLEAVED_H2D_MIN_AVERAGE_BYTES = 1024 * 1024
# One rank can stage the input and distribute it faster than every TP rank
# repeating the host copy once the payload reaches this TP-scaled threshold.
# Local B200 measurements put the break-even points near 128 MiB for TP2 and
# 64 MiB for TP4.
_TP_BROADCAST_BASE_MIN_BYTES = 256 * 1024 * 1024


class DeviceFeatureItem(Protocol):
    """Structural type for a multimodal item materialized before encoding."""

    feature: torch.Tensor | None
    feature_shm: ShmTensorHandle | None


def _packed_layout(
    handles: Sequence[ShmTensorHandle],
) -> tuple[list[tuple[ShmTensorHandle, int, int]], int]:
    """Lay out heterogeneous tensors in one byte buffer with dtype alignment."""
    layout = []
    cursor = 0
    for handle in handles:
        alignment = torch.empty((), dtype=handle.dtype).element_size()
        cursor = (cursor + alignment - 1) // alignment * alignment
        nbytes = handle.nbytes()
        layout.append((handle, cursor, nbytes))
        cursor += nbytes
    return layout, cursor


def _build_owner_transfers(
    items: Sequence[DeviceFeatureItem],
    item_indices_by_rank: Sequence[Sequence[int]],
    owner_ranks: tuple[int, ...],
) -> dict[tuple[int, int], list[ShmTensorHandle]]:
    if len(item_indices_by_rank) != len(owner_ranks):
        raise ValueError(
            "item-DP assignment and owner group sizes differ: "
            f"{len(item_indices_by_rank)} != {len(owner_ranks)}"
        )

    transfers: dict[tuple[int, int], list[ShmTensorHandle]] = {}
    for owner_index, item_indices in enumerate(item_indices_by_rank):
        owner_rank = owner_ranks[owner_index]
        for item_index in item_indices:
            handle = items[item_index].feature_shm
            if (
                not isinstance(handle, ShmTensorHandle)
                or not handle.is_cross_node
                or handle.is_reachable_from(owner_rank)
            ):
                continue
            source_rank = handle.source_rank
            if source_rank is None:
                raise RuntimeError("Cross-node SHM feature has no source rank")
            if source_rank not in owner_ranks:
                raise RuntimeError(
                    f"SHM source rank {source_rank} is outside item-DP group "
                    f"{owner_ranks}"
                )
            transfers.setdefault((source_rank, owner_rank), []).append(handle)
    return transfers


@dataclass(frozen=True)
class _TpBroadcastEntry:
    item: DeviceFeatureItem
    handle: ShmTensorHandle
    length: int


@dataclass(frozen=True)
class _TpBroadcastBatch:
    source_rank: int
    dtype: torch.dtype
    entries: tuple[_TpBroadcastEntry, ...]
    stage_via_pinned: bool = False

    @property
    def numel(self) -> int:
        return sum(entry.length for entry in self.entries)

    @property
    def nbytes(self) -> int:
        return sum(entry.handle.nbytes() for entry in self.entries)


class EncoderFeatureTransport:
    """Own CPU routing, H2D staging, and vision-TP input broadcasts."""

    def __init__(self, encoder_mapping: VisionTowerMapping | None = None) -> None:
        self._encoder_dp_group = (
            encoder_mapping.dp_group if encoder_mapping is not None else None
        )
        self._h2d_stream: torch.cuda.Stream | None = None

        vision_tp_group = (
            encoder_mapping.tp_group if encoder_mapping is not None else None
        )
        self._vision_tp_group = vision_tp_group
        self._has_vision_tp = vision_tp_group is not None and len(vision_tp_group) > 1
        self._vision_tp_process_group = None
        self._vision_tp_src_rank: int | None = None
        if (
            vision_tp_group is not None
            and len(vision_tp_group) > 1
            and process_group_manager.has_process_group("nccl", vision_tp_group)
        ):
            self._vision_tp_process_group = process_group_manager.get_process_group(
                "nccl", vision_tp_group
            )
            self._vision_tp_src_rank = vision_tp_group[0]

    def route_to_item_owners(
        self,
        items: Sequence[DeviceFeatureItem],
        item_indices_by_rank: Sequence[Sequence[int]],
    ) -> None:
        """Pack and send cross-node SHM payloads to their item-DP owners."""
        if not any(
            item.feature_shm is not None and item.feature_shm.is_cross_node
            for item in items
        ):
            return
        if self._encoder_dp_group is None:
            raise RuntimeError("Item-owner SHM routing has no encoder DP group")
        owner_ranks = tuple(self._encoder_dp_group)
        transfers = _build_owner_transfers(
            items,
            item_indices_by_rank,
            owner_ranks,
        )
        if not transfers:
            return

        cpu_group = process_group_manager.get_process_group("gloo", owner_ranks)
        started = time.perf_counter() if LOG_MM_TIMING else None
        rank = torch.distributed.get_rank()
        transferred_handles = 0
        transferred_bytes = 0
        for (source_rank, owner_rank), handles in sorted(transfers.items()):
            layout, packed_nbytes = _packed_layout(handles)
            transferred_handles += len(layout)
            transferred_bytes += sum(nbytes for _, _, nbytes in layout)
            if rank == source_rank:
                payload = torch.empty(packed_nbytes, dtype=torch.uint8)
                for handle, offset, nbytes in layout:
                    handle.copy_bytes_into(payload.narrow(0, offset, nbytes))
                torch.distributed.send(payload, dst=owner_rank, group=cpu_group)
            elif rank == owner_rank:
                payload = torch.empty(packed_nbytes, dtype=torch.uint8, pin_memory=True)
                torch.distributed.recv(payload, src=source_rank, group=cpu_group)
                for handle, offset, nbytes in layout:
                    handle.set_remote(payload.narrow(0, offset, nbytes))

        if LOG_MM_TIMING and started is not None:
            logger.info(
                "mm_timing shm_owner_route_ms transfers=%d handles=%d bytes=%d "
                "elapsed=%.3f",
                len(transfers),
                transferred_handles,
                transferred_bytes,
                (time.perf_counter() - started) * 1000,
            )

    def move_to_device(
        self, items: Sequence[DeviceFeatureItem], device: torch.device
    ) -> None:
        """Materialize pending features on ``device`` with bounded host copies."""
        pending = [
            item
            for item in items
            if item.feature_shm is not None
            or (
                isinstance(item.feature, torch.Tensor) and item.feature.device != device
            )
        ]
        if not pending:
            return

        cross_node_tp_items = [
            item
            for item in pending
            if self._has_vision_tp
            and item.feature_shm is not None
            and item.feature_shm.is_cross_node
        ]
        if cross_node_tp_items:
            if device.type != "cuda":
                raise RuntimeError(
                    "Cross-node weight-TP SHM transport requires a CUDA device"
                )
            batches = self._build_cross_node_tp_batches(cross_node_tp_items)
            self._execute_tp_broadcasts(batches, device, mode="cross_node")
            pending = [
                item
                for item in pending
                if item.feature_shm is not None
                or (
                    isinstance(item.feature, torch.Tensor)
                    and item.feature.device != device
                )
            ]
            if not pending:
                return

        if device.type != "cuda":
            for item in pending:
                handle = item.feature_shm
                if handle is not None:
                    try:
                        item.feature = handle.consume()
                    finally:
                        item.feature_shm = None
                if isinstance(item.feature, torch.Tensor):
                    item.feature = item.feature.to(device, non_blocking=True)
            return

        shm_items = [item for item in pending if item.feature_shm is not None]
        shm_count = len(shm_items)
        shm_nbytes = sum(
            item.feature_shm.nbytes()
            for item in shm_items
            if item.feature_shm is not None
        )
        use_tp_broadcast = self._should_use_tp_broadcast(pending)
        interleave_h2d = shm_nbytes > shm_count * _INTERLEAVED_H2D_MIN_AVERAGE_BYTES
        defer_shm_cleanup = shm_count == 1 and interleave_h2d
        if not use_tp_broadcast and not interleave_h2d:
            for item in shm_items:
                handle = item.feature_shm
                assert handle is not None
                try:
                    item.feature = handle.consume()
                finally:
                    item.feature_shm = None

        # Keep collectives on the model stream so their ordering matches other
        # TP collectives queued by the forward pass on every rank.
        if use_tp_broadcast:
            batch = self._build_local_tp_batch(pending)
            self._execute_tp_broadcasts((batch,), device, mode="local")
            return

        h2d = self._h2d_stream_on(device)
        current = torch.cuda.current_stream(device)
        with torch.cuda.stream(h2d):
            for item in pending:
                handle = item.feature_shm
                if handle is not None:
                    if defer_shm_cleanup:
                        try:
                            item.feature = handle.copy_to_pinned()
                            item.feature = item.feature.to(device, non_blocking=True)
                        finally:
                            handle.release()
                            item.feature_shm = None
                        continue
                    try:
                        item.feature = handle.consume()
                    finally:
                        item.feature_shm = None
                if isinstance(item.feature, torch.Tensor):
                    item.feature = item.feature.to(device, non_blocking=True)
        current.wait_stream(h2d)
        for item in pending:
            if isinstance(item.feature, torch.Tensor):
                item.feature.record_stream(current)

    def _should_use_tp_broadcast(self, items: Sequence[DeviceFeatureItem]) -> bool:
        """Return whether a same-host SHM batch should use one TP H2D source."""
        tp_group = self._vision_tp_group
        if (
            tp_group is None
            or self._vision_tp_process_group is None
            or self._vision_tp_src_rank is None
            or not items
            or not all(item.feature_shm is not None for item in items)
        ):
            return False

        handles = []
        for item in items:
            handle = item.feature_shm
            assert handle is not None
            handles.append(handle)
        dtype = handles[0].dtype
        if any(handle.dtype != dtype for handle in handles):
            return False
        total_nbytes = sum(handle.nbytes() for handle in handles)
        return total_nbytes >= _TP_BROADCAST_BASE_MIN_BYTES // len(tp_group)

    def _h2d_stream_on(self, device: torch.device) -> torch.cuda.Stream:
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(device=device)
        return self._h2d_stream

    def _build_local_tp_batch(
        self, items: Sequence[DeviceFeatureItem]
    ) -> _TpBroadcastBatch:
        source_rank = self._vision_tp_src_rank
        if source_rank is None:
            raise RuntimeError("Local TP SHM broadcast has no source rank")
        entries = []
        for item in items:
            handle = item.feature_shm
            if handle is None:
                raise RuntimeError("Local TP SHM broadcast received an inline feature")
            entries.append(_TpBroadcastEntry(item, handle, math.prod(handle.shape)))
        return _TpBroadcastBatch(
            source_rank=source_rank,
            dtype=entries[0].handle.dtype,
            entries=tuple(entries),
            stage_via_pinned=len(entries) > 1,
        )

    def _build_cross_node_tp_batches(
        self, items: Sequence[DeviceFeatureItem]
    ) -> tuple[_TpBroadcastBatch, ...]:
        tp_group = self._vision_tp_group
        if tp_group is None or self._vision_tp_process_group is None:
            raise RuntimeError(
                "Cross-node weight-TP SHM transport has no vision TP process group"
            )

        grouped: dict[
            tuple[int, torch.dtype],
            list[_TpBroadcastEntry],
        ] = {}
        for item in items:
            handle = item.feature_shm
            assert isinstance(handle, ShmTensorHandle)
            source_rank = handle.source_rank
            if source_rank is None:
                raise RuntimeError("Cross-node TP item has no SHM broadcast source")
            if source_rank not in tp_group:
                raise RuntimeError(
                    f"SHM source rank {source_rank} is outside vision TP group "
                    f"{tp_group}"
                )
            grouped.setdefault((source_rank, handle.dtype), []).append(
                _TpBroadcastEntry(item, handle, math.prod(handle.shape))
            )

        batches = []
        for (source_rank, dtype), entries in sorted(
            grouped.items(), key=lambda pair: (pair[0][0], str(pair[0][1]))
        ):
            batches.append(
                _TpBroadcastBatch(
                    source_rank=source_rank,
                    dtype=dtype,
                    entries=tuple(entries),
                )
            )
        return tuple(batches)

    def _execute_tp_broadcasts(
        self,
        batches: Sequence[_TpBroadcastBatch],
        device: torch.device,
        *,
        mode: str,
    ) -> None:
        process_group = self._vision_tp_process_group
        if process_group is None:
            raise RuntimeError("SHM TP broadcast has no vision TP process group")

        started = time.perf_counter() if LOG_MM_TIMING else None
        rank = torch.distributed.get_rank()
        for batch in batches:
            base = torch.empty(batch.numel, dtype=batch.dtype, device=device)
            offset = 0
            if rank == batch.source_rank:
                for entry in batch.entries:
                    destination = base.narrow(0, offset, entry.length)
                    try:
                        if batch.stage_via_pinned:
                            source = entry.handle.consume().reshape(-1)
                            destination.copy_(source, non_blocking=True)
                        else:
                            entry.handle.copy_into(destination)
                    finally:
                        entry.item.feature_shm = None
                    offset += entry.length
            else:
                for entry in batch.entries:
                    try:
                        entry.handle.release()
                    finally:
                        entry.item.feature_shm = None

            torch.distributed.broadcast(
                base,
                src=batch.source_rank,
                group=process_group,
            )
            offset = 0
            for entry in batch.entries:
                entry.item.feature = base.narrow(0, offset, entry.length).view(
                    entry.handle.shape
                )
                offset += entry.length

        if LOG_MM_TIMING and started is not None:
            logger.info(
                "mm_timing shm_tp_broadcast_ms mode=%s groups=%d handles=%d "
                "bytes=%d elapsed=%.3f",
                mode,
                len(batches),
                sum(len(batch.entries) for batch in batches),
                sum(batch.nbytes for batch in batches),
                (time.perf_counter() - started) * 1000,
            )
