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

"""Event- and weakref-fenced ring of reusable pinned host staging buffers."""

from __future__ import annotations

import math
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tokenspeed.runtime.utils.common import is_pin_memory_available


def copy_host_values(dst: torch.Tensor, values) -> None:
    """Copy a Python sequence, NumPy array, or tensor into pre-staged host memory.

    Writing a Python list through ``dst.numpy()[:]`` avoids the fresh
    ``aten::empty`` allocation incurred by ``torch.as_tensor(list, ...)`` on the
    forward critical path.
    """
    if isinstance(values, torch.Tensor):
        dst.copy_(values)
        return
    if isinstance(values, np.ndarray):
        np.copyto(dst.numpy(), values)
        return
    dst.numpy()[:] = values


def _default_event_factory(device: torch.device) -> Callable[[], Any] | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return lambda: torch.cuda.Event(enable_timing=False)
    return None


@dataclass
class _PinnedSlot:
    """One host staging buffer and its asynchronous reuse fences."""

    slot_id: int
    storage: torch.Tensor
    capacity_bytes: int
    event: Any | None
    event_armed: bool = False
    live_refs: list[weakref.ref[torch.Tensor]] = field(default_factory=list)


class PinnedHostRing:
    """Bounded ring of pinned host staging buffers fenced by events and refs.

    Eliminates per-step ``cudaHostAlloc`` allocator stalls without creating
    overlap-scheduling data races: a slot cannot be reused while any returned
    tensor view is still referenced or while its recorded device copy event is
    still in flight. When all slots in the ring are busy simultaneously, falls
    back cleanly to a standalone host buffer.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        max_slots: int = 8,
        min_capacity_bytes: int = 4096,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        if max_slots <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots}")
        if min_capacity_bytes <= 0:
            raise ValueError(
                f"min_capacity_bytes must be positive, got {min_capacity_bytes}"
            )
        self.device = torch.device(device)
        self._max_slots = max_slots
        self._min_capacity_bytes = min_capacity_bytes
        self._pin_memory = (
            self.device.type == "cuda" and is_pin_memory_available()
        ) or self.device.type == "npu"
        self._event_factory = (
            event_factory
            if event_factory is not None
            else _default_event_factory(self.device)
        )
        self._slots: list[_PinnedSlot] = []
        self._cursor: int = 0
        self._pending_seal: set[int] = set()
        self._lock = threading.Lock()
        self.alloc_count: int = 0
        self.fallback_alloc_count: int = 0

    @staticmethod
    def _numel(shape_or_numel: tuple[int, ...] | int) -> int:
        if isinstance(shape_or_numel, int):
            return shape_or_numel
        return math.prod(shape_or_numel)

    def _round_capacity(self, needed_bytes: int) -> int:
        target = max(needed_bytes, self._min_capacity_bytes)
        return 1 << (target - 1).bit_length()

    def _alloc_backing(self, capacity_bytes: int) -> torch.Tensor:
        self.alloc_count += 1
        return torch.empty(
            capacity_bytes,
            dtype=torch.int8,
            device="cpu",
            pin_memory=self._pin_memory,
        )

    def _slot_is_free_locked(self, slot: _PinnedSlot) -> bool:
        if slot.slot_id in self._pending_seal:
            return False
        live = [ref for ref in slot.live_refs if ref() is not None]
        slot.live_refs = live
        if live:
            return False
        if slot.event_armed:
            if slot.event is not None and not slot.event.query():
                return False
            slot.event_armed = False
        return True

    def _acquire_backing_locked(
        self, needed_bytes: int
    ) -> tuple[_PinnedSlot | None, torch.Tensor]:
        count = len(self._slots)
        for offset in range(count):
            idx = (self._cursor + offset) % count
            slot = self._slots[idx]
            if self._slot_is_free_locked(slot):
                self._cursor = (idx + 1) % count
                if slot.capacity_bytes < needed_bytes:
                    cap = self._round_capacity(needed_bytes)
                    slot.storage = self._alloc_backing(cap)
                    slot.capacity_bytes = cap
                return slot, slot.storage

        if count < self._max_slots:
            cap = self._round_capacity(needed_bytes)
            slot = _PinnedSlot(
                slot_id=count,
                storage=self._alloc_backing(cap),
                capacity_bytes=cap,
                event=(
                    self._event_factory() if self._event_factory is not None else None
                ),
            )
            self._slots.append(slot)
            self._cursor = (
                0 if len(self._slots) == self._max_slots else len(self._slots)
            )
            return slot, slot.storage

        self.fallback_alloc_count += 1
        self.alloc_count += 1
        return None, torch.empty(
            needed_bytes,
            dtype=torch.int8,
            device="cpu",
            pin_memory=self._pin_memory,
        )

    def acquire_views(
        self,
        specs: Sequence[tuple[tuple[int, ...] | int, torch.dtype]],
        *,
        align: int = 8,
    ) -> list[torch.Tensor]:
        """Allocate aligned typed host views sharing one ring slot.

        The caller should call ``seal_pending()`` once async device copies using
        the returned views have been enqueued on the active stream.
        """
        if align <= 0:
            raise ValueError(f"align must be positive, got {align}")
        layout: list[tuple[int, tuple[int, ...] | int, int, torch.dtype]] = []
        off = 0
        for shape_or_numel, dtype in specs:
            numel = self._numel(shape_or_numel)
            nbytes = numel * dtype.itemsize
            eff_align = max(align, dtype.itemsize)
            if off % eff_align != 0:
                off = ((off + eff_align - 1) // eff_align) * eff_align
            layout.append((off, shape_or_numel, numel, dtype))
            off += nbytes

        if off == 0:
            return [
                torch.empty(shape_or_numel, dtype=dtype, device="cpu")
                for shape_or_numel, dtype in specs
            ]

        with self._lock:
            slot, storage = self._acquire_backing_locked(off)
            views = [
                storage[byte_off : byte_off + numel * dtype.itemsize]
                .view(dtype)
                .view(shape_or_numel)
                for byte_off, shape_or_numel, numel, dtype in layout
            ]
            if slot is not None:
                slot.live_refs.extend(weakref.ref(view) for view in views)
                self._pending_seal.add(slot.slot_id)
            return views

    def seal_pending(self) -> None:
        """Record stream events on all unsealed slots staged in this batch."""
        with self._lock:
            if not self._pending_seal:
                return
            for slot_id in self._pending_seal:
                slot = self._slots[slot_id]
                if slot.event is not None:
                    slot.event.record()
                slot.event_armed = True
            self._pending_seal.clear()

    @contextmanager
    def staging(self) -> Iterator[None]:
        """Fence every slot acquired inside the block on exit."""
        try:
            yield
        finally:
            self.seal_pending()

    def copy_packed_d2h(
        self,
        src: torch.Tensor,
        views_spec: Sequence[tuple[int, tuple[int, ...] | int, torch.dtype]],
    ) -> tuple[torch.Tensor, ...]:
        """Copy contiguous ``src`` asynchronously into one fenced host slot and slice views."""
        total_elements = src.numel()
        if total_elements == 0:
            return tuple(
                torch.empty(shape_or_numel, dtype=dtype, device="cpu")
                for _, shape_or_numel, dtype in views_spec
            )

        needed_bytes = total_elements * src.element_size()
        with self._lock:
            slot, storage = self._acquire_backing_locked(needed_bytes)
            if slot is not None:
                self._pending_seal.add(slot.slot_id)

        try:
            flat_dst = storage[:needed_bytes].view(src.dtype).view(src.shape)
            flat_dst.copy_(src, non_blocking=True)
            views = tuple(
                storage[
                    elem_off
                    * dtype.itemsize : (elem_off + self._numel(shape_or_numel))
                    * dtype.itemsize
                ]
                .view(dtype)
                .view(shape_or_numel)
                for elem_off, shape_or_numel, dtype in views_spec
            )
        finally:
            if slot is not None:
                with self._lock:
                    self._pending_seal.discard(slot.slot_id)
                    if "views" in locals():
                        slot.live_refs.extend(weakref.ref(view) for view in views)
                    if slot.event is not None:
                        slot.event.record()
                    slot.event_armed = True
        return views

    def copy_tensors_d2h(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Copy arbitrary contiguous device tensors into one fenced pinned host slot."""
        specs = [(tensor.shape, tensor.dtype) for tensor in tensors]
        views = self.acquire_views(specs, align=8)
        try:
            for dst, src in zip(views, tensors, strict=True):
                if src.numel() > 0:
                    dst.copy_(src, non_blocking=True)
        finally:
            self.seal_pending()
        return tuple(views)


_DEVICE_RINGS_LOCK = threading.Lock()
_DEVICE_RINGS: dict[tuple[str, int | None], PinnedHostRing] = {}


def get_shared_pinned_host_ring(device: torch.device | str) -> PinnedHostRing:
    """Return a shared ``PinnedHostRing`` for ``device`` (for stateless bridges)."""
    dev = torch.device(device)
    index = dev.index
    if index is None and dev.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
    key = (dev.type, index)
    with _DEVICE_RINGS_LOCK:
        ring = _DEVICE_RINGS.get(key)
        if ring is None:
            ring = PinnedHostRing(dev)
            _DEVICE_RINGS[key] = ring
        return ring
