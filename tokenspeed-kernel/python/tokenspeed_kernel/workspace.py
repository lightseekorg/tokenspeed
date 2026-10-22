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

"""Kernel state that must start zeroed, including under CUDA/HIP graph capture.

Some kernels keep device state across their own programs, such as split-K
arrival counters or semaphores. Such state must be zero before a launch and
every launch leaves it zero again, so it needs zeroing once, not per launch.
:func:`zeroed_state` hands out these buffers.

Eager launches share one buffer per stream: same-stream launches are ordered
and each leaves the buffer zeroed for the next.

A graph captured inside :func:`capture_session` takes its buffers from an
arena the session zeroes before capture starts, so replays carry no zeroing
nodes. The arena lives outside the graph's memory pool on purpose: during
capture the pool hands out memory that tensors freed earlier in the capture
used, and their kernels write it again on every replay, before a later
launch's state would be read. The session keeps the arena alive as long as an
owner object, normally the graph, which replays the addresses it captured but
holds no reference to them. Without a session, or once the arena is used up, a
captured launch falls back to zeroing a fresh buffer inside the graph.
"""

from __future__ import annotations

import contextlib
import threading
import warnings
from collections.abc import Iterator
from weakref import WeakKeyDictionary

import torch

_eager_buffers: dict[tuple[int, int, torch.dtype, int], torch.Tensor] = {}
# Allocations and prepared plans captured graphs use, kept alive by their owners.
_owned_resources: WeakKeyDictionary[object, dict[int, object]] = WeakKeyDictionary()

# Slices start on this many bytes.
_ALIGN = 256
# A session's arena holds at least this many bytes (1024 slices), or twice the
# most any earlier session asked for.
# The floor keeps the size independent of capture order: a small graph captured
# first must not shrink the arena of a larger one after it.
_MIN_ARENA_BYTES = 256 << 10


class _Arena:
    """Zeroed bytes a capture session carves state from."""

    def __init__(self, num_bytes: int, device: torch.device) -> None:
        self.storage = torch.zeros((num_bytes,), dtype=torch.uint8, device=device)
        self.used = 0
        # Bytes every take asked for, including those that did not fit.
        self.wanted = 0

    def take(self, numel: int, dtype: torch.dtype) -> torch.Tensor | None:
        size = numel * dtype.itemsize
        start = self.used
        end = start + (size + _ALIGN - 1) // _ALIGN * _ALIGN
        self.wanted += end - start
        if end > self.storage.numel():
            return None
        self.used = end
        return self.storage[start : start + size].view(dtype)


class _Sessions(threading.local):
    def __init__(self) -> None:
        # The arena of each open session, innermost last.
        self.arenas: list[_Arena] = []
        self.owners: list[object] = []
        self.peak_bytes = 0


_sessions = _Sessions()


def retain_for_capture(resource: object) -> None:
    """Keep a prepared kernel resource alive until its graph owner is released.

    Args:
        resource: Plan or allocation used by captured launches. Call inside
            ``capture_session(owner)``; the resource must not retain the owner.
    """
    if not _sessions.owners:
        raise RuntimeError("capture kernel plans inside capture_session(owner)")
    _owned_resources.setdefault(_sessions.owners[-1], {})[id(resource)] = resource


def zeroed_state(
    numel: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return a zeroed buffer that the caller's launch leaves zeroed again.

    Args:
        numel: Number of elements.
        dtype: Element type.
        device: CUDA/HIP device the launch runs on.

    Returns:
        A 1-D tensor of ``numel`` zeros. Eager calls on one stream may return
        the same buffer, so the launch must restore every element to zero
        before it completes. A launch captured in a :func:`capture_session`
        gets a buffer that only that launch uses.
    """
    if torch.cuda.is_current_stream_capturing():
        arena = _sessions.arenas[-1] if _sessions.arenas else None
        index = torch.cuda.current_device() if device.index is None else device.index
        buffer = None
        if arena is not None and arena.storage.device.index == index:
            buffer = arena.take(numel, dtype)
        if buffer is None:
            # No pre-zeroed memory: zero a fresh buffer inside the graph.
            return torch.zeros((numel,), dtype=dtype, device=device)
        return buffer

    device_index = torch.cuda.current_device() if device.index is None else device.index
    stream_id = torch.cuda.current_stream(device_index).cuda_stream
    key = (device_index, stream_id, dtype, numel)
    buffer = _eager_buffers.get(key)
    if buffer is None:
        buffer = torch.zeros((numel,), dtype=dtype, device=device)
        _eager_buffers[key] = buffer
    return buffer


@contextlib.contextmanager
def capture_session(owner: object) -> Iterator[None]:
    """Supply zeroed state and own prepared plans for captured launches.

    Enter it before the capture starts: it allocates and zeroes the arena the
    captured launches carve their :func:`zeroed_state` from, on the current
    device, and keeps it alive as long as ``owner``.

    Args:
        owner: Object that outlives every replay of the captured graph or
            graphs, normally the ``torch.cuda.CUDAGraph``; it must support
            weak references.
    """
    if not torch.cuda.is_available():
        yield
        return
    num_bytes = max(_MIN_ARENA_BYTES, 2 * _sessions.peak_bytes)
    arena = _Arena(num_bytes, torch.device("cuda", torch.cuda.current_device()))
    _sessions.arenas.append(arena)
    _sessions.owners.append(owner)
    try:
        yield
    finally:
        _sessions.arenas.pop()
        _sessions.owners.pop()
    _sessions.peak_bytes = max(_sessions.peak_bytes, arena.wanted)
    if arena.wanted > num_bytes:
        warnings.warn(
            f"capture_session arena of {num_bytes} bytes ran out; the rest of the "
            "captured launches zero their state inside the graph",
            stacklevel=3,
        )
    if arena.used:
        _owned_resources.setdefault(owner, {})[id(arena.storage)] = arena.storage


__all__ = ["capture_session", "retain_for_capture", "zeroed_state"]
