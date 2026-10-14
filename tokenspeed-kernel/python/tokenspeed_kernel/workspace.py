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
and each leaves the buffer zeroed for the next. A graph captured inside
:func:`capture_session` gets buffers of its own that are zeroed once after
capture, outside the graph, so replays carry no zeroing nodes. The session
ties their lifetime to an owner object, normally the graph: a graph replays
the addresses it captured but holds no reference to them. Without a session,
a captured launch falls back to zeroing a fresh buffer inside the graph.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from weakref import WeakKeyDictionary

import torch

_eager_buffers: dict[tuple[int, int, torch.dtype, int], torch.Tensor] = {}
# Buffers captured graphs use, kept alive by their owners.
_owned_buffers: WeakKeyDictionary[object, list[torch.Tensor]] = WeakKeyDictionary()


class _Sessions(threading.local):
    def __init__(self) -> None:
        # Buffers awaiting zeroing for each open session, innermost last.
        self.pending: list[list[torch.Tensor]] = []


_sessions = _Sessions()


def zeroed_state(numel: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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
        if not _sessions.pending:
            # No session to zero it after capture: zero it inside the graph.
            return torch.zeros((numel,), dtype=dtype, device=device)
        buffer = torch.empty((numel,), dtype=dtype, device=device)
        _sessions.pending[-1].append(buffer)
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
    """Give the launches captured in this scope pre-zeroed state.

    Enter it around the whole capture, outside ``torch.cuda.graph``, so that
    the capture has ended when the session closes. The buffers captured
    launches got from :func:`zeroed_state` are then zeroed outside the graph
    and kept alive as long as ``owner``.

    Args:
        owner: Object that outlives every replay of the captured graph or
            graphs, normally the ``torch.cuda.CUDAGraph``; it must support
            weak references.
    """
    pending: list[torch.Tensor] = []
    _sessions.pending.append(pending)
    try:
        yield
    finally:
        _sessions.pending.pop()
    if not pending:
        return
    for buffer in pending:
        buffer.zero_()
    # Replays may run on other streams; make the zeros visible to them.
    torch.cuda.synchronize(pending[0].device)
    _owned_buffers.setdefault(owner, []).extend(pending)


__all__ = ["capture_session", "zeroed_state"]
