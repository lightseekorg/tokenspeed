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

from contextlib import contextmanager
from typing import Any

import torch


def new_device_stream():
    """Create a stream for the active accelerator, or None on CPU-only hosts."""
    try:
        device_module = torch.get_device_module()
    except (AttributeError, RuntimeError):
        return None
    is_available = getattr(device_module, "is_available", None)
    if callable(is_available) and not is_available():
        return None
    return device_module.Stream()


@contextmanager
def limit_stream_cores(
    stream: Any | None,
    *,
    cube_num: int,
    vector_num: int,
    enable: bool,
):
    """Temporarily limit accelerator cores available to one stream.

    Accelerator backends without a per-stream limit API keep their normal
    scheduling.  Supporting backends restore the stream's previous limits,
    rather than resetting to a device-wide default that may discard an outer
    scheduling scope.
    """
    if not enable or stream is None:
        yield
        return
    device_module = torch.get_device_module(getattr(stream, "device", None))
    get_limit = getattr(device_module, "get_stream_limit", None)
    set_limit = getattr(device_module, "set_stream_limit", None)
    if not callable(get_limit) or not callable(set_limit):
        yield
        return
    previous = get_limit(stream)
    set_limit(stream, cube_num=cube_num, vector_num=vector_num)
    try:
        yield
    finally:
        set_limit(
            stream,
            cube_num=int(previous["cube_core_num"]),
            vector_num=int(previous["vector_core_num"]),
        )


class StreamFork:
    def __init__(self, aux_stream: Any | None):
        self.aux_stream = aux_stream
        self.device_module = None
        if aux_stream is not None:
            device = getattr(aux_stream, "device", None)
            self.device_module = torch.get_device_module(device)
        self.fork_event = (
            self.device_module.Event() if self.device_module is not None else None
        )
        self.join_event = (
            self.device_module.Event() if self.device_module is not None else None
        )
        self._active = False
        self._overlap = True
        self._current: Any | None = None

    @contextmanager
    def scope(self, *, enable: bool, overlap: bool = True):
        """Configure auxiliary-stream execution for work in this scope.

        Args:
            enable: Whether calls to ``branch()`` use the auxiliary stream.
            overlap: Whether the main stream may overlap with auxiliary-stream
                work. If false, the main stream waits after each branch.

        Yields:
            This stream fork, ready to create branches with ``branch()``.
        """
        self._active = enable and self.aux_stream is not None
        self._overlap = overlap
        if self._active:
            self._current = self.device_module.current_stream()
            self.fork_event.record(self._current)
        try:
            yield self
        finally:
            if self._active:
                self.join_event.wait(self._current)
                self._active = False
                self._overlap = True
                self._current = None

    @contextmanager
    def branch(self):
        if not self._active:
            yield
            return
        with self.device_module.stream(self.aux_stream):
            self.fork_event.wait(self.aux_stream)
            yield
            self.join_event.record(self.aux_stream)
        if not self._overlap:
            self.join_event.wait(self._current)
