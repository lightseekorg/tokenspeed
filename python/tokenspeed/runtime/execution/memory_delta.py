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

"""Observe device-driver free memory across named regions of execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Protocol


class MemoryDeltaObserver(Protocol):
    """Record how many bytes each named region takes from driver-free memory."""

    samples: dict[str, list[int]]

    def measure(self, series: str) -> AbstractContextManager[None]: ...


@dataclass
class _DriverMemoryDeltaObserver:
    """Measure regions against synchronized driver-reported free memory."""

    device_module: object
    gpu_id: int
    samples: dict[str, list[int]] = field(default_factory=dict)

    @contextmanager
    def measure(self, series: str) -> Iterator[None]:
        self.device_module.synchronize()
        free_before = int(self.device_module.mem_get_info(self.gpu_id)[0])

        yield

        self.device_module.synchronize()
        free_after = int(self.device_module.mem_get_info(self.gpu_id)[0])
        self.samples.setdefault(series, []).append(free_before - free_after)


@dataclass(frozen=True)
class _NullMemoryDeltaObserver:
    """Measure nothing, for the paths that only ever capture."""

    samples: dict[str, list[int]] = field(default_factory=dict)

    @contextmanager
    def measure(self, series: str) -> Iterator[None]:
        del series

        yield


_NULL_OBSERVER: MemoryDeltaObserver = _NullMemoryDeltaObserver()


def memory_delta_observer(
    *, record: bool, device_module: object | None = None, gpu_id: int | None = None
) -> MemoryDeltaObserver:
    """Return an observer that records driver-memory deltas, or one that ignores them."""
    if not record:
        return _NULL_OBSERVER
    return _DriverMemoryDeltaObserver(device_module, gpu_id)
