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

"""Transport-independent metric events and plugin lifecycle contracts."""

from dataclasses import dataclass
from typing import Literal, Protocol

MetricConfig = dict[str, object]
MetricOperation = Literal["inc", "set", "observe"]


@dataclass(frozen=True)
class MetricEvent:
    """One metric update, treated as read-only by every sink.

    ``inc`` carries a delta; ``set`` carries the current value; ``observe``
    carries an observation with ``weight`` samples. Async sinks must retain
    a stable snapshot of labels instead of the caller's mutable dictionary.
    """

    timestamp_ns: int
    metric: str
    operation: MetricOperation
    value: float
    labels: dict[str, str]
    weight: int = 1


class MetricEventSink(Protocol):
    """A process-local, thread-safe asynchronous metric output."""

    def emit(self, event: MetricEvent) -> None:
        """Accept an event without waiting for I/O or queue capacity.

        Calls may race with ``close``. After closing starts, drop new events
        without reopening resources or blocking the caller.
        """
        ...

    def close(self, timeout: float) -> None:
        """Stop accepting events and drain within the budget; safe to repeat."""
        ...


class MetricNodeSession(Protocol):
    """Launcher-owned resources shared by a node's metric producers."""

    @property
    def producer_config(self) -> MetricConfig:
        """Return JSON-only configuration for creating process-local sinks."""
        ...

    def close(self, timeout: float) -> None:
        """Release shared resources within the timeout; safe to call repeatedly."""
        ...
