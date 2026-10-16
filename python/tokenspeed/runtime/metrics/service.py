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

"""Process-local ownership of registered raw metric outputs.

Collectors share one protected sink. The launcher separately owns node sessions
and closes its local sinks before those sessions within a shared time budget.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import TYPE_CHECKING

from tokenspeed.runtime.metrics.config import resolve_reporter_configs
from tokenspeed.runtime.metrics.events import (
    MetricConfig,
    MetricEvent,
    MetricEventSink,
    MetricNodeSession,
)
from tokenspeed.runtime.metrics.registry import (
    copy_metric_config,
    ensure_metric_reporters_loaded,
    get_metric_reporter,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)
_STARTUP_CLEANUP_TIMEOUT = 2.0


def _close_resources(
    resources: list[tuple[str, MetricEventSink | MetricNodeSession]],
    deadline: float,
) -> None:
    for name, resource in reversed(resources):
        try:
            resource.close(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            logger.exception(f"Metric reporter {name!r} failed during close")


class _ProtectedSink:
    """Serialize delivery and isolate an output that raises during emit.

    Disabled outputs remain owned until shutdown. Closing never runs from emit,
    and retained collector references cannot send to an already closed output.
    """

    def __init__(self, sinks: list[tuple[str, MetricEventSink]]) -> None:
        self._sinks = sinks
        self._disabled: set[str] = set()
        self._closed = False
        self._lock = threading.Lock()

    def emit(self, event: MetricEvent) -> None:
        with self._lock:
            if self._closed:
                return
            for name, sink in self._sinks:
                if name in self._disabled:
                    continue
                try:
                    sink.emit(event)
                except Exception:
                    self._disabled.add(name)
                    logger.exception(
                        f"Metric reporter {name!r} disabled after an emit failure"
                    )

    def close(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            if self._closed:
                return
            self._closed = True
        _close_resources(self._sinks, deadline)


class MetricsService:
    def __init__(self) -> None:
        self._prepared_configs: dict[str, MetricConfig] | None = None
        self._producer_configs: dict[str, MetricConfig] | None = None
        self._sessions: list[tuple[str, MetricNodeSession]] = []
        self._sink: _ProtectedSink | None = None
        self._started = False
        self._sinks_closed = False
        self._nodes_closed = False
        self._closed = False
        self._lock = threading.RLock()

    def prepare(self, args: ServerArgs) -> None:
        """Validate selected outputs before expensive model preparation starts."""
        with self._lock:
            if self._prepared_configs is None:
                self._prepared_configs = resolve_reporter_configs(args)

    def start(self, args: ServerArgs, ports: PortArgs) -> None:
        """Create this node's sessions before spawning producer processes.

        An explicit start may begin a new lifecycle after close. Merely looking
        up a sink or connecting after shutdown cannot recreate output resources.
        """
        with self._lock:
            if self._started and not self._closed:
                raise RuntimeError(
                    "A metrics service is already running in this process"
                )
            self.prepare(args)
            if self._closed:
                self._producer_configs = None
                self._sink = None
                self._sinks_closed = False
                self._nodes_closed = False
                self._closed = False
            self._started = True
            configs = {}
            try:
                for name, config in self._prepared_configs.items():
                    reporter = get_metric_reporter(name)
                    if reporter.start_node is None:
                        producer_config = config
                    else:
                        session = reporter.start_node(copy_metric_config(config))
                        self._sessions.append((name, session))
                        producer_config = session.producer_config
                    configs[name] = copy_metric_config(producer_config)
                self.connect(configs)
                # Each consumer receives a snapshot, never plugin-owned objects.
                ports.metrics_producer_configs = {
                    name: copy_metric_config(config) for name, config in configs.items()
                }
            except Exception:
                self.close(timeout=_STARTUP_CLEANUP_TIMEOUT)
                raise

    def connect(self, configs: dict[str, MetricConfig]) -> None:
        """Configure lazy process-local outputs from serializable producer data."""
        with self._lock:
            if self._sinks_closed or self._nodes_closed or self._closed:
                raise RuntimeError("Cannot connect a closed metrics service")
            copied = copy_metric_config(configs)
            for config in copied.values():
                if not isinstance(config, dict):
                    raise ValueError(
                        "Each metric reporter needs a configuration object"
                    )
            if self._producer_configs is not None:
                if self._producer_configs != copied:
                    raise RuntimeError("Cannot replace an active metrics configuration")
                return
            if copied:
                ensure_metric_reporters_loaded()
                for name in copied:
                    get_metric_reporter(name)
            self._producer_configs = copied

    def get_event_sink(self) -> MetricEventSink | None:
        """Return the process's shared output, creating it only during collection."""
        with self._lock:
            if self._sinks_closed or not self._producer_configs:
                return None
            if self._sink is None:
                sinks = []
                try:
                    for name, config in self._producer_configs.items():
                        sink = get_metric_reporter(name).create_sink(
                            copy_metric_config(config)
                        )
                        sinks.append((name, sink))
                except Exception:
                    deadline = time.monotonic() + _STARTUP_CLEANUP_TIMEOUT
                    self._sinks_closed = True
                    _close_resources(sinks, deadline)
                    self.close(timeout=max(0.0, deadline - time.monotonic()))
                    raise
                self._sink = _ProtectedSink(sinks)
            return self._sink

    def close_sinks(self, timeout: float) -> None:
        """Stop local collection and drain local sinks, leaving node sessions alive."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            self._sinks_closed = True
            sink, self._sink = self._sink, None
            if sink is not None:
                sink.close(timeout=max(0.0, deadline - time.monotonic()))

    def close_nodes(self, timeout: float) -> None:
        """Close launcher-owned sessions within the remaining time budget."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            self._nodes_closed = True
            sessions, self._sessions = self._sessions, []
            _close_resources(sessions, deadline)
            if self._sinks_closed:
                self._closed = True
                self._prepared_configs = None

    def close(self, timeout: float) -> None:
        """Close local sinks then node sessions within one shared time budget."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            self.close_sinks(timeout=max(0.0, deadline - time.monotonic()))
            self.close_nodes(timeout=max(0.0, deadline - time.monotonic()))


metrics_service = MetricsService()
atexit.register(metrics_service.close, timeout=2.0)
