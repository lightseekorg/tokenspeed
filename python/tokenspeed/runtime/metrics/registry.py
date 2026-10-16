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

"""Metric output registration using the existing runtime plugin loader."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from tokenspeed.runtime.metrics.events import (
    MetricConfig,
    MetricEventSink,
    MetricNodeSession,
)

logger = logging.getLogger(__name__)

METRIC_REPORTER_API_VERSION = 1


@dataclass(frozen=True)
class MetricReporter:
    """Creation callbacks registered without constructing any resources."""

    validate: Callable[[MetricConfig], MetricConfig]
    create_sink: Callable[[MetricConfig], MetricEventSink]
    start_node: Callable[[MetricConfig], MetricNodeSession] | None


_REPORTERS: dict[str, MetricReporter] = {}
_lock = threading.RLock()


def register_metric_reporter(
    name: str,
    *,
    validate: Callable[[MetricConfig], MetricConfig],
    create_sink: Callable[[MetricConfig], MetricEventSink],
    start_node: Callable[[MetricConfig], MetricNodeSession] | None,
    override: bool,
) -> None:
    """Register factories, participating in plugin failure rollback.

    Args:
        name: Reporter name selected in the server's metrics configuration.
        validate: Resource-free validation returning normalized JSON config.
        create_sink: Create one output per participating producer process.
        start_node: Create shared node resources, or None for a direct sink.
        override: Explicitly permit replacing a previously registered name.

    Raises:
        ValueError: The name is invalid, reserved, or already registered.
        TypeError: A supplied callback is not callable.
    """
    from tokenspeed.runtime.plugins.registry import record_external

    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("Metric reporter name must be a nonempty, unpadded string")
    if name == "prometheus":
        raise ValueError("Metric reporter name 'prometheus' is reserved")
    if not callable(validate) or not callable(create_sink):
        raise TypeError("Metric reporter validate and create_sink must be callable")
    if start_node is not None and not callable(start_node):
        raise TypeError("Metric reporter start_node must be callable or None")

    with _lock:
        previous = _REPORTERS.get(name)
        if previous is not None:
            if not override:
                raise ValueError(
                    f"Metric reporter {name!r} is already registered; "
                    "pass override=True to replace it"
                )
            logger.warning(f"Plugin registration overrides metric reporter {name!r}")
        _REPORTERS[name] = MetricReporter(validate, create_sink, start_node)

        def undo() -> None:
            with _lock:
                if previous is None:
                    del _REPORTERS[name]
                else:
                    _REPORTERS[name] = previous

        record_external("metric reporter", name, undo)


def get_metric_reporter(name: str) -> MetricReporter:
    """Look up an already loaded reporter, rejecting unknown selections."""
    with _lock:
        try:
            return _REPORTERS[name]
        except KeyError:
            raise ValueError(f"Unknown metric reporter {name!r}") from None


def ensure_metric_reporters_loaded() -> None:
    """Discover metric reporters through the existing runtime plugin loader."""
    from tokenspeed.runtime import plugins

    plugins.ensure_loaded()


def copy_metric_config(config: MetricConfig) -> MetricConfig:
    """Validate JSON-only config and return an independent recursive copy.

    Unlike a JSON round trip, this rejects non-string keys and tuples rather
    than silently converting them. Non-finite floats, cycles and runtime
    objects are rejected before configuration crosses a process boundary.
    Error messages omit configuration values, which may contain credentials.
    """
    if type(config) is not dict:
        raise ValueError("Metric configuration must be a JSON object")
    active: set[int] = set()

    def copy_value(value: object) -> object:
        value_type = type(value)
        if value is None or value_type in (str, bool, int):
            return value
        if value_type is float:
            if not math.isfinite(value):
                raise ValueError("Metric configuration requires finite floats")
            return value
        if value_type not in (dict, list):
            raise ValueError("Metric configuration must contain only JSON values")
        identity = id(value)
        if identity in active:
            raise ValueError("Metric configuration must not contain cyclic values")
        active.add(identity)
        try:
            if value_type is list:
                return [copy_value(item) for item in value]
            if any(type(key) is not str for key in value):
                raise ValueError("Metric configuration requires string object keys")
            return {key: copy_value(item) for key, item in value.items()}
        finally:
            active.remove(identity)

    return cast(MetricConfig, copy_value(config))
