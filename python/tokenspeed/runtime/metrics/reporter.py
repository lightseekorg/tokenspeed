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

"""Dispatch metric operations to Prometheus and raw event reporters.

Prometheus objects are created lazily, after its multiprocess directory is
configured. Plugins receive counter deltas, gauge snapshots and histogram samples;
batched histograms carry one per-token value and its token count as weight.
"""

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from tokenspeed.runtime.metrics.events import (
    MetricEvent,
    MetricEventSink,
    MetricOperation,
)

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

_GAUGE_SNAPSHOT_INTERVAL_SECONDS = 1.0


class Metric:
    """A metric binding shared by the selected reporters."""

    def __init__(
        self,
        *,
        name: str,
        operation: MetricOperation,
        labelnames: tuple[str, ...],
        labels: dict[str, str] | None,
        prometheus_metric: "Counter | Gauge | Histogram | None",
        event_sink: MetricEventSink | None,
    ) -> None:
        self._name = name
        self._operation = operation
        self._labelnames = labelnames
        self._labels = labels
        self._prometheus_metric = prometheus_metric
        self._event_sink = event_sink
        self._children: dict[tuple[str, ...], Metric] = {}
        self._last_gauge_value: float | None = None
        self._last_gauge_emit_at = 0.0

    def labels(self, **labels: str) -> "Metric":
        """Bind the declared labels, retaining all dynamic label values."""
        if set(labels) != set(self._labelnames):
            raise ValueError(f"Incorrect labels for metric {self._name}")
        if not self._labelnames:
            return self
        key = tuple(str(labels[name]) for name in self._labelnames)
        if key not in self._children:
            bound_labels = dict(zip(self._labelnames, key, strict=True))
            prometheus_metric = (
                self._prometheus_metric.labels(**bound_labels)
                if self._prometheus_metric is not None
                else None
            )
            self._children[key] = Metric(
                name=self._name,
                operation=self._operation,
                labelnames=self._labelnames,
                labels=bound_labels,
                prometheus_metric=prometheus_metric,
                event_sink=self._event_sink,
            )
        return self._children[key]

    def _check_operation(self, operation: MetricOperation) -> None:
        if self._operation != operation:
            raise ValueError(f"Metric {self._name} does not support {operation}")
        if self._labels is None:
            raise ValueError(f"Metric {self._name} requires labels")

    def _emit(self, value: float, *, weight: int) -> None:
        if self._event_sink is not None:
            self._event_sink.emit(
                MetricEvent(
                    timestamp_ns=time.time_ns(),
                    metric=self._name,
                    operation=self._operation,
                    value=float(value),
                    labels=dict(self._labels),
                    weight=weight,
                )
            )

    def inc(self, amount: float = 1) -> None:
        """Record a counter increment, never a cumulative snapshot."""
        self._check_operation("inc")
        if amount < 0:
            raise ValueError("Counters cannot decrease")
        if self._prometheus_metric is not None:
            self._prometheus_metric.inc(amount)
        self._emit(amount, weight=1)

    def set(self, value: float) -> None:
        """Update Prometheus; emit gauge changes or a snapshot on the next set."""
        self._check_operation("set")
        if self._prometheus_metric is not None:
            self._prometheus_metric.set(value)
        if self._event_sink is None:
            return
        value = float(value)
        now = time.monotonic()
        # Idle scheduler polls must not construct and queue duplicate events.
        if (
            value == self._last_gauge_value
            and now - self._last_gauge_emit_at < _GAUGE_SNAPSHOT_INTERVAL_SECONDS
        ):
            return
        self._emit(value, weight=1)
        self._last_gauge_value = value
        self._last_gauge_emit_at = now

    def observe(self, value: float) -> None:
        """Record one histogram observation."""
        self._check_operation("observe")
        if self._prometheus_metric is not None:
            self._prometheus_metric.observe(value)
        self._emit(value, weight=1)

    def observe_batched(self, interval: float, num_new_tokens: int) -> None:
        """Record a per-token interval with its token count as weight."""
        self._check_operation("observe")
        if num_new_tokens <= 0:
            return
        value = interval / num_new_tokens
        histogram = self._prometheus_metric
        if histogram is not None:
            # Retain the original total interval without multiply-back rounding.
            histogram._sum.inc(interval)
            for index, bound in enumerate(histogram._upper_bounds):
                if value <= bound:
                    histogram._buckets[index].inc(num_new_tokens)
                    break
        self._emit(value, weight=num_new_tokens)


class MetricFactory:
    """Create metric bindings without importing unused reporters."""

    def __init__(
        self,
        *,
        enabled: bool,
        reporters: list[str],
        event_sink: MetricEventSink | None = None,
        registry=None,
    ) -> None:
        raw_output = any(name != "prometheus" for name in reporters)
        self._event_sink = event_sink if enabled and raw_output else None
        if enabled and raw_output and event_sink is None:
            raise ValueError("Raw metric reporters require an event sink")
        self._registry = registry
        self._prometheus_classes = {}
        if enabled and "prometheus" in reporters:
            from prometheus_client import Counter, Gauge, Histogram

            self._prometheus_classes = {
                "inc": Counter,
                "set": Gauge,
                "observe": Histogram,
            }

    def _make(
        self,
        *,
        name: str,
        documentation: str,
        labelnames: Sequence[str],
        operation: MetricOperation,
        options: dict,
    ) -> Metric:
        prometheus_metric = None
        if self._prometheus_classes:
            registry_kw = (
                {"registry": self._registry} if self._registry is not None else {}
            )
            prometheus_metric = self._prometheus_classes[operation](
                name=name,
                documentation=documentation,
                labelnames=labelnames,
                **options,
                **registry_kw,
            )
        if operation == "inc" and not name.endswith("_total"):
            name = f"{name}_total"
        return Metric(
            name=name,
            operation=operation,
            labelnames=tuple(labelnames),
            labels=None if labelnames else {},
            prometheus_metric=prometheus_metric,
            event_sink=self._event_sink,
        )

    def counter(
        self, *, name: str, documentation: str, labelnames: Sequence[str]
    ) -> Metric:
        """Create a counter that records increments to the selected reporters."""
        return self._make(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            operation="inc",
            options={},
        )

    def gauge(
        self,
        *,
        name: str,
        documentation: str,
        labelnames: Sequence[str],
        multiprocess_mode: str,
    ) -> Metric:
        """Create a gauge, preserving its Prometheus multiprocess mode."""
        return self._make(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            operation="set",
            options={"multiprocess_mode": multiprocess_mode},
        )

    def histogram(
        self,
        *,
        name: str,
        documentation: str,
        labelnames: Sequence[str],
        buckets: Sequence[float],
    ) -> Metric:
        """Create a histogram using the supplied Prometheus bucket boundaries."""
        return self._make(
            name=name,
            documentation=documentation,
            labelnames=labelnames,
            operation="observe",
            options={"buckets": buckets},
        )
