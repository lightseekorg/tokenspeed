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

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

__all__ = [
    "GraphBenchmarkConfig",
    "GraphBenchmarkError",
    "GraphMeasurement",
    "GraphTimer",
    "PreparedInvocation",
]

GraphBenchmarkPhase = Literal[
    "configuration",
    "environment",
    "warmup",
    "capture",
    "first_replay",
    "measurement",
    "cleanup",
]


@dataclass(frozen=True)
class GraphBenchmarkConfig:
    """Fixed work and sampling configuration for a graph benchmark."""

    calls_per_graph: int
    eager_warmup_iterations: int
    replay_warmup_iterations: int
    measurement_blocks: int

    def __post_init__(self) -> None:
        _validate_positive_int("calls_per_graph", self.calls_per_graph)
        _validate_positive_int("eager_warmup_iterations", self.eager_warmup_iterations)
        _validate_nonnegative_int(
            "replay_warmup_iterations", self.replay_warmup_iterations
        )
        _validate_positive_int("measurement_blocks", self.measurement_blocks)


@dataclass(frozen=True)
class PreparedInvocation:
    """A graph-capturable call and its reset hook.

    ``reset`` restores static device state before each eager call, capture, and
    replay. It is always called outside the timed event interval. Setting
    ``repeat_safe`` confirms that multiple calls may be captured in one graph.
    """

    invoke: Callable[[], object]
    reset: Callable[[], None] | None = None
    repeat_safe: bool = False


@dataclass(frozen=True)
class GraphMeasurement:
    """Device timing samples and aggregate statistics for one invocation."""

    samples_us: tuple[float, ...]
    median_us: float
    p90_us: float
    min_us: float
    max_us: float
    relative_mad: float
    calls_per_graph: int
    eager_warmup_iterations: int
    replay_warmup_iterations: int
    warmup_time_ms: float
    capture_time_ms: float
    first_replay_time_ms: float
    measurement_time_ms: float


class GraphBenchmarkError(RuntimeError):
    """Failure in a named graph benchmark phase."""

    def __init__(
        self,
        phase: GraphBenchmarkPhase,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(f"{phase}: {message}")
        self.phase = phase
        self.cause = cause


class _GraphBackend(Protocol):
    def ensure_available(self) -> None: ...

    def monotonic(self) -> float: ...

    def current_stream(self) -> object: ...

    def new_stream(self) -> object: ...

    def use_stream(self, stream: object) -> AbstractContextManager[None]: ...

    def wait_stream(self, stream: object, other: object) -> None: ...

    def synchronize_stream(self, stream: object) -> None: ...

    def new_graph(self) -> object: ...

    def capture(
        self,
        graph: object,
        stream: object,
        *,
        capture_error_mode: str,
    ) -> AbstractContextManager[None]: ...

    def replay(self, graph: object) -> None: ...

    def cleanup_graph(self, graph: object) -> None: ...

    def new_event(self) -> object: ...

    def record_event(self, event: object, stream: object) -> None: ...

    def elapsed_time_ms(self, start: object, end: object) -> float: ...


class _TorchCudaBackend:
    def ensure_available(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("graph benchmarks require a CUDA-compatible device")

    def monotonic(self) -> float:
        return time.perf_counter()

    def current_stream(self) -> object:
        return torch.cuda.current_stream()

    def new_stream(self) -> object:
        return torch.cuda.Stream()

    def use_stream(self, stream: object) -> AbstractContextManager[None]:
        return torch.cuda.stream(stream)

    def wait_stream(self, stream: object, other: object) -> None:
        stream.wait_stream(other)

    def synchronize_stream(self, stream: object) -> None:
        stream.synchronize()

    def new_graph(self) -> object:
        return torch.cuda.CUDAGraph()

    def capture(
        self,
        graph: object,
        stream: object,
        *,
        capture_error_mode: str,
    ) -> AbstractContextManager[None]:
        return torch.cuda.graph(
            graph,
            stream=stream,
            capture_error_mode=capture_error_mode,
        )

    def replay(self, graph: object) -> None:
        graph.replay()

    def cleanup_graph(self, graph: object) -> None:
        graph.reset()

    def new_event(self) -> object:
        return torch.cuda.Event(enable_timing=True)

    def record_event(self, event: object, stream: object) -> None:
        event.record(stream)

    def elapsed_time_ms(self, start: object, end: object) -> float:
        return float(start.elapsed_time(end))


class GraphTimer:
    """Measure an opaque device invocation through graph replay."""

    def __init__(
        self,
        config: GraphBenchmarkConfig,
        *,
        backend: _GraphBackend | None = None,
    ) -> None:
        self.config = config
        self._backend = backend or _TorchCudaBackend()
        self._stream: object | None = None

    def measure(self, prepared: PreparedInvocation) -> GraphMeasurement:
        """Capture and measure one prepared invocation."""
        if self.config.calls_per_graph > 1 and prepared.repeat_safe is not True:
            cause = ValueError(
                "calls_per_graph > 1 requires an invocation with repeat_safe=True"
            )
            raise GraphBenchmarkError(
                "configuration",
                "calls_per_graph exceeds one while repeat_safe is false",
                cause=cause,
            ) from cause

        backend = self._backend
        try:
            backend.ensure_available()
        except Exception as error:
            raise GraphBenchmarkError(
                "environment",
                "graph timing is unavailable in this environment",
                cause=error,
            ) from error

        graph: object | None = None
        active_error: BaseException | None = None

        try:
            with torch.no_grad():
                warmup_started = backend.monotonic()
                stream, event_pairs = self._warm_up(prepared)
                warmup_time_ms = _elapsed_wall_ms(backend, warmup_started)

                capture_started = backend.monotonic()
                graph = self._capture(prepared, stream)
                capture_time_ms = _elapsed_wall_ms(backend, capture_started)

                first_replay_started = backend.monotonic()
                self._first_replay(prepared, graph, stream)
                first_replay_time_ms = _elapsed_wall_ms(backend, first_replay_started)

                measurement_started = backend.monotonic()
                samples_us = self._measure_replays(
                    prepared,
                    graph,
                    stream,
                    event_pairs,
                )
                measurement_time_ms = _elapsed_wall_ms(backend, measurement_started)

                return _summarize(
                    samples_us,
                    calls_per_graph=self.config.calls_per_graph,
                    eager_warmup_iterations=self.config.eager_warmup_iterations,
                    replay_warmup_iterations=self.config.replay_warmup_iterations,
                    warmup_time_ms=warmup_time_ms,
                    capture_time_ms=capture_time_ms,
                    first_replay_time_ms=first_replay_time_ms,
                    measurement_time_ms=measurement_time_ms,
                )
        except GraphBenchmarkError as error:
            active_error = error
            raise
        except Exception as error:
            active_error = error
            raise GraphBenchmarkError(
                "warmup",
                "failed to initialize the graph benchmark",
                cause=error,
            ) from error
        finally:
            if active_error is not None and self._stream is not None:
                failed_stream = self._stream
                self._stream = None
                try:
                    backend.synchronize_stream(failed_stream)
                except Exception as error:  # noqa: BLE001
                    active_error.add_note(f"failed stream drain also failed: {error}")
            if graph is not None:
                try:
                    backend.cleanup_graph(graph)
                except Exception as error:
                    self._stream = None
                    if active_error is not None:
                        active_error.add_note(f"graph cleanup also failed: {error}")
                    else:
                        raise GraphBenchmarkError(
                            "cleanup",
                            "failed to release the captured graph",
                            cause=error,
                        ) from error

    def _warm_up(
        self,
        prepared: PreparedInvocation,
    ) -> tuple[object, list[tuple[object, object]]]:
        backend = self._backend
        try:
            source_stream = backend.current_stream()
            if self._stream is None:
                self._stream = backend.new_stream()
            stream = self._stream
            backend.wait_stream(stream, source_stream)
            event_pairs = [
                (backend.new_event(), backend.new_event())
                for _ in range(self.config.measurement_blocks)
            ]

            with backend.use_stream(stream):
                for _ in range(self.config.eager_warmup_iterations):
                    _reset(prepared)
                    prepared.invoke()
            backend.synchronize_stream(stream)

            # Event allocation can be lazy. Record every event before capture so
            # allocation and initialization are never part of a timed block.
            with backend.use_stream(stream):
                for start, end in event_pairs:
                    backend.record_event(start, stream)
                    backend.record_event(end, stream)
            backend.synchronize_stream(stream)
            return stream, event_pairs
        except Exception as error:
            raise GraphBenchmarkError(
                "warmup",
                "eager warmup or event initialization failed",
                cause=error,
            ) from error

    def _capture(
        self,
        prepared: PreparedInvocation,
        stream: object,
    ) -> object:
        backend = self._backend
        graph: object | None = None
        try:
            with backend.use_stream(stream):
                _reset(prepared)
            backend.synchronize_stream(stream)

            graph = backend.new_graph()
            with backend.capture(
                graph,
                stream,
                capture_error_mode="global",
            ):
                for _ in range(self.config.calls_per_graph):
                    prepared.invoke()
            backend.synchronize_stream(stream)
            return graph
        except Exception as error:
            if graph is not None:
                try:
                    backend.synchronize_stream(stream)
                except Exception as synchronize_error:  # noqa: BLE001
                    error.add_note(
                        f"partial graph stream drain failed: {synchronize_error}"
                    )
                self._stream = None
                try:
                    backend.cleanup_graph(graph)
                except Exception as cleanup_error:  # noqa: BLE001
                    error.add_note(f"partial graph cleanup failed: {cleanup_error}")
            raise GraphBenchmarkError(
                "capture",
                "process-wide graph capture failed",
                cause=error,
            ) from error

    def _first_replay(
        self,
        prepared: PreparedInvocation,
        graph: object,
        stream: object,
    ) -> None:
        backend = self._backend
        try:
            with backend.use_stream(stream):
                _reset(prepared)
                backend.replay(graph)
            backend.synchronize_stream(stream)
        except Exception as error:
            raise GraphBenchmarkError(
                "first_replay",
                "the first untimed graph replay failed",
                cause=error,
            ) from error

    def _measure_replays(
        self,
        prepared: PreparedInvocation,
        graph: object,
        stream: object,
        event_pairs: list[tuple[object, object]],
    ) -> tuple[float, ...]:
        backend = self._backend
        try:
            with backend.use_stream(stream):
                for _ in range(self.config.replay_warmup_iterations):
                    _reset(prepared)
                    backend.replay(graph)
            backend.synchronize_stream(stream)

            with backend.use_stream(stream):
                for start, end in event_pairs:
                    _reset(prepared)
                    backend.record_event(start, stream)
                    backend.replay(graph)
                    backend.record_event(end, stream)
            backend.synchronize_stream(stream)

            samples: list[float] = []
            for start, end in event_pairs:
                elapsed_ms = backend.elapsed_time_ms(start, end)
                sample_us = elapsed_ms * 1000.0 / self.config.calls_per_graph
                if not math.isfinite(sample_us) or sample_us <= 0.0:
                    raise ValueError(
                        "device event produced an invalid per-invocation sample "
                        f"({sample_us!r} us); increase calls_per_graph if the "
                        "operation is below the event timer resolution"
                    )
                samples.append(sample_us)
            return tuple(samples)
        except Exception as error:
            raise GraphBenchmarkError(
                "measurement",
                "graph replay measurement failed",
                cause=error,
            ) from error


def _reset(prepared: PreparedInvocation) -> None:
    if prepared.reset is not None:
        prepared.reset()


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _elapsed_wall_ms(backend: _GraphBackend, started: float) -> float:
    elapsed = (backend.monotonic() - started) * 1000.0
    return max(0.0, elapsed)


def _percentile(sorted_values: tuple[float, ...], percentile: float) -> float:
    rank = (len(sorted_values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _summarize(
    samples_us: tuple[float, ...],
    *,
    calls_per_graph: int,
    eager_warmup_iterations: int,
    replay_warmup_iterations: int,
    warmup_time_ms: float,
    capture_time_ms: float,
    first_replay_time_ms: float,
    measurement_time_ms: float,
) -> GraphMeasurement:
    sorted_samples = tuple(sorted(samples_us))
    median_us = float(statistics.median(sorted_samples))
    absolute_deviations = tuple(
        sorted(abs(sample - median_us) for sample in sorted_samples)
    )
    mad_us = float(statistics.median(absolute_deviations))
    return GraphMeasurement(
        samples_us=samples_us,
        median_us=median_us,
        p90_us=float(_percentile(sorted_samples, 90.0)),
        min_us=sorted_samples[0],
        max_us=sorted_samples[-1],
        relative_mad=mad_us / median_us,
        calls_per_graph=calls_per_graph,
        eager_warmup_iterations=eager_warmup_iterations,
        replay_warmup_iterations=replay_warmup_iterations,
        warmup_time_ms=warmup_time_ms,
        capture_time_ms=capture_time_ms,
        first_replay_time_ms=first_replay_time_ms,
        measurement_time_ms=measurement_time_ms,
    )
