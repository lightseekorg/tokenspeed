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

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.benchmark.graph import (
    GraphBenchmarkConfig,
    GraphBenchmarkError,
    GraphTimer,
    PreparedInvocation,
)


@dataclass
class _FakeEvent:
    index: int


@dataclass
class _FakeGraph:
    index: int


class _FakeBackend:
    def __init__(self, elapsed_times_ms: list[float] | None = None) -> None:
        self.elapsed_times_ms = elapsed_times_ms or [1.0, 2.0, 3.0]
        self.log: list[str] = []
        self._event_count = 0
        self._graph_count = 0
        self._elapsed_index = 0
        self._clock = 0.0
        self.fail_cleanup = False
        self.fail_replay = False

    def ensure_available(self) -> None:
        self.log.append("available")

    def monotonic(self) -> float:
        current = self._clock
        self._clock += 0.001
        return current

    def current_stream(self) -> object:
        self.log.append("current_stream")
        return "default"

    def new_stream(self) -> object:
        self.log.append("new_stream")
        return "benchmark"

    @contextmanager
    def use_stream(self, stream: object):
        self.log.append(f"stream_enter:{stream}")
        try:
            yield
        finally:
            self.log.append(f"stream_exit:{stream}")

    def wait_stream(self, stream: object, other: object) -> None:
        self.log.append(f"wait:{stream}:{other}")

    def synchronize_stream(self, stream: object) -> None:
        self.log.append(f"sync:{stream}")

    def new_graph(self) -> object:
        graph = _FakeGraph(self._graph_count)
        self._graph_count += 1
        self.log.append(f"new_graph:{graph.index}")
        return graph

    @contextmanager
    def capture(
        self,
        graph: object,
        stream: object,
        *,
        capture_error_mode: str,
    ):
        self.log.append(f"capture_enter:{graph.index}:{stream}:{capture_error_mode}")
        try:
            yield
        finally:
            self.log.append(f"capture_exit:{graph.index}")

    def replay(self, graph: object) -> None:
        self.log.append(f"replay:{graph.index}")
        if self.fail_replay:
            raise RuntimeError("replay broke")

    def cleanup_graph(self, graph: object) -> None:
        self.log.append(f"cleanup:{graph.index}")
        if self.fail_cleanup:
            raise RuntimeError("cleanup broke")

    def new_event(self) -> object:
        event = _FakeEvent(self._event_count)
        self._event_count += 1
        self.log.append(f"new_event:{event.index}")
        return event

    def record_event(self, event: object, stream: object) -> None:
        self.log.append(f"record:{event.index}:{stream}")

    def elapsed_time_ms(self, start: object, end: object) -> float:
        self.log.append(f"elapsed:{start.index}:{end.index}")
        elapsed = self.elapsed_times_ms[self._elapsed_index]
        self._elapsed_index += 1
        return elapsed


def _positions(log: list[str], prefix: str) -> list[int]:
    return [index for index, item in enumerate(log) if item.startswith(prefix)]


def test_measure_uses_fixed_graph_work_and_reports_statistics() -> None:
    backend = _FakeBackend([1.0, 3.0, 2.0])
    config = GraphBenchmarkConfig(
        calls_per_graph=10,
        eager_warmup_iterations=2,
        replay_warmup_iterations=2,
        measurement_blocks=3,
    )
    calls: list[str] = []

    def invoke() -> object:
        assert not torch.is_grad_enabled()
        value = object()
        calls.append("invoke")
        return value

    measurement = GraphTimer(config, backend=backend).measure(
        PreparedInvocation(invoke, repeat_safe=True)
    )

    assert len(calls) == 2 + 10
    assert measurement.samples_us == (100.0, 300.0, 200.0)
    assert measurement.median_us == 200.0
    assert measurement.p90_us == pytest.approx(280.0)
    assert measurement.min_us == 100.0
    assert measurement.max_us == 300.0
    assert measurement.relative_mad == 0.5
    assert measurement.calls_per_graph == 10
    assert measurement.eager_warmup_iterations == 2
    assert measurement.replay_warmup_iterations == 2
    assert measurement.warmup_time_ms == pytest.approx(1.0)
    assert measurement.capture_time_ms == pytest.approx(1.0)
    assert measurement.first_replay_time_ms == pytest.approx(1.0)
    assert measurement.measurement_time_ms == pytest.approx(1.0)

    assert "wait:benchmark:default" in backend.log
    assert "capture_enter:0:benchmark:global" in backend.log
    assert len(_positions(backend.log, "replay:")) == 1 + 2 + 3
    assert backend.log[-1] == "cleanup:0"


def test_events_are_primed_before_capture_and_reused_for_measurement() -> None:
    backend = _FakeBackend([1.0, 1.0])
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=2,
    )

    GraphTimer(config, backend=backend).measure(PreparedInvocation(lambda: None))

    capture_index = _positions(backend.log, "capture_enter:")[0]
    record_positions = _positions(backend.log, "record:")
    assert len(record_positions) == 8
    assert all(index < capture_index for index in record_positions[:4])
    assert all(index > capture_index for index in record_positions[4:])
    assert backend._event_count == 4


def test_timer_reuses_its_benchmark_stream_across_measurements() -> None:
    backend = _FakeBackend([1.0, 1.0])
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )
    timer = GraphTimer(config, backend=backend)

    timer.measure(PreparedInvocation(lambda: None))
    timer.measure(PreparedInvocation(lambda: None))

    assert backend.log.count("new_stream") == 1
    assert len(_positions(backend.log, "cleanup:")) == 2


def test_reset_is_outside_each_timed_interval() -> None:
    backend = _FakeBackend([1.0, 1.0])
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=1,
        measurement_blocks=2,
    )

    def reset() -> None:
        backend.log.append("reset")

    GraphTimer(config, backend=backend).measure(
        PreparedInvocation(lambda: None, reset=reset)
    )

    measurement_records = _positions(backend.log, "record:")[-4:]
    measurement_resets = _positions(backend.log, "reset")[-2:]
    assert measurement_resets[0] < measurement_records[0]
    assert measurement_records[1] < measurement_resets[1] < measurement_records[2]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calls_per_graph", 0),
        ("calls_per_graph", True),
        ("eager_warmup_iterations", 0),
        ("replay_warmup_iterations", -1),
        ("measurement_blocks", 0),
    ],
)
def test_config_rejects_invalid_counts(field: str, value: object) -> None:
    values = {
        "calls_per_graph": 1,
        "eager_warmup_iterations": 1,
        "replay_warmup_iterations": 0,
        "measurement_blocks": 1,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        GraphBenchmarkConfig(**values)


def test_timer_requires_explicit_configuration() -> None:
    with pytest.raises(TypeError):
        GraphBenchmarkConfig()
    with pytest.raises(TypeError):
        GraphTimer()


@pytest.mark.parametrize("repeat_safe", [False, "true"])
def test_multiple_calls_require_repeat_safe_invocation(repeat_safe: object) -> None:
    config = GraphBenchmarkConfig(
        calls_per_graph=2,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )

    with pytest.raises(GraphBenchmarkError, match="repeat_safe") as raised:
        GraphTimer(config, backend=_FakeBackend([1.0])).measure(
            PreparedInvocation(lambda: None, repeat_safe=repeat_safe)
        )

    assert raised.value.phase == "configuration"
    assert isinstance(raised.value.cause, ValueError)


def test_unavailable_device_is_an_environment_error() -> None:
    class _UnavailableBackend(_FakeBackend):
        def ensure_available(self) -> None:
            raise RuntimeError("no device")

    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )

    with pytest.raises(GraphBenchmarkError) as raised:
        GraphTimer(config, backend=_UnavailableBackend()).measure(
            PreparedInvocation(lambda: None)
        )

    assert raised.value.phase == "environment"
    assert isinstance(raised.value.cause, RuntimeError)


@pytest.mark.parametrize("sample", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_event_sample_is_a_measurement_error(sample: float) -> None:
    backend = _FakeBackend([sample])
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )

    with pytest.raises(GraphBenchmarkError) as raised:
        GraphTimer(config, backend=backend).measure(PreparedInvocation(lambda: None))

    assert raised.value.phase == "measurement"
    assert isinstance(raised.value.cause, ValueError)
    assert backend.log[-1] == "cleanup:0"


def test_capture_failure_is_typed_and_partial_graph_is_cleaned() -> None:
    backend = _FakeBackend([1.0])
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )
    invocation_count = 0

    def invoke() -> None:
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count == 2:
            raise RuntimeError("capture broke")

    with pytest.raises(GraphBenchmarkError) as raised:
        GraphTimer(config, backend=backend).measure(PreparedInvocation(invoke))

    assert raised.value.phase == "capture"
    assert isinstance(raised.value.cause, RuntimeError)
    assert backend.log[-1] == "cleanup:0"


def test_first_replay_failure_is_typed_and_graph_is_cleaned() -> None:
    backend = _FakeBackend([1.0])
    backend.fail_replay = True
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )

    with pytest.raises(GraphBenchmarkError) as raised:
        GraphTimer(config, backend=backend).measure(PreparedInvocation(lambda: None))

    assert raised.value.phase == "first_replay"
    assert isinstance(raised.value.cause, RuntimeError)
    assert raised.value.__cause__ is raised.value.cause
    assert backend.log[-1] == "cleanup:0"


def test_failed_run_discards_the_persistent_stream() -> None:
    backend = _FakeBackend([1.0])
    backend.fail_replay = True
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )
    timer = GraphTimer(config, backend=backend)

    with pytest.raises(GraphBenchmarkError):
        timer.measure(PreparedInvocation(lambda: None))

    backend.fail_replay = False
    timer.measure(PreparedInvocation(lambda: None))

    assert backend.log.count("new_stream") == 2


def test_cleanup_failure_is_typed() -> None:
    backend = _FakeBackend([1.0])
    backend.fail_cleanup = True
    config = GraphBenchmarkConfig(
        calls_per_graph=1,
        eager_warmup_iterations=1,
        replay_warmup_iterations=0,
        measurement_blocks=1,
    )

    with pytest.raises(GraphBenchmarkError) as raised:
        GraphTimer(config, backend=backend).measure(PreparedInvocation(lambda: None))

    assert raised.value.phase == "cleanup"
    assert isinstance(raised.value.cause, RuntimeError)
