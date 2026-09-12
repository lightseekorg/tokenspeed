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

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import (
    GraphBenchmarkConfig,
    GraphBenchmarkError,
    GraphMeasurement,
    GraphTimer,
    PreparedInvocation,
)
from tokenspeed_kernel.benchmark.validation import (
    OutputValidationSpec,
    ValidationDatum,
    get_output_validator,
    validate_output,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelSpec

__all__ = [
    "BenchmarkCaseError",
    "BenchmarkRequest",
    "BenchmarkStatus",
    "KernelBenchmarkHarness",
    "KernelBenchmarkResult",
    "PreparedBenchmark",
    "PreparedValidation",
    "ValidationInvocation",
    "set_benchmark_generator",
]


class BenchmarkStatus(str, Enum):
    """Outcome of one benchmark request."""

    SUCCESS = "success"
    NOT_APPLICABLE = "not_applicable"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    REGISTRATION_MISSING = "registration_missing"
    INVALID_CASE = "invalid_case"
    ENVIRONMENT_INVALID = "environment_invalid"
    SETUP_FAILURE = "setup_failure"
    CAPTURE_FAILURE = "capture_failure"
    EXECUTION_FAILURE = "execution_failure"
    CORRECTNESS_FAILURE = "correctness_failure"


class BenchmarkCaseError(RuntimeError):
    """Expected preparation failure with a machine-readable outcome."""

    def __init__(self, status: BenchmarkStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class BenchmarkRequest:
    """Complete operation request accepted by the benchmark harness."""

    family: str
    mode: str
    parameters: dict[str, Any]
    solution: str | None
    registration: str | None
    seed: int
    definition_version: int

    def __post_init__(self) -> None:
        if self.solution is not None and self.registration is not None:
            raise ValueError("solution and registration are mutually exclusive")
        object.__setattr__(self, "parameters", dict(self.parameters))

    @property
    def selection_mode(self) -> str:
        if self.registration is not None:
            return "registration"
        if self.solution is not None:
            return "solution"
        return "normal"


@dataclass(frozen=True)
class ValidationInvocation:
    """Candidate and reference calls over one set of fresh inputs."""

    candidate: Callable[[], tuple[object, ...]]
    reference: Callable[[], tuple[object, ...]]


@dataclass(frozen=True)
class PreparedValidation:
    """Operation-owned correctness specification and fresh-input factory."""

    output_specs: Callable[[], Sequence[OutputValidationSpec | None]]
    prepare_run: Callable[[int], ValidationInvocation]


@dataclass(frozen=True)
class PreparedBenchmark:
    """Operation-owned state handed to the shared graph timer."""

    registration: KernelSpec
    invocation: PreparedInvocation
    parameters: dict[str, Any]
    validation: PreparedValidation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))


BenchmarkGenerator = Callable[[BenchmarkRequest, PlatformInfo], PreparedBenchmark]
_BENCHMARK_GENERATORS: dict[tuple[str, str], BenchmarkGenerator] = {}


def set_benchmark_generator(
    family: str,
    mode: str,
    generator: BenchmarkGenerator,
) -> None:
    """Associate an operation family and mode with a benchmark generator.

    Args:
        family: Operation family accepted by the generator.
        mode: Operation mode within ``family`` accepted by the generator.
        generator: Callable that receives a complete benchmark request and the
            detected platform, then returns the registration, invocation,
            parameters, and optional correctness work for the shared harness.

    Returns:
        None.
    """

    _BENCHMARK_GENERATORS[(family, mode)] = generator


def _load_builtin_generators() -> None:
    from tokenspeed_kernel.benchmark.generators.gemm import prepare_dense_bmm

    _BENCHMARK_GENERATORS.setdefault(("gemm", "bmm"), prepare_dense_bmm)


@dataclass(frozen=True)
class KernelBenchmarkResult:
    """Structured outcome for one registration-level benchmark request."""

    status: BenchmarkStatus
    family: str
    mode: str
    parameters: dict[str, Any]
    selection_mode: str
    requested_solution: str | None
    requested_registration: str | None
    seed: int
    definition_version: int
    platform_vendor: str
    platform_arch: str
    device_name: str
    registration_name: str | None = None
    solution: str | None = None
    timing_mode: str = "graph_replay"
    metric: str = "device_time_per_invocation"
    unit: str = "us"
    samples_us: tuple[float, ...] = ()
    median_us: float | None = None
    p90_us: float | None = None
    min_us: float | None = None
    max_us: float | None = None
    relative_mad: float | None = None
    calls_per_graph: int = 0
    eager_warmup_iterations: int = 0
    replay_warmup_iterations: int = 0
    measurement_blocks: int = 0
    setup_time_ms: float = 0.0
    correctness_time_ms: float = 0.0
    warmup_time_ms: float = 0.0
    capture_time_ms: float = 0.0
    first_replay_time_ms: float = 0.0
    measurement_time_ms: float = 0.0
    total_time_ms: float = 0.0
    correctness: dict[str, Any] | None = None
    error_phase: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = 1

    @property
    def succeeded(self) -> bool:
        return self.status is BenchmarkStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["samples_us"] = list(self.samples_us)
        return payload


_GRAPH_STATUS_BY_PHASE = {
    "configuration": BenchmarkStatus.INVALID_CASE,
    "environment": BenchmarkStatus.ENVIRONMENT_INVALID,
    "warmup": BenchmarkStatus.SETUP_FAILURE,
    "capture": BenchmarkStatus.CAPTURE_FAILURE,
    "first_replay": BenchmarkStatus.EXECUTION_FAILURE,
    "measurement": BenchmarkStatus.EXECUTION_FAILURE,
    "cleanup": BenchmarkStatus.EXECUTION_FAILURE,
}


class KernelBenchmarkHarness:
    """Prepare and measure one registered operation with graph replay."""

    def __init__(
        self,
        config: GraphBenchmarkConfig | None,
        *,
        timer: GraphTimer | None,
        platform_provider: Callable[[], PlatformInfo],
    ) -> None:
        if timer is not None and config is not None:
            raise ValueError("config cannot be provided with an explicit timer")
        if timer is None:
            if config is None:
                raise ValueError("config is required when no timer is provided")
            timer = GraphTimer(config)
        self._timer = timer
        self._platform_provider = platform_provider

    def run(self, request: BenchmarkRequest) -> KernelBenchmarkResult:
        """Run one request and return a success or explicit failure result."""

        started = time.perf_counter()
        try:
            platform = self._platform_provider()
        except Exception as exc:  # noqa: BLE001 - failures are result data
            return self._failure_result(
                request,
                None,
                BenchmarkStatus.ENVIRONMENT_INVALID,
                started,
                phase="environment",
                error=exc,
            )

        _load_builtin_generators()
        generator = _BENCHMARK_GENERATORS.get((request.family, request.mode))
        if generator is None:
            error = KeyError(
                f"No benchmark generator registered for {request.family}.{request.mode}"
            )
            return self._failure_result(
                request,
                platform,
                BenchmarkStatus.INVALID_CASE,
                started,
                phase="preparation",
                error=error,
            )

        try:
            prepared = generator(request, platform)
            self._validate_prepared(request, prepared)
        except BenchmarkCaseError as exc:
            return self._failure_result(
                request,
                platform,
                exc.status,
                started,
                phase="preparation",
                error=exc,
            )
        except Exception as exc:  # noqa: BLE001 - failures are result data
            return self._failure_result(
                request,
                platform,
                BenchmarkStatus.SETUP_FAILURE,
                started,
                phase="preparation",
                error=exc,
            )

        setup_time_ms = (time.perf_counter() - started) * 1000.0
        correctness: dict[str, Any] | None = None
        correctness_time_ms = 0.0
        if prepared.validation is not None:
            correctness_started = time.perf_counter()
            try:
                correctness = self._run_validation(prepared.validation)
            except Exception as exc:  # noqa: BLE001 - failures are result data
                correctness_time_ms = (
                    time.perf_counter() - correctness_started
                ) * 1000.0
                return self._failure_result(
                    request,
                    platform,
                    BenchmarkStatus.CORRECTNESS_FAILURE,
                    started,
                    phase="validation",
                    error=exc,
                    prepared=prepared,
                    setup_time_ms=setup_time_ms,
                    correctness_time_ms=correctness_time_ms,
                )
            correctness_time_ms = (time.perf_counter() - correctness_started) * 1000.0
            assert correctness is not None
            if not correctness["passed"]:
                error = RuntimeError(self._correctness_failure_message(correctness))
                return self._failure_result(
                    request,
                    platform,
                    BenchmarkStatus.CORRECTNESS_FAILURE,
                    started,
                    phase="validation",
                    error=error,
                    prepared=prepared,
                    setup_time_ms=setup_time_ms,
                    correctness=correctness,
                    correctness_time_ms=correctness_time_ms,
                )

        try:
            measurement = self._timer.measure(prepared.invocation)
        except GraphBenchmarkError as exc:
            status = _GRAPH_STATUS_BY_PHASE.get(
                exc.phase, BenchmarkStatus.EXECUTION_FAILURE
            )
            return self._failure_result(
                request,
                platform,
                status,
                started,
                phase=exc.phase,
                error=exc.cause or exc,
                prepared=prepared,
                setup_time_ms=setup_time_ms,
                correctness=correctness,
                correctness_time_ms=correctness_time_ms,
            )
        except Exception as exc:  # noqa: BLE001 - failures are result data
            return self._failure_result(
                request,
                platform,
                BenchmarkStatus.EXECUTION_FAILURE,
                started,
                phase="measurement",
                error=exc,
                prepared=prepared,
                setup_time_ms=setup_time_ms,
                correctness=correctness,
                correctness_time_ms=correctness_time_ms,
            )

        return self._success_result(
            request,
            platform,
            prepared,
            measurement,
            setup_time_ms,
            correctness,
            correctness_time_ms,
            started,
        )

    @classmethod
    def _run_validation(
        cls,
        validation: PreparedValidation | None,
    ) -> dict[str, Any] | None:
        specs = cls._resolve_validation_specs(validation)
        if not specs:
            return None
        assert validation is not None

        maximum_runs = max(spec.runs for spec in specs if spec is not None)
        output_data: list[list[ValidationDatum]] = [[] for _ in specs]
        with torch.no_grad():
            for run_index in range(maximum_runs):
                invocation = validation.prepare_run(run_index)

                expected = cls._require_output_tuple(
                    invocation.reference(),
                    f"reference correctness run {run_index}",
                )
                actual = cls._require_output_tuple(
                    invocation.candidate(),
                    f"candidate correctness run {run_index}",
                )
                if len(expected) != len(specs) or len(actual) != len(specs):
                    raise ValueError(
                        f"correctness run {run_index} returned candidate/reference "
                        f"output counts {len(actual)}/{len(expected)}; "
                        f"expected {len(specs)}"
                    )

                for output_index, spec in enumerate(specs):
                    if spec is None or run_index >= spec.runs:
                        continue
                    output_data[output_index].append(
                        ValidationDatum(
                            actual=actual[output_index],
                            expected=expected[output_index],
                        )
                    )

        return cls._evaluate_validation(
            specs,
            tuple(tuple(data) for data in output_data),
        )

    @staticmethod
    def _resolve_validation_specs(
        validation: PreparedValidation | None,
    ) -> tuple[OutputValidationSpec | None, ...]:
        if validation is None:
            return ()
        specs = tuple(validation.output_specs())
        if not specs or not any(spec is not None for spec in specs):
            raise ValueError("output_specs must select at least one output")
        for spec in specs:
            if spec is None:
                continue
            get_output_validator(spec.validator)
        return specs

    @staticmethod
    def _require_output_tuple(value: object, context: str) -> tuple[object, ...]:
        if not isinstance(value, tuple):
            raise TypeError(f"{context} must return an output tuple")
        return value

    @staticmethod
    def _evaluate_validation(
        specs: tuple[OutputValidationSpec | None, ...],
        output_data: tuple[tuple[ValidationDatum, ...], ...],
    ) -> dict[str, Any]:
        outputs: list[dict[str, Any]] = []
        passed = True
        maximum_runs = max(spec.runs for spec in specs if spec is not None)
        for index, spec in enumerate(specs):
            if spec is None:
                outputs.append(
                    {
                        "index": index,
                        "validator": None,
                        "runs": 0,
                        "kwargs": {},
                        "passed": None,
                        "diagnostic": None,
                    }
                )
                continue
            outcome = validate_output(spec, output_data[index])
            passed = passed and outcome.passed
            outputs.append(
                {
                    "index": index,
                    "validator": spec.validator,
                    "runs": spec.runs,
                    "kwargs": dict(spec.kwargs),
                    "passed": outcome.passed,
                    "diagnostic": outcome.diagnostic,
                }
            )
        return {"passed": passed, "runs": maximum_runs, "outputs": outputs}

    @staticmethod
    def _correctness_failure_message(correctness: dict[str, Any]) -> str:
        failures = [
            f"output {output['index']} ({output['validator']}): "
            f"{output['diagnostic'] or 'validation failed'}"
            for output in correctness["outputs"]
            if output["passed"] is False
        ]
        return "; ".join(failures) or "correctness validation failed"

    @staticmethod
    def _validate_prepared(
        request: BenchmarkRequest,
        prepared: PreparedBenchmark,
    ) -> None:
        spec = prepared.registration
        if (spec.family, spec.mode) != (request.family, request.mode):
            raise BenchmarkCaseError(
                BenchmarkStatus.INVALID_CASE,
                f"Generator prepared {spec.family}.{spec.mode} for "
                f"{request.family}.{request.mode}",
            )
        if request.solution is not None and spec.solution != request.solution:
            raise BenchmarkCaseError(
                BenchmarkStatus.INVALID_CASE,
                f"Generator selected solution {spec.solution!r}; "
                f"requested {request.solution!r}",
            )
        if request.registration is not None and spec.name != request.registration:
            raise BenchmarkCaseError(
                BenchmarkStatus.INVALID_CASE,
                f"Generator selected registration {spec.name!r}; "
                f"requested {request.registration!r}",
            )

    @staticmethod
    def _base_fields(
        request: BenchmarkRequest,
        platform: PlatformInfo | None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "family": request.family,
            "mode": request.mode,
            "parameters": dict(
                parameters if parameters is not None else request.parameters
            ),
            "selection_mode": request.selection_mode,
            "requested_solution": request.solution,
            "requested_registration": request.registration,
            "seed": request.seed,
            "definition_version": request.definition_version,
            "platform_vendor": platform.vendor if platform is not None else "",
            "platform_arch": platform.arch if platform is not None else "",
            "device_name": platform.device_name if platform is not None else "",
        }

    def _success_result(
        self,
        request: BenchmarkRequest,
        platform: PlatformInfo,
        prepared: PreparedBenchmark,
        measurement: GraphMeasurement,
        setup_time_ms: float,
        correctness: dict[str, Any] | None,
        correctness_time_ms: float,
        started: float,
    ) -> KernelBenchmarkResult:
        return KernelBenchmarkResult(
            status=BenchmarkStatus.SUCCESS,
            **self._base_fields(request, platform, prepared.parameters),
            registration_name=prepared.registration.name,
            solution=prepared.registration.solution,
            samples_us=measurement.samples_us,
            median_us=measurement.median_us,
            p90_us=measurement.p90_us,
            min_us=measurement.min_us,
            max_us=measurement.max_us,
            relative_mad=measurement.relative_mad,
            calls_per_graph=measurement.calls_per_graph,
            eager_warmup_iterations=measurement.eager_warmup_iterations,
            replay_warmup_iterations=measurement.replay_warmup_iterations,
            measurement_blocks=len(measurement.samples_us),
            setup_time_ms=setup_time_ms,
            correctness_time_ms=correctness_time_ms,
            warmup_time_ms=measurement.warmup_time_ms,
            capture_time_ms=measurement.capture_time_ms,
            first_replay_time_ms=measurement.first_replay_time_ms,
            measurement_time_ms=measurement.measurement_time_ms,
            total_time_ms=(time.perf_counter() - started) * 1000.0,
            correctness=correctness,
        )

    def _failure_result(
        self,
        request: BenchmarkRequest,
        platform: PlatformInfo | None,
        status: BenchmarkStatus,
        started: float,
        *,
        phase: str,
        error: BaseException,
        prepared: PreparedBenchmark | None = None,
        setup_time_ms: float = 0.0,
        correctness: dict[str, Any] | None = None,
        correctness_time_ms: float = 0.0,
    ) -> KernelBenchmarkResult:
        return KernelBenchmarkResult(
            status=status,
            **self._base_fields(
                request,
                platform,
                prepared.parameters if prepared is not None else None,
            ),
            registration_name=(
                prepared.registration.name if prepared is not None else None
            ),
            solution=prepared.registration.solution if prepared is not None else None,
            setup_time_ms=setup_time_ms,
            correctness=correctness,
            correctness_time_ms=correctness_time_ms,
            total_time_ms=(time.perf_counter() - started) * 1000.0,
            error_phase=phase,
            error_type=type(error).__name__,
            error_message=str(error),
        )
