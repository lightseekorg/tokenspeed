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

import pytest
from tokenspeed_kernel.benchmark.graph import (
    GraphBenchmarkError,
    GraphMeasurement,
    PreparedInvocation,
)
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    KernelBenchmarkHarness,
    PreparedBenchmark,
    PreparedValidation,
    ValidationInvocation,
    set_benchmark_generator,
)
from tokenspeed_kernel.benchmark.validation import (
    OutputValidationSpec,
    ValidationOutcome,
    set_output_validator,
)
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo
from tokenspeed_kernel.registry import KernelSpec
from utils import FakeTimer


class _FailingTimer:
    def __init__(self, phase: str) -> None:
        self.phase = phase

    def measure(
        self,
        prepared: PreparedInvocation,
        *,
        cold_cache: bool,
        measurement_blocks: int,
    ) -> GraphMeasurement:
        _ = prepared, cold_cache, measurement_blocks
        cause = RuntimeError("timing failed")
        raise GraphBenchmarkError(self.phase, "timing failed", cause=cause)


def _platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="amd",
        arch_version=ArchVersion(9, 5),
        device_name="test device",
        device_count=1,
        total_memory=1,
        memory_bandwidth=1.0,
        sm_count=1,
        max_threads_per_sm=1,
        max_shared_memory_per_sm=1,
    )


def _request(family: str) -> BenchmarkRequest:
    return BenchmarkRequest(
        family=family,
        mode="test",
        parameters={"size": 8},
        solution="test_solution",
        registration=None,
        cold_cache=True,
        seed=7,
    )


def _prepared(request: BenchmarkRequest, platform: PlatformInfo) -> PreparedBenchmark:
    _ = platform
    spec = KernelSpec(
        name="test_registration",
        family=request.family,
        mode="test",
        solution="test_solution",
    )
    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=lambda: "output"),
        parameters={"size": 8, "dtype": "test"},
    )


def test_request_selection_modes_and_parameter_copy():
    parameters = {"size": 8}
    normal = BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )
    solution = BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters=parameters,
        solution="gluon",
        registration=None,
        cold_cache=True,
        seed=42,
    )
    exact = BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters=parameters,
        solution=None,
        registration="gluon_bmm",
        cold_cache=False,
        seed=42,
    )
    parameters["size"] = 16

    assert normal.selection_mode == "normal"
    assert solution.selection_mode == "solution"
    assert exact.selection_mode == "registration"
    assert normal.parameters == {"size": 8}
    assert normal.cold_cache is True
    assert exact.cold_cache is False


def test_request_rejects_ambiguous_selection() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        BenchmarkRequest(
            "gemm",
            "bmm",
            {},
            solution="gluon",
            registration="exact",
            cold_cache=True,
            seed=42,
        )


def test_request_requires_identity_and_selection_fields() -> None:
    with pytest.raises(TypeError):
        BenchmarkRequest("gemm", "bmm", {})


def test_harness_returns_measurement_and_actual_registration():
    set_benchmark_generator("unit_success", "test", _prepared)
    timer = FakeTimer()
    result = KernelBenchmarkHarness(timer, platform_provider=_platform).run(
        _request("unit_success"), measurement_blocks=3
    )

    assert result.status is BenchmarkStatus.SUCCESS
    assert result.registration_name == "test_registration"
    assert result.solution == "test_solution"
    assert result.selection_mode == "solution"
    assert result.cold_cache is True
    assert result.parameters == {"size": 8, "dtype": "test"}
    assert result.samples_us == (2.0, 3.0, 4.0)
    assert result.median_us == 3.0
    assert result.eager_warmup_iterations == 5
    assert result.replay_warmup_iterations == 3
    assert result.measurement_blocks == 3
    assert result.correctness is None
    assert result.to_dict()["status"] == "success"
    assert result.to_dict()["samples_us"] == [2.0, 3.0, 4.0]
    assert timer.cold_cache == [True]
    assert timer.measurement_blocks == [3]


def test_harness_requires_explicit_measurement_blocks() -> None:
    harness = KernelBenchmarkHarness(
        FakeTimer(),
        platform_provider=_platform,
    )

    with pytest.raises(TypeError):
        harness.run(_request("unit_success"))


def test_harness_routes_fresh_runs_to_each_output_validator() -> None:
    received: dict[str, tuple[object, ...]] = {}
    prepared_runs: list[int] = []

    def record(name):
        def validator(_spec, data):
            received[name] = tuple((datum.actual, datum.expected) for datum in data)
            return ValidationOutcome(True, f"{name} checked")

        return validator

    set_output_validator("unit_first", record("first"))
    set_output_validator("unit_third", record("third"))
    specs = (
        OutputValidationSpec("unit_first", {}),
        None,
        OutputValidationSpec("unit_third", {"setting": 1}),
    )

    def prepare_run(run_index):
        prepared_runs.append(run_index)
        return ValidationInvocation(
            candidate=lambda: (f"c1-{run_index}", object(), f"c3-{run_index}"),
            reference=lambda: (f"r1-{run_index}", object(), f"r3-{run_index}"),
        )

    def generator(request, platform):
        prepared = _prepared(request, platform)
        return PreparedBenchmark(
            registration=prepared.registration,
            invocation=prepared.invocation,
            parameters=prepared.parameters,
            validation=PreparedValidation(specs, 2, prepare_run),
        )

    timer = FakeTimer()
    set_benchmark_generator("unit_output_routing", "test", generator)
    result = KernelBenchmarkHarness(timer, platform_provider=_platform).run(
        _request("unit_output_routing"), measurement_blocks=3
    )

    assert result.status is BenchmarkStatus.SUCCESS
    assert prepared_runs == [0, 1]
    assert received["first"] == (("c1-0", "r1-0"), ("c1-1", "r1-1"))
    assert received["third"] == (("c3-0", "r3-0"), ("c3-1", "r3-1"))
    assert result.correctness == {
        "passed": True,
        "runs": 2,
        "outputs": [
            {
                "index": 0,
                "validator": "unit_first",
                "kwargs": {},
                "passed": True,
                "diagnostic": "first checked",
            },
            {
                "index": 1,
                "validator": None,
                "kwargs": {},
                "passed": None,
                "diagnostic": None,
            },
            {
                "index": 2,
                "validator": "unit_third",
                "kwargs": {"setting": 1},
                "passed": True,
                "diagnostic": "third checked",
            },
        ],
    }
    assert result.correctness_time_ms >= 0.0
    assert timer.calls == 1


def test_correctness_failure_skips_timing() -> None:
    set_output_validator(
        "unit_failure",
        lambda _spec, _data: ValidationOutcome(False, "values differ"),
    )

    def generator(request, platform):
        prepared = _prepared(request, platform)
        validation = PreparedValidation(
            (OutputValidationSpec("unit_failure", {}),),
            1,
            lambda _index: ValidationInvocation(
                candidate=lambda: (2,),
                reference=lambda: (1,),
            ),
        )
        return PreparedBenchmark(
            prepared.registration,
            prepared.invocation,
            prepared.parameters,
            validation,
        )

    timer = FakeTimer()
    set_benchmark_generator("unit_correctness_failure", "test", generator)
    result = KernelBenchmarkHarness(timer, platform_provider=_platform).run(
        _request("unit_correctness_failure"), measurement_blocks=3
    )

    assert result.status is BenchmarkStatus.CORRECTNESS_FAILURE
    assert result.error_phase == "validation"
    assert result.error_type == "RuntimeError"
    assert "values differ" in (result.error_message or "")
    assert result.correctness is not None
    assert result.correctness["passed"] is False
    assert result.correctness_time_ms >= 0.0
    assert result.samples_us == ()
    assert timer.calls == 0


def test_correctness_exception_skips_timing() -> None:
    set_output_validator(
        "unit_exception",
        lambda _spec, _data: ValidationOutcome(True),
    )

    def fail_reference():
        raise LookupError("reference failed")

    def generator(request, platform):
        prepared = _prepared(request, platform)
        return PreparedBenchmark(
            prepared.registration,
            prepared.invocation,
            prepared.parameters,
            PreparedValidation(
                (OutputValidationSpec("unit_exception", {}),),
                1,
                lambda _index: ValidationInvocation(
                    candidate=lambda: (1,),
                    reference=fail_reference,
                ),
            ),
        )

    timer = FakeTimer()
    set_benchmark_generator("unit_correctness_exception", "test", generator)
    result = KernelBenchmarkHarness(timer, platform_provider=_platform).run(
        _request("unit_correctness_exception"), measurement_blocks=3
    )

    assert result.status is BenchmarkStatus.CORRECTNESS_FAILURE
    assert result.error_phase == "validation"
    assert result.error_type == "LookupError"
    assert result.error_message == "reference failed"
    assert timer.calls == 0


@pytest.mark.parametrize(
    "phase, status",
    [
        ("configuration", BenchmarkStatus.INVALID_CASE),
        ("environment", BenchmarkStatus.ENVIRONMENT_INVALID),
        ("warmup", BenchmarkStatus.SETUP_FAILURE),
        ("capture", BenchmarkStatus.CAPTURE_FAILURE),
        ("first_replay", BenchmarkStatus.EXECUTION_FAILURE),
        ("measurement", BenchmarkStatus.EXECUTION_FAILURE),
        ("cleanup", BenchmarkStatus.EXECUTION_FAILURE),
    ],
)
def test_harness_classifies_graph_failures(phase, status):
    set_benchmark_generator("unit_graph_failure", "test", _prepared)
    result = KernelBenchmarkHarness(
        _FailingTimer(phase), platform_provider=_platform
    ).run(_request("unit_graph_failure"), measurement_blocks=3)

    assert result.status is status
    assert result.error_phase == phase
    assert result.error_type == "RuntimeError"
    assert result.error_message == "timing failed"


def test_harness_preserves_expected_preparation_outcome():
    def unavailable(request, platform):
        _ = request, platform
        raise BenchmarkCaseError(
            BenchmarkStatus.NOT_APPLICABLE, "backend is not installed"
        )

    set_benchmark_generator("unit_unavailable", "test", unavailable)
    result = KernelBenchmarkHarness(FakeTimer(), platform_provider=_platform).run(
        _request("unit_unavailable"), measurement_blocks=3
    )

    assert result.status is BenchmarkStatus.NOT_APPLICABLE
    assert result.error_phase == "preparation"
    assert result.error_message == "backend is not installed"


def test_harness_reports_missing_generator_as_invalid_case():
    result = KernelBenchmarkHarness(FakeTimer(), platform_provider=_platform).run(
        _request("unit_missing_generator"), measurement_blocks=3
    )

    assert result.status is BenchmarkStatus.INVALID_CASE
    assert "No benchmark generator" in (result.error_message or "")
