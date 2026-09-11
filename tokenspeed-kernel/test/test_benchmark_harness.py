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
import tokenspeed_kernel.benchmark.generators.gemm as gemm_generator
import torch
from tokenspeed_kernel.benchmark.graph import (
    GraphBenchmarkConfig,
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
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo, current_platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


class _FakeTimer:
    def __init__(self) -> None:
        self.calls = 0

    def measure(self, prepared: PreparedInvocation) -> GraphMeasurement:
        self.calls += 1
        prepared.invoke()
        return GraphMeasurement(
            samples_us=(2.0, 3.0, 4.0),
            median_us=3.0,
            p90_us=3.8,
            min_us=2.0,
            max_us=4.0,
            relative_mad=1.0 / 3.0,
            calls_per_graph=100,
            eager_warmup_iterations=5,
            replay_warmup_iterations=3,
            warmup_time_ms=1.0,
            capture_time_ms=2.0,
            first_replay_time_ms=3.0,
            measurement_time_ms=4.0,
        )


class _FailingTimer:
    def __init__(self, phase: str) -> None:
        self.phase = phase

    def measure(self, prepared: PreparedInvocation) -> GraphMeasurement:
        _ = prepared
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
        seed=7,
        definition_version=2,
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
        invocation=PreparedInvocation(
            invoke=lambda: "output",
            repeat_safe=True,
        ),
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
        seed=42,
        definition_version=1,
    )
    solution = BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters=parameters,
        solution="gluon",
        registration=None,
        seed=42,
        definition_version=1,
    )
    exact = BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters=parameters,
        solution=None,
        registration="gluon_bmm",
        seed=42,
        definition_version=1,
    )
    parameters["size"] = 16

    assert normal.selection_mode == "normal"
    assert solution.selection_mode == "solution"
    assert exact.selection_mode == "registration"
    assert normal.parameters == {"size": 8}


def test_request_rejects_ambiguous_selection() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        BenchmarkRequest(
            "gemm",
            "bmm",
            {},
            solution="gluon",
            registration="exact",
            seed=42,
            definition_version=1,
        )


def test_request_requires_identity_and_selection_fields() -> None:
    with pytest.raises(TypeError):
        BenchmarkRequest("gemm", "bmm", {})


def test_harness_returns_measurement_and_actual_registration():
    set_benchmark_generator("unit_success", "test", _prepared)
    result = KernelBenchmarkHarness(
        None, timer=_FakeTimer(), platform_provider=_platform
    ).run(_request("unit_success"))

    assert result.status is BenchmarkStatus.SUCCESS
    assert result.registration_name == "test_registration"
    assert result.solution == "test_solution"
    assert result.selection_mode == "solution"
    assert result.parameters == {"size": 8, "dtype": "test"}
    assert result.samples_us == (2.0, 3.0, 4.0)
    assert result.median_us == 3.0
    assert result.calls_per_graph == 100
    assert result.eager_warmup_iterations == 5
    assert result.replay_warmup_iterations == 3
    assert result.measurement_blocks == 3
    assert result.correctness is None
    assert result.to_dict()["status"] == "success"
    assert result.to_dict()["samples_us"] == [2.0, 3.0, 4.0]


def test_harness_routes_fresh_runs_to_each_output_validator() -> None:
    received: dict[str, tuple[object, ...]] = {}
    prepared_runs: list[int] = []
    candidate_calls: list[int] = []
    reference_calls: list[int] = []

    def record(name):
        def validator(_spec, data):
            received[name] = tuple((datum.actual, datum.expected) for datum in data)
            return ValidationOutcome(True, f"{name} checked")

        return validator

    set_output_validator("unit_one_run", record("short"))
    set_output_validator("unit_three_runs", record("long"))
    specs = (
        OutputValidationSpec("unit_one_run", 1, {}),
        None,
        OutputValidationSpec("unit_three_runs", 3, {}),
    )

    def prepare_run(run_index):
        prepared_runs.append(run_index)

        def candidate():
            candidate_calls.append(run_index)
            return (f"candidate-short-{run_index}", object(), f"candidate-{run_index}")

        def reference():
            reference_calls.append(run_index)
            return (f"reference-short-{run_index}", object(), f"reference-{run_index}")

        return ValidationInvocation(candidate=candidate, reference=reference)

    def generator(request, platform):
        prepared = _prepared(request, platform)
        return PreparedBenchmark(
            registration=prepared.registration,
            invocation=prepared.invocation,
            parameters=prepared.parameters,
            validation=PreparedValidation(lambda: specs, prepare_run),
        )

    timer = _FakeTimer()
    set_benchmark_generator("unit_output_routing", "test", generator)
    result = KernelBenchmarkHarness(None, timer=timer, platform_provider=_platform).run(
        _request("unit_output_routing")
    )

    assert result.status is BenchmarkStatus.SUCCESS
    assert prepared_runs == [0, 1, 2]
    assert candidate_calls == [0, 1, 2]
    assert reference_calls == [0, 1, 2]
    assert received["short"] == (("candidate-short-0", "reference-short-0"),)
    assert received["long"] == (
        ("candidate-0", "reference-0"),
        ("candidate-1", "reference-1"),
        ("candidate-2", "reference-2"),
    )
    assert result.correctness == {
        "passed": True,
        "runs": 3,
        "outputs": [
            {
                "index": 0,
                "validator": "unit_one_run",
                "runs": 1,
                "kwargs": {},
                "passed": True,
                "diagnostic": "short checked",
            },
            {
                "index": 1,
                "validator": None,
                "runs": 0,
                "kwargs": {},
                "passed": None,
                "diagnostic": None,
            },
            {
                "index": 2,
                "validator": "unit_three_runs",
                "runs": 3,
                "kwargs": {},
                "passed": True,
                "diagnostic": "long checked",
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
            lambda: (OutputValidationSpec("unit_failure", 1, {}),),
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

    timer = _FakeTimer()
    set_benchmark_generator("unit_correctness_failure", "test", generator)
    result = KernelBenchmarkHarness(None, timer=timer, platform_provider=_platform).run(
        _request("unit_correctness_failure")
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
                lambda: (OutputValidationSpec("unit_exception", 1, {}),),
                lambda _index: ValidationInvocation(
                    candidate=lambda: (1,),
                    reference=fail_reference,
                ),
            ),
        )

    timer = _FakeTimer()
    set_benchmark_generator("unit_correctness_exception", "test", generator)
    result = KernelBenchmarkHarness(None, timer=timer, platform_provider=_platform).run(
        _request("unit_correctness_exception")
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
        None, timer=_FailingTimer(phase), platform_provider=_platform
    ).run(_request("unit_graph_failure"))

    assert result.status is status
    assert result.error_phase == phase
    assert result.error_type == "RuntimeError"
    assert result.error_message == "timing failed"


def test_harness_preserves_expected_preparation_outcome():
    def unavailable(request, platform):
        _ = request, platform
        raise BenchmarkCaseError(
            BenchmarkStatus.BACKEND_UNAVAILABLE, "backend is not installed"
        )

    set_benchmark_generator("unit_unavailable", "test", unavailable)
    result = KernelBenchmarkHarness(
        None, timer=_FakeTimer(), platform_provider=_platform
    ).run(_request("unit_unavailable"))

    assert result.status is BenchmarkStatus.BACKEND_UNAVAILABLE
    assert result.error_phase == "preparation"
    assert result.error_message == "backend is not installed"


def test_harness_reports_missing_generator_as_invalid_case():
    result = KernelBenchmarkHarness(
        None, timer=_FakeTimer(), platform_provider=_platform
    ).run(_request("unit_missing_generator"))

    assert result.status is BenchmarkStatus.INVALID_CASE
    assert "No benchmark generator" in (result.error_message or "")


@pytest.mark.parametrize(
    "parameters, match",
    [
        (
            {"batch": 12, "M": 1, "N": 512, "K": 128, "extra": 1},
            "Unknown",
        ),
        ({"batch": 0, "M": 1, "N": 512, "K": 128}, "batch"),
        (
            {"batch": 12, "M": 1, "N": 512, "K": 128, "dtype": "float16"},
            "dtype",
        ),
    ],
)
def test_dense_bmm_rejects_invalid_generator_parameters(parameters, match):
    request = BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters=parameters,
        solution=None,
        registration=None,
        seed=42,
        definition_version=1,
    )

    with pytest.raises(BenchmarkCaseError, match=match) as raised:
        gemm_generator.prepare_dense_bmm(request, _platform())

    assert raised.value.status is BenchmarkStatus.INVALID_CASE


def test_dense_bmm_validation_configuration_is_opt_in() -> None:
    assert gemm_generator._parse_validation(None, dtype=torch.bfloat16, K=128) is None
    assert gemm_generator._parse_validation({}, dtype=torch.bfloat16, K=128) == {
        "runs": 5,
        "atol": 0.015,
        "rtol": 0.015,
    }
    assert gemm_generator._parse_validation(
        {"runs": 3, "atol": 0.02, "rtol": 0.01},
        dtype=torch.bfloat16,
        K=128,
    ) == {"runs": 3, "atol": 0.02, "rtol": 0.01}


@pytest.mark.parametrize(
    ("validation", "match"),
    [
        ({"runs": 0}, "between 1 and 100"),
        ({"runs": 101}, "between 1 and 100"),
        ({"atol": True}, "must be a number"),
        ({"rtol": float("inf")}, "finite and nonnegative"),
        ({"unexpected": 1}, "Unknown"),
    ],
)
def test_dense_bmm_rejects_invalid_validation_configuration(validation, match):
    with pytest.raises(BenchmarkCaseError, match=match) as raised:
        gemm_generator._parse_validation(
            validation,
            dtype=torch.bfloat16,
            K=128,
        )

    assert raised.value.status is BenchmarkStatus.INVALID_CASE


@pytest.mark.parametrize("corrupt", [False, True])
def test_dense_bmm_uses_registered_reference_for_local_correctness(
    fresh_registry,
    monkeypatch,
    corrupt,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    candidate_spec = KernelSpec(
        name="unit_candidate_bmm",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
    )
    reference_spec = KernelSpec(
        name="unit_reference_bmm",
        family="gemm",
        mode="bmm",
        solution="reference",
        format_signatures=frozenset({signature}),
    )
    calls = {"candidate": 0, "reference": 0}
    candidate_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    reference_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seeds: list[int] = []

    def product(kwargs):
        return torch.bmm(
            kwargs["A"].float(),
            kwargs["B"].float().transpose(1, 2),
        ).to(torch.bfloat16)

    def candidate(**kwargs):
        calls["candidate"] += 1
        candidate_inputs.append((kwargs["A"], kwargs["B"], kwargs["out"]))
        output = product(kwargs)
        if corrupt:
            output.add_(1)
        kwargs["out"].copy_(output)
        return kwargs["out"]

    def reference(**kwargs):
        calls["reference"] += 1
        reference_inputs.append((kwargs["A"], kwargs["B"], kwargs["out"]))
        kwargs["out"].copy_(product(kwargs))
        return kwargs["out"]

    KernelRegistry.get().register(candidate_spec, candidate)
    KernelRegistry.get().register(reference_spec, reference)

    class InputGenerator:
        def __init__(self, seed):
            self.seed = seed

        def generate(self, *, batch, M, N, K):
            value = float(self.seed + 1)
            A = torch.full((batch, M, K), value, dtype=torch.bfloat16)
            B = torch.ones((batch, K, N), dtype=torch.bfloat16)
            return {"A": A, "B": B.transpose(1, 2), "out_dtype": torch.bfloat16}

    def get_generator(*_args, seed, **_kwargs):
        seeds.append(seed)
        return InputGenerator(seed)

    monkeypatch.setattr(gemm_generator, "get_input_generator", get_generator)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    timer = _FakeTimer()
    result = KernelBenchmarkHarness(None, timer=timer, platform_provider=_platform).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={
                "batch": 2,
                "M": 1,
                "N": 4,
                "K": 8,
                "dtype": "bfloat16",
                "validation": {"runs": 2, "atol": 0.0, "rtol": 0.0},
            },
            solution=None,
            registration=candidate_spec.name,
            seed=7,
            definition_version=1,
        )
    )

    assert seeds == [7, 8, 9]
    assert calls["reference"] == 2
    for candidate_args, reference_args in zip(
        candidate_inputs[:2], reference_inputs, strict=True
    ):
        assert candidate_args[0] is reference_args[0]
        assert candidate_args[1] is reference_args[1]
        assert candidate_args[2] is not reference_args[2]
    assert result.correctness is not None
    assert result.correctness["outputs"][0]["validator"] == "close"
    if corrupt:
        assert result.status is BenchmarkStatus.CORRECTNESS_FAILURE
        assert result.correctness["passed"] is False
        assert calls["candidate"] == 2
        assert timer.calls == 0
    else:
        assert result.status is BenchmarkStatus.SUCCESS
        assert result.correctness["passed"] is True
        assert calls["candidate"] == 3
        assert timer.calls == 1


@pytest.mark.parametrize("register_incompatible_reference", [False, True])
def test_dense_bmm_validation_requires_a_compatible_registered_reference(
    fresh_registry,
    monkeypatch,
    register_incompatible_reference,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    candidate_spec = KernelSpec(
        name="unit_candidate_without_reference",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
    )
    KernelRegistry.get().register(
        candidate_spec,
        lambda **kwargs: kwargs["out"],
    )
    if register_incompatible_reference:
        reference_spec = KernelSpec(
            name="unit_incompatible_reference",
            family="gemm",
            mode="bmm",
            solution="reference",
            format_signatures=frozenset({signature}),
            traits={"m": frozenset({99})},
        )
        KernelRegistry.get().register(
            reference_spec,
            lambda **kwargs: kwargs["out"],
        )

    class InputGenerator:
        def generate(self, *, batch, M, N, K):
            A = torch.ones((batch, M, K), dtype=torch.bfloat16)
            B = torch.ones((batch, K, N), dtype=torch.bfloat16).transpose(1, 2)
            return {"A": A, "B": B, "out_dtype": torch.bfloat16}

    monkeypatch.setattr(
        gemm_generator,
        "get_input_generator",
        lambda *_args, **_kwargs: InputGenerator(),
    )
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    timer = _FakeTimer()

    result = KernelBenchmarkHarness(None, timer=timer, platform_provider=_platform).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={
                "batch": 2,
                "M": 1,
                "N": 4,
                "K": 8,
                "dtype": "bfloat16",
                "validation": {"runs": 1},
            },
            solution=None,
            registration=candidate_spec.name,
            seed=42,
            definition_version=1,
        )
    )

    assert result.status is BenchmarkStatus.REGISTRATION_MISSING
    assert result.error_phase == "preparation"
    assert "No compatible registered reference" in (result.error_message or "")
    assert timer.calls == 0


def test_exact_dense_bmm_rejects_incompatible_shape(
    fresh_registry,
    monkeypatch,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    spec = KernelSpec(
        name="unit_exact_bmm",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
        traits={
            "batch": frozenset({12}),
            "m": frozenset({1}),
            "n": frozenset({512}),
            "k": frozenset({128}),
        },
    )
    KernelRegistry.get().register(spec, lambda **_kwargs: None)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)

    result = KernelBenchmarkHarness(
        None, timer=_FakeTimer(), platform_provider=_platform
    ).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={"batch": 12, "M": 2, "N": 512, "K": 128},
            solution=None,
            registration="unit_exact_bmm",
            seed=42,
            definition_version=1,
        )
    )

    assert result.status is BenchmarkStatus.INVALID_CASE
    assert result.registration_name is None
    assert "does not support parameters" in (result.error_message or "")


def test_dense_bmm_solution_shape_miss_is_invalid_not_backend_unavailable(
    fresh_registry,
    monkeypatch,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    spec = KernelSpec(
        name="unit_solution_bmm",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
        traits={"m": frozenset({1})},
    )
    KernelRegistry.get().register(spec, lambda **_kwargs: None)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)

    result = KernelBenchmarkHarness(
        None, timer=_FakeTimer(), platform_provider=_platform
    ).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={"batch": 12, "M": 2, "N": 512, "K": 128},
            solution="unit",
            registration=None,
            seed=42,
            definition_version=1,
        )
    )

    assert result.status is BenchmarkStatus.INVALID_CASE
    assert "No kernel found" in (result.error_message or "")


def test_dense_bmm_missing_solution_reports_backend_unavailable(
    fresh_registry,
    monkeypatch,
):
    _ = fresh_registry
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)

    result = KernelBenchmarkHarness(
        None, timer=_FakeTimer(), platform_provider=_platform
    ).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={"batch": 12, "M": 1, "N": 512, "K": 128},
            solution="missing",
            registration=None,
            seed=42,
            definition_version=1,
        )
    )

    assert result.status is BenchmarkStatus.BACKEND_UNAVAILABLE


@pytest.mark.parametrize(
    ("selection", "selection_mode"),
    [
        ({"solution": None, "registration": None}, "normal"),
        ({"solution": "gluon", "registration": None}, "solution"),
        (
            {"solution": None, "registration": "gluon_bmm_a16w16_gfx950"},
            "registration",
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is required")
def test_dense_bmm_gluon_registration_graph_replay(selection, selection_mode):
    platform = current_platform()
    if not platform.is_cdna4:
        pytest.skip("Gluon dense BMM benchmark requires an AMD CDNA4 GPU")

    harness = KernelBenchmarkHarness(
        GraphBenchmarkConfig(
            calls_per_graph=100,
            eager_warmup_iterations=2,
            replay_warmup_iterations=1,
            measurement_blocks=7,
        ),
        timer=None,
        platform_provider=current_platform,
    )
    result = harness.run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={
                "batch": 12,
                "M": 1,
                "N": 512,
                "K": 128,
                "dtype": "bfloat16",
                "validation": {"runs": 3},
            },
            seed=42,
            definition_version=1,
            **selection,
        )
    )

    assert result.status is BenchmarkStatus.SUCCESS, result.to_dict()
    assert result.registration_name == "gluon_bmm_a16w16_gfx950"
    assert result.solution == "gluon"
    assert result.selection_mode == selection_mode
    assert result.calls_per_graph == 100
    assert result.measurement_blocks == 7
    assert result.median_us is not None and result.median_us > 0.0
    assert all(sample > 0.0 for sample in result.samples_us)
    assert result.correctness is not None
    assert result.correctness["passed"] is True
    assert result.correctness["runs"] == 3
