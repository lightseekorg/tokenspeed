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

import json
from pathlib import Path

import pytest
import tokenspeed_kernel.benchmark.ci as benchmark_ci
from tokenspeed_kernel.benchmark.ci import (
    SuiteConfigError,
    load_suite,
    main,
    run_suite,
)
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkStatus,
    KernelBenchmarkResult,
)

_REVISION = "a" * 40
_ENVIRONMENT = {
    "vendor": "amd",
    "arch": "9.5",
    "device_name": "test accelerator",
    "device_count": 1,
}


def _definition(*, seed: int = 42) -> dict:
    return {
        "family": "gemm",
        "mode": "bmm",
        "parameters": {
            "batch": 12,
            "M": 1,
            "N": 512,
            "K": 128,
            "dtype": "bfloat16",
        },
        "registration": "gluon_bmm_a16w16_gfx950",
        "seed": seed,
        "definition_version": 1,
    }


def _policy() -> dict:
    return {
        "max_regression_relative": 0.1,
        "max_regression_absolute_us": 0.5,
        "max_relative_mad": 0.05,
    }


def _suite_payload(cases: list[dict] | None = None) -> dict:
    if cases is None:
        cases = [
            {
                "id": "gemm.bmm/example/v1",
                "definition": _definition(),
                "policy": _policy(),
            }
        ]
    return {
        "schema_version": 1,
        "suite_id": "unit-amd-gfx950",
        "environment": {"vendor": "amd", "arch": "9.5"},
        "timer": {
            "calls_per_graph": 100,
            "eager_warmup_iterations": 5,
            "replay_warmup_iterations": 3,
            "measurement_blocks": 5,
        },
        "cases": cases,
    }


def _write_suite(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _success_result(request) -> KernelBenchmarkResult:
    return KernelBenchmarkResult(
        status=BenchmarkStatus.SUCCESS,
        family=request.family,
        mode=request.mode,
        parameters=request.parameters,
        selection_mode=request.selection_mode,
        requested_solution=request.solution,
        requested_registration=request.registration,
        seed=request.seed,
        definition_version=request.definition_version,
        platform_vendor="amd",
        platform_arch="9.5",
        device_name="test accelerator",
        registration_name="gluon_bmm_a16w16_gfx950",
        solution="gluon",
        samples_us=(1.0, 1.1, 1.2, 1.1, 1.0),
        median_us=1.1,
        p90_us=1.18,
        min_us=1.0,
        max_us=1.2,
        relative_mad=0.09,
        calls_per_graph=100,
        eager_warmup_iterations=5,
        replay_warmup_iterations=3,
        measurement_blocks=5,
        correctness={"passed": True, "runs": 5},
    )


def test_starter_suite_selects_exact_gfx950_registration():
    suite_path = Path(__file__).parents[1] / "benchmarks" / "amd" / "gfx950.json"
    suite = load_suite(suite_path)

    assert suite.suite_id == "amd-gfx950-registration-kernels"
    assert suite.required_environment == {"vendor": "amd", "arch": "9.5"}
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert case.id == ("gemm.bmm/gluon_bmm_a16w16_gfx950/b12-m1-n512-k128-bfloat16/v1")
    assert case.request.parameters == {
        "batch": 12,
        "M": 1,
        "N": 512,
        "K": 128,
        "dtype": "bfloat16",
        "validation": {
            "runs": 5,
            "atol": 0.015,
            "rtol": 0.015,
        },
    }
    assert case.request.registration == "gluon_bmm_a16w16_gfx950"
    assert case.request.solution is None
    assert case.request.seed == 42
    assert case.request.definition_version == 1
    assert case.policy == _policy()


def test_run_suite_uses_one_timer_and_emits_deterministic_envelope(tmp_path):
    cases = [
        {
            "id": "gemm.bmm/z-case/v1",
            "definition": _definition(seed=43),
            "policy": _policy(),
        },
        {
            "id": "gemm.bmm/a-case/v1",
            "definition": _definition(seed=42),
            "policy": _policy(),
        },
    ]
    suite = load_suite(_write_suite(tmp_path, _suite_payload(cases)))
    created_configs = []
    requests = []

    class Harness:
        def run(self, request):
            requests.append(request)
            return _success_result(request)

    def harness_factory(config):
        created_configs.append(config)
        return Harness()

    payload = run_suite(
        suite,
        _REVISION,
        harness_factory=harness_factory,
        environment_provider=lambda: _ENVIRONMENT,
    )

    assert payload.keys() == {
        "schema_version",
        "suite_id",
        "revision",
        "environment",
        "timer",
        "cases",
    }
    assert payload["schema_version"] == 1
    assert payload["revision"] == _REVISION
    assert payload["environment"] == _ENVIRONMENT
    assert len(created_configs) == 1
    assert [case["id"] for case in payload["cases"]] == [
        "gemm.bmm/a-case/v1",
        "gemm.bmm/z-case/v1",
    ]
    assert [request.seed for request in requests] == [42, 43]
    assert payload["cases"][0]["result"]["status"] == "success"
    assert payload["cases"][0]["result"] == {
        "status": "success",
        "registration_name": "gluon_bmm_a16w16_gfx950",
        "timing_mode": "graph_replay",
        "metric": "device_time_per_invocation",
        "unit": "us",
        "samples_us": [1.0, 1.1, 1.2, 1.1, 1.0],
        "correctness": {"passed": True, "runs": 5},
        "error_phase": None,
        "error_type": None,
        "error_message": None,
    }
    assert payload["cases"][0]["policy"] == _policy()
    assert payload["cases"][0].keys() == {"id", "definition", "policy", "result"}


def test_run_suite_rejects_success_with_the_wrong_measurement_context(tmp_path):
    suite = load_suite(_write_suite(tmp_path, _suite_payload()))

    class Harness:
        def run(self, request):
            result = _success_result(request)
            return KernelBenchmarkResult(
                **{
                    **result.to_dict(),
                    "status": BenchmarkStatus.SUCCESS,
                    "calls_per_graph": 1,
                }
            )

    payload = run_suite(
        suite,
        _REVISION,
        harness_factory=lambda _config: Harness(),
        environment_provider=lambda: _ENVIRONMENT,
    )

    result = payload["cases"][0]["result"]
    assert result["status"] == "execution_failure"
    assert result["error_message"] == "successful benchmark reported the wrong context"


def test_benchmark_exception_is_result_data_and_later_cases_run(tmp_path):
    cases = [
        {
            "id": "gemm.bmm/fails/v1",
            "definition": _definition(seed=42),
            "policy": _policy(),
        },
        {
            "id": "gemm.bmm/succeeds/v1",
            "definition": _definition(seed=43),
            "policy": _policy(),
        },
    ]
    suite = load_suite(_write_suite(tmp_path, _suite_payload(cases)))

    class Harness:
        def run(self, request):
            if request.seed == 42:
                raise RuntimeError("launch failed")
            return _success_result(request)

    payload = run_suite(
        suite,
        _REVISION,
        harness_factory=lambda _config: Harness(),
        environment_provider=lambda: _ENVIRONMENT,
    )

    assert [case["result"]["status"] for case in payload["cases"]] == [
        "execution_failure",
        "success",
    ]
    assert payload["cases"][0]["result"]["error_message"] == "launch failed"


def test_harness_failure_result_is_preserved(tmp_path):
    suite = load_suite(_write_suite(tmp_path, _suite_payload()))

    class Harness:
        def run(self, request):
            result = _success_result(request)
            return KernelBenchmarkResult(
                **{
                    **result.to_dict(),
                    "status": BenchmarkStatus.CAPTURE_FAILURE,
                    "samples_us": (),
                    "median_us": None,
                    "error_phase": "capture",
                    "error_type": "RuntimeError",
                    "error_message": "capture failed",
                }
            )

    payload = run_suite(
        suite,
        _REVISION,
        harness_factory=lambda _config: Harness(),
        environment_provider=lambda: _ENVIRONMENT,
    )

    result = payload["cases"][0]["result"]
    assert result["status"] == "capture_failure"
    assert result["error_phase"] == "capture"
    assert result["error_message"] == "capture failed"


def test_environment_mismatch_produces_complete_results_without_timing(tmp_path):
    suite = load_suite(_write_suite(tmp_path, _suite_payload()))
    factory_called = False

    def harness_factory(_config):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("must not construct a harness")

    payload = run_suite(
        suite,
        _REVISION,
        harness_factory=harness_factory,
        environment_provider=lambda: {
            "vendor": "nvidia",
            "arch": "9.0",
            "device_name": "other accelerator",
        },
    )

    assert factory_called is False
    result = payload["cases"][0]["result"]
    assert result["status"] == "environment_invalid"
    assert payload["environment"]["vendor"] == "nvidia"
    assert payload["environment"]["arch"] == "9.0"
    assert "required 'amd'" in result["error_message"]


def test_builtin_harness_factory_passes_explicit_dependencies(tmp_path, monkeypatch):
    config = load_suite(_write_suite(tmp_path, _suite_payload())).timer
    expected = object()

    def fake_harness(config_arg, *, timer, platform_provider):
        assert config_arg is config
        assert timer is None
        assert platform_provider is benchmark_ci.current_platform
        return expected

    monkeypatch.setattr(benchmark_ci, "KernelBenchmarkHarness", fake_harness)

    assert benchmark_ci._create_harness(config) is expected


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda payload: payload.update(schema_version=2), "unsupported"),
        (lambda payload: payload.update(cases=[]), "non-empty"),
        (
            lambda payload: payload["timer"].update(calls_per_graph=0),
            "calls_per_graph",
        ),
        (
            lambda payload: payload["cases"][0]["policy"].update(
                max_regression_absolute_us=-0.1
            ),
            "max_regression_absolute_us",
        ),
        (
            lambda payload: payload["cases"][0]["definition"].pop("seed"),
            "seed",
        ),
    ],
)
def test_load_suite_rejects_invalid_invariant_family(tmp_path, mutate, match):
    payload = _suite_payload()
    mutate(payload)

    with pytest.raises(SuiteConfigError, match=match):
        load_suite(_write_suite(tmp_path, payload))


def test_load_suite_rejects_duplicate_case_ids(tmp_path):
    case = _suite_payload()["cases"][0]
    payload = _suite_payload([case, dict(case)])

    with pytest.raises(SuiteConfigError, match="duplicate case IDs"):
        load_suite(_write_suite(tmp_path, payload))


def test_load_suite_ignores_unknown_fields(tmp_path):
    payload = _suite_payload()
    payload["note"] = "future suite metadata"
    payload["environment"]["runner"] = "self-hosted"
    payload["timer"]["future_timer_option"] = True
    payload["cases"][0]["note"] = "future case metadata"
    payload["cases"][0]["definition"]["note"] = "future definition metadata"
    payload["cases"][0]["policy"]["note"] = "future policy metadata"

    suite = load_suite(_write_suite(tmp_path, payload))

    assert suite.suite_id == "unit-amd-gfx950"
    assert suite.cases[0].policy == _policy()


def test_main_returns_nonzero_for_invalid_suite(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        main(["--suite", str(path), "--revision", _REVISION, "--output", "-"])

    assert raised.value.code == 2


def test_main_writes_successful_run_document(tmp_path, monkeypatch):
    suite_path = _write_suite(tmp_path, _suite_payload())
    output_path = tmp_path / "output" / "result.json"
    expected = {"schema_version": 1, "cases": []}

    def fake_run_suite(
        _suite,
        _revision,
        *,
        harness_factory,
        environment_provider,
    ):
        assert harness_factory is benchmark_ci._create_harness
        assert environment_provider is benchmark_ci._collect_environment
        return expected

    monkeypatch.setattr(benchmark_ci, "run_suite", fake_run_suite)

    exit_code = main(
        [
            "--suite",
            str(suite_path),
            "--revision",
            _REVISION,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == expected
