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
                "id": "gemm.bmm/example",
                "comparison_epoch": 1,
                "definition": _definition(),
                "policy": _policy(),
            }
        ]
    return {
        "schema_version": 1,
        "suite_id": "unit-amd-gfx950",
        "environment": {"vendor": "amd", "arch": "9.5"},
        "timer": {
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
        cold_cache=request.cold_cache,
        seed=request.seed,
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
        eager_warmup_iterations=5,
        replay_warmup_iterations=3,
        measurement_blocks=5,
        correctness={"passed": True, "runs": 5},
    )


def test_gfx950_suite_selects_exact_registrations():
    suite_path = Path(__file__).parents[1] / "benchmarks" / "amd" / "gfx950.json"
    suite = load_suite(suite_path)

    assert suite.suite_id == "amd-gfx950-registration-kernels"
    assert suite.required_environment == {"vendor": "amd", "arch": "9.5"}
    assert suite.timer.eager_warmup_iterations == 5
    assert suite.timer.replay_warmup_iterations == 3
    assert suite.default_measurement_blocks == 30
    gemm_cases = [case for case in suite.cases if case.request.family == "gemm"]
    assert len(gemm_cases) == 7
    case = gemm_cases[0]
    assert case.id == ("gemm.bmm/gluon_bmm_a16w16_gfx950/b12-m1-n512-k128-bfloat16")
    assert case.comparison_epoch == 1
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
    assert case.request.cold_cache is True
    assert case.request.seed == 42
    assert case.measurement_blocks == 30
    assert case.policy == _policy()

    mxfp8_shapes = (
        (1024, 1792, 5120),
        (1024, 4096, 1280),
        (1024, 5120, 1024),
        (4096, 1792, 5120),
        (4096, 4096, 1280),
        (4096, 5120, 1024),
    )
    for mxfp8_case, (m, n, k) in zip(gemm_cases[1:], mxfp8_shapes, strict=True):
        assert mxfp8_case.id == (
            "gemm.mm/gluon_mm_mxfp8_gfx950/" f"m{m}-n{n}-k{k}-mxfp8-bfloat16"
        )
        assert mxfp8_case.comparison_epoch == 1
        assert mxfp8_case.request.parameters == {
            "M": m,
            "N": n,
            "K": k,
            "quant": "mxfp8",
            "block_size": [1, 32],
            "out_dtype": "bfloat16",
            "validation": {"runs": 1, "atol": 0.0, "rtol": 0.0},
        }
        assert mxfp8_case.request.registration == "gluon_mm_mxfp8_gfx950"
        assert mxfp8_case.request.cold_cache is True
        assert mxfp8_case.policy == _policy()


def test_load_suite_includes_case_files(tmp_path):
    fragment_path = tmp_path / "operation.json"
    fragment_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "common_parameters": {
                    "model_profile": "test-model",
                    "M": 4,
                },
                "cases": [
                    {
                        "id": "gemm.bmm/included",
                        "comparison_epoch": 1,
                        "definition": _definition(seed=43),
                        "policy": _policy(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = _suite_payload()
    payload["case_files"] = [fragment_path.name]

    suite = load_suite(_write_suite(tmp_path, payload))

    assert [case.id for case in suite.cases] == [
        "gemm.bmm/example",
        "gemm.bmm/included",
    ]
    included = next(case for case in suite.cases if case.id.endswith("included"))
    assert included.request.parameters["model_profile"] == "test-model"
    assert included.request.parameters["M"] == 1


def test_load_suite_applies_case_measurement_block_overrides(tmp_path):
    cases = [
        {
            "id": "gemm.bmm/default",
            "comparison_epoch": 1,
            "definition": _definition(seed=42),
            "policy": _policy(),
        },
        {
            "id": "gemm.bmm/override",
            "comparison_epoch": 1,
            "definition": _definition(seed=43),
            "measurement_blocks": 30,
            "policy": _policy(),
        },
    ]

    suite = load_suite(_write_suite(tmp_path, _suite_payload(cases)))

    assert {case.id: case.measurement_blocks for case in suite.cases} == {
        "gemm.bmm/default": 5,
        "gemm.bmm/override": 30,
    }


def test_load_suite_expands_parameter_lists_as_cartesian_product(tmp_path):
    definition = _definition()
    definition["parameters"]["N"] = [32, 64]
    definition["parameters"]["M"] = [1, 2]
    definition["parameters"]["metadata"] = {"values": [7, 8]}
    definition["parameters"]["literal_shape"] = [[7, 8]]
    payload = _suite_payload(
        [
            {
                "id": "gemm.bmm/listed",
                "comparison_epoch": 1,
                "definition": definition,
                "policy": _policy(),
            }
        ]
    )

    suite = load_suite(_write_suite(tmp_path, payload))

    assert {
        case.id: (case.request.parameters["M"], case.request.parameters["N"])
        for case in suite.cases
    } == {
        "gemm.bmm/listed_0": (1, 32),
        "gemm.bmm/listed_1": (1, 64),
        "gemm.bmm/listed_2": (2, 32),
        "gemm.bmm/listed_3": (2, 64),
    }
    assert all(
        case.request.parameters["metadata"] == {"values": [7, 8]}
        for case in suite.cases
    )
    assert all(
        case.request.parameters["literal_shape"] == [7, 8] for case in suite.cases
    )


def test_load_suite_preserves_id_for_one_literal_list_value(tmp_path):
    definition = _definition()
    definition["parameters"]["literal_shape"] = [[7, 8]]

    suite = load_suite(_write_suite(tmp_path, _suite_payload()))
    payload = _suite_payload()
    payload["cases"][0]["definition"] = definition
    suite = load_suite(_write_suite(tmp_path, payload))

    assert suite.cases[0].id == "gemm.bmm/example"
    assert suite.cases[0].request.parameters["literal_shape"] == [7, 8]


def test_load_suite_rejects_empty_parameter_list(tmp_path):
    definition = _definition()
    definition["parameters"]["M"] = []
    payload = _suite_payload(
        [
            {
                "id": "gemm.bmm/empty",
                "comparison_epoch": 1,
                "definition": definition,
                "policy": _policy(),
            }
        ]
    )

    with pytest.raises(SuiteConfigError, match="M must not be an empty list"):
        load_suite(_write_suite(tmp_path, payload))


def test_run_suite_uses_one_timer_and_emits_deterministic_envelope(tmp_path):
    cases = [
        {
            "id": "gemm.bmm/z-case",
            "comparison_epoch": 2,
            "definition": _definition(seed=43),
            "policy": _policy(),
        },
        {
            "id": "gemm.bmm/a-case",
            "comparison_epoch": 1,
            "definition": _definition(seed=42),
            "policy": _policy(),
        },
    ]
    suite = load_suite(_write_suite(tmp_path, _suite_payload(cases)))
    created_configs = []
    requests = []
    measurement_counts = []

    class Harness:
        def run(self, request, *, measurement_blocks):
            requests.append(request)
            measurement_counts.append(measurement_blocks)
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
    assert payload["timer"] == {
        "eager_warmup_iterations": 5,
        "replay_warmup_iterations": 3,
        "measurement_blocks": 5,
    }
    assert len(created_configs) == 1
    assert [case["id"] for case in payload["cases"]] == [
        "gemm.bmm/a-case",
        "gemm.bmm/z-case",
    ]
    assert [case["comparison_epoch"] for case in payload["cases"]] == [1, 2]
    assert [request.seed for request in requests] == [42, 43]
    assert measurement_counts == [5, 5]
    assert payload["cases"][0]["result"]["status"] == "success"
    assert payload["cases"][0]["result"] == {
        "status": "success",
        "registration_name": "gluon_bmm_a16w16_gfx950",
        "cold_cache": True,
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
    assert payload["cases"][0].keys() == {
        "id",
        "comparison_epoch",
        "definition",
        "policy",
        "measurement_blocks",
        "result",
    }
    assert payload["cases"][0]["measurement_blocks"] == 5


def test_run_suite_uses_one_harness_for_mixed_measurement_blocks(tmp_path):
    cases = [
        {
            "id": "gemm.bmm/default",
            "comparison_epoch": 1,
            "definition": _definition(seed=42),
            "policy": _policy(),
        },
        {
            "id": "gemm.bmm/override",
            "comparison_epoch": 1,
            "definition": _definition(seed=43),
            "measurement_blocks": 30,
            "policy": _policy(),
        },
    ]
    suite = load_suite(_write_suite(tmp_path, _suite_payload(cases)))
    runs = []

    class Harness:
        def run(self, request, *, measurement_blocks):
            runs.append((request.seed, measurement_blocks))
            return _success_result(request)

    payload = run_suite(
        suite,
        _REVISION,
        harness_factory=lambda _config: Harness(),
        environment_provider=lambda: _ENVIRONMENT,
    )

    assert runs == [(42, 5), (43, 30)]
    assert [case["measurement_blocks"] for case in payload["cases"]] == [5, 30]


def test_suite_defaults_to_cold_cache_and_can_disable_it(tmp_path):
    default_suite = load_suite(_write_suite(tmp_path, _suite_payload()))
    assert default_suite.cases[0].request.cold_cache is True
    assert default_suite.cases[0].definition["cold_cache"] is True

    payload = _suite_payload()
    payload["cases"][0]["definition"]["cold_cache"] = False
    hot_suite = load_suite(_write_suite(tmp_path, payload))
    assert hot_suite.cases[0].request.cold_cache is False
    assert hot_suite.cases[0].definition["cold_cache"] is False


def test_benchmark_exception_is_result_data_and_later_cases_run(tmp_path):
    cases = [
        {
            "id": "gemm.bmm/fails",
            "comparison_epoch": 1,
            "definition": _definition(seed=42),
            "policy": _policy(),
        },
        {
            "id": "gemm.bmm/succeeds",
            "comparison_epoch": 1,
            "definition": _definition(seed=43),
            "policy": _policy(),
        },
    ]
    suite = load_suite(_write_suite(tmp_path, _suite_payload(cases)))

    class Harness:
        def run(self, request, *, measurement_blocks):
            assert measurement_blocks == 5
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
        def run(self, request, *, measurement_blocks):
            assert measurement_blocks == 5
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
    assert result["cold_cache"] is True
    assert payload["environment"]["vendor"] == "nvidia"
    assert payload["environment"]["arch"] == "9.0"
    assert "required 'amd'" in result["error_message"]


def test_builtin_harness_factory_passes_explicit_dependencies(tmp_path, monkeypatch):
    config = load_suite(_write_suite(tmp_path, _suite_payload())).timer
    expected = object()

    def fake_harness(timer, *, platform_provider):
        assert isinstance(timer, benchmark_ci.GraphTimer)
        assert timer.config is config
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
            lambda payload: payload["timer"].update(calls_per_graph=1),
            "calls_per_graph was removed",
        ),
        (
            lambda payload: payload["cases"][0].update(measurement_blocks=4),
            "measurement_blocks",
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
        (
            lambda payload: payload["cases"][0]["definition"].update(cold_cache="yes"),
            "cold_cache",
        ),
        (
            lambda payload: payload["cases"][0].update(comparison_epoch=0),
            "comparison_epoch",
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
