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

import copy
from pathlib import Path
from types import SimpleNamespace

import kernel_benchmark_ci as benchmark_ci
import pytest
import yaml
from kernel_benchmark_ci import (
    CoordinatorError,
    _baseline_supports_suite,
    _prepare_python_environment,
    compare_runs,
    comparison_exit_code,
    render_summary,
    validate_run_document,
)

BASE_SHA = "1" * 40
CANDIDATE_SHA = "2" * 40
TARGET_SHA = "3" * 40
MERGE_SHA = "4" * 40
REPO_ROOT = Path(__file__).resolve().parents[2]


def _samples(center: float, spread: float = 0.02) -> list[float]:
    return [center + spread * offset for offset in (-4, -3, -2, -1, 0, 1, 2, 3, 4)]


def _result(
    center: float,
    *,
    status: str = "success",
    spread: float = 0.02,
    validated: bool = True,
) -> dict:
    if status != "success":
        return {
            "status": status,
            "error_phase": "measurement",
            "error_type": "RuntimeError",
            "error_message": "test failure",
        }
    samples = _samples(center, spread)
    return {
        "status": "success",
        "samples_us": samples,
        "correctness": {"passed": True} if validated else None,
        "registration_name": "gluon_bmm_a16w16_gfx950",
        "timing_mode": "graph_replay",
        "metric": "device_time_per_invocation",
        "unit": "us",
    }


def _policy(
    *,
    relative: float = 0.10,
    absolute_us: float = 0.5,
    max_relative_mad: float = 0.05,
) -> dict:
    return {
        "max_regression_relative": relative,
        "max_regression_absolute_us": absolute_us,
        "max_relative_mad": max_relative_mad,
    }


def _case(
    center: float,
    *,
    case_id: str = "amd.gfx950.gemm.bmm.bf16.b12-m1-n512-k128.gluon",
    definition_version: int = 1,
    status: str = "success",
    spread: float = 0.02,
    policy: dict | None = None,
    validated: bool = True,
) -> dict:
    definition = {
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
        "seed": 42,
        "definition_version": definition_version,
    }
    if validated:
        definition["parameters"]["validation"] = {
            "runs": 5,
            "atol": 0.015,
            "rtol": 0.015,
        }
    return {
        "id": case_id,
        "definition": definition,
        "policy": policy or _policy(),
        "result": _result(
            center,
            status=status,
            spread=spread,
            validated=validated,
        ),
    }


def _run(revision: str, cases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "suite_id": "amd-gfx950-kernel-pr",
        "revision": revision,
        "environment": {
            "vendor": "amd",
            "arch": "9.5",
            "device_name": "AMD Instinct MI350X",
        },
        "timer": {
            "calls_per_graph": 100,
            "eager_warmup_iterations": 5,
            "replay_warmup_iterations": 3,
            "measurement_blocks": 9,
        },
        "cases": cases,
    }


def _compare(base: dict | None, candidate: dict, **overrides) -> dict:
    arguments = {
        "repository": "lightseekorg/tokenspeed",
        "pull_request_number": 123,
        "github_run_id": 456,
        "github_run_attempt": 1,
        "target_sha": TARGET_SHA,
        "merge_base_sha": BASE_SHA,
        "candidate_sha": CANDIDATE_SHA,
        "merge_sha": MERGE_SHA,
    }
    arguments.update(overrides)
    return compare_runs(base, candidate, **arguments)


def test_compare_reports_regression_with_minimal_schema():
    report = _compare(
        _run(BASE_SHA, [_case(10.0)]),
        _run(CANDIDATE_SHA, [_case(12.0)]),
    )

    assert set(report) == {
        "schema_version",
        "suite_id",
        "repository",
        "pull_request_number",
        "github_run_id",
        "github_run_attempt",
        "target_sha",
        "merge_base_sha",
        "candidate_sha",
        "merge_sha",
        "comparisons",
    }
    comparison = report["comparisons"][0]
    assert set(comparison) == {
        "id",
        "classification",
        "base_median_us",
        "candidate_median_us",
        "delta_us",
        "delta_percent",
        "detail",
    }
    assert comparison["classification"] == "regression"
    assert comparison["delta_percent"] == pytest.approx(20.0)
    assert comparison_exit_code(report) == 1


@pytest.mark.parametrize(
    ("base_time", "candidate_time", "classification"),
    [
        (10.0, 8.0, "improvement"),
        (2.0, 2.4, "within_budget"),
    ],
)
def test_compare_classifies_non_regressions(base_time, candidate_time, classification):
    report = _compare(
        _run(BASE_SHA, [_case(base_time)]),
        _run(CANDIDATE_SHA, [_case(candidate_time)]),
    )

    assert report["comparisons"][0]["classification"] == classification
    assert comparison_exit_code(report) == 0


def test_compare_marks_noisy_measurement_inconclusive():
    report = _compare(
        _run(BASE_SHA, [_case(10.0, spread=1.0)]),
        _run(CANDIDATE_SHA, [_case(13.0, spread=1.0)]),
    )

    assert report["comparisons"][0]["classification"] == "inconclusive"
    assert "MAD" in report["comparisons"][0]["detail"]
    assert comparison_exit_code(report) == 0


def test_compare_uses_merge_base_policy():
    base_case = _case(10.0, policy=_policy(relative=0.10, absolute_us=0.5))
    candidate_case = _case(12.0, policy=_policy(relative=0.50, absolute_us=5.0))

    report = _compare(
        _run(BASE_SHA, [base_case]),
        _run(CANDIDATE_SHA, [candidate_case]),
    )

    comparison = report["comparisons"][0]
    assert comparison["classification"] == "regression"
    assert "baseline policy was used" in comparison["detail"]


def test_compare_bootstrap_requires_candidate_success():
    report = _compare(None, _run(CANDIDATE_SHA, [_case(10.0)]), merge_sha=None)

    assert report["comparisons"][0]["classification"] == "added"
    assert report["comparisons"][0]["candidate_median_us"] == 10.0
    assert comparison_exit_code(report) == 0

    failed = _run(
        CANDIDATE_SHA,
        [_case(10.0, status="capture_failure")],
    )
    failed_report = _compare(None, failed, merge_sha=None)
    assert failed_report["comparisons"][0]["classification"] == "invalid"
    assert comparison_exit_code(failed_report) == 2


def test_compare_reports_added_changed_and_missing_cases():
    report = _compare(
        _run(
            BASE_SHA,
            [
                _case(10.0, case_id="removed"),
                _case(10.0, case_id="changed"),
            ],
        ),
        _run(
            CANDIDATE_SHA,
            [
                _case(10.0, case_id="changed", definition_version=2),
                _case(10.0, case_id="added"),
            ],
        ),
    )

    assert {item["classification"] for item in report["comparisons"]} == {
        "added",
        "changed",
        "missing",
    }
    assert comparison_exit_code(report) == 0


@pytest.mark.parametrize("context", ["environment", "timer", "registration"])
def test_compare_requires_matching_measurement_context(context):
    base = _run(BASE_SHA, [_case(10.0)])
    candidate = _run(CANDIDATE_SHA, [_case(10.0)])
    if context == "environment":
        candidate["environment"]["device_name"] = "AMD Instinct MI355X"
    elif context == "timer":
        candidate["timer"]["calls_per_graph"] += 1
    else:
        candidate["cases"][0]["result"]["registration_name"] = "different"

    report = _compare(base, candidate)

    expected = "changed" if context == "timer" else "invalid"
    assert report["comparisons"][0]["classification"] == expected


@pytest.mark.parametrize("samples", [[0.0] * 9, [float("nan")] * 9])
def test_validate_run_rejects_invalid_samples(samples):
    run = _run(CANDIDATE_SHA, [_case(10.0)])
    run["cases"][0]["result"]["samples_us"] = samples

    with pytest.raises(CoordinatorError, match="timing samples"):
        validate_run_document(run)


def test_validate_run_requires_configured_sample_count():
    run = _run(CANDIDATE_SHA, [_case(10.0)])
    run["cases"][0]["result"]["samples_us"].pop()

    with pytest.raises(CoordinatorError, match="measurement blocks"):
        validate_run_document(run)


def test_validate_run_requires_identity_and_unique_case_ids():
    run = _run(CANDIDATE_SHA, [_case(10.0)])
    with pytest.raises(CoordinatorError, match="does not match"):
        validate_run_document(run, expected_revision=BASE_SHA)

    duplicate = copy.deepcopy(run["cases"][0])
    run["cases"].append(duplicate)
    with pytest.raises(CoordinatorError, match="duplicate"):
        validate_run_document(run)

    run = _run(CANDIDATE_SHA, [_case(10.0)])
    run["schema_version"] = 2
    with pytest.raises(CoordinatorError, match="schema version"):
        validate_run_document(run)

    run = _run(CANDIDATE_SHA, [_case(10.0)])
    run["cases"][0]["id"] = 1
    with pytest.raises(CoordinatorError, match="benchmark id"):
        validate_run_document(run)


def test_validate_run_requires_device_timing_semantics():
    run = _run(CANDIDATE_SHA, [_case(10.0)])
    run["cases"][0]["result"]["timing_mode"] = "host_wall_clock"

    with pytest.raises(CoordinatorError, match="timing semantics"):
        validate_run_document(run)


def test_validate_run_requires_local_correctness_when_configured():
    run = _run(CANDIDATE_SHA, [_case(10.0)])
    run["cases"][0]["result"]["correctness"] = None

    with pytest.raises(CoordinatorError, match="required by the benchmark definition"):
        validate_run_document(run)


def test_validate_run_preserves_environment_failure_results():
    run = _run(
        CANDIDATE_SHA,
        [_case(10.0, status="environment_invalid")],
    )
    run["environment"] = {
        "vendor": "",
        "arch": "",
        "device_name": "",
        "detection_error": {"type": "RuntimeError", "message": "probe failed"},
    }

    validated = validate_run_document(run)
    assert validated["cases"] == run["cases"]
    assert not any(validated["environment"].values())

    run["cases"] = [_case(10.0)]
    with pytest.raises(CoordinatorError, match="complete hardware"):
        validate_run_document(run)


def test_validate_run_rejects_failing_local_correctness():
    run = _run(CANDIDATE_SHA, [_case(10.0)])
    run["cases"][0]["result"]["correctness"] = {"passed": False}

    with pytest.raises(CoordinatorError, match="must report passed=true"):
        validate_run_document(run)


def test_validate_run_allows_opt_out_and_ignores_unknown_metadata():
    run = _run(CANDIDATE_SHA, [_case(10.0, validated=False)])
    run["producer_metadata"] = {"new": "field"}
    run["timer"]["future_setting"] = 10
    run["cases"][0]["result"]["median_us"] = -1000.0

    validated = validate_run_document(run)
    assert validated["producer_metadata"] == {"new": "field"}
    assert validated["cases"][0]["result"]["median_us"] == -1000.0


def test_render_summary_prioritizes_regressions_and_bounds_rows():
    base_cases = [_case(10.0, case_id=f"case-{index:02d}") for index in range(60)]
    candidate_cases = [
        _case(12.0 if index == 59 else 10.0, case_id=f"case-{index:02d}")
        for index in range(60)
    ]
    summary = render_summary(
        _compare(_run(BASE_SHA, base_cases), _run(CANDIDATE_SHA, candidate_cases))
    )

    assert summary.index("case-59") < summary.index("case-00")
    assert summary.count("| `case-") == 50
    assert "10 additional results" in summary


def test_bootstrap_requires_both_runner_and_suite_to_be_absent(tmp_path):
    suite = Path("tokenspeed-kernel/benchmarks/amd/gfx950.json")
    worker = Path("tokenspeed-kernel/python/tokenspeed_kernel/benchmark/ci.py")

    assert _baseline_supports_suite(tmp_path, suite) is False

    (tmp_path / worker).parent.mkdir(parents=True)
    (tmp_path / worker).touch()
    with pytest.raises(CoordinatorError, match="not the requested suite"):
        _baseline_supports_suite(tmp_path, suite)

    (tmp_path / worker).unlink()
    (tmp_path / suite).parent.mkdir(parents=True)
    (tmp_path / suite).touch()
    with pytest.raises(CoordinatorError, match="not its revision-local runner"):
        _baseline_supports_suite(tmp_path, suite)

    (tmp_path / worker).touch()
    assert _baseline_supports_suite(tmp_path, suite) is True


def test_orchestrate_runs_each_revision_once(monkeypatch, tmp_path):
    suite = Path("tokenspeed-kernel/benchmarks/amd/gfx950.json")
    calls: list[tuple[str, str]] = []

    def add_worktree(repo, path, revision):
        del repo, revision
        (path / "tokenspeed-kernel/python/tokenspeed_kernel/benchmark").mkdir(
            parents=True
        )
        (path / "tokenspeed-kernel/python/tokenspeed_kernel/benchmark/ci.py").touch()
        (path / suite).parent.mkdir(parents=True)
        (path / suite).touch()

    def run_revision(checkout, python, **kwargs):
        del python
        calls.append((checkout.name, kwargs["output_path"].name))
        revision = kwargs["revision"]
        return _run(revision, [_case(10.0)])

    monkeypatch.setattr(benchmark_ci, "_add_worktree", add_worktree)
    monkeypatch.setattr(benchmark_ci, "_remove_worktree", lambda *args: None)
    monkeypatch.setattr(benchmark_ci, "_run_revision", run_revision)
    args = SimpleNamespace(
        repo=tmp_path,
        output_dir=tmp_path / "output",
        base_ref=BASE_SHA,
        candidate_ref=CANDIDATE_SHA,
        suite=suite,
        work_dir=None,
        environment_mode="current",
        repository="lightseekorg/tokenspeed",
        pull_request_number=123,
        github_run_id=456,
        github_run_attempt=1,
        merge_sha=None,
    )

    report = benchmark_ci.orchestrate(
        args, revisions=(TARGET_SHA, BASE_SHA, CANDIDATE_SHA)
    )

    assert report["comparisons"][0]["classification"] == "within_budget"
    assert calls == [
        ("base", "base.json"),
        ("candidate", "candidate.json"),
    ]


def test_revision_environment_inherits_and_pins_prepared_rocm(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    requirements = checkout / "tokenspeed-kernel/python/requirements/rocm.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("torch\nnumpy\n", encoding="utf-8")
    inherited = tmp_path / "prepared/site-packages"
    inherited.mkdir(parents=True)
    environment_dir = tmp_path / "revision-venv"
    calls: list[list[str]] = []

    def run_logged(command, **kwargs):
        del kwargs
        calls.append(list(command))
        if command[1:3] == ["-m", "venv"]:
            (environment_dir / "bin").mkdir(parents=True)
            (environment_dir / "bin/python").touch()
            (environment_dir / "lib/python3.13/site-packages").mkdir(parents=True)

    monkeypatch.setattr(benchmark_ci, "_run_logged", run_logged)
    monkeypatch.setattr(benchmark_ci.sys, "path", [str(inherited)])
    monkeypatch.setattr(
        benchmark_ci.importlib.metadata, "version", lambda package: "2.13.0+rocm7.2"
    )

    python = _prepare_python_environment(
        checkout, environment_dir, tmp_path / "setup.log"
    )

    assert python == environment_dir / "bin/python"
    overlay = environment_dir / (
        "lib/python3.13/site-packages/tokenspeed-benchmark-base.pth"
    )
    assert overlay.read_text(encoding="utf-8") == f"{inherited.resolve()}\n"
    constraint = environment_dir / "benchmark-constraints.txt"
    assert constraint.read_text(encoding="utf-8") == "torch==2.13.0+rocm7.2\n"
    pip_call = next(command for command in calls if "pip" in command)
    assert pip_call[pip_call.index("--constraint") + 1] == str(constraint)
    assert sum("torch.cuda.is_available" in " ".join(command) for command in calls) == 2


def test_shared_task_runner_preserves_paired_benchmark_outputs():
    path = REPO_ROOT / ".github/workflows/run-pr-test-stage.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)
    inputs = triggers["workflow_call"]["inputs"]
    job = workflow["jobs"]["test"]

    for name, environment_name in (
        ("comparison_base_ref", "BASE_REF"),
        ("comparison_candidate_ref", "CANDIDATE_REF"),
    ):
        assert inputs[name]["type"] == "string"
        assert job["env"][environment_name] == f"${{{{ inputs.{name} }}}}"
    assert "github.event.pull_request.number" in job["env"]["PR_NUMBER"]
    assert "github.sha" in job["env"]["MERGE_SHA"]

    checkout = next(
        step for step in job["steps"] if step.get("name") == "Checkout code"
    )
    assert checkout["with"]["fetch-depth"] == (
        "${{ matrix.workflow_stage == 'kernel-benchmark' && '0' || '1' }}"
    )
    assert checkout["with"]["persist-credentials"] == (
        "${{ matrix.workflow_stage != 'kernel-benchmark' }}"
    )

    upload = next(
        step for step in job["steps"] if step.get("name") == "Upload task result"
    )
    assert ".ci-artifacts/result.json" in upload["with"]["path"]
    assert ".ci-artifacts/published/" in upload["with"]["path"]
    assert upload["with"]["include-hidden-files"] is True

    assert not (REPO_ROOT / ".github/workflows/kernel-benchmark-amd.yml").exists()

    cancel_workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/cancel-pr-tests-on-close.yml").read_text(
            encoding="utf-8"
        )
    )
    groups = {
        item["group"]
        for item in cancel_workflow["jobs"]["cancel"]["strategy"]["matrix"]["include"]
    }
    assert "kernel-benchmark-amd" not in groups
