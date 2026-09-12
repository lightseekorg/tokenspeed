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

"""Run and compare registration-level kernel benchmarks across two revisions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COMPARISON_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1

CLASSIFICATIONS = (
    "regression",
    "improvement",
    "within_budget",
    "inconclusive",
    "added",
    "missing",
    "changed",
    "invalid",
)

_TIMER_FIELDS = (
    "calls_per_graph",
    "eager_warmup_iterations",
    "replay_warmup_iterations",
    "measurement_blocks",
)
_TIMING_SEMANTICS = {
    "timing_mode": "graph_replay",
    "metric": "device_time_per_invocation",
    "unit": "us",
}
_WORKER_RELATIVE_PATH = Path(
    "tokenspeed-kernel/python/tokenspeed_kernel/benchmark/ci.py"
)
_KERNEL_REQUIREMENTS = Path("tokenspeed-kernel/python/requirements/rocm.txt")


class CoordinatorError(RuntimeError):
    """A failure to produce a valid two-revision benchmark comparison."""


def _load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoordinatorError(f"Invalid benchmark result {path}: {exc}") from exc
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _definition_requires_correctness(definition: Mapping[str, Any]) -> bool:
    parameters = definition.get("parameters")
    return isinstance(parameters, dict) and parameters.get("validation") is not None


def validate_run_document(
    value: object,
    *,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Check the revision-local facts needed for a meaningful comparison."""

    try:
        run = dict(value)
        if run["schema_version"] != RUN_SCHEMA_VERSION:
            raise CoordinatorError(
                f"benchmark run schema version must be {RUN_SCHEMA_VERSION}"
            )
        if expected_revision is not None and run["revision"] != expected_revision:
            raise CoordinatorError(
                f"benchmark run revision {run['revision']} does not match "
                f"{expected_revision}"
            )
        timer = {field: run["timer"][field] for field in _TIMER_FIELDS}
        environment = {
            field: run["environment"][field]
            for field in ("vendor", "arch", "device_name")
        }
        cases = run["cases"]
        if not cases:
            raise CoordinatorError("benchmark run has no cases")

        seen: set[str] = set()
        successful = False
        for case in cases:
            case_id = case["id"]
            if not isinstance(case_id, str) or not case_id or case_id in seen:
                raise CoordinatorError(f"invalid or duplicate benchmark id {case_id!r}")
            seen.add(case_id)

            result = case["result"]
            if result["status"] != "success":
                continue
            successful = True
            samples = tuple(float(sample) for sample in result["samples_us"])
            if len(samples) != timer["measurement_blocks"]:
                raise CoordinatorError("sample count does not match measurement blocks")
            if not all(math.isfinite(sample) and sample > 0.0 for sample in samples):
                raise CoordinatorError("timing samples must be finite and positive")
            if not result["registration_name"]:
                raise CoordinatorError("successful benchmark has no registration")
            if any(
                result[field] != expected
                for field, expected in _TIMING_SEMANTICS.items()
            ):
                raise CoordinatorError(
                    "successful benchmark has unsupported timing semantics"
                )

            correctness = result.get("correctness")
            if (
                _definition_requires_correctness(case["definition"])
                and correctness is None
            ):
                raise CoordinatorError(
                    "correctness is required by the benchmark definition"
                )
            if correctness is not None and correctness.get("passed") is not True:
                raise CoordinatorError("correctness must report passed=true")

        if successful and not all(environment.values()):
            raise CoordinatorError(
                "successful benchmark runs require complete hardware information"
            )
        run["timer"] = timer
        run["environment"] = environment
        return run
    except CoordinatorError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CoordinatorError(f"malformed benchmark run: {exc}") from exc


def _relative_mad(samples: Sequence[float]) -> float:
    median = float(statistics.median(samples))
    deviations = [abs(sample - median) for sample in samples]
    return float(statistics.median(deviations)) / median


def _empty_comparison(case_id: str, classification: str, detail: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "classification": classification,
        "base_median_us": None,
        "candidate_median_us": None,
        "delta_us": None,
        "delta_percent": None,
        "detail": detail,
    }


def _result_median(case: Mapping[str, Any]) -> float | None:
    result = case.get("result")
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    return float(statistics.median(result["samples_us"]))


def _measurement_comparison(
    case_id: str,
    base_case: Mapping[str, Any],
    candidate_case: Mapping[str, Any],
) -> dict[str, Any]:
    base_result = base_case["result"]
    candidate_result = candidate_case["result"]
    policy = base_case["policy"]
    candidate_policy = candidate_case["policy"]

    for label, result in (("baseline", base_result), ("candidate", candidate_result)):
        if result.get("status") != "success":
            return _empty_comparison(
                case_id,
                "invalid",
                f"{label} benchmark returned {result.get('status', 'unknown')}",
            )

    if base_result["registration_name"] != candidate_result["registration_name"]:
        return _empty_comparison(
            case_id,
            "invalid",
            "selected kernel registrations differ between revisions",
        )
    if any(
        base_result[field] != candidate_result[field]
        for field in ("timing_mode", "metric", "unit")
    ):
        return _empty_comparison(
            case_id,
            "invalid",
            "timing result formats differ between revisions",
        )

    base_samples = tuple(float(value) for value in base_result["samples_us"])
    candidate_samples = tuple(float(value) for value in candidate_result["samples_us"])
    base_median = float(statistics.median(base_samples))
    candidate_median = float(statistics.median(candidate_samples))
    delta_us = candidate_median - base_median
    delta_relative = candidate_median / base_median - 1.0

    relative_mad = max(_relative_mad(base_samples), _relative_mad(candidate_samples))
    if relative_mad > policy["max_relative_mad"]:
        classification = "inconclusive"
        detail = (
            f"relative timing MAD {relative_mad:.2%} exceeds "
            f"the {policy['max_relative_mad']:.2%} limit"
        )
    elif (
        delta_relative > policy["max_regression_relative"]
        and delta_us > policy["max_regression_absolute_us"]
    ):
        classification = "regression"
        detail = "slowdown exceeds both the relative and absolute budgets"
    elif (
        delta_relative < -policy["max_regression_relative"]
        and delta_us < -policy["max_regression_absolute_us"]
    ):
        classification = "improvement"
        detail = "speedup exceeds both the relative and absolute budgets"
    else:
        classification = "within_budget"
        detail = "change does not exceed both regression budgets"

    if policy != candidate_policy:
        detail += "; candidate policy changed, so the baseline policy was used"

    return {
        "id": case_id,
        "classification": classification,
        "base_median_us": base_median,
        "candidate_median_us": candidate_median,
        "delta_us": delta_us,
        "delta_percent": delta_relative * 100.0,
        "detail": detail,
    }


def _case_map(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["id"]): case for case in run["cases"]}


def _case_succeeded(case: Mapping[str, Any]) -> bool:
    result = case.get("result")
    return isinstance(result, dict) and result.get("status") == "success"


def compare_runs(
    base_run: Mapping[str, Any] | None,
    candidate_run: Mapping[str, Any],
    *,
    repository: str,
    pull_request_number: int | None,
    github_run_id: int | None,
    github_run_attempt: int | None,
    target_sha: str,
    merge_base_sha: str,
    candidate_sha: str,
    merge_sha: str | None,
    bootstrap_reason: str | None = None,
) -> dict[str, Any]:
    """Compare one merge-base run with one candidate run."""

    candidate = validate_run_document(candidate_run, expected_revision=candidate_sha)
    base = (
        validate_run_document(base_run, expected_revision=merge_base_sha)
        if base_run is not None
        else None
    )
    candidate_cases = _case_map(candidate)
    base_cases = _case_map(base) if base is not None else {}
    comparisons: list[dict[str, Any]] = []

    if base is None:
        reason = (
            bootstrap_reason or "the baseline does not contain this benchmark suite"
        )
        for case_id, candidate_case in candidate_cases.items():
            if _case_succeeded(candidate_case):
                comparison = _empty_comparison(
                    case_id, "added", f"{reason}; not compared"
                )
                comparison["candidate_median_us"] = _result_median(candidate_case)
            else:
                comparison = _empty_comparison(
                    case_id,
                    "invalid",
                    "candidate benchmark returned "
                    f"{candidate_case['result'].get('status', 'unknown')}",
                )
            comparisons.append(comparison)
    else:
        if base["suite_id"] != candidate["suite_id"]:
            raise CoordinatorError(
                "baseline and candidate suite_id values do not match: "
                f"{base['suite_id']!r} != {candidate['suite_id']!r}"
            )

        environments_match = base["environment"] == candidate["environment"]
        timers_match = base["timer"] == candidate["timer"]

        for case_id in sorted(candidate_cases.keys() | base_cases.keys()):
            base_case = base_cases.get(case_id)
            candidate_case = candidate_cases.get(case_id)
            if base_case is None:
                assert candidate_case is not None
                classification = (
                    "added" if _case_succeeded(candidate_case) else "invalid"
                )
                detail = (
                    "new candidate benchmark; no baseline entry"
                    if classification == "added"
                    else "new candidate benchmark did not complete successfully"
                )
                comparison = _empty_comparison(case_id, classification, detail)
                comparison["candidate_median_us"] = _result_median(candidate_case)
                comparisons.append(comparison)
                continue
            if candidate_case is None:
                comparison = _empty_comparison(
                    case_id,
                    "missing",
                    "baseline benchmark is absent from the candidate suite",
                )
                comparison["base_median_us"] = _result_median(base_case)
                comparisons.append(comparison)
                continue
            if (
                base_case["definition"] != candidate_case["definition"]
                or not timers_match
            ):
                classification = (
                    "changed" if _case_succeeded(candidate_case) else "invalid"
                )
                detail = (
                    "benchmark definition or timing configuration changed; "
                    "measurements were not compared"
                    if classification == "changed"
                    else "changed candidate benchmark did not complete successfully"
                )
                comparison = _empty_comparison(case_id, classification, detail)
                comparison["base_median_us"] = _result_median(base_case)
                comparison["candidate_median_us"] = _result_median(candidate_case)
                comparisons.append(comparison)
                continue
            if not environments_match:
                comparisons.append(
                    _empty_comparison(
                        case_id,
                        "invalid",
                        "baseline and candidate hardware environments differ",
                    )
                )
                continue
            comparisons.append(
                _measurement_comparison(case_id, base_case, candidate_case)
            )

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "suite_id": candidate["suite_id"],
        "repository": repository,
        "pull_request_number": pull_request_number,
        "github_run_id": github_run_id,
        "github_run_attempt": github_run_attempt,
        "target_sha": target_sha,
        "merge_base_sha": merge_base_sha,
        "candidate_sha": candidate_sha,
        "merge_sha": merge_sha,
        "comparisons": comparisons,
    }


def _comparison_counts(report: Mapping[str, Any]) -> dict[str, int]:
    counts = {classification: 0 for classification in CLASSIFICATIONS}
    for comparison in report.get("comparisons", []):
        classification = comparison.get("classification")
        if classification in counts:
            counts[classification] += 1
    return counts


def _comparison_status(report: Mapping[str, Any]) -> str:
    counts = _comparison_counts(report)
    if counts["invalid"]:
        return "invalid"
    if counts["regression"]:
        return "regression"
    if counts["added"] and sum(counts.values()) == counts["added"]:
        return "bootstrap"
    comparable = sum(
        counts[classification]
        for classification in (
            "regression",
            "improvement",
            "within_budget",
            "inconclusive",
        )
    )
    if (
        counts["inconclusive"]
        or counts["changed"]
        or counts["missing"]
        or not comparable
    ):
        return "inconclusive"
    return "passed"


def _summary_text(status: str, counts: Mapping[str, int]) -> str:
    if status == "invalid":
        return "The required benchmark comparison was invalid."
    if status == "regression":
        return f"Detected {counts['regression']} performance regression(s)."
    if status == "bootstrap":
        return "Candidate smoke benchmarks succeeded; no baseline suite was available."
    if status == "inconclusive":
        return "No regressions found; some benchmarks could not be compared."
    return "No kernel performance regressions were found."


def comparison_exit_code(report: Mapping[str, Any]) -> int:
    status = _comparison_status(report)
    if status in {"passed", "bootstrap", "inconclusive"}:
        return 0
    if status == "regression":
        return 1
    return 2


def render_summary(report: Mapping[str, Any]) -> str:
    """Render a bounded GitHub job summary from a comparison report."""

    status = _comparison_status(report)
    counts = _comparison_counts(report)
    status_labels = {
        "passed": "Passed",
        "regression": "Regression detected",
        "bootstrap": "Bootstrap run",
        "inconclusive": "Inconclusive",
        "invalid": "Invalid comparison",
    }
    lines = [
        "# AMD kernel benchmark comparison",
        "",
        f"**{status_labels[status]}:** {_summary_text(status, counts)}",
        "",
        f"- Target branch head: `{str(report['target_sha'])[:12]}`",
        f"- Merge base: `{str(report['merge_base_sha'])[:12]}`",
        f"- Candidate: `{str(report['candidate_sha'])[:12]}`",
        f"- Suite: `{report['suite_id']}`",
        "",
        "| Benchmark | Result | Base (us) | Candidate (us) | Change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    comparisons = list(report.get("comparisons", []))
    priority = {
        "regression": 0,
        "invalid": 1,
        "missing": 2,
        "inconclusive": 3,
        "improvement": 4,
        "changed": 5,
        "added": 6,
        "within_budget": 7,
    }
    comparisons.sort(
        key=lambda item: (priority.get(item["classification"], 99), item["id"])
    )
    for item in comparisons[:50]:
        lines.append(
            f"| `{item['id']}` | {item['classification']} | "
            f"{_format_number(item.get('base_median_us'))} | "
            f"{_format_number(item.get('candidate_median_us'))} | "
            f"{_format_percent(item.get('delta_percent'))} |"
        )
    if len(comparisons) > 50:
        lines.extend(
            ["", f"{len(comparisons) - 50} additional results are in the artifact."]
        )
    lines.extend(
        [
            "",
            (
                "Comparisons require matching benchmark definitions, timing settings, "
                "registrations, and hardware. The merge-base policy supplies the "
                "regression and noise budgets."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _format_number(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _format_percent(value: object) -> str:
    return "-" if value is None else f"{float(value):+.2f}%"


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise CoordinatorError(f"git {' '.join(args)} failed: {message}") from exc
    return completed.stdout.strip()


def resolve_revisions(
    repo: Path, base_ref: str, candidate_ref: str
) -> tuple[str, str, str]:
    target_sha = _git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{base_ref}^{{commit}}",
    )
    candidate_sha = _git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{candidate_ref}^{{commit}}",
    )
    merge_base_sha = _git(repo, "merge-base", target_sha, candidate_sha)
    return target_sha, merge_base_sha, candidate_sha


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    attempts: int = 1,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            check=False,
            capture_output=True,
            text=True,
        )
        transcript = (
            f"attempt {attempt}/{attempts}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(transcript)
        if completed.returncode == 0:
            return
        failures.append(
            (completed.stderr or completed.stdout or f"exit {completed.returncode}")[
                -2000:
            ]
        )
    raise CoordinatorError(
        f"Command failed after {attempts} attempt(s); see {log_path}: "
        + failures[-1].strip()
    )


def _prepare_python_environment(
    checkout: Path,
    environment_dir: Path,
    log_path: Path,
) -> Path:
    _run_logged(
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment_dir)],
        cwd=checkout,
        log_path=log_path,
    )
    python = environment_dir / "bin" / "python"
    site_directories = list((environment_dir / "lib").glob("python*/site-packages"))
    if len(site_directories) != 1:
        raise CoordinatorError(
            f"Cannot locate the Python package directory in {environment_dir}"
        )
    inherited_packages: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if path.is_dir() and any(
            part in {"site-packages", "dist-packages"} for part in path.parts
        ):
            inherited_packages.append(path)
    inherited_packages = list(dict.fromkeys(inherited_packages))
    if not inherited_packages:
        raise CoordinatorError("The prepared Python environment has no package paths")
    overlay = site_directories[0] / "tokenspeed-benchmark-base.pth"
    _atomic_write(
        overlay,
        "".join(f"{path}\n" for path in inherited_packages),
    )
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CoordinatorError(
            "The prepared Python environment does not contain PyTorch"
        ) from exc
    constraint = environment_dir / "benchmark-constraints.txt"
    _atomic_write(constraint, f"torch=={torch_version}\n")
    rocm_probe = [
        str(python),
        "-c",
        (
            "import torch; "
            "assert torch.cuda.is_available(); "
            "assert torch.version.hip; "
            "print(torch.__version__, torch.version.hip)"
        ),
    ]
    _run_logged(rocm_probe, cwd=checkout, log_path=log_path)
    requirements = checkout / _KERNEL_REQUIREMENTS
    if not requirements.is_file():
        raise CoordinatorError(f"ROCm requirements are missing: {requirements}")
    _run_logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--constraint",
            str(constraint),
            "-r",
            str(requirements),
        ],
        cwd=checkout,
        log_path=log_path,
        attempts=3,
    )
    _run_logged(rocm_probe, cwd=checkout, log_path=log_path)
    return python


def _worker_environment(checkout: Path, cache_dir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(checkout / "tokenspeed-kernel/python"),
            str(checkout / "tokenspeed-kernel-amd/python"),
        ]
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
    environment["TORCH_EXTENSIONS_DIR"] = str(cache_dir / "torch-extensions")
    environment["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    for directory in (
        environment["TRITON_CACHE_DIR"],
        environment["TORCH_EXTENSIONS_DIR"],
        environment["XDG_CACHE_HOME"],
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
    return environment


def _run_revision(
    checkout: Path,
    python: Path,
    *,
    suite_relative: Path,
    revision: str,
    output_path: Path,
    log_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    suite = checkout / suite_relative
    command = [
        str(python),
        "-m",
        "tokenspeed_kernel.benchmark.ci",
        "--suite",
        str(suite),
        "--revision",
        revision,
        "--output",
        str(output_path),
    ]
    _run_logged(
        command,
        cwd=checkout,
        log_path=log_path,
        env=_worker_environment(checkout, cache_dir),
    )
    return _load_json(output_path)


def _add_worktree(repo: Path, path: Path, revision: str) -> None:
    _git(repo, "worktree", "add", "--detach", "--force", str(path), revision)


def _remove_worktree(repo: Path, path: Path) -> None:
    if path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )


def _invalid_report(
    *,
    args: argparse.Namespace,
    target_sha: str,
    merge_base_sha: str,
    candidate_sha: str,
    error: BaseException,
) -> dict[str, Any]:
    case = _empty_comparison(
        "infrastructure",
        "invalid",
        f"{type(error).__name__}: {str(error)[:1000]}",
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "suite_id": "unknown",
        "repository": args.repository,
        "pull_request_number": args.pull_request_number,
        "github_run_id": args.github_run_id,
        "github_run_attempt": args.github_run_attempt,
        "target_sha": target_sha,
        "merge_base_sha": merge_base_sha,
        "candidate_sha": candidate_sha,
        "merge_sha": args.merge_sha,
        "comparisons": [case],
    }


def _validate_suite_path(path: Path) -> Path:
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise CoordinatorError("suite must be a relative path within each checkout")
    return path


def _baseline_supports_suite(checkout: Path, suite_relative: Path) -> bool:
    worker_exists = (checkout / _WORKER_RELATIVE_PATH).is_file()
    suite_exists = (checkout / suite_relative).is_file()
    if worker_exists and suite_exists:
        return True
    if not worker_exists and not suite_exists:
        return False
    if worker_exists:
        raise CoordinatorError(
            "merge base contains the benchmark runner but not the requested suite; "
            "this is not an initial bootstrap"
        )
    raise CoordinatorError(
        "merge base contains the requested suite but not its revision-local runner"
    )


def orchestrate(
    args: argparse.Namespace,
    *,
    revisions: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_sha, merge_base_sha, candidate_sha = revisions or resolve_revisions(
        repo,
        args.base_ref,
        args.candidate_ref,
    )
    suite_relative = _validate_suite_path(args.suite)
    temporary_parent = args.work_dir.resolve() if args.work_dir else None
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tokenspeed-kernel-benchmark-",
        dir=temporary_parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        base_checkout = temporary / "base"
        candidate_checkout = temporary / "candidate"
        _add_worktree(repo, base_checkout, merge_base_sha)
        try:
            _add_worktree(repo, candidate_checkout, candidate_sha)
            try:
                candidate_worker = candidate_checkout / _WORKER_RELATIVE_PATH
                candidate_suite = candidate_checkout / suite_relative
                if not candidate_worker.is_file() or not candidate_suite.is_file():
                    raise CoordinatorError(
                        "candidate revision must contain the benchmark worker and suite"
                    )

                base_supported = _baseline_supports_suite(base_checkout, suite_relative)
                if args.environment_mode == "venv":
                    candidate_python = _prepare_python_environment(
                        candidate_checkout,
                        temporary / "candidate-venv",
                        output_dir / "candidate-setup.log",
                    )
                    base_python = (
                        _prepare_python_environment(
                            base_checkout,
                            temporary / "base-venv",
                            output_dir / "base-setup.log",
                        )
                        if base_supported
                        else None
                    )
                else:
                    candidate_python = Path(sys.executable)
                    base_python = Path(sys.executable) if base_supported else None

                base_run = None
                if base_python is not None:
                    base_run = _run_revision(
                        base_checkout,
                        base_python,
                        suite_relative=suite_relative,
                        revision=merge_base_sha,
                        output_path=output_dir / "base.json",
                        log_path=output_dir / "base.log",
                        cache_dir=temporary / "base-cache",
                    )
                candidate_run = _run_revision(
                    candidate_checkout,
                    candidate_python,
                    suite_relative=suite_relative,
                    revision=candidate_sha,
                    output_path=output_dir / "candidate.json",
                    log_path=output_dir / "candidate.log",
                    cache_dir=temporary / "candidate-cache",
                )
                return compare_runs(
                    base_run,
                    candidate_run,
                    repository=args.repository,
                    pull_request_number=args.pull_request_number,
                    github_run_id=args.github_run_id,
                    github_run_attempt=args.github_run_attempt,
                    target_sha=target_sha,
                    merge_base_sha=merge_base_sha,
                    candidate_sha=candidate_sha,
                    merge_sha=args.merge_sha,
                    bootstrap_reason=(
                        None
                        if base_supported
                        else "the merge base does not contain the revision-local runner "
                        "and suite"
                    ),
                )
            finally:
                _remove_worktree(repo, candidate_checkout)
        finally:
            _remove_worktree(repo, base_checkout)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("tokenspeed-kernel/benchmarks/amd/gfx950.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--environment-mode",
        choices=("venv", "current"),
        default="venv",
        help="Use isolated virtual environments or the invoking Python environment",
    )
    parser.add_argument("--repository", default="")
    parser.add_argument("--pull-request-number", type=int)
    parser.add_argument("--github-run-id", type=int)
    parser.add_argument("--github-run-attempt", type=int)
    parser.add_argument("--merge-sha")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_sha = "0" * 40
    merge_base_sha = "0" * 40
    candidate_sha = "0" * 40
    try:
        target_sha, merge_base_sha, candidate_sha = resolve_revisions(
            args.repo.resolve(), args.base_ref, args.candidate_ref
        )
        report = orchestrate(
            args,
            revisions=(target_sha, merge_base_sha, candidate_sha),
        )
    except Exception as exc:  # noqa: BLE001 - failures must become CI artifacts
        report = _invalid_report(
            args=args,
            target_sha=target_sha,
            merge_base_sha=merge_base_sha,
            candidate_sha=candidate_sha,
            error=exc,
        )
    summary = render_summary(report)
    _write_json(output_dir / "comparison.json", report)
    _atomic_write(output_dir / "summary.md", summary)
    print(summary, end="")
    return comparison_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
