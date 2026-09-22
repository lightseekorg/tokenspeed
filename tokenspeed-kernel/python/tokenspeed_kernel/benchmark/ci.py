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

"""Execute one revision's registration-level benchmark suite for CI."""

from __future__ import annotations

import argparse
import json
import math
import platform as host_platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import GraphBenchmarkConfig, GraphTimer
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkRequest,
    BenchmarkStatus,
    KernelBenchmarkHarness,
    KernelBenchmarkResult,
)
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "BenchmarkSuite",
    "SuiteCase",
    "SuiteConfigError",
    "load_suite",
    "main",
    "run_suite",
]

_SCHEMA_VERSION = 1


class SuiteConfigError(ValueError):
    """A benchmark suite cannot be interpreted without guessing intent."""


@dataclass(frozen=True)
class SuiteCase:
    """One stable benchmark identity and its revision-local request."""

    id: str
    comparison_epoch: int
    definition: dict[str, Any]
    policy: dict[str, float]
    measurement_blocks: int
    request: BenchmarkRequest


@dataclass(frozen=True)
class BenchmarkSuite:
    """Validated input consumed by one revision-local benchmark process."""

    suite_id: str
    required_environment: dict[str, str]
    timer: GraphBenchmarkConfig
    default_measurement_blocks: int
    cases: tuple[SuiteCase, ...]
    schema_version: int = _SCHEMA_VERSION


def _object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuiteConfigError(f"{location} must be a JSON object")
    return value


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SuiteConfigError(f"{location} must be a non-empty string")
    return value


def _number(value: object, location: str) -> float:
    if not isinstance(value, (int, float)):
        raise SuiteConfigError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SuiteConfigError(f"{location} must be finite")
    return result


def _parse_timer(raw: object) -> tuple[GraphBenchmarkConfig, int]:
    timer = _object(raw, "timer")
    if "calls_per_graph" in timer:
        raise SuiteConfigError(
            "timer.calls_per_graph was removed; each graph contains one operation"
        )
    config = GraphBenchmarkConfig(
        eager_warmup_iterations=timer["eager_warmup_iterations"],
        replay_warmup_iterations=timer["replay_warmup_iterations"],
    )
    measurement_blocks = timer["measurement_blocks"]
    if not isinstance(measurement_blocks, int) or measurement_blocks < 5:
        raise SuiteConfigError("timer.measurement_blocks must be at least 5")
    return config, measurement_blocks


def _parse_environment(raw: object) -> dict[str, str]:
    environment = _object(raw, "environment")
    return {
        "vendor": environment["vendor"],
        "arch": environment["arch"],
    }


def _parse_policy(raw: object, location: str) -> dict[str, float]:
    policy = _object(raw, location)
    parsed = {
        field: _number(policy[field], f"{location}.{field}")
        for field in (
            "max_regression_relative",
            "max_regression_absolute_us",
            "max_relative_mad",
        )
    }
    if not 0.0 <= parsed["max_regression_relative"] < 1.0:
        raise SuiteConfigError(f"{location}.max_regression_relative must be in [0, 1)")
    if parsed["max_regression_absolute_us"] < 0.0:
        raise SuiteConfigError(
            f"{location}.max_regression_absolute_us must be nonnegative"
        )
    if not 0.0 < parsed["max_relative_mad"] < 1.0:
        raise SuiteConfigError(f"{location}.max_relative_mad must be in (0, 1)")
    return parsed


def _parse_definition(
    raw: object, location: str
) -> tuple[dict[str, Any], BenchmarkRequest]:
    definition = _object(raw, location)
    cold_cache = definition.get("cold_cache", True)
    if not isinstance(cold_cache, bool):
        raise SuiteConfigError(f"{location}.cold_cache must be a boolean")
    request = BenchmarkRequest(
        family=definition["family"],
        mode=definition["mode"],
        parameters=definition["parameters"],
        solution=definition.get("solution"),
        registration=definition.get("registration"),
        cold_cache=cold_cache,
        seed=definition["seed"],
    )

    normalized = {
        "family": request.family,
        "mode": request.mode,
        "parameters": request.parameters,
        "solution": request.solution,
        "registration": request.registration,
        "cold_cache": request.cold_cache,
        "seed": request.seed,
    }
    return normalized, request


def _parse_case(
    raw: object,
    index: int,
    suite_measurement_blocks: int,
) -> SuiteCase:
    location = f"cases[{index}]"
    case = _object(raw, location)
    case_id = _nonempty_string(case["id"], f"{location}.id")
    comparison_epoch = case["comparison_epoch"]
    if not isinstance(comparison_epoch, int) or comparison_epoch <= 0:
        raise SuiteConfigError(
            f"{location}.comparison_epoch must be a positive integer"
        )
    definition, request = _parse_definition(
        case["definition"], f"{location}.definition"
    )
    policy = _parse_policy(case["policy"], f"{location}.policy")
    measurement_blocks = case.get("measurement_blocks", suite_measurement_blocks)
    if not isinstance(measurement_blocks, int) or measurement_blocks < 5:
        raise SuiteConfigError(f"{location}.measurement_blocks must be at least 5")
    return SuiteCase(
        id=case_id,
        comparison_epoch=comparison_epoch,
        definition=definition,
        policy=policy,
        measurement_blocks=measurement_blocks,
        request=request,
    )


def _expand_case(raw: object, index: int) -> list[dict[str, Any]]:
    location = f"cases[{index}]"
    case = _object(raw, location)
    definition = _object(case.get("definition"), f"{location}.definition")
    parameters = _object(
        definition.get("parameters"), f"{location}.definition.parameters"
    )
    dimensions = sorted(
        (name, values)
        for name, values in parameters.items()
        if isinstance(values, list)
    )
    if not dimensions:
        return [case]

    case_id = _nonempty_string(case.get("id"), f"{location}.id")
    for name, values in dimensions:
        if not values:
            raise SuiteConfigError(
                f"{location}.definition.parameters.{name} must not be an empty list"
            )

    combinations = tuple(product(*(values for _, values in dimensions)))
    expanded = []
    for expansion_index, combination in enumerate(combinations):
        expanded_parameters = dict(parameters)
        for (name, _), value in zip(dimensions, combination, strict=True):
            expanded_parameters[name] = value
        expanded.append(
            {
                **case,
                "id": (
                    case_id
                    if len(combinations) == 1
                    else f"{case_id}_{expansion_index}"
                ),
                "definition": {
                    **definition,
                    "parameters": expanded_parameters,
                },
            }
        )
    return expanded


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SuiteConfigError(f"cannot read {path}: {error}") from error


def load_suite(path: str | Path) -> BenchmarkSuite:
    """Load the fields needed to execute a benchmark suite."""

    suite_path = Path(path)
    suite = _object(_read_json(suite_path), "suite")
    try:
        schema_version = suite["schema_version"]
        if schema_version != _SCHEMA_VERSION:
            raise SuiteConfigError(
                f"unsupported schema_version {schema_version}; expected {_SCHEMA_VERSION}"
            )
        suite_id = _nonempty_string(suite["suite_id"], "suite_id")
        required_environment = _parse_environment(suite["environment"])
        timer, default_measurement_blocks = _parse_timer(suite["timer"])
        root_cases = suite["cases"]
        if not isinstance(root_cases, list):
            raise SuiteConfigError("cases must be a JSON array")
        cases_raw = list(root_cases)
        case_files = suite.get("case_files", [])
        if not isinstance(case_files, list) or not all(
            isinstance(name, str) and name for name in case_files
        ):
            raise SuiteConfigError("case_files must be an array of file names")
        for name in case_files:
            fragment = _object(
                _read_json(suite_path.parent / name), f"case file {name}"
            )
            if fragment.get("schema_version") != schema_version:
                raise SuiteConfigError(
                    f"case file {name} must use schema_version {schema_version}"
                )
            fragment_cases = fragment.get("cases")
            if not isinstance(fragment_cases, list):
                raise SuiteConfigError(f"case file {name} must contain a cases array")
            cases_raw.extend(fragment_cases)
        if not cases_raw:
            raise SuiteConfigError("cases must be a non-empty JSON array")
        expanded_cases = [
            expanded
            for index, case in enumerate(cases_raw)
            for expanded in _expand_case(case, index)
        ]
        cases = tuple(
            _parse_case(case, index, default_measurement_blocks)
            for index, case in enumerate(expanded_cases)
        )
    except SuiteConfigError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SuiteConfigError(f"invalid benchmark suite: {error}") from error

    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise SuiteConfigError("duplicate case IDs are not allowed")

    return BenchmarkSuite(
        suite_id=suite_id,
        required_environment=required_environment,
        timer=timer,
        default_measurement_blocks=default_measurement_blocks,
        cases=tuple(sorted(cases, key=lambda case: case.id)),
        schema_version=schema_version,
    )


def _collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "vendor": "",
        "arch": "",
        "device_name": "",
        "device_count": 0,
        "total_memory_bytes": 0,
        "compute_units": 0,
        "python_version": host_platform.python_version(),
        "torch_version": str(torch.__version__),
        "hip_runtime_version": (
            str(torch.version.hip) if torch.version.hip is not None else None
        ),
        "cuda_runtime_version": (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
    }
    try:
        detected = current_platform()
    except Exception as error:  # noqa: BLE001 - reported as environment data
        environment["detection_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        return environment

    environment.update(
        {
            "vendor": detected.vendor,
            "arch": detected.arch,
            "device_name": detected.device_name,
            "device_count": detected.device_count,
            "total_memory_bytes": detected.total_memory,
            "compute_units": detected.sm_count,
        }
    )
    return environment


def _timer_payload(
    timer: GraphBenchmarkConfig,
    default_measurement_blocks: int,
) -> dict[str, int]:
    return {
        "eager_warmup_iterations": timer.eager_warmup_iterations,
        "replay_warmup_iterations": timer.replay_warmup_iterations,
        "measurement_blocks": default_measurement_blocks,
    }


def _failure_payload(
    status: BenchmarkStatus,
    phase: str,
    error: BaseException,
    *,
    cold_cache: bool,
) -> dict[str, Any]:
    return {
        "status": status.value,
        "registration_name": None,
        "cold_cache": cold_cache,
        "timing_mode": "graph_replay",
        "metric": "device_time_per_invocation",
        "unit": "us",
        "samples_us": [],
        "correctness": None,
        "error_phase": phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _result_payload(result: KernelBenchmarkResult) -> dict[str, Any]:
    """Keep only measurements and diagnostics used across revisions."""

    return {
        "status": result.status.value,
        "registration_name": result.registration_name,
        "cold_cache": result.cold_cache,
        "timing_mode": result.timing_mode,
        "metric": result.metric,
        "unit": result.unit,
        "samples_us": list(result.samples_us),
        "correctness": result.correctness,
        "error_phase": result.error_phase,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def _environment_mismatch(
    required: dict[str, str],
    actual: dict[str, Any],
) -> RuntimeError | None:
    mismatches = [
        f"{key}={actual.get(key)!r} (required {expected!r})"
        for key, expected in required.items()
        if actual.get(key) != expected
    ]
    if not mismatches:
        return None
    return RuntimeError("benchmark environment mismatch: " + ", ".join(mismatches))


def _create_harness(config: GraphBenchmarkConfig) -> KernelBenchmarkHarness:
    return KernelBenchmarkHarness(
        GraphTimer(config),
        platform_provider=current_platform,
    )


def run_suite(
    suite: BenchmarkSuite,
    revision: str,
    *,
    harness_factory: Callable[[GraphBenchmarkConfig], KernelBenchmarkHarness],
    environment_provider: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run a validated suite and return its coordinator-facing envelope."""

    try:
        environment = dict(environment_provider())
    except Exception as error:  # noqa: BLE001 - reported as environment data
        environment = {
            "vendor": "",
            "arch": "",
            "device_name": "",
            "detection_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    mismatch = _environment_mismatch(suite.required_environment, environment)
    harness: KernelBenchmarkHarness | None = None
    harness_error: Exception | None = None
    if mismatch is None:
        try:
            harness = harness_factory(suite.timer)
        except Exception as error:  # noqa: BLE001 - returned for every case
            harness_error = error

    case_payloads: list[dict[str, Any]] = []
    for case in suite.cases:
        if mismatch is not None:
            result_payload = _failure_payload(
                BenchmarkStatus.ENVIRONMENT_INVALID,
                "environment",
                mismatch,
                cold_cache=case.request.cold_cache,
            )
        elif harness_error is not None:
            result_payload = _failure_payload(
                BenchmarkStatus.SETUP_FAILURE,
                "runner_setup",
                harness_error,
                cold_cache=case.request.cold_cache,
            )
        else:
            assert harness is not None
            try:
                result_payload = _result_payload(
                    harness.run(
                        case.request,
                        measurement_blocks=case.measurement_blocks,
                    )
                )
            except Exception as error:  # noqa: BLE001 - benchmark failures are data
                result_payload = _failure_payload(
                    BenchmarkStatus.EXECUTION_FAILURE,
                    "runner",
                    error,
                    cold_cache=case.request.cold_cache,
                )

        case_payloads.append(
            {
                "id": case.id,
                "comparison_epoch": case.comparison_epoch,
                "definition": case.definition,
                "policy": case.policy,
                "measurement_blocks": case.measurement_blocks,
                "result": result_payload,
            }
        )

    return {
        "schema_version": _SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "revision": revision,
        "environment": environment,
        "timer": _timer_payload(suite.timer, suite.default_measurement_blocks),
        "cases": case_payloads,
    }


def _write_output(payload: dict[str, Any], path: str | Path) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(path) == "-":
        sys.stdout.write(serialized)
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a revision-local kernel benchmark suite for CI"
    )
    parser.add_argument("--suite", required=True, help="Benchmark suite JSON path")
    parser.add_argument(
        "--revision",
        required=True,
        help="Full lowercase Git object ID for this checkout",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Result JSON path, or '-' for standard output",
    )
    args = parser.parse_args(argv)

    try:
        suite = load_suite(args.suite)
        payload = run_suite(
            suite,
            args.revision,
            harness_factory=_create_harness,
            environment_provider=_collect_environment,
        )
        _write_output(payload, args.output)
    except (OSError, SuiteConfigError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
