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
import os
import platform as host_platform
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import GraphBenchmarkConfig
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
    definition: dict[str, Any]
    policy: dict[str, float]
    request: BenchmarkRequest


@dataclass(frozen=True)
class BenchmarkSuite:
    """Validated input consumed by one revision-local benchmark process."""

    suite_id: str
    required_environment: dict[str, str]
    timer: GraphBenchmarkConfig
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


def _number(
    value: object,
    location: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuiteConfigError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SuiteConfigError(f"{location} must be finite")
    return result


def _parse_timer(raw: object) -> GraphBenchmarkConfig:
    timer = _object(raw, "timer")
    config = GraphBenchmarkConfig(
        calls_per_graph=timer["calls_per_graph"],
        eager_warmup_iterations=timer["eager_warmup_iterations"],
        replay_warmup_iterations=timer["replay_warmup_iterations"],
        measurement_blocks=timer["measurement_blocks"],
    )
    if config.measurement_blocks < 5:
        raise SuiteConfigError("timer.measurement_blocks must be at least 5")
    return config


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
    request = BenchmarkRequest(
        family=definition["family"],
        mode=definition["mode"],
        parameters=definition["parameters"],
        solution=definition.get("solution"),
        registration=definition.get("registration"),
        seed=definition["seed"],
        definition_version=definition["definition_version"],
    )

    normalized = {
        "family": request.family,
        "mode": request.mode,
        "parameters": request.parameters,
        "solution": request.solution,
        "registration": request.registration,
        "seed": request.seed,
        "definition_version": request.definition_version,
    }
    return normalized, request


def _parse_case(raw: object, index: int) -> SuiteCase:
    location = f"cases[{index}]"
    case = _object(raw, location)
    case_id = _nonempty_string(case["id"], f"{location}.id")
    definition, request = _parse_definition(
        case["definition"], f"{location}.definition"
    )
    policy = _parse_policy(case["policy"], f"{location}.policy")
    return SuiteCase(case_id, definition, policy, request)


def load_suite(path: str | Path) -> BenchmarkSuite:
    """Load the fields needed to execute a versioned benchmark suite."""

    suite_path = Path(path)
    try:
        raw = json.loads(suite_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SuiteConfigError(f"cannot read {suite_path}: {error}") from error

    suite = _object(raw, "suite")
    try:
        schema_version = suite["schema_version"]
        if isinstance(schema_version, bool) or schema_version != _SCHEMA_VERSION:
            raise SuiteConfigError(
                f"unsupported schema_version {schema_version}; expected {_SCHEMA_VERSION}"
            )
        suite_id = _nonempty_string(suite["suite_id"], "suite_id")
        required_environment = _parse_environment(suite["environment"])
        timer = _parse_timer(suite["timer"])
        cases_raw = suite["cases"]
        if not isinstance(cases_raw, list) or not cases_raw:
            raise SuiteConfigError("cases must be a non-empty JSON array")
        cases = tuple(_parse_case(case, index) for index, case in enumerate(cases_raw))
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


def _timer_payload(timer: GraphBenchmarkConfig) -> dict[str, int]:
    return {
        "calls_per_graph": timer.calls_per_graph,
        "eager_warmup_iterations": timer.eager_warmup_iterations,
        "replay_warmup_iterations": timer.replay_warmup_iterations,
        "measurement_blocks": timer.measurement_blocks,
    }


def _failure_payload(
    status: BenchmarkStatus,
    phase: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "status": status.value,
        "registration_name": None,
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
        config,
        timer=None,
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
                BenchmarkStatus.ENVIRONMENT_INVALID, "environment", mismatch
            )
        elif harness_error is not None:
            result_payload = _failure_payload(
                BenchmarkStatus.SETUP_FAILURE, "runner_setup", harness_error
            )
        else:
            assert harness is not None
            try:
                result = harness.run(case.request)
                expected_context = (
                    suite.timer.calls_per_graph,
                    suite.timer.eager_warmup_iterations,
                    suite.timer.replay_warmup_iterations,
                    suite.timer.measurement_blocks,
                    environment.get("vendor"),
                    environment.get("arch"),
                    environment.get("device_name"),
                )
                actual_context = (
                    result.calls_per_graph,
                    result.eager_warmup_iterations,
                    result.replay_warmup_iterations,
                    result.measurement_blocks,
                    result.platform_vendor,
                    result.platform_arch,
                    result.device_name,
                )
                if result.succeeded and actual_context != expected_context:
                    raise RuntimeError(
                        "successful benchmark reported the wrong context"
                    )
                result_payload = _result_payload(result)
            except Exception as error:  # noqa: BLE001 - benchmark failures are data
                result_payload = _failure_payload(
                    BenchmarkStatus.EXECUTION_FAILURE, "runner", error
                )

        case_payloads.append(
            {
                "id": case.id,
                "definition": case.definition,
                "policy": case.policy,
                "result": result_payload,
            }
        )

    return {
        "schema_version": _SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "revision": revision,
        "environment": environment,
        "timer": _timer_payload(suite.timer),
        "cases": case_payloads,
    }


def _write_output(payload: dict[str, Any], path: str | Path) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(path) == "-":
        sys.stdout.write(serialized)
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a revision-local kernel benchmark suite for CI"
    )
    parser.add_argument("--suite", required=True, help="Versioned suite JSON path")
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
