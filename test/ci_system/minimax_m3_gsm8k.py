#!/usr/bin/env python3
"""Run and validate the pinned MiniMax-M3 GSM8K release workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

EXPECTED_GSM8K_SAMPLES = 1319
MAX_RECORDED_FAILURES = 50


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_failure(failures: list[str], message: str) -> None:
    if len(failures) < MAX_RECORDED_FAILURES:
        failures.append(message)


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        gpu_ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be comma-separated integers"
        ) from exc
    if (
        len(gpu_ids) != 4
        or len(set(gpu_ids)) != 4
        or any(gpu_id < 0 for gpu_id in gpu_ids)
    ):
        raise argparse.ArgumentTypeError(
            "exactly four distinct non-negative GPU IDs are required"
        )
    return gpu_ids


def _parse_probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("score threshold must be numeric") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("score threshold must be finite and in [0, 1]")
    return parsed


@dataclass(frozen=True)
class Gsm8kConfig:
    evalscope: Path
    model: str
    base_url: str
    control_url: str
    output_dir: Path
    gpu_ids: tuple[int, ...]
    expected_samples: int
    minimum_score: float
    eval_batch_size: int
    max_tokens: int
    seed: int


@dataclass
class MemoryMonitor:
    process: subprocess.Popen[Any]

    def stop(self) -> None:
        """Stop only the sampler process created for this workload."""

        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def start_memory_monitor(
    output_path: Path,
    gpu_ids: tuple[int, ...],
    *,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> MemoryMonitor:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    process = popen_factory(
        [
            "nvidia-smi",
            "-i",
            ",".join(str(gpu_id) for gpu_id in gpu_ids),
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
            "-l",
            "1",
            "-f",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return MemoryMonitor(process)


def run_evalscope(
    config: Gsm8kConfig,
    *,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> int:
    """Run EvalScope while retaining and forwarding its complete stdout."""

    work_dir = config.output_dir / "evalscope"
    stdout_path = config.output_dir / "evalscope.stdout.log"
    shutil.rmtree(work_dir, ignore_errors=True)
    stdout_path.unlink(missing_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    generation_config = json.dumps(
        {"do_sample": False, "temperature": 0, "max_tokens": config.max_tokens},
        separators=(",", ":"),
    )
    command = [
        str(config.evalscope),
        "eval",
        "--model",
        config.model,
        "--api-url",
        f"{config.base_url.rstrip('/')}/v1",
        "--api-key",
        "EMPTY_TOKEN",
        "--datasets",
        "gsm8k",
        "--eval-batch-size",
        str(config.eval_batch_size),
        "--stream",
        "--generation-config",
        generation_config,
        "--seed",
        str(config.seed),
        "--work-dir",
        str(work_dir),
        "--no-timestamp",
    ]
    print("$ " + " ".join(command), flush=True)
    process = popen_factory(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("EvalScope process did not expose stdout")
    with stdout_path.open("w") as retained:
        for line in process.stdout:
            retained.write(line)
            retained.flush()
            print(line, end="", flush=True)
    return_code = process.wait()
    (config.output_dir / "evalscope.exit_code").write_text(f"{return_code}\n")
    return return_code


def _read_jsonl(path: Path, failures: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_failure(failures, f"{label} file is missing: {path}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _append_failure(
                failures, f"{label} line {line_number} is invalid JSON: {exc}"
            )
            continue
        if not isinstance(value, dict):
            _append_failure(failures, f"{label} line {line_number} is not an object")
            continue
        rows.append(value)
    return rows


def _validate_indices(
    rows: list[dict[str, Any]],
    *,
    label: str,
    expected_samples: int,
    failures: list[str],
) -> set[int]:
    indices: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            _append_failure(
                failures, f"{label} row {row_number} has a non-integer index"
            )
            continue
        indices.append(index)
    unique = set(indices)
    if len(unique) != len(indices):
        _append_failure(failures, f"{label} contains duplicate indices")
    expected = set(range(expected_samples))
    missing = sorted(expected - unique)
    unexpected = sorted(unique - expected)
    if missing:
        _append_failure(failures, f"{label} is missing indices: {missing[:10]}")
    if unexpected:
        _append_failure(failures, f"{label} has unexpected indices: {unexpected[:10]}")
    return unique


def validate_gsm8k_outputs(
    predictions_path: Path,
    reviews_path: Path,
    *,
    expected_samples: int,
    minimum_score: float,
) -> dict[str, Any]:
    failures: list[str] = []
    predictions = _read_jsonl(predictions_path, failures, "predictions")
    reviews = _read_jsonl(reviews_path, failures, "reviews")
    if len(predictions) != expected_samples:
        _append_failure(
            failures,
            f"expected {expected_samples} predictions, found {len(predictions)}",
        )
    if len(reviews) != expected_samples:
        _append_failure(
            failures, f"expected {expected_samples} reviews, found {len(reviews)}"
        )

    prediction_indices = _validate_indices(
        predictions,
        label="predictions",
        expected_samples=expected_samples,
        failures=failures,
    )
    review_indices = _validate_indices(
        reviews,
        label="reviews",
        expected_samples=expected_samples,
        failures=failures,
    )

    nonempty_outputs = 0
    positive_usage = 0
    model_errors = 0
    for row_number, row in enumerate(predictions, start=1):
        output = row.get("model_output")
        if not isinstance(output, dict):
            _append_failure(
                failures, f"prediction row {row_number} has no model_output"
            )
            continue
        if output.get("error") not in (None, ""):
            model_errors += 1
            _append_failure(
                failures, f"prediction row {row_number} reports a model error"
            )
        try:
            content = output["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if isinstance(content, str) and content.strip():
            nonempty_outputs += 1
        else:
            _append_failure(failures, f"prediction row {row_number} has empty content")
        usage = output.get("usage")
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        if (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens > 0
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens > 0
        ):
            positive_usage += 1
        else:
            _append_failure(failures, f"prediction row {row_number} has invalid usage")

    scores: list[float] = []
    for row_number, row in enumerate(reviews, start=1):
        try:
            score = row["sample_score"]["score"]["value"]["acc"]
        except (KeyError, TypeError):
            _append_failure(failures, f"review row {row_number} has no accuracy score")
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            _append_failure(
                failures, f"review row {row_number} has a non-numeric score"
            )
            continue
        numeric_score = float(score)
        if not math.isfinite(numeric_score) or numeric_score not in (0.0, 1.0):
            _append_failure(
                failures, f"review row {row_number} has invalid score {score!r}"
            )
            continue
        scores.append(numeric_score)

    score = sum(scores) / expected_samples if len(scores) == expected_samples else None
    if score is None:
        _append_failure(failures, "could not compute an exact 1319-sample score")
    elif score < minimum_score:
        _append_failure(
            failures,
            f"GSM8K score {score:.12g} is below minimum {minimum_score:.12g}",
        )

    return {
        "counts": {
            "predictions": len(predictions),
            "reviews": len(reviews),
            "nonempty_outputs": nonempty_outputs,
            "positive_usage": positive_usage,
            "model_errors": model_errors,
        },
        "indices": {
            "expected_min": 0,
            "expected_max": expected_samples - 1,
            "predictions_complete": prediction_indices == set(range(expected_samples)),
            "reviews_complete": review_indices == set(range(expected_samples)),
        },
        "quality": {
            "correct": int(sum(scores)),
            "score": score,
            "finite_binary_scores": len(scores),
        },
        "failures": failures,
    }


def parse_gpu_memory(path: Path, expected_gpu_ids: tuple[int, ...]) -> dict[str, Any]:
    failures: list[str] = []
    samples: dict[int, list[int]] = {gpu_id: [] for gpu_id in expected_gpu_ids}
    if not path.is_file():
        failures.append(f"GPU memory file is missing: {path}")
    else:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                index_text, memory_text = line.split(",", 1)
                gpu_id = int(index_text.strip())
                memory_mib = int(memory_text.strip())
            except (ValueError, TypeError):
                _append_failure(
                    failures, f"invalid GPU memory row {line_number}: {line!r}"
                )
                continue
            if gpu_id not in samples:
                _append_failure(
                    failures, f"unexpected GPU ID in memory trace: {gpu_id}"
                )
                continue
            if memory_mib < 0:
                _append_failure(failures, f"negative memory sample for GPU {gpu_id}")
                continue
            samples[gpu_id].append(memory_mib)
    for gpu_id, values in samples.items():
        if not values:
            _append_failure(failures, f"no memory samples for GPU {gpu_id}")
    return {
        "gpu_ids": list(expected_gpu_ids),
        "sample_counts": {
            str(gpu_id): len(samples[gpu_id]) for gpu_id in expected_gpu_ids
        },
        "peak_mib": {
            str(gpu_id): max(samples[gpu_id]) if samples[gpu_id] else None
            for gpu_id in expected_gpu_ids
        },
        "failures": failures,
    }


def probe_health(
    base_url: str,
    control_url: str,
    *,
    session: requests.Session,
) -> dict[str, Any]:
    probes = {
        "gateway_readiness": f"{base_url.rstrip('/')}/readiness",
        "control_health": f"{control_url.rstrip('/')}/health",
        "engine_health": f"{control_url.rstrip('/')}/health_check",
    }
    result: dict[str, Any] = {}
    failures: list[str] = []
    for name, url in probes.items():
        try:
            response = session.get(url, timeout=30)
            status_code = response.status_code
            result[name] = {"status_code": status_code}
            if status_code != 200:
                failures.append(f"{name} returned HTTP {status_code}")
        except requests.RequestException as exc:
            result[name] = {"status_code": None, "error": type(exc).__name__}
            failures.append(f"{name} request failed: {type(exc).__name__}")
    result["failures"] = failures
    return result


def run(
    config: Gsm8kConfig,
    *,
    session: requests.Session | None = None,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    monitor_factory: Callable[
        [Path, tuple[int, ...]], MemoryMonitor
    ] = start_memory_monitor,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    memory_path = config.output_dir / "memory.csv"
    monitor: MemoryMonitor | None = None
    evalscope_exit_code: int | None = None
    orchestration_failures: list[str] = []
    try:
        monitor = monitor_factory(memory_path, config.gpu_ids)
        evalscope_exit_code = run_evalscope(config, popen_factory=popen_factory)
        if evalscope_exit_code != 0:
            orchestration_failures.append(
                f"EvalScope exited with status {evalscope_exit_code}"
            )
    except Exception as exc:  # Preserve an artifact for infrastructure failures.
        orchestration_failures.append(
            f"workload execution failed: {type(exc).__name__}: {exc}"
        )
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception as exc:
                orchestration_failures.append(
                    f"memory sampler stop failed: {type(exc).__name__}: {exc}"
                )

    evalscope_dir = config.output_dir / "evalscope"
    predictions_path = evalscope_dir / "predictions" / config.model / "gsm8k_main.jsonl"
    reviews_path = evalscope_dir / "reviews" / config.model / "gsm8k_main.jsonl"
    validation = validate_gsm8k_outputs(
        predictions_path,
        reviews_path,
        expected_samples=config.expected_samples,
        minimum_score=config.minimum_score,
    )
    memory = parse_gpu_memory(memory_path, config.gpu_ids)
    owned_session = session is None
    http = requests.Session() if session is None else session
    try:
        health = probe_health(config.base_url, config.control_url, session=http)
    finally:
        if owned_session:
            http.close()

    failures = [
        *orchestration_failures,
        *validation.pop("failures"),
        *memory.pop("failures"),
        *health.pop("failures"),
    ]
    result = {
        "schema_version": 1,
        "workload": "minimax_m3_gsm8k",
        "ok": not failures,
        "model": config.model,
        "expected_samples": config.expected_samples,
        "minimum_score": config.minimum_score,
        "evalscope": {
            "exit_code": evalscope_exit_code,
            "predictions_path": str(predictions_path),
            "reviews_path": str(reviews_path),
            "stdout_path": str(config.output_dir / "evalscope.stdout.log"),
        },
        **validation,
        "gpu_memory": memory,
        "health": health,
        "failures": failures[:MAX_RECORDED_FAILURES],
        "failure_count": len(failures),
    }
    _write_json_atomic(config.output_dir / "validation.json", result)
    score = result["quality"]["score"]
    print(
        "GSM8K validation: "
        f"{'PASS' if result['ok'] else 'FAIL'} score={score} "
        f"samples={result['counts']['reviews']}/{config.expected_samples}",
        flush=True,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evalscope", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids, required=True)
    parser.add_argument(
        "--expected-samples",
        type=int,
        choices=(EXPECTED_GSM8K_SAMPLES,),
        default=EXPECTED_GSM8K_SAMPLES,
    )
    parser.add_argument("--minimum-score", type=_parse_probability, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.eval_batch_size <= 0 or args.max_tokens <= 0:
        print("eval batch size and max tokens must be positive", file=sys.stderr)
        return 2
    config = Gsm8kConfig(
        evalscope=args.evalscope,
        model=args.model,
        base_url=args.base_url,
        control_url=args.control_url,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        expected_samples=args.expected_samples,
        minimum_score=args.minimum_score,
        eval_batch_size=args.eval_batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    return 0 if run(config)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
