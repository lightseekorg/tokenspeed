#!/usr/bin/env python3
"""Run the immutable MiniMax-M3 1,048,575 + 1 release boundary check."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from minimax_m3_gsm8k import (
    MemoryMonitor,
    parse_gpu_memory,
    probe_health,
    start_memory_monitor,
)

PROMPT_TOKENS = 1_048_575
OUTPUT_TOKENS = 1
EXPECTED_OUTPUT_IDS = (123,)
EXPECTED_TEXT = "{"
CHUNK_SIZE = 8192
EXPECTED_FULL_CHUNKS = 127
EXPECTED_TAIL_TOKENS = 8191
MAX_RECORDED_FAILURES = 50
MAX_RECORDED_LOG_MATCHES = 20

HARD_FAILURE_PATTERNS = (
    ("retract_failed", re.compile(r"Retract failed", re.IGNORECASE)),
    ("host_capacity_exhausted", re.compile(r"host capacity exhausted", re.IGNORECASE)),
    ("aborting_request", re.compile(r"aborting request", re.IGNORECASE)),
    (
        "allocator_or_buffer_overflow",
        re.compile(
            r"(?:AlignedAllocator|allocator).*?(?:overflow|failed)|"
            r"Buffer overflow(?: when allocating memory)?",
            re.IGNORECASE,
        ),
    ),
    (
        "scheduler_exception",
        re.compile(r"Scheduler hit an exception:|scheduler exception", re.IGNORECASE),
    ),
    (
        "traceback",
        re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    ),
    (
        "cuda_failure",
        re.compile(
            r"CUDA (?:out of memory|error)|OutOfMemoryError|illegal memory access",
            re.IGNORECASE,
        ),
    ),
    (
        "nccl_failure",
        re.compile(r"NCCL.*(?:abort|unhandled|error|failure)", re.IGNORECASE),
    ),
    ("worker_restart", re.compile(r"worker.*(?:restart|restarted)", re.IGNORECASE)),
    (
        "fatal_process_failure",
        re.compile(r"segmentation fault|segfault|\bFATAL\b", re.IGNORECASE),
    ),
)


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


@dataclass(frozen=True)
class ExactLongContextContract:
    prompt_tokens: int = PROMPT_TOKENS
    output_tokens: int = OUTPUT_TOKENS
    expected_output_ids: tuple[int, ...] = EXPECTED_OUTPUT_IDS
    expected_text: str = EXPECTED_TEXT
    chunk_size: int = CHUNK_SIZE
    expected_full_chunks: int = EXPECTED_FULL_CHUNKS
    expected_tail_tokens: int = EXPECTED_TAIL_TOKENS


RELEASE_CONTRACT = ExactLongContextContract()


@dataclass(frozen=True)
class ExactLongContextConfig:
    model: str
    base_url: str
    control_url: str
    server_log: Path
    output_dir: Path
    gpu_ids: tuple[int, ...]
    request_timeout_seconds: float
    memory_limit_mib: int


def validate_response(
    status_code: int | None,
    body: Any,
    contract: ExactLongContextContract,
) -> dict[str, Any]:
    failures: list[str] = []
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    output_ids: Any = None
    text: Any = None
    if status_code != 200:
        failures.append(f"generate request returned HTTP {status_code}")
    if not isinstance(body, list) or len(body) != 1 or not isinstance(body[0], dict):
        failures.append("generate response must be a one-item object list")
    else:
        output = body[0]
        output_ids = output.get("output_ids")
        text = output.get("text")
        meta = output.get("meta_info")
        if not isinstance(meta, dict):
            failures.append("generate response has no meta_info object")
            meta = {}
        raw_request_id = meta.get("id")
        request_id = raw_request_id if isinstance(raw_request_id, str) else None
        prompt_tokens = meta.get("prompt_tokens")
        completion_tokens = meta.get("completion_tokens")
        cached_tokens = meta.get("cached_tokens")
        if output_ids != list(contract.expected_output_ids):
            failures.append(
                f"expected output_ids {list(contract.expected_output_ids)}, found {output_ids!r}"
            )
        if text != contract.expected_text:
            failures.append(f"expected text {contract.expected_text!r}, found {text!r}")
        if prompt_tokens != contract.prompt_tokens:
            failures.append(
                f"expected {contract.prompt_tokens} prompt tokens, found {prompt_tokens!r}"
            )
        if completion_tokens != contract.output_tokens:
            failures.append(
                f"expected {contract.output_tokens} completion token, found {completion_tokens!r}"
            )
        if cached_tokens != 0:
            failures.append(f"expected cached_tokens=0, found {cached_tokens!r}")
        if not request_id:
            failures.append("generate response has no request ID")
    return {
        "status_code": status_code,
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "output_ids": output_ids,
        "text": text,
        "failures": failures,
    }


def _read_log_window(server_log: Path, start_offset: int) -> str:
    if not server_log.is_file():
        return ""
    data = server_log.read_bytes()
    return data[min(start_offset, len(data)) :].decode(errors="replace")


def wait_for_request_finish(
    server_log: Path,
    *,
    start_offset: int,
    request_id: str,
    timeout_seconds: float = 30,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    deadline = monotonic() + timeout_seconds
    finish = re.compile(rf"Req:\s*{re.escape(request_id)}\s+Finish!")
    window = ""
    while True:
        window = _read_log_window(server_log, start_offset)
        if finish.search(window) or monotonic() >= deadline:
            return window
        sleep(0.25)


def analyze_request_log(
    log_window: str,
    *,
    request_id: str | None,
    contract: ExactLongContextContract,
) -> dict[str, Any]:
    failures: list[str] = []
    full_marker = f"#new-token: {contract.chunk_size},"
    tail_marker = f"#new-token: {contract.expected_tail_tokens},"
    full_count = log_window.count(full_marker)
    tail_count = log_window.count(tail_marker)
    if full_count != contract.expected_full_chunks:
        failures.append(
            f"expected {contract.expected_full_chunks} full chunks, found {full_count}"
        )
    if tail_count != 1:
        failures.append(
            f"expected one {contract.expected_tail_tokens}-token tail, found {tail_count}"
        )
    finish_seen = bool(
        request_id
        and re.search(rf"Req:\s*{re.escape(request_id)}\s+Finish!", log_window)
    )
    if not finish_seen:
        failures.append("request Finish! log line was not observed")

    matches: list[dict[str, Any]] = []
    total_matches = 0
    for line_number, line in enumerate(log_window.splitlines(), start=1):
        for name, pattern in HARD_FAILURE_PATTERNS:
            if pattern.search(line):
                total_matches += 1
                if len(matches) < MAX_RECORDED_LOG_MATCHES:
                    matches.append(
                        {
                            "pattern": name,
                            "line_number": line_number,
                            "line": line[:500],
                        }
                    )
    if total_matches:
        failures.append(f"request log contains {total_matches} critical match(es)")
    return {
        "chunk_log": {
            "chunk_size": contract.chunk_size,
            "expected_full_count": contract.expected_full_chunks,
            "actual_full_count": full_count,
            "expected_tail_tokens": contract.expected_tail_tokens,
            "actual_tail_count": tail_count,
            "finish_seen": finish_seen,
        },
        "critical_log_check": {
            "passed": total_matches == 0,
            "total_matches": total_matches,
            "matches": matches,
        },
        "failures": failures,
    }


def run(
    config: ExactLongContextConfig,
    *,
    contract: ExactLongContextContract = RELEASE_CONTRACT,
    session: requests.Session | None = None,
    monitor_factory: Callable[
        [Path, tuple[int, ...]], MemoryMonitor
    ] = start_memory_monitor,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    response_path = config.output_dir / "response.json"
    health_path = config.output_dir / "post_health.json"
    memory_path = config.output_dir / "memory.csv"
    request_log_path = config.output_dir / "request_server_window.log"
    failures: list[str] = []
    start_offset = (
        config.server_log.stat().st_size if config.server_log.is_file() else 0
    )
    if not config.server_log.is_file():
        failures.append(f"server log is missing before request: {config.server_log}")

    owned_session = session is None
    http = requests.Session() if session is None else session
    monitor: MemoryMonitor | None = None
    status_code: int | None = None
    body: Any = None
    elapsed_seconds: float | None = None
    try:
        monitor = monitor_factory(memory_path, config.gpu_ids)
        payload = {
            "model": config.model,
            "input_ids": [1] * contract.prompt_tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": contract.output_tokens,
            },
        }
        started = monotonic()
        try:
            response = http.post(
                f"{config.base_url.rstrip('/')}/generate",
                json=payload,
                timeout=config.request_timeout_seconds,
            )
            elapsed_seconds = monotonic() - started
            status_code = response.status_code
            try:
                body = response.json()
            except ValueError:
                failures.append("generate response is not valid JSON")
        except requests.RequestException as exc:
            elapsed_seconds = monotonic() - started
            failures.append(f"generate request failed: {type(exc).__name__}")

        response_artifact = {
            "status_code": status_code,
            "elapsed_seconds": elapsed_seconds,
            "body": body,
        }
        _write_json_atomic(response_path, response_artifact)
        response_check = validate_response(status_code, body, contract)
        failures.extend(response_check.pop("failures"))
        request_id = response_check["request_id"]
        if request_id:
            log_window = wait_for_request_finish(
                config.server_log,
                start_offset=start_offset,
                request_id=request_id,
                monotonic=monotonic,
                sleep=sleep,
            )
        else:
            log_window = _read_log_window(config.server_log, start_offset)
        request_log_path.write_text(log_window)
        log_check = analyze_request_log(
            log_window,
            request_id=request_id,
            contract=contract,
        )
        failures.extend(log_check.pop("failures"))
        health = probe_health(config.base_url, config.control_url, session=http)
        failures.extend(health.pop("failures"))
        _write_json_atomic(health_path, health)
    except Exception as exc:  # Preserve artifacts for infrastructure failures.
        failures.append(f"workload execution failed: {type(exc).__name__}: {exc}")
        response_check = validate_response(status_code, body, contract)
        log_check = analyze_request_log("", request_id=None, contract=contract)
        health = {}
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception as exc:
                failures.append(
                    f"memory sampler stop failed: {type(exc).__name__}: {exc}"
                )
        if owned_session:
            http.close()

    memory = parse_gpu_memory(memory_path, config.gpu_ids)
    failures.extend(memory.pop("failures"))
    within_limit = True
    for gpu_id, peak in memory["peak_mib"].items():
        if peak is not None and peak > config.memory_limit_mib:
            within_limit = False
            failures.append(
                f"GPU {gpu_id} peak {peak} MiB exceeds {config.memory_limit_mib} MiB"
            )
    memory["limit_mib"] = config.memory_limit_mib
    memory["within_limit"] = within_limit

    request_result = {
        **response_check,
        "elapsed_seconds": elapsed_seconds,
    }
    result = {
        "schema_version": 1,
        "workload": "minimax_m3_exact_longctx",
        "ok": not failures,
        "request": request_result,
        **log_check,
        "gpu_memory": memory,
        "health": health,
        "artifacts": {
            "response": str(response_path),
            "post_health": str(health_path),
            "request_server_window": str(request_log_path),
            "memory": str(memory_path),
        },
        "failures": failures[:MAX_RECORDED_FAILURES],
        "failure_count": len(failures),
    }
    _write_json_atomic(config.output_dir / "validation.json", result)
    print(
        "Exact 1M validation: "
        f"{'PASS' if result['ok'] else 'FAIL'} status={status_code} "
        f"elapsed_s={elapsed_seconds}",
        flush=True,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids, required=True)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600)
    parser.add_argument("--memory-limit-mib", type=int, default=140000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        not math.isfinite(args.request_timeout_seconds)
        or args.request_timeout_seconds <= 0
        or args.memory_limit_mib <= 0
    ):
        raise SystemExit("request timeout and memory limit must be positive")
    config = ExactLongContextConfig(
        model=args.model,
        base_url=args.base_url,
        control_url=args.control_url,
        server_log=args.server_log,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        request_timeout_seconds=args.request_timeout_seconds,
        memory_limit_mib=args.memory_limit_mib,
    )
    return 0 if run(config)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
