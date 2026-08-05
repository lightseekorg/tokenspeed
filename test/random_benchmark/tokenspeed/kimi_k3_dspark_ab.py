#!/usr/bin/env python3
"""Paired Kimi-K3 no-spec/DSpark serving benchmark.

The harness launches one arm at a time, drives both TokenSpeed and SGLang
through the same OpenAI-compatible client, and writes raw JSON plus a compact
paired summary.  It intentionally uses a non-blocking process lock; callers on
shared machines must still acquire their site's GPU lease before invoking it.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOCK_PATH = Path("/tmp/k3-grid.lock")
DEFAULT_TARGETS = {
    "tokenspeed": "/sgl-workspace/models/Kimi-K3-flat",
    "sglang": "moonshotai/Kimi-K3",
}
DEFAULT_DRAFTS = {
    "tokenspeed": (
        "/sgl-workspace/models/hub/models--Inferact--Kimi-K3-DSpark/"
        "snapshots/cf6b8244620e7ea4b0651d214f28e89eac75bed6"
    ),
    "sglang": "RadixArk/Kimi-K3-DSpark",
}
ACCEPT_LENGTH_RE = re.compile(
    r"\b(?:avg_accept_len|accept len):\s*([0-9]+(?:\.[0-9]+)?)"
)
ACCEPT_RATE_RE = re.compile(r"\baccept(?:_| )rate:\s*([0-9]+(?:\.[0-9]+)?)")
REQUEST_ACCEPT_RE = re.compile(
    r"Req:\s+(\S+)\s+Finish!\s+Accept_num_tokens_avg:\s+" r"([0-9]+(?:\.[0-9]+)?)"
)
SUMMARY_FIELDS = (
    "engine",
    "group",
    "point",
    "isl",
    "osl",
    "concurrency",
    "repeats_no_spec",
    "repeats_dspark",
    "no_spec_memory_fraction",
    "dspark_memory_fraction",
    "no_spec_graph_max",
    "dspark_graph_max",
    "no_spec_tpot_ms",
    "dspark_tpot_ms",
    "tpot_speedup",
    "no_spec_tpot_p99_ms",
    "dspark_tpot_p99_ms",
    "dspark_tpot_tail_ratio",
    "spec_iteration_ms",
    "break_even_accept",
    "accept_margin",
    "no_spec_output_tps",
    "dspark_output_tps",
    "output_speedup",
    "no_spec_total_tps",
    "dspark_total_tps",
    "total_speedup",
    "no_spec_ttft_p99_ms",
    "dspark_ttft_p99_ms",
    "ttft_p99_ratio",
    "no_spec_e2e_p99_ms",
    "dspark_e2e_p99_ms",
    "e2e_p99_speedup",
    "dspark_accept_length",
    "dspark_accept_rate",
    "status",
)
REFERENCE_METRICS = (
    "median_tpot_ms",
    "output_throughput",
    "total_token_throughput",
)
REQUEST_TAIL_FIELDS = (
    "engine",
    "arm",
    "group",
    "point",
    "repeat",
    "request_id",
    "latency_ms",
    "ttft_ms",
    "mean_itl_ms",
    "output_tokens",
    "accept_length",
    "error",
)


@dataclass(frozen=True)
class Point:
    group: str
    name: str
    isl: int
    osl: int
    concurrency: int
    prompts: int
    warmups: int


def _point(
    group: str,
    name: str,
    isl: int,
    osl: int,
    concurrency: int,
    *,
    prompts: int | None = None,
    warmups: int = 2,
) -> Point:
    return Point(
        group=group,
        name=name,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        prompts=prompts if prompts is not None else max(8, concurrency * 2),
        warmups=warmups,
    )


FULL_MATRIX = (
    *(
        _point(
            "concurrency",
            f"1k-1k-c{conc}",
            1024,
            1024,
            conc,
            prompts=max(8, conc * 2 if conc < 16 else conc),
        )
        for conc in (1, 4, 8, 16, 32, 48)
    ),
    *(
        _point(
            "throughput",
            f"8k-1k-c{conc}",
            8192,
            1024,
            conc,
            prompts=max(8, conc * 2 if conc < 16 else conc),
            warmups=4,
        )
        for conc in (1, 8, 16, 48)
    ),
    *(
        _point(
            "throughput-core",
            f"8k-1k-core-c{conc}",
            8192,
            1024,
            conc,
            prompts=max(2, conc),
            warmups=1,
        )
        for conc in (1, 8, 16)
    ),
    _point("input", "8k-512-c1", 8192, 512, 1, prompts=2, warmups=1),
    _point("input", "32k-512-c1", 32768, 512, 1, prompts=2, warmups=1),
    _point("input", "128k-512-c1", 131072, 512, 1, prompts=1, warmups=0),
    _point("output", "1k-128-c1", 1024, 128, 1, prompts=2, warmups=1),
    _point("output", "1k-512-c1", 1024, 512, 1, prompts=2, warmups=1),
    _point("output", "1k-4k-c1", 1024, 4096, 1, prompts=1, warmups=0),
)
CI_MATRIX = (
    _point("ci", "4k-1k-c1", 4096, 1024, 1),
    _point("ci", "4k-1k-c4", 4096, 1024, 4),
    _point("ci", "4k-1k-c8", 4096, 1024, 8),
    _point("ci", "4k-1k-c16", 4096, 1024, 16),
)
SMOKE_MATRIX = (_point("smoke", "1k-128-c1", 1024, 128, 1, prompts=2),)


class GpuLock:
    """Non-blocking lock compatible with the workspace's flock harnesses."""

    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(f"another run holds {self.path}") from exc
        os.ftruncate(self.fd, 0)
        os.write(self.fd, f"{os.getpid()}\n".encode())
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def selected_points(
    profile: str, groups: set[str], names: set[str] | None = None
) -> tuple[Point, ...]:
    matrix = {
        "full": FULL_MATRIX,
        "ci": CI_MATRIX,
        "smoke": SMOKE_MATRIX,
    }[profile]
    if groups:
        matrix = tuple(point for point in matrix if point.group in groups)
    if names:
        matrix = tuple(point for point in matrix if point.name in names)
    return matrix


def _capture_sizes(max_concurrency: int) -> list[int]:
    return [size for size in (1, 2, 4, 8, 16, 32, 48) if size <= max_concurrency]


def build_server_command(
    args: argparse.Namespace, arm: str
) -> tuple[list[str], dict[str, str], str]:
    target = args.target or DEFAULT_TARGETS[args.engine]
    draft = args.draft or DEFAULT_DRAFTS[args.engine]
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/sgl-workspace/models")
    extra = shlex.split(args.server_extra_args)

    if args.engine == "tokenspeed":
        model_name = args.served_model_name
        command = [
            args.tokenspeed_command,
            "serve",
            "--model",
            target,
            "--served-model-name",
            model_name,
            "--tp",
            "8",
            "--ep-size",
            "8",
            "--attention-backend",
            "mla",
            "--moe-backend",
            "auto",
            "--kv-cache-dtype",
            "fp8",
            "--mm-encoder-tp-mode",
            "data",
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(args.max_concurrency),
            "--gpu-memory-utilization",
            str(args.memory_fraction),
            "--trust-remote-code",
            "--sampling-backend",
            "greedy",
            "--disable-kvstore",
            "--kvstore-ratio",
            "0.0",
            "--no-enable-prefix-caching",
            "--enable-cache-report",
            "--disable-weight-loader-prefetch-checkpoints",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        if args.eager:
            command.append("--enforce-eager")
        else:
            graph_max = args.graph_max_concurrency or args.max_concurrency
            sizes = _capture_sizes(graph_max)
            command += [
                "--max-cudagraph-capture-size",
                str(max(sizes)),
                "--cudagraph-capture-sizes",
                *(str(size) for size in sizes),
                "--disable-prefill-graph",
            ]
        if args.enable_mixed_batch:
            command.append("--enable-mixed-batch")
        if arm == "dspark":
            command += [
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                draft,
                "--speculative-num-draft-tokens",
                str(args.speculative_width),
                "--drafter-attention-backend",
                "mla",
            ]
    else:
        model_name = target
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("SGLANG_USE_AITER", "1")
        env.setdefault("SGLANG_AITER_K3_OPT", "1")
        env.setdefault("AITER_FLYDSL_FORCE", "1")
        env.setdefault("AITER_SITUV2_A8W4", "1")
        command = [
            args.sglang_command,
            "serve",
            "--model-path",
            target,
            "--trust-remote-code",
            "--tp",
            "8",
            "--attention-backend",
            "triton",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            str(args.memory_fraction),
            "--cuda-graph-max-bs-decode",
            str(args.graph_max_concurrency or args.max_concurrency),
            "--max-running-requests",
            str(args.max_concurrency),
            "--context-length",
            str(args.max_model_len),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--reasoning-parser",
            "kimi_k3",
            "--tool-call-parser",
            "kimi_k3",
            "--disable-radix-cache",
        ]
        if arm == "dspark":
            command += [
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                draft,
            ]

    return command + extra, env, model_name


def wait_for_port_free(port: int, timeout: int = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                time.sleep(1)
                continue
            return
    raise TimeoutError(f"port {port} is still in use after {timeout}s")


def _url_ok(url: str, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def _last_lines(path: Path, count: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-count:])


def wait_for_ready(
    process: subprocess.Popen,
    url: str,
    log_path: Path,
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with {process.returncode}\n{_last_lines(log_path)}"
            )
        if _url_ok(url):
            return
        time.sleep(5)
    raise TimeoutError(
        f"server was not ready after {timeout}s\n{_last_lines(log_path)}"
    )


def stop_server(process: subprocess.Popen | None, timeout: int = 30) -> None:
    if process is None:
        return
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_benchmark_monitored(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    server_process: subprocess.Popen,
    ready_url: str,
    timeout: int,
) -> int:
    """Run a client while aborting promptly if the serving workers die."""
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    deadline = time.monotonic() + timeout
    readiness_failures = 0
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        if server_process.poll() is not None:
            stop_server(process, timeout=5)
            return 125
        if _url_ok(ready_url, timeout=2):
            readiness_failures = 0
        else:
            readiness_failures += 1
            # A single-rank long prefill can make a health handler unresponsive
            # for tens of seconds without killing the serving workers. Twelve
            # consecutive failures still detects a dead gateway within a minute.
            if readiness_failures >= 12:
                stop_server(process, timeout=5)
                return 125
        time.sleep(5)
    stop_server(process, timeout=5)
    return 124


def flush_cache(port: int) -> None:
    url = f"http://127.0.0.1:{port}/flush_cache"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"flush cache returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            request = urllib.request.Request(url, data=b"{}", method="POST")
            with urllib.request.urlopen(request, timeout=60):
                return
        raise


def parse_acceptance(text: str) -> dict[str, Any]:
    lengths = [float(value) for value in ACCEPT_LENGTH_RE.findall(text)]
    rates = [float(value) for value in ACCEPT_RATE_RE.findall(text)]
    request_acceptance = [
        {"request_id": request_id, "accept_length": float(value)}
        for request_id, value in REQUEST_ACCEPT_RE.findall(text)
    ]
    result: dict[str, Any] = {}
    if lengths:
        result["accept_length"] = statistics.median(lengths)
        result["accept_length_samples"] = len(lengths)
    if rates:
        result["accept_rate"] = statistics.median(rates)
        result["accept_rate_samples"] = len(rates)
    if request_acceptance:
        result["request_acceptance"] = request_acceptance
    return result


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _snapshot_id(model: str) -> str:
    path = Path(model)
    if path.exists():
        resolved = path.resolve()
        if resolved.parent.name == "snapshots":
            return resolved.name
        return str(resolved)
    return model


def benchmark_command(
    args: argparse.Namespace,
    point: Point,
    model_name: str,
    output_path: Path,
) -> list[str]:
    command = [
        args.bench_command,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        model_name,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(point.isl),
        "--random-output-len",
        str(point.osl),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(point.prompts),
        "--max-concurrency",
        str(point.concurrency),
        "--num-warmups",
        str(point.warmups),
        "--request-rate",
        "inf",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,99",
        "--extra-request-body",
        '{"temperature":0}',
        "--disable-tqdm",
        "--request-id-prefix",
        f"{output_path.stem}-",
        "--output-file",
        str(output_path),
    ]
    if args.save_detailed:
        command.append("--save-detailed")
    return command


def _read_log_segment(path: Path, offset: int) -> str:
    with path.open(errors="replace") as stream:
        stream.seek(offset)
        return stream.read()


def _is_complete_result(path: Path) -> bool:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(result.get("ab_metadata"), dict)
        and int(result.get("failed", 0)) == 0
    )


def _handle_benchmark_failure(
    *,
    returncode: int,
    point_name: str,
    bench_log: Path,
    continue_on_error: bool,
) -> None:
    """Abort on infrastructure loss even when point failures are tolerated."""
    if returncode in (124, 125):
        reason = "timed out" if returncode == 124 else "lost its serving process"
        raise RuntimeError(
            f"benchmark infrastructure {reason} at {point_name}; see {bench_log}"
        )
    if not continue_on_error:
        raise RuntimeError(f"benchmark failed for {point_name}; see {bench_log}")


def run_arm(
    args: argparse.Namespace,
    arm: str,
    points: Iterable[Point],
    run_dir: Path,
) -> None:
    arm_dir = run_dir / args.engine / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    server_log = arm_dir / "server.log"
    command, env, model_name = build_server_command(args, arm)
    metadata = {
        "engine": args.engine,
        "arm": arm,
        "target": args.target or DEFAULT_TARGETS[args.engine],
        "draft": args.draft or DEFAULT_DRAFTS[args.engine],
        "target_snapshot": _snapshot_id(args.target or DEFAULT_TARGETS[args.engine]),
        "draft_snapshot": _snapshot_id(args.draft or DEFAULT_DRAFTS[args.engine]),
        "speculative_width": args.speculative_width if arm == "dspark" else None,
        "enable_mixed_batch": args.enable_mixed_batch,
        "server_command": command,
        "tokenspeed_sha": _git_sha(args.repo_root),
        "sglang_sha": _git_sha(Path("/sgl-workspace/sglang")),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (arm_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.dry_run:
        print(shlex.join(command))
        for point in points:
            for repeat in range(1, args.repeats + 1):
                output_path = arm_dir / f"{point.group}-{point.name}-r{repeat}.json"
                print(
                    shlex.join(benchmark_command(args, point, model_name, output_path))
                )
        return

    wait_for_port_free(args.port)
    process: subprocess.Popen | None = None
    try:
        with server_log.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=args.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        ready_url = (
            f"http://127.0.0.1:{args.port}"
            f"{'/readiness' if args.engine == 'tokenspeed' else '/health'}"
        )
        wait_for_ready(process, ready_url, server_log, args.startup_timeout)

        for point in points:
            for repeat in range(1, args.repeats + 1):
                output_path = arm_dir / f"{point.group}-{point.name}-r{repeat}.json"
                if (
                    output_path.exists()
                    and args.resume
                    and _is_complete_result(output_path)
                ):
                    print(f"[resume] keeping {output_path}")
                    continue
                # SGLang runs with radix cache disabled in this protocol, so
                # there is no reusable prefix state to flush. Its endpoint can
                # also return 400 while post-startup warmup is still draining.
                if args.engine == "tokenspeed":
                    flush_cache(args.port)
                log_offset = server_log.stat().st_size
                bench_log = output_path.with_suffix(".log")
                bench = benchmark_command(args, point, model_name, output_path)
                print(
                    f"[{args.engine}/{arm}] {point.name} repeat "
                    f"{repeat}/{args.repeats}"
                )
                returncode = run_benchmark_monitored(
                    bench,
                    cwd=args.repo_root,
                    env=env,
                    log_path=bench_log,
                    server_process=process,
                    ready_url=ready_url,
                    timeout=args.point_timeout,
                )
                segment = _read_log_segment(server_log, log_offset)
                acceptance = parse_acceptance(segment)
                if returncode != 0 or not output_path.exists():
                    failure = {
                        "ab_metadata": {
                            **asdict(point),
                            "engine": args.engine,
                            "arm": arm,
                            "repeat": repeat,
                            "returncode": returncode,
                            "speculative_width": (
                                args.speculative_width if arm == "dspark" else None
                            ),
                            "memory_fraction": args.memory_fraction,
                            "server_max_concurrency": args.max_concurrency,
                            "graph_max_concurrency": (
                                args.graph_max_concurrency or args.max_concurrency
                            ),
                            "enable_mixed_batch": args.enable_mixed_batch,
                            **acceptance,
                        },
                        "failed": 1,
                        "error": _last_lines(bench_log),
                    }
                    output_path.write_text(json.dumps(failure, indent=2))
                    _handle_benchmark_failure(
                        returncode=returncode,
                        point_name=point.name,
                        bench_log=bench_log,
                        continue_on_error=args.continue_on_error,
                    )
                    continue
                result = json.loads(output_path.read_text())
                result["ab_metadata"] = {
                    **asdict(point),
                    "engine": args.engine,
                    "arm": arm,
                    "repeat": repeat,
                    "returncode": returncode,
                    "speculative_width": (
                        args.speculative_width if arm == "dspark" else None
                    ),
                    "memory_fraction": args.memory_fraction,
                    "server_max_concurrency": args.max_concurrency,
                    "graph_max_concurrency": (
                        args.graph_max_concurrency or args.max_concurrency
                    ),
                    "enable_mixed_batch": args.enable_mixed_batch,
                    **acceptance,
                }
                output_path.write_text(json.dumps(result, indent=2))
    finally:
        stop_server(process)
        wait_for_port_free(args.port)


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and float(row.get("failed", 0)) == 0
    ]
    return statistics.median(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def load_raw_results(run_dir: Path, engine: str | None = None) -> list[dict[str, Any]]:
    rows = []
    roots = [run_dir / engine] if engine else [path for path in run_dir.iterdir()]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.json")):
            if path.name in {"metadata.json", "summary.json", "reference.json"}:
                continue
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            metadata = row.get("ab_metadata")
            if not isinstance(metadata, dict):
                continue
            row["_path"] = str(path)
            rows.append(row)
    return rows


def aggregate_results(
    raw_rows: list[dict[str, Any]], num_gpus: int = 8
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    points: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        meta = row["ab_metadata"]
        key = (meta["engine"], meta["group"], meta["name"], meta["arm"])
        grouped.setdefault(key, []).append(row)
        points[(meta["engine"], meta["group"], meta["name"])] = meta

    arm_summaries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        metadata_rows = [row["ab_metadata"] for row in rows]
        arm_summaries[key] = {
            "repeats": len(rows),
            "successful_repeats": sum(int(row.get("failed", 0)) == 0 for row in rows),
            "median_tpot_ms": _median(rows, "median_tpot_ms"),
            "p99_tpot_ms": _median(rows, "p99_tpot_ms"),
            "p99_ttft_ms": _median(rows, "p99_ttft_ms"),
            "p99_e2el_ms": _median(rows, "p99_e2el_ms"),
            "output_throughput": _median(rows, "output_throughput"),
            "output_tps_per_gpu": (
                _median(rows, "output_throughput") / num_gpus
                if _median(rows, "output_throughput") is not None
                else None
            ),
            "total_token_throughput": _median(rows, "total_token_throughput"),
            "accept_length": _median(metadata_rows, "accept_length"),
            "accept_rate": _median(metadata_rows, "accept_rate"),
            "memory_fraction": _median(metadata_rows, "memory_fraction"),
            "graph_max_concurrency": _median(metadata_rows, "graph_max_concurrency"),
        }

    paired_rows = []
    for point_key, meta in sorted(points.items()):
        engine, group, name = point_key
        no_spec = arm_summaries.get((*point_key, "no-spec"))
        dspark = arm_summaries.get((*point_key, "dspark"))
        status = "paired"
        if no_spec is None or dspark is None:
            status = "partial"
        elif (
            no_spec["successful_repeats"] != no_spec["repeats"]
            or dspark["successful_repeats"] != dspark["repeats"]
        ):
            status = "failed"
        no_spec_tpot = no_spec["median_tpot_ms"] if no_spec else None
        dspark_tpot = dspark["median_tpot_ms"] if dspark else None
        accept_length = dspark["accept_length"] if dspark else None
        spec_iteration_ms = (
            dspark_tpot * accept_length
            if dspark_tpot is not None and accept_length is not None
            else None
        )
        break_even_accept = _ratio(spec_iteration_ms, no_spec_tpot)
        paired_rows.append(
            {
                "engine": engine,
                "group": group,
                "point": name,
                "isl": meta["isl"],
                "osl": meta["osl"],
                "concurrency": meta["concurrency"],
                "repeats_no_spec": no_spec["successful_repeats"] if no_spec else 0,
                "repeats_dspark": dspark["successful_repeats"] if dspark else 0,
                "no_spec_memory_fraction": (
                    no_spec["memory_fraction"] if no_spec else None
                ),
                "dspark_memory_fraction": (
                    dspark["memory_fraction"] if dspark else None
                ),
                "no_spec_graph_max": (
                    no_spec["graph_max_concurrency"] if no_spec else None
                ),
                "dspark_graph_max": (
                    dspark["graph_max_concurrency"] if dspark else None
                ),
                "no_spec_tpot_ms": no_spec_tpot,
                "dspark_tpot_ms": dspark_tpot,
                "tpot_speedup": _ratio(no_spec_tpot, dspark_tpot),
                "no_spec_tpot_p99_ms": no_spec["p99_tpot_ms"] if no_spec else None,
                "dspark_tpot_p99_ms": dspark["p99_tpot_ms"] if dspark else None,
                "dspark_tpot_tail_ratio": _ratio(
                    dspark["p99_tpot_ms"] if dspark else None,
                    dspark_tpot,
                ),
                "spec_iteration_ms": spec_iteration_ms,
                "break_even_accept": break_even_accept,
                "accept_margin": (
                    accept_length - break_even_accept
                    if accept_length is not None and break_even_accept is not None
                    else None
                ),
                "no_spec_output_tps": (
                    no_spec["output_throughput"] if no_spec else None
                ),
                "dspark_output_tps": (dspark["output_throughput"] if dspark else None),
                "output_speedup": _ratio(
                    dspark["output_throughput"] if dspark else None,
                    no_spec["output_throughput"] if no_spec else None,
                ),
                "no_spec_total_tps": (
                    no_spec["total_token_throughput"] if no_spec else None
                ),
                "dspark_total_tps": (
                    dspark["total_token_throughput"] if dspark else None
                ),
                "total_speedup": _ratio(
                    dspark["total_token_throughput"] if dspark else None,
                    no_spec["total_token_throughput"] if no_spec else None,
                ),
                "no_spec_ttft_p99_ms": no_spec["p99_ttft_ms"] if no_spec else None,
                "dspark_ttft_p99_ms": dspark["p99_ttft_ms"] if dspark else None,
                "ttft_p99_ratio": _ratio(
                    dspark["p99_ttft_ms"] if dspark else None,
                    no_spec["p99_ttft_ms"] if no_spec else None,
                ),
                "no_spec_e2e_p99_ms": no_spec["p99_e2el_ms"] if no_spec else None,
                "dspark_e2e_p99_ms": dspark["p99_e2el_ms"] if dspark else None,
                "e2e_p99_speedup": _ratio(
                    no_spec["p99_e2el_ms"] if no_spec else None,
                    dspark["p99_e2el_ms"] if dspark else None,
                ),
                "dspark_accept_length": accept_length,
                "dspark_accept_rate": dspark["accept_rate"] if dspark else None,
                "status": status,
            }
        )

    reference = {"|".join(key): value for key, value in sorted(arm_summaries.items())}
    return paired_rows, reference


def check_results(
    paired_rows: list[dict[str, Any]],
    reference: dict[str, Any],
    *,
    min_accept_length: float,
    baseline: dict[str, Any] | None,
    reference_threshold: float,
) -> dict[str, Any]:
    failures = []
    for row in paired_rows:
        if row["status"] == "failed":
            failures.append(f"{row['engine']}/{row['point']}: failed repeats")
        elif row["status"] == "partial":
            failures.append(f"{row['engine']}/{row['point']}: missing A/B arm")
        accept_length = row.get("dspark_accept_length")
        if accept_length is not None and accept_length < min_accept_length:
            failures.append(
                f"{row['engine']}/{row['point']}: accept length "
                f"{accept_length:.3f} < {min_accept_length:.3f}"
            )

    if baseline:
        for key, current in reference.items():
            previous = baseline.get(key)
            if not isinstance(previous, dict):
                continue
            arm = key.rsplit("|", 1)[-1]
            for metric in REFERENCE_METRICS:
                actual = current.get(metric)
                expected = previous.get(metric)
                if actual is None or expected in (None, 0):
                    continue
                ratio = (
                    expected / actual
                    if metric == "median_tpot_ms"
                    else actual / expected
                )
                if ratio < reference_threshold:
                    failures.append(
                        f"{key}/{metric}: {ratio:.3f}x reference "
                        f"(minimum {reference_threshold:.3f}x)"
                    )
            if arm == "dspark":
                actual = current.get("accept_length")
                expected = previous.get("accept_length")
                if actual is not None and expected not in (None, 0):
                    ratio = actual / expected
                    if ratio < reference_threshold:
                        failures.append(
                            f"{key}/accept_length: {ratio:.3f}x reference "
                            f"(minimum {reference_threshold:.3f}x)"
                        )
    return {"passed": not failures, "failures": failures}


def request_tail_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in raw_rows:
        request_ids = result.get("request_ids")
        if not isinstance(request_ids, list):
            continue
        metadata = result["ab_metadata"]
        acceptance = {
            item["request_id"]: item["accept_length"]
            for item in metadata.get("request_acceptance", [])
        }
        latencies = result.get("latencies", [])
        ttfts = result.get("ttfts", [])
        itls = result.get("itls", [])
        output_lens = result.get("output_lens", [])
        errors = result.get("errors", [])
        for index, request_id in enumerate(request_ids):
            request_itls = itls[index] if index < len(itls) else []
            accept_length = acceptance.get(request_id)
            if accept_length is None and request_id:
                accept_length = next(
                    (
                        value
                        for logged_id, value in acceptance.items()
                        if logged_id.startswith(f"{request_id}-")
                    ),
                    None,
                )
            rows.append(
                {
                    "engine": metadata["engine"],
                    "arm": metadata["arm"],
                    "group": metadata["group"],
                    "point": metadata["name"],
                    "repeat": metadata["repeat"],
                    "request_id": request_id,
                    "latency_ms": (
                        float(latencies[index]) * 1000
                        if index < len(latencies)
                        else None
                    ),
                    "ttft_ms": (
                        float(ttfts[index]) * 1000 if index < len(ttfts) else None
                    ),
                    "mean_itl_ms": (
                        statistics.mean(request_itls) * 1000 if request_itls else None
                    ),
                    "output_tokens": (
                        output_lens[index] if index < len(output_lens) else None
                    ),
                    "accept_length": accept_length,
                    "error": errors[index] if index < len(errors) else None,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["engine"],
            row["arm"],
            row["group"],
            row["point"],
            row["repeat"],
            row["request_id"] or "",
        ),
    )


def write_request_tail(run_dir: Path, raw_rows: list[dict[str, Any]]) -> None:
    rows = request_tail_rows(raw_rows)
    if not rows:
        return
    with (run_dir / "request_tail.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUEST_TAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    run_dir: Path,
    paired_rows: list[dict[str, Any]],
    reference: dict[str, Any],
    checks: dict[str, Any],
) -> None:
    with (run_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(paired_rows)
    (run_dir / "summary.json").write_text(
        json.dumps({"rows": paired_rows, "checks": checks}, indent=2)
    )
    (run_dir / "reference.json").write_text(json.dumps(reference, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("tokenspeed", "sglang"))
    parser.add_argument("--arm", choices=("no-spec", "dspark", "both"), default="both")
    parser.add_argument("--profile", choices=("full", "ci", "smoke"), default="full")
    parser.add_argument(
        "--group",
        action="append",
        choices=(
            "concurrency",
            "throughput",
            "throughput-core",
            "input",
            "output",
            "ci",
            "smoke",
        ),
        default=[],
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Run only the named matrix point; repeat to select multiple points.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analyze-only", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-threshold", type=float, default=0.9)
    parser.add_argument("--min-accept-length", type=float, default=1.1)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--target")
    parser.add_argument("--draft")
    parser.add_argument("--tokenizer", default="moonshotai/Kimi-K3")
    parser.add_argument("--served-model-name", default="kimi-k3")
    parser.add_argument("--port", type=int, default=21000)
    parser.add_argument("--max-concurrency", type=int, default=48)
    parser.add_argument(
        "--graph-max-concurrency",
        type=int,
        help="Maximum captured batch size; max scheduling concurrency is unchanged.",
    )
    parser.add_argument("--max-model-len", type=int, default=131584)
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--speculative-width", type=int, default=8)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--point-timeout", type=int, default=3600)
    parser.add_argument("--server-extra-args", default="")
    parser.add_argument("--tokenspeed-command", default="ts")
    parser.add_argument("--sglang-command", default="sglang")
    parser.add_argument("--bench-command", default="tokenspeed")
    parser.add_argument(
        "--repo-root", type=Path, default=Path("/sgl-workspace/tokenspeed")
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--enable-mixed-batch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.analyze_only is None and args.engine is None:
        parser.error("--engine is required unless --analyze-only is used")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.graph_max_concurrency is not None and args.graph_max_concurrency <= 0:
        parser.error("--graph-max-concurrency must be positive")
    if args.output_dir is None and args.analyze_only is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("ab_results") / f"kimi-k3-dspark-{stamp}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = (args.analyze_only or args.output_dir).resolve()
    run_dir = (
        args.output_dir.resolve()
        if args.analyze_only is not None and args.output_dir is not None
        else source_dir
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only is None:
        points = selected_points(args.profile, set(args.group), set(args.point))
        if not points:
            raise SystemExit("selected matrix is empty")
        arms = ("no-spec", "dspark") if args.arm == "both" else (args.arm,)
        if args.dry_run:
            for arm in arms:
                run_arm(args, arm, points, run_dir)
        else:
            with GpuLock():
                for arm in arms:
                    run_arm(args, arm, points, run_dir)

    raw_rows = load_raw_results(source_dir, args.engine)
    write_request_tail(run_dir, raw_rows)
    paired_rows, reference = aggregate_results(raw_rows, args.num_gpus)
    baseline = json.loads(args.reference.read_text()) if args.reference else None
    checks = check_results(
        paired_rows,
        reference,
        min_accept_length=args.min_accept_length,
        baseline=baseline,
        reference_threshold=args.reference_threshold,
    )
    write_summary(run_dir, paired_rows, reference, checks)
    print(f"summary: {run_dir / 'summary.csv'}")
    if not checks["passed"]:
        for failure in checks["failures"]:
            print(f"REGRESSION: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
