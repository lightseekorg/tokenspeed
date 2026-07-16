#!/usr/bin/env python3
"""Run the paired MiniMax-M3 encoder Graph A/B on one clean TP4 runner.

The benchmark collector intentionally owns no server lifecycle.  This CI-only
orchestrator supplies that outer lifecycle with the shared root-SIGTERM
validator, runs two launch pairs in E1-G1-G2-E2 order on the same GPUs and
installation, then executes the artifact comparator so performance gates affect
the CI exit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from server_lifecycle import (
    probe_nvidia_gpus,
    start_managed_server,
    stop_managed_server,
)

REPO_ROOT = Path(__file__).parents[2]
BENCHMARK_PATH = (
    REPO_ROOT / "test/quality_benchmark/tokenspeed/minimax_m3_encoder_graph_ab.py"
)
BENCHMARK_MODULE_NAME = "minimax_m3_encoder_graph_ab_ci_benchmark"
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    BENCHMARK_MODULE_NAME, BENCHMARK_PATH
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
benchmark = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_MODULE_NAME] = benchmark
BENCHMARK_SPEC.loader.exec_module(benchmark)

DEFAULT_BASE_GPU_ID = 0
DEFAULT_GPU_ID_STEP = 1
HTTP_PORT = 8000
CONTROL_PORT = 8001
LAUNCH_SEQUENCE = (
    ("launch-1", "eager"),
    ("launch-1", "graph"),
    ("launch-2", "graph"),
    ("launch-2", "eager"),
)
MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
REVISION = "c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"
SERVED_MODEL = "minimax-m3"
FORBIDDEN_LOG_PATTERNS = (
    r"Prefill graph capture failed",
    r"Encoder CUDA graph capture failed",
    r"Scheduler hit an exception:",
    r"Buffer overflow when allocating memory",
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"Retract failed",
    r"NCCL.*(?:abort|unhandled)",
    r"CancelledError",
    r"Task was destroyed",
    r"Traceback \(most recent call last\)",
    r"Exception ignored in",
    r"unhandled exception during asyncio.run\(\) shutdown",
    r"resource_tracker:.*leaked",
)
HARDWARE_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "pci.bus_id",
    "compute_cap",
    "persistence_mode",
    "power.limit",
    "clocks.applications.graphics",
    "clocks.applications.memory",
    "clocks.max.sm",
    "clocks.max.memory",
    "compute_mode",
    "mig.mode.current",
)


class OrchestratorError(RuntimeError):
    """Raised when paired CI cannot produce trustworthy A/B evidence."""


def derive_gpu_indices(base_gpu_id: int, gpu_id_step: int) -> tuple[int, ...]:
    """Return the four physical GPU indices selected by CLI base/step."""

    if (
        not isinstance(base_gpu_id, int)
        or isinstance(base_gpu_id, bool)
        or base_gpu_id < 0
    ):
        raise OrchestratorError("--base-gpu-id must be a non-negative integer")
    if (
        not isinstance(gpu_id_step, int)
        or isinstance(gpu_id_step, bool)
        or gpu_id_step <= 0
    ):
        raise OrchestratorError("--gpu-id-step must be a positive integer")
    indices = tuple(
        base_gpu_id + rank * gpu_id_step for rank in range(benchmark.EXPECTED_TP_RANKS)
    )
    if len(indices) != benchmark.EXPECTED_TP_RANKS or len(set(indices)) != len(indices):
        raise OrchestratorError("base/step must select four distinct physical GPUs")
    return indices


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_server_argv(
    *,
    graph: bool,
    base_gpu_id: int = DEFAULT_BASE_GPU_ID,
    gpu_id_step: int = DEFAULT_GPU_ID_STEP,
) -> list[str]:
    """Return an arm command; graph adds exactly one flag to the eager argv."""
    derive_gpu_indices(base_gpu_id, gpu_id_step)
    argv = [
        "tokenspeed",
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--served-model-name",
        SERVED_MODEL,
        "--trust-remote-code",
        "--tensor-parallel-size",
        "4",
        "--base-gpu-id",
        str(base_gpu_id),
        "--gpu-id-step",
        str(gpu_id_step),
        "--max-model-len",
        "32768",
        "--max-total-tokens",
        "32768",
        "--max-num-seqs",
        "2",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "8192",
        "--block-size",
        "128",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--kv-cache-quant-method",
        "none",
        "--attention-backend",
        "mha",
        "--mm-attention-backend",
        "triton_attn",
        "--moe-backend",
        "triton",
        "--sampling-backend",
        "greedy",
        "--seed",
        "20260715",
        "--enable-log-mm-timing",
        "--enforce-eager",
        "--disable-prefill-graph",
        "--no-enable-prefix-caching",
        "--disable-kvstore",
        "--enable-output-logprobs",
        "--epd-pixel-shm",
        "--epd-ingest-offloop",
        "--unlink-mm-shm-after-read",
        "--multimodal-tensor-transport",
        "shm",
        "--multimodal-shm-min-bytes",
        "65536",
        "--multimodal-pixel-cache-mb",
        "0",
        "--multimodal-image-max-input-bytes",
        "268435456",
        "--multimodal-image-encoder-input-dtype",
        "bfloat16",
        "--host",
        "127.0.0.1",
        "--port",
        str(HTTP_PORT),
        "--control-port",
        str(CONTROL_PORT),
    ]
    if graph:
        seed_value_index = argv.index("20260715")
        argv.insert(seed_value_index + 1, "--enable-mm-encoder-cuda-graph")
    return argv


def _server_command(*, graph: bool, base_gpu_id: int, gpu_id_step: int) -> str:
    return shlex.join(
        build_server_argv(
            graph=graph,
            base_gpu_id=base_gpu_id,
            gpu_id_step=gpu_id_step,
        )
    )


def capture_hardware_identity(gpu_indices: Sequence[int]) -> dict[str, Any]:
    """Capture stable identity, driver, clock ceilings, and power limit."""
    if len(gpu_indices) != benchmark.EXPECTED_TP_RANKS or len(set(gpu_indices)) != len(
        gpu_indices
    ):
        raise OrchestratorError("hardware identity requires four distinct GPU indices")
    selected = tuple(gpu_indices)
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(HARDWARE_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OrchestratorError(f"cannot query GPU identity: {exc}") from exc
    if result.returncode != 0:
        raise OrchestratorError(
            f"GPU identity query failed ({result.returncode}): {result.stderr.strip()}"
        )
    rows: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != len(HARDWARE_QUERY_FIELDS):
            raise OrchestratorError(f"unexpected GPU identity row: {raw_line!r}")
        row = dict(zip(HARDWARE_QUERY_FIELDS, fields))
        try:
            index = int(row["index"])
        except ValueError as exc:
            raise OrchestratorError(f"invalid GPU index in {raw_line!r}") from exc
        if index in selected:
            row["index"] = index
            rows.append(row)
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != sorted(selected):
        raise OrchestratorError(
            f"hardware query did not return GPU indices {selected}: {rows}"
        )
    if len({row["uuid"] for row in rows}) != len(rows):
        raise OrchestratorError("selected GPUs do not have unique UUIDs")
    return {
        "schema_version": 1,
        "query_fields": list(HARDWARE_QUERY_FIELDS),
        "gpus": rows,
    }


def _require_idle_gpus(
    gpu_indices: Sequence[int], max_memory_mib: int = 16
) -> dict[str, Any]:
    state = probe_nvidia_gpus(gpu_indices)
    if not state.get("ok"):
        raise OrchestratorError(f"cannot query idle GPUs: {state.get('error')}")
    failures: list[str] = []
    for index in gpu_indices:
        details = state["gpus"].get(str(index), {})
        if details.get("compute_pids"):
            failures.append(f"GPU {index} compute PIDs={details['compute_pids']}")
        if details.get("memory_used_mib", max_memory_mib + 1) > max_memory_mib:
            failures.append(f"GPU {index} memory={details.get('memory_used_mib')} MiB")
    if failures:
        raise OrchestratorError("selected GPUs are not idle: " + "; ".join(failures))
    return state


def _url_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for_server_ready(
    process: subprocess.Popen[str], timeout_seconds: float
) -> None:
    """Require both gateway readiness and the later-started control sidecar."""
    deadline = time.monotonic() + timeout_seconds
    gateway_url = f"http://127.0.0.1:{HTTP_PORT}/readiness"
    control_url = f"http://127.0.0.1:{CONTROL_PORT}/get_server_info"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise OrchestratorError(
                f"server exited with code {returncode} before readiness"
            )
        if _url_ready(gateway_url) and _url_ready(control_url):
            return
        time.sleep(1.0)
    raise OrchestratorError(
        f"server did not expose gateway and control readiness in {timeout_seconds}s"
    )


def _scan_server_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in FORBIDDEN_LOG_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                matches.append(
                    {
                        "pattern": pattern,
                        "line_number": line_number,
                        "line": line[:1000],
                    }
                )
                break
    return {"passed": not matches, "match_count": len(matches), "matches": matches[:50]}


def _shutdown_config(
    arm_dir: Path, work_dir: Path, gpu_indices: Sequence[int]
) -> dict[str, Any]:
    try:
        relative_output = (arm_dir / "shutdown-validation.json").relative_to(work_dir)
    except ValueError as exc:
        raise OrchestratorError("arm output directory must be inside work dir") from exc
    return {
        "target": "root",
        "signal": "SIGTERM",
        "timeout_seconds": 120,
        "expected_exit_code": 0,
        "ports": [HTTP_PORT, CONTROL_PORT],
        "gpu_indices": list(gpu_indices),
        "max_memory_mib": 16,
        "output": str(relative_output),
    }


def _run_arm(
    *,
    arm: str,
    launch_id: str,
    work_dir: Path,
    arm_dir: Path,
    dog: Path,
    reference: Path,
    server_sha: str,
    provenance_files: tuple[tuple[str, Path], ...],
    request_timeout_seconds: float,
    base_gpu_id: int,
    gpu_id_step: int,
) -> dict[str, Any]:
    graph = arm == "graph"
    gpu_indices = derive_gpu_indices(base_gpu_id, gpu_id_step)
    # Validate the durable shutdown destination before a server is allowed to
    # start.  In particular, an arbitrary external --output-dir must not turn
    # a late artifact-path error into a leaked TP4 process tree.
    shutdown_config = _shutdown_config(arm_dir, work_dir, gpu_indices)
    arm_dir.mkdir(parents=True, exist_ok=True)
    log_path = arm_dir / "server.log"
    log_path.write_text("")
    _require_idle_gpus(gpu_indices)
    server = start_managed_server(
        _server_command(
            graph=graph,
            base_gpu_id=base_gpu_id,
            gpu_id_step=gpu_id_step,
        ),
        env=os.environ,
        cwd=work_dir,
        log_path=log_path,
        gpu_indices=gpu_indices,
    )
    assert server is not None
    collection: dict[str, Any] | None = None
    workload_error: BaseException | None = None
    lifecycle_failures: list[str] = []
    try:
        try:
            wait_for_server_ready(server.process, 3600)
            collection = benchmark.collect_arm(
                benchmark.CollectConfig(
                    arm=arm,
                    launch_id=launch_id,
                    base_url=f"http://127.0.0.1:{HTTP_PORT}",
                    server_info_base_url=f"http://127.0.0.1:{CONTROL_PORT}",
                    model=SERVED_MODEL,
                    dog=dog,
                    reference=reference,
                    server_log=log_path,
                    server_sha=server_sha,
                    output=arm_dir / "arm.json",
                    request_timeout_seconds=request_timeout_seconds,
                    provenance_files=provenance_files,
                )
            )
            if not collection["ok"]:
                raise OrchestratorError(f"{arm} collector gates failed")
        except BaseException as exc:
            workload_error = exc
    finally:
        try:
            shutdown = stop_managed_server(
                server,
                shutdown_config,
                cwd=work_dir,
            )
        except BaseException as exc:
            shutdown = None
            lifecycle_failures.append(f"shutdown raised {type(exc).__name__}: {exc}")
    log_check = _scan_server_log(log_path)
    _write_json_atomic(arm_dir / "server-log-validation.json", log_check)
    if shutdown is not None and not shutdown["passed"]:
        lifecycle_failures.append(
            "shutdown: " + "; ".join(shutdown.get("failures", []))
        )
    if not log_check["passed"]:
        lifecycle_failures.append(
            f"server log: {log_check['match_count']} forbidden match(es)"
        )
    if workload_error is not None or lifecycle_failures:
        details = []
        if workload_error is not None:
            details.append(
                f"workload: {type(workload_error).__name__}: {workload_error}"
            )
        details.extend(lifecycle_failures)
        raise OrchestratorError(f"{arm} arm failed: " + " | ".join(details))
    assert collection is not None
    return collection


def run(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    gpu_indices = derive_gpu_indices(args.base_gpu_id, args.gpu_id_step)
    try:
        output_dir.relative_to(work_dir)
    except ValueError as exc:
        raise OrchestratorError("--output-dir must resolve inside --work-dir") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    required_files = {
        "dog": args.dog,
        "reference": args.reference,
        "runtime_environment": args.runtime_environment,
        "smg_packages": args.smg_packages,
        "pip_install_report": args.pip_install_report,
    }
    missing = [
        f"{name}={path}" for name, path in required_files.items() if not path.is_file()
    ]
    if missing:
        raise OrchestratorError("required files are missing: " + ", ".join(missing))

    common_provenance = (
        ("runtime_environment", args.runtime_environment),
        ("smg_packages", args.smg_packages),
        ("pip_install_report", args.pip_install_report),
    )
    launch_ids = tuple(dict.fromkeys(launch_id for launch_id, _ in LAUNCH_SEQUENCE))
    if len(launch_ids) < benchmark.MIN_LAUNCH_PAIRS:
        raise OrchestratorError(
            f"launch sequence needs at least {benchmark.MIN_LAUNCH_PAIRS} pairs"
        )
    for launch_id in launch_ids:
        arms = [arm for candidate, arm in LAUNCH_SEQUENCE if candidate == launch_id]
        if sorted(arms) != ["eager", "graph"]:
            raise OrchestratorError(
                f"launch {launch_id!r} must contain one eager and one graph arm"
            )

    baseline_hardware: dict[str, Any] | None = None
    collections: dict[tuple[str, str], dict[str, Any]] = {}
    launch_artifacts: dict[str, dict[str, dict[str, str]]] = {}
    execution_order: list[dict[str, Any]] = []
    for execution_index, (launch_id, arm) in enumerate(LAUNCH_SEQUENCE):
        arm_dir = output_dir / launch_id / arm
        hardware_path = arm_dir / "hardware.json"
        hardware = capture_hardware_identity(gpu_indices)
        _write_json_atomic(hardware_path, hardware)
        if baseline_hardware is None:
            baseline_hardware = hardware
        elif hardware != baseline_hardware:
            raise OrchestratorError(
                "stable GPU identity/driver/clock/power state changed before "
                f"{launch_id}/{arm}"
            )
        collection = _run_arm(
            arm=arm,
            launch_id=launch_id,
            work_dir=work_dir,
            arm_dir=arm_dir,
            dog=args.dog,
            reference=args.reference,
            server_sha=args.server_sha,
            provenance_files=((*common_provenance, ("hardware", hardware_path))),
            request_timeout_seconds=args.request_timeout_seconds,
            base_gpu_id=args.base_gpu_id,
            gpu_id_step=args.gpu_id_step,
        )
        collections[(launch_id, arm)] = collection
        execution_order.append(
            {
                "index": execution_index,
                "launch_id": launch_id,
                "arm": arm,
                "arm_artifact": str(arm_dir / "arm.json"),
            }
        )
        launch_artifacts.setdefault(launch_id, {})[arm] = {
            "arm": str(arm_dir / "arm.json"),
            "hardware": str(hardware_path),
            "log_validation": str(arm_dir / "server-log-validation.json"),
            "shutdown": str(arm_dir / "shutdown-validation.json"),
        }

    assert baseline_hardware is not None
    eager_paths = [
        output_dir / launch_id / "eager/arm.json" for launch_id in launch_ids
    ]
    graph_paths = [
        output_dir / launch_id / "graph/arm.json" for launch_id in launch_ids
    ]

    comparison_path = output_dir / "comparison.json"
    comparison = benchmark.compare_arms(
        eager_paths,
        graph_paths,
        output=comparison_path,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result = {
        "schema_version": 1,
        "workload": "minimax_m3_encoder_graph_ab_ci",
        "ok": all(collection["ok"] for collection in collections.values())
        and comparison["ok"],
        "server_sha": args.server_sha,
        "gpu_selection": {
            "base_gpu_id": args.base_gpu_id,
            "gpu_id_step": args.gpu_id_step,
            "physical_indices": list(gpu_indices),
        },
        "launch_sequence": execution_order,
        "hardware": baseline_hardware,
        "artifacts": {
            "launches": launch_artifacts,
            "comparison": str(comparison_path),
            "runtime_environment": str(args.runtime_environment),
            "smg_packages": str(args.smg_packages),
            "pip_install_report": str(args.pip_install_report),
        },
    }
    _write_json_atomic(output_dir / "ci-validation.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runtime-environment", type=Path, required=True)
    parser.add_argument("--smg-packages", type=Path, required=True)
    parser.add_argument("--pip-install-report", type=Path, required=True)
    parser.add_argument("--server-sha", required=True)
    parser.add_argument("--base-gpu-id", type=int, default=DEFAULT_BASE_GPU_ID)
    parser.add_argument("--gpu-id-step", type=int, default=DEFAULT_GPU_ID_STEP)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=benchmark.DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=benchmark.DEFAULT_BOOTSTRAP_SEED
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (OrchestratorError, benchmark.BenchmarkError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"MiniMax-M3 encoder Graph paired CI: {'PASS' if result['ok'] else 'FAIL'}",
        flush=True,
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
