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

"""Paired ordinary-generation observations, with an independent client clock.

Run ``python examples/profiling/service_tpot.py --help``. All workload and
measurement choices come from one frozen JSON file. This module imports no
runtime/GPU package until the explicitly requested child run starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


@dataclass
class StreamSample:
    """Real arrival chunks; a multi-token chunk has one timestamp."""

    chunks: list[dict] = field(default_factory=list)
    rid: str | None = None
    _cumulative: int = 0
    logical_output_ids: list[int] | None = None
    logical_output_sha256: str | None = None
    logical_output_status: str = "unfinished"

    def add(self, chunk: dict, arrival_ns: int) -> None:
        meta = chunk["meta_info"]
        rid = meta["id"]
        if self.rid is not None and self.rid != rid:
            raise ValueError("one sample must contain exactly one request")
        self.rid = rid
        cumulative = int(meta["completion_tokens"])
        if cumulative < self._cumulative:
            raise ValueError("completion_tokens decreased")
        token_ids = chunk.get("output_ids")
        self.chunks.append(
            {
                "arrival_ns": arrival_ns,
                "completion_tokens": cumulative,
                "new_tokens": cumulative - self._cumulative,
                "output_ids": token_ids,
                "output_ids_sha256": None,
                "finish_reason": meta["finish_reason"],
                "cached_tokens": meta.get("cached_tokens"),
            }
        )
        self._cumulative = cumulative

    def finish(self) -> None:
        # Hash after the last arrival so evidence bookkeeping does not delay
        # draining the next streaming chunk.
        for chunk in self.chunks:
            ids = chunk["output_ids"]
            chunk["output_ids_sha256"] = digest(ids) if ids is not None else None
        logical = []
        for chunk in self.chunks:
            ids = chunk["output_ids"]
            if ids is None:
                self.logical_output_status = "missing_raw_token_ids"
                return
            # Validate the two real client contracts (delta and cumulative).
            # If both apply they must yield the exact same logical sequence.
            candidates = set()
            if len(ids) == chunk["new_tokens"]:
                candidates.add(tuple(logical + ids))
            if (
                len(ids) == chunk["completion_tokens"]
                and ids[: len(logical)] == logical
            ):
                candidates.add(tuple(ids))
            if len(candidates) != 1:
                self.logical_output_status = "inconsistent_token_count_contract"
                return
            logical = list(candidates.pop())
            if len(logical) != chunk["completion_tokens"]:
                self.logical_output_status = "inconsistent_token_count_contract"
                return
        self.logical_output_ids = logical
        self.logical_output_sha256 = digest(logical)
        self.logical_output_status = "verified"

    def window(self, first_count: int | None, last_count: int | None) -> dict:
        chunks = [c for c in self.chunks if c["new_tokens"] > 0]
        if first_count is not None:
            chunks = [
                c for c in chunks if first_count <= c["completion_tokens"] <= last_count
            ]
        if len(chunks) < 2:
            return {"status": "insufficient_distinct_arrival_chunks"}
        first, last = chunks[0], chunks[-1]
        if first_count is not None and (
            first["completion_tokens"] != first_count
            or last["completion_tokens"] != last_count
        ):
            return {"status": "requested_endpoint_not_received"}
        count = last["completion_tokens"] - first["completion_tokens"]
        elapsed = last["arrival_ns"] - first["arrival_ns"]
        return {
            "status": "observed",
            "first_count": first["completion_tokens"],
            "last_count": last["completion_tokens"],
            "first_ns": first["arrival_ns"],
            "last_ns": last["arrival_ns"],
            "token_intervals": count,
            "elapsed_ns": elapsed,
            "tpot_ns": elapsed / count,
        }


def profile_option(command: list, name: str) -> str | None:
    """Read one exact Nsight option, accepting both CLI value spellings."""
    values = []
    for index, argument in enumerate(command):
        argument = str(argument)
        if argument == name:
            values.append(str(command[index + 1]) if index + 1 < len(command) else "")
        elif argument.startswith(name + "="):
            values.append(argument.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"duplicate Nsight option {name}")
    return values[0] if values else None


def verify_profile_launch(command: list) -> None:
    """Keep the service alive when CUDA profiler API collection ends."""
    if (
        profile_option(command, "--capture-range") == "cudaProfilerApi"
        and profile_option(command, "--kill") != "none"
    ):
        raise ValueError("cudaProfilerApi launch requires explicit --kill=none")


def verify_capture_collection() -> list[dict]:
    """Require actual same-run early-start receipts before request warmup."""
    timeline = json.loads(Path(os.environ["TOKENSPEED_TIMELINE_CONFIG"]).read_text())
    ranks = timeline.get("ranks")
    if not ranks or 0 not in ranks:
        raise ValueError(
            "cudaProfilerApi requires explicit timeline ranks including rank 0"
        )
    receipts = []
    for rank in ranks:
        path = Path(timeline["output_dir"]) / f"rank-{rank}.collection.json"
        receipt = json.loads(path.read_text())
        capture = Path(timeline["output_dir"]) / (
            f"rank-{rank}.capture-{receipt['next_capture_sequence']}.json"
        )
        report = json.loads(capture.read_text())
        if (
            receipt.get("schema") != "tokenspeed.profile_collection/v1"
            or timeline["run_id"] != os.environ["TOKENSPEED_PROFILE_RUN_ID"]
            or report.get("run_id") != timeline["run_id"]
            or report.get("process_id") != receipt.get("process_id")
            or report.get("capture_id")
            != f"{timeline['run_id']}:r{rank}:c{receipt['next_capture_sequence']}"
            or report.get("graph_key") != receipt.get("graph_key")
            or receipt.get("run_id") != os.environ["TOKENSPEED_PROFILE_RUN_ID"]
            or receipt.get("rank") != rank
            or receipt.get("event") != "cuda_profiler_started_before_capture_rendezvous"
        ):
            raise ValueError(f"missing actual same-run early capture for rank {rank}")
        receipts.append(
            {
                **receipt,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "capture_path": str(capture),
                "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
            }
        )
    return receipts


def load_case(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    case = json.loads(raw)
    required = {
        "engine_args",
        "sampling_params",
        "input_ids",
        "warmup_requests",
        "measured_requests",
        "sequence",
        "statistical_window",
        "profile_environment",
        "cache_policy",
        "instrumentation_scope",
        "coverage_mode",
    }
    missing = required - case.keys()
    if missing:
        raise ValueError(f"case is missing {sorted(missing)}")
    if case["coverage_mode"] != "decode":
        raise ValueError("the current pilot requires explicit coverage_mode=decode")
    if case["cache_policy"] != "flush_before_each_request":
        raise ValueError("unsupported cache_policy; use flush_before_each_request")
    if case["instrumentation_scope"] not in {"service_only", "model_layer_op"}:
        raise ValueError("unsupported instrumentation_scope")
    full_scope = case["instrumentation_scope"] == "model_layer_op"
    has_collector = "TOKENSPEED_TIMELINE_CONFIG" in case["profile_environment"]
    if full_scope != has_collector:
        raise ValueError(
            "instrumentation_scope must explicitly match collector configuration"
        )
    command = case.get("profile_command", [])
    capture_range = profile_option(command, "--capture-range")
    if capture_range == "cudaProfilerApi" and not full_scope:
        raise ValueError("cudaProfilerApi requires the early-start graph collector")
    if full_scope:
        if capture_range not in {"none", "cudaProfilerApi"}:
            raise ValueError(
                "full collector requires explicit Nsight "
                "capture-range=none or cudaProfilerApi"
            )
        if profile_option(command, "--cuda-graph-trace") != "node":
            raise ValueError("full collector requires Nsight --cuda-graph-trace=node")
    if full_scope and capture_range == "cudaProfilerApi":
        if profile_option(command, "--capture-range-end") != "stop":
            raise ValueError("cudaProfilerApi requires --capture-range-end=stop")
        args = case["engine_args"]
        sizes = args.get("cudagraph_capture_sizes")
        if args.get("enforce_eager") is not False or (
            sizes is not None and 1 not in sizes
        ):
            raise ValueError(
                "cudaProfilerApi requires enabled decode graphs "
                "including batch size 1"
            )
    if not case["sequence"] or any(
        arm not in {"baseline", "profiled"} for arm in case["sequence"]
    ):
        raise ValueError("sequence must explicitly list baseline/profiled arms")
    if (
        case["sequence"].count("baseline") < 2
        or "profiled" not in case["sequence"]
        or case["sequence"][0] != "baseline"
    ):
        raise ValueError(
            "the paired protocol needs two reference arms and a subsequent profiled arm"
        )
    if case["warmup_requests"] < 0 or case["measured_requests"] < 1:
        raise ValueError("invalid request counts")
    if case["engine_args"].get("load_format") == "dummy":
        raise ValueError("baseline requires real weights")
    if any("acceptance" in key or "dummy" in key for key in case["engine_args"]):
        raise ValueError(
            "synthetic acceptance/dummy settings are not a service baseline"
        )
    if (
        case["sampling_params"].get("temperature") != 0
        or case["sampling_params"].get("ignore_eos") is not True
    ):
        raise ValueError(
            "freeze temperature=0 and ignore_eos=true for the core fixed-length baseline"
        )
    if case["sampling_params"].get("max_new_tokens", 0) < 2:
        raise ValueError("the baseline requires at least two generated tokens")
    if case["engine_args"].get("stream_interval") != 1:
        raise ValueError("freeze stream_interval=1 for the core baseline")
    window = case["statistical_window"]
    if set(window) != {"first_count", "last_count"}:
        raise ValueError(
            "window requires first_count/last_count (both null means full observed span)"
        )
    first, last = window["first_count"], window["last_count"]
    if (first is None) != (last is None) or (
        first is not None and not 0 < first < last
    ):
        raise ValueError("invalid cumulative-token statistical window")
    if first is None or first < 2 or last > case["sampling_params"]["max_new_tokens"]:
        raise ValueError(
            "decode coverage requires explicit received token endpoints within 2..max_new_tokens"
        )
    if (
        "TOKENSPEED_SERVICE_NVTX_LIBRARY" in case["profile_environment"]
        or "TOKENSPEED_SERVICE_NVTX_LIBRARY" in os.environ
    ):
        raise ValueError(
            "TOKENSPEED_SERVICE_NVTX_LIBRARY was removed; use the profiler command with Torch NVTX support"
        )
    forbidden = {
        "TOKENSPEED_SERVICE_TRACE",
        "TOKENSPEED_PROFILE_RUN_ID",
        "TOKENSPEED_SERVICE_NVTX",
    }
    if forbidden & case["profile_environment"].keys():
        raise ValueError("driver owns trace output and run identity")
    return case, hashlib.sha256(raw).hexdigest()


def observe_before_engine(
    case: dict, case_hash: str, *, repetition: int
) -> dict | None:
    """Write automatic data-only startup provenance before any Engine import."""
    if case["instrumentation_scope"] != "model_layer_op":
        return None
    path = Path(os.environ["TOKENSPEED_TIMELINE_CONFIG"])
    timeline = json.loads(path.read_text())
    run_id = os.environ["TOKENSPEED_PROFILE_RUN_ID"]
    if (
        timeline.get("run_id") != run_id
        or run_id != f"{case_hash[:16]}-{repetition}"
        or "startup_observation" in timeline
    ):
        raise ValueError(
            "Before-Engine observation requires the fresh same-arm timeline"
        )
    source = artifact_source(Path(__file__).resolve())
    clock_identity = {}
    try:
        clock_identity = dict(
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            time_namespace=os.readlink("/proc/self/ns/time"),
            pid_namespace=os.readlink("/proc/self/ns/pid"),
        )
    except OSError as error:
        clock_identity["unavailable_reason"] = f"{type(error).__name__}: {error}"
    before = time.time_ns()
    monotonic = time.monotonic_ns()
    after = time.time_ns()
    body = dict(
        schema="tokenspeed.before_engine_observation/v1",
        run_id=run_id,
        arm="profiled",
        repetition=repetition,
        case_sha256=case_hash,
        process_id=os.getpid(),
        parent_process_id=os.getppid(),
        realtime_before_ns=before,
        monotonic_ns=monotonic,
        realtime_after_ns=after,
        clock_identity=clock_identity,
        driver_source=source,
        timeline_config=timeline,
        timeline_config_sha256=digest(timeline),
    )
    observation = dict(body=body, sha256=digest(body))
    path.write_text(
        json.dumps({**timeline, "startup_observation": observation}, sort_keys=True)
        + "\n"
    )
    return observation


def run_arm(case_path: Path, output: Path, arm: str, repetition: int) -> None:
    case, case_hash = load_case(case_path)
    if arm == "profiled":
        verify_profile_launch(case.get("profile_command", []))
    if (
        type(repetition) is not int
        or not 0 <= repetition < len(case["sequence"])
        or case["sequence"][repetition] != arm
    ):
        raise ValueError("Run arm/repetition differs from the frozen case")
    observation = (
        observe_before_engine(case, case_hash, repetition=repetition)
        if arm == "profiled"
        else None
    )
    if arm == "profiled":
        prepare_service_nvtx(case, output.parent, repetition=repetition)
    # This observation precedes serving/model imports, worker creation and JIT.
    # Baseline children retain their ordinary unobserved startup.
    from tokenspeed.runtime.entrypoints.engine import Engine

    frozen_model_config = case_path.parent / "model-config.json"
    if case["instrumentation_scope"] == "model_layer_op":
        actual_config = Path(case["engine_args"]["model"]) / "config.json"
        if (
            not frozen_model_config.is_file()
            or actual_config.read_bytes() != frozen_model_config.read_bytes()
        ):
            raise ValueError(
                "actual model config differs from the frozen model-depth source"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", buffering=1) as stream:

        def write(event):
            stream.write(json.dumps(event, sort_keys=True) + "\n")

        write(
            {
                "kind": "run",
                "case_sha256": case_hash,
                "arm": arm,
                "repetition": repetition,
                "case": case,
                "run_id": os.environ["TOKENSPEED_PROFILE_RUN_ID"],
                "clock": "time.perf_counter_ns/client_stream_chunk_arrival",
                "startup_observation_sha256": (
                    observation["sha256"] if observation else None
                ),
            }
        )
        engine = Engine(**case["engine_args"])
        try:
            if (
                arm == "profiled"
                and case["instrumentation_scope"] == "model_layer_op"
                and profile_option(case.get("profile_command", []), "--capture-range")
                == "cudaProfilerApi"
            ):
                write(
                    {
                        "kind": "early_collection",
                        "receipts": verify_capture_collection(),
                        "native_ownership": "requires_same_run_capture_instantiate_clone_proof",
                    }
                )
            run_protocol(
                engine,
                case,
                arm=arm,
                run_id=os.environ["TOKENSPEED_PROFILE_RUN_ID"],
                write=write,
            )
        finally:
            engine.shutdown()


def run_protocol(engine, case: dict, *, arm: str, run_id: str, write) -> None:
    """Use the real service control API; cache and profile actions precede timing."""
    armed = False
    try:
        for phase, count in (
            ("warmup", case["warmup_requests"]),
            ("measured", case["measured_requests"]),
        ):
            if phase == "measured" and arm == "profiled":
                reply = engine.start_profile(
                    activities=["CUDA_PROFILER"], profile_id=run_id
                )
                write(
                    {
                        "kind": "profile_control",
                        "action": "start",
                        "success": reply.success,
                        "message": reply.message,
                        "requested_scope": case["instrumentation_scope"],
                    }
                )
                if reply.success is not True:
                    raise RuntimeError("service rejected profiler start")
                armed = True
            for index in range(count):
                # flush_cache clears service prefix/KV cache, not model weights,
                # captured CUDA graphs, or compilation caches.
                reply = engine.flush_cache()
                write(
                    {
                        "kind": "cache_control",
                        "phase": phase,
                        "index": index,
                        "policy": case["cache_policy"],
                        "success": reply.success,
                    }
                )
                if reply.success is not True:
                    raise RuntimeError("service rejected cache flush")
                sample = StreamSample()
                requested_ns = time.perf_counter_ns()
                for chunk in engine.generate(
                    input_ids=case["input_ids"],
                    sampling_params=case["sampling_params"],
                    stream=True,
                ):
                    arrival_ns = time.perf_counter_ns()
                    sample.add(chunk, arrival_ns)
                sample.finish()
                window = sample.window(**case["statistical_window"])
                cache_status = (
                    "verified"
                    if sample.chunks
                    and all(chunk["cached_tokens"] == 0 for chunk in sample.chunks)
                    else "incomplete"
                )
                write(
                    {
                        "kind": "request",
                        "phase": phase,
                        "index": index,
                        "requested_ns": requested_ns,
                        "client_ttft_ns": next(
                            (
                                chunk["arrival_ns"] - requested_ns
                                for chunk in sample.chunks
                                if chunk["new_tokens"] > 0
                            ),
                            None,
                        ),
                        "sample": asdict(sample),
                        "window": window,
                        "cache_policy_status": cache_status,
                    }
                )
    finally:
        if armed:
            reply = engine.stop_profile()
            write(
                {
                    "kind": "profile_control",
                    "action": "stop",
                    "success": reply.success,
                    "message": reply.message,
                }
            )
            if reply.success is not True:
                raise RuntimeError("service rejected profiler stop/flush")


def artifact_source(path: Path) -> dict:
    """Fingerprint a finished source without loading large reports into memory."""
    path = path.resolve(strict=True)
    fingerprint = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    return {
        "path": str(path),
        "sha256": fingerprint.hexdigest(),
        "bytes": path.stat().st_size,
    }


def prepare_service_nvtx(case: dict, output_dir: Path, *, repetition: int) -> None:
    """Prepare Python marker imports in the actual profiled child, before Engine.

    JSON-only children do not import Torch/NVTX. The service's first start
    marker validates its binding before any measured request callback.
    """
    enabled = bool(case.get("profile_command"))
    os.environ["TOKENSPEED_SERVICE_NVTX"] = "1" if enabled else "0"
    receipt = dict(
        schema="tokenspeed.service_nvtx_provider/v1",
        run_id=os.environ["TOKENSPEED_PROFILE_RUN_ID"],
        process_id=os.getpid(),
        enabled=enabled,
        selection="json_only",
        sources=[],
    )
    if enabled:
        import inspect

        import torch

        from tokenspeed.runtime.engine import service_trace

        marker = service_trace.prepare_nvtx_marker()
        source = inspect.getsourcefile(marker)
        if source is None:
            raise RuntimeError("Torch NVTX Python source identity unavailable")
        receipt.update(
            selection="torch.cuda.nvtx.mark",
            torch_version=str(torch.__version__),
            torch_cuda_version=torch.version.cuda,
            torch_git_version=torch.version.git_version,
            binding_validation="first synchronous ServiceTrace start marker before request warmup",
            sources=[
                artifact_source(Path(source)),
                artifact_source(Path(service_trace.__file__)),
            ],
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{repetition}-service-nvtx.json"
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


def write_run_manifest(output_dir: Path, index: int, run_id: str) -> Path:
    """Seal one completed profiled arm, including transitive replay sources.

    This inventory is integrity/provenance, never a selection of executable cost
    providers. Missing/incomplete GPU coverage is left to the existing analyzer.
    """
    artifacts = {}

    def retain(path: Path) -> str:
        source = artifact_source(path)
        artifacts[source["path"]] = source
        return source["path"]

    ranks = []
    graph_dir = output_dir / f"{index}-graph"
    for path in sorted(graph_dir.glob("rank-*.window.json")):
        window = json.loads(path.read_text())
        rank = window["rank"]
        if type(rank) is not int or rank < 0 or window["run_id"] != run_id:
            raise ValueError("Cross-run/rank collector window in run manifest")
        replay = graph_dir / f"rank-{rank}.replays.jsonl"
        for line in replay.read_text().splitlines():
            row = json.loads(line)
            if type(row.get("rank")) is not int or (
                row.get("run_id"),
                row.get("rank"),
            ) != (run_id, rank):
                raise ValueError("Cross-run/rank replay in run manifest")
            if row.get("capture_file"):
                retain(Path(row["capture_file"]))
            if row.get("binary_payload"):
                retain(Path(row["binary_payload"]["path"]))
            if row.get("eager_sidecar") is not None:
                from tokenspeed_kernel.profile_eager import read_eager_sidecar

                read_eager_sidecar(
                    row, config_sha256=window.get("profile_config_sha256")
                )
                eager = row["eager_sidecar"]
                name = retain(Path(eager["path"]))
                if artifacts[name] != eager:
                    raise ValueError("Run manifest eager sidecar fingerprint mismatch")
        ranks.append(
            {
                "rank": rank,
                "window": retain(path),
                "replays": retain(replay),
                "config": retain(graph_dir / f"rank-{rank}.config.json"),
            }
        )
    if len({row["rank"] for row in ranks}) != len(ranks):
        raise ValueError("Duplicate collector rank in run manifest")
    manifest = {
        "schema": "tokenspeed.profile_run/v1",
        "run_id": run_id,
        "arm_index": index,
        "report": (
            retain(output_dir / f"{index}-nsys.nsys-rep")
            if json.loads((output_dir / "case.json").read_text()).get("profile_command")
            else None
        ),
        "client": retain(output_dir / f"{index}-profiled.jsonl"),
        "case": retain(output_dir / "case.json"),
        "metadata": retain(output_dir / f"{index}-metadata-coverage.json"),
        "model_config": (
            retain(output_dir / "model-config.json")
            if (output_dir / "model-config.json").is_file()
            else None
        ),
        "ranks": ranks,
        "artifacts": artifacts,
    }
    communication = sorted(
        set(graph_dir.glob("rank-*.communication.json"))
        | set(graph_dir.glob("rank-*.mnnvl-allocation-*.json"))
    )
    if communication:
        from tokenspeed_kernel.profile_communication_inventory import (
            communication_inventory,
        )

        sealed = communication_inventory(
            [path.absolute() for path in communication], run_id=run_id
        )
        artifacts.update(sealed)
        manifest["communication_evidence"] = sorted(sealed)
    for path in sorted(output_dir.glob(f"{index}-service-*.jsonl")):
        retain(path)
    for path in (
        output_dir / f"{index}-service-nvtx.json",
        output_dir / f"{index}-timeline.json",
    ):
        if path.is_file():
            retain(path)
            if path.name == f"{index}-timeline.json":
                observation = json.loads(path.read_text()).get("startup_observation")
                if observation is not None:
                    driver = observation["body"]["driver_source"]
                    driver_path = retain(Path(driver["path"]))
                    if artifacts[driver_path] != driver:
                        raise ValueError("Before-Engine driver source changed")
                    manifest["timeline"] = str(path.resolve())
                    manifest["startup_observation_sha256"] = observation["sha256"]
    provider = json.loads((output_dir / f"{index}-service-nvtx.json").read_text())
    if provider["run_id"] != run_id:
        raise ValueError("Service NVTX provider receipt belongs to another run")
    for source in provider["sources"]:
        path = retain(Path(source["path"]))
        if artifacts[path] != source:
            raise ValueError("Service NVTX provider source changed after preparation")
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / f"run-{index}.{hashlib.sha256(raw).hexdigest()}.json"
    with path.open("xb") as stream:
        stream.write(raw)
    return path


def run_pair(case_path: Path, output_dir: Path) -> None:
    case, case_hash = load_case(case_path)
    verify_profile_launch(case.get("profile_command", []))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = output_dir / "case.json"
    frozen.write_bytes(case_path.read_bytes())
    if case["instrumentation_scope"] == "model_layer_op":
        config_source = Path(case["engine_args"]["model"]) / "config.json"
        config_raw = config_source.read_bytes()
        model_layer_ids(config_raw)
        (output_dir / "model-config.json").write_bytes(config_raw)
    timeline_template = None
    timeline_path = case["profile_environment"].get("TOKENSPEED_TIMELINE_CONFIG")
    if timeline_path is not None:
        source = Path(timeline_path)
        raw_timeline = source.read_bytes()
        (output_dir / "timeline-template.json").write_bytes(raw_timeline)
        timeline_template = json.loads(raw_timeline)
    # Explicitly reject leaked observation settings in the reference arm.
    profile_keys = set(case["profile_environment"]) | {
        "TOKENSPEED_SERVICE_TRACE",
        "TOKENSPEED_SERVICE_NVTX",
        "TOKENSPEED_TIMELINE_CONFIG",
    }
    for index, arm in enumerate(case["sequence"]):
        env = os.environ.copy()
        for key in profile_keys:
            env.pop(key, None)
        # The run id is deterministic provenance, not a request-order join.
        env["TOKENSPEED_PROFILE_RUN_ID"] = f"{case_hash[:16]}-{index}"
        if arm == "profiled":
            env.update(
                {key: str(value) for key, value in case["profile_environment"].items()}
            )
            env["TOKENSPEED_SERVICE_TRACE"] = str(
                (output_dir / f"{index}-service-{{pid}}.jsonl").resolve()
            )
            if timeline_template is not None:
                timeline = {
                    **timeline_template,
                    "run_id": env["TOKENSPEED_PROFILE_RUN_ID"],
                    "output_dir": str((output_dir / f"{index}-graph").resolve()),
                }
                config_path = output_dir / f"{index}-timeline.json"
                config_path.write_text(json.dumps(timeline, sort_keys=True) + "\n")
                env["TOKENSPEED_TIMELINE_CONFIG"] = str(config_path.resolve())
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--case",
            str(frozen.resolve()),
            "--output",
            str((output_dir / f"{index}-{arm}.jsonl").resolve()),
            "--arm",
            arm,
            "--repetition",
            str(index),
        ]
        # Optional nsys command prefix is frozen in the case. It is applied
        # only to profiled runs, never to the reference/noise observations.
        if arm == "profiled":
            command = [
                str(value).format(output=str((output_dir / f"{index}-nsys").resolve()))
                for value in case.get("profile_command", [])
            ] + command
        with (output_dir / f"{index}-{arm}.log").open("xb") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summarize_pair(output_dir), indent=2, sort_keys=True) + "\n"
    )

    for index, arm in enumerate(case["sequence"]):
        if arm == "profiled":
            path = write_run_manifest(output_dir, index, f"{case_hash[:16]}-{index}")
            print(f"Run manifest: {path}", flush=True)


def model_layer_ids(raw: bytes) -> list[int]:
    """Read model depth from the frozen checkpoint config, not observed rows."""
    config = json.loads(raw)
    config = config.get("text_config", config)
    canonical, alias = config.get("num_hidden_layers"), config.get("num_layers")
    # ModelConfig accepts num_layers when the canonical spelling is absent;
    # released Lite checkpoints use this alias. Conflicting raw declarations
    # are ambiguous across config adapters, so do not guess the model depth.
    declared = [value for value in (canonical, alias) if value is not None]
    if not declared or any(type(value) is not int or value < 1 for value in declared):
        raise ValueError(
            "model config must declare a positive integer num_hidden_layers or num_layers"
        )
    if canonical is not None and alias is not None and canonical != alias:
        raise ValueError(
            "model config has conflicting num_hidden_layers and num_layers"
        )
    depth = canonical if canonical is not None else alias
    return list(range(depth))


def verify_instrumentation(
    output_dir: Path, index: int, records: list[dict], *, requested_scope: str
) -> dict:
    """Verify actual ready metadata against true committed token intervals.

    GPU node/clone ownership is a separate offline Nsight analysis. Even fully
    verified host metadata is not promoted to full Model/Layer/Op GPU coverage.
    """
    run_id = records[0]["run_id"]
    measured = [
        row
        for row in records
        if row.get("kind") == "request" and row["phase"] == "measured"
    ]
    controls = {
        row["action"]: row for row in records if row.get("kind") == "profile_control"
    }
    lifecycle = all(
        controls.get(action, {}).get("success") is True for action in ("start", "stop")
    )
    result = {
        "schema": "tokenspeed.metadata_coverage/v1",
        "run_id": run_id,
        "requested_scope": requested_scope,
        "requested_coverage_mode": "decode",
        "verified_scope": "unverified",
        "metadata_status": "incomplete",
        "status": "incomplete",
        "profile_lifecycle_verified": lifecycle,
        "requests": {},
        "native_graph_ownership": "requires_offline_nsight_node_clone_analysis",
    }
    service = []
    service_sources = []
    for path in output_dir.glob(f"{index}-service-*.jsonl"):
        raw_service = path.read_bytes()
        service_sources.append(
            {"path": str(path), "sha256": hashlib.sha256(raw_service).hexdigest()}
        )
        service.extend(json.loads(line) for line in raw_service.splitlines())
    result["service_sources"] = service_sources
    same_run = [row for row in service if row.get("run_id") == run_id]
    native_markers = any(
        row.get("boundary") == "start" and row.get("nvtx") is True for row in same_run
    )
    result["native_marker_requested_and_loaded"] = native_markers
    if same_run:
        result["verified_scope"] = "service_marker_observations"
    if requested_scope == "service_only":
        result["model_layer_op_overhead"] = "unavailable_service_only"
        for row in measured:
            result["requests"][str(row["index"])] = {
                "status": "unavailable_service_only"
            }
        return result
    frozen_config = output_dir / "model-config.json"
    if not frozen_config.is_file():
        result["reason"] = "missing_frozen_model_depth_source"
        return result
    raw_config = frozen_config.read_bytes()
    expected_model_layers = model_layer_ids(raw_config)
    result["model_config_source"] = {
        "path": str(frozen_config),
        "sha256": hashlib.sha256(raw_config).hexdigest(),
    }
    result["expected_model_layer_ids"] = expected_model_layers
    graph_dir = output_dir / f"{index}-graph"
    window_path = graph_dir / "rank-0.window.json"
    replay_path = graph_dir / "rank-0.replays.jsonl"
    if not window_path.is_file() or not replay_path.is_file():
        result["reason"] = "missing_actual_collector_window_or_ready_index"
        return result
    manifest = json.loads(window_path.read_text())
    result["window_manifest"] = manifest
    result["window_manifest_source"] = {
        "path": str(window_path),
        "sha256": hashlib.sha256(window_path.read_bytes()).hexdigest(),
    }
    result["ready_index_source"] = {
        "path": str(replay_path),
        "sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
    }
    window_ok = (
        manifest.get("run_id") == run_id
        and manifest.get("rank") == 0
        and manifest.get("writer_flushed") is True
        and manifest.get("writer_error") is None
        and manifest.get("unresolved_result_metadata") == 0
    )
    replay_rows = [json.loads(line) for line in replay_path.read_text().splitlines()]
    ready = [
        row
        for row in replay_rows
        if row.get("event") == "device_results_ready"
        and row.get("run_id") == run_id
        and row.get("rank") == 0
    ]
    bindings = [
        row for row in replay_rows if row.get("event") == "forward_identity_bound"
    ]
    if bindings:
        from tokenspeed_kernel.profile_eager import merge_forward_binding

        by_step = {row["step_id"]: row for row in ready}
        for binding in bindings:
            row = by_step.get(binding["step_id"])
            if row is None:
                raise ValueError("Late ForwardId binding lacks a device-ready step")
            merge_forward_binding(row, binding)
    commits = [
        row
        for row in same_run
        if row.get("boundary") == "commit"
        and row.get("graph") is not None
        and row["graph"].get("rank") == 0
    ]
    dropped = [row for row in ready if row.get("unavailable_reason") is not None]
    measured_forward_ids = {
        row["forward_id"]
        for request in measured
        for row in commits
        if request["window"]["status"] == "observed"
        and row["rid"] == request["sample"]["rid"]
        and request["window"]["first_count"]
        <= row["completion_tokens"]
        <= request["window"]["last_count"]
    }
    drop_keys = {(row.get("forward_id"), row.get("step_id")) for row in dropped}
    drops_identified = (
        type(manifest.get("dropped_metadata")) is int
        and manifest["dropped_metadata"] == len(dropped) == len(drop_keys)
        and all(
            row.get("forward_id") is not None and row.get("request_ids")
            for row in dropped
        )
    )
    drops_disjoint = (
        all(row["forward_id"] not in measured_forward_ids for row in dropped)
        if drops_identified
        else False
    )
    result["dropped_steps"] = [
        {
            key: row.get(key)
            for key in ("forward_id", "step_id", "request_ids", "unavailable_reason")
        }
        for row in dropped
    ]
    result["dropped_metadata_identified"] = drops_identified
    result["dropped_metadata_disjoint_from_measured_decode"] = drops_disjoint
    result["whole_run_scope"] = (
        "partial" if dropped or not drops_identified else "not_verified"
    )
    window_ok = bool(window_ok and drops_identified and drops_disjoint)
    for request in measured:
        key = str(request["index"])
        window = request["window"]
        report = {
            "status": "incomplete",
            "forward_ids": [],
            "forward_evidence": [],
            "rid": request["sample"]["rid"],
            "generation": None,
            "first_count": window.get("first_count"),
            "last_count": window.get("last_count"),
        }
        result["requests"][key] = report
        if window["status"] != "observed":
            report["reason"] = "client_window_unavailable"
            continue
        first, last = window["first_count"], window["last_count"]
        rows = [
            row
            for row in commits
            if row["rid"] == request["sample"]["rid"]
            and first <= row["completion_tokens"] <= last
        ]
        rows.sort(key=lambda row: row["completion_tokens"])
        if (
            not rows
            or rows[0]["completion_tokens"] != first
            or rows[-1]["completion_tokens"] != last
        ):
            report["reason"] = "missing_exact_committed_token_endpoints"
            continue
        generations = {(row["pid"], row["generation"]) for row in rows}
        if len(generations) != 1 or rows[0]["generation"] is None:
            report["reason"] = "ambiguous_commit_generation"
            continue
        report["generation"] = rows[0]["generation"]
        previous = first
        problems = []
        for offset, commit in enumerate(rows):
            cumulative = commit["completion_tokens"]
            if offset and cumulative - len(commit["output_ids"]) != previous:
                problems.append("missing_or_duplicate_committed_token_interval")
            previous = cumulative
            graph = commit["graph"]
            matches = [
                row
                for row in ready
                if row.get("forward_id") == commit["forward_id"]
                and row.get("step_id") == graph.get("step_id")
                and commit["rid"] in row.get("request_ids", [])
            ]
            if len(matches) != 1:
                problems.append("missing_or_ambiguous_same_run_forward_ready_record")
                continue
            row = matches[0]
            report["forward_ids"].append(commit["forward_id"])
            report["forward_evidence"].append(
                {
                    "forward_id": commit["forward_id"],
                    "step_id": row.get("step_id"),
                    "capture_id": row.get("capture_id"),
                }
            )
            if row.get("eager_sidecar") is not None:
                from tokenspeed_kernel.profile_eager import read_eager_sidecar

                capture = read_eager_sidecar(
                    row, config_sha256=manifest.get("profile_config_sha256")
                )
                expected_scopes = sorted(
                    item["nvtx_id"] for item in capture["device_analysis"]["scopes"]
                )
                expected_layers = sorted(
                    {
                        item["layer_id"]
                        for item in capture["layers"]
                        if type(item.get("layer_id")) is int
                    }
                )
                if not capture["ops"] or expected_layers != expected_model_layers:
                    problems.append("eager_scope_metadata_incomplete")
                report["forward_evidence"][-1].update(
                    eager_sidecar=row["eager_sidecar"],
                    attribution_mode="eager_nvtx_gpu_projection",
                    expected_scope_ids=expected_scopes,
                    expected_layer_ids=expected_layers,
                )
                continue
            payload = row.get("binary_payload")
            capture_file = row.get("capture_file")
            if (
                row.get("unavailable_reason") is not None
                or payload is None
                or capture_file is None
            ):
                problems.append("collector_metadata_unavailable")
                continue
            payload_path = Path(payload["path"])
            capture_path = Path(capture_file)
            if not payload_path.is_file() or not capture_path.is_file():
                problems.append("missing_persisted_payload_or_capture")
                continue
            raw = payload_path.read_bytes()
            if (
                len(raw) != payload["bytes"]
                or hashlib.sha256(raw).hexdigest() != payload["sha256"]
            ):
                problems.append("payload_hash_or_size_mismatch")
                continue
            capture = json.loads(capture_path.read_text())
            if capture.get("run_id") != run_id or capture.get("capture_id") != row.get(
                "capture_id"
            ):
                problems.append("capture_identity_mismatch")
            scopes = capture.get("graph_scopes", [])
            expected_scopes = sorted({item["scope_id"] for item in scopes})
            expected_layers = sorted(
                {
                    item["layer_id"]
                    for item in capture.get("layers", [])
                    if type(item.get("layer_id")) is int
                }
            )
            if (
                not expected_scopes
                or not capture.get("ops")
                or expected_layers != expected_model_layers
                or capture.get("unavailable_integer_metadata")
            ):
                problems.append("capture_scope_or_dynamic_metadata_incomplete")
            report["forward_evidence"][-1].update(
                capture_file={
                    "path": str(capture_path),
                    "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                },
                binary_payload=payload,
                expected_scope_ids=expected_scopes,
                expected_layer_ids=expected_layers,
            )
        if (
            not problems
            and lifecycle
            and window_ok
            and request.get("cache_policy_status") == "verified"
        ):
            report["status"] = "metadata_verified"
        else:
            report["reasons"] = sorted(set(problems))
            report["window_lifecycle_and_cache_gate"] = bool(
                lifecycle
                and window_ok
                and request.get("cache_policy_status") == "verified"
            )
    if measured and all(
        item["status"] == "metadata_verified" for item in result["requests"].values()
    ):
        result["verified_scope"] = "service_and_model_layer_op_metadata"
        result["metadata_status"] = "verified"
    else:
        result["metadata_status"] = "incomplete"
    result["model_layer_op_overhead"] = (
        "unavailable_until_exact_native_graph_ownership_verified"
    )
    return result


def verify_native_coverage(
    output_dir: Path, index: int, metadata: dict, metadata_path: Path
) -> dict:
    """Consume the fixed offline analyzer receipt, retaining its source hashes."""
    result = dict(metadata)
    path = output_dir / f"{index}-graph/rank-0.native-coverage.json"
    if metadata["metadata_status"] != "verified" or not path.is_file():
        return result
    native = json.loads(path.read_text())
    expected = {
        "schema": "tokenspeed.native_coverage/v1",
        "run_id": metadata["run_id"],
        "rank": 0,
        "scope": "model_layer_op",
        "coverage_mode": metadata["requested_coverage_mode"],
        "source_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
    }
    attribution = native.get("attribution")
    mixed = attribution == "same_run_graph_and_eager_native_activities"
    if (
        any(native.get(key) != value for key, value in expected.items())
        or attribution
        not in {
            "same_run_cuda_graph_node_ids",
            "same_run_graph_and_eager_native_activities",
        }
        or native.get("status") not in {"partial", "verified"}
        or (mixed and not native.get("source_provenance", {}).get("eager_projection"))
    ):
        result["native_graph_ownership"] = (
            "invalid_native_receipt_identity_or_metadata_hash"
        )
        return result
    source = native.get("source_nsight_report", {})
    report_path = Path(source.get("path", ""))
    if not report_path.is_file() or artifact_source(report_path)[
        "sha256"
    ] != source.get("sha256"):
        result["native_graph_ownership"] = "native_report_hash_unverified"
        return result
    selected = {
        item["forward_id"]
        for request in metadata["requests"].values()
        for item in request["forward_evidence"]
    }
    if mixed and native.get("selected_forward_ids") != sorted(selected):
        result["native_graph_ownership"] = "native_forward_identity_mismatch"
        return result
    for key, request in metadata["requests"].items():
        evidence = native.get("requests", {}).get(key, {})
        if any(
            evidence.get(name) != request[name]
            for name in ("rid", "generation", "first_count", "last_count")
        ):
            result["native_graph_ownership"] = "native_request_identity_mismatch"
            return result
        rows = evidence.get("forward_evidence", [])
        if len(rows) != len(request["forward_evidence"]):
            result["native_graph_ownership"] = "native_forward_coverage_incomplete"
            return result
        for expected_forward in request["forward_evidence"]:
            identity = {
                name: expected_forward[name]
                for name in ("forward_id", "step_id", "capture_id")
            }
            matches = [
                row
                for row in rows
                if all(row.get(name) == value for name, value in identity.items())
            ]
            if len(matches) != 1:
                result["native_graph_ownership"] = (
                    "native_scope_or_node_attribution_incomplete"
                )
                return result
            row = matches[0]
            eager = expected_forward.get("eager_sidecar")
            if (
                eager is not None
                and (
                    not mixed
                    or row.get("attribution_mode") != "eager_nvtx_gpu_projection"
                    or row.get("eager_sidecar") != eager
                )
                or eager is None
                and mixed
                and row.get("attribution_mode") != "cuda_graph_node_ids"
            ):
                result["native_graph_ownership"] = "native_attribution_source_mismatch"
                return result
            if native["status"] == "partial":
                continue
            layers, scopes = row.get("layer_coverage", {}), row.get(
                "scope_coverage", {}
            )
            if (
                (
                    not row.get("graph_node_ids")
                    if eager is None
                    else not row.get("eager_native_timing", {}).get("activities")
                )
                or set(row.get("scope_kinds", [])) != {"model", "layer", "op"}
                or row.get("critical_unknowns") != []
                or layers.get("expected_ids") != expected_forward["expected_layer_ids"]
                or layers.get("attributed_ids")
                != expected_forward["expected_layer_ids"]
                or layers.get("missing_ids") != []
                or scopes.get("expected_scope_ids")
                != expected_forward["expected_scope_ids"]
                or scopes.get("attributed_scope_ids")
                != expected_forward["expected_scope_ids"]
                or scopes.get("missing_scope_ids") != []
            ):
                result["native_graph_ownership"] = (
                    "native_scope_or_node_attribution_incomplete"
                )
                return result
    result["native_evidence"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": native["status"],
        "attribution": attribution,
        "critical_unknowns": native.get("critical_unknowns", []),
        "whole_run_status": native.get("whole_run_status", "not_verified"),
        "whole_run_critical_unknowns": native.get("whole_run_critical_unknowns", []),
    }
    result["native_graph_ownership"] = native["status"]
    if native["status"] == "verified":
        result.update(
            status="verified",
            verified_scope="model_layer_op",
            model_layer_op_overhead="eligible_with_matching_token_and_cache_windows",
        )
    return result


def summarize_pair(output_dir: Path) -> dict:
    """Separate baseline noise and predeclared B-minus-previous-A overhead.

    This reports observations, never a pass/fail threshold. Per-request indices
    are paired only by the frozen workload protocol; request identities across
    two independent runs are deliberately different.
    """
    case, case_hash = load_case(output_dir / "case.json")
    runs = []
    coverage = {}
    for index, arm in enumerate(case["sequence"]):
        records = [
            json.loads(line)
            for line in (output_dir / f"{index}-{arm}.jsonl").read_text().splitlines()
        ]
        if records[0]["case_sha256"] != case_hash:
            raise ValueError("paired arms have different frozen case bytes")
        requests = {
            record["index"]: record
            for record in records[1:]
            if record.get("kind") == "request" and record["phase"] == "measured"
        }
        runs.append((arm, requests))
        if arm == "profiled":
            metadata = verify_instrumentation(
                output_dir,
                index,
                records,
                requested_scope=case["instrumentation_scope"],
            )
            metadata_path = output_dir / f"{index}-metadata-coverage.json"
            metadata_bytes = (
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            ).encode()
            if (
                not metadata_path.exists()
                or metadata_path.read_bytes() != metadata_bytes
            ):
                metadata_path.write_bytes(metadata_bytes)
            coverage[index] = verify_native_coverage(
                output_dir, index, metadata, metadata_path
            )
    baseline = [
        record["window"]["tpot_ns"]
        for arm, requests in runs
        if arm == "baseline"
        for record in requests.values()
        if record["window"]["status"] == "observed"
    ]
    pairs, noise_pairs = [], []
    previous_baseline = None
    for run_index, (arm, requests) in enumerate(runs):
        if previous_baseline is not None:
            reference_index, reference = previous_baseline
            for request_index, sample in requests.items():
                a, b = reference[request_index], sample
                if (
                    a["window"]["status"] != "observed"
                    or b["window"]["status"] != "observed"
                ):
                    continue
                aw, bw = a["window"], b["window"]
                same_window = (aw["first_count"], aw["last_count"]) == (
                    bw["first_count"],
                    bw["last_count"],
                )
                item = {
                    "reference_run": reference_index,
                    "run": run_index,
                    "request_index": request_index,
                    "same_received_window": same_window,
                    "logical_token_parity": (
                        a["sample"].get("logical_output_sha256") is not None
                        and a["sample"].get("logical_output_sha256")
                        == b["sample"].get("logical_output_sha256")
                    ),
                    "cache_policy_verified": (
                        a.get("cache_policy_status") == "verified"
                        and b.get("cache_policy_status") == "verified"
                    ),
                    "tpot_difference_ns": (
                        bw["tpot_ns"] - aw["tpot_ns"] if same_window else None
                    ),
                    "reference_tpot_ns": aw["tpot_ns"],
                    "tpot_ns": bw["tpot_ns"],
                    "reference_token_chunks": [
                        chunk["output_ids_sha256"] for chunk in a["sample"]["chunks"]
                    ],
                    "token_chunks": [
                        chunk["output_ids_sha256"] for chunk in b["sample"]["chunks"]
                    ],
                }
                if arm == "profiled":
                    gate = coverage[run_index]
                    item["requested_instrumentation_scope"] = case[
                        "instrumentation_scope"
                    ]
                    item["verified_instrumentation_scope"] = gate["verified_scope"]
                    item["coverage"] = gate["requests"].get(
                        str(request_index), {"status": "incomplete"}
                    )
                    item["model_layer_op_overhead_ns"] = (
                        item["tpot_difference_ns"]
                        if gate["status"] == "verified"
                        and item["coverage"]["status"] == "metadata_verified"
                        and item["logical_token_parity"]
                        and item["cache_policy_verified"]
                        else None
                    )
                    item["observed_arm_difference_ns"] = item["tpot_difference_ns"]
                    item["difference_scope"] = gate["verified_scope"]
                (pairs if arm == "profiled" else noise_pairs).append(item)
        if arm == "baseline":
            previous_baseline = (run_index, requests)
    return {
        "case_sha256": case_hash,
        "baseline_tpot_ns": {
            "samples": baseline,
            "mean": statistics.mean(baseline) if baseline else None,
            "stdev": statistics.stdev(baseline) if len(baseline) > 1 else None,
        },
        "baseline_noise_pairs": noise_pairs,
        "profiled_overhead_pairs": pairs,
        "instrumentation_coverage": coverage,
        "threshold_status": "not_set_user_confirms_after_real_data",
    }


def nsight_events(path: Path) -> list[dict]:
    """Read Nsight's own NVTX timestamps, without a client-clock input."""
    events = []
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
        if not {"start", "text"} <= columns:
            raise ValueError("Nsight export lacks NVTX_EVENTS.start/text")
        if "textId" in columns:
            query = "SELECT n.start, COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id"
        else:
            query = "SELECT start,text FROM NVTX_EVENTS"
        for timestamp, text in db.execute(query):
            if not text or not text.startswith('{"boundary":'):
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if event.get("schema") == "tokenspeed.service/v1":
                events.append({"timestamp_ns": int(timestamp), **event})
    return events


def align_window(
    sample: dict,
    window: dict,
    events: list[dict],
    *,
    run_id: str,
    boundary: str,
    service_pid: int | None = None,
) -> dict:
    """Join only exact run/RID/count endpoints. Unknown or ambiguous stays so."""
    if window["status"] != "observed":
        return {"status": "unknown", "reason": window["status"]}
    endpoints = []
    for count in (window["first_count"], window["last_count"]):
        candidates = [
            event
            for event in events
            if event["run_id"] == run_id
            and event["boundary"] == boundary
            and event.get("rid") == sample["rid"]
            and event.get("completion_tokens") == count
            and event.get("generation") is not None
            and event.get("forward_id") is not None
            and (service_pid is None or event["pid"] == service_pid)
        ]
        if len(candidates) != 1:
            return {
                "status": "unknown",
                "reason": "missing_or_ambiguous_exact_endpoint",
                "count": count,
            }
        endpoints.append(candidates[0])
    first, last = endpoints
    if first["pid"] != last["pid"] or first["generation"] != last["generation"]:
        return {"status": "unknown", "reason": "request_generation_changed"}
    elapsed = last["timestamp_ns"] - first["timestamp_ns"]
    tpot = elapsed / window["token_intervals"]
    return {
        "status": "observed",
        "boundary": boundary,
        "first": first,
        "last": last,
        "nsight_elapsed_ns": elapsed,
        "nsight_tpot_ns": tpot,
        "client_tpot_ns": window["tpot_ns"],
        "tpot_difference_ns": window["tpot_ns"] - tpot,
        "endpoint_residual_change_ns": window["elapsed_ns"] - elapsed,
        "first_observer_delay_ns": first.get("observer_delay_ns"),
        "last_observer_delay_ns": last.get("observer_delay_ns"),
        "note": "difference of endpoint delays; absolute clock origins are not compared",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--case", required=True, type=Path)
    pair.add_argument("--output-dir", required=True, type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output-dir", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--case", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--arm", required=True, choices=("baseline", "profiled"))
    run.add_argument("--repetition", required=True, type=int)
    align = sub.add_parser("align")
    align.add_argument("--client", required=True, type=Path)
    align.add_argument("--nsight-sqlite", required=True, type=Path)
    align.add_argument("--service-pid", type=int)
    align.add_argument(
        "--boundary",
        required=True,
        choices=("commit", "output_ready", "frontend_send_returned"),
    )
    args = parser.parse_args()
    if args.command == "pair":
        run_pair(args.case, args.output_dir)
    elif args.command == "summarize":
        print(json.dumps(summarize_pair(args.output_dir), indent=2, sort_keys=True))
    elif args.command == "run":
        run_arm(args.case, args.output, args.arm, args.repetition)
    else:
        records = [json.loads(line) for line in args.client.read_text().splitlines()]
        run_id = records[0]["run_id"]
        events = nsight_events(args.nsight_sqlite)
        for record in records[1:]:
            if record.get("kind") == "request" and record["phase"] == "measured":
                print(
                    json.dumps(
                        {
                            "index": record["index"],
                            **align_window(
                                record["sample"],
                                record["window"],
                                events,
                                run_id=run_id,
                                boundary=args.boundary,
                                service_pid=args.service_pid,
                            ),
                        },
                        sort_keys=True,
                    )
                )


if __name__ == "__main__":
    main()
