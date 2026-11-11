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

"""Join recorded op costs to GPU timestamps exported by Nsight Systems."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import sqlite3
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tokenspeed_kernel.op_profile import CostRegistry
from tokenspeed_kernel.profile_utilization import (
    ScopeTiming,
    analyze_utilization,
    hardware_for_analysis,
)


def _read_projection(path: Path) -> dict[int, dict[str, str]]:
    """Index the official projection CSV by its Nsight range ID."""
    projected = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            range_id = int(row["RangeId"])
            if range_id in projected:
                raise ValueError("Duplicate Nsight projection range")
            projected[range_id] = row
    return projected


def _read_nvtx_ranges(
    database: Path,
    *,
    capture: dict[str, Any],
    scope_ids: set[int | str],
    payload_type: str | None,
) -> dict[int | str, sqlite3.Row]:
    """Read this session's ranges, excluding other processes and threads."""
    ranges = {}
    labels = {scope["nvtx_label"] for scope in capture["scopes"]}
    process_id = capture.get("process_id")
    payload_column = {"int64": "n.int64Value", "uint64": "n.uint64Value", None: "NULL"}[
        payload_type
    ]
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        domain_join = signed_range = ""
        if payload_type == "int64":
            if type(process_id) is not int or not 0 < process_id < (1 << 24):
                raise ValueError("Signed TokenSpeed NVTX capture requires an exact PID")
            columns = {
                column[1] for column in db.execute("PRAGMA table_info(NVTX_EVENTS)")
            }
            if not {"domainId", "eventType", "int64Value"} <= columns:
                raise ValueError(
                    "Signed NVTX export lacks TokenSpeed domain or push/pop evidence"
                )
            # Domain IDs are process-local. Use the full packed process key
            # for the declaration, including any Nsight namespace bits.
            domain_join = """
            JOIN (
                SELECT d.domainId, (d.globalTid >> 24) AS process_key
                FROM NVTX_EVENTS d LEFT JOIN StringIds ds ON d.textId = ds.id
                WHERE d.eventType = 75
                GROUP BY d.domainId, process_key
                HAVING COUNT(*) = 1 AND MAX(COALESCE(ds.value, d.text)) = 'TokenSpeed'
            ) domains ON domains.domainId = n.domainId
                      AND domains.process_key = (n.globalTid >> 24)
            """
            signed_range = """
                AND n.eventType = 59
                AND typeof(n.start) = 'integer' AND typeof(n.end) = 'integer'
                AND n.end >= n.start
                AND (n.endGlobalTid IS NULL OR n.endGlobalTid = n.globalTid)
            """
        # Nsight packs PID/TID into 24-bit fields within globalTid.
        for row in db.execute(
            f"""
            SELECT n.rowid AS range_id, n.start, n.end, n.globalTid, n.endGlobalTid,
                   ((n.globalTid >> 24) & 0xFFFFFF) AS pid,
                   COALESCE(s.value, n.text) AS nvtx_label,
                   {payload_column} AS nvtx_id
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
            {domain_join}
            WHERE (n.globalTid & 0xFFFFFF) = ?
              AND (? IS NULL OR ((n.globalTid >> 24) & 0xFFFFFF) = ?)
            {signed_range}
            """,
            (capture["thread_id"], process_id, process_id),
        ):
            label = row["nvtx_label"]
            if label not in labels:
                continue
            scope_id = label
            if payload_type is not None:
                scope_id = row["nvtx_id"]
                if type(scope_id) is not int:
                    continue
                # SQLite stores the unsigned payload's bits in a signed INTEGER.
                if payload_type == "uint64" and scope_id < 0:
                    scope_id += 1 << 64
            if scope_id not in scope_ids:
                continue
            if scope_id in ranges:
                raise ValueError(f"Ambiguous NVTX range for {label}")
            if (
                row["end"] is None
                or row["end"] < row["start"]
                or row["endGlobalTid"] not in (None, row["globalTid"])
            ):
                raise ValueError(f"Incomplete or cross-thread NVTX range for {label}")
            ranges[scope_id] = row
    return ranges


def _join_nsight_projection(
    report: dict[str, Any], *, database: Path, projection: Path
) -> tuple[list[ScopeTiming], dict[str, dict[str, Any]]]:
    """Join signed/legacy NVTX identities to original native scope boundaries."""
    capture = report["device_analysis"]
    recorded = capture["scopes"]
    if len({scope["trace_label"] for scope in recorded}) != len(recorded):
        raise ValueError("Duplicate metadata scope identity")
    # Python NVTX uses signed payloads. C sidecars have unsigned IDs and no type.
    # The oldest captures identify ranges by unique labels.
    payload_type = capture.get("nvtx_payload_type")
    has_id = ["nvtx_id" in scope for scope in recorded]
    if payload_type is None and any(has_id):
        payload_type = "uint64"
    if payload_type not in (None, "int64", "uint64"):
        raise ValueError(f"Unsupported NVTX payload type: {payload_type!r}")
    identity_field = "nvtx_label"
    if payload_type is not None:
        limit = 1 << (63 if payload_type == "int64" else 64)
        if any(
            type(scope.get("nvtx_id")) is not int or not 0 <= scope["nvtx_id"] < limit
            for scope in recorded
        ):
            raise ValueError(f"Expected exact {payload_type} NVTX scope IDs")
        identity_field = "nvtx_id"
    scopes = {scope[identity_field]: scope for scope in recorded}
    if len(scopes) != len(recorded):
        raise ValueError("NVTX identities must uniquely identify calls in this session")
    projected = _read_projection(projection)
    ranges = _read_nvtx_ranges(
        database, capture=capture, scope_ids=set(scopes), payload_type=payload_type
    )
    timings = []
    evidence = {}
    ops = {op["trace_label"]: op for op in report["ops"]}
    for scope_id, scope in scopes.items():
        trace_label, label = scope["trace_label"], scope["nvtx_label"]
        if scope_id not in ranges:
            detail = (
                " (expected a closed same-thread TokenSpeed push/pop range)"
                if payload_type == "int64"
                else ""
            )
            raise ValueError(f"NVTX range missing from Nsight capture: {label}{detail}")
        native = ranges[scope_id]
        if native["nvtx_label"] != label:
            raise ValueError("NVTX payload identity has an unexpected label")
        start, end = native["start"], native["end"]
        gpu = projected.get(native["range_id"])
        duration = 0
        gpu_start = None
        count = 0
        reason = "no GPU activity correlated by Nsight to this NVTX range"
        if gpu is not None:
            projection_name = gpu["Name"]
            name_matches = (
                projection_name == f"TokenSpeed:{label}"
                if payload_type == "int64"
                else projection_name.split(":", 1)[-1] == label
            )
            if (
                int(gpu["Orig Start (ns)"]) != start
                or int(gpu["Orig Duration (ns)"]) != end - start
                or int(gpu["PID"]) != native["pid"]
                or int(gpu["TID"]) != capture["thread_id"]
                or not name_matches
            ):
                raise ValueError("Nsight projection and NVTX capture do not match")
            duration = int(gpu["Projected Duration (ns)"])
            gpu_start = int(gpu["Projected Start (ns)"])
            count = int(gpu["NumGPUOps"])
            if duration <= 0 or count <= 0 or gpu_start < 0:
                raise ValueError("Invalid Nsight GPU projection")
            reason = None
            if trace_label in ops and "device" in capture:
                op = ops[trace_label]
                devices = {
                    tensor["device"]
                    for tensor in [*op["inputs"].values(), *op["outputs"].values()]
                    if tensor["device"].startswith("cuda:")
                }
                if devices != {capture["device"]}:
                    reason = "op tensor metadata does not establish work on the declared CUDA device"
        timings.append(ScopeTiming(trace_label, start, end, duration, reason))
        evidence[trace_label] = {
            "native_nvtx_label": label,
            "nsight_range_id": native["range_id"],
            "cpu_start_ns": start,
            "cpu_end_ns": end,
            "gpu_start_ns": gpu_start,
            "gpu_end_ns": None if gpu_start is None else gpu_start + duration,
            "gpu_activity_count": count,
        }
    return timings, evidence


def analyze_nsight(
    report: dict[str, Any], *, database: Path, projection: Path
) -> dict[str, Any]:
    """Join cost metadata to the official Nsight NVTX GPU projection export.

    The SQLite database and nvtx_gpu_proj_trace CSV must come from the same
    capture. CPU range timestamps identify nesting only. The denominator is
    Projected Duration: last associated GPU activity end minus first start,
    including gaps between activities. Missing GPU work stays unavailable.
    No Event, CPU duration or previously calculated metric is a fallback.
    """
    capture = report["device_analysis"]
    timings, evidence = _join_nsight_projection(
        report, database=database, projection=projection
    )
    rows = analyze_utilization(
        report, timings, hardware_for_analysis(report["hardware"], None)
    )
    for row in rows:
        row.update(evidence[row["trace_label"]])
    return {
        **report,
        "device_analysis": {
            **capture,
            "timing_kind": "nsight_gpu_projection",
            "timing_source": "Nsight Systems nvtx_gpu_proj_trace",
            "timing_definition": "last GPU activity end minus first GPU activity start",
            "timing_includes": ["associated GPU activities", "gaps between activities"],
            "timestamp_unit": "ns",
            "scopes": rows,
        },
    }


def finalize_nsight(
    *,
    nsys: str,
    trace: Path,
    metadata: Path,
    output: Path,
    hardware: Path | None = None,
) -> dict[str, Any]:
    """Export a finished nsys-rep and write metrics without modifying the capture.

    Intermediate SQLite/CSV exports live in a temporary directory. Output is a
    new metrics JSON; its native_nvtx_label links every row to the original
    timeline. Computed display labels do not rewrite the original NVTX events.
    """
    if output.resolve() in (metadata.resolve(), trace.resolve()):
        raise ValueError("Output must not overwrite metadata or the Nsight capture")
    recorded = json.loads(metadata.read_text())
    recorded["hardware"] = asdict(hardware_for_analysis(recorded["hardware"], hardware))
    with tempfile.TemporaryDirectory(prefix="tokenspeed-nsight-") as directory:
        root = Path(directory)
        database = root / "trace.sqlite"
        subprocess.run(
            [
                nsys,
                "export",
                "--type=sqlite",
                f"--output={database}",
                str(trace.resolve()),
            ],
            check=True,
        )
        subprocess.run(
            [
                nsys,
                "stats",
                "--report=nvtx_gpu_proj_trace",
                "--format=csv",
                f"--output={root / 'projection'}",
                str(database),
            ],
            check=True,
        )
        result = analyze_nsight(
            recorded,
            database=database,
            projection=root / "projection_nvtx_gpu_proj_trace.csv",
        )
    result["device_analysis"]["source_report"] = str(trace.resolve())
    with trace.open("rb") as source:
        fingerprint = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    result["device_analysis"]["source_report_sha256"] = fingerprint.hexdigest()
    result["device_analysis"]["source_metadata"] = str(metadata.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _load_cost_provider(
    selection: str, *, hbm_policy: str
) -> tuple[CostRegistry | None, dict]:
    """Load only an explicitly selected installed CLI registration function."""
    from tokenspeed_kernel import profile_costs
    from tokenspeed_kernel.profile_graph_analysis import (
        default_cost_registry,
        file_source,
    )

    provenance = {
        "selection": selection,
        "hbm_policy": hbm_policy,
        "builtin_source": file_source(Path(profile_costs.__file__)),
    }
    if selection == "builtin":
        return None, provenance
    module_name, separator, function_name = selection.partition(":")
    if (
        not separator
        or not all(part.isidentifier() for part in module_name.split("."))
        or not function_name.isidentifier()
    ):
        raise ValueError("Cost provider must be builtin or installed module:function")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(
            f"Cost provider {selection!r} is unavailable; install the provider and its CPU dependencies"
        ) from error
    callback = getattr(module, function_name, None)
    if not inspect.isfunction(callback):
        raise ValueError(
            f"Cost provider {selection!r} is not a callable registration function"
        )
    source = inspect.getsourcefile(module) or inspect.getfile(module)
    provenance["source"] = file_source(Path(source))
    provenance["callable"] = f"{callback.__module__}:{callback.__qualname__}"
    registry = default_cost_registry(hbm_policy=hbm_policy)
    try:
        callback(registry, hbm_policy=hbm_policy)
    except ImportError as error:
        raise ValueError(
            f"Cost provider {selection!r} could not register; install its CPU dependencies"
        ) from error
    return registry, provenance


def finalize_graph_nsight(
    *,
    nsys: str,
    trace: Path,
    replays: Path,
    window: Path,
    output: Path,
    hbm_policy: str,
    hardware: Path | None,
    service_metadata: Path | None,
    model_config: Path | None,
    coverage_output: Path | None,
    scheduler_mode: str,
    registry: CostRegistry | None,
    cost_provider_source: dict | None = None,
    sealed_communication_artifacts: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Export and analyze collector graph artifacts using public CPU cost rules.

    Optional service proof requires all three service inputs. Output files are
    exclusive new artifacts; their provenance retains hashes of the temporary
    SQLite export and every consumed capture/binary source. Explicit
    scheduler_mode="collector" adds scheduler_analysis from the same export and
    validated per-step service status to native scope labels; "disabled" omits it.
    registry is an explicitly trusted Python CostRegistry; None selects builtin
    public costs only. cost_provider_source optionally records caller-supplied
    provenance; no provider is ever loaded from captured/serialized metadata.
    """
    from tokenspeed_kernel.profile_graph_analysis import (
        analyze_graph_replays,
        default_cost_registry,
        file_source,
        native_coverage_receipt,
    )

    if scheduler_mode not in ("disabled", "collector"):
        raise ValueError("Select disabled or collector scheduler mode explicitly")
    if registry is not None and not isinstance(registry, CostRegistry):
        raise TypeError("registry must be a trusted CostRegistry or explicit None")
    if cost_provider_source is None:
        if registry is None:
            _, cost_provider_source = _load_cost_provider(
                "builtin", hbm_policy=hbm_policy
            )
        else:
            cost_provider_source = {
                "selection": "trusted_python_registry",
                "source": "explicit CostRegistry supplied by Python caller",
                "hbm_policy": hbm_policy,
            }
    if scheduler_mode == "collector" and service_metadata is None:
        raise ValueError("Collector scheduler analysis requires service metadata")
    proof = (service_metadata, model_config, coverage_output)
    if any(item is not None for item in proof) and any(item is None for item in proof):
        raise ValueError(
            "Service proof requires metadata, model config and coverage output"
        )
    outputs = [output] + ([] if coverage_output is None else [coverage_output])
    if len({path.resolve() for path in outputs}) != len(outputs) or any(
        path.exists() for path in outputs
    ):
        raise ValueError("Graph analysis output files must be distinct and new")
    selected_registry = (
        default_cost_registry(hbm_policy=hbm_policy) if registry is None else registry
    )
    source_trace_hash = file_source(trace)["sha256"]
    with tempfile.TemporaryDirectory(prefix="tokenspeed-graph-nsight-") as directory:
        database = Path(directory) / "trace.sqlite"
        subprocess.run(
            [
                nsys,
                "export",
                "--type=sqlite",
                f"--output={database}",
                str(trace.resolve()),
            ],
            check=True,
        )
        eager_projection = None
        if any(
            json.loads(line).get("eager_sidecar") is not None
            for line in replays.read_text().splitlines()
            if line.strip()
        ):
            projection = database.with_name("projection")
            subprocess.run(
                [
                    nsys,
                    "stats",
                    "--report=nvtx_gpu_proj_trace",
                    "--format=csv",
                    f"--output={projection}",
                    str(database),
                ],
                check=True,
            )
            eager_projection = projection.with_name(
                "projection_nvtx_gpu_proj_trace.csv"
            )
        result = analyze_graph_replays(
            database=database,
            report=trace,
            replays=replays,
            window=window,
            registry=selected_registry,
            hardware=hardware,
            evidence_kind="nsight_capture",
            eager_projection=eager_projection,
            sealed_communication_artifacts=sealed_communication_artifacts,
        )
        result["provenance"]["cost_provider"] = cost_provider_source
        result["provenance"]["database"][
            "origin"
        ] = "matching_nsys_export_from_source_report"
        result["provenance"]["database"]["export_command"] = [
            nsys,
            "export",
            "--type=sqlite",
        ]
        if result["device_analysis"]["source_report_sha256"] != source_trace_hash:
            raise ValueError("Source report changed during export/analysis")
        if scheduler_mode == "collector":
            from tokenspeed_kernel.profile_scheduler import analyze_scheduler

            scheduler = analyze_scheduler(
                database,
                selection_mode="collector",
                missing_launch_policy="report_unavailable",
                service_metadata=service_metadata,
                report=trace,
                expected_report_sha256=source_trace_hash,
                registry=selected_registry,
                eager_steps={
                    step["step_id"]: step
                    for step in result["steps"]
                    if step.get("attribution_mode") == "eager_nvtx_gpu_projection"
                },
            )
            for name, path in (("ready_index", replays), ("window", window)):
                if (
                    scheduler["provenance"][name]["sha256"]
                    != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        f"Scheduler service metadata references a different {name}"
                    )
            scheduler["provenance"]["database"] = dict(result["provenance"]["database"])
            scheduler["provenance"]["cost_provider"] = cost_provider_source
            result["scheduler_analysis"] = scheduler
            statuses = {step["step_id"]: step for step in scheduler["steps"]}
            for scope in result["device_analysis"]["scopes"]:
                step = statuses.get(scope["step_id"])
                if step is None or step["forward_id"] != scope["forward_id"]:
                    raise ValueError("Graph/scheduler exact step/ForwardId mismatch")
                scope["scheduler_status"] = {
                    name: step[name]
                    for name in (
                        "scheduled",
                        "gpu_observed",
                        "device_ready",
                        "authoritative_commit",
                        "completion_status",
                        "token_commits",
                    )
                }
        coverage = (
            None
            if service_metadata is None
            else native_coverage_receipt(
                result, metadata=service_metadata, model_config=model_config
            )
        )
    for path, content in [(output, result)] + (
        [] if coverage is None else [(coverage_output, coverage)]
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(content, indent=2, allow_nan=False) + "\n")
    return result


def read_run_manifest(path: Path, *, rank: int | None) -> dict[str, Any]:
    """Validate a sealed pair-driver inventory and select one exact rank.

    Every consumed source must be present in the hash inventory, including
    capture and binary paths referenced by replay rows. No serialized field
    selects a provider, policy or Python import.
    """
    if rank is not None and (type(rank) is not int or rank < 0):
        raise ValueError("Select a nonnegative integer rank")
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if not path.name.endswith(f".{fingerprint}.json"):
        raise ValueError("Run manifest filename must contain its content SHA256")
    manifest = json.loads(raw)
    run_id = manifest.get("run_id")
    if (
        manifest.get("schema") != "tokenspeed.profile_run/v1"
        or not isinstance(run_id, str)
        or not run_id
    ):
        raise ValueError("Unsupported run manifest schema/identity")
    artifacts = manifest["artifacts"]
    for name, source in artifacts.items():
        source_path = Path(name)
        if (
            not source_path.is_absolute()
            or source_path.resolve() != source_path
            or source.get("path") != name
        ):
            raise ValueError(
                "Run manifest sources must use exact resolved absolute paths"
            )
        digest = hashlib.sha256()
        size = 0
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if (
            type(source.get("bytes")) is not int
            or source["bytes"] != size
            or source.get("sha256") != digest.hexdigest()
        ):
            raise ValueError(f"Run artifact hash/size mismatch: {name}")

    def source(name: str) -> Path:
        if name not in artifacts:
            raise ValueError(f"Run source absent from hash inventory: {name}")
        return Path(name)

    sealed_communication = None
    if "communication_evidence" in manifest:
        from tokenspeed_kernel.profile_communication_inventory import (
            communication_inventory,
        )

        directories = {source(item["config"]).parent for item in manifest["ranks"]}
        names = manifest["communication_evidence"]
        if not isinstance(names, list) or not names:
            raise ValueError("Invalid sealed communication inventory")
        paths = [source(name) for name in names]
        if any(item.parent not in directories for item in paths):
            raise ValueError("Communication receipt outside sealed rank directory")
        observed = communication_inventory(paths, run_id=run_id)
        if any(artifacts[name] != item for name, item in observed.items()):
            raise ValueError("Sealed communication receipt fingerprint mismatch")
        sealed_communication = {
            **observed,
            **{item["config"]: artifacts[item["config"]] for item in manifest["ranks"]},
        }

    if manifest["report"] is None:
        raise ValueError("JSON-only run has no completed Nsight report")
    report = source(manifest["report"])
    client = source(manifest["client"])
    with client.open() as stream:
        header = json.loads(next(stream))
    case = source(manifest["case"])
    if (
        header.get("run_id"),
        header.get("arm"),
        header.get("repetition"),
        header.get("case_sha256"),
    ) != (run_id, "profiled", manifest["arm_index"], artifacts[str(case)]["sha256"]):
        raise ValueError("Run manifest/client/case identity mismatch")
    observation = None
    if manifest.get("startup_observation_sha256") is not None:
        from tokenspeed_kernel.op_profile import (
            _profile_observed_configs,
            _profile_validate_startup_observation,
        )

        timeline_path = source(manifest["timeline"])
        timeline = json.loads(timeline_path.read_text())
        observation = timeline.pop("startup_observation")
        body = _profile_validate_startup_observation(
            observation, run_id=run_id, timeline_config=timeline
        )
        driver = source(body["driver_source"]["path"])
        if (
            body["case_sha256"] != artifacts[str(case)]["sha256"]
            or body["repetition"] != manifest["arm_index"]
            or artifacts[str(driver)] != body["driver_source"]
            or observation["sha256"] != manifest["startup_observation_sha256"]
            or header.get("startup_observation_sha256") != observation["sha256"]
        ):
            raise ValueError(
                "Sealed startup observation case/arm/source identity mismatch"
            )
    metadata_path = source(manifest["metadata"])
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("schema") != "tokenspeed.metadata_coverage/v1"
        or metadata.get("run_id") != run_id
        or type(metadata.get("rank", 0)) is not int
        or metadata.get("rank", 0) != 0
    ):
        raise ValueError("Run manifest/service metadata identity mismatch")
    for item in metadata.get("service_sources", []):
        service = source(str(Path(item["path"]).resolve()))
        if item.get("sha256") != artifacts[str(service)]["sha256"]:
            raise ValueError("Service metadata source hash mismatch")
        for line in service.read_text().splitlines():
            if json.loads(line).get("run_id") != run_id:
                raise ValueError("Cross-run service source")
    model_config = (
        source(manifest["model_config"])
        if manifest["model_config"] is not None
        else None
    )
    ranks = manifest["ranks"]
    rank_ids = [item["rank"] for item in ranks]
    if (
        not ranks
        or any(type(value) is not int or value < 0 for value in rank_ids)
        or len(set(rank_ids)) != len(rank_ids)
    ):
        raise ValueError("Run manifest requires distinct nonnegative collector ranks")
    if rank is None:
        if len(ranks) != 1:
            raise ValueError(
                f"Run manifest has multiple ranks {rank_ids}; select --rank explicitly"
            )
        rank = rank_ids[0]
    if rank not in rank_ids:
        raise ValueError(f"Rank {rank} is absent from the run manifest")
    selected = None
    for item in ranks:
        identity = (run_id, item["rank"])
        window = source(item["window"])
        replays = source(item["replays"])
        recorded = json.loads(window.read_text())
        if (
            recorded.get("schema") != "tokenspeed.profile_window/v1"
            or type(recorded.get("rank")) is not int
            or (recorded.get("run_id"), recorded.get("rank")) != identity
        ):
            raise ValueError("Cross-run/rank manifest window")
        semantic = recorded.get("loaded_library_manifest", {}).get(
            "semantic_provenance", {}
        )
        if observation is not None:
            config = json.loads(source(item["config"]).read_text())
            if config["config"] not in _profile_observed_configs(
                observation, run_id=run_id
            ):
                raise ValueError(
                    "Actual expanded runtime config differs from startup observation"
                )
            config_sha = hashlib.sha256(
                json.dumps(
                    config["config"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if (
                (config.get("run_id"), config.get("rank")) != identity
                or config.get("sha256") != config_sha
                or config["config"].get("startup_observation") != observation
            ):
                raise ValueError("Sealed startup observation/runtime config mismatch")
            for snapshot in (semantic, semantic.get("baseline", {})):
                context = snapshot.get("source_context", {})
                if (
                    context.get("startup_observation") != observation
                    or context.get("profile_config_sha256") != config_sha
                    or (context.get("run_id"), context.get("rank")) != identity
                ):
                    raise ValueError(
                        "Sealed startup observation/window identity mismatch"
                    )
        replay_rows = [json.loads(line) for line in replays.read_text().splitlines()]
        eager_rows = [
            row for row in replay_rows if row.get("eager_sidecar") is not None
        ]
        if eager_rows:
            config = json.loads(source(item["config"]).read_text())
            config_sha = hashlib.sha256(
                json.dumps(
                    config["config"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if (
                config.get("schema") != "tokenspeed.profile_config/v1"
                or (config.get("run_id"), config.get("rank")) != identity
                or config.get("sha256") != config_sha
                or recorded.get("profile_config_sha256") != config_sha
            ):
                raise ValueError("Eager sidecar/window/rank config mismatch")
        for row in replay_rows:
            if (
                type(row.get("rank")) is not int
                or (row.get("run_id"), row.get("rank")) != identity
            ):
                raise ValueError("Cross-run/rank manifest replay")
            if row.get("capture_file"):
                capture_path = source(str(Path(row["capture_file"]).resolve()))
                capture = json.loads(capture_path.read_text())
                if any(
                    capture.get(key) != row.get(key)
                    for key in ("run_id", "rank", "capture_id", "process_id")
                ):
                    raise ValueError("Manifest capture/replay identity mismatch")
                if observation is not None and (
                    capture.get("static_metadata", {}).get("startup_observation_sha256")
                    != observation["sha256"]
                    or capture.get("static_metadata", {}).get("profile_config_sha256")
                    != semantic.get("source_context", {}).get("profile_config_sha256")
                ):
                    raise ValueError(
                        "Sealed startup observation/capture config mismatch"
                    )
            if row.get("eager_sidecar") is not None:
                from tokenspeed_kernel.profile_eager import read_eager_sidecar

                eager = row["eager_sidecar"]
                eager_path = source(eager["path"])
                if eager != artifacts[str(eager_path)]:
                    raise ValueError("Manifest/replay eager fingerprint mismatch")
                read_eager_sidecar(row, config_sha256=config_sha)
            if row.get("binary_payload"):
                payload = row["binary_payload"]
                payload_path = source(str(Path(payload["path"]).resolve()))
                if any(
                    payload.get(key) != artifacts[str(payload_path)][key]
                    for key in ("sha256", "bytes")
                ):
                    raise ValueError("Manifest/replay binary fingerprint mismatch")
        if item["rank"] == rank:
            selected = {
                "report": report,
                "graph_replays": replays,
                "graph_window": window,
                **(
                    {"sealed_communication_artifacts": sealed_communication}
                    if sealed_communication is not None
                    else {}
                ),
            }
    # The pair driver's fixed service proof is rank 0. Other ranks remain valid
    # graph analyses, but cannot borrow rank 0's scheduler/coverage evidence.
    if rank == 0:
        if model_config is None:
            raise ValueError("Rank 0 service proof requires the frozen model config")
        for field, value in (
            ("ready_index_source", selected["graph_replays"]),
            ("window_manifest_source", selected["graph_window"]),
            ("model_config_source", model_config),
        ):
            recorded = metadata.get(field)
            if (
                recorded is None
                or Path(recorded["path"]).resolve() != value
                or recorded.get("sha256") != artifacts[str(value)]["sha256"]
            ):
                raise ValueError(f"Manifest/service metadata source mismatch: {field}")
        selected.update(
            metadata=metadata_path,
            model_config=model_config,
            native_coverage=selected["graph_window"].parent
            / "rank-0.native-coverage.json",
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate utilization from Nsight GPU timestamps"
    )
    parser.add_argument(
        "--nsys", default="nsys", help="Nsight CLI matching the capture version"
    )
    parser.add_argument(
        "--report", type=Path, help="Completed .nsys-rep (or use --run-manifest)"
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Hash-addressed manifest emitted by service_tpot pair",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="Exact rank from --run-manifest; required when ambiguous",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Eager cost metadata, or service metadata for graph proof",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New Nsight-timed metrics JSON"
    )
    parser.add_argument(
        "--hardware", type=Path, help="Verify the captured hardware selection"
    )
    parser.add_argument(
        "--graph-replays",
        type=Path,
        help="Collector rank-N.replays.jsonl; selects graph analysis",
    )
    parser.add_argument(
        "--graph-window", type=Path, help="Final collector rank-N.window.json"
    )
    parser.add_argument(
        "--hbm-policy",
        choices=("unavailable", "logical_io_once"),
        help="Explicit graph cost estimate policy",
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=("disabled", "collector"),
        help="Explicit scheduler selection for graph analysis; collector uses --metadata sources",
    )
    parser.add_argument(
        "--cost-provider",
        help="Explicit graph cost provider: builtin or installed module:function; executes only this user-selected registration function",
    )
    parser.add_argument(
        "--model-config", type=Path, help="Frozen model config for service proof"
    )
    parser.add_argument(
        "--native-coverage", type=Path, help="New service native-coverage receipt"
    )
    parser.add_argument(
        "--annotated-report",
        type=Path,
        help="New native report with utilization labels",
    )
    parser.add_argument(
        "--import-nvtxt", type=Path, help="Matching Nsight ImportNvtxt executable"
    )
    parser.add_argument(
        "--target-id", help="Actual Hw:Vm ID from ImportNvtxt report info"
    )
    args = parser.parse_args()
    sealed_communication = None
    if args.run_manifest is not None:
        if any(
            value is not None
            for value in (
                args.report,
                args.graph_replays,
                args.graph_window,
                args.metadata,
                args.model_config,
            )
        ):
            parser.error(
                "--run-manifest supplies report/replays/window/metadata/model-config; do not mix source options"
            )
        try:
            sources = read_run_manifest(args.run_manifest, rank=args.rank)
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
            parser.error(str(error))
        args.report = sources["report"]
        args.graph_replays = sources["graph_replays"]
        args.graph_window = sources["graph_window"]
        args.metadata = sources.get("metadata")
        args.model_config = sources.get("model_config")
        sealed_communication = sources.get("sealed_communication_artifacts")
        if args.native_coverage is None:
            args.native_coverage = sources.get("native_coverage")
    elif args.rank is not None:
        parser.error("--rank requires --run-manifest")
    if args.report is None:
        parser.error("Supply --report or --run-manifest")
    annotation = (args.annotated_report, args.import_nvtxt, args.target_id)
    if any(item is not None for item in annotation) and any(
        item is None for item in annotation
    ):
        parser.error(
            "--annotated-report, --import-nvtxt and --target-id must be supplied together"
        )
    if args.graph_replays is not None:
        if (
            args.graph_window is None
            or args.hbm_policy is None
            or args.scheduler_mode is None
            or args.cost_provider is None
        ):
            parser.error(
                "Graph mode requires --graph-window, --hbm-policy, --scheduler-mode and --cost-provider"
            )
        try:
            registry, provider_source = _load_cost_provider(
                args.cost_provider, hbm_policy=args.hbm_policy
            )
        except ValueError as error:
            parser.error(str(error))
        finalize_graph_nsight(
            nsys=args.nsys,
            trace=args.report,
            replays=args.graph_replays,
            window=args.graph_window,
            output=args.output,
            hbm_policy=args.hbm_policy,
            hardware=args.hardware,
            service_metadata=args.metadata,
            model_config=args.model_config,
            coverage_output=args.native_coverage,
            scheduler_mode=args.scheduler_mode,
            registry=registry,
            cost_provider_source=provider_source,
            sealed_communication_artifacts=sealed_communication,
        )
    else:
        if args.metadata is None:
            parser.error("Eager analysis requires --metadata")
        if any(
            value is not None
            for value in (
                args.graph_window,
                args.hbm_policy,
                args.scheduler_mode,
                args.cost_provider,
                args.model_config,
                args.native_coverage,
            )
        ):
            parser.error("Graph options require --graph-replays")
        finalize_nsight(
            nsys=args.nsys,
            trace=args.report,
            metadata=args.metadata,
            output=args.output,
            hardware=args.hardware,
        )
    if args.annotated_report is not None:
        from tokenspeed_kernel.profile_native_report import annotate_report

        annotate_report(
            report=args.report,
            metrics=args.output,
            output=args.annotated_report,
            importer=args.import_nvtxt,
            target_id=args.target_id,
            nsys=args.nsys,
        )


if __name__ == "__main__":
    main()
