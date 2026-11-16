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

"""Offline current-replay costs and exact Nsight CUDA graph provenance.

This module never imports model execution or initializes CUDA. Serialized rule
names are looked up only in an explicitly constructed CostRegistry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tokenspeed_kernel.op_profile import CostRegistry
from tokenspeed_kernel.profile_communication import bind_capture_communication
from tokenspeed_kernel.profile_costs import (
    register_collective_costs,
    register_compute_costs,
)
from tokenspeed_kernel.profile_eager import analyze_eager_step, merge_forward_binding
from tokenspeed_kernel.profile_native_activity import (
    activity_resource_columns,
    decode_activity_resources,
)
from tokenspeed_kernel.profile_utilization import hardware_for_analysis, scope_metrics


def file_source(path: Path) -> dict[str, str]:
    """Return the resolved source path and a streaming SHA256 fingerprint."""
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def default_cost_registry(*, hbm_policy: str) -> CostRegistry:
    """Build public CPU-safe rules; callers register trusted model rules explicitly."""
    registry = CostRegistry()
    register_compute_costs(registry, hbm_policy=hbm_policy)
    register_collective_costs(registry, hbm_policy=hbm_policy)
    return registry


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def decode_replay(capture: dict, row: dict, *, payload_path: Path) -> dict:
    """Validate one owned lease and decode its exact binary integer snapshots.

    Byte counts, SHA256, offset alignment/bounds, nonoverlap, integer dtype and
    declared tensor element counts must all agree. Capture-time tensor values
    are cleared before loading this replay's values, including missing snapshots.
    Returns a fresh report with real OpRecord-compatible metadata.
    """
    for name in ("run_id", "rank", "capture_id", "process_id"):
        if capture.get(name) != row.get(name):
            raise ValueError(f"Capture/replay {name} mismatch")
    if not _integer(row.get("serving_step")):
        raise ValueError("Invalid serving step identity")
    step = row["step_id"]
    if step != f"{row['run_id']}:r{row['rank']}:s{row['serving_step']}":
        raise ValueError("Step identity mismatch")
    lease = row.get("metadata_lease")
    if (
        not isinstance(lease, dict)
        or lease.get("step_id") != step
        or not (
            _integer(lease.get("slot")) and _integer(lease.get("generation"), minimum=1)
        )
    ):
        raise ValueError("Missing or mismatched metadata lease generation")
    payload = row["binary_payload"]
    raw = payload_path.read_bytes()
    if (
        not _integer(payload.get("bytes"))
        or len(raw) != payload["bytes"]
        or len(raw) != capture["integer_metadata_bytes"]
        or hashlib.sha256(raw).hexdigest() != payload.get("sha256")
    ):
        raise ValueError("Replay payload size/SHA256 mismatch")
    report = copy.deepcopy(capture)
    ops = {op["trace_label"]: op for op in report["ops"]}
    if len(ops) != len(report["ops"]):
        raise ValueError("Duplicate op identity")
    for op in ops.values():
        op["tensor_values"] = {name: None for name in op.get("tensor_values", {})}
    occupied, identities = [], set()
    for item in capture["integer_snapshots"]:
        label, name = item["trace_label"], item["name"]
        if label not in ops or (label, name) in identities:
            raise ValueError("Unknown or duplicate integer snapshot identity")
        identities.add((label, name))
        op = ops[label]
        tensor = op["inputs"].get(name)
        dtype = item["dtype"]
        if dtype not in {"torch.int32", "torch.int64"} or tensor is None:
            raise ValueError("Unsupported integer snapshot dtype or tensor")
        size, code = (4, "i") if dtype == "torch.int32" else (8, "q")
        count, offset = item["count"], item["offset"]
        if not _integer(count, minimum=1) or count > 4096 or not _integer(offset):
            raise ValueError("Invalid integer snapshot count/offset")
        shape = tensor["shape"]
        elements = 1
        for dim in shape:
            if not _integer(dim, minimum=1):
                raise ValueError("Invalid integer tensor shape")
            elements *= dim
        end = offset + count * size
        if (
            tensor["dtype"] != dtype
            or tensor["logical_bytes"] != count * size
            or elements != count
            or offset % size
            or end > len(raw)
            or any(offset < stop and start < end for start, stop in occupied)
        ):
            raise ValueError(
                "Integer snapshot alignment, bounds, shape or overlap mismatch"
            )
        occupied.append((offset, end))
        op["tensor_values"][name] = tuple(
            struct.unpack_from(f"<{count}{code}", raw, offset)
        )
    return report


def _reevaluate(
    report: dict,
    registry: CostRegistry,
    *,
    engine_context: dict[str, dict] | None = None,
) -> dict:
    return registry.analyze_recorded(report, engine_context=engine_context)


def _union(intervals: list[tuple[int, int]]) -> int:
    total, end = 0, -1
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def _scopes(capture: dict) -> tuple[dict, dict, dict]:
    scopes = {item["scope_id"]: item for item in capture["graph_scopes"]}
    if len(scopes) != len(capture["graph_scopes"]):
        raise ValueError("Duplicate graph scope ID")
    records = {item["trace_label"]: ("op", item) for item in capture["ops"]}
    records.update({item["trace_label"]: ("layer", item) for item in capture["layers"]})
    labels = {item["trace_label"] for item in scopes.values()}
    if (
        len(labels) != len(scopes)
        or labels != set(records)
        or len(records) != len(capture["ops"]) + len(capture["layers"])
    ):
        raise ValueError("Every graph scope needs exactly one metadata record")
    ancestors = {}
    for identity, scope in scopes.items():
        chain, current = set(), identity
        while current is not None:
            if current in chain or current not in scopes:
                raise ValueError("Cycle or missing parent in graph scope ancestry")
            chain.add(current)
            current = scopes[current]["parent_scope_id"]
        ancestors[identity] = chain
        if any(not _integer(node, minimum=1) for node in scope["graph_node_ids"]):
            raise ValueError("Invalid capture graph node ID")
    return scopes, records, ancestors


def _capture_node_requirements(
    capture: dict, scopes: dict, observer: set
) -> tuple[dict, dict]:
    """Index static types and scope node sets once for an immutable capture.

    CUDA 12.9.1: Empty=5, WaitEvent=6, EventRecord=7 require no independent
    activity. Every other or unavailable type remains activity-required.
    https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__TYPES.html
    """
    containing = {}
    for scope in scopes.values():
        for node in scope["graph_node_ids"]:
            containing.setdefault(node, set()).add(scope["graph_id"])
    graphs = {scope["graph_id"] for scope in scopes.values()}
    evidence, dependency, tables = {}, set(), {}
    activity_tables = {
        0: "CUPTI_ACTIVITY_KIND_KERNEL",
        1: "CUPTI_ACTIVITY_KIND_MEMCPY",
        2: "CUPTI_ACTIVITY_KIND_MEMSET",
    }
    for item in capture.get("graph_node_types", []):
        node, graph, kind = (
            item.get("graph_node_id"),
            item.get("graph_id"),
            item.get("cuda_graph_node_type"),
        )
        if (
            not _integer(node, minimum=1)
            or node in evidence
            or not _integer(graph)
            or (kind is not None and not _integer(kind))
        ):
            raise ValueError("Invalid or duplicate capture graph node type identity")
        if graph not in containing.get(node, graphs if node in observer else set()):
            raise ValueError("Graph node type belongs to another capture graph")
        evidence[node] = item
        if capture.get("graph_node_type_api") != "cudaGraphNodeGetType" or item.get(
            "unavailable_reason"
        ):
            continue
        if kind in {5, 6, 7}:
            dependency.add(node)
        elif kind in activity_tables:
            tables[node] = activity_tables[kind]
    requirements = {}
    for identity, scope in scopes.items():
        owned = set(scope["graph_node_ids"]) - observer
        requirements[identity] = dict(
            owned=owned,
            dependency=owned & dependency,
            activity_required=owned - dependency,
        )
    return requirements, dict(dependency=dependency, activity_tables=tables)


def _node_activity_conflicts(index: dict, activities: list) -> set:
    return {
        item["source"]
        for item in activities
        if item["source"] in index["dependency"]
        or (
            item["source"] in index["activity_tables"]
            and item["table"] != index["activity_tables"][item["source"]]
        )
    }


@dataclass(frozen=True)
class _NativeRange:
    start: int
    end: int | None
    global_tid: int
    end_global_tid: int | None


@dataclass(frozen=True)
class _NativeLaunch:
    start: int | None
    end: int | None
    correlation: int | None


@dataclass(frozen=True)
class _NativeActivity:
    start: int
    end: int
    node: int | None
    table: str
    source_rowid: int | None = None
    kernel: dict | None = None
    memory: dict | None = None


@dataclass
class _NativeIndex:
    tables: set[str]
    ranges: dict[tuple[int, int, str], list[_NativeRange]]
    launches: dict[int, list[_NativeLaunch]]
    runtime_counts: dict[tuple[int, int], int]
    clones: dict[int, dict[int, int]]
    clone_conflicts: set[int]
    activities: dict[tuple[int, int], list[_NativeActivity]]
    source_nodes: dict[tuple[int, int], int]
    device_uuids: dict[int, dict]


def _read_native_index(db: sqlite3.Connection, replays: list[dict]) -> _NativeIndex:
    """Read each native table once; retain only recorded replay identities."""
    tables = {
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    index = _NativeIndex(tables, {}, {}, {}, {}, set(), {}, {}, {})
    # Nsight device IDs are not Torch CUDA_VISIBLE_DEVICES ordinals. Bind them
    # through the native device UUID table, retaining unknown/ambiguous mappings.
    for table in ("TARGET_INFO_GPU", "CUPTI_ACTIVITY_KIND_DEVICE"):
        if table not in tables:
            continue
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        identity = "id" if "id" in columns else "deviceId"
        if identity not in columns or "uuid" not in columns:
            continue
        major = "computeMajor" if "computeMajor" in columns else "smMajor"
        minor = "computeMinor" if "computeMinor" in columns else "smMinor"
        capability = (
            f",{major},{minor}"
            if major in columns and minor in columns
            else ",NULL,NULL"
        )
        for device_id, raw_uuid, compute_major, compute_minor in db.execute(
            f"SELECT {identity},uuid{capability} FROM {table}"
        ):
            if isinstance(raw_uuid, bytes):
                value = raw_uuid.hex()
            elif isinstance(raw_uuid, str):
                value = raw_uuid.lower().removeprefix("gpu-").replace("-", "")
            else:
                continue
            if len(value) != 32 or any(
                char not in "0123456789abcdef" for char in value
            ):
                continue
            previous = index.device_uuids.get(device_id)
            if previous is not None and previous["uuid"] != value:
                index.device_uuids[device_id] = {
                    "uuid": None,
                    "table": "conflicting native UUID mappings",
                }
            else:
                index.device_uuids[device_id] = {
                    "uuid": value,
                    "table": table,
                    "compute_capability": (
                        f"{compute_major}.{compute_minor}"
                        if compute_major is not None and compute_minor is not None
                        else previous.get("compute_capability") if previous else None
                    ),
                }
    required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME"}
    if required - tables:
        return index
    identities = {
        (
            row.get("process_id"),
            row.get("replay_thread_id"),
            row.get("replay_nvtx_label"),
        )
        for row in replays
        if _integer(row.get("process_id"), minimum=1)
        and _integer(row.get("replay_thread_id"), minimum=1)
        and isinstance(row.get("replay_nvtx_label"), str)
    }
    pids = sorted({identity[0] for identity in identities})
    if not pids:
        return index
    parameters = ",".join("?" for _ in pids)
    for row in db.execute(
        f"""SELECT n.start,n.end,n.globalTid,n.endGlobalTid,
        COALESCE(s.value,n.text) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE ((n.globalTid >> 24)&16777215) IN ({parameters})""",
        pids,
    ):
        start, end, global_tid, end_global_tid, label = row
        key = ((global_tid >> 24) & 16777215, global_tid & 16777215, label)
        if key in identities:
            index.ranges.setdefault(key, []).append(
                _NativeRange(start, end, global_tid, end_global_tid)
            )
    # Count every runtime row, including non-launch calls, to retain the strict
    # per-process correlation uniqueness check. SQL keeps the original API-name
    # predicate; no kernel name or GPU timestamp assigns an activity to an op.
    for row in db.execute(
        f"""SELECT r.start,r.end,r.globalTid,r.correlationId,
        (s.value='cudaGraphLaunch' OR s.value LIKE 'cudaGraphLaunch_v%'
        OR s.value LIKE 'cudaGraphLaunch_ptsz%') FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        LEFT JOIN StringIds s ON r.nameId=s.id
        WHERE ((r.globalTid >> 24)&16777215) IN ({parameters})""",
        pids,
    ):
        start, end, global_tid, correlation, is_launch = row
        pid = (global_tid >> 24) & 16777215
        if correlation is not None:
            key = (pid, correlation)
            index.runtime_counts[key] = index.runtime_counts.get(key, 0) + 1
        if is_launch:
            index.launches.setdefault(global_tid, []).append(
                _NativeLaunch(start, end, correlation)
            )
    correlations = set()
    for ranges in index.ranges.values():
        if len(ranges) != 1:
            continue
        native = ranges[0]
        if (
            native.end is None
            or native.end < native.start
            or native.end_global_tid not in (None, native.global_tid)
        ):
            continue
        launches = [
            launch
            for launch in index.launches.get(native.global_tid, [])
            if launch.start is not None
            and launch.end is not None
            and launch.start >= native.start
            and launch.end <= native.end
        ]
        if len(launches) == 1 and launches[0].correlation is not None:
            correlations.add(launches[0].correlation)
    if "CUDA_GRAPH_NODE_EVENTS" in tables:
        for node, original, global_tid in db.execute(
            f"""SELECT graphNodeId,originalGraphNodeId,globalTid
            FROM CUDA_GRAPH_NODE_EVENTS WHERE originalGraphNodeId IS NOT NULL
            AND originalGraphNodeId!=0 AND ((globalTid >> 24)&16777215) IN ({parameters})""",
            pids,
        ):
            pid = (global_tid >> 24) & 16777215
            clones = index.clones.setdefault(pid, {})
            if node in clones and clones[node] != original:
                index.clone_conflicts.add(pid)
            clones[node] = original
    if not correlations:
        return index
    correlation_parameters = ",".join("?" for _ in correlations)
    values = [*pids, *sorted(correlations)]
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        if table not in tables:
            continue
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        resource_columns = activity_resource_columns(table)
        selected = [
            name if name in columns else f"NULL AS {name}" for name in resource_columns
        ]
        extra = "," + ",".join(selected)
        strings = (
            dict(db.execute("SELECT id,value FROM StringIds"))
            if table == "CUPTI_ACTIVITY_KIND_KERNEL"
            else {}
        )
        for raw in db.execute(
            f"""SELECT rowid,start,end,graphNodeId,correlationId,globalPid{extra}
            FROM {table} WHERE ((globalPid >> 24)&16777215) IN ({parameters})
            AND correlationId IN ({correlation_parameters})""",
            values,
        ):
            source_rowid, start, end, node, correlation, global_pid = raw[:6]
            kernel, memory = decode_activity_resources(
                table, resource_columns, raw[6:], strings
            )
            key = ((global_pid >> 24) & 16777215, correlation)
            index.activities.setdefault(key, []).append(
                _NativeActivity(start, end, node, table, source_rowid, kernel, memory)
            )
    return index


def _join(
    index: _NativeIndex, capture: dict, replay: dict
) -> tuple[list, dict, list[str]]:
    required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME"}
    if required - index.tables:
        return [], {}, ["missing_native_sqlite_provenance_tables"]
    pid, tid = replay["process_id"], replay["replay_thread_id"]
    label = replay["replay_nvtx_label"]
    expected_label = f"tokenspeed.profile.replay | {replay['step_id']}"
    if (
        label != expected_label
        or not _integer(pid, minimum=1)
        or not _integer(tid, minimum=1)
    ):
        raise ValueError("Invalid replay label/process/thread identity")
    ranges = index.ranges.get((pid, tid, label), [])
    evidence = {
        "step_id": replay["step_id"],
        "forward_id": replay.get("forward_id"),
        "capture_id": replay["capture_id"],
        "replay_nvtx_label": label,
    }
    if len(ranges) != 1:
        return [], evidence, ["missing_or_ambiguous_exact_replay_nvtx"]
    native = ranges[0]
    if (
        native.end is None
        or native.end < native.start
        or native.end_global_tid not in (None, native.global_tid)
    ):
        return [], evidence, ["incomplete_or_cross_thread_replay_nvtx"]
    launches = [
        launch
        for launch in index.launches.get(native.global_tid, [])
        if launch.start is not None
        and launch.end is not None
        and launch.start >= native.start
        and launch.end <= native.end
    ]
    if len(launches) != 1:
        return [], evidence, ["missing_or_ambiguous_exact_cudaGraphLaunch"]
    correlation = launches[0].correlation
    evidence["launch_correlation_id"] = correlation
    if index.runtime_counts.get((pid, correlation), 0) != 1:
        return [], evidence, ["ambiguous_runtime_correlation_id"]
    if pid in index.clone_conflicts:
        raise ValueError("Ambiguous same-process graph clone relation")
    clones = index.clones.get(pid, {})

    def source(node: int) -> int:
        key = (pid, node)
        if key in index.source_nodes:
            return index.source_nodes[key]
        seen = set()
        original = node
        while node in clones:
            if node in seen:
                raise ValueError("Cycle in graph clone relation")
            seen.add(node)
            node = clones[node]
        index.source_nodes[(pid, original)] = node
        return node

    activities = []
    for activity in index.activities.get((pid, correlation), []):
        if (
            not _integer(activity.start)
            or not _integer(activity.end)
            or activity.end <= activity.start
        ):
            raise ValueError("Invalid correlated activity timestamps")
        activities.append(
            dict(
                start=activity.start,
                end=activity.end,
                node=activity.node,
                source=source(activity.node) if activity.node else None,
                table=activity.table,
                source_rowid=activity.source_rowid,
                kernel=activity.kernel,
                memory=activity.memory,
                process_id=pid,
            )
        )
    evidence["clone_relations"] = {str(key): value for key, value in clones.items()}
    evidence["correlated_activity_count"] = len(activities)
    return (
        activities,
        evidence,
        [] if activities else ["graph_launch_has_no_correlated_activities"],
    )


def _partial(metrics: dict, reason: str) -> None:
    for key in (
        "mfu",
        "mfu_total",
        "estimated_mbu",
        "algbw_bytes_per_second",
        "busbw_bytes_per_second",
        "estimated_comm_utilization",
        "estimated_comm_reference_efficiency",
        "estimated_comm_tx_payload_bytes_per_second",
        "estimated_comm_rx_payload_bytes_per_second",
        "estimated_comm_tx_utilization",
        "estimated_comm_rx_utilization",
    ):
        if (
            key in {"mfu", "mfu_total", "estimated_mbu"}
            and metrics.get(key) is not None
        ):
            metrics[f"{key}_known"] = metrics[key]
        metrics[key] = None
    for key in (
        "mfu_missing_reason",
        "mfu_total_missing_reason",
        "mbu_missing_reason",
        "busbw_missing_reason",
        "estimated_comm_utilization_missing_reason",
    ):
        metrics[key] = "; ".join(filter(None, (metrics.get(key), reason)))

    metrics["estimated_comm_utilization_warning"] = None
    metrics["estimated_comm_reference_efficiency_warning"] = None


def analyze_graph_replays(
    *,
    database: Path,
    report: Path,
    replays: Path,
    window: Path,
    registry: CostRegistry,
    hardware: Path | None,
    evidence_kind: str,
    eager_projection: Path | None = None,
    sealed_communication_artifacts: dict[str, dict] | None = None,
) -> dict:
    """Analyze actual ready collector records against a same-run SQLite export.

    Args:
        database: SQLite export made from report by the matching nsys CLI.
        report: Original .nsys-rep (fingerprinted, never modified).
        replays: Collector rank-N.replays.jsonl, including late ForwardId bindings.
        window: Final flushed rank-N.window.json.
        registry: Explicit trusted CPU cost rules; serialized imports are forbidden.
        hardware: Optional equivalent hardware profile verification path.
        evidence_kind: 'nsight_capture' or 'synthetic_cpu_fixture'. The latter
            exercises serialization and joins, never produces verified GPU proof.
    Returns:
        Per-step Model/Layer/Op metrics and provenance; absent ownership or costs
        remain explicit critical unknowns. Quantities are rank-local only.
    """
    if evidence_kind not in {"nsight_capture", "synthetic_cpu_fixture"}:
        raise ValueError("Select the evidence kind explicitly")
    manifest = json.loads(window.read_text())
    if (
        manifest.get("schema") != "tokenspeed.profile_window/v1"
        or not isinstance(manifest.get("run_id"), str)
        or not manifest["run_id"]
        or not _integer(manifest.get("rank"))
    ):
        raise ValueError("Unsupported collector window schema")
    events = [
        json.loads(line) for line in replays.read_text().splitlines() if line.strip()
    ]
    ready, bindings, leases = {}, {}, {}
    forward_steps = {}
    for index, row in enumerate(events):
        if (row.get("run_id"), row.get("rank")) != (
            manifest["run_id"],
            manifest["rank"],
        ):
            raise ValueError("Cross-run/rank replay index")
        event = row.get("event")
        if event == "window_started":
            if index != 0 or any(
                not _integer(row.get(name), minimum=1)
                or row[name] != manifest.get(name)
                for name in ("start_step", "num_steps")
            ):
                raise ValueError("Invalid or misplaced collector window header")
            continue
        if event not in {"forward_identity_bound", "device_results_ready"}:
            raise ValueError(f"Unsupported replay index event: {event!r}")
        step = row["step_id"]
        if not isinstance(step, str) or not step:
            raise ValueError("Invalid replay step identity")
        if event == "forward_identity_bound":
            if step in bindings and bindings[step] != row:
                raise ValueError("Ambiguous late ForwardId binding")
            bindings[step] = row
        elif event == "device_results_ready":
            if step in ready:
                raise ValueError("Duplicate device-ready step identity")
            ready[step] = row
    all_rows, steps, sources = [], [], {}
    lifecycle_unknowns = []
    unavailable = [
        row for row in ready.values() if row.get("unavailable_reason") is not None
    ]
    persisted = [
        row
        for row in ready.values()
        if row.get("unavailable_reason") is None
        and row.get("binary_payload")
        and row.get("capture_file")
    ]
    eager = [row for row in ready.values() if row.get("eager_sidecar") is not None]
    mixed = bool(eager)
    if mixed and eager_projection is None:
        raise ValueError(
            "Mixed native analysis requires the same-report eager projection"
        )
    if mixed and set(bindings) - set(ready):
        raise ValueError("Late ForwardId binding lacks a device-ready step")
    claimed_activities = {}
    source_report = file_source(report)
    # Account for explicit dropped rows separately. A flushed, fingerprinted
    # payload remains valid when a different step could not be persisted.
    if (
        manifest.get("writer_flushed") is not True
        or manifest.get("writer_error") is not None
        or manifest.get("unresolved_result_metadata") != 0
        or manifest.get("selected_steps") != len(ready)
        or manifest.get("device_results_ready") != len(ready)
        or manifest.get("persisted_payloads") != len(persisted)
        or manifest.get("dropped_metadata") != len(unavailable)
        or manifest.get("persisted_eager", 0) != len(eager)
        or len(persisted) + len(unavailable) + len(eager) != len(ready)
        or any(row.get("binary_payload") for row in unavailable)
    ):
        lifecycle_unknowns.append("collector_window_incomplete")
    whole_window_unknowns = list(lifecycle_unknowns)
    if unavailable:
        whole_window_unknowns.append("collector_metadata_dropped")
    communication_bindings = {}
    typed_capture_cache = {}
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        native_index = _read_native_index(
            db,
            [
                row
                for row in ready.values()
                if row.get("unavailable_reason") is None
                and row.get("binary_payload")
                and row.get("capture_file")
            ],
        )
        for step_id, original in ready.items():
            row = dict(original)
            if step_id in bindings:
                merge_forward_binding(row, bindings[step_id])
            unknowns = list(lifecycle_unknowns)
            if not _integer(row.get("forward_id")):
                unknowns.append("forward_identity_unavailable")
            elif row["forward_id"] in forward_steps:
                raise ValueError("One ForwardId maps to multiple replay steps")
            else:
                forward_steps[row["forward_id"]] = step_id
            if row.get("eager_sidecar") is not None:
                eager_step, scopes, timing = analyze_eager_step(
                    row,
                    config_sha256=manifest.get("profile_config_sha256"),
                    database=database,
                    projection=eager_projection,
                    hardware=hardware,
                    device_uuids=native_index.device_uuids,
                    registry=registry,
                    loaded_library_manifest=manifest.get("loaded_library_manifest"),
                    source_report_sha256=source_report["sha256"],
                    sealed_communication_artifacts=sealed_communication_artifacts,
                )
                for scope in scopes:
                    for item in scope["native_activity_rows"]:
                        identity = item["table"], item["rowid"]
                        if claimed_activities.setdefault(identity, step_id) != step_id:
                            raise ValueError(
                                "Native GPU activity belongs to multiple serving steps"
                            )
                eager_step["critical_unknowns"] = sorted(
                    set(unknowns) | set(eager_step["critical_unknowns"])
                )
                eager_step["eager_native_timing"] = timing
                steps.append(eager_step)
                all_rows.extend(scopes)
                artifact = row["eager_sidecar"]
                sources[artifact["path"]] = artifact
                continue
            if (
                row.get("unavailable_reason") is not None
                or not row.get("binary_payload")
                or not row.get("capture_file")
            ):
                steps.append(
                    {
                        **row,
                        "critical_unknowns": unknowns
                        + [
                            row.get("unavailable_reason")
                            or "collector_metadata_unavailable"
                        ],
                        "graph_node_ids": [],
                    }
                )
                continue
            capture_path = Path(row["capture_file"])
            capture = json.loads(capture_path.read_text())
            if (
                capture.get("schema_version") != 1
                or capture.get("execution_mode") != "cuda_graph_capture"
            ):
                raise ValueError("Unsupported graph capture schema")
            payload_path = Path(row["binary_payload"]["path"])
            current = decode_replay(capture, row, payload_path=payload_path)
            capture_key = str(capture_path.resolve())
            if capture_key not in communication_bindings:
                communication_bindings[capture_key] = bind_capture_communication(
                    current,
                    capture_path.parent,
                    sealed_artifacts=sealed_communication_artifacts,
                )
            for op in current["ops"]:
                if op["trace_label"] in communication_bindings[capture_key]:
                    op["scalar_args"].update(
                        communication_bindings[capture_key][op["trace_label"]]
                    )
            slot, generation = (
                row["metadata_lease"]["slot"],
                row["metadata_lease"]["generation"],
            )
            if generation <= leases.get(slot, 0):
                raise ValueError("Metadata lease generation reused or out of order")
            if "slot_states" in manifest and slot >= sum(
                manifest["slot_states"].values()
            ):
                raise ValueError("Metadata lease slot outside collector capacity")
            leases[slot] = generation
            scopes, records, ancestry = _scopes(current)
            activities, evidence, join_unknowns = _join(native_index, current, row)
            if mixed:
                for activity in activities:
                    identity = activity["table"], activity["source_rowid"]
                    if claimed_activities.setdefault(identity, step_id) != step_id:
                        raise ValueError(
                            "Native GPU activity belongs to multiple serving steps"
                        )
                evidence["attribution_mode"] = "cuda_graph_node_ids"
            evidence.update(
                {
                    name: row.get(name)
                    for name in ("step_id", "forward_id", "capture_id")
                }
            )
            unknowns.extend(join_unknowns)
            observer = set(current["observer_graph_node_ids"])
            activities = [item for item in activities if item["source"] not in observer]
            source_nodes = {item["source"] for item in activities}
            captured_nodes = (
                set().union(
                    *(set(scope["graph_node_ids"]) for scope in scopes.values())
                )
                - observer
            )
            outside = source_nodes - captured_nodes
            if outside:
                unknowns.append("correlated_nodes_outside_capture_scopes")
            hw = hardware_for_analysis(current["hardware"], hardware)
            capture_source = file_source(capture_path)
            type_key = (capture_source["path"], capture_source["sha256"])
            if type_key not in typed_capture_cache:
                typed_capture_cache[type_key] = _capture_node_requirements(
                    current, scopes, observer
                )
            requirements, node_type_index = typed_capture_cache[type_key]
            type_conflicts = _node_activity_conflicts(node_type_index, activities)
            engine_context = {}
            for identity, scope in scopes.items():
                label = scope["trace_label"]
                kind, record = records[label]
                if kind != "op":
                    continue
                node_requirements = requirements[identity]
                expected = node_requirements["owned"]
                required = node_requirements["activity_required"]
                selected = [item for item in activities if item["source"] in expected]
                engine_context[label] = dict(
                    complete_join=not join_unknowns
                    and bool(required)
                    and not expected & type_conflicts
                    and {item["source"] for item in selected} == required,
                    capture_sha256=capture_source["sha256"],
                    replay_identity={
                        key: row.get(key)
                        for key in ("step_id", "forward_id", "capture_id", "process_id")
                    },
                    run_id=current["run_id"],
                    rank=current["rank"],
                    scope_identity={"scope_id": identity, "trace_label": label},
                    source_node_ids=sorted(expected),
                    activity_required_source_node_ids=sorted(required),
                    dependency_source_node_ids=sorted(node_requirements["dependency"]),
                    activity_bindings=selected,
                    join_evidence=evidence,
                    loaded_library_manifest=manifest.get("loaded_library_manifest"),
                    profile_config_sha256=current.get("static_metadata", {}).get(
                        "profile_config_sha256"
                    ),
                    startup_observation_sha256=current.get("static_metadata", {}).get(
                        "startup_observation_sha256"
                    ),
                    hardware={"device": hw.device, "selection": hw.selection},
                    native_device_uuids=native_index.device_uuids,
                )
            current["analysis"] = _reevaluate(
                current, registry, engine_context=engine_context
            )
            costs = {
                op["trace_label"]: entry
                for op, entry in zip(
                    current["ops"], current["analysis"]["ops"], strict=True
                )
            }
            attributed_scopes, attributed_layers, kinds = [], set(), set()
            for identity, scope in scopes.items():
                label = scope["trace_label"]
                kind, record = records[label]
                selected = [
                    item
                    for item in activities
                    if item["source"] in scope["graph_node_ids"]
                ]
                nodes = {item["source"] for item in selected}
                node_requirements = requirements[identity]
                expected_nodes = node_requirements["owned"]
                missing_nodes = node_requirements["activity_required"] - nodes
                unexpected_activity = expected_nodes & type_conflicts
                op_labels = [
                    candidate["trace_label"]
                    for key, candidate in scopes.items()
                    if identity in ancestry[key]
                    and records[candidate["trace_label"]][0] == "op"
                ]
                entries = [costs[name] for name in op_labels]
                owned = (
                    set().union(
                        *(
                            set(scopes[key]["graph_node_ids"])
                            for key in scopes
                            if identity in ancestry[key]
                            and records[scopes[key]["trace_label"]][0] == "op"
                        )
                    )
                    - observer
                )
                uncovered = nodes - owned
                intervals = [(item["start"], item["end"]) for item in selected]
                start = min((a for a, _ in intervals), default=None)
                end = max((b for _, b in intervals), default=None)
                duration = 0 if start is None else end - start
                metrics = scope_metrics(
                    entries,
                    duration,
                    hw,
                    scope_kind=kind,
                    communication_observation=(
                        record["scalar_args"] if kind == "op" else None
                    ),
                )
                scope_invalidity = []
                if not nodes or missing_nodes:
                    scope_invalidity.append("missing_captured_scope_node_activities")
                if unexpected_activity:
                    scope_invalidity.append(
                        "activity_conflicts_with_captured_node_type"
                    )
                if uncovered:
                    scope_invalidity.append("gpu_nodes_outside_costed_ops")
                if any(entry["cost"] is None for entry in entries) or not entries:
                    scope_invalidity.append("cost_coverage_incomplete")
                metric_unknowns = []
                if metrics.get("useful_flops") and metrics.get("mfu_total") is None:
                    metric_unknowns.append("compute_normalization_unavailable")
                if (
                    metrics.get("estimated_hbm_bytes")
                    and metrics.get("estimated_mbu") is None
                ):
                    metric_unknowns.append("hbm_normalization_unavailable")
                if (
                    entries
                    and all(entry["cost"] is not None for entry in entries)
                    and any(
                        entry["cost"]["estimated_hbm_bytes"] is None
                        for entry in entries
                    )
                ):
                    metric_unknowns.append("hbm_estimate_unavailable")
                if record.get("error"):
                    scope_invalidity.append("recorded_scope_error")
                scope_invalidity.extend(lifecycle_unknowns)
                scope_invalidity.extend(join_unknowns)
                # Shared scope_metrics already leaves each unavailable metric null.
                # Only invalid scope evidence invalidates independent metrics too.
                if scope_invalidity:
                    _partial(metrics, "; ".join(scope_invalidity))
                scope_unknowns = scope_invalidity + metric_unknowns
                if scope_unknowns:
                    unknowns.extend(
                        f"scope={identity}:{reason}" for reason in scope_unknowns
                    )
                if nodes and not missing_nodes and not unexpected_activity:
                    attributed_scopes.append(identity)
                    if kind == "op":
                        kinds.add("op")
                    elif (
                        record["layer_id"] is None and scope["parent_scope_id"] is None
                    ):
                        kinds.add("model")
                    else:
                        kinds.add("layer")
                        if type(record["layer_id"]) is int:
                            attributed_layers.add(record["layer_id"])
                all_rows.append(
                    {
                        **metrics,
                        "kind": kind,
                        "scope_id": identity,
                        "parent_scope_id": scope["parent_scope_id"],
                        "rank": current["rank"],
                        "iter_id": row["serving_step"],
                        "serving_step": row["serving_step"],
                        "step_id": step_id,
                        "forward_id": row.get("forward_id"),
                        "capture_id": row["capture_id"],
                        "layer_id": record["layer_id"],
                        "occurrence": record["occurrence"],
                        "trace_label": label,
                        "native_nvtx_label": scope["display_name"],
                        "gpu_start_ns": start,
                        "gpu_end_ns": end,
                        "gpu_duration_ns": duration,
                        "gpu_activity_sum_ns": sum(b - a for a, b in intervals),
                        "gpu_activity_union_ns": _union(intervals),
                        "gpu_activity_count": len(selected),
                        "graph_node_ids": sorted(nodes),
                        "owned_graph_node_ids": sorted(expected_nodes),
                        "activity_required_graph_node_ids": sorted(
                            node_requirements["activity_required"]
                        ),
                        "dependency_graph_node_ids": sorted(
                            node_requirements["dependency"]
                        ),
                        "missing_graph_node_ids": sorted(missing_nodes),
                        "replay_node_ids": sorted({item["node"] for item in selected}),
                        "source_nodes_outside_opted_in_ops": sorted(uncovered),
                        "critical_unknowns": scope_unknowns,
                        "quantity_scope": "per_rank",
                        "coverage_scope": "opted_in_registered_calls",
                        **(
                            {"attribution_mode": "cuda_graph_node_ids"} if mixed else {}
                        ),
                    }
                )
            capture_source, binary_source = file_source(capture_path), file_source(
                payload_path
            )
            sources[str(capture_path.resolve())] = capture_source
            sources[str(payload_path.resolve())] = binary_source
            if current.get("unavailable_integer_metadata"):
                unknowns.append("capture_integer_metadata_unavailable")
            steps.append(
                {
                    **evidence,
                    "rank": current["rank"],
                    "serving_step": row["serving_step"],
                    "graph_node_ids": sorted(
                        node for node in source_nodes if node is not None
                    ),
                    "uncovered_graph_node_ids": sorted(
                        node for node in outside if node is not None
                    ),
                    "unidentified_activity_count": sum(
                        item["source"] is None for item in activities
                    ),
                    "observer_graph_node_ids": sorted(observer),
                    "scope_kinds": sorted(kinds),
                    "expected_scope_ids": sorted(scopes),
                    "attributed_scope_ids": sorted(attributed_scopes),
                    "attributed_layer_ids": sorted(attributed_layers),
                    "critical_unknowns": sorted(set(unknowns)),
                    "capture_file": capture_source,
                    "binary_payload": binary_source,
                    "replay_analysis": current["analysis"],
                    "hardware": asdict(hw),
                }
            )
    if not ready:
        lifecycle_unknowns.append("no_selected_device_ready_steps")
        whole_window_unknowns.append("no_selected_device_ready_steps")
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "rank": manifest["rank"],
        "execution_mode": (
            "cuda_graph_replay_and_eager" if mixed else "cuda_graph_replay"
        ),
        "evidence_kind": evidence_kind,
        "quantity_scope": "per_rank",
        "steps": steps,
        "critical_unknowns": sorted(set(whole_window_unknowns)),
        "global_critical_unknowns": sorted(set(lifecycle_unknowns)),
        "whole_run_status": (
            "partial"
            if whole_window_unknowns or any(step["critical_unknowns"] for step in steps)
            else "complete"
        ),
        "collector_window": {
            "manifest": manifest,
            "dropped_steps": [
                {
                    name: step.get(name)
                    for name in (
                        "forward_id",
                        "step_id",
                        "request_ids",
                        "unavailable_reason",
                    )
                }
                for step in steps
                if step.get("unavailable_reason") is not None
            ],
        },
        "provenance": {
            "report": source_report,
            "database": file_source(database),
            "replays": file_source(replays),
            "window": file_source(window),
            "artifacts": list(sources.values()),
            **(
                {
                    "sealed_communication_sources": list(
                        sealed_communication_artifacts.values()
                    )
                }
                if sealed_communication_artifacts is not None
                else {}
            ),
            **({"eager_projection": file_source(eager_projection)} if mixed else {}),
        },
        "device_analysis": {
            "timing_kind": (
                "nsight_mixed_graph_eager" if mixed else "nsight_graph_node_ids"
            ),
            "source_report": str(report.resolve()),
            "source_report_sha256": source_report["sha256"],
            "timing_definition": (
                "last exactly attributed GPU activity end minus first start"
                if mixed
                else "last associated graph-node activity end minus first start"
            ),
            "timestamp_unit": "ns",
            "scopes": all_rows,
        },
    }


def native_coverage_receipt(
    result: dict, *, metadata: Path, model_config: Path
) -> dict:
    """Build the service's fixed v1 receipt over its entire measured ForwardId set.

    Model depth is read from the frozen config; observed layer count never defines
    expected coverage. Native status is verified only when every requested step,
    capture, layer, scope and critical cost/node provenance check succeeds.
    """
    source = json.loads(metadata.read_text())
    config = json.loads(model_config.read_text())
    config = config.get("text_config", config)
    values = [
        config[key] for key in ("num_hidden_layers", "num_layers") if key in config
    ]
    if (
        not values
        or any(not _integer(value, minimum=1) for value in values)
        or len(set(values)) != 1
    ):
        raise ValueError("Frozen model config has missing or conflicting depth")
    expected_layers = list(range(values[0]))
    if source.get("model_config_source") != file_source(model_config):
        raise ValueError("Frozen model config differs from service metadata source")
    problems = list(result["critical_unknowns"])
    problems.extend(result.get("global_critical_unknowns", []))
    for source_name, result_name in (
        ("ready_index_source", "replays"),
        ("window_manifest_source", "window"),
    ):
        if source_name in source and (
            source[source_name].get("sha256")
            != result["provenance"][result_name]["sha256"]
        ):
            raise ValueError(f"Service metadata references a different {result_name}")
    if result["evidence_kind"] != "nsight_capture":
        problems.append("synthetic_cpu_fixture_is_not_gpu_evidence")
    if (source.get("run_id"), source.get("rank", 0)) != (
        result["run_id"],
        result["rank"],
    ):
        raise ValueError("Service metadata run/rank mismatch")
    if source.get("metadata_status") != "verified":
        problems.append("service_metadata_not_verified")
    selected_forward_ids = {
        item["forward_id"]
        for request in source["requests"].values()
        for item in request["forward_evidence"]
    }
    selected_step_ids = {
        item["step_id"]
        for request in source["requests"].values()
        for item in request["forward_evidence"]
    }
    collector = result.get("collector_window", {})
    dropped = collector.get("dropped_steps", [])
    # This exception is limited to independently identified, disjoint drops.
    # All global failures and every requested step's critical checks still gate
    # native verification; whole-run coverage remains partial in the receipt.
    if "collector_metadata_dropped" in problems:

        def drop_identity(item: dict) -> tuple:
            return (
                item.get("forward_id"),
                item.get("step_id"),
                tuple(item.get("request_ids") or []),
                item.get("unavailable_reason"),
            )

        if (
            source.get("requested_coverage_mode") == "decode"
            and source.get("metadata_status") == "verified"
            and source.get("dropped_metadata_identified") is True
            and source.get("dropped_metadata_disjoint_from_measured_decode") is True
            and collector.get("manifest") == source.get("window_manifest")
            and len(dropped) == collector["manifest"]["dropped_metadata"]
            and sorted(map(drop_identity, dropped), key=repr)
            == sorted(map(drop_identity, source.get("dropped_steps", [])), key=repr)
            and all(
                _integer(item.get("forward_id"))
                and item["forward_id"] not in selected_forward_ids
                and item.get("step_id") not in selected_step_ids
                and item.get("request_ids")
                and item.get("unavailable_reason")
                for item in dropped
            )
            and all(
                source.get(source_name, {}).get("sha256")
                == result["provenance"][result_name]["sha256"]
                for source_name, result_name in (
                    ("ready_index_source", "replays"),
                    ("window_manifest_source", "window"),
                )
            )
        ):
            problems.remove("collector_metadata_dropped")
    requests = {}
    for key, request in source["requests"].items():
        evidence = []
        requested_ids = [item["forward_id"] for item in request["forward_evidence"]]
        if (
            len(set(requested_ids)) != len(requested_ids)
            or request.get("forward_ids", requested_ids) != requested_ids
        ):
            problems.append("service_requested_forward_set_mismatch")
        for expected in request["forward_evidence"]:
            matches = [
                item
                for item in result["steps"]
                if all(
                    item.get(name) == expected[name]
                    for name in ("forward_id", "step_id", "capture_id")
                )
            ]
            unknowns = []
            if len(matches) != 1:
                item = {
                    name: expected[name]
                    for name in ("forward_id", "step_id", "capture_id")
                }
                unknowns.append("missing_or_ambiguous_measured_forward")
            else:
                item = copy.deepcopy(matches[0])
                unknowns.extend(item["critical_unknowns"])
            expected_scopes = expected.get("expected_scope_ids")
            actual_scopes = item.get("attributed_scope_ids", [])
            actual_layers = item.get("attributed_layer_ids", [])
            if (
                expected.get("expected_layer_ids") != expected_layers
                or actual_layers != expected_layers
            ):
                unknowns.append("frozen_model_layer_coverage_incomplete")
            if not isinstance(expected_scopes, list) or not expected_scopes:
                # A dropped forward still has an exact serving identity. Its
                # per-forward scope expectations were never established.
                expected_scopes = None
                unknowns.append("measured_scope_expectations_unavailable")
            elif (
                item.get("expected_scope_ids") != expected_scopes
                or actual_scopes != expected_scopes
            ):
                unknowns.append("measured_scope_coverage_incomplete")
            eager = expected.get("eager_sidecar")
            for name in (
                ("eager_sidecar",)
                if eager is not None
                else ("capture_file", "binary_payload")
            ):
                recorded, required = item.get(name), expected.get(name)
                if (
                    not isinstance(recorded, dict)
                    or not isinstance(required, dict)
                    or any(
                        not required.get(field)
                        or recorded.get(field) != required[field]
                        for field in ("path", "sha256")
                    )
                ):
                    unknowns.append(f"measured_{name}_hash_mismatch")
            if set(item.get("scope_kinds", [])) != {
                "model",
                "layer",
                "op",
            } or (
                not item.get("graph_node_ids")
                if eager is None
                else item.get("attribution_mode") != "eager_nvtx_gpu_projection"
                or not item.get("eager_native_timing", {}).get("activities")
            ):
                unknowns.append("missing_model_layer_op_native_coverage")
            item.update(
                layer_coverage={
                    "expected_ids": expected_layers,
                    "attributed_ids": actual_layers,
                    "missing_ids": sorted(set(expected_layers) - set(actual_layers)),
                },
                scope_coverage={
                    "expected_scope_ids": expected_scopes,
                    "attributed_scope_ids": actual_scopes,
                    "missing_scope_ids": (
                        sorted(set(expected_scopes) - set(actual_scopes))
                        if expected_scopes is not None
                        else None
                    ),
                },
                critical_unknowns=sorted(set(unknowns)),
            )
            problems.extend(unknowns)
            evidence.append(item)
        if not evidence:
            problems.append("request_has_no_measured_forward_evidence")
        requests[key] = {
            name: request[name]
            for name in ("rid", "generation", "first_count", "last_count")
        }
        requests[key]["forward_evidence"] = evidence
    if not requests:
        problems.append("no_measured_requests")
    return {
        "schema": "tokenspeed.native_coverage/v1",
        "run_id": result["run_id"],
        "rank": result["rank"],
        "scope": "model_layer_op",
        "coverage_mode": source["requested_coverage_mode"],
        "attribution": (
            "same_run_graph_and_eager_native_activities"
            if result["device_analysis"]["timing_kind"] == "nsight_mixed_graph_eager"
            else "same_run_cuda_graph_node_ids"
        ),
        "status": "partial" if problems else "verified",
        "whole_run_status": result.get("whole_run_status", "not_verified"),
        "whole_run_critical_unknowns": sorted(
            set(result["critical_unknowns"])
            | {
                reason
                for item in result["steps"]
                for reason in item["critical_unknowns"]
            }
        ),
        "selected_forward_ids": sorted(selected_forward_ids),
        "critical_unknowns": sorted(set(problems)),
        "source_metadata_sha256": file_source(metadata)["sha256"],
        "source_model_config": file_source(model_config),
        "source_nsight_report": result["provenance"]["report"],
        "source_provenance": result["provenance"],
        "requests": requests,
    }
