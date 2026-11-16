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

"""Offline scheduler graph-launch identity and GPU gap analysis.

Legacy selection uses only the standard library and can run as a script.
Collector selection reuses the CPU-safe graph reader without initializing CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspeed_kernel.op_profile import CostRegistry

_GPU_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "kernel",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "memcpy",
    "CUPTI_ACTIVITY_KIND_MEMSET": "memset",
}
_GRAPH_LAUNCH = re.compile(r"cudaGraphLaunch(?:_v[0-9]+|_ptsz)*(?:\(.*\))?$")


def _digest(path: Path) -> dict[str, str]:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            sha.update(block)
    return {"path": str(path.resolve()), "sha256": sha.hexdigest()}


def _merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("GPU activity ends before it starts")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


class _Coverage:
    def __init__(self, intervals):
        self.intervals = _merged(intervals)
        self.starts = [start for start, _ in self.intervals]
        self.ends = [end for _, end in self.intervals]

    def within(self, start: int, end: int) -> int:
        left = bisect_right(self.ends, start)
        right = bisect_left(self.starts, end)
        return sum(min(b, end) - max(a, start) for a, b in self.intervals[left:right])


@dataclass(frozen=True, slots=True)
class _Activity:
    start: int
    end: int
    process_key: int
    correlation: int | None
    node: int | None
    device: int | None
    context: int | None
    stream: int | None
    kind: str


def _timing(activities: list[_Activity]) -> dict[str, Any]:
    if not activities:
        return {
            "gpu_start_ns": None,
            "gpu_end_ns": None,
            "gpu_span_ns": None,
            "gpu_activity_sum_ns": None,
            "gpu_activity_union_ns": None,
            "gpu_activity_count": 0,
        }
    intervals = [(a.start, a.end) for a in activities]
    start, end = min(a for a, _ in intervals), max(b for _, b in intervals)
    return {
        "gpu_start_ns": start,
        "gpu_end_ns": end,
        "gpu_span_ns": end - start,
        "gpu_activity_sum_ns": sum(b - a for a, b in intervals),
        "gpu_activity_union_ns": sum(b - a for a, b in _merged(intervals)),
        "gpu_activity_count": len(activities),
    }


def _stats(values) -> dict[str, int | float | None]:
    values = sorted(values)
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    def quantile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "count": len(values),
        "min": values[0],
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "max": values[-1],
    }


def _schema(db):
    return {
        row[0]: {column[1] for column in db.execute(f'PRAGMA table_info("{row[0]}")')}
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if re.fullmatch(r"[A-Za-z0-9_]+", row[0])
    }


def _require(schema, table, columns):
    missing = set(columns) - schema.get(table, set())
    if missing:
        raise ValueError(f"{table} lacks required columns: {sorted(missing)}")


def _nvtx(db, schema, label):
    _require(schema, "NVTX_EVENTS", ("start", "end", "globalTid"))
    fields = schema["NVTX_EVENTS"]
    text = "n.text" if "text" in fields else "NULL"
    join = ""
    if "textId" in fields and {"id", "value"} <= schema.get("StringIds", set()):
        text = f"COALESCE({text}, s.value)"
        join = "LEFT JOIN StringIds s ON n.textId=s.id"
    end_tid = "n.endGlobalTid" if "endGlobalTid" in fields else "NULL"
    return db.execute(
        f"SELECT n.rowid AS range_id,n.start,n.end,n.globalTid,{end_tid} AS end_tid "
        f"FROM NVTX_EVENTS n {join} WHERE {text}=? ORDER BY n.start,n.rowid",
        (label,),
    ).fetchall()


def _complete_range(row):
    if (
        row["end"] is None
        or row["end"] < row["start"]
        or row["end_tid"] not in (None, row["globalTid"])
    ):
        raise ValueError("Incomplete or cross-thread NVTX range")


def _captures(paths, process_id):
    prepared, ranks, provenance = [], {}, []
    for path in paths:
        data = json.loads(path.read_text())
        if data["process_id"] != process_id:
            raise ValueError("Capture metadata belongs to another process")
        nodes = {
            node for scope in data["graph_scopes"] for node in scope["graph_node_ids"]
        }
        if not nodes:
            raise ValueError("Capture metadata contains no model graph nodes")
        if process_id in ranks and ranks[process_id] != data["rank"]:
            raise ValueError("Capture metadata has inconsistent process rank")
        prepared.append((str(path.resolve()), nodes))
        ranks[process_id] = data["rank"]
        provenance.append(_digest(path))
    return prepared, ranks, provenance


def _intersection(left, right):
    """Intersect activity unions, without treating their envelopes as activity."""
    a, b = _merged(left), _merged(right)
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            result.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _duration(intervals):
    return sum(end - start for start, end in _merged(intervals))


def _verified_source(source):
    path = Path(source["path"])
    actual = _digest(path)
    if actual["sha256"] != source["sha256"]:
        raise ValueError(f"Source SHA-256 mismatch: {path}")
    return path


def _collector_metadata(path):
    """Resolve existing service provenance, including late ForwardId bindings."""
    from tokenspeed_kernel.profile_eager import merge_forward_binding

    metadata = json.loads(path.read_text())
    if metadata.get("schema") != "tokenspeed.metadata_coverage/v1":
        raise ValueError("Unsupported service metadata schema")
    manifest_path = _verified_source(metadata["window_manifest_source"])
    manifest = json.loads(manifest_path.read_text())
    run, rank = metadata["run_id"], manifest["rank"]
    if manifest.get("run_id") != run or manifest != metadata["window_manifest"]:
        raise ValueError("Collector window/service metadata mismatch")
    ready_path = _verified_source(metadata["ready_index_source"])
    ready, bindings = {}, {}
    for line in ready_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("run_id"), row.get("rank")) != (run, rank):
            raise ValueError("Cross-run/rank ready index")
        if row["event"] == "window_started":
            continue
        step = row["step_id"]
        if row["event"] == "forward_identity_bound":
            if step in bindings and bindings[step] != row:
                raise ValueError("Conflicting late ForwardId binding")
            bindings[step] = row
        elif row["event"] == "device_results_ready":
            if step in ready:
                raise ValueError("Duplicate device-ready step identity")
            ready[step] = row
        else:
            raise ValueError("Unsupported collector event")
    forwards = set()
    for step, row in ready.items():
        if step in bindings:
            merge_forward_binding(row, bindings[step])
        if row.get("forward_id") is not None:
            if row["forward_id"] in forwards:
                raise ValueError("One ForwardId maps to multiple ready steps")
            forwards.add(row["forward_id"])
    commits = defaultdict(list)
    for source in metadata["service_sources"]:
        for line in _verified_source(source).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            graph = row.get("graph")
            if row.get("run_id") != run or row.get("boundary") != "commit" or not graph:
                continue
            if (graph.get("run_id"), graph.get("rank")) != (run, rank):
                continue
            step = graph.get("step_id")
            if step not in ready:
                continue
            replay = ready[step]
            if (row.get("forward_id"), row.get("pid")) != (
                replay.get("forward_id"),
                replay.get("process_id"),
            ) or row.get("rid") not in replay.get("request_ids", []):
                raise ValueError("Service commit/ready identity mismatch")
            if (
                type(row.get("generation")) is not int
                or type(row.get("observed_ns")) is not int
            ):
                raise ValueError(
                    "Authoritative commit lacks request generation or timestamp"
                )
            count, ids = row.get("completion_tokens"), row.get("output_ids")
            if (
                type(count) is not int
                or not isinstance(ids, list)
                or not ids
                or count < len(ids)
            ):
                raise ValueError("Invalid committed token interval")
            if any(item["rid"] == row["rid"] for item in commits[step]):
                raise ValueError("Duplicate authoritative request commit for a step")
            commits[step].append(row)
    return metadata, manifest, ready, commits


def _semantic_intervals(capture, row, activities, *, registry: CostRegistry):
    """Classify exact Op node owners using trusted costs, retaining unknowns."""
    from tokenspeed_kernel.profile_graph_analysis import (
        _reevaluate,
        _scopes,
        decode_replay,
    )

    decoded = decode_replay(
        capture, row, payload_path=Path(row["binary_payload"]["path"])
    )
    _scopes(decoded)
    costs = _reevaluate(decoded, registry)
    entries = {
        op["trace_label"]: entry["cost"]
        for op, entry in zip(decoded["ops"], costs["ops"], strict=True)
    }
    owners = defaultdict(list)
    for scope in decoded["graph_scopes"]:
        if scope["trace_label"] not in entries:
            continue
        cost = entries[scope["trace_label"]]
        category = "unclassified"
        if cost is not None:
            comm = cost.get("communication") is not None
            compute = cost["useful_flops"] > 0
            category = (
                "fused_compute_communication"
                if comm and compute
                else (
                    "pure_communication"
                    if comm
                    else "compute" if compute else "other_known_op"
                )
            )
        for node in scope["graph_node_ids"]:
            owners[node].append(category)
    groups = defaultdict(list)
    for activity in activities:
        categories = owners.get(activity["source"], [])
        # Even equal categories cannot legitimize ambiguous Op ownership.
        category = categories[0] if len(categories) == 1 else "unclassified"
        groups[category].append((activity["start"], activity["end"]))
    return dict(groups)


def _semantic_summary(groups):
    comm, compute = groups.get("pure_communication", []), groups.get("compute", [])
    overlap = _duration(_intersection(comm, compute))
    return {
        "activity_union_ns_by_class": {
            name: _duration(groups.get(name, []))
            for name in (
                "pure_communication",
                "compute",
                "fused_compute_communication",
                "other_known_op",
                "unclassified",
            )
        },
        "pure_communication_compute_activity_intersection_ns": overlap,
        "pure_communication_outside_classified_compute_ns": _duration(comm) - overlap,
        "classification_method": "unique captured Op node ownership and trusted cost semantics",
        "interpretation": "classified activity coverage, not causal critical path; fused Op intervals have no standalone communication duration",
    }


def _token_commits(rows):
    return [
        {
            "rid": row["rid"],
            "generation": row["generation"],
            "completion_tokens": row["completion_tokens"],
            "accepted_token_count": len(row["output_ids"]),
            "accepted_token_interval": [
                row["completion_tokens"] - len(row["output_ids"]),
                row["completion_tokens"],
            ],
            "observed_ns": row["observed_ns"],
            "output_ids": row["output_ids"],
        }
        for row in rows
    ]


def _collector_requests(metadata, steps, gaps):
    by_step = {step["step_id"]: step for step in steps}
    pairs = {(pair["previous_step_id"], pair["next_step_id"]): pair for pair in gaps}
    requests = {}
    for key, expected in metadata["requests"].items():
        rid, generation = expected.get("rid"), expected.get("generation")
        endpoints, unknowns = [], []
        for identity in expected.get("forward_evidence", []):
            step = by_step.get(identity["step_id"])
            if step is None or any(
                step.get(name) != identity[name]
                for name in ("forward_id", "capture_id")
            ):
                unknowns.append("missing_or_ambiguous_measured_forward")
                continue
            matches = [
                c
                for c in step["token_commits"]
                if (c["rid"], c["generation"]) == (rid, generation)
            ]
            if len(matches) != 1:
                unknowns.append("missing_exact_authoritative_commit")
                continue
            commit = matches[0]
            if (
                not expected["first_count"]
                <= commit["completion_tokens"]
                <= expected["last_count"]
            ):
                raise ValueError(
                    "Measured forward lies outside committed request window"
                )
            for field in ("capture_file", "binary_payload", "eager_sidecar"):
                actual = step["collector_sources"].get(field)
                if field in identity and actual is None:
                    unknowns.append(f"measured_{field}_unavailable")
                elif (
                    field in identity and actual["sha256"] != identity[field]["sha256"]
                ):
                    raise ValueError(
                        "Measured source hash differs from service metadata"
                    )
            endpoints.append({"step_id": step["step_id"], **commit})
        endpoints.sort(key=lambda item: item["completion_tokens"])
        counts = [item["completion_tokens"] for item in endpoints]
        if (
            not counts
            or counts[0] != expected.get("first_count")
            or counts[-1] != expected.get("last_count")
            or len(set(counts)) != len(counts)
        ):
            unknowns.append("missing_or_duplicate_exact_window_endpoints")
        service_unknowns = [
            reason
            for reason in unknowns
            if reason
            not in (
                "measured_capture_file_unavailable",
                "measured_binary_payload_unavailable",
                "measured_eager_sidecar_unavailable",
            )
        ]
        for previous, following in zip(endpoints, endpoints[1:]):
            if (
                following["accepted_token_interval"][0] != previous["completion_tokens"]
                or following["observed_ns"] < previous["observed_ns"]
            ):
                service_unknowns.append(
                    "missing_or_invalid_authoritative_commit_interval"
                )
        selected = []
        for previous, following in zip(endpoints, endpoints[1:]):
            pair = pairs.get((previous["step_id"], following["step_id"]))
            if (
                following["accepted_token_interval"][0] != previous["completion_tokens"]
                or pair is None
            ):
                unknowns.append("missing_adjacent_executed_or_committed_step")
                continue
            selected.append(
                {
                    **pair,
                    "service_commit_period_ns": following["observed_ns"]
                    - previous["observed_ns"],
                    "accepted_token_interval": [
                        previous["completion_tokens"],
                        following["completion_tokens"],
                    ],
                }
            )
        window_steps = [by_step[item["step_id"]] for item in endpoints]
        semantic = defaultdict(list)
        for step in window_steps:
            for name, intervals in step.get(
                "semantic_activity_intervals_ns", {}
            ).items():
                semantic[name].extend(intervals)
        requests[key] = {
            "executed_step_ids_by_ready_request_membership": [
                step["step_id"] for step in steps if rid in step.get("request_ids", [])
            ],
            "executed_request_membership_definition": "ready request ID membership; uncommitted steps have unknown request generation",
            "measured_model_gpu_activity_union_ns": _duration(
                interval
                for step in window_steps
                for interval in step.get("gpu_activity_intervals_ns", [])
            ),
            "semantic_activity": _semantic_summary(semantic),
            "rid": rid,
            "generation": generation,
            "first_count": expected.get("first_count"),
            "last_count": expected.get("last_count"),
            "committed_endpoint_count": len(endpoints),
            "endpoints": endpoints,
            "consecutive_pairs": selected,
            "pair_count": len(selected),
            "status": "partial" if unknowns else "observed",
            "critical_unknowns": sorted(set(unknowns)),
            "summary_ns": {
                "gpu_period": _stats(pair["gpu_period_ns"] for pair in selected),
                "service_commit_period": _stats(
                    pair["service_commit_period_ns"] for pair in selected
                ),
                "gpu_activity_intersection": _stats(
                    pair["gpu_activity_intersection_ns"] for pair in selected
                ),
            },
            "service_window_status": "partial" if service_unknowns else "observed",
            "service_window_critical_unknowns": sorted(set(service_unknowns)),
            "service_tpot_ns": (
                (endpoints[-1]["observed_ns"] - endpoints[0]["observed_ns"])
                / (counts[-1] - counts[0])
                if not service_unknowns
                and len(endpoints) > 1
                and counts[-1] > counts[0]
                else None
            ),
            "service_tpot_definition": "authoritative host commit elapsed time / accepted token count between exact endpoints; not GPU start-start period",
        }
    return requests


def analyze_scheduler(
    database: Path,
    *,
    selection_mode: str,
    missing_launch_policy: str,
    registry: CostRegistry | None,
    process_id: int | None = None,
    request_label: str | None = None,
    replay_label: str | None = None,
    service_metadata: Path | None = None,
    captures: tuple[Path, ...] = (),
    completion: dict[str, Any] | None = None,
    whole_capture_source: str | None = None,
    report: Path | None = None,
    expected_database_sha256: str | None = None,
    expected_report_sha256: str | None = None,
    eager_steps: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Analyze scheduled launches with legacy NVTX or collector selection.

    Args:
        database: Nsight SQLite export, opened read-only.
        selection_mode: Explicit "legacy" NVTX selection or "collector" exact
            per-step labels and authoritative service commit selection.
        service_metadata: Collector mode metadata coverage JSON. Its existing
            source paths and hashes identify ready/window/raw service evidence.
            No request range or generic replay label is used in collector mode.
        process_id: OS PID of the worker whose scheduled replays are measured.
        request_label: Unique, complete CPU NVTX request marker; it may belong
            to a driver process distinct from the worker.
        replay_label: CPU NVTX marker enclosing each worker cudaGraphLaunch.
        missing_launch_policy: "error" rejects a marker lacking a launch;
            "report_unavailable" retains it with unavailable identity and timing.
            Multiple enclosed launches always fail as ambiguous.
        registry: Explicit trusted collector cost rules; None selects builtin
            costs only. Legacy mode requires None because it does not classify
            Op semantics. No registry/provider is loaded from trace metadata.
        whole_capture_source: Evidence that the entire export contains one
            request. Required when request_label is None; selects all worker
            replay markers without inventing a request wall-time boundary.
        captures: Same-run capture JSONs containing process_id, rank and
            graph_scopes[].graph_node_ids. Their explicit clone ancestry selects
            model activities; omission reports whole graph launches instead.
        completion: Optional request-level evidence with completed_verify_count,
            output_token_count and source. It never identifies individual replay
            completions. The caller must supply evidence from this request.
        report: Optional original Nsight report, recorded with its hash.
        expected_database_sha256: Optional expected input database digest.
        expected_report_sha256: Optional expected report digest; requires report.

    Returns:
        JSON-compatible per-replay span/sum/union, consecutive period and signed
        gap, gap coverage by every captured process/device/context/stream/type,
        statistics, identities, provenance and limitations. All times are ns.
        Coverage uses timestamp intersection only to measure busy time, never
        to attribute an activity to an op, rank, replay, or completion.
    """
    if missing_launch_policy not in ("error", "report_unavailable"):
        raise ValueError("Unknown missing launch policy")
    if selection_mode not in ("legacy", "collector"):
        raise ValueError("Select legacy or collector mode explicitly")
    collector = selection_mode == "collector"
    eager_steps = {} if eager_steps is None else eager_steps
    if eager_steps and not collector:
        raise ValueError("Eager projection requires collector scheduling")
    registry_selection = "builtin" if registry is None else "trusted_python_registry"
    if collector:
        from tokenspeed_kernel.op_profile import CostRegistry
        from tokenspeed_kernel.profile_graph_analysis import default_cost_registry

        if registry is None:
            registry = default_cost_registry(hbm_policy="unavailable")
        elif not isinstance(registry, CostRegistry):
            raise TypeError("registry must be a trusted CostRegistry or explicit None")
    elif registry is not None:
        raise ValueError("Legacy selection does not use cost rules; pass registry=None")
    metadata, manifest, ready, commits = None, None, {}, {}
    if collector:
        if (
            service_metadata is None
            or any(
                value is not None
                for value in (
                    request_label,
                    replay_label,
                    whole_capture_source,
                    completion,
                )
            )
            or captures
        ):
            raise ValueError(
                "Collector selection requires service metadata and no legacy selectors"
            )
        metadata, manifest, ready, commits = _collector_metadata(service_metadata)
        if eager_steps and set(eager_steps) != {
            key for key, row in ready.items() if row.get("eager_sidecar") is not None
        }:
            raise ValueError("Scheduler eager projection and ready step sets differ")
        pids = {row["process_id"] for row in ready.values()}
        if len(pids) != 1 or (process_id is not None and process_id not in pids):
            raise ValueError("Collector selection requires one unambiguous worker PID")
        process_id = pids.pop()
    elif service_metadata is not None or process_id is None or replay_label is None:
        raise ValueError(
            "Legacy mode requires process/replay selectors and no service metadata"
        )
    elif (request_label is None) == (whole_capture_source is None):
        raise ValueError("Select exactly one request label or whole-capture source")
    database_provenance = _digest(database)
    if (
        expected_database_sha256 is not None
        and database_provenance["sha256"] != expected_database_sha256
    ):
        raise ValueError("Database SHA-256 does not match expected provenance")
    report_provenance = _digest(report) if report is not None else None
    if expected_report_sha256 is not None and (
        report_provenance is None
        or report_provenance["sha256"] != expected_report_sha256
    ):
        raise ValueError("Report SHA-256 does not match expected provenance")
    prepared, ranks, capture_provenance = _captures(captures, process_id)
    collector_rows, collector_unknowns = {}, {}
    collector_captures, collector_sources = {}, {}
    if collector:
        ranks[process_id] = manifest["rank"]
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        schema = _schema(db)
        request = None
        if request_label is not None:
            requests = _nvtx(db, schema, request_label)
            if len(requests) != 1:
                raise ValueError(f"Expected one request marker, found {len(requests)}")
            request = requests[0]
            _complete_range(request)
        replays = []
        if collector:
            from tokenspeed_kernel.profile_graph_analysis import (
                _join,
                _read_native_index,
            )

            # Keep scheduled executions even when the collector has no ready receipt.
            _require(schema, "NVTX_EVENTS", ("text", "textId", "globalTid"))
            prefix = f"tokenspeed.profile.replay | {metadata['run_id']}:r{manifest['rank']}:s"
            ranges_by_label = defaultdict(list)
            for native in db.execute(
                "SELECT n.rowid AS range_id,n.start,n.end,n.endGlobalTid AS end_tid,"
                "n.globalTid,COALESCE(s.value,n.text) AS label FROM NVTX_EVENTS n "
                "LEFT JOIN StringIds s ON n.textId=s.id WHERE ((n.globalTid >> 24)&16777215)=?",
                (process_id,),
            ):
                label = native["label"]
                if not isinstance(label, str) or not label.startswith(prefix):
                    continue
                ranges_by_label[label].append(dict(native))
                number = label[len(prefix) :]
                if not number.isdecimal():
                    raise ValueError("Malformed collector replay step label")
                step = label.split(" | ", 1)[1]
                if step not in ready:
                    ready[step] = {
                        "step_id": step,
                        "run_id": metadata["run_id"],
                        "rank": manifest["rank"],
                        "process_id": process_id,
                        "replay_thread_id": native["globalTid"] & 16777215,
                        "replay_nvtx_label": label,
                        "serving_step": int(number),
                        "unavailable_reason": "no_device_ready_receipt",
                        "request_ids": [],
                        "capture_id": None,
                        "forward_id": None,
                        "event": "scheduled_without_device_ready_receipt",
                    }
            candidates = [
                row for row in ready.values() if row.get("replay_thread_id") is not None
            ]
            native_index = _read_native_index(db, candidates)
            for row in ready.values():
                step = row["step_id"]
                if row.get("eager_sidecar") is not None:
                    native_step = eager_steps.get(step)
                    if native_step is None:
                        collector_unknowns[step] = [
                            "eager_native_projection_requires_mixed_analysis"
                        ]
                        continue
                    if any(
                        native_step.get(key) != row.get(key)
                        for key in (
                            "run_id",
                            "rank",
                            "process_id",
                            "forward_id",
                            "request_ids",
                            "eager_sidecar",
                        )
                    ):
                        raise ValueError(
                            "Scheduler eager projection/ready identity mismatch"
                        )
                    timing = native_step["eager_native_timing"]
                    models = timing["models"]
                    if not models:
                        collector_unknowns[step] = [
                            "eager_model_native_range_unavailable"
                        ]
                        continue
                    first = min(models, key=lambda item: item["cpu_start_ns"])
                    replay = {
                        "range_id": first["nsight_range_id"],
                        "globalTid": timing["global_tid"],
                        "start": first["cpu_start_ns"],
                        "end": max(item["cpu_end_ns"] for item in models),
                        "eager_step_id": step,
                    }
                    if replay["range_id"] in collector_rows:
                        raise ValueError("Duplicate native scheduler range")
                    collector_rows[replay["range_id"]] = row
                    collector_sources[step] = {"eager_sidecar": row["eager_sidecar"]}
                    replays.append(replay)
                    continue
                activities, evidence, unknowns = (
                    [],
                    {},
                    ["replay_thread_identity_unavailable"],
                )
                if row.get("replay_thread_id") is not None:
                    activities, evidence, unknowns = _join(native_index, {}, row)
                fatal = [
                    reason
                    for reason in unknowns
                    if reason
                    not in (
                        "replay_thread_identity_unavailable",
                        "graph_launch_has_no_correlated_activities",
                        "missing_or_ambiguous_exact_replay_nvtx",
                        "missing_or_ambiguous_exact_cudaGraphLaunch",
                    )
                ]
                if fatal:
                    raise ValueError(f"Collector exact join failed for {step}: {fatal}")
                collector_unknowns[step] = unknowns + (
                    [row["unavailable_reason"]]
                    if row.get("unavailable_reason") is not None
                    else []
                )
                matches = native_index.ranges.get(
                    (process_id, row.get("replay_thread_id"), row["replay_nvtx_label"]),
                    [],
                )
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous exact collector range: {step}")
                if len(matches) == 1:
                    native = matches[0]
                    # Range IDs remain native row locators, never synthetic IDs.
                    ranges = ranges_by_label[row["replay_nvtx_label"]]
                    matches_native = [
                        r for r in ranges if r["globalTid"] == native.global_tid
                    ]
                    if len(matches_native) != 1:
                        raise ValueError(
                            "Ambiguous full-process collector range identity"
                        )
                    replay = dict(matches_native[0])
                    _complete_range(replay)
                else:
                    if (
                        missing_launch_policy == "error"
                        and row.get("unavailable_reason") is None
                    ):
                        raise ValueError(
                            f"Missing or ambiguous exact collector range: {step}"
                        )
                    # An unavailable device-ready record is still preserved below.
                    continue
                if replay["range_id"] in collector_rows:
                    raise ValueError("Duplicate native scheduler range")
                collector_rows[replay["range_id"]] = row
                replays.append(replay)
                if row.get("capture_file") and row.get("binary_payload"):
                    capture_path = Path(row["capture_file"])
                    capture = json.loads(capture_path.read_text())
                    if (
                        capture.get("schema_version") != 1
                        or capture.get("execution_mode") != "cuda_graph_capture"
                    ):
                        raise ValueError("Unsupported graph capture schema")
                    if any(
                        capture.get(name) != row.get(name)
                        for name in ("run_id", "rank", "process_id", "capture_id")
                    ):
                        raise ValueError("Collector capture identity mismatch")
                    collector_captures[step] = capture
                    collector_sources[step] = {
                        "capture_file": _digest(capture_path),
                        "binary_payload": _digest(
                            _verified_source(row["binary_payload"])
                        ),
                    }
            replays.sort(key=lambda row: (row["start"], row["range_id"]))
        else:
            for row in _nvtx(db, schema, replay_label):
                if ((row["globalTid"] >> 24) & 0xFFFFFF) != process_id:
                    continue
                if (
                    request is None
                    or request["start"] <= row["start"] <= request["end"]
                ):
                    _complete_range(row)
                    if request is not None and row["end"] > request["end"]:
                        raise ValueError("Replay NVTX range crosses request boundary")
                    replays.append(row)
        if not replays and (not collector or missing_launch_policy == "error"):
            raise ValueError("No worker replay markers within traced request")
        process_keys = {r["globalTid"] >> 24 for r in replays}
        if len(process_keys) > 1:
            raise ValueError("OS PID is ambiguous across captured process namespaces")
        process_key = process_keys.pop() if process_keys else None
        _require(
            schema,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            ("start", "end", "globalTid", "correlationId", "nameId"),
        )
        _require(schema, "StringIds", ("id", "value"))
        launches = []
        for replay in replays:
            if "eager_step_id" in replay:
                launches.append(None)
                continue
            matches = [
                row
                for row in db.execute(
                    "SELECT r.rowid AS runtime_id,r.start,r.end,r.correlationId,s.value "
                    "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
                    "WHERE r.globalTid=? AND r.start>=? AND r.end<=?",
                    (replay["globalTid"], replay["start"], replay["end"]),
                )
                if _GRAPH_LAUNCH.fullmatch(row["value"])
            ]
            if len(matches) > 1 or (not matches and missing_launch_policy == "error"):
                raise ValueError(
                    f"Replay {replay['range_id']} has {len(matches)} graph launches"
                )
            launches.append(matches[0] if matches else None)
        correlations = [row["correlationId"] for row in launches if row is not None]
        if len(set(correlations)) != len(correlations) or None in correlations:
            raise ValueError("Reused or missing graph launch correlation identity")

        clones = {}
        if prepared or (
            collector and any("eager_step_id" not in replay for replay in replays)
        ):
            _require(
                schema,
                "CUDA_GRAPH_NODE_EVENTS",
                ("graphNodeId", "originalGraphNodeId", "globalTid"),
            )
            for row in db.execute(
                "SELECT graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                "WHERE (globalTid >> 24)=? AND originalGraphNodeId IS NOT NULL AND originalGraphNodeId!=0",
                (process_key,),
            ):
                node, parent = row
                if node in clones and clones[node] != parent:
                    raise ValueError("Ambiguous graph clone ancestry")
                clones[node] = parent

        def original(node):
            seen = set()
            while node in clones:
                if node in seen:
                    raise ValueError("Cyclic graph clone ancestry")
                seen.add(node)
                node = clones[node]
            return node

        gpu_tables = _GPU_TABLES
        if eager_steps:
            from tokenspeed_kernel.profile_eager import GPU_TABLES

            gpu_tables = GPU_TABLES
        present = [table for table in gpu_tables if table in schema]
        if not present:
            raise ValueError("No captured CUDA kernel/memcpy/memset tables")
        by_correlation = defaultdict(list)
        placeholders = ",".join("?" for _ in correlations)

        def read_activity(table, where, parameters):
            columns = schema[table]
            _require(schema, table, ("start", "end", "globalPid", "correlationId"))
            fields = [
                name if name in columns else f"NULL AS {name}"
                for name in (
                    "start",
                    "end",
                    "globalPid",
                    "correlationId",
                    "graphNodeId",
                    "deviceId",
                    "contextId",
                    "streamId",
                )
            ]
            for row in db.execute(
                f"SELECT {','.join(fields)} FROM {table} WHERE {where}", parameters
            ):
                if (
                    row["end"] is None
                    or row["start"] is None
                    or row["end"] < row["start"]
                ):
                    raise ValueError(f"Invalid GPU activity duration in {table}")
                yield _Activity(
                    row["start"],
                    row["end"],
                    row["globalPid"] >> 24,
                    row["correlationId"],
                    row["graphNodeId"],
                    row["deviceId"],
                    row["contextId"],
                    row["streamId"],
                    gpu_tables[table],
                )

        for table in present:
            for activity in read_activity(
                table,
                f"(globalPid >> 24)=? AND correlationId IN ({placeholders})",
                (process_key, *correlations),
            ):
                by_correlation[activity.correlation].append(activity)
        steps = []
        semantic_groups = defaultdict(list)
        for index, (replay, launch) in enumerate(zip(replays, launches, strict=True)):
            if "eager_step_id" in replay:
                step_id = replay["eager_step_id"]
                row = collector_rows[replay["range_id"]]
                timing = eager_steps[step_id]["eager_native_timing"]
                selected = [
                    _Activity(
                        item["start"],
                        item["end"],
                        item["process_key"],
                        item["correlationId"],
                        None,
                        item["deviceId"],
                        item["contextId"],
                        item["streamId"],
                        item["kind"],
                    )
                    for item in timing["activities"]
                ]
                groups = timing["semantic_groups"]
                for name, intervals in groups.items():
                    semantic_groups[name].extend(intervals)
                token_commits = _token_commits(commits[step_id])
                steps.append(
                    {
                        "attribution_mode": "eager_nvtx_gpu_projection",
                        "scheduled_index": index,
                        "replay_marker_index": None,
                        "nvtx_range_id": replay["range_id"],
                        "eager_model_range_ids": [
                            item["nsight_range_id"] for item in timing["models"]
                        ],
                        "launch_identity_status": "identified_eager_model_ranges",
                        "global_tid": replay["globalTid"],
                        "process_key": process_key,
                        "process_id": process_id,
                        "rank": row["rank"],
                        "cpu_start_ns": replay["start"],
                        "cpu_end_ns": replay["end"],
                        "runtime_row_id": None,
                        "runtime_start_ns": None,
                        "runtime_end_ns": None,
                        "launch_correlation_id": None,
                        "capture_path": None,
                        "associated_model_source_node_count": None,
                        "graph_source_nodes_outside_model_count": None,
                        "whole_graph_launch": None,
                        "timing_scope": "projected_eager_model_ranges",
                        "step_id": step_id,
                        "forward_id": row.get("forward_id"),
                        "capture_id": None,
                        "serving_step": row["serving_step"],
                        "request_ids": row["request_ids"],
                        "scheduled": True,
                        "gpu_observed": bool(selected),
                        "device_ready": True,
                        "authoritative_commit": bool(token_commits),
                        "token_commits": token_commits,
                        "completion_status": (
                            "token_committed"
                            if token_commits
                            else (
                                "executed_without_observed_token_commit"
                                if selected
                                else "device_ready_without_observed_gpu_activity"
                            )
                        ),
                        "critical_unknowns": (
                            []
                            if selected
                            else ["eager_model_native_activity_unavailable"]
                        ),
                        "collector_sources": collector_sources[step_id],
                        "gpu_activity_intervals_ns": _merged(
                            (a.start, a.end) for a in selected
                        ),
                        "semantic_activity": _semantic_summary(groups),
                        "semantic_activity_intervals_ns": groups,
                        **_timing(selected),
                    }
                )
                continue
            activities = (
                by_correlation[launch["correlationId"]] if launch is not None else []
            )
            selected = [] if collector else activities
            collector_row = collector_rows.get(replay["range_id"])
            step_id = collector_row["step_id"] if collector_row else None
            capture_path = None
            source_nodes = {
                original(a.node) for a in activities if a.node not in (None, 0)
            }
            model_nodes = set()
            if collector and step_id in collector_captures:
                current_capture = collector_captures[step_id]
                model_nodes = {
                    node
                    for scope in current_capture["graph_scopes"]
                    for node in scope["graph_node_ids"]
                }
                observer = set(current_capture["observer_graph_node_ids"])
                model_nodes -= observer
                capture_path = collector_row["capture_file"]
                selected = [a for a in activities if original(a.node) in model_nodes]
            elif prepared and activities:
                matches = [
                    (path, nodes) for path, nodes in prepared if nodes & source_nodes
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Replay cannot uniquely match same-run model capture nodes"
                    )
                capture_path, model_nodes = matches[0]
                selected = [a for a in activities if original(a.node) in model_nodes]
            steps.append(
                {
                    "scheduled_index": index if launch is not None else None,
                    "replay_marker_index": index,
                    "nvtx_range_id": replay["range_id"],
                    "launch_identity_status": (
                        "identified"
                        if launch is not None
                        else "unavailable: no enclosed same-thread graph launch in export"
                    ),
                    "global_tid": replay["globalTid"],
                    "process_key": process_key,
                    "process_id": process_id,
                    "rank": ranks.get(process_id),
                    "cpu_start_ns": replay["start"],
                    "cpu_end_ns": replay["end"],
                    "runtime_row_id": (
                        launch["runtime_id"] if launch is not None else None
                    ),
                    "runtime_start_ns": launch["start"] if launch is not None else None,
                    "runtime_end_ns": launch["end"] if launch is not None else None,
                    "launch_correlation_id": (
                        launch["correlationId"] if launch is not None else None
                    ),
                    "capture_path": capture_path,
                    "associated_model_source_node_count": len(
                        source_nodes & model_nodes
                    ),
                    "graph_source_nodes_outside_model_count": (
                        len(source_nodes - model_nodes)
                        if prepared or collector
                        else None
                    ),
                    "completion_status": "not_identified_by_trace",
                    "timing_scope": (
                        "captured_model_nodes"
                        if model_nodes
                        else (
                            "model_boundary_unavailable"
                            if collector
                            else "whole_graph_launch"
                        )
                    ),
                    "whole_graph_launch": _timing(activities),
                    **_timing(selected),
                }
            )
            steps[-1]["gpu_activity_intervals_ns"] = _merged(
                (a.start, a.end) for a in selected
            )
            if collector:
                token_commits = _token_commits(commits[step_id])
                classified = {}
                if step_id in collector_captures:
                    observer = set(
                        collector_captures[step_id]["observer_graph_node_ids"]
                    )
                    classified = _semantic_intervals(
                        collector_captures[step_id],
                        collector_row,
                        [
                            {"start": a.start, "end": a.end, "source": original(a.node)}
                            for a in activities
                            if original(a.node) not in observer
                        ],
                        registry=registry,
                    )
                    for name, intervals in classified.items():
                        semantic_groups[name].extend(intervals)
                else:
                    classified = {
                        "unclassified": [(a.start, a.end) for a in activities]
                    }
                    semantic_groups["unclassified"].extend(classified["unclassified"])
                if eager_steps:
                    steps[-1]["attribution_mode"] = "cuda_graph_node_ids"
                steps[-1].update(
                    step_id=step_id,
                    forward_id=collector_row.get("forward_id"),
                    capture_id=collector_row.get("capture_id"),
                    serving_step=collector_row.get("serving_step"),
                    request_ids=collector_row.get("request_ids", []),
                    scheduled=launch is not None,
                    gpu_observed=bool(activities),
                    device_ready=collector_row["event"] == "device_results_ready",
                    authoritative_commit=bool(token_commits),
                    token_commits=token_commits,
                    completion_status=(
                        "token_committed"
                        if token_commits
                        else (
                            "executed_without_observed_token_commit"
                            if activities
                            else (
                                "device_ready_without_observed_gpu_activity"
                                if collector_row["event"] == "device_results_ready"
                                else "scheduled_without_observed_gpu_activity"
                            )
                        )
                    ),
                    critical_unknowns=collector_unknowns[step_id],
                    collector_sources=collector_sources.get(step_id, {}),
                    semantic_activity=_semantic_summary(classified),
                    semantic_activity_intervals_ns=classified,
                )
        timed = [s for s in steps if s["gpu_start_ns"] is not None]
        if not timed and missing_launch_policy == "error":
            raise ValueError("No GPU activities have selected launch identities")
        window_start = min((s["gpu_start_ns"] for s in timed), default=None)
        window_end = max((s["gpu_end_ns"] for s in timed), default=None)
        # Temporal intersection below measures coverage only. Identity was fixed
        # above by same-thread runtime correlation and explicit capture node IDs.
        groups = defaultdict(list)
        for table in present if timed else []:
            for a in read_activity(
                table, "end>? AND start<?", (window_start, window_end)
            ):
                groups[(a.process_key, a.device, a.context, a.stream, a.kind)].append(
                    (a.start, a.end)
                )
        coverage = {key: _Coverage(intervals) for key, intervals in groups.items()}
        all_coverage = _Coverage(
            interval for intervals in groups.values() for interval in intervals
        )
        kind_coverage = {
            kind: _Coverage(
                interval
                for key, intervals in groups.items()
                if key[4] == kind
                for interval in intervals
            )
            for kind in {key[4] for key in groups}
        }
        worker_coverage = _Coverage(
            interval
            for key, intervals in groups.items()
            if key[0] == process_key
            for interval in intervals
        )
        other_coverage = _Coverage(
            interval
            for key, intervals in groups.items()
            if key[0] != process_key
            for interval in intervals
        )
        device_coverage = {
            device: _Coverage(
                interval
                for key, intervals in groups.items()
                if key[1] == device
                for interval in intervals
            )
            for device in {key[1] for key in groups}
        }
        gaps = []
        for previous, following in zip(steps, steps[1:]):
            if collector:
                previous_keys = {
                    (c["rid"], c["generation"]) for c in previous["token_commits"]
                }
                following_keys = {
                    (c["rid"], c["generation"]) for c in following["token_commits"]
                }
                if (
                    not previous_keys & following_keys
                    or following["serving_step"] != previous["serving_step"] + 1
                ):
                    continue
            if previous["gpu_end_ns"] is None or following["gpu_start_ns"] is None:
                continue
            start, end = previous["gpu_end_ns"], following["gpu_start_ns"]
            gap = end - start
            detail = []
            if gap > 0:
                for key, series in coverage.items():
                    covered = series.within(start, end)
                    if covered:
                        pkey, device, context, stream, kind = key
                        pid = pkey & 0xFFFFFF
                        detail.append(
                            {
                                "process_key": pkey,
                                "process_id": pid,
                                "rank": ranks.get(pid) if pkey == process_key else None,
                                "device_id": device,
                                "context_id": context,
                                "stream_id": stream,
                                "activity_kind": kind,
                                "activity_union_ns": covered,
                            }
                        )
            total = all_coverage.within(start, end) if gap > 0 else None
            gaps.append(
                {
                    "previous_scheduled_index": previous["scheduled_index"],
                    "next_scheduled_index": following["scheduled_index"],
                    "gpu_period_ns": end - previous["gpu_start_ns"],
                    "signed_gap_ns": gap,
                    "positive_gap_ns": max(gap, 0),
                    "overlap_ns": max(-gap, 0),
                    "span_overlap_ns": max(
                        0,
                        min(previous["gpu_end_ns"], following["gpu_end_ns"])
                        - max(previous["gpu_start_ns"], following["gpu_start_ns"]),
                    ),
                    "gpu_activity_intersection_ns": _duration(
                        _intersection(
                            previous["gpu_activity_intervals_ns"],
                            following["gpu_activity_intervals_ns"],
                        )
                    ),
                    "start_ns": start,
                    "end_ns": end,
                    "all_captured_activity_union_ns": total,
                    "no_captured_activity_ns": (
                        gap - total if total is not None else None
                    ),
                    "selected_worker_activity_union_ns": (
                        worker_coverage.within(start, end) if gap > 0 else None
                    ),
                    "other_process_activity_union_ns": (
                        other_coverage.within(start, end) if gap > 0 else None
                    ),
                    "by_activity_kind": (
                        {
                            kind: series.within(start, end)
                            for kind, series in kind_coverage.items()
                        }
                        if gap > 0
                        else {}
                    ),
                    "by_device": (
                        [
                            {
                                "device_id": device,
                                "activity_union_ns": series.within(start, end),
                                "no_captured_activity_ns": gap
                                - series.within(start, end),
                            }
                            for device, series in device_coverage.items()
                        ]
                        if gap > 0
                        else []
                    ),
                    "by_process_device_stream_kind": detail,
                }
            )
            if collector:
                gaps[-1].update(
                    previous_step_id=previous["step_id"],
                    next_step_id=following["step_id"],
                )
    if collector:
        represented = {step["step_id"] for step in steps}
        for step_id, row in ready.items():
            if step_id in represented:
                continue
            steps.append(
                {
                    "step_id": step_id,
                    "forward_id": row.get("forward_id"),
                    "capture_id": row.get("capture_id"),
                    "serving_step": row.get("serving_step"),
                    "request_ids": row.get("request_ids", []),
                    "scheduled": None,
                    "gpu_observed": None,
                    "device_ready": True,
                    "authoritative_commit": bool(commits[step_id]),
                    "completion_status": (
                        "token_committed"
                        if commits[step_id]
                        else "collector_metadata_unavailable"
                    ),
                    "token_commits": _token_commits(commits[step_id]),
                    "collector_sources": {},
                    "critical_unknowns": collector_unknowns.get(step_id, [])
                    + [row.get("unavailable_reason") or "collector_range_unavailable"],
                    "gpu_start_ns": None,
                    "gpu_end_ns": None,
                }
            )
        steps.sort(key=lambda row: row["serving_step"])
    completion_evidence = None
    if completion is not None:
        required = {"completed_verify_count", "output_token_count", "source"}
        if not required <= completion.keys() or not completion["source"]:
            raise ValueError("Completion evidence requires counts and source")
        for name in ("completed_verify_count", "output_token_count"):
            if type(completion[name]) is not int or completion[name] < 0:
                raise ValueError("Completion counts must be nonnegative integers")
        completion_evidence = {
            **completion,
            "scheduled_minus_completed_verify_count": (
                len(steps) - completion["completed_verify_count"]
                if len(correlations) == len(steps)
                else None
            ),
            "per_replay_completion_identity": "unavailable; aggregate counts do not identify a replay",
        }
    graph_marker_count = sum("eager_step_id" not in row for row in replays)
    eager_marker_count = len(replays) - graph_marker_count
    gpu_candidates = sorted(
        name
        for name, columns in schema.items()
        if name.startswith("CUPTI_ACTIVITY_KIND_")
        and {"globalPid", "deviceId", "streamId", "start", "end"} <= columns
    )
    result = {
        "schema_version": 2,
        "selection_mode": selection_mode,
        "timing_scope": (
            "captured_model_nodes_and_projected_eager_model_ranges"
            if eager_steps
            else (
                "captured_model_nodes"
                if prepared or collector
                else "whole_graph_launch; model boundary unavailable"
            )
        ),
        "identity_method": (
            "exact Graph launch/clone identities and eager NVTX scopes with same-thread CUDA correlations"
            if eager_steps
            else "NVTX same globalTid -> enclosed cudaGraphLaunch -> correlationId + full process key; model uses explicit graph clone ancestry"
        ),
        "request": {
            "selection": (
                "collector_per_step_identity"
                if collector
                else "nvtx_marker" if request is not None else "whole_capture"
            ),
            "label": request_label,
            "source": whole_capture_source,
            "nvtx_range_id": request["range_id"] if request is not None else None,
            "global_tid": request["globalTid"] if request is not None else None,
            "start_ns": request["start"] if request is not None else None,
            "end_ns": request["end"] if request is not None else None,
            "replay_cpu_envelope_start_ns": min(
                (s["cpu_start_ns"] for s in steps if s.get("cpu_start_ns") is not None),
                default=None,
            ),
            "replay_cpu_envelope_end_ns": max(
                (s["cpu_end_ns"] for s in steps if s.get("cpu_end_ns") is not None),
                default=None,
            ),
        },
        "replay_marker_count": graph_marker_count,
        **(
            {
                "eager_marker_count": eager_marker_count,
                "gpu_timed_step_count": len(timed),
            }
            if eager_steps
            else {}
        ),
        "device_ready_step_count": (
            sum(row["event"] == "device_results_ready" for row in ready.values())
            if collector
            else None
        ),
        "identified_graph_launch_count": len(correlations),
        "scheduled_replay_count": (
            graph_marker_count if len(correlations) == graph_marker_count else None
        ),
        "missing_launch_identity_count": graph_marker_count - len(correlations),
        "statistics_coverage": (
            "complete_for_selected_markers"
            if len(timed) == len(steps)
            else "partial; unidentified or GPU-untimed markers excluded without bridging pairs"
        ),
        "gpu_timed_replay_count": sum(
            step.get("attribution_mode") != "eager_nvtx_gpu_projection"
            for step in timed
        ),
        "completion": completion_evidence,
        "steps": steps,
        "consecutive_pairs": gaps,
        "summary_ns": {
            "gpu_span": _stats(s["gpu_span_ns"] for s in timed),
            "gpu_activity_sum": _stats(s["gpu_activity_sum_ns"] for s in timed),
            "gpu_activity_union": _stats(s["gpu_activity_union_ns"] for s in timed),
            "gpu_period": _stats(g["gpu_period_ns"] for g in gaps),
            "span_overlap": _stats(g["span_overlap_ns"] for g in gaps),
            "gpu_activity_intersection": _stats(
                g["gpu_activity_intersection_ns"] for g in gaps
            ),
            "signed_gap": _stats(g["signed_gap_ns"] for g in gaps),
            "positive_gap": _stats(
                g["positive_gap_ns"] for g in gaps if g["signed_gap_ns"] > 0
            ),
            "no_captured_activity_in_positive_gap": _stats(
                g["no_captured_activity_ns"] for g in gaps if g["signed_gap_ns"] > 0
            ),
        },
        "coverage": {
            "activity_tables": present,
            "missing_activity_tables": sorted(set(gpu_tables) - set(present)),
            "unhandled_gpu_activity_tables": sorted(set(gpu_candidates) - set(present)),
            "captured_devices": sorted(
                (device for device in device_coverage if device is not None)
            ),
            "captured_process_keys": sorted({key[0] for key in groups}),
            "known_ranks_by_process_id": ranks,
            "semantic_activity_classification": "unclassified; activity kind and process/stream identity do not establish state update, sampling, communication, or wait",
            "window_start_ns": window_start,
            "window_end_ns": window_end,
            "observed_streams": [
                {
                    "process_key": key[0],
                    "process_id": key[0] & 0xFFFFFF,
                    "device_id": key[1],
                    "context_id": key[2],
                    "stream_id": key[3],
                    "activity_kind": key[4],
                    "activity_count": len(intervals),
                }
                for key, intervals in groups.items()
            ],
        },
        "provenance": {
            "database": database_provenance,
            "report": report_provenance,
            "captures": capture_provenance,
        },
        "limitations": [
            "Scheduled replay and captured GPU completion are not completed request verify steps; aggregate completion evidence cannot label individual replays.",
            "No captured activity means only no kernel/memcpy/memset evidence in this export; GPU physical idle is not established.",
            "Coverage intersects timestamps across all captured processes and streams; it never assigns semantic op/rank identity by time or kernel name.",
            "Activity union is wall-time coverage, including across devices; sum may double-count concurrency. Per-device coverage is separate.",
            "Same-run capture provenance is a caller contract, checked against PID and unique explicit graph node ancestry; no cross-run mapping is inferred.",
            "Unknown process ranks remain unknown; no rank is inferred from PID order or device number.",
            "Diagnostic capture timing must not replace the original low-interference performance baseline.",
        ],
    }

    if collector:
        result["provenance"]["cost_provider"] = {"selection": registry_selection}
        result["requests"] = _collector_requests(metadata, steps, gaps)
        result["semantic_activity"] = _semantic_summary(semantic_groups)
        result["coverage"]["semantic_activity_classification"] = (
            "unique Graph Op node ownership and exact eager Op activity ownership; unknown ownership/cost remains unclassified"
            if eager_steps
            else "unique captured Op node ownership and trusted costs; unknown ownership/cost remains unclassified"
        )
        result["provenance"]["service_metadata"] = _digest(service_metadata)
        result["provenance"]["service_sources"] = metadata["service_sources"]
        result["provenance"]["ready_index"] = metadata["ready_index_source"]
        result["provenance"]["window"] = metadata["window_manifest_source"]
        result["provenance"]["collector_sources"] = collector_sources
        result["collector_window"] = manifest
        result["critical_unknowns"] = sorted(
            set(
                [
                    reason
                    for step in steps
                    for reason in step.get("critical_unknowns", [])
                ]
                + (
                    []
                    if manifest.get("writer_flushed") is True
                    and manifest.get("writer_error") is None
                    and manifest.get("unresolved_result_metadata") == 0
                    else ["collector_window_incomplete"]
                )
            )
        )
        result["limitations"][
            0
        ] = "Authoritative token commits join exact run/rank/request/generation/ForwardId/step; missing commit observation does not prove never committed. Executed FLOPs are never scaled by accepted tokens."
        result["limitations"].append(
            "GPU start-start periods are not service TPOT; span overlap is distinct from activity-union intersection. Communication outside classified compute is coverage, not a causal exposed critical path."
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--selection-mode", choices=("legacy", "collector"), required=True
    )
    parser.add_argument("--service-metadata", type=Path)
    parser.add_argument(
        "--cost-provider",
        help="Explicit collector cost provider: builtin or installed module:function",
    )
    parser.add_argument("--process-id", type=int)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--request-label")
    selection.add_argument("--whole-capture-source")
    parser.add_argument("--replay-label")
    parser.add_argument(
        "--missing-launch-policy",
        choices=("error", "report_unavailable"),
        required=True,
    )
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--completion", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-database-sha256")
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry, provider_source = None, None
    if args.selection_mode == "collector":
        if args.cost_provider is None:
            parser.error("Collector mode requires explicit --cost-provider")
        from tokenspeed_kernel.profile_nsys import _load_cost_provider

        try:
            registry, provider_source = _load_cost_provider(
                args.cost_provider, hbm_policy="unavailable"
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.cost_provider is not None:
        parser.error("--cost-provider requires collector selection")
    result = analyze_scheduler(
        args.database,
        selection_mode=args.selection_mode,
        service_metadata=args.service_metadata,
        process_id=args.process_id,
        request_label=args.request_label,
        replay_label=args.replay_label,
        captures=tuple(args.capture),
        missing_launch_policy=args.missing_launch_policy,
        registry=registry,
        completion=json.loads(args.completion.read_text()) if args.completion else None,
        whole_capture_source=args.whole_capture_source,
        report=args.report,
        expected_database_sha256=args.expected_database_sha256,
        expected_report_sha256=args.expected_report_sha256,
    )
    if provider_source is not None:
        result["provenance"]["cost_provider"] = provider_source
    result["provenance"]["completion_file"] = (
        _digest(args.completion) if args.completion else None
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "scheduled": result["scheduled_replay_count"],
                "summary_ns": result["summary_ns"],
            }
        )
    )


if __name__ == "__main__":
    main()
