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

This module uses only the standard library and can also run as a script without
importing the kernel package (or requiring CUDA).
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
from typing import Any

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


def analyze_scheduler(
    database: Path,
    *,
    process_id: int,
    request_label: str | None,
    replay_label: str,
    missing_launch_policy: str,
    captures: tuple[Path, ...] = (),
    completion: dict[str, Any] | None = None,
    whole_capture_source: str | None = None,
    report: Path | None = None,
    expected_database_sha256: str | None = None,
    expected_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Analyze one traced request using exact CUDA graph launch identities.

    Args:
        database: Nsight SQLite export, opened read-only.
        process_id: OS PID of the worker whose scheduled replays are measured.
        request_label: Unique, complete CPU NVTX request marker; it may belong
            to a driver process distinct from the worker.
        replay_label: CPU NVTX marker enclosing each worker cudaGraphLaunch.
        missing_launch_policy: "error" rejects a marker lacking a launch;
            "report_unavailable" retains it with unavailable identity and timing.
            Multiple enclosed launches always fail as ambiguous.
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
    if (request_label is None) == (whole_capture_source is None):
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
        for row in _nvtx(db, schema, replay_label):
            if ((row["globalTid"] >> 24) & 0xFFFFFF) != process_id:
                continue
            if request is None or request["start"] <= row["start"] <= request["end"]:
                _complete_range(row)
                if request is not None and row["end"] > request["end"]:
                    raise ValueError("Replay NVTX range crosses request boundary")
                replays.append(row)
        if not replays:
            raise ValueError("No worker replay markers within traced request")
        process_keys = {r["globalTid"] >> 24 for r in replays}
        if len(process_keys) != 1:
            raise ValueError("OS PID is ambiguous across captured process namespaces")
        process_key = process_keys.pop()
        _require(
            schema,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            ("start", "end", "globalTid", "correlationId", "nameId"),
        )
        _require(schema, "StringIds", ("id", "value"))
        launches = []
        for replay in replays:
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
        if prepared:
            _require(
                schema,
                "CUDA_GRAPH_NODE_EVENTS",
                ("graphNodeId", "originalGraphNodeId", "globalTid"),
            )
            for row in db.execute(
                "SELECT graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                "WHERE (globalTid >> 24)=? AND originalGraphNodeId IS NOT NULL",
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

        present = [table for table in _GPU_TABLES if table in schema]
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
                    _GPU_TABLES[table],
                )

        for table in present:
            for activity in read_activity(
                table,
                f"(globalPid >> 24)=? AND correlationId IN ({placeholders})",
                (process_key, *correlations),
            ):
                by_correlation[activity.correlation].append(activity)
        steps = []
        for index, (replay, launch) in enumerate(zip(replays, launches, strict=True)):
            activities = (
                by_correlation[launch["correlationId"]] if launch is not None else []
            )
            selected = activities
            capture_path = None
            source_nodes = {
                original(a.node) for a in activities if a.node not in (None, 0)
            }
            model_nodes = set()
            if prepared and activities:
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
                        len(source_nodes - model_nodes) if prepared else None
                    ),
                    "completion_status": "not_identified_by_trace",
                    "whole_graph_launch": _timing(activities),
                    **_timing(selected),
                }
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
    gpu_candidates = sorted(
        name
        for name, columns in schema.items()
        if name.startswith("CUPTI_ACTIVITY_KIND_")
        and {"globalPid", "deviceId", "streamId", "start", "end"} <= columns
    )
    return {
        "schema_version": 1,
        "timing_scope": (
            "captured_model_nodes"
            if prepared
            else "whole_graph_launch; model boundary unavailable"
        ),
        "identity_method": "NVTX same globalTid -> enclosed cudaGraphLaunch -> correlationId + full process key; model uses explicit graph clone ancestry",
        "request": {
            "selection": "nvtx_marker" if request is not None else "whole_capture",
            "label": request_label,
            "source": whole_capture_source,
            "nvtx_range_id": request["range_id"] if request is not None else None,
            "global_tid": request["globalTid"] if request is not None else None,
            "start_ns": request["start"] if request is not None else None,
            "end_ns": request["end"] if request is not None else None,
            "replay_cpu_envelope_start_ns": min(s["cpu_start_ns"] for s in steps),
            "replay_cpu_envelope_end_ns": max(s["cpu_end_ns"] for s in steps),
        },
        "replay_marker_count": len(steps),
        "identified_graph_launch_count": len(correlations),
        "scheduled_replay_count": (
            len(steps) if len(correlations) == len(steps) else None
        ),
        "missing_launch_identity_count": len(steps) - len(correlations),
        "statistics_coverage": (
            "complete_for_selected_markers"
            if len(timed) == len(steps)
            else "partial; unidentified or GPU-untimed markers excluded without bridging pairs"
        ),
        "gpu_timed_replay_count": len(timed),
        "completion": completion_evidence,
        "steps": steps,
        "consecutive_pairs": gaps,
        "summary_ns": {
            "gpu_span": _stats(s["gpu_span_ns"] for s in timed),
            "gpu_activity_sum": _stats(s["gpu_activity_sum_ns"] for s in timed),
            "gpu_activity_union": _stats(s["gpu_activity_union_ns"] for s in timed),
            "gpu_period": _stats(g["gpu_period_ns"] for g in gaps),
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
            "missing_activity_tables": sorted(set(_GPU_TABLES) - set(present)),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--request-label")
    selection.add_argument("--whole-capture-source")
    parser.add_argument("--replay-label", required=True)
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
    result = analyze_scheduler(
        args.database,
        process_id=args.process_id,
        request_label=args.request_label,
        replay_label=args.replay_label,
        captures=tuple(args.capture),
        missing_launch_policy=args.missing_launch_policy,
        completion=json.loads(args.completion.read_text()) if args.completion else None,
        whole_capture_source=args.whole_capture_source,
        report=args.report,
        expected_database_sha256=args.expected_database_sha256,
        expected_report_sha256=args.expected_report_sha256,
    )
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
