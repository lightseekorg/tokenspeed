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
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tokenspeed_kernel.profile_utilization import (
    HardwareProfile,
    ScopeTiming,
    analyze_utilization,
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
    identity_field: str,
) -> dict[int | str, sqlite3.Row]:
    """Read this session's ranges, excluding other processes and threads."""
    ranges = {}
    labels = {scope["nvtx_label"] for scope in capture["scopes"]}
    process_id = capture.get("process_id")
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        # Nsight packs PID/TID into 24-bit fields within globalTid.
        for row in db.execute(
            """
            SELECT n.rowid AS range_id, n.start, n.end, n.globalTid, n.endGlobalTid,
                   ((n.globalTid >> 24) & 0xFFFFFF) AS pid,
                   COALESCE(s.value, n.text) AS nvtx_label,
                   n.uint64Value AS nvtx_id
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
            WHERE (n.globalTid & 0xFFFFFF) = ?
              AND (? IS NULL OR ((n.globalTid >> 24) & 0xFFFFFF) = ?)
            """,
            (capture["thread_id"], process_id, process_id),
        ):
            scope_id = row[identity_field]
            if scope_id not in scope_ids or row["nvtx_label"] not in labels:
                continue
            label = row["nvtx_label"]
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
    recorded = capture["scopes"]
    if len({scope["trace_label"] for scope in recorded}) != len(recorded):
        raise ValueError("Duplicate metadata scope identity")
    # New captures carry payload IDs; historical captures used unique labels.
    identity_field = (
        "nvtx_id" if all("nvtx_id" in scope for scope in recorded) else "nvtx_label"
    )
    scopes = {scope[identity_field]: scope for scope in recorded}
    if len(scopes) != len(recorded):
        raise ValueError("NVTX identities must uniquely identify calls in this session")
    projected = _read_projection(projection)
    ranges = _read_nvtx_ranges(
        database, capture=capture, scope_ids=set(scopes), identity_field=identity_field
    )
    timings = []
    evidence = {}
    ops = {op["trace_label"]: op for op in report["ops"]}
    for scope_id, scope in scopes.items():
        trace_label, label = scope["trace_label"], scope["nvtx_label"]
        if scope_id not in ranges:
            raise ValueError(f"NVTX range missing from Nsight capture: {label}")
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
            if (
                int(gpu["Orig Start (ns)"]) != start
                or int(gpu["Orig Duration (ns)"]) != end - start
                or int(gpu["PID"]) != native["pid"]
                or int(gpu["TID"]) != capture["thread_id"]
                or gpu["Name"].split(":", 1)[-1] != label
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
    rows = analyze_utilization(report, timings, HardwareProfile(**report["hardware"]))
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
    *, nsys: str, trace: Path, metadata: Path, output: Path
) -> dict[str, Any]:
    """Export a finished nsys-rep and write metrics without modifying the capture.

    Intermediate SQLite/CSV exports live in a temporary directory. Output is a
    new metrics JSON; its native_nvtx_label links every row to the original
    timeline. Computed display labels do not rewrite the original NVTX events.
    """
    if output.resolve() in (metadata.resolve(), trace.resolve()):
        raise ValueError("Output must not overwrite metadata or the Nsight capture")
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
            json.loads(metadata.read_text()),
            database=database,
            projection=root / "projection_nvtx_gpu_proj_trace.csv",
        )
    result["device_analysis"]["source_report"] = str(trace.resolve())
    result["device_analysis"]["source_metadata"] = str(metadata.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate utilization from Nsight GPU timestamps"
    )
    parser.add_argument(
        "--nsys", default="nsys", help="Nsight CLI matching the capture version"
    )
    parser.add_argument(
        "--report", type=Path, required=True, help="Completed .nsys-rep"
    )
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Runtime cost metadata JSON"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New Nsight-timed metrics JSON"
    )
    args = parser.parse_args()
    finalize_nsight(
        nsys=args.nsys, trace=args.report, metadata=args.metadata, output=args.output
    )


if __name__ == "__main__":
    main()
