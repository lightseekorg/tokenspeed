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

"""Cost and timing aggregation shared by metadata export and NVTX labels."""

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
    scopes = {scope["trace_label"]: scope for scope in capture["scopes"]}
    if len(scopes) != len(capture["scopes"]):
        raise ValueError("Duplicate metadata scope identity")
    labels = {scope["nvtx_label"] for scope in scopes.values()}
    use_payload = all("nvtx_id" in scope for scope in scopes.values())
    identities = {
        scope["nvtx_id"] if use_payload else scope["nvtx_label"]
        for scope in scopes.values()
    }
    if len(identities) != len(scopes):
        raise ValueError("NVTX identities must uniquely identify calls in this session")
    with projection.open(newline="") as source:
        projected = {}
        for row in csv.DictReader(source):
            identity = int(row["RangeId"])
            if identity in projected:
                raise ValueError("Duplicate Nsight projection range")
            projected[identity] = row
    ranges = {}
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        for identity, start, end, tid, end_tid, label, payload in db.execute(
            "SELECT n.rowid,n.start,n.end,n.globalTid,n.endGlobalTid,"
            "COALESCE(s.value,n.text),n.uint64Value FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId=s.id"
        ):
            if label not in labels or tid is None:
                continue
            key = payload if use_payload else label
            if key not in identities:
                continue
            pid = (tid >> 24) & 0xFFFFFF
            if tid & 0xFFFFFF != capture["thread_id"]:
                continue
            if capture.get("process_id") is not None and pid != capture["process_id"]:
                continue
            if key in ranges:
                raise ValueError(f"Ambiguous NVTX range for {label}")
            if end is None or end < start or end_tid not in (None, tid):
                raise ValueError(f"Incomplete or cross-thread NVTX range for {label}")
            ranges[key] = (identity, start, end, pid, label)
    timings = []
    evidence = {}
    ops = {op["trace_label"]: op for op in report["ops"]}
    for trace_label, scope in scopes.items():
        label = scope["nvtx_label"]
        key = scope["nvtx_id"] if use_payload else label
        if key not in ranges:
            raise ValueError(f"NVTX range missing from Nsight capture: {label}")
        identity, start, end, pid, native_label = ranges[key]
        if native_label != label:
            raise ValueError("NVTX payload identity has an unexpected label")
        gpu = projected.get(identity)
        duration = 0
        gpu_start = None
        count = 0
        reason = "no GPU activity correlated by Nsight to this NVTX range"
        if gpu is not None:
            if (
                int(gpu["Orig Start (ns)"]) != start
                or int(gpu["Orig Duration (ns)"]) != end - start
                or int(gpu["PID"]) != pid
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
            "nsight_range_id": identity,
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
