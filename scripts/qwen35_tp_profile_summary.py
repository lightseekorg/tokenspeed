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

"""Separate device compute, communication, overlap and idle with interval unions."""

import argparse
import collections
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from qwen35_profile_summary import family


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative-duration interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(b - a for a, b in merge_intervals(intervals))


def summarize(path, root):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    complete = [e for e in events if e.get("ph") == "X"]
    computing = [e for e in complete if e.get("name") == "Computing"]
    communication = [e for e in complete if e.get("name") == "Communication"]
    assert computing, f"No device computing intervals: {path}"
    origin = min(Decimal(str(e["ts"])) for e in computing + communication)

    def intervals(group):
        result = []
        for e in group:
            start = float(Decimal(str(e["ts"])) - origin) / 1000
            result.append((start, start + float(e["dur"]) / 1000))
        return result

    ci, ki = intervals(computing), intervals(communication)
    c, k, active = duration(ci), duration(ki), duration(ci + ki)
    span = max(end for start, end in ci + ki)
    relative = path.relative_to(root)
    request_dir = next(p for p in path.parents if p.name.startswith("length-"))
    profile_dir = next(p for p in path.parents if p.name.startswith("profile-"))
    step = int(profile_dir.name.split("-")[-1])
    records = [
        json.loads(line)
        for line in (request_dir / "steps.jsonl").read_text().splitlines()
    ]
    forward = next(r for r in records if r["step"] == step)
    kernel_groups = collections.defaultdict(list)
    for e in complete:
        task_type = e.get("args", {}).get("Task Type", "")
        if task_type.startswith("KERNEL") or task_type in {
            "AI_CORE",
            "AI_VECTOR_CORE",
            "AI_CPU",
            "MIX_AIV",
            "MIX_AIC",
        }:
            kernel_groups[family(e["name"])].append(e)
    hccl = []
    communication_path = path.parent / "communication.json"
    if communication_path.exists():
        for step_data in json.loads(communication_path.read_text()).values():
            for name, operation in step_data.get("collective", {}).items():
                if not name.startswith("hcom_") or "total" in name.lower():
                    continue
                info = operation.get("Communication Time Info", {})
                sdma = operation.get("Communication Bandwidth Info", {}).get("SDMA", {})
                hccl.append(
                    dict(
                        name=name,
                        elapsed_ms=info.get("Elapse Time(ms)", 0),
                        wait_ms=info.get("Wait Time(ms)", 0),
                        synchronization_ms=info.get("Synchronization Time(ms)", 0),
                        sdma_copy_time_ms=sdma.get("Transit Time(ms)", 0),
                        sdma_copy_MB=sdma.get("Transit Size(MB)", 0),
                    )
                )
    collective_summary = collections.defaultdict(
        lambda: dict(calls=0, logical_input_bytes=0)
    )
    for operation in forward["collectives"]:
        entry = collective_summary[operation["name"]]
        entry["calls"] += 1
        entry["logical_input_bytes"] += operation["payload_bytes"]
    return dict(
        trace=str(path),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rank=next((p for p in relative.parts if p.startswith("rank-")), "rank-0"),
        request=request_dir.name,
        step=step,
        mode=forward["mode"],
        tokens=forward["tokens"],
        batch_size=forward["batch_size"],
        diagnostic_forward_wall_ms=forward["wall_ms"],
        diagnostic_event_span_ms=forward["event_span_ms"],
        compute_ms=c,
        communication_ms=k,
        overlap_ms=c + k - active,
        exposed_communication_ms=active - c,
        device_active_union_ms=active,
        device_first_to_last_ms=span,
        idle_between_tasks_ms=max(0.0, span - active),
        kernel_families={
            name: dict(count=len(es), duration_sum_ms=sum(e["dur"] for e in es) / 1000)
            for name, es in kernel_groups.items()
        },
        collectives=dict(collective_summary),
        hccl_op_count=len(hccl),
        hccl_wait_sum_ms=sum(x["wait_ms"] for x in hccl),
        hccl_sdma_copy_time_sum_ms=sum(x["sdma_copy_time_ms"] for x in hccl),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    traces = []
    for root in args.work_dirs:
        paths = sorted(root.rglob("trace_view.json"))
        assert paths, root
        for path in paths:
            record = summarize(path, root)
            record["experiment"] = root.name
            traces.append(record)
    result = dict(
        method="Union of official profiler Computing and Communication device intervals; overlap is their intersection; idle is the complement between first and last device task. Never add ranks or use profiler latency as benchmark latency.",
        traces=traces,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for t in traces:
        print(
            t["experiment"],
            t["rank"],
            t["request"],
            t["mode"],
            {
                k: round(t[k], 3)
                for k in [
                    "compute_ms",
                    "communication_ms",
                    "overlap_ms",
                    "idle_between_tasks_ms",
                ]
            },
        )


if __name__ == "__main__":
    main()
