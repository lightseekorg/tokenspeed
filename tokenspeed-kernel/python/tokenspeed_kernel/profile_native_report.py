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

"""Append utilization labels to a native Nsight report using its offline writer.

Uses NVIDIA's ``ImportNvtxt`` utility from the matching Nsight installation.
The input is the Nsight-timed JSON produced by ``profile_nsys`` for this capture.
Ranges occupy the correlated GPU spans, including their gaps. They are derived
annotations, not new kernel measurements or replacements for the CPU ranges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def _display_groups(rows, default_rank):
    """Separate levels using recorded ancestry, never GPU-time containment."""

    def key(row):
        return row.get("rank", default_rank), row["iter_id"], row.get("scope_id")

    scopes = {key(row): row for row in rows if row.get("scope_id") is not None}

    def ancestors(row):
        rank, iteration, _ = key(row)
        seen = set()
        parent = row.get("parent_scope_id")
        while parent is not None:
            if parent in seen:
                raise ValueError("Cycle in recorded scope ancestry")
            seen.add(parent)
            scope = scopes.get((rank, iteration, parent))
            if scope is None:
                raise ValueError("Missing recorded parent scope")
            yield scope
            parent = scope.get("parent_scope_id")

    layer_ids = defaultdict(set)
    for row in rows:
        if row["kind"] == "layer" and row.get("scope_id") is not None:
            for scope in (row, *ancestors(row)):
                layer_ids[key(scope)].add(row["layer_id"])
    model_keys = {identity for identity, ids in layer_ids.items() if len(ids) > 1}
    groups = defaultdict(list)
    for row in rows:
        rank = row.get("rank", default_rank)
        if row["kind"] == "op":
            level = (2, 0, "Op")
        elif key(row) in model_keys or rank == "all":
            level = (0, 0, "Model")
        else:
            depth = sum(
                parent["kind"] == "layer" and key(parent) not in model_keys
                for parent in ancestors(row)
            )
            label = "Layer" if depth == 0 else f"Module depth={depth}"
            level = (1, depth, label)
        groups[(str(rank), *level)].append(row)
    return groups


def _mfu_display(row):
    """Use the explicit ideal-time total, retaining older graph report fields."""
    if "mfu_total" in row:
        value, known = row["mfu_total"], row.get("mfu_total_known")
        reason = row.get("mfu_total_missing_reason")
    else:
        value, known = row["mfu"], row.get("mfu_known")
        reason = row["mfu_missing_reason"]
    if value is not None:
        return f"{value * 100:.3g}%", reason
    if known is not None:
        return f"{known * 100:.3g}% (partial)", reason
    return "N/A", reason


def annotate_report(*, report: Path, metrics: Path, output: Path, importer: Path):
    """Copy a capture and append native NVTX ranges with MFU/Estimated MBU.

    All paths must be explicit. ``output`` must be a new ``.nsys-rep`` path;
    ``importer`` is the matching Nsight installation's ImportNvtxt executable.
    GPU timestamps are Nsight nanoseconds relative to the analysis start, as
    exported by ``profile_nsys``. Missing metrics stay N/A with their reasons.
    """
    report, metrics, output, importer = (
        path.expanduser().resolve() for path in (report, metrics, output, importer)
    )
    if output.suffix != ".nsys-rep" or output.exists():
        raise ValueError("Output must be a new .nsys-rep file")
    if output in (report, metrics):
        raise ValueError("Output must not overwrite capture or metrics")
    if not report.is_file() or not importer.is_file():
        raise ValueError("Capture and ImportNvtxt executable must exist")
    data = json.loads(metrics.read_text())
    analysis = data["device_analysis"]
    if analysis["timing_kind"] not in {
        "nsight_gpu_projection",
        "nsight_graph_node_ids",
    }:
        raise ValueError("Native annotation requires completed Nsight GPU timing")
    rows = analysis["scopes"]
    if not rows:
        raise ValueError("No analyzed scopes")
    for row in rows:
        start, end = row["gpu_start_ns"], row["gpu_end_ns"]
        if start is None or end is None or end <= start:
            raise ValueError("Every annotated scope must have correlated GPU work")
        if end - start != row["gpu_duration_ns"]:
            raise ValueError("Scope duration does not match its GPU boundaries")

    def fingerprint(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    original_hash = fingerprint(report)
    if analysis.get("source_report_sha256", original_hash) != original_hash:
        raise ValueError("Metrics were calculated from a different capture")
    output.parent.mkdir(parents=True, exist_ok=True)
    nvtxt = output.with_suffix(".nvtxt")
    if nvtxt.exists():
        raise ValueError("NVTXT sidecar already exists")
    # NVTXT Relative uses the same origin as Nsight's SQLite activity times.
    # Synthetic process/thread IDs identify analysis lanes, not new GPU work.
    lines = [
        "@NameProcess,ProcessId,Name",
        "@NameOsThread,ProcessId,ThreadId,Name",
        "@RangePush,Time,ProcessId,ThreadId,CategoryId,Color,Message",
        "@RangePop,Time,ProcessId,ThreadId",
        "TimeBase=Relative",
        "Payload=0",
    ]
    events = []
    # Imported thread rows can flatten nested ranges in the native viewer.
    # Put each hierarchy level on its own visible lane, and split any overlap
    # within a level so no longer range can cover a shorter one.
    lane_rows = []
    groups = _display_groups(rows, data["rank"])
    for (rank, _, _, level), grouped in sorted(groups.items()):
        tracks = []
        for row in sorted(
            grouped,
            key=lambda item: (item["gpu_start_ns"], -item["gpu_end_ns"]),
        ):
            start, end = row["gpu_start_ns"], row["gpu_end_ns"]
            for assigned in tracks:
                if start >= assigned[-1]["gpu_end_ns"]:
                    assigned.append(row)
                    break
            else:
                tracks.append([row])
        for index, assigned in enumerate(tracks, 1):
            name = f"rank={rank} | {level}"
            if len(tracks) > 1:
                name += f" track={index}"
            # Synthetic process groups are independently collapsible in the
            # native viewer. Keep all nonzero ranks together, without dropping
            # their annotations or changing any measurement boundaries.
            process = 2 if rank == "all" else (0 if rank == "0" else 1)
            lane_rows.append((process, name, assigned))
    process_names = {
        0: "Rank 0 | MFU / Est.MBU (GPU projection)",
        1: "Other ranks | expand to inspect MFU / Est.MBU",
        2: "All-rank model | MFU / Est.MBU",
    }
    for process in sorted({entry[0] for entry in lane_rows}):
        lines.append(f'NameProcess,{process},"{process_names[process]}"')
    for lane, (process, name, assigned) in enumerate(
        sorted(lane_rows, key=lambda item: item[0]), 1
    ):
        lines.append(f'NameOsThread,{process},{lane},"{name}"')
        for order, row in enumerate(assigned):
            start, end = row["gpu_start_ns"], row["gpu_end_ns"]
            mfu, mfu_reason = _mfu_display(row)
            mbu = (
                "N/A"
                if row["estimated_mbu"] is None
                else f"{row['estimated_mbu'] * 100:.3g}%"
            )
            if (
                row["estimated_mbu"] is None
                and row.get("estimated_mbu_known") is not None
            ):
                mbu = f"{row['estimated_mbu_known'] * 100:.3g}% (partial)"
            message = f"{row['native_nvtx_label']} | MFU_total={mfu} Est.MBU={mbu}"
            message += f" | iter={row['iter_id']} layer={row['layer_id']}"
            for compute_type, value in row["mfu_by_compute_type"].items():
                if row["mfu"] is None:
                    message += f" | MFU[{compute_type}]={value * 100:.3g}%"
            reasons = {
                reason
                for reason in (mfu_reason, row["mbu_missing_reason"])
                if reason is not None
            }
            if reasons:
                message += f" | N/A: {'; '.join(sorted(reasons))}"
            if any(char in message for char in ('"', "\r", "\n")):
                raise ValueError(
                    "NVTXT message contains unsupported quoting characters"
                )
            color = "LightBlue" if row["kind"] == "layer" else "PaleGreen"
            # Preserve parent-first input order for identical GPU spans, and
            # close them in reverse order. Label sorting would invert scopes.
            events.append(
                (
                    start,
                    1,
                    -end,
                    order,
                    f'RangePush,{start},{process},{lane},0,{color},"{message}"',
                )
            )
            events.append((end, 0, -start, -order, f"RangePop,{end},{process},{lane}"))
    lines.extend(event[-1] for event in sorted(events))
    nvtxt.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            str(importer),
            "--cmd",
            "merge",
            "-i",
            str(report),
            "-n",
            str(nvtxt),
            "-o",
            str(output),
        ],
        check=True,
    )
    if fingerprint(report) != original_hash:
        raise RuntimeError("Source capture changed during annotation")
    if not output.is_file():
        raise RuntimeError("Writer did not produce a report")
    return {
        "source_sha256": original_hash,
        "annotated_scopes": len(rows),
        "annotation_lanes": len(lane_rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-nvtxt", type=Path, required=True)
    args = parser.parse_args()
    result = annotate_report(
        report=args.report,
        metrics=args.metrics,
        output=args.output,
        importer=args.import_nvtxt,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
