#!/usr/bin/env python3
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

"""Join GSM8K wire measurements to native rank-0 RequestStats logs."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from test_gsm8k_pause_resume import summary_stats

REQUEST_STATS = re.compile(
    r"Req: (?P<rid>[a-f0-9]{32}) Finish! RequestStats\(.*?"
    r"output_tokens=(?P<tokens>\d+),.*?decode_tps=(?P<rate>\d+(?:\.\d+)?),"
)


def native_stats(path, rows):
    """Summarize approximate host-side TPOT from the logged one-decimal rate."""
    if not path.is_file():
        return {}
    values = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        match = REQUEST_STATS.search(line)
        if match:
            values[match["rid"]].append((int(match["tokens"]), float(match["rate"])))
    metrics = {"matched": 0, "missing": 0, "duplicates": 0, "token_count_mismatch": 0}
    times = []
    timed_tokens = 0
    timed_ms = 0.0
    for row in rows:
        candidates = values.get(row.get("request_id"), [])
        if not candidates:
            metrics["missing"] += 1
            continue
        if len(candidates) != 1:
            metrics["duplicates"] += 1
            continue
        output_tokens, rate = candidates[0]
        if output_tokens != row["completion_tokens"]:
            metrics["token_count_mismatch"] += 1
            continue
        metrics["matched"] += 1
        if rate > 0.0 and output_tokens >= 2:
            ms = 1000.0 / rate
            times.append(ms)
            timed_tokens += output_tokens - 1
            timed_ms += ms * (output_tokens - 1)
    return {
        **metrics,
        "tpot_ms": summary_stats(times),
        "token_weighted_tpot_ms": timed_ms / timed_tokens if timed_tokens else None,
    }


def load_run(path):
    # Each run directory contains engine logs and a benchmark subdirectory.
    bench = path / "benchmark"
    if not (bench / "summary.json").is_file():
        return {"run": path.name, "status": "no_summary"}
    report = json.loads((bench / "summary.json").read_text())
    report["run"] = path.name
    inputs = path / "inputs.log"
    if inputs.is_file():
        for line in inputs.read_text().splitlines():
            if line.startswith(("mode=", "cards=")):
                key, value = line.split("=", 1)
                report[key] = value
    for phase in ("before", "after"):
        records = bench / f"{phase}.jsonl"
        if records.is_file():
            rows = [json.loads(line) for line in records.read_text().splitlines()]
            report[f"{phase}_native"] = native_stats(path / "runner.log", rows)
    return report


def csv_row(report):
    row = {
        "run": report["run"],
        "status": report["status"],
        "mode": report.get("mode"),
        "cards": report.get("cards"),
        "tp_size": report.get("tp_size"),
        "questions": report.get("measured_limit"),
        "concurrency": report.get("concurrency"),
        "max_output_tokens": report.get("max_tokens"),
    }
    for phase in ("before", "after"):
        measured = report.get(phase, {})
        for key in (
            "correct",
            "accuracy",
            "wall_s",
            "completion_tokens",
            "length_capped",
        ):
            row[f"{phase}_{key}"] = measured.get(key)
        row[f"{phase}_wire_mean_tpot_ms"] = measured.get("tpot_ms", {}).get("mean")
        row[f"{phase}_wire_p95_tpot_ms"] = measured.get("tpot_ms", {}).get("p95")
        native = report.get(f"{phase}_native", {})
        row[f"{phase}_native_mean_tpot_ms"] = native.get("tpot_ms", {}).get("mean")
        row[f"{phase}_native_matched"] = native.get("matched")
    for key in (
        "changed_output_ids",
        "changed_correctness",
        "accuracy_percentage_points",
        "wall_s_delta_pct",
        "mean_tpot_delta_pct",
    ):
        row[key] = report.get("comparison", {}).get(key)
    for key in ("pause_wire_s", "resume_wire_s"):
        row[key] = report.get(key)
    row["recapture_count"] = report.get("graph", {}).get("recapture_count")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [load_run(path) for path in args.runs]
    (args.output_dir / "gsm8k-matrix.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    rows = [csv_row(report) for report in reports]
    with (args.output_dir / "gsm8k-matrix.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
