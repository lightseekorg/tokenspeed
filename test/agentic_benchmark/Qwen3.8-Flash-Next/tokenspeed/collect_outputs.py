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

"""Collect Qwen TP4EP4 agentic runs with output throughput in tok/min/GPU."""

import argparse
import csv
import json
import math
import sqlite3
import sys
from pathlib import Path

CONFIG = "attn_tp4_moe_ep4"
NUM_GPUS = 4
COLUMNS = [
    "config",
    "Conc.",
    "Latency (tps/user)",
    "Output Throughput (tok/min/gpu)",
    "Approx Cache Hit",
    "Decoded Tok/Iter",
]


def decoded_tok_per_iter(run_dir: Path) -> float:
    """Return sum(completion_tokens - 1) / sum(streaming gaps), or -1 if absent.

    This weights acceptance length by decode iterations, as in Kimi-K3's
    collector, rather than using EvalScope's unweighted per-request mean.
    """
    db_path = run_dir / "benchmark_data.db"
    if not db_path.is_file():
        return -1.0
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT completion_tokens, inter_token_latencies "
            "FROM result WHERE success = 1"
        ).fetchall()
    finally:
        connection.close()
    tokens = iterations = 0
    for completion_tokens, latencies in rows:
        gaps = len(json.loads(latencies))
        if completion_tokens and completion_tokens > 1 and gaps > 0:
            tokens += completion_tokens - 1
            iterations += gaps
    return tokens / iterations if iterations else -1.0


def collect(sweep_dir: Path, *, expected_concurrencies: list[int]) -> list[dict]:
    """Return CI-compatible metric rows, rejecting failed or incomplete sweeps."""
    rows = []
    seen = set()
    for summary_path in sorted((sweep_dir / CONFIG).rglob("benchmark_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["Failed Requests"]:
            raise ValueError(f"{summary_path}: benchmark contains failed requests")
        concurrency = int(summary["Concurrency"])
        if concurrency in seen:
            raise ValueError(f"{summary_path}: duplicate concurrency {concurrency}")
        seen.add(concurrency)
        # EvalScope renamed TPOT in 1.10; preserve the same calculation.
        tpot_key = "Avg TPOT (ms)" if "Avg TPOT (ms)" in summary else "TPOT (ms)"
        tpot_ms = float(summary[tpot_key])
        output_throughput = float(summary["Output Throughput (tok/s)"])
        if not math.isfinite(tpot_ms) or tpot_ms <= 0:
            raise ValueError(f"{summary_path}: TPOT must be finite and positive")
        if not math.isfinite(output_throughput) or output_throughput < 0:
            raise ValueError(f"{summary_path}: invalid output throughput")
        rows.append(
            {
                "config": CONFIG,
                "Conc.": concurrency,
                "Latency (tps/user)": round(1000.0 / tpot_ms, 2),
                "Output Throughput (tok/min/gpu)": round(
                    output_throughput * 60 / NUM_GPUS, 2
                ),
                "Approx Cache Hit": round(summary["KV Cache Hit Rate (%)"], 2),
                "Decoded Tok/Iter": round(decoded_tok_per_iter(summary_path.parent), 4),
            }
        )
    if not rows or seen != set(expected_concurrencies):
        raise ValueError(
            f"Expected concurrencies {sorted(set(expected_concurrencies))}, "
            f"found {sorted(seen)} in {sweep_dir / CONFIG}"
        )
    return sorted(rows, key=lambda row: row["Conc."])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--expected-concurrencies", type=int, nargs="+", required=True)
    parser.add_argument("-o", "--output", type=Path, help="CSV path; omit for stdout")
    args = parser.parse_args()
    rows = collect(args.sweep_dir, expected_concurrencies=args.expected_concurrencies)
    output = (
        args.output.open("w", newline="", encoding="utf-8")
        if args.output
        else sys.stdout
    )
    try:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
