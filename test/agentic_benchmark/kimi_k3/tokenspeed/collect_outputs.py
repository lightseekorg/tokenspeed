#!/usr/bin/env python3
"""Collect a sweep's per-config benchmark_summary.json files into one CSV."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

COLUMNS = [
    "config",
    "Conc.",
    "Latency (tps/user)",
    "Throughput (tps/gpu)",
    "Approx Cache Hit",
    "Decoded Tok/Iter",
]


def num_gpus_from_config(config: str) -> int:
    m = re.search(r"attn_(?:tp|dp)(\d+)", config)
    if not m:
        raise ValueError(f"Cannot infer GPU count from config name: {config}")
    return int(m.group(1))


def collect(sweep_dir: Path):
    rows = []
    for config_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        config = config_dir.name
        n_gpus = num_gpus_from_config(config)
        for run_dir in sorted(config_dir.iterdir()):
            summary_path = run_dir / "benchmark_summary.json"
            if not summary_path.is_file():
                continue
            s = json.loads(summary_path.read_text())
            tpot_ms = s["TPOT (ms)"]
            decode_tps_user = 1000.0 / tpot_ms if tpot_ms else 0.0
            tps_gpu = s["Total Throughput (tok/s)"] / n_gpus
            rows.append(
                {
                    "config": config,
                    "Conc.": s["Concurrency"],
                    "Latency (tps/user)": round(decode_tps_user, 2),
                    "Throughput (tps/gpu)": round(tps_gpu, 2),
                    "Approx Cache Hit": round(s["KV Cache Hit Rate (%)"], 2),
                    "Decoded Tok/Iter": round(s["Decoded Tok/Iter"], 4),
                }
            )
    rows.sort(key=lambda r: (r["config"], r["Conc."]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", type=Path, help="e.g. outputs/20260505_152734")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: stdout)",
    )
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"Not a directory: {args.sweep_dir}")

    rows = collect(args.sweep_dir)
    out = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        w = csv.DictWriter(out, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
