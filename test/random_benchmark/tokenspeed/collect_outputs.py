#!/usr/bin/env python3
"""Collect random perf sweeps into one CI summary table."""

import argparse
import json
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

INPUT_ORDER = {
    "input_1k": 1024,
    "input_2k": 2048,
    "input_4k": 4096,
    "input_8k": 8192,
}


def _config_from_path(sweep_dir: Path, summary_path: Path) -> str | None:
    try:
        rel_parts = summary_path.relative_to(sweep_dir).parts
    except ValueError:
        return None
    for part in rel_parts:
        if part.startswith("input_"):
            return part
    return None


def _sort_key(row: dict) -> tuple[int, int]:
    config = row["config"]
    return (INPUT_ORDER.get(config, 0), int(row["Conc."]))


def _float(summary: dict, key: str) -> float:
    value = summary.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def collect(sweep_dir: Path, num_gpus: int):
    rows = []
    for summary_path in sorted(sweep_dir.rglob("benchmark_summary.json")):
        config = _config_from_path(sweep_dir, summary_path)
        if config is None:
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] skip {summary_path}: {exc}", file=sys.stderr)
            continue

        tpot_ms = _float(summary, "TPOT (ms)")
        tps_user = 1000.0 / tpot_ms if tpot_ms else 0.0
        total_tps = _float(summary, "Total Throughput (tok/s)")
        cache_hit = _float(summary, "KV Cache Hit Rate (%)")
        decoded_per_iter = _float(summary, "Decoded Tok/Iter") or _float(
            summary, "Avg Decoded Tokens/Iter"
        )

        rows.append(
            {
                "config": config,
                "Conc.": int(_float(summary, "Concurrency")),
                "Latency (tps/user)": round(tps_user, 2),
                "Throughput (tps/gpu)": round(total_tps / num_gpus, 2),
                "Approx Cache Hit": round(cache_hit, 2),
                "Decoded Tok/Iter": round(decoded_per_iter, 4),
            }
        )
    rows.sort(key=_sort_key)
    return rows


def print_table(rows):
    # The CI pipeline recognizes this marker and adds the following block to
    # the GitHub step summary.
    print("\nOverall perf table:")
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in COLUMNS
    }
    header = "  ".join(column.rjust(widths[column]) for column in COLUMNS)
    sep = "  ".join("-" * widths[column] for column in COLUMNS)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row[column]).rjust(widths[column]) for column in COLUMNS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"Not a directory: {args.sweep_dir}")
    if args.num_gpus <= 0:
        sys.exit("--num-gpus must be positive")

    rows = collect(args.sweep_dir, args.num_gpus)
    print_table(rows)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
