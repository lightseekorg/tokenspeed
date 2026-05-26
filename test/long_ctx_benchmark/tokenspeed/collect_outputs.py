#!/usr/bin/env python3
"""Aggregate long-context perf runs into one summary table.

Expected sweep layout (produced by the long-ctx perf yaml):

    <sweep_dir>/run<R>/<len>/<model_name>/benchmark_summary.json

For each prompt length, this script computes per-run decode TPS (= 1000 / TPOT)
and MTP acceptance rate (= "Decoded Tok/Iter"), then prints a summary table
aggregated across all runs.

Usage:
    python3 collect_outputs.py <sweep_dir> [-o out.csv]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Order matters: shorter -> longer
LEN_ORDER = ["32k", "64k", "128k", "256k", "512k", "1024k"]


def _len_key(length: str) -> int:
    m = re.match(r"(\d+)k", length)
    return int(m.group(1)) if m else 0


def _summary_files(sweep_dir: Path):
    """Yield (run_id, prompt_len, summary_json_path)."""
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir() and p.name.startswith("run")):
        try:
            run_id = int(run_dir.name[3:])
        except ValueError:
            continue
        for len_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            length = len_dir.name
            for summary_path in len_dir.rglob("benchmark_summary.json"):
                yield run_id, length, summary_path


def collect(sweep_dir: Path):
    """Return per-run rows, sorted by (length, run_id)."""
    rows = []
    for run_id, length, summary_path in _summary_files(sweep_dir):
        try:
            s = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] skip {summary_path}: {e}", file=sys.stderr)
            continue
        tpot_ms = s.get("TPOT (ms)") or 0.0
        tps_user = 1000.0 / tpot_ms if tpot_ms else 0.0
        ar = float(s.get("Decoded Tok/Iter") or 0.0)
        total_tps = float(s.get("Total Throughput (tok/s)") or 0.0)
        rows.append(
            {
                "prompt_len": length,
                "run": run_id,
                "tps_per_user": round(tps_user, 2),
                "acceptance_rate": round(ar, 4),
                "total_throughput": round(total_tps, 2),
                "tpot_ms": round(tpot_ms, 3),
            }
        )
    rows.sort(key=lambda r: (_len_key(r["prompt_len"]), r["run"]))
    return rows


def aggregate(rows):
    """Group rows by prompt_len and return summary rows."""
    by_len: dict = {}
    for r in rows:
        by_len.setdefault(r["prompt_len"], []).append(r)
    summary = []
    for length in sorted(by_len.keys(), key=_len_key):
        group = by_len[length]
        tps_vals = [r["tps_per_user"] for r in group]
        ar_vals = [r["acceptance_rate"] for r in group]
        thr_vals = [r["total_throughput"] for r in group]
        n = len(group)
        summary.append(
            {
                "prompt_len": length,
                "n_runs": n,
                "avg_tps_per_user": round(sum(tps_vals) / n, 2) if n else 0.0,
                "min_tps_per_user": min(tps_vals) if tps_vals else 0.0,
                "max_tps_per_user": max(tps_vals) if tps_vals else 0.0,
                "avg_acceptance_rate": round(sum(ar_vals) / n, 4) if n else 0.0,
                "min_acceptance_rate": min(ar_vals) if ar_vals else 0.0,
                "max_acceptance_rate": max(ar_vals) if ar_vals else 0.0,
                "avg_total_throughput": round(sum(thr_vals) / n, 2) if n else 0.0,
            }
        )
    return summary


def print_table(rows, summary):
    """Pretty-print aggregated summary."""
    if not rows:
        print("[long-ctx perf] no benchmark_summary.json found", file=sys.stderr)
        return

    # Aggregated
    print("\n=== Long-context perf (aggregated, avg of N runs) ===")
    print(f"  {'Prompt':<8} {'N':>3} "
          f"{'avg TPS/user':>13} {'TPS range':>16} "
          f"{'avg AR':>9} {'AR range':>14} "
          f"{'avg Total TPS':>15}")
    print("  " + "-" * 80)
    for s in summary:
        tps_rng = f"{s['min_tps_per_user']:.1f}-{s['max_tps_per_user']:.1f}"
        ar_rng = f"{s['min_acceptance_rate']:.2f}-{s['max_acceptance_rate']:.2f}"
        print(f"  {s['prompt_len']:<8} {s['n_runs']:>3} "
              f"{s['avg_tps_per_user']:>13.2f} {tps_rng:>16} "
              f"{s['avg_acceptance_rate']:>9.3f} {ar_rng:>14} "
              f"{s['avg_total_throughput']:>15.2f}")
    print()


def write_csv(out_path: Path, rows, summary):
    with out_path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["section", "prompt_len", "run", "tps_per_user",
                    "acceptance_rate", "total_throughput", "tpot_ms"])
        for r in rows:
            w.writerow(["per_run", r["prompt_len"], r["run"],
                        r["tps_per_user"], r["acceptance_rate"],
                        r["total_throughput"], r["tpot_ms"]])
        w.writerow([])
        w.writerow(["section", "prompt_len", "n_runs",
                    "avg_tps_per_user", "min_tps_per_user", "max_tps_per_user",
                    "avg_acceptance_rate", "min_acceptance_rate", "max_acceptance_rate",
                    "avg_total_throughput"])
        for s in summary:
            w.writerow(["agg", s["prompt_len"], s["n_runs"],
                        s["avg_tps_per_user"], s["min_tps_per_user"], s["max_tps_per_user"],
                        s["avg_acceptance_rate"], s["min_acceptance_rate"], s["max_acceptance_rate"],
                        s["avg_total_throughput"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", type=Path,
                    help="Top-level dir containing run<N>/<len>/<model>/benchmark_summary.json")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Optional CSV output path")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"Not a directory: {args.sweep_dir}")

    rows = collect(args.sweep_dir)
    summary = aggregate(rows)
    print_table(rows, summary)
    if args.output:
        write_csv(args.output, rows, summary)
        print(f"[long-ctx perf] wrote CSV: {args.output}")

    # Treat completely empty sweep as failure for CI visibility
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
