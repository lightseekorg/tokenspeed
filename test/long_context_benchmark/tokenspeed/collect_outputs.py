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
import math
import re
import sys
from pathlib import Path

# Order matters: shorter -> longer
LEN_ORDER = ["32k", "64k", "128k", "256k", "512k", "1024k"]

# Speculative-decoding acceptance keys produced by different evalscope versions.
# Current evalscope[perf] reports "Avg Decoded Tok/Iter". Other versions used
# "Avg Decoded Tokens/Iter" or "Decoded Tok/Iter".
# "Spec Decode Acceptance (%)" is a percent-form alternative on some builds.
AR_KEYS = (
    "Avg Decoded Tok/Iter",
    "Avg Decoded Tokens/Iter",
    "Decoded Tok/Iter",
    "Spec Decode Acceptance (%)",
)


def _extract_ar(summary: dict, source: str) -> float | None:
    """Return MTP acceptance rate (avg decoded tokens per iter).

    Prefers the new evalscope key, falls back to the old one. Treats the
    percent-form acceptance as `1 + pct/100` only if it's the sole signal
    available. Missing metrics remain unavailable instead of becoming zero.
    """
    for key in AR_KEYS[:-1]:
        v = summary.get(key)
        if v is not None:
            try:
                value = float(v)
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
    pct = summary.get("Spec Decode Acceptance (%)")
    if pct is not None:
        try:
            # acceptance% is the fraction of draft tokens accepted; convert to
            # an avg-decoded-tokens-per-iter approximation only as a fallback.
            value = 1.0 + float(pct) / 100.0
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    print(
        f"[warn] {source}: none of {AR_KEYS} found in benchmark_summary.json; "
        f"acceptance rate will be reported as N/A",
        file=sys.stderr,
    )
    return None


def _extract_tpot(summary: dict, source: Path) -> float:
    key = "Avg TPOT (ms)" if "Avg TPOT (ms)" in summary else "TPOT (ms)"
    raw = summary.get(key)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        value = float("nan")
    if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{source}: {key} must be a finite positive number, got {raw!r}"
        )
    return value


def _len_key(length: str) -> int:
    m = re.match(r"(\d+)k", length)
    return int(m.group(1)) if m else 0


def _summary_files(sweep_dir: Path):
    """Yield (run_id, prompt_len, summary_json_path)."""
    for run_dir in sorted(
        p for p in sweep_dir.iterdir() if p.is_dir() and p.name.startswith("run")
    ):
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
            raise ValueError(
                f"{summary_path}: cannot read benchmark summary: {e}"
            ) from e
        if not isinstance(s, dict):
            raise ValueError(f"{summary_path}: benchmark summary must be a JSON object")
        tpot_ms = _extract_tpot(s, summary_path)
        tps_user = 1000.0 / tpot_ms
        ar = _extract_ar(s, str(summary_path))
        total_tps = float(s.get("Total Throughput (tok/s)") or 0.0)
        rows.append(
            {
                "prompt_len": length,
                "run": run_id,
                "tps_per_user": round(tps_user, 2),
                "acceptance_rate": round(ar, 4) if ar is not None else None,
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
        complete_ar = all(value is not None for value in ar_vals)
        thr_vals = [r["total_throughput"] for r in group]
        n = len(group)
        summary.append(
            {
                "prompt_len": length,
                "n_runs": n,
                "avg_tps_per_user": round(sum(tps_vals) / n, 2) if n else 0.0,
                "min_tps_per_user": min(tps_vals) if tps_vals else 0.0,
                "max_tps_per_user": max(tps_vals) if tps_vals else 0.0,
                "avg_acceptance_rate": (
                    round(sum(ar_vals) / n, 4) if complete_ar else None
                ),
                "min_acceptance_rate": min(ar_vals) if complete_ar else None,
                "max_acceptance_rate": max(ar_vals) if complete_ar else None,
                "avg_total_throughput": round(sum(thr_vals) / n, 2) if n else 0.0,
            }
        )
    return summary


def print_table(rows, summary):
    """Pretty-print aggregated summary.

    Uses the "Overall perf table:" marker so the CI pipeline
    (pipeline.py extract_evalscope_table) auto-extracts it into
    the GitHub Step Summary.
    """
    if not rows:
        print("[long-ctx perf] no benchmark_summary.json found", file=sys.stderr)
        return

    print("\nOverall perf table:")
    header = (
        f"{'Prompt':<8}  {'Runs':>4}  "
        f"{'Avg TPS/user':>13}  {'TPS Range':>17}  "
        f"{'Avg AR':>8}  {'AR Range':>15}  "
        f"{'Avg Total TPS':>14}"
    )
    sep = (
        f"{'--------':<8}  {'----':>4}  "
        f"{'-------------':>13}  {'-----------------':>17}  "
        f"{'--------':>8}  {'---------------':>15}  "
        f"{'--------------':>14}"
    )
    print(header)
    print(sep)
    for s in summary:
        tps_rng = f"{s['min_tps_per_user']:.1f} - {s['max_tps_per_user']:.1f}"
        ar_avg = "N/A"
        ar_rng = "N/A"
        if s["avg_acceptance_rate"] is not None:
            ar_avg = f"{s['avg_acceptance_rate']:.3f}"
            ar_rng = f"{s['min_acceptance_rate']:.2f} - {s['max_acceptance_rate']:.2f}"
        print(
            f"{s['prompt_len']:<8}  {s['n_runs']:>4}  "
            f"{s['avg_tps_per_user']:>13.2f}  {tps_rng:>17}  "
            f"{ar_avg:>8}  {ar_rng:>15}  "
            f"{s['avg_total_throughput']:>14.2f}"
        )
    print()


def write_csv(out_path: Path, rows, summary):
    with out_path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "section",
                "prompt_len",
                "run",
                "tps_per_user",
                "acceptance_rate",
                "total_throughput",
                "tpot_ms",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    "per_run",
                    r["prompt_len"],
                    r["run"],
                    r["tps_per_user"],
                    r["acceptance_rate"],
                    r["total_throughput"],
                    r["tpot_ms"],
                ]
            )
        w.writerow([])
        w.writerow(
            [
                "section",
                "prompt_len",
                "n_runs",
                "avg_tps_per_user",
                "min_tps_per_user",
                "max_tps_per_user",
                "avg_acceptance_rate",
                "min_acceptance_rate",
                "max_acceptance_rate",
                "avg_total_throughput",
            ]
        )
        for s in summary:
            w.writerow(
                [
                    "agg",
                    s["prompt_len"],
                    s["n_runs"],
                    s["avg_tps_per_user"],
                    s["min_tps_per_user"],
                    s["max_tps_per_user"],
                    s["avg_acceptance_rate"],
                    s["min_acceptance_rate"],
                    s["max_acceptance_rate"],
                    s["avg_total_throughput"],
                ]
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dir",
        type=Path,
        help="Top-level dir containing run<N>/<len>/<model>/benchmark_summary.json",
    )
    ap.add_argument(
        "-o", "--output", type=Path, default=None, help="Optional CSV output path"
    )
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"Not a directory: {args.sweep_dir}")

    try:
        rows = collect(args.sweep_dir)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
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
