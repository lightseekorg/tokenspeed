#!/usr/bin/env python3
"""Collect prefill conversation replay and decode simulation measurements.

Reports cache-hit rates for the user to assess. Accepts multiple sweep dirs
(e.g. one P sweep and one D sweep); the P:D sizing helper needs one of each.
"""

import argparse
import json
import re
import sys
from pathlib import Path

METRIC = {
    "p_measure": "Computed Throughput (tok/s)",
    "d_measure": "Output Throughput (tok/s)",
}


def num_gpus(config: str) -> int:
    m = re.search(r"attn_(?:tp|dp)(\d+)", config)
    if not m:
        raise ValueError(f"Cannot infer GPU count from config name: {config}")
    return int(m.group(1))


def collect(sweep_dir: Path):
    rows = []
    for summary in sorted(sweep_dir.glob("*/parallel_*/benchmark_summary.json")):
        run_name = summary.parent.parent.name  # <config>_<phase>
        m = re.match(r"(.+)_(p_measure|d_measure)$", run_name)
        if not m:
            continue
        config, phase = m.group(1), m.group(2)
        s = json.loads(summary.read_text())
        hit = s.get("KV Cache Hit Rate (%)", -1.0)
        metric = s.get(METRIC[phase], 0.0)
        conc = s.get("Concurrency")
        target_tokens = s["Target Response Tokens"]
        target_requests_per_second = (
            s["Output Throughput (tok/s)"] / target_tokens
            if phase == "d_measure"
            else s["Requests/s"]
        )

        row = {
            "phase": phase,
            "config": config,
            "Conc.": conc,
            f"{METRIC[phase]} /gpu": round(metric / num_gpus(config), 2),
            "Cache Hit (%)": hit,
            "Target Response Tokens": target_tokens,
            "Requests/s @ target OSL": round(target_requests_per_second, 4),
            "Latency mean (s)": s.get("Latency mean (s)"),
            "Latency p50 (s)": s.get("Latency p50 (s)"),
            "Latency p99 (s)": s.get("Latency p99 (s)"),
        }
        if phase == "d_measure":
            row["TTFT p50 (s)"] = s.get("TTFT p50 (s)")
            row["TTFT p99 (s)"] = s.get("TTFT p99 (s)")
            row["TPOT p50 (ms)"] = s.get("TPOT p50 (ms)")
            row["TPOT p99 (ms)"] = s.get("TPOT p99 (ms)")
        rows.append(row)
    return rows


def print_tables(rows):
    print("# Boundaries: no KV-transfer cost modeled; prime-as-transfer is the")
    print("# core approximation; single deployment; P latency includes queueing.")
    print("# P uses synthetic assistant text; D throughput includes the full wave.")
    for phase in ("p_measure", "d_measure"):
        sub = [r for r in rows if r["phase"] == phase]
        if not sub:
            continue
        print(f"\n== {phase} ==")
        cols = [c for c in sub[0].keys() if c != "phase"]
        print(",".join(cols))
        for r in sorted(sub, key=lambda x: (x["config"], x["Conc."])):
            print(",".join(str(r.get(c, "")) for c in cols))


def sizing_helper(rows):
    """Estimate fractional P instances per D instance for each decode point."""
    print("\n== P:D sizing from measured P request throughput ==")
    for p in (r for r in rows if r["phase"] == "p_measure"):
        config = p["config"]
        decode_rows = [
            r
            for r in rows
            if r["config"] == config
            and r["phase"] == "d_measure"
            and r["Target Response Tokens"] == p["Target Response Tokens"]
        ]
        if not decode_rows:
            print(f"{config}: no matching D measurement")
            continue
        mu_p = p["Requests/s @ target OSL"]
        output_tokens = p["Target Response Tokens"]
        if not mu_p or not output_tokens:
            print(f"{config}: zero P rate or response length, cannot size")
            continue
        for d in sorted(decode_rows, key=lambda r: r["Conc."]):
            mu_d = d["Requests/s @ target OSL"]
            print(
                f"{config}: P cc={p['Conc.']}, D cc={d['Conc.']}, "
                f"target OSL={output_tokens}, mu_P={mu_p:.4f} req/s, "
                f"estimated mu_D={mu_d:.4f} req/s "
                f"-> {mu_d / mu_p:.2f} P instances per D instance"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dirs",
        type=Path,
        nargs="+",
        help="one or more sweep dirs (e.g. outputs/p_<ts> and " "outputs/d_<ts>)",
    )
    args = ap.parse_args()

    rows = []
    for d in args.sweep_dirs:
        rows.extend(collect(d))
    if not rows:
        sys.exit(f"no benchmark_summary.json under {args.sweep_dirs}")
    print_tables(rows)
    sizing_helper(rows)


if __name__ == "__main__":
    main()
