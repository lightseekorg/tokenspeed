#!/usr/bin/env python3
"""Collect pd_sim sweeps into three tables (P-fresh / P-cached / D-sim).

Applies the validity guards from the README: a rung is VOID if its cache-hit
guard fails, if any request failed, or (D-sim) if measure-phase memory kept
climbing past the post-prime level. Accepts multiple sweep dirs (e.g. one P
sweep and one D sweep); the P:D sizing helper needs one of each.
"""

import argparse
import json
import re
import sys
from pathlib import Path

MEM_CLIMB_MIB = 4096  # measure-phase peak more than this above post-prime -> flag

METRIC = {
    "p_fresh": "Prefill Throughput (tok/s)",
    "p_cached": "Computed Throughput (tok/s)",
    "d_measure": "Output Throughput (tok/s)",
}


def hit_guard(phase, hit):
    if hit < 0:
        return False  # missing data must not masquerade as a reading
    if phase == "p_fresh":
        return hit <= 5.0
    return hit >= 95.0


def num_gpus(config: str) -> int:
    m = re.search(r"attn_(?:tp|dp)(\d+)", config)
    if not m:
        raise ValueError(f"Cannot infer GPU count from config name: {config}")
    return int(m.group(1))


def load_ledgers(sweep_dir: Path):
    ledger = {}
    for f in sweep_dir.glob("*_memory_ledger.jsonl"):
        config = f.name.replace("_memory_ledger.jsonl", "")
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ledger[(config, e.get("parallel"))] = e
    return ledger


def collect(sweep_dir: Path):
    rows = []
    ledger = load_ledgers(sweep_dir)
    for summary in sorted(sweep_dir.glob("*/parallel_*/benchmark_summary.json")):
        run_name = summary.parent.parent.name  # <config>_<phase>
        m = re.match(r"(.+)_(p_fresh|p_cached|d_measure)$", run_name)
        if not m:
            continue
        config, phase = m.group(1), m.group(2)
        s = json.loads(summary.read_text())
        hit = s.get("KV Cache Hit Rate (%)", -1.0)
        failed = s.get("Failed Requests", 0)
        metric = s.get(METRIC[phase], 0.0)
        conc = s.get("Concurrency")

        problems = []
        if not hit_guard(phase, hit):
            problems.append("hit")
        if failed:
            problems.append(f"{failed}failed")

        row = {
            "phase": phase,
            "config": config,
            "Conc.": conc,
            f"{METRIC[phase]} /gpu": round(metric / num_gpus(config), 2),
            "Cache Hit (%)": round(hit, 2),
            "Requests/s": s.get("Requests/s"),
            "Latency p50 (s)": s.get("Latency p50 (s)"),
            "Latency p99 (s)": s.get("Latency p99 (s)"),
        }
        if phase == "d_measure":
            row["TPOT p50 (ms)"] = s.get("TPOT p50 (ms)")
            row["TPOT p99 (ms)"] = s.get("TPOT p99 (ms)")
            led = ledger.get((config, conc))
            if led:
                after = led.get("mem_after_prime", -1)
                peak = led.get("mem_peak_during_measure", -1)
                row["Mem after prime (MiB)"] = after
                row["Mem peak measure (MiB)"] = peak
                if peak < 0 or after < 0:
                    problems.append("mem-unsampled")
                elif peak - after > MEM_CLIMB_MIB:
                    problems.append("mem-climb")
            else:
                problems.append("no-ledger")
        row["valid"] = "ok" if not problems else "VOID(" + "+".join(problems) + ")"
        rows.append(row)
    return rows


def print_tables(rows):
    print("# Boundaries: no KV-transfer cost modeled; prime-as-transfer is the")
    print("# core approximation; single machine; TTFT ~= full-request latency")
    print("# at max_tokens 1; TPOT amortizes the cache-hit KV load.")
    for phase in ("p_fresh", "p_cached", "d_measure"):
        sub = [r for r in rows if r["phase"] == phase]
        if not sub:
            continue
        print(f"\n== {phase} ==")
        cols = [c for c in sub[0].keys() if c != "phase"]
        print(",".join(cols))
        for r in sorted(sub, key=lambda x: (x["config"], x["Conc."])):
            print(",".join(str(r.get(c, "")) for c in cols))


def sizing_helper(rows, fresh_tok, cached_tok, decode_tok):
    """P:D node ratio for a target per-conversation token mix."""
    print(
        f"\n== P:D sizing (mix per conversation: fresh={fresh_tok} "
        f"cached={cached_tok} decode={decode_tok} tokens) =="
    )
    for config in sorted({r["config"] for r in rows}):
        picks = {}
        for phase in ("p_fresh", "p_cached", "d_measure"):
            valid = [
                r
                for r in rows
                if r["config"] == config and r["phase"] == phase and r["valid"] == "ok"
            ]
            if valid:
                picks[phase] = max(valid, key=lambda r: r["Conc."])
        if len(picks) < 3:
            print(f"{config}: insufficient valid rungs")
            continue
        fresh = picks["p_fresh"][f"{METRIC['p_fresh']} /gpu"]
        cached = picks["p_cached"][f"{METRIC['p_cached']} /gpu"]
        d = picks["d_measure"][f"{METRIC['d_measure']} /gpu"]
        if not (fresh and cached and d):
            print(f"{config}: zero rate in a phase, cannot size")
            continue
        # GPU-seconds per conversation on each side; their ratio is the
        # P-GPU : D-GPU provisioning ratio for this workload mix.
        p_sec = fresh_tok / fresh + cached_tok / cached
        d_sec = decode_tok / d
        print(
            f"{config}: P {p_sec:.2f} gpu-s/conv, D {d_sec:.2f} gpu-s/conv "
            f"-> {p_sec / d_sec:.2f} P-GPUs per D-GPU"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dirs",
        type=Path,
        nargs="+",
        help="one or more sweep dirs (e.g. outputs/p_<ts> and " "outputs/d_<ts>)",
    )
    ap.add_argument(
        "--mix-fresh-tokens",
        type=int,
        default=50000,
        help="fresh prefill tokens per conversation (agentic "
        "anchor: the 50K first turn)",
    )
    ap.add_argument(
        "--mix-cached-tokens",
        type=int,
        default=15000,
        help="cached-prefill computed tokens per conversation "
        "(agentic anchor: ~11-14 increments x ~1.3K)",
    )
    ap.add_argument(
        "--mix-decode-tokens",
        type=int,
        default=6000,
        help="decoded tokens per conversation (agentic anchor: " "~12 turns x 500)",
    )
    args = ap.parse_args()

    rows = []
    for d in args.sweep_dirs:
        rows.extend(collect(d))
    if not rows:
        sys.exit(f"no benchmark_summary.json under {args.sweep_dirs}")
    print_tables(rows)
    sizing_helper(
        rows, args.mix_fresh_tokens, args.mix_cached_tokens, args.mix_decode_tokens
    )


if __name__ == "__main__":
    main()
