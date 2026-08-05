#!/usr/bin/env python3
"""Aggregate fixed-width Kimi-K3 DSpark sweeps against one no-spec baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("kimi_k3_dspark_ab.py")
SPEC = importlib.util.spec_from_file_location("kimi_k3_dspark_ab", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ab
SPEC.loader.exec_module(ab)

FIELDS = (
    "engine",
    "group",
    "point",
    "verify_width",
    "draft_count",
    "baseline_repeats",
    "dspark_repeats",
    "baseline_tpot_ms",
    "dspark_tpot_ms",
    "tpot_speedup",
    "baseline_output_tps",
    "dspark_output_tps",
    "output_speedup",
    "accept_length",
    "accept_rate",
    "spec_iteration_ms",
    "break_even_accept",
    "accept_margin",
)


def _reference(directory: Path) -> dict[str, Any]:
    rows = ab.load_raw_results(directory, "tokenspeed")
    _, reference = ab.aggregate_results(rows)
    return reference


def aggregate_widths(root: Path, widths: list[int]) -> list[dict[str, Any]]:
    baseline = _reference(root / "baseline")
    rows = []
    for width in widths:
        candidate = _reference(root / f"w{width}")
        for key, dspark in sorted(candidate.items()):
            if not key.endswith("|dspark"):
                continue
            baseline_key = key.removesuffix("|dspark") + "|no-spec"
            no_spec = baseline.get(baseline_key)
            if not no_spec:
                continue
            engine, group, point, _ = key.split("|", 3)
            baseline_tpot = no_spec.get("median_tpot_ms")
            dspark_tpot = dspark.get("median_tpot_ms")
            accept = dspark.get("accept_length")
            iteration = (
                dspark_tpot * accept
                if dspark_tpot is not None and accept is not None
                else None
            )
            break_even = ab._ratio(iteration, baseline_tpot)
            rows.append(
                {
                    "engine": engine,
                    "group": group,
                    "point": point,
                    "verify_width": width,
                    "draft_count": width - 1,
                    "baseline_repeats": no_spec.get("successful_repeats", 0),
                    "dspark_repeats": dspark.get("successful_repeats", 0),
                    "baseline_tpot_ms": baseline_tpot,
                    "dspark_tpot_ms": dspark_tpot,
                    "tpot_speedup": ab._ratio(baseline_tpot, dspark_tpot),
                    "baseline_output_tps": no_spec.get("output_throughput"),
                    "dspark_output_tps": dspark.get("output_throughput"),
                    "output_speedup": ab._ratio(
                        dspark.get("output_throughput"),
                        no_spec.get("output_throughput"),
                    ),
                    "accept_length": accept,
                    "accept_rate": dspark.get("accept_rate"),
                    "spec_iteration_ms": iteration,
                    "break_even_accept": break_even,
                    "accept_margin": (
                        accept - break_even
                        if accept is not None and break_even is not None
                        else None
                    ),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = aggregate_widths(args.root, args.widths)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "width_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "width_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n"
    )
    print(json.dumps({"rows": rows}, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
