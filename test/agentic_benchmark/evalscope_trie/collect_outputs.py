#!/usr/bin/env python3
"""Collect EvalScope trie perf outputs into CSV and a simple Pareto SVG."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import xml.sax.saxutils as sx
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "phase",
    "model",
    "parallel",
    "number",
    "success",
    "total_requests",
    "success_requests",
    "failed_requests",
    "steady_completion_tok_s",
    "completion_tps_per_user",
    "output_token_min_per_gpu",
    "avg_latency_s",
    "avg_ttft_s",
    "avg_tpot_s",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_turns",
    "cache_hit_pct",
    "first_turn_ttft_s",
    "subsequent_turn_ttft_s",
    "decoded_tok_iter",
    "spec_accept_rate",
    "trace_count",
    "run_dir",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ms_to_s(value: Any) -> float | None:
    value = safe_float(value)
    return value / 1000.0 if value is not None else None


def workload_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {row.get("metric", ""): row for row in data.get("rows", [])}


def db_counts(db_path: Path) -> tuple[int | None, int | None]:
    if not db_path.exists():
        return None, None
    with sqlite3.connect(db_path) as con:
        total = con.execute("select count(*) from result").fetchone()[0]
        success = con.execute("select count(*) from result where success=1").fetchone()[
            0
        ]
    return total, success


def parse_parallel_number(run_dir: Path) -> tuple[int, int]:
    match = re.match(r"parallel_(\d+)_number_(\d+)$", run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse parallel/number from {run_dir}")
    return int(match.group(1)), int(match.group(2))


def infer_phase_and_model(root: Path, run_dir: Path) -> tuple[str, str]:
    rel = run_dir.relative_to(root)
    parts = rel.parts
    phase = parts[0] if len(parts) >= 3 else "unknown"
    model = parts[-2] if len(parts) >= 2 else "unknown"
    return phase, model


def parse_run(root: Path, run_dir: Path, num_gpus: int) -> dict[str, Any]:
    summary = load_json(run_dir / "benchmark_summary.json")
    trace_summary = load_json(run_dir / "trace_summary.json")
    rows = workload_rows(run_dir / "workload_throughput.json")
    parallel, number = parse_parallel_number(run_dir)
    phase, model = infer_phase_and_model(root, run_dir)
    db_total, db_success = db_counts(run_dir / "benchmark_data.db")

    def wl(metric: str, field: str) -> float | None:
        return safe_float(rows.get(metric, {}).get(field))

    total = int(summary.get("Total Requests") or db_total or 0)
    success = int(summary.get("Success Requests") or db_success or 0)
    failed = int(summary.get("Failed Requests") or max(total - success, 0))
    steady_completion = wl("Completion tok/s", "steady_state")
    completion_tps_user = (
        steady_completion / parallel
        if steady_completion is not None and parallel
        else None
    )
    output_token_min_gpu = (
        steady_completion * 60.0 / num_gpus
        if steady_completion is not None and num_gpus
        else None
    )

    return {
        "phase": phase,
        "model": model,
        "parallel": parallel,
        "number": number,
        "success": f"{success}/{total}" if total else "",
        "total_requests": total,
        "success_requests": success,
        "failed_requests": failed,
        "steady_completion_tok_s": steady_completion,
        "completion_tps_per_user": completion_tps_user,
        "output_token_min_per_gpu": output_token_min_gpu,
        "avg_latency_s": safe_float(summary.get("Avg Latency (s)")),
        "avg_ttft_s": ms_to_s(summary.get("TTFT (ms)")),
        "avg_tpot_s": ms_to_s(summary.get("TPOT (ms)")),
        "avg_input_tokens": safe_float(summary.get("Avg Input Tokens")),
        "avg_output_tokens": safe_float(summary.get("Avg Output Tokens")),
        "avg_turns": safe_float(summary.get("Avg Turns/Request")),
        "cache_hit_pct": safe_float(summary.get("KV Cache Hit Rate (%)")),
        "first_turn_ttft_s": ms_to_s(summary.get("First-Turn TTFT (ms)")),
        "subsequent_turn_ttft_s": ms_to_s(summary.get("Subsequent-Turn TTFT (ms)")),
        "decoded_tok_iter": safe_float(summary.get("Decoded Tok/Iter")),
        "spec_accept_rate": safe_float(summary.get("Spec. Accept Rate")),
        "trace_count": trace_summary.get("n_traces"),
        "run_dir": str(run_dir),
    }


def collect(root: Path, num_gpus: int) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(root.glob("**/parallel_*_number_*")):
        if (run_dir / "benchmark_summary.json").exists():
            rows.append(parse_run(root, run_dir, num_gpus))
    rows.sort(key=lambda r: (r["phase"] != "sweep", r["parallel"], r["number"]))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def finite(values: list[float | None]) -> list[float]:
    return [
        value
        for value in values
        if value is not None and not math.isnan(value) and not math.isinf(value)
    ]


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def write_svg(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    points = [
        row
        for row in rows
        if row["phase"] == "sweep"
        and row["completion_tps_per_user"] is not None
        and row["output_token_min_per_gpu"] is not None
    ]
    if not points:
        return

    width, height = 1120, 720
    left, right, top, bottom = 95, 55, 86, 96
    xs = finite([row["completion_tps_per_user"] for row in points])
    ys = finite([row["output_token_min_per_gpu"] for row in points])
    x_min, x_max = min(xs) * 0.88, max(xs) * 1.08
    y_min, y_max = 0.0, max(ys) * 1.15

    def x_px(v: float) -> float:
        return scale(v, x_min, x_max, left, width - right)

    def y_px(v: float) -> float:
        return scale(v, y_min, y_max, height - bottom, top)

    ordered = sorted(points, key=lambda row: row["parallel"])
    line_path = " ".join(
        ("M" if idx == 0 else "L")
        + f"{x_px(row['completion_tps_per_user']):.1f},"
        + f"{y_px(row['output_token_min_per_gpu']):.1f}"
        for idx, row in enumerate(ordered)
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#333}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".axis{stroke:#555;stroke-width:1.5}",
        ".line{fill:none;stroke:#8e24aa;stroke-width:3}",
        ".pt{fill:#8e24aa;stroke:white;stroke-width:2}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="560" y="36" text-anchor="middle" font-size="24" font-weight="700">{sx.escape(title)}</text>',
        '<text x="560" y="64" text-anchor="middle" font-size="14">EvalScope trie workload, formal sweep only</text>',
    ]
    for idx in range(7):
        x = left + (width - left - right) * idx / 6
        value = x_min + (x_max - x_min) * idx / 6
        svg.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle" font-size="12">{value:.0f}</text>'
        )
    for idx in range(7):
        y = height - bottom - (height - top - bottom) * idx / 6
        value = y_min + (y_max - y_min) * idx / 6
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>'
        )
        svg.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.0f}</text>'
        )
    svg.append(
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>'
    )
    svg.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>'
    )
    svg.append(f'<path class="line" d="{line_path}"/>')
    for row in ordered:
        x = x_px(row["completion_tps_per_user"])
        y = y_px(row["output_token_min_per_gpu"])
        failed = int(row.get("failed_requests") or 0)
        label = f"p={row['parallel']} n={row['number']}"
        if failed:
            label += f" ({failed} failed)"
        svg.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="6"/>')
        svg.append(
            f'<text x="{x+8:.1f}" y="{y-8:.1f}" font-size="12">{sx.escape(label)}</text>'
        )
    svg.append(
        f'<text x="{(left + width - right) / 2:.1f}" y="{height-34}" text-anchor="middle" font-size="15">Completion TPS/user = steady completion tok/s / parallel</text>'
    )
    svg.append(
        f'<text transform="translate(28,{(top + height - bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="15">Output Token/Min/GPU = steady completion tok/s * 60 / num_gpus</text>'
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="EvalScope sweep output root")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--svg", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="Agentic Trie Performance",
        help="SVG chart title",
    )
    args = parser.parse_args()

    rows = collect(args.output_dir, args.num_gpus)
    csv_path = args.csv or args.output_dir / "sweep.csv"
    svg_path = args.svg or args.output_dir / "sweep.svg"
    write_csv(rows, csv_path)
    write_svg(rows, svg_path, args.title)
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"svg={svg_path}")


if __name__ == "__main__":
    main()
