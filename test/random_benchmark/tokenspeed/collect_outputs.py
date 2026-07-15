#!/usr/bin/env python3
"""Collect random perf sweeps into one CI summary table."""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
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
    "input_32k": 32768,
}

ExpectedCell = tuple[str, int, int]


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
    """Collect rows using the historical best-effort behavior."""
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
        output_tps = _float(summary, "Output Throughput (tok/s)")
        cache_hit = _float(summary, "KV Cache Hit Rate (%)")
        decoded_per_iter = _float(summary, "Decoded Tok/Iter") or _float(
            summary, "Avg Decoded Tokens/Iter"
        )

        rows.append(
            {
                "config": config,
                "Conc.": int(_float(summary, "Concurrency")),
                "Latency (tps/user)": round(tps_user, 2),
                "Throughput (tps/gpu)": round(output_tps / num_gpus, 2),
                "Approx Cache Hit": round(cache_hit, 2),
                "Decoded Tok/Iter": round(decoded_per_iter, 4),
            }
        )
    rows.sort(key=_sort_key)
    return rows


def parse_expected_cell(value: str) -> ExpectedCell:
    """Parse ``CONFIG:CONCURRENCY:REQUESTS`` from the command line."""
    parts = value.split(":")
    if len(parts) != 3 or not parts[0].strip():
        raise argparse.ArgumentTypeError(
            "expected CONFIG:CONCURRENCY:REQUESTS, " f"got {value!r}"
        )

    config = parts[0].strip()
    try:
        concurrency = int(parts[1])
        requests = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "concurrency and requests must be integers in " f"{value!r}"
        ) from exc
    if concurrency <= 0 or requests <= 0:
        raise argparse.ArgumentTypeError(
            "concurrency and requests must be positive in " f"{value!r}"
        )
    return config, concurrency, requests


def _normalize_expected_cells(
    expected_cells: list[ExpectedCell],
) -> dict[tuple[str, int], int]:
    expected = {}
    for config, concurrency, requests in expected_cells:
        key = (config, concurrency)
        if key in expected:
            raise ValueError("duplicate --expect-cell for " f"{config}:{concurrency}")
        expected[key] = requests
    return expected


def _strict_int(summary: dict, key: str, source: Path) -> int:
    if key not in summary:
        raise ValueError(f"{source}: missing {key!r}")
    value = summary[key]
    if type(value) is not int:
        raise ValueError(f"{source}: {key!r} must be an integer, got {value!r}")
    return value


def _strict_positive_number(summary: dict, key: str, source: Path) -> float:
    if key not in summary:
        raise ValueError(f"{source}: missing {key!r}")
    value = summary[key]
    if isinstance(value, bool):
        raise ValueError(
            f"{source}: {key!r} must be a positive finite number, got {value!r}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: {key!r} must be a positive finite number, got {value!r}"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{source}: {key!r} must be a positive finite number, got {value!r}"
        )
    return number


def _strict_decoded_per_iter(summary: dict, source: Path) -> float:
    # EvalScope 1.8 emits the first key. Keep the older spelling readable,
    # but never hide a malformed primary value by falling back to the alias.
    if "Decoded Tok/Iter" in summary:
        key = "Decoded Tok/Iter"
    else:
        key = "Avg Decoded Tokens/Iter"
    return _strict_positive_number(summary, key, source)


def _cell_sort_key(key: tuple[str, int]) -> tuple[int, str, int]:
    config, concurrency = key
    return (INPUT_ORDER.get(config, 0), config, concurrency)


def collect_and_validate(
    sweep_dir: Path,
    num_gpus: int,
    expected_cells: list[ExpectedCell],
) -> tuple[list[dict], dict]:
    """Collect and strictly validate an explicitly declared sweep matrix."""
    expected = _normalize_expected_cells(expected_cells)
    failures = []
    rows = []
    cells = []
    observed_keys: dict[tuple[str, int], list[Path]] = {}
    summary_paths = sorted(sweep_dir.rglob("benchmark_summary.json"))

    for summary_path in summary_paths:
        config = _config_from_path(sweep_dir, summary_path)
        if config is None:
            failures.append(
                f"{summary_path}: cannot derive an input_* config from path"
            )
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"{summary_path}: cannot read benchmark summary: {exc}")
            continue
        if not isinstance(summary, dict):
            failures.append(f"{summary_path}: benchmark summary must be an object")
            continue

        try:
            concurrency = _strict_int(summary, "Concurrency", summary_path)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if concurrency <= 0:
            failures.append(
                f"{summary_path}: 'Concurrency' must be positive, got {concurrency}"
            )
            continue

        key = (config, concurrency)
        prior_paths = observed_keys.setdefault(key, [])
        if prior_paths:
            failures.append(
                "duplicate cell "
                f"{config}:{concurrency}: {prior_paths[0]} and {summary_path}"
            )
        prior_paths.append(summary_path)

        cell = {
            "config": config,
            "concurrency": concurrency,
            "summary_path": str(summary_path.relative_to(sweep_dir)),
            "total": None,
            "success": None,
            "failed": None,
            "tpot_ms": None,
            "output_throughput_tok_s": None,
            "output_throughput_per_gpu_tok_s": None,
            "decoded_tok_per_iter": None,
        }

        for field, output_key in (
            ("Total Requests", "total"),
            ("Success Requests", "success"),
            ("Failed Requests", "failed"),
        ):
            try:
                cell[output_key] = _strict_int(summary, field, summary_path)
            except ValueError as exc:
                failures.append(str(exc))

        metric_values = {}
        for field, output_key in (
            ("TPOT (ms)", "tpot_ms"),
            ("Output Throughput (tok/s)", "output_throughput_tok_s"),
        ):
            try:
                metric_values[output_key] = _strict_positive_number(
                    summary, field, summary_path
                )
                cell[output_key] = metric_values[output_key]
            except ValueError as exc:
                failures.append(str(exc))
        try:
            metric_values["decoded_tok_per_iter"] = _strict_decoded_per_iter(
                summary, summary_path
            )
            cell["decoded_tok_per_iter"] = metric_values["decoded_tok_per_iter"]
        except ValueError as exc:
            failures.append(str(exc))

        output_tps = metric_values.get("output_throughput_tok_s")
        if output_tps is not None:
            cell["output_throughput_per_gpu_tok_s"] = output_tps / num_gpus

        expected_requests = expected.get(key)
        if expected_requests is not None:
            for output_key, expected_value in (
                ("total", expected_requests),
                ("success", expected_requests),
                ("failed", 0),
            ):
                actual_value = cell[output_key]
                if actual_value is not None and actual_value != expected_value:
                    failures.append(
                        f"{summary_path}: {output_key} requests "
                        f"{actual_value} != expected {expected_value}"
                    )

        required_metrics = (
            "tpot_ms",
            "output_throughput_tok_s",
            "decoded_tok_per_iter",
        )
        if all(metric in metric_values for metric in required_metrics):
            cache_hit = _float(summary, "KV Cache Hit Rate (%)")
            if not math.isfinite(cache_hit):
                cache_hit = 0.0
            rows.append(
                {
                    "config": config,
                    "Conc.": concurrency,
                    "Latency (tps/user)": round(1000.0 / metric_values["tpot_ms"], 2),
                    "Throughput (tps/gpu)": round(output_tps / num_gpus, 2),
                    "Approx Cache Hit": round(cache_hit, 2),
                    "Decoded Tok/Iter": round(metric_values["decoded_tok_per_iter"], 4),
                }
            )
        cells.append(cell)

    observed_set = set(observed_keys)
    expected_set = set(expected)
    for config, concurrency in sorted(expected_set - observed_set, key=_cell_sort_key):
        failures.append(f"missing expected cell {config}:{concurrency}")
    for config, concurrency in sorted(observed_set - expected_set, key=_cell_sort_key):
        failures.append(f"unexpected cell {config}:{concurrency}")

    expected_requests = sum(expected.values())
    request_totals = {
        "expected": expected_requests,
        "total": sum(cell["total"] or 0 for cell in cells),
        "success": sum(cell["success"] or 0 for cell in cells),
        "failed": sum(cell["failed"] or 0 for cell in cells),
    }
    for key, expected_value in (
        ("total", expected_requests),
        ("success", expected_requests),
        ("failed", 0),
    ):
        if request_totals[key] != expected_value:
            failures.append(
                f"aggregate {key} requests {request_totals[key]} "
                f"!= expected {expected_value}"
            )

    rows.sort(key=_sort_key)
    cells.sort(
        key=lambda cell: (
            *_cell_sort_key((cell["config"], cell["concurrency"])),
            cell["summary_path"],
        )
    )
    expected_specs = [
        {"config": config, "concurrency": concurrency, "requests": requests}
        for (config, concurrency), requests in sorted(
            expected.items(), key=lambda item: _cell_sort_key(item[0])
        )
    ]
    report = {
        "schema_version": 1,
        "mode": "strict",
        "status": "pass" if not failures else "fail",
        "expected_cells": len(expected),
        "observed_summary_files": len(summary_paths),
        "observed_cells": len(cells),
        "observed_unique_cells": len(observed_set),
        "expected_cell_specs": expected_specs,
        "cells": cells,
        "request_totals": request_totals,
        "failures": failures,
    }
    return rows, report


def write_json_atomic(path: Path, value: dict) -> None:
    """Atomically write a JSON artifact in ``path``'s directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def print_table(rows):
    # The CI pipeline recognizes this marker, parses the following CSV for
    # perf-reference checks, and adds the block to the GitHub step summary.
    print("\nOverall perf table:")
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--expect-cell",
        action="append",
        default=[],
        type=parse_expected_cell,
        metavar="CONFIG:CONCURRENCY:REQUESTS",
        help="require exactly this benchmark cell; repeat for the full matrix",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        help="atomically write the collection and validation result",
    )
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"Not a directory: {args.sweep_dir}")
    if args.num_gpus <= 0:
        sys.exit("--num-gpus must be positive")

    strict = bool(args.expect_cell)
    if strict:
        try:
            rows, validation = collect_and_validate(
                args.sweep_dir, args.num_gpus, args.expect_cell
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        rows = collect(args.sweep_dir, args.num_gpus)
        validation = {
            "schema_version": 1,
            "mode": "legacy",
            "status": "pass" if rows else "fail",
            "observed_cells": len(rows),
            "failures": [] if rows else ["no benchmark summary rows found"],
        }
    print_table(rows)
    if args.validation_json is not None:
        write_json_atomic(args.validation_json, validation)
    if strict and validation["status"] != "pass":
        for failure in validation["failures"]:
            print(f"[error] {failure}", file=sys.stderr)
        sys.exit(1)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
