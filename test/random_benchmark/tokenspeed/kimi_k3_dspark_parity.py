#!/usr/bin/env python3
"""Summarize and compare env-gated DSpark parity tensor dumps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def load_manifest(root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    records = {}
    for manifest in sorted(root.glob("manifest-rank*.jsonl")):
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["stage"], int(record["index"]), int(record["rank"]))
            if key in records:
                raise ValueError(f"duplicate parity record: {key}")
            records[key] = record
    return records


def _tensor(root: Path, record: dict[str, Any]) -> torch.Tensor:
    path = Path(record["path"])
    if not path.exists():
        path = root / path.name
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["tensor"]


def compare_dumps(
    actual_root: Path,
    reference_root: Path,
    *,
    hidden_cosine_floor: float = 0.999,
    max_abs_floor: float = 0.02,
) -> dict[str, Any]:
    actual = load_manifest(actual_root)
    reference = load_manifest(reference_root)
    keys = sorted(actual.keys() | reference.keys())
    rows = []
    passed = True

    for key in keys:
        actual_record = actual.get(key)
        reference_record = reference.get(key)
        row: dict[str, Any] = {
            "stage": key[0],
            "index": key[1],
            "rank": key[2],
        }
        if actual_record is None or reference_record is None:
            row["verdict"] = "missing"
            row["missing_from"] = "actual" if actual_record is None else "reference"
            passed = False
            rows.append(row)
            continue

        actual_tensor = _tensor(actual_root, actual_record)
        reference_tensor = _tensor(reference_root, reference_record)
        row["actual_shape"] = list(actual_tensor.shape)
        row["reference_shape"] = list(reference_tensor.shape)
        if actual_tensor.shape != reference_tensor.shape:
            row["verdict"] = "shape_mismatch"
            passed = False
            rows.append(row)
            continue

        if not (
            actual_tensor.is_floating_point() or reference_tensor.is_floating_point()
        ):
            equal = torch.equal(actual_tensor, reference_tensor)
            row.update(
                {
                    "exact_equal": equal,
                    "mismatch_count": int((actual_tensor != reference_tensor).sum()),
                    "verdict": "pass" if equal else "token_mismatch",
                }
            )
            passed &= equal
            rows.append(row)
            continue

        lhs = actual_tensor.float().reshape(-1)
        rhs = reference_tensor.float().reshape(-1)
        diff = (lhs - rhs).abs()
        lhs_norm = torch.linalg.vector_norm(lhs)
        rhs_norm = torch.linalg.vector_norm(rhs)
        if lhs.numel() == 0:
            cosine = 1.0
        elif lhs_norm == 0 or rhs_norm == 0:
            cosine = 1.0 if torch.equal(lhs, rhs) else 0.0
        else:
            cosine = float(torch.dot(lhs, rhs) / (lhs_norm * rhs_norm))
        max_abs = float(diff.max()) if diff.numel() else 0.0
        mean_abs = float(diff.mean()) if diff.numel() else 0.0
        finite = math.isfinite(cosine) and math.isfinite(max_abs)
        tensor_passed = (
            finite and cosine >= hidden_cosine_floor and max_abs <= max_abs_floor
        )
        row.update(
            {
                "cosine": cosine,
                "max_abs_err": max_abs,
                "mean_abs_err": mean_abs,
                "verdict": "pass" if tensor_passed else "numeric_mismatch",
            }
        )
        passed &= tensor_passed
        rows.append(row)

    return {
        "actual_root": str(actual_root),
        "reference_root": str(reference_root),
        "hidden_cosine_floor": hidden_cosine_floor,
        "max_abs_floor": max_abs_floor,
        "passed": passed,
        "records": rows,
    }


def summarize_dump(root: Path) -> dict[str, Any]:
    records = load_manifest(root)
    by_stage: dict[str, int] = {}
    for stage, _, _ in records:
        by_stage[stage] = by_stage.get(stage, 0) + 1
    return {
        "root": str(root),
        "record_count": len(records),
        "records_by_stage": dict(sorted(by_stage.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("root", type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("actual", type=Path)
    compare.add_argument("reference", type=Path)
    compare.add_argument("--hidden-cosine-floor", type=float, default=0.999)
    compare.add_argument("--max-abs-floor", type=float, default=0.02)
    compare.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "summarize":
        report = summarize_dump(args.root)
    else:
        report = compare_dumps(
            args.actual,
            args.reference,
            hidden_cosine_floor=args.hidden_cosine_floor,
            max_abs_floor=args.max_abs_floor,
        )
        if args.output is not None:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
