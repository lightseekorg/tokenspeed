# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fit nonnegative GDS costs and validate held-out geometries in seconds."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from test.cache_gds_calibration.data import integer, positive, read_rows

import numpy as np


def features(row, kind):
    if kind == "prefill":
        # Persisted timing records retain the original mathematical field names.
        cached_tokens = integer(row["a"], 0)
        prompt_tokens = integer(row["b"], 1)
        if prompt_tokens <= cached_tokens:
            raise ValueError("prefill sample requires prompt_tokens > cached_tokens")
        uncached_tokens = prompt_tokens - cached_tokens
        return (
            uncached_tokens,
            uncached_tokens * (prompt_tokens + cached_tokens) / 200000,
        )
    if kind != "disk":
        raise ValueError(f"unknown cost model: {kind}")
    n, size = integer(row["read_blocks"], 1), integer(row["aligned_bytes"], 4096)
    if size % 4096:
        raise ValueError("Disk bytes must include 4096-byte alignment")
    return n, size


def constrained_fit(x, y, weighting):
    """Two-column nonnegative least squares, with normalized columns."""
    if weighting not in ("absolute", "relative"):
        raise ValueError(f"unknown weighting: {weighting}")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape != (len(y), 2) or y.ndim != 1:
        raise ValueError("fit requires two feature columns and one target per row")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("fit requires finite features and positive finite times")
    weights = np.ones(len(y)) if weighting == "absolute" else 1 / y
    weighted = x * weights[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    if np.any(norms == 0):
        raise ValueError("unidentifiable coefficients: zero feature column")
    scaled = weighted / norms
    condition = float(np.linalg.cond(scaled))
    if not math.isfinite(condition) or condition > 1e5:
        raise ValueError(
            "unidentifiable coefficients: vary context lengths / Disk block sizes"
        )
    target = y * weights
    candidates = [np.zeros(2)]
    both = np.linalg.lstsq(scaled, target, rcond=None)[0]
    if np.all(both >= 0):
        candidates.append(both)
    for col in range(2):
        candidate = np.zeros(2)
        candidate[col] = max(0.0, float(scaled[:, col] @ target))
        candidates.append(candidate)
    best = min(candidates, key=lambda c: float(np.linalg.norm(scaled @ c - target)))
    return best / norms, condition


def fit_one(rows, kind, weighting):
    if not rows or {row["kind"] for row in rows} != {kind}:
        raise ValueError(f"expected nonempty {kind} samples")
    if len({row["deployment"] for row in rows}) != 1:
        raise ValueError("do not mix deployments in one fit")
    if kind == "disk":
        fingerprints = {
            (
                r["time_basis"],
                r["native"]["backend"],
                r["native"]["inflight"],
                r["native"]["queue_depth"],
                r["native"]["max_io_bytes"],
            )
            for r in rows
        }
        if len(fingerprints) != 1 or any(not r["native"]["direct_io"] for r in rows):
            raise ValueError(
                "do not mix Disk backends, concurrency, time bases, or buffered I/O"
            )
    elif len({r["timing_source"] for r in rows}) != 1:
        raise ValueError("do not mix prefill timing sources")
    grouped = {}
    for row in rows:
        key = (row["split"], *features(row, kind))
        if key[0] not in ("train", "validation"):
            raise ValueError("every sample must be train or validation")
        grouped.setdefault(key, []).append(positive(row["seconds"]))
    train_keys = {k[1:] for k in grouped if k[0] == "train"}
    validation_keys = {k[1:] for k in grouped if k[0] == "validation"}
    if train_keys & validation_keys:
        raise ValueError(
            "hold out whole geometries, not repeats of training geometries"
        )
    train = [(key, values) for key, values in grouped.items() if key[0] == "train"]
    validation = [
        (key, values) for key, values in grouped.items() if key[0] == "validation"
    ]
    if len(train) < 3 or len(validation) < 2:
        raise ValueError("need >=3 training and >=2 held-out geometries")
    if min(map(len, grouped.values())) < 3:
        raise ValueError("need >=3 repeats per geometry")
    coefs, condition = constrained_fit(
        [k[1:] for k, _ in train], [statistics.median(v) for _, v in train], weighting
    )
    if not np.all(np.isfinite(coefs)) or coefs[0 if kind == "prefill" else 1] <= 0:
        raise ValueError("fit reached a forbidden zero coefficient; inspect data/model")
    residuals = []
    for key, values in grouped.items():
        median = statistics.median(values)
        prediction = float(np.dot(key[1:], coefs))
        residuals.append(
            {
                "split": key[0],
                "features": key[1:],
                "repeats": len(values),
                "median_s": median,
                "predicted_s": prediction,
                "relative_error": abs(prediction - median) / median,
                "relative_mad": statistics.median(abs(x - median) for x in values)
                / median,
            }
        )
    errors = [r["relative_error"] for r in residuals if r["split"] == "validation"]
    return {
        "coefficients": coefs.tolist(),
        "condition_number": condition,
        "validation_mape": statistics.mean(errors),
        "validation_max_error": max(errors),
        "max_relative_mad": max(r["relative_mad"] for r in residuals),
        "geometry_medians": residuals,
    }


def fit(args):
    for value in (
        args.disk_scale,
        args.max_mape,
        args.max_error,
        args.max_relative_mad,
    ):
        positive(value)
    prefill, disk = read_rows(args.prefill), read_rows(args.disk)
    if {r["deployment"] for r in prefill} != {r["deployment"] for r in disk}:
        raise ValueError("prefill and Disk must target the same deployment")
    p, d = (
        fit_one(prefill, "prefill", args.weighting),
        fit_one(disk, "disk", args.weighting),
    )
    p["observed_range"] = {
        "min_a": min(r["a"] for r in prefill),
        "max_b": max(r["b"] for r in prefill),
        "min_suffix": min(r["b"] - r["a"] for r in prefill),
        "max_suffix": max(r["b"] - r["a"] for r in prefill),
    }
    p["timing_source"] = prefill[0]["timing_source"]
    d["observed_range"] = {
        "min_block_bytes": min(r["aligned_bytes"] / r["read_blocks"] for r in disk),
        "max_block_bytes": max(r["aligned_bytes"] / r["read_blocks"] for r in disk),
    }
    d["measurement"] = {
        "time_basis": disk[0]["time_basis"],
        **{
            key: disk[0]["native"][key]
            for key in ("backend", "inflight", "queue_depth", "max_io_bytes")
        },
    }
    report = {
        "schema": 1,
        "units": "seconds",
        "deployment": prefill[0]["deployment"],
        "weighting": args.weighting,
        "disk_scale": positive(args.disk_scale),
        "acceptance_limits": {
            "validation_mape": args.max_mape,
            "validation_max_error": args.max_error,
            "max_relative_mad": args.max_relative_mad,
        },
        "prefill": p,
        "disk": d,
        "sources_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in args.prefill + args.disk
        },
    }
    report["accepted"] = all(
        r["validation_mape"] <= args.max_mape
        and r["validation_max_error"] <= args.max_error
        and r["max_relative_mad"] <= args.max_relative_mad
        for r in (p, d)
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if not report["accepted"]:
        raise ValueError(
            f"fit failed acceptance thresholds; inspect {output / 'report.json'}"
        )
    config = {
        "prefix_cache_evict_policy": "gds",
        "prefix_cache_gds_prefill_cost": p["coefficients"],
        "prefix_cache_gds_disk_read_cost": [
            value * args.disk_scale for value in d["coefficients"]
        ],
    }
    if not all(
        math.isfinite(value) for value in config["prefix_cache_gds_disk_read_cost"]
    ):
        raise ValueError("scaled Disk coefficients are not finite")
    (output / "gds_args.json").write_text(json.dumps(config, indent=2) + "\n")
    D, E = config["prefix_cache_gds_prefill_cost"]
    A, B = config["prefix_cache_gds_disk_read_cost"]
    (output / "gds_args.txt").write_text(
        f"--prefix-cache-evict-policy gds\n--prefix-cache-gds-prefill-cost {D:.12g} {E:.12g}\n"
        f"--prefix-cache-gds-disk-read-cost {A:.12g} {B:.12g}\n"
    )
    print(json.dumps(config, indent=2))
