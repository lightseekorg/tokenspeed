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

"""Input contracts and command-line options for the MLP channel example."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputCase:
    id: str
    token_ids: tuple[int, ...]
    text: str | None = None


def validate_cases(value: object, *, max_tokens: int) -> tuple[InputCase, ...]:
    """Reject ambiguous, malformed, duplicate, or silently truncated inputs."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(value, list) or not value:
        raise ValueError("Input file must contain a nonempty JSON array")
    cases: list[InputCase] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not {"id", "token_ids"} <= set(item):
            raise ValueError(f"Input {index} requires id and token_ids")
        if set(item) - {"id", "token_ids", "text"}:
            raise ValueError(f"Input {index} has unknown fields")
        case_id = item["id"]
        if type(case_id) is not str or not case_id.strip() or case_id in seen:
            raise ValueError(f"Input {index} has an empty or duplicate id")
        ids = item["token_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= max_tokens:
            raise ValueError(f"Input {case_id} needs 1..{max_tokens} token IDs")
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"Input {case_id} has a non-integer or negative token ID")
        text = item.get("text")
        if text is not None and type(text) is not str:
            raise ValueError(f"Input {case_id} text must be a string")
        cases.append(InputCase(case_id, tuple(ids), text))
        seen.add(case_id)
    return tuple(cases)


def parse_histogram_edges(value: str) -> tuple[float, ...]:
    """Parse explicit, positive, ascending absolute-activation bin boundaries."""
    try:
        parts = value.split(",")
        if not parts or any(not part.strip() for part in parts):
            raise ValueError
        edges = tuple(float(part) for part in parts)
    except ValueError as error:
        raise ValueError("histogram edges must be comma-separated numbers") from error
    if any(not math.isfinite(edge) or edge <= 0 for edge in edges) or any(
        right <= left for left, right in zip(edges, edges[1:])
    ):
        raise ValueError("histogram edges must be positive and strictly ascending")
    return edges


def _nonnegative(value: float, name: str, *, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")


def validate_mode_options(args: argparse.Namespace) -> tuple[float, ...] | None:
    """Validate every profile threshold before initializing the GPU runtime."""
    if args.max_tokens <= 0 or args.cache_mib <= 0 or args.top_k <= 0:
        raise ValueError("max_tokens, cache_mib and top_k must be positive")
    if args.gpu_id < 0 or not args.model_identity.strip():
        raise ValueError("gpu_id must be nonnegative and model_identity nonempty")
    if args.rows not in ("all", "last"):
        raise ValueError("rows must be all or last")
    threshold_names = (
        "activation_threshold",
        "histogram_edges",
        "max_active_fraction",
        "max_mean_contribution",
        "max_observed_contribution",
        "min_observed_rows",
        "runtime_multiplier",
        "contribution_floor",
        "min_down_norm_ratio",
        "max_outlier_active_fraction",
    )
    if args.mode == "watch":
        if args.watchlist is None:
            raise ValueError("watch mode requires --watchlist")
        if any(getattr(args, name) is not None for name in threshold_names):
            raise ValueError("profile thresholds are only accepted in profile mode")
        return None
    if args.watchlist is not None:
        raise ValueError("profile mode does not accept --watchlist")
    missing = [name for name in threshold_names if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "profile mode requires explicit thresholds: " + ", ".join(missing)
        )
    edges = parse_histogram_edges(args.histogram_edges)
    for name in (
        "activation_threshold",
        "max_active_fraction",
        "max_mean_contribution",
        "max_observed_contribution",
        "runtime_multiplier",
        "min_down_norm_ratio",
        "max_outlier_active_fraction",
    ):
        _nonnegative(getattr(args, name), name)
    _nonnegative(args.contribution_floor, "contribution_floor", positive=True)
    _nonnegative(args.min_down_norm_ratio, "min_down_norm_ratio", positive=True)
    if not 0 <= args.max_active_fraction <= 1:
        raise ValueError("max_active_fraction must be in [0, 1]")
    if args.runtime_multiplier < 1:
        raise ValueError("runtime_multiplier must be at least 1")
    if args.max_outlier_active_fraction > 1:
        raise ValueError("max_outlier_active_fraction must be in [0, 1]")
    if args.min_observed_rows <= 0:
        raise ValueError("min_observed_rows must be positive")
    return edges


def build_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=("profile", "watch"), required=True)
    parser.add_argument(
        "--rows",
        choices=("all", "last"),
        required=True,
        help="Measure every physical token row, or only each input's final row",
    )
    parser.add_argument("--model", required=True, help="Checkpoint path or Hub ID")
    parser.add_argument(
        "--revision", required=True, help="Declared checkpoint revision"
    )
    parser.add_argument(
        "--model-identity",
        required=True,
        help="Caller-supplied identity of loaded weights; watchlist must match",
    )
    parser.add_argument(
        "--inputs", required=True, type=Path, help="JSON array of token-ID records"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New output directory"
    )
    parser.add_argument(
        "--watchlist", type=Path, help="Frozen profile watchlist, for watch mode"
    )
    parser.add_argument("--norm-mode", choices=("fused", "unfused"), required=True)
    parser.add_argument("--attention-backend", choices=("mha", "triton"), required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cache-mib", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5, help="Next-token IDs to report")
    parser.add_argument("--activation-threshold", type=float)
    parser.add_argument(
        "--histogram-edges", help="Comma-separated positive bin boundaries"
    )
    parser.add_argument("--max-active-fraction", type=float)
    parser.add_argument("--max-mean-contribution", type=float)
    parser.add_argument("--max-observed-contribution", type=float)
    parser.add_argument("--min-observed-rows", type=int)
    parser.add_argument("--runtime-multiplier", type=float)
    parser.add_argument("--contribution-floor", type=float)
    parser.add_argument("--min-down-norm-ratio", type=float)
    parser.add_argument("--max-outlier-active-fraction", type=float)
    return parser
