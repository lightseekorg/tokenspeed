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

"""Pure report construction and channel selection for activity profiles."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerHostTotals:
    """One layer's aggregates after the profiler copies them to the host."""

    rows: int
    sum_abs: list[float]
    sum_square: list[float]
    max_abs: list[float]
    active_count: list[int]
    sum_contribution: list[float]
    max_contribution: list[float]
    histograms: list[list[int]]
    down_column_norms: list[float]


@dataclass(frozen=True)
class WatchlistThresholds:
    max_active_fraction: float
    max_mean_contribution: float
    max_observed_contribution: float
    min_down_norm_ratio: float
    max_outlier_active_fraction: float
    min_observed_rows: int
    runtime_multiplier: float
    contribution_floor: float


def nonnegative_finite(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _channel_row(
    layer_index: int, channel: int, totals: LayerHostTotals, down_norm_median: float
) -> dict:
    rows = totals.rows
    down_norm = totals.down_column_norms[channel]
    return {
        "layer": layer_index,
        "channel": channel,
        "sample_count": rows,
        "mean_abs": totals.sum_abs[channel] / rows,
        "rms": math.sqrt(totals.sum_square[channel] / rows),
        "max_abs": totals.max_abs[channel],
        "active_count": totals.active_count[channel],
        "active_fraction": totals.active_count[channel] / rows,
        "down_column_norm": down_norm,
        "down_norm_ratio_to_layer_median": (
            down_norm / down_norm_median if down_norm_median > 0 else None
        ),
        "mean_contribution": totals.sum_contribution[channel] / rows,
        "max_contribution": totals.max_contribution[channel],
        "histogram_counts": totals.histograms[channel],
    }


def build_layer_report(layer_index: int, totals: LayerHostTotals) -> dict:
    down_norm_median = statistics.median(totals.down_column_norms)
    return {
        "layer": layer_index,
        "down_norm_median": down_norm_median,
        "channels": [
            _channel_row(layer_index, channel, totals, down_norm_median)
            for channel in range(len(totals.sum_abs))
        ],
    }


def validate_candidate_limit(limit: int | None) -> int | None:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    return limit


def rank_profile_candidates(profile: dict, *, limit: int | None) -> list[dict]:
    rows = [channel for layer in profile["layers"] for channel in layer["channels"]]
    rows.sort(
        key=lambda channel: (
            channel["active_fraction"],
            channel["mean_contribution"],
            channel["layer"],
            channel["channel"],
        )
    )
    return rows if limit is None else rows[:limit]


def validate_watchlist_thresholds(
    *,
    max_active_fraction: float,
    max_mean_contribution: float,
    max_observed_contribution: float,
    min_down_norm_ratio: float,
    max_outlier_active_fraction: float,
    min_observed_rows: int,
    runtime_multiplier: float,
    contribution_floor: float,
) -> WatchlistThresholds:
    max_active_fraction = nonnegative_finite(max_active_fraction, "max_active_fraction")
    max_mean_contribution = nonnegative_finite(
        max_mean_contribution, "max_mean_contribution"
    )
    max_observed_contribution = nonnegative_finite(
        max_observed_contribution, "max_observed_contribution"
    )
    min_down_norm_ratio = nonnegative_finite(min_down_norm_ratio, "min_down_norm_ratio")
    max_outlier_active_fraction = nonnegative_finite(
        max_outlier_active_fraction, "max_outlier_active_fraction"
    )
    runtime_multiplier = nonnegative_finite(runtime_multiplier, "runtime_multiplier")
    contribution_floor = nonnegative_finite(contribution_floor, "contribution_floor")
    if (
        max_active_fraction > 1
        or min_down_norm_ratio <= 0
        or max_outlier_active_fraction > 1
        or runtime_multiplier < 1
        or contribution_floor <= 0
    ):
        raise ValueError(
            "Require fractions <= 1, norm ratio > 0, multiplier >= 1, " "and floor > 0"
        )
    if type(min_observed_rows) is not int or min_observed_rows <= 0:
        raise ValueError("min_observed_rows must be a positive integer")
    return WatchlistThresholds(
        max_active_fraction=max_active_fraction,
        max_mean_contribution=max_mean_contribution,
        max_observed_contribution=max_observed_contribution,
        min_down_norm_ratio=min_down_norm_ratio,
        max_outlier_active_fraction=max_outlier_active_fraction,
        min_observed_rows=min_observed_rows,
        runtime_multiplier=runtime_multiplier,
        contribution_floor=contribution_floor,
    )


def selection_reasons(
    channel: dict,
    *,
    max_active_fraction: float,
    max_mean_contribution: float,
    max_observed_contribution: float,
    min_down_norm_ratio: float,
    max_outlier_active_fraction: float,
) -> list[str]:
    """Explain the same two eligibility gates used by the frozen watchlist."""
    reasons = []
    if (
        channel["active_fraction"] <= max_active_fraction
        and channel["mean_contribution"] <= max_mean_contribution
        and channel["max_contribution"] <= max_observed_contribution
    ):
        reasons.append("quiet_activity")
    ratio = channel["down_norm_ratio_to_layer_median"]
    if (
        ratio is not None
        and ratio >= min_down_norm_ratio
        and channel["active_fraction"] <= max_outlier_active_fraction
    ):
        reasons.append("large_down_column_with_low_activity")
    return reasons


def select_watchlist(profile: dict, thresholds: WatchlistThresholds) -> dict:
    if any(
        layer["channels"][0]["sample_count"] < thresholds.min_observed_rows
        for layer in profile["layers"]
    ):
        raise ValueError("Profile has fewer rows than min_observed_rows")
    entries = []
    for layer in profile["layers"]:
        for channel in layer["channels"]:
            if not selection_reasons(
                channel,
                max_active_fraction=thresholds.max_active_fraction,
                max_mean_contribution=thresholds.max_mean_contribution,
                max_observed_contribution=thresholds.max_observed_contribution,
                min_down_norm_ratio=thresholds.min_down_norm_ratio,
                max_outlier_active_fraction=thresholds.max_outlier_active_fraction,
            ):
                continue
            contribution_limit = max(
                channel["max_contribution"] * thresholds.runtime_multiplier,
                thresholds.contribution_floor,
            )
            if not math.isfinite(contribution_limit):
                raise ValueError("Calculated contribution limit is nonfinite")
            entries.append(
                {
                    "layer": channel["layer"],
                    "channel": channel["channel"],
                    "contribution_limit": contribution_limit,
                }
            )
    entries.sort(key=lambda entry: (entry["layer"], entry["channel"]))
    return {
        "schema": "channel-watchlist-v1",
        "model_identity": profile["model_identity"],
        "entries": entries,
    }
