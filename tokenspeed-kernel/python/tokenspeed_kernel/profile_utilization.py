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

"""Cost and timing aggregation shared by metadata export and NVTX labels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenspeed_kernel.profile_costs import CollectiveTraffic


@dataclass(frozen=True)
class HardwareProfile:
    """Rank-local, dense compute peaks and HBM bandwidth in FLOP/s and byte/s.

    Compute keys identify both precision and engine, for example bf16_tensor.
    Missing peaks are unavailable, never inferred from tensor storage dtype.
    source records where these device/partition-specific capacities came from.
    """

    name: str
    compute_flops_per_second: dict[str, float]
    hbm_bytes_per_second: float | None
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.source.strip():
            raise ValueError("Hardware name and peak source must be recorded")
        values = list(self.compute_flops_per_second.values())
        if self.hbm_bytes_per_second is not None:
            values.append(self.hbm_bytes_per_second)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Hardware peaks must be finite and positive")

    @classmethod
    def from_json(cls, path: str | Path) -> HardwareProfile:
        """Read an explicit hardware profile; return its validated capacities."""
        return cls(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ScopeTiming:
    trace_label: str
    # Native tool clock values, not perf_counter_ns or report-relative time.
    start_timestamp: int
    end_timestamp: int
    gpu_duration_ns: float
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.start_timestamp < 0 or self.end_timestamp < self.start_timestamp:
            raise ValueError("Invalid native scope timestamps")


def _bus_metrics(entries: list[dict[str, Any]], duration_ns: float) -> dict[str, Any]:
    costs = [entry["cost"] for entry in entries if entry["cost"] is not None]
    traffic = [
        CollectiveTraffic(**cost["communication"])
        for cost in costs
        if cost.get("communication") is not None
    ]
    result: dict[str, Any] = {
        "collective_count": len(traffic),
        "communication_group": None,
        "communication_world_size": None,
        "collective_total_bytes": None,
        "bus_bytes": None,
        "algbw_bytes_per_second": None,
        "busbw_bytes_per_second": None,
        "busbw_missing_reason": None,
        "busbw_definition": "nccl_tests_normalized_bytes_over_scope_window",
    }
    if len(costs) != len(entries):
        result["busbw_missing_reason"] = "incomplete cost coverage"
    elif not traffic:
        result["busbw_missing_reason"] = "no recorded collective traffic"
    elif len({(item.group, item.world_size) for item in traffic}) != 1:
        result["busbw_missing_reason"] = (
            "mixed communication groups; no aggregate busbw"
        )
    else:
        result["communication_group"] = traffic[0].group
        result["communication_world_size"] = traffic[0].world_size
        result["collective_total_bytes"] = sum(item.total_bytes for item in traffic)
        result["bus_bytes"] = sum(item.bus_bytes for item in traffic)
        if not math.isfinite(duration_ns) or duration_ns <= 0:
            result["busbw_missing_reason"] = "positive finite GPU duration unavailable"
        else:
            seconds = duration_ns * 1e-9
            result["algbw_bytes_per_second"] = (
                result["collective_total_bytes"] / seconds
            )
            result["busbw_bytes_per_second"] = result["bus_bytes"] / seconds
    return result


def _metrics(
    entries: list[dict[str, Any]], duration_ns: float, hardware: HardwareProfile
) -> dict[str, Any]:
    costs = [entry["cost"] for entry in entries if entry["cost"] is not None]
    result: dict[str, Any] = {
        "op_count": len(entries),
        "covered_op_count": len(costs),
        "useful_flops": None,
        "estimated_hbm_bytes": None,
        "mfu": None,
        "mfu_by_compute_type": {},
        "estimated_mbu": None,
        "mfu_missing_reason": None,
        "mbu_missing_reason": None,
        **_bus_metrics(entries, duration_ns),
    }
    reason = None
    if not entries:
        reason = "no opted-in operations"
    elif len(costs) != len(entries):
        reason = "incomplete cost coverage: " + "; ".join(
            sorted(
                {entry["missing_reason"] for entry in entries if entry["cost"] is None}
            )
        )
    elif not math.isfinite(duration_ns) or duration_ns <= 0:
        reason = "positive finite GPU duration unavailable"
    elif any(
        value is not None and (not math.isfinite(value) or value < 0)
        for cost in costs
        for value in (cost["useful_flops"], cost["estimated_hbm_bytes"])
    ):
        reason = "costs must be finite and nonnegative"
    if reason is not None:
        result["mfu_missing_reason"] = result["mbu_missing_reason"] = reason
        return result

    seconds = duration_ns * 1e-9
    result["useful_flops"] = sum(cost["useful_flops"] for cost in costs)
    flops: dict[str, int] = {}
    for cost in costs:
        compute_type = cost.get("compute_type")
        if cost.get("communication") is not None and cost["useful_flops"] == 0:
            continue
        if compute_type is None:
            result["mfu_missing_reason"] = "compute type unavailable"
            continue
        flops[compute_type] = flops.get(compute_type, 0) + cost["useful_flops"]
    for compute_type, amount in flops.items():
        peak = hardware.compute_flops_per_second.get(compute_type)
        if peak is None:
            result["mfu_missing_reason"] = f"peak unavailable for {compute_type}"
        else:
            result["mfu_by_compute_type"][compute_type] = amount / (peak * seconds)
    if len(flops) > 1:
        result["mfu_missing_reason"] = "mixed compute types; no overall normalization"
    if not flops and result["mfu_missing_reason"] is None:
        result["mfu_missing_reason"] = "no model compute FLOPs in collective-only scope"
    if result["mfu_missing_reason"] is None and len(flops) == 1:
        result["mfu"] = next(iter(result["mfu_by_compute_type"].values()))

    if any(cost["estimated_hbm_bytes"] is None for cost in costs):
        result["mbu_missing_reason"] = "HBM traffic estimate unavailable"
    else:
        result["estimated_hbm_bytes"] = sum(
            cost["estimated_hbm_bytes"] for cost in costs
        )
        if hardware.hbm_bytes_per_second is None:
            result["mbu_missing_reason"] = "HBM bandwidth peak unavailable"
        else:
            result["estimated_mbu"] = result["estimated_hbm_bytes"] / (
                hardware.hbm_bytes_per_second * seconds
            )
    return result


def analyze_utilization(
    report: dict[str, Any], timings: list[ScopeTiming], hardware: HardwareProfile
) -> list[dict[str, Any]]:
    """Aggregate unique op costs over each actual call's GPU event window.

    Nested layers include descendant ops once. Repeated calls remain separate.
    Coverage is of opted-in ops, not all device work inside a layer or model.
    """
    by_label = {timing.trace_label: timing for timing in timings}
    if len(by_label) != len(timings):
        raise ValueError("Duplicate scope timing")
    records = [("op", op) for op in report["ops"]] + [
        ("layer", layer) for layer in report["layers"]
    ]
    if set(by_label) != {record["trace_label"] for _, record in records}:
        raise ValueError("Every recorded op/layer needs exactly one timing")
    op_costs = dict(
        zip(
            (op["trace_label"] for op in report["ops"]),
            report["analysis"]["ops"],
            strict=True,
        )
    )
    for label, entry in op_costs.items():
        if by_label[label].unavailable_reason is not None:
            op_costs[label] = {
                **entry,
                "cost": None,
                "missing_reason": by_label[label].unavailable_reason,
            }
    rows = []
    for kind, record in records:
        timing = by_label[record["trace_label"]]
        if kind == "op":
            entries = [op_costs[record["trace_label"]]]
        else:
            entries = [
                entry
                for label, entry in op_costs.items()
                if timing.start_timestamp <= by_label[label].start_timestamp
                and by_label[label].end_timestamp <= timing.end_timestamp
            ]
        metrics = _metrics(entries, timing.gpu_duration_ns, hardware)
        if timing.unavailable_reason is not None:
            metrics["mfu"] = metrics["estimated_mbu"] = None
            metrics["mfu_by_compute_type"] = {}
            metrics["mfu_missing_reason"] = metrics["mbu_missing_reason"] = (
                timing.unavailable_reason
            )
        if record["error"] is not None:
            metrics["mfu"] = metrics["estimated_mbu"] = None
            metrics["mfu_by_compute_type"] = {}
            metrics["mfu_missing_reason"] = metrics["mbu_missing_reason"] = (
                f"{kind} failed: {record['error']}"
            )
        invalid_reason = timing.unavailable_reason
        if record["error"] is not None:
            invalid_reason = f"{kind} failed: {record['error']}"
        if invalid_reason is not None:
            metrics["busbw_bytes_per_second"] = None
            metrics["algbw_bytes_per_second"] = None
            metrics["busbw_missing_reason"] = invalid_reason

        def percentage(value: float | None) -> str:
            return "N/A" if value is None else f"{100 * value:.2f}%"

        # Layer ranges organize the timeline; full identities and aggregate
        # metrics remain in metadata. Only op ranges display utilization.
        label = f"L{record['layer_id']}" if kind == "layer" else record["op_name"]
        collective_only = (
            metrics["collective_count"] > 0
            and metrics["collective_count"] == metrics["op_count"]
        )
        if kind == "op":
            if collective_only:
                label += f" | Est.MBU={percentage(metrics['estimated_mbu'])}"
            else:
                label += f" | MFU={percentage(metrics['mfu'])}"
                label += f" Est.MBU={percentage(metrics['estimated_mbu'])}"
            if metrics["collective_count"] and not collective_only:
                busbw = metrics["busbw_bytes_per_second"]
                display = "N/A" if busbw is None else f"{busbw / 1e9:.2f} GB/s"
                label += f" busbw={display}"
        rows.append(
            {
                "kind": kind,
                "trace_label": record["trace_label"],
                "nvtx_label": label,
                "iter_id": record["iter_id"],
                "layer_id": record["layer_id"],
                "occurrence": record["occurrence"],
                "gpu_duration_ns": timing.gpu_duration_ns,
                "coverage_scope": "opted_in_registered_calls",
                **metrics,
            }
        )
    return rows
