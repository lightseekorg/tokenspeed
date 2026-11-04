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

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from tokenspeed_kernel.profile_costs import CollectiveTraffic


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _read_hardware(content: str, suffix: str) -> dict[str, Any]:
    if suffix.lower() in {".yaml", ".yml"}:
        try:
            from yaml import safe_load
        except ImportError as exc:
            raise ImportError(
                "YAML hardware profiles require tokenspeed-kernel[profiling]"
            ) from exc
        data = safe_load(content)
    else:
        data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Hardware profile must be a mapping")
    return data


def _partition_from_inventory(cuda_uuid: str, inventory: str) -> dict[str, Any]:
    """Match CUDA's UUID to a full GPU or MIG UUID; absence is not proof."""

    def normalized(value: str) -> str:
        return value.removeprefix("GPU-").removeprefix("MIG-").lower()

    matches = []
    for line in inventory.splitlines():
        identity = re.search(r"\(UUID:\s*([^\s)]+)\)", line)
        if identity is None or normalized(identity.group(1)) != normalized(cuda_uuid):
            continue
        if line.lstrip().startswith("MIG "):
            # CI prefixes distinguish a subdivided compute instance from its GI.
            # Unknown label suffixes must not yield a partial GI identity.
            shape = re.match(
                r"\s*MIG\s+((?:\d+c\.)?\d+g\.\d+gb)\s+Device\s+\d+\s*:", line
            )
            matches.append(
                {"partition": "mig", "mig_profile": shape.group(1) if shape else None}
            )
        elif line.startswith("GPU "):
            matches.append({"partition": "full_device"})
    if len(matches) != 1:
        raise ValueError(
            "CUDA UUID does not uniquely establish full-device/MIG identity"
        )
    return {
        **matches[0],
        "partition_evidence": {
            "method": "CUDA UUID matched to nvidia-smi -L",
            "inventory": inventory,
        },
    }


@dataclass(frozen=True)
class HardwareProfile:
    """Rank-local dense peaks; legacy profiles remain readable without SKU proof.

    Version 2 records exact device scope, explicit units and published-to-dense
    conversions. Input bytes and their hash travel with every report; runtime
    observations audit the explicit selection, never supply theoretical peaks.
    """

    name: str
    compute_flops_per_second: dict[str, float]
    hbm_bytes_per_second: float | None
    source: str
    schema_version: int = 1
    device: dict[str, Any] | None = None
    units: dict[str, str] | None = None
    compute_metadata: dict[str, dict[str, Any]] | None = None
    sources: list[dict[str, Any]] | None = None
    bandwidth_source_id: str | None = None
    input_snapshot: dict[str, Any] | None = field(default=None, compare=False)
    selection: dict[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.source, str)
            or not self.source.strip()
        ):
            raise ValueError("Hardware name and peak source must be recorded")
        if not isinstance(self.compute_flops_per_second, dict) or any(
            not isinstance(key, str) or not key.strip()
            for key in self.compute_flops_per_second
        ):
            raise ValueError("Compute peaks must have explicit compute type keys")
        values = list(self.compute_flops_per_second.values())
        if self.hbm_bytes_per_second is not None:
            values.append(self.hbm_bytes_per_second)
        if not all(_positive(value) for value in values):
            raise ValueError("Hardware peaks must be finite and positive")
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError("Unsupported hardware schema_version")
        if self.schema_version == 2:
            self._validate_v2()
        elif any(
            value is not None
            for value in (
                self.device,
                self.units,
                self.compute_metadata,
                self.sources,
                self.bandwidth_source_id,
            )
        ):
            raise ValueError("Structured hardware identity requires schema_version: 2")
        if self.input_snapshot is not None:
            snapshot = self.input_snapshot
            if not isinstance(snapshot, dict) or not isinstance(
                snapshot.get("content"), str
            ):
                raise ValueError("Invalid hardware input snapshot")
            if hashlib.sha256(
                snapshot["content"].encode("utf-8")
            ).hexdigest() != snapshot.get("sha256"):
                raise ValueError("Hardware input snapshot SHA256 mismatch")
            original = _read_hardware(snapshot["content"], snapshot["suffix"])
            original.pop("input_snapshot", None)
            original.pop("selection", None)
            if replace(self, input_snapshot=None, selection=None) != HardwareProfile(
                **original
            ):
                raise ValueError(
                    "Hardware capacities differ from the saved input snapshot"
                )

    def _validate_v2(self) -> None:
        device = self.device
        if not isinstance(device, dict) or device.get("vendor") != "NVIDIA":
            raise ValueError("Version 2 requires an explicit NVIDIA device identity")
        for key in ("sku", "form_factor"):
            if not isinstance(device.get(key), str) or not device[key].strip():
                raise ValueError(f"Missing device {key}")
        if not _positive(device.get("memory_gb")) or not _positive(
            device.get("power_watts")
        ):
            raise ValueError(
                "Device memory_gb and nominal power_watts must be positive"
            )
        if device.get("partition") not in {"full_device", "mig"}:
            raise ValueError("Device partition must be full_device or mig")
        if device["partition"] == "mig":
            if (
                not isinstance(device.get("mig_profile"), str)
                or not device["mig_profile"].strip()
            ):
                raise ValueError("MIG profiles require the exact mig_profile")
        elif device.get("mig_profile") is not None:
            raise ValueError("Full-device profiles cannot carry a MIG profile")
        names = device.get("match_names")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("match_names must contain unique exact device names")
        if self.units != {"compute": "FLOP/s", "hbm": "byte/s"}:
            raise ValueError("Hardware units must be FLOP/s and byte/s")
        if not isinstance(self.sources, list) or not self.sources:
            raise ValueError("Structured official sources are required")
        ids = []
        for source in self.sources:
            if not isinstance(source, dict) or any(
                not isinstance(source.get(key), str) or not source[key].strip()
                for key in ("id", "title", "url", "accessed")
            ):
                raise ValueError("Sources require id, title, url and accessed")
            if not source["url"].startswith("https://"):
                raise ValueError("Hardware sources must use HTTPS URLs")
            ids.append(source["id"])
        if len(ids) != len(set(ids)):
            raise ValueError("Hardware source IDs must be unique")
        if (
            self.hbm_bytes_per_second is not None
            and self.bandwidth_source_id not in ids
        ):
            raise ValueError("Bandwidth needs an explicit source ID")
        if not isinstance(self.compute_metadata, dict) or set(
            self.compute_metadata
        ) != set(self.compute_flops_per_second):
            raise ValueError("Every compute peak needs matching explicit metadata")
        for key, metadata in self.compute_metadata.items():
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid compute metadata for {key}")
            precision, engine = metadata.get("precision"), metadata.get("engine")
            if (
                not isinstance(precision, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", precision) is None
                or engine not in {"tensor", "simt"}
                or key != f"{precision}_{engine}"
            ):
                raise ValueError(
                    f"Compute key {key} must identify precision and engine"
                )
            if (
                metadata.get("sparsity") != "dense"
                or metadata.get("source_id") not in ids
            ):
                raise ValueError(
                    "Compute peaks require dense semantics and a source ID"
                )
            published_sparsity = metadata.get("published_sparsity")
            factor = metadata.get("dense_conversion_factor")
            if (
                published_sparsity not in {"dense", "sparse"}
                or type(factor) not in (int, float)
                or factor != (0.5 if published_sparsity == "sparse" else 1)
            ):
                raise ValueError(
                    "Published sparsity requires an explicit dense conversion (1 or 0.5)"
                )
            if engine == "simt" and published_sparsity != "dense":
                raise ValueError(
                    "SIMT peaks cannot use Tensor Core sparsity conversion"
                )
            unit = metadata.get("published_unit")
            value = metadata.get("published_value")
            if unit not in {"FLOP/s", "TFLOP/s"} or not _positive(value):
                raise ValueError(
                    "Published compute peaks need explicit values and units"
                )
            expected = value * (1e12 if unit == "TFLOP/s" else 1) * factor
            if not math.isclose(
                self.compute_flops_per_second[key], expected, rel_tol=1e-12
            ):
                raise ValueError(
                    f"Dense peak for {key} disagrees with its published conversion"
                )

    @classmethod
    def from_file(cls, path: str | Path) -> HardwareProfile:
        """Read the existing JSON/YAML format, retaining exact UTF-8 input bytes."""
        path = Path(path).resolve()
        raw = path.read_bytes()
        content = raw.decode("utf-8")
        data = _read_hardware(content, path.suffix)
        profile = cls(**data)
        return replace(
            profile,
            input_snapshot={
                "path": str(path),
                "suffix": path.suffix,
                "content": content,
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> HardwareProfile:
        """Retain the existing JSON entry point through the common reader."""
        return cls.from_file(path)

    @classmethod
    def from_selection(cls, path: str | Path | None) -> HardwareProfile:
        """Choose an explicit path or TOKENSPEED_PROFILE_HARDWARE; never a GPU default."""
        selected = (
            path if path is not None else os.environ.get("TOKENSPEED_PROFILE_HARDWARE")
        )
        if not selected:
            raise ValueError("Supply --hardware or TOKENSPEED_PROFILE_HARDWARE")
        profile = cls.from_file(selected)
        environment_path = os.environ.get("TOKENSPEED_PROFILE_HARDWARE")
        matches_environment = (
            environment_path is not None
            and Path(environment_path).resolve() == Path(selected).resolve()
        )
        expected = (
            os.environ.get("TOKENSPEED_PROFILE_HARDWARE_SHA256")
            if path is None or matches_environment
            else None
        )
        if expected is not None and profile.input_snapshot["sha256"] != expected:
            raise ValueError(
                "Worker hardware profile SHA256 differs from launcher selection"
            )
        return profile

    def validate_device(self, observed: dict[str, Any]) -> HardwareProfile:
        """Audit an explicit selection against known observed identity fields.

        CUDA cannot establish every board's form factor or nominal TDP. Those
        remain recorded as unverified; configured power limits are observations,
        never substituted for published peak specifications.
        """
        if self.schema_version == 1:
            return replace(
                self,
                selection={
                    "status": "legacy_unverified",
                    "observed": observed,
                    "unverified_fields": [
                        "sku",
                        "form_factor",
                        "memory_gb",
                        "power_watts",
                        "partition",
                    ],
                },
            )
        device = self.device
        if observed.get("name") not in device["match_names"]:
            raise ValueError(
                f"Selected hardware {self.name} does not match observed device {observed.get('name')}"
            )
        if observed.get("partition") != device["partition"]:
            raise ValueError(
                "Selected full-device/MIG scope does not match observed device"
            )
        if (
            device["partition"] == "mig"
            and observed.get("mig_profile") != device["mig_profile"]
        ):
            raise ValueError("Selected MIG profile does not match observed device")
        verified = ["name", "partition"]
        for key in ("sku", "form_factor", "memory_gb", "power_watts"):
            if key in observed:
                if observed[key] != device[key]:
                    raise ValueError(
                        f"Selected hardware {key} does not match observed device"
                    )
                verified.append(key)
        return replace(
            self,
            selection={
                "status": "explicit_selection_checked",
                "observed": observed,
                "verified_fields": verified,
                "unverified_fields": [
                    key
                    for key in ("sku", "form_factor", "memory_gb", "power_watts")
                    if key not in verified
                ],
            },
        )

    def validate_cuda_device(self, device: Any) -> HardwareProfile:
        """Read CUDA identity outside capture, then reject known selection mismatches."""
        import torch

        properties = torch.cuda.get_device_properties(device)
        name = properties.name
        try:
            cuda_uuid = properties.uuid
        except AttributeError:
            cuda_uuid = None
        if isinstance(cuda_uuid, bytes) and len(cuda_uuid) == 16:
            cuda_uuid = str(uuid.UUID(bytes=cuda_uuid))
        else:
            cuda_uuid = str(cuda_uuid) if cuda_uuid is not None else None
        observed = {
            "name": name,
            "total_memory_bytes": properties.total_memory,
            "uuid": cuda_uuid,
            "partition": "unknown",
        }
        if self.schema_version == 2:
            if cuda_uuid is None:
                raise ValueError(
                    "CUDA device UUID unavailable; supply an independently verified device identity"
                )
            try:
                inventory = subprocess.run(
                    ["nvidia-smi", "-L"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError(
                    "Cannot verify full-device/MIG identity with nvidia-smi -L"
                ) from exc
            observed.update(_partition_from_inventory(cuda_uuid, inventory))
        if self.schema_version == 2:
            # CUDA reports allocatable memory, below nominal capacity. Reject
            # different memory SKUs, but permit reserved memory and decimal/GiB labels.
            actual_gb = properties.total_memory / 1e9
            expected_gb = self.device["memory_gb"]
            if not expected_gb * 0.9 <= actual_gb <= expected_gb * 1.1:
                raise ValueError(
                    "Selected hardware memory capacity does not match CUDA device"
                )
        return self.validate_device(observed)


def hardware_for_analysis(
    recorded: dict[str, Any], path: str | Path | None
) -> HardwareProfile:
    """Preserve capture selection; an explicit offline path must be equivalent."""
    profile = HardwareProfile(**recorded)
    if profile.schema_version == 1 and profile.selection is None:
        profile = profile.validate_device({})
    if path is not None:
        selected = HardwareProfile.from_file(path)
        if profile != selected:
            raise ValueError("Offline hardware selection differs from the capture")
        if (
            profile.input_snapshot is not None
            and profile.input_snapshot["sha256"] != selected.input_snapshot["sha256"]
        ):
            raise ValueError("Offline hardware input SHA256 differs from the capture")
        if profile.input_snapshot is None:
            profile = replace(selected, selection=profile.selection)
    return profile


@dataclass(frozen=True)
class ScopeTiming:
    trace_label: str
    # Nsight-exported CPU range timestamps; used only for nesting.
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
        "useful_flops_per_second": None,
        "estimated_hbm_bytes": None,
        "estimated_hbm_bytes_per_second": None,
        "mfu": None,
        "mfu_total": None,
        "mfu_total_known": None,
        "mfu_total_missing_reason": None,
        "mfu_total_definition": "sum(useful_flops[compute_type]/peak_flops_per_second[compute_type])/selected_gpu_duration_seconds",
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
        result["mfu_total_missing_reason"] = reason
        return result

    seconds = duration_ns * 1e-9
    result["useful_flops"] = sum(cost["useful_flops"] for cost in costs)
    result["useful_flops_per_second"] = result["useful_flops"] / seconds
    flops: dict[str, int] = {}
    for cost in costs:
        compute_type = cost.get("compute_type")
        if cost["useful_flops"] == 0:
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
    if not flops and result["mfu_missing_reason"] is None:
        result["mfu_missing_reason"] = "no model compute FLOPs in scope"
    result["mfu_total_known"] = (
        sum(result["mfu_by_compute_type"].values())
        if result["mfu_by_compute_type"]
        else None
    )
    result["mfu_total_missing_reason"] = result["mfu_missing_reason"]
    if result["mfu_total_missing_reason"] is None:
        result["mfu_total"] = result["mfu_total_known"]
    if len(flops) > 1:
        result["mfu_missing_reason"] = "mixed compute types; no overall normalization"
    if not flops and result["mfu_missing_reason"] is None:
        result["mfu_missing_reason"] = "no model compute FLOPs in scope"
    if result["mfu_missing_reason"] is None and len(flops) == 1:
        result["mfu"] = next(iter(result["mfu_by_compute_type"].values()))

    if any(cost["estimated_hbm_bytes"] is None for cost in costs):
        result["mbu_missing_reason"] = "HBM traffic estimate unavailable"
    else:
        result["estimated_hbm_bytes"] = sum(
            cost["estimated_hbm_bytes"] for cost in costs
        )
        result["estimated_hbm_bytes_per_second"] = (
            result["estimated_hbm_bytes"] / seconds
        )
        if hardware.hbm_bytes_per_second is None:
            result["mbu_missing_reason"] = "HBM bandwidth peak unavailable"
        else:
            result["estimated_mbu"] = (
                result["estimated_hbm_bytes_per_second"] / hardware.hbm_bytes_per_second
            )
    return result


def analyze_utilization(
    report: dict[str, Any], timings: list[ScopeTiming], hardware: HardwareProfile
) -> list[dict[str, Any]]:
    """Aggregate unique op costs over each actual call's Nsight GPU projection.

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
        invalid_reason = timing.unavailable_reason
        if record["error"] is not None:
            invalid_reason = f"{kind} failed: {record['error']}"
        if invalid_reason is not None:
            metrics["useful_flops_per_second"] = None
            metrics["estimated_hbm_bytes_per_second"] = None
            metrics["mfu"] = metrics["estimated_mbu"] = None
            metrics["mfu_total"] = metrics["mfu_total_known"] = None
            metrics["mfu_total_missing_reason"] = invalid_reason
            metrics["mfu_by_compute_type"] = {}
            metrics["mfu_missing_reason"] = metrics["mbu_missing_reason"] = (
                invalid_reason
            )
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
            and all(entry["cost"]["useful_flops"] == 0 for entry in entries)
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
