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


def _communication_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in {
        "",
        "unknown",
        "unavailable",
        "n/a",
        "auto",
        "not_recorded",
        "*",
    }


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
    communication_capacities: list[dict[str, Any]] | None = None
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
                self.communication_capacities,
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
        self._validate_communication_capacities(set(ids))
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
            if metadata.get("kind") == "derived":
                derivation = metadata.get("derivation")
                if not isinstance(derivation, dict):
                    raise ValueError("Derived peaks require an explicit derivation")
                base = derivation.get("base_compute_type")
                ratio = derivation.get("throughput_ratio")
                if (
                    key != "bf16_simt"
                    or base != "fp32_simt"
                    or ratio != 2
                    or derivation.get("compute_capability") != "9.0"
                    or self.device.get("compute_capability") != "9.0"
                    or derivation.get("source_id") not in ids
                    or derivation.get("reference") != "cc9_scalar_fma_throughput_ratio"
                    or self.compute_metadata.get(base, {}).get("kind") == "derived"
                    or not _positive(self.compute_flops_per_second.get(base))
                ):
                    raise ValueError("Unsupported derived scalar throughput contract")
                expected = self.compute_flops_per_second[base] * ratio
                if not math.isclose(
                    self.compute_flops_per_second[key], expected, rel_tol=1e-12
                ):
                    raise ValueError(
                        "Derived peak disagrees with source throughput ratio"
                    )
                continue
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

    def _validate_communication_capacities(self, source_ids: set[str]) -> None:
        capacities = self.communication_capacities
        if capacities is None:
            return
        if not isinstance(capacities, list):
            raise ValueError("communication_capacities must be a list")
        identities = set()
        for capacity in capacities:
            if not isinstance(capacity, dict):
                raise ValueError("Communication capacity must be a mapping")
            for key in (
                "id",
                "transport",
                "topology",
                "backend",
                "algorithm",
                "protocol",
                "normalization_reason",
                "conversion_reason",
            ):
                if not _communication_text(capacity.get(key)):
                    raise ValueError(f"Communication capacity requires {key}")
            if capacity["id"] in identities:
                raise ValueError("Communication capacity IDs must be unique")
            identities.add(capacity["id"])
            if (
                capacity.get("scope") != "per_rank"
                or capacity.get("direction") != "one_direction"
                or capacity.get("normalization")
                not in {
                    "nccl_tests_bus_bytes",
                    "trtllm_mnnvl_oneshot_remote_payload",
                }
                or self.device["partition"] != "full_device"
            ):
                raise ValueError(
                    "Communication capacity requires full-device per-rank one-direction bandwidth and a supported traffic model"
                )
            if capacity["normalization"] == "trtllm_mnnvl_oneshot_remote_payload" and (
                capacity["backend"] != "trtllm"
                or capacity["algorithm"] != "oneshot"
                or capacity["protocol"] != "mnnvl"
                or capacity["transport"] != "nvlink"
                or capacity.get("placement") != "single_node"
                or capacity.get("addressing_source_id") not in source_ids
            ):
                raise ValueError(
                    "MNNVL directional payload model requires a sourced single-node NVLink one-shot multicast route"
                )
            if capacity.get("placement") not in {"single_node", "multi_node"}:
                raise ValueError("Communication capacity requires explicit placement")
            if (
                type(capacity.get("world_size")) is not int
                or capacity["world_size"] < 2
            ):
                raise ValueError("Communication capacity requires world_size >= 2")
            for key in ("source_id", "conversion_source_id", "normalization_source_id"):
                if capacity.get(key) not in source_ids:
                    raise ValueError(f"Communication capacity requires a valid {key}")
            value, factor = capacity.get("published_value"), capacity.get(
                "conversion_factor"
            )
            unit, direction = capacity.get("published_unit"), capacity.get(
                "published_direction"
            )
            published_scope = capacity.get("published_scope")
            board = capacity.get("nvlink_board")
            if "nvlink_board" in capacity and (
                not isinstance(board, dict)
                or published_scope != "per_link"
                or capacity["placement"] != "single_node"
                or capacity["topology"] != "single_node_nvswitch"
                or "link_version" in capacity
                or "link_version_source_id" in capacity
                or not _communication_text(capacity.get("link_generation"))
                or any(
                    type(board.get(key)) is not int or not 0 < board[key] <= 0xFFFFFFFF
                    for key in ("pci_device_id", "pci_subsystem_id")
                )
                or type(board.get("switch_count")) is not int
                or board["switch_count"] < 1
                or any(
                    not _communication_text(board.get(key))
                    or board[key] not in source_ids
                    for key in (
                        "pci_source_id",
                        "generation_source_id",
                        "switch_source_id",
                    )
                )
            ):
                raise ValueError(
                    "NVLink board evidence requires sourced PCI IDs, generation and switch count for a single-node per-link capacity"
                )
            links = 1
            if published_scope == "per_link":
                links = capacity.get("active_link_count")
                if (
                    type(links) is not int
                    or links < 1
                    or capacity["transport"] != "nvlink"
                    or not _positive(
                        capacity.get("required_link_rate_bytes_per_second")
                    )
                ):
                    raise ValueError(
                        "Per-link capacity requires an active NVLink count and observed full-rate predicate"
                    )
                if board is None:
                    # This enum namespace only applies to profiles selecting it.
                    generations = {
                        1: "1.0",
                        2: "2.0",
                        3: "2.2",
                        4: "3.0",
                        5: "3.1",
                        6: "4.0",
                        7: "5.0",
                        8: "6.0",
                    }
                    version = capacity.get("link_version")
                    source_id = capacity.get("link_version_source_id")
                    if (
                        type(version) is not int
                        or version not in generations
                        or generations[version] != capacity.get("link_generation")
                        or not _communication_text(source_id)
                        or source_id not in source_ids
                    ):
                        raise ValueError(
                            "NVML link_version enum does not match its sourced link_generation"
                        )
            if (
                not _positive(value)
                or not _positive(factor)
                or unit not in {"byte/s", "GB/s"}
                or direction not in {"one_direction", "bidirectional_sum"}
                or published_scope not in {"per_rank", "per_link"}
                or factor != (0.5 if direction == "bidirectional_sum" else 1)
                or not _positive(capacity.get("bytes_per_second"))
                or not math.isclose(
                    capacity["bytes_per_second"],
                    value * (1e9 if unit == "GB/s" else 1) * factor * links,
                    rel_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Communication capacity needs an explicit supported directional conversion"
                )
            if "bidirectional_bytes_per_second" in capacity and (
                not _positive(capacity["bidirectional_bytes_per_second"])
                or not math.isclose(
                    capacity["bidirectional_bytes_per_second"],
                    2 * capacity["bytes_per_second"],
                    rel_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Symmetric bidirectional capacity must equal twice the directional reference"
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
    # Cost assumptions use this exact prefix for positively known, unmodeled
    # communication. Ordinary missing traffic does not imply communication.
    prefix = "comm_unavailable_reason: "
    unmodeled_reasons = sorted(
        {
            assumption.removeprefix(prefix)
            for cost in costs
            for assumption in cost.get("assumptions", ())
            if assumption.startswith(prefix) and assumption.removeprefix(prefix)
        }
    )
    result: dict[str, Any] = {
        "comm_unavailable_reason": "; ".join(unmodeled_reasons) or None,
        "collective_count": len(traffic),
        "communication_collectives": [item.collective for item in traffic],
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


def scope_metrics(
    entries: list[dict[str, Any]],
    duration_ns: float,
    hardware: HardwareProfile,
    *,
    scope_kind: str | None = None,
    communication_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate rank-local metrics for explicit unique cost entries and GPU duration.

    Callers own scope membership and choose the measured GPU denominator. Returns
    the shared compute, HBM and collective metrics with missing-value reasons.
    Only scope_kind="op" enables communication peak normalization. Optional
    communication_observation is that OpRecord's scalar_args, with comm_* host
    evidence described in docs/hardware-profiles/communication.md. Omission keeps
    effective bandwidth available without claiming a utilization denominator.
    """
    costs = [entry["cost"] for entry in entries if entry["cost"] is not None]
    result: dict[str, Any] = {
        "op_count": len(entries),
        "covered_op_count": len(costs),
        "useful_flops": None,
        "useful_flops_per_second": None,
        "estimated_hbm_bytes": None,
        "estimated_hbm_bytes_per_second": None,
        "recorded_host_mapped_read_bytes": None,
        "no_model_float_compute": None,
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
    result.update(
        _communication_utilization(
            result, hardware, scope_kind, communication_observation
        )
    )
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
    result["recorded_host_mapped_read_bytes"] = sum(
        cost.get("host_mapped_read_bytes") or 0 for cost in costs
    )
    result["useful_flops"] = sum(cost["useful_flops"] for cost in costs)
    result["useful_flops_per_second"] = result["useful_flops"] / seconds
    flops: dict[str, int] = {}
    unresolved = []
    for cost in costs:
        components = cost.get("compute_components")
        if components is None:
            components = [
                {
                    "useful_flops": cost["useful_flops"],
                    "compute_type": cost.get("compute_type"),
                }
            ]
        if (
            any(
                type(item.get("useful_flops")) is not int or item["useful_flops"] < 0
                for item in components
            )
            or sum(item["useful_flops"] for item in components) != cost["useful_flops"]
        ):
            result["mfu_missing_reason"] = result["mfu_total_missing_reason"] = (
                "invalid compute component partition"
            )
            return result
        for item in components:
            amount, compute_type = item["useful_flops"], item.get("compute_type")
            if not amount:
                continue
            if compute_type is None:
                unresolved.append(
                    item.get("missing_reason")
                    or "positive FLOPs have no proven compute type"
                )
                continue
            flops[compute_type] = flops.get(compute_type, 0) + amount
    for compute_type, amount in flops.items():
        peak = hardware.compute_flops_per_second.get(compute_type)
        if peak is None:
            unresolved.append(f"peak unavailable for {compute_type}")
        else:
            result["mfu_by_compute_type"][compute_type] = amount / (peak * seconds)
    result["no_model_float_compute"] = result["useful_flops"] == 0
    result["mfu_total_known"] = sum(result["mfu_by_compute_type"].values())
    if unresolved:
        reason = "; ".join(sorted(set(unresolved)))
        result["mfu_missing_reason"] = result["mfu_total_missing_reason"] = reason
    else:
        result["mfu_total"] = result["mfu_total_known"]
        result["mfu"] = result["mfu_total"]

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


def _nvlink_capacity_missing_reason(
    capacity: dict[str, Any], observation: dict[str, Any], size: int
) -> str | None:
    link_source = observation.get("comm_link_source")
    if not _communication_text(link_source):
        return "active NVLink evidence source unavailable"

    exact_rank_fields = [
        (
            "active link count",
            "comm_rank_active_link_counts",
            capacity["active_link_count"],
        ),
    ]
    rate_fields = ["comm_rank_min_link_rate_bytes_per_second"]
    board = capacity.get("nvlink_board")
    if board is None:
        exact_rank_fields.append(
            ("link version", "comm_rank_link_versions", capacity["link_version"])
        )
    else:
        board_source = observation.get("comm_nvlink_board_source")
        if not _communication_text(board_source):
            return "same-run NVLink board evidence unavailable"
        snapshot = link_source.partition("#rank_evidence")[0]
        if (
            re.fullmatch(r".+#sha256=[0-9a-f]{64}", snapshot) is None
            or any(
                observation[field].partition("#rank_evidence")[0] != snapshot
                for field in (
                    "comm_nvlink_board_source",
                    "comm_topology_source",
                    "comm_placement_source",
                )
            )
            or observation["comm_route_source"].partition("#route_sources")[0]
            != snapshot
        ):
            return "NVLink board, link, topology and route snapshots differ"
        exact_rank_fields.extend(
            (
                (
                    "GPU PCI device ID",
                    "comm_rank_pci_device_ids",
                    board["pci_device_id"],
                ),
                (
                    "GPU PCI subsystem ID",
                    "comm_rank_pci_subsystem_ids",
                    board["pci_subsystem_id"],
                ),
                ("full GPU device", "comm_rank_full_devices", True),
            )
        )
        switches = observation.get("comm_switch_pci_addresses")
        if (
            not isinstance(switches, (list, tuple))
            or len(switches) != board["switch_count"]
            or any(
                not isinstance(address, str)
                or re.fullmatch(
                    r"(?:[0-9a-f]{4}){1,2}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                    address.lower(),
                )
                is None
                for address in switches
            )
            or len({address.lower() for address in switches}) != len(switches)
        ):
            return "observed NVLink switch endpoints do not match capacity"
        peer_counts = observation.get("comm_peer_nvlink_counts")
        if (
            not isinstance(peer_counts, (list, tuple))
            or len(peer_counts) != 1
            or type(peer_counts[0]) is not int
            or peer_counts[0] != capacity["active_link_count"]
        ):
            return "observed NVLink peer matrix does not match capacity"
        rate_fields.append("comm_rank_max_link_rate_bytes_per_second")

    for label, field, expected in exact_rank_fields:
        values = observation.get(field)
        if (
            not isinstance(values, (tuple, list))
            or len(values) != size
            or any(
                type(value) is not type(expected) or value != expected
                for value in values
            )
        ):
            return f"observed rank NVLink {label} does not match capacity"
    for field in rate_fields:
        rates = observation.get(field)
        if (
            not isinstance(rates, (tuple, list))
            or len(rates) != size
            or any(
                not _positive(rate)
                or not math.isclose(
                    rate, capacity["required_link_rate_bytes_per_second"], rel_tol=1e-6
                )
                for rate in rates
            )
        ):
            return "observed rank NVLink rates do not match required full-rate state"
    return None


def _communication_utilization(
    metrics: dict[str, Any],
    hardware: HardwareProfile,
    scope_kind: str | None,
    observation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Match the selected route's traffic model to a directional reference."""
    result: dict[str, Any] = {
        "estimated_comm_utilization": None,
        "estimated_comm_utilization_missing_reason": None,
        "estimated_comm_utilization_warning": None,
        "estimated_comm_utilization_definition": "max(modeled_tx_payload_rate, modeled_remote_rx_payload_rate) / matching_per_rank_one_direction_capacity; whole_Op_GPU_span; excludes unmodeled wire/protocol overhead; not physical saturation",
        "estimated_comm_reference_efficiency": None,
        "estimated_comm_reference_efficiency_definition": "nccl_tests_normalized_busbw / matching_one_direction_reference; not directional payload utilization",
        "estimated_comm_reference_efficiency_warning": None,
        "communication_payload_model": None,
        "estimated_comm_tx_payload_bytes": None,
        "estimated_comm_rx_payload_bytes": None,
        "estimated_comm_tx_payload_bytes_per_second": None,
        "estimated_comm_rx_payload_bytes_per_second": None,
        "estimated_comm_tx_utilization": None,
        "estimated_comm_rx_utilization": None,
        "estimated_comm_tx_capacity_bytes_per_second": None,
        "estimated_comm_rx_capacity_bytes_per_second": None,
        "communication_bottleneck_direction": None,
        "communication_capacity": None,
        "communication_observation": observation,
    }

    def unavailable(reason: str) -> dict[str, Any]:
        result["estimated_comm_utilization_missing_reason"] = reason
        return result

    if scope_kind != "op":
        return unavailable("communication utilization is defined only for an Op")
    if metrics["op_count"] != 1:
        return unavailable(
            "communication utilization requires exactly one Op cost entry"
        )
    if metrics["busbw_missing_reason"] is not None:
        return unavailable(metrics["busbw_missing_reason"])
    if metrics["communication_world_size"] == 1:
        return unavailable(
            "single-rank collective has no inter-rank bandwidth denominator"
        )
    if not hardware.communication_capacities:
        return unavailable(
            "no sourced communication capacity in selected hardware profile"
        )
    if not isinstance(observation, dict):
        return unavailable(
            "selected communication route and placement evidence unavailable"
        )
    for key in (
        "group",
        "transport",
        "topology",
        "backend",
        "algorithm",
        "protocol",
        "transport_source",
        "topology_source",
        "placement_source",
        "route_source",
    ):
        value = observation.get(f"comm_{key}")
        if not _communication_text(value):
            if observation.get("comm_algorithm") == "twoshot" and _communication_text(
                observation.get("comm_offline_binding_missing_reason")
            ):
                return unavailable(observation["comm_offline_binding_missing_reason"])
            return unavailable(f"communication evidence unavailable: comm_{key}")
    size = metrics["communication_world_size"]
    ranks = observation.get("comm_ranks")
    hosts = observation.get("comm_rank_hosts")
    devices = observation.get("comm_rank_device_uuids")
    rank = observation.get("comm_rank")
    if (
        observation["comm_group"] != metrics["communication_group"]
        or type(observation.get("comm_world_size")) is not int
        or observation["comm_world_size"] != size
        or not isinstance(ranks, (tuple, list))
        or len(ranks) != size
        or any(type(item) is not int or item < 0 for item in ranks)
        or len(set(ranks)) != size
        or type(rank) is not int
        or rank not in ranks
    ):
        return unavailable("communication group/rank identity does not match cost")
    if "ranks" in observation and (
        not isinstance(observation["ranks"], (tuple, list))
        or tuple(observation["ranks"]) != tuple(ranks)
    ):
        return unavailable(
            "communication evidence ranks differ from recorded collective ranks"
        )
    for name, values in (("hosts", hosts), ("device UUIDs", devices)):
        if (
            not isinstance(values, (tuple, list))
            or len(values) != size
            or any(not _communication_text(item) for item in values)
        ):
            return unavailable(f"ordered communication rank {name} unavailable")
    normalized_devices = [device.removeprefix("GPU-").lower() for device in devices]
    if len(set(zip(hosts, normalized_devices, strict=True))) != size:
        return unavailable("communication ranks do not map to distinct devices")
    selected_uuid = (hardware.selection or {}).get("observed", {}).get("uuid")
    if (
        selected_uuid is not None
        and normalized_devices[ranks.index(rank)]
        != selected_uuid.removeprefix("GPU-").lower()
    ):
        return unavailable("communication rank device differs from selected hardware")
    placement = "single_node" if len(set(hosts)) == 1 else "multi_node"
    if observation.get("comm_placement") != placement:
        return unavailable("communication placement disagrees with ordered rank hosts")
    twoshot = (
        observation.get("comm_normalization") == "trtllm_mnnvl_twoshot_remote_payload"
    )
    if twoshot:
        if not (
            observation.get("comm_algorithm") == "twoshot"
            and observation.get("comm_offline_binding_verified") is True
            and observation.get("comm_all_rank_mapping_verified") is True
        ):
            return unavailable("MNNVL two-shot route lacks sealed all-rank binding")
    match_fields = (
        "scope",
        "direction",
        "placement",
        "transport",
        "topology",
        "backend",
        "protocol",
        "world_size",
    )
    # Use the compatible physical endpoint peak already in the capture.
    # Two-shot normalization remains tied to the observed native route.
    reference_contract = (
        ("oneshot", "trtllm_mnnvl_oneshot_remote_payload")
        if twoshot
        else (observation.get("comm_algorithm"), observation.get("comm_normalization"))
    )
    matches = [
        capacity
        for capacity in hardware.communication_capacities
        if (capacity["algorithm"], capacity["normalization"]) == reference_contract
        and all(observation.get(f"comm_{key}") == capacity[key] for key in match_fields)
    ]
    if len(matches) != 1:
        return unavailable(
            "selected route/scope/direction/topology does not uniquely match a sourced communication capacity"
        )
    capacity = matches[0]
    if twoshot:
        capacity = {
            **capacity,
            "id": f"{capacity['id']}/two-shot-endpoint-reference",
            "algorithm": "twoshot",
            "normalization": "trtllm_mnnvl_twoshot_remote_payload",
            "endpoint_reference_profile_entry_id": capacity["id"],
            "normalization_reason": "reviewed two-shot peer-unicast scatter and reduced multicast payloads divided by the whole Op GPU span",
        }
    result["communication_capacity"] = capacity
    if capacity["published_scope"] == "per_link":
        reason = _nvlink_capacity_missing_reason(capacity, observation, size)
        if reason is not None:
            return unavailable(reason)
    if capacity["normalization"] == "nccl_tests_bus_bytes":
        value = metrics["busbw_bytes_per_second"] / capacity["bytes_per_second"]
        if not math.isfinite(value) or value < 0:
            return unavailable(
                "communication reference efficiency is not finite and nonnegative"
            )
        result["estimated_comm_reference_efficiency"] = value
        if value > 1:
            result["estimated_comm_reference_efficiency_warning"] = (
                "equivalent bus bandwidth exceeds reference capacity; not physical directional utilization"
            )
        return unavailable(
            "only NCCL-equivalent reference efficiency is available; selected route has no reviewed directional payload model"
        )
    expected_oneshot = not twoshot
    if (
        observation.get("comm_payload_model") != capacity["normalization"]
        or observation.get("comm_addressing")
        != ("nvlink_peer_unicast_then_nvls_multicast" if twoshot else "nvls_multicast")
        or observation.get("comm_allocator_backend") != "CUDA"
        or not _communication_text(observation.get("comm_allocation_source"))
        or not _communication_text(observation.get("comm_payload_model_source"))
        or type(observation.get("comm_pattern")) is not int
        or observation["comm_pattern"] not in (0, 1)
        or observation.get("comm_use_oneshot") is not expected_oneshot
        or observation.get("comm_workspace") != "mnnvl"
        or type(observation.get("comm_payload_element_size")) is not int
        or observation["comm_payload_element_size"] != 2
        or type(observation.get("comm_payload_bytes")) is not int
        or observation["comm_payload_bytes"] != metrics["collective_total_bytes"]
        or observation["comm_payload_bytes"] <= 0
        or observation["comm_payload_bytes"] % 16
        or metrics["communication_collectives"] != ["all_reduce_sum"]
    ):
        algorithm = "two-shot" if twoshot else "one-shot"
        return unavailable(
            f"selected MNNVL {algorithm} directional payload contract is incomplete or differs from collective cost"
        )
    payload = metrics["collective_total_bytes"]
    rx_payload = (size - 1) * payload
    model_details = {
        "tx_definition": "one complete S-byte input stored once to the NVLS multicast mapping",
        "rx_definition": "(P-1)*S useful bytes from the other ranks delivered to this rank's local replica",
    }
    if twoshot:
        tokens = observation.get("comm_payload_tokens")
        row_bytes = observation.get("comm_payload_bytes_per_token")
        if (
            type(tokens) is not int
            or tokens <= 0
            or type(row_bytes) is not int
            or row_bytes <= 0
            or row_bytes % 16
            or tokens * row_bytes != payload
            or not _communication_text(observation.get("comm_native_kernel"))
            or type(observation.get("comm_native_kernel_rowid")) is not int
            or observation["comm_native_kernel_rowid"] <= 0
        ):
            return unavailable("MNNVL two-shot native payload geometry unavailable")
        owned = (tokens + size - 1 - ranks.index(rank)) // size
        scatter_tx = (tokens - owned) * row_bytes
        scatter_rx = (size - 1) * owned * row_bytes
        multicast_tx = owned * row_bytes
        multicast_rx = scatter_tx
        rx_payload = scatter_rx + multicast_rx
        model_details = {
            "tx_definition": "remote owner scatter plus one multicast injection per owned reduced token",
            "rx_definition": "peer contributions to owned tokens plus multicast deliveries for nonowned tokens",
            "owned_tokens": owned,
            "scatter_tx_payload_bytes": scatter_tx,
            "scatter_rx_payload_bytes": scatter_rx,
            "multicast_tx_payload_bytes": multicast_tx,
            "multicast_rx_payload_bytes": multicast_rx,
        }
    tx_rate = metrics["algbw_bytes_per_second"]
    rx_rate = tx_rate * (rx_payload / payload)
    tx_util = tx_rate / capacity["bytes_per_second"]
    rx_util = rx_rate / capacity["bytes_per_second"]
    result.update(
        communication_payload_model={
            "id": capacity["normalization"],
            **model_details,
            "unmodeled": [
                "self-replica physical switch loopback",
                "packet/encoding overhead",
                "retries and protocol control traffic",
            ],
            "local_only": [
                "Lamport polling of local replicas",
                "buffer_flags and dirty-buffer clear",
                "local register reduction and fused epilogue",
            ],
            "source": observation["comm_payload_model_source"],
        },
        estimated_comm_tx_payload_bytes=payload,
        estimated_comm_rx_payload_bytes=rx_payload,
        estimated_comm_tx_payload_bytes_per_second=tx_rate,
        estimated_comm_rx_payload_bytes_per_second=rx_rate,
        estimated_comm_tx_utilization=tx_util,
        estimated_comm_rx_utilization=rx_util,
        estimated_comm_tx_capacity_bytes_per_second=capacity["bytes_per_second"],
        estimated_comm_rx_capacity_bytes_per_second=capacity["bytes_per_second"],
        communication_bottleneck_direction=(
            "rx" if rx_payload > payload else "tx_and_rx"
        ),
    )
    value = max(tx_util, rx_util)
    if not math.isfinite(value) or value < 0:
        return unavailable(
            "communication utilization ratio is not finite and nonnegative"
        )
    # Preserve diagnostic evidence when the payload model exceeds the bound.
    if value > 1:
        result["estimated_comm_utilization_warning"] = (
            "modeled directional payload bandwidth exceeds reference capacity; review timing/model/peak; not measured physical saturation"
        )
    result["estimated_comm_utilization"] = value
    return result


def _communication_utilization_label(metrics: dict[str, Any]) -> str:
    value = metrics.get("estimated_comm_utilization")
    return "Est.CommBWUtil=" + ("N/A" if value is None else f"{value * 100:.3g}%")


def _gpu_duration_label(duration_ns) -> str:
    if (
        type(duration_ns) not in (int, float)
        or not math.isfinite(duration_ns)
        or duration_ns <= 0
    ):
        return "GPU=N/A"
    return f"GPU={duration_ns / 1000:.5g}us"


def _communication_details(metrics: dict[str, Any]) -> str:
    """Return effective communication rates and estimated utilization for one Op."""
    fields = (
        ("algbw_bytes_per_second", "commAlgBW", 1e9, "GB/s"),
        ("busbw_bytes_per_second", "commBusBW", 1e9, "GB/s"),
    )
    parts = []
    if metrics.get("comm_unavailable_reason"):
        parts.append(f"Comm=N/A: {metrics['comm_unavailable_reason']}")
        if not metrics.get("collective_count"):
            return " | ".join(parts)
    for key, title, scale, unit in fields:
        value = metrics.get(key)
        display = "N/A" if value is None else f"{value / scale:.3g}{unit}"
        parts.append(f"{title}={display}")
    if metrics.get("collective_count"):
        parts.append("comm BW denominator=whole Op GPU span")
    if metrics.get("communication_bottleneck_direction"):
        parts.append(
            f"modeled payload direction={metrics['communication_bottleneck_direction']}"
        )
    for key, title in (
        ("busbw_missing_reason", "comm"),
        ("estimated_comm_utilization_missing_reason", "comm util"),
    ):
        if metrics.get(key):
            parts.append(f"{title} N/A: {metrics[key]}")
    if metrics.get("estimated_comm_utilization_warning"):
        parts.append(metrics["estimated_comm_utilization_warning"])
    return " | ".join(parts)


def communication_label(metrics: dict[str, Any]) -> str:
    """Lead with directional utilization, retaining rates and evidence details."""
    return " | ".join(
        filter(
            None,
            (
                _communication_utilization_label(metrics),
                _communication_details(metrics),
            ),
        )
    )


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
        metrics = scope_metrics(
            entries,
            timing.gpu_duration_ns,
            hardware,
            scope_kind=kind,
            communication_observation=(
                record.get("scalar_args") if kind == "op" else None
            ),
        )
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
            metrics["estimated_comm_utilization"] = None
            metrics["estimated_comm_utilization_missing_reason"] = invalid_reason
            metrics["estimated_comm_utilization_warning"] = None
            metrics["estimated_comm_reference_efficiency"] = None
            metrics["estimated_comm_reference_efficiency_warning"] = None
            for key in (
                "estimated_comm_tx_payload_bytes_per_second",
                "estimated_comm_rx_payload_bytes_per_second",
                "estimated_comm_tx_utilization",
                "estimated_comm_rx_utilization",
            ):
                metrics[key] = None

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
            communication = bool(
                metrics["collective_count"] or metrics["comm_unavailable_reason"]
            )
            if communication:
                label += f" | {_communication_utilization_label(metrics)} | {_gpu_duration_label(timing.gpu_duration_ns)}"
            if collective_only:
                label += f" | Est.MBU(HBM)={percentage(metrics['estimated_mbu'])}"
            else:
                label += f" | MFU={percentage(metrics['mfu'])}"
                label += f" Est.MBU(HBM)={percentage(metrics['estimated_mbu'])}"
            if communication:
                label += f" | {_communication_details(metrics)}"
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
