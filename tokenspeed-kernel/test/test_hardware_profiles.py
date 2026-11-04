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

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from tokenspeed_kernel.op_profile import OpRecord, TensorDescription
from tokenspeed_kernel.profile_costs import DenseMatmulCost
from tokenspeed_kernel.profile_utilization import (
    HardwareProfile,
    _partition_from_inventory,
    hardware_for_analysis,
)

PROFILES = Path(__file__).resolve().parents[2] / "docs" / "hardware-profiles"


@pytest.mark.parametrize("path", sorted(PROFILES.glob("*.yaml")), ids=lambda p: p.stem)
def test_published_profiles_roundtrip_with_input_provenance(path):
    profile = HardwareProfile.from_file(path)
    assert profile.schema_version == 2
    assert (
        profile.input_snapshot["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert profile.input_snapshot["content"] == path.read_text()
    report = json.loads(json.dumps(asdict(profile)))
    assert hardware_for_analysis(report, path) == profile
    assert hardware_for_analysis(report, None).input_snapshot == profile.input_snapshot


@pytest.fixture
def hardware():
    return HardwareProfile.from_file(PROFILES / "nvidia-h100-sxm-80gb.yaml")


@pytest.mark.parametrize(
    "change", ["units", "engine", "conversion", "sparse", "source", "scope", "names"]
)
def test_invalid_structured_profiles_rejected(hardware, change):
    data = asdict(replace(hardware, input_snapshot=None))
    if change == "units":
        data["units"]["compute"] = "TFLOP/s"
    elif change == "engine":
        data["compute_metadata"]["bf16_tensor"]["engine"] = "simt"
    elif change == "conversion":
        data["compute_metadata"]["bf16_tensor"]["dense_conversion_factor"] = 1
    elif change == "sparse":
        data["compute_metadata"]["bf16_tensor"]["sparsity"] = "sparse"
    elif change == "source":
        data["compute_metadata"]["bf16_tensor"]["source_id"] = "missing"
    elif change == "scope":
        data["device"]["partition"] = "mig"
    else:
        data["device"]["match_names"] *= 2
    with pytest.raises(ValueError):
        HardwareProfile(**data)


def test_explicit_selection_identity_and_mig_checks(hardware):
    observed = {"name": hardware.device["match_names"][0], "partition": "full_device"}
    selected = hardware.validate_device(observed)
    assert selected.selection["status"] == "explicit_selection_checked"
    assert "form_factor" in selected.selection["unverified_fields"]
    assert selected.input_snapshot == hardware.input_snapshot
    for mismatch in (
        {"name": "NVIDIA H200"},
        {"partition": "mig"},
        {"memory_gb": 94},
        {"form_factor": "PCIe"},
    ):
        with pytest.raises(ValueError, match="match"):
            hardware.validate_device({**observed, **mismatch})
    mig_data = asdict(replace(hardware, input_snapshot=None))
    mig_data["device"].update(partition="mig", mig_profile="1g.10gb", memory_gb=10)
    mig = HardwareProfile(**mig_data)
    with pytest.raises(ValueError, match="MIG profile"):
        mig.validate_device({**observed, "partition": "mig", "mig_profile": "2g.20gb"})


def test_snapshot_tampering_and_offline_override_rejected(hardware, tmp_path):
    changed = asdict(hardware)
    changed["input_snapshot"]["content"] += "\n"
    with pytest.raises(ValueError, match="SHA256"):
        HardwareProfile(**changed)
    changed = asdict(hardware)
    changed["hbm_bytes_per_second"] *= 2
    with pytest.raises(ValueError, match="saved input"):
        HardwareProfile(**changed)
    alternate = tmp_path / "hardware.yaml"
    alternate.write_text(hardware.input_snapshot["content"] + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        hardware_for_analysis(asdict(hardware), alternate)
    changed = asdict(replace(hardware, input_snapshot=None))
    changed["hbm_bytes_per_second"] *= 2
    alternate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs"):
        hardware_for_analysis(asdict(hardware), alternate)


def test_explicit_environment_selection_has_no_default(hardware, monkeypatch):
    monkeypatch.delenv("TOKENSPEED_PROFILE_HARDWARE", raising=False)
    monkeypatch.delenv("TOKENSPEED_PROFILE_HARDWARE_SHA256", raising=False)
    with pytest.raises(ValueError, match="Supply --hardware"):
        HardwareProfile.from_selection(None)
    monkeypatch.setenv("TOKENSPEED_PROFILE_HARDWARE", hardware.input_snapshot["path"])
    assert HardwareProfile.from_selection(None) == hardware
    monkeypatch.setenv("TOKENSPEED_PROFILE_HARDWARE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="launcher"):
        HardwareProfile.from_selection(None)


def test_legacy_four_fields_remain_readable_and_identity_unverified(tmp_path):
    path = tmp_path / "legacy.json"
    data = dict(
        name="fixture",
        compute_flops_per_second={"bf16_tensor": 1000},
        hbm_bytes_per_second=None,
        source="analytical fixture",
    )
    path.write_text(json.dumps(data))
    profile = HardwareProfile.from_json(path)
    assert profile.schema_version == 1
    assert profile.compute_flops_per_second == data["compute_flops_per_second"]
    assert (
        profile.validate_device({"name": "arbitrary"}).selection["status"]
        == "legacy_unverified"
    )
    assert hardware_for_analysis(data, path).input_snapshot == profile.input_snapshot


def test_matmul_engine_requires_explicit_contract_not_storage_dtype():
    a = TensorDescription(
        shape=(2, 3),
        dtype="torch.bfloat16",
        stride=(3, 1),
        device="cuda:0",
        logical_bytes=12,
    )
    b = replace(a, shape=(3, 4), stride=(4, 1), logical_bytes=24)
    out = replace(a, shape=(2, 4), stride=(4, 1), logical_bytes=16)
    op = OpRecord(
        iter_id=0,
        layer_id=0,
        occurrence=0,
        inner_id=0,
        op_name="gemm",
        implementation="fixture",
        cost_rule="dense",
        inputs={"a": a, "b": b},
        outputs={"out": out},
        cpu_start_ns=0,
        cpu_duration_ns=0,
        error=None,
        trace_label="fixture",
    )
    rule = DenseMatmulCost("a", "b", "out")
    cost = rule(op)
    assert cost.compute_type is None
    assert cost.useful_flops == 48
    explicit = rule(replace(op, static_traits={"compute_type": "bf16_tensor"}))
    assert explicit.compute_type == "bf16_tensor"
    assert explicit.useful_flops == cost.useful_flops
    dynamic = rule(
        replace(
            op,
            static_traits={"compute_type": "bf16_tensor"},
            scalar_args={"compute_type": "bf16_simt"},
        )
    )
    assert dynamic.compute_type == "bf16_simt"


def test_partition_matching_requires_exact_uuid_evidence():
    inventory = (
        "GPU 0: NVIDIA H100 (UUID: GPU-aaa)\n  MIG 1g.10gb Device 0: (UUID: MIG-bbb)\n"
    )
    assert _partition_from_inventory("aaa", inventory)["partition"] == "full_device"
    assert _partition_from_inventory("bbb", inventory)["partition"] == "mig"
    assert _partition_from_inventory("MIG-bbb", inventory)["mig_profile"] == "1g.10gb"
    with pytest.raises(ValueError, match="uniquely"):
        _partition_from_inventory("ccc", inventory)
    with pytest.raises(ValueError, match="uniquely"):
        _partition_from_inventory("aaa", inventory + inventory)


def test_native_display_prefers_explicit_total_and_labels_partial():
    from tokenspeed_kernel.profile_native_report import _mfu_display

    old = dict(mfu=None, mfu_known=0.2, mfu_missing_reason="unknown peak")
    assert _mfu_display(old) == ("20% (partial)", "unknown peak")
    complete = {
        **old,
        "mfu_total": 0.36,
        "mfu_total_known": 0.36,
        "mfu_total_missing_reason": None,
    }
    assert _mfu_display(complete) == ("36%", None)
    partial = {
        **complete,
        "mfu_total": None,
        "mfu_total_missing_reason": "peak unavailable",
    }
    assert _mfu_display(partial) == ("36% (partial)", "peak unavailable")


def test_cuda_uuid_mig_partition_cannot_masquerade_as_full_gpu(hardware, monkeypatch):
    import subprocess
    from dataclasses import dataclass

    import torch

    @dataclass
    class DeviceProperties:
        name: str
        total_memory: int
        uuid: str

    # A full-size MIG instance can have the full board's nominal memory.
    properties = DeviceProperties(
        hardware.device["match_names"][0], 80_000_000_000, "bbb"
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    inventory = (
        "GPU 0: NVIDIA H100 (UUID: GPU-aaa)\n  MIG 7g.80gb Device 0: (UUID: MIG-bbb)\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=inventory
        ),
    )
    with pytest.raises(ValueError, match="full-device/MIG"):
        hardware.validate_cuda_device(0)
    properties.uuid = "aaa"
    selected = hardware.validate_cuda_device(0)
    assert selected.selection["observed"]["partition"] == "full_device"
    assert (
        selected.selection["observed"]["partition_evidence"]["inventory"] == inventory
    )


def test_explicit_downstream_path_preserves_frozen_launcher_hash(hardware, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_PROFILE_HARDWARE", hardware.input_snapshot["path"])
    monkeypatch.setenv("TOKENSPEED_PROFILE_HARDWARE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="launcher"):
        HardwareProfile.from_selection(hardware.input_snapshot["path"])


def test_future_explicit_precision_can_be_described_without_schema_code_changes(
    hardware,
):
    data = asdict(replace(hardware, input_snapshot=None))
    value = data["compute_flops_per_second"].pop("fp8_e4m3_tensor")
    metadata = data["compute_metadata"].pop("fp8_e4m3_tensor")
    metadata["precision"] = "fp4_e2m1"
    data["compute_flops_per_second"]["fp4_e2m1_tensor"] = value
    data["compute_metadata"]["fp4_e2m1_tensor"] = metadata
    # Synthetic schema exercise; this does not publish an FP4 peak for H100.
    assert (
        HardwareProfile(**data).compute_metadata["fp4_e2m1_tensor"]["precision"]
        == "fp4_e2m1"
    )


@pytest.mark.parametrize("label", ["1c.3g.40gb", "2c.3g.40gb", "1g.10gb"])
def test_mig_compute_instance_label_matches_exact_selected_profile(hardware, label):
    inventory = f"GPU 0: NVIDIA H100 (UUID: GPU-parent)\n  MIG {label}  Device  0: (UUID: MIG-child)\n"
    observed = _partition_from_inventory("MIG-child", inventory)
    assert observed["mig_profile"] == label
    device = {**hardware.device, "partition": "mig", "mig_profile": label}
    # Synthetic capacities test identity only; no MIG peaks are derived or published.
    mig = replace(hardware, device=device, input_snapshot=None)
    observed["name"] = device["match_names"][0]
    assert mig.validate_device(observed).selection["observed"]["mig_profile"] == label
    if "c." in label:
        full_gi = replace(
            mig, device={**device, "mig_profile": label.split("c.", 1)[1]}
        )
        with pytest.raises(ValueError, match="MIG profile"):
            full_gi.validate_device(observed)


@pytest.mark.parametrize("label", ["unknown", "prefix.3g.40gb", "1c.3g.40gb.unknown"])
def test_unknown_mig_label_does_not_supply_partial_profile_identity(hardware, label):
    inventory = f"GPU 0: NVIDIA H100 (UUID: GPU-parent)\n  MIG {label} Device 0: (UUID: MIG-child)\n"
    observed = _partition_from_inventory("MIG-child", inventory)
    assert observed["mig_profile"] is None
    device = {**hardware.device, "partition": "mig", "mig_profile": "3g.40gb"}
    mig = replace(hardware, device=device, input_snapshot=None)
    with pytest.raises(ValueError, match="MIG profile"):
        mig.validate_device({**observed, "name": device["match_names"][0]})
