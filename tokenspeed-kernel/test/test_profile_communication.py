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
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from tokenspeed_kernel import profile_communication as communication
from tokenspeed_kernel.op_profile import OpCost
from tokenspeed_kernel.profile_costs import CollectiveTraffic
from tokenspeed_kernel.profile_utilization import HardwareProfile, scope_metrics


def communication_receipts(directory, *, rows=None, pattern=0):
    """CPU-authored rank receipts with real installed source identities."""

    def write(name, value):
        contents = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(contents).hexdigest()
        path = directory / name.format(sha=digest)
        path.write_bytes(contents)
        return path, digest

    source_root = Path(communication.__file__).parent
    source_paths = ("ops/communication/trtllm.py", *communication._MNNVL_ROUTE_SOURCES)
    sources = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in source_paths
    }
    if rows is None:
        rows = [
            dict(
                rank=rank,
                pid=100 + rank,
                host="fixture-host",
                uuid=f"GPU-fixture-{rank}",
                physical_index=rank,
                switch_device_type=1,
                links=[
                    dict(
                        active=True,
                        remote_device_type=1,
                        remote_pci="switch:0",
                        version=4,
                        raw_status_bytes_per_second=25e9,
                    )
                ],
            )
            for rank in range(2)
        ]
        rows[0]["topology"] = {
            "returncode": 0,
            "stdout": "\x1b[4mGPU0 GPU1 CPU Affinity\x1b[0m\nGPU0 X NV18 0-7\nGPU1 NV18 X 0-7\n",
        }
    ranks = list(range(len(rows)))
    config = {"world_size": len(ranks)}
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    allocation_paths = []
    snapshot_sha = None
    for rank in ranks:
        write(
            f"rank-{rank}.config.json",
            dict(run_id="fixture", rank=rank, config=config, sha256=config_sha),
        )
        _, peer_sha = write(
            f"rank-{rank}.communication.json",
            dict(
                schema="tokenspeed.communication_evidence/v1",
                run_id="fixture",
                rank=rank,
                config_sha256=config_sha,
                rank_evidence=rows,
                route_sources=sources,
            ),
        )
        if rank == 0:
            snapshot_sha = peer_sha
        allocation = dict(
            mapping_verified=True,
            allocator_backend="CUDA",
            world_size=len(ranks),
            peer_count=len(ranks),
            rank=rank,
            ranks=ranks,
            group_name=None,
            uuid=f"GPU-fixture-{rank}",
            local_ptr=4096 + rank * 4096,
            multicast_ptr=16384,
            buffer_size=8192,
        )
        path, _ = write(
            f"rank-{rank}.mnnvl-allocation-{{sha}}.json",
            dict(
                schema="tokenspeed.mnnvl_allocation_evidence/v1",
                run_id="fixture",
                rank=rank,
                config_sha256=config_sha,
                communication_snapshot_sha256=peer_sha,
                allocation=allocation,
            ),
        )
        allocation_paths.append(path)
    capture = dict(
        run_id="fixture",
        rank=0,
        process_id=100,
        ops=[
            dict(
                trace_label="fixture-op",
                scalar_args=dict(
                    ranks=ranks,
                    group_name=None,
                    comm_backend="trtllm",
                    comm_workspace="mnnvl",
                    comm_protocol="mnnvl",
                    comm_algorithm="oneshot",
                    comm_pattern=pattern,
                    comm_use_oneshot=True,
                    comm_payload_element_size=2,
                    comm_placement_source=f"{directory / 'rank-0.communication.json'}#sha256={snapshot_sha}#rank_evidence",
                    logical_batch=1,
                    query_width=1,
                    profile_argument_num_global_tokens=1,
                    comm_payload_bytes=8192,
                ),
            )
        ],
    )
    return capture, rows, allocation_paths


def test_ansi_nvswitch_evidence_requires_real_peer_links(tmp_path):
    _, rows, _ = communication_receipts(tmp_path)
    assert communication._nvlink_group(rows, rows[0])
    # Formatting is harmless; a PCIe-only peer route is contrary evidence.
    contrary = deepcopy(rows[0])
    contrary["topology"]["stdout"] = contrary["topology"]["stdout"].replace(
        "NV18", "PIX"
    )
    assert not communication._nvlink_group(rows, contrary)


def test_cached_communication_binding_preserves_each_replays_live_scalars(tmp_path):
    capture, _, allocations = communication_receipts(tmp_path)
    original = deepcopy(capture)
    bindings = communication.bind_capture_communication(capture, tmp_path)
    delta = bindings["fixture-op"]
    assert delta["comm_transport"] == "nvlink"
    assert delta["comm_addressing"] == "nvls_multicast"
    assert delta["comm_offline_binding_verified"] is True
    live = dict(
        logical_batch=7,
        query_width=3,
        profile_argument_num_global_tokens=99,
        comm_payload_bytes=123456,
    )
    expected = live.copy()
    assert set(delta).isdisjoint(live)
    live.update(delta)
    assert {key: live[key] for key in expected} == expected
    assert capture == original

    # Incomplete evidence yields a diagnostic delta, never captured scalars.
    allocations[1].unlink()
    missing = communication.bind_capture_communication(capture, tmp_path)["fixture-op"]
    assert set(missing) == {"comm_offline_binding_missing_reason"}
    live.update(missing)
    assert {key: live[key] for key in expected} == expected
    assert capture == original


@pytest.fixture
def h200_rows():
    ranks = range(8)
    switches = ["00000000:05:00.0"] * 4 + ["00000000:06:00.0"] * 5
    switches += ["00000000:07:00.0"] * 5 + ["00000000:08:00.0"] * 4
    rows = [
        dict(
            rank=rank,
            pid=100 + rank,
            host="fixture-host",
            uuid=f"GPU-fixture-{rank}",
            nvml_uuid=f"GPU-fixture-{rank}",
            physical_index=rank,
            pci_device_id=0x233510DE,
            pci_subsystem_id=0x18BE10DE,
            mig_mode=0,
            full_device=True,
            switch_device_type=1,
            links=[
                dict(
                    link=index,
                    active=True,
                    remote_device_type=1,
                    remote_pci=address,
                    version=7,
                    raw_status_bytes_per_second=26.562e9,
                )
                for index, address in enumerate(switches)
            ],
        )
        for rank in ranks
    ]
    header = " ".join(f"GPU{rank}" for rank in ranks)
    topology = [f"\x1b[4m{header} CPU Affinity\x1b[0m"]
    for rank in ranks:
        widths = " ".join("X" if peer == rank else "NV18" for peer in ranks)
        topology.append(f"GPU{rank} {widths} 0-7")
    rows[0]["topology"] = {"returncode": 0, "stdout": "\n".join(topology)}
    return rows


def h200_metrics(directory, rows, *, pattern=0, remove_allocation=False):
    hardware = HardwareProfile.from_file(
        Path(__file__).resolve().parents[2]
        / "docs/hardware-profiles/nvidia-h200-sxm-141gb.yaml"
    ).validate_device(
        {"name": "NVIDIA H200", "partition": "full_device", "uuid": rows[0]["uuid"]}
    )
    capture, _, allocations = communication_receipts(
        directory, rows=rows, pattern=pattern
    )
    if remove_allocation:
        allocations[-1].unlink()
    observation = capture["ops"][0]["scalar_args"] | (
        communication.bind_capture_communication(capture, directory)["fixture-op"]
    )
    group = f"nccl:None:{tuple(range(len(rows)))}"
    cost = OpCost(
        useful_flops=4096 if pattern == 1 else 0,
        executed_flops=None,
        logical_read_bytes=8192,
        logical_write_bytes=8192,
        estimated_hbm_bytes=16384,
        rule="fixture",
        assumptions=(),
        compute_type="fp32_simt" if pattern == 1 else None,
        communication=CollectiveTraffic("all_reduce_sum", 8192, len(rows), group),
    )
    return scope_metrics(
        [{"cost": asdict(cost)}],
        8000,
        hardware,
        scope_kind="op",
        communication_observation=observation,
    )


@pytest.mark.parametrize("pattern", [0, 1])
def test_h200_board_selects_directional_capacity(tmp_path, h200_rows, pattern):
    row = h200_metrics(tmp_path, h200_rows, pattern=pattern)
    assert row["communication_observation"]["comm_rank_link_versions"] == (7,) * 8
    assert row["communication_capacity"]["link_generation"] == "4.0"
    assert row["communication_capacity"]["bidirectional_bytes_per_second"] == 900e9
    assert row["estimated_comm_tx_capacity_bytes_per_second"] == 450e9
    assert row["estimated_comm_rx_capacity_bytes_per_second"] == 450e9
    assert row["estimated_comm_tx_payload_bytes"] == 8192
    assert row["estimated_comm_rx_payload_bytes"] == 7 * 8192
    assert row["algbw_bytes_per_second"] == pytest.approx(8192 / 8e-6)
    assert row["busbw_bytes_per_second"] == pytest.approx(14336 / 8e-6)
    assert row["estimated_comm_utilization"] == pytest.approx(7 * 8192 / 8e-6 / 450e9)


@pytest.mark.parametrize(
    "pattern, failure",
    [(0, None), (1, None), (1, "native"), (0, "unsealed")],
)
def test_h200_twoshot_uses_owned_token_payload(tmp_path, h200_rows, pattern, failure):
    from tokenspeed_kernel.profile_communication_inventory import (
        communication_inventory,
    )

    capture, _, allocations = communication_receipts(
        tmp_path, rows=h200_rows, pattern=pattern
    )
    op = capture["ops"][0]
    name, implementation, arg = (
        (
            "communication.all_reduce_residual_rmsnorm",
            "trtllm_allreduce_residual_rmsnorm",
            "input_tensor",
        )
        if pattern
        else ("communication.all_reduce", "trtllm_all_reduce", "tensor")
    )
    payload = 59 * 8192
    op.update(
        op_name=name,
        implementation=implementation,
        inputs={
            arg: dict(
                shape=[59, 4096],
                stride=[4096, 1],
                dtype="torch.bfloat16",
                logical_bytes=payload,
            )
        },
    )
    op["scalar_args"].update(
        comm_algorithm="twoshot", comm_use_oneshot=False, comm_payload_bytes=payload
    )
    native_name = "mnnvl_allreduce_fusion_kernel_twoshot"
    native = {
        "fixture-op": [
            dict(
                kind="kernel",
                table="CUPTI_ACTIVITY_KIND_KERNEL",
                source_rowid=5,
                kernel=dict(
                    short_name=native_name,
                    name=f"fixture_{native_name}",
                    demangled_name=(
                        f"void flashinfer::trtllm_mnnvl_allreduce_fusion::{native_name}"
                        "<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)"
                        f"{pattern}, __nv_bfloat16, (int)8, (bool)0, (bool)0>"
                    ),
                    grid=[58 if failure == "native" else 59, 2, 1],
                    block=[256, 1, 1],
                    cluster=[1, 2, 1],
                ),
            )
        ]
    }
    sealed = communication_inventory(
        [*tmp_path.glob("rank-*.communication.json"), *allocations], run_id="fixture"
    )
    config = tmp_path / "rank-0.config.json"
    raw = config.read_bytes()
    sealed[str(config)] = dict(
        path=str(config), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    observation = (
        op["scalar_args"]
        | communication.bind_capture_communication(
            capture,
            tmp_path,
            sealed_artifacts=None if failure == "unsealed" else sealed,
            native_op_activities=native,
        )["fixture-op"]
    )
    hardware = HardwareProfile.from_file(
        Path(__file__).resolve().parents[2]
        / "docs/hardware-profiles/nvidia-h200-sxm-141gb.yaml"
    ).validate_device(
        {
            "name": "NVIDIA H200",
            "partition": "full_device",
            "uuid": h200_rows[0]["uuid"],
        }
    )
    cost = OpCost(
        useful_flops=4096 if pattern else 0,
        executed_flops=None,
        logical_read_bytes=payload,
        logical_write_bytes=payload,
        estimated_hbm_bytes=payload * 2,
        rule="fixture",
        assumptions=(),
        compute_type="fp32_simt" if pattern else None,
        communication=CollectiveTraffic(
            "all_reduce_sum", payload, 8, f"nccl:None:{tuple(range(8))}"
        ),
    )
    row = scope_metrics(
        [{"cost": asdict(cost)}],
        8000,
        hardware,
        scope_kind="op",
        communication_observation=observation,
    )
    assert row["algbw_bytes_per_second"] == pytest.approx(payload / 8e-6)
    assert row["busbw_bytes_per_second"] == pytest.approx(payload * 1.75 / 8e-6)
    if failure:
        assert row["estimated_comm_utilization"] is None
        reason = "native kernel/launch" if failure == "native" else "sealed routes"
        assert reason in row["estimated_comm_utilization_missing_reason"]
        return
    assert row["communication_capacity"]["algorithm"] == "twoshot"
    assert row["estimated_comm_tx_payload_bytes"] == payload
    assert row["estimated_comm_rx_payload_bytes"] == (7 * 8 + 51) * 8192
    assert row["communication_payload_model"]["owned_tokens"] == 8
    assert row["estimated_comm_utilization"] == pytest.approx(
        (7 * 8 + 51) * 8192 / 8e-6 / 450e9
    )


@pytest.mark.parametrize(
    "mismatch, reason",
    [
        ("missing_pci", "board evidence unavailable"),
        ("device", "GPU PCI device ID"),
        ("subsystem", "GPU PCI subsystem ID"),
        ("mig", "full GPU device"),
        ("uuid", "board evidence unavailable"),
        ("link_count", "active link count"),
        ("link_rate", "rates do not match"),
        ("peer_width", "peer matrix does not match"),
        ("switch_count", "switch endpoints do not match"),
        ("allocation", "communication evidence unavailable: comm_group"),
    ],
)
def test_h200_board_requires_all_rank_evidence(tmp_path, h200_rows, mismatch, reason):
    peer = h200_rows[-1]
    if mismatch == "missing_pci":
        del peer["pci_device_id"]
    elif mismatch == "device":
        peer["pci_device_id"] = 0x233010DE
    elif mismatch == "subsystem":
        peer["pci_subsystem_id"] = 0x18BF10DE
    elif mismatch == "mig":
        peer.update(mig_mode=1, full_device=False)
    elif mismatch == "uuid":
        peer["nvml_uuid"] = "GPU-another-device"
    elif mismatch == "link_count":
        peer["links"][0]["active"] = False
    elif mismatch == "link_rate":
        peer["links"][0]["raw_status_bytes_per_second"] = 53.124e9
    elif mismatch == "peer_width":
        topology = h200_rows[0]["topology"]
        topology["stdout"] = topology["stdout"].replace("NV18", "NV16", 1)
    elif mismatch == "switch_count":
        for rank in h200_rows:
            for link in rank["links"]:
                if link["remote_pci"] == "00000000:08:00.0":
                    link["remote_pci"] = "00000000:07:00.0"
    row = h200_metrics(tmp_path, h200_rows, remove_allocation=mismatch == "allocation")
    assert row["estimated_comm_utilization"] is None
    assert reason in row["estimated_comm_utilization_missing_reason"]
    if mismatch == "allocation":
        assert (
            "allocation missing or ambiguous"
            in row["communication_observation"]["comm_offline_binding_missing_reason"]
        )
    assert row["busbw_bytes_per_second"] == pytest.approx(14336 / 8e-6)


def test_manifest_inventory_binds_exact_communication_inputs(tmp_path):
    import pytest
    from tokenspeed_kernel.profile_communication_inventory import (
        communication_inventory,
    )

    capture, _, allocations = communication_receipts(tmp_path)
    snapshots = sorted(tmp_path.glob("rank-*.communication.json"))
    sealed = communication_inventory([*snapshots, *allocations], run_id="fixture")
    for path in tmp_path.glob("rank-*.config.json"):
        raw = path.read_bytes()
        sealed[str(path)] = dict(
            path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
        )

    # A different allocation outside the inventory cannot participate.
    body = json.loads(allocations[0].read_text())
    body["allocation"]["local_ptr"] += 1024
    raw = json.dumps(body).encode()
    extra = tmp_path / f"rank-0.mnnvl-allocation-{hashlib.sha256(raw).hexdigest()}.json"
    extra.write_bytes(raw)
    result = communication.bind_capture_communication(
        capture, tmp_path, sealed_artifacts=sealed
    )
    assert result["fixture-op"]["comm_offline_binding_verified"]
    capture["ops"][0]["scalar_args"].update(result["fixture-op"])
    assert not communication.bind_capture_communication(
        capture, tmp_path, sealed_artifacts=sealed
    )
    snapshots[0].write_bytes(snapshots[0].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs from sealed run inventory"):
        communication.bind_capture_communication(
            capture, tmp_path, sealed_artifacts=sealed
        )
    with pytest.raises(ValueError, match="lacks its sealed communication snapshot"):
        communication_inventory([*snapshots, *allocations], run_id="fixture")
