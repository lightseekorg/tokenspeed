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
from pathlib import Path

from tokenspeed_kernel import profile_communication as communication


def communication_receipts(directory):
    """Two CPU-authored rank receipts with real installed source identities."""

    def write(name, value):
        contents = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(contents).hexdigest()
        path = directory / name.format(sha=digest)
        path.write_bytes(contents)
        return path, digest

    config = {"world_size": 2}
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_root = Path(communication.__file__).parent
    source_paths = ("ops/communication/trtllm.py", *communication._MNNVL_ROUTE_SOURCES)
    sources = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in source_paths
    }
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
    allocation_paths = []
    snapshot_sha = None
    for rank in range(2):
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
            world_size=2,
            peer_count=2,
            rank=rank,
            ranks=[0, 1],
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
                    ranks=[0, 1],
                    group_name=None,
                    comm_backend="trtllm",
                    comm_workspace="mnnvl",
                    comm_protocol="mnnvl",
                    comm_algorithm="oneshot",
                    comm_pattern=0,
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
