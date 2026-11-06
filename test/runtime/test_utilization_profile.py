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

"""Native timeline integration with the real Qwen shared-expert implementation."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.op_profile import profiled
from tokenspeed_kernel.profile_nsys import finalize_nsight
from tokenspeed_kernel.profile_nvtx import build_nvtx_bridge

from tokenspeed.runtime.execution.utilization_profile import RuntimeUtilizationProfile


@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA native timeline")
def test_reject_non_cuda_execution(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_TIMELINE_CONFIG", "unused.json")
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        RuntimeUtilizationProfile.from_env(
            model=torch.nn.Module(),
            model_id="qwen",
            rank=0,
            device="cpu",
            enforce_eager=True,
            disable_prefill_graph=True,
            speculative=False,
        )


@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA native timeline")
def test_standalone_from_env_without_process_group(tmp_path, monkeypatch):
    if torch.distributed.is_initialized():
        pytest.skip("Requires a standalone process without an initialized group")
    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps(
            {
                "name": "CPU construction fixture",
                "compute_flops_per_second": {},
                "hbm_bytes_per_second": None,
                "source": "synthetic setup-only fixture",
            }
        )
    )
    config = dict(
        run_id="standalone",
        output_dir=str(tmp_path),
        hardware=str(hardware),
        layers="layers",
        start_step=1,
        num_steps=1,
        max_scopes=16,
        hbm_policy="unavailable",
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("TOKENSPEED_TIMELINE_CONFIG", str(path))
    monkeypatch.delenv("TOKENSPEED_PROFILE_RUN_ID", raising=False)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    # CUDA is the configured execution device; constructing its CPU-side
    # observer must not allocate CUDA resources or initialize a process group.
    initialized = torch.cuda.is_initialized()
    profile = RuntimeUtilizationProfile.from_env(
        model=model,
        model_id="fixture",
        rank=0,
        device="cuda",
        enforce_eager=True,
        disable_prefill_graph=True,
        speculative=False,
    )
    assert isinstance(profile, RuntimeUtilizationProfile)
    assert profile.model is model and profile.rank == 0
    assert profile.graph_enabled is False
    assert profile.output == tmp_path / "rank-0.metrics.json"
    snapshot = json.loads((tmp_path / "rank-0.config.json").read_text())
    assert snapshot["run_id"] == "standalone" and snapshot["rank"] == 0
    assert snapshot["graph_enabled"] is False
    assert not (tmp_path / "rank-0.communication.json").exists()
    assert not torch.distributed.is_initialized()
    assert torch.cuda.is_initialized() == initialized


def _workload(root: Path) -> None:
    # Keep GPU/model dependencies local to the native workload so the observer
    # construction regression above remains a CPU-only test.
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP

    class ProfiledSharedExpert(Qwen3_5MoeMLP):
        @profiled(level="layer")
        def forward(self, x):
            return super().forward(x)

    class SharedExpertStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Qwen3.5-397B-A17B's actual shared-expert dimensions. Random weights
            # test the runtime/kernel path; this is not a checkpoint-quality test.
            self.layers = torch.nn.ModuleList(
                [
                    ProfiledSharedExpert(
                        hidden_size=4096,
                        intermediate_size=1024,
                        hidden_act="silu",
                        mapping=Mapping(rank=0, world_size=1),
                        quant_config=None,
                        reduce_results=False,
                        parallelism="moe_shared",
                    )
                    for _ in range(2)
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(7)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = SharedExpertStack()
        for weight in model.parameters():
            torch.nn.init.normal_(weight, std=0.01)
        prefill = torch.randn(32, 4096)
        decode = prefill[:1].clone()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    profiler = RuntimeUtilizationProfile.from_env(
        model=model,
        model_id="Qwen3.5-397B-A17B/shared-expert/random-weights",
        rank=0,
        device="cuda",
        enforce_eager=True,
        disable_prefill_graph=True,
        speculative=False,
    )
    assert profiler is not None
    assert profiler.output == root.resolve() / "rank-0.metrics.json"

    def forward(step, x):
        identity = profiler.begin_step(
            step=step, request_ids=("shared-expert-test",), pool_indices=(0,)
        )
        with torch.inference_mode(), torch.cuda.stream(stream), profiler.iteration(
            step=step, identity=identity
        ):
            result = model(x)
            if identity is not None:
                event = torch.cuda.Event()
                event.record()
                identity.bind_result_event(event)
        if identity is not None:
            event.synchronize()
            identity.device_results_ready(event)
        return result

    def reference(x):
        with torch.inference_mode():
            for layer in model.layers:
                gate, up = torch.nn.functional.linear(
                    x, layer.gate_up_proj.weight
                ).chunk(2, -1)
                activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                    x.dtype
                )
                x = torch.nn.functional.linear(activated, layer.down_proj.weight)
        return x

    # Warm both input shapes before the bounded serving window.
    for _ in range(3):
        forward(0, prefill)
        forward(0, decode)
    profiler.arm()
    forward(1, decode)
    assert not profiler.output.exists()
    prefill_output = forward(2, prefill)
    assert not profiler.output.exists()
    decode_output = forward(3, decode)
    # Output is ready immediately after the final batch, before shutdown.
    saved = profiler.output.read_bytes()
    forward(4, decode)
    assert profiler.output.read_bytes() == saved
    stream.synchronize()
    torch.testing.assert_close(prefill_output, reference(prefill), rtol=0.01, atol=2e-5)
    torch.testing.assert_close(decode_output, reference(decode), rtol=0.01, atol=2e-5)
    report = json.loads(saved)
    assert report["device_analysis"]["thread_id"] == threading.get_native_id()
    assert {op["iter_id"] for op in report["ops"]} == {2, 3}
    assert len(report["ops"]) == 8
    assert {layer["layer_id"] for layer in report["layers"]} >= {0, 1}
    profiler.stop()


def test_qwen_shared_expert_native_timeline(tmp_path):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("Requires NVIDIA CUDA and TOKENSPEED_TEST_NSYS (2026.5+)")
    root = tmp_path
    library = build_nvtx_bridge(
        include_dir=Path(nsys).resolve().parent / "nvtx/include",
        output=root / "nvtx_profile.so",
        compiler="cc",
    )
    (root / "hardware.json").write_text(
        json.dumps(
            {
                "name": "test normalization",
                "compute_flops_per_second": {"bf16_tensor": 1e12},
                "hbm_bytes_per_second": 1e12,
                "source": "synthetic test peaks, not device specifications",
            }
        )
    )
    config = {
        "run_id": "qwen-shared-expert",
        "output_dir": str(root),
        "hardware": str(root / "hardware.json"),
        "library": str(library),
        "layers": "layers",
        "start_step": 2,
        "num_steps": 2,
        "max_scopes": 256,
        "hbm_policy": "logical_io_once",
    }
    (root / "capture.json").write_text(json.dumps(config))
    repo = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(repo / "python"), str(repo / "tokenspeed-kernel/python")]
        ),
        TOKENSPEED_TIMELINE_CONFIG=str(root / "capture.json"),
    )
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={root / 'timeline'}",
            sys.executable,
            str(Path(__file__).resolve()),
            str(root),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    report = finalize_nsight(
        nsys=nsys,
        trace=root / "timeline.nsys-rep",
        metadata=root / "rank-0.metrics.json",
        output=root / "rank-0.nsight.json",
    )
    rows = report["device_analysis"]["scopes"]
    assert len(rows) == len(report["ops"]) + len(report["layers"])
    assert {op["implementation"] for op in report["ops"]} == {"torch_mm"}
    # Generic torch matmul has useful algebra but no instruction-engine proof.
    assert all(row["mfu"] is None for row in rows)
    assert any(
        row["estimated_mbu"] is not None and row["estimated_mbu"] > 0 for row in rows
    )
    subprocess.run(
        [
            nsys,
            "stats",
            "--report=nvtx_gpu_proj_trace",
            "--format=csv",
            f"--output={root / 'projection'}",
            str(root / "timeline.nsys-rep"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with (root / "projection_nvtx_gpu_proj_trace.csv").open() as source:
        projected = defaultdict(list)
        for row in csv.DictReader(source):
            projected[row["Name"].split(":", 1)[-1]].append(row)
    labels = {row["native_nvtx_label"] for row in rows}
    with sqlite3.connect(f"file:{root / 'timeline.sqlite'}?mode=ro", uri=True) as db:
        events = [
            event
            for event in db.execute("SELECT eventType,globalTid,text FROM NVTX_EVENTS")
            if event[2] in labels
        ]
    assert len(events) == len(rows)
    assert {event[0] for event in events} == {59}
    assert {event[1] % (1 << 24) for event in events} == {
        report["device_analysis"]["thread_id"]
    }
    for row in rows:
        native = projected[row["native_nvtx_label"]].pop(0)
        assert row["gpu_duration_ns"] == int(native["Projected Duration (ns)"])
        assert int(native["TID"]) == report["device_analysis"]["thread_id"]
        assert int(native["NumGPUOps"]) >= 1
        if row["kind"] == "op":
            assert "Est.MBU=" in row["nvtx_label"] and "MFU=" in row["nvtx_label"]
        else:
            assert row["nvtx_label"] == f"L{row['layer_id']}"


if __name__ == "__main__":
    _workload(Path(sys.argv[1]))
