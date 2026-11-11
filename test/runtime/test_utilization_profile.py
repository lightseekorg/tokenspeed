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
from collections import Counter
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.op_profile import profiled
from tokenspeed_kernel.profile_nsys import finalize_graph_nsight, finalize_nsight

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


def test_disabled_ranges_work_without_nvtx():
    # Simulate NVIDIA dispatch and a missing optional package in a fresh process.
    # Checking only the CPU platform would miss an eager NVIDIA-only import.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from unittest.mock import patch
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo
platform = PlatformInfo(
    vendor="nvidia",
    arch_version=ArchVersion(9, 0),
    device_name="fixture",
    device_count=1,
    total_memory=0,
    memory_bandwidth=0,
    sm_count=0,
    max_threads_per_sm=0,
    max_shared_memory_per_sm=0,
)
with patch("tokenspeed_kernel.platform.current_platform", return_value=platform):
    sys.modules["nvtx"] = None
    from tokenspeed.runtime.utils.nvtx import nvtx_range, set_nvtx_enabled
    set_nvtx_enabled(False)
    with nvtx_range("disabled"):
        pass
    @nvtx_range("disabled")
    def run():
        return 42
    assert run() == 42
""",
        ],
        env=dict(os.environ, TOKENSPEED_NVTX="0"),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA native timeline")
@pytest.mark.parametrize("key", ["library", "nvtx_library"])
def test_eager_config_rejects_legacy_library(tmp_path, monkeypatch, key):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({key: "obsolete.so"}))
    monkeypatch.setenv("TOKENSPEED_TIMELINE_CONFIG", str(path))
    with pytest.raises(ValueError, match="Remove the legacy library or nvtx_library"):
        RuntimeUtilizationProfile.from_env(
            model=torch.nn.Module(),
            model_id="fixture",
            rank=0,
            device="cuda",
            enforce_eager=True,
            disable_prefill_graph=True,
            speculative=False,
        )


def _workload(root: Path, mode: str) -> None:
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
        enforce_eager=mode == "eager",
        disable_prefill_graph=True,
        speculative=False,
    )
    assert profiler is not None
    fallback = mode != "eager"
    mixed = mode == "mixed"
    assert profiler.graph_enabled == fallback
    graph = None
    graph_output = None
    graph_key = ("proof", 1)
    assert profiler.output == root.resolve() / "rank-0.metrics.json"

    def forward(step, x):
        identity = profiler.begin_step(
            step=step, request_ids=("shared-expert-test",), pool_indices=(0,)
        )
        with torch.inference_mode(), torch.cuda.stream(stream), profiler.iteration(
            step=step, identity=identity
        ):
            if graph is not None and step == 3:
                with profiler.replay(
                    graph_key=graph_key,
                    batch_size=1,
                    padded_batch_size=1,
                    query_width=1,
                    query_shard=None,
                ):
                    graph.replay()
                result = graph_output
            else:
                with profiler.eager(
                    batch_size=1,
                    input_tokens=len(x),
                    decode_query_width=None,
                    query_sharded=False,
                ):
                    result = model(x)
            if identity is not None:
                event = torch.cuda.Event()
                event.record()
                identity.bind_result_event(event)
                if fallback:
                    assert identity.unavailable_reason is None
                    identity.bind_forward_id(step * 10)
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
    if mixed:
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.stream(stream), profiler.capture(
            graph_key=graph_key, batch_size=1, query_width=1
        ), torch.cuda.graph(graph, stream=stream):
            graph_output = model(decode)
    profiler.arm()
    forward(1, decode)
    assert not profiler.output.exists()
    prefill_output = forward(2, prefill)
    assert not profiler.output.exists()
    decode_output = forward(3, decode)
    # Output is ready immediately after the final batch, before shutdown.
    outputs = (
        [
            root / f"rank-0.step-{step}.eager.json"
            for step in ((2,) if mixed else (2, 3))
        ]
        if fallback
        else [profiler.output]
    )
    if fallback:
        forward(4, decode)
        window = profiler.stop()
        assert window["persisted_eager"] == (1 if mixed else 2)
        assert window["persisted_payloads"] == int(mixed)
        assert window["dropped_metadata"] == 0
        saved = [path.read_bytes() for path in outputs]
        from tokenspeed_kernel.profile_eager import (
            merge_forward_binding,
            read_eager_sidecar,
        )

        rows = [
            json.loads(line)
            for line in (root / "rank-0.replays.jsonl").read_text().splitlines()
        ]
        ready = {
            row["step_id"]: row
            for row in rows
            if row["event"] == "device_results_ready"
        }
        for row in rows:
            if row["event"] == "forward_identity_bound":
                merge_forward_binding(ready[row["step_id"]], row)
        assert sorted(row["forward_id"] for row in ready.values()) == [20, 30]
        for row in ready.values():
            if row.get("eager_sidecar") is not None:
                report = read_eager_sidecar(
                    row, config_sha256=window["profile_config_sha256"]
                )
                assert report["eager_step"]["serving_step"] == row["serving_step"]
    else:
        saved = [path.read_bytes() for path in outputs]
        forward(4, decode)
        assert [path.read_bytes() for path in outputs] == saved
        profiler.stop()
    assert not (root / "rank-0.step-4.eager.json").exists()
    stream.synchronize()
    torch.testing.assert_close(prefill_output, reference(prefill), rtol=0.01, atol=2e-5)
    torch.testing.assert_close(decode_output, reference(decode), rtol=0.01, atol=2e-5)
    reports = [json.loads(content) for content in saved]
    for report in reports:
        assert report["device_analysis"]["thread_id"] == threading.get_native_id()
        assert report["device_analysis"]["nvtx_payload_type"] == "int64"
        assert {layer["layer_id"] for layer in report["layers"]} >= {0, 1}
    ops = [op for report in reports for op in report["ops"]]
    expected_steps = (2,) if mixed else (2, 3)
    assert {op["iter_id"] for op in ops} == set(expected_steps)
    for step in expected_steps:
        assert Counter(op["implementation"] for op in ops if op["iter_id"] == step) == {
            "torch_mm": 4,
            "silu_and_mul": 2,
        }


@pytest.mark.parametrize("mode", ["eager", "graph-fallback", "mixed"])
def test_qwen_shared_expert_native_timeline(tmp_path, mode):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("Requires NVIDIA CUDA and TOKENSPEED_TEST_NSYS (2026.5+)")
    root = tmp_path
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
        "layers": "layers",
        "start_step": 2,
        "num_steps": 2,
        "max_scopes": 256,
        "hbm_policy": "logical_io_once",
    }
    if mode == "mixed":
        cuda_runtime = os.environ.get("TOKENSPEED_TEST_CUDA_RUNTIME")
        cupti = os.environ.get("TOKENSPEED_TEST_CUPTI")
        if not cuda_runtime or not cupti:
            pytest.skip(
                "Mixed native proof needs exact CUDA runtime and profiler CUPTI paths"
            )
        config.update(cuda_runtime_library=cuda_runtime, cupti_library=cupti)
    elif mode == "graph-fallback":
        config.update(cuda_runtime_library="libcudart.so", cupti_library="libcupti.so")
    (root / "capture.json").write_text(json.dumps(config))
    repo = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(repo / "python"), str(repo / "tokenspeed-kernel/python")]
        ),
        TOKENSPEED_TIMELINE_CONFIG=str(root / "capture.json"),
    )
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        *(["--cuda-graph-trace=node"] if mode == "mixed" else []),
        "--sample=none",
        "--cpuctxsw=none",
        f"--output={root / 'timeline'}",
        sys.executable,
        str(Path(__file__).resolve()),
        str(root),
        mode,
    ]
    try:
        subprocess.run(
            command, env=env, check=True, capture_output=True, text=True, timeout=240
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(
            f"Nsight workload exited {error.returncode}\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        )
    if mode == "mixed":
        report = finalize_graph_nsight(
            nsys=nsys,
            trace=root / "timeline.nsys-rep",
            replays=root / "rank-0.replays.jsonl",
            window=root / "rank-0.window.json",
            output=root / "mixed.metrics.json",
            hbm_policy="logical_io_once",
            hardware=None,
            service_metadata=None,
            model_config=None,
            coverage_output=None,
            scheduler_mode="disabled",
            registry=None,
        )
        assert report["collector_window"]["dropped_steps"] == []
        eager, graph = report["steps"]
        assert eager["attribution_mode"] == "eager_nvtx_gpu_projection"
        assert graph["attribution_mode"] == "cuda_graph_node_ids"
        assert eager["forward_id"] == 20 and graph["forward_id"] == 30
        for step in (eager, graph):
            assert step["scope_kinds"] == ["layer", "model", "op"]
            assert step["attributed_layer_ids"] == [0, 1]
        return
    metadata_files = (
        [root / f"rank-0.step-{step}.eager.json" for step in (2, 3)]
        if mode == "graph-fallback"
        else [root / "rank-0.metrics.json"]
    )
    reports = [
        finalize_nsight(
            nsys=nsys,
            trace=root / "timeline.nsys-rep",
            metadata=metadata,
            output=metadata.with_suffix(".nsight.json"),
        )
        for metadata in metadata_files
    ]
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
        projected = {int(row["RangeId"]): row for row in csv.DictReader(source)}
    with sqlite3.connect(f"file:{root / 'timeline.sqlite'}?mode=ro", uri=True) as db:
        events = {event[0]: event[1:] for event in db.execute("""
                SELECT n.rowid, n.eventType, n.globalTid, COALESCE(s.value, n.text),
                       n.int64Value
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
                WHERE n.int64Value IS NOT NULL
                """)}
    for report, metadata in zip(reports, metadata_files, strict=True):
        rows = report["device_analysis"]["scopes"]
        assert len(rows) == len(report["ops"]) + len(report["layers"])
        assert {op["implementation"] for op in report["ops"]} == {
            "torch_mm",
            "silu_and_mul",
        }
        # Generic torch matmul has useful algebra but no instruction-engine proof.
        mm_scopes = {
            op["trace_label"]
            for op in report["ops"]
            if op["implementation"] == "torch_mm"
        }
        mm_rows = [row for row in rows if row["trace_label"] in mm_scopes]
        assert len(mm_rows) == len(mm_scopes)
        assert all(row["mfu"] is None for row in mm_rows)
        assert any(
            row["estimated_mbu"] is not None and row["estimated_mbu"] > 0
            for row in rows
        )
        sidecar = json.loads(metadata.read_text())["device_analysis"]
        scopes = {item["trace_label"]: item for item in sidecar["scopes"]}
        for row in rows:
            range_id = row["nsight_range_id"]
            event_type, global_tid, label, payload = events[range_id]
            assert event_type == 59
            assert global_tid % (1 << 24) == sidecar["thread_id"]
            assert label == row["native_nvtx_label"]
            assert payload == scopes[row["trace_label"]]["nvtx_id"]
            native = projected[range_id]
            assert native["Name"] == f"TokenSpeed:{label}"
            assert row["gpu_duration_ns"] == int(native["Projected Duration (ns)"])
            assert int(native["TID"]) == sidecar["thread_id"]
            assert int(native["NumGPUOps"]) >= 1
            if row["kind"] == "op":
                assert "Est.MBU=" in row["nvtx_label"] and "MFU=" in row["nvtx_label"]
            else:
                assert row["nvtx_label"] == f"L{row['layer_id']}"


if __name__ == "__main__":
    _workload(Path(sys.argv[1]), sys.argv[2])
