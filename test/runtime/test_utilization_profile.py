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
from tokenspeed_kernel.profile_nvtx import build_nvtx_bridge

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_thread import ForwardThread
from tokenspeed.runtime.execution.utilization_profile import RuntimeUtilizationProfile
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeMLP


@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA native timeline")
@pytest.mark.parametrize(
    "device,eager,prefill_off,speculative,reason",
    [
        ("cpu", True, True, False, "NVIDIA CUDA"),
        ("cuda", False, True, False, "enforce-eager"),
        ("cuda", True, False, False, "disable-prefill-graph"),
        ("cuda", True, True, True, "draft models"),
    ],
)
def test_reject_unsupported_execution(
    monkeypatch, device, eager, prefill_off, speculative, reason
):
    monkeypatch.setenv("TOKENSPEED_TIMELINE_CONFIG", "unused.json")
    with pytest.raises(ValueError, match=reason):
        RuntimeUtilizationProfile.from_env(
            model=torch.nn.Module(),
            model_id="qwen",
            rank=0,
            device=device,
            enforce_eager=eager,
            disable_prefill_graph=prefill_off,
            speculative=speculative,
        )


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


def _workload(root: Path) -> None:
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
    thread = ForwardThread("cuda")

    def forward(step, x):
        with torch.inference_mode(), torch.cuda.stream(stream), profiler.iteration(
            step=step
        ):
            return model(x)

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

    try:
        # Warm both input shapes before the bounded serving window.
        for _ in range(3):
            thread.run(lambda: forward(0, prefill))
            thread.run(lambda: forward(0, decode))
        thread.run(lambda: forward(1, decode))
        assert not profiler.output.exists()
        prefill_output = thread.run(lambda: forward(2, prefill))
        assert not profiler.output.exists()
        decode_output = thread.run(lambda: forward(3, decode))
        # Output is ready immediately after the final batch, before shutdown.
        saved = profiler.output.read_bytes()
        thread.run(lambda: forward(4, decode))
        assert profiler.output.read_bytes() == saved
        stream.synchronize()
        torch.testing.assert_close(
            prefill_output, reference(prefill), rtol=0.01, atol=2e-5
        )
        torch.testing.assert_close(
            decode_output, reference(decode), rtol=0.01, atol=2e-5
        )
        report = json.loads(saved)
        assert report["device_analysis"]["thread_id"] != threading.get_native_id()
        assert {op["iter_id"] for op in report["ops"]} == {2, 3}
        assert len(report["ops"]) == 8 and len(report["layers"]) == 4
    finally:
        thread.shutdown()


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
        "max_scopes": 12,
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
    report = json.loads((root / "rank-0.metrics.json").read_text())
    rows = report["device_analysis"]["scopes"]
    assert len(rows) == 12
    assert {op["implementation"] for op in report["ops"]} == {"torch_mm"}
    assert all(row["mfu"] > 0 and row["estimated_mbu"] > 0 for row in rows)
    for row in rows:
        assert row["mfu"] == pytest.approx(
            row["useful_flops"] / (row["gpu_duration_ns"] * 1e3)
        )
        if row["kind"] == "layer":
            tokens = 32 if row["iter_id"] == 2 else 1
            assert row["useful_flops"] == 6 * tokens * 4096 * 1024
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
    labels = {row["nvtx_label"] for row in rows}
    with sqlite3.connect(f"file:{root / 'timeline.sqlite'}?mode=ro", uri=True) as db:
        events = [
            event
            for event in db.execute("SELECT eventType,globalTid,text FROM NVTX_EVENTS")
            if event[2] in labels
        ]
    assert len(events) == 12
    assert {event[0] for event in events} == {59}
    assert {event[1] % (1 << 24) for event in events} == {
        report["device_analysis"]["thread_id"]
    }
    for row in rows:
        native = projected[row["nvtx_label"]].pop(0)
        assert int(native["TID"]) == report["device_analysis"]["thread_id"]
        assert int(native["NumGPUOps"]) >= (1 if row["kind"] == "op" else 3)
        if row["kind"] == "op":
            assert "Est.MBU=" in native["Name"] and "MFU=" in native["Name"]
        else:
            assert row["nvtx_label"] == f"L{row['layer_id']}"


if __name__ == "__main__":
    _workload(Path(sys.argv[1]))
