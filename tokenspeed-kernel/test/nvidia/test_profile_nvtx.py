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

import csv
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.op_profile import CostRegistry, OpProfile, ProfileSession
from tokenspeed_kernel.profile_costs import (
    AllReduceRMSNormCost,
    AttentionMatmulCost,
    DenseMatmulCost,
    register_collective_costs,
    register_compute_costs,
)
from tokenspeed_kernel.profile_nsys import finalize_nsight
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec


@pytest.mark.parametrize(
    "route,partial",
    [
        ("ipc", False),
        ("ipc", True),
        ("mnnvl", False),
        ("unfused", False),
        ("unfused", True),
    ],
)
def test_allreduce_rmsnorm_interception_without_gpu(monkeypatch, route, partial):
    from tokenspeed_kernel.ops.communication import trtllm as comm

    # Only process-group/workspace/launch facilities are substituted. Tensor
    # arithmetic, dispatch, recording and cost analysis all execute on the CPU.
    class Group:
        group_name = "profile-test"

        def size(self):
            return 2

    group = Group()
    mnnvl = SimpleNamespace(resolve_use_oneshot=lambda *args: False)
    workspace = None if route == "unfused" else mnnvl if route == "mnnvl" else object()
    manager = SimpleNamespace(mnnvl_workspace=mnnvl if route == "mnnvl" else None)
    monkeypatch.setattr(comm, "ensure_workspace_initialized", lambda **kwargs: True)
    monkeypatch.setattr(comm, "_manager_for_group", lambda group: manager)
    monkeypatch.setattr(comm, "_ar_fusion_workspace", lambda *args, **kwargs: workspace)
    monkeypatch.setattr(comm.dist, "get_process_group_ranks", lambda group: [3, 7])
    monkeypatch.setattr(comm.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        comm.dist, "all_reduce", lambda tensor, **kwargs: tensor.mul_(2)
    )

    def cpu_launch(**kwargs):
        summed = kwargs["allreduce_in"] * kwargs["world_size"] + kwargs["residual_in"]
        output = summed * torch.rsqrt(
            summed.square().mean(-1, keepdim=True) + kwargs["rms_eps"]
        )
        output *= kwargs["rms_gamma"]
        kwargs["residual_out"].copy_(summed)
        if kwargs["partial_norm_out"] is not None:
            kwargs["partial_norm_out"].copy_(output[3:])
        else:
            kwargs["norm_out"].copy_(output)

    monkeypatch.setattr(comm, "trtllm_allreduce_fusion", cpu_launch)
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    residual, weight = torch.full_like(x, 0.125), torch.arange(1, 5, dtype=x.dtype)
    session = ProfileSession(run_id="fusion", model_id="test", rank=7, backend="none")

    def run():
        return comm.allreduce_residual_rmsnorm(
            x,
            residual,
            weight,
            1,
            group,
            eps=1e-5,
            use_oneshot=True,
            has_partial_norm_out=partial,
        )

    with session.iteration(iter_id=0):
        output, residual_out, scale_out, partial_out = run()
    summed = 2 * x + residual
    expected = torch.nn.functional.rms_norm(summed, (4,), weight, eps=1e-5)
    if not partial or route == "unfused":
        torch.testing.assert_close(output, expected)
    torch.testing.assert_close(residual_out, summed)
    assert scale_out is None
    if partial:
        torch.testing.assert_close(partial_out, expected[3:])
    else:
        assert partial_out is None
    assert len(session.ops) == 1
    record = session.ops[0]
    fused = route != "unfused"
    assert record.implementation == "trtllm_allreduce_residual_rmsnorm"
    assert record.scalar_args["ranks"] == (3, 7)
    assert record.scalar_args["group_name"] == group.group_name
    assert record.scalar_args["fused"] is fused
    costs = CostRegistry()
    register_collective_costs(costs, hbm_policy="logical_io_once")
    cost = session.report(costs=costs)["analysis"]["ops"][0]["cost"]
    assert cost["useful_flops"] == 100  # Full epilogue, including two-shot.
    assert cost["logical_read_bytes"] == 176
    assert cost["logical_write_bytes"] == (112 if fused and partial else 160)
    assert cost["estimated_hbm_bytes"] == ((288 if partial else 336) if fused else None)
    assert cost["communication"]["total_bytes"] == 80
    assert cost["communication"]["world_size"] == 2
    assert cost["compute_type"] is None  # CPU execution has no GPU peak.
    rule = AllReduceRMSNormCost(hbm_policy="logical_io_once")
    gpu_metadata = replace(
        record,
        inputs={k: replace(v, device="cuda:0") for k, v in record.inputs.items()},
        outputs={k: replace(v, device="cuda:0") for k, v in record.outputs.items()},
    )
    assert rule(gpu_metadata).compute_type == "fp32_simt"
    for flag in ("block_quant_fp8", "residual_reduce_scattered"):
        with pytest.raises(ValueError, match="not modeled"):
            rule(replace(record, scalar_args={**record.scalar_args, flag: True}))
    if fused:

        def fail_launch(**kwargs):
            raise RuntimeError("launch failed")

        monkeypatch.setattr(comm, "trtllm_allreduce_fusion", fail_launch)
        with session.iteration(iter_id=1), pytest.raises(
            RuntimeError, match="launch failed"
        ):
            run()
        assert session.ops[-1].error == "RuntimeError"
        assert session.report(costs=costs)["analysis"]["ops"][-1]["cost"] is None


def test_compute_type_uses_gpu_operands_not_output_dtype():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    from tokenspeed_kernel.numerics.torch.gemm import torch_bmm, torch_mm

    def dense(a, b):
        return (a @ b).double()

    def attention(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, enable_gqa=True
        ).double()

    registry = KernelRegistry()
    for name, fn in (("dense", dense), ("attention", attention)):
        registry.register(
            KernelSpec(name=name, family=name, mode="test"),
            fn,
            profile=OpProfile(cost_rule=name),
        )
    costs = CostRegistry()
    register_compute_costs(costs, hbm_policy="unavailable")
    costs.register("dense", DenseMatmulCost(lhs="a", rhs="b", output="return"))
    costs.register(
        "attention",
        AttentionMatmulCost(q="q", k="k", v="v", output="return", mask="full"),
    )
    cases = (
        (torch.bfloat16, False, "bf16_tensor"),
        (torch.float16, False, "fp16_tensor"),
        (torch.float32, False, None),
        (torch.bfloat16, True, None),
    )
    for dtype, autocast_enabled, expected in cases:
        a = torch.randn(16, 32, device="cuda", dtype=dtype)
        b = torch.randn(24, 32, device="cuda", dtype=dtype)
        q = torch.randn(1, 4, 2, 8, device="cuda", dtype=dtype)
        k = torch.randn(1, 2, 4, 8, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        session = ProfileSession(
            run_id="inference", model_id="test", rank=0, backend="none"
        )
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
            with session.iteration(iter_id=0):
                registry.get_impl("dense")(a, b.T)
                torch_mm(a, b, None, None, torch.float64)
                torch_bmm(a.unsqueeze(0), b.unsqueeze(0), None, None, torch.float64)
                registry.get_impl("attention")(q, k, v)
        report = session.report(costs=costs)
        assert len(session.ops) == 4
        assert all(op.outputs["return"].dtype == "torch.float64" for op in session.ops)
        assert all(op.autocast_enabled == autocast_enabled for op in session.ops)
        assert all(
            row["cost"]["compute_type"] == expected for row in report["analysis"]["ops"]
        )


def test_native_ranges_preserve_thread_nesting_and_gpu_projection(tmp_path):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and TOKENSPEED_TEST_NSYS pointing to Nsight 2026.5+")
    kernel_root = Path(__file__).resolve().parents[2]
    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps(
            {
                "name": "analytical test peaks",
                "compute_flops_per_second": {"bf16_tensor": 1e12},
                "hbm_bytes_per_second": 1e12,
                "source": "synthetic capacities for export consistency test",
            }
        )
    )
    output = tmp_path / "capture"
    output.mkdir()
    script = tmp_path / "workload.py"
    script.write_text(textwrap.dedent("""\
        import sys
        from itertools import count
        from pathlib import Path
        import torch
        from tokenspeed_kernel import profile_nvtx
        from tokenspeed_kernel.numerics.torch.gemm import torch_mm
        from tokenspeed_kernel.op_profile import CostRegistry, ProfileSession, profiled
        from tokenspeed_kernel.profile_costs import register_compute_costs
        from tokenspeed_kernel.profile_nvtx import NvtxTimeline
        from tokenspeed_kernel.profile_utilization import HardwareProfile

        class TwoMatmulLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("up", torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
                self.register_buffer("down", torch.randn_like(self.up))

            @profiled(level="layer")
            def forward(self, x):
                hidden = torch_mm(x, self.up, None, None, torch.bfloat16)
                return torch_mm(hidden, self.down, None, None, torch.bfloat16)

        root = Path(sys.argv[1])
        output = root / "capture"
        profile_nvtx._scope_ids = count((1 << 53) + 1)
        layer = TwoMatmulLayer()
        x = torch.randn_like(layer.up)
        reference = torch.mm(torch.mm(x, layer.up.T), layer.down.T)
        for _ in range(5):
            layer(x)
        torch.cuda.synchronize()
        session = ProfileSession(run_id="native-test", model_id="two-matmul", rank=0, backend="nvtx")
        session.bind_layer(layer, layer_id=7)
        costs = CostRegistry()
        register_compute_costs(costs, hbm_policy="logical_io_once")
        with NvtxTimeline(
            session, costs=costs,
            hardware=HardwareProfile.from_json(root / "hardware.json"),
            stream=torch.cuda.current_stream(), max_scopes=6,
            output=output / "metrics.json",
        ):
            for iteration in range(2):
                with session.iteration(iter_id=iteration):
                    result = layer(x)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    """))
    env = dict(os.environ, PYTHONPATH=str(kernel_root / "python"))
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        f"--output={output / 'timeline'}",
        sys.executable,
        str(script),
        str(tmp_path),
    ]
    subprocess.run(
        command, env=env, check=True, capture_output=True, text=True, timeout=180
    )
    report = finalize_nsight(
        nsys=nsys,
        trace=output / "timeline.nsys-rep",
        metadata=output / "metrics.json",
        output=output / "nsight.metrics.json",
    )
    assert {op["implementation"] for op in report["ops"]} == {"torch_mm"}
    assert {op["cost_rule"] for op in report["ops"]} == {"torch_mm.v1"}
    rows = report["device_analysis"]["scopes"]
    assert len(rows) == 6
    for row in rows:
        assert row["useful_flops"] > 0
        assert row["gpu_duration_ns"] > 0
        assert row["estimated_mbu"] > 0
        assert row["mfu"] is None
        assert "positive FLOPs have no proven compute type" in row["mfu_missing_reason"]
    projection = output / "projection"
    subprocess.run(
        [
            nsys,
            "stats",
            "--report=nvtx_gpu_proj_trace",
            "--format=csv",
            f"--output={projection}",
            str(output / "timeline.nsys-rep"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with (output / "projection_nvtx_gpu_proj_trace.csv").open() as source:
        projected = {int(row["RangeId"]): row for row in csv.DictReader(source)}
    metadata = json.loads((output / "metrics.json").read_text())
    assert metadata["device_analysis"]["nvtx_payload_type"] == "int64"
    expected = metadata["device_analysis"]["scopes"]
    assert [item["nvtx_id"] for item in expected] == list(
        range((1 << 53) + 1, (1 << 53) + 7)
    )
    with sqlite3.connect(f"file:{output / 'timeline.sqlite'}?mode=ro", uri=True) as db:
        events = {
            event[0]: event[1:]
            for event in db.execute(
                """
                SELECT n.rowid, n.eventType, n.globalTid, n.endGlobalTid,
                       COALESCE(s.value, n.text), n.int64Value, n.uint64Value
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
                WHERE n.int64Value BETWEEN ? AND ?
                """,
                (expected[0]["nvtx_id"], expected[-1]["nvtx_id"]),
            )
        }
    assert len(events) == 6  # finish must not duplicate ranges.
    expected = {item["trace_label"]: item for item in expected}
    for row in rows:
        range_id = row["nsight_range_id"]
        event_type, tid, end_tid, label, payload, unsigned_payload = events[range_id]
        assert event_type == 59  # Actual thread NVTX push/pop range.
        assert tid % (1 << 24) == report["device_analysis"]["thread_id"]
        assert end_tid in (None, tid)
        assert (
            label
            == row["native_nvtx_label"]
            == expected[row["trace_label"]]["nvtx_label"]
        )
        assert payload == expected[row["trace_label"]]["nvtx_id"]
        assert unsigned_payload is None
        projection_row = projected[range_id]
        assert projection_row["Name"] == f"TokenSpeed:{label}"
        assert int(projection_row["NumGPUOps"]) == (1 if row["kind"] == "op" else 2)
        assert int(projection_row["Lvl"]) == (1 if row["kind"] == "op" else 0)
        assert int(projection_row["TID"]) == report["device_analysis"]["thread_id"]
        if row["kind"] == "op":
            assert row["gpu_duration_ns"] == int(
                projection_row["Projected Duration (ns)"]
            )
        else:
            assert row["nvtx_label"] == f"L{row['layer_id']}"


def test_finalize_once_and_recover_after_workload_exception(tmp_path):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and TOKENSPEED_TEST_NSYS pointing to Nsight 2026.5+")
    script = tmp_path / "workload.py"
    script.write_text(textwrap.dedent("""\
        import json
        import sys
        from dataclasses import replace
        from pathlib import Path
        import torch
        from tokenspeed_kernel.op_profile import (
            ProfileSession, CostRegistry, OpProfile, profiled,
        )
        from tokenspeed_kernel.profile_costs import DenseMatmulCost
        from tokenspeed_kernel.profile_nvtx import NvtxTimeline
        from tokenspeed_kernel.profile_utilization import HardwareProfile
        from tokenspeed_kernel.registry import KernelRegistry, KernelSpec

        root = Path(sys.argv[1])
        def work(a, b, lengths=None):
            if b is None:
                raise ValueError("intentional workload failure")
            return a @ b
        registry = KernelRegistry()
        registry.register(
            KernelSpec(name="test", family="gemm", mode="mm"),
            work,
            profile=OpProfile(cost_rule="mm", integer_tensor_args=("lengths",)),
        )
        mm = registry.get_impl("test")
        class FailingLayer:
            @profiled(level="layer")
            def forward(self, a, lengths):
                return mm(a, None, lengths)
        failing_layer = FailingLayer()
        dense = DenseMatmulCost(lhs="a", rhs="b", output="return")
        costs = CostRegistry()
        costs.register("mm", lambda op: replace(dense(op), compute_type="test",
                                                estimated_hbm_bytes=100))
        hardware = HardwareProfile("synthetic", {"test": 1e12}, 1e12, "test")
        a = torch.ones(16, 16, device="cuda")
        reference = mm(a, a)
        for case in ("failed", "valid", "cpu"):
            lengths = torch.tensor([2, 4], dtype=torch.int64 if case == "valid" else torch.int32,
                device="cpu" if case == "cpu" else "cuda")
            session = ProfileSession(run_id=case, model_id=case, rank=0, backend="nvtx")
            session.bind_layer(failing_layer, layer_id=7)
            timeline = NvtxTimeline(session, costs=costs, hardware=hardware,
                stream=torch.cuda.current_stream(), max_scopes=2, max_metadata_bytes=32,
                output=root / f"{case}.json")
            try:
                with timeline:
                    with session.iteration(iter_id=0):
                        if case == "failed":
                            failing_layer.forward(a, lengths)
                        elif case == "cpu":
                            mm(torch.ones(16, 16), torch.ones(16, 16), lengths)
                        else:
                            result = mm(a, a, lengths)
                            lengths.add_(3)
                            result = mm(a, a, lengths)
                            lengths.fill_(99)
                    timeline.finish()
                    before = (root / f"{case}.json").stat().st_mtime_ns
                    timeline.finish()
                    assert (root / f"{case}.json").stat().st_mtime_ns == before
            except ValueError as error:
                assert case == "failed" and str(error) == "intentional workload failure"
            else:
                assert case != "failed"
            report = json.loads((root / f"{case}.json").read_text())
            assert [op["tensor_values"]["lengths"] for op in report["ops"]] == (
                [[2, 4], [5, 7]] if case == "valid" else [[2, 4]])
            assert report["device_analysis"]["metadata_snapshot_bytes"] == (
                32 if case == "valid" else 8 if case == "failed" else 0)
            assert report["device_analysis"]["timing_kind"] == "pending_nsight"
            assert all("gpu_duration_ns" not in row for row in report["device_analysis"]["scopes"])
        torch.testing.assert_close(result, reference)
        # A call outside the context still has its original behavior.
        torch.testing.assert_close(mm(a, a), reference)
        # CUDA metadata stays unread in the ordinary metadata-only collector.
        plain = ProfileSession(run_id="plain", model_id="plain", rank=0, backend="none")
        with plain.iteration(iter_id=0):
            mm(a, a, torch.tensor([3], device="cuda", dtype=torch.int32))
        assert plain.ops[0].tensor_values["lengths"] is None
    """))
    kernel_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(kernel_root / "python"))
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={tmp_path / 'recovery'}",
            sys.executable,
            str(script),
            str(tmp_path),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(
        [
            nsys,
            "export",
            "--type=sqlite",
            f"--output={tmp_path / 'recovery.sqlite'}",
            str(tmp_path / "recovery.nsys-rep"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    windows = {}
    for case in ("valid", "failed", "cpu"):
        report = finalize_nsight(
            nsys=nsys,
            trace=tmp_path / "recovery.nsys-rep",
            metadata=tmp_path / f"{case}.json",
            output=tmp_path / f"{case}.nsight.json",
        )
        scopes = report["device_analysis"]["scopes"]
        row = next(item for item in scopes if item["kind"] == "op")
        if case == "failed":
            layer = next(item for item in scopes if item["kind"] == "layer")
            assert layer["cpu_start_ns"] <= row["cpu_start_ns"]
            assert row["cpu_end_ns"] <= layer["cpu_end_ns"]
            windows[case] = (layer["cpu_start_ns"], layer["cpu_end_ns"])
        else:
            windows[case] = (row["cpu_start_ns"], row["cpu_end_ns"])
        assert (row["mfu"] is not None) == (case == "valid")
        if case == "cpu":
            assert "no GPU activity" in row["mfu_missing_reason"]
    assert windows["failed"][0] < windows["failed"][1] <= windows["valid"][0]


def test_real_collective_busbw_in_native_timeline(tmp_path):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices and TOKENSPEED_TEST_NSYS")
    script = tmp_path / "collectives.py"
    script.write_text(textwrap.dedent("""\
        import os
        import sys
        from datetime import timedelta
        from pathlib import Path
        import torch
        import torch.distributed as dist
        from tokenspeed_kernel.op_profile import CostRegistry, OpProfile, ProfileSession, profiled
        from tokenspeed_kernel.profile_costs import CollectiveCost, register_collective_costs
        from tokenspeed_kernel.profile_nvtx import NvtxTimeline
        from tokenspeed_kernel.profile_utilization import HardwareProfile
        from tokenspeed_kernel.registry import KernelRegistry, KernelSpec

        root = Path(sys.argv[1])
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=90), device_id=torch.device("cuda", rank))
        size = dist.get_world_size()
        def reduce(x):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x
        def scatter(x, out):
            dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM, async_op=True).wait()
            return out
        def gather(x, out):
            dist.all_gather_into_tensor(out, x, async_op=True).wait()
            return out
        functions = {"all_reduce_sum": reduce, "reduce_scatter_sum": scatter, "all_gather": gather}
        registry = KernelRegistry()
        wrapped = {}
        costs = CostRegistry()
        register_collective_costs(costs, hbm_policy="logical_io_once")
        for name, fn in functions.items():
            registry.register(
                KernelSpec(name=name, family="collective", mode=name),
                fn,
                profile=OpProfile(cost_rule=name),
            )
            wrapped[name] = registry.get_impl(name)
            is_gather = name == "all_gather"
            costs.register(name, CollectiveCost(
                collective=name,
                tensor="return" if is_gather else "x",
                tensor_source="output" if is_gather else "input",
                world_size=size,
                group=f"nccl:{None!r}:{tuple(range(size))}",
            ))
        full = torch.full((16384,), rank + 1., device="cuda")
        shard = torch.full((16384 // size,), rank + 1., device="cuda")
        scattered = torch.empty_like(shard)
        gathered = torch.empty_like(full)
        class Layer:
            @profiled(level="layer")
            def forward(self):
                reduced = wrapped["all_reduce_sum"](full.clone())
                wrapped["reduce_scatter_sum"](full, scattered)
                wrapped["all_gather"](shard, gathered)
                return reduced
        layer = Layer()
        for _ in range(3):
            layer.forward()
        torch.cuda.synchronize()
        session = ProfileSession(run_id="collectives", model_id="collective-validation", rank=rank, backend="nvtx")
        session.bind_layer(layer, layer_id=0)
        with NvtxTimeline(session, costs=costs, hardware=HardwareProfile("test", {}, 1e12, "synthetic HBM peak for metric validation"), stream=torch.cuda.current_stream(), max_scopes=4, output=root / f"metrics.rank{rank}.json"):
            with session.iteration(iter_id=0):
                reduced = layer.forward()
        torch.testing.assert_close(reduced, torch.full_like(full, 3.), rtol=0, atol=0)
        torch.testing.assert_close(scattered, torch.full_like(shard, 3.), rtol=0, atol=0)
        expected = torch.cat([torch.full_like(shard, float(i+1)) for i in range(size)])
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        dist.destroy_process_group()
    """))
    kernel_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(kernel_root / "python"), OMP_NUM_THREADS="1")
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={tmp_path / 'collectives'}",
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(script),
            str(tmp_path),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(
        [
            nsys,
            "stats",
            "--report=nvtx_gpu_proj_trace",
            "--format=csv",
            f"--output={tmp_path / 'projection'}",
            str(tmp_path / "collectives.nsys-rep"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with (tmp_path / "projection_nvtx_gpu_proj_trace.csv").open() as source:
        projected = defaultdict(list)
        for r in csv.DictReader(source):
            projected[(int(r["TID"]), r["Name"].split(":", 1)[-1])].append(r)
    labels = set()
    for rank in range(2):
        report = finalize_nsight(
            nsys=nsys,
            trace=tmp_path / "collectives.nsys-rep",
            metadata=tmp_path / f"metrics.rank{rank}.json",
            output=tmp_path / f"nsight.rank{rank}.json",
        )
        assert report["ops"][0]["implementation"] == "all_reduce_sum"
        scopes = report["device_analysis"]["scopes"]
        assert len(scopes) == 4
        for row, expected_bytes in zip(
            scopes, (65536, 32768, 32768, 131072), strict=True
        ):
            assert row["bus_bytes"] == expected_bytes
            assert row["busbw_bytes_per_second"] == pytest.approx(
                expected_bytes / (row["gpu_duration_ns"] * 1e-9)
            )
            assert row["mfu"] is None
            if row["trace_label"] == report["ops"][0]["trace_label"]:
                assert row["estimated_mbu"] == pytest.approx(
                    2 * 65536 / (row["gpu_duration_ns"] * 1e-9) / 1e12
                )
            else:
                assert row["estimated_mbu"] is None
            label = row["nvtx_label"]
            labels.add(row["native_nvtx_label"])
            if row["kind"] == "op":
                assert "Est.MBU(HBM)=" in label
            else:
                assert label == f"L{row['layer_id']}"
            assert "MFU=" not in label and "busbw=" not in label
            native = projected[
                (report["device_analysis"]["thread_id"], row["native_nvtx_label"])
            ].pop(0)
            assert row["gpu_duration_ns"] == int(native["Projected Duration (ns)"])
            assert int(native["TID"]) == report["device_analysis"]["thread_id"]
            assert int(native["NumGPUOps"]) >= 1
    with sqlite3.connect(
        f"file:{tmp_path / 'collectives.sqlite'}?mode=ro", uri=True
    ) as db:
        events = [
            event
            for event in db.execute("SELECT eventType,globalTid,text FROM NVTX_EVENTS")
            if event[2] in labels
        ]
    assert len(events) == 8
    assert {event[0] for event in events} == {59}
    assert len({event[1] for event in events}) == 2
