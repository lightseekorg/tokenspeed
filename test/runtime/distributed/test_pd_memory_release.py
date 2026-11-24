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

"""Tests for NCCL sleep/wake, including DP projection on four Blackwell GPUs."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch

from tokenspeed.runtime.distributed import nccl_suspend

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize(
    "version, bindings, supported",
    [
        ((2, 29, 6), ("suspend", "resume"), False),
        ((2, 29, 7), (), False),
        ((2, 30, 4), ("suspend",), False),
        ((2, 30, 4), ("resume",), False),
        ((2, 29, 7), ("suspend", "resume"), True),
    ],
)
def test_nccl_region_requires_both_bindings_and_a_supported_version(
    monkeypatch, version, bindings, supported
):
    # Stand in for a torch build's bindings and CUDA runtime, without GPUs.
    backend = type("NcclBackend", (), {name: lambda self: None for name in bindings})
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "ProcessGroupNCCL", backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: version)

    assert nccl_suspend.nccl_suspend_supported() is supported


def test_nccl_suspend_without_a_process_group_is_a_no_op():
    import torch.distributed as dist

    from tokenspeed.runtime.distributed.nccl_suspend import (
        resume_nccl_communicators,
        suspend_nccl_communicators,
    )

    if dist.is_initialized():
        pytest.skip("another test initialized torch.distributed in this process")
    assert suspend_nccl_communicators() == 0
    assert resume_nccl_communicators() == 0


@torch.no_grad()
def _projection_sleep_worker(rank, rendezvous):
    import torch.distributed as dist

    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.engine.memory_occupation import memory_regions
    from tokenspeed.runtime.engine.rust_host import HostDataPlane
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.device import DeviceHandle
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.models.kimi_k3_projection import KimiMLAColumnProj
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    world, rows, features, columns = 4, 512, 7168, 14400
    ranks = tuple(range(world))
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    process_group_manager.register_process_group("nccl", ranks, dist.group.WORLD)
    process_group_manager.init_process_group(ranks, backend="gloo")
    cpu_group = process_group_manager.get_process_group("gloo", ranks)
    initialize_comm_backend(use_pynccl=False)
    global_server_args_dict["enable_memory_saver"] = True
    saver = TorchMemorySaverAdapter.create(enable=True)

    @dataclass(frozen=True)
    class ModelConfig:
        dtype: torch.dtype = torch.bfloat16

    @dataclass(frozen=True)
    class ExecutionConfig:
        enforce_eager: bool = False

    class ProjectionRunner(ModelRunner):
        """Exercise the real communication methods with one layer and no checkpoint."""

        def __init__(self, model):
            self.model = model
            self.model_config = ModelConfig()
            self.dp_linear_max_tokens = None

    class ProjectionExecutor:
        """Capture only the projection, with no KV arena or speculative draft."""

        def __init__(self, runner, inputs, ctx):
            self.model_runner = runner
            self.draft_model_runner = None
            self.config = ExecutionConfig()
            self.device = device
            self.device_module = torch.cuda
            self.inputs = inputs
            self.ctx = ctx
            self.stream = torch.cuda.Stream(device=device)
            self.graph = self.output = None

        def complete_pending_drafting(self):
            pass

        def forget_served_inputs(self):
            pass

        def release_graphs(self):
            if self.graph is not None:
                self.graph.reset()
            self.graph = self.output = None

        def capture_graphs(self, *, entries, observer):
            assert entries is None
            self.graph = torch.cuda.CUDAGraph()
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.model_runner.model.forward_intermediate(self.inputs, ctx=self.ctx)
            with saver.cuda_graph(
                self.graph, pool=None, stream=self.stream, tag="cuda_graph"
            ):
                self.output = self.model_runner.model.forward_intermediate(
                    self.inputs, ctx=self.ctx
                ).clone()

        def replay(self):
            self.graph.replay()
            return self.output.cpu()

    values = (torch.arange(features).remainder(127) + rank).to(torch.bfloat16)
    values = values.expand(rows, features).contiguous()
    expected = values[:, torch.arange(columns) % features]
    with saver.region(tag="weights", enable_cpu_backup=True), torch.device(device):
        model = KimiMLAColumnProj(
            features,
            columns,
            DenseLayerMapping(rank=rank, world_size=world, tp_size=world, dp_size=1),
            False,
            "qkv_a",
        )
        local = torch.arange(model.weight.shape[0])
        model.weight.zero_()
        model.weight[local, (rank * model.weight.shape[0] + local) % features] = 1
        model.quant_method.process_weights_after_loading(model)
        inputs = values.to(device)
    del local
    weight_pointer, input_pointer = model.weight.data_ptr(), inputs.data_ptr()
    runner = ProjectionRunner(model)
    assert runner.prepare_dp_linear_communication(rows)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=rows,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, rows, 1),
        global_num_tokens=[rows] * world,
        collective_global_num_tokens=[rows] * world,
    )
    executor = ProjectionExecutor(runner, inputs, ctx)
    handle = DeviceHandle(executor)
    host = HostDataPlane(
        handle,
        profiler=None,
        regions=memory_regions(handle, saver, nccl_suspendable=True),
        report_compiles=False,
        max_encoder_tokens=None,
        attn_tp_cpu_group=cpu_group,
        epd_publisher=None,
        starting=False,
    )
    executor.capture_graphs(entries=None, observer=None)
    torch.testing.assert_close(executor.replay(), expected, rtol=0, atol=0)
    for cycle in range(2):
        workspace = model.projection_workspace
        native = workspace.fused_column
        assert native is not None
        symmetric_bytes = native.a.nbytes + native.receive.nbytes
        host.release_memory(["weights", "cuda_graph"])
        torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info(device)[0]
        assert (
            nccl_suspend._nccl_backends()
        ), "expected an initialized NCCL communicator"
        host.release_memory(["nccl"])
        assert workspace.closed and executor.graph is None
        assert model.projection_workspace is None
        released_bytes = torch.cuda.mem_get_info(device)[0] - free_before
        assert released_bytes >= symmetric_bytes
        ordinary_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == rank and tuple(segment["segment_pool_id"]) == (0, 0)
        )
        assert ordinary_bytes == 0, ordinary_bytes

        host.resume_memory(["nccl"])
        assert model.projection_workspace is None and executor.graph is None
        host.resume_memory(["cuda_graph", "weights"])
        dist.barrier(group=cpu_group)
        host.finish_memory_resume()
        assert model.projection_workspace is not workspace
        assert model.projection_workspace.fused_column is not None
        assert (model.weight.data_ptr(), inputs.data_ptr()) == (
            weight_pointer,
            input_pointer,
        )
        inputs.add_(cycle + 1)
        expected.add_(cycle + 1)
        torch.testing.assert_close(executor.replay(), expected, rtol=0, atol=0)
        if rank == 0:
            print(
                f"sleep/wake {cycle + 1}: reclaimed {released_bytes} bytes, ordinary={ordinary_bytes}, exact replay",
                flush=True,
            )
    executor.release_graphs()
    runner.release_dp_linear_communication()
    dist.destroy_process_group()


def test_bf16_projection_two_nccl_sleep_wake_cycles(tmp_path):
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    if (
        torch.cuda.device_count() < 4
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or not nccl_suspend.nccl_suspend_supported()
    ):
        pytest.skip("requires four node-local Blackwell GPUs and NCCL suspend/resume")
    pytest.importorskip("torch_memory_saver")
    with TorchMemorySaverAdapter.create(enable=True).configure_subprocess():
        torch.multiprocessing.spawn(
            _projection_sleep_worker,
            args=((tmp_path / "rendezvous").as_uri(),),
            nprocs=4,
            join=True,
        )
