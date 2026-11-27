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

"""Exercise TokenSpeed allocation owners with the standalone saver on an NPU."""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_NPU_MEMORY_SAVER_TESTS") != "1",
    reason="requires RUN_NPU_MEMORY_SAVER_TESTS=1 and an exclusively assigned NPU",
)
@pytest.mark.parametrize("graph_policy", ["discard", "cpu_backup"])
def test_runtime_owners_survive_memory_release(graph_policy):
    import torch

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    pytest.importorskip("torch_npu_memory_saver")

    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
        initialize_communication,
    )

    torch.npu.set_device(0)
    saver = TorchMemorySaverAdapter.create(enable=True, device="npu")
    workspace_saver = TorchMemorySaverAdapter.create(enable=True, device="npu")
    communication = []
    model = torch.nn.Module()
    with saver.region(tag="weights", enable_cpu_backup=False):
        model.weight = torch.nn.Parameter(torch.ones(128, device="npu"))
    with saver.region(tag="kv_cache", enable_cpu_backup=False):
        kv = torch.zeros(128, device="npu")
    with saver.named_region("attention_state"):
        constant = torch.full((128,), 7.0, device="npu")
        with workspace_saver.named_region("workspace"):
            workspace = torch.full((128,), 3.0, device="npu")
            initialize_communication(
                lambda: communication.append(torch.full((128,), 11.0, device="npu"))
            )
        assert communication == []
    (comm_buffer,) = communication

    graph = torch.npu.NPUGraph()
    inputs = torch.arange(128, device="npu", dtype=torch.float32)
    expected = torch.arange(128, dtype=torch.float32).square() + 10
    for _ in range(3):
        inputs * inputs + constant + workspace
    torch.npu.synchronize()

    try:
        with saver.npu_graph(
            graph,
            pool=None,
            stream=torch.npu.Stream(),
            tag="acl_graph",
            policy=graph_policy,
            pool_scope="decode",
            auto_dispatch_capture=True,
        ):
            output = inputs * inputs + constant + workspace
        saver.validate_startup(
            models=[("target", model)], kv_regions={"arena": kv}, graphs=[graph]
        )
        model.unmanaged = torch.nn.Parameter(torch.ones(128, device="npu"))
        with pytest.raises(RuntimeError, match="target.unmanaged.*bypassed"):
            saver.validate_startup(
                models=[("target", model)], kv_regions={"arena": kv}, graphs=[graph]
            )
        del model.unmanaged

        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        tensors = (constant, workspace, output)
        addresses = [tensor.data_ptr() for tensor in tensors]
        output_ptr = output.data_ptr()
        for _ in range(3):
            saver.pause("acl_graph")
            # The saver can retain pools from earlier captures in this process.
            (output_block,) = [
                block
                for block in saver.stats()["blocks"]
                if 0 <= output_ptr - int(block["ptr"], 16) < block["aligned_bytes"]
            ]
            assert output_block["tag"] == "acl_graph"
            assert not output_block["mapped"] and not output_block["physical_live"]
            assert output_block["backup_valid"] == (graph_policy == "cpu_backup")
            saver.pause("kv_cache")
            saver.pause("weights")
            torch.testing.assert_close(comm_buffer.cpu(), torch.full((128,), 11.0))
            saver.resume("weights")
            saver.resume("acl_graph")
            if graph_policy == "cpu_backup":
                # Check restored bytes before replay can regenerate the output.
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(constant.cpu(), torch.full((128,), 7.0))
            torch.testing.assert_close(workspace.cpu(), torch.full((128,), 3.0))
            saver.resume("kv_cache")
            graph.replay()
            torch.npu.synchronize()
            assert [tensor.data_ptr() for tensor in tensors] == addresses
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    finally:
        saver.resume()
        graph.reset()
        torch.npu.synchronize()


@pytest.mark.skipif(
    os.environ.get("RUN_NPU_MEMORY_SAVER_TESTS") != "1",
    reason="requires RUN_NPU_MEMORY_SAVER_TESTS=1 and an exclusively assigned NPU",
)
def test_moe_workspace_uses_ascend_pool(monkeypatch):
    import torch

    pytest.importorskip("torch_npu")
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    torch.npu.set_device(0)
    monkeypatch.setitem(global_server_args_dict, "enable_memory_saver", True)
    monkeypatch.setitem(global_server_args_dict, "device", "npu")
    saver = TorchMemorySaverAdapter.create(enable=True, device="npu")

    def prepare_input_buffers(plan, layer):
        # Supply a minimal kernel plan with the same four-view factory as MegaMoE.
        storage = torch.arange(32, dtype=torch.float32, device="npu").view(4, 8)
        return lambda rows, capacity: tuple(row[:rows] for row in storage)

    experts = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(experts)
    experts.plan = {"prepare_input_buffers": prepare_input_buffers}
    experts._input_buffer_factory = None
    views = experts.input_buffers(4, 8)
    ptr = views[0].untyped_storage().data_ptr()

    def backing():
        return next(
            b
            for b in saver.stats()["blocks"]
            if 0 <= ptr - int(b["ptr"], 16) < b["aligned_bytes"]
        )

    expected = torch.stack([v.cpu() for v in views])
    assert backing()["policy"] == "backup"
    assert backing()["tag"] == "weights"
    try:
        saver.pause("weights")
        assert not backing()["mapped"] and backing()["backup_valid"]
        saver.resume("weights")
        result = experts.input_buffers(4, 8)
        assert result[0].untyped_storage().data_ptr() == ptr
        torch.testing.assert_close(torch.stack([v.cpu() for v in result]), expected)
    finally:
        saver.resume("weights")


@pytest.mark.skipif(
    os.environ.get("RUN_NPU_MEMORY_SAVER_TESTS") != "1",
    reason="requires RUN_NPU_MEMORY_SAVER_TESTS=1 and an exclusively assigned NPU",
)
def test_npu_host_transfer_staging_uses_discardable_kv_pool():
    import torch

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    pytest.importorskip("torch_npu_memory_saver")

    from tokenspeed_kernel.ops.kvcache import host_transfer

    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    torch.npu.set_device(0)
    saver = TorchMemorySaverAdapter.create(enable=True, device="npu")
    payload = 4096
    with saver.region(tag="kv_cache", enable_cpu_backup=False):
        device_buf = torch.zeros(payload * 4, dtype=torch.uint8, device="npu")
    with saver.named_region("cache_transfer"):
        geometry = host_transfer.build_host_transfer_geometry(
            rows=((0, 0, 0, payload, payload, 0, 1, payload),),
            layer_slices=((0, 1),),
            group_packing=(1,),
            host_group_packing=(1,),
            host_lcm_block_bytes=payload,
            num_host_lcm_blocks=2,
            num_device_lcm_blocks=3,
            num_device_buffers=1,
        ).bind(device_buf.device, non_blocking=False)

    def _alloc_staging(nbytes: int, target_device: torch.device) -> torch.Tensor:
        with saver.region(tag="kv_cache", enable_cpu_backup=False):
            return torch.empty(nbytes, dtype=torch.uint8, device=target_device)

    staging = host_transfer.StagingBuffer(allocator=_alloc_staging, default_bytes=65536)
    staging_ptr = staging.acquire(payload, device_buf.device).data_ptr()
    host = torch.zeros(payload * 2, dtype=torch.uint8, pin_memory=True)
    workspace = host_transfer.HostTransferWorkspace()
    workspace.bind_staging_buffer(staging)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream), saver.named_region("cache_transfer"):
        workspace.prepare_backend((device_buf,), host, backend="auto")
    stream.synchronize()

    try:
        saver.pause("kv_cache")
        matching = [
            block
            for block in saver.stats()["blocks"]
            if 0 <= staging_ptr - int(block["ptr"], 16) < block["aligned_bytes"]
        ]
        assert len(matching) == 1
        assert matching[0]["tag"] == "kv_cache"
        assert not matching[0]["mapped"] and not matching[0]["physical_live"]
        assert not matching[0]["backup_valid"]
        saver.resume("kv_cache")

        pattern = (
            torch.arange(payload, dtype=torch.int64).remainder(251).to(torch.uint8)
        )
        device_buf[payload : 2 * payload].copy_(pattern.npu())
        num_blocks, _ = workspace.load_block_transfers(((0, 1, 1),), geometry=geometry)
        with torch.npu.stream(stream), saver.named_region("cache_transfer"):
            workspace.commit_block_transfers(
                num_blocks, device_buf.device, non_blocking=True
            )
        host_transfer.transfer_cache_blocks(
            "d2h",
            (device_buf,),
            host,
            geometry,
            workspace,
            stream,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=1,
            backend="auto",
            grid_cap=None,
            layer_ready_flags=None,
        )
        stream.synchronize()
        assert torch.equal(host[:payload], pattern)
    finally:
        saver.resume("kv_cache")
