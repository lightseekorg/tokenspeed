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
def test_runtime_owners_survive_memory_release():
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
            policy="discard",
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
        for _ in range(3):
            saver.pause("acl_graph")
            saver.pause("kv_cache")
            saver.pause("weights")
            torch.testing.assert_close(comm_buffer.cpu(), torch.full((128,), 11.0))
            saver.resume("weights")
            saver.resume("acl_graph")
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
