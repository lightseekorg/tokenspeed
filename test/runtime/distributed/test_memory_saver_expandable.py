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

"""Verify scoped allocator compatibility with the real memory saver in a child."""

import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch
from packaging.version import Version

from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
    TorchMemorySaverAdapter,
    initialize_communication,
)

register_cuda_ci(est_time=30, suite="runtime-1gpu")


_CONFIG = (
    "max_split_size_mb:128,roundup_power2_divisions:[256:1,512:2,>:4],"
    "expandable_segments:True"
)
_ENV_NAMES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


def worker(configuration):
    torch.cuda.set_device(0)
    if configuration in ("runtime_override", "sticky"):
        torch.cuda.memory._set_allocator_settings(_CONFIG)
    if configuration == "sticky":
        torch.cuda.memory._set_allocator_settings("max_split_size_mb:128")
    original_settings = torch.cuda.memory._snapshot()["allocator_settings"]
    original_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    if (
        configuration == "PYTORCH_ALLOC_CONF"
        and not original_settings["expandable_segments"]
    ):
        return False
    assert original_settings["expandable_segments"]
    saver = TorchMemorySaverAdapter.create(enable=True)

    def assert_restored():
        assert torch.cuda.memory._snapshot()["allocator_settings"] == original_settings
        assert {name: os.environ.get(name) for name in _ENV_NAMES} == original_env

    before = torch.full((24 << 20,), 17, dtype=torch.uint8, device="cuda")
    # Even a failed load must restore settings and leave the next scope usable.
    with pytest.raises(RuntimeError, match="load failed"):
        with saver.region(tag="weights", enable_cpu_backup=True):
            weights = torch.full((8 << 20,), 5, dtype=torch.float32, device="cuda")
            raise RuntimeError("load failed")
    assert_restored()
    communication = []
    with saver.region(tag="kv_cache", enable_cpu_backup=False):
        kv = torch.full((32 << 20,), 9, dtype=torch.uint8, device="cuda")
        initialize_communication(
            lambda: communication.append(
                torch.full((24 << 20,), 19, dtype=torch.uint8, device="cuda")
            )
        )
    assert_restored()
    after = communication[0]
    with saver.named_region("attention_state"):
        with saver.named_region("workspace"):
            constant = torch.full((1 << 20,), 11, dtype=torch.uint8, device="cuda")
    assert_restored()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup = weights[:1024] * 2
    torch.cuda.current_stream().wait_stream(stream)
    del warmup
    graph = torch.cuda.CUDAGraph()
    with saver.cuda_graph(
        graph, pool=torch.cuda.graph_pool_handle(), stream=stream, tag="cuda_graph"
    ):
        output = weights[:1024] * 2
    assert_restored()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(output == 10).item()

    segments = torch.cuda.memory_snapshot()
    for tensor, expandable in (
        (before, True),
        (after, True),
        (weights, False),
        (kv, False),
        (output, False),
        (constant, False),
    ):
        segment = next(
            s
            for s in segments
            if s["address"] <= tensor.data_ptr() < s["address"] + s["total_size"]
        )
        assert segment["is_expandable"] is expandable
    tensors = (weights, kv, output, before, after)
    pointers = [t.data_ptr() for t in tensors]
    for cycle in range(3):
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info()
        for tag in ("cuda_graph", "kv_cache", "weights"):
            saver.pause(tag=tag)
            assert_restored()
        free_paused, _ = torch.cuda.mem_get_info()
        assert free_paused - free_before >= 32 << 20
        assert torch.all(before == 17).item()
        assert torch.all(after == 19).item()
        for tag in ("weights", "kv_cache", "cuda_graph"):
            saver.resume(tag=tag)
            assert_restored()
        assert pointers == [t.data_ptr() for t in tensors]
        assert torch.all(weights == 5).item()
        assert torch.all(constant == 11).item()
        kv.fill_(cycle + 1)
        assert torch.all(kv == cycle + 1).item()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(output == 10).item()
        print(
            f"cycle={cycle}: released={free_paused - free_before} bytes; data, pointers, graph and configuration restored",
            flush=True,
        )
    print("Scoped expandable allocator compatibility passed", flush=True)
    return True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA memory saver",
)
@pytest.mark.parametrize("configuration", [*_ENV_NAMES, "runtime_override", "sticky"])
def test_expandable_allocations_around_managed_storage(configuration):
    pytest.importorskip("torch_memory_saver")
    if Version(version("torch-memory-saver")) < Version("0.0.10"):
        pytest.skip("requires memory saver with managed-pointer teardown support")
    with TorchMemorySaverAdapter.create(enable=True).configure_subprocess():
        env = dict(os.environ)
        for name in _ENV_NAMES:
            env.pop(name, None)
        if configuration in _ENV_NAMES:
            env[configuration] = _CONFIG
        else:
            env["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "test.runtime.distributed.test_memory_saver_expandable",
                configuration,
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    if result.returncode == 77 and configuration == "PYTORCH_ALLOC_CONF":
        pytest.skip("this PyTorch build only reads PYTORCH_CUDA_ALLOC_CONF")
    assert result.returncode == 0, result.stdout
    assert "Scoped expandable allocator compatibility passed" in result.stdout
    print(result.stdout, end="")


if __name__ == "__main__":
    sys.exit(0 if worker(sys.argv[1]) else 77)
