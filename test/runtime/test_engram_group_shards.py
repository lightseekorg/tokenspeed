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

"""Cross-instance lookups on two GPUs sharing an IMEX channel."""

import gc
import math
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import save_file
from tokenspeed_kernel import host_memory
from tokenspeed_kernel.ops.communication.fabric import (
    FabricRowAllocator,
    fabric_allocation_supported,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=45,
    suite="runtime-2gpu",
    disabled_on_runners=["amd-*", "b200-*", "h100-*"],
    disabled_on_runners_reason="Engram group FABRIC coverage runs on GB GPUs with a shared IMEX channel",
)


def _engram_group_worker(device_index, workdir, host, boundary):
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.models.deepseek_v41_engram import (
        EngramGroup,
        RowShardedEngramEmbedding,
    )

    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    group = EngramGroup(
        2,
        str(workdir / "store"),
        device,
        host=host,
        timeout=timedelta(seconds=60),
    )
    handles = [b"", b"\x00\xff"]
    assert group.exchange("test-handles", handles[group.rank]) == handles
    with safe_open(
        str(workdir / "table.safetensors"), framework="pt", device="cpu"
    ) as checkpoint:
        rows, width = checkpoint.get_slice("weight").get_shape()
        table = RowShardedEngramEmbedding(
            rows,
            width,
            Mapping(rank=0),
            device,
            group,
            1,
        )
        ranges = ((0, boundary), (boundary, rows))
        assert (table.row_start, table.row_end) == ranges[group.rank]
        for field in ("weight", "scale"):
            table.load_sharded(field, checkpoint.get_slice(field), 65536)
        ids = torch.tensor([[0, boundary - 1, boundary, rows - 1]])
        codes = checkpoint.get_tensor("weight").view(torch.uint8)[ids]
        scales = checkpoint.get_tensor("scale")[ids]
    group.publish_loaded()

    ids = ids.to(device)
    actual_codes, actual_scales = table.mxfp8_rows(ids)
    assert torch.equal(actual_codes.view(torch.uint8).cpu().reshape_as(codes), codes)
    assert torch.equal(actual_scales.cpu().reshape_as(scales), scales)
    expected = (
        codes.view(torch.float8_e4m3fn).float()
        * scales.view(torch.float8_e8m0fnu).float()
    ).bfloat16()
    assert torch.equal(table(ids, query_shard=None).cpu(), expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = table(ids, query_shard=None)
    if group.rank == 0:
        first_row = table.packed.detach()[0, :width]
        del graph, result, table
        gc.collect()
        assert torch.equal(first_row.cpu(), codes[0, 0])
        del first_row
        group.store.set("owner-gone", "1")
    else:
        group.store.wait(["owner-gone"], timedelta(seconds=60))
        ids.copy_(ids.flip(1))
        graph.replay()
        assert torch.equal(result.cpu(), expected.flip(1))


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.version.cuda is None,
    reason="two NVIDIA GPUs",
)
@pytest.mark.parametrize("host", [False, True], ids=["hbm", "host-numa"])
def test_two_independent_engram_instances(tmp_path, host):
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("GPUs need peer access")
    minima = []
    for index in range(2):
        with torch.cuda.device(index):
            if not fabric_allocation_supported(index):
                pytest.skip(f"cuda:{index} needs CUDA FABRIC allocation access (IMEX)")
            if host and (
                error := host_memory.probe_host_memory(index, exportable=True)
            ):
                pytest.skip(error)
            minima.append(
                FabricRowAllocator(torch.device("cuda", index), host=host).granularity
            )

    width = 32  # One MXFP8 scale byte per row.
    row_bytes = width + 1
    granularity = math.lcm(*minima)
    boundary = granularity // math.gcd(granularity, row_bytes)
    row = torch.arange(boundary + 2, dtype=torch.int32)[:, None]
    codes = (
        (row % 97).to(torch.uint8) + torch.arange(width, dtype=torch.uint8) % 23
    ).view(torch.float8_e4m3fn)
    # A full-scale value in each block makes the expected MXFP8 output the checkpoint bytes.
    codes[:, 0] = torch.finfo(codes.dtype).max
    scales = (row % 7 + 120).to(torch.uint8)
    save_file({"weight": codes, "scale": scales}, str(tmp_path / "table.safetensors"))
    mp.spawn(_engram_group_worker, args=(tmp_path, host, boundary), nprocs=2)
