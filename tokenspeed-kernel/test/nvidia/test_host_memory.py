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

"""Host-NUMA cuMem memory: one mapping for the CPU and the GPU, and its sharing."""

from __future__ import annotations

import gc

import pytest
import torch
import torch.multiprocessing as mp
from utils import is_nvidia

if not (is_nvidia() and torch.cuda.is_available()):
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from tokenspeed_kernel import host_memory  # noqa: E402
from tokenspeed_kernel.ops.embedding import host_gather  # noqa: E402

_MIB = 1 << 20


@pytest.fixture(autouse=True)
def _device() -> int:
    # A runtime call makes the primary context current before the driver calls.
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    error = host_memory.probe_host_memory(device, False)
    if error is not None:
        pytest.skip(f"host-NUMA cuMem memory unavailable: {error}")
    return device


def _pattern(nbytes: int) -> torch.Tensor:
    return (torch.arange(nbytes, dtype=torch.int64) * 7 % 251).to(torch.uint8)


def _gpu_rows(memory: host_memory.HostMemory, width: int, rows: list[int]):
    """Rows of the memory as a [rows, width] table, read by a GPU kernel."""
    table = memory.tensor[: memory.tensor.numel() // width * width].view(-1, width)
    ids = torch.tensor(rows, device="cuda")
    return host_gather.uint8_row_gather(table, ids, None).cpu()


def test_chunks_map_for_the_cpu_and_the_gpu_at_one_address(_device):
    node = host_memory.host_numa_node(_device)
    # Not a granule multiple: the tensor keeps the requested size.
    nbytes = 3 * 2 * _MIB + 5
    memory = host_memory.create_host_memory(nbytes, (node,), _device, False, 1, 0.05)
    assert memory.tensor.numel() == nbytes and memory.tensor.device.type == "cpu"
    assert memory.numa_nodes == (node,) and memory.handles == ()
    # A 1-byte chunk size rounds up to the 2 MiB granularity: four chunks.
    assert memory.granularity == 2 * _MIB
    assert [length for _, length, _ in memory.chunks] == [2 * _MIB] * 4
    memory.tensor.copy_(_pattern(nbytes))
    assert torch.equal(memory.tensor[-5:], _pattern(nbytes)[-5:])
    width = 4096
    # Rows on either side of each chunk boundary the full rows reach.
    rows = [0, 511, 512, 1023, 1024, 1535]
    expected = _pattern(nbytes)[: 1536 * width].view(-1, width)[rows]
    assert torch.equal(_gpu_rows(memory, width, rows), expected)


def test_chunks_interleave_over_numa_nodes(_device):
    nodes = sorted(
        {host_memory.host_numa_node(i) for i in range(torch.cuda.device_count())}
        - {None}
    )
    if len(nodes) < 2:
        pytest.skip("the GPUs share one host NUMA node")
    memory = host_memory.create_host_memory(
        8 * _MIB, nodes, _device, False, 2 * _MIB, 0.05
    )
    assert [node for _, _, node in memory.chunks] == [
        nodes[index % len(nodes)] for index in range(4)
    ]
    memory.tensor.copy_(_pattern(8 * _MIB))
    expected = _pattern(8 * _MIB).view(-1, _MIB)
    assert torch.equal(_gpu_rows(memory, _MIB, [0, 2, 4, 7]), expected[[0, 2, 4, 7]])


def test_the_reserve_stops_creation_before_anything_is_allocated(_device):
    node = host_memory.host_numa_node(_device)
    with pytest.raises(MemoryError, match="reserve"):
        # Nearly the whole node is reserved for other allocations.
        host_memory.create_host_memory(2 * _MIB, (node,), _device, False, 1, 0.999)


def test_the_last_view_unmaps_the_memory(_device):
    node = host_memory.host_numa_node(_device)
    memory = host_memory.create_host_memory(
        2 * _MIB, (node,), _device, False, 2 * _MIB, 0.05
    )
    memory.tensor.fill_(9)
    view = memory.tensor[100:200]
    finalizer = memory._finalizer
    del memory
    gc.collect()
    assert finalizer.alive and torch.all(view == 9)
    del view
    gc.collect()
    assert not finalizer.alive


def _import_and_mark(rank: int, exported, device: int) -> None:
    torch.cuda.set_device(device)
    torch.cuda.synchronize()
    memory = host_memory.import_host_memory(exported, device)
    assert memory.handles == () and memory.chunks == exported.chunks
    nbytes = exported.nbytes
    assert torch.equal(memory.tensor, _pattern(nbytes))
    width = 4096
    # Rows of both chunks, on either side of the boundary.
    rows = [0, 511, 512, 1023]
    expected = _pattern(nbytes).view(-1, width)[rows]
    assert torch.equal(_gpu_rows(memory, width, rows), expected)
    # The creator sees what this process writes.
    memory.tensor[nbytes // 2 :].fill_(173)


def test_exported_memory_maps_in_another_process(_device):
    error = host_memory.probe_host_memory(_device, True)
    if error is not None:
        pytest.skip(f"FABRIC handles unavailable: {error}")
    node = host_memory.host_numa_node(_device)
    nbytes = 4 * _MIB
    memory = host_memory.create_host_memory(
        nbytes, (node,), _device, True, 2 * _MIB, 0.05
    )
    assert len(memory.handles) == 2
    memory.tensor.copy_(_pattern(nbytes))
    mp.spawn(_import_and_mark, args=(memory.export(), _device), nprocs=1, join=True)
    assert torch.equal(memory.tensor[: nbytes // 2], _pattern(nbytes)[: nbytes // 2])
    assert torch.all(memory.tensor[nbytes // 2 :] == 173)
