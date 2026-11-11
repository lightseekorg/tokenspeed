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

"""Run with pytest on two CUDA devices; registered in the runtime-2gpu CI suite."""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-2gpu")


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    # Runtime imports may initialize CUDA, so select the rank's device first.
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as groups,
    )

    group = (0, 1)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=30),
    )
    groups.register_process_group("nccl", group, dist.group.WORLD)
    groups.register_process_group("gloo", group, dist.new_group(backend="gloo"))
    backend = NcclBackend()
    backend.configure(use_pynccl=True)
    comm = backend._get_or_create_resources(group)["pynccl_comm"]
    assert comm is not None and comm.available
    producer = torch.cuda.Stream()
    results = []
    for stream in (producer, None):
        tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.stream(producer), comm.change_state(enable=True, stream=stream):
            # Widen the overlap: the input value is ready only on the producer stream.
            torch.cuda._sleep(200_000_000)
            tensor.fill_(rank + 1)
            backend.all_reduce(tensor, group)
        torch.cuda.synchronize()
        results.append(tensor.item())
    print(
        f"rank={rank}, explicit producer stream / default PyNCCL stream: {results}",
        flush=True,
    )
    # Exercise each operation on the current side stream, including capture.
    inputs = torch.empty(2, dtype=torch.int32, device="cuda")
    reduced = torch.empty(1, dtype=torch.int32, device="cuda")
    gathered = torch.empty(2, dtype=torch.int32, device="cuda")
    scattered = torch.empty(1, dtype=torch.int32, device="cuda")
    broadcast = torch.empty(1, dtype=torch.int32, device="cuda")
    received = torch.empty(1, dtype=torch.int32, device="cuda")

    def collectives():
        torch.cuda._sleep(20_000_000)
        inputs.fill_(rank + 1)
        reduced.fill_(rank + 1)
        backend.all_reduce(reduced, group)
        backend.all_gather_single(gathered, inputs[:1], group)
        scattered.copy_(backend.reduce_scatter(inputs, group))
        broadcast.fill_(rank + 7)
        comm.broadcast(broadcast, src=0)
        if rank == 0:
            comm.send(inputs[:1], dst=1)
            comm.recv(received, src=1)
        else:
            comm.recv(received, src=0)
            comm.send(inputs[:1], dst=0)

    for captured in (False, True):
        with comm.change_state(enable=True), torch.cuda.stream(producer):
            if captured:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=producer):
                    collectives()
                graph.replay()
            else:
                collectives()
        producer.synchronize()
        assert reduced.tolist() == [3]
        assert gathered.tolist() == [1, 2]
        assert scattered.tolist() == [3]
        assert broadcast.tolist() == [7]
        assert received.tolist() == [2 - rank]
        print(f"rank={rank}, graph={captured}, all collectives correct", flush=True)

    try:
        with comm.change_state(enable=True, stream=producer):
            raise RuntimeError("leave the communication context")
    except RuntimeError:
        pass
    assert comm.disabled and comm.stream is None
    dist.destroy_process_group()
    assert results == [
        3,
        3,
    ], "collective missed the writes queued on the calling CUDA stream"


def test_pynccl_current_stream(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
