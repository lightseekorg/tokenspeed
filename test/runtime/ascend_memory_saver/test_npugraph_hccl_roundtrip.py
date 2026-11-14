#!/usr/bin/env python3
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

"""Verify real HCCL collectives in eager, ACL Graph, and remapped ACL Graph modes."""

import argparse
import gc
import json
import os
from datetime import timedelta
from pathlib import Path

from test_gsm8k_pause_resume import pta_git_version, require
from test_npugraph_vmm_roundtrip import Saver, make_pool, verify_owned


def run(args):
    import torch
    import torch.distributed as dist
    import torch_npu
    from torch_npu.npu import memory

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    require(size in (2, 4, 8), f"Expected TP2, TP4, or TP8, got {size}")
    require(
        len(visible) == len(set(visible)) == size
        and all(card.isdecimal() for card in visible)
        and torch.npu.device_count() == size,
        f"Expose exactly {size} distinct physical NPUs",
    )
    require(rank == local_rank, "Run the probe on one node")
    require(os.environ.get("TASK_QUEUE_ENABLE") == "1", "Set TASK_QUEUE_ENABLE=1")
    if args.mode == "vmm":
        require(args.library is not None, "--library is required in VMM mode")
        require(
            os.environ.get("PYTORCH_NPU_ALLOC_CONF") == "expandable_segments:False",
            "Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:False",
        )
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    report = {
        "rank": rank,
        "world_size": size,
        "physical_npu": visible[local_rank],
        "mode": args.mode,
    }

    def record(stage, **values):
        line = json.dumps(dict(report, stage=stage, **values))
        os.write(1, f"{line}\n".encode())

    record(
        "start",
        pta_commit=pta_git_version(torch_npu, args.pta_commit),
        torch_npu=torch_npu.__version__,
        device_name=torch.npu.get_device_name(local_rank),
        zero_copy=os.environ.get("TORCH_HCCL_ZERO_COPY", "0"),
        expansion=os.environ.get("HCCL_OP_EXPANSION_MODE"),
    )
    timeout = timedelta(seconds=args.timeout)
    dist.init_process_group("hccl", timeout=timeout)
    group = dist.new_group(backend="hccl", timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    owner = None
    pool = None
    stream = torch.npu.Stream()
    values = torch.full((args.elements,), rank + 1, dtype=torch.float32, device=device)
    rank_sum = size * (size + 1) // 2

    def collect(input_tensor):
        reduced = input_tensor.clone()
        dist.all_reduce(reduced, group=group)
        gathered = input_tensor.new_empty(size * input_tensor.numel())
        dist.all_gather_into_tensor(gathered, input_tensor, group=group)
        return reduced, gathered

    def check(outputs, offset):
        reduced, gathered = outputs
        torch.npu.synchronize()
        expected_sum = torch.full(
            reduced.shape, rank_sum + size * offset, dtype=reduced.dtype
        )
        require(
            torch.equal(reduced.cpu(), expected_sum),
            "AllReduce output differs from sum of inputs on every NPU",
        )
        expected = (
            torch.arange(1 + offset, size + 1 + offset, dtype=torch.float32)
            .unsqueeze(1)
            .expand(size, args.elements)
        )
        require(
            torch.equal(gathered.cpu().view(size, args.elements), expected),
            "AllGather output differs from the rank-ordered NPU inputs",
        )

    torch.npu.synchronize()
    with torch.npu.stream(stream):
        outputs = collect(values)
    check(outputs, 0)
    record("eager_collectives_verified", all_reduce=rank_sum)
    del outputs
    if args.mode == "eager":
        dist.destroy_process_group()
        record("finished")
        return

    saver = Saver(args.library) if args.mode == "vmm" else None
    if saver is not None:
        owner, pool = make_pool(memory, args.library, "graph_backup_malloc")
    hccl = group._get_backend(device)
    comm_name = hccl.get_hccl_comm_name(rank, init_comm=False)
    require(bool(comm_name), "Warmup did not initialize the HCCL communicator")
    graph = torch.npu.NPUGraph()
    dist.barrier(group=control)
    record("capture_begin")
    with torch.npu.graph(
        graph,
        pool=None if pool is None else pool.id,
        stream=stream,
        auto_dispatch_capture=True,
    ):
        outputs = collect(values)
    torch.npu.synchronize()
    record("capture_end")
    if saver is not None:
        require(graph.pool() == pool.id, "Capture did not use the VMM pool")
        for output in outputs:
            verify_owned(saver.stats(), output, "acl_graph")
        del output
    pointers = [output.data_ptr() for output in outputs]

    graph.replay()
    check(outputs, 0)
    record("initial_replay_verified", all_reduce=rank_sum)
    for cycle in range(1, args.cycles + 1):
        if saver is not None:
            backup = [output.cpu() for output in outputs]
            dist.barrier(group=control)
            released = saver.transition("pause", "acl_graph")
            stats = saver.stats()
            require(
                released > 0
                and stats["physical_bytes"] == stats["mapped_bytes"] == 0
                and stats["reserved_bytes"] == released,
                "Pause did not release the physical graph-pool pages",
            )
            record("released", cycle=cycle, released_bytes=released)
            dist.barrier(group=control)
            require(
                saver.transition("resume", "acl_graph") == released,
                "Resume changed the graph-pool size",
            )
            require(
                [output.data_ptr() for output in outputs] == pointers,
                "Resume changed a graph tensor address",
            )
            require(
                all(
                    torch.equal(output.cpu(), saved)
                    for output, saved in zip(outputs, backup)
                ),
                "Resume did not restore backed-up graph bytes",
            )
            require(
                hccl.get_hccl_comm_name(rank, init_comm=False) == comm_name,
                "Resume changed the captured HCCL communicator",
            )
            record("remapped", cycle=cycle, remapped_bytes=released)
        values.fill_(rank + 1 + cycle)
        torch.npu.synchronize()
        dist.barrier(group=control)
        graph.replay()
        check(outputs, cycle)
        record("replay_verified", cycle=cycle, all_reduce=rank_sum + size * cycle)

    dist.barrier(group=control)
    graph.reset()
    del graph, outputs, pool
    if saver is not None:
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        stats = saver.stats()
        require(
            all(
                stats[key] == 0
                for key in (
                    "segments",
                    "reserved_bytes",
                    "physical_bytes",
                    "mapped_bytes",
                    "host_bytes",
                )
            )
            and not stats["callback_fault"],
            f"Graph teardown left saver segments or a callback fault: {stats}",
        )
    del owner
    dist.destroy_process_group()
    record("finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eager", "graph", "vmm"))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--pta-commit")
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    require(
        args.elements > 0 and args.cycles > 0 and args.timeout > 0,
        "Use positive probe parameters",
    )
    run(args)


if __name__ == "__main__":
    main()
