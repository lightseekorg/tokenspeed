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

"""Probe Ascend pool policies on one NPU with synthetic, fully overwritten activations."""

import argparse
import json
import os
from pathlib import Path

from test_gsm8k_pause_resume import pta_git_version


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def owning_segment(stats, tensor):
    ptr = tensor.data_ptr()
    end = ptr + tensor.numel() * tensor.element_size()
    for block in stats["blocks"]:
        base = int(block["ptr"], 16)
        if base <= ptr and end <= base + block["aligned_bytes"]:
            return block
    return None


def check_value(torch, tensor, value):
    expected = torch.full(tensor.shape, value, dtype=tensor.dtype, device="cpu")
    torch.testing.assert_close(tensor.cpu(), expected, atol=0, rtol=0)


def check_transition(stats, initial, *, paused):
    require(not stats["callback_fault"], stats["callback_error"])
    before = {block["ptr"]: block for block in initial["blocks"]}
    after = {block["ptr"]: block for block in stats["blocks"]}
    require(before.keys() == after.keys(), "A managed segment lost its reserved VA")
    for ptr, block in after.items():
        require(block["reserved"], f"Segment {ptr} lost its VA reservation")
        require(
            (block["tag"], block["policy"], block["aligned_bytes"])
            == (
                before[ptr]["tag"],
                before[ptr]["policy"],
                before[ptr]["aligned_bytes"],
            ),
            f"Segment {ptr} changed ownership or size",
        )
        require(
            block["mapped"] == block["physical_live"] == (not paused),
            f"Segment {ptr} has an unexpected physical allocation or mapping",
        )
        require(
            block["backup_valid"] == (paused and block["policy"] == "backup"),
            f"Segment {ptr} has an unexpected host backup",
        )


def run(torch, report, cycles):
    from tokenspeed.runtime.utils.ascend_memory_saver_adapter import (
        AscendMemorySaverAdapter,
    )
    from tokenspeed.runtime.utils.ascend_memory_saver_state import (
        AllocationPolicy,
        PoolKey,
    )

    require(torch.npu.device_count() == 1, "Expected one visible NPU")
    torch.npu.set_device(0)
    report["device"] = torch.npu.get_device_name(0)
    report["physical_npu"] = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    adapter = AscendMemorySaverAdapter()

    def allocate(tag, policy, scope, shape, value):
        with adapter.allocation_region(PoolKey(tag, policy, scope)):
            return torch.full(shape, value, dtype=torch.float16, device="npu")

    weight = allocate("weights", AllocationPolicy.DISCARD, "target", (256, 256), 1)
    rope = allocate("weights", AllocationPolicy.CPU_BACKUP, "rope", (1, 256), 3)
    draft = allocate("weights", AllocationPolicy.REMAIN, "draft", (1, 256), 4)
    kv = allocate("kv_cache", AllocationPolicy.DISCARD, "kv", (32, 256), 1)
    inputs = [
        torch.ones((rows, 256), dtype=torch.float16, device="npu")
        for rows in (32, 64, 96, 32)
    ]

    # Each graph allocation is written by matmul/add/relu before any read in
    # that replay. The persistent weight, RoPE and draft inputs live elsewhere.
    def forward(item):
        return torch.relu(item @ weight + rope + draft)

    def resident_forward(item):
        return torch.relu(item + draft)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for item in inputs[:3]:
            forward(item)
        resident_forward(inputs[3])
    stream.synchronize()

    def capture(item, operation, *, scope, policy, pool=None):
        graph = torch.npu.NPUGraph()
        with adapter.npu_graph(
            graph,
            pool=pool,
            stream=stream,
            tag="acl_graph",
            policy=policy,
            pool_scope=scope,
            auto_dispatch_capture=True,
        ):
            output = operation(item)
        return graph, output

    first = capture(inputs[0], forward, scope="shared", policy="discard")
    shared = capture(
        inputs[1], forward, scope="shared", policy="discard", pool=first[0].pool()
    )
    split = capture(inputs[2], forward, scope="split", policy="discard")
    resident = capture(inputs[3], resident_forward, scope="resident", policy="remain")
    managed_graphs = (first, shared, split)
    graphs = (*managed_graphs, resident)
    pool_ids = [graph.pool() for graph, _ in graphs]
    require(pool_ids[0] == pool_ids[1], "Captures in the same scope must share a pool")
    require(len({pool_ids[0], pool_ids[2], pool_ids[3]}) == 3, "Pool scopes alias")
    report["pool_ids"] = pool_ids
    torch.npu.synchronize()

    initial = adapter.stats()
    report["initial"] = initial
    managed = [
        (weight, "weights", "discard"),
        (rope, "weights", "backup"),
        (kv, "kv_cache", "discard"),
        *((out, "acl_graph", "discard") for _, out in managed_graphs),
    ]
    for tensor, tag, policy in managed:
        block = owning_segment(initial, tensor)
        require(
            block is not None and (block["tag"], block["policy"]) == (tag, policy),
            f"{tag}/{policy} tensor at {tensor.data_ptr():#x} has the wrong allocation",
        )
    for tensor in (draft, resident[1], *inputs):
        require(
            owning_segment(initial, tensor) is None,
            f"Resident tensor at {tensor.data_ptr():#x} joined a managed pool",
        )

    tensors = (weight, rope, kv, draft, *inputs, *(out for _, out in graphs))
    pointers = [tensor.data_ptr() for tensor in tensors]
    report["tensor_virtual_addresses"] = {
        name: f"{tensor.data_ptr():#x}"
        for name, tensor in zip(
            (
                "target_weight",
                "rope",
                "kv",
                "resident_draft_weight",
                "input_0",
                "input_1",
                "input_2",
                "input_3",
                "shared_graph_output_0",
                "shared_graph_output_1",
                "split_graph_output",
                "resident_graph_output",
            ),
            tensors,
            strict=True,
        )
    }

    def replay(graph, out, expected):
        graph.replay()
        torch.npu.synchronize()
        check_value(torch, out, expected)

    for graph, out in managed_graphs:
        replay(graph, out, 263)
    replay(*resident, 5)

    for cycle in range(cycles):
        adapter.pause()
        paused = adapter.stats()
        report["last_pause"] = paused
        check_transition(paused, initial, paused=True)
        require(
            paused["physical_bytes"] == paused["mapped_bytes"] == 0,
            "Pause left managed physical memory live",
        )
        check_value(torch, draft, 4)
        check_value(torch, resident[1], 5)

        adapter.resume()
        resumed = adapter.stats()
        report["last_resume"] = resumed
        check_transition(resumed, initial, paused=False)
        adapter.assert_active()
        require(
            [tensor.data_ptr() for tensor in tensors] == pointers,
            "A tensor's virtual address changed",
        )
        check_value(torch, rope, 3)

        # Rebuild discarded input storage in place before reading it, and let
        # replay rebuild the discarded graph allocations before checking output.
        value = 2 + cycle % 2
        weight.copy_(torch.full(weight.shape, value, dtype=weight.dtype, device="cpu"))
        kv.fill_(value)
        torch.npu.synchronize()
        check_value(torch, kv, value)
        for graph, out in managed_graphs:
            replay(graph, out, 256 * value + 7)
        replay(*resident, 5)
        report["cycles"].append(
            {
                "cycle": cycle + 1,
                "released_bytes": initial["physical_bytes"],
                "backup_bytes": paused["valid_backup_bytes"],
                "managed_replays": len(managed_graphs),
            }
        )

    for graph, _ in graphs:
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--pta-commit", help="Require this PTA Git commit if specified")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal():
        parser.error("set ASCEND_RT_VISIBLE_DEVICES to one physical NPU index")
    if os.environ.get("TASK_QUEUE_ENABLE") != "1":
        parser.error("set TASK_QUEUE_ENABLE=1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "cycles": []}
    try:
        import torch
        import torch_npu

        report["torch_npu"] = torch_npu.__version__
        report["pta_commit"] = pta_git_version(torch_npu, args.pta_commit)
        with torch.inference_mode():
            run(torch, report, args.cycles)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "evidence": str(args.output)}))


if __name__ == "__main__":
    main()
