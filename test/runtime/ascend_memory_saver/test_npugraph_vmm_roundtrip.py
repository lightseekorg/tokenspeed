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

"""Exercise the selected PTA wheel, real ACL VMM and two shared NPUGraphs."""

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path

from test_gsm8k_pause_resume import POLICIES, pta_git_version, require


def transition_bytes(event):
    tag = event["tag"]
    policy = POLICIES[tag]
    pausing = event["event"] == "pause"
    before = event["before"]
    after = event["after"]
    require(
        not before["callback_fault"] and not after["callback_fault"],
        f"{tag}: native callback fault",
    )
    old_blocks = {
        block["ptr"]: block for block in before["blocks"] if block["tag"] == tag
    }
    new_blocks = {
        block["ptr"]: block for block in after["blocks"] if block["tag"] == tag
    }
    require(old_blocks and old_blocks.keys() == new_blocks.keys(), f"{tag}: VA changed")
    for ptr, old in old_blocks.items():
        new = new_blocks[ptr]
        require(
            old["reserved"]
            and new["reserved"]
            and old["policy"] == new["policy"] == policy
            and old["aligned_bytes"] == new["aligned_bytes"],
            f"{tag} at {ptr}: storage metadata changed",
        )
        require(
            old["physical_live"] == old["mapped"] == pausing
            and new["physical_live"] == new["mapped"] == (not pausing),
            f"{tag} at {ptr}: unexpected physical allocation or mapping",
        )
        if pausing:
            require(
                new["backup_valid"] == (policy == "backup"),
                f"{tag} at {ptr}: incorrect backup state",
            )
    return sum(block["aligned_bytes"] for block in new_blocks.values())


class Saver:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        signatures = {
            "saver_abi_version": ([], ctypes.c_int),
            "saver_last_error": ([], ctypes.c_char_p),
            "saver_stats": ([], ctypes.c_char_p),
            "saver_pause": ([ctypes.c_char_p], ctypes.c_int),
            "saver_resume": ([ctypes.c_char_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result
        require(self.library.saver_abi_version() == 2, "Unexpected native ABI")

    def stats(self):
        return json.loads(self.library.saver_stats())

    def transition(self, operation, tag):
        before = self.stats()
        status = getattr(self.library, f"saver_{operation}")(tag.encode())
        require(status == 0, self.library.saver_last_error().decode())
        after = self.stats()
        amount = transition_bytes(
            {
                "event": operation,
                "tag": tag,
                "before": before,
                "after": after,
            }
        )
        return amount


def make_pool(memory, path, symbol):
    owner = memory.NPUPluggableAllocator(str(path), symbol, "saver_free")
    return owner, memory.MemPool(owner.allocator())


def free_bytes(torch):
    return int(torch.npu.mem_get_info(0)[0])


def verify_owned(stats, tensor, tag):
    ptr = tensor.data_ptr()
    end = ptr + tensor.numel() * tensor.element_size()
    require(
        any(
            block["tag"] == tag
            and int(block["ptr"], 16) <= ptr
            and end <= int(block["ptr"], 16) + block["aligned_bytes"]
            for block in stats["blocks"]
        ),
        f"{tag} tensor bypassed the real VMM allocator at {ptr:#x}",
    )


def run(args, report):
    import torch
    import torch_npu
    from torch_npu.npu import memory

    git_version = pta_git_version(torch_npu, args.pta_commit)
    require(torch.npu.device_count() == 1, "Expected a single visible NPU")
    torch.npu.set_device(0)
    saver = Saver(args.library)
    report.update(
        torch_npu=torch_npu.__version__,
        pta_commit=git_version,
        physical_npu=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        device=torch.npu.get_device_name(0),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
    )
    symbols = ("weights_backup_malloc", "kv_malloc", "graph_backup_malloc")
    weight_owner, weight_pool = make_pool(memory, args.library, symbols[0])
    kv_owner, kv_pool = make_pool(memory, args.library, symbols[1])
    graph_owner, graph_pool = make_pool(memory, args.library, symbols[2])
    with memory.use_mem_pool(weight_pool):
        weight = torch.full((256, 256), 2.0, dtype=torch.float16, device="npu")
    with memory.use_mem_pool(kv_pool):
        kv = torch.full((2**20,), 3, dtype=torch.uint8, device="npu")
    inputs = [
        torch.ones((rows, 256), dtype=torch.float16, device="npu") for rows in (32, 64)
    ]
    capture_stream = torch.npu.Stream()
    require(capture_stream != torch.npu.default_stream(), "Expected a side stream")
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(capture_stream):
        for item in inputs:
            torch.relu(item @ weight)
    capture_stream.synchronize()
    del item

    graphs = []
    outputs = []
    capture_calls = []
    capture_begin = torch.npu.NPUGraph.capture_begin

    def record_capture(graph, *args, **kwargs):
        result = capture_begin(graph, *args, **kwargs)
        capture_calls.append(id(graph))
        return result

    torch.npu.NPUGraph.capture_begin = record_capture
    try:
        for item in inputs:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                graph,
                pool=graph_pool.id,
                stream=capture_stream,
                auto_dispatch_capture=True,
            ):
                output = torch.relu(item @ weight)
            graphs.append(graph)
            outputs.append(output)
        torch.npu.synchronize()
        del item, graph, output
        require(
            len(capture_calls) == 2
            and all(graph.pool() == graph_pool.id for graph in graphs),
            "Each bucket must capture once in the same custom pool",
        )

        stats = saver.stats()
        verify_owned(stats, weight, "weights")
        verify_owned(stats, kv, "kv_cache")
        for output in outputs:
            verify_owned(stats, output, "acl_graph")
        del output
        managed_bytes = stats["physical_bytes"]
        report["initial"] = stats
        report["shared_pool"] = list(graph_pool.id)
        saved_pointers = [t.data_ptr() for t in (weight, kv, *inputs, *outputs)]

        with torch.inference_mode():
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            del graph
            for cycle in range(args.cycles):
                old_weight = weight.cpu()
                old_outputs = [output.cpu() for output in outputs]
                for output in old_outputs:
                    require(
                        torch.isfinite(output).all().item(), "Invalid pre-sleep output"
                    )
                del output
                before = free_bytes(torch)
                freed = {tag: saver.transition("pause", tag) for tag in POLICIES}
                paused = saver.stats()
                require(
                    paused["physical_bytes"] == paused["mapped_bytes"] == 0
                    and paused["reserved_bytes"] == managed_bytes,
                    "Pause must free all managed physical pages and keep their VA",
                )
                asleep = free_bytes(torch)
                require(asleep > before, "CANN did not report released HBM")
                for tag in ("acl_graph", "weights", "kv_cache"):
                    require(
                        saver.transition("resume", tag) == freed[tag],
                        f"{tag} size changed",
                    )
                awake = free_bytes(torch)
                require(awake < asleep, "CANN did not report remapped HBM")

                # Inspect backed bytes before writing any new input or weight,
                # and before any replay can regenerate the graph output.
                require(
                    torch.equal(weight.cpu(), old_weight), "Weight backup data changed"
                )
                require(
                    all(
                        torch.equal(output.cpu(), expected)
                        for output, expected in zip(outputs, old_outputs)
                    ),
                    "Graph output backup data changed",
                )
                require(
                    [t.data_ptr() for t in (weight, kv, *inputs, *outputs)]
                    == saved_pointers,
                    "Managed tensor virtual address changed",
                )
                # Keep expected outputs in range for long FP16 stress runs.
                weight_value = float(cycle % 8 + 3)
                weight.copy_(torch.full(weight.shape, weight_value, dtype=weight.dtype))
                for bucket, (graph, item, output) in enumerate(
                    zip(graphs, inputs, outputs), 1
                ):
                    input_value = float(cycle % 8 + bucket)
                    item.fill_(input_value)
                    torch.npu.synchronize()
                    graph.replay()
                    torch.npu.synchronize()
                    expected = torch.full(
                        output.shape,
                        256 * weight_value * input_value,
                        dtype=output.dtype,
                    )
                    require(
                        torch.equal(output.cpu(), expected),
                        f"Bucket {bucket}: bad replay",
                    )
                del graph, item, output
                require(len(capture_calls) == 2, "Graph was recaptured")
                report["cycles"].append(
                    {
                        "cycle": cycle + 1,
                        "freed_by_tag": freed,
                        "device_hbm_before": before,
                        "device_hbm_paused": asleep,
                        "device_hbm_resumed": awake,
                        "capture_calls": len(capture_calls),
                        "successful_replays": len(graphs),
                        "backup_verified_before_overwrite": True,
                    }
                )
                print(json.dumps(report["cycles"][-1]), flush=True)

        if args.teardown_paused:
            for tag in POLICIES:
                saver.transition("pause", tag)
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
        del graph, weight, kv
        graphs.clear()
        inputs.clear()
        outputs.clear()
        del graph_pool, weight_pool, kv_pool
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        final = saver.stats()
        report["final"] = final
        for key in (
            "segments",
            "reserved_bytes",
            "physical_bytes",
            "mapped_bytes",
            "host_bytes",
        ):
            require(final[key] == 0, f"Teardown left {key}={final[key]}")
        require(not final["callback_fault"], final["callback_error"])
        del graph_owner, weight_owner, kv_owner
    finally:
        torch.npu.NPUGraph.capture_begin = capture_begin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--pta-commit", help="Require this PTA Git commit if specified")
    parser.add_argument("--teardown-paused", action="store_true")
    args = parser.parse_args()
    args.library = args.library.resolve(strict=True)
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal():
        parser.error("set ASCEND_RT_VISIBLE_DEVICES to one physical NPU index")
    if os.environ.get("TASK_QUEUE_ENABLE") != "1":
        parser.error("set TASK_QUEUE_ENABLE=1")
    if os.environ.get("PYTORCH_NPU_ALLOC_CONF") != "expandable_segments:False":
        parser.error("set PYTORCH_NPU_ALLOC_CONF=expandable_segments:False")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "teardown_paused": args.teardown_paused, "cycles": []}
    try:
        run(args, report)
        report["status"] = "passed"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "evidence": str(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
