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

"""Real one-NPU owner/workspace persistence, with explicit weights and graph modes."""

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

ALIASES = ("executor_inputs", "runtime_states", "attention_state", "workspace")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def owning_block(stats, tensor):
    pointer = tensor.data_ptr()
    end = pointer + tensor.numel() * tensor.element_size()
    matches = [
        block
        for block in stats["blocks"]
        if int(block["ptr"], 16) <= pointer
        and end <= int(block["ptr"], 16) + block["aligned_bytes"]
    ]
    require(
        len(matches) == 1, "Tensor did not belong to exactly one real native segment"
    )
    return matches[0]


def tensor_pool_id(torch, tensor):
    pointer = tensor.data_ptr()
    end = pointer + tensor.numel() * tensor.element_size()
    matches = [
        segment
        for segment in torch.npu.memory_snapshot()
        if segment["address"] <= pointer
        and end <= segment["address"] + segment["total_size"]
    ]
    require(len(matches) == 1, "Tensor did not belong to exactly one real PTA segment")
    return tuple(matches[0]["segment_pool_id"])


def persistent_pool_evidence(torch, adapter):
    """Read-only pool/block join, also reusable by a later real model-host trace."""
    from tokenspeed.runtime.utils.ascend_memory_saver_state import (
        AllocationPolicy,
        PoolKey,
    )

    key = PoolKey("weights", AllocationPolicy.CPU_BACKUP, "persistent_buffers")
    state = adapter._state()
    pool_id = tuple(state.state.pools[key].pool.id)
    segments = [
        segment
        for segment in torch.npu.memory_snapshot()
        if tuple(segment["segment_pool_id"]) == pool_id
    ]
    require(segments, "The production persistent PTA pool has no live allocations")
    blocks = state.native.stats()["blocks"]
    for segment in segments:
        require(
            any(
                block["tag"] == "weights"
                and block["policy"] == "backup"
                and block["reserved"]
                and block["mapped"]
                and block["physical_live"]
                and block["phase"] == "active"
                and int(block["ptr"], 16) <= segment["address"]
                and segment["address"] + segment["total_size"]
                <= int(block["ptr"], 16) + block["aligned_bytes"]
                for block in blocks
            ),
            "Persistent PTA segment bypassed native weights backup",
        )
    return {
        "pool_id": pool_id,
        "PTA_segments": len(segments),
        "PTA_bytes": sum(segment["total_size"] for segment in segments),
    }


def check_transition(before, after, tag, *, pausing):
    require(
        not before["callback_fault"] and not after["callback_fault"],
        "Native callback fault",
    )
    old = {block["ptr"]: block for block in before["blocks"]}
    new = {block["ptr"]: block for block in after["blocks"]}
    require(
        old.keys() == new.keys(),
        "A transition changed the native segment reservation set",
    )
    released = 0
    for pointer, block in new.items():
        previous = old[pointer]
        require(
            (block["tag"], block["policy"], block["aligned_bytes"], block["reserved"])
            == (previous["tag"], previous["policy"], previous["aligned_bytes"], True),
            "A transition changed ownership, size, or virtual reservation",
        )
        if block["tag"] != tag:
            require(
                block == previous, "A tag-specific transition touched an unrelated pool"
            )
            continue
        released += block["aligned_bytes"]
        require(
            block["mapped"] == block["physical_live"] == (not pausing),
            "Expected actual unmap/physical release or remap/physical allocation",
        )
        require(
            block["backup_valid"] == (pausing and block["policy"] == "backup"),
            "Unexpected per-block CPU backup validity",
        )
        require(
            block["contents_valid"] == (not pausing), "Unexpected native content phase"
        )
    require(released > 0, f"No real allocation was exercised for {tag}")
    direction = -1 if pausing else 1
    require(
        after["physical_bytes"] - before["physical_bytes"] == direction * released
        and after["mapped_bytes"] - before["mapped_bytes"] == direction * released,
        "Native physical/mapped byte totals did not reflect the actual transition",
    )
    backup = sum(
        block["aligned_bytes"] for block in new.values() if block["backup_valid"]
    )
    require(
        after["host_bytes"] == after["valid_backup_bytes"] == backup,
        "Native host allocation and valid-backup byte totals disagree",
    )
    require(
        after["reserved_bytes"] == before["reserved_bytes"], "Reserved VA bytes changed"
    )
    return {
        "tag": tag,
        "pausing": pausing,
        "transition_bytes": released,
        "host_bytes": after["host_bytes"],
        "valid_backup_bytes": backup,
    }


def run(torch, args, report):
    from tokenspeed.runtime.execution import workspace as workspace_module
    from tokenspeed.runtime.utils import ascend_memory_saver_adapter as adapter_module
    from tokenspeed.runtime.utils import ascend_memory_saver_state as state_module
    from tokenspeed.runtime.utils import torch_memory_saver_adapter as factory_module
    from tokenspeed.runtime.utils.ascend_memory_saver_state import (
        AllocationPolicy,
        PoolKey,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict
    from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
        TorchMemorySaverAdapter,
    )

    root = args.source.resolve()
    for module, relative in (
        (adapter_module, "utils/ascend_memory_saver_adapter.py"),
        (state_module, "utils/ascend_memory_saver_state.py"),
        (factory_module, "utils/torch_memory_saver_adapter.py"),
        (workspace_module, "execution/workspace.py"),
    ):
        require(
            Path(module.__file__).resolve()
            == root / "python/tokenspeed/runtime" / relative,
            f"Runtime module came from outside the qualified source: {module.__file__}",
        )
    require(torch.npu.device_count() == 1, "Exactly one leased visible NPU is required")
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    adapter = TorchMemorySaverAdapter.create(enable=True, device="npu")
    other = TorchMemorySaverAdapter.create(enable=True, device="npu")
    require(
        isinstance(adapter, adapter_module.AscendMemorySaverAdapter)
        and isinstance(other, adapter_module.AscendMemorySaverAdapter)
        and adapter is not other,
        "The explicit-device production factory must provide two real Ascend adapters",
    )
    state = adapter._state()
    require(
        state is other._state(),
        "Adapters did not share the actual process/device state",
    )
    require(
        adapter.stats()["segments"] == 0,
        "Use a fresh dedicated process for each weight policy",
    )
    original_route = state.use_mem_pool
    route_events = []

    @contextmanager
    def observe_real_route(pool):
        route_events.append(("enter", tuple(pool.id)))
        with original_route(pool):
            yield
        route_events.append(("exit", tuple(pool.id)))

    def pattern(size, offset):
        return ((torch.arange(size, dtype=torch.int64) + offset) % 251).to(torch.uint8)

    def equal(tensor, expected):
        require(torch.equal(tensor.cpu(), expected), "Full tensor contents changed")

    def transition(tag, pausing):
        before = adapter.stats()
        (adapter.pause if pausing else adapter.resume)(tag)
        report["transitions"].append(
            check_transition(before, adapter.stats(), tag, pausing=pausing)
        )

    buffers, expected = {}, {}
    graph = None
    previous_args = dict(global_server_args_dict)
    state.use_mem_pool = observe_real_route
    try:
        for index, name in enumerate(ALIASES):
            expected[name] = pattern(131101 + index * 257, 17 * index)
        # Observe, but never replace, the actual PTA scope. Inner aliases must
        # share the outer scope even when they use different adapter instances.
        with ExitStack() as stack:
            for index, name in enumerate(ALIASES):
                stack.enter_context(
                    (adapter if index % 2 == 0 else other).named_region(name)
                )
                buffers[name] = expected[name].to(device)
        persistent_key = PoolKey(
            "weights", AllocationPolicy.CPU_BACKUP, "persistent_buffers"
        )
        persistent_id = tuple(state.state.pools[persistent_key].pool.id)
        require(
            route_events == [("enter", persistent_id), ("exit", persistent_id)],
            "Nested aliases entered an additional real PTA allocation scope",
        )
        report["nested_alias_PTA_scope_entries"] = 1

        global_server_args_dict["enable_memory_saver"] = True
        # Construct and grow OUTSIDE an outer named scope. Otherwise inherited
        # routing could conceal a Noop production WorkspacePool adapter.
        workspace = workspace_module.WorkspacePool(device, initial_nbytes=4096)
        require(
            isinstance(
                workspace._memory_saver, adapter_module.AscendMemorySaverAdapter
            ),
            "Production NPU WorkspacePool routing is pending; do not inject a substitute adapter",
        )
        (small,) = workspace.allocate(((4096,), torch.uint8))
        require(
            tensor_pool_id(torch, small) == persistent_id,
            "Initial workspace bypassed the persistent pool",
        )
        del small
        (grown,) = workspace.allocate(((2 * 1024 * 1024 + 257,), torch.uint8))
        del grown
        (work,) = workspace.allocate(((workspace.capacity_bytes,), torch.uint8))
        expected["workspace_backing"] = pattern(work.numel(), 113)
        work.copy_(expected["workspace_backing"])
        buffers["workspace_backing"] = work
        workspace.freeze()
        try:
            workspace.allocate(((workspace.capacity_bytes + 1,), torch.uint8))
        except RuntimeError as error:
            require(
                "frozen" in str(error),
                "Frozen workspace failed for an unrelated reason",
            )
        else:
            raise AssertionError("Frozen WorkspacePool allowed growth")

        weight_expected = pattern(524321, 7)
        with adapter.region(
            "weights", enable_cpu_backup=args.weight_policy == "cpu_backup"
        ):
            weight = weight_expected.to(device)
        with other.region("weights", enable_cpu_backup=True):
            ordinary_backup = pattern(262177, 19).to(device)
        ordinary_expected = ordinary_backup.cpu()
        with adapter.region("kv_cache", enable_cpu_backup=False):
            kv = pattern(524309, 29).to(device)
        graph_expected = (expected["executor_inputs"] + 3).to(torch.uint8)
        if args.execution_mode == "graph":
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                warm = buffers["executor_inputs"] + 3
            stream.synchronize()
            del warm
            graph = torch.npu.NPUGraph()
            with adapter.npu_graph(
                graph,
                pool=None,
                stream=stream,
                tag="acl_graph",
                policy="discard",
                pool_scope="persistent_regression_graph",
                auto_dispatch_capture=True,
            ):
                graph_output = buffers["executor_inputs"] + 3
        else:
            with adapter.region("acl_graph", enable_cpu_backup=False):
                graph_output = graph_expected.to(device)
        torch.npu.synchronize()
        initial = adapter.stats()
        for name, tensor in buffers.items():
            block = owning_block(initial, tensor)
            require(
                (block["tag"], block["policy"]) == ("weights", "backup")
                and tensor_pool_id(torch, tensor) == persistent_id,
                f"{name} did not use the real persistent CPU-backup pool",
            )
        for tensor in (weight, ordinary_backup, kv, graph_output):
            require(
                tensor_pool_id(torch, tensor) != persistent_id,
                "Persistent pool mixed with weights/KV/graph storage",
            )
        require(
            (
                owning_block(initial, weight)["tag"],
                owning_block(initial, weight)["policy"],
            )
            == (
                "weights",
                "backup" if args.weight_policy == "cpu_backup" else "discard",
            ),
            "Parameter weight tag/policy was not applied",
        )
        require(
            (
                owning_block(initial, ordinary_backup)["tag"],
                owning_block(initial, ordinary_backup)["policy"],
            )
            == ("weights", "backup"),
            "Ordinary backup weights used the wrong tag/policy",
        )
        require(
            (
                owning_block(initial, kv)["tag"],
                owning_block(initial, graph_output)["tag"],
            )
            == ("kv_cache", "acl_graph"),
            "KV or graph storage used the wrong native tag",
        )
        report["persistent_pool"] = persistent_pool_evidence(torch, adapter)
        pointers = {name: tensor.data_ptr() for name, tensor in buffers.items()}

        def check_persistent():
            for name, tensor in buffers.items():
                require(tensor.data_ptr() == pointers[name], "Persistent VA moved")
                equal(tensor, expected[name])

        check_persistent()
        transition("kv_cache", True)
        check_persistent()
        transition("weights", True)
        report["paused_weights"] = adapter.stats()
        transition("weights", False)
        # KV is still asleep; restored weights buffers must nevertheless allow
        # their production named scope. No global all-tags-active bypass is used.
        with other.named_region("attention_state"):
            pass
        check_persistent()
        equal(ordinary_backup, ordinary_expected)
        if args.weight_policy == "cpu_backup":
            equal(weight, weight_expected)
        require(
            owning_block(adapter.stats(), weight)["mapped"],
            "Weight write attempted before mapping",
        )
        replacement = pattern(weight.numel(), 71)
        weight.copy_(replacement)
        equal(weight, replacement)
        transition("kv_cache", False)
        kv.copy_(pattern(kv.numel(), 83))
        transition("acl_graph", True)
        check_persistent()
        transition("acl_graph", False)
        # Discarded graph output is undefined until replay/full overwrite.
        if graph is not None:
            graph.replay()
        else:
            graph_output.copy_(graph_expected)
        torch.npu.synchronize()
        equal(graph_output, graph_expected)
        check_persistent()
        report.update(
            full_persistent_values_equal=True,
            same_persistent_VA=True,
            workspace_capacity_bytes=workspace.capacity_bytes,
            workspace_frozen=workspace.frozen,
            actual_graph_replayed=graph is not None,
            weight_rewritten_only_after_remap=True,
        )
    finally:
        state.use_mem_pool = original_route
        global_server_args_dict.clear()
        global_server_args_dict.update(previous_args)
        # There are no child processes, RPCs, watchdog threads or signal cleanup.
        # Return each owned tag to active before resetting the owned graph.
        original_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        for tag in ("weights", "kv_cache", "acl_graph"):
            if tag in state.state.paused | state.state.failed:
                try:
                    adapter.resume(tag)
                except BaseException as error:
                    cleanup_errors.append(f"resume {tag}: {error!r}")
        try:
            if graph is not None:
                graph.reset()
            torch.npu.synchronize()
        except BaseException as error:
            cleanup_errors.append(f"graph/synchronize: {error!r}")
        report["cleanup"] = {
            "paused_tags": sorted(state.state.paused),
            "failed_tags": sorted(state.state.failed),
            "signals_sent": 0,
            "errors": cleanup_errors,
            "pool_lifetime": "Production owned PTA pools remain alive until normal process exit",
        }
        if cleanup_errors and not original_error:
            raise RuntimeError(f"Owned cleanup failed: {cleanup_errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--weight-policy", choices=("cpu_backup", "discard"), required=True
    )
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--expected-pta-git", required=True)
    parser.add_argument("--expected-library-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new output file")
    require(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal(),
        "Expose one leased physical NPU",
    )
    require(os.environ.get("TASK_QUEUE_ENABLE") == "1", "Set TASK_QUEUE_ENABLE=1")
    library = Path(os.environ["TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY"])
    require(
        library.is_absolute()
        and hashlib.sha256(library.read_bytes()).hexdigest()
        == args.expected_library_sha256,
        "Real saver library differs from the qualified artifact",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "failed",
        "weight_policy": args.weight_policy,
        "execution_mode": args.execution_mode,
        "transitions": [],
        "scope": "Standalone real owner/workspace buffer persistence; not model TP2 constant qualification",
    }
    try:
        import torch
        import torch_npu

        require(
            torch_npu.version.git_version == args.expected_pta_git,
            "PTA Git version differs",
        )
        report.update(
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            pta_git=torch_npu.version.git_version,
            library_sha256=args.expected_library_sha256,
        )
        with torch.inference_mode():
            run(torch, args, report)
        report["status"] = "passed"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
