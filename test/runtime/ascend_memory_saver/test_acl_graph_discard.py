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


"""Check serving ACL Graph replay after overwriting the remapped private pool."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from test_gsm8k_pause_resume import (
    ephemeral_loopback_address,
    make_server_args,
    read_trace,
    require,
    write_json,
)


@dataclass(frozen=True)
class ServerConfig:
    model: Path
    dist_init_addr: str
    tp_size: int = 1
    concurrency: int = 4
    seed: int = 17


def launch(args, *, trace_env):
    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

    flags = make_server_args(ServerConfig(args.model, ephemeral_loopback_address()))
    flags[flags.index("--sampling-backend") + 1] = args.sampler
    flags[flags.index("--cudagraph-capture-sizes") + 1 :] = ["1", "4"]
    if args.drafter:
        flags += [
            "--speculative-algorithm",
            args.drafter,
            "--speculative-num-steps",
            "3",
            "--speculative-num-draft-tokens",
            "4",
        ]
    if args.draft_checkpoint:
        flags += ["--speculative-draft-model-path", str(args.draft_checkpoint)]
    if trace_env is None:
        flags.remove("--enable-memory-saver")
        index = flags.index("--weight-memory-policy")
        del flags[index : index + 2]
    return WireDriver(
        flags,
        num_engines=1,
        routing="round_robin",
        startup_timeout_s=150,
        env=trace_env,
    )


def infer(driver, prompts, sizes, *, expect_verification):
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    sampling = {"temperature": 0, "ignore_eos": True, "max_new_tokens": 16}
    results = {}
    for size in sizes:
        outputs = driver.generate(
            [generate_request(prompt, sampling) for prompt in prompts[:size]]
        )
        require(len(outputs) == size, f"Expected {size} outputs")
        results[size] = []
        for output in outputs:
            require(
                output.finish_reason == "length" and len(output.output_ids) == 16,
                f"Incomplete generation: {output}",
            )
            if expect_verification:
                require(output.spec_verify_ct, f"Missing verification: {output}")
            results[size].append(
                {
                    "ids": output.output_ids,
                    "verify_count": output.spec_verify_ct,
                    "accept_length": output.accept_draft_tokens,
                }
            )
    return results


def graph_id(row):
    return row["pid"], row["graph"], tuple(row["pool"])


def graph_blocks(stats, *, mapped, policy):
    blocks = {}
    for block in stats["blocks"]:
        if block["tag"] == "acl_graph":
            require(
                block["reserved"]
                and block["policy"] == policy
                and block["physical_live"] == block["mapped"] == mapped
                and block["backup_valid"] == (not mapped and policy == "backup"),
                f"Unexpected graph allocation state: {block}",
            )
            blocks[block["ptr"]] = block["aligned_bytes"]
    require(blocks, "No private pool allocations")
    require(not stats["callback_fault"], "Allocator callback failed")
    snapshot_bytes = sum(blocks.values()) if not mapped and policy == "backup" else 0
    require(
        stats["host_bytes"] == stats["valid_backup_bytes"] == snapshot_bytes,
        f"Expected {snapshot_bytes} graph snapshot bytes",
    )
    return blocks


def run(args, report):
    import tokenspeed_engine
    import torch
    import torch_npu
    from transformers import AutoTokenizer

    from tokenspeed.runtime.engine.io_struct import (
        IsSleepingReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )

    require(torch.npu.device_count() == 1, "Expose exactly one NPU")
    report["pta_commit"] = torch_npu.version.git_version
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in (
            "The capital of France is",
            "Answer the question. " * 22 + "Two plus two equals",
            "Here is some context. " * 29 + "The largest ocean is",
            "A red apple and a green apple are both",
        )
    ]
    sizes = (1, 3, 4)
    trace_path = args.output_dir / "graph-trace.jsonl"
    trace_dir = Path(__file__).resolve().parent / "trace"
    policy = "backup" if args.expect_backup else "discard"
    trace_env = {
        "GSM8K_GRAPH_TRACE": str(trace_path),
        "GSM8K_TRACE_HOST_EXE": str(
            Path(
                os.environ.get("TOKENSPEED_HOST_BINARY")
                or tokenspeed_engine.host_binary()
            ).resolve(strict=True)
        ),
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(trace_dir), os.environ.get("PYTHONPATH")))
        ),
        "TOKENSPEED_GRAPH_DEBUG": "1",
        "TOKENSPEED_TEST_ACL_GRAPH_POLICY": policy,
    }

    def reply(driver, req):
        results = driver.control(req)
        require(len(results) == 1, f"Expected one reply: {results}")
        return results[0]

    with launch(args, trace_env=None) as reference:
        expected = infer(
            reference, prompts, sizes, expect_verification=bool(args.drafter)
        )
    report["fresh"] = expected
    with launch(args, trace_env=trace_env) as driver:
        before = infer(driver, prompts, sizes, expect_verification=bool(args.drafter))
        require(before == expected, "Serving baseline differs from fresh outputs")
        startup = read_trace(trace_path)
        captures = [row for row in startup if row["event"] == "capture_end"]
        graph_ids = {graph_id(row) for row in captures}
        require(
            len(captures) == len(graph_ids) == 2
            and len({(pid, pool) for pid, _, pool in graph_ids}) == 1
            and all(row["fia_tasks"] > 0 for row in captures)
            and {graph_id(row) for row in startup if row["event"] == "first_replay"}
            == graph_ids,
            f"Expected two auto-dispatched startup graphs in one pool: {captures}",
        )
        report["graphs"] = captures
        offset = len(startup)
        addresses = None
        for cycle in range(1, args.cycles + 1):
            for req, sleeping in (
                (ReleaseMemoryOccupationReqInput(tags=["acl_graph"]), True),
                (ResumeMemoryOccupationReqInput(tags=["acl_graph"]), True),
                (ResumeMemoryOccupationReqInput(tags=["kv_cache"]), False),
            ):
                result = reply(driver, req)
                require(result.success, f"Graph transition failed: {result}")
                result = reply(driver, IsSleepingReqInput())
                require(
                    result.is_sleeping == sleeping, f"Unexpected sleep state: {result}"
                )
            order = sizes if cycle % 2 else sizes[::-1]
            actual = infer(
                driver, prompts, order, expect_verification=bool(args.drafter)
            )
            require(
                actual == expected,
                f"Cycle {cycle}: output differs from fresh: {actual}",
            )
            rows = read_trace(trace_path)[offset:]
            offset += len(rows)
            kv_rows = [row for row in rows if row.get("tag") == "kv_cache"]
            require(
                [row["event"] for row in kv_rows] == ["pause", "resume"],
                f"Cycle {cycle}: graph suspend must discard and rebuild device KV",
            )
            # The graph assertions below inspect its independent private pool.
            rows = [row for row in rows if row.get("tag") != "kv_cache"]
            require(
                not any(
                    row["event"] in ("capture_begin", "capture_end", "reset")
                    for row in rows
                ),
                f"Cycle {cycle}: graph capture or reset after startup",
            )
            graph_events = ["first_replay", "first_replay"]
            if policy == "discard":
                graph_events = ["graph_pool_poison", "first_replay"] * 2
            require(
                [row["event"] for row in rows] == ["pause", "resume", *graph_events]
                and all(row["phase"] == f"cycle-{cycle}" for row in rows[2:]),
                f"Cycle {cycle}: unexpected transition or replay trace",
            )
            pause, resume, *graph_rows = rows
            replays = [row for row in graph_rows if row["event"] == "first_replay"]
            poisons = [row for row in graph_rows if row["event"] == "graph_pool_poison"]
            require(
                {graph_id(row) for row in replays} == graph_ids,
                f"Cycle {cycle}: expected both startup graphs",
            )
            layout = graph_blocks(pause["before"], mapped=True, policy=policy)
            if addresses is None:
                addresses = layout
            require(
                layout == addresses
                and graph_blocks(pause["after"], mapped=False, policy=policy)
                == addresses
                and graph_blocks(resume["before"], mapped=False, policy=policy)
                == addresses
                and graph_blocks(resume["after"], mapped=True, policy=policy)
                == addresses,
                f"Cycle {cycle}: graph addresses changed or poison did not cover the pool",
            )
            if poisons:
                require(
                    all(row["blocks"] == len(addresses) for row in poisons)
                    and poisons[-1]["graph"] == replays[-1]["graph"],
                    f"Cycle {cycle}: poison did not cover the pool before both buckets",
                )
            graph_bytes = sum(addresses.values())
            require(
                pause["before"]["physical_bytes"] - pause["after"]["physical_bytes"]
                == resume["after"]["physical_bytes"]
                - resume["before"]["physical_bytes"]
                == graph_bytes,
                f"Cycle {cycle}: unexpected physical memory delta",
            )
            report["cycles"].append(
                {
                    "cycle": cycle,
                    "order": order,
                    "poison_byte": poisons[0]["byte"] if poisons else None,
                    "graph_bytes": graph_bytes,
                    "graph_policy": policy,
                    "replays": [graph_id(row) for row in replays],
                    "outputs": actual,
                }
            )
    require(
        driver.process.returncode == 0, f"Launcher exit: {driver.process.returncode}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument(
        "--drafter", choices=("MTP", "EAGLE3", "DFLASH", "DSPARK", "DUMMY")
    )
    parser.add_argument("--draft-checkpoint", type=Path)
    parser.add_argument("--sampler", choices=("greedy", "ascend_full"), required=True)
    parser.add_argument("--expect-backup", action="store_true")
    args = parser.parse_args()
    if args.cycles < 2:
        parser.error("--cycles must be at least 2")
    if (args.drafter not in (None, "DUMMY")) != (args.draft_checkpoint is not None):
        parser.error("Supply a draft checkpoint for a model-backed drafter")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "cycles": []}
    try:
        run(args, report)
        report["status"] = "passed"
    finally:
        write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
