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

"""Verify TP2 weight replacement, two local refusals and replica commit ordering."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from test_gsm8k_pause_resume import (
    ephemeral_loopback_address,
    make_server_args,
    read_trace,
    require,
    verify_graph_trace,
    write_json,
)
from transformers import AutoTokenizer

from tokenspeed.runtime.engine.io_struct import (
    IsSleepingReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
)
from tokenspeed.runtime.entrypoints.wire_driver import WireDriver, generate_request

TAGS = ["kv_cache", "weights", "acl_graph"]
Q = "model.layers.0.self_attn.q_proj.weight"


def prepare_binary_inputs(initial, updated, directory):
    """Create isolated binary fixtures from A/B without modifying either model."""
    good = directory / "good-bin"
    bad = directory / "oversized-bin"
    stale = directory / "stale-npcache"
    for path in (good, bad, stale):
        path.mkdir()
    tensors = load_file(updated / "model.safetensors", device="cpu")
    torch.save(tensors, good / "pytorch_model.bin")
    original_shape = tuple(tensors[Q].shape)
    tensors[Q] = torch.cat((tensors[Q], tensors[Q][:8]))
    oversized_shape = tuple(tensors[Q].shape)
    torch.save(tensors, bad / "pytorch_model.bin")
    del tensors
    gc.collect()

    # NPCACHE uses a name manifest and per-tensor .npy payloads. Float32 arrays
    # represent the BF16 A values exactly and can be read by NumPy.
    arrays = stale / "np"
    arrays.mkdir()
    with safe_open(initial / "model.safetensors", framework="pt", device="cpu") as f:
        names = f.keys()
        for name in names:
            with (arrays / name).open("wb") as out:
                np.save(out, f.get_tensor(name).float().numpy())
    (arrays / "weight_names.json").write_text(json.dumps(names))
    (stale / "pytorch_model.bin").symlink_to(good / "pytorch_model.bin")
    # Prove the old complete cache and the replacement checkpoint differ.
    changed = []
    with (
        safe_open(initial / "model.safetensors", framework="pt", device="cpu") as a,
        safe_open(updated / "model.safetensors", framework="pt", device="cpu") as b,
    ):
        require(set(a.keys()) == set(b.keys()), "A/B checkpoint inventories differ")
        for name in b.keys():
            if not torch.equal(a.get_tensor(name), b.get_tensor(name)):
                changed.append(name)
    require(changed, "A and B have identical tensors")
    with (arrays / changed[0]).open("rb") as cached:
        with safe_open(
            updated / "model.safetensors", framework="pt", device="cpu"
        ) as b:
            require(
                not torch.equal(
                    torch.from_numpy(np.load(cached)), b.get_tensor(changed[0]).float()
                ),
                "The seeded NumPy cache unexpectedly contains B values",
            )
    return (
        good,
        bad,
        stale,
        {
            "a_b_different_tensors": changed,
            "oversized_source": Q,
            "valid_shape": original_shape,
            "oversized_shape": oversized_shape,
            "stale_cache_names": len(names),
        },
    )


def verify_eager_trace(startup, before, after, tp_size):
    """Eager requires no graphs and the same real weight/KV mapping invariants."""
    from collections import Counter

    require(
        not any(
            row["event"] in {"capture_begin", "capture_end", "reset", "first_replay"}
            for row in after
        ),
        "Eager unexpectedly captured/reset/replayed a graph",
    )
    all_transitions = [row for row in after if row["event"] in {"pause", "resume"}]
    transitions = [
        row for row in all_transitions if row["tag"] in {"weights", "kv_cache"}
    ]
    pids = {row["pid"] for row in transitions}
    require(len(pids) == tp_size, "Eager missing real rank transitions")
    require(
        Counter((row["pid"], row["event"], row["tag"]) for row in transitions)
        == Counter(
            (pid, op, tag)
            for pid in pids
            for op in ("pause", "resume")
            for tag in ("weights", "kv_cache")
        ),
        "Eager missing/duplicate rank transitions",
    )
    for row in all_transitions:
        tag = row["tag"]
        require(
            tag in {"weights", "kv_cache", "acl_graph"}, "Unexpected eager mapping tag"
        )
        old = {b["ptr"]: b for b in row["before"]["blocks"] if b["tag"] == tag}
        new = {b["ptr"]: b for b in row["after"]["blocks"] if b["tag"] == tag}
        require(
            not row["before"]["callback_fault"] and not row["after"]["callback_fault"],
            "Native saver callback fault",
        )
        if tag == "acl_graph":
            require(not old and not new, "Eager allocated an ACL graph pool")
            continue
        require(
            old and old.keys() == new.keys(),
            f"{tag}: VA changed or tag has no allocations",
        )
        expected_policies = {"discard", "backup"} if tag == "weights" else {"discard"}
        require(
            {b["policy"] for b in new.values()} == expected_policies,
            "Unexpected classified eager mapping policies",
        )
        pausing = row["event"] == "pause"
        for address, b in old.items():
            a = new[address]
            require(
                b["reserved"]
                and a["reserved"]
                and b["policy"] == a["policy"]
                and b["aligned_bytes"] == a["aligned_bytes"]
                and b["physical_live"] == b["mapped"] == pausing
                and a["physical_live"] == a["mapped"] == (not pausing)
                and a["backup_valid"] == (pausing and a["policy"] == "backup"),
                "Invalid eager saver transition",
            )
    return {
        "ranks": sorted(pids),
        "graph_lifecycle_or_replay_count": 0,
        "mapping_transitions": len(transitions),
        "policy": {
            "model_parameters": "discard",
            "persistent_buffers": "backup",
            "kv_cache": "discard",
        },
        "virtual_reservations_preserved": True,
    }


def main():
    from test_timeline import install

    install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    args = parser.parse_args()
    args.output_dir.mkdir()
    report = {"status": "failed", "tp_size": 2, "execution_mode": args.execution_mode}
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        write_json(args.output_dir / "report.json", report)


def run(args, report):
    """Run fresh B and independent positive and negative TP2 engine sessions."""
    require(torch.npu.device_count() == 2, "Expected two visible Ascend devices")
    tokenizer = AutoTokenizer.from_pretrained(args.initial, local_files_only=True)
    prompts = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in ("The capital of France is", "Two plus two equals")
    ]
    sampling = {"temperature": 0, "ignore_eos": True, "max_new_tokens": 12}
    fixtures = args.output_dir / "fixtures"
    fixtures.mkdir()
    good, bad, stale, report["fixtures"] = prepare_binary_inputs(
        args.initial, args.updated, fixtures
    )

    def launch(model, trace=None, *, rank1_bad=None, use_l3=False):
        launch_args = make_server_args(
            SimpleNamespace(
                model=model,
                tp_size=2,
                concurrency=2,
                seed=17,
                dist_init_addr=ephemeral_loopback_address(),
            )
        )
        launch_args[launch_args.index("--weight-memory-policy") + 1] = "discard"
        if args.execution_mode == "eager":
            launch_args.append("--enforce-eager")
        if use_l3:
            launch_args.remove("--disable-kvstore")
            launch_args.remove("--disable-prefix-caching")
            launch_args.extend(
                (
                    "--kvstore-storage-backend",
                    "memory",
                    "--weight-version",
                    "fixture-A",
                )
            )
        env = None
        if trace is not None:
            env = {
                "GSM8K_GRAPH_TRACE": str(trace),
                "GSM8K_TRACE_HOST_EXE": str(
                    Path(os.environ["TOKENSPEED_HOST_BINARY"]).resolve()
                ),
                "PYTHONPATH": os.pathsep.join(
                    (str(args.trace_dir), os.environ["PYTHONPATH"])
                ),
            }
            if rank1_bad is not None:
                env["TS_WEIGHT_BAD_RANK1"] = str(rank1_bad)
        return WireDriver(
            launch_args,
            num_engines=1,
            routing="round_robin",
            startup_timeout_s=180,
            env=env,
        )

    def infer(driver):
        outputs = driver.generate([generate_request(ids, sampling) for ids in prompts])
        require(
            all(out.finish_reason == "length" for out in outputs),
            f"Unexpected finish reasons: {[out.finish_details for out in outputs]}",
        )
        return [out.output_ids for out in outputs]

    def control(driver, request, *, failure=None):
        replies = driver.control(request)
        require(len(replies) == 1, f"Expected one reply to {request}: {replies}")
        reply = replies[0]
        if failure is None:
            require(reply.success, f"Failed {request}: {reply}")
        else:
            require(
                not reply.success and failure in reply.message,
                f"Expected refusal containing {failure!r}: {reply}",
            )
        return reply.message

    def suspend(driver):
        control(driver, ReleaseMemoryOccupationReqInput(tags=TAGS))
        control(driver, ResumeMemoryOccupationReqInput(tags=["acl_graph", "weights"]))

    def refuse_kv_wake(driver, trace, message):
        before = read_trace(trace)
        refusal = control(
            driver, ResumeMemoryOccupationReqInput(tags=["kv_cache"]), failure=message
        )
        require(read_trace(trace) == before, "Refused KV wake changed the trace")
        (reply,) = driver.control(IsSleepingReqInput())
        require(reply.is_sleeping, "Refused KV wake changed sleeping state")
        return refusal

    def disk(path, fmt=None, *, version=None, flush=True):
        return UpdateWeightFromDiskReqInput(
            model_path=str(path),
            load_format=fmt,
            weight_version=version,
            flush_cache=flush,
        )

    def weights_trace(path, success):
        events = read_trace(path)
        entries = {x["rank"]: x for x in events if x["event"] == "disk_update_enter"}
        exits = {x["rank"]: x for x in events if x["event"] == "disk_update_exit"}
        require(
            set(entries) == set(exits) == {0, 1},
            "Both ranks must enter and finish reload",
        )
        require(
            sum(x["event"] == "disk_update_enter" for x in events)
            == sum(x["event"] == "disk_update_exit" for x in events)
            == 2,
            "Expected one write attempt per rank",
        )
        for rank in (0, 1):
            require(exits[rank]["success"] is success, f"Unexpected rank {rank} result")
            require(
                entries[rank]["parameter_va"] == exits[rank]["parameter_va"],
                f"Rank {rank} changed parameter addresses",
            )
        starts = [x for x in events if x["event"] == "begin_weight_update"]
        ends = [x for x in events if x["event"] == "end_weight_update"]
        expected = {x["pid"] for x in exits.values()} if success else set()
        require(
            {x["pid"] for x in starts} == {x["pid"] for x in ends} == expected,
            "Unexpected session or post-load entry",
        )
        return {
            "ranks": exits,
            "begin_count": len(starts),
            "post_load_entry_count": len(ends),
            "parameter_va_equal": True,
        }

    with launch(args.updated) as fresh:
        report["fresh_b"] = infer(fresh)
    trace = args.output_dir / "legal.trace.jsonl"
    with launch(args.initial, trace) as driver:
        startup = read_trace(trace)
        report["before_a"] = infer(driver)
        before = read_trace(trace)
        suspend(driver)
        report["early_kv_refusal"] = refuse_kv_wake(
            driver, trace, "complete weight update"
        )
        report["legal_commit"] = control(driver, disk(good, version="fixture-B"))
        control(driver, ResumeMemoryOccupationReqInput(tags=["kv_cache"]))
        report["after_b"] = infer(driver)
    require(
        report["before_a"] != report["fresh_b"], "Probe outputs cannot distinguish A/B"
    )
    require(
        report["after_b"] == report["fresh_b"], "Updated output differs from fresh B"
    )
    from test_persistent_model_ownership import validate_trace_ownership

    report["model_pool_ownership"] = validate_trace_ownership(
        read_trace(trace), tp_size=2
    )
    if args.execution_mode == "graph":
        report["graph"] = verify_graph_trace(
            startup,
            before,
            read_trace(trace),
            TAGS,
            2,
            graph_policy="discard",
            weight_policies=("discard", "backup"),
        )
    else:
        report["eager_mappings"] = verify_eager_trace(
            startup, before, read_trace(trace), 2
        )
    report["legal_weight_trace"] = weights_trace(trace, True)
    write_json(args.output_dir / "partial.json", report)

    # Admission refuses NPCACHE before flush or transaction; same-version AUTO retry is healthy.
    trace = args.output_dir / "npcache.trace.jsonl"
    with launch(args.initial, trace, use_l3=True) as driver:
        suspend(driver)
        before_refusal = read_trace(trace)
        refusal = control(
            driver, disk(stale, "npcache", version="fixture-B"), failure="npcache"
        )
        require(
            read_trace(trace) == before_refusal,
            "NPCACHE preflight reached parameter writes or mappings",
        )
        require(
            driver.control(IsSleepingReqInput())[0].is_sleeping,
            "NPCACHE refusal woke KV",
        )
        report["npcache"] = {
            "refusal": refusal,
            "healthy_same_version_retry": control(
                driver, disk(good, "auto", version="fixture-B")
            ),
        }
        control(driver, ResumeMemoryOccupationReqInput(tags=["kv_cache"]))
        report["npcache"]["after_b"] = infer(driver)
        require(
            report["npcache"]["after_b"] == report["fresh_b"],
            "NPCACHE corrected retry differs from fresh B",
        )
    report["npcache"]["weights"] = weights_trace(trace, True)
    write_json(args.output_dir / "partial.json", report)

    for name, request, rank1_bad, expected in (
        ("oversized", disk(good, version="fixture-bad"), bad, "before TP slicing"),
    ):
        trace = args.output_dir / f"{name}.trace.jsonl"
        with launch(args.initial, trace, rank1_bad=rank1_bad, use_l3=True) as driver:
            suspend(driver)
            refusal = control(driver, request, failure=expected)
            wake = refuse_kv_wake(driver, trace, "restart")
            old = control(
                driver, disk(good, version="fixture-A", flush=False), failure="restart"
            )
            failed = control(
                driver,
                disk(good, version="fixture-bad", flush=False),
                failure="cannot change without flush_cache",
            )
        report[name] = {
            "refusal": refusal,
            "wake_refusal": wake,
            "old_namespace_probe": old,
            "failed_namespace_probe": failed,
            "weights": weights_trace(trace, False),
        }
        if name == "oversized":
            rank0 = report[name]["weights"]["ranks"][0]
            require(
                "group rank 1" in rank0["message"],
                "Healthy rank missed the local refusal",
            )
        write_json(args.output_dir / "partial.json", report)


if __name__ == "__main__":
    main()
