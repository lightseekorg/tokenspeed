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

"""Verify a complete Ascend checkpoint replacement after weight discard."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from test_gsm8k_pause_resume import (
    ephemeral_loopback_address,
    make_server_args,
    read_trace,
    require,
    write_json,
)


def run(args, report):
    import tokenspeed_engine
    from transformers import AutoTokenizer

    from tokenspeed.runtime.engine.io_struct import (
        IsSleepingReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightFromDiskReqInput,
        UpdateWeightsFromMooncakeReqInput,
    )
    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver, generate_request

    host = Path(
        os.environ.get("TOKENSPEED_HOST_BINARY") or tokenspeed_engine.host_binary()
    ).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.initial, local_files_only=True)
    prompts = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in ("The capital of France is", "Two plus two equals")
    ]
    if args.verify_cache_flush:
        # A repeated prompt must publish at least one complete KV page.
        prefix = tokenizer.encode("A long prefix for cache reuse. " * 32)
        prompts[0] = prefix + prompts[0]
    sampling = {"temperature": 0, "ignore_eos": True, "max_new_tokens": 12}
    trace_path = args.output_dir / "graph-trace.jsonl"
    trace_dir = Path(__file__).resolve().parent / "trace"
    env = {
        "GSM8K_GRAPH_TRACE": str(trace_path),
        "GSM8K_TRACE_HOST_EXE": str(host),
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                (str(trace_dir), str(trace_dir.parent), os.environ.get("PYTHONPATH")),
            )
        ),
    }

    draft_policy = "refresh" if args.updated_draft is not None else "retain"
    sdk_directory = args.output_dir.resolve() / "sdk"
    version = 2
    snapshot = {"version": version} if args.snapshot or args.store_master else None
    sdk_config = {
        "directory": str(sdk_directory),
        "expect_snapshot": snapshot is not None,
    }
    ambiguous_checkpoint = args.output_dir.resolve() / "ambiguous-auto"
    if args.verify_disk_format_preflight:
        ambiguous_checkpoint.mkdir()
        for checkpoint_file in args.updated.iterdir():
            if checkpoint_file.name == "config.json" or checkpoint_file.name.endswith(
                (".safetensors", ".safetensors.index.json")
            ):
                (ambiguous_checkpoint / checkpoint_file.name).symlink_to(
                    checkpoint_file.resolve()
                )
        require(
            any(ambiguous_checkpoint.glob("*.safetensors")),
            "Format preflight requires a safetensors replacement checkpoint",
        )
        # This marker must be rejected as a second family before it is read.
        (ambiguous_checkpoint / "pytorch_model.bin").write_text(
            "Ambiguous checkpoint family marker; do not deserialize.\n"
        )
    publisher = None
    if args.mooncake_fixture:
        checkpoints = {"target": args.incomplete or args.updated}
        if args.updated_draft is not None:
            checkpoints["draft"] = args.updated_draft
        if args.store_master:
            from mooncake_sdk_fixture import open_store
            from safetensors import safe_open
            from safetensors.torch import save

            sdk_config["master"] = args.store_master
            publisher = open_store("127.0.0.1", args.store_master, 2 << 30)
            objects = {}
            for role, directory in checkpoints.items():
                keys = []
                with safe_open(
                    directory / "model.safetensors", framework="pt", device="cpu"
                ) as checkpoint:
                    for index, name in enumerate(checkpoint.keys()):
                        key = f"weights-fixture-{os.getpid()}-{version}-{role}-{index}"
                        result = publisher.put(
                            key, save({name: checkpoint.get_tensor(name)})
                        )
                        require(
                            result == 0,
                            f"Failed to publish fixture weight {name}: {result}",
                        )
                        keys.append(key)
                objects[role] = keys
            snapshot["fixture_objects"] = objects
        else:
            for role, directory in checkpoints.items():
                link = sdk_directory / f"version-{version}" / role
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(directory.resolve(), target_is_directory=True)
        report["transport"] = (
            "mooncake-tcp-fixture" if publisher is not None else "file-sdk-fixture"
        )
        report["draft_policy"] = draft_policy

    def launch(model, policy, trace=False):
        launch_args = make_server_args(
            SimpleNamespace(
                model=model,
                tp_size=1,
                concurrency=2,
                dist_init_addr=ephemeral_loopback_address(),
                seed=17,
            )
        )
        launch_args[launch_args.index("--weight-memory-policy") + 1] = policy
        if args.draft is not None:
            draft = (
                (args.updated_draft or args.draft)
                if policy == "reference"
                else args.draft
            )
            launch_args.extend(
                (
                    "--speculative-algorithm",
                    "MTP",
                    "--speculative-draft-model-path",
                    str(draft),
                    "--speculative-num-steps",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                )
            )
            if policy == "discard" and not args.mooncake_fixture:
                launch_args.extend(("--model-update-draft-weights", draft_policy))
        if args.mooncake_fixture and policy != "reference":
            launch_args.extend(
                (
                    "--model-update-config",
                    json.dumps(
                        {
                            "weight_store_config": sdk_config,
                            "local_host": "127.0.0.1",
                            "hf_type": "qwen3",
                            "hf_safetensors_path": str(args.initial),
                        }
                    ),
                    "--model-update-sdk-module",
                    "mooncake_sdk_fixture",
                    "--model-update-engine-type",
                    "fluent_llm",
                    "--model-update-draft-weights",
                    draft_policy,
                )
            )
        if args.verify_namespace or args.verify_cache_flush:
            launch_args.remove("--disable-kvstore")
            launch_args.remove("--disable-prefix-caching")
            launch_args.extend(
                (
                    "--kvstore-storage-backend",
                    "memory",
                    "--weight-version",
                    "fixture-initial",
                )
            )
        if args.eager:
            launch_args.append("--enforce-eager")
        if policy == "reference":
            launch_args.remove("--enable-memory-saver")
            index = launch_args.index("--weight-memory-policy")
            del launch_args[index : index + 2]
        return WireDriver(
            launch_args,
            num_engines=1,
            routing="round_robin",
            startup_timeout_s=150,
            env=(
                env
                if trace or (args.mooncake_fixture and policy != "reference")
                else None
            ),
        )

    def infer(driver):
        outputs = driver.generate([generate_request(ids, sampling) for ids in prompts])
        require(
            all(out.finish_reason == "length" for out in outputs),
            f"Unexpected finish reasons: {[out.finish_details for out in outputs]}",
        )
        return [
            {
                "ids": out.output_ids,
                "cached_tokens": out.cached_tokens,
                "verify_count": out.spec_verify_ct,
                "accept_length": out.accept_draft_tokens,
            }
            for out in outputs
        ]

    def control(driver, request):
        replies = driver.control(request)
        require(len(replies) == 1 and replies[0].success, f"{request}: {replies}")

    def token_ids(outputs):
        return [out["ids"] for out in outputs]

    def weight_update(*, flush_cache=True):
        if args.mooncake_fixture:
            return UpdateWeightsFromMooncakeReqInput(
                version=version,
                flush_cache=flush_cache,
                weight_version=None,
                snapshot=snapshot,
            )
        return UpdateWeightFromDiskReqInput(
            model_path=str(args.incomplete or args.updated),
            load_format="auto" if args.verify_disk_format_preflight else None,
            flush_cache=flush_cache,
            weight_version=(
                str(version) if args.verify_cache_flush and flush_cache else None
            ),
        )

    if args.incomplete:
        with launch(args.initial, "discard") as driver:
            control(
                driver, ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            )
            control(driver, ResumeMemoryOccupationReqInput(tags=["weights"]))
            replies = driver.control(weight_update())
            require(
                len(replies) == 1 and not replies[0].success,
                f"Incomplete update: {replies}",
            )
            report["refusal"] = replies[0].message
            sleeping = driver.control(IsSleepingReqInput())
            require(
                len(sleeping) == 1 and sleeping[0].is_sleeping,
                "Engine must stay asleep after refusing the incomplete update",
            )
            if args.mooncake_fixture:
                require(
                    "missing checkpoint weights" in replies[0].message, str(replies)
                )
                if args.verify_namespace:
                    for namespace, expected in (
                        ("fixture-initial", "restart"),
                        (str(version), "cannot change without flush_cache"),
                    ):
                        replies = driver.control(
                            UpdateWeightsFromMooncakeReqInput(
                                version=version,
                                flush_cache=False,
                                weight_version=namespace,
                            )
                        )
                        require(
                            len(replies) == 1
                            and not replies[0].success
                            and expected in replies[0].message,
                            f"Unexpected namespace after failure: {replies}",
                        )
                    report["namespace_after_failure"] = "fixture-initial"
                replies = driver.control(
                    ResumeMemoryOccupationReqInput(tags=["kv_cache"])
                )
                require(
                    len(replies) == 1
                    and not replies[0].success
                    and "restart" in replies[0].message,
                    f"KV wake must fail after a partial stream: {replies}",
                )
                report["kv_wake_refusal"] = replies[0].message
            else:
                require(
                    "Incomplete checkpoint" in replies[0].message
                    and "nothing was written" in replies[0].message,
                    f"Incomplete checkpoint was not refused before writes: {replies}",
                )
        return

    with launch(args.updated, "reference") as reference:
        report["fresh"] = infer(reference)

    graph_tags = [] if args.eager else ["acl_graph"]
    with launch(args.initial, "discard", trace=True) as driver:
        report["before"] = infer(driver)
        if args.verify_cache_flush:
            # With unchanged weights a KV-only cycle removes L1 and lets
            # the first resumed request prove the prefix survived off-device.
            control(driver, ReleaseMemoryOccupationReqInput(tags=["kv_cache"]))
            control(driver, ResumeMemoryOccupationReqInput(tags=["kv_cache"]))
            report["retained_cache_tokens"] = infer(driver)[0]["cached_tokens"]
            require(
                report["retained_cache_tokens"] > 0,
                "No retained prefix after KV-only wake",
            )
        control(
            driver,
            ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights", *graph_tags]),
        )
        control(
            driver,
            ResumeMemoryOccupationReqInput(tags=[*graph_tags, "weights"]),
        )
        if args.verify_cache_flush:
            replies = driver.control(weight_update(flush_cache=False))
            require(
                len(replies) == 1
                and not replies[0].success
                and "require flush_cache=true" in replies[0].message,
                f"Replacement without cache flush must fail: {replies}",
            )
            report["cache_flush_refusal"] = replies[0].message
        if args.verify_disk_format_preflight:
            report["disk_format_refusals"] = []
            for model_path, load_format, expected in (
                (args.updated, "npcache", "load_format 'npcache' is unsupported"),
                (
                    ambiguous_checkpoint,
                    "auto",
                    "load_format 'auto' is ambiguous",
                ),
            ):
                replies = driver.control(
                    UpdateWeightFromDiskReqInput(
                        model_path=str(model_path),
                        load_format=load_format,
                        flush_cache=True,
                        weight_version=str(version),
                    )
                )
                require(
                    len(replies) == 1
                    and not replies[0].success
                    and expected in replies[0].message
                    and "nothing was written" in replies[0].message,
                    f"Unexpected disk format preflight response: {replies}",
                )
                report["disk_format_refusals"].append(
                    {
                        "load_format": load_format,
                        "weight_version": str(version),
                        "message": replies[0].message,
                    }
                )
                sleeping = driver.control(IsSleepingReqInput())
                require(
                    len(sleeping) == 1 and sleeping[0].is_sleeping,
                    "KV must remain asleep after a disk format refusal",
                )
        control(driver, weight_update())
        if args.verify_disk_format_preflight:
            report["disk_format_retry_weight_version"] = str(version)
        control(driver, ResumeMemoryOccupationReqInput(tags=["kv_cache"]))
        report["after"] = infer(driver)
        if args.verify_cache_flush:
            require(
                all(row["cached_tokens"] == 0 for row in report["after"]),
                "Replacement reused a prefix from the previous weights",
            )

    events = read_trace(trace_path)
    transitions = [
        event
        for event in events
        if event.get("tag") == "weights" and event["event"] in ("pause", "resume")
    ]
    require(len(transitions) == 2, "Expected a weights pause and resume")
    allocations = []
    for event in transitions:
        before, after = (
            {
                block["ptr"]: block
                for block in event[state]["blocks"]
                if block["tag"] == "weights"
            }
            for state in ("before", "after")
        )
        require(
            before and before.keys() == after.keys(), "Weight virtual addresses changed"
        )
        paused = event["event"] == "pause"
        require(
            all(
                block["mapped"] == (not paused)
                and block["backup_valid"] == (paused and block["policy"] == "backup")
                for block in after.values()
            ),
            "Incorrect weight mapping or Host snapshot",
        )
        allocations = list(after.values())
    report["weight_virtual_addresses_unchanged"] = True
    policies = {block["policy"] for block in allocations}
    require(
        policies
        == (
            {"discard", "backup"}
            if args.draft and draft_policy == "retain"
            else {"discard"}
        ),
        f"Unexpected weight policies {policies}",
    )
    report["weight_bytes"] = {
        policy: sum(
            block["aligned_bytes"] for block in allocations if block["policy"] == policy
        )
        for policy in sorted(policies)
    }
    if not args.eager:
        replays = {
            phase: {
                (event["pid"], event["graph"])
                for event in events
                if event["event"] == "first_replay" and event["phase"] == phase
            }
            for phase in ("before", "after")
        }
        require(
            replays["before"] & replays["after"],
            "No original graph replayed after wake",
        )
        require(
            not any(
                event["event"] == "capture_begin" and event["phase"] == "after"
                for event in events
            ),
            "Unexpected recapture after wake",
        )
        report["original_graph_replayed"] = True
        report["recaptured_after_wake"] = False
    require(
        token_ids(report["before"]) != token_ids(report["fresh"]),
        "Checkpoint B does not change the probe outputs",
    )
    require(
        token_ids(report["after"]) == token_ids(report["fresh"]),
        "Reload differs from fresh checkpoint B",
    )
    if args.draft:
        require(
            all(
                out["verify_count"]
                for phase in ("before", "after", "fresh")
                for out in report[phase]
            ),
            "No speculative verify rounds recorded",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument(
        "--updated-draft",
        type=Path,
        help="Refresh an opted-in draft through the SDK fixture",
    )
    parser.add_argument(
        "--mooncake-fixture",
        action="store_true",
        help="Use the file-backed Model Updater fixture",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Send snapshot metadata to the SDK fixture",
    )
    parser.add_argument(
        "--store-master", help="Use an existing local Mooncake master for the fixture"
    )
    parser.add_argument(
        "--verify-namespace",
        action="store_true",
        help="Verify a failed SDK update leaves the in-memory L3 namespace unchanged",
    )
    parser.add_argument(
        "--incomplete", type=Path, help="Expect an incomplete checkpoint refusal"
    )
    parser.add_argument(
        "--verify-cache-flush",
        action="store_true",
        help="Warm Host prefix caches and check their invalidation on replacement",
    )
    parser.add_argument(
        "--verify-disk-format-preflight",
        action="store_true",
        help="Refuse unsupported and ambiguous disk formats before a valid retry",
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.updated_draft or args.snapshot or args.store_master
    ) and not args.mooncake_fixture:
        parser.error(
            "--updated-draft, --snapshot and --store-master require --mooncake-fixture"
        )
    if args.updated_draft and not args.draft:
        parser.error("--updated-draft requires --draft")
    if args.verify_namespace and not (args.mooncake_fixture and args.incomplete):
        parser.error("--verify-namespace requires --mooncake-fixture and --incomplete")
    if args.verify_cache_flush and args.incomplete:
        parser.error("--verify-cache-flush requires a complete replacement checkpoint")
    if args.verify_disk_format_preflight:
        if args.mooncake_fixture or args.incomplete:
            parser.error(
                "--verify-disk-format-preflight requires a complete disk replacement"
            )
        if not args.verify_cache_flush:
            parser.error("--verify-disk-format-preflight requires --verify-cache-flush")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed"}
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
