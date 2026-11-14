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

"""Rerun selected GSM8K questions with A/A or a graph-only pause/resume."""

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

from test_gsm8k_pause_resume import (
    DATASET_SHA256,
    compare,
    encode_prompt,
    ephemeral_loopback_address,
    infer,
    load_dataset,
    make_server_args,
    pta_git_version,
    read_trace,
    require,
    summarize,
    verify_graph_trace,
    write_json,
    write_rows,
)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sample_evenly(ids, count):
    """Pick the midpoint of each equal-size segment of a sorted stratum."""
    return [ids[(2 * i + 1) * len(ids) // (2 * count)] for i in range(count)]


def sample_stratified(strata, count):
    """Balance the quotas, then redistribute unused quota from small strata."""
    quota = [0] * len(strata)
    remaining = min(count, sum(map(len, strata)))
    while remaining:
        for i, ids in enumerate(strata):
            if quota[i] < len(ids):
                quota[i] += 1
                remaining -= 1
                if remaining == 0:
                    break
    return [
        index
        for ids, size in zip(strata, quota, strict=True)
        for index in sample_evenly(ids, size)
    ]


def select_cases(original, selection, target_count, correctness_cap=None):
    before = {row["index"]: row for row in load_jsonl(original / "before.jsonl")}
    after = {row["index"]: row for row in load_jsonl(original / "after.jsonl")}
    require(
        before and before.keys() == after.keys(), "Original run lacks paired questions"
    )
    changed = sorted(
        i for i, a in before.items() if a["output_ids"] != after[i]["output_ids"]
    )
    correct_to_wrong = [
        i for i in changed if before[i]["correct"] and not after[i]["correct"]
    ]
    wrong_to_correct = [
        i for i in changed if not before[i]["correct"] and after[i]["correct"]
    ]
    correctness = sorted(correct_to_wrong + wrong_to_correct)
    unchanged = sorted(set(before) - set(changed))
    changed_same_correctness = sorted(set(changed) - set(correctness))
    strata = {
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "changed_tokens_same_correctness": changed_same_correctness,
        "unchanged_tokens": unchanged,
    }
    if selection == "correctness":
        selected = (
            correctness
            if correctness_cap is None
            else sample_stratified(
                (correct_to_wrong, wrong_to_correct), correctness_cap
            )
        )
    elif len(changed) <= target_count:
        # Keep all originally changed questions when they fit, and fill the balance with controls.
        controls = min(len(unchanged), target_count - len(changed))
        selected = changed + sample_evenly(unchanged, controls)
    else:
        selected = sample_stratified(tuple(strata.values()), target_count)
    require(selected, "The selection produced no questions")
    selected_set = set(selected)
    distribution = {
        label: {"available": len(ids), "selected": len(selected_set.intersection(ids))}
        for label, ids in strata.items()
    }
    return sorted(selected), before, after, changed, correctness, distribution


def check_aa_trace(startup, last):
    names = {"capture_begin", "capture_end", "reset"}
    original = [row for row in startup if row["event"] in names]
    require(
        [row for row in last if row["event"] in names] == original,
        "Unexpected graph capture/reset after startup",
    )
    require(
        not any(row["event"] in {"pause", "resume"} for row in last),
        "Unexpected VMM transition in A/A",
    )
    captures = {
        (row["pid"], row["graph"], tuple(row["pool"]))
        for row in original
        if row["event"] == "capture_end"
    }
    replays = {
        (row["pid"], row["graph"], tuple(row["pool"]))
        for row in last
        if row["event"] == "first_replay"
    }
    require(
        replays and replays <= captures,
        "Expected replay of an originally captured graph",
    )
    return {
        "captured_graphs": sorted(captures),
        "replayed_graphs": sorted(replays),
        "recapture_count": 0,
    }


def run(args, report):
    import tokenspeed_engine
    import torch
    import torch_npu
    from transformers import AutoTokenizer

    import tokenspeed
    from tokenspeed.runtime.engine import zmq_wire
    from tokenspeed.runtime.engine.io_struct import (
        IsSleepingReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )
    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

    reference = json.loads((args.original / "summary.json").read_text())
    require(
        reference["status"] == "complete", "Expected a completed original benchmark"
    )
    require(
        reference["dataset_sha256"] == DATASET_SHA256,
        "Original run used a different test set",
    )
    git_version = pta_git_version(torch_npu, args.pta_commit)
    require(
        torch_npu.__version__ == reference["torch_npu"]
        and git_version == reference["torch_npu_git"],
        "PTA version mismatch",
    )
    require(
        torch.npu.device_count() == args.tp_size == reference["tp_size"],
        "Unexpected TP topology",
    )
    if args.max_tokens is not None:
        require(
            args.max_tokens == reference["max_tokens"],
            "Use the original run's max output tokens",
        )
    args.max_tokens = reference["max_tokens"]
    args.seed = reference["seed"]
    server_concurrency = reference["concurrency"]
    require(
        1 <= args.concurrency <= server_concurrency <= 64,
        "Unsupported client concurrency",
    )
    selected, original_before, original_after, changed, correctness, strata = (
        select_cases(
            args.original,
            args.selection,
            args.target_count,
            args.correctness_cap,
        )
    )
    rows = load_dataset(args.dataset)
    require(
        set(selected) <= set(range(len(rows))),
        "Selected question index outside dataset",
    )
    model = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [(i, encode_prompt(model, rows[i]["question"])) for i in selected]
    warm_ids = sorted(set(range(len(rows))) - set(selected), reverse=True)[
        : args.warmup
    ]
    warm_prompts = [(i, encode_prompt(model, rows[i]["question"])) for i in warm_ids]
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.max_tokens,
        "stream_interval": 1,
    }
    server_args = make_server_args(
        argparse.Namespace(**{**vars(args), "concurrency": server_concurrency})
    )
    host = Path(
        os.environ.get("TOKENSPEED_HOST_BINARY") or tokenspeed_engine.host_binary()
    ).resolve(strict=True)
    trace_path = args.output_dir / "graph-trace.jsonl"
    fixture = Path(__file__).resolve().parent / "trace"
    env = {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(fixture), os.environ.get("PYTHONPATH")))
        ),
        "GSM8K_GRAPH_TRACE": str(trace_path),
        "GSM8K_TRACE_HOST_EXE": str(host),
    }
    report.update(
        {
            "question_ids": selected,
            "original_changed_ids": changed,
            "original_correctness_changed_ids": correctness,
            "selection_strata": strata,
            "warmup_question_ids": warm_ids,
            "sampling": sampling,
            "seed": args.seed,
            "server_args": server_args,
            "server_max_concurrency": server_concurrency,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "torch_npu_git": git_version,
            "tokenspeed_path": str(next(iter(tokenspeed.__path__))),
            "host": str(host),
        }
    )

    def save():
        write_json(args.output_dir / "summary.json", report)

    async def control_reply(driver, request):
        payload = [zmq_wire.encode_control(request)]
        replies = await asyncio.wait_for(
            driver.frontend.control(payload), timeout=args.control_timeout
        )
        require(
            len(replies) == 1,
            f"Expected one engine response for {type(request).__name__}",
        )
        return zmq_wire.decode_control_output(replies[0][0])

    def sleeping(driver, expected):
        reply = asyncio.run(control_reply(driver, IsSleepingReqInput()))
        require(
            reply.is_sleeping == expected, f"Expected is_sleeping={expected}: {reply!r}"
        )

    def transition(driver, request):
        start = time.perf_counter()
        reply = asyncio.run(control_reply(driver, request(tags=["acl_graph"])))
        require(reply.success, f"{request.__name__}: {reply!r}")
        return time.perf_counter() - start

    def phase(driver, name):
        print(
            f"{name}: evaluating {len(prompts)} questions at concurrency {args.concurrency}",
            flush=True,
        )
        results, wall_s = asyncio.run(
            infer(
                driver,
                prompts,
                sampling,
                concurrency=args.concurrency,
                timeout=args.request_timeout,
                phase=name,
            )
        )
        scored = [
            sample.score(rows[sample.index]["answer"], model) for sample in results
        ]
        write_rows(args.output_dir / f"{name}.jsonl", scored)
        report[name] = summarize(scored, wall_s)
        report["original_token_matches"][name] = {
            f"original_{label}": sum(
                row["output_ids"] == baseline[row["index"]]["output_ids"]
                for row in scored
            )
            for label, baseline in (
                ("before", original_before),
                ("after", original_after),
            )
        }
        save()
        print(json.dumps({"phase": name, **report[name]}), flush=True)
        return scored

    def pair(a, b, left, right):
        comparison, diffs = compare(left, right, report[a], report[b])
        comparison["changed_ids"] = [row["index"] for row in diffs]
        comparison["changed_correctness_ids"] = [
            row["index"]
            for row in diffs
            if row["before"]["correct"] != row["after"]["correct"]
        ]
        report["comparisons"][f"{a}_vs_{b}"] = comparison
        write_rows(args.output_dir / f"{a}-vs-{b}-differences.jsonl", diffs)
        save()

    with WireDriver(
        server_args,
        num_engines=1,
        routing="round_robin",
        startup_timeout_s=args.startup_timeout,
        env=env,
    ) as driver:
        report["launcher_pid"] = driver.process.pid
        startup = read_trace(trace_path)
        sleeping(driver, False)
        if warm_prompts:
            asyncio.run(
                infer(
                    driver,
                    warm_prompts,
                    {**sampling, "max_new_tokens": 16},
                    concurrency=args.concurrency,
                    timeout=args.request_timeout,
                    phase="warmup",
                )
            )
        baseline = phase(driver, "a1")
        preceding = "a1"
        if args.mode in {"aa", "aa-then-pause"}:
            second = phase(driver, "a2")
            pair("a1", "a2", baseline, second)
            report["aa_graph"] = check_aa_trace(startup, read_trace(trace_path))
            preceding = "a2"
        if args.mode in {"pause", "aa-then-pause"}:
            before = read_trace(trace_path)
            report["pause_wire_s"] = transition(driver, ReleaseMemoryOccupationReqInput)
            sleeping(driver, True)
            report["resume_wire_s"] = transition(driver, ResumeMemoryOccupationReqInput)
            sleeping(driver, False)
            restored = phase(driver, "b")
            pair("a1", "b", baseline, restored)
            if preceding == "a2":
                pair("a2", "b", second, restored)
            report["graph"] = verify_graph_trace(
                startup,
                before,
                read_trace(trace_path),
                ["acl_graph"],
                args.tp_size,
                graph_policy="discard",
            )
    report["launcher_exit_code"] = driver.process.returncode
    require(
        driver.process.returncode == 0,
        f"Launcher exit status {driver.process.returncode}",
    )
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Directory containing the full run's before/after JSONL and summary",
    )
    parser.add_argument(
        "--selection",
        choices=("correctness", "changed-plus-control"),
        default="correctness",
    )
    parser.add_argument(
        "--mode", choices=("aa", "pause", "aa-then-pause"), default="aa-then-pause"
    )
    parser.add_argument("--target-count", type=int, default=96)
    parser.add_argument(
        "--correctness-cap",
        type=int,
        help="Cap the correctness-flip subset, balanced by flip direction",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--max-tokens", type=int, help="If provided, must match the full run"
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--dist-init-addr")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--control-timeout", type=float, default=180)
    parser.add_argument("--pta-commit", help="Require this PTA Git commit if specified")
    args = parser.parse_args()
    if not (
        1 <= args.concurrency <= 64 and args.target_count >= 1 and args.warmup >= 0
    ):
        parser.error("Invalid concurrency, target count or warmup")
    if args.correctness_cap is not None and (
        args.correctness_cap < 1 or args.selection != "correctness"
    ):
        parser.error(
            "--correctness-cap requires a positive count and --selection correctness"
        )
    args.dist_init_addr = args.dist_init_addr or ephemeral_loopback_address()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "model": str(args.model),
        "original": str(args.original),
        "selection": args.selection,
        "mode": args.mode,
        "client_concurrency": args.concurrency,
        "target_count": args.target_count,
        "correctness_cap": args.correctness_cap,
        "tp_size": args.tp_size,
        "dist_init_addr": args.dist_init_addr,
        "original_token_matches": {},
        "comparisons": {},
    }
    path = args.output_dir / "summary.json"
    write_json(path, report)
    try:
        run(args, report)
    except BaseException:
        report.update({"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        write_json(path, report)
    print(
        json.dumps({"status": report["status"], "comparisons": report["comparisons"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
