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

"""Compare GSM8K before and after real graph-pool sleep/wake on one TokenSpeed service."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import socket
import statistics
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

DATASET_SOURCE = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
)
DATASET_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
DATASET_SIZE = 1319
# Match TokenSpeed's test/single_request_benchmark/gsm8k_decode.py prompt.
PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
POLICIES = {"weights": "backup", "kv_cache": "discard", "acl_graph": "backup"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pta_git_version(torch_npu, expected):
    version = getattr(getattr(torch_npu, "version", None), "git_version", None)
    if expected is not None:
        require(version == expected, f"Expected PTA commit {expected}, got {version}")
    return version


def load_dataset(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == DATASET_SHA256, f"Unexpected GSM8K source SHA-256: {digest}")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(
        len(rows) == DATASET_SIZE, f"Expected {DATASET_SIZE} questions, got {len(rows)}"
    )
    return rows


def last_boxed(text):
    """Return the last balanced LaTeX boxed answer, as in TokenSpeed's scorer."""
    opening = "\\boxed{"
    index = text.rfind(opening)
    if index < 0:
        return None
    start = index + len(opening)
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end]
    return None


def number(text):
    if text is None:
        return None
    text = text.strip().split("\\text{", 1)[0]
    for token in ("\\$", "$", "\\!", "\\,", ",", " "):
        text = text.replace(token, "")
    try:
        value = Decimal(text.rstrip("."))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def encode_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + PROMPT_SUFFIX}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@dataclass
class Result:
    index: int
    request_id: str
    sent: float
    first: float | None = None
    last: float | None = None
    ended: float | None = None
    first_chunk_tokens: int = 0
    prompt_tokens: int = 0
    finish_reason: str | None = None
    output_ids: list[int] = field(default_factory=list)
    verify_count: int | None = None
    accept_length: float | None = None

    def add(self, chunk, now):
        self.finish_reason = chunk["finish_reason"]
        if self.finish_reason is not None:
            self.ended = now
            self.verify_count = chunk.get("spec_verify_ct")
            self.accept_length = chunk.get("accept_draft_tokens")
            require(
                self.finish_reason in {"stop", "length"},
                f"Question {self.index} failed: {chunk['finish_details']!r}",
            )
        self.prompt_tokens = chunk["prompt_tokens"]
        output = chunk["output_ids"]
        if output:
            if self.first is None:
                self.first = now
                self.first_chunk_tokens = len(output)
            self.last = now
            self.output_ids.extend(output)

    def score(self, expected, tokenizer):
        text = tokenizer.decode(self.output_ids, skip_special_tokens=True)
        prediction = last_boxed(text)
        reference = expected.rsplit("####", 1)[-1].strip()
        expected_number = number(reference)
        require(expected_number is not None, f"Unexpected reference {reference!r}")
        # TPOT is reported only where the first output token was observed alone.
        decode_s = None
        tpot_ms = None
        if self.first_chunk_tokens == 1 and len(self.output_ids) >= 2:
            decode_s = self.last - self.first
            tpot_ms = decode_s * 1000 / (len(self.output_ids) - 1)
        return {
            "index": self.index,
            "request_id": self.request_id,
            "reference": reference,
            "prediction": prediction,
            "correct": number(prediction) == expected_number,
            "output_text": text,
            "output_ids": self.output_ids,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": len(self.output_ids),
            "first_chunk_tokens": self.first_chunk_tokens,
            "finish_reason": self.finish_reason,
            "ttft_ms": None if self.first is None else (self.first - self.sent) * 1000,
            "decode_s": decode_s,
            "tpot_ms": tpot_ms,
            "request_s": self.ended - self.sent,
            "verify_count": self.verify_count,
            "accept_length": self.accept_length,
        }


async def infer(driver, prompts, sampling, *, concurrency, timeout, phase):
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    results = []
    pending = {}
    work = iter(prompts)
    start = time.perf_counter()

    async def submit():
        item = next(work, None)
        if item is None:
            return
        index, ids = item
        request = generate_request(ids, sampling, stream=True)
        frames = [bytes(frame) for frame in driver._encoder.encode(request)]
        pending[request.rid] = Result(index, request.rid, time.perf_counter())
        await asyncio.wait_for(driver.frontend.generate(frames), timeout=timeout)

    for _ in range(min(concurrency, len(prompts))):
        await submit()
    while pending:
        deadline = min(item.sent for item in pending.values()) + timeout
        remaining = max(0.0, deadline - time.perf_counter())
        try:
            chunks = await asyncio.wait_for(
                driver.frontend.next_outputs(), timeout=remaining
            )
        except TimeoutError as exc:
            pending_indexes = [item.index for item in pending.values()]
            raise RuntimeError(
                f"{phase}: deadline elapsed; pending questions {pending_indexes}"
            ) from exc
        now = time.perf_counter()
        complete_count = 0
        for rid, chunk in chunks:
            if rid not in pending:
                raise RuntimeError(f"Unexpected output for request {rid}")
            if isinstance(chunk, Exception):
                raise chunk
            item = pending[rid]
            item.add(chunk, now)
            if item.finish_reason is not None:
                results.append(pending.pop(rid))
                complete_count += 1
                if len(results) % 100 == 0:
                    print(
                        f"{phase}: {len(results)}/{len(prompts)} complete", flush=True
                    )
        for _ in range(complete_count):
            await submit()
    wall_s = time.perf_counter() - start
    require(len(results) == len(prompts), f"{phase}: incomplete result")
    return sorted(results, key=lambda item: item.index), wall_s


def summary_stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": values[math.ceil(len(values) * 0.95) - 1],
    }


def summarize(rows, wall_s):
    correct = sum(row["correct"] for row in rows)
    tokens = sum(row["completion_tokens"] for row in rows)
    timed = [row for row in rows if row["tpot_ms"] is not None]
    timed_tokens = sum(row["completion_tokens"] - 1 for row in timed)
    return {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "wall_s": wall_s,
        "completion_tokens": tokens,
        "e2e_output_tokens_per_s": tokens / wall_s,
        "questions_per_s": len(rows) / wall_s,
        "length_capped": sum(row["finish_reason"] == "length" for row in rows),
        "first_chunk_multi_token": sum(row["first_chunk_tokens"] > 1 for row in rows),
        "ttft_ms": summary_stats(row["ttft_ms"] for row in rows),
        "tpot_ms": summary_stats(row["tpot_ms"] for row in rows),
        "verify_count": summary_stats(row.get("verify_count") for row in rows),
        "accept_length": summary_stats(row.get("accept_length") for row in rows),
        "token_weighted_tpot_ms": (
            sum(row["decode_s"] for row in timed) * 1000 / timed_tokens
            if timed_tokens
            else None
        ),
    }


def compare(before, after, before_summary, after_summary):
    diff = []
    counts = Counter()
    for a, b in zip(before, after, strict=True):
        require(a["index"] == b["index"], "Mismatched question pairing")
        counts["exact_output_ids"] += a["output_ids"] == b["output_ids"]
        counts["exact_output_text"] += a["output_text"] == b["output_text"]
        counts["changed_correctness"] += a["correct"] != b["correct"]
        counts["correct_to_wrong"] += a["correct"] and not b["correct"]
        counts["wrong_to_correct"] += b["correct"] and not a["correct"]
        if a["output_ids"] != b["output_ids"]:
            diff.append({"index": a["index"], "before": a, "after": b})

    def percent_delta(old, new):
        return (new - old) * 100 / old if old and new is not None else None

    comparison = {
        **counts,
        "changed_output_ids": len(diff),
        "accuracy_percentage_points": (
            after_summary["accuracy"] - before_summary["accuracy"]
        )
        * 100,
        "wall_s_delta_pct": percent_delta(
            before_summary["wall_s"], after_summary["wall_s"]
        ),
        "mean_tpot_delta_pct": percent_delta(
            before_summary["tpot_ms"]["mean"], after_summary["tpot_ms"]["mean"]
        ),
        "p95_tpot_delta_pct": percent_delta(
            before_summary["tpot_ms"]["p95"], after_summary["tpot_ms"]["p95"]
        ),
    }
    return comparison, diff


def make_server_args(args):
    caps = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= args.concurrency]
    if caps[-1] != args.concurrency:
        caps.append(args.concurrency)
    return [
        "--model",
        str(args.model),
        "--device",
        "npu",
        "--tensor-parallel-size",
        str(args.tp_size),
        "--dist-init-addr",
        args.dist_init_addr,
        "--dtype",
        "bfloat16",
        "--seed",
        str(args.seed),
        "--attention-backend",
        "mha",
        "--sampling-backend",
        "greedy",
        "--enable-memory-saver",
        "--weight-memory-policy",
        "cpu_backup",
        "--disable-prefill-graph",
        "--disable-cudagraph-memory-reserve",
        "--disable-pdl",
        "--disable-autotune",
        "--disable-prefix-caching",
        "--disable-kvstore",
        "--enable-log-request-stats",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(args.concurrency),
        "--max-total-tokens",
        str(max(16384, args.concurrency * 1024)),
        "--chunked-prefill-size",
        "4096",
        "--max-cudagraph-capture-size",
        str(args.concurrency),
        "--stream-interval",
        "1",
        "--cudagraph-capture-sizes",
        *map(str, caps),
    ]


def ephemeral_loopback_address():
    """Choose a local port and leave space for TokenSpeed's derived port cluster."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        if port <= 65529:
            return f"127.0.0.1:{port}"


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_graph_trace(
    startup,
    before,
    after,
    tags,
    tp_size,
    *,
    graph_policy,
    weight_policies=("backup",),
    no_pause_control=False,
):
    lifecycle = {"capture_begin", "capture_end", "reset"}
    expected = [event for event in startup if event["event"] in lifecycle]
    require(
        [event for event in after if event["event"] in lifecycle] == expected,
        "Graph capture/reset occurred after startup",
    )
    captures = {
        (event["pid"], event["graph"], tuple(event["pool"]))
        for event in expected
        if event["event"] == "capture_end"
    }
    pids = {pid for pid, _, _ in captures}
    require(len(pids) == tp_size, f"Expected {tp_size} captured ranks, saw {pids}")

    def replays(rows, phase):
        return {
            (event["pid"], event["graph"], tuple(event["pool"]))
            for event in rows
            if event["event"] == "first_replay" and event["phase"] == phase
        }

    baseline_replays = replays(before, "before")
    require({row[0] for row in baseline_replays} == pids, "Missing baseline replay")
    transitions = [event for event in after if event["event"] in {"pause", "resume"}]
    if no_pause_control:
        all_replays = replays(after, "before")
        require(all_replays <= captures, "Replay used an uncaptured graph")
        require(not transitions, "A/A control unexpectedly paused or resumed")
        return {
            "captured_graphs": sorted(captures),
            "before_replays": sorted(baseline_replays),
            "all_replays": sorted(all_replays),
            "recapture_count": 0,
            "transitions": [],
        }
    resumed_replays = replays(after, "after")
    require({row[0] for row in resumed_replays} == pids, "Missing post-resume replay")
    require(
        baseline_replays | resumed_replays <= captures,
        "Replay used an uncaptured graph",
    )
    shared_replays = baseline_replays & resumed_replays
    require(
        {row[0] for row in shared_replays} == pids,
        "No shared replayed graph id on one or more ranks",
    )

    require(
        Counter((row["pid"], row["event"], row["tag"]) for row in transitions)
        == Counter(
            (pid, op, tag) for pid in pids for op in ("pause", "resume") for tag in tags
        ),
        "Unexpected missing or duplicate rank transitions",
    )
    amounts = []
    for event in transitions:
        tag = event["tag"]
        pausing = event["event"] == "pause"
        old = {b["ptr"]: b for b in event["before"]["blocks"] if b["tag"] == tag}
        new = {b["ptr"]: b for b in event["after"]["blocks"] if b["tag"] == tag}
        require(
            old and old.keys() == new.keys(),
            f"{tag}: VA changed or tag has no allocations",
        )
        require(
            not event["before"]["callback_fault"]
            and not event["after"]["callback_fault"],
            "Native saver callback fault",
        )
        expected_policies = {
            "weights": set(weight_policies),
            "acl_graph": {graph_policy},
            "kv_cache": {"discard"},
        }[tag]
        require(
            {block["policy"] for block in new.values()} == expected_policies,
            f"{tag}: expected allocation policies {expected_policies}",
        )
        for addr, b in old.items():
            a = new[addr]
            require(
                b["reserved"]
                and a["reserved"]
                and b["policy"] == a["policy"]
                and b["aligned_bytes"] == a["aligned_bytes"]
                and b["physical_live"] == b["mapped"] == pausing
                and a["physical_live"] == a["mapped"] == (not pausing)
                and a["backup_valid"] == (pausing and a["policy"] == "backup"),
                f"{event['pid']} {event['event']} {tag} {addr}: invalid saver transition",
            )
        amounts.append(
            {
                "event": event["event"],
                "pid": event["pid"],
                "tag": tag,
                "managed_bytes": sum(b["aligned_bytes"] for b in new.values()),
                "bytes_by_policy": {
                    policy: sum(
                        block["aligned_bytes"]
                        for block in new.values()
                        if block["policy"] == policy
                    )
                    for policy in sorted(expected_policies)
                },
                "seconds": event["seconds"],
            }
        )
    return {
        "captured_graphs": sorted(captures),
        "before_replays": sorted(baseline_replays),
        "after_replays": sorted(resumed_replays),
        "shared_replayed_graphs": sorted(shared_replays),
        "recapture_count": 0,
        "transitions": amounts,
    }


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_rows(path, rows):
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        UpdateWeightFromDiskReqInput,
    )
    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

    report.update(
        {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "torch_npu_git": pta_git_version(torch_npu, args.pta_commit),
            "tokenspeed_path": str(next(iter(tokenspeed.__path__))),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        }
    )
    require(
        torch.npu.device_count() == args.tp_size,
        "Visible device count does not match TP size",
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
    rows = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        (i, encode_prompt(tokenizer, row["question"]))
        for i, row in enumerate(rows[: args.limit])
    ]
    warm_prompts = [
        (-i, encode_prompt(tokenizer, rows[-i]["question"]))
        for i in range(1, args.warmup + 1)
    ]
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.max_tokens,
        "stream_interval": 1,
    }
    server_args = make_server_args(args)
    policy_index = server_args.index("--weight-memory-policy") + 1
    server_args[policy_index] = args.weight_memory_policy
    if args.draft:
        server_args.extend(
            (
                "--speculative-algorithm",
                "MTP",
                "--speculative-draft-model-path",
                str(args.draft),
                "--speculative-num-steps",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--model-update-draft-weights",
                "retain",
            )
        )
    report.update(
        {
            "server_args": server_args,
            "host_binary": str(host),
            "sampling": sampling,
        }
    )

    def phase(driver, name):
        print(
            f"{name}: evaluating {len(prompts)} questions at concurrency {args.concurrency}",
            flush=True,
        )
        samples, wall_s = asyncio.run(
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
            result.score(rows[result.index]["answer"], tokenizer) for result in samples
        ]
        write_rows(args.output_dir / f"{name}.jsonl", scored)
        report[name] = summarize(scored, wall_s)
        if args.draft:
            require(
                any((row["verify_count"] or 0) > 0 for row in scored),
                f"{name}: no MTP verifier calls recorded",
            )
        write_json(args.output_dir / "summary.json", report)
        print(json.dumps({"phase": name, **report[name]}), flush=True)
        return scored

    async def control_reply(driver, request):
        encoded = [zmq_wire.encode_control(request)]
        replies = await asyncio.wait_for(
            driver.frontend.control(encoded), timeout=args.control_timeout
        )
        require(
            len(replies) == 1,
            f"Expected one engine response for {type(request).__name__}",
        )
        return zmq_wire.decode_control_output(replies[0][0])

    def control(driver, request):
        started = time.perf_counter()
        reply = asyncio.run(control_reply(driver, request))
        elapsed = time.perf_counter() - started
        require(reply.success, f"{type(request).__name__}: {reply!r}")
        return elapsed

    def refused(driver, request, expected):
        reply = asyncio.run(control_reply(driver, request))
        require(
            not reply.success and expected in reply.message,
            f"Expected {type(request).__name__} refusal containing {expected!r}: {reply!r}",
        )
        return reply.message

    def sleeping(driver, expected):
        reply = asyncio.run(control_reply(driver, IsSleepingReqInput()))
        require(
            reply.is_sleeping == expected, f"Expected is_sleeping={expected}: {reply!r}"
        )

    with WireDriver(
        report["server_args"],
        num_engines=1,
        routing="round_robin",
        startup_timeout_s=args.startup_timeout,
        env=env,
    ) as driver:
        report.update({"engines": driver.engines(), "launcher_pid": driver.process.pid})
        startup_trace = read_trace(trace_path)
        require(
            any(row["event"] == "capture_end" for row in startup_trace),
            "No startup graph trace",
        )
        sleeping(driver, False)
        if warm_prompts:
            asyncio.run(
                infer(
                    driver,
                    warm_prompts,
                    {**sampling, "max_new_tokens": min(16, args.max_tokens)},
                    concurrency=args.concurrency,
                    timeout=args.request_timeout,
                    phase="warmup",
                )
            )
        before = phase(driver, "before")
        before_trace = read_trace(trace_path)
        if not args.no_pause_control:
            if args.weight_memory_policy == "discard":
                report["early_weights_release_refusal"] = refused(
                    driver,
                    # Graph release now includes KV; weight-only release must
                    # still reject discarding weights beneath live KV.
                    ReleaseMemoryOccupationReqInput(tags=["weights"]),
                    "Release KV before discarding weights",
                )
                sleeping(driver, False)
            report["pause_wire_s"] = control(
                driver, ReleaseMemoryOccupationReqInput(tags=args.tags)
            )
            sleeping(driver, True)
            if args.weight_memory_policy == "discard":
                report["early_kv_resume_refusal"] = refused(
                    driver,
                    ResumeMemoryOccupationReqInput(tags=args.tags),
                    "a complete weight update is required",
                )
                sleeping(driver, True)
                resume_started = time.perf_counter()
                report["remap_wire_s"] = control(
                    driver,
                    ResumeMemoryOccupationReqInput(tags=["acl_graph", "weights"]),
                )
                sleeping(driver, True)
                # Load the same checkpoint to hold model quality constant.
                report["checkpoint_commit_wire_s"] = control(
                    driver, UpdateWeightFromDiskReqInput(model_path=str(args.model))
                )
                sleeping(driver, True)
                report["resume_kv_wire_s"] = control(
                    driver, ResumeMemoryOccupationReqInput(tags=["kv_cache"])
                )
                report["resume_wire_s"] = time.perf_counter() - resume_started
            else:
                report["resume_wire_s"] = control(
                    driver, ResumeMemoryOccupationReqInput(tags=args.tags)
                )
            print(
                json.dumps(
                    {
                        "pause_wire_s": report["pause_wire_s"],
                        "resume_wire_s": report["resume_wire_s"],
                    }
                ),
                flush=True,
            )
        sleeping(driver, False)
        after = phase(driver, "after")
        report["comparison"], diffs = compare(
            before, after, report["before"], report["after"]
        )
        write_rows(args.output_dir / "output-differences.jsonl", diffs)
        request_fields = (
            "index",
            "completion_tokens",
            "ttft_ms",
            "tpot_ms",
            "request_s",
        )
        report["first_submitted_request"] = {
            name: {field: records[0][field] for field in request_fields}
            for name, records in (("before", before), ("after", after))
        }
        weight_policies = ("backup",)
        if args.weight_memory_policy == "discard":
            weight_policies = ("backup", "discard") if args.draft else ("discard",)
        report["graph"] = verify_graph_trace(
            startup_trace,
            before_trace,
            read_trace(trace_path),
            args.tags,
            args.tp_size,
            graph_policy=args.expected_graph_policy,
            weight_policies=weight_policies,
            no_pause_control=args.no_pause_control,
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
    parser.add_argument("--draft", type=Path, help="MTP draft checkpoint to retain")
    parser.add_argument(
        "--no-pause-control",
        action="store_true",
        help="Repeat the workload on the same running service without pausing.",
    )
    parser.add_argument(
        "--weight-memory-policy",
        choices=("cpu_backup", "discard"),
        default="cpu_backup",
    )
    parser.add_argument(
        "--expected-graph-policy", choices=("discard", "backup"), default="discard"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--dist-init-addr",
        help="TokenSpeed distributed address, defaults to a local ephemeral port",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Server seed; greedy sampling pins its own tie-breaking seed",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=DATASET_SIZE,
        help="Use a smaller value for smoke runs only",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Short pre-run warmup requests; resume run starts cold",
    )
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--control-timeout", type=float, default=180)
    parser.add_argument("--pta-commit", help="Require this PTA Git commit if specified")
    parser.add_argument(
        "--tags", nargs="+", choices=list(POLICIES), default=["acl_graph"]
    )
    args = parser.parse_args()
    if args.draft and args.weight_memory_policy != "discard":
        parser.error("--draft requires --weight-memory-policy discard")
    if args.weight_memory_policy == "discard" and set(args.tags) != set(POLICIES):
        parser.error("discard benchmark requires --tags kv_cache weights acl_graph")
    if (
        not 1 <= args.limit <= DATASET_SIZE
        or not 1 <= args.concurrency <= 64
        or not 2 <= args.max_tokens <= 2048
        or not 0 <= args.warmup <= DATASET_SIZE
        or len(set(args.tags)) != len(args.tags)
        or "acl_graph" not in args.tags
    ):
        parser.error(
            "Invalid limit, concurrency, token cap, warmup or tags; acl_graph is required"
        )
    if args.dist_init_addr is None:
        args.dist_init_addr = ephemeral_loopback_address()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "model": str(args.model),
        "draft": str(args.draft) if args.draft else None,
        "no_pause_control": args.no_pause_control,
        "weight_memory_policy": args.weight_memory_policy,
        "tp_size": args.tp_size,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "tags": args.tags,
        "maximum_measured_inflight_requests": min(args.concurrency, args.limit),
        "dist_init_addr": args.dist_init_addr,
        "dataset_source": DATASET_SOURCE,
        "dataset_sha256": DATASET_SHA256,
        "dataset_size": DATASET_SIZE,
        "measured_limit": args.limit,
        "warmup_requests_before_baseline": args.warmup,
        "warmup_requests_after_resume": 0,
        "prompt": {"shots": 0, "suffix": PROMPT_SUFFIX, "enable_thinking": False},
        "batching": "bounded continuous inflight requests; actual scheduler membership may vary by timing",
        "metric_definition": {
            "accuracy": "last balanced boxed answer compared numerically with GSM8K #### target",
            "tpot": "wire client: (last minus first token chunk receipt time) / (output tokens minus 1)",
            "wall": "first request submission through last finish, excluding load, warmup, pause and resume",
        },
    }
    write_json(args.output_dir / "summary.json", report)
    try:
        run(args, report)
    except BaseException:
        report.update({"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        write_json(args.output_dir / "summary.json", report)
    print(
        json.dumps({"comparison": report["comparison"], "status": report["status"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
