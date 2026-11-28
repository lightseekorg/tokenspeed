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

"""Collect isolated requests and import server-side prefill timing."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from test.cache_gds_calibration.data import (
    PrefillCase,
    integer,
    positive,
    prefill_samples,
    read_rows,
    validate_prefix,
    write_rows,
)


def request(url, ids, rid, dp_rank, timeout):
    payload = {
        "rid": rid,
        "input_ids": ids,
        "data_parallel_rank": dp_rank,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
        "stream": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or "meta_info" not in result:
        raise ValueError(
            "expected a successful native /generate response with meta_info"
        )
    meta = result["meta_info"]
    if meta.get("id") != rid:
        raise ValueError(
            f"response id {meta.get('id')} does not match {rid}; cannot join server timing"
        )
    reason = meta.get("finish_reason", {})
    if not isinstance(reason, dict) or reason.get("type") in (None, "abort"):
        raise ValueError(f"request did not finish successfully: {reason}")
    return meta


def collect_prefill(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code == "yes"
    )
    corpus = Path(args.text_file).read_text()
    corpus_sha256 = hashlib.sha256(corpus.encode()).hexdigest()
    pool = tokenizer.encode(corpus, add_special_tokens=False)
    if args.repeats < 1 or args.warmup_repeats < 1 or args.data_parallel_rank < 0:
        raise ValueError("positive repeats/warmups and nonnegative DP rank required")
    if min(args.prefix_lengths) < 0 or min(args.suffix_lengths) <= 0:
        raise ValueError("invalid context lengths")
    cases = [
        PrefillCase(
            prefix_tokens=prefix_tokens, new_tokens=new_tokens, split=args.split
        )
        for prefix_tokens in args.prefix_lengths
        for new_tokens in args.suffix_lengths
    ]
    with Path(args.output).open("x") as output:
        for sample in prefill_samples(
            cases,
            pool,
            repeats=args.repeats,
            warmup_repeats=args.warmup_repeats,
            seed=args.seed,
        ):
            prefix_tokens = sample.case.prefix_tokens
            prompt_tokens = sample.case.prompt_tokens
            repeat, ids = sample.repeat, sample.input_ids
            rid = f"gds-cal-{uuid.uuid4().hex}"
            if prefix_tokens:
                request(
                    args.url,
                    ids[:prefix_tokens],
                    rid + "-prime",
                    args.data_parallel_rank,
                    args.timeout,
                )
            started = time.perf_counter()
            meta = request(args.url, ids, rid, args.data_parallel_rank, args.timeout)
            row = {
                "kind": "request",
                "deployment": args.deployment,
                "rid": rid,
                "split": args.split,
                "expected_a": prefix_tokens,
                "expected_b": prompt_tokens,
                "warmup": repeat < 0,
                "repeat": repeat,
                "seed": args.seed,
                "model_path": args.model_path,
                "corpus_sha256": corpus_sha256,
                "dp_rank": args.data_parallel_rank,
                "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "http_elapsed_s": time.perf_counter() - started,
                "meta_info": meta,
            }
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            print(
                f"{rid}: prefix_tokens={prefix_tokens} new_tokens={sample.case.new_tokens} "
                f"prompt_tokens={prompt_tokens} warmup={sample.warmup}",
                flush=True,
            )


STATS = re.compile(r"Req: (\S+) Finish! RequestStats\((.*)\)")


def parse_server_logs(paths):
    stats, traces, arrivals = {}, {}, {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if "event=ts_request_accepted" in line:
                plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
                arrival = re.search(
                    r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?"
                    r'(?:Z|[+-]\d\d:\d\d))\s+.*event=ts_request_accepted.*rid="([^"]+)"',
                    plain,
                )
                if arrival:
                    stamp = datetime.fromisoformat(
                        arrival[1].replace("Z", "+00:00")
                    ).timestamp()
                    arrivals[arrival[2]] = stamp
            match = STATS.search(line)
            if match:
                rid, body = match.groups()
                # Parse literal keyword values, never execute log text.
                call = ast.parse(f"record({body})", mode="eval").body
                record = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                if rid in stats and stats[rid] != record:
                    raise ValueError(
                        f"conflicting stats for {rid}; use only one serving role's logs"
                    )
                stats[rid] = record
            if "TTFTBreakdown " in line:
                record, _ = json.JSONDecoder().raw_decode(
                    line.split("TTFTBreakdown ", 1)[1]
                )
                if record["role"] == "prefill":
                    rid = record["rid"]
                    if rid in traces and traces[rid] != record:
                        raise ValueError(f"conflicting prefill traces for {rid}")
                    traces[rid] = record
    for rid, record in stats.items():
        if rid in arrivals:
            record["accepted_unix_s"] = arrivals[rid]
    return stats, traces


def import_prefill(args):
    if args.max_prefix_shortfall < 0:
        raise ValueError("prefix shortfall must be nonnegative")
    if (args.timing_source == "pd-prefill-trace") != (
        args.queue_source == "pd-admission"
    ):
        raise ValueError(
            "PD timing requires the pd-admission queue source; plain timing requires a plain queue source"
        )
    stats, traces = parse_server_logs(args.server_logs)
    accepted, rejected = [], []
    seen = set()
    for row in read_rows(args.requests):
        if row["warmup"]:
            continue
        rid = row["rid"]
        if rid in seen:
            raise ValueError(f"duplicate request {rid}")
        seen.add(rid)
        try:
            s = stats[rid]
            if (
                s["status"] != "finished"
                or s["preempt_count"] != 0
                or s["preempt_ms"] != 0
            ):
                raise ValueError("aborted request or reported preemption")
            if s["output_tokens"] != 1:
                raise ValueError("expected exactly one output token")
            cached_tokens = integer(s["cache_tokens"], 0)
            prompt_tokens = integer(s["prompt_tokens"], 1)
            validate_prefix(
                expected_prefix_tokens=row["expected_a"],
                expected_prompt_tokens=row["expected_b"],
                cached_tokens=cached_tokens,
                prompt_tokens=prompt_tokens,
                max_prefix_shortfall=args.max_prefix_shortfall,
            )
            if args.timing_source == "plain-request-stats":
                if s["decode_wait_ms"] != 0:
                    raise ValueError("decode-role timing is not prefill timing")
                elapsed = s["prefill_ms"]
                queue_source = args.queue_source
                if queue_source == "server-accepted":
                    if "accepted_unix_s" not in s:
                        raise ValueError(
                            "no ts_request_accepted arrival parsed; server-accepted "
                            "requires engine acceptance logs with UTC (Z) or "
                            "numeric-offset (+/-HH:MM) timestamps"
                        )
                    # created_time precedes client encoding and transport; the
                    # accepted event isolates waiting inside the serving engine.
                    queue = (s["commit_ts"] - s["accepted_unix_s"]) * 1000
                    if queue < -1:
                        raise ValueError("dispatch precedes the server accepted event")
                    queue = max(0.0, queue)  # commit_ts is rounded to milliseconds.
                else:
                    if s["recv_ts"] == 0:
                        raise ValueError(
                            "request creation time is missing; select server-accepted explicitly"
                        )
                    queue = s["queue_ms"]
            else:
                t = traces[rid]
                if (
                    t["status"] != "first_output"
                    or t["cached_tokens"] != cached_tokens
                    or t["prompt_tokens"] != prompt_tokens
                ):
                    raise ValueError("PD prefill trace disagrees with request stats")
                elapsed, queue = t["phases_ms"]["prefill"], t["phases_ms"]["admission"]
                queue_source = "pd-admission-phase"
            if (
                queue is None
                or not math.isfinite(queue)
                or queue < 0
                or queue > args.max_queue_ms
            ):
                raise ValueError("admission wait exceeds isolation threshold")
            seconds = positive(elapsed) / 1000
            accepted.append(
                {
                    "kind": "prefill",
                    "deployment": row["deployment"],
                    "rid": rid,
                    "split": row["split"],
                    "a": cached_tokens,
                    "b": prompt_tokens,
                    "seconds": seconds,
                    "timing_source": args.timing_source,
                    "queue_ms": queue,
                    "queue_source": queue_source,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"rid": rid, "reason": str(error)})
    write_rows(args.output, accepted)
    write_rows(args.rejected, rejected)
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}))
    if rejected or not accepted:
        raise ValueError(
            "incomplete/invalid calibration; inspect rejection file before collecting again"
        )
