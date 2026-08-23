#!/usr/bin/env python3
"""Phased client for the PD-disaggregation simulation bench.

evalscope's swe_smith plugin has no prime/measure phase control, so the
prime waves, the P-cached turn replay, and the measured waves all run
through this thin client. It replays the frozen agentic dataset, drives a
fixed concurrency with rolling admission, and emits a benchmark_summary.json
consumed by pd_sim's own collect_outputs.py (NOT column-compatible with the
parent bench's collect — pd_sim has its own).

Phases (--phase):
  p-fresh   unique first turns, max_tokens 1        -> prefill tok/s
  p-cached  turn-1 prime (500) then turn-2 measure  -> computed tok/s
  d-prime   first turns, max_tokens 1, low parallel  (no summary emitted)
  d-measure resend first turns, max_tokens N        -> output tok/s
"""

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import threading
import time
import urllib.request


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="chat/completions endpoint")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--dataset", required=True, help="frozen agentic_dataset.json")
    p.add_argument(
        "--phase",
        required=True,
        choices=["p-fresh", "p-cached", "d-prime", "d-measure"],
    )
    p.add_argument("--parallel", type=int, required=True)
    p.add_argument(
        "--number",
        type=int,
        required=True,
        help="requests to issue (conversations consumed)",
    )
    p.add_argument(
        "--offset", type=int, default=0, help="first conversation index to use"
    )
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--name", default="run", help="label for the output dir")
    p.add_argument("--outputs-dir", default="outputs")
    return p.parse_args()


def first_turn_messages(conv):
    return conv[0]["messages"]


class Runner:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.results = []

    def request(self, messages, max_tokens):
        body = json.dumps(
            {
                "model": self.args.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "ignore_eos": True,
            }
        ).encode()
        req = urllib.request.Request(
            self.args.url, data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.args.timeout) as r:
                out = json.load(r)
        except Exception:
            # One retry; a request that fails twice is recorded, not fatal —
            # the rung completes and collect VOIDs it on failures.
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(
                        self.args.url,
                        data=body,
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=self.args.timeout,
                ) as r:
                    out = json.load(r)
            except Exception as e:
                return {"failed": True, "error": str(e)[:200]}
        t1 = time.monotonic()
        usage = out.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        msg = out["choices"][0]["message"]
        return {
            "latency_s": t1 - t0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": details.get("cached_tokens") or 0,
            "content": msg.get("content"),
            "reasoning_content": msg.get("reasoning_content"),
        }

    def prime_item(self, conv):
        # P-cached prime: turn 1 with up to 500 generated tokens builds a
        # realistic prefix (50K prompt + completion). Excluded from timing.
        # BOTH reasoning_content and content are passed back verbatim: K3's
        # chat template renders them into the think/response channels, so the
        # re-rendered assistant turn retokenizes onto the cached token stream
        # instead of diverging at the <think> open tag.
        t1_msgs = list(first_turn_messages(conv))
        prime = self.request(t1_msgs, 500)
        if prime.get("failed") or len(conv) < 2:
            return None
        assistant = {"role": "assistant", "content": prime["content"] or ""}
        if prime.get("reasoning_content"):
            assistant["reasoning_content"] = prime["reasoning_content"]
        return t1_msgs + [assistant] + list(conv[1]["messages"])

    def one_item(self, payload):
        a = self.args
        if a.phase == "p-cached":
            msgs = payload  # pre-assembled turn-2 message list
        else:
            msgs = list(first_turn_messages(payload))
        res = self.request(msgs, a.max_tokens)
        with self.lock:
            self.results.append(res)
        return res

    # NOTE: failed requests carry {'failed': True} and are excluded from the
    # token/latency aggregates but counted in 'Failed Requests'.

    def run(self, convs):
        a = self.args
        picked = convs[a.offset : a.offset + a.number]
        if len(picked) < a.number:
            raise SystemExit(
                f"dataset has only {len(picked)} conversations at offset "
                f"{a.offset}; need {a.number}"
            )
        if a.phase == "p-cached":
            # Phase 1 (untimed): prime every conversation's turn-1 prefix.
            with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
                payloads = [m for m in ex.map(self.prime_item, picked) if m]
            if not payloads:
                raise SystemExit("no conversation has a second turn")
        else:
            payloads = picked
        # Phase 2 (timed): only the measured requests count toward wall time.
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
            list(ex.map(self.one_item, payloads))
        wall = time.monotonic() - t0
        return wall


BOUNDARIES = (
    "no KV-transfer cost modeled; prime-as-transfer is the core "
    "approximation; single machine; TTFT approximated by "
    "full-request latency at max_tokens 1; TPOT amortizes the "
    "cache-hit KV load into per-token time"
)


def _nearest_rank(sorted_vals, p):
    import math

    n = len(sorted_vals)
    if not n:
        return 0.0
    return sorted_vals[max(0, min(n - 1, math.ceil(p * n) - 1))]


def summarize(args, results, wall_s):
    failed = [r for r in results if r.get("failed")]
    ok = [r for r in results if not r.get("failed")]
    n = len(ok)
    prompt = sum(r["prompt_tokens"] for r in ok)
    cached = sum(r["cached_tokens"] for r in ok)
    output = sum(r["completion_tokens"] for r in ok)
    computed = prompt - cached
    lat = sorted(r["latency_s"] for r in ok)
    hit = 100.0 * cached / prompt if prompt else 0.0

    summary = {
        "Phase": args.phase,
        "Concurrency": args.parallel,
        "Requests": n,
        "Failed Requests": len(failed),
        "Wall (s)": round(wall_s, 3),
        "Prompt Tokens": prompt,
        "Cached Tokens": cached,
        "Computed Prompt Tokens": computed,
        "Output Tokens": output,
        "KV Cache Hit Rate (%)": round(hit, 4),
        "Prefill Throughput (tok/s)": round(prompt / wall_s, 4),
        "Computed Throughput (tok/s)": round(computed / wall_s, 4),
        "Output Throughput (tok/s)": round(output / wall_s, 4),
        "Requests/s": round(n / wall_s, 4),
        "Latency p50 (s)": round(_nearest_rank(lat, 0.50), 4),
        "Latency p99 (s)": round(_nearest_rank(lat, 0.99), 4),
        "Boundaries": BOUNDARIES,
    }
    if args.phase == "d-measure" and n:
        tpots = sorted(
            1000.0 * r["latency_s"] / max(1, r["completion_tokens"]) for r in ok
        )
        summary["TPOT p50 (ms)"] = round(_nearest_rank(tpots, 0.50), 4)
        summary["TPOT p99 (ms)"] = round(_nearest_rank(tpots, 0.99), 4)
    return summary


def main():
    args = parse_args()
    data = json.load(open(args.dataset))
    convs = data["conversations"]

    runner = Runner(args)
    wall = runner.run(convs)

    if args.phase == "d-prime":
        hit = summarize(args, runner.results, wall)["KV Cache Hit Rate (%)"]
        print(
            f"primed {len(runner.results)} conversations in {wall:.1f}s "
            f"(hit during prime: {hit:.1f}%)"
        )
        return

    summary = summarize(args, runner.results, wall)
    outdir = (
        f"{args.outputs_dir}/{args.name}/"
        f"parallel_{args.parallel}_number_{args.number}"
    )
    import os

    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())
