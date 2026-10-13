#!/usr/bin/env python3
"""Prefill conversation replay and cached-prompt decode simulation.

p-measure replays every user turn with one generated token, inserting prepared
500-token synthetic assistant messages between turns. Each worker advances one
conversation sequentially, then takes the next conversation from the pool.
d-prime caches the concatenated user-turn prompt minus its last rendered token;
d-measure generates from the full prompt.

Every request carries rid=pdsim-conv-<index>, so a gateway configured with
sticky sessions keeps a conversation on the same attention-DP rank.
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

from transformers import AutoTokenizer

TARGET_RESPONSE_TOKENS = 500


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="chat/completions endpoint")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--dataset", required=True, help="agentic_dataset_x128.json")
    p.add_argument(
        "--phase",
        required=True,
        choices=["p-measure", "d-prime", "d-measure"],
    )
    p.add_argument("--parallel", type=int, required=True)
    p.add_argument(
        "--number",
        type=int,
        required=True,
        help="conversations to replay",
    )
    p.add_argument(
        "--offset", type=int, default=0, help="first conversation index to use"
    )
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--name", default="run", help="label for the output dir")
    p.add_argument("--outputs-dir", default="outputs")
    args = p.parse_args()
    if args.phase == "p-measure" and args.max_tokens != 1:
        p.error("p-measure requires --max-tokens 1")
    return args


def unique_first_turn_conversations(convs):
    """Deduplicate first-turn prefixes while preserving conversation order."""
    seen = set()
    out = []
    for conv in convs:
        key = json.dumps(conv[0]["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(conv)
    return out


class Runner:
    def __init__(self, args):
        self.args = args
        self.opener = urllib.request.build_opener()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.results = []

    def request(self, messages, max_tokens, rid):
        stream = self.args.phase == "d-measure"
        body = json.dumps(
            {
                "model": self.args.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "rid": rid,
                "stream": stream,
                "stream_options": {"include_usage": True} if stream else None,
            }
        ).encode()
        req = urllib.request.Request(
            self.args.url, data=body, headers={"Content-Type": "application/json"}
        )
        first_token = last_token = None
        usage = None
        t0 = time.monotonic()
        try:
            with self.opener.open(req, timeout=self.args.timeout) as r:
                if stream:
                    for line in r:
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        chunk = json.loads(data)
                        if "error" in chunk:
                            raise RuntimeError(chunk["error"])
                        if chunk.get("usage") is not None:
                            usage = chunk["usage"]
                        for choice in chunk["choices"]:
                            delta = choice["delta"]
                            if delta.get("content") or delta.get("reasoning_content"):
                                last_token = time.monotonic()
                                if first_token is None:
                                    first_token = last_token
                    else:
                        raise RuntimeError("Response stream ended before [DONE]")
                    if usage is None or first_token is None:
                        raise RuntimeError("Response stream lacks usage or output text")
                    if usage["completion_tokens"] < 2:
                        raise RuntimeError(
                            "Decode TPOT requires at least two output tokens"
                        )
                else:
                    usage = json.load(r)["usage"]
        except Exception as e:
            raise SystemExit(f"request {rid} failed: {str(e)[:200]}") from e
        t1 = time.monotonic()
        details = usage.get("prompt_tokens_details") or {}
        result = {
            "latency_s": t1 - t0,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cached_tokens": details.get("cached_tokens") or 0,
        }
        if stream:
            result["ttft_s"] = first_token - t0
            result["decode_s"] = last_token - first_token
        return result

    def prime(self, item):
        if self.stop_event.is_set():
            return
        index, tokens = item
        rid = f"pdsim-conv-{index}"
        body = json.dumps(
            {
                "model": self.args.model,
                "input_ids": tokens,
                "sampling_params": {
                    "max_new_tokens": self.args.max_tokens,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                "rid": rid,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            self.args.url.removesuffix("/v1/chat/completions") + "/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(req, timeout=self.args.timeout) as r:
                reason = json.load(r)[0]["meta_info"]["finish_reason"]
                if reason["type"] not in ("length", "stop"):
                    raise RuntimeError(reason)
        except Exception as e:
            raise SystemExit(f"request {rid} failed: {str(e)[:200]}") from e

    def prepare_prime(self, items):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model, trust_remote_code=True, local_files_only=True
        )
        prepared = []
        for index, conv in items:
            messages = [message for turn in conv for message in turn["messages"]]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            # Save the state before the final token that prefix reuse must replay.
            prepared.append((index, tokens[:-1]))
        return prepared

    def prepare_prefill(self, items):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model, trust_remote_code=True, local_files_only=True
        )
        vocab = sorted(
            set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids)
        )
        prepared = []
        for index, conv in items:
            rng = random.Random(index)
            responses = []
            for _ in range(len(conv) - 1):
                for _ in range(10):
                    text = tokenizer.decode(
                        rng.choices(vocab, k=2 * TARGET_RESPONSE_TOKENS)
                    )
                    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
                    text = tokenizer.decode(tokens[:TARGET_RESPONSE_TOKENS])
                    if (
                        len(tokenizer(text, add_special_tokens=False)["input_ids"])
                        == TARGET_RESPONSE_TOKENS
                    ):
                        break
                else:
                    raise RuntimeError(
                        f"Could not prepare a {TARGET_RESPONSE_TOKENS}-token fake response"
                    )
                responses.append({"role": "assistant", "content": text})
            prepared.append((index, (conv, responses)))
        return prepared

    def one_item(self, item):
        a = self.args
        index, payload = item
        if self.stop_event.is_set():
            return
        rid = f"pdsim-conv-{index}"
        if a.phase == "p-measure":
            conv, responses = payload
            messages = []
            for turn_index, turn in enumerate(conv):
                if self.stop_event.is_set():
                    return
                messages.extend(turn["messages"])
                res = self.request(messages, a.max_tokens, rid)
                with self.lock:
                    self.results.append(res)
                if turn_index < len(responses):
                    messages.append(responses[turn_index])
        else:
            messages = [message for turn in payload for message in turn["messages"]]
            res = self.request(messages, a.max_tokens, rid)
            with self.lock:
                self.results.append(res)

    def _map(self, fn, items):
        # A failed request raises; cancel the queued ones so the run stops
        # after the in-flight requests instead of sending the rest.
        with cf.ThreadPoolExecutor(max_workers=self.args.parallel) as ex:
            futures = [ex.submit(fn, item) for item in items]
            try:
                for future in cf.as_completed(futures):
                    future.result()
            except BaseException:
                self.stop_event.set()
                ex.shutdown(cancel_futures=True)
                raise

    def run(self, convs):
        a = self.args
        picked = convs[a.offset : a.offset + a.number]
        if len(picked) < a.number:
            raise SystemExit(
                f"dataset has only {len(picked)} conversations at offset "
                f"{a.offset}; need {a.number}"
            )
        # Items carry the conversation's absolute index so every phase sends
        # the same rid for the same conversation.
        items = list(enumerate(picked, start=a.offset))
        request = self.one_item
        if a.phase == "p-measure":
            payloads = self.prepare_prefill(items)
        elif a.phase == "d-prime":
            payloads = self.prepare_prime(items)
            request = self.prime
        else:
            payloads = items
        # Tokenization and synthetic response generation are outside the timed run.
        t0 = time.monotonic()
        self._map(request, payloads)
        wall = time.monotonic() - t0
        return wall


BOUNDARIES = (
    "no KV-transfer cost modeled; prime-as-transfer is the core "
    "approximation; single deployment; P request latency includes queueing "
    "at max_tokens 1; D TPOT excludes TTFT; throughput includes the full wave"
)


def _nearest_rank(sorted_vals, p):
    n = len(sorted_vals)
    if not n:
        return 0.0
    return sorted_vals[max(0, min(n - 1, math.ceil(p * n) - 1))]


def summarize(args, results, wall_s):
    ok = results
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
        "Conversations": args.number,
        "Target Response Tokens": TARGET_RESPONSE_TOKENS,
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
        "Latency mean (s)": round(statistics.mean(lat), 4) if lat else 0.0,
        "Latency p50 (s)": round(_nearest_rank(lat, 0.50), 4),
        "Latency p99 (s)": round(_nearest_rank(lat, 0.99), 4),
        "Boundaries": BOUNDARIES,
    }
    if args.phase == "d-measure" and n:
        ttfts = sorted(r["ttft_s"] for r in ok)
        summary["TTFT p50 (s)"] = round(_nearest_rank(ttfts, 0.50), 4)
        summary["TTFT p99 (s)"] = round(_nearest_rank(ttfts, 0.99), 4)
        tpots = sorted(
            1000.0 * r["decode_s"] / (r["completion_tokens"] - 1) for r in ok
        )
        summary["TPOT p50 (ms)"] = round(_nearest_rank(tpots, 0.50), 4)
        summary["TPOT p99 (ms)"] = round(_nearest_rank(tpots, 0.99), 4)
    return summary


def main():
    args = parse_args()
    with open(args.dataset) as f:
        data = json.load(f)
    convs = unique_first_turn_conversations(data["conversations"])

    runner = Runner(args)
    wall = runner.run(convs)

    if args.phase == "d-prime":
        print(f"Primed {args.number} conversations in {wall:.1f}s")
        return

    summary = summarize(args, runner.results, wall)
    outdir = (
        f"{args.outputs_dir}/{args.name}/"
        f"parallel_{args.parallel}_number_{args.number}"
    )
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())
