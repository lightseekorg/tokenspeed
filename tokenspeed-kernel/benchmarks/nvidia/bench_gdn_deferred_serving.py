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

"""Measure deferred GDN serving with exact native-stream token counts.

Run against separately restarted off/on servers with identical configurations.
After warmup responses finish, wait for the engine's running requests, waiting
requests and active KV usage to reach zero through its metrics endpoint. An
overlapped stopped forward can outlive its client's last response.
ITL divides each stream interval by its emitted token count and weights that
interval by those tokens; speculation can emit several tokens together.
Acceptance counts are emitted tokens per stream update, including the target
token. Server logs also retain accepted lengths for queued stopped rows.
"""

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path

import aiohttp


def engine_is_idle(metrics: str) -> bool:
    """Require every rank's request queues and active KV pages to be empty."""
    activity = {
        "tokenspeed:num_requests_running": [],
        "tokenspeed:num_requests_waiting": [],
        "tokenspeed:kv_cache_usage_perc": [],
    }
    for line in metrics.splitlines():
        sample, separator, value = line.strip().rpartition(" ")
        name = sample.partition("{")[0]
        if separator and name in activity:
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"invalid engine activity metric: {name}")
            activity[name].append(number)
    missing = [name for name, values in activity.items() if not values]
    if missing:
        raise ValueError(f"engine metrics are missing: {', '.join(missing)}")
    return all(value == 0 for values in activity.values() for value in values)


async def wait_until_idle(
    session: aiohttp.ClientSession, metrics_url: str, *, timeout_s: float
) -> float:
    """Wait outside measurement for warmup's GPU work and ownership to drain."""
    start = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_s):
            while True:
                async with session.get(metrics_url) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"engine metrics returned HTTP {response.status}"
                        )
                    if engine_is_idle(await response.text()):
                        return (time.perf_counter() - start) * 1000
                await asyncio.sleep(0.005)
    except TimeoutError as error:
        raise TimeoutError("engine did not become idle after warmup") from error


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (
        position - low
    )


async def measure(args):
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=1200)
    # Match both client limits; the HTTP pool otherwise caps concurrency at 100.
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def request(index, tokens):
            async with semaphore:
                prompt = (
                    f"Task {index}: Continue listing consecutive positive integers "
                    "in ascending order, separated by commas. Write at least 500 integers.\n"
                    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,"
                )
                payload = {
                    "model": args.model,
                    "text": prompt,
                    "stream": True,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": tokens,
                        "ignore_eos": True,
                    },
                }
                start = time.perf_counter()
                previous_time = None
                previous_tokens = 0
                first = None
                intervals = []
                counts = Counter()
                finish = None
                async with session.post(
                    args.url + "/generate", json=payload
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"generation returned HTTP {response.status}: {await response.text()}"
                        )
                    async for raw in response.content:
                        if (
                            not raw.startswith(b"data: ")
                            or raw.strip() == b"data: [DONE]"
                        ):
                            continue
                        item = json.loads(raw[6:])
                        info = item["meta_info"]
                        count = info["completion_tokens"]
                        now = time.perf_counter()
                        delta = count - previous_tokens
                        if delta > 0:
                            if previous_time is None:
                                first = now - start
                            else:
                                intervals.extend(
                                    [(now - previous_time) * 1000 / delta] * delta
                                )
                                counts[delta] += 1
                            previous_time, previous_tokens = now, count
                        if info.get("finish_reason") is not None:
                            finish = info["finish_reason"]
                if previous_tokens != tokens or finish != "length":
                    raise RuntimeError(
                        f"incomplete generation: {previous_tokens} tokens, finish {finish}"
                    )
                return previous_tokens, first * 1000, intervals, counts

        await asyncio.gather(
            *(request(-index - 1, 32) for index in range(args.concurrency))
        )
        warmup_drain_ms = await wait_until_idle(session, args.metrics_url, timeout_s=30)
        start = time.perf_counter()
        results = await asyncio.gather(
            *(request(index, args.output_tokens) for index in range(args.requests))
        )
        elapsed = time.perf_counter() - start
    intervals = [interval for result in results for interval in result[2]]
    accepted = sum((result[3] for result in results), Counter())
    tokens = sum(result[0] for result in results)
    first_tokens = [row[1] for row in results]
    result = {
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "tokens_per_request": args.output_tokens,
        "warmup_completion": "engine_idle",
        "warmup_drain_ms": warmup_drain_ms,
        "elapsed_s": elapsed,
        "output_tokens": tokens,
        "output_tokens_s": tokens / elapsed,
        "itl_p50_ms": percentile(intervals, 0.5),
        "itl_p95_ms": percentile(intervals, 0.95),
        "ttft_p50_ms": percentile(first_tokens, 0.5),
        "ttft_p95_ms": percentile(first_tokens, 0.95),
        "ttft_p99_ms": percentile(first_tokens, 0.99),
        "acceptance_histogram": dict(sorted(accepted.items())),
        "mean_emitted_tokens_per_decode": sum(k * v for k, v in accepted.items())
        / sum(accepted.values()),
    }
    print(json.dumps(result, indent=2), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--metrics-url",
        required=True,
        help="Engine /metrics URL printed at startup; requires --enable-metrics.",
    )
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(measure(parser.parse_args()))
