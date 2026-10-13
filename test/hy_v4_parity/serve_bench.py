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
"""Minimal, engine-agnostic serving benchmark (OpenAI /v1/completions).

Both engines receive byte-identical requests: random token ids from a fixed
seed, decoded to text with the model tokenizer (smg rejects token-id prompts),
``ignore_eos`` so every request decodes exactly ``--output-len`` tokens, and
streaming so TTFT and per-token latency can be measured.

    python -m test.hy_v4_parity.serve_bench --port 8200 --model M \
        --tokenizer /path/to/checkpoint --input-len 1024 --output-len 256 \
        --concurrency 1,8,32 --num-prompts 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

_TOK = None


def make_prompts(n: int, input_len: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        # Stay inside the regular-token range of the 120832-entry Hy4 vocab;
        # decode->encode drifts a little, so trim to the requested length.
        ids = [rng.randint(1000, 100000) for _ in range(input_len + 64)]
        ids = _TOK.encode(_TOK.decode(ids), add_special_tokens=False)[:input_len]
        out.append(_TOK.decode(ids))
    return out


async def one_request(session, url, model, prompt, output_len):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    last = None
    complete = False
    usage = None
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                complete = True
                break
            chunk = json.loads(data)
            if chunk.get("error") is not None:
                raise RuntimeError("completion stream returned an error event")
            if chunk.get("usage"):
                usage = chunk["usage"]
            if not chunk.get("choices"):
                continue
            text = chunk["choices"][0].get("text")
            if not isinstance(text, str) or text == "":
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
    end = time.perf_counter()
    if not complete or first is None or last is None:
        raise RuntimeError("incomplete completion stream or no visible output")
    if not isinstance(usage, dict):
        raise RuntimeError("completion stream omitted token usage")
    n_out = usage.get("completion_tokens")
    n_in = usage.get("prompt_tokens")
    if type(n_out) is not int or n_out <= 0 or type(n_in) is not int or n_in < 0:
        raise RuntimeError("completion stream returned invalid token usage")
    if n_out != output_len:
        raise RuntimeError(
            f"completion produced {n_out} tokens, expected {output_len} with ignore_eos"
        )
    ttft = first - start
    # Do not charge a delayed usage/footer frame to the token generation span.
    tpot = (last - first) / max(n_out - 1, 1)
    return ttft, tpot, end - start, n_out, n_in


async def run_level(url, model, prompts, output_len, concurrency):
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    results = []

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def guarded(p):
            async with sem:
                results.append(await one_request(session, url, model, p, output_len))

        t0 = time.perf_counter()
        await asyncio.gather(*(guarded(p) for p in prompts))
        wall = time.perf_counter() - t0
    ttfts = sorted(r[0] for r in results)
    tpots = sorted(r[1] for r in results)
    out_tokens = sum(r[3] for r in results)
    in_tokens = sum(r[4] for r in results)
    pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {
        "metrics_version": 2,
        "concurrency": concurrency,
        "num_prompts": len(prompts),
        "avg_in": round(in_tokens / len(results), 1),
        "avg_out": round(out_tokens / len(results), 1),
        "wall_s": round(wall, 2),
        "output_tok_per_s": round(out_tokens / wall, 1),
        "total_tok_per_s": round((out_tokens + in_tokens) / wall, 1),
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1e3, 1),
        "ttft_ms_p99": round(pct(ttfts, 0.99) * 1e3, 1),
        "tpot_ms_mean": round(statistics.mean(tpots) * 1e3, 2),
        "tpot_ms_p99": round(pct(tpots, 0.99) * 1e3, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--num-prompts", type=int, default=0, help="0 = 4x concurrency")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    global _TOK
    import os

    from transformers import PreTrainedTokenizerFast

    # Load tokenizer.json directly: AutoTokenizer would validate the Hy4 config.
    _TOK = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(args.tokenizer, "tokenizer.json")
    )
    url = f"http://{args.host}:{args.port}/v1/completions"
    rows = []
    # Warmup: one short request so lazy JIT / graph capture is not timed.
    asyncio.run(run_level(url, args.model, make_prompts(2, 64, seed=99), 8, 2))
    for c in [int(x) for x in args.concurrency.split(",")]:
        n = args.num_prompts or max(4 * c, 8)
        # A distinct seed per level, so no level reuses another's prefix cache.
        prompts = make_prompts(n, args.input_len, seed=c)
        row = run_level(url, args.model, prompts, args.output_len, c)
        row = asyncio.run(row)
        row["tag"] = args.tag
        print(json.dumps(row), flush=True)
        rows.append(row)
    if args.out:
        with open(args.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
