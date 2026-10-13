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

"""Batch-invariance probe against a concurrency-1 greedy GSM8K run.

``--ref`` is a ``gsm8k_eval.py`` result taken at ``--conc 1``. ``prefill`` mode
sends every prompt at once for one token, so the first token is decided inside
a large prefill batch; ``decode`` mode staggers arrivals by ``--gap`` so each
request decodes alongside a changing batch. A deterministic engine reproduces
the reference text in both modes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from test.hy_v4_parity.gsm8k_eval import complete, fewshot_prefix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8200/v1/completions")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--ref", required=True, help="gsm8k_eval.py --conc 1 output")
    parser.add_argument("--train", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--mode", choices=["prefill", "decode"], required=True)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--gap", type=float, default=0.6)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prefix = fewshot_prefix(args.train, 5)
    with open(args.data) as f:
        test = [json.loads(line) for line in f]
    with open(args.ref) as f:
        reference = {r["i"]: r["text"] for r in json.load(f)["res"]}
    ids = list(range(args.n))

    def call(index: int, max_tokens: int) -> str:
        return complete(
            args.url,
            {
                "model": args.model,
                "prompt": prefix + f"Question: {test[index]['question']}\nAnswer:",
                "max_tokens": max_tokens,
                "temperature": 0,
                "stop": ["Question:", "\n\n"],
            },
            retries=1,
        )

    outputs: dict[int, str] = {}
    with cf.ThreadPoolExecutor(len(ids)) as pool:
        futures = {}
        for index in ids:
            max_tokens = 1 if args.mode == "prefill" else args.max_tokens
            futures[pool.submit(call, index, max_tokens)] = index
            if args.mode == "decode":
                time.sleep(args.gap)
        for future, index in futures.items():
            outputs[index] = future.result()
    agree = sum(reference[i][: len(outputs[i])] == outputs[i] for i in ids)
    with open(args.out, "w") as f:
        json.dump(outputs, f, ensure_ascii=False)
    print(f"mode={args.mode} n={len(ids)} agree_with_c1={agree}")


if __name__ == "__main__":
    main()
