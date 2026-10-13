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

"""Greedy few-shot GSM8K over an OpenAI ``/v1/completions`` endpoint.

Engine-agnostic, so the same request stream scores TokenSpeed and a reference
engine side by side:

    python -m test.hy_v4_parity.gsm8k_eval --data test.jsonl --train train.jsonl \
        --model hy4 --out ts.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8200/v1/completions")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--data", required=True, help="GSM8K test jsonl")
    parser.add_argument("--train", required=True, help="GSM8K train jsonl (shots)")
    parser.add_argument("--n", type=int, default=1319)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--conc", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def fewshot_prefix(train_path: str, shots: int) -> str:
    with open(train_path) as f:
        train = [json.loads(line) for line in f][:shots]
    return "".join(
        f"Question: {t['question']}\n"
        f"Answer: {t['answer'].replace('####', 'The answer is')}\n\n"
        for t in train
    )


def extract_number(text: str) -> float | None:
    text = text.replace(",", "")
    matches = re.findall(r"The answer is\s*\$?(-?[\d.]+)", text)
    if not matches:
        matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return None
    try:
        return float(matches[-1].rstrip("."))
    except ValueError:
        return None


def complete(url: str, body: dict, retries: int = 5) -> str:
    for _ in range(retries):
        try:
            request = urllib.request.Request(
                url, json.dumps(body).encode(), {"Content-Type": "application/json"}
            )
            response = json.loads(urllib.request.urlopen(request, timeout=600).read())
            return response["choices"][0]["text"]
        except Exception as exc:  # noqa: BLE001 -- recorded as a failed sample
            error = f"ERR {exc}"
            time.sleep(2)
    return error


def main() -> None:
    args = _parse_args()
    prefix = fewshot_prefix(args.train, args.shots)
    with open(args.data) as f:
        test = [json.loads(line) for line in f][: args.n]

    def run(index: int) -> dict:
        sample = test[index]
        text = complete(
            args.url,
            {
                "model": args.model,
                "prompt": prefix + f"Question: {sample['question']}\nAnswer:",
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stop": ["Question:", "\n\n"],
            },
        )
        gold = float(sample["answer"].split("####")[-1].strip().replace(",", ""))
        pred = extract_number(text.split("Question:")[0])
        return {
            "i": index,
            "gold": gold,
            "pred": pred,
            "ok": pred is not None and abs(pred - gold) < 1e-4,
            "text": text,
        }

    start = time.time()
    with cf.ThreadPoolExecutor(args.conc) as pool:
        results = sorted(pool.map(run, range(len(test))), key=lambda r: r["i"])
    accuracy = sum(r["ok"] for r in results) / len(results)
    errors = sum(r["text"].startswith("ERR") for r in results)
    with open(args.out, "w") as f:
        json.dump(
            {"acc": accuracy, "n": len(results), "errs": errors, "res": results},
            f,
            ensure_ascii=False,
        )
    print(
        f"GSM8K acc={accuracy:.4f} n={len(results)} errs={errors} "
        f"time={time.time() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
