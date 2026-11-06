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

"""Single-request decode speed and accuracy on GSM8K.

Sends the first ``--num-prompts`` questions of the GSM8K test split to an
OpenAI-compatible chat endpoint one at a time (greedy, streamed) and reports:

- the per-request decode rate: output tokens after the first, divided by the
  time from the first to the last streamed token, averaged over requests
  (time to first token is not part of it);
- the end-to-end output rate: all output tokens over the wall time, per GPU;
- the accuracy: the last ``\\boxed{}`` answer against the reference answer.

The summary ends with the CSV block that ``test/ci_system/pipeline.py`` gates
a perf task's ``perf_reference`` on. The script exits with status 1 when the
accuracy is below ``--min-accuracy``.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# `pipeline.py` reads the perf summary from this exact CSV header.
CSV_HEADER = (
    "config,Conc.,Latency (tps/user),Throughput (tps/gpu),"
    "Approx Cache Hit,Decoded Tok/Iter"
)


@dataclass
class RequestResult:
    output_tokens: int
    first_token_s: float
    last_token_s: float
    text: str

    @property
    def decode_tok_s(self) -> float | None:
        """Output tokens after the first over the time they took to stream."""
        span = self.last_token_s - self.first_token_s
        if self.output_tokens < 2 or span <= 0:
            return None
        return (self.output_tokens - 1) / span


def last_boxed(text: str) -> str | None:
    """The contents of the last ``\\boxed{...}`` in ``text``, braces balanced."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def normalize_number(answer: str) -> float | None:
    """A numeric answer as a float: drops ``$``, thousands separators,
    spaces, a trailing ``.`` and a ``\\text{...}`` unit."""
    text = answer.strip()
    unit = text.find("\\text{")
    if unit >= 0:
        text = text[:unit]
    for token in ("\\$", "$", "\\!", "\\,", ",", " "):
        text = text.replace(token, "")
    text = text.rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def is_correct(response: str, reference: str) -> bool:
    """Whether the last boxed answer of ``response`` equals ``reference``.

    ``reference`` is a GSM8K answer field; its value follows ``####``.
    """
    boxed = last_boxed(response)
    if boxed is None:
        return False
    got = normalize_number(boxed)
    want = normalize_number(reference.split("####")[-1])
    return got is not None and want is not None and got == want


def load_gsm8k(num_prompts: int) -> list[tuple[str, str]]:
    """The first ``num_prompts`` (question, answer) pairs of the test split,
    from the Hugging Face cache (``HF_HOME``)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"
    )
    rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    if len(rows) < num_prompts:
        raise ValueError(f"GSM8K test has {len(rows)} questions, not {num_prompts}")
    return [(row["question"], row["answer"]) for row in rows[:num_prompts]]


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    chat_template_kwargs: dict,
    timeout_s: float,
) -> RequestResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": chat_template_kwargs,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = last = None
    parts: list[str] = []
    output_tokens = None
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                output_tokens = chunk["usage"]["completion_tokens"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("reasoning_content") or "") + (
                    delta.get("content") or ""
                )
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    parts.append(piece)
    if output_tokens is None or first is None:
        raise RuntimeError("the stream ended without tokens or usage")
    return RequestResult(output_tokens, first, last, "".join(parts))


def summary_csv(
    config: str, results: list[RequestResult], wall_s: float, num_gpus: int
) -> str:
    rates = [r.decode_tok_s for r in results if r.decode_tok_s is not None]
    per_user = statistics.mean(rates)
    per_gpu = sum(r.output_tokens for r in results) / wall_s / num_gpus
    return f"{CSV_HEADER}\n{config},1,{per_user:.2f},{per_gpu:.2f},N/A,N/A\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True, help="Label of the CSV row.")
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument(
        "--chat-template-kwargs", type=json.loads, required=True, help="JSON object."
    )
    parser.add_argument("--min-accuracy", type=float, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Per-request JSONL.")
    args = parser.parse_args()

    prompts = load_gsm8k(args.num_prompts)
    results: list[RequestResult] = []
    correct = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with args.output.open("w") as out:
        for i, (question, answer) in enumerate(prompts):
            result = stream_chat(
                args.base_url,
                args.model,
                question + PROMPT_SUFFIX,
                args.max_tokens,
                args.chat_template_kwargs,
                args.timeout_s,
            )
            ok = is_correct(result.text, answer)
            correct += ok
            results.append(result)
            out.write(
                json.dumps(
                    {
                        "index": i,
                        "correct": ok,
                        "output_tokens": result.output_tokens,
                        "decode_tok_s": result.decode_tok_s,
                        "reference": answer.split("####")[-1].strip(),
                        "boxed": last_boxed(result.text),
                    }
                )
                + "\n"
            )
    wall_s = time.perf_counter() - start

    accuracy = correct / len(prompts)
    at_cap = sum(r.output_tokens >= args.max_tokens for r in results)
    rates = [r.decode_tok_s for r in results if r.decode_tok_s is not None]
    print(
        f"GSM8K single request: {correct}/{len(prompts)} correct "
        f"(accuracy {accuracy:.4f}, minimum {args.min_accuracy}), "
        f"{at_cap} at the {args.max_tokens}-token cap"
    )
    print(
        f"decode tok/s per request: mean {statistics.mean(rates):.1f}, "
        f"median {statistics.median(rates):.1f}; "
        f"mean output tokens {statistics.mean(r.output_tokens for r in results):.1f}"
    )
    print()
    print(summary_csv(args.config, results, wall_s, args.num_gpus))
    if accuracy < args.min_accuracy:
        print(f"accuracy {accuracy:.4f} is below {args.min_accuracy}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
