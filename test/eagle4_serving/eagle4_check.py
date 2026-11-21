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

"""EAGLE-4 serving checks for Qwen3.8-27B on one GPU.

Three speculative modes, each served with ``--attention-backend mha``:

- ``chain``: static chain, top-k 1, 7 steps, 8 draft tokens;
- ``tree``: static tree, top-k 4, 5 steps, 16 draft tokens;
- ``harness``: EAGLE-Harness (``--speculative-exec-mode 0``), top-k 4,
  12 steps, 16 draft tokens. Its cost table is measured first
  (``eagle_harness.measure_run``) on the same GPU, for batch sizes up to the
  server's ``--max-num-seqs``, unless ``--harness-data`` names one.

Subcommands:

``smoke``
    Serve each mode in turn, send the prompts with ``--concurrency`` requests
    in flight, and fail when a request fails or a mode's mean acceptance
    length (accepted tokens per verify step, from the server log) is below
    ``--min-accept-length``.

``parity``
    Greedy decode the prompts once per mode and then once without
    speculation (the reference), and compare each mode's tokens with the
    reference up to the first difference. Greedy speculative decoding should
    emit the reference tokens; a difference passes when it is a numerical tie:

    - ``near_tie``: the reference server's logprobs of its token and the
      mode's token at that position differ by at most ``TIE_GAP`` (one bf16 ulp
      of a logit in [16, 32) is 0.125). Both logprobs are read as prompt
      logprobs of the shared prefix followed by either token, since the
      server returns top-k logprobs only for a prefill;
    - otherwise the target model recomputes the logits of that position in
      fp32 (Hugging Face Transformers, eager attention): ``fp32_argmax`` when
      the mode's token is the fp32 argmax, ``fp32_pass`` when the fp32 logit
      gap between the two tokens is at most ``TOLERANCE_ULPS`` bf16 ulps of
      their magnitude, and ``fail`` otherwise.

``serve``
    Serve one mode in the foreground (measuring the harness cost table
    first), for a task's managed server.

``accept-length``
    Print the mean acceptance length of a server log, and fail below
    ``--min-accept-length``.

All servers run on one GPU, one after another; ``parity`` loads the fp32
model after the last server stops.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

TARGET_MODEL = "Qwen/Qwen3.8-27B"
DRAFT_MODEL = "EAGLE-Corp/Qwen3.8-27B-EAGLE4"

# Reference logprob gap at or below which a different token is a tie.
TIE_GAP = 0.13
# fp32 logit gap, in bf16 ulps of the logits, at or below which a different token passes.
TOLERANCE_ULPS = 2

PASSING_STATUSES = ("identical", "length", "near_tie", "fp32_argmax", "fp32_pass")

# Self-written prompts across task kinds, so draft acceptance varies.
PROMPTS = (
    "Write a Python function that returns the n-th Fibonacci number using "
    "memoization, and explain its time complexity.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip in "
    "minutes? Show your reasoning.",
    "Explain the difference between a process and a thread to a new programmer.",
    "Write a short story of about 150 words about a lighthouse keeper who "
    "finds a message in a bottle.",
    "Translate into French, German and Spanish: 'The library opens at eight "
    "and closes at six on weekdays.'",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Write a SQL query that returns the three customers with the highest total "
    "order value from tables customers(id, name) and orders(id, customer_id, "
    "amount).",
    "If a rectangle has a perimeter of 36 cm and its length is twice its width, "
    "what is its area?",
    "You are a travel guide in Kyoto. Recommend a one-day itinerary for a "
    "first-time visitor.",
    "List the planets of the solar system in order from the Sun, with one fact "
    "about each.",
    "What is the derivative of x^3 * sin(x)? Explain each step.",
    "Write a bash one-liner that counts the lines of every .py file under the "
    "current directory and prints the total.",
    "Compare the advantages and disadvantages of electric cars and petrol cars.",
    "请用中文解释什么是机器学习,并举两个日常生活中的例子。",
    "Write a haiku about winter mornings, then explain its imagery.",
    "Prove that the sum of two even integers is even.",
)


class SpecMode(NamedTuple):
    topk: int
    num_steps: int
    num_draft_tokens: int
    harness: bool


MODES = {
    "chain": SpecMode(topk=1, num_steps=7, num_draft_tokens=8, harness=False),
    "tree": SpecMode(topk=4, num_steps=5, num_draft_tokens=16, harness=False),
    "harness": SpecMode(topk=4, num_steps=12, num_draft_tokens=16, harness=True),
}
HARNESS_COMPUTE_INTENSITY = "high_intensity"

_ACCEPT_RE = re.compile(r"Accept_num_tokens_avg:\s*([0-9]+(?:\.[0-9]+)?)")


def bf16_ulp(magnitude: float) -> float:
    """The spacing of bf16 values (8 significand bits) around ``magnitude``."""
    return 2.0 ** (math.floor(math.log2(max(abs(magnitude), 2.0**-126))) - 7)


def first_difference(reference: list[int], candidate: list[int]) -> int | None:
    """The first position where the two token lists differ, or None when one
    is a prefix of the other."""
    for index, (ref, cand) in enumerate(zip(reference, candidate)):
        if ref != cand:
            return index
    return None


def fp32_status(logits: list[float], ref: int, cand: int) -> str:
    """Judge a difference by the fp32 logits of its position."""
    if max(range(len(logits)), key=logits.__getitem__) == cand:
        return "fp32_argmax"
    gap = logits[ref] - logits[cand]
    ulp = bf16_ulp(max(abs(logits[ref]), abs(logits[cand])))
    return "fp32_pass" if gap <= TOLERANCE_ULPS * ulp else "fail"


def accept_lengths(log_text: str) -> list[float]:
    """Each finished request's mean acceptance length in a server log."""
    return [float(value) for value in _ACCEPT_RE.findall(log_text)]


def serve_command(
    mode: str | None,
    port: int,
    max_num_seqs: int,
    harness_data: Path | None,
    output_logprobs: bool,
) -> list[str]:
    command = [
        "ts",
        "serve",
        "--model",
        TARGET_MODEL,
        "--tensor-parallel-size",
        "1",
        "--attention-backend",
        "mha",
        "--sampling-backend",
        "greedy",
        "--mamba-ssm-dtype",
        "float32",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        str(max_num_seqs),
        "--disable-kvstore",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if output_logprobs:
        command.append("--enable-output-logprobs")
    if mode is None:
        return command
    spec = MODES[mode]
    command += [
        "--speculative-algorithm",
        "EAGLE4",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--speculative-eagle-topk",
        str(spec.topk),
        "--speculative-num-steps",
        str(spec.num_steps),
        "--speculative-num-draft-tokens",
        str(spec.num_draft_tokens),
    ]
    if spec.harness:
        command += [
            "--speculative-exec-mode",
            "0",
            "--speculative-compute-intensity",
            HARNESS_COMPUTE_INTENSITY,
            "--speculative-data-path",
            str(harness_data),
        ]
    return command


def measure_command(run_dir: Path, max_bs: int) -> list[str]:
    spec = MODES["harness"]
    return [
        sys.executable,
        "-m",
        "tokenspeed.runtime.execution.drafter.eagle_harness.measure_run",
        "--model",
        TARGET_MODEL,
        "--draft-model",
        DRAFT_MODEL,
        "--run-dir",
        str(run_dir),
        "--tp-size",
        "1",
        "--attention-backend",
        "mha",
        "--compute-intensity",
        HARNESS_COMPUTE_INTENSITY,
        "--topk",
        str(spec.topk),
        "--num-steps",
        str(spec.num_steps),
        "--num-draft-tokens",
        str(spec.num_draft_tokens),
        "--max-bs",
        str(max_bs),
        "--mamba-ssm-dtype",
        "float32",
        "--fast",
    ]


def run_logged(command: list[str], log_path: Path) -> None:
    print(f"[eagle4] $ {' '.join(command)} > {log_path}", flush=True)
    start = time.monotonic()
    with log_path.open("w") as log:
        returncode = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    if returncode != 0:
        print_tail(log_path)
        raise RuntimeError(f"{command[0]} exited with {returncode}; log: {log_path}")
    print(f"[eagle4] done in {time.monotonic() - start:.0f} s", flush=True)


def print_tail(log_path: Path, lines: int = 40) -> None:
    tail = log_path.read_text(errors="replace").splitlines()[-lines:]
    print("\n".join(f"  | {line}" for line in tail), flush=True)


class Server:
    """One ``ts serve`` process group, its output in ``log_path``."""

    def __init__(self, command: list[str], port: int, log_path: Path) -> None:
        self.command = command
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> Server:
        print(f"[eagle4] $ {' '.join(self.command)} > {self.log_path}", flush=True)
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.wait_ready(timeout_s=1800)
        return self

    def wait_ready(self, timeout_s: float) -> None:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            if self.process.poll() is not None:
                print_tail(self.log_path)
                raise RuntimeError(
                    f"server exited with {self.process.returncode} during startup"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/readiness", timeout=5):
                    print(
                        f"[eagle4] server ready in {time.monotonic() - start:.0f} s",
                        flush=True,
                    )
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(5)
        raise TimeoutError(f"server not ready after {timeout_s:.0f} s")

    def __exit__(self, *exc_info) -> None:
        os.killpg(self.process.pid, signal.SIGINT)
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        self.log.close()

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace")

    def last_token_logprob(self, input_ids: list[int]) -> float:
        """The logprob of ``input_ids[-1]`` after the tokens before it."""
        # From the first token: a later start splits the prefill in two, and
        # the second part makes a hybrid model's GDN prefill synchronize.
        result = self.generate(input_ids, 1, logprob_start_len=0)
        return float(result["meta_info"]["input_token_logprobs"][-1][0])

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        logprob_start_len: int | None = None,
    ) -> dict:
        body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "top_k": 1,
                "max_new_tokens": max_new_tokens,
            },
        }
        if logprob_start_len is not None:
            body.update(return_logprob=True, logprob_start_len=logprob_start_len)
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"/generate failed with {error.code}: {error.read()[:1000]!r}"
            ) from error


def encode_prompts() -> list[list[int]]:
    """The prompts as chat-formatted token ids (thinking on)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    encoded = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded.append(tokenizer(text, add_special_tokens=False).input_ids)
    return encoded


def generate_all(
    server: Server,
    prompts: list[list[int]],
    max_new_tokens: int,
    concurrency: int,
) -> tuple[list[dict], float]:
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(lambda ids: server.generate(ids, max_new_tokens), prompts)
        )
    return results, time.monotonic() - start


def harness_data_dir(args: argparse.Namespace) -> Path:
    """The harness cost table: ``--harness-data``, or measured now."""
    if args.harness_data is not None:
        return args.harness_data
    run_dir = args.out_dir / "harness-data"
    run_logged(
        measure_command(run_dir, args.max_num_seqs),
        args.out_dir / "harness-measure.log",
    )
    return run_dir


def mode_server(args: argparse.Namespace, mode: str | None, data: Path | None):
    name = mode or "reference"
    return Server(
        serve_command(
            mode,
            args.port,
            args.max_num_seqs,
            data,
            output_logprobs=mode is None,
        ),
        args.port,
        args.out_dir / f"server-{name}.log",
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    prompts = encode_prompts()
    failures = []
    rows = []
    for mode in args.modes:
        data = harness_data_dir(args) if MODES[mode].harness else None
        with mode_server(args, mode, data) as server:
            results, wall_s = generate_all(
                server, prompts, args.max_new_tokens, args.concurrency
            )
            # The log line follows the response; let it land.
            time.sleep(2)
            lengths = accept_lengths(server.log_text())
        tokens = sum(r["meta_info"]["completion_tokens"] for r in results)
        mean = statistics.mean(lengths) if lengths else 0.0
        rows.append((mode, len(results), tokens, tokens / wall_s, mean))
        if any(r["meta_info"]["completion_tokens"] == 0 for r in results):
            failures.append(f"{mode}: a request returned no tokens")
        if len(lengths) < len(results):
            failures.append(
                f"{mode}: {len(lengths)} acceptance lengths logged for "
                f"{len(results)} requests"
            )
        if mean < args.min_accept_length:
            failures.append(
                f"{mode}: mean acceptance length {mean:.2f} < "
                f"{args.min_accept_length}"
            )
    print("mode,requests,output tokens,tok/s,accept length")
    for mode, count, tokens, rate, mean in rows:
        print(f"{mode},{count},{tokens},{rate:.1f},{mean:.2f}")
    for failure in failures:
        print(f"[eagle4] FAIL {failure}", flush=True)
    return 1 if failures else 0


def classify(reference: list[int], candidate: list[int]) -> tuple[str, int | None]:
    """``identical``, ``length`` (one is a prefix of the other) or
    ``diverged``, with the first differing position."""
    index = first_difference(reference, candidate)
    if index is not None:
        return "diverged", index
    return ("identical" if reference == candidate else "length"), None


def gap_status(gap: float) -> str:
    """``near_tie`` for a reference logprob gap within ``TIE_GAP`` either
    way, else ``fp32`` (needs the fp32 logits). The gap is read from prefill
    logprobs, which may rank the mode's token first."""
    return "near_tie" if abs(gap) <= TIE_GAP else "fp32"


def fp32_logits(prefixes: list[list[int]]) -> list[list[float]]:
    """The target model's fp32 next-token logits after each prefix."""
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(TARGET_MODEL)
    model_cls = getattr(transformers, config.architectures[0])
    start = time.monotonic()
    model = model_cls.from_pretrained(
        TARGET_MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
        device_map="cuda",
    ).eval()
    print(f"[eagle4] fp32 model loaded in {time.monotonic() - start:.0f} s")
    logits = []
    with torch.inference_mode():
        for prefix in prefixes:
            input_ids = torch.tensor([prefix], device="cuda")
            output = model(input_ids=input_ids, logits_to_keep=1)
            logits.append(output.logits[0, -1].float().cpu().tolist())
    return logits


def cmd_parity(args: argparse.Namespace) -> int:
    prompts = encode_prompts()
    outputs = {}
    lengths = {}
    for mode in args.modes:
        data = harness_data_dir(args) if MODES[mode].harness else None
        with mode_server(args, mode, data) as server:
            results, _ = generate_all(
                server, prompts, args.max_new_tokens, args.concurrency
            )
            time.sleep(2)
            lengths[mode] = accept_lengths(server.log_text())
        outputs[mode] = [r["output_ids"] for r in results]

    records = []
    with mode_server(args, None, None) as server:
        reference, _ = generate_all(
            server, prompts, args.max_new_tokens, args.concurrency
        )
        ref_ids = [r["output_ids"] for r in reference]
        for mode in args.modes:
            for index, candidate in enumerate(outputs[mode]):
                status, position = classify(ref_ids[index], candidate)
                record = {"mode": mode, "prompt": index, "status": status}
                if status == "diverged":
                    prefix = prompts[index] + ref_ids[index][:position]
                    ref = ref_ids[index][position]
                    cand = candidate[position]
                    gap = server.last_token_logprob(
                        prefix + [ref]
                    ) - server.last_token_logprob(prefix + [cand])
                    record.update(
                        status=gap_status(gap),
                        position=position,
                        reference_token=ref,
                        candidate_token=cand,
                        reference_gap=gap,
                    )
                records.append(record)
    pending = [r for r in records if r["status"] == "fp32"]
    if pending:
        prefixes = [
            prompts[r["prompt"]] + ref_ids[r["prompt"]][: r["position"]]
            for r in pending
        ]
        for record, logits in zip(pending, fp32_logits(prefixes)):
            ref = record["reference_token"]
            cand = record["candidate_token"]
            record["status"] = fp32_status(logits, ref, cand)
            record["fp32_gap"] = logits[ref] - logits[cand]

    (args.out_dir / "parity.json").write_text(
        json.dumps(
            {
                "records": records,
                "reference": ref_ids,
                "outputs": outputs,
                "accept_lengths": lengths,
            },
            indent=1,
        )
    )
    failed = False
    print("mode,prompts," + ",".join(PASSING_STATUSES) + ",fail,accept length")
    for mode in args.modes:
        counts = {s: 0 for s in (*PASSING_STATUSES, "fail")}
        for record in records:
            if record["mode"] == mode:
                counts[record["status"]] += 1
        failed |= counts["fail"] > 0
        mean = statistics.mean(lengths[mode]) if lengths[mode] else 0.0
        print(
            f"{mode},{len(prompts)},"
            + ",".join(str(counts[s]) for s in (*PASSING_STATUSES, "fail"))
            + f",{mean:.2f}"
        )
    for record in records:
        if record["status"] not in ("identical", "length"):
            print(f"[eagle4] {json.dumps(record)}")
    return 1 if failed else 0


def cmd_serve(args: argparse.Namespace) -> int:
    data = harness_data_dir(args) if MODES[args.mode].harness else None
    command = serve_command(
        args.mode, args.port, args.max_num_seqs, data, output_logprobs=False
    )
    print(f"[eagle4] $ {' '.join(command)}", flush=True)
    os.execvp(command[0], command)


def cmd_accept_length(args: argparse.Namespace) -> int:
    lengths = accept_lengths(args.server_log.read_text(errors="replace"))
    if not lengths:
        print(f"[eagle4] FAIL no acceptance lengths in {args.server_log}")
        return 1
    mean = statistics.mean(lengths)
    print(f"Mean acceptance length: {mean:.2f} over {len(lengths)} requests")
    if mean < args.min_accept_length:
        print(f"[eagle4] FAIL mean acceptance length < {args.min_accept_length}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "parity", "serve"):
        p = sub.add_parser(name)
        if name == "serve":
            p.add_argument("--mode", choices=list(MODES), required=True)
        else:
            p.add_argument(
                "--modes", nargs="+", choices=list(MODES), default=list(MODES)
            )
            p.add_argument("--concurrency", type=int, default=8)
            p.add_argument("--max-new-tokens", type=int, default=256)
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--max-num-seqs", type=int, default=8)
        p.add_argument(
            "--harness-data",
            type=Path,
            help="an EAGLE-Harness cost table to serve instead of measuring one",
        )
    sub.choices["smoke"].add_argument("--min-accept-length", type=float, default=1.5)
    p = sub.add_parser("accept-length")
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--min-accept-length", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "accept-length":
        return cmd_accept_length(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = {"smoke": cmd_smoke, "parity": cmd_parity, "serve": cmd_serve}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
