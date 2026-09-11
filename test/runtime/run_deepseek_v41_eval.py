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

"""Bounded real-checkpoint smoke/GSM8K run; cleans up only its own server tree.

Activate the development venv and explicitly select available GPUs before use.
All paths/ports/time limits, --execution-mode eager|graph, --batch-size,
--max-total-tokens, --max-model-len and --chunked-prefill-size are explicit;
the supplied snapshot is never modified.
Batch size controls server admission, decode capture and EvalScope concurrency.
Graph mode enables decode capture only.
The output directory must exist and be empty. With --run-eval, EvalScope must
finish successfully and report all 1,319 GSM8K test samples without errors.
A complete run writes public result.json and fails unless accuracy > 0.90;
malformed/incomplete reports and failed processes never produce a result.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import psutil

MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"


def read_gsm8k_result(work_dir: Path, evalscope_version: str) -> dict:
    """Validate EvalScope 1.11.1's v2 report and return only public result fields."""
    # EvalScope adds a timestamp, then reports/<model_id>/gsm8k.json.
    reports = list(work_dir.glob("*/reports/*/gsm8k.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected exactly one GSM8K report, found {len(reports)}")
    try:
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        [metric] = report["metrics"]
        [category] = metric["categories"]
        [subset] = category["subsets"]
        identity = {"name": "accuracy", "aggregation": "mean", "dimensions": {}}
        if (
            type(report["schema_version"]) is not int
            or report["schema_version"] != 2
            or report["dataset_name"] != "gsm8k"
            or report["model_name"] != MODEL_NAME.rsplit("/", 1)[1]
            or report["primary_metric_identity"] != identity
            or report["primary_metric_unavailable_reason"] is not None
            or metric["identity"] != identity
            or category["name"] != ["default"]
            or subset["name"] != "main"
            or subset["is_aggregate"] is not False
        ):
            raise ValueError(
                "Expected the GSM8K main accuracy:mean report for this model"
            )
        execution = report["execution_summary"]
        if execution["incomplete"] is not False or set(execution["subsets"]) != {
            "main"
        }:
            raise ValueError("GSM8K execution is incomplete or has unexpected subsets")
        for counts in (execution, execution["subsets"]["main"]):
            for key, expected in (
                ("requested", 1319),
                ("succeeded", 1319),
                ("errored", 0),
            ):
                if type(counts[key]) is not int or counts[key] != expected:
                    raise ValueError(f"GSM8K execution {key} must be {expected}")
        for node in (report, metric, category, subset):
            if type(node["num"]) is not int or node["num"] != 1319:
                raise ValueError(
                    "GSM8K must score exactly 1319 samples at every report level"
                )
        accuracy = metric["score"]
        for score in (
            accuracy,
            metric["macro_score"],
            category["score"],
            category["macro_score"],
            subset["score"],
        ):
            if (
                type(score) not in (int, float)
                or not 0 <= score <= 1
                or score != accuracy
            ):
                raise ValueError(
                    "GSM8K accuracy must be a consistent finite number in [0, 1]"
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid or incomplete GSM8K report: {exc}") from exc
    return {
        "model": MODEL_NAME,
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "test",
        "evalscope_version": evalscope_version,
        "samples": 1319,
        "accuracy": accuracy,
        "threshold": 0.90,
        "passed": accuracy > 0.90,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "output-dir", "port", "server-timeout", "eval-timeout"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    for name in (
        "batch-size",
        "max-total-tokens",
        "max-model-len",
        "chunked-prefill-size",
    ):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--run-eval", action="store_true")
    args = parser.parse_args()
    if not 0 < args.batch_size <= args.max_total_tokens:
        parser.error("Require 0 < batch-size <= max-total-tokens")
    if args.max_model_len <= 0 or args.chunked_prefill_size <= 0:
        parser.error("max-model-len and chunked-prefill-size must be positive")
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 4:
        parser.error("Explicitly select four free GPUs with CUDA_VISIBLE_DEVICES")
    output = Path(args.output_dir).resolve()
    if not output.is_dir():
        parser.error("Create the output directory before running this harness")
    if any(output.iterdir()):
        parser.error(
            "Output directory must be empty; use a new directory to preserve previous logs"
        )
    port = int(args.port)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    name = MODEL_NAME
    command = [
        "ts",
        "serve",
        "--model",
        args.model,
        "--served-model-name",
        name,
        "--tensor-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--moe-backend",
        "mega_moe",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-num-seqs",
        str(args.batch_size),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--gpu-memory-utilization",
        "0.9",
        "--disable-kvstore",
        "--disable-prefill-graph",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if args.execution_mode == "eager":
        command.append("--enforce-eager")
    else:
        command.extend(("--max-cudagraph-capture-size", str(args.batch_size)))
    # Store a reproducible public model ID rather than a developer's local path.
    public_command = [name if item == args.model else item for item in command]
    (output / "server-command.json").write_text(json.dumps(public_command, indent=2))
    base = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while time.monotonic() - started < int(args.server_timeout):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {process.returncode}; see {output / 'server.log'}"
                    )
                try:
                    with urlopen(base + "/readiness", timeout=3) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(5)
            else:
                raise TimeoutError("Server readiness deadline exceeded")
            print(f"Server ready in {time.monotonic() - started:.1f}s", flush=True)
            request = Request(
                base + "/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": name,
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is 2 + 2? Answer briefly.",
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 32,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=600) as response:
                smoke = json.load(response)
            (output / "smoke.json").write_text(
                json.dumps(smoke, ensure_ascii=False, indent=2)
            )
            print(json.dumps(smoke, ensure_ascii=False), flush=True)
            answer = smoke["choices"][0]["message"]["content"] or ""
            if "4" not in answer and "four" not in answer.lower():
                raise RuntimeError(
                    "Arithmetic smoke failed; inspect smoke.json before evaluating"
                )
            if args.run_eval:
                evaluation = [
                    "evalscope",
                    "eval",
                    "--model",
                    name,
                    "--api-url",
                    base + "/v1",
                    "--api-key",
                    "EMPTY_TOKEN",
                    "--datasets",
                    "gsm8k",
                    "--dataset-hub",
                    "huggingface",
                    "--dataset-args",
                    json.dumps({"gsm8k": {"dataset_id": "openai/gsm8k"}}),
                    "--eval-batch-size",
                    str(args.batch_size),
                    "--generation-config",
                    json.dumps(
                        {"do_sample": False, "temperature": 0.0, "max_tokens": 512}
                    ),
                    "--work-dir",
                    str(output / "evalscope"),
                ]
                (output / "eval-command.json").write_text(
                    json.dumps(evaluation, indent=2)
                )
                with (output / "evalscope.log").open("w") as eval_log:
                    subprocess.run(
                        evaluation,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=int(args.eval_timeout),
                    )
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {process.returncode} during evaluation; see server.log"
                    )
                result = read_gsm8k_result(output / "evalscope", version("evalscope"))
                result["execution_mode"] = args.execution_mode
                result["batch_size"] = args.batch_size
                result["max_total_tokens"] = args.max_total_tokens
                result["max_model_len"] = args.max_model_len
                result["chunked_prefill_size"] = args.chunked_prefill_size
                (output / "result.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(result), flush=True)
                if not result["passed"]:
                    raise RuntimeError(
                        f"GSM8K accuracy {result['accuracy']:.4f} must be > 0.90"
                    )
        finally:
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass


if __name__ == "__main__":
    main()
