# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Run one standard AgentX concurrency point against a TokenSpeed server."""

import argparse
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

AGENTX_CONTEXT_LENGTH = 262144
AGENTX_DURATION_SECONDS = 1800


def validate_models(payload: dict, model: str) -> None:
    """Check the OpenAI model ID before starting AgentX."""
    models = [item for item in payload["data"] if item["id"] == model]
    if len(models) != 1:
        raise ValueError(f"{model} must be advertised exactly once by /v1/models")


def validate_model_info(payload: dict, model: str, context_length: int) -> None:
    """Check the engine's served model and effective context window."""
    if payload.get("served_model_name") != model:
        raise ValueError(f"/get_model_info must report served_model_name {model!r}")
    max_context_length = payload.get("max_context_length")
    if type(max_context_length) is not int or max_context_length < context_length:
        raise ValueError(
            f"{model} must report at least {context_length} context tokens"
        )


def validate_result(output_dir: Path, parallel: int) -> dict:
    """Reject incomplete or incomparable AgentX runs while retaining their files."""
    paths = list(output_dir.rglob("agentx_summary.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one AgentX summary, found {len(paths)}")
    result = json.loads(paths[0].read_text())
    # Include invalid-run reasons in the CI log before raising an error.
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "completed" or result["submission_valid"] is not True:
        raise ValueError("AgentX did not produce a completed, valid benchmark")
    if (
        result["concurrency"] != parallel
        or result["duration"] != AGENTX_DURATION_SECONDS
    ):
        raise ValueError("AgentX result does not match the requested benchmark")
    if result["scenario"]["mode"] != "benchmark":
        raise ValueError("AgentX smoke results cannot satisfy this benchmark")
    if result["dataset"]["verified"] is not True:
        raise ValueError("AgentX dataset hash was not verified")
    if not result["metrics"]:
        raise ValueError("AgentX did not report any metrics")
    return result


def run(args: argparse.Namespace) -> None:
    """Record the endpoint and command, execute AgentX, and validate its output."""
    if args.context_length != AGENTX_CONTEXT_LENGTH:
        raise ValueError(f"AgentX 256k requires {AGENTX_CONTEXT_LENGTH} context tokens")
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    output_dir = args.outputs_dir.resolve()
    # Reject stale results instead of allowing a failed run to reuse them.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"benchmark output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    with urlopen(f"{base_url}/v1/models", timeout=30) as response:
        models = json.load(response)
    (output_dir / "server-models.json").write_text(f"{json.dumps(models, indent=2)}\n")
    validate_models(models, args.model)
    control_url = args.control_url.rstrip("/")
    with urlopen(f"{control_url}/get_model_info", timeout=30) as response:
        model_info = json.load(response)
    (output_dir / "server-model-info.json").write_text(
        f"{json.dumps(model_info, indent=2)}\n"
    )
    validate_model_info(model_info, args.model, args.context_length)

    scenario = {
        "name": "agentx",
        "mode": "benchmark",
        "variant": "256k",
        "num_gpus": args.num_gpus,
        "engine": "tokenspeed",
        "engine_version": args.engine_version,
        "tokenizer_revision": args.revision,
        "tokenizer_trust_remote_code": True,
        "request_timeout_seconds": 1800,
        "benchmark_grace_period": 600,
    }
    command = [
        args.evalscope,
        "perf",
        "--scenario",
        json.dumps(scenario, separators=(",", ":")),
        "--model",
        args.model,
        "--tokenizer-path",
        args.model,
        "--url",
        f"{base_url}/v1/chat/completions",
        "--api",
        "openai",
        "--data-source",
        "huggingface",
        "--parallel",
        str(args.parallel),
        "--duration",
        str(AGENTX_DURATION_SECONDS),
        "--seed",
        "20260707",
        "--outputs-dir",
        str(output_dir),
        "--no-timestamp",
    ]
    (output_dir / "command.json").write_text(f"{json.dumps(command, indent=2)}\n")
    completed = subprocess.run(command, check=False)
    # EvalScope can preserve useful diagnostics when its subprocess fails.
    validate_result(output_dir, args.parallel)
    completed.check_returncode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel", type=int, choices=(1, 2, 4, 8, 16, 32), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--evalscope", required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
