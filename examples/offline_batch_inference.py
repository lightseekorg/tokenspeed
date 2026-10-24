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

"""Offline batched inference recipe for TokenSpeed Engine.

This example demonstrates how to initialize the TokenSpeed inference engine
in-process for offline batch evaluation, benchmarking, or synthetic data
generation, without requiring an external HTTP server daemon.

Usage:
    python examples/offline_batch_inference.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --tensor-parallel-size 1 \\
        --max-model-len 8192
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline batched inference using TokenSpeed Engine."
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        type=str,
        required=True,
        help="Path or HuggingFace repo ID of the model checkpoint.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        dest="tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel degree across available GPUs.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU VRAM allocated for model weights and KV cache.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic evaluation).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Attention backend kernel implementation (e.g., 'triton', 'flashinfer', 'fa3').",
    )
    parser.add_argument(
        "--sampling-backend",
        type=str,
        default=None,
        help="Sampling backend implementation (e.g., 'triton', 'flashinfer', 'greedy').",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save generated results in JSON format.",
    )
    return parser.parse_args()


def run_batch_inference(
    model: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    temperature: float,
    max_new_tokens: int,
    prompts: list[str],
    attention_backend: str | None = None,
    sampling_backend: str | None = None,
) -> list[dict[str, Any]]:
    """Initialize TokenSpeed Engine, run batch generation, and report
    metrics.
    """
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import ServerArgs

    server_args = ServerArgs(
        model=model,
        attn_tp_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_backend=attention_backend,
        sampling_backend=sampling_backend,
        log_level="error",
    )

    print(
        f"Initializing TokenSpeed Engine for {model} " f"(TP={tensor_parallel_size})..."
    )
    engine = Engine(server_args=server_args)

    sampling_params: dict[str, Any] = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": False,
    }

    try:
        print(f"Executing batch generation for {len(prompts)} prompts...")
        start_time = time.perf_counter()
        outputs = engine.generate(prompt=prompts, sampling_params=sampling_params)
        duration = time.perf_counter() - start_time

        total_output_tokens = sum(len(out.get("output_ids", [])) for out in outputs)
        throughput = total_output_tokens / duration if duration > 0.0 else 0.0

        print(
            f"Generation finished in {duration:.2f}s "
            f"({total_output_tokens} tokens total, "
            f"{throughput:.2f} tokens/s)\n"
        )
        return outputs

    finally:
        # Guarantee scheduler subprocess cleanup
        engine.shutdown()


def main() -> int:
    args = parse_args()

    # Evaluation prompts representing diverse production reasoning/coding
    prompts: list[str] = [
        (
            "Explain the architectural split between C++ control plane and "
            "Python execution plane in high-performance inference engines."
        ),
        (
            "Implement a thread-safe LRU cache in Python with O(1) "
            "get and put operations."
        ),
        (
            "Compare tensor parallelism vs expert parallelism for sparse "
            "Mixture-of-Experts (MoE) architectures."
        ),
        (
            "Summarize how prefix caching works and why Radix trees improve "
            "multi-turn agent response latency."
        ),
    ]

    outputs = run_batch_inference(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        prompts=prompts,
        attention_backend=args.attention_backend,
        sampling_backend=args.sampling_backend,
    )

    results_to_save: list[dict[str, Any]] = []
    for index, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
        generated_text = output.get("text", "").strip()
        num_tokens = len(output.get("output_ids", []))
        print(f"=== [Item {index}/{len(prompts)}] ({num_tokens} tokens) ===")
        print(f"Prompt: {prompt}")
        print(f"Output:\n{generated_text}\n")

        results_to_save.append(
            {
                "prompt": prompt,
                "text": generated_text,
                "token_count": num_tokens,
            }
        )

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as file:
            json.dump(results_to_save, file, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
