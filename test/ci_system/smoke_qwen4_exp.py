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

"""Smoke the public Qwen4Exp checkpoint in one fresh Engine process per mode.

The companion manual Slurm task runs non-spec, MTP eager, and MTP graph in
sequence. Each invocation reuses its Engine for two rounds of batch sizes 1
and 4, exercising decode and repeated requests with the same graph captures.
This checks completion and output lengths, not numerical parity or accuracy.
Generated text and token IDs are never printed by this script.
"""

from __future__ import annotations

import argparse
import gzip
import json
import signal
import tempfile
import time
from pathlib import Path
from types import FrameType
from typing import NoReturn

MODEL = "Qwen/Qwen3.8-Flash-Next-FP8"
MODES = ("non-spec", "mtp-eager", "mtp-graph")
BATCH_SIZES = (1, 4)
REPETITIONS = 2
NEW_TOKENS = 32
PROMPTS = (
    "The capital of France is",
    "Explain why the sky appears blue in one sentence.",
    "Continue the sequence: 1, 2, 3, 4,",
    "Write a short sentence about a tree.",
)


def _engine_kwargs(mode: str) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError("Unknown smoke mode")
    speculative = mode != "non-spec"
    return {
        "model": MODEL,
        "trust_remote_code": True,
        # tensor_parallel_size is a CLI alias; Engine takes attn_tp_size.
        "attn_tp_size": 2,
        "quantization": "fp8",
        "dtype": "bfloat16",
        "moe_backend": "flashinfer_trtllm",
        "sampling_backend": "greedy",
        "speculative_algorithm": "MTP" if speculative else None,
        "speculative_num_steps": 3 if speculative else 0,
        "speculative_num_draft_tokens": 4 if speculative else 1,
        "enforce_eager": mode == "mtp-eager",
        "max_model_len": 2048,
        "max_num_seqs": max(BATCH_SIZES),
        "max_cudagraph_capture_size": max(BATCH_SIZES),
        "cudagraph_capture_sizes": [1, 2, 4],
        # Profile only decode graph launches, after capture and one warm round.
        "disable_prefill_graph": True,
        "enable_prefix_caching": True,
        "seed": 0,
        "log_level": "info",
        "enable_log_requests": False,
        # Records the scheduler's actual batch sizes without request contents.
        "decode_log_interval": 1,
    }


def _check_outputs(
    outputs: object, batch_size: int, new_tokens: int, speculative: bool
) -> tuple[int, int]:
    results = [outputs] if isinstance(outputs, dict) else outputs
    assert isinstance(results, list), "Engine did not return completed responses"
    assert len(results) == batch_size, "Completed request count differs from batch size"
    generated_tokens = 0
    verify_rounds = 0
    for index, result in enumerate(results):
        label = f"Response {index}"
        assert isinstance(result, dict), f"{label} is not a response dictionary"
        text = result.get("text")
        assert isinstance(text, str) and text.strip(), f"{label} has no text"
        output_ids = result.get("output_ids")
        assert isinstance(output_ids, list) and output_ids, f"{label} has no token IDs"
        meta = result.get("meta_info")
        assert isinstance(meta, dict), f"{label} has no metadata"
        finish_reason = meta.get("finish_reason")
        assert isinstance(finish_reason, dict), f"{label} has no finish reason"
        assert (
            finish_reason.get("type") == "length"
        ), f"{label} did not finish by length"
        assert (
            meta.get("completion_tokens") == new_tokens
        ), f"{label} token count differs"
        assert len(output_ids) == new_tokens, f"{label} token IDs have the wrong length"
        generated_tokens += len(output_ids)
        if speculative:
            count = meta.get("spec_verify_ct")
            assert (
                isinstance(count, int) and count > 0
            ), f"{label} did not verify drafts"
            verify_rounds += count
    return generated_tokens, verify_rounds


def _check_graph_profile(trace_dir: str, graph_expected: bool) -> int:
    traces = sorted(Path(trace_dir).glob("*.trace.json.gz"))
    assert len(traces) == 2, "Expected one completed profiler trace per TP rank"
    graph_launches = 0
    for path in traces:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            events = json.load(handle)["traceEvents"]
        assert any(
            event.get("cat") == "kernel" for event in events
        ), "Profiler did not record GPU kernels"
        count = sum("cudaGraphLaunch" in event.get("name", "") for event in events)
        if graph_expected:
            assert count > 0, "Decode did not replay a CUDA graph on every TP rank"
        else:
            assert count == 0, "Eager decode unexpectedly launched a CUDA graph"
        graph_launches += count
    return graph_launches


def _exit_on_sigterm(signum: int, frame: FrameType | None) -> NoReturn:
    # Let Engine's finally/atexit shutdown clean up workers on CI timeouts.
    raise SystemExit(128 + signum)


def _run_smoke(mode: str) -> None:
    from tokenspeed.runtime.entrypoints.engine import Engine

    start = time.monotonic()
    engine = Engine(**_engine_kwargs(mode))
    completed_requests = 0
    generated_tokens = 0
    verify_rounds = 0
    graph_launches = 0
    try:
        print(
            json.dumps(
                {
                    "event": "engine_ready",
                    "mode": mode,
                    "elapsed_seconds": round(time.monotonic() - start, 3),
                }
            ),
            flush=True,
        )
        for repetition in range(1, REPETITIONS + 1):
            for batch_size in BATCH_SIZES:
                batch_start = time.monotonic()
                # Keep traces temporary: publish counts, never captured data.
                with tempfile.TemporaryDirectory(
                    prefix="qwen4-exp-smoke-"
                ) as trace_dir:
                    profile = repetition == REPETITIONS
                    if profile:
                        engine.llm.run(
                            engine.tokenizer_manager.start_profile(
                                output_dir=trace_dir,
                                start_step=None,
                                num_steps=None,
                                activities=["CPU", "GPU"],
                                with_stack=False,
                                record_shapes=False,
                                profile_by_stage=False,
                                profile_id=f"{mode}-bs{batch_size}",
                            )
                        )
                    try:
                        outputs = engine.generate(
                            prompt=list(PROMPTS[:batch_size]),
                            sampling_params={
                                "temperature": 0.0,
                                "top_k": 1,
                                "seed": 0,
                                "max_new_tokens": NEW_TOKENS,
                                "min_new_tokens": NEW_TOKENS,
                                "ignore_eos": True,
                            },
                            stream=False,
                        )
                    finally:
                        if profile:
                            engine.stop_profile()
                    batch_graph_launches = (
                        _check_graph_profile(trace_dir, mode != "mtp-eager")
                        if profile
                        else None
                    )
                batch_tokens, batch_verify_rounds = _check_outputs(
                    outputs, batch_size, NEW_TOKENS, mode != "non-spec"
                )
                completed_requests += batch_size
                generated_tokens += batch_tokens
                verify_rounds += batch_verify_rounds
                graph_launches += batch_graph_launches or 0
                print(
                    json.dumps(
                        {
                            "event": "batch_complete",
                            "mode": mode,
                            "batch_size": batch_size,
                            "repetition": repetition,
                            "completed_requests": batch_size,
                            "generated_tokens": batch_tokens,
                            "spec_verify_rounds": batch_verify_rounds,
                            "cuda_graph_launches": batch_graph_launches,
                            "elapsed_seconds": round(time.monotonic() - batch_start, 3),
                        }
                    ),
                    flush=True,
                )
    finally:
        engine.shutdown()
    print(
        json.dumps(
            {
                "event": "smoke_passed",
                "mode": mode,
                "completed_requests": completed_requests,
                "generated_tokens": generated_tokens,
                "spec_verify_rounds": verify_rounds,
                "cuda_graph_launches": graph_launches,
                "elapsed_seconds": round(time.monotonic() - start, 3),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen4Exp model smoke requires CUDA")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Qwen4Exp model smoke requires two visible CUDA GPUs")
    print(
        json.dumps(
            {
                "event": "cuda_preflight",
                "mode": args.mode,
                "devices": [torch.cuda.get_device_name(index) for index in range(2)],
            }
        ),
        flush=True,
    )
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    _run_smoke(args.mode)


if __name__ == "__main__":
    main()
