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

"""Real Qwen3.5 FP8 DeepEP BCG qualification on four Blackwell GPUs.

    TOKENSPEED_DEEPEP_BCG_E2E=1 python3 -m pytest -q \
        test/runtime/models/test_deepep_prefill_bcg_e2e.py

Two fresh eager-prefill engines establish repeatability, then a third engine
enables BCG. Decode graphs, default serving kernels, text-only configuration,
and all other settings match. Requests explicitly address both DP replicas.
The test requires first-serving-replay evidence from every EP rank; startup
capture and smoke replay cannot satisfy that assertion.

The public engine currently exposes sampled-token logprobs, not full-vocabulary
logits. This test compares every greedy token (including the first) and its
logprob unconditionally. It does not claim a full-logit oracle; the distributed
MoE tests supply the complementary complete-output numerical check.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import unittest
from pathlib import Path

_ENABLED = os.environ.get("TOKENSPEED_DEEPEP_BCG_E2E") == "1"
_MODEL = os.environ.get("TOKENSPEED_DEEPEP_BCG_E2E_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
_ARTIFACTS = Path(
    os.environ.get("TOKENSPEED_DEEPEP_BCG_E2E_ARTIFACTS", ".ci-artifacts/deepep-bcg")
).resolve()
_TIMEOUT = int(os.environ.get("TOKENSPEED_DEEPEP_BCG_E2E_TIMEOUT", "1800"))
_PORT = int(os.environ.get("TOKENSPEED_DEEPEP_BCG_E2E_PORT", "23620"))
_CAP = 512
_CHUNK = 1024
# Match the existing ModelCase prefill/decode tolerance. Eager repeatability
# must independently pass this bound; observed eager error is reported, never
# used to silently relax the BCG gate.
_LOGPROB_ATOL = 5e-2
_REPLAY_PATTERN = re.compile(
    r"DeepEP prefill BCG replay: rank=(\d+) bucket=(\d+) real_tokens=(\d+)"
)


def _assert_replay_log(log: str) -> set[int]:
    replay_ranks = {int(match[0]) for match in _REPLAY_PATTERN.findall(log)}
    if replay_ranks != {0, 1, 2, 3}:
        raise AssertionError(
            "all EP ranks must prove a real serving BCG replay; "
            f"observed ranks: {sorted(replay_ranks)}"
        )
    return replay_ranks


def _sampled_tokens(response: dict) -> tuple[list[int], list[float]]:
    token_ids = response["output_ids"]
    sampled = response["meta_info"]["output_token_logprobs"]
    if not token_ids or len(sampled) != len(token_ids):
        raise AssertionError("missing generated tokens or sampled-token logprobs")
    if [int(item[1]) for item in sampled] != token_ids:
        raise AssertionError("sampled-token logprob IDs do not match output_ids")
    logprobs = [float(item[0]) for item in sampled]
    if not all(math.isfinite(value) for value in logprobs):
        raise AssertionError("nonfinite sampled-token logprob")
    return token_ids, logprobs


def _compare_records(reference: list[dict], actual: list[dict]) -> float:
    if len(reference) != len(actual):
        raise AssertionError("different request counts")
    max_error = 0.0
    for expected, observed in zip(reference, actual, strict=True):
        label = expected["case"]
        if label != observed["case"]:
            raise AssertionError("different request ordering")
        expected_ids, expected_lps = _sampled_tokens(expected["response"])
        actual_ids, actual_lps = _sampled_tokens(observed["response"])
        # Do not use test.runners.check_close_model_outputs: its first-token
        # divergence path can skip the logprob comparison entirely.
        if expected_ids[0] != actual_ids[0]:
            raise AssertionError(f"{label}: first token differs")
        if expected_ids != actual_ids:
            raise AssertionError(f"{label}: greedy token sequence differs")
        error = max(abs(a - b) for a, b in zip(expected_lps, actual_lps, strict=True))
        max_error = max(max_error, error)
        if error > _LOGPROB_ATOL:
            raise AssertionError(
                f"{label}: sampled-token logprob error {error:.6g} "
                f"exceeds {_LOGPROB_ATOL}"
            )
    return max_error


def _render(tokenizer, context: str, question: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _workload(tokenizer) -> list[tuple[str, list[list[int]], int]]:
    short = "Answer the final question briefly. "
    shared = "Remember the following note. " + "A blue square has four sides. " * 24
    long = "Ignore these notes when answering. " + "A red circle is round. " * 220
    cases = []
    for name, context, question in (
        ("short", short, "What is 2 + 2? Reply with just the number."),
        ("prefix_first", shared, "What is 1 + 1? Reply with just the number."),
        ("prefix_second", shared, "What is 2 + 2? Reply with just the number."),
        ("prefix_third", shared, "What is 3 + 3? Reply with just the number."),
        ("chunked_overcap", long, "What is 4 + 4? Reply with just the number."),
        ("short_after_chunk", short, "What is 5 + 5? Reply with just the number."),
    ):
        # Distinct replicas exercise unequal real counts in the same submitted
        # batch. The focused distributed test controls exact scheduler steps.
        prompts = [
            _render(tokenizer, context, question),
            _render(tokenizer, context, "Please calculate carefully. " + question),
        ]
        if name == "chunked_overcap":
            if not all(_CHUNK < len(prompt) < 4096 - 16 for prompt in prompts):
                raise AssertionError("chunked case no longer spans multiple chunks")
        elif not all(0 < len(prompt) <= _CAP for prompt in prompts):
            raise AssertionError(f"{name}: prompt no longer fits the graph cap")
        cases.append((name, prompts, 16))
    return cases


def _run_arm(arm: str, result_path: Path) -> None:
    import torch
    from tokenspeed_kernel.platform import current_platform
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    if torch.cuda.device_count() < 4 or not current_platform().is_blackwell:
        raise RuntimeError("DeepEP BCG E2E requires four NVIDIA Blackwell GPUs")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)
    cases = _workload(tokenizer)
    engine = Engine(
        model=_MODEL,
        language_model_only=True,
        trust_remote_code=True,
        dtype="bfloat16",
        world_size=4,
        attn_tp_size=2,
        data_parallel_size=2,
        dense_tp_size=4,
        moe_tp_size=1,
        ep_size=4,
        dist_init_addr=f"127.0.0.1:{_PORT + 1}",
        port=_PORT,
        quantization="fp8",
        moe_backend="deep_gemm",
        all2all_backend="deepep",
        deepep_mode="auto",
        max_model_len=4096,
        max_num_seqs=16,
        max_total_tokens=16384,
        max_prefill_tokens=_CHUNK,
        chunked_prefill_size=_CHUNK,
        max_cudagraph_capture_size=16,
        gpu_memory_utilization=0.20,
        sampling_backend="greedy",
        enable_output_logprobs=True,
        disable_kvstore=True,
        enable_prefix_caching=True,
        disable_prefill_graph=arm != "bcg",
        prefill_graph_max_tokens=_CAP if arm == "bcg" else 0,
        prefill_graph_capture_sizes=[32, 128, 256, 512],
        seed=42,
        distributed_timeout_seconds=120,
        log_level="info",
    )
    records = []
    try:
        # The same prefix is used on both replicas, and the third access is
        # required for hybrid Full-KV/state boundary promotion.
        for name, prompts, max_new_tokens in cases:
            responses = engine.generate(
                input_ids=prompts,
                sampling_params={
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
                return_logprob=True,
                top_logprobs_num=0,
                logprob_start_len=-1,
                stream=False,
                data_parallel_rank=[0, 1],
            )
            if not isinstance(responses, list) or len(responses) != 2:
                raise AssertionError("expected one response per DP replica")
            for rank, response in enumerate(responses):
                ids, _ = _sampled_tokens(response)
                if len(ids) != max_new_tokens:
                    raise AssertionError("generation did not exercise decode")
                cached = response["meta_info"]["cached_tokens"]
                if name == "prefix_first" and cached != 0:
                    raise AssertionError("first shared-prefix request was not cold")
                if name == "prefix_third" and cached <= 0:
                    raise AssertionError("third shared-prefix request missed cache")
                records.append({"case": f"{name}/dp{rank}", "response": response})
    finally:
        engine.shutdown()
    result_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _launch_arm(arm: str) -> tuple[list[dict], str]:
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result_path = _ARTIFACTS / f"{arm}.json"
    log_path = _ARTIFACTS / f"{arm}.log"
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-arm",
        arm,
        str(result_path),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise AssertionError(f"{arm} exceeded {_TIMEOUT}s; see {log_path}") from exc
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if returncode:
        raise AssertionError(f"{arm} failed; see {log_path}\n{log[-6000:]}")
    return json.loads(result_path.read_text(encoding="utf-8")), log


class TestDeepEPPrefillBCGOracle(unittest.TestCase):
    def test_capture_messages_do_not_prove_serving_replay(self):
        with self.assertRaisesRegex(AssertionError, "all EP ranks"):
            _assert_replay_log("Prefill CUDA graphs captured: 4 buckets")
        with self.assertRaisesRegex(AssertionError, "all EP ranks"):
            _assert_replay_log(
                "DeepEP prefill BCG replay: rank=0 bucket=128 real_tokens=53"
            )
        log = "\n".join(
            f"DeepEP prefill BCG replay: rank={rank} bucket=128 real_tokens=53"
            for rank in range(4)
        )
        self.assertEqual(_assert_replay_log(log), {0, 1, 2, 3})

    def test_first_token_divergence_fails(self):
        def record(token: int) -> list[dict]:
            return [
                {
                    "case": "first",
                    "response": {
                        "output_ids": [token],
                        "meta_info": {"output_token_logprobs": [[-0.1, token, None]]},
                    },
                }
            ]

        with self.assertRaisesRegex(AssertionError, "first token differs"):
            _compare_records(record(1), record(2))

    def test_missing_or_nonfinite_logprobs_fail(self):
        for logprobs in ([], [[float("nan"), 1, None]]):
            with self.subTest(logprobs=logprobs), self.assertRaises(AssertionError):
                _sampled_tokens(
                    {
                        "output_ids": [1],
                        "meta_info": {"output_token_logprobs": logprobs},
                    }
                )


@unittest.skipUnless(_ENABLED, "set TOKENSPEED_DEEPEP_BCG_E2E=1 on four B200 GPUs")
class TestDeepEPPrefillBCGE2E(unittest.TestCase):
    def test_eager_equivalence_with_cache_chunking_and_decode(self):
        eager, eager_log = _launch_arm("eager")
        repeated, repeat_log = _launch_arm("eager_repeat")
        candidate, candidate_log = _launch_arm("bcg")
        self.assertFalse(_REPLAY_PATTERN.search(eager_log))
        self.assertFalse(_REPLAY_PATTERN.search(repeat_log))
        replay_ranks = _assert_replay_log(candidate_log)
        report = {
            "eager_repeat_max_logprob_error": _compare_records(eager, repeated),
            "bcg_max_logprob_error": _compare_records(eager, candidate),
            "sampled_logprob_atol": _LOGPROB_ATOL,
            "full_vocabulary_logits_checked": False,
            "replay_ranks": sorted(replay_ranks),
        }
        (_ARTIFACTS / "comparison.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--run-arm":
        _run_arm(sys.argv[2], Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--check-replay-log":
        _assert_replay_log(
            Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
        )
        print("DeepEP prefill BCG serving replay verified on all four EP ranks")
    else:
        unittest.main()
