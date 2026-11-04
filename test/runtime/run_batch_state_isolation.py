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

"""Compare one request alone and in deliberately varied native batches.

The runner drives one headless engine through ``WireDriver`` and writes a
self-contained JSON report with token IDs, selected-token logprobs, finish
metadata, prefix-cache counts, first divergences, and frontend-observed load
samples.  It uses greedy decoding and the launch's declared numerical contract;
with ``--numerics rl-bitwise`` every comparison is exact.

The default command is a first-pass single-GPU experiment:
``Qwen/Qwen3-0.6B``, BF16, TP 1, a 4096-token context and scheduling budget,
disabled prefix caching and CUDA graphs. It includes independent repeats,
equal-length and long/short two-request batches, reversed submission order,
shared-prefix requests, and a four-request batch. See
``docs/guides/batch-state-isolation.md`` for commands and interpretation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TARGET_PROMPT = """You are a careful assistant. Answer the following arithmetic question with one short sentence.

Question: If a box has 17 blue marbles and 25 red marbles, how many marbles are in the box?
Answer:"""
DEFAULT_SHARED_PREFIX = """You are a careful assistant. Answer the question with one short sentence.

Context: A library catalog stores books by author and title. """


class Driver(Protocol):
    """The narrow native-driver interface the experiment needs."""

    server_args: list[str]
    num_engines: int
    routing: str
    startup_timeout_s: float

    def engines(self) -> list[dict]: ...

    def generate_with_loads(
        self, requests: list[Any], *, poll_interval_s: float
    ) -> tuple[list[Any], list[Any]]: ...


@dataclasses.dataclass(frozen=True)
class PromptSet:
    """Predeclared prompts whose positions create the required batch shapes."""

    target: str
    equal: str
    short: str
    long: str
    shared_target: str
    shared_peer: str
    fourth: str

    def by_name(self) -> dict[str, str]:
        return {
            "target": self.target,
            "equal": self.equal,
            "short": self.short,
            "long": self.long,
            "shared_target": self.shared_target,
            "shared_peer": self.shared_peer,
            "fourth": self.fourth,
        }


@dataclasses.dataclass(frozen=True)
class BatchCase:
    """One planned submission order and the target request's position in it."""

    name: str
    request_names: tuple[str, ...]
    target_index: int
    reference_case: str | None
    comparison_policy: str | None

    @property
    def target_name(self) -> str:
        return self.request_names[self.target_index]


def default_prompt_set() -> PromptSet:
    """Build nonempty prompt variants without relying on a chat template."""
    return PromptSet(
        target=DEFAULT_TARGET_PROMPT,
        equal=(
            "You are a careful assistant. Answer the following arithmetic question with one "
            "short sentence.\n\nQuestion: If 9 birds are on a branch and 4 fly away, how "
            "many birds remain?\nAnswer:"
        ),
        short="Answer with one word: What is the capital of France?",
        long=(
            "Read the following notes, then answer the final question in one short sentence.\n\n"
            + " ".join(
                f"Note {index}: a deterministic scheduler must retain request state separately."
                for index in range(1, 91)
            )
            + "\n\nQuestion: What must remain separate for every request?\nAnswer:"
        ),
        shared_target=(
            DEFAULT_SHARED_PREFIX
            + "Question: What information identifies a catalog entry?\nAnswer:"
        ),
        shared_peer=(
            DEFAULT_SHARED_PREFIX
            + "Question: What two fields can a reader use to find a book?\nAnswer:"
        ),
        fourth="Answer in one short sentence: Why should test evidence retain raw token IDs?",
    )


def default_cases() -> list[BatchCase]:
    """Cover the first-pass isolation matrix with explicit submission orders."""
    return [
        BatchCase("single-1", ("target",), 0, None, None),
        BatchCase("single-2", ("target",), 0, "single-1", "exact"),
        BatchCase(
            "equal-length", ("target", "equal"), 0, "single-1", "allow-numeric-variance"
        ),
        BatchCase(
            "long-short",
            ("target", "long", "short"),
            0,
            "single-1",
            "allow-numeric-variance",
        ),
        BatchCase("reversed", ("equal", "target"), 1, "equal-length", "exact"),
        BatchCase("shared-prefix-single", ("shared_target",), 0, None, None),
        BatchCase(
            "shared-prefix",
            ("shared_target", "shared_peer"),
            0,
            "shared-prefix-single",
            "allow-numeric-variance",
        ),
        BatchCase(
            "shared-prefix-reversed",
            ("shared_peer", "shared_target"),
            1,
            "shared-prefix",
            "exact",
        ),
        BatchCase(
            "four-request",
            ("target", "equal", "long", "fourth"),
            0,
            "single-1",
            "allow-numeric-variance",
        ),
    ]


def validate_cases(cases: list[BatchCase], prompts: PromptSet) -> None:
    """Reject an ambiguous experiment plan before an engine starts."""
    available = prompts.by_name()
    known = set()
    for case in cases:
        if not case.name or case.name in known:
            raise ValueError(
                f"Every batch case needs a unique nonempty name: {case.name!r}"
            )
        known.add(case.name)
        if not case.request_names:
            raise ValueError(f"Batch case {case.name!r} has no requests")
        if not 0 <= case.target_index < len(case.request_names):
            raise ValueError(
                f"Batch case {case.name!r} has target_index {case.target_index} outside "
                f"its {len(case.request_names)} requests"
            )
        missing = set(case.request_names) - set(available)
        if missing:
            raise ValueError(
                f"Batch case {case.name!r} references unknown prompts: {missing}"
            )
        if (case.reference_case is None) != (case.comparison_policy is None):
            raise ValueError(
                f"Batch case {case.name!r} must set both reference_case and "
                "comparison_policy, or neither"
            )
        if case.reference_case is not None and case.reference_case not in known:
            raise ValueError(
                f"Batch case {case.name!r} must reference an earlier case, not "
                f"{case.reference_case!r}"
            )
        if case.comparison_policy not in (None, "exact", "allow-numeric-variance"):
            raise ValueError(
                f"Batch case {case.name!r} has unsupported comparison policy "
                f"{case.comparison_policy!r}"
            )


def completion_record(completion: Any) -> dict[str, Any]:
    """Copy the comparable native fields into JSON-safe, immutable evidence."""
    return {
        "output_ids": list(completion.output_ids),
        "output_token_logprobs": list(completion.output_token_logprobs),
        "finish_reason": completion.finish_reason,
        "finish_details": completion.finish_details,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "cached_tokens": completion.cached_tokens,
        "input_token_logprobs": (
            None
            if completion.input_token_logprobs is None
            else [
                list(completion.input_token_logprobs[0]),
                list(completion.input_token_logprobs[1]),
            ]
        ),
    }


def first_difference(expected: list[Any], actual: list[Any]) -> dict[str, Any] | None:
    """Locate the first unequal value, including a length-only difference."""
    for position, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return {"position": position, "expected": left, "actual": right}
    if len(expected) != len(actual):
        position = min(len(expected), len(actual))
        return {
            "position": position,
            "expected": expected[position] if position < len(expected) else None,
            "actual": actual[position] if position < len(actual) else None,
        }
    return None


def comparison_record(
    reference: dict[str, Any], candidate: dict[str, Any], *, comparison_policy: str
) -> dict[str, Any]:
    """Compare targets under an explicit exact or numeric-variance policy."""
    token_difference = first_difference(
        reference["output_ids"], candidate["output_ids"]
    )
    logprob_difference = first_difference(
        reference["output_token_logprobs"], candidate["output_token_logprobs"]
    )
    finish_equal = (
        reference["finish_reason"] == candidate["finish_reason"]
        and reference["finish_details"] == candidate["finish_details"]
    )
    token_prefix = (
        len(reference["output_ids"])
        if token_difference is None
        else token_difference["position"]
    )
    logprob_before_token_divergence = first_difference(
        reference["output_token_logprobs"][:token_prefix],
        candidate["output_token_logprobs"][:token_prefix],
    )
    values_equal = token_difference is None and logprob_difference is None
    if values_equal and finish_equal:
        status = "PASS"
    elif comparison_policy == "allow-numeric-variance" and finish_equal:
        status = "NUMERIC_VARIANCE"
    else:
        status = "FAIL"
    return {
        "status": status,
        "comparison_policy": comparison_policy,
        "tokens_equal": token_difference is None,
        "logprobs_equal": logprob_difference is None,
        "finish_equal": finish_equal,
        "first_token_difference": token_difference,
        "first_logprob_difference": logprob_difference,
        "first_logprob_difference_before_token_divergence": logprob_before_token_divergence,
        "reference_finish": {
            "reason": reference["finish_reason"],
            "details": reference["finish_details"],
        },
        "candidate_finish": {
            "reason": candidate["finish_reason"],
            "details": candidate["finish_details"],
        },
    }


def load_samples_record(samples: list[Any]) -> list[dict[str, Any]]:
    """Normalize WireDriver scheduler observations while preserving all samples."""
    return [
        {
            "elapsed_s": sample.elapsed_s,
            "engines": [list(engine) for engine in sample.engines],
            "in_flight": list(sample.in_flight),
        }
        for sample in samples
    ]


def batch_observation(
    samples: list[dict[str, Any]], expected_requests: int
) -> dict[str, Any]:
    """State exactly what the sampled load evidence can and cannot establish."""
    max_running = max(
        (engine[0] for sample in samples for engine in sample["engines"]),
        default=0,
    )
    observed_concurrent_requests = any(
        engine[0] >= 2 and in_flight >= 2
        for sample in samples
        for engine, in_flight in zip(
            sample["engines"], sample["in_flight"], strict=True
        )
    )
    return {
        "samples": len(samples),
        "max_running_requests": max_running,
        "observed_concurrent_requests": observed_concurrent_requests,
        "expected_requests": expected_requests,
        "conclusion": (
            "observed at least two running requests in one frontend load sample"
            if observed_concurrent_requests
            else "inconclusive: no sampled load snapshot showed two running requests"
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--numerics",
        choices=("auto", "rl-bitwise"),
        default="auto",
        help="engine numerical contract; rl-bitwise requires exact comparisons",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--load-poll-interval", type=float, default=0.01)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-prefix-cache-contrast",
        action="store_true",
        help="also run cold and warm shared-prefix cases in separate cache-enabled engines",
    )
    args = parser.parse_args(argv)
    if not args.model.strip():
        parser.error("model must be nonempty")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.max_model_len < 1 or args.max_num_seqs < 4:
        parser.error(
            "max-model-len must be positive and max-num-seqs must be at least 4"
        )
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("startup-timeout must be positive and finite")
    if not math.isfinite(args.load_poll_interval) or args.load_poll_interval <= 0:
        parser.error("load-poll-interval must be positive and finite")
    if (
        not math.isfinite(args.gpu_memory_utilization)
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        parser.error("gpu-memory-utilization must be in (0, 1]")
    return args


def engine_args(args: argparse.Namespace, *, prefix_caching: bool) -> list[str]:
    """Make every first-pass engine choice explicit and reproducible."""
    return [
        "--model",
        args.model,
        "--dtype",
        "bfloat16",
        "--numerics",
        args.numerics,
        "--tensor-parallel-size",
        "1",
        "--seed",
        str(args.seed),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        # The scheduler requires a positive per-round budget. A full-context
        # budget preserves the first pass's non-chunked behavior for this run.
        "--chunked-prefill-size",
        str(args.max_model_len),
        "--disable-prefix-caching" if not prefix_caching else "--enable-prefix-caching",
        "--disable-prefill-graph",
        "--enforce-eager",
        "--disable-kvstore",
        "--enable-output-logprobs",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]


def tokenizer_for(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=False)


def encode_prompts(tokenizer: Any, prompts: PromptSet) -> dict[str, list[int]]:
    input_ids = {}
    for name, prompt in prompts.by_name().items():
        encoded = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if (
            not isinstance(encoded, list)
            or not encoded
            or any(type(token) is not int or token < 0 for token in encoded)
        ):
            raise ValueError(
                f"Tokenizer did not return nonempty integer IDs for {name!r}"
            )
        input_ids[name] = encoded
    return input_ids


def run_cases(
    driver: Driver,
    *,
    input_ids: dict[str, list[int]],
    cases: list[BatchCase],
    max_new_tokens: int,
    poll_interval_s: float,
    records: dict[str, Any],
    args_numerics: str,
) -> None:
    """Submit each predeclared case and compare its named target with its reference."""
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    sampling_params = {"temperature": 0.0, "max_new_tokens": max_new_tokens}
    for case in cases:
        requests = [
            generate_request(input_ids[name], sampling_params, return_logprob=True)
            for name in case.request_names
        ]
        record = {
            "request_names": list(case.request_names),
            "target_index": case.target_index,
            "target_name": case.target_name,
            "requests": [
                {"rid": request.rid, "input_ids": request.input_ids}
                for request in requests
            ],
            "target_completion": None,
            "elapsed_s": None,
            "scheduler_load_samples": [],
            "batch_observation": None,
            "comparison": None,
        }
        records[case.name] = record
        started = time.monotonic()
        try:
            completions, samples = driver.generate_with_loads(
                requests, poll_interval_s=poll_interval_s
            )
            if len(completions) != len(requests):
                raise RuntimeError(
                    f"{case.name}: expected {len(requests)} completions, got {len(completions)}"
                )
            target = completion_record(completions[case.target_index])
            sample_records = load_samples_record(samples)
            record["target_completion"] = target
            record["scheduler_load_samples"] = sample_records
            record["batch_observation"] = batch_observation(
                sample_records, len(requests)
            )
            if case.reference_case is not None:
                assert case.comparison_policy is not None
                comparison_policy = (
                    "exact" if args_numerics == "rl-bitwise" else case.comparison_policy
                )
                record["comparison"] = comparison_record(
                    records[case.reference_case]["target_completion"],
                    target,
                    comparison_policy=comparison_policy,
                )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - started


def run_prefix_cache_contrast(
    args: argparse.Namespace,
    *,
    input_ids: dict[str, list[int]],
) -> dict[str, Any]:
    """Run cold and warm shared-prefix pairs in separate cache-enabled engines."""
    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver, generate_request

    result = {}
    sampling_params = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens}
    for phase in ("cold", "warm"):
        with WireDriver(
            engine_args(args, prefix_caching=True),
            num_engines=1,
            routing="round_robin",
            startup_timeout_s=args.startup_timeout,
        ) as driver:
            if phase == "warm":
                driver.generate(
                    [
                        generate_request(
                            input_ids["shared_peer"],
                            sampling_params,
                            return_logprob=True,
                        )
                    ]
                )
            requests = [
                generate_request(input_ids[name], sampling_params, return_logprob=True)
                for name in ("shared_target", "shared_peer")
            ]
            completions, samples = driver.generate_with_loads(
                requests, poll_interval_s=args.load_poll_interval
            )
            target = completion_record(completions[0])
            result[phase] = {
                "target_completion": target,
                "scheduler_load_samples": load_samples_record(samples),
                "batch_observation": batch_observation(
                    load_samples_record(samples), len(requests)
                ),
            }
    result["comparison"] = comparison_record(
        result["cold"]["target_completion"],
        result["warm"]["target_completion"],
        comparison_policy="exact",
    )
    result["cached_tokens"] = {
        phase: result[phase]["target_completion"]["cached_tokens"]
        for phase in ("cold", "warm")
    }
    cold_cached = result["cached_tokens"]["cold"]
    warm_cached = result["cached_tokens"]["warm"]
    result["cache_evidence"] = {
        "status": "PASS" if warm_cached > cold_cached else "INCONCLUSIVE",
        "conclusion": (
            "warm target served more prompt tokens from cache than cold target"
            if warm_cached > cold_cached
            else "inconclusive: warm target did not report more cached prompt tokens"
        ),
    }
    return result


def run(args: argparse.Namespace, *, driver: Driver | None = None) -> dict[str, Any]:
    """Run the first-pass matrix and report exact failures separately from variance."""
    directory = args.output_dir.resolve()
    if not directory.is_dir() or any(directory.iterdir()):
        raise ValueError("output-dir must already exist and be empty")
    prompts = default_prompt_set()
    cases = default_cases()
    validate_cases(cases, prompts)
    engine_configuration = engine_args(args, prefix_caching=False)
    report = {
        "status": "INVALID",
        "model": args.model,
        "first_pass_scope": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "prefix_caching": False,
            "chunked_prefill_size": args.max_model_len,
            "cuda_graphs": False,
            "numerics": args.numerics,
        },
        "engine_args": engine_configuration,
        "cases": {},
        "prefix_cache_contrast": None,
    }
    write_json(directory / "run.json", {**vars(args), "output_dir": str(directory)})
    try:
        tokenizer = tokenizer_for(args.model)
        input_ids = encode_prompts(tokenizer, prompts)
        write_json(directory / "prompts.json", input_ids)
        write_json(
            directory / "plan.json",
            [dataclasses.asdict(case) for case in cases],
        )
        from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

        engine_context = (
            nullcontext(driver)
            if driver is not None
            else WireDriver(
                engine_configuration,
                num_engines=1,
                routing="round_robin",
                startup_timeout_s=args.startup_timeout,
            )
        )
        with engine_context as active_driver:
            if (
                active_driver.server_args != engine_configuration
                or active_driver.num_engines != 1
                or active_driver.routing != "round_robin"
                or active_driver.startup_timeout_s != args.startup_timeout
            ):
                raise ValueError(
                    "borrowed driver does not match the declared first-pass engine"
                )
            report["engines"] = active_driver.engines()
            run_cases(
                active_driver,
                input_ids=input_ids,
                cases=cases,
                max_new_tokens=args.max_new_tokens,
                poll_interval_s=args.load_poll_interval,
                records=report["cases"],
                args_numerics=args.numerics,
            )
        if args.run_prefix_cache_contrast:
            report["prefix_cache_contrast"] = run_prefix_cache_contrast(
                args, input_ids=input_ids
            )
        comparisons = [
            case["comparison"]
            for case in report["cases"].values()
            if case["comparison"] is not None
        ]
        if report["prefix_cache_contrast"] is not None:
            comparisons.append(report["prefix_cache_contrast"]["comparison"])
        report["status"] = (
            "FAIL"
            if any(item["status"] == "FAIL" for item in comparisons)
            else (
                "NUMERIC_VARIANCE"
                if any(item["status"] == "NUMERIC_VARIANCE" for item in comparisons)
                else "PASS"
            )
        )
    except Exception as exc:  # noqa: BLE001 - preserve the original experiment failure.
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_json(directory / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (OSError, ValueError) as exc:
        print(f"BATCH_STATE_ISOLATION status=INVALID error={exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return {"PASS": 0, "NUMERIC_VARIANCE": 0, "FAIL": 1, "INVALID": 2}[
        report["status"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
