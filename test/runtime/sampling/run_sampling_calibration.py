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

"""Fixed-sample native calibration of generated tokens against reported probabilities.

Requires --enable-output-logprobs --enable-output-cumulative-probs. Chain
speculative verification is supported; draft trees are not. Statistical
analysis uses only the standard library; the runner tokenizes prompts and
drives the engine with WireDriver.
See docs/guides/sampling-calibration.md for the hypotheses and power limits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import secrets
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspeed.runtime.entrypoints.wire_driver import Completion, WireDriver


class CalibrationError(ValueError):
    """The experiment did not produce the predeclared, aligned observations."""


class BinomialDistribution:
    """Inclusive binomial tails computed by normalized probability recurrences."""

    def __init__(self, trials: int, probability: float):
        if type(trials) is not int or trials < 1:
            raise ValueError("Binomial trials must be a positive integer")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Binomial probability must be in [0, 1]")
        self.trials = trials
        self.probability = probability
        mass = [0.0] * (trials + 1)
        mode = min(trials, int((trials + 1) * probability))
        mass[mode] = 1.0
        if 0 < probability < 1:
            odds = probability / (1 - probability)
            for k in range(mode, trials):
                mass[k + 1] = mass[k] * (trials - k) / (k + 1) * odds
            for k in range(mode, 0, -1):
                mass[k - 1] = mass[k] * k / (trials - k + 1) / odds
        normalizer = math.fsum(mass)
        self.mass = [value / normalizer for value in mass]
        self.lower = [0.0] * (trials + 1)
        self.upper = [0.0] * (trials + 1)
        total = 0.0
        for k in range(trials + 1):
            total += self.mass[k]
            self.lower[k] = min(1.0, total)
        total = 0.0
        for k in range(trials, -1, -1):
            total += self.mass[k]
            self.upper[k] = min(1.0, total)

    def at_most(self, count: int) -> float:
        if count < 0:
            return 0.0
        if count >= self.trials:
            return 1.0
        return self.lower[count]

    def at_least(self, count: int) -> float:
        if count <= 0:
            return 1.0
        if count > self.trials:
            return 0.0
        return self.upper[count]

    def critical_counts(self, tail_alpha: float) -> tuple[int, int]:
        """Return inclusive rejection bounds; -1/n+1 mean unreachable tails."""
        if not math.isfinite(tail_alpha) or not 0 < tail_alpha < 0.5:
            raise ValueError("Each binomial tail alpha must be in (0, 0.5)")
        low = -1
        high = self.trials + 1
        for k, value in enumerate(self.lower):
            if value > tail_alpha:
                break
            low = k
        for k, value in enumerate(self.upper):
            if value <= tail_alpha:
                high = k
                break
        return low, high


@dataclass(frozen=True)
class RequestPlan:
    index: int
    prompt: str
    tokens: int


def plan_requests(
    prompts: list[str], *, mode: str, samples: int, tokens_per_request: int
) -> list[RequestPlan]:
    """Assign every counted output position before observing any response."""
    if samples < 1 or tokens_per_request < 1:
        raise ValueError("Samples and tokens per request must be positive")
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Prompts must be nonempty strings")
    if mode == "prefill":
        if len(prompts) != 1 or tokens_per_request != 1:
            raise ValueError(
                "Prefill mode requires one prompt and one token per request"
            )
        return [RequestPlan(i, prompts[0], 1) for i in range(samples)]
    if mode != "gsm8k":
        raise ValueError(f"Unknown experiment mode: {mode}")
    needed = (samples + tokens_per_request - 1) // tokens_per_request
    if len(prompts) < needed:
        raise ValueError(f"Need {needed} dataset prompts, received {len(prompts)}")
    return [
        RequestPlan(
            i, prompts[i], min(tokens_per_request, samples - i * tokens_per_request)
        )
        for i in range(needed)
    ]


def sampling_parameters(plan: RequestPlan) -> dict:
    """Keep the full temperature-one distribution and every sampled token."""
    return {
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": plan.tokens,
        "ignore_eos": True,
        "no_stop_trim": True,
        "skip_special_tokens": False,
    }


@dataclass(frozen=True)
class Observation:
    request_index: int
    output_position: int
    phase: str
    token_id: int
    logprob: float
    cumulative_probability: float
    jitter: float
    quantile: float
    bin_index: int


@dataclass(frozen=True)
class SamplingResponse:
    """The aligned native output columns consumed by the statistical collector."""

    output_ids: list[int]
    output_token_logprobs: list[float]
    output_token_cumulative_probs: list[float]
    finish_reason: str | None
    finish_details: dict | None

    @classmethod
    def from_completion(cls, completion: Completion) -> SamplingResponse:
        return cls(
            output_ids=completion.output_ids,
            output_token_logprobs=completion.output_token_logprobs,
            output_token_cumulative_probs=completion.output_token_cumulative_probs,
            finish_reason=completion.finish_reason,
            finish_details=completion.finish_details,
        )


class QuantileCalibration:
    """Validate complete responses, then test one predeclared pooled sample."""

    def __init__(self, *, samples: int, bins: int, alpha: float, client_rng_seed: int):
        if type(samples) is not int or samples < 1:
            raise ValueError("Samples must be a positive integer")
        if type(bins) is not int or bins < 2:
            raise ValueError("Bins must be an integer of at least two")
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("Alpha must be in (0, 1)")
        self.samples = samples
        self.bins = bins
        self.alpha = alpha
        self.client_rng_seed = client_rng_seed
        self.rng = random.Random(client_rng_seed)
        self.counts = [0] * bins
        self.phase_counts = {"prefill": [0] * bins, "decode": [0] * bins}
        self.observations: list[Observation] = []
        self.next_request_index = 0

    def add_response(
        self, plan: RequestPlan, response: SamplingResponse
    ) -> list[Observation]:
        """Reject missing/misaligned metadata rather than selecting usable tokens."""
        if plan.index != self.next_request_index:
            raise CalibrationError(
                "Responses must be consumed in predeclared request order"
            )
        if len(self.observations) + plan.tokens > self.samples:
            raise CalibrationError("Response exceeds the predeclared sample size")
        if response.finish_reason not in {"length", "stop"}:
            raise CalibrationError(
                f"Request {plan.index} did not finish successfully: {response.finish_details}"
            )
        ids = response.output_ids
        logprobs = response.output_token_logprobs
        cumulative = response.output_token_cumulative_probs
        arrays = (ids, logprobs, cumulative)
        if any(not isinstance(values, list) for values in arrays):
            raise CalibrationError(
                f"Request {plan.index} needs output_ids, output_token_logprobs and "
                "output_token_cumulative_probs arrays; enable both engine diagnostics"
            )
        if any(len(values) != plan.tokens for values in arrays):
            raise CalibrationError(
                f"Request {plan.index} expected {plan.tokens} aligned tokens, "
                f"received lengths {[len(values) for values in arrays]}"
            )
        validated = []
        for position, (token, lp, prefix) in enumerate(
            zip(ids, logprobs, cumulative, strict=True)
        ):
            if type(token) is not int or token < 0:
                raise CalibrationError(
                    f"Request {plan.index} has an invalid output token"
                )
            if (
                type(lp) not in (int, float)
                or not math.isfinite(lp)
                or lp > 0
                or type(prefix) not in (int, float)
                or not math.isfinite(prefix)
                or not 0 <= prefix <= 1
            ):
                raise CalibrationError(
                    f"Request {plan.index}, token {position}: invalid probability metadata"
                )
            probability = math.exp(lp)
            # Logprob and prefix mass use different floating-point reductions.
            # Permit only a small endpoint error, never a missing probability.
            if probability == 0 or prefix + probability > 1 + 2e-6:
                raise CalibrationError(
                    f"Request {plan.index}, token {position}: inconsistent CDF interval"
                )
            validated.append((position, token, float(lp), float(prefix), probability))

        added = []
        for position, token, lp, prefix, probability in validated:
            jitter = self.rng.random()
            quantile = min(math.nextafter(1.0, 0.0), prefix + jitter * probability)
            bucket = min(self.bins - 1, int(quantile * self.bins))
            phase = "prefill" if position == 0 else "decode"
            observation = Observation(
                plan.index, position, phase, token, lp, prefix, jitter, quantile, bucket
            )
            self.counts[bucket] += 1
            self.phase_counts[phase][bucket] += 1
            self.observations.append(observation)
            added.append(observation)
        self.next_request_index += 1
        return added

    def report(self) -> dict:
        if len(self.observations) != self.samples:
            raise CalibrationError(
                f"Need exactly {self.samples} samples, received {len(self.observations)}"
            )
        null_probability = 1 / self.bins
        null = BinomialDistribution(self.samples, null_probability)
        tail_alpha = self.alpha / (2 * self.bins)
        low, high = null.critical_counts(tail_alpha)
        rows = []
        for bucket, count in enumerate(self.counts):
            lower = null.at_most(count)
            upper = null.at_least(count)
            rows.append(
                {
                    "bin": bucket,
                    "mass_interval": [bucket / self.bins, (bucket + 1) / self.bins],
                    "count": count,
                    "observed_fraction": count / self.samples,
                    "lower_tail_p": lower,
                    "upper_tail_p": upper,
                    "rejected": lower <= tail_alpha or upper <= tail_alpha,
                }
            )
        familywise_p = min(
            1.0,
            2
            * self.bins
            * min(min(row["lower_tail_p"], row["upper_tail_p"]) for row in rows),
        )
        power = []
        for increase in (0.01, 0.02):
            alternative = null_probability + increase
            distribution = BinomialDistribution(self.samples, alternative)
            power.append(
                {
                    "actual_bin_probability": alternative,
                    "absolute_increase": increase,
                    "power": min(
                        1.0, distribution.at_most(low) + distribution.at_least(high)
                    ),
                }
            )
        return {
            "status": "FAIL" if familywise_p <= self.alpha else "PASS",
            "scope": "one pooled fixed-sample probability-quantile test",
            "samples": self.samples,
            "bins": self.bins,
            "alpha": self.alpha,
            "familywise_pvalue": familywise_p,
            "expected_fraction_per_bin": null_probability,
            "tail_alpha": tail_alpha,
            "reject_count_at_or_below": low,
            "reject_count_at_or_above": high,
            "bin_results": rows,
            "phase_counts": self.phase_counts,
            "phase_counts_are_descriptive_only": True,
            "phase_definition": "first output token=prefill; subsequent output tokens=decode",
            "reference_power": power,
            "power_assumption": "independent trials with a stable actual bin probability",
            "client_rng_seed": self.client_rng_seed,
            "selection": "predeclared request order and exact per-request token quotas",
            "server_configuration_verified": False,
        }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def json_snapshot(value: Any) -> Any:
    """Keep invalid native float diagnostics readable in strict raw JSON records."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_snapshot(item) for item in value]
    return value


def read_dataset(path: Path, *, question_field: str, prompt_template: str) -> list[str]:
    if "{question}" not in prompt_template:
        raise ValueError("Prompt template must contain {question}")
    prompts = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        question = row.get(question_field) if isinstance(row, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Dataset line {line_number} has no nonempty {question_field} string"
            )
        prompts.append(prompt_template.replace("{question}", question))
    return prompts


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("gsm8k", "prefill"))
    for name in ("model", "run-name", "client-rng-seed"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("samples", "bins", "batch-size", "num-engines"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument(
        "--routing",
        choices=("round_robin", "least_in_flight", "minimum_cache_usage"),
        required=True,
    )
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument(
        "--add-special-tokens", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--startup-timeout", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--question-field")
    parser.add_argument("--prompt-template")
    parser.add_argument("--tokens-per-request", type=int)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_engines < 1:
        parser.error("batch-size and num-engines must be positive")
    if not args.model.strip() or not args.run_name.strip():
        parser.error("model and run-name must be nonempty")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("startup-timeout must be finite and positive")
    if args.mode == "gsm8k":
        if any(
            value is None
            for value in (
                args.dataset,
                args.question_field,
                args.prompt_template,
                args.tokens_per_request,
            )
        ):
            parser.error(
                "gsm8k requires dataset, question-field, prompt-template and tokens-per-request"
            )
        if args.prompt is not None or args.prompt_file is not None:
            parser.error("prompt and prompt-file apply only to prefill mode")
    else:
        if args.prompt is None and args.prompt_file is None:
            parser.error("prefill requires prompt or prompt-file")
        if any(
            value is not None
            for value in (
                args.dataset,
                args.question_field,
                args.prompt_template,
                args.tokens_per_request,
            )
        ):
            parser.error("dataset arguments apply only to gsm8k mode")
    return args


def run(
    args: argparse.Namespace,
    *,
    driver: WireDriver | None = None,
) -> dict:
    """Run one fixed-sample experiment, optionally borrowing an existing engine.

    Without a driver, the run starts its engine from ``args`` and closes it.
    A borrowed driver is neither entered nor closed; it must have been started
    exactly as ``args`` describes (engine-config, --num-engines, --routing and
    --startup-timeout), or the run is invalid before it submits anything.
    """
    directory = args.output_dir.resolve()
    if not directory.is_dir() or any(directory.iterdir()):
        raise ValueError("Output directory must already exist and be empty")
    rng_seed = (
        secrets.randbits(128)
        if args.client_rng_seed == "random"
        else int(args.client_rng_seed)
    )
    calibration = QuantileCalibration(
        samples=args.samples, bins=args.bins, alpha=args.alpha, client_rng_seed=rng_seed
    )
    raw_directory = directory / "raw"
    raw_directory.mkdir()
    write_json(
        directory / "run.json",
        {
            **vars(args),
            "output_dir": str(directory),
            "engine_config": str(args.engine_config),
            "dataset": str(args.dataset) if args.dataset is not None else None,
            "prompt_file": (
                str(args.prompt_file) if args.prompt_file is not None else None
            ),
            "resolved_client_rng_seed": rng_seed,
            "borrowed_driver": driver is not None,
        },
    )
    try:
        if args.mode == "gsm8k":
            prompts = read_dataset(
                args.dataset,
                question_field=args.question_field,
                prompt_template=args.prompt_template,
            )
            tokens_per_request = args.tokens_per_request
        else:
            prompts = [
                (
                    args.prompt_file.read_text(encoding="utf-8")
                    if args.prompt_file is not None
                    else args.prompt
                )
            ]
            tokens_per_request = 1
        plans = plan_requests(
            prompts,
            mode=args.mode,
            samples=args.samples,
            tokens_per_request=tokens_per_request,
        )
        write_json(directory / "plan.json", [asdict(plan) for plan in plans])
        server_args = json.loads(args.engine_config.read_text(encoding="utf-8"))
        if not isinstance(server_args, list) or any(
            not isinstance(value, str) for value in server_args
        ):
            raise ValueError(
                "engine-config must contain a JSON list of ServerArgs strings"
            )
        for flag in ("--enable-output-logprobs", "--enable-output-cumulative-probs"):
            if flag not in server_args:
                raise ValueError(f"engine-config must explicitly enable {flag}")
        if driver is not None:
            # A borrowed engine must be the one these arguments describe: they
            # are recorded as the experiment's configuration.
            started = {
                "engine-config": driver.server_args,
                "--num-engines": driver.num_engines,
                "--routing": driver.routing,
                "--startup-timeout": driver.startup_timeout_s,
            }
            requested = {
                "engine-config": server_args,
                "--num-engines": args.num_engines,
                "--routing": args.routing,
                "--startup-timeout": args.startup_timeout,
            }
            mismatched = [
                name for name in requested if started[name] != requested[name]
            ]
            if mismatched:
                raise ValueError(
                    "the borrowed driver was not started with this run's "
                    f"{', '.join(mismatched)}"
                )
        write_json(directory / "engine-args.json", server_args)
        from transformers import AutoTokenizer

        from tokenspeed.runtime.entrypoints.wire_driver import (
            WireDriver,
            generate_request,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.tokenizer_trust_remote_code
        )
        input_ids = {}
        for plan in plans:
            # Cache tokenization only; every request still gets a fresh engine rid.
            if plan.prompt not in input_ids:
                encoded = tokenizer(
                    plan.prompt, add_special_tokens=args.add_special_tokens
                )["input_ids"]
                if (
                    not isinstance(encoded, list)
                    or not encoded
                    or any(type(token) is not int or token < 0 for token in encoded)
                ):
                    raise ValueError("Tokenizer must return nonempty integer input_ids")
                input_ids[plan.prompt] = encoded
        batches = [
            plans[i : i + args.batch_size]
            for i in range(0, len(plans), args.batch_size)
        ]
        with (directory / "observations.jsonl").open(
            "w", encoding="utf-8"
        ) as observations:
            engine_context = (
                nullcontext(driver)
                if driver is not None
                else WireDriver(
                    server_args,
                    num_engines=args.num_engines,
                    routing=args.routing,
                    startup_timeout_s=args.startup_timeout,
                )
            )
            with engine_context as active_driver:
                write_json(directory / "engines.json", active_driver.engines())
                progress = -1
                for batch_index, batch in enumerate(batches):
                    requests = [
                        generate_request(
                            input_ids[plan.prompt],
                            sampling_parameters(plan),
                            return_logprob=True,
                        )
                        for plan in batch
                    ]
                    record = {
                        "request_indices": [plan.index for plan in batch],
                        "requests": [
                            {
                                "rid": request.rid,
                                "input_ids": request.input_ids,
                                "sampling_params": sampling_parameters(plan),
                            }
                            for plan, request in zip(batch, requests, strict=True)
                        ],
                    }
                    path = raw_directory / f"{batch_index:06d}.json"
                    write_json(path, record)
                    try:
                        completions = active_driver.generate(requests)
                        record["completions"] = json_snapshot(
                            [asdict(completion) for completion in completions]
                        )
                        write_json(path, record)
                        if len(completions) != len(batch):
                            raise CalibrationError(
                                "Native completions do not match the requested batch size"
                            )
                        # WireDriver returns submission order, not completion order.
                        for plan, output in zip(batch, completions, strict=True):
                            response = SamplingResponse.from_completion(output)
                            for observation in calibration.add_response(plan, response):
                                observations.write(
                                    json.dumps(asdict(observation), allow_nan=False)
                                    + "\n"
                                )
                            observations.flush()
                        current_progress = (
                            len(calibration.observations) * 10 // args.samples
                        )
                        if current_progress > progress:
                            print(
                                f"CALIBRATION collected={len(calibration.observations)}/{args.samples}",
                                file=sys.stderr,
                                flush=True,
                            )
                            progress = current_progress
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        write_json(path, record)
                        raise
        report = calibration.report()
        report.update(run_name=args.run_name, mode=args.mode, requests=len(plans))
    except Exception as exc:
        report = {
            "status": "INVALID",
            "run_name": args.run_name,
            "mode": args.mode,
            "error": f"{type(exc).__name__}: {exc}",
            "validated_samples": len(calibration.observations),
            "requested_samples": args.samples,
        }
    write_json(directory / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (ValueError, OSError) as exc:
        print(f"CALIBRATION status=INVALID error={exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
