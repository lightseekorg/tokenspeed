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

"""Collect isolated prefill requests using TokenSpeed's native WireDriver."""

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from test.cache_gds_calibration.data import (
    PrefillCase,
    completed_repeats,
    positive,
    prefill_samples,
    validate_cases,
    validate_prefix,
)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "engine-args",
        "model-path",
        "text-file",
        "deployment",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).resolve().with_name("cases_quick.json")),
        help="prefix/new-token matrix (default: cases_quick.json); select cases.json for the full sweep",
    )
    for name in (
        "num-engines",
        "data-parallel-rank",
        "repeats",
        "warmup-repeats",
        "seed",
        "max-prefix-shortfall",
    ):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument(
        "--routing", choices=("round_robin", "least_in_flight"), required=True
    )
    parser.add_argument("--trust-remote-code", choices=("yes", "no"), required=True)
    parser.add_argument("--startup-timeout", type=positive, required=True)
    parser.add_argument(
        "--resume-requests",
        nargs="+",
        default=[],
        help="completed native request JSONL files from the same deployment; reuse matching formal repeats",
    )
    return parser


def main():
    args = parser().parse_args()
    if (
        args.repeats < 1
        or args.warmup_repeats < 1
        or not 0 <= args.data_parallel_rank < args.num_engines
        or args.max_prefix_shortfall < 0
    ):
        raise ValueError("invalid repeats or DP rank")
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver, generate_request

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    engine_args = json.loads(Path(args.engine_args).read_text())
    cases = [PrefillCase(**case) for case in json.loads(Path(args.cases).read_text())]
    validate_cases(cases)
    completed = completed_repeats(
        cases,
        args.resume_requests,
        repeats=args.repeats,
        deployment=args.deployment,
        max_prefix_shortfall=args.max_prefix_shortfall,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code == "yes"
    )
    corpus = Path(args.text_file).read_text()
    pool = tokenizer.encode(corpus, add_special_tokens=False)
    if len(set(pool)) < 32:
        raise ValueError(
            "representative corpus requires at least 32 distinct token IDs"
        )
    manifest = {
        **vars(args),
        "engine_args": engine_args,
        "cases": [dataclasses.asdict(case) for case in cases],
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "timing_source": "plain-request-stats",
        "started_unix_s": time.time(),
        "reused_formal_requests": len(completed),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    samples = prefill_samples(
        cases,
        pool,
        repeats=args.repeats,
        warmup_repeats=args.warmup_repeats,
        seed=args.seed,
    )
    collected = 0
    with WireDriver(
        engine_args,
        num_engines=args.num_engines,
        routing=args.routing,
        startup_timeout_s=args.startup_timeout,
    ) as driver, (output / "requests.jsonl").open("x") as records:
        (output / "engines.json").write_text(json.dumps(driver.engines(), indent=2))
        smoke_request = generate_request(
            tokenizer.encode("The sum of 1 and 1 is", add_special_tokens=False),
            {"max_new_tokens": 8, "temperature": 0},
            data_parallel_rank=args.data_parallel_rank,
            created_time=time.time(),
        )
        smoke = driver.generate([smoke_request])[0]
        (output / "smoke.json").write_text(
            json.dumps(
                {
                    **dataclasses.asdict(smoke),
                    "text": tokenizer.decode(smoke.output_ids),
                },
                indent=2,
            )
        )
        if smoke.finish_reason == "abort":
            raise RuntimeError(f"smoke failed: {smoke.finish_details}")
        for sample in samples:
            prefix_tokens = sample.case.prefix_tokens
            prompt_tokens = sample.case.prompt_tokens
            repeat, ids = sample.repeat, sample.input_ids
            if (
                not sample.warmup
                and (prefix_tokens, prompt_tokens, repeat) in completed
            ):
                continue
            sampling = {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True}
            if prefix_tokens:
                prime_req = generate_request(
                    ids[:prefix_tokens],
                    sampling,
                    data_parallel_rank=args.data_parallel_rank,
                    created_time=time.time(),
                )
                prime = driver.generate([prime_req])[0]
                if prime.finish_reason == "abort":
                    raise RuntimeError(f"prime failed: {prime.finish_details}")
            req = generate_request(
                ids,
                sampling,
                data_parallel_rank=args.data_parallel_rank,
                created_time=time.time(),
            )
            print(
                f"CAL_START {req.rid} prefix_tokens={prefix_tokens} "
                f"new_tokens={sample.case.new_tokens} prompt_tokens={prompt_tokens} repeat={repeat}",
                flush=True,
            )
            started = time.perf_counter()
            completion = driver.generate([req])[0]
            row = {
                "kind": "request",
                "deployment": args.deployment,
                "rid": req.rid,
                "split": sample.case.split,
                "expected_a": prefix_tokens,
                "expected_b": prompt_tokens,
                "warmup": repeat < 0,
                "repeat": repeat,
                "seed": args.seed,
                "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "wire_elapsed_s": time.perf_counter() - started,
                "completion": dataclasses.asdict(completion),
            }
            records.write(json.dumps(row) + "\n")
            records.flush()
            print(
                f"CAL_DONE {req.rid} cached={completion.cached_tokens} seconds={row['wire_elapsed_s']:.6f}",
                flush=True,
            )
            if completion.finish_reason == "abort" or completion.completion_tokens != 1:
                raise RuntimeError(f"measurement failed: {completion.finish_details}")
            validate_prefix(
                expected_prefix_tokens=prefix_tokens,
                expected_prompt_tokens=prompt_tokens,
                prompt_tokens=completion.prompt_tokens,
                cached_tokens=completion.cached_tokens,
                max_prefix_shortfall=args.max_prefix_shortfall,
            )
            collected += 1
    (output / "complete.json").write_text(
        json.dumps(
            {
                "finished_unix_s": time.time(),
                "requests": collected,
                "reused_formal_requests": len(completed),
            }
        )
    )


if __name__ == "__main__":
    main()
