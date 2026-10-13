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

"""Check HY V4's draft step-0 prefill graph against eager on real weights.

    python -m test.hy_v4_parity.draft_graph_check --source /path/to/hy_v4

Builds a truncated checkpoint (released layers bit for bit, plus the MTP
block), then serves one greedy MTP batch twice on one GPU: with the draft's
step-0 prefill graph, and with step 0 eager. The target keeps its graphs in
both runs. A small chunked-prefill size splits the long prompts across
forwards; ``--enable-mixed-batch`` also lets those chunks share a batch with
decodes, so step 0 follows mixed batches. The graph declines those, so they
run eager in both runs and the check judges the pure extends among them.

Only actual step-0 replays count as graph evidence; mixed/eager controls do
not dilute their verdict. Every replay's proposed id and top-2 gap is checked,
as are identical live inputs, full logits and all latent/index-K rows written
by that step. The snapshots are retained for diagnosis. Missing, truncated,
non-finite or non-comparable evidence fails, and generated token sequences must
match. This is a strict regression oracle, not an accuracy benchmark: a near
tie or divergent free-running schedule fails rather than being called a pass.
See test/runtime/test_draft_graph_validation.py for adversarial oracle tests.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test.hy_v4_parity.hy_v4_fixtures import (  # noqa: E402
    Truncation,
    build_truncated_checkpoint,
)


def _prompts(count: int, seed: int) -> list[list[int]]:
    """Token-id prompts spread from 9 to 1500 tokens; repeated spans let MTP accept."""
    rng = random.Random(seed)
    prompts = []
    for index in range(count):
        span = [rng.randrange(1000, 30000) for _ in range(rng.randrange(8, 40))]
        length = 9 + 1491 * index // max(count - 1, 1)
        prompts.append((span * (length // len(span) + 1))[:length])
    return prompts


def _run(args) -> int:
    """One engine run in this process; the parent sets the sitecustomize env."""
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=args.model,
        # The scheduler loads a tokenizer even with ids in and out; the
        # truncated checkpoint carries none.
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        kv_cache_quant_method="none",
        skip_tokenizer_init=True,
        speculative_algorithm="MTP",
        speculative_num_steps=args.steps,
        max_model_len=4096,
        max_num_seqs=args.prompts,
        chunked_prefill_size=256,
        enable_mixed_batch=args.enable_mixed_batch,
        enable_prefix_caching=False,
        disable_kvstore=True,
        disable_autotune=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        log_level="warning",
    )
    try:
        outputs = engine.generate(
            input_ids=_prompts(args.prompts, args.seed),
            sampling_params={"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        )
    finally:
        engine.shutdown()
    Path(args.out).write_text(
        json.dumps(
            [
                {
                    "output_ids": [int(token) for token in out["output_ids"]],
                    "verify_ct": int(out["meta_info"].get("spec_verify_ct", 0)),
                }
                for out in outputs
            ]
        )
    )
    return 0


def _spawn(mode: str, model: Path, workdir: Path, args) -> dict:
    """Serve the batch in a fresh interpreter; return outputs, counts and trace."""
    inject = workdir / "inject"
    inject.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_HERE / "draft_graph_sitecustomize.py", inject / "sitecustomize.py")
    stats = workdir / f"stats_{mode}"
    for stale in glob.glob(f"{stats}.*"):
        os.remove(stale)
    out = workdir / f"outputs_{mode}.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(inject), str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")]
        ),
        "HY_V4_DRAFT_GRAPH": mode,
        "HY_V4_DRAFT_GRAPH_STATS": str(stats),
    }
    command = [
        sys.executable,
        "-m",
        "test.hy_v4_parity.draft_graph_check",
        "--run",
        "--model",
        str(model),
        "--tokenizer",
        str(args.source),
        "--out",
        str(out),
        "--steps",
        str(args.steps),
        "--prompts",
        str(args.prompts),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ] + (["--enable-mixed-batch"] if args.enable_mixed_batch else [])
    print(f"\n== draft graph {mode}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(_REPO_ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"draft graph {mode} run exited with {completed.returncode}")
    counts = {"replays": 0, "eager_extends": 0}
    for path in glob.glob(f"{stats}.*[0-9]"):
        for key, value in json.loads(Path(path).read_text()).items():
            counts[key] += value
    trace = []
    for path in sorted(glob.glob(f"{stats}.*.trace")):
        trace += [json.loads(line) for line in Path(path).read_text().splitlines()]
    return {
        "outputs": json.loads(out.read_text()),
        "counts": counts,
        "trace": trace,
    }


def _output_summary(graph: list[dict], eager: list[dict]) -> str:
    """How many requests produced the same tokens and verify steps."""
    if len(graph) != len(eager):
        return f"request counts differ: {len(graph)} vs {len(eager)}"
    tokens = sum(g["output_ids"] == e["output_ids"] for g, e in zip(graph, eager))
    verify = sum(g["verify_ct"] == e["verify_ct"] for g, e in zip(graph, eager))
    return (
        f"outputs match for {tokens}/{len(graph)} requests, verify steps for "
        f"{verify}/{len(graph)}"
    )


def _compare_proposals(graph, eager, min_match, max_gap_delta):
    from test.runtime.draft_graph_validation import compare_proposals

    return compare_proposals(graph, eager, min_match, max_gap_delta)


def _compare_tensor_evidence(graph, eager, args):
    from test.runtime.draft_graph_validation import compare_payloads

    import torch

    problems = []
    for number, (g, e) in enumerate(zip(graph, eager)):
        first_g = g.get("first_step")
        first_e = e.get("first_step")
        if not isinstance(first_g, dict) or not isinstance(first_e, dict):
            problems.append(f"call {number}: missing full tensor evidence")
            continue
        if first_g.get("path") != "graph":
            continue
        try:
            left = torch.load(
                first_g["artifact"], map_location="cpu", weights_only=True
            )
            right = torch.load(
                first_e["artifact"], map_location="cpu", weights_only=True
            )
            errors = compare_payloads(
                left,
                right,
                logit_atol=args.logit_atol,
                logit_rtol=args.logit_rtol,
                kv_atol=args.kv_atol,
                kv_rtol=args.kv_rtol,
            )
            problems.extend(f"call {number}: {error}" for error in errors)
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            EOFError,
        ) as error:
            problems.append(f"call {number}: unreadable tensor evidence: {error}")
    return problems


def _accept_length(outputs: list[dict]) -> float:
    tokens = sum(len(out["output_ids"]) for out in outputs)
    steps = sum(out["verify_ct"] for out in outputs)
    return tokens / steps if steps else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="HY V4 checkpoint directory to truncate.")
    parser.add_argument("--truncation", default="3:16", help="LAYERS:EXPERTS")
    parser.add_argument("--workdir", default="/tmp/hy_v4_draft_graph")
    parser.add_argument("--steps", type=int, default=3, help="MTP draft steps")
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--enable-mixed-batch",
        action="store_true",
        help="Let the scheduler put prefill chunks and decodes in one batch.",
    )
    parser.add_argument(
        "--min-match",
        type=float,
        default=1.0,
        help="Compatibility option; must be 1.0. No arbitrary errors are allowed.",
    )
    parser.add_argument(
        "--max-gap-delta",
        type=float,
        default=0.05,
        help="Largest per-row step-0 top-2 logit gap difference on actual replays.",
    )
    parser.add_argument("--logit-atol", type=float, default=0.05)
    parser.add_argument("--logit-rtol", type=float, default=0.01)
    parser.add_argument("--kv-atol", type=float, default=0.004)
    parser.add_argument("--kv-rtol", type=float, default=0.02)
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.min_match != 1.0:
        parser.error("--min-match must be 1.0; aggregate error allowances are unsafe")
    if args.steps < 1 or args.prompts < 1 or args.max_new_tokens < 1:
        parser.error("steps, prompts and max-new-tokens must be positive")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (
            args.max_gap_delta,
            args.logit_atol,
            args.logit_rtol,
            args.kv_atol,
            args.kv_rtol,
        )
    ):
        parser.error("all tolerances must be finite and non-negative")
    if args.run:
        return _run(args)
    if not args.source:
        parser.error("--source is required")

    layers, _, experts = args.truncation.partition(":")
    truncation = Truncation(
        num_layers=int(layers),
        num_experts=None if not experts or experts == "all" else int(experts),
    )
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model = build_truncated_checkpoint(
        workdir / f"hy4-{truncation.describe().replace('/', '-')}",
        truncation,
        repo_id=args.source,
        log=lambda message: print(f"[fixture] {message}", flush=True),
    )

    graph = _spawn("on", model, workdir, args)
    eager = _spawn("off", model, workdir, args)
    print()
    for name, run in (("graph", graph), ("eager", eager)):
        trace = run["trace"]
        mixed = sum(0 < call["num_extends"] < call["bs"] for call in trace)
        print(
            f"{name} run: {run['counts']['replays']} step-0 replays, "
            f"{run['counts']['eager_extends']} eager extends, "
            f"{len(trace)} eager-step draft calls ({mixed} mixed), "
            f"accept length {_accept_length(run['outputs']):.3f}"
        )
    print(_output_summary(graph["outputs"], eager["outputs"]))
    problems, summary = _compare_proposals(
        graph["trace"], eager["trace"], args.min_match, args.max_gap_delta
    )
    from test.runtime.draft_graph_validation import validate_trace_counts

    for name, run in (("graph", graph), ("eager", eager)):
        problems.extend(f"{name}: {error}" for error in validate_trace_counts(run))
        if len(run["outputs"]) != args.prompts:
            problems.append(
                f"{name}: output count does not match the submitted prompts"
            )
    problems.extend(_compare_tensor_evidence(graph["trace"], eager["trace"], args))
    if len(graph["outputs"]) != len(eager["outputs"]) or any(
        g["output_ids"] != e["output_ids"]
        for g, e in zip(graph["outputs"], eager["outputs"])
    ):
        problems.append(
            "generated token sequences differ; end-to-end parity is not established"
        )
    if graph["counts"]["replays"] == 0:
        problems.append("the graph run never replayed the draft step-0 graph")
    if eager["counts"]["replays"] != 0 or eager["counts"]["eager_extends"] == 0:
        problems.append("the eager run did not run step 0 eagerly")
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"PASS {len(graph['trace'])} served draft calls: {summary}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
