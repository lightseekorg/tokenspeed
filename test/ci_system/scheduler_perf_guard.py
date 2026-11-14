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

"""Guard the C++ scheduler's CPU cost against regressions.

Compares a base and a candidate build of the scheduler's CPU benchmarks.

``instructions`` counts the instructions the scheduler's API calls execute
under Callgrind. The count is deterministic for a given binary, so any change
beyond a small budget is a real change in work: this suite blocks. It also
checks that a full-cache admission does not get costlier as the cache grows,
and that a decode round's admission does not get costlier as contexts grow.

``timing`` measures thread CPU time of alternating base and candidate
processes on the full-size workloads. It also sees memory effects that
instruction counts miss (cache misses, prefetching), but shared CI machines
are noisy, so it only reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FULL_CACHE = "tokenspeed_scheduler_full_cache_benchmark"
DECODE_ROUNDS = "tokenspeed_scheduler_decode_rounds_benchmark"

# The scheduler API calls a benchmark process makes inside its measured region.
API_FUNCTIONS = (
    "tokenspeed::Scheduler::SubmitRequests(",
    "tokenspeed::Scheduler::NextExecutionPlan(",
    "tokenspeed::Scheduler::Advance(",
)
ADMIT_FUNCTION = "tokenspeed::CacheCoordinator::Admit("

# Instruction counts are exact, so the budget only absorbs incidental code
# changes (a check, a branch) rather than measurement noise.
MAX_INSTRUCTION_REGRESSION = 0.02


@dataclass(frozen=True)
class InstructionCase:
    """A benchmark run whose per-sample instructions are compared.

    The process runs ``low`` and then ``high`` samples; the per-sample count
    is the difference divided by ``high - low``, which cancels setup work
    (filling the cache, prefilling the prompts) that the scheduler API calls
    also perform.
    """

    name: str
    binary: str
    args: tuple[str, ...]
    low: int
    high: int

    def command(self, bin_dir: Path, samples: int) -> list[str]:
        if self.binary == FULL_CACHE:
            # POOL PAGE REQUEST BATCH HIT, then WARMUPS REPEATS MODE.
            return [
                str(bin_dir / self.binary),
                *self.args,
                "0",
                str(samples),
                "lifecycle",
            ]
        # PAGE CONTEXT REQUESTS DECODE_TOKENS, then WARMUP_ROUNDS ROUNDS.
        return [str(bin_dir / self.binary), *self.args, "0", str(samples)]


# Small enough for Callgrind, large enough that per-page and per-block work
# dominates. The scaling pairs differ only in cache size or context length.
INSTRUCTION_CASES = (
    InstructionCase(
        "full_cache_256k_hit0", FULL_CACHE, ("262144", "64", "32768", "16", "0"), 1, 3
    ),
    InstructionCase(
        "full_cache_256k_hit90", FULL_CACHE, ("262144", "64", "32768", "16", "90"), 1, 3
    ),
    InstructionCase(
        "full_cache_1m_hit0", FULL_CACHE, ("1048576", "64", "32768", "16", "0"), 1, 3
    ),
    InstructionCase("decode_8k", DECODE_ROUNDS, ("16", "8192", "64", "8"), 10, 40),
    InstructionCase("decode_64k", DECODE_ROUNDS, ("16", "65536", "64", "8"), 10, 40),
)


@dataclass(frozen=True)
class ScalingCheck:
    """``larger`` may cost at most ``max_ratio`` times ``smaller`` per sample."""

    name: str
    metric: str
    smaller: str
    larger: str
    max_ratio: float
    reason: str


SCALING_CHECKS = (
    ScalingCheck(
        "admission_vs_cache_size",
        "api",
        "full_cache_256k_hit0",
        "full_cache_1m_hit0",
        1.10,
        "a full-cache admission evicts the same pages whatever the cache size",
    ),
    ScalingCheck(
        "decode_admit_vs_context",
        "admit",
        "decode_8k",
        "decode_64k",
        1.15,
        "a decode round's admission appends and publishes the same blocks whatever the context length",
    ),
)


@dataclass(frozen=True)
class TimingCase:
    name: str
    binary: str
    args: tuple[str, ...]
    metrics: tuple[str, ...]


# The full-size workloads: a full 200 GiB-equivalent prefix cache admitting
# 16 x 128K prompts, and 256 x 128K-token requests decoding with a verify
# width of 6.
TIMING_CASES = (
    TimingCase(
        "full_cache_hit0",
        FULL_CACHE,
        ("3770160", "64", "131072", "16", "0", "3", "12", "lifecycle"),
        ("first_plan", "publish_plan", "finish"),
    ),
    TimingCase(
        "full_cache_hit90",
        FULL_CACHE,
        ("3770160", "64", "131072", "16", "90", "3", "12", "lifecycle"),
        ("first_plan", "publish_plan", "finish"),
    ),
    TimingCase(
        "full_cache_events_hit0",
        FULL_CACHE,
        ("3770160", "64", "131072", "16", "0", "3", "12", "events"),
        ("first_plan", "publish_plan", "events"),
    ),
    TimingCase(
        "decode_256x128k",
        DECODE_ROUNDS,
        ("64", "131072", "256", "6", "10", "40"),
        ("plan", "advance"),
    ),
)

# Timing budgets: a slowdown must exceed both to count as a regression.
MAX_TIMING_REGRESSION = 0.05
MIN_TIMING_REGRESSION_MS = 0.1


class GuardError(RuntimeError):
    """A benchmark could not be run or its output could not be read."""


def parse_callgrind_totals(text: str) -> int:
    """The total cost a Callgrind output file reports."""
    for line in text.splitlines():
        if line.startswith("totals:") or line.startswith("summary:"):
            return int(line.split()[1])
    raise GuardError("Callgrind output has no totals line")


# "  1,234 (12.34%)  file:function [object]", as callgrind_annotate lists them.
_ANNOTATE_LINE = re.compile(r"^\s*([\d,]+)\s+\([^)]*\)\s+(\S.*)$")


def parse_inclusive_cost(annotate_output: str, function_prefix: str) -> int:
    """The inclusive cost callgrind_annotate lists for a function.

    Only names that start with ``function_prefix`` count: a lambda or template
    instantiated inside the function also mentions its name.
    """
    total = 0
    found = False
    for line in annotate_output.splitlines():
        match = _ANNOTATE_LINE.match(line)
        if not match:
            continue
        _, _, function = match.group(2).partition(":")
        if function.startswith(function_prefix):
            total += int(match.group(1).replace(",", ""))
            found = True
    if not found:
        raise GuardError(f"callgrind_annotate lists no {function_prefix}")
    return total


def per_sample(low_count: int, high_count: int, low: int, high: int) -> float:
    if high <= low:
        raise GuardError("the larger run must have more samples")
    return (high_count - low_count) / (high - low)


def _run(command: Sequence[str], log: Path) -> str:
    completed = subprocess.run(list(command), capture_output=True, text=True)
    log.write_text(f"$ {' '.join(command)}\n{completed.stderr}", encoding="utf-8")
    if completed.returncode != 0:
        raise GuardError(f"{command[0]} exited {completed.returncode}; see {log}")
    return completed.stdout


def measure_instructions(
    case: InstructionCase, bin_dir: Path, valgrind: str, annotate: str, work: Path
) -> dict[str, float]:
    counts: dict[int, dict[str, int]] = {}
    for samples in (case.low, case.high):
        out = work / f"{case.name}-{samples}.callgrind"
        command = [
            valgrind,
            "--tool=callgrind",
            "--collect-atstart=no",
            *[f"--toggle-collect={function}*" for function in API_FUNCTIONS],
            f"--callgrind-out-file={out}",
            *case.command(bin_dir, samples),
        ]
        _run(command, work / f"{case.name}-{samples}.log")
        annotated = _run(
            [annotate, "--inclusive=yes", "--threshold=100", str(out)],
            work / f"{case.name}-{samples}.annotate.log",
        )
        counts[samples] = {
            "api": parse_callgrind_totals(
                out.read_text(encoding="utf-8", errors="replace")
            ),
            "admit": parse_inclusive_cost(annotated, ADMIT_FUNCTION),
        }
    return {
        metric: per_sample(
            counts[case.low][metric], counts[case.high][metric], case.low, case.high
        )
        for metric in ("api", "admit")
    }


def classify_instructions(base: float, candidate: float) -> tuple[str, float]:
    change = candidate / base - 1 if base else float("inf")
    if change > MAX_INSTRUCTION_REGRESSION:
        return "regression", change
    if change < -MAX_INSTRUCTION_REGRESSION:
        return "improvement", change
    return "within_budget", change


def evaluate_scaling(
    results: Mapping[str, Mapping[str, float]], check: ScalingCheck
) -> float:
    return results[check.larger][check.metric] / results[check.smaller][check.metric]


def instruction_report(
    base: Mapping[str, Mapping[str, float]] | None,
    candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    comparisons = []
    for case in INSTRUCTION_CASES:
        for metric in ("api", "admit"):
            entry: dict[str, Any] = {
                "case": case.name,
                "metric": metric,
                "candidate": candidate[case.name][metric],
            }
            if base is None or case.name not in base:
                entry["classification"] = "bootstrap"
            else:
                entry["base"] = base[case.name][metric]
                entry["classification"], entry["change"] = classify_instructions(
                    entry["base"], entry["candidate"]
                )
            comparisons.append(entry)
    scaling = []
    for check in SCALING_CHECKS:
        ratio = evaluate_scaling(candidate, check)
        base_ratio = (
            evaluate_scaling(base, check)
            if base is not None and check.smaller in base and check.larger in base
            else None
        )
        if ratio <= check.max_ratio:
            status = "passed"
        elif base_ratio is not None and base_ratio > check.max_ratio:
            # Not this change's doing: report it without blocking.
            status = "preexisting"
        else:
            status = "regression"
        scaling.append(
            {
                "check": check.name,
                "ratio": ratio,
                "base_ratio": base_ratio,
                "max_ratio": check.max_ratio,
                "status": status,
                "reason": check.reason,
            }
        )
    regressed = any(
        item["classification"] == "regression" for item in comparisons
    ) or any(item["status"] == "regression" for item in scaling)
    bootstrap = any(item["classification"] == "bootstrap" for item in comparisons)
    status = "regression" if regressed else ("bootstrap" if bootstrap else "passed")
    return {
        "suite": "instructions",
        "status": status,
        "comparisons": comparisons,
        "scaling": scaling,
    }


def render_instructions(report: Mapping[str, Any]) -> str:
    lines = [
        "## Scheduler instruction counts",
        "",
        f"Status: **{report['status']}**. Instructions per sample in the scheduler's API calls (Callgrind); "
        f"a change above {MAX_INSTRUCTION_REGRESSION:.0%} blocks.",
        "",
        "| Case | Metric | Base | Candidate | Change | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["comparisons"]:
        base = f"{item['base']:,.0f}" if "base" in item else "-"
        change = f"{item['change']:+.2%}" if "change" in item else "-"
        lines.append(
            f"| {item['case']} | {item['metric']} | {base} | {item['candidate']:,.0f} | {change} | "
            f"{item['classification']} |"
        )
    lines += [
        "",
        "| Scaling check | Ratio | Base ratio | Limit | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["scaling"]:
        base_ratio = (
            f"{item['base_ratio']:.3f}" if item["base_ratio"] is not None else "-"
        )
        lines.append(
            f"| {item['check']} ({item['reason']}) | {item['ratio']:.3f} | {base_ratio} | "
            f"{item['max_ratio']:.2f} | {item['status']} |"
        )
    return "\n".join(lines) + "\n"


def _available(bin_dir: Path | None, binary: str) -> bool:
    return bin_dir is not None and (bin_dir / binary).is_file()


def run_instructions(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = Path(args.candidate)
    base_dir = Path(args.base) if args.base else None
    for case in INSTRUCTION_CASES:
        if not _available(candidate_dir, case.binary):
            raise GuardError(f"candidate is missing {case.binary}")
    results: dict[str, dict[str, dict[str, float]]] = {"base": {}, "candidate": {}}
    for side, bin_dir in (("base", base_dir), ("candidate", candidate_dir)):
        work = output / side
        work.mkdir(exist_ok=True)
        for case in INSTRUCTION_CASES:
            # A base revision that predates a benchmark has no number for it.
            if _available(bin_dir, case.binary):
                results[side][case.name] = measure_instructions(
                    case, bin_dir, args.valgrind, args.callgrind_annotate, work
                )
    report = instruction_report(results["base"] or None, results["candidate"])
    (output / "instructions.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = render_instructions(report)
    (output / "instructions.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 1 if report["status"] == "regression" else 0


def round_change(
    base_samples: Sequence[float], candidate_samples: Sequence[float]
) -> tuple[float, float]:
    """Relative and absolute change of one round's median."""
    base = statistics.median(base_samples)
    candidate = statistics.median(candidate_samples)
    return candidate / base - 1, candidate - base


def classify_timing(rounds: Sequence[tuple[float, float]]) -> tuple[str, float, float]:
    """Classify by the round with the median relative change.

    Rounds pair fresh base and candidate processes, so a one-off process-level
    shift moves only its own round.
    """
    relative, absolute = sorted(rounds)[len(rounds) // 2]
    labels = set()
    for change, delta in rounds:
        if change > MAX_TIMING_REGRESSION and delta > MIN_TIMING_REGRESSION_MS:
            labels.add("regression")
        elif change < -MAX_TIMING_REGRESSION and delta < -MIN_TIMING_REGRESSION_MS:
            labels.add("improvement")
    if {"regression", "improvement"} <= labels:
        return "inconclusive", relative, absolute
    if relative > MAX_TIMING_REGRESSION and absolute > MIN_TIMING_REGRESSION_MS:
        return "regression", relative, absolute
    if relative < -MAX_TIMING_REGRESSION and absolute < -MIN_TIMING_REGRESSION_MS:
        return "improvement", relative, absolute
    return "within_budget", relative, absolute


def _cpu_samples(result: Mapping[str, Any], metric: str) -> list[float]:
    try:
        return [float(value) for value in result[metric]["thread_cpu_ms"]["samples"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise GuardError(f"benchmark output has no {metric} samples") from exc


def run_timing(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = Path(args.candidate)
    base_dir = Path(args.base) if args.base else None
    for case in TIMING_CASES:
        if not _available(candidate_dir, case.binary):
            raise GuardError(f"candidate is missing {case.binary}")
    sides = [("base", base_dir), ("candidate", candidate_dir)]
    pin = ["taskset", "-c", args.cpu] if args.cpu else []
    samples: dict[tuple[str, str, str], list[list[float]]] = {}
    for round_index in range(args.rounds):
        # Alternate which revision runs first, so a drift within a round
        # favors neither.
        order = sides if round_index % 2 == 0 else list(reversed(sides))
        for case in TIMING_CASES:
            for side, bin_dir in order:
                if not _available(bin_dir, case.binary):
                    continue
                stdout = _run(
                    [*pin, str(bin_dir / case.binary), *case.args],
                    output / f"{side}-{case.name}-r{round_index}.log",
                )
                result = json.loads(stdout)
                for metric in case.metrics:
                    samples.setdefault((side, case.name, metric), []).append(
                        _cpu_samples(result, metric)
                    )
    comparisons = []
    for case in TIMING_CASES:
        for metric in case.metrics:
            candidate = samples[("candidate", case.name, metric)]
            entry: dict[str, Any] = {
                "case": case.name,
                "metric": metric,
                "candidate_ms": statistics.median(
                    [x for run in candidate for x in run]
                ),
            }
            if ("base", case.name, metric) in samples:
                base = samples[("base", case.name, metric)]
                entry["base_ms"] = statistics.median([x for run in base for x in run])
                rounds = [round_change(b, c) for b, c in zip(base, candidate)]
                entry["round_changes"] = [change for change, _ in rounds]
                entry["classification"], entry["change"], entry["delta_ms"] = (
                    classify_timing(rounds)
                )
            else:
                entry["classification"] = "bootstrap"
            comparisons.append(entry)
    report = {"suite": "timing", "rounds": args.rounds, "comparisons": comparisons}
    (output / "timing.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = render_timing(report)
    (output / "timing.md").write_text(summary, encoding="utf-8")
    print(summary)
    for item in comparisons:
        if item["classification"] == "regression":
            print(
                f"::warning title=Scheduler timing::{item['case']} {item['metric']} "
                f"{item['change']:+.1%} ({item['delta_ms']:+.3f} ms)"
            )
    return 0


def render_timing(report: Mapping[str, Any]) -> str:
    lines = [
        "## Scheduler CPU time (report only)",
        "",
        f"Thread CPU P50 in ms per call, {report['rounds']} alternating rounds; the round with the median change "
        f"decides. A regression exceeds both {MAX_TIMING_REGRESSION:.0%} and {MIN_TIMING_REGRESSION_MS} ms. "
        "Shared CI machines are noisy: confirm a reported regression locally.",
        "",
        "| Case | Metric | Base | Candidate | Change | Rounds | Result |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["comparisons"]:
        base = f"{item['base_ms']:.3f}" if "base_ms" in item else "-"
        change = f"{item['change']:+.1%}" if "change" in item else "-"
        rounds = " ".join(f"{value:+.1%}" for value in item.get("round_changes", []))
        lines.append(
            f"| {item['case']} | {item['metric']} | {base} | {item['candidate_ms']:.3f} | {change} | {rounds} | "
            f"{item['classification']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="suite", required=True)
    for name in ("instructions", "timing"):
        command = sub.add_parser(name)
        command.add_argument(
            "--candidate", required=True, help="directory with the candidate benchmarks"
        )
        command.add_argument(
            "--base",
            required=True,
            help="directory with the base benchmarks ('' if none)",
        )
        command.add_argument("--output", required=True)
    sub.choices["instructions"].add_argument("--valgrind", required=True)
    sub.choices["instructions"].add_argument("--callgrind-annotate", required=True)
    sub.choices["timing"].add_argument("--rounds", type=int, required=True)
    sub.choices["timing"].add_argument(
        "--cpu", required=True, help="CPU to pin benchmarks to ('' for none)"
    )
    args = parser.parse_args(argv)
    try:
        return (
            run_instructions(args) if args.suite == "instructions" else run_timing(args)
        )
    except GuardError as exc:
        print(f"::error title=Scheduler perf guard::{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
