#!/usr/bin/env python3
"""Summarize a Pier DeepSWE result and enforce infrastructure/score gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _reward_score(data: dict[str, Any]) -> float | None:
    evals = data.get("stats", {}).get("evals", {})
    for eval_stats in evals.values():
        for metric in eval_stats.get("metrics", []):
            value = metric.get("reward")
            if isinstance(value, (int, float)):
                return float(value)

    rewards: list[float] = []
    for trial in data.get("trial_results", []):
        verifier = trial.get("verifier_result") or {}
        value = (verifier.get("rewards") or {}).get("reward")
        if isinstance(value, (int, float)):
            rewards.append(float(value))
    return sum(rewards) / len(rewards) if rewards else None


def build_summary(data: dict[str, Any], minimum_score: float) -> tuple[str, list[str]]:
    stats = data.get("stats", {})
    total = int(data.get("n_total_trials", 0))
    completed = int(stats.get("n_completed_trials", 0))
    errors = int(stats.get("n_errored_trials", 0))
    cancelled = int(stats.get("n_cancelled_trials", 0))
    score = _reward_score(data)

    lines = [
        "### B300 Kimi K3 DeepSWE",
        "",
        f"- Trials completed: {completed}/{total}",
        f"- Trial errors: {errors}",
        f"- Cancelled trials: {cancelled}",
        (
            f"- Binary reward: {score:.4f}"
            if score is not None
            else "- Binary reward: unavailable"
        ),
        f"- Minimum score: {minimum_score:.4f}",
    ]

    exception_counts: dict[str, int] = {}
    for eval_stats in stats.get("evals", {}).values():
        for exception_type, trials in eval_stats.get("exception_stats", {}).items():
            exception_counts[exception_type] = exception_counts.get(
                exception_type, 0
            ) + len(trials)
    if exception_counts:
        lines.extend(["", "#### Exceptions", ""])
        lines.extend(
            f"- `{name}`: {count}" for name, count in sorted(exception_counts.items())
        )

    problems = []
    if total <= 0:
        problems.append("Pier reported no trials")
    if completed != total:
        problems.append(f"only {completed}/{total} trials completed")
    if errors:
        problems.append(f"{errors} trial(s) ended with agent or infrastructure errors")
    if cancelled:
        problems.append(f"{cancelled} trial(s) were cancelled")
    if score is None:
        problems.append("the binary reward metric is missing")
    elif score < minimum_score:
        problems.append(
            f"binary reward {score:.4f} is below minimum {minimum_score:.4f}"
        )

    if problems:
        lines.extend(["", "#### Gate failures", ""])
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.extend(["", "DeepSWE gate passed."])

    return "\n".join(lines) + "\n", problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.minimum_score <= 1.0:
        parser.error("--minimum-score must be between 0 and 1")
    if not args.result.is_file():
        parser.error(f"Pier result does not exist: {args.result}")

    data = json.loads(args.result.read_text(encoding="utf-8"))
    summary, problems = build_summary(data, args.minimum_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(summary, end="")

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with Path(github_summary).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
