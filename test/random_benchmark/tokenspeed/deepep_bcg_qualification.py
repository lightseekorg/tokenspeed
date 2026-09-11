#!/usr/bin/env python3
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

"""Plan paired DeepEP BCG trials and report their real EvalScope output.

This tool does not launch servers. Execute each schedule entry with a fresh
server, in its listed order. Both arms inherit identical server/client arguments;
only the prefill graph flags and client output directory differ. One schedule
covers one workload. Keep raw server logs, memory measurements, and replay
proof under each run's evidence/ directory. These files are hashed, not treated
as measurements whose meaning the tool can establish.

Client TTFT includes queueing and transport: it is never labeled server prefill.
Optional server-prefill samples must be exported from a separately instrumented
run to evidence/server_prefill.csv (latency_ms column) with their measurement
boundary and raw source described in evidence/server_prefill_source.txt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shlex
import statistics
from pathlib import Path

GRAPH_OPTIONS = {
    "--disable-prefill-graph",
    "--prefill-graph-max-tokens",
    "--prefill-graph-capture-sizes",
}
# EvalScope v1.9.1 perf_constants.py; v1.10 adds the Avg prefixes.
SUMMARY_METRICS = {
    "client_ttft_mean_ms": ("Avg TTFT (ms)", "TTFT (ms)"),
    "client_tpot_mean_ms": ("Avg TPOT (ms)", "TPOT (ms)"),
    "output_tokens_per_second": ("Output Throughput (tok/s)",),
    "total_tokens_per_second": ("Total Throughput (tok/s)",),
    "input_tokens_mean": ("Avg Input Tokens",),
    "output_tokens_mean": ("Avg Output Tokens",),
    "cache_hit_percent": ("KV Cache Hit Rate (%)",),
}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def argument_names(argv: list[str]) -> set[str]:
    return {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}


def validate_config(config: dict) -> None:
    for key in ("server_argv", "client_argv"):
        argv = config[key]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(arg, str) and arg for arg in argv)
        ):
            raise ValueError(f"{key} must be a nonempty argv string array")
    server_options = argument_names(config["server_argv"])
    if server_options & (GRAPH_OPTIONS | {"--enforce-eager"}):
        raise ValueError(
            "server_argv must omit prefill graph flags and --enforce-eager"
        )
    client_options = argument_names(config["client_argv"])
    if "--outputs-dir" in client_options:
        raise ValueError(
            "client_argv must omit --outputs-dir; the schedule supplies it"
        )
    if not {"--stream", "--no-timestamp", "--seed"} <= client_options:
        raise ValueError("client_argv requires --stream, --no-timestamp, and --seed")
    if "--no-stream" in client_options:
        raise ValueError("client_argv must use streaming for TTFT")
    if type(config["pairs"]) is not int or config["pairs"] < 8:
        raise ValueError("pairs must be an integer >= 8")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    cap, sizes = config["prefill_graph_max_tokens"], config["capture_sizes"]
    if type(cap) is not int or cap <= 0:
        raise ValueError("prefill_graph_max_tokens must be a positive integer")
    if (
        not isinstance(sizes, list)
        or not sizes
        or any(type(size) is not int or size <= 0 or size > cap for size in sizes)
        or sizes != sorted(set(sizes))
    ):
        raise ValueError("capture_sizes must be sorted unique positive integers <= cap")
    ranks = config["expected_ranks"]
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(type(rank) is not int or rank < 0 for rank in ranks)
        or ranks != sorted(set(ranks))
    ):
        raise ValueError(
            "expected_ranks must list every participating global rank once"
        )
    provenance = config["provenance"]
    for key in ("tokenspeed_revision", "model_revision", "dependencies", "hardware"):
        if (
            not isinstance(provenance.get(key), str)
            or not provenance[key].strip()
            or provenance[key].startswith("REPLACE_")
        ):
            raise ValueError(f"provenance.{key} must describe the actual pinned setup")


def make_schedule(config: dict, root: Path) -> dict:
    """Return a seeded paired schedule without executing any command."""
    validate_config(config)
    rng = random.Random(config["seed"])
    runs = []
    for pair in range(1, config["pairs"] + 1):
        arms = ["eager", "bcg"]
        rng.shuffle(arms)
        for arm in arms:
            directory = f"pair{pair:02d}/{arm}"
            graph_args = (
                ["--disable-prefill-graph"]
                if arm == "eager"
                else [
                    "--prefill-graph-max-tokens",
                    str(config["prefill_graph_max_tokens"]),
                    "--prefill-graph-capture-sizes",
                    *map(str, config["capture_sizes"]),
                ]
            )
            runs.append(
                {
                    "pair": pair,
                    "arm": arm,
                    "directory": directory,
                    "server_argv": config["server_argv"] + graph_args,
                    "client_argv": config["client_argv"]
                    + ["--outputs-dir", str(root / directory / "evalscope")],
                }
            )
    return {
        "schema_version": 1,
        "output_root": str(root),
        "config": config,
        "runs": runs,
    }


def finite_number(value, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{description} must be finite and nonnegative")
    return number


def metric(summary: dict, names: tuple[str, ...]) -> float | None:
    for name in names:
        if summary.get(name) is not None:
            return finite_number(summary[name], name)
    return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_reduction(
    eager: list[float], bcg: list[float], seed: int, bootstrap_samples: int
) -> dict:
    """Mean per-pair relative reduction and percentile-bootstrap 95% interval.

    The unit of resampling is a fresh-server pair, not individual requests.
    Positive reduction is an improvement only for latency metrics.
    """
    if not eager or len(eager) != len(bcg) or any(value <= 0 for value in eager):
        raise ValueError(
            "paired reduction requires equal nonempty pairs and positive baselines"
        )
    changes = [100.0 * (1.0 - b / a) for a, b in zip(eager, bcg, strict=True)]
    rng = random.Random(seed)
    samples = [
        statistics.fmean(rng.choices(changes, k=len(changes)))
        for _ in range(bootstrap_samples)
    ]
    return {
        "pairs": len(changes),
        "eager_mean": statistics.fmean(eager),
        "bcg_mean": statistics.fmean(bcg),
        "mean_paired_reduction_percent": statistics.fmean(changes),
        "paired_reduction_95_percent_interval": [
            percentile(samples, 0.025),
            percentile(samples, 0.975),
        ],
    }


def fingerprint(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(root)), "sha256": digest.hexdigest()}


def read_run(root: Path, run: dict, expected_ranks: list[int]) -> dict:
    directory = root / run["directory"]
    summaries = sorted((directory / "evalscope").rglob("benchmark_summary.json"))
    if len(summaries) != 1:
        raise ValueError(f"{directory}: expected one summary, found {len(summaries)}")
    summary_path = summaries[0]
    summary = read_json(summary_path)
    success = finite_number(summary.get("Success Requests"), "Success Requests")
    failed = finite_number(summary.get("Failed Requests"), "Failed Requests")
    if failed != 0 or success <= 0:
        raise ValueError(f"{summary_path}: failed requests or no successful requests")
    values = {key: metric(summary, names) for key, names in SUMMARY_METRICS.items()}
    if values["client_ttft_mean_ms"] is None or values["client_ttft_mean_ms"] <= 0:
        raise ValueError(f"{summary_path}: missing or nonpositive TTFT")
    sources = [fingerprint(summary_path, root)]
    args_path = summary_path.with_name("benchmark_args.json")
    client_args = read_json(args_path)
    sources.append(fingerprint(args_path, root))
    # EvalScope's serialized output location is expected to differ across arms.
    client_args.pop("outputs_dir", None)
    if client_args.get("stream") is not True:
        raise ValueError(f"{args_path}: streaming must be enabled for TTFT")
    percentiles_path = summary_path.with_name("benchmark_percentile.json")
    for level in (50, 95, 99):
        values[f"client_ttft_p{level}_ms"] = None
    if percentiles_path.exists():
        percentiles = read_json(percentiles_path)
        if not isinstance(percentiles, list):
            raise ValueError(
                f"{percentiles_path}: expected EvalScope percentile row list"
            )
        sources.append(fingerprint(percentiles_path, root))
        for row in percentiles:
            for level in (50, 95, 99):
                if row.get("Percentiles") == f"{level}%":
                    values[f"client_ttft_p{level}_ms"] = metric(row, ("TTFT (ms)",))
    evidence_dir = directory / "evidence"
    server_log = (evidence_dir / "server.log").read_text()
    replay_ranks = sorted(
        {
            int(match.group(1))
            for match in re.finditer(
                r"DeepEP prefill BCG replay: rank=(\d+) bucket=\d+ real_tokens=\d+",
                server_log,
            )
        }
    )
    if run["arm"] == "bcg" and replay_ranks != expected_ranks:
        raise ValueError(
            f"{directory}: expected serving BCG replay on ranks {expected_ranks}, "
            f"observed {replay_ranks}"
        )
    if run["arm"] == "eager" and replay_ranks:
        raise ValueError(f"{directory}: eager baseline unexpectedly replayed BCG")
    sources.extend(
        fingerprint(path, root)
        for path in sorted(evidence_dir.rglob("*"))
        if path.is_file()
    )
    for level in ("mean", "p50", "p95", "p99"):
        values[f"server_prefill_{level}_ms"] = None
    prefill_path = evidence_dir / "server_prefill.csv"
    if prefill_path.exists():
        description = (evidence_dir / "server_prefill_source.txt").read_text().strip()
        if not description:
            raise ValueError(
                "server_prefill_source.txt must describe boundary and raw source"
            )
        with prefill_path.open(newline="") as stream:
            samples = [
                finite_number(row["latency_ms"], "server prefill sample")
                for row in csv.DictReader(stream)
            ]
        if not samples:
            raise ValueError(f"{prefill_path}: no samples")
        values["server_prefill_mean_ms"] = statistics.fmean(samples)
        for level in (50, 95, 99):
            values[f"server_prefill_p{level}_ms"] = percentile(samples, level / 100)
    return {
        "pair": run["pair"],
        "arm": run["arm"],
        "successful_requests": success,
        "serving_replay_ranks": replay_ranks,
        "client_args": client_args,
        "metrics": values,
        "sources": sources,
    }


def build_report(root: Path, schedule: dict) -> dict:
    """Read complete paired runs; reject missing, failed, or mismatched trials."""
    # Artifacts may be copied off the GPU host before reporting. Validate the
    # scheduled command paths against their original root, then read locally.
    if schedule != make_schedule(schedule["config"], Path(schedule["output_root"])):
        raise ValueError("schedule was modified; regenerate it from its config")
    runs = [
        read_run(root, run, schedule["config"]["expected_ranks"])
        for run in schedule["runs"]
    ]
    by_pair = {}
    for run in runs:
        by_pair.setdefault(run["pair"], {})[run["arm"]] = run
    for pair, arms in by_pair.items():
        if arms["eager"]["client_args"] != arms["bcg"]["client_args"]:
            raise ValueError(f"pair {pair}: EvalScope arguments differ between arms")
        if arms["eager"]["successful_requests"] != arms["bcg"]["successful_requests"]:
            raise ValueError(f"pair {pair}: successful request counts differ")
    comparisons = {}
    for name in runs[0]["metrics"]:
        eager = [arms["eager"]["metrics"][name] for arms in by_pair.values()]
        bcg = [arms["bcg"]["metrics"][name] for arms in by_pair.values()]
        comparisons[name] = (
            None
            if any(value is None or value <= 0 for value in eager)
            or any(value is None for value in bcg)
            else paired_reduction(eager, bcg, schedule["config"]["seed"], 10000)
        )
    return {
        "schema_version": 1,
        "provenance": schedule["config"]["provenance"],
        "schedule": fingerprint(root / "schedule.json", root),
        "comparisons": comparisons,
        "runs": runs,
        "limitations": [
            "Commands describe the intended setup; verify actual server flags/revision in raw logs.",
            "Client TTFT includes queueing and transport; server prefill requires separate samples.",
            "Total token throughput is not uncached input throughput.",
            "Missing metrics remain null; zero TPOT with one output token has no reduction estimate.",
            "Serving replay rank markers establish engagement; hashes do not prove memory bounds or correctness.",
            "Bootstrap intervals resample independent server-restart pairs; no automatic promotion verdict.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Write a randomized schedule; execute it manually"
    )
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Summarize completed paired trials")
    report.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            root = args.output.resolve()
            schedule = make_schedule(read_json(args.config), root)
            root.mkdir(parents=True, exist_ok=False)
            write_json(root / "schedule.json", schedule)
            lines = [
                "# Fresh server for each entry; wait for health before running its client."
            ]
            for run in schedule["runs"]:
                directory = root / run["directory"]
                (directory / "evidence").mkdir(parents=True)
                lines.extend(
                    [
                        f"\n# Pair {run['pair']} / {run['arm']}; record server stdout/stderr in {directory / 'evidence/server.log'}",
                        shlex.join(run["server_argv"]),
                        shlex.join(run["client_argv"]),
                    ]
                )
            (root / "commands.txt").write_text("\n".join(lines) + "\n")
            print(
                f"Wrote {root / 'schedule.json'} and commands.txt; no commands executed."
            )
        else:
            root = args.directory.resolve()
            result = build_report(root, read_json(root / "schedule.json"))
            write_json(root / "report.json", result)
            print("Client TTFT reduction (%), positive means lower latency:")
            print(json.dumps(result["comparisons"]["client_ttft_mean_ms"], indent=2))
            print(
                f"Full metrics, missing values, provenance and evidence: {root / 'report.json'}"
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
