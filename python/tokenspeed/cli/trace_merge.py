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

"""Merge Proton chrome traces into one or more VizTracer timelines.

Both profilers can run in the same scheduler process (``/start_profile``
with ``{"activities": ["VIZTRACER", "PROTON"]}``) but write separate files
with timestamps relative to their own start. Each file also records the
absolute wall-clock time its ``ts=0`` refers to — ``viztracer_metadata.
baseTimeNanoseconds`` in VizTracer reports, top-level ``baseTimeNanoseconds``
in Proton chrome traces — so the two timelines can be shifted onto a shared
axis after the fact.

Usage (one file pair per scheduler rank):

    tokenspeed merge-traces <run>-TP0.viztracer.json \
        <run>-TP0.proton.chrome_trace -o <run>-TP0-merged.json

Usage (one final report for multiple scheduler ranks):

    tokenspeed merge-traces --all-ranks \
        --rank 0 <run>-TP0.viztracer.json <run>-TP0.proton.chrome_trace \
        --rank 1 <run>-TP1.viztracer.json <run>-TP1.proton.chrome_trace \
        -o <run>-all-ranks-merged.json

In all-ranks mode, the TP rank is encoded into each numeric Proton flow ID.
This keeps CPU-to-GPU flow arrows within a rank while preventing matching
launch IDs on different ranks from cross-linking.

Open the merged report in vizviewer or https://ui.perfetto.dev — Python
frames and Proton's kernel/scope lanes share one time axis. Alignment
accuracy is microsecond-to-millisecond grade (the profilers reconcile GPU
and CPU clocks independently); use it to correlate host activity with GPU
gaps, not for sub-microsecond attribution.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

__all__ = ["merge_all_ranks", "merge_proton_viztracer", "main"]

# Proton labels its single synthetic process (pid 0) "Trace"; rename it so
# the lanes are recognizable next to the VizTracer process in a merged view.
_PROTON_PROCESS_NAME = "Proton"
_PROTON_PID_BASE = 10_000
_FLOW_PHASES = frozenset({"s", "t", "f"})
_FLOW_ID_RANK_SHIFT = 32
_FLOW_ID_LOCAL_MASK = (1 << _FLOW_ID_RANK_SHIFT) - 1
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1

TracePair = tuple[int, str | Path, str | Path]


def _load_json(path: str | Path, kind: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{kind} file {path} is not valid JSON: {exc}") from exc


def _viztracer_base_time_ns(
    viztracer_json: dict[str, Any], viztracer_path: str | Path
) -> int | float:
    base_time_ns = viztracer_json.get("viztracer_metadata", {}).get(
        "baseTimeNanoseconds"
    )
    if base_time_ns is None:
        raise ValueError(
            f"{viztracer_path} has no viztracer_metadata.baseTimeNanoseconds; "
            "re-record with a viztracer version that stores the report's "
            "absolute time base."
        )
    return base_time_ns


def _proton_base_time_ns(
    proton_json: dict[str, Any], proton_path: str | Path
) -> int | float:
    base_time_ns = proton_json.get("baseTimeNanoseconds")
    if base_time_ns is None:
        raise ValueError(
            f"{proton_path} has no baseTimeNanoseconds; Upgrade "
            "tokenspeed-proton and re-record."
        )
    return base_time_ns


def _namespace_proton_flow_id(event: dict[str, Any], rank: int) -> None:
    """Make a numeric Proton flow ID globally unique for an all-ranks trace."""
    if event.get("ph") not in _FLOW_PHASES:
        return

    flow_id = event.get("id")
    if flow_id is None:
        return
    if (
        not isinstance(flow_id, int)
        or isinstance(flow_id, bool)
        or not 0 <= flow_id <= _FLOW_ID_LOCAL_MASK
    ):
        raise ValueError(
            f"Proton flow ID must be an unsigned 32-bit integer: {flow_id!r}"
        )

    namespaced_id = (rank << _FLOW_ID_RANK_SHIFT) | flow_id
    if namespaced_id > _MAX_SAFE_JSON_INTEGER:
        raise ValueError(
            f"TP rank {rank} is too large to encode in a JSON-safe flow ID"
        )
    event["id"] = namespaced_id


def _prepare_proton_events(
    proton_json: dict[str, Any],
    proton_path: str | Path,
    viztracer_base_ns: int | float,
    *,
    rank: int | None = None,
) -> list[dict[str, Any]]:
    """Align Proton events to a VizTracer base and optionally scope a rank."""
    proton_base_ns = _proton_base_time_ns(proton_json, proton_path)
    offset_us = (proton_base_ns - viztracer_base_ns) / 1000.0
    process_name = (
        _PROTON_PROCESS_NAME if rank is None else f"{_PROTON_PROCESS_NAME} TP{rank}"
    )

    proton_events = proton_json.get("traceEvents", [])
    for event in proton_events:
        if rank is not None:
            _namespace_proton_flow_id(event, rank)
            if event.get("pid") == 0:
                event["pid"] = _PROTON_PID_BASE + rank

        if event.get("ph") == "M":
            # Metadata events carry no meaningful timestamp; drop any so
            # they cannot clobber the report's process/thread names.
            event.pop("ts", None)
            if (
                event.get("name") == "process_name"
                and event.get("args", {}).get("name") == "Trace"
            ):
                event["args"]["name"] = process_name
        elif "ts" in event:
            event["ts"] += offset_us

    return proton_events


def _write_json(output_path: str | Path, trace: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace, f)


def merge_proton_viztracer(
    viztracer_path: str | Path,
    proton_path: str | Path,
    output_path: str | Path,
) -> int:
    """Merge a Proton chrome trace into a VizTracer report.

    Shifts every Proton event by the difference between the two files'
    ``baseTimeNanoseconds`` anchors and appends them to the VizTracer
    report's ``traceEvents``, producing one chrome-trace JSON on a shared
    time axis.

    Args:
        viztracer_path: VizTracer report (``.json``) saved with a viztracer
            version that records ``viztracer_metadata.baseTimeNanoseconds``.
        proton_path: Proton trace (``data="trace"`` session dumped as
            ``chrome_trace``) with a top-level ``baseTimeNanoseconds``.
        output_path: Where to write the merged chrome-trace JSON.

    Returns:
        The number of Proton events merged into the report.

    Raises:
        ValueError: If either input lacks its absolute time anchor or is
            not valid JSON.
    """
    viztracer_json = _load_json(viztracer_path, "VizTracer report")
    proton_json = _load_json(proton_path, "Proton trace")
    viztracer_base_ns = _viztracer_base_time_ns(viztracer_json, viztracer_path)
    proton_events = _prepare_proton_events(proton_json, proton_path, viztracer_base_ns)

    viztracer_json.setdefault("traceEvents", []).extend(proton_events)
    _write_json(output_path, viztracer_json)
    return len(proton_events)


def merge_all_ranks(rank_traces: Iterable[TracePair], output_path: str | Path) -> int:
    """Merge and align Proton/VizTracer trace pairs from several TP ranks.

    Every input pair is first aligned using its own VizTracer time base. All
    resulting events are then shifted to the earliest VizTracer time base, so
    the final report has one shared absolute timeline. Proton always uses a
    synthetic pid 0 and launch IDs that restart in every scheduler process;
    the all-ranks output gives it a pid of ``10000 + rank`` and encodes its TP
    rank in each numeric flow ID. The latter preserves CPU-to-GPU arrows
    within a rank without allowing equal launch IDs from two ranks to
    cross-link.

    Args:
        rank_traces: ``(rank, viztracer_path, proton_path)`` triples. Ranks
            must be unique non-negative integers.
        output_path: Destination for the consolidated Chrome trace JSON.

    Returns:
        The total number of Proton events added to the final trace.

    Raises:
        ValueError: If no inputs are supplied, a rank is negative or
            duplicated, or an input trace lacks an absolute time anchor.
    """
    pairs = sorted(rank_traces, key=lambda pair: pair[0])
    if not pairs:
        raise ValueError("at least one --rank trace pair is required")

    ranks = [rank for rank, _, _ in pairs]
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"ranks must be non-negative: {ranks}")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks are not allowed: {ranks}")

    loaded_pairs: list[
        tuple[int, dict[str, Any], int | float, dict[str, Any], str | Path]
    ] = []
    for rank, viztracer_path, proton_path in pairs:
        viztracer_json = _load_json(viztracer_path, "VizTracer report")
        viztracer_base_ns = _viztracer_base_time_ns(viztracer_json, viztracer_path)
        proton_json = _load_json(proton_path, "Proton trace")
        loaded_pairs.append(
            (rank, viztracer_json, viztracer_base_ns, proton_json, proton_path)
        )

    global_base_ns = min(pair[2] for pair in loaded_pairs)
    merged_trace = copy.deepcopy(loaded_pairs[0][1])
    merged_trace["traceEvents"] = []
    merged_trace.setdefault("viztracer_metadata", {})[
        "baseTimeNanoseconds"
    ] = global_base_ns

    proton_event_count = 0
    for (
        rank,
        viztracer_json,
        viztracer_base_ns,
        proton_json,
        proton_path,
    ) in loaded_pairs:
        rank_offset_us = (viztracer_base_ns - global_base_ns) / 1000.0
        proton_events = _prepare_proton_events(
            proton_json, proton_path, viztracer_base_ns, rank=rank
        )
        rank_events = [*viztracer_json.get("traceEvents", []), *proton_events]
        for event in rank_events:
            if event.get("ph") != "M" and "ts" in event:
                event["ts"] += rank_offset_us
        merged_trace["traceEvents"].extend(rank_events)
        proton_event_count += len(proton_events)

    _write_json(output_path, merged_trace)
    return proton_event_count


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``tokenspeed merge-traces``.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        prog="tokenspeed merge-traces",
        description="Merge one or more Proton/VizTracer trace pairs onto a "
        "shared timeline.",
    )
    parser.add_argument(
        "viztracer_json",
        nargs="?",
        help="VizTracer report (.json; single-rank mode)",
    )
    parser.add_argument(
        "proton_trace",
        nargs="?",
        help="Proton chrome trace (.chrome_trace; single-rank mode)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output path (required with --all-ranks; default: "
            "<viztracer stem>-merged.json)"
        ),
    )
    parser.add_argument(
        "--all-ranks",
        action="store_true",
        help="Merge several rank pairs into one Chrome trace.",
    )
    parser.add_argument(
        "--rank",
        dest="rank_traces",
        action="append",
        nargs=3,
        metavar=("RANK", "VIZTRACER_JSON", "PROTON_TRACE"),
        help="TP rank and its VizTracer/Proton inputs; repeat with --all-ranks.",
    )
    args = parser.parse_args(argv)

    if args.all_ranks:
        if args.viztracer_json is not None or args.proton_trace is not None:
            parser.error("--all-ranks accepts inputs only through --rank")
        if not args.rank_traces:
            parser.error("--all-ranks requires at least one --rank triple")
        if args.output is None:
            parser.error("--all-ranks requires --output")

        rank_traces: list[TracePair] = []
        for rank, viztracer_path, proton_path in args.rank_traces:
            try:
                rank_traces.append((int(rank), viztracer_path, proton_path))
            except ValueError:
                parser.error(f"--rank expects an integer rank, got {rank!r}")

        merged = merge_all_ranks(rank_traces, args.output)
        print(
            f"Merged {merged} Proton events across {len(rank_traces)} ranks into "
            f"{args.output}"
        )
        return

    if args.rank_traces:
        parser.error("--rank requires --all-ranks")
    if args.viztracer_json is None or args.proton_trace is None:
        parser.error("single-rank mode requires VIZTRACER_JSON and PROTON_TRACE")

    output = args.output
    if output is None:
        viztracer_path = Path(args.viztracer_json)
        output = viztracer_path.with_name(f"{viztracer_path.stem}-merged.json")

    merged = merge_proton_viztracer(args.viztracer_json, args.proton_trace, output)
    print(f"Merged {merged} Proton events into {output}")


if __name__ == "__main__":
    main()
