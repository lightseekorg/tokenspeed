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

"""Merge a Proton chrome trace into a VizTracer report on one timeline.

Both profilers can run in the same scheduler process (``/start_profile``
with ``{"activities": ["VIZTRACER", "PROTON"]}``) but write separate files
with timestamps relative to their own start. Each file records enough clock
metadata to place its events on the VizTracer monotonic time axis —
``viztracer_metadata.baseTimeNanoseconds`` in VizTracer reports and
top-level ``baseTimeNanoseconds`` in Proton chrome traces. Proton ROCm
activity traces use a monotonic GPU-clock anchor while their CPU scope events
retain an absolute clock; the merger handles both timestamp forms.

On AMD, launch the server with ``ROCR_VISIBLE_DEVICES`` rather than
``HIP_VISIBLE_DEVICES`` when using Proton. If the host cannot attach a ROCm
activity backend, set ``TOKENSPEED_KERNEL_PROFILE_BACKEND=instrumentation`` to
collect mergeable Proton kernel scopes without acquiring the ROCm profiler.

Usage (one file pair per scheduler rank):

    tokenspeed merge-traces <run>-TP0.viztracer.json \
        <run>-TP0.proton.chrome_trace -o <run>-TP0-merged.json

Open the merged report in vizviewer or https://ui.perfetto.dev — Python
frames and Proton's kernel/scope lanes share one time axis. Alignment
accuracy is microsecond-to-millisecond grade (the profilers reconcile GPU
and CPU clocks independently); use it to correlate host activity with GPU
gaps, not for sub-microsecond attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["merge_proton_viztracer", "main"]

# Proton labels its single synthetic process (pid 0) "Trace"; rename it so
# the lanes are recognizable next to the VizTracer process in a merged view.
_PROTON_PROCESS_NAME = "Proton"


def _load_json(path: str | Path, kind: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{kind} file {path} is not valid JSON: {exc}") from exc


def _proton_timestamp_offset_us(
    proton_base_ns: int | float,
    viztracer_base_ns: int | float,
    timestamp_us: int | float,
) -> float:
    """Return the shift that puts one Proton event on the VizTracer axis.

    Most Proton writers use an absolute clock anchor and relative event
    timestamps. ROCm activity traces instead use a monotonic GPU-clock anchor
    for kernel/flow timestamps while CPU scope timestamps retain the absolute
    clock. The latter anchor is far smaller than the VizTracer wall-to-monotonic
    conversion offset, which lets us distinguish the formats without relying
    on a backend-specific marker in the JSON.
    """
    if proton_base_ns >= viztracer_base_ns / 2:
        return (proton_base_ns - viztracer_base_ns) / 1000.0

    proton_base_us = proton_base_ns / 1000.0
    viztracer_base_us = viztracer_base_ns / 1000.0
    if timestamp_us >= viztracer_base_us / 2:
        # CPU scope timestamp on the absolute clock.
        return proton_base_us - viztracer_base_us
    # GPU activity or flow timestamp relative to the monotonic anchor.
    return proton_base_us


def merge_proton_viztracer(
    viztracer_path: str | Path,
    proton_path: str | Path,
    output_path: str | Path,
) -> int:
    """Merge a Proton chrome trace into a VizTracer report.

    Converts Proton relative and absolute timestamp forms onto the VizTracer
    monotonic axis and appends them to the report ``traceEvents``, producing
    one chrome-trace JSON on a shared time axis.

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

    viztracer_base_ns = viztracer_json.get("viztracer_metadata", {}).get(
        "baseTimeNanoseconds"
    )
    if viztracer_base_ns is None:
        raise ValueError(
            f"{viztracer_path} has no viztracer_metadata.baseTimeNanoseconds; "
            "re-record with a viztracer version that stores the report's "
            "absolute time base."
        )

    proton_base_ns = proton_json.get("baseTimeNanoseconds")
    if proton_base_ns is None:
        raise ValueError(
            f"{proton_path} has no baseTimeNanoseconds; the trace was written "
            "by a tokenspeed-triton without absolute-anchor support. Upgrade "
            "tokenspeed-triton and re-record."
        )

    proton_events = proton_json.get("traceEvents", [])
    for event in proton_events:
        if event.get("ph") == "M":
            # Metadata events carry no meaningful timestamp; drop any so
            # they cannot clobber the report's process/thread names.
            event.pop("ts", None)
            if (
                event.get("name") == "process_name"
                and event.get("args", {}).get("name") == "Trace"
            ):
                event["args"]["name"] = _PROTON_PROCESS_NAME
        elif "ts" in event:
            event["ts"] += _proton_timestamp_offset_us(
                proton_base_ns, viztracer_base_ns, event["ts"]
            )

    viztracer_json.setdefault("traceEvents", []).extend(proton_events)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(viztracer_json, f)
    return len(proton_events)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``tokenspeed merge-traces``.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        prog="tokenspeed merge-traces",
        description="Merge a Proton chrome trace into a VizTracer report "
        "on a shared timeline (one file pair per scheduler rank).",
    )
    parser.add_argument("viztracer_json", help="VizTracer report (.json)")
    parser.add_argument("proton_trace", help="Proton chrome trace (.chrome_trace)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <viztracer stem>-merged.json)",
    )
    args = parser.parse_args(argv)

    output = args.output
    if output is None:
        viztracer_path = Path(args.viztracer_json)
        output = viztracer_path.with_name(f"{viztracer_path.stem}-merged.json")

    merged = merge_proton_viztracer(args.viztracer_json, args.proton_trace, output)
    print(f"Merged {merged} Proton events into {output}")


if __name__ == "__main__":
    main()
