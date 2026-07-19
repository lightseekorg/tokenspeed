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

"""Tests for :mod:`tokenspeed.cli.trace_merge` using synthetic trace files.

The synthetic files encode the on-disk contracts this tool depends on:
VizTracer reports store their absolute time base in
``viztracer_metadata.baseTimeNanoseconds``; Proton chrome traces store it
top-level as ``baseTimeNanoseconds`` with relative-microsecond event
timestamps, metadata (``ph == "M"``) events without timestamps, and a
synthetic pid-0 process named "Trace".
"""

from __future__ import annotations

import json

import pytest

from tokenspeed.cli.trace_merge import (
    main,
    merge_all_ranks,
    merge_proton_viztracer,
)

VIZTRACER_BASE_NS = 1_000_000_000_000_000
# Proton session started 2.5 ms after the viztracer clock origin.
PROTON_BASE_NS = VIZTRACER_BASE_NS + 2_500_000
EXPECTED_OFFSET_US = 2_500.0


def _viztracer_report() -> dict:
    return {
        "displayTimeUnit": "us",
        "traceEvents": [
            {
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 4242,
                "tid": 4242,
                "name": "forward",
                "cat": "fee",
            },
        ],
        "viztracer_metadata": {
            "version": "1.1.1",
            "baseTimeNanoseconds": VIZTRACER_BASE_NS,
        },
    }


def _proton_trace() -> dict:
    return {
        "displayTimeUnit": "us",
        "baseTimeNanoseconds": PROTON_BASE_NS,
        "traceEvents": [
            {
                "ph": "M",
                "pid": 0,
                "tid": 0,
                "name": "process_name",
                "args": {"name": "Trace"},
            },
            {
                "ph": "M",
                "pid": 0,
                "tid": 100,
                "name": "thread_name",
                "args": {"name": "GPU Stream 7"},
            },
            {
                "ph": "X",
                "ts": 10.0,
                "dur": 5.0,
                "pid": 0,
                "tid": 100,
                "name": "gemm.mm[triton_mm]",
                "cat": "kernel",
            },
            {
                "ph": "s",
                "ts": 8.0,
                "pid": 0,
                "tid": 1,
                "id": 1,
                "cat": "flow",
                "name": "launch->kernel",
            },
            {
                "ph": "f",
                "ts": 10.0,
                "pid": 0,
                "tid": 100,
                "id": 1,
                "cat": "flow",
                "name": "launch->kernel",
            },
        ],
    }


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_merge_shifts_proton_events_onto_viztracer_axis(tmp_path):
    viztracer_path = _write(tmp_path, "run.viztracer.json", _viztracer_report())
    proton_path = _write(tmp_path, "run.proton.chrome_trace", _proton_trace())
    output_path = tmp_path / "merged.json"

    merged_count = merge_proton_viztracer(viztracer_path, proton_path, output_path)
    assert merged_count == 5

    merged = json.loads(output_path.read_text())
    events = merged["traceEvents"]
    assert len(events) == 6

    # VizTracer events are untouched; its metadata anchor is preserved.
    assert events[0] == _viztracer_report()["traceEvents"][0]
    assert merged["viztracer_metadata"]["baseTimeNanoseconds"] == VIZTRACER_BASE_NS

    by_name = {e["name"]: e for e in events[1:]}
    # Proton's synthetic process is renamed for the merged view, and
    # metadata events stay timestamp-free.
    assert by_name["process_name"]["args"]["name"] == "Proton"
    assert "ts" not in by_name["process_name"]
    assert by_name["thread_name"]["args"]["name"] == "GPU Stream 7"

    # Timed events (complete and flow) are shifted by the anchor delta.
    assert by_name["gemm.mm[triton_mm]"]["ts"] == 10.0 + EXPECTED_OFFSET_US
    assert by_name["gemm.mm[triton_mm]"]["dur"] == 5.0
    flow_ts = sorted(e["ts"] for e in events if e.get("cat") == "flow")
    assert flow_ts == [8.0 + EXPECTED_OFFSET_US, 10.0 + EXPECTED_OFFSET_US]
    assert {e["id"] for e in events if e.get("cat") == "flow"} == {1}


def test_merge_rejects_viztracer_report_without_anchor(tmp_path):
    report = _viztracer_report()
    del report["viztracer_metadata"]["baseTimeNanoseconds"]
    viztracer_path = _write(tmp_path, "run.viztracer.json", report)
    proton_path = _write(tmp_path, "run.proton.chrome_trace", _proton_trace())

    with pytest.raises(ValueError, match="viztracer_metadata.baseTimeNanoseconds"):
        merge_proton_viztracer(viztracer_path, proton_path, tmp_path / "out.json")


def test_merge_rejects_proton_trace_without_anchor(tmp_path):
    trace = _proton_trace()
    del trace["baseTimeNanoseconds"]
    viztracer_path = _write(tmp_path, "run.viztracer.json", _viztracer_report())
    proton_path = _write(tmp_path, "run.proton.chrome_trace", trace)

    with pytest.raises(ValueError, match="baseTimeNanoseconds"):
        merge_proton_viztracer(viztracer_path, proton_path, tmp_path / "out.json")


def test_cli_defaults_output_next_to_viztracer_report(tmp_path, capsys):
    viztracer_path = _write(tmp_path, "run.viztracer.json", _viztracer_report())
    proton_path = _write(tmp_path, "run.proton.chrome_trace", _proton_trace())

    main([viztracer_path, proton_path])

    default_output = tmp_path / "run.viztracer-merged.json"
    assert default_output.exists()
    assert "Merged 5 Proton events" in capsys.readouterr().out


def _rank_viztracer_report(*, base_time_ns: int, pid: int) -> dict:
    report = _viztracer_report()
    event = report["traceEvents"][0]
    event["pid"] = pid
    event["tid"] = pid
    report["viztracer_metadata"]["baseTimeNanoseconds"] = base_time_ns
    return report


def _rank_proton_trace(*, base_time_ns: int) -> dict:
    trace = _proton_trace()
    trace["baseTimeNanoseconds"] = base_time_ns
    return trace


def test_merge_all_ranks_aligns_timelines_and_namespaces_flow_ids(tmp_path):
    rank0_viz = _write(
        tmp_path,
        "run-TP0.viztracer.json",
        _rank_viztracer_report(base_time_ns=VIZTRACER_BASE_NS, pid=4242),
    )
    rank0_proton = _write(
        tmp_path,
        "run-TP0.proton.chrome_trace",
        _rank_proton_trace(base_time_ns=PROTON_BASE_NS),
    )

    rank1_base = VIZTRACER_BASE_NS + 10_000
    rank1_viz = _write(
        tmp_path,
        "run-TP1.viztracer.json",
        _rank_viztracer_report(base_time_ns=rank1_base, pid=4243),
    )
    rank1_proton = _write(
        tmp_path,
        "run-TP1.proton.chrome_trace",
        _rank_proton_trace(base_time_ns=rank1_base + 2_500_000),
    )
    output_path = tmp_path / "all-ranks.json"

    merged_count = merge_all_ranks(
        [(1, rank1_viz, rank1_proton), (0, rank0_viz, rank0_proton)],
        output_path,
    )
    assert merged_count == 10

    merged = json.loads(output_path.read_text())
    assert merged["viztracer_metadata"]["baseTimeNanoseconds"] == VIZTRACER_BASE_NS
    assert len(merged["traceEvents"]) == 12

    forwards = sorted(
        (event["pid"], event["ts"])
        for event in merged["traceEvents"]
        if event["name"] == "forward"
    )
    assert forwards == [(4242, 100.0), (4243, 110.0)]

    proton_processes = {
        event["pid"]: event["args"]["name"]
        for event in merged["traceEvents"]
        if event["ph"] == "M" and event["name"] == "process_name"
    }
    assert proton_processes == {10000: "Proton TP0", 10001: "Proton TP1"}

    flow_events = [
        event
        for event in merged["traceEvents"]
        if event.get("name") == "launch->kernel"
    ]
    assert {(event["pid"], event["id"]) for event in flow_events} == {
        (10000, 1),
        (10001, (1 << 32) | 1),
    }
    assert {event["id"] for event in flow_events if event["ph"] == "s"} == {
        1,
        (1 << 32) | 1,
    }
    assert {event["id"] for event in flow_events if event["ph"] == "f"} == {
        1,
        (1 << 32) | 1,
    }
    assert all(isinstance(event["id"], int) for event in flow_events)


def test_all_ranks_cli_merges_explicit_rank_triples(tmp_path, capsys):
    rank0_viz = _write(
        tmp_path,
        "run-TP0.viztracer.json",
        _rank_viztracer_report(base_time_ns=VIZTRACER_BASE_NS, pid=4242),
    )
    rank0_proton = _write(
        tmp_path,
        "run-TP0.proton.chrome_trace",
        _rank_proton_trace(base_time_ns=PROTON_BASE_NS),
    )
    rank1_viz = _write(
        tmp_path,
        "run-TP1.viztracer.json",
        _rank_viztracer_report(base_time_ns=VIZTRACER_BASE_NS, pid=4243),
    )
    rank1_proton = _write(
        tmp_path,
        "run-TP1.proton.chrome_trace",
        _rank_proton_trace(base_time_ns=PROTON_BASE_NS),
    )
    output_path = tmp_path / "all-ranks.json"

    main(
        [
            "--all-ranks",
            "--rank",
            "0",
            rank0_viz,
            rank0_proton,
            "--rank",
            "1",
            rank1_viz,
            rank1_proton,
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    assert "Merged 10 Proton events across 2 ranks" in capsys.readouterr().out
