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

"""Consumer behavior against real rotating JSONL files and checkpoints."""

import json
import os
import runpy
import select
import subprocess
import sys
from pathlib import Path

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/metrics/read_jsonl.py"


def _line(stream, sequence):
    return (
        json.dumps(
            {
                "schema_version": 1,
                "stream_id": stream,
                "sequence": sequence,
                "timestamp_ns": 1_000_000_000 + sequence,
                "metric": "tokenspeed:inter_token_latency_seconds",
                "operation": "observe",
                "value": 0.01,
                "labels": {"model_name": "模型"},
                "weight": 3,
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _reader(directory, checkpoint, warnings):
    reader_type = runpy.run_path(str(SCRIPT))["JsonlMetricsReader"]
    return reader_type(directory, checkpoint, warn=warnings.append)


def test_reads_shards_and_streams_then_resumes_at_byte_offset(tmp_path):
    (tmp_path / "metrics-a-000000.jsonl").write_bytes(_line("a", 0))
    (tmp_path / "metrics-a-000001.jsonl").write_bytes(_line("a", 1))
    (tmp_path / "metrics-b-000000.jsonl").write_bytes(_line("b", 0))
    checkpoint = tmp_path / "checkpoint.json"
    records, warnings = [], []

    assert _reader(tmp_path, checkpoint, warnings).drain(records.append) == 3
    assert [(r["stream_id"], r["sequence"]) for r in records] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
    ]
    assert records[0]["labels"] == {"model_name": "模型"}
    assert warnings == []
    assert _reader(tmp_path, checkpoint, warnings).drain(records.append) == 0
    with (tmp_path / "metrics-a-000001.jsonl").open("ab") as output:
        output.write(_line("a", 2))
    assert _reader(tmp_path, checkpoint, warnings).drain(records.append) == 1
    assert records[-1]["sequence"] == 2


def test_failed_consumption_does_not_advance_checkpoint(tmp_path):
    (tmp_path / "metrics-a-000000.jsonl").write_bytes(_line("a", 0) + _line("a", 1))
    checkpoint = tmp_path / "checkpoint.json"

    def consume(record):
        if record["sequence"] == 1:
            raise RuntimeError("destination unavailable")

    with pytest.raises(RuntimeError, match="destination unavailable"):
        _reader(tmp_path, checkpoint, []).drain(consume)

    records = []
    assert _reader(tmp_path, checkpoint, []).drain(records.append) == 1
    assert [r["sequence"] for r in records] == [1]


def test_waits_for_a_complete_line_without_advancing_checkpoint(tmp_path):
    path = tmp_path / "metrics-a-000000.jsonl"
    line = _line("a", 1)
    path.write_bytes(_line("a", 0) + line[:-2])
    checkpoint = tmp_path / "checkpoint.json"
    records, warnings = [], []
    reader = _reader(tmp_path, checkpoint, warnings)

    assert reader.drain(records.append) == 1
    saved = checkpoint.read_bytes()
    assert reader.drain(records.append) == 0
    assert checkpoint.read_bytes() == saved
    with path.open("ab") as output:
        output.write(line[-2:])
    assert reader.drain(records.append) == 1
    assert [r["sequence"] for r in records] == [0, 1]
    assert warnings == []


def test_abandoned_partial_tail_does_not_block_later_shards_or_streams(tmp_path):
    (tmp_path / "metrics-a-000000.jsonl").write_bytes(_line("a", 0) + b'{"partial":')
    (tmp_path / "metrics-a-000001.jsonl").write_bytes(_line("a", 2))
    (tmp_path / "metrics-b-000000.jsonl").write_bytes(b'{"partial":')
    (tmp_path / "metrics-c-000000.jsonl").write_bytes(_line("c", 0))
    records, warnings = [], []
    reader = _reader(tmp_path, tmp_path / "checkpoint.json", warnings)

    assert reader.drain(records.append) == 3
    assert [(r["stream_id"], r["sequence"]) for r in records] == [
        ("a", 0),
        ("a", 2),
        ("c", 0),
    ]
    assert any("incomplete" in warning for warning in warnings)
    assert any("sequence gap" in warning for warning in warnings)


def test_warns_and_continues_when_retention_removes_checkpoint_file(tmp_path):
    path = tmp_path / "metrics-a-000000.jsonl"
    path.write_bytes(_line("a", 0))
    checkpoint = tmp_path / "checkpoint.json"
    _reader(tmp_path, checkpoint, []).drain(lambda record: None)
    path.unlink()
    (tmp_path / "metrics-a-000002.jsonl").write_bytes(_line("a", 3))
    records, warnings = [], []

    assert _reader(tmp_path, checkpoint, warnings).drain(records.append) == 1
    assert [r["sequence"] for r in records] == [3]
    assert any("missing" in warning for warning in warnings)
    assert any("sequence gap" in warning for warning in warnings)


def test_cli_runs_without_tokenspeed_and_checkpoints_stdout(tmp_path):
    (tmp_path / "metrics-a-000000.jsonl").write_bytes(_line("a", 0))
    command = [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--directory",
        str(tmp_path),
        "--checkpoint",
        str(tmp_path / "checkpoint.json"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sequence"] == 0
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_follow_discovers_new_streams(tmp_path):
    command = [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--directory",
        str(tmp_path),
        "--checkpoint",
        str(tmp_path / "checkpoint.json"),
        "--follow",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for stream in ("a", "b"):
            (tmp_path / f"metrics-{stream}-000000.jsonl").write_bytes(_line(stream, 0))
            ready, _, _ = select.select([process.stdout], [], [], 5)
            assert ready, f"reader did not discover stream {stream}"
            line = process.stdout.readline()
            assert line, f"reader exited with code {process.poll()}"
            assert json.loads(line)["stream_id"] == stream
    finally:
        process.terminate()
        process.wait(timeout=5)
