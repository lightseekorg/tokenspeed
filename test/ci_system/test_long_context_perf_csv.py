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

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

COLLECTOR = (
    Path(__file__).resolve().parents[1]
    / "long_context_benchmark"
    / "tokenspeed"
    / "collect_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("long_context_collect_outputs", COLLECTOR)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def write_summary(root, summary, *, run, length):
    path = root / f"run{run}" / length / "model" / "parallel_1_number_1"
    path.mkdir(parents=True)
    path = path / "benchmark_summary.json"
    path.write_text(json.dumps(summary))
    return path


@pytest.mark.parametrize(
    "metrics",
    [
        {"TPOT (ms)": 2.0, "Decoded Tok/Iter": 3.4},
        {"Avg TPOT (ms)": "2.0", "Avg Decoded Tok/Iter": 3.4},
        {"Avg TPOT (ms)": 2.0, "TPOT (ms)": 10.0, "Avg Decoded Tokens/Iter": 3.4},
    ],
)
def test_current_and_legacy_metrics_produce_nonzero_report(tmp_path, metrics):
    write_summary(tmp_path, metrics, run=1, length="128k")
    rows = collector.collect(tmp_path)
    assert rows[0]["tps_per_user"] == 500.0
    assert rows[0]["acceptance_rate"] == 3.4


def test_three_runs_at_each_length_preserve_current_metrics_in_csv(tmp_path):
    # Synthetic representative values based on CI log tables, not original JSON.
    tpot_by_length = {"128k": 1.82, "256k": 1.78, "512k": 1.84, "1024k": 2.11}
    for run in range(1, 4):
        for length, tpot in tpot_by_length.items():
            write_summary(
                tmp_path,
                {"Avg TPOT (ms)": tpot, "Avg Decoded Tok/Iter": 3.4},
                run=run,
                length=length,
            )
    output = tmp_path / "summary.csv"
    process = subprocess.run(
        [sys.executable, COLLECTOR, tmp_path, "-o", output],
        capture_output=True,
        text=True,
        check=True,
    )
    with output.open() as stream:
        rows = list(csv.reader(stream))
    runs = [row for row in rows if row and row[0] == "per_run"]
    aggregates = [row for row in rows if row and row[0] == "agg"]
    assert len(runs) == 12
    assert len(aggregates) == 4
    for row in runs:
        assert float(row[3]) == round(1000 / tpot_by_length[row[1]], 2)
        assert float(row[4]) == 3.4
    assert all(int(row[2]) == 3 for row in aggregates)
    assert "Overall perf table:" in process.stdout


@pytest.mark.parametrize(
    "bad", [None, True, False, "bad", 0, -1, "NaN", "Infinity", float("inf")]
)
def test_invalid_current_metric_does_not_fall_back_or_hide_run(tmp_path, bad):
    write_summary(tmp_path, {"Avg TPOT (ms)": 2}, run=1, length="128k")
    path = write_summary(
        tmp_path,
        {"Avg TPOT (ms)": bad, "TPOT (ms)": 2},
        run=2,
        length="128k",
    )
    process = subprocess.run(
        [sys.executable, COLLECTOR, tmp_path], capture_output=True, text=True
    )
    assert process.returncode != 0
    assert str(path) in process.stderr
    assert "Avg TPOT (ms)" in process.stderr
    assert "Overall perf table:" not in process.stdout


@pytest.mark.parametrize("summary", [{}, [], "invalid JSON"])
def test_missing_metric_or_invalid_summary_fails_collection(tmp_path, summary):
    path = write_summary(tmp_path, summary, run=1, length="128k")
    if summary == "invalid JSON":
        path.write_text("{")
    with pytest.raises(ValueError, match="benchmark_summary.json"):
        collector.collect(tmp_path)


def test_missing_acceptance_is_unavailable_for_whole_aggregate(tmp_path, capsys):
    write_summary(
        tmp_path, {"Avg TPOT (ms)": 2, "Avg Decoded Tok/Iter": 3}, run=1, length="128k"
    )
    write_summary(tmp_path, {"Avg TPOT (ms)": 2}, run=2, length="128k")
    rows = collector.collect(tmp_path)
    summary = collector.aggregate(rows)
    assert summary[0]["avg_acceptance_rate"] is None
    assert summary[0]["min_acceptance_rate"] is None
    assert summary[0]["max_acceptance_rate"] is None
    collector.print_table(rows, summary)
    assert "N/A" in capsys.readouterr().out
    output = tmp_path / "summary.csv"
    collector.write_csv(output, rows, summary)
    with output.open() as stream:
        aggregate = next(row for row in csv.reader(stream) if row and row[0] == "agg")
    assert aggregate[6:9] == ["", "", ""]


def test_empty_sweep_exits_with_failure(tmp_path):
    result = subprocess.run(
        [sys.executable, COLLECTOR, tmp_path], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "no benchmark_summary.json found" in result.stderr
