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

"""Data-only qualification-tool checks; these are not GPU performance results."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "random_benchmark/tokenspeed/deepep_bcg_qualification.py"
)
SPEC = importlib.util.spec_from_file_location("deepep_bcg_qualification", SCRIPT)
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


@pytest.fixture
def config():
    return {
        "server_argv": [
            "tokenspeed",
            "serve",
            "test-model",
            "--max-cudagraph-capture-size",
            "16",
            "--max-total-tokens",
            "8192",
        ],
        "client_argv": [
            "evalscope",
            "perf",
            "--stream",
            "--no-timestamp",
            "--seed",
            "7",
        ],
        "pairs": 8,
        "expected_ranks": [0, 1, 2, 3],
        "seed": 7,
        "prefill_graph_max_tokens": 8192,
        "capture_sizes": [32, 128, 512, 2048, 8192],
        "provenance": {
            "tokenspeed_revision": "test-revision",
            "model_revision": "test-model-revision",
            "dependencies": "fixture, not a hardware run",
            "hardware": "fixture, no GPUs",
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_trial(root, run, ttft_key):
    directory = root / run["directory"] / "evalscope" / "parallel_1_number_2"
    directory.mkdir(parents=True)
    write_json(
        directory / "benchmark_summary.json",
        {
            "Success Requests": 2,
            "Failed Requests": 0,
            "Concurrency": 1,
            ttft_key: 100.0 if run["arm"] == "eager" else 90.0,
            "TPOT (ms)": 0.0,
            "Output Throughput (tok/s)": 20.0,
        },
    )
    write_json(
        directory / "benchmark_args.json",
        {"seed": 7, "stream": True, "outputs_dir": str(directory)},
    )
    write_json(
        directory / "benchmark_percentile.json",
        [
            {"Percentiles": "50%", "TTFT (ms)": 80.0},
            {"Percentiles": "95%", "TTFT (ms)": 110.0},
            {"Percentiles": "99%", "TTFT (ms)": 120.0},
        ],
    )
    evidence = root / run["directory"] / "evidence"
    evidence.mkdir()
    markers = "\n".join(
        f"DeepEP prefill BCG replay: rank={rank} bucket=128 real_tokens=123"
        for rank in range(4)
    )
    (evidence / "server.log").write_text(markers if run["arm"] == "bcg" else "")
    return directory


def test_seeded_schedule_keeps_decode_and_kv_settings(config, tmp_path):
    schedule = qualification.make_schedule(config, tmp_path)
    assert schedule == qualification.make_schedule(config, tmp_path)
    assert {schedule["runs"][index]["arm"] for index in range(0, 16, 2)} == {
        "bcg",
        "eager",
    }
    for run in schedule["runs"]:
        assert run["server_argv"][: len(config["server_argv"])] == config["server_argv"]
        assert run["client_argv"][: len(config["client_argv"])] == config["client_argv"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "option",
    ["--enforce-eager", "--disable-prefill-graph", "--prefill-graph-max-tokens=128"],
)
def test_rejects_server_configuration_confounds(config, tmp_path, option):
    config["server_argv"].append(option)
    with pytest.raises(ValueError, match="must omit"):
        qualification.make_schedule(config, tmp_path)


@pytest.mark.parametrize("ttft_key", ["TTFT (ms)", "Avg TTFT (ms)"])
def test_ttft_is_direct_and_missing_metrics_stay_missing(config, tmp_path, ttft_key):
    schedule = qualification.make_schedule(config, tmp_path)
    write_json(tmp_path / "schedule.json", schedule)
    for run in schedule["runs"]:
        write_trial(tmp_path, run, ttft_key)
    report = qualification.build_report(tmp_path, schedule)
    ttft = report["comparisons"]["client_ttft_mean_ms"]
    assert ttft["mean_paired_reduction_percent"] == pytest.approx(10)
    assert ttft["paired_reduction_95_percent_interval"] == pytest.approx([10, 10])
    assert report["comparisons"]["server_prefill_mean_ms"] is None
    assert report["comparisons"]["client_tpot_mean_ms"] is None
    assert report["comparisons"]["cache_hit_percent"] is None
    assert report["runs"][0]["metrics"]["client_ttft_p99_ms"] == 120
    assert all(source["sha256"] for source in report["runs"][0]["sources"])


def test_server_prefill_requires_separate_samples_and_boundary(config, tmp_path):
    run = qualification.make_schedule(config, tmp_path)["runs"][0]
    write_trial(tmp_path, run, "TTFT (ms)")
    evidence = tmp_path / run["directory"] / "evidence"
    (evidence / "server_prefill.csv").write_text("latency_ms\n1\n2\n3\n")
    with pytest.raises(FileNotFoundError):
        qualification.read_run(tmp_path, run, config["expected_ranks"])
    (evidence / "server_prefill_source.txt").write_text(
        "Test fixture: three wall-clock forward samples, no hardware measurements."
    )
    result = qualification.read_run(tmp_path, run, config["expected_ranks"])
    assert result["metrics"]["server_prefill_mean_ms"] == 2
    assert result["metrics"]["server_prefill_p50_ms"] == 2
    assert result["metrics"]["client_ttft_mean_ms"] != 2


@pytest.mark.parametrize("mutation", ["missing_ttft", "nan", "failed", "duplicate"])
def test_incomplete_or_failed_runs_do_not_produce_speedup(config, tmp_path, mutation):
    run = qualification.make_schedule(config, tmp_path)["runs"][0]
    directory = write_trial(tmp_path, run, "TTFT (ms)")
    path = directory / "benchmark_summary.json"
    summary = json.loads(path.read_text())
    if mutation == "missing_ttft":
        summary.pop("TTFT (ms)")
    elif mutation == "nan":
        summary["TTFT (ms)"] = float("nan")
    elif mutation == "failed":
        summary["Failed Requests"] = 1
    else:
        duplicate = directory.parent / "duplicate"
        duplicate.mkdir()
        write_json(duplicate / "benchmark_summary.json", summary)
    write_json(path, summary)
    with pytest.raises(ValueError):
        qualification.read_run(tmp_path, run, config["expected_ranks"])


def test_actual_evalscope_arguments_must_match(config, tmp_path):
    schedule = qualification.make_schedule(config, tmp_path)
    write_json(tmp_path / "schedule.json", schedule)
    for run in schedule["runs"]:
        directory = write_trial(tmp_path, run, "TTFT (ms)")
        if run["pair"] == 1 and run["arm"] == "bcg":
            write_json(directory / "benchmark_args.json", {"seed": 999, "stream": True})
    with pytest.raises(ValueError, match="arguments differ"):
        qualification.build_report(tmp_path, schedule)


def test_schedule_tampering_rejected(config, tmp_path):
    schedule = qualification.make_schedule(config, tmp_path)
    schedule["runs"][0]["server_argv"].append("--enforce-eager")
    with pytest.raises(ValueError, match="schedule was modified"):
        qualification.build_report(tmp_path, schedule)


def test_paired_bootstrap_resamples_pairs_and_has_deterministic_uncertainty():
    eager = [100.0] * 8
    bcg = [50.0, 150.0] * 4
    result = qualification.paired_reduction(eager, bcg, 17, 10000)
    assert result == qualification.paired_reduction(eager, bcg, 17, 10000)
    assert result["mean_paired_reduction_percent"] == 0
    lower, upper = result["paired_reduction_95_percent_interval"]
    assert lower < 0 < upper


@pytest.mark.parametrize(
    "arm, ranks", [("bcg", []), ("bcg", [0, 1, 2]), ("eager", [0])]
)
def test_report_requires_candidate_replay_on_every_rank(config, tmp_path, arm, ranks):
    run = next(
        run
        for run in qualification.make_schedule(config, tmp_path)["runs"]
        if run["arm"] == arm
    )
    write_trial(tmp_path, run, "TTFT (ms)")
    server_log = tmp_path / run["directory"] / "evidence/server.log"
    server_log.write_text(
        "\n".join(
            f"DeepEP prefill BCG replay: rank={rank} bucket=128 real_tokens=123"
            for rank in ranks
        )
    )
    with pytest.raises(ValueError, match="replay"):
        qualification.read_run(tmp_path, run, config["expected_ranks"])


def test_can_report_artifacts_copied_from_gpu_host(config, tmp_path):
    schedule = qualification.make_schedule(config, Path("/original/gpu-host/output"))
    write_json(tmp_path / "schedule.json", schedule)
    for run in schedule["runs"]:
        write_trial(tmp_path, run, "TTFT (ms)")
    report = qualification.build_report(tmp_path, schedule)
    assert report["comparisons"]["client_ttft_mean_ms"]["pairs"] == 8
