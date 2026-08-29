import json
import subprocess
import sys
from pathlib import Path

from pipeline import check_perf_reference, extract_perf_summary_rows


def test_random_benchmark_output_can_be_gated(tmp_path):
    run_dir = tmp_path / "input_4k" / "parallel_8"
    run_dir.mkdir(parents=True)
    (run_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "Concurrency": 8,
                "TPOT (ms)": 10.0,
                "Output Throughput (tok/s)": 4000.0,
                "KV Cache Hit Rate (%)": 0.0,
                "Decoded Tok/Iter": 1.0,
            }
        )
    )

    collector = (
        Path(__file__).resolve().parents[1]
        / "random_benchmark"
        / "tokenspeed"
        / "collect_outputs.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            collector,
            tmp_path,
            "--num-gpus",
            "8",
            "--emit-csv",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    rows = extract_perf_summary_rows(result.stdout)
    assert rows[0]["Conc."] == "8"
    assert rows[0]["Latency (tps/user)"] == "100.0"
    assert rows[0]["Throughput (tps/gpu)"] == "500.0"

    check = check_perf_reference(
        {"perf_threshold": 0.9, "perf_reference": {8: [100, 500]}},
        [{"perf_summary_rows": rows}],
        ["perf"],
    )
    assert check["passed"]


def test_random_benchmark_accepts_evalscope_1_10_metric_names(tmp_path):
    run_dir = tmp_path / "input_4k" / "parallel_1"
    run_dir.mkdir(parents=True)
    (run_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "Concurrency": 1,
                "Avg TPOT (ms)": 25.0,
                "Output Throughput (tok/s)": 32.0,
                "KV Cache Hit Rate (%)": 0.0,
                "Avg Decoded Tok/Iter": 2.0,
            }
        )
    )

    collector = (
        Path(__file__).resolve().parents[1]
        / "random_benchmark"
        / "tokenspeed"
        / "collect_outputs.py"
    )
    result = subprocess.run(
        [sys.executable, collector, tmp_path, "--num-gpus", "8", "--emit-csv"],
        capture_output=True,
        text=True,
        check=True,
    )

    rows = extract_perf_summary_rows(result.stdout)
    assert rows[0]["Latency (tps/user)"] == "40.0"
    assert rows[0]["Throughput (tps/gpu)"] == "4.0"
    assert rows[0]["Decoded Tok/Iter"] == "2.0"
