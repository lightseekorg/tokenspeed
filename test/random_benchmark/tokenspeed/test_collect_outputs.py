import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).with_name("collect_outputs.py")
SPEC = importlib.util.spec_from_file_location(
    "random_benchmark_collect_outputs", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
collect_outputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect_outputs)

REPO_ROOT = MODULE_PATH.parents[3]

EXPECTED_CELLS = [
    ("input_1k", 1, 4),
    ("input_1k", 4, 16),
    ("input_1k", 16, 64),
    ("input_8k", 1, 4),
    ("input_8k", 4, 16),
    ("input_8k", 16, 64),
    ("input_32k", 1, 4),
    ("input_32k", 4, 16),
]


def _write_summary(
    root,
    config,
    concurrency,
    requests,
    *,
    run="run",
    overrides=None,
):
    summary_dir = root / config / run
    summary_dir.mkdir(parents=True)
    summary = {
        "TPOT (ms)": 20.0,
        "Output Throughput (tok/s)": 400.0,
        "Concurrency": concurrency,
        "Total Requests": requests,
        "Success Requests": requests,
        "Failed Requests": 0,
        "KV Cache Hit Rate (%)": 75.0,
        "Decoded Tok/Iter": 1.0,
    }
    summary.update(overrides or {})
    path = summary_dir / "benchmark_summary.json"
    path.write_text(json.dumps(summary))
    return path


def _write_expected_matrix(root):
    for config, concurrency, requests in EXPECTED_CELLS:
        _write_summary(
            root,
            config,
            concurrency,
            requests,
            run=f"parallel_{concurrency}_number_{requests}",
        )


def test_collect_uses_ci_throughput_column(tmp_path):
    summary_dir = tmp_path / "input_1k" / "run"
    summary_dir.mkdir(parents=True)
    (summary_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "TPOT (ms)": 20,
                "Output Throughput (tok/s)": 400,
                "Concurrency": 8,
                "KV Cache Hit Rate (%)": 75,
                "Avg Decoded Tokens/Iter": 2.5,
            }
        )
    )

    rows = collect_outputs.collect(tmp_path, num_gpus=4)

    assert rows == [
        {
            "config": "input_1k",
            "Conc.": 8,
            "Latency (tps/user)": 50.0,
            "Throughput (tps/gpu)": 100.0,
            "Approx Cache Hit": 75.0,
            "Decoded Tok/Iter": 2.5,
        }
    ]


def test_print_table_uses_ci_throughput_header(capsys):
    row = {column: 1 for column in collect_outputs.COLUMNS}

    collect_outputs.print_table([row])

    output = capsys.readouterr().out
    lines = output.strip().splitlines()
    assert lines[0] == "Overall perf table:"
    assert lines[1] == ",".join(collect_outputs.COLUMNS)
    assert lines[2] == "1,1,1,1,1,1"
    assert "Output Throughput (tps/gpu)" not in output


def test_input_32k_sorts_after_shorter_known_configs():
    rows = [
        {"config": "input_32k", "Conc.": 1},
        {"config": "input_8k", "Conc.": 1},
        {"config": "input_1k", "Conc.": 1},
    ]

    rows.sort(key=collect_outputs._sort_key)

    assert [row["config"] for row in rows] == [
        "input_1k",
        "input_8k",
        "input_32k",
    ]


def test_strict_validation_accepts_exact_matrix(tmp_path):
    _write_expected_matrix(tmp_path)

    rows, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=EXPECTED_CELLS
    )

    assert len(rows) == 8
    assert report["status"] == "pass"
    assert report["expected_cells"] == 8
    assert report["observed_summary_files"] == 8
    assert report["observed_cells"] == 8
    assert report["observed_unique_cells"] == 8
    assert report["request_totals"] == {
        "expected": 188,
        "total": 188,
        "success": 188,
        "failed": 0,
    }
    assert report["failures"] == []


def test_strict_validation_rejects_missing_cell(tmp_path):
    _write_expected_matrix(tmp_path)
    missing = tmp_path / "input_32k" / "parallel_4_number_16" / "benchmark_summary.json"
    missing.unlink()

    _, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=EXPECTED_CELLS
    )

    assert report["status"] == "fail"
    assert "missing expected cell input_32k:4" in report["failures"]
    assert report["request_totals"]["total"] == 172


def test_strict_validation_rejects_unexpected_cell(tmp_path):
    _write_expected_matrix(tmp_path)
    _write_summary(tmp_path, "input_2k", 2, 8)

    _, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=EXPECTED_CELLS
    )

    assert report["status"] == "fail"
    assert "unexpected cell input_2k:2" in report["failures"]
    assert report["request_totals"]["total"] == 196


def test_strict_validation_rejects_duplicate_cell(tmp_path):
    _write_expected_matrix(tmp_path)
    _write_summary(tmp_path, "input_1k", 1, 4, run="duplicate")

    _, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=EXPECTED_CELLS
    )

    assert report["status"] == "fail"
    assert any("duplicate cell input_1k:1" in error for error in report["failures"])
    assert report["observed_summary_files"] == 9
    assert report["observed_cells"] == 9
    assert report["observed_unique_cells"] == 8


def test_strict_validation_rejects_request_count_mismatch(tmp_path):
    _write_summary(
        tmp_path,
        "input_1k",
        1,
        4,
        overrides={"Total Requests": 3, "Success Requests": 3},
    )

    _, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=[("input_1k", 1, 4)]
    )

    assert report["status"] == "fail"
    assert any(
        "total requests 3 != expected 4" in error for error in report["failures"]
    )
    assert any(
        "success requests 3 != expected 4" in error for error in report["failures"]
    )


def test_strict_validation_rejects_failed_request(tmp_path):
    _write_summary(
        tmp_path,
        "input_1k",
        1,
        4,
        overrides={"Success Requests": 3, "Failed Requests": 1},
    )

    _, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=[("input_1k", 1, 4)]
    )

    assert report["status"] == "fail"
    assert any(
        "success requests 3 != expected 4" in error for error in report["failures"]
    )
    assert any(
        "failed requests 1 != expected 0" in error for error in report["failures"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("TPOT (ms)", "nan"),
        ("Output Throughput (tok/s)", "inf"),
        ("Decoded Tok/Iter", 0.0),
        ("Decoded Tok/Iter", -1.0),
    ],
)
def test_strict_validation_requires_positive_finite_perf(tmp_path, field, value):
    _write_summary(tmp_path, "input_1k", 1, 4, overrides={field: value})

    rows, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=[("input_1k", 1, 4)]
    )

    assert rows == []
    assert report["status"] == "fail"
    assert any(
        repr(field) in error and "positive finite number" in error
        for error in report["failures"]
    )


def test_strict_validation_requires_exact_success_field(tmp_path):
    _write_summary(
        tmp_path,
        "input_1k",
        1,
        4,
        overrides={"Success Requests": None, "Successful Requests": 4},
    )
    path = tmp_path / "input_1k" / "run" / "benchmark_summary.json"
    summary = json.loads(path.read_text())
    del summary["Success Requests"]
    path.write_text(json.dumps(summary))

    _, report = collect_outputs.collect_and_validate(
        tmp_path, num_gpus=4, expected_cells=[("input_1k", 1, 4)]
    )

    assert report["status"] == "fail"
    assert any("missing 'Success Requests'" in error for error in report["failures"])


def test_collect_without_expectations_keeps_legacy_loose_behavior(tmp_path):
    for run in ("first", "second"):
        summary_dir = tmp_path / "input_1k" / run
        summary_dir.mkdir(parents=True)
        (summary_dir / "benchmark_summary.json").write_text(
            json.dumps(
                {
                    "TPOT (ms)": 0,
                    "Output Throughput (tok/s)": "not-a-number",
                    "Concurrency": 1,
                }
            )
        )

    rows = collect_outputs.collect(tmp_path, num_gpus=4)

    assert len(rows) == 2
    assert all(row["Latency (tps/user)"] == 0.0 for row in rows)
    assert all(row["Throughput (tps/gpu)"] == 0.0 for row in rows)


def test_parse_expected_cell():
    assert collect_outputs.parse_expected_cell("input_1k:16:64") == (
        "input_1k",
        16,
        64,
    )


@pytest.mark.parametrize("value", ["input_1k", ":1:4", "input_1k:x:4", "input_1k:0:4"])
def test_parse_expected_cell_rejects_invalid_value(value):
    with pytest.raises(
        collect_outputs.argparse.ArgumentTypeError,
        match="expected|integers|positive",
    ):
        collect_outputs.parse_expected_cell(value)


def test_duplicate_expected_cell_is_a_cli_error(tmp_path, monkeypatch):
    _write_summary(tmp_path, "input_1k", 1, 4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            str(tmp_path),
            "--expect-cell",
            "input_1k:1:4",
            "--expect-cell",
            "input_1k:1:4",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        collect_outputs.main()

    assert exc_info.value.code == 2


def test_strict_cli_writes_failure_validation_before_exit(tmp_path, monkeypatch):
    _write_summary(
        tmp_path,
        "input_1k",
        1,
        4,
        overrides={"Failed Requests": 1, "Success Requests": 3},
    )
    validation_path = tmp_path / "artifacts" / "validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            str(tmp_path),
            "--expect-cell",
            "input_1k:1:4",
            "--validation-json",
            str(validation_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        collect_outputs.main()

    assert exc_info.value.code == 1
    report = json.loads(validation_path.read_text())
    assert report["status"] == "fail"
    assert report["request_totals"] == {
        "expected": 4,
        "total": 4,
        "success": 3,
        "failed": 1,
    }
    assert list(validation_path.parent.glob(".validation.json.*.tmp")) == []


@pytest.mark.parametrize(
    ("task_name", "artifact_name", "reference"),
    [
        (
            "minimax-m3-mxfp8-evalscope-random.yaml",
            "minimax-m3-fp8-random",
            {
                "input_1k": {1: [11.49, 2.85], 4: [11.63, 11.36], 16: [10.58, 39.44]},
                "input_8k": {1: [11.50, 2.74], 4: [11.10, 9.78], 16: [8.24, 25.08]},
                "input_32k": {1: [11.47, 2.39], 4: [9.17, 6.48]},
            },
        ),
        (
            "minimax-m3-bf16-evalscope-random.yaml",
            "minimax-m3-bf16-random",
            {
                "input_1k": {1: [11.55, 2.87], 4: [11.68, 11.42], 16: [10.64, 39.71]},
                "input_8k": {1: [11.56, 2.76], 4: [11.17, 9.88], 16: [8.33, 25.58]},
                "input_32k": {1: [11.55, 2.43], 4: [9.29, 6.68]},
            },
        ),
    ],
)
def test_minimax_m3_random_task_locks_release_matrix(
    task_name, artifact_name, reference
):
    task_path = REPO_ROOT / "test" / "ci" / "perf" / task_name
    task = yaml.safe_load(task_path.read_text())
    command = task["perf"]["command"]
    expected_patterns = [
        "Prefill graph capture failed",
        "Scheduler hit an exception:",
        "Buffer overflow when allocating memory",
        "CUDA out of memory",
        "OutOfMemoryError",
        "Retract failed",
        "NCCL.*(?:abort|unhandled)",
    ]

    assert task["timeout_minutes"] == 120
    assert task["server"]["forbidden_log_patterns"] == expected_patterns
    assert task["perf_threshold"] == 0.9
    assert task["perf_reference"] == reference
    assert f"artifact_dir=.ci-artifacts/{artifact_name}" in command
    assert (
        f"--output .ci-artifacts/{artifact_name}/runtime_environment.json"
        in " ".join(task["install"])
    )
    assert '--validation-json "$artifact_dir/validation.json"' in command
    for config, concurrency, requests in EXPECTED_CELLS:
        assert f"--expect-cell {config}:{concurrency}:{requests}" in command
    assert sum(requests for _, _, requests in EXPECTED_CELLS) == 188
