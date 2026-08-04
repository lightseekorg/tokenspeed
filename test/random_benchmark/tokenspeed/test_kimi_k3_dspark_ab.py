from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("kimi_k3_dspark_ab.py")
SPEC = importlib.util.spec_from_file_location("kimi_k3_dspark_ab", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ab
SPEC.loader.exec_module(ab)


def _row(arm: str, repeat: int, *, tpot: float, output_tps: float):
    metadata = {
        "engine": "tokenspeed",
        "group": "ci",
        "name": "4k-1k-c16",
        "arm": arm,
        "repeat": repeat,
        "isl": 4096,
        "osl": 1024,
        "concurrency": 16,
        "prompts": 32,
        "warmups": 2,
    }
    if arm == "dspark":
        metadata.update({"accept_length": 2.6, "accept_rate": 0.23})
    return {
        "ab_metadata": metadata,
        "failed": 0,
        "median_tpot_ms": tpot,
        "p99_ttft_ms": 1000 + repeat,
        "output_throughput": output_tps,
        "total_token_throughput": output_tps * 5,
    }


def test_full_matrix_covers_each_axis_without_duplicate_points():
    keys = {
        (point.group, point.isl, point.osl, point.concurrency)
        for point in ab.FULL_MATRIX
    }
    assert len(keys) == len(ab.FULL_MATRIX)
    assert {point.group for point in ab.FULL_MATRIX} == {
        "concurrency",
        "throughput",
        "throughput-core",
        "input",
        "output",
    }
    coordinates = {
        (point.isl, point.osl, point.concurrency) for point in ab.FULL_MATRIX
    }
    assert (1024, 1024, 48) in coordinates
    assert (131072, 512, 1) in coordinates
    assert (1024, 4096, 1) in coordinates


def test_point_filter_slices_a_group_for_short_gpu_leases():
    points = ab.selected_points("full", {"concurrency"}, {"1k-1k-c8", "1k-1k-c16"})
    assert [point.name for point in points] == ["1k-1k-c8", "1k-1k-c16"]


def test_server_commands_only_add_native_dspark_flags():
    token_args = ab.parse_args(["--engine", "tokenspeed", "--dry-run"])
    baseline, _, _ = ab.build_server_command(token_args, "no-spec")
    dspark, _, _ = ab.build_server_command(token_args, "dspark")
    assert baseline == dspark[: len(baseline)]
    assert dspark[len(baseline) :] == [
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        ab.DEFAULT_DRAFTS["tokenspeed"],
        "--speculative-num-draft-tokens",
        "8",
        "--drafter-attention-backend",
        "mla",
    ]

    sgl_args = ab.parse_args(["--engine", "sglang", "--dry-run"])
    baseline, _, _ = ab.build_server_command(sgl_args, "no-spec")
    dspark, _, _ = ab.build_server_command(sgl_args, "dspark")
    assert baseline == dspark[: len(baseline)]
    assert dspark[len(baseline) :] == [
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        ab.DEFAULT_DRAFTS["sglang"],
    ]


def test_aggregate_reports_paired_speedups_and_acceptance():
    rows = [
        _row("no-spec", 1, tpot=100, output_tps=80),
        _row("no-spec", 2, tpot=102, output_tps=82),
        _row("no-spec", 3, tpot=98, output_tps=78),
        _row("dspark", 1, tpot=50, output_tps=160),
        _row("dspark", 2, tpot=52, output_tps=164),
        _row("dspark", 3, tpot=48, output_tps=156),
    ]
    paired, reference = ab.aggregate_results(rows)
    assert len(paired) == 1
    result = paired[0]
    assert result["status"] == "paired"
    assert result["tpot_speedup"] == pytest.approx(2.0)
    assert result["output_speedup"] == pytest.approx(2.0)
    assert result["dspark_accept_length"] == pytest.approx(2.6)
    assert "tokenspeed|ci|4k-1k-c16|dspark" in reference


def test_reference_check_catches_throughput_and_acceptance_regressions():
    current = {
        "tokenspeed|ci|4k-1k-c16|dspark": {
            "median_tpot_ms": 60.0,
            "output_throughput": 80.0,
            "total_token_throughput": 400.0,
            "accept_length": 2.0,
        }
    }
    baseline = {
        "tokenspeed|ci|4k-1k-c16|dspark": {
            "median_tpot_ms": 50.0,
            "output_throughput": 100.0,
            "total_token_throughput": 500.0,
            "accept_length": 2.6,
        }
    }
    check = ab.check_results(
        [],
        current,
        min_accept_length=1.1,
        baseline=baseline,
        reference_threshold=0.9,
    )
    assert check["passed"] is False
    assert any("output_throughput" in failure for failure in check["failures"])
    assert any("accept_length" in failure for failure in check["failures"])


def test_gpu_lock_refuses_a_second_holder(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    with ab.GpuLock(lock_path), pytest.raises(RuntimeError, match="another run holds"):
        with ab.GpuLock(lock_path):
            pass


def test_resume_only_keeps_results_with_ab_metadata(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"completed": 16}')
    assert ab._is_complete_result(result) is False
    result.write_text('{"ab_metadata": {"repeat": 1}, "completed": 16}')
    assert ab._is_complete_result(result) is True
    result.write_text('{"ab_metadata": {"repeat": 1}, "failed": 1}')
    assert ab._is_complete_result(result) is False
