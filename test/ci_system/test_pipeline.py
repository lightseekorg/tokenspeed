import re

import pytest
from pipeline import (
    PGM_RUNNER_PREFIXES,
    STALE_PROCESS_PATTERNS,
    build_step_summary_lines,
    check_perf_reference,
    extract_evalscope_score,
    extract_perf_summary_rows,
    runner_uses_pgm,
)


def test_stale_process_patterns_match_smg_router_proctitle():
    """`smg launch` rewrites its cmdline to `smg::router` via setproctitle;
    the cleanup list must still match after that, otherwise stale routers
    survive between runs and the next run hits port-bind conflicts."""
    sample_cmdlines = [
        "smg::router",
        "smg::router --worker-urls grpc://127.0.0.1:1234",
    ]
    for cmdline in sample_cmdlines:
        assert any(
            re.search(pat, cmdline) for pat in STALE_PROCESS_PATTERNS
        ), f"no STALE_PROCESS_PATTERNS entry matched cmdline: {cmdline!r}"


def test_stale_process_patterns_match_existing_targets():
    cmdlines = [
        "/usr/bin/python /usr/local/bin/ts serve --model foo",
        "/usr/bin/python -m smg launch --worker-urls grpc://127.0.0.1:1234",
        "/usr/bin/python -m smg_grpc_servicer.tokenspeed --host 127.0.0.1",
        "/usr/bin/python /repo/test/runtime/run_ci_suite.py --device cuda",
    ]
    for cmdline in cmdlines:
        assert any(
            re.search(pat, cmdline) for pat in STALE_PROCESS_PATTERNS
        ), f"no STALE_PROCESS_PATTERNS entry matched cmdline: {cmdline!r}"


@pytest.mark.parametrize(
    "runner,expected",
    [
        ("b200-1gpu", True),
        ("b200-4gpu", True),
        ("b300-4gpu", True),
        ("gb200-1gpu", True),
        ("h100-1gpu", False),
        ("h200-1gpu", False),
        ("linux-mi355-2gpu-lightseek", False),
        ("ubuntu-latest", False),
    ],
)
def test_runner_uses_pgm_covers_bare_metal_blackwell(runner, expected):
    """Self-hosted Blackwell pools (b200/b300/gb200) share the host's
    network namespace and process table — they need pgid-scoped
    cleanup so concurrent jobs on the same physical runner don't pkill
    each other's smg / engine / ts serve processes via the broad
    cmdline-pattern fallback."""
    assert runner_uses_pgm(runner) is expected


def test_pgm_runner_prefixes_are_kebab_terminated():
    """Trailing dash prevents `gb200` from accidentally matching a
    future `gb2000-*` runner label, etc."""
    for prefix in PGM_RUNNER_PREFIXES:
        assert prefix.endswith("-"), prefix


def test_extract_evalscope_score_from_pipe_table():
    report_table = """
| Model           | Dataset | Metric   | Subset  | Num | Score  | Cat.0   |
|-----------------|---------|----------|---------|-----|--------|---------|
| Kimi-K2.5-NVFP4 | aime25  | mean_acc | default | 30  | 0.9667 | default |
"""

    assert extract_evalscope_score(report_table) == 0.9667


def test_extract_evalscope_score_from_box_table():
    report_table = """
┌─────────────────┬───────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model           │ Dataset   │ Metric   │ Subset   │   Num │   Score │ Cat.0   │
├─────────────────┼───────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ Kimi-K2.5-NVFP4 │ aime25    │ mean_acc │ default  │    30 │  0.9667 │ default │
└─────────────────┴───────────┴──────────┴──────────┴───────┴─────────┴─────────┘
"""

    assert extract_evalscope_score(report_table) == 0.9667


PERF_CSV_FIXTURE = """\
some unrelated log line
config,Conc.,Latency (tps/user),Throughput (tps/gpu),Approx Cache Hit,Decoded Tok/Iter
attn_tp4_moe_tp4,1,40.0,2500.0,82.5,3.1
attn_tp4_moe_tp4,2,38.0,4500.0,82.5,3.1
attn_tp4_moe_tp4,4,35.0,8000.0,82.5,3.1
attn_tp4_moe_tp4,8,32.0,14000.0,82.5,3.1
attn_tp4_moe_tp4,16,30.0,24000.0,82.5,3.1

2026-05-08 12:00:00 - root - INFO - done
"""


def test_extract_perf_summary_rows_parses_csv_block():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    assert rows is not None
    assert len(rows) == 5
    assert rows[0]["Conc."] == "1"
    assert rows[-1]["Latency (tps/user)"] == "30.0"
    assert rows[-1]["Throughput (tps/gpu)"] == "24000.0"


def test_extract_perf_summary_rows_returns_none_when_missing():
    assert extract_perf_summary_rows("nothing relevant here") is None


def _command_results_with(rows):
    return [{"perf_summary_rows": rows}]


def test_check_perf_reference_passes_when_actual_meets_floor():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {
        "perf_threshold": 0.9,
        "perf_reference": {16: [33.0, 26000.0]},
    }
    result = check_perf_reference(task, _command_results_with(rows), ["perf"])
    assert result is not None
    assert result["passed"] is True
    assert result["failures"] == []


def test_check_perf_reference_fails_when_metric_below_floor():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {
        "perf_threshold": 0.9,
        "perf_reference": {16: [40.0, 26000.0]},
    }
    result = check_perf_reference(task, _command_results_with(rows), ["perf"])
    assert result is not None
    assert result["passed"] is False
    assert any("Latency (tps/user)" in f for f in result["failures"])


def test_check_perf_reference_reports_missing_row():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {"perf_reference": {64: [10.0, 100.0]}}
    result = check_perf_reference(task, _command_results_with(rows), ["perf"])
    assert result is not None
    assert result["passed"] is False
    assert any("no matching row" in f for f in result["failures"])


def test_check_perf_reference_skips_when_perf_stage_not_run():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {"perf_reference": {16: [40.0, 26000.0]}}
    assert check_perf_reference(task, _command_results_with(rows), ["server"]) is None


def test_check_perf_reference_returns_none_when_unconfigured():
    assert check_perf_reference({}, [], ["perf"]) is None


def test_check_perf_reference_raises_when_no_rows_found():
    task = {"perf_reference": {16: [40.0, 26000.0]}}
    with pytest.raises(ValueError, match="no perf summary rows"):
        check_perf_reference(task, [], ["perf"])


def test_check_perf_reference_raises_on_malformed_pair():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {"perf_reference": {16: [40.0]}}
    with pytest.raises(ValueError, match=r"\[tps_user, tps_gpu\]"):
        check_perf_reference(task, _command_results_with(rows), ["perf"])


def _base_result(**extras):
    base = {
        "ok": True,
        "task": "perf-task",
        "runner": "b200-4gpu",
        "executed_stages": ["server", "perf.install", "perf"],
        "targets": {},
        "command_results": [],
    }
    base.update(extras)
    return base


def test_step_summary_includes_perf_reference_pass():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {
        "perf_threshold": 0.9,
        "perf_reference": {16: [33.0, 26000.0]},
    }
    check = check_perf_reference(task, _command_results_with(rows), ["perf"])
    summary = "\n".join(
        build_step_summary_lines(_base_result(perf_reference_check=check))
    )
    assert "- Perf reference: `pass`" in summary
    assert "threshold `0.9`" in summary
    assert "1 concurrency levels" in summary


def test_step_summary_includes_perf_reference_failures():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {
        "perf_threshold": 0.9,
        "perf_reference": {16: [40.0, 26000.0]},
    }
    check = check_perf_reference(task, _command_results_with(rows), ["perf"])
    summary = "\n".join(
        build_step_summary_lines(_base_result(perf_reference_check=check))
    )
    assert "- Perf reference: `fail`" in summary
    assert "Latency (tps/user)" in summary


def test_step_summary_omits_perf_reference_when_unconfigured():
    summary = "\n".join(build_step_summary_lines(_base_result()))
    assert "Perf reference" not in summary
