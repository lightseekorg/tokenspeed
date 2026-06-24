import re
import textwrap
from pathlib import Path

import pytest
from pipeline import (
    STALE_PROCESS_PATTERNS,
    build_matrix,
    build_step_summary_lines,
    check_eval_score_threshold,
    check_perf_reference,
    extract_evalscope_score,
    extract_perf_summary_rows,
    format_perf_reference_markdown_table,
    format_perf_reference_table,
    is_amd_runner,
    is_gb200_runner,
    resolve_score_threshold_for_runner,
    should_run_nvidia_gpu_cleanup,
    validate_task,
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


def test_amd_runner_prefixes_cover_legacy_and_arc_labels():
    assert is_amd_runner("amd-mi35x-1gpu-test")
    assert is_amd_runner("amd-mi35x-4gpu-test")
    assert is_amd_runner("amd-mi355-1gpu-bench")
    assert is_amd_runner("amd-mi350-1gpu-bench")
    assert is_amd_runner("amd-mi350-4gpu-bench")
    assert not is_amd_runner("b200-1gpu")
    assert not is_amd_runner("gb200-4gpu-perf")


def test_nvidia_gpu_cleanup_runner_prefixes_cover_gb200_and_b300():
    assert is_gb200_runner("gb200-1gpu")
    assert is_gb200_runner("gb200-4gpu-perf")
    assert not is_gb200_runner("b300-4gpu")

    assert should_run_nvidia_gpu_cleanup("gb200-1gpu")
    assert should_run_nvidia_gpu_cleanup("gb200-4gpu-perf")
    assert should_run_nvidia_gpu_cleanup("b300-4gpu")
    assert not should_run_nvidia_gpu_cleanup("b200-4gpu")
    assert not should_run_nvidia_gpu_cleanup("h100-1gpu")
    assert not should_run_nvidia_gpu_cleanup("amd-mi35x-2gpu-test")
    assert not should_run_nvidia_gpu_cleanup("amd-mi355-1gpu-bench")
    assert not should_run_nvidia_gpu_cleanup("amd-mi350-1gpu-bench")


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


def test_resolve_score_threshold_passes_through_scalar():
    assert resolve_score_threshold_for_runner(0.7, "b200-2gpu") == 0.7


def test_resolve_score_threshold_passes_through_range_list():
    assert resolve_score_threshold_for_runner([0.6, 0.8], "b200-2gpu") == [0.6, 0.8]


def test_resolve_score_threshold_picks_per_runner_value():
    threshold = {"b200-2gpu": 0.7, "amd-mi35x-2gpu-test": 0.69}
    assert resolve_score_threshold_for_runner(threshold, "b200-2gpu") == 0.7
    assert resolve_score_threshold_for_runner(threshold, "amd-mi35x-2gpu-test") == 0.69


def test_resolve_score_threshold_returns_none_for_unknown_runner():
    threshold = {"b200-2gpu": 0.7}
    assert resolve_score_threshold_for_runner(threshold, "h100-2gpu") is None


def _eval_command_results(score):
    return [{"stage": "eval", "evalscope_score": score}]


def test_check_eval_score_threshold_uses_per_runner_mapping_pass():
    task = {
        "score_threshold": {
            "b200-2gpu": 0.7,
            "amd-mi35x-2gpu-test": 0.69,
        }
    }
    check = check_eval_score_threshold(
        task, _eval_command_results(0.695), ["eval"], "amd-mi35x-2gpu-test"
    )
    assert check is not None
    assert check["passed"] is True
    assert check["min"] == 0.69


def test_check_eval_score_threshold_uses_per_runner_mapping_fail():
    task = {
        "score_threshold": {
            "b200-2gpu": 0.7,
            "amd-mi35x-2gpu-test": 0.69,
        }
    }
    check = check_eval_score_threshold(
        task, _eval_command_results(0.695), ["eval"], "b200-2gpu"
    )
    assert check is not None
    assert check["passed"] is False
    assert check["min"] == 0.7


def test_check_eval_score_threshold_skips_runner_without_mapping_entry():
    task = {"score_threshold": {"b200-2gpu": 0.7}}
    assert (
        check_eval_score_threshold(
            task, _eval_command_results(0.5), ["eval"], "h100-2gpu"
        )
        is None
    )


def test_check_eval_score_threshold_still_supports_scalar():
    task = {"score_threshold": 0.7}
    check = check_eval_score_threshold(
        task, _eval_command_results(0.71), ["eval"], "b200-2gpu"
    )
    assert check is not None
    assert check["passed"] is True
    assert check["min"] == 0.7


def _write_task_yaml(tmp_path: Path, filename: str, body: str) -> Path:
    path = tmp_path / filename
    path.write_text(textwrap.dedent(body).lstrip())
    return path


_DEFAULT_BODY_TEMPLATE = """\
api_version: ci.tokenspeed.io/v1
name: {name}
type: ut
triggers:
  - per-commit
runner:
  labels:
{labels}
"""


def _default_body(name: str, labels: list[str], extra: str = "") -> str:
    label_block = "\n".join(f"    - {label}" for label in labels)
    body = _DEFAULT_BODY_TEMPLATE.format(name=name, labels=label_block)
    if extra:
        body += extra
    return body


def test_validate_task_accepts_known_priorities(tmp_path):
    for priority in ("low", "normal", "high"):
        body = _default_body("ut-a", ["b300-1gpu"], extra=f"priority: {priority}\n")
        path = _write_task_yaml(tmp_path, f"{priority}.yaml", body)
        import yaml as _yaml

        validate_task(_yaml.safe_load(path.read_text()), path)


def test_validate_task_rejects_unknown_priority(tmp_path):
    body = _default_body("ut-a", ["b300-1gpu"], extra="priority: urgent\n")
    path = _write_task_yaml(tmp_path, "bad.yaml", body)
    import yaml as _yaml

    with pytest.raises(ValueError, match=r"priority must be one of"):
        validate_task(_yaml.safe_load(path.read_text()), path)


def test_build_matrix_default_priority_preserves_existing_order(tmp_path):
    # Two tasks; both omit `priority`. Order must match the existing
    # behaviour: alphabetical by file path, then label order from the yaml.
    _write_task_yaml(
        tmp_path,
        "a-first.yaml",
        _default_body("ut-a", ["b300-1gpu", "h100-1gpu"]),
    )
    _write_task_yaml(
        tmp_path,
        "b-second.yaml",
        _default_body("ut-b", ["b200-1gpu"]),
    )
    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")
    assert [(e["name"], e["runner"]) for e in matrix["include"]] == [
        ("ut-a", "b300-1gpu"),
        ("ut-a", "h100-1gpu"),
        ("ut-b", "b200-1gpu"),
    ]
    assert all(e["priority"] == "normal" for e in matrix["include"])


def test_build_matrix_sorts_high_priority_before_low(tmp_path):
    # b300-4gpu evals are marked `high`, the b300-1gpu unit-test stays
    # default (normal). After the sort the heavy 4gpu jobs land at the
    # head of the include list and GitHub Actions dispatches them first.
    _write_task_yaml(
        tmp_path,
        "eval-heavy.yaml",
        _default_body("eval-heavy", ["b300-4gpu"], extra="priority: high\n"),
    )
    _write_task_yaml(
        tmp_path,
        "ut-kernel.yaml",
        _default_body("ut-kernel", ["b300-1gpu"]),
    )
    _write_task_yaml(
        tmp_path,
        "ut-flaky.yaml",
        _default_body("ut-flaky", ["b300-1gpu"], extra="priority: low\n"),
    )
    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")
    assert [e["name"] for e in matrix["include"]] == [
        "eval-heavy",
        "ut-kernel",
        "ut-flaky",
    ]


def test_validate_task_accepts_per_label_priority_dict(tmp_path):
    body = _default_body(
        "ut-a",
        ["b300-1gpu", "h100-1gpu"],
        extra="priority:\n  b300-1gpu: low\n",
    )
    path = _write_task_yaml(tmp_path, "per-label.yaml", body)
    import yaml as _yaml

    validate_task(_yaml.safe_load(path.read_text()), path)


def test_validate_task_rejects_per_label_priority_with_unknown_label(tmp_path):
    body = _default_body(
        "ut-a",
        ["b300-1gpu"],
        extra="priority:\n  h100-1gpu: low\n",
    )
    path = _write_task_yaml(tmp_path, "unknown.yaml", body)
    import yaml as _yaml

    with pytest.raises(ValueError, match=r"priority contains unknown labels"):
        validate_task(_yaml.safe_load(path.read_text()), path)


def test_validate_task_rejects_per_label_priority_with_unknown_value(tmp_path):
    body = _default_body(
        "ut-a",
        ["b300-1gpu"],
        extra="priority:\n  b300-1gpu: urgent\n",
    )
    path = _write_task_yaml(tmp_path, "bad-value.yaml", body)
    import yaml as _yaml

    with pytest.raises(ValueError, match=r"priority values must each be one of"):
        validate_task(_yaml.safe_load(path.read_text()), path)


def test_build_matrix_per_label_priority_only_affects_listed_label(tmp_path):
    # `priority: { b300-1gpu: low }` lowers only the b300-1gpu instance.
    # The same task running on h100-1gpu / b200-1gpu stays at default
    # `normal`, so the heavy 4gpu eval still leads, then both default
    # labels of the kernel ut, then the b300-1gpu kernel ut last.
    _write_task_yaml(
        tmp_path,
        "eval-heavy.yaml",
        _default_body("eval-heavy", ["b300-4gpu"]),
    )
    _write_task_yaml(
        tmp_path,
        "ut-kernel.yaml",
        _default_body(
            "ut-kernel",
            ["h100-1gpu", "b300-1gpu", "b200-1gpu"],
            extra="priority:\n  b300-1gpu: low\n",
        ),
    )
    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")
    assert [(e["name"], e["runner"], e["priority"]) for e in matrix["include"]] == [
        ("eval-heavy", "b300-4gpu", "normal"),
        ("ut-kernel", "h100-1gpu", "normal"),
        ("ut-kernel", "b200-1gpu", "normal"),
        ("ut-kernel", "b300-1gpu", "low"),
    ]


def test_build_matrix_sort_is_stable_within_priority(tmp_path):
    # Same priority across both files: alphabetical file order plus
    # within-file label order must be preserved.
    _write_task_yaml(
        tmp_path,
        "a.yaml",
        _default_body("a", ["b300-4gpu", "b200-4gpu"], extra="priority: high\n"),
    )
    _write_task_yaml(
        tmp_path,
        "b.yaml",
        _default_body("b", ["gb200-4gpu"], extra="priority: high\n"),
    )
    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")
    assert [(e["name"], e["runner"]) for e in matrix["include"]] == [
        ("a", "b300-4gpu"),
        ("a", "b200-4gpu"),
        ("b", "gb200-4gpu"),
    ]


def _checks_fixture():
    def mk(conc, la, lr, ta, tr, threshold=0.95):
        return {
            "conc": conc,
            "Latency (tps/user)": {
                "actual": la,
                "ref": lr,
                "floor": lr * threshold,
                "passed": la >= lr * threshold,
            },
            "Throughput (tps/gpu)": {
                "actual": ta,
                "ref": tr,
                "floor": tr * threshold,
                "passed": ta >= tr * threshold,
            },
        }

    return [
        mk(1, 446.43, 423.21, 10014.97, 9679.21),
        mk(2, 315.46, 312.51, 14877.08, 14635.51),
        mk(16, 76.63, 78.31, 29807.71, 30845.64),
    ]


def test_format_perf_reference_table_columns_and_pct():
    lines = format_perf_reference_table(_checks_fixture())
    header, rule, *body = lines
    assert "Conc" in header
    assert "Lat actual" in header
    assert "Lat ref" in header
    assert "Lat floor" in header
    # Header makes the comparison base explicit so readers do not have to
    # guess whether the percentage is against `ref` or the threshold floor.
    assert "Lat actual/ref" in header
    assert "Thru actual" in header
    assert "Thru ref" in header
    assert "Thru floor" in header
    assert "Thru actual/ref" in header
    assert set(rule) == {"-"}
    assert len(body) == 3
    assert "446.43" in body[0]  # actual
    assert "423.21" in body[0]  # ref
    assert "402.05" in body[0]  # floor = 423.21 * 0.95
    # 446.43 / 423.21 = 1.0549... -> 105.5%
    assert "105.5%" in body[0]
    # 76.63 / 78.31 = 0.9785... -> 97.9% (below 100%, sanity)
    assert "97.9%" in body[2]


def test_format_perf_reference_table_empty_when_no_checks():
    assert format_perf_reference_table([]) == []


def test_format_perf_reference_markdown_table_has_header_and_alignment():
    lines = format_perf_reference_markdown_table(_checks_fixture())
    assert lines[0].startswith("| Conc |")
    assert "Lat ref" in lines[0]
    assert "Lat floor" in lines[0]
    assert "Lat actual/ref" in lines[0]
    assert "Thru ref" in lines[0]
    assert "Thru floor" in lines[0]
    assert "Thru actual/ref" in lines[0]
    # Alignment row: all-right-aligned (`---:`)
    assert "---:" in lines[1]
    # Body rows
    assert lines[2].startswith("| 1 |")
    assert "446.43" in lines[2]  # actual
    assert "423.21" in lines[2]  # ref
    assert "402.05" in lines[2]  # floor
    assert "105.5%" in lines[2]
    assert "97.9%" in lines[-1]


def test_format_perf_reference_markdown_table_empty_when_no_checks():
    assert format_perf_reference_markdown_table([]) == []


def test_step_summary_embeds_perf_reference_table():
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {
        "perf_threshold": 0.9,
        "perf_reference": {16: [33.0, 26000.0]},
    }
    check = check_perf_reference(task, _command_results_with(rows), ["perf"])
    summary = "\n".join(
        build_step_summary_lines(_base_result(perf_reference_check=check))
    )
    # Comparison table interleaved so a passing run still shows actual,
    # raw ref (non-threshold), threshold-adjusted floor, and actual/ref %.
    assert "| Conc | Lat actual | Lat ref | Lat floor | Lat actual/ref" in summary
    assert "Thru floor" in summary
    assert "Thru actual/ref" in summary
    assert "| 16 |" in summary
    assert "%" in summary


def test_perf_reference_table_rendered_for_passing_check(capsys):
    rows = extract_perf_summary_rows(PERF_CSV_FIXTURE)
    task = {
        "perf_threshold": 0.9,
        "perf_reference": {16: [33.0, 26000.0]},
    }
    check_perf_reference(task, _command_results_with(rows), ["perf"])
    out = capsys.readouterr().out
    # Even when status=passed, the per-conc comparison table is now printed
    # to stdout (previously only failures were detailed).
    assert "[perf-ref] threshold=0.9, status=passed" in out
    assert "[perf-ref]   Conc" in out
    assert "[perf-ref]   ---" in out
    assert "%" in out
