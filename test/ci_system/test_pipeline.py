import importlib.util
import json
import re
import textwrap
from pathlib import Path

import pipeline as pipeline_module
import pytest
from pipeline import (
    STALE_PROCESS_PATTERNS,
    build_matrix,
    build_step_summary_lines,
    check_eval_score_threshold,
    check_perf_reference,
    check_server_log_patterns,
    extract_evalscope_score,
    extract_perf_summary_rows,
    format_perf_reference_markdown_table,
    format_perf_reference_table,
    get_excluded_runner_labels,
    get_runner_specific_env,
    is_amd_runner,
    is_gb200_runner,
    is_nvidia_arm_runner,
    normalize_task,
    normalize_task_names,
    resolve_score_threshold_for_runner,
    runner_matches_group,
    should_run_nvidia_gpu_cleanup,
    summarize_command_output,
    validate_task,
)

RANDOM_COLLECTOR_PATH = (
    Path(__file__).parents[1] / "random_benchmark/tokenspeed/collect_outputs.py"
)
RANDOM_COLLECTOR_SPEC = importlib.util.spec_from_file_location(
    "ci_random_benchmark_collect_outputs", RANDOM_COLLECTOR_PATH
)
assert RANDOM_COLLECTOR_SPEC is not None and RANDOM_COLLECTOR_SPEC.loader is not None
random_collect_outputs = importlib.util.module_from_spec(RANDOM_COLLECTOR_SPEC)
RANDOM_COLLECTOR_SPEC.loader.exec_module(random_collect_outputs)

REPO_ROOT = Path(__file__).parents[2]
PR_TEST_WORKFLOWS = (
    ".github/workflows/pr-test-nvidia.yml",
    ".github/workflows/pr-test-nvidia-arm.yml",
    ".github/workflows/pr-test-amd.yml",
)


@pytest.mark.parametrize("workflow_path", PR_TEST_WORKFLOWS)
def test_pr_workflows_use_matrix_timeout_and_upload_all_ci_artifacts(workflow_path):
    import yaml as _yaml

    workflow = _yaml.safe_load((REPO_ROOT / workflow_path).read_text())
    job = workflow["jobs"]["unit-test"]
    upload_step = next(
        step
        for step in job["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )

    assert job["timeout-minutes"] == "${{ matrix.timeout_minutes }}"
    assert upload_step["with"]["path"] == "${{ env.WORK_DIR }}/.ci-artifacts/"
    assert upload_step["with"]["include-hidden-files"] is True


@pytest.mark.parametrize("workflow_path", PR_TEST_WORKFLOWS)
def test_pr_workflows_require_fail_closed_manual_task_names(workflow_path):
    import yaml as _yaml

    workflow = _yaml.safe_load((REPO_ROOT / workflow_path).read_text())
    # PyYAML 1.1 parses the unquoted workflow key `on` as boolean true.
    events = workflow.get("on", workflow.get(True))
    task_names_input = events["workflow_dispatch"]["inputs"]["task_names"]
    build_step = next(
        step
        for step in workflow["jobs"]["scan"]["steps"]
        if step.get("name") == "Build task matrix"
    )

    assert task_names_input["required"] is True
    assert task_names_input["type"] == "string"
    assert (
        build_step["env"]["CI_SELECTED_TASK_NAMES"]
        == "${{ github.event.inputs.task_names }}"
    )
    assert 'if [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]]' in build_step["run"]
    assert (
        'TASK_FILTER_ARGS=(--task-names "$CI_SELECTED_TASK_NAMES")' in build_step["run"]
    )
    assert '"${TASK_FILTER_ARGS[@]}"' in build_step["run"]


def test_minimax_m3_exact_longctx_locks_known_output_token():
    import yaml as _yaml
    from minimax_m3_exact_longctx import EXPECTED_OUTPUT_IDS

    task_path = REPO_ROOT / "test/ci/perf/minimax-m3-mxfp8-exact-longctx.yaml"
    task = _yaml.safe_load(task_path.read_text())

    assert (
        "python3 test/ci_system/minimax_m3_exact_longctx.py" in task["perf"]["command"]
    )
    assert EXPECTED_OUTPUT_IDS == (123,)


def test_server_log_check_passes_without_forbidden_matches(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "server ready\nTraceback during shutdown\nasyncio.exceptions.CancelledError\n"
    )
    task = {
        "server": {
            "forbidden_log_patterns": [
                r"CUDA out of memory",
                r"NCCL.*(?:abort|unhandled)",
            ]
        }
    }

    check = check_server_log_patterns(task, ["server", "perf"], log_path)

    assert check is not None
    assert check["passed"] is True
    assert check["match_count"] == 0
    assert check["matches"] == []
    assert check["omitted_match_count"] == 0


def test_server_log_check_caps_match_details_and_line_length(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "".join(f"CUDA out of memory {i} " + "x" * 600 + "\n" for i in range(25))
    )
    task = {"server": {"forbidden_log_patterns": [r"CUDA out of memory"]}}

    check = check_server_log_patterns(task, ["server"], log_path)

    assert check is not None
    assert check["passed"] is False
    assert check["match_count"] == 25
    assert len(check["matches"]) == 20
    assert check["omitted_match_count"] == 5
    assert all(len(match["line"]) <= 500 for match in check["matches"])


def test_server_log_check_fails_when_executed_server_log_is_missing(tmp_path):
    missing_path = tmp_path / "missing.log"
    task = {"server": {"forbidden_log_patterns": [r"fatal"]}}

    check = check_server_log_patterns(task, ["server"], missing_path)

    assert check is not None
    assert check["passed"] is False
    assert check["match_count"] == 0
    assert "server log is missing" in check["error"]


@pytest.mark.parametrize(
    ("stages_run", "dry_run"),
    [
        (["install"], False),
        (["server"], True),
    ],
)
def test_server_log_check_skips_install_only_and_dry_run(tmp_path, stages_run, dry_run):
    task = {"server": {"forbidden_log_patterns": [r"fatal"]}}

    check = check_server_log_patterns(
        task,
        stages_run,
        tmp_path / "missing.log",
        dry_run=dry_run,
    )

    assert check is None


def test_execute_task_scans_after_cleanup_and_preserves_workload_error(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(textwrap.dedent("""\
            api_version: ci.tokenspeed.io/v1
            name: perf-server-log-order
            type: perf
            timeout_minutes: 75
            triggers:
              - manual
            runner:
              labels:
                - h100-1gpu
            server:
              command: fake-server
              forbidden_log_patterns:
                - CUDA out of memory
              ready:
                url: http://127.0.0.1:8000/readiness
            perf:
              command: fake-workload
            """))
    result_path = tmp_path / "result.json"
    server_log_path = tmp_path / ".ci-artifacts" / "server.log"
    events = []

    monkeypatch.setattr(
        pipeline_module,
        "setup_runner",
        lambda *args, **kwargs: ({}, None),
    )
    monkeypatch.setattr(
        pipeline_module, "kill_ready_port_listener", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline_module, "start_server", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(pipeline_module, "poll_readiness", lambda *args, **kwargs: None)

    def fail_workload(command, **kwargs):
        events.append("workload")
        raise RuntimeError("workload boom")

    def stop_server(_process):
        events.append("stop")
        with server_log_path.open("a") as handle:
            handle.write("CUDA out of memory during shutdown\n")

    def cleanup_runner(*args, **kwargs):
        events.append("cleanup")

    original_check = check_server_log_patterns

    def tracked_check(task, stages_run, log_path, *, dry_run=False):
        events.append("scan")
        return original_check(
            task,
            stages_run,
            log_path,
            dry_run=dry_run,
        )

    monkeypatch.setattr(pipeline_module, "shell_run", fail_workload)
    monkeypatch.setattr(pipeline_module, "stop_server", stop_server)
    monkeypatch.setattr(pipeline_module, "cleanup_runner", cleanup_runner)
    monkeypatch.setattr(pipeline_module, "check_server_log_patterns", tracked_check)

    returncode = pipeline_module.execute_task(
        config=config_path.name,
        runner="h100-1gpu",
        work_dir=str(tmp_path),
        dry_run=False,
        print_plan=False,
        result_json=str(result_path),
    )

    result = json.loads(result_path.read_text())
    assert returncode == 1
    assert events == ["workload", "stop", "cleanup", "scan"]
    assert result["error"] == "workload boom"
    assert result["config"] == config_path.name
    assert result["timeout_minutes"] == 75
    assert result["server_log_check"]["passed"] is False
    assert result["server_log_check"]["match_count"] == 1


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


def test_nvidia_runner_groups_split_arm_from_x86():
    assert is_nvidia_arm_runner("gb200-1gpu")
    assert not is_nvidia_arm_runner("b200-1gpu")
    assert not is_nvidia_arm_runner("amd-mi35x-1gpu-test")

    assert runner_matches_group("gb200-1gpu", "nvidia")
    assert runner_matches_group("gb200-1gpu", "nvidia-arm")
    assert not runner_matches_group("gb200-1gpu", "nvidia-x86")
    assert runner_matches_group("b200-1gpu", "nvidia-x86")
    assert runner_matches_group("b300-4gpu", "nvidia-x86")
    assert not runner_matches_group("amd-mi35x-1gpu-test", "nvidia-arm")
    assert not runner_matches_group("amd-mi35x-1gpu-test", "nvidia-x86")


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


def test_runner_specific_env_uses_original_label_after_b200_override(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_B200_RUNNER_LABEL", "b200v2")
    task = {
        "runner": {
            "labels": ["b200-2gpu"],
            "env": {
                "b200-2gpu": {
                    "GPT_OSS_EVAL_MODEL": "openai/gpt-oss-120b",
                },
            },
        },
    }

    assert get_runner_specific_env(task, "b200v2-2gpu") == {
        "GPT_OSS_EVAL_MODEL": "openai/gpt-oss-120b",
    }


def test_runner_specific_env_prefers_exact_label(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_B200_RUNNER_LABEL", "b200v2")
    task = {
        "runner": {
            "labels": ["b200-2gpu", "b200v2-2gpu"],
            "env": {
                "b200-2gpu": {"MODEL": "original"},
                "b200v2-2gpu": {"MODEL": "exact"},
            },
        },
    }

    assert get_runner_specific_env(task, "b200v2-2gpu") == {"MODEL": "exact"}


def test_excluded_runner_labels_parse_comma_separated_terms(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", " B300, mi355, ,")
    assert get_excluded_runner_labels() == ["b300", "mi355"]

    monkeypatch.setenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", " , , ")
    assert get_excluded_runner_labels() == []

    monkeypatch.delenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS")
    assert get_excluded_runner_labels() == []


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

PERF_MULTI_CONFIG_CSV_FIXTURE = """\
config,Conc.,Latency (tps/user),Throughput (tps/gpu),Approx Cache Hit,Decoded Tok/Iter
input_1k,16,100.0,1000.0,80.0,1.0
input_4k,16,50.0,500.0,70.0,1.0
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


def test_random_collector_output_is_parsed_and_gated_by_config(capsys):
    collector_rows = [
        {
            "config": "input_1k",
            "Conc.": 16,
            "Latency (tps/user)": 100.0,
            "Throughput (tps/gpu)": 1000.0,
            "Approx Cache Hit": 80.0,
            "Decoded Tok/Iter": 1.0,
        },
        {
            "config": "input_32k",
            "Conc.": 16,
            "Latency (tps/user)": 25.0,
            "Throughput (tps/gpu)": 250.0,
            "Approx Cache Hit": 60.0,
            "Decoded Tok/Iter": 1.0,
        },
    ]
    random_collect_outputs.print_table(collector_rows)
    output = capsys.readouterr().out

    command_summary = summarize_command_output("collect random sweep", output)
    parsed_rows = command_summary["perf_summary_rows"]
    assert [(row["config"], row["Conc."]) for row in parsed_rows] == [
        ("input_1k", "16"),
        ("input_32k", "16"),
    ]

    task = {
        "perf_threshold": 1.0,
        "perf_reference": {
            "input_1k": {16: [100.0, 1000.0]},
            "input_32k": {16: [25.0, 250.0]},
        },
    }
    result = check_perf_reference(task, [command_summary], ["perf"])
    assert result is not None
    assert result["passed"] is True


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


def test_check_perf_reference_matches_config_and_concurrency():
    rows = extract_perf_summary_rows(PERF_MULTI_CONFIG_CSV_FIXTURE)
    task = {
        "perf_threshold": 1.0,
        "perf_reference": {
            "input_1k": {16: [100.0, 1000.0]},
            "input_4k": {16: [50.0, 500.0]},
        },
    }

    result = check_perf_reference(task, _command_results_with(rows), ["perf"])

    assert result is not None
    assert result["passed"] is True
    assert [check["config"] for check in result["checks"]] == [
        "input_1k",
        "input_4k",
    ]


def test_check_perf_reference_config_failure_identifies_reference_point():
    rows = extract_perf_summary_rows(PERF_MULTI_CONFIG_CSV_FIXTURE)
    task = {
        "perf_threshold": 1.0,
        "perf_reference": {"input_4k": {16: [60.0, 500.0]}},
    }

    result = check_perf_reference(task, _command_results_with(rows), ["perf"])

    assert result is not None
    assert result["passed"] is False
    assert any("config=input_4k, conc=16" in failure for failure in result["failures"])


def test_check_perf_reference_reports_missing_config_row():
    rows = extract_perf_summary_rows(PERF_MULTI_CONFIG_CSV_FIXTURE)
    task = {"perf_reference": {"input_8k": {16: [10.0, 100.0]}}}

    result = check_perf_reference(task, _command_results_with(rows), ["perf"])

    assert result is not None
    assert result["passed"] is False
    assert result["failures"] == [
        "config=input_8k, conc=16: no matching row in perf summary"
    ]


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


def test_check_perf_reference_raises_on_malformed_config_pair():
    rows = extract_perf_summary_rows(PERF_MULTI_CONFIG_CSV_FIXTURE)
    task = {"perf_reference": {"input_1k": {16: [40.0]}}}
    with pytest.raises(ValueError, match=r"\[tps_user, tps_gpu\]"):
        check_perf_reference(task, _command_results_with(rows), ["perf"])


def _base_result(**extras):
    base = {
        "ok": True,
        "task": "perf-task",
        "config": "test/ci/perf/perf-task.yaml",
        "runner": "b200-4gpu",
        "timeout_minutes": 60,
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


def test_normalize_task_defaults_timeout_minutes_to_60(tmp_path):
    path = _write_task_yaml(
        tmp_path,
        "default-timeout.yaml",
        _default_body("ut-default-timeout", ["b300-1gpu"]),
    )

    task = normalize_task(path, tmp_path)

    assert task["timeout_minutes"] == 60


@pytest.mark.parametrize("name", ["", " task-a", "task-a ", "task,a", ["task-a"]])
def test_validate_task_rejects_name_incompatible_with_exact_filter(tmp_path, name):
    import yaml as _yaml

    path = tmp_path / "invalid-name.yaml"
    task = _yaml.safe_load(_default_body("task-a", ["b300-1gpu"]))
    task["name"] = name

    with pytest.raises(ValueError, match=r"name must be a non-empty string"):
        validate_task(task, path)


@pytest.mark.parametrize("timeout_minutes", [1, 60, 360])
def test_validate_task_accepts_timeout_minutes_bounds(tmp_path, timeout_minutes):
    import yaml as _yaml

    path = tmp_path / "timeout.yaml"
    task = _yaml.safe_load(_default_body("ut-timeout", ["b300-1gpu"]))
    task["timeout_minutes"] = timeout_minutes

    validate_task(task, path)


@pytest.mark.parametrize(
    "timeout_minutes",
    [True, False, 0, 361, 60.0, "60", None],
)
def test_validate_task_rejects_invalid_timeout_minutes(tmp_path, timeout_minutes):
    import yaml as _yaml

    path = tmp_path / "invalid-timeout.yaml"
    task = _yaml.safe_load(_default_body("ut-timeout", ["b300-1gpu"]))
    task["timeout_minutes"] = timeout_minutes

    with pytest.raises(ValueError, match=r"timeout_minutes must be an integer"):
        validate_task(task, path)


def test_build_matrix_emits_default_and_explicit_timeout_minutes(tmp_path):
    _write_task_yaml(
        tmp_path,
        "a-default.yaml",
        _default_body("ut-default", ["b300-1gpu"]),
    )
    _write_task_yaml(
        tmp_path,
        "b-explicit.yaml",
        _default_body(
            "ut-explicit",
            ["b300-1gpu"],
            extra="timeout_minutes: 120\n",
        ),
    )

    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")

    assert {entry["name"]: entry["timeout_minutes"] for entry in matrix["include"]} == {
        "ut-default": 60,
        "ut-explicit": 120,
    }


def test_scan_task_names_parser_normalizes_exact_comma_separated_names():
    args = pipeline_module.parse_args(["scan", "--task-names", " task-a,task-b "])

    assert args.task_names == ["task-a", "task-b"]


@pytest.mark.parametrize(
    "task_names",
    [[], [""], ["task-a", ""], ["task-a", "task-a"]],
)
def test_normalize_task_names_rejects_empty_or_duplicate_selection(task_names):
    with pytest.raises(ValueError):
        normalize_task_names(task_names)


def test_build_matrix_filters_exact_task_names(tmp_path):
    _write_task_yaml(
        tmp_path,
        "a.yaml",
        _default_body("task-a", ["b200-1gpu"]),
    )
    _write_task_yaml(
        tmp_path,
        "b.yaml",
        _default_body("task-b", ["b200-1gpu"]),
    )

    matrix = build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        task_names=["task-b"],
    )

    assert [entry["name"] for entry in matrix["include"]] == ["task-b"]


def test_build_matrix_rejects_any_unknown_selected_task(tmp_path):
    _write_task_yaml(
        tmp_path,
        "known.yaml",
        _default_body("known", ["b200-1gpu"]),
    )

    with pytest.raises(ValueError, match=r"unknown task names: \['missing'\]"):
        build_matrix(
            tmp_path,
            tmp_path,
            trigger="per-commit",
            task_names=["known", "missing"],
        )


def test_build_matrix_rejects_selected_task_without_requested_trigger(tmp_path):
    _write_task_yaml(
        tmp_path,
        "known.yaml",
        _default_body("known", ["b200-1gpu"]),
    )

    with pytest.raises(ValueError, match=r"do not support trigger 'manual'"):
        build_matrix(
            tmp_path,
            tmp_path,
            trigger="manual",
            task_names=["known"],
        )


def test_build_matrix_rejects_partial_runner_group_match(tmp_path):
    _write_task_yaml(
        tmp_path,
        "amd.yaml",
        _default_body("amd", ["amd-mi35x-1gpu-test"]),
    )
    _write_task_yaml(
        tmp_path,
        "nvidia.yaml",
        _default_body("nvidia", ["b200-1gpu"]),
    )

    with pytest.raises(
        ValueError,
        match=r"no matrix entries.*\['amd'\]",
    ):
        build_matrix(
            tmp_path,
            tmp_path,
            trigger="per-commit",
            runner_group="nvidia-x86",
            task_names=["nvidia", "amd"],
        )


def test_build_matrix_rejects_selected_task_excluded_from_all_runners(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", "b200")
    _write_task_yaml(
        tmp_path,
        "known.yaml",
        _default_body("known", ["b200-1gpu"]),
    )

    with pytest.raises(ValueError, match=r"no matrix entries.*\['known'\]"):
        build_matrix(
            tmp_path,
            tmp_path,
            trigger="per-commit",
            task_names=["known"],
        )


def test_scan_main_returns_nonzero_for_unknown_selected_task(tmp_path, capsys):
    _write_task_yaml(
        tmp_path,
        "known.yaml",
        _default_body("known", ["b200-1gpu"]),
    )

    returncode = pipeline_module.main(
        [
            "scan",
            "--repo-root",
            str(tmp_path),
            "--root",
            ".",
            "--trigger",
            "per-commit",
            "--task-names",
            "missing",
        ]
    )

    captured = capsys.readouterr()
    assert returncode == 2
    assert captured.out == ""
    assert "unknown task names: ['missing']" in captured.err


def test_execute_task_print_plan_includes_config_and_timeout(tmp_path, capsys):
    config_path = _write_task_yaml(
        tmp_path,
        "plan.yaml",
        _default_body(
            "ut-plan",
            ["b300-1gpu"],
            extra="timeout_minutes: 90\n",
        ),
    )

    returncode = pipeline_module.execute_task(
        config=config_path.name,
        runner="b300-1gpu",
        work_dir=str(tmp_path),
        dry_run=True,
        print_plan=True,
        result_json=None,
    )

    plan = json.loads(capsys.readouterr().out)
    assert returncode == 0
    assert plan["config"] == config_path.name
    assert plan["timeout_minutes"] == 90


def test_validate_task_accepts_compilable_forbidden_log_patterns(tmp_path):
    import yaml as _yaml

    path = tmp_path / "patterns.yaml"
    task = _yaml.safe_load(_default_body("ut-patterns", ["b300-1gpu"]))
    task["server"] = {
        "forbidden_log_patterns": [
            r"CUDA out of memory",
            r"NCCL.*(?:abort|unhandled)",
        ]
    }

    validate_task(task, path)


@pytest.mark.parametrize(
    "patterns",
    ["fatal", [1], [""]],
)
def test_validate_task_rejects_malformed_forbidden_log_patterns(tmp_path, patterns):
    import yaml as _yaml

    path = tmp_path / "invalid-patterns.yaml"
    task = _yaml.safe_load(_default_body("ut-patterns", ["b300-1gpu"]))
    task["server"] = {"forbidden_log_patterns": patterns}

    with pytest.raises(ValueError, match=r"server\.forbidden_log_patterns"):
        validate_task(task, path)


def test_validate_task_rejects_invalid_forbidden_log_regex(tmp_path):
    import yaml as _yaml

    path = tmp_path / "invalid-regex.yaml"
    task = _yaml.safe_load(_default_body("ut-patterns", ["b300-1gpu"]))
    task["server"] = {"forbidden_log_patterns": ["["]}

    with pytest.raises(ValueError, match=r"is not a valid regex"):
        validate_task(task, path)


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


def test_validate_task_accepts_boolean_optional(tmp_path):
    body = _default_body("ut-a", ["b300-1gpu"], extra="optional: true\n")
    path = _write_task_yaml(tmp_path, "optional.yaml", body)
    import yaml as _yaml

    validate_task(_yaml.safe_load(path.read_text()), path)


def test_validate_task_rejects_non_boolean_optional(tmp_path):
    body = _default_body("ut-a", ["b300-1gpu"], extra="optional: flaky\n")
    path = _write_task_yaml(tmp_path, "bad-optional.yaml", body)
    import yaml as _yaml

    with pytest.raises(ValueError, match=r"optional must be a boolean"):
        validate_task(_yaml.safe_load(path.read_text()), path)


def test_validate_task_accepts_per_label_optional_dict(tmp_path):
    body = _default_body(
        "ut-a",
        ["b300-1gpu", "h100-1gpu"],
        extra="optional:\n  b300-1gpu: true\n",
    )
    path = _write_task_yaml(tmp_path, "per-label-optional.yaml", body)
    import yaml as _yaml

    validate_task(_yaml.safe_load(path.read_text()), path)


def test_validate_task_rejects_per_label_optional_with_unknown_label(tmp_path):
    body = _default_body(
        "ut-a",
        ["b300-1gpu"],
        extra="optional:\n  h100-1gpu: true\n",
    )
    path = _write_task_yaml(tmp_path, "unknown-optional.yaml", body)
    import yaml as _yaml

    with pytest.raises(ValueError, match=r"optional contains unknown labels"):
        validate_task(_yaml.safe_load(path.read_text()), path)


def test_validate_task_rejects_per_label_optional_with_non_boolean_value(tmp_path):
    body = _default_body(
        "ut-a",
        ["b300-1gpu"],
        extra="optional:\n  b300-1gpu: flaky\n",
    )
    path = _write_task_yaml(tmp_path, "bad-optional-value.yaml", body)
    import yaml as _yaml

    with pytest.raises(ValueError, match=r"optional values must be booleans"):
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
    assert all(e["optional"] is False for e in matrix["include"])


def test_build_matrix_excludes_runner_label_substrings_case_insensitively(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", " B300, mi355, ,")
    _write_task_yaml(
        tmp_path,
        "mixed.yaml",
        _default_body(
            "mixed",
            [
                "b300-1gpu",
                "gb300-4gpu",
                "amd-mi355-1gpu-bench",
                "h100-1gpu",
            ],
        ),
    )

    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")

    assert [entry["runner"] for entry in matrix["include"]] == ["h100-1gpu"]


def test_build_matrix_excludes_resolved_runner_label(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKENSPEED_B200_RUNNER_LABEL", "blackwell")
    monkeypatch.setenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", "blackwell")
    _write_task_yaml(
        tmp_path,
        "mixed.yaml",
        _default_body("mixed", ["b200-1gpu", "h100-1gpu"]),
    )

    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")

    assert [entry["runner"] for entry in matrix["include"]] == ["h100-1gpu"]


def test_build_matrix_empty_exclusion_restores_all_runners(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", " , , ")
    _write_task_yaml(
        tmp_path,
        "mixed.yaml",
        _default_body(
            "mixed",
            ["b300-1gpu", "amd-mi355-1gpu-bench"],
        ),
    )

    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")

    assert [entry["runner"] for entry in matrix["include"]] == [
        "b300-1gpu",
        "amd-mi355-1gpu-bench",
    ]


def test_build_matrix_all_excluded_returns_empty_include(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", "gpu")
    _write_task_yaml(
        tmp_path,
        "mixed.yaml",
        _default_body(
            "mixed",
            ["b300-1gpu", "amd-mi355-1gpu-bench"],
        ),
    )

    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")

    assert matrix == {"include": []}


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


def test_build_matrix_per_label_optional_only_affects_listed_label(tmp_path):
    _write_task_yaml(
        tmp_path,
        "ut-kernel.yaml",
        _default_body(
            "ut-kernel",
            ["h100-1gpu", "amd-mi355-1gpu-bench"],
            extra="optional:\n  amd-mi355-1gpu-bench: true\n",
        ),
    )
    matrix = build_matrix(tmp_path, tmp_path, trigger="per-commit")
    assert [(e["runner"], e["optional"]) for e in matrix["include"]] == [
        ("h100-1gpu", False),
        ("amd-mi355-1gpu-bench", True),
    ]


def test_build_matrix_splits_nvidia_arm_from_x86(tmp_path):
    _write_task_yaml(
        tmp_path,
        "mixed-nvidia.yaml",
        _default_body("mixed-nvidia", ["h100-1gpu", "b200-1gpu", "gb200-1gpu"]),
    )

    x86_matrix = build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        runner_group="nvidia-x86",
    )
    arm_matrix = build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        runner_group="nvidia-arm",
    )

    assert [entry["runner"] for entry in x86_matrix["include"]] == [
        "h100-1gpu",
        "b200-1gpu",
    ]
    assert [entry["runner"] for entry in arm_matrix["include"]] == [
        "gb200-1gpu",
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


def test_format_perf_reference_tables_include_config_for_nested_references():
    rows = extract_perf_summary_rows(PERF_MULTI_CONFIG_CSV_FIXTURE)
    task = {
        "perf_threshold": 1.0,
        "perf_reference": {
            "input_1k": {16: [100.0, 1000.0]},
            "input_4k": {16: [50.0, 500.0]},
        },
    }
    check = check_perf_reference(task, _command_results_with(rows), ["perf"])
    assert check is not None

    text_table = format_perf_reference_table(check["checks"])
    markdown_table = format_perf_reference_markdown_table(check["checks"])

    assert "Config" in text_table[0]
    assert "input_1k" in text_table[2]
    assert markdown_table[0].startswith("| Conc | Config |")
    assert "input_4k" in markdown_table[-1]


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


def test_step_summary_counts_config_concurrency_points():
    rows = extract_perf_summary_rows(PERF_MULTI_CONFIG_CSV_FIXTURE)
    task = {
        "perf_threshold": 1.0,
        "perf_reference": {
            "input_1k": {16: [100.0, 1000.0]},
            "input_4k": {16: [50.0, 500.0]},
        },
    }
    check = check_perf_reference(task, _command_results_with(rows), ["perf"])
    summary = "\n".join(
        build_step_summary_lines(_base_result(perf_reference_check=check))
    )

    assert "2 config/concurrency points" in summary


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
