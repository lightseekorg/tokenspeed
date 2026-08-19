import re
import subprocess
import textwrap
from pathlib import Path

import pipeline
import pytest
from pipeline import (
    STALE_PROCESS_PATTERNS,
    build_matrix,
    build_step_summary_lines,
    check_eval_score_threshold,
    check_perf_reference,
    configure_slurm_server_command,
    ensure_ready_port_available,
    extract_evalscope_score,
    extract_perf_summary_rows,
    filter_stage_commands,
    format_perf_reference_markdown_table,
    format_perf_reference_table,
    get_excluded_runner_labels,
    get_jit_cache_env,
    get_runner_specific_env,
    get_stage_commands,
    is_amd_runner,
    is_gb200_runner,
    is_nvidia_arm_runner,
    parse_args,
    poll_readiness,
    resolve_score_threshold_for_runner,
    runner_matches_group,
    setup_runner,
    should_run_nvidia_gpu_cleanup,
    validate_task,
    wrap_command_with_log,
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


def test_poll_readiness_fails_when_server_process_exits(monkeypatch, tmp_path):
    class ServerProcess:
        calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 7

    def unavailable(*_args, **_kwargs):
        raise pipeline.URLError("not ready")

    log = tmp_path / "server.log"
    log.write_text("first line\nfatal startup error\n")
    monkeypatch.setattr(pipeline, "urlopen", unavailable)

    with pytest.raises(RuntimeError, match="exit code 7") as exc_info:
        poll_readiness(
            {"url": "http://127.0.0.1:8000/readiness", "interval": 10},
            False,
            process=ServerProcess(),
            log_path=log,
        )
    assert "fatal startup error" in str(exc_info.value)


def test_server_log_wrapper_preserves_server_exit_code(tmp_path):
    command = wrap_command_with_log("exit 9", tmp_path / "server.log")
    assert "set -o pipefail" in command
    assert subprocess.run(command, shell=True).returncode == 9


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
    assert is_nvidia_arm_runner("gb300-4gpu")
    assert not is_nvidia_arm_runner("b200-1gpu")
    assert not is_nvidia_arm_runner("amd-mi35x-1gpu-test")

    assert runner_matches_group("gb200-1gpu", "nvidia")
    assert runner_matches_group("gb200-1gpu", "nvidia-arm")
    assert not runner_matches_group("gb200-1gpu", "nvidia-x86")
    assert runner_matches_group("gb300-4gpu", "nvidia")
    assert runner_matches_group("gb300-4gpu", "nvidia-arm")
    assert not runner_matches_group("gb300-4gpu", "nvidia-x86")
    assert runner_matches_group("slurm-b300-4gpu", "nvidia-arm")
    assert runner_matches_group("slurm-gb200-4node-4gpu", "nvidia-arm")
    assert runner_matches_group("slurm-gb300-4gpu", "nvidia-arm")
    assert not runner_matches_group("slurm-b300-4gpu", "nvidia-x86")
    assert not runner_matches_group("slurm-gb200-4node-4gpu", "nvidia-x86")
    assert pipeline.get_default_runner_env("slurm-b300-4gpu")["SM"] == "sm103"
    assert pipeline.uses_isolated_jit_cache("slurm-gb300-4gpu")
    assert runner_matches_group("b200-1gpu", "nvidia-x86")
    assert runner_matches_group("b300-4gpu", "nvidia-x86")
    assert not runner_matches_group("amd-mi35x-1gpu-test", "nvidia-arm")
    assert not runner_matches_group("amd-mi35x-1gpu-test", "nvidia-x86")


def test_slurm_runner_override_is_narrow_and_gpu_counted():
    assert (
        pipeline.apply_slurm_runner_override("b300-1gpu", "gb300-1gpu", "slurm", "ut")
        == "gb300-1gpu"
    )
    assert (
        pipeline.apply_slurm_runner_override(
            "slurm-b300-4gpu", "slurm-gb300-4gpu", "slurm", "eval"
        )
        == "slurm-gb300-4gpu"
    )
    with pytest.raises(ValueError, match="GPU counts"):
        pipeline.apply_slurm_runner_override("b300-4gpu", "gb300-1gpu", "slurm", "ut")
    with pytest.raises(ValueError, match="b300"):
        pipeline.apply_slurm_runner_override("b200-1gpu", "gb300-1gpu", "slurm", "ut")
    with pytest.raises(ValueError, match="setup-mode"):
        pipeline.apply_slurm_runner_override("b300-1gpu", "gb300-1gpu", "ci", "ut")
    with pytest.raises(ValueError, match="perf"):
        pipeline.apply_slurm_runner_override("b300-4gpu", "gb300-4gpu", "slurm", "perf")


def test_nvidia_gpu_cleanup_runner_prefixes_cover_gb200_and_b300():
    assert is_gb200_runner("gb200-1gpu")
    assert is_gb200_runner("gb200-4gpu-perf")
    assert is_gb200_runner("slurm-gb200-4node-4gpu")
    assert not is_gb200_runner("b300-4gpu")
    assert not is_gb200_runner("gb300-4gpu")

    assert should_run_nvidia_gpu_cleanup("gb200-1gpu")
    assert should_run_nvidia_gpu_cleanup("gb200-4gpu-perf")
    assert should_run_nvidia_gpu_cleanup("b300-4gpu")
    assert not should_run_nvidia_gpu_cleanup("gb300-4gpu")
    assert not should_run_nvidia_gpu_cleanup("b200-4gpu")
    assert not should_run_nvidia_gpu_cleanup("h100-1gpu")
    assert not should_run_nvidia_gpu_cleanup("amd-mi35x-2gpu-test")
    assert not should_run_nvidia_gpu_cleanup("amd-mi355-1gpu-bench")
    assert not should_run_nvidia_gpu_cleanup("amd-mi350-1gpu-bench")


def test_b200v2_setup_forces_all_apt_invocations_to_ipv4(tmp_path, capsys):
    setup_runner("b200v2-4gpu", {}, tmp_path, dry_run=True)

    output = capsys.readouterr().out
    assert (
        'Acquire::ForceIPv4 "true";'
        "' | sudo tee /etc/apt/apt.conf.d/99tokenspeed-force-ipv4"
    ) in output


def test_execute_cli_defaults_to_ci_setup_mode():
    args = parse_args(
        [
            "execute",
            "--config",
            "test/ci/eval/task.yaml",
            "--runner",
            "gb200-4gpu",
        ]
    )

    assert args.setup_mode == "ci"


def test_scan_cli_excludes_multi_node_tasks_by_default():
    args = parse_args(["scan"])

    assert args.runner_group == "all"
    assert args.multi_node == "exclude"


def test_execute_cli_accepts_slurm_setup_mode():
    args = parse_args(
        [
            "execute",
            "--config",
            "test/ci/eval/task.yaml",
            "--runner",
            "gb200-4gpu",
            "--setup-mode",
            "slurm",
        ]
    )

    assert args.setup_mode == "slurm"


def test_execute_cli_accepts_slurm_server_modes():
    serve = parse_args(
        [
            "execute",
            "--config",
            "test/ci/eval/task.yaml",
            "--runner",
            "slurm-gb200-4node-4gpu",
            "--setup-mode",
            "slurm",
            "--serve-only",
        ]
    )
    client = parse_args(
        [
            "execute",
            "--config",
            "test/ci/eval/task.yaml",
            "--runner",
            "slurm-gb200-4node-4gpu",
            "--setup-mode",
            "slurm",
            "--external-server",
        ]
    )

    assert serve.serve_only is True
    assert serve.external_server is False
    assert client.serve_only is False
    assert client.external_server is True


def test_multi_node_slurm_task_validation():
    task = {
        "api_version": "ci.tokenspeed.io/v1",
        "name": "multi-node",
        "type": "eval",
        "workflow_stage": "model-test",
        "triggers": ["slurm"],
        "runner": {"labels": ["slurm-gb200-4node-4gpu"]},
        "slurm": {"nodes": 4, "gpus_per_node": 4},
        "server": {
            "command": "ts serve --model example/model",
            "ready": {"url": "http://127.0.0.1:8000/readiness"},
        },
        "eval": {"command": "run eval"},
    }

    validate_task(task, Path("task.yaml"))

    task["triggers"] = ["per-commit"]
    validate_task(task, Path("task.yaml"))

    task["triggers"] = ["manual"]
    with pytest.raises(ValueError, match="exactly one"):
        validate_task(task, Path("task.yaml"))

    task["triggers"] = ["per-commit", "slurm"]
    with pytest.raises(ValueError, match="exactly one"):
        validate_task(task, Path("task.yaml"))


def test_slurm_server_modes_split_server_from_client(monkeypatch, tmp_path):
    task = {
        "name": "multi-node",
        "type": "eval",
        "runner": {"labels": ["slurm-gb200-4node-4gpu"]},
        "install": ["install project"],
        "server": {
            "command": "ts serve --model example/model",
            "ready": {"url": "http://127.0.0.1:8000/readiness"},
        },
        "eval": {
            "install": ["install eval"],
            "command": "run eval",
        },
    }

    class FakeProcessGroupManager:
        def __init__(self):
            self.commands = []

        def run(self, command, *, cwd, env, dry_run):
            self.commands.append(command)
            return {"returncode": 0, "output": ""}

        def start(self, *args, **kwargs):
            raise AssertionError("server modes must not start a managed server")

        def terminate_all(self, *, dry_run):
            pass

    monkeypatch.setattr(pipeline, "normalize_task", lambda path, root: task)
    monkeypatch.setattr(pipeline, "summarize_task_targets", lambda *_: {})
    monkeypatch.setattr(pipeline, "get_jit_cache_env", lambda *_: {})

    serve_manager = FakeProcessGroupManager()
    monkeypatch.setattr(
        pipeline,
        "setup_runner",
        lambda *args, **kwargs: (args[1], serve_manager),
    )
    assert (
        pipeline.execute_task(
            config="task.yaml",
            runner="slurm-gb200-4node-4gpu",
            work_dir=str(tmp_path),
            dry_run=False,
            print_plan=False,
            result_json=None,
            setup_mode="slurm",
            serve_only=True,
        )
        == 0
    )
    assert serve_manager.commands == [
        "install project",
        "ts serve --model example/model --engine-startup-timeout 7200",
    ]

    ready_calls = []
    monkeypatch.setattr(
        pipeline,
        "poll_readiness",
        lambda ready, dry_run, **kwargs: ready_calls.append((ready, dry_run)),
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_ready_port_available",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("external server mode must not check for a free port")
        ),
    )
    client_manager = FakeProcessGroupManager()
    monkeypatch.setattr(
        pipeline,
        "setup_runner",
        lambda *args, **kwargs: (args[1], client_manager),
    )
    assert (
        pipeline.execute_task(
            config="task.yaml",
            runner="slurm-gb200-4node-4gpu",
            work_dir=str(tmp_path),
            dry_run=False,
            print_plan=False,
            result_json=None,
            setup_mode="slurm",
            external_server=True,
        )
        == 0
    )
    assert client_manager.commands == ["install project", "install eval", "run eval"]
    assert ready_calls == [
        (
            {
                "url": "http://127.0.0.1:8000/readiness",
                "timeout": 7200,
            },
            False,
        )
    ]


def test_slurm_setup_is_scoped_to_job_id(monkeypatch, tmp_path):
    class FakeProcessGroupManager:
        def __init__(self, runner_id):
            self.runner_id = runner_id

    monkeypatch.setattr(pipeline, "ProcessGroupManager", FakeProcessGroupManager)
    original_env = {"SLURM_JOB_ID": "12345", "CI_VENV_PATH": "/shared/venv"}

    local_env, manager = setup_runner(
        "gb200-4gpu",
        original_env,
        tmp_path,
        dry_run=False,
        setup_mode="slurm",
    )

    assert manager.runner_id == "slurm-12345"
    assert local_env["CI_RUNNER_ID"] == "slurm-12345"
    assert local_env["CI_VENV_PATH"] == "/shared/venv"
    assert "CI_RUNNER_ID" not in original_env


def test_slurm_setup_isolates_overlapping_steps(monkeypatch, tmp_path):
    class FakeProcessGroupManager:
        def __init__(self, runner_id):
            self.runner_id = runner_id

    monkeypatch.setattr(pipeline, "ProcessGroupManager", FakeProcessGroupManager)

    local_env, manager = setup_runner(
        "slurm-gb200-4node-4gpu",
        {"SLURM_JOB_ID": "12345", "SLURM_STEP_ID": "7", "SLURM_PROCID": "3"},
        tmp_path,
        dry_run=False,
        setup_mode="slurm",
    )

    assert manager.runner_id == "slurm-12345-step-7-proc-3"
    assert local_env["CI_RUNNER_ID"] == manager.runner_id


def test_slurm_setup_requires_job_id(tmp_path):
    with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
        setup_runner(
            "gb200-4gpu",
            {},
            tmp_path,
            dry_run=False,
            setup_mode="slurm",
        )


def test_slurm_ready_port_check_rejects_an_existing_listener(monkeypatch):
    class ConnectedSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    calls = []

    def connect(address, timeout):
        calls.append((address, timeout))
        return ConnectedSocket()

    monkeypatch.setattr(pipeline.socket, "create_connection", connect)

    with pytest.raises(RuntimeError, match="already in use"):
        ensure_ready_port_available(
            {"url": "http://127.0.0.1:8123/readiness"}, dry_run=False
        )

    assert calls == [(("127.0.0.1", 8123), 0.5)]


def test_slurm_ready_port_check_accepts_an_unused_port(monkeypatch):
    def refuse_connection(address, timeout):
        raise ConnectionRefusedError

    monkeypatch.setattr(pipeline.socket, "create_connection", refuse_connection)

    ensure_ready_port_available(
        {"url": "http://127.0.0.1:8123/readiness"}, dry_run=False
    )


def test_skipping_top_level_install_keeps_eval_install():
    task = {
        "type": "eval",
        "install": ["install project"],
        "server": {
            "command": "serve",
            "ready": {"url": "http://127.0.0.1:8000/readiness"},
        },
        "eval": {
            "install": ["install eval dependencies"],
            "command": "run eval",
        },
    }

    stages = filter_stage_commands(
        get_stage_commands(task), only_stages=None, skip_stages={"install"}
    )

    assert [name for name, _ in stages] == ["server", "eval.install", "eval"]


def test_slurm_execution_only_cleans_its_process_group(monkeypatch, tmp_path):
    task = {
        "name": "slurm-unit-test",
        "type": "ut",
        "runner": {"labels": ["gb200-1gpu"]},
        "ut": {"commands": ["run test"]},
        "report": {"github_step_summary": True},
    }
    result_json = tmp_path / "result.json"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "missing" / "summary.md"))

    class FakeProcessGroupManager:
        terminated = False

        def run(self, command, *, cwd, env, dry_run):
            return {"returncode": 0, "output": ""}

        def terminate_all(self, *, dry_run):
            self.terminated = True

    manager = FakeProcessGroupManager()
    monkeypatch.setattr(pipeline, "normalize_task", lambda path, root: task)
    monkeypatch.setattr(
        pipeline,
        "setup_runner",
        lambda runner, env, cwd, dry_run, reuse_state, setup_mode: (env, manager),
    )

    def reject_global_cleanup(*args, **kwargs):
        raise AssertionError("Slurm mode must not run global runner cleanup")

    monkeypatch.setattr(pipeline, "cleanup_runner", reject_global_cleanup)

    return_code = pipeline.execute_task(
        config="task.yaml",
        runner="gb200-1gpu",
        work_dir=str(tmp_path),
        dry_run=False,
        print_plan=False,
        result_json=str(result_json),
        setup_mode="slurm",
    )

    assert return_code == 0
    assert manager.terminated is True
    assert result_json.exists()


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


def test_step_summary_write_failure_is_non_fatal(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "missing" / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(missing))

    pipeline.write_detailed_step_summary(_base_result())

    assert "warning: could not write GitHub step summary" in capsys.readouterr().err


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
workflow_stage: unit-test
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


def test_build_matrix_can_select_or_exclude_multi_node_tasks(tmp_path):
    _write_task_yaml(
        tmp_path,
        "single.yaml",
        _default_body("single", ["gb300-4gpu"]),
    )
    _write_task_yaml(
        tmp_path,
        "multi.yaml",
        """
        api_version: ci.tokenspeed.io/v1
        name: multi
        type: eval
        workflow_stage: model-test
        triggers: [per-commit]
        runner:
          labels: [slurm-b300-4gpu]
        slurm:
          nodes: 2
          gpus_per_node: 4
        server:
          command: ts serve example/model
          ready:
            url: http://127.0.0.1:8000/readiness
        eval:
          command: run eval
        """,
    )

    only = build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        runner_group="nvidia-arm",
        multi_node="only",
    )
    excluded = build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        runner_group="nvidia-arm",
        multi_node="exclude",
    )
    default = build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        runner_group="nvidia-arm",
    )

    assert [entry["name"] for entry in only["include"]] == ["multi"]
    assert [entry["name"] for entry in excluded["include"]] == ["single"]
    assert default == excluded
    assert build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        runner_group="nvidia-x86",
    ) == {"include": []}
    assert build_matrix(
        tmp_path,
        tmp_path,
        trigger="per-commit",
        runner_group="amd",
    ) == {"include": []}


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


def test_slurm_server_command_inherits_readiness_timeout():
    assert configure_slurm_server_command("ts serve --model example", 7200) == (
        "ts serve --model example --engine-startup-timeout 7200"
    )
    explicit = "ts serve --model example --engine-startup-timeout 300"
    assert configure_slurm_server_command(explicit, 7200) == explicit
    assert configure_slurm_server_command("python serve.py", 7200) == "python serve.py"


def test_jit_caches_are_redirected_off_the_work_dir():
    env = {"RUNNER_NAME": "gb200-1gpu-0"}
    resolved = get_jit_cache_env(env)
    assert resolved == {
        "TRITON_CACHE_DIR": "/tmp/ci-jit-cache-gb200-1gpu-0/triton",
        "CUTE_DSL_CACHE_DIR": "/tmp/ci-jit-cache-gb200-1gpu-0/cute_dsl",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/ci-jit-cache-gb200-1gpu-0/torchinductor",
        "TORCH_EXTENSIONS_DIR": "/tmp/ci-jit-cache-gb200-1gpu-0/torch_extensions",
    }


def test_stale_torch_extension_build_is_removed(tmp_path):
    cache_root = tmp_path / "torch_extensions"
    locked_build = cache_root / "ext_a"
    complete_build = cache_root / "ext_b"
    locked_build.mkdir(parents=True)
    complete_build.mkdir()
    (locked_build / "lock").touch()
    (locked_build / "partial.so").touch()
    (complete_build / "complete.so").touch()

    pipeline.remove_stale_torch_extension_builds(
        {"TORCH_EXTENSIONS_DIR": str(cache_root)}, dry_run=False
    )

    assert not locked_build.exists()
    assert (complete_build / "complete.so").exists()


def test_stale_torch_extension_build_survives_dry_run(tmp_path, capsys):
    pipeline.remove_stale_torch_extension_builds({}, dry_run=False)
    pipeline.remove_stale_torch_extension_builds(
        {"TORCH_EXTENSIONS_DIR": str(tmp_path / "missing")}, dry_run=True
    )
    locked_build = tmp_path / "torch_extensions" / "ext_a"
    locked_build.mkdir(parents=True)
    (locked_build / "lock").touch()

    pipeline.remove_stale_torch_extension_builds(
        {"TORCH_EXTENSIONS_DIR": str(locked_build.parent)}, dry_run=True
    )

    assert locked_build.exists()
    assert "[dry-run] remove stale torch extension build" in capsys.readouterr().out


def test_stale_torch_extension_cleanup_fails_if_lock_survives(tmp_path, monkeypatch):
    locked_build = tmp_path / "torch_extensions" / "ext_a"
    locked_build.mkdir(parents=True)
    (locked_build / "lock").touch()
    monkeypatch.setattr(pipeline.shutil, "rmtree", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="failed to remove stale"):
        pipeline.remove_stale_torch_extension_builds(
            {"TORCH_EXTENSIONS_DIR": str(locked_build.parent)}, dry_run=False
        )


def test_jit_cache_env_keeps_explicit_overrides():
    env = {"RUNNER_NAME": "gb200-1gpu-0", "TRITON_CACHE_DIR": "/mnt/triton"}
    assert "TRITON_CACHE_DIR" not in get_jit_cache_env(env)
