import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import pytest
from slurm_submit import (
    Submission,
    Task,
    gpu_count,
    load_task,
    parse_pr_number,
    pr_worktree,
    print_progress,
    print_target,
    queued_states,
    render_script,
    result_detail,
    select_tasks,
    source_pr_url,
    submit,
    write_report,
)


def write_task(
    repo: Path,
    runner: str = "gb200-1gpu",
    task_type: str = "eval",
    model: str = "example/model",
    nodes: int | None = None,
    gpus_per_node: int | None = None,
) -> str:
    workflow_stage = "unit-test" if task_type == "ut" else "model-test"
    trigger = "slurm" if nodes and nodes > 1 else "manual"
    relative = Path(f"test/ci/{task_type}/example.yaml")
    path = repo / relative
    path.parent.mkdir(parents=True)
    lines = [
        "api_version: ci.tokenspeed.io/v1",
        "name: example",
        f"type: {task_type}",
        f"workflow_stage: {workflow_stage}",
        f"triggers: [{trigger}]",
    ]
    if nodes is not None:
        lines.extend(
            [
                "slurm:",
                f"  nodes: {nodes}",
                f"  gpus_per_node: {gpus_per_node}",
            ]
        )
    lines.extend(
        [
            "runner:",
            f"  labels: [{runner}]",
            "server:",
            f"  command: ts serve --model {model}",
            "  ready:",
            "    url: http://127.0.0.1:8000/readiness",
            f"{task_type}:",
            "  command: run-eval",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return relative.as_posix()


def test_gpu_count():
    assert gpu_count("gb200-4gpu") == 4
    with pytest.raises(ValueError):
        gpu_count("gb200")


def test_load_task(tmp_path):
    config = write_task(tmp_path)
    assert load_task(tmp_path, config) == Task(
        config, "example", "eval", "gb200-1gpu", 1
    )


def test_load_task_reads_multi_node_topology(tmp_path):
    config = write_task(
        tmp_path,
        runner="slurm-gb200-4node-4gpu",
        nodes=4,
        gpus_per_node=4,
    )

    assert load_task(tmp_path, config) == Task(
        config, "example", "eval", "slurm-gb200-4node-4gpu", 4, 4
    )


def test_load_task_rejects_gpu_topology_mismatch(tmp_path):
    config = write_task(
        tmp_path,
        runner="slurm-gb200-4node-4gpu",
        nodes=4,
        gpus_per_node=2,
    )

    with pytest.raises(ValueError, match="does not match runner"):
        load_task(tmp_path, config)


def test_load_task_checks_runner(tmp_path):
    config = write_task(tmp_path)
    with pytest.raises(ValueError, match="not declared"):
        load_task(tmp_path, config, "gb200-4gpu")


def test_load_task_supports_optional_gb300_alias(tmp_path):
    config = write_task(tmp_path, runner="b300-1gpu", task_type="ut")

    assert load_task(tmp_path, config, "b300-1gpu", "gb300-1gpu") == Task(
        config, "example", "ut", "gb300-1gpu", 1, 1, "b300-1gpu"
    )


def test_render_script_passes_declared_and_effective_gb300_runners():
    script = render_script(
        Task(
            "test/ci/ut/example.yaml",
            "example",
            "ut",
            "gb300-1gpu",
            1,
            declared_runner="b300-1gpu",
        ),
        Path("/shared/source.tar"),
        Path("/shared/runs"),
        Path("/shared/cache"),
        "ghcr.io/example/image@sha256:abc",
    )

    assert "--runner=b300-1gpu" in script
    assert "--runner-override=gb300-1gpu" in script
    assert 'gpu_ids="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"' in script


@pytest.mark.parametrize(
    ("declared", "effective", "message"),
    [
        ("b300-4gpu", "gb300-1gpu", "GPU counts"),
        ("b200-1gpu", "gb300-1gpu", "b300"),
    ],
)
def test_load_task_rejects_invalid_gb300_alias(tmp_path, declared, effective, message):
    config = write_task(tmp_path, runner=declared, task_type="ut")

    with pytest.raises(ValueError, match=message):
        load_task(tmp_path, config, declared, effective)


def test_load_task_rejects_gb300_perf_alias(tmp_path):
    config = write_task(tmp_path, runner="b300-4gpu", task_type="perf")

    with pytest.raises(ValueError, match="perf"):
        load_task(tmp_path, config, "b300-4gpu", "gb300-4gpu")


def test_select_all_filters_exact_runner(monkeypatch, tmp_path):
    config = write_task(tmp_path)
    monkeypatch.setattr(
        "slurm_submit.build_matrix",
        lambda *_: {
            "include": [
                {
                    "config": config,
                    "type": "eval",
                    "runner": "gb200-1gpu",
                }
            ]
        },
    )
    args = argparse.Namespace(
        config=None,
        runner=["gb200-1gpu"],
        task_types=None,
        match=None,
        trigger="manual",
    )
    assert select_tasks(args, tmp_path) == [
        Task(config, "example", "eval", "gb200-1gpu", 1)
    ]


def test_select_all_rejects_runner_alias_even_with_runner(tmp_path):
    args = argparse.Namespace(
        config=None,
        runner=["gb200-1gpu"],
        runner_alias=["b300-1gpu=gb300-1gpu"],
        task_types=None,
        match=None,
        trigger="manual",
    )

    with pytest.raises(ValueError, match="requires --config"):
        select_tasks(args, tmp_path)


def test_select_all_supports_multiple_runners_types_and_model_match(
    monkeypatch, tmp_path
):
    config = write_task(tmp_path, model="moonshotai/Kimi-K2")
    unit_config = write_task(tmp_path, "b200-2gpu", "ut")
    monkeypatch.setattr(
        "slurm_submit.build_matrix",
        lambda *_: {
            "include": [
                {"config": config, "type": "eval", "runner": "gb200-1gpu"},
                {"config": unit_config, "type": "ut", "runner": "b200-2gpu"},
            ]
        },
    )
    args = argparse.Namespace(
        config=None,
        runner=["gb200-1gpu", "b200-2gpu"],
        task_types=["eval", "ut"],
        match=["kimi"],
        trigger=None,
    )
    assert select_tasks(args, tmp_path) == [
        Task(config, "example", "eval", "gb200-1gpu", 1)
    ]


def test_select_all_supports_exclude_match(monkeypatch, tmp_path):
    config = write_task(tmp_path, model="example/mmlu")
    monkeypatch.setattr(
        "slurm_submit.build_matrix",
        lambda *_: {
            "include": [
                {"config": config, "type": "eval", "runner": "gb200-1gpu"},
            ]
        },
    )
    args = argparse.Namespace(
        config=None,
        runner=["gb200-1gpu"],
        task_types=["eval"],
        match=None,
        exclude_match=["mmlu"],
        trigger=None,
    )
    with pytest.raises(ValueError, match="no tasks match"):
        select_tasks(args, tmp_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("795", 795),
        ("https://github.com/lightseekorg/tokenspeed/pull/795", 795),
        ("https://github.com/lightseekorg/tokenspeed/pull/795/", 795),
    ],
)
def test_parse_pr_number(value, expected):
    assert parse_pr_number(value) == expected


def test_parse_pr_number_rejects_other_urls():
    with pytest.raises(ValueError, match="pull request"):
        parse_pr_number("https://example.com/pull/795")


def test_source_pr_url_uses_repository_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
    assert source_pr_url("884") == "https://github.com/lightseekorg/tokenspeed/pull/884"


def test_print_target_distinguishes_pr_head_from_merge(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
    commits = {"HEAD^2": "pr-head", "HEAD^1": "base-head"}
    monkeypatch.setattr("slurm_submit.git", lambda _repo, *_args: commits[_args[-1]])

    print_target(tmp_path, "884", "merged-test")

    assert capsys.readouterr().out.splitlines() == [
        "Target: PR #884",
        "Link: https://github.com/lightseekorg/tokenspeed/pull/884",
        "Target commit: pr-head",
        "Merged test commit: merged-test",
        "Base commit: base-head",
    ]


def test_pr_worktree_rejects_shallow_checkout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", source], check=True)
    subprocess.run(["git", "-C", source, "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", source, "config", "user.email", "test@example.com"], check=True
    )
    (source / "README.md").write_text("test\n")
    subprocess.run(["git", "-C", source, "add", "README.md"], check=True)
    subprocess.run(["git", "-C", source, "commit", "-qm", "initial"], check=True)
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth=1", f"file://{source}", shallow],
        check=True,
    )

    with pytest.raises(ValueError, match="fetch-depth: 0"):
        with pr_worktree(shallow, "794"):
            pass


def test_render_script_contains_cluster_requirements():
    script = render_script(
        Task("test/ci/eval/example.yaml", "example", "eval", "gb200-1gpu", 1),
        Path("/shared/source.tar"),
        Path("/shared/runs"),
        Path("/shared/cache"),
        "ghcr.io/example/image@sha256:abc",
    )
    assert "--setup-mode=slurm" in script
    assert "--container-remap-root" in script
    assert 'gpu_ids="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"' not in script
    assert "libcuda.so.1" in script
    assert "libcudart.so.13" in script
    assert "/usr/bin/nvidia-smi" in script
    assert "/shared/cache:/home/runner/.cache" in script
    unset = (
        "unset GITHUB_STEP_SUMMARY GITHUB_OUTPUT GITHUB_ENV GITHUB_PATH "
        "GITHUB_STATE   GITHUB_EVENT_PATH"
    )
    assert unset in script
    assert script.index(unset) < script.index('srun "${srun_args[@]}"')
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_render_script_mounts_only_allocated_gb300_devices():
    script = render_script(
        Task("test/ci/ut/example.yaml", "example", "ut", "gb300-1gpu", 1),
        Path("/shared/source.tar"),
        Path("/shared/runs"),
        Path("/shared/cache"),
        "ghcr.io/example/image@sha256:abc",
    )

    assert 'gpu_ids="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"' in script
    assert 'device="/dev/nvidia$gpu"' in script
    assert 'elif [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]' in script
    assert "No NVIDIA device nodes matched Slurm GPU allocation" in script
    assert "\n  /dev/nvidiactl \\\n" in script
    assert "\n  /dev/nvidia-uvm \\\n" in script
    assert "\n  /dev/nvidia-uvm-tools \\\n" in script
    assert "\n  /dev/nvidia-nvswitchctl \\\n" in script
    assert "\n  /dev/nvidia-caps \\\n" in script
    assert "\n  /dev/nvidia-caps-imex-channels; do\n" in script
    assert '"${gpu_mounts[@]}" "${mounts[@]}"' in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    multinode_script = render_script(
        Task(
            "test/ci/eval/example.yaml",
            "example",
            "eval",
            "slurm-gb300-4node-4gpu",
            16,
            nodes=4,
        ),
        Path("/shared/source.tar"),
        Path("/shared/runs"),
        Path("/shared/cache"),
        "ghcr.io/example/image@sha256:abc",
    )
    assert 'gpu_ids="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"' in multinode_script


def test_render_multinode_gb300_keeps_devices_out_of_client_step():
    script = render_script(
        Task(
            "test/ci/eval/example.yaml",
            "example",
            "eval",
            "gb300-4gpu",
            4,
            nodes=2,
        ),
        Path("/shared/source.tar"),
        Path("/shared/runs"),
        Path("/shared/cache"),
        "ghcr.io/example/image@sha256:abc",
    )

    assert (
        'server_mounts=("$server_src:/workspace" "$server_tmp:/tmp" '
        '"${gpu_mounts[@]}" "${mounts[@]}")' in script
    )
    assert (
        'client_mounts=("$client_src:/workspace" "$client_tmp:/tmp" '
        '"${mounts[@]}")' in script
    )
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_render_script_orchestrates_multi_node_server_and_head_client():
    script = render_script(
        Task(
            "test/ci/eval/example.yaml",
            "example",
            "eval",
            "slurm-gb200-4node-4gpu",
            4,
            4,
        ),
        Path("/shared/source.tar"),
        Path("/shared/runs"),
        Path("/shared/cache"),
        "ghcr.io/example/image@sha256:abc",
    )

    assert "--nodes=4" in script
    assert "--ntasks=4" in script
    assert "--ntasks-per-node=1" in script
    assert "--gres=gpu:4" in script
    assert (
        'head_node="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n \'1p\')"'
        in script
    )
    assert 'client_prepare_args+=(--nodelist="$head_node")' in script
    assert 'client_srun_args+=(--nodelist="$head_node")' in script
    assert "--serve-only" in script
    assert "--external-server" in script
    assert "SLURM_STEP_NUM_NODES" in script
    assert "tokenspeed-prepare" in script
    assert "tokenspeed-client-prepare" in script
    assert 'server_src="$scratch/server-src"' in script
    assert 'client_src="$scratch/client-src"' in script
    assert "tokenspeed-cleanup" in script
    assert "trap cleanup EXIT" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_submit_requests_multi_node_allocation(capsys, tmp_path):
    task = Task(
        "test/ci/eval/example.yaml",
        "example",
        "eval",
        "slurm-gb200-4node-4gpu",
        4,
        4,
    )
    args = argparse.Namespace(
        partition="batch",
        time="01:00:00",
        nodelist=None,
        render=True,
    )

    submit(task, "#!/bin/bash\n", tmp_path, args, "a" * 40)

    command = capsys.readouterr().out.splitlines()[0]
    assert "--nodes=4" in command
    assert "--ntasks=4" in command
    assert "--ntasks-per-node=1" in command
    assert "--gres=gpu:4" in command


def test_result_detail_reports_eval_score(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        '{"eval_score_check": {"score": 0.95, "threshold": 0.9, "passed": true}}'
    )
    assert result_detail(result) == "score=0.95, threshold=0.9"


def test_write_report_collects_logs_and_results(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("task output\n")
    run_root = tmp_path / "runs"
    result = run_root / "123" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"error": ""}')
    task = Task("test/ci/eval/example.yaml", "example", "eval", "gb200-1gpu", 1)

    report = tmp_path / "report"
    write_report(
        [Submission(task, "123", log)],
        {
            "123": {
                "state": "COMPLETED",
                "elapsed": "00:01:00",
                "exit_code": "0:0",
            }
        },
        run_root,
        report,
    )

    assert (report / "123.log").read_text() == "task output\n"
    assert (report / "123-result.json").exists()
    manifest_task = json.loads((report / "manifest.json").read_text())[0]["task"]
    assert manifest_task == {
        "config": "test/ci/eval/example.yaml",
        "name": "example",
        "task_type": "eval",
        "runner": "gb200-1gpu",
        "gpus": 1,
        "nodes": 1,
    }
    assert (
        "| 123 | eval | gb200-1gpu | example | ✅ |"
        in (report / "summary.md").read_text()
    )


def test_queued_states_queries_only_requested_jobs(monkeypatch):
    def fake_run(command, **kwargs):
        assert command == [
            "squeue",
            "--noheader",
            "--jobs=123,456",
            "--format=%i|%T|%M|%R",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "123|RUNNING|00:01|node-a\n"
                "456|PENDING|00:00|Resources\n"
                "999|RUNNING|12:00|node-private\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("slurm_submit.subprocess.run", fake_run)
    assert queued_states(["123", "456"]) == {
        "123": {
            "state": "RUNNING",
            "elapsed": "00:01",
            "exit_code": "",
            "reason": "",
        },
        "456": {
            "state": "PENDING",
            "elapsed": "00:00",
            "exit_code": "",
            "reason": "Resources",
        },
    }


def test_print_progress_omits_running_node(capsys, tmp_path):
    submission = Submission(
        Task("test/ci/eval/example.yaml", "example", "eval", "gb200-1gpu", 1),
        "123",
        tmp_path / "job.log",
    )
    print_progress(
        [submission],
        {
            "123": {
                "state": "RUNNING",
                "elapsed": "00:01",
                "exit_code": "",
                "reason": "",
            }
        },
    )
    output = capsys.readouterr().out
    assert "123" in output
    assert "example" in output
    assert "node" not in output


def test_print_progress_handles_accounting_delay(capsys, tmp_path):
    submission = Submission(
        Task("test/ci/ut/example.yaml", "example", "ut", "gb300-1gpu", 1),
        "123",
        tmp_path / "job.log",
    )

    print_progress([submission], {"123": {}})

    output = capsys.readouterr().out
    assert "UNKNOWN" in output
    assert "123" in output
