import argparse
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
    render_script,
    result_detail,
    select_tasks,
    write_report,
)


def write_task(
    repo: Path,
    runner: str = "gb200-1gpu",
    task_type: str = "eval",
    model: str = "example/model",
) -> str:
    relative = Path(f"test/ci/{task_type}/example.yaml")
    path = repo / relative
    path.parent.mkdir(parents=True)
    path.write_text(textwrap.dedent(f"""\
            api_version: ci.tokenspeed.io/v1
            name: example
            type: {task_type}
            triggers: [manual]
            runner:
              labels: [{runner}]
            server:
              command: ts serve --model {model}
              ready:
                url: http://127.0.0.1:8000/readiness
            {task_type}:
              command: run-eval
            """))
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


def test_load_task_checks_runner(tmp_path):
    config = write_task(tmp_path)
    with pytest.raises(ValueError, match="not declared"):
        load_task(tmp_path, config, "gb200-4gpu")


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
    assert "libcuda.so.1" in script
    assert "libcudart.so.13" in script
    assert "/usr/bin/nvidia-smi" in script
    assert "/shared/cache:/home/runner/.cache" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


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
    assert (
        "| 123 | eval | gb200-1gpu | example | COMPLETED |"
        in (report / "summary.md").read_text()
    )
