import argparse
import subprocess
import textwrap
from pathlib import Path

import pytest
from slurm_submit import Task, gpu_count, load_task, render_script, select_tasks


def write_task(repo: Path, runner: str = "gb200-1gpu") -> str:
    relative = Path("test/ci/eval/example.yaml")
    path = repo / relative
    path.parent.mkdir(parents=True)
    path.write_text(textwrap.dedent(f"""\
            api_version: ci.tokenspeed.io/v1
            name: example
            type: eval
            triggers: [manual]
            runner:
              labels: [{runner}]
            server:
              command: ts serve --model example/model
              ready:
                url: http://127.0.0.1:8000/readiness
            eval:
              command: run-eval
            """))
    return relative.as_posix()


def test_gpu_count():
    assert gpu_count("gb200-4gpu") == 4
    with pytest.raises(ValueError):
        gpu_count("gb200")


def test_load_task(tmp_path):
    config = write_task(tmp_path)
    assert load_task(tmp_path, config) == Task(config, "example", "gb200-1gpu", 1)


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
    args = argparse.Namespace(config=None, runner="gb200-1gpu", trigger="manual")
    assert select_tasks(args, tmp_path) == [Task(config, "example", "gb200-1gpu", 1)]


def test_render_script_contains_cluster_requirements():
    script = render_script(
        Task("test/ci/eval/example.yaml", "example", "gb200-1gpu", 1),
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
