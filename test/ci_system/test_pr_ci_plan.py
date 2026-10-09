# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Keep semantic validation priorities focused and bound to existing targets."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".github/scripts"))
spec = importlib.util.spec_from_file_location(
    "pr_ci_plan", REPO / ".github/scripts/pr_ci_plan.py"
)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def test_model_change_keeps_focused_tests_and_manual_ci(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
    monkeypatch.setenv("PR_NUMBER", "1")
    monkeypatch.setenv("TOKENSPEED_B200_RUNNER_LABEL", "b200v2")
    test = "test/runtime/test_deepseek_v41_engram.py"
    package_test = "tokenspeed-scheduler/python/tests/test_kv_cache.py"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **k: SimpleNamespace(
            stdout=(
                test + "\n" + package_test + "\n"
                if args[1] == "ls-files"
                else "python/tokenspeed/runtime/engram.py\n"
            )
        ),
    )
    data = planner.context(REPO, "a" * 40, "b" * 40)
    assert package_test in data["test_files"]
    config = "test/ci/ut/deepseek-v4.1-flash-pd-1p1d.yaml"
    task = next(t for t in data["catalog"] if t["config"] == config)
    assert task["triggers"] == ["manual"]
    assert task["runners"] == ["b200v2-4gpu"]
    assert task["slurm_runners"] == {"gb200": ["b200-4gpu"], "gb300": ["b200-4gpu"]}
    assert "test_deepseek_v41_pd_1p1d.py" in task["targets"]["commands"][0]
    assert any("qwen" in t["config"] for t in data["catalog"])
    result = planner.proposal(
        "Evidence summary from the CLI.\n\n```json\n"
        + json.dumps(
            {
                "summary": "Engram changes affect DeepSeek V4.1 cache history.",
                "tests": [
                    {
                        "path": test,
                        "label": "Engram inputs",
                        "reason": "History commits | graph\npadding [scrub]",
                    }
                ],
                "tasks": [
                    {
                        "config": config,
                        "runner": "b200-4gpu",
                        "cluster": "gb200",
                        "label": "PD handoff",
                        "reason": "Verify history across PD cache handoff.",
                    }
                ],
                "conflicts": "",
            }
        )
        + "\n```",
        data,
    )
    assert [t["path"] for t in result["tests"]] == [test]
    assert [t["config"] for t in result["tasks"]] == [config]
    body = planner.render(result)
    assert "qwen" not in body
    assert "| Order | Check | Verifies | Run on |" in body
    assert "History commits \\| graph padding \\[scrub\\]" in body
    assert (
        f"[Engram inputs](https://github.com/lightseekorg/tokenspeed/blob/{'a' * 40}/{test})"
        in body
    )
    assert "Slurm GB200 / 4 GPU" in body
    assert "GB300 if full" in body
    assert "tests not run; required CI unchanged" in body


def test_proposal_cannot_invent_runner_or_command():
    task = {
        "config": "test/ci/ut/example.yaml",
        "runners": ["b200v2-1gpu"],
        "name": "example",
    }
    data = {
        "version": 1,
        "repository": "lightseekorg/tokenspeed",
        "pr": 1,
        "head": "a" * 40,
        "base": "b" * 40,
        "catalog": [task],
        "test_files": [],
    }
    response = {
        "summary": "Focused validation.",
        "tests": [],
        "tasks": [
            {
                "config": task["config"],
                "runner": "arbitrary-command",
                "cluster": "gb200",
                "label": "Example CI",
                "reason": "Changed caller.",
            }
        ],
        "conflicts": "",
    }
    with pytest.raises(ValueError):
        planner.proposal(json.dumps(response), data)
    response["tasks"] = []
    response["tests"] = [
        {
            "path": "python/tokenspeed/__init__.py",
            "label": "Example test",
            "reason": "Not a test.",
        }
    ]
    with pytest.raises(ValueError, match="existing test file"):
        planner.proposal(json.dumps(response), data)


def test_recommended_runtime_ut_requires_its_ci_task(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
    monkeypatch.setenv("PR_NUMBER", "1")
    test = "test/runtime/test_multimodal_encoded_offload.py"
    cpu_test = "test/ci_system/test_pr_ci_plan.py"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **k: SimpleNamespace(stdout=f"{test}\n{cpu_test}\n"),
    )
    data = planner.context(REPO, "a" * 40, "b" * 40)
    configs = [
        "test/ci/ut/ut-runtime-2gpu.yaml",
        "test/ci/ut/ut-runtime-1gpu.yaml",
    ]
    choices = []
    for config in configs:
        task = next(t for t in data["catalog"] if t["config"] == config)
        choices.append(
            dict(
                config=config,
                runner=task["slurm_runners"]["gb200"][0],
                cluster="gb200",
                label="Runtime regression",
                reason="Bounded encoder calls preserve packed item order.",
            )
        )
    response = dict(
        summary="Multimodal encoder packing respects the forward token bound.",
        tests=[
            dict(path=path, label="Regression", reason="Verify selected behavior.")
            for path in (test, cpu_test)
        ],
        tasks=choices[:1],
        conflicts="",
    )
    with pytest.raises(planner.CoverageError, match="test_multimodal_encoded_offload"):
        planner.proposal(json.dumps(response), data)
    response["tasks"] = choices
    result = planner.proposal(json.dumps(response), data)
    assert [t["path"] for t in result["tests"]] == [test, cpu_test]
    assert [t["config"] for t in result["tasks"]] == configs


def test_scheduler_native_checks_precede_gpu_recommendations():
    checks = planner.native_checks(["tokenspeed-scheduler/src/capacity_model.cpp"])
    assert [c["workflow"] for c in checks] == [
        "scheduler-cpp-test.yml",
        "scheduler-python-test.yml",
    ]
    assert planner.native_checks(["python/tokenspeed/runtime/models/qwen3_5.py"]) == []
    assert (
        planner.native_checks([".github/workflows/scheduler-cpp-test.yml"])
        == checks[:1]
    )
    data = dict(
        version=1,
        repository="lightseekorg/tokenspeed",
        pr=1,
        head="a" * 40,
        base="b" * 40,
        native_checks=checks,
        catalog=[],
        test_files=[],
    )
    plan = planner.proposal(
        json.dumps(
            dict(summary="Scheduler classification", tests=[], tasks=[], conflicts="")
        ),
        data,
    )
    body = planner.render(plan)
    assert "| 1 | [Scheduler C++](" in body
    assert "| 2 | [Scheduler Python](" in body
    assert "Native CPU CI" in body and "no task recommended" in body


def test_mla_plan_includes_native_gpu_ci_and_requires_serving(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
    monkeypatch.setenv("PR_NUMBER", "2048")
    source = "tokenspeed-mla/python/tokenspeed_mla/mla_decode_fp8.py"
    test = "tokenspeed-mla/tests/test_mla_decode.py"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            stdout=f"{test}\n" if args[1] == "ls-files" else f"{source}\n"
        ),
    )
    data = planner.context(REPO, "a" * 40, "b" * 40)
    assert [c["workflow"] for c in data["native_checks"]] == [
        "nvidia-kernel-library-tests.yml"
    ]
    assert planner.native_checks(["tokenspeed-kernel/test/amd/test_mla.py"]) == []
    assert planner._matches_path(
        "test/amd/test_mla.py", ["test/**", "!test/amd/**", "test/amd/test_mla.py"]
    )
    response = dict(
        summary="MLA decode compilation and serving compatibility",
        tests=[
            dict(path=test, label="MLA GPU UT", reason="Numerics and CUDA Graph replay")
        ],
        tasks=[],
        conflicts="",
    )
    with pytest.raises(planner.CoverageError, match="serving CI task"):
        planner.proposal(json.dumps(response), data)
    config = (
        "test/ci/eval/kimi-k3-nvfp4-dp16-four-node-evalscope-aime26-gb300-slurm.yaml"
    )
    task = next(t for t in data["catalog"] if t["config"] == config)
    response["tasks"] = [
        dict(
            config=config,
            runner=task["slurm_runners"]["gb300"][0],
            cluster="gb300",
            label="K3 serving e2e",
            reason="Target and drafter MLA decode with FP8 KV and EAGLE3",
        )
    ]
    plan = planner.proposal(json.dumps(response), data)
    body = planner.render(plan)
    assert "| 1 | [NVIDIA kernel libraries](" in body
    assert "Native GPU CI" in body and "K3 serving e2e" in body
    # Documentation still triggers native CI, without forcing model serving.
    data["paths"] = ["tokenspeed-mla/README.md"]
    response["tasks"] = []
    planner.proposal(json.dumps(response), data)
