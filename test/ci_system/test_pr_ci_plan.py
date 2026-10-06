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
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
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
