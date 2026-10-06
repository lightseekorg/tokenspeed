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

"""Regression coverage for the deterministic floor around model proposals."""

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


def test_shared_change_cannot_drop_baseline(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
    monkeypatch.setenv("PR_NUMBER", "1")
    monkeypatch.setenv("TOKENSPEED_B200_RUNNER_LABEL", "b200v2")
    monkeypatch.setenv(
        "TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", "slurm-gb200,slurm-gb300"
    )
    monkeypatch.setenv("TOKENSPEED_CI_GB300_SLURM_PER_COMMIT_ENABLED", "true")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="python/tokenspeed/__init__.py\n"),
    )
    data = planner.context(REPO, "a" * 40, "b" * 40)
    result = planner.proposal(
        json.dumps({"summary": "Shared runtime change.", "tasks": [], "conflicts": ""}),
        data,
    )
    assert data["catalog"]
    assert any(t["runner"].startswith("slurm-gb200-") for t in data["catalog"])
    assert any(t["runner"].startswith("slurm-gb300-") for t in data["catalog"])
    assert set(data["floor"]) == {planner.task_key(t) for t in data["catalog"]}
    assert {planner.task_key(t) for t in result["tasks"]} == set(data["floor"])


def test_proposal_cannot_invent_runner_or_command():
    task = {
        "config": "test/ci/ut/example.yaml",
        "runner": "b200v2-1gpu",
        "name": "example",
    }
    data = {
        "version": 1,
        "repository": "lightseekorg/tokenspeed",
        "pr": 1,
        "head": "a" * 40,
        "base": "b" * 40,
        "catalog": [task],
        "floor": [],
        "broad_groups": [],
    }
    response = {
        "summary": "Focused validation.",
        "tasks": [
            {
                "config": task["config"],
                "runner": "arbitrary-command",
                "reason": "Changed caller.",
            }
        ],
        "conflicts": "",
    }
    with pytest.raises(ValueError):
        planner.proposal(json.dumps(response), data)
