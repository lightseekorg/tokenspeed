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

import copy

import pytest
import retry_ci

REPOSITORY = "lightseekorg/tokenspeed"


@pytest.mark.parametrize(
    "value",
    [
        "34550905154",
        " 34550905154 ",
        "https://github.com/lightseekorg/tokenspeed/actions/runs/34550905154/",
    ],
)
def test_parse_run_url_and_id(value):
    assert retry_ci.parse_run_id(value, REPOSITORY) == 34550905154


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "123; echo unsafe",
        "https://github.com/other/repo/actions/runs/123",
        "https://github.com.evil.invalid/lightseekorg/tokenspeed/actions/runs/123",
        "https://github.com/lightseekorg/tokenspeed/actions/runs/123/jobs/456",
    ],
)
def test_reject_invalid_run(value):
    with pytest.raises(ValueError):
        retry_ci.parse_run_id(value, REPOSITORY)


@pytest.fixture
def api(monkeypatch):
    run = {
        "status": "completed",
        "path": ".github/workflows/slurm-dispatch.yml",
        "event": "workflow_dispatch",
        "run_attempt": 2,
    }
    jobs = [{"labels": ["slurm-dispatch"], "conclusion": "failure"}]
    artifacts = [
        {"name": "slurm-123-1", "id": 11, "expired": False},
        {"name": "slurm-123-2", "id": 12, "expired": False},
    ]
    calls = []

    def request(endpoint):
        calls.append(endpoint)
        if endpoint.endswith("/123"):
            return copy.deepcopy(run)
        if "/attempts/2/jobs?" in endpoint:
            return {"jobs": jobs}
        if "/artifacts?" in endpoint:
            return {"artifacts": artifacts}
        raise AssertionError(endpoint)

    monkeypatch.setattr(retry_ci, "gh_json", request)
    return run, jobs, artifacts, calls


def test_resolve_uses_current_attempt_and_recorded_coordinator(api):
    _, jobs, _, calls = api
    jobs[0]["labels"] = ["self-hosted", "slurm-dispatch-gb300"]
    result = retry_ci.resolve_run("123", REPOSITORY)
    assert result == {
        "source_run_id": "123",
        "source_attempt": "2",
        "artifact_name": "slurm-123-2",
        "artifact_id": "12",
        "coordinator": "slurm-dispatch-gb300",
    }
    assert any("/attempts/2/jobs" in endpoint for endpoint in calls)
    assert not any("/attempts/1/jobs" in endpoint for endpoint in calls)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("running", "finish"),
        ("unsupported", "Slurm Dispatch"),
        ("expired", "expired"),
        ("missing", "missing"),
        ("no_coordinator", "coordinator"),
        ("two_coordinators", "coordinator"),
    ],
)
def test_resolve_fails_without_replay_context(api, mutation, error):
    run, jobs, artifacts, _ = api
    if mutation == "running":
        run["status"] = "in_progress"
    elif mutation == "unsupported":
        run["path"] = ".github/workflows/k8s-dispatch.yml"
    elif mutation == "expired":
        artifacts[-1]["expired"] = True
    elif mutation == "missing":
        artifacts.pop()
    elif mutation == "no_coordinator":
        jobs.clear()
    elif mutation == "two_coordinators":
        jobs.append({"labels": ["slurm-dispatch-gb300"]})
    with pytest.raises(ValueError, match=error):
        retry_ci.resolve_run("123", REPOSITORY)


def test_resolve_rejects_attempt_race(api, monkeypatch):
    original = retry_ci.gh_json
    reads = 0

    def request(endpoint):
        nonlocal reads
        result = original(endpoint)
        if endpoint.endswith("/123"):
            reads += 1
            if reads == 2:
                result["run_attempt"] = 3
        return result

    monkeypatch.setattr(retry_ci, "gh_json", request)
    with pytest.raises(ValueError, match="changed"):
        retry_ci.resolve_run("123", REPOSITORY)


def test_paginated_job_lookup(monkeypatch):
    calls = []

    def request(endpoint):
        calls.append(endpoint)
        return {
            "jobs": (
                [{"labels": []}] * 100
                if endpoint.endswith("page=1")
                else [{"labels": ["slurm-dispatch"]}]
            )
        }

    monkeypatch.setattr(retry_ci, "gh_json", request)
    assert len(retry_ci.gh_items("example/jobs", "jobs")) == 101
    assert calls == [
        "example/jobs?per_page=100&page=1",
        "example/jobs?per_page=100&page=2",
    ]


def test_main_writes_only_validated_outputs(api, tmp_path):
    output = tmp_path / "output"
    assert (
        retry_ci.main(
            [
                "--source-run",
                "123",
                "--repository",
                REPOSITORY,
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["artifact_id"] == "12"
    assert values["coordinator"] == "slurm-dispatch"
