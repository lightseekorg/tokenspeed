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

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from kernel_benchmark_comment import (
    BOT_LOGIN,
    COMMENT_MARKER,
    MAX_COMMENT_CHARS,
    MAX_COMPARISONS,
    REPORT_ARCHIVE_PATH,
    ValidationError,
    _SafeRedirectHandler,
    download_report,
    extract_report,
    publish,
    render_comment,
    validate_report_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "lightseekorg/tokenspeed"
RUN_ID = 12345
RUN_ATTEMPT = 2
PULL_REQUEST = 77
TARGET_SHA = "1" * 40
BASE_SHA = "2" * 40
CANDIDATE_SHA = "3" * 40


def comparison(
    benchmark_id: str = "gemm.bmm/gluon/b12-m1-n512-k128-bfloat16/v1",
    classification: str = "regression",
    delta_percent: float | None = 12.5,
) -> dict[str, Any]:
    return {
        "id": benchmark_id,
        "classification": classification,
        "base_median_us": 10.0,
        "candidate_median_us": 11.25,
        "delta_us": 1.25,
        "delta_percent": delta_percent,
        "detail": "threshold exceeded",
    }


def report(
    *,
    comparisons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entries = [comparison()] if comparisons is None else comparisons
    return {
        "schema_version": 1,
        "suite_id": "amd-kernel-registrations",
        "repository": REPOSITORY,
        "pull_request_number": PULL_REQUEST,
        "github_run_id": RUN_ID,
        "github_run_attempt": RUN_ATTEMPT,
        "target_sha": TARGET_SHA,
        "merge_base_sha": BASE_SHA,
        "candidate_sha": CANDIDATE_SHA,
        "merge_sha": None,
        "comparisons": entries,
    }


def report_bytes(value: dict[str, Any] | None = None) -> bytes:
    return json.dumps(report() if value is None else value).encode("utf-8")


def artifact_zip(value: dict[str, Any] | None = None, **extra: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(REPORT_ARCHIVE_PATH, report_bytes(value))
        for name, contents in extra.items():
            archive.writestr(name, contents)
    return buffer.getvalue()


class FakeGitHub:
    def __init__(
        self,
        *,
        report_archive: bytes | None = None,
        pull_state: str = "open",
        head_sha: str = CANDIDATE_SHA,
        base_sha: str = TARGET_SHA,
        comments: list[dict[str, Any]] | None = None,
    ) -> None:
        self.report_archive = report_archive
        self.pull_state = pull_state
        self.head_sha = head_sha
        self.base_sha = base_sha
        self.comments = list(comments or [])
        self.reads: list[str] = []
        self.writes: list[tuple[str, str, Any]] = []

    def get_json(self, path: str) -> Any:
        self.reads.append(path)
        if "/actions/runs/" in path and "/artifacts?" in path:
            artifacts = []
            if self.report_archive is not None:
                artifacts.append(
                    {
                        "id": 9001,
                        "name": (
                            "pr-test-kernel-benchmark-amd-gfx950-"
                            f"amd-mi355-1gpu-bench-{RUN_ID}-{RUN_ATTEMPT}"
                        ),
                        "expired": False,
                    }
                )
            return {"total_count": len(artifacts), "artifacts": artifacts}
        if path == f"/repos/{REPOSITORY}/pulls/{PULL_REQUEST}":
            return {
                "state": self.pull_state,
                "head": {"sha": self.head_sha},
                "base": {"sha": self.base_sha},
            }
        if f"/repos/{REPOSITORY}/issues/{PULL_REQUEST}/comments?" in path:
            return self.comments
        raise AssertionError(f"unexpected GET {path}")

    def get_bytes(self, path: str, limit: int) -> bytes:
        self.reads.append(path)
        assert path == f"/repos/{REPOSITORY}/actions/artifacts/9001/zip"
        assert len(self.report_archive or b"") <= limit
        assert self.report_archive is not None
        return self.report_archive

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.writes.append(("POST", path, payload))
        return {"id": 1}

    def patch_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.writes.append(("PATCH", path, payload))
        return {"id": int(path.rsplit("/", 1)[1])}

    def delete(self, path: str) -> None:
        self.writes.append(("DELETE", path, None))


def bot_comment(
    identifier: int, *, run_id: int = 1, attempt: int = 1
) -> dict[str, Any]:
    return {
        "id": identifier,
        "body": (
            f"{COMMENT_MARKER}\n"
            "<!-- tokenspeed-kernel-benchmark-comment-metadata:v1 "
            f"run_id={run_id} run_attempt={attempt} -->"
        ),
        "user": {"login": BOT_LOGIN, "type": "Bot"},
    }


def test_validate_report_accepts_required_fields_and_ignores_extensions():
    value = report()
    del value["suite_id"]
    del value["merge_sha"]
    del value["comparisons"][0]["detail"]
    value["future_report_field"] = {"untrusted": True}
    value["comparisons"][0]["future_comparison_field"] = [1, 2, 3]

    validated = validate_report_bytes(report_bytes(value))

    assert validated["schema_version"] == 1
    assert validated["comparisons"][0]["classification"] == "regression"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["comparisons"][0].update(classification="arbitrary"),
            "classification",
        ),
        (
            lambda value: value["comparisons"][0].update(classification=[]),
            "classification",
        ),
        (lambda value: value["comparisons"][0].update(delta_percent=[]), "number"),
    ],
)
def test_validate_report_rejects_malformed_fields(mutation, message):
    value = report()
    mutation(value)

    with pytest.raises(ValidationError, match=message):
        validate_report_bytes(report_bytes(value))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_report_rejects_non_finite_numbers(value):
    malformed = report()
    malformed["comparisons"][0]["delta_us"] = value

    with pytest.raises(ValidationError, match="finite"):
        validate_report_bytes(report_bytes(malformed))


def test_validate_report_enforces_entry_bound():
    too_many = report(comparisons=[comparison()] * (MAX_COMPARISONS + 1))
    with pytest.raises(ValidationError, match="entry limit"):
        validate_report_bytes(report_bytes(too_many))


def test_extract_report_reads_only_published_comparison_from_the_artifact():
    assert extract_report(artifact_zip())["candidate_sha"] == CANDIDATE_SHA
    archive = artifact_zip(
        **{
            "comparison.json": b"not the published report",
            "run-me.sh": b"exit 1",
        }
    )

    assert extract_report(archive)["candidate_sha"] == CANDIDATE_SHA


def test_extract_report_ignores_artifact_without_a_published_comparison():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("comparison.json", report_bytes())

    assert extract_report(buffer.getvalue()) is None


def test_artifact_redirect_does_not_forward_the_github_token():
    request = urllib.request.Request(
        "https://api.github.com/repos/example/project/actions/artifacts/1/zip",
        headers={"Authorization": "Bearer secret", "Accept": "application/zip"},
    )

    redirected = _SafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://artifact-storage.example/report.zip",
    )

    assert redirected is not None
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Accept") == "application/zip"


def test_download_report_treats_a_missing_artifact_as_a_normal_skip():
    client = FakeGitHub()

    assert download_report(client, REPOSITORY, RUN_ID, RUN_ATTEMPT) == (None, False)
    assert client.writes == []


def test_download_report_treats_an_artifact_without_a_report_as_a_normal_skip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result.json", b"{}")
    client = FakeGitHub(report_archive=buffer.getvalue())

    assert download_report(client, REPOSITORY, RUN_ID, RUN_ATTEMPT) == (None, True)
    assert client.writes == []


def test_render_comment_uses_a_fixed_bounded_template_and_escapes_ids():
    entries = [
        comparison(
            benchmark_id=f"<script>@team|case-{index}",
            delta_percent=float(index),
        )
        for index in range(12)
    ]
    value = report(comparisons=entries)
    value["summary"] = "untrusted summary"
    value["merge_base_sha"] = "`<script>@team"

    body = render_comment(value, RUN_ID, RUN_ATTEMPT)

    assert body.startswith(COMMENT_MARKER)
    assert "<script>" not in body
    assert "@team" not in body
    assert "Showing 10 of 12 entries." in body
    assert value["summary"] not in body
    assert "Regressions: **12**" in body
    assert "Confidence interval" not in body
    assert "actions/runs/12345/attempts/2" in body

    too_large = report(comparisons=[comparison(benchmark_id="x" * MAX_COMMENT_CHARS)])
    with pytest.raises(ValidationError, match="comment exceeds"):
        render_comment(too_large, RUN_ID, RUN_ATTEMPT)


@pytest.mark.parametrize(
    ("classifications", "expected"),
    [
        (["regression", "invalid"], "Invalid measurement"),
        (["regression"], "Regression detected"),
        (["missing"], "Inconclusive"),
        (["added"], "Baseline unavailable"),
        (["improvement", "within_budget"], "Passed"),
        ([], "Baseline unavailable"),
    ],
)
def test_render_comment_derives_status_from_comparisons(classifications, expected):
    entries = [
        comparison(benchmark_id=f"case-{index}", classification=classification)
        for index, classification in enumerate(classifications)
    ]

    body = render_comment(report(comparisons=entries), RUN_ID, RUN_ATTEMPT)

    assert f"**Result:** {expected}" in body


def test_publish_creates_a_comment_for_the_current_open_pull_request():
    client = FakeGitHub(report_archive=artifact_zip())

    result = publish(client, REPOSITORY, RUN_ID, RUN_ATTEMPT, PULL_REQUEST)

    assert result == "created"
    assert client.writes[0][0] == "POST"
    assert client.writes[0][1] == f"/repos/{REPOSITORY}/issues/{PULL_REQUEST}/comments"
    assert COMMENT_MARKER in client.writes[0][2]["body"]


def test_publish_leaves_comments_unchanged_when_the_benchmark_task_is_missing():
    client = FakeGitHub(
        comments=[
            bot_comment(10, run_id=RUN_ID - 1),
            bot_comment(20, run_id=RUN_ID + 1),
        ]
    )

    result = publish(client, REPOSITORY, RUN_ID, RUN_ATTEMPT, PULL_REQUEST)

    assert result == "task_missing"
    assert client.writes == []


def test_publish_removes_obsolete_comments_when_task_has_no_report():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result.json", b"{}")
    client = FakeGitHub(
        report_archive=buffer.getvalue(),
        comments=[
            bot_comment(10, run_id=RUN_ID - 1),
            bot_comment(20, run_id=RUN_ID + 1),
        ],
    )

    result = publish(client, REPOSITORY, RUN_ID, RUN_ATTEMPT, PULL_REQUEST)

    assert result == "artifact_missing"
    assert client.writes == [
        ("DELETE", f"/repos/{REPOSITORY}/issues/comments/10", None)
    ]


@pytest.mark.parametrize(
    ("pull_state", "head_sha", "base_sha", "expected", "removes_old_comment"),
    [
        ("closed", CANDIDATE_SHA, TARGET_SHA, "pull_request_closed", False),
        ("open", "4" * 40, TARGET_SHA, "candidate_stale", True),
        ("open", CANDIDATE_SHA, "4" * 40, "target_stale", True),
    ],
)
def test_publish_skips_closed_or_stale_pull_requests(
    pull_state, head_sha, base_sha, expected, removes_old_comment
):
    client = FakeGitHub(
        report_archive=artifact_zip(),
        pull_state=pull_state,
        head_sha=head_sha,
        base_sha=base_sha,
        comments=[bot_comment(10)],
    )

    assert publish(client, REPOSITORY, RUN_ID, RUN_ATTEMPT, PULL_REQUEST) == expected
    expected_writes = (
        [("DELETE", f"/repos/{REPOSITORY}/issues/comments/10", None)]
        if removes_old_comment
        else []
    )
    assert client.writes == expected_writes
    assert any("/comments?" in path for path in client.reads) is removes_old_comment


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "attacker/repository"),
        ("pull_request_number", PULL_REQUEST + 1),
        ("github_run_id", RUN_ID + 1),
        ("github_run_attempt", RUN_ATTEMPT + 1),
    ],
)
def test_publish_rejects_artifact_identity_mismatches(field, value):
    malformed = report()
    malformed[field] = value
    client = FakeGitHub(report_archive=artifact_zip(malformed))

    with pytest.raises(ValidationError, match="does not match the triggering run"):
        publish(client, REPOSITORY, RUN_ID, RUN_ATTEMPT, PULL_REQUEST)
    assert client.writes == []


def test_publish_updates_one_bot_comment_and_deletes_older_duplicates():
    other_author = {
        "id": 30,
        "body": COMMENT_MARKER,
        "user": {"login": "contributor", "type": "User"},
    }
    unrelated_bot = {
        "id": 31,
        "body": "another automated comment",
        "user": {"login": BOT_LOGIN, "type": "Bot"},
    }
    client = FakeGitHub(
        report_archive=artifact_zip(),
        comments=[bot_comment(10), bot_comment(20), other_author, unrelated_bot],
    )

    result = publish(client, REPOSITORY, RUN_ID, RUN_ATTEMPT, PULL_REQUEST)

    assert result == "updated"
    assert [write[:2] for write in client.writes] == [
        ("PATCH", f"/repos/{REPOSITORY}/issues/comments/20"),
        ("DELETE", f"/repos/{REPOSITORY}/issues/comments/10"),
    ]


def test_publish_does_not_overwrite_a_newer_completed_run():
    client = FakeGitHub(
        report_archive=artifact_zip(),
        comments=[bot_comment(10, run_id=RUN_ID + 1, attempt=1)],
    )

    result = publish(client, REPOSITORY, RUN_ID, RUN_ATTEMPT, PULL_REQUEST)

    assert result == "newer_result_present"
    assert client.writes == []


def test_comment_workflow_has_a_minimal_trusted_contract():
    path = REPO_ROOT / ".github/workflows/kernel-benchmark-comment.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)

    assert triggers == {
        "workflow_run": {
            "workflows": ["PR Test AMD"],
            "types": ["completed"],
        }
    }
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "write",
    }
    job = workflow["jobs"]["publish"]
    assert "workflow_run.event == 'pull_request'" in job["if"]
    checkout = next(
        step
        for step in job["steps"]
        if step.get("name") == "Checkout trusted publisher"
    )
    assert checkout["with"] == {
        "ref": "${{ github.event.repository.default_branch }}",
        "persist-credentials": False,
    }
    publisher = next(
        step for step in job["steps"] if step.get("name") == "Publish benchmark report"
    )
    assert publisher["env"]["EXPECTED_PULL_REQUEST"] == (
        "${{ github.event.workflow_run.pull_requests[0].number }}"
    )
    assert publisher["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert "test/ci_system/kernel_benchmark_comment.py" in publisher["run"]
    assert all("download-artifact" not in step.get("uses", "") for step in job["steps"])
