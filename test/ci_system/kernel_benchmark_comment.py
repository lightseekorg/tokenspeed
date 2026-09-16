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

"""Publish an untrusted kernel benchmark report as a bounded PR comment."""

from __future__ import annotations

import argparse
import collections
import html
import io
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any

SCHEMA_VERSION = 1
REPORT_ARCHIVE_PATH = "published/comparison.json"
COMMENT_MARKER = "<!-- tokenspeed-kernel-benchmark-comment:v1 -->"
COMMENT_METADATA_PREFIX = "tokenspeed-kernel-benchmark-comment-metadata:v1"
COMMENT_METADATA_RE = re.compile(
    rf"<!-- {COMMENT_METADATA_PREFIX} " r"run_id=(\d{1,19}) run_attempt=(\d{1,19}) -->"
)
BOT_LOGIN = "github-actions[bot]"

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_API_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_COMPARISONS = 10_000
MAX_COMMENT_CHARS = 60_000
MAX_COMMENT_ROWS_PER_CLASS = 10
MAX_COMMENT_PAGES = 10
MAX_ARTIFACT_PAGES = 10
PER_PAGE = 100

CLASSIFICATIONS = {
    "regression",
    "improvement",
    "within_budget",
    "inconclusive",
    "added",
    "missing",
    "changed",
    "invalid",
}
CLASSIFICATION_LABELS = {
    "regression": "Regressions",
    "improvement": "Improvements",
    "within_budget": "Within budget",
    "inconclusive": "Inconclusive",
    "added": "Added",
    "missing": "Missing",
    "changed": "Changed",
    "invalid": "Invalid",
}


class PublisherError(RuntimeError):
    """Base class for safe, user-facing publisher failures."""


class ValidationError(PublisherError):
    """Raised when an untrusted report violates its data contract."""


class GitHubError(PublisherError):
    """Raised when a GitHub API operation fails."""


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow artifact redirects without forwarding the GitHub token."""

    def redirect_request(self, request, file, code, message, headers, new_url):
        redirected = super().redirect_request(
            request, file, code, message, headers, new_url
        )
        if redirected is None:
            return None
        old_url = urllib.parse.urlsplit(request.full_url)
        new_url_parts = urllib.parse.urlsplit(new_url)
        old_origin = (old_url.scheme, old_url.netloc)
        new_origin = (new_url_parts.scheme, new_url_parts.netloc)
        if old_origin != new_origin:
            redirected.remove_header("Authorization")
        return redirected


def _read_bounded(response: Any, limit: int, description: str) -> bytes:
    data = response.read(limit + 1)
    if len(data) > limit:
        raise GitHubError(f"{description} exceeds its size limit")
    return data


class GitHubClient:
    """Small GitHub REST client with bounded responses."""

    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        if not token:
            raise GitHubError("GITHUB_TOKEN is required")
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._opener = urllib.request.build_opener(_SafeRedirectHandler())

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        limit: int = MAX_API_RESPONSE_BYTES,
    ) -> bytes:
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self._api_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "tokenspeed-kernel-benchmark-publisher",
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            with self._opener.open(request, timeout=30) as response:
                return _read_bounded(response, limit, "GitHub API response")
        except urllib.error.HTTPError as exc:
            raise GitHubError(
                f"GitHub API request {method} {path} failed with status {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GitHubError(f"GitHub API request {method} {path} failed") from exc

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        raw = self._request(method, path, payload=payload)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GitHubError("GitHub API returned invalid JSON") from exc

    def get_json(self, path: str) -> Any:
        return self.request_json("GET", path)

    def get_bytes(self, path: str, limit: int) -> bytes:
        return self._request("GET", path, limit=limit)

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        return self.request_json("POST", path, payload=payload)

    def patch_json(self, path: str, payload: dict[str, Any]) -> Any:
        return self.request_json("PATCH", path, payload=payload)

    def delete(self, path: str) -> None:
        self._request("DELETE", path)


def _expect_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must be an object")
    return value


def _expect_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{field} must be finite")
    return result


def _expect_optional_number(
    value: Any,
    field: str,
) -> float | None:
    if value is None:
        return None
    return _expect_number(value, field)


def _validate_comparison(value: Any, index: int) -> dict[str, Any]:
    field = f"comparisons[{index}]"
    comparison = _expect_mapping(value, field)

    if not isinstance(comparison["id"], str):
        raise ValidationError(f"{field}.id must be a string")
    classification = comparison["classification"]
    if not isinstance(classification, str) or classification not in CLASSIFICATIONS:
        raise ValidationError(f"{field}.classification is invalid")
    _expect_optional_number(comparison["base_median_us"], f"{field}.base_median_us")
    _expect_optional_number(
        comparison["candidate_median_us"],
        f"{field}.candidate_median_us",
    )
    _expect_optional_number(comparison["delta_us"], f"{field}.delta_us")
    _expect_optional_number(comparison["delta_percent"], f"{field}.delta_percent")
    return comparison


def validate_report_bytes(raw: bytes) -> dict[str, Any]:
    """Parse and validate a comparison report received from an untrusted run."""
    if len(raw) > MAX_REPORT_BYTES:
        raise ValidationError("comparison report exceeds its size limit")
    try:
        report = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValidationError("comparison report is not UTF-8") from exc
    except (RecursionError, ValueError) as exc:
        raise ValidationError("comparison report is not valid JSON") from exc

    report = _expect_mapping(report, "report")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"schema_version must be {SCHEMA_VERSION}")

    comparisons = report["comparisons"]
    if not isinstance(comparisons, list):
        raise ValidationError("comparisons must be an array")
    if len(comparisons) > MAX_COMPARISONS:
        raise ValidationError("comparisons exceeds its entry limit")
    report["comparisons"] = [
        _validate_comparison(value, index) for index, value in enumerate(comparisons)
    ]
    return report


def extract_report(archive: bytes) -> dict[str, Any] | None:
    """Read the published comparison report without extracting artifact files."""
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValidationError("report artifact exceeds its size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as artifact:
            reports = [
                entry
                for entry in artifact.infolist()
                if not entry.is_dir() and entry.filename == REPORT_ARCHIVE_PATH
            ]
            if not reports:
                return None
            if len(reports) != 1:
                raise ValidationError(
                    f"report artifact must contain one {REPORT_ARCHIVE_PATH} file"
                )
            entry = reports[0]
            if entry.flag_bits & 0x1:
                raise ValidationError("comparison report must not be encrypted")
            if entry.file_size > MAX_REPORT_BYTES:
                raise ValidationError("comparison report exceeds its size limit")
            with artifact.open(entry) as report_file:
                raw = report_file.read(MAX_REPORT_BYTES + 1)
    except zipfile.BadZipFile as exc:
        raise ValidationError("report artifact is not a valid ZIP file") from exc
    return validate_report_bytes(raw)


def _artifact_name(run_id: int, run_attempt: int) -> str:
    return (
        "pr-test-kernel-benchmark-amd-gfx950-amd-mi355-1gpu-bench-"
        f"{run_id}-{run_attempt}"
    )


def download_report(
    client: Any, repository: str, run_id: int, run_attempt: int
) -> tuple[dict[str, Any] | None, bool]:
    """Return the report and whether its task artifact exists for this attempt."""
    expected_name = _artifact_name(run_id, run_attempt)
    matches: list[dict[str, Any]] = []
    for page in range(1, MAX_ARTIFACT_PAGES + 1):
        response = client.get_json(
            f"/repos/{repository}/actions/runs/{run_id}/artifacts"
            f"?per_page={PER_PAGE}&page={page}"
        )
        artifacts = response["artifacts"]
        for artifact in artifacts:
            if artifact.get("name") == expected_name:
                matches.append(artifact)
        if len(artifacts) < PER_PAGE:
            break
    else:
        raise GitHubError("artifact list exceeds its page limit")

    if not matches:
        return None, False
    if len(matches) != 1:
        raise ValidationError("benchmark run has duplicate report artifacts")
    artifact = matches[0]
    if artifact.get("expired") is True:
        return None, True
    artifact_id = artifact["id"]
    archive = client.get_bytes(
        f"/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        MAX_ARCHIVE_BYTES,
    )
    return extract_report(archive), True


def _markdown_text(value: str) -> str:
    escaped = html.escape(value.replace("@", "(at)"), quote=False)
    for character in "\\`*_{}[]()#+-.!|":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped.replace("\n", " ").replace("\r", " ")


def _format_number(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.2f}{suffix}"


def _format_time(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _comparison_sort_key(comparison: dict[str, Any]) -> tuple[float, str]:
    delta = comparison["delta_percent"]
    return (abs(float(delta)) if delta is not None else -1.0, comparison["id"])


def _comparison_table(title: str, comparisons: list[dict[str, Any]]) -> list[str]:
    if not comparisons:
        return []
    selected = sorted(comparisons, key=_comparison_sort_key, reverse=True)[
        :MAX_COMMENT_ROWS_PER_CLASS
    ]
    lines = [
        f"### {title}",
        "| Benchmark | Base (us) | Candidate (us) | Change (us) | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(comparison["id"]),
                    _format_time(comparison["base_median_us"]),
                    _format_time(comparison["candidate_median_us"]),
                    _format_number(comparison["delta_us"]),
                    _format_number(comparison["delta_percent"], "%"),
                ]
            )
            + " |"
        )
    if len(comparisons) > len(selected):
        lines.append(f"Showing {len(selected)} of {len(comparisons)} entries.")
    return lines


def _display_status(counts: collections.Counter[str]) -> str:
    if counts["invalid"]:
        return "Invalid measurement"
    if counts["regression"]:
        return "Regression detected"
    if counts["inconclusive"] or counts["changed"] or counts["missing"]:
        return "Inconclusive"
    if sum(counts.values()) == counts["added"]:
        return "Baseline unavailable"
    return "Passed"


def render_comment(report: dict[str, Any], run_id: int, run_attempt: int) -> str:
    """Render only validated values into a fixed Markdown template."""
    counts = collections.Counter(
        comparison["classification"] for comparison in report["comparisons"]
    )
    count_parts = [
        f"{CLASSIFICATION_LABELS[classification]}: **{counts[classification]}**"
        for classification in CLASSIFICATION_LABELS
        if counts.get(classification, 0)
    ]
    if not count_parts:
        count_parts = ["No comparable benchmark entries"]

    repository = report["repository"]
    merge_base = _markdown_text(str(report["merge_base_sha"])[:12])
    candidate = _markdown_text(str(report["candidate_sha"])[:12])
    lines = [
        COMMENT_MARKER,
        f"<!-- {COMMENT_METADATA_PREFIX} run_id={run_id} run_attempt={run_attempt} -->",
        "## AMD kernel benchmark comparison",
        f"**Result:** {_display_status(counts)}",
        "",
        (f"Merge base `{merge_base}` compared with candidate `{candidate}`."),
        " | ".join(count_parts),
    ]
    if not any(
        counts[classification]
        for classification in (
            "regression",
            "improvement",
            "within_budget",
            "inconclusive",
        )
    ):
        lines.extend(["", "The performance gate was not evaluated for this run."])

    regressions = [
        comparison
        for comparison in report["comparisons"]
        if comparison["classification"] == "regression"
    ]
    improvements = [
        comparison
        for comparison in report["comparisons"]
        if comparison["classification"] == "improvement"
    ]
    for title, selected in (
        ("Largest regressions", regressions),
        ("Largest improvements", improvements),
    ):
        table = _comparison_table(title, selected)
        if table:
            lines.extend(["", *table])

    lines.extend(
        [
            "",
            f"[Measurement run](https://github.com/{repository}/actions/runs/{run_id}/attempts/{run_attempt})",
        ]
    )
    body = "\n".join(lines)
    if len(body) > MAX_COMMENT_CHARS:
        raise ValidationError("rendered comment exceeds its length limit")
    return body


def _get_pull_request(client: Any, repository: str, number: int) -> dict[str, Any]:
    return client.get_json(f"/repos/{repository}/pulls/{number}")


def _list_comments(client: Any, repository: str, number: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in range(1, MAX_COMMENT_PAGES + 1):
        comments = client.get_json(
            f"/repos/{repository}/issues/{number}/comments"
            f"?per_page={PER_PAGE}&page={page}"
        )
        result.extend(comments)
        if len(comments) < PER_PAGE:
            return result
    raise GitHubError("pull request comment list exceeds its page limit")


def _owned_comment(comment: dict[str, Any]) -> bool:
    user = comment.get("user")
    return (
        isinstance(comment.get("id"), int)
        and not isinstance(comment.get("id"), bool)
        and comment["id"] > 0
        and isinstance(comment.get("body"), str)
        and COMMENT_MARKER in comment["body"]
        and isinstance(user, dict)
        and user.get("login") == BOT_LOGIN
        and user.get("type") == "Bot"
    )


def _comment_run_key(comment: dict[str, Any]) -> tuple[int, int] | None:
    match = COMMENT_METADATA_RE.search(comment["body"])
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def update_comment(
    client: Any,
    repository: str,
    pull_request_number: int,
    body: str,
    run_id: int,
    run_attempt: int,
) -> str:
    """Create or update the single bot-owned comment for this report family."""
    comments = _list_comments(client, repository, pull_request_number)
    owned = [comment for comment in comments if _owned_comment(comment)]
    current_key = (run_id, run_attempt)
    previous_keys = [key for comment in owned if (key := _comment_run_key(comment))]
    if previous_keys and max(previous_keys) > current_key:
        return "newer_result_present"

    if not owned:
        client.post_json(
            f"/repos/{repository}/issues/{pull_request_number}/comments", {"body": body}
        )
        return "created"

    retained = max(owned, key=lambda comment: comment["id"])
    client.patch_json(
        f"/repos/{repository}/issues/comments/{retained['id']}", {"body": body}
    )
    for duplicate in owned:
        if duplicate["id"] != retained["id"]:
            client.delete(f"/repos/{repository}/issues/comments/{duplicate['id']}")
    return "updated"


def remove_obsolete_comments(
    client: Any,
    repository: str,
    pull_request_number: int,
    run_id: int,
    run_attempt: int,
) -> None:
    """Remove stale benchmark summaries when the current run has no report."""

    current_key = (run_id, run_attempt)
    for comment in _list_comments(client, repository, pull_request_number):
        if not _owned_comment(comment):
            continue
        comment_key = _comment_run_key(comment)
        if comment_key is None or comment_key <= current_key:
            client.delete(f"/repos/{repository}/issues/comments/{comment['id']}")


def publish(
    client: Any,
    repository: str,
    run_id: int,
    run_attempt: int,
    pull_request_number: int,
) -> str:
    """Validate the run artifact and publish it only to its current open PR."""
    report, artifact_exists = download_report(client, repository, run_id, run_attempt)
    if report is None:
        if not artifact_exists:
            return "task_missing"
        remove_obsolete_comments(
            client,
            repository,
            pull_request_number,
            run_id,
            run_attempt,
        )
        return "artifact_missing"
    expected = {
        "repository": repository,
        "github_run_id": run_id,
        "github_run_attempt": run_attempt,
        "pull_request_number": pull_request_number,
    }
    for field, value in expected.items():
        if report[field] != value:
            raise ValidationError(f"report {field} does not match the triggering run")

    pull_request = _get_pull_request(client, repository, pull_request_number)
    if pull_request.get("state") != "open":
        return "pull_request_closed"
    head = pull_request["head"]
    base = pull_request["base"]
    stale_result = None
    if head["sha"] != report["candidate_sha"]:
        stale_result = "candidate_stale"
    elif base["sha"] != report["target_sha"]:
        stale_result = "target_stale"
    if stale_result is not None:
        remove_obsolete_comments(
            client,
            repository,
            pull_request_number,
            run_id,
            run_attempt,
        )
        return stale_result

    body = render_comment(report, run_id, run_attempt)
    return update_comment(
        client,
        repository,
        pull_request_number,
        body,
        run_id,
        run_attempt,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument("--pull-request-number", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        client = GitHubClient(
            os.environ.get("GITHUB_TOKEN", ""),
            os.environ.get("GITHUB_API_URL", "https://api.github.com"),
        )
        result = publish(
            client,
            args.repository,
            args.run_id,
            args.run_attempt,
            args.pull_request_number,
        )
    except PublisherError as exc:
        print(f"Kernel benchmark comment publisher failed: {exc}", file=sys.stderr)
        return 1

    messages = {
        "artifact_missing": "No benchmark report artifact was produced; nothing to publish.",
        "task_missing": "The benchmark task did not run; leaving its previous comment unchanged.",
        "pull_request_closed": "The pull request is closed; skipping benchmark comment.",
        "candidate_stale": "The pull request has a newer head commit; skipping stale results.",
        "target_stale": "The target branch has advanced; skipping stale results.",
        "newer_result_present": "A newer benchmark result is already published; skipping.",
        "created": "Created the kernel benchmark pull request comment.",
        "updated": "Updated the kernel benchmark pull request comment.",
    }
    print(messages[result])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
