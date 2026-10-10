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

"""Shared helpers for the PR CI assist, repair, and planning scripts."""

import base64
import json
import os
import re
import subprocess
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pr_ci_plan import source_url
from pr_ci_state import BOT


def upsert_comment(
    command: Callable[..., str],
    api: Callable[[str], dict],
    repo: str,
    number: str,
    body: str,
    comment_id: int | None,
) -> dict:
    """Create or update a PR comment and return the live record.

    With ``comment_id`` None a new comment is created, notifying subscribers;
    otherwise that comment is patched in place, producing no notification.
    """
    if comment_id is None:
        url = command("gh", "pr", "comment", number, "--repo", repo, "--body", body)
        comment_id = int(url.rsplit("issuecomment-", 1)[-1])
    else:
        token = command("gh", "auth", "token", "--hostname", "github.com").strip()
        request = Request(
            f"https://api.github.com/repos/{repo}/issues/comments/{comment_id}",
            data=json.dumps({"body": body}).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
            method="PATCH",
        )
        with urlopen(request, timeout=30) as response:
            response.read()
    return api(f"issues/comments/{comment_id}")


def run_command(
    args: tuple[str, ...], *, cwd: Path | None, strip: bool, failure: str | None
) -> str:
    """Run a checked command and return its stdout.

    With ``failure`` None, a non-zero exit propagates CalledProcessError with
    the raw captured output. Otherwise the failure is sanitized into a
    SystemExit carrying only the given public label and an HTTP or exit status,
    keeping provider configuration and API response bodies out of public logs.
    """
    try:
        output = subprocess.run(
            args, cwd=cwd, check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as error:
        if failure is None:
            raise
        status = re.search(r"HTTP [0-9]{3}", error.stderr or "")
        detail = status[0] if status else f"exit {error.returncode}"
        raise SystemExit(f"{failure}: {' '.join(args[:2])} ({detail}).") from None
    return output.strip() if strip else output


def require_bot(
    command: Callable[..., str], *, error: type[BaseException], message: str
) -> None:
    """Require the gh CLI to be authenticated as the bot account."""
    command("gh", "auth", "status")
    if json.loads(command("gh", "api", "user"))["login"] != BOT:
        raise error(message)


def require_public_repo(
    command: Callable[..., str],
    repo: str,
    *,
    error: type[BaseException],
    message: str,
) -> None:
    """Require the destination repository to be public."""
    if (
        command(
            "gh", "repo", "view", repo, "--json", "visibility", "--jq", ".visibility"
        ).strip()
        != "PUBLIC"
    ):
        raise error(message)


def org_variables(command: Callable[..., str], repo: str) -> dict[str, str]:
    """Fetch the repository's Actions organization variables by name."""
    rows = json.loads(
        command(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/actions/organization-variables?per_page=100",
        )
    )
    return {v["name"]: v["value"] for page in rows for v in page["variables"]}


def mask_secret(value: str) -> None:
    """Mask a secret in GitHub Actions log output."""
    escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::add-mask::{escaped}", flush=True)


def planner_config(url: str, model: str) -> str:
    """Render the bounded CLI configuration for the planner model."""
    return f"""default_model = "planner"
telemetry = false
[providers.planner]
type = "openai"
base_url = {json.dumps(url)}
api_key_env = "KIMI_API_KEY"
[models.planner]
provider = "planner"
model = {json.dumps(model)}
max_context_size = 262144
capabilities = ["thinking", "tool_use"]
"""


def screen_public_output(
    text: str,
    *,
    substitute: Iterable[tuple[str, str]] = (),
    extra_private: Iterable[str] = (),
    strip: Iterable[str] = (),
    stripped_private: Iterable[str] = (),
    allowed_urls: Collection[str] = (),
    url_indicators: str,
    secret_indicators: str,
    max_length: int | None,
) -> str | None:
    """Screen text bound for public comments; return the rejection or None.

    ``substitute`` regex replacements hide approved source links before every
    scan. ``extra_private`` literal secrets are rejected in the substituted
    text. ``strip`` public identifiers are removed only for the
    ``stripped_private`` scan, so a public catalog identifier that embeds a
    private name stays publishable without hiding that name in free text.
    ``allowed_urls`` exact links are permitted by the ``url_indicators`` scan.
    Returns "private" for a leaked indicator, "length" over ``max_length``.
    """
    scanned = text
    for pattern, replacement in substitute:
        scanned = re.sub(pattern, replacement, scanned)
    if any(value and value in scanned for value in extra_private):
        return "private"
    stripped = scanned
    for literal in strip:
        stripped = stripped.replace(literal, "")
    if any(value and value in stripped for value in stripped_private):
        return "private"
    links = scanned
    if allowed_urls:
        links = re.sub(
            r"https?://[^\s)<>]+",
            lambda match: "ALLOWED_LINK" if match[0] in allowed_urls else match[0],
            links,
        )
    if re.search(url_indicators, links) or re.search(secret_indicators, scanned):
        return "private"
    if max_length is not None and len(scanned) > max_length:
        return "length"
    return None


def check_model_output(
    body: str, root: Path, *, source_links: bool = False, max_length: int = 60000
) -> None:
    """Reject planner or repair text leaking provider or runner details."""
    # isort and ruff classify tomllib's stdlib status differently.
    import tomllib

    config = tomllib.loads(
        Path(os.environ["KIMI_CODE_HOME"], "config.toml").read_text()
    )
    url = config["providers"]["planner"]["base_url"]
    key = os.environ["KIMI_API_KEY"]
    private = [
        key,
        key[:8],
        base64.b64encode(key.encode()).decode(),
        url,
        urlparse(url).hostname,
        os.environ["RUNNER_TEMP"],
        os.environ["GITHUB_WORKSPACE"],
        str(Path.cwd()),
    ]
    # Public task identifiers can contain the configured model's name. Allow
    # only exact catalog identifiers; free text still cannot identify it.
    data = json.loads(root.joinpath("context.json").read_text())
    allowed = set()
    if source_links:
        allowed = {source_url(data)} | {
            source_url(data, path)
            for path in [
                *data["test_files"],
                *(t["config"] for t in data["catalog"]),
                *(
                    f".github/workflows/{c['workflow']}"
                    for c in data.get("native_checks", [])
                ),
            ]
        }
    issue = screen_public_output(
        body,
        extra_private=private,
        strip=[task[field] for task in data["catalog"] for field in ("config", "name")],
        stripped_private=[config["models"]["planner"]["model"]],
        allowed_urls=allowed,
        url_indicators=r"https?://|github\.com|\bwww\.",
        secret_indicators=(
            r"\b(?:sk-|ghp_|gho_|github_pat_)|"
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|/(?:home|root|tmp|proc)/"
        ),
        max_length=max_length,
    )
    if issue == "length":
        raise SystemExit("CI plan exceeds the comment size limit.")
    if issue is not None:
        raise SystemExit(
            "CI plan failed the public-output check; no plan was published."
        )


def published_matches(published: str, expected: str) -> bool:
    """Compare a live comment with the reviewed body, ignoring trailing space."""
    return published.rstrip() == expected.rstrip()


def result_status(result: dict, task: dict, sha: str, runner: str) -> str:
    if (
        result.get("source_sha") != sha
        or result.get("config") != task["config"]
        or result.get("task") != task["name"]
        or result.get("runner") != runner
    ):
        return "missing"
    stages = result.get("executed_stages", [])
    if result.get("ok") is True and any(
        stage not in {"install", "server", "cleanup"} for stage in stages
    ):
        return "passed"
    return "failed" if result.get("ok") is False else "missing"


def manifest_task_result(
    target: Path, task: dict, sha: str, *, runner: str | None
) -> str:
    """Evaluate the task's single Slurm manifest row in a downloaded artifact.

    ``runner`` pins the manifest row and its proof; None accepts the runner the
    row recorded. Returns "passed" only with a completed Slurm row whose
    source-bound proof executed real stages.
    """
    manifest = json.loads((target / "manifest.json").read_text())
    rows = [
        r
        for r in manifest
        if r["task"]["config"] == task["config"]
        and (runner is None or r["task"]["runner"] == runner)
    ]
    if len(rows) != 1 or not re.fullmatch(r"[0-9]+", rows[0]["job_id"]):
        return "missing"
    row = rows[0]
    result = {
        "source_sha": sha,
        "config": task["config"],
        **json.loads((target / f"{row['job_id']}-result.json").read_text()),
    }
    status = result_status(
        result, task, sha, runner if runner is not None else row["task"]["runner"]
    )
    if status == "passed" and row["state"] == "COMPLETED" and row["exit_code"] == "0:0":
        return "passed"
    return "failed" if status == "failed" else "missing"
