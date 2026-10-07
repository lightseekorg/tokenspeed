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

"""Prepare, generate, and publish the PR CI plan in separate credential scopes."""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from pr_ci_plan import context, proposal, render, source_url
from pr_ci_state import marker


def _command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as error:
        # Keep provider configuration and API response bodies out of public logs.
        status = re.search(r"HTTP [0-9]{3}", error.stderr or "")
        detail = status[0] if status else f"exit {error.returncode}"
        raise SystemExit(
            f"CI planning command failed: {' '.join(args[:2])} ({detail})."
        ) from None


def _check_bot() -> None:
    _command("gh", "auth", "status")
    if _command("gh", "api", "user", "--jq", ".login").strip() != "lightseek-bot":
        raise SystemExit("GitHub authentication must use lightseek-bot.")


def prepare(root: Path) -> None:
    if _command("git", "rev-parse", "HEAD").strip() != os.environ["PR_HEAD_SHA"]:
        raise SystemExit("Checkout differs from the reviewed commit.")
    _check_bot()
    rows = _command(
        "gh",
        "api",
        "--paginate",
        f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/organization-variables?per_page=30",
        "--jq",
        ".variables[] | @json",
    )
    variables = {
        row["name"]: row["value"]
        for row in (json.loads(line) for line in rows.splitlines() if line.strip())
    }
    url = variables.get("KIMI_API_URL", "")
    model = variables.get("KIMI_MODEL", "")
    if not url or not model:
        raise SystemExit("Set the KIMI_API_URL and KIMI_MODEL organization variables.")
    for value in (url, model):
        mask = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::add-mask::{mask}", flush=True)

    home = Path(os.environ["RUNNER_TEMP"], "ci-assist-home")
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    with root.joinpath("pr.diff").open("w") as diff:
        subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                f"{os.environ['PR_BASE_SHA']}...{os.environ['PR_HEAD_SHA']}",
            ],
            check=True,
            stdout=diff,
        )
    config = f"""default_model = "planner"
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
    home.joinpath("config.toml").write_text(config)
    shutil.copyfile(Path(__file__).with_name("pr-ci-planner.md"), root / "planner.md")
    data = context(
        Path.cwd(),
        os.environ["PR_HEAD_SHA"],
        os.environ["PR_BASE_SHA"],
    )
    pr = json.loads(
        _command(
            "gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/pulls/{data['pr']}"
        )
    )
    if (
        pr["head"]["sha"] != data["head"]
        or pr["base"]["sha"] != data["base"]
        or pr["head"]["repo"]["full_name"] != data["repository"]
    ):
        raise SystemExit("The PR source changed; retry with a fresh event.")
    data["mergeable"] = pr["mergeable"]
    data["title"] = pr["title"]
    data["body"] = pr["body"] or ""
    root.joinpath("context.json").write_text(json.dumps(data, indent=2))


def _model_body(root: Path) -> str:
    events = [
        json.loads(line)
        for line in root.joinpath("events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # CLI 2.1.1 appends a session resume hint after the final assistant message.
    messages = [event for event in events if event.get("role") != "meta"]
    final = messages[-1] if messages else {}
    body = final.get("content", "")
    if (
        final.get("role") != "assistant"
        or final.get("tool_calls")
        or not isinstance(body, str)
        or not body.strip()
    ):
        raise SystemExit("The planner did not produce a final response.")
    _check_public_output(body, root)
    return body.strip()


def _check_public_output(body: str, root: Path, *, source_links: bool = False) -> None:
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
    scanned = body
    for task in data["catalog"]:
        for field in ("config", "name"):
            scanned = scanned.replace(task[field], "")
    model = config["models"]["planner"]["model"]
    links = body
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
        links = re.sub(
            r"https?://[^\s)<>]+",
            lambda match: "SOURCE_LINK" if match[0] in allowed else match[0],
            links,
        )
    if (
        any(value and value in body for value in private)
        or (model and model in scanned)
        or re.search(r"https?://|github\.com|\bwww\.", links)
        or re.search(
            r"\b(?:sk-|ghp_|gho_|github_pat_)|"
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|/(?:home|root|tmp|proc)/",
            body,
        )
    ):
        raise SystemExit(
            "CI plan failed the public-output check; no plan was published."
        )
    if len(body) > 60000:
        raise SystemExit("CI plan exceeds the comment size limit.")


def _generate(root: Path, source: str, correction: str = "") -> str:
    # A neutral working directory avoids loading the PR's CLI/MCP configuration.
    with (
        root.joinpath("events.jsonl").open("w") as events,
        root.joinpath("cli.stderr").open("w") as errors,
    ):
        result = subprocess.run(
            [
                "timeout",
                "120" if correction else "600",
                "kimi",
                "--agent-file",
                str(root / "planner.md"),
                "--output-format",
                "stream-json",
                "-p",
                f"Plan CI coverage using {root}/context.json and {root}/pr.diff. "
                f"Source root: {source}. Read relevant callers and CI task specs. "
                "Return only the JSON schema in your instructions. " + correction,
            ],
            cwd=root,
            stdout=events,
            stderr=errors,
        )
    if result.returncode:
        raise SystemExit("CI planning failed or timed out; no plan was published.")
    return _model_body(root)


def plan(root: Path) -> None:
    if not os.environ.get("KIMI_API_KEY"):
        raise SystemExit("Set the KIMI_API_KEY organization secret.")
    source = str(Path.cwd())
    data = json.loads(root.joinpath("context.json").read_text())
    raw = _generate(root, source)
    for attempt in range(2):
        try:
            plan = proposal(raw, data)
            break
        except ValueError as error:
            if attempt:
                # Validation errors are fixed text, never raw model content.
                raise SystemExit(f"Invalid CI proposal: {error}") from None
            root.joinpath("previous-response.txt").write_text(raw)
            raw = _generate(
                root,
                source,
                f"The previous response in {root}/previous-response.txt failed validation: {error}. "
                "Correct its JSON once. Use a summary under 200 characters, labels under 60, "
                "reasons under 120, and no @ mentions. Preserve the selected scope and use only catalog identifiers.",
            )
    # Check decoded text before presentation escaping can change its spelling.
    editorial = [plan["summary"], plan["conflicts"]]
    for item in [*plan["tests"], *plan["tasks"]]:
        editorial += [item["label"], item["reason"]]
    _check_public_output("\n".join(editorial), root)
    body = render(plan)
    _check_public_output(body, root, source_links=True)
    data = json.loads(root.joinpath("context.json").read_text())
    metadata = {k: data[k] for k in ("version", "repository", "pr", "head", "base")}
    metadata.update(
        run=int(os.environ["GITHUB_RUN_ID"]),
        tests=[t["path"] for t in plan["tests"]],
        tasks=[
            {k: t[k] for k in ("config", "runner", "cluster")} for t in plan["tasks"]
        ],
    )
    root.joinpath("comment.md").write_text(body + marker("plan", metadata))


def publish(root: Path) -> None:
    _check_bot()
    repo = os.environ["GITHUB_REPOSITORY"]
    number = os.environ["PR_NUMBER"]
    if (
        _command(
            "gh", "repo", "view", repo, "--json", "visibility", "--jq", ".visibility"
        ).strip()
        != "PUBLIC"
    ):
        raise SystemExit("The plan destination must be a public repository.")
    head = _command(
        "gh",
        "pr",
        "view",
        number,
        "--repo",
        repo,
        "--json",
        "headRefOid",
        "--jq",
        ".headRefOid",
    ).strip()
    live = json.loads(_command("gh", "api", f"repos/{repo}/pulls/{number}"))
    if (
        head != os.environ["PR_HEAD_SHA"]
        or live["base"]["sha"]
        != json.loads(root.joinpath("context.json").read_text())["base"]
    ):
        print("PR head changed; skipping the obsolete plan.")
        return
    comment_url = _command(
        "gh",
        "pr",
        "comment",
        number,
        "--repo",
        repo,
        "--body-file",
        str(root / "comment.md"),
    ).strip()
    comment_id = comment_url.rsplit("issuecomment-", 1)[-1]
    published = _command(
        "gh", "api", f"repos/{repo}/issues/comments/{comment_id}", "--jq", ".body"
    )
    root.joinpath("published.md").write_text(published)
    # gh's --jq output adds a newline; compare after trimming it.
    if published.rstrip() != root.joinpath("comment.md").read_text().rstrip():
        raise SystemExit("Published plan differs from the checked body.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "publish"))
    args = parser.parse_args()
    stages = {"prepare": prepare, "plan": plan, "publish": publish}
    try:
        stages[args.stage](Path(os.environ["RUNNER_TEMP"], "ci-plan"))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        # Configuration and raw model output must not appear in public tracebacks.
        raise SystemExit(
            f"CI planning {args.stage} failed; raw output withheld."
        ) from None
