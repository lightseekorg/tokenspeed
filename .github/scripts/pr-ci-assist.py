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

"""Report CI coverage and dispatch bounded, exact-source diagnostic tasks."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from pr_ci_plan import context, task_key

WORKFLOWS = {
    "amd": "amd-tests.yml",
    "nvidia-arm": "pr-test-nvidia-arm.yml",
    "nvidia-x86": "nvidia-b200-tests.yml",
    "nvidia-gb200-slurm": "nvidia-gb200-tests.yml",
    "nvidia-gb300-slurm": "nvidia-gb300-tests.yml",
}


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def api(repo: str, endpoint: str) -> dict:
    return json.loads(command("gh", "api", f"repos/{repo}/{endpoint}"))


def public_bot(repo: str) -> None:
    command("gh", "auth", "status")
    if command("gh", "api", "user", "--jq", ".login").strip() != "lightseek-bot":
        raise ValueError("The configured GitHub identity must be lightseek-bot.")
    if (
        command(
            "gh", "repo", "view", repo, "--json", "visibility", "--jq", ".visibility"
        ).strip()
        != "PUBLIC"
    ):
        raise ValueError("The destination must be public.")


def pages(repo: str, endpoint: str, field: str) -> list:
    rows = command(
        "gh",
        "api",
        "--paginate",
        f"repos/{repo}/{endpoint}",
        "--jq",
        f".{field}[] | @json",
    )
    return [json.loads(line) for line in rows.splitlines() if line.strip()]


def current_pr(repo: str, number: int) -> dict:
    pr = api(repo, f"pulls/{number}")
    if pr["state"] != "open" or pr["head"]["repo"]["full_name"] != repo:
        raise ValueError("Only open repository-branch PRs are supported.")
    return pr


def event_pr(repo: str) -> int | None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    run = event["workflow_run"]
    if run["head_repository"]["full_name"] != repo:
        return None
    head = run["head_sha"]
    if run["event"] == "workflow_dispatch":
        match = re.fullmatch(
            r"ci-diagnostic-([0-9a-f]{40})-[0-9a-f]{12}-a[12]", run["display_title"]
        )
        if not match or Path(run["path"]).name not in {
            "slurm-dispatch.yml",
            "k8s-dispatch.yml",
        }:
            return None
        head = match[1]
    elif run["event"] != "pull_request":
        return None
    prs = api(repo, f"commits/{head}/pulls")
    for pr in prs:
        if (
            pr["state"] == "open"
            and pr["head"]["sha"] == head
            and pr["head"]["repo"]["full_name"] == repo
        ):
            return pr["number"]
    return None


def load_plan(repo: str, pr: dict, root: Path) -> dict | None:
    head = pr["head"]["sha"]
    runs = pages(
        repo,
        f"actions/workflows/pr-ci-plan.yml/runs?event=pull_request&head_sha={head}&per_page=100",
        "workflow_runs",
    )
    if not runs:
        return None
    run = max(runs, key=lambda item: item["id"])
    if (
        run["conclusion"] != "success"
        or run["head_repository"]["full_name"] != repo
        or run["path"] != ".github/workflows/pr-ci-plan.yml"
    ):
        return None
    directory = root / "proposal"
    directory.mkdir(parents=True, exist_ok=True)
    command(
        "gh",
        "run",
        "download",
        str(run["id"]),
        "--repo",
        repo,
        "--name",
        f"pr-ci-plan-{pr['number']}-{head}",
        "--dir",
        str(directory),
    )
    path = directory / "plan.json"
    if path.stat().st_size > 250000:
        raise ValueError("Oversized CI proposal.")
    plan = json.loads(path.read_text())
    if (
        plan["version"] != 1
        or plan["repository"] != repo
        or plan["pr"] != pr["number"]
        or plan["head"] != head
        or plan["base"] != pr["base"]["sha"]
    ):
        return None
    # Parse the candidate's data with main's tooling, never import its code.
    source = root / "source"
    command("git", "fetch", "--no-tags", "origin", head, pr["base"]["sha"])
    command("git", "worktree", "add", "--detach", str(source), head)
    try:
        data = context(source, head, pr["base"]["sha"])
    finally:
        command("git", "worktree", "remove", "--force", str(source))
    catalog = {task_key(task): task for task in data["catalog"]}
    keys = [task_key(task) for task in plan["tasks"]]
    if len(keys) != len(set(keys)) or not set(data["floor"]).issubset(keys):
        raise ValueError("CI proposal omits mandatory coverage.")
    # Ignore all model-supplied execution metadata and text in this stage.
    plan["tasks"] = [catalog[key] for key in keys]
    plan["policy_changes"] = data["policy_changes"]
    return plan


def task_workflow(task: dict) -> str:
    return task["workflow"]


def task_states(repo: str, pr: dict, plan: dict) -> dict:
    runs = pages(
        repo,
        f"actions/runs?event=pull_request&head_sha={pr['head']['sha']}&per_page=100",
        "workflow_runs",
    )
    latest = {}
    for run in runs:
        if run["head_repository"]["full_name"] != repo:
            continue
        workflow = Path(run["path"]).name
        if workflow not in latest or run["id"] > latest[workflow]["id"]:
            latest[workflow] = run
    jobs = {
        workflow: pages(
            repo, f"actions/runs/{run['id']}/jobs?filter=latest&per_page=100", "jobs"
        )
        for workflow, run in latest.items()
        if workflow in WORKFLOWS.values()
    }
    states = {}
    for task in plan["tasks"]:
        workflow = task_workflow(task)
        suffix = (
            task["name"]
            if task["runner"].startswith("slurm-")
            else f"{task['name']} ({task['runner']})"
        )
        matched = [
            job
            for job in jobs.get(workflow, [])
            if job["name"] == suffix or job["name"].endswith(f" / {suffix}")
        ]
        state = "pending"
        if matched and all(job["conclusion"] == "success" for job in matched):
            state = "passed"
        elif any(job["conclusion"] in {"failure", "timed_out"} for job in matched):
            state = "failed"
        states[task_key(task)] = state
    return states


def dispatch(repo: str, pr: dict, task: dict, backend: str, attempt: int) -> str:
    if pr["draft"]:
        raise ValueError("Diagnostics require a ready PR.")
    if backend == "auto":
        backend = (
            "gb300"
            if task["workflow"] == "nvidia-gb300-tests.yml"
            else "gb200" if task["runner"].startswith("slurm-") else "k8s"
        )
    workflow = "k8s-dispatch.yml" if backend == "k8s" else "slurm-dispatch.yml"
    key = hashlib.sha256(f"{task_key(task)}:{backend}".encode()).hexdigest()[:12]
    prefix = f"ci-diagnostic-{pr['head']['sha']}-{key}-a"
    runs = pages(
        repo,
        f"actions/workflows/{workflow}/runs?event=workflow_dispatch&per_page=100",
        "workflow_runs",
    )
    prior = [run for run in runs if run["display_title"] == f"{prefix}{attempt}"]
    if prior:
        return "already requested"
    if attempt == 2 and not any(
        run["display_title"] == f"{prefix}1"
        and run["status"] == "completed"
        and run["conclusion"] == "failure"
        for run in runs
    ):
        raise ValueError("A second diagnostic requires a failed first attempt.")
    if current_pr(repo, pr["number"])["head"]["sha"] != pr["head"]["sha"]:
        raise ValueError("PR head changed before dispatch.")
    fields = {
        "commit": pr["head"]["sha"],
        "yaml": task["config"],
        "runner": task["runner"],
        "task_types": task["type"],
        "trigger": "all",
        "include_mmlu": "true",
        "correlation": f"{prefix}{attempt}",
    }
    if backend == "k8s":
        if task["runner"].startswith("slurm-"):
            raise ValueError("Slurm tasks require a Slurm backend.")
        fields["runner_pool"] = "all"
    else:
        # B200 logical labels on Slurm are cross-hardware diagnosis only.
        fields["cluster"] = backend
        fields["runner"] = task["runner"].replace("b200v2-", "b200-", 1)
        if not fields["runner"].startswith(
            ("b200-", "gb200-", "slurm-gb200-", "gb300-", "slurm-gb300-")
        ):
            raise ValueError("This task cannot be diagnosed on the selected cluster.")
    args = ["gh", "workflow", "run", workflow, "--repo", repo, "--ref", "main"]
    for name, value in fields.items():
        args += ["-f", f"{name}={value}"]
    command(*args)
    # Read the published request back; absence never counts as validation.
    found = pages(
        repo,
        f"actions/workflows/{workflow}/runs?event=workflow_dispatch&per_page=100",
        "workflow_runs",
    )
    return (
        "requested"
        if any(run["display_title"] == fields["correlation"] for run in found)
        else "requested; awaiting run registration"
    )


def diagnostic_states(repo: str, pr: dict, tasks: list, root: Path) -> dict:
    """Only report results with an artifact proving the tested source/task."""
    results = {}
    for workflow, backends in (
        ("k8s-dispatch.yml", ("k8s",)),
        ("slurm-dispatch.yml", ("gb200", "gb300")),
    ):
        runs = pages(
            repo,
            f"actions/workflows/{workflow}/runs?event=workflow_dispatch&per_page=100",
            "workflow_runs",
        )
        for task in tasks:
            for backend in backends:
                key = hashlib.sha256(
                    f"{task_key(task)}:{backend}".encode()
                ).hexdigest()[:12]
                prefix = f"ci-diagnostic-{pr['head']['sha']}-{key}-a"
                matches = [
                    run
                    for run in runs
                    if run["head_repository"]["full_name"] == repo
                    and run["path"] == f".github/workflows/{workflow}"
                    and run["display_title"] in {f"{prefix}1", f"{prefix}2"}
                ]
                if not matches:
                    continue
                run = max(matches, key=lambda item: item["id"])
                if run["status"] != "completed":
                    results.setdefault(task_key(task), []).append(
                        f"{backend}: diagnostic pending"
                    )
                    continue
                directory = root / f"diagnostic-{run['id']}"
                directory.mkdir(exist_ok=True)
                artifact = (
                    f"dispatch-source-{run['id']}-{run['run_attempt']}"
                    if backend == "k8s"
                    else f"slurm-{run['id']}-{run['run_attempt']}"
                )
                try:
                    command(
                        "gh",
                        "run",
                        "download",
                        str(run["id"]),
                        "--repo",
                        repo,
                        "--name",
                        artifact,
                        "--dir",
                        str(directory),
                    )
                    path = directory / (
                        "dispatch-source.json" if backend == "k8s" else "source.json"
                    )
                    if path.stat().st_size > 10000:
                        continue
                    evidence = json.loads(path.read_text())
                except (OSError, ValueError, subprocess.CalledProcessError):
                    continue
                expected_runner = (
                    task["runner"]
                    if backend == "k8s"
                    else task["runner"].replace("b200v2-", "b200-", 1)
                )
                if evidence.get("commit") != pr["head"]["sha"] or evidence.get(
                    "tasks"
                ) != [{"config": task["config"], "runner": expected_runner}]:
                    continue
                if backend != "k8s" and evidence.get("cluster") != backend:
                    continue
                outcome = (
                    "retry passed; possible flake"
                    if run["conclusion"] == "success"
                    else "diagnostic failed; inspect logs before proposing a repair"
                )
                results.setdefault(task_key(task), []).append(f"{backend}: {outcome}")
    return results


def report(repo: str, pr: dict, plan: dict, root: Path) -> None:
    states = task_states(repo, pr, plan)
    diagnostics = diagnostic_states(repo, pr, plan["tasks"], root)
    checks = json.loads(
        command(
            "gh",
            "pr",
            "view",
            str(pr["number"]),
            "--repo",
            repo,
            "--json",
            "statusCheckRollup,reviewDecision",
        )
    )
    passed = {
        check.get("name") or check.get("context")
        for check in checks["statusCheckRollup"]
        if check.get("conclusion") == "SUCCESS" or check.get("state") == "SUCCESS"
    }
    cpu = {"lint", "DCO", "Commit trailers"}.issubset(passed)
    covered = (
        cpu
        and not plan["policy_changes"]
        and all(state == "passed" for state in states.values())
    )
    status = (
        "Selected CI passed; explicit plan acceptance and merge authorization are still required."
        if covered
        else "Waiting for selected CI."
    )
    if pr["mergeable"] is False:
        status = "Merge conflicts require resolution and validation at the new head."
    if plan["policy_changes"]:
        status = "Task scheduling or acceptance metadata changed; a coverage proposal cannot establish merge readiness. Policy review and full existing CI remain required."
    if checks["reviewDecision"] != "APPROVED":
        status += " Review approval is pending."
    lines = [
        f"<!-- pr-ci-state:{plan['head']} -->",
        f"Reviewed commit: `{plan['head']}`",
        "",
        "### CI readiness",
        "",
        status,
        "",
        f"CPU checks: {'passed' if cpu else 'pending or failed'}.",
    ]
    for task in plan["tasks"]:
        lines.append(
            f"- `{task['name']}` on `{task['runner']}`: {states[task_key(task)]}"
        )
        for diagnostic in diagnostics.get(task_key(task), []):
            lines.append(f"  - {diagnostic}")
    lines += [
        "",
        "A retry pass alone is a possible flake. Different hardware cannot replace the original hardware check. Remaining PR tests are cancelled by the existing workflow after an authorized merge.",
    ]
    body = "\n".join(lines) + "\n"
    if len(body) > 60000 or re.search(
        r"https?://|github\.com|\b(?:sk-|ghp_|gho_|github_pat_)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|/(?:home|root|tmp|proc)/",
        body,
    ):
        raise ValueError("Readiness output failed the public-output check.")
    fresh = current_pr(repo, pr["number"])
    if fresh["head"]["sha"] != plan["head"] or fresh["base"]["sha"] != plan["base"]:
        return
    comments = json.loads(
        command(
            "gh", "pr", "view", str(pr["number"]), "--repo", repo, "--json", "comments"
        )
    )["comments"]
    own = [
        comment for comment in comments if comment["author"]["login"] == "lightseek-bot"
    ]
    if any(comment["body"].rstrip() == body.rstrip() for comment in own):
        return
    path = root / "readiness.md"
    path.write_text(body)
    args = [
        "gh",
        "pr",
        "comment",
        str(pr["number"]),
        "--repo",
        repo,
        "--body-file",
        str(path),
    ]
    if own and own[-1]["body"].startswith(f"<!-- pr-ci-state:{plan['head']} -->"):
        args += ["--edit-last"]
    command(*args)
    live = json.loads(
        command(
            "gh", "pr", "view", str(pr["number"]), "--repo", repo, "--json", "comments"
        )
    )["comments"]
    if not any(
        comment["author"]["login"] == "lightseek-bot"
        and comment["body"].rstrip() == body.rstrip()
        for comment in live
    ):
        raise ValueError("Published readiness differs from the checked body.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("resolve", "report", "dispatch"),
        default=os.environ.get("CI_ACTION", "report"),
    )
    parser.add_argument("--pr", type=int, default=os.environ.get("PR") or None)
    parser.add_argument("--config", default=os.environ.get("CI_CONFIG"))
    parser.add_argument("--runner", default=os.environ.get("CI_RUNNER"))
    parser.add_argument(
        "--backend",
        choices=("auto", "k8s", "gb200", "gb300"),
        default=os.environ.get("CI_BACKEND", "auto"),
    )
    parser.add_argument(
        "--attempt", type=int, choices=(1, 2), default=os.environ.get("CI_ATTEMPT", "1")
    )
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    number = args.pr or event_pr(repo)
    if args.action == "resolve":
        if number:
            current_pr(repo, number)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"pr={number or ''}\n")
        return
    if not number:
        return
    public_bot(repo)
    pr = current_pr(repo, number)
    root = Path(os.environ["RUNNER_TEMP"], "pr-ci-assist")
    root.mkdir(parents=True, exist_ok=True)
    os.environ["PR_NUMBER"] = str(number)
    plan = load_plan(repo, pr, root)
    if plan is None:
        print("No fresh coverage proposal; full existing CI remains required.")
        return
    if args.action == "report":
        report(repo, pr, plan, root)
    else:
        task = next(
            t
            for t in plan["tasks"]
            if t["config"] == args.config and t["runner"] == args.runner
        )
        if task_states(repo, pr, plan)[task_key(task)] != "failed":
            raise ValueError("Dispatch requires a failed selected CI task.")
        print(dispatch(repo, pr, task, args.backend, args.attempt))


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        StopIteration,
        subprocess.CalledProcessError,
    ):
        raise SystemExit("CI assistance failed; raw output withheld.") from None
