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

"""Follow a PR's selected CI tasks, using the existing immutable dispatchers."""

import argparse
import datetime
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

from pr_ci_plan import CoverageError, context, task_key, validate_test_coverage
from pr_ci_state import BOT, BOT_ID, COMMAND, NATIVE_CHECKS, REPO, SHA, marker, record

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "pr-ci-assist"
FINISHED_PHASES = {"done", "manual", "stale", "promoted"}


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def api(path: str):
    return json.loads(command("gh", "api", f"repos/{REPO}/{path}"))


def pages(path: str, field: str | None):
    separator = "&" if "?" in path else "?"
    data = json.loads(
        command(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{REPO}/{path}{separator}per_page=100",
        )
    )
    return [row for page in data for row in (page[field] if field else page)]


def output(key: str, value: str):
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"{key}={value}\n")


def _require_open_pr(pr: dict) -> dict:
    if (
        pr["state"] != "open"
        or not pr["head"]["repo"]
        or pr["head"]["repo"]["full_name"] != REPO
        or pr["base"]["ref"] != "main"
    ):
        raise ValueError("An open same-repository PR into main is required.")
    return pr


def pull(number: int) -> dict:
    return _require_open_pr(api(f"pulls/{number}"))


def permitted(comment: dict) -> str | None:
    match = COMMAND.fullmatch(comment["body"] or "")
    if not match:
        return None
    permission = api(f"collaborators/{comment['user']['login']}/permission")[
        "permission"
    ]
    return match[1].lower() if permission in {"admin", "maintain", "write"} else None


def latest_command(comments: list[dict]) -> dict | None:
    for comment in reversed(comments):
        if COMMAND.fullmatch(comment["body"] or "") and permitted(comment):
            return comment
    return None


def resolve():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    name = os.environ["GITHUB_EVENT_NAME"]
    number = None
    if (
        name == "issue_comment"
        and event["action"] == "created"
        and "pull_request" in event["issue"]
    ):
        if permitted(event["comment"]):
            number = event["issue"]["number"]
    elif name == "workflow_dispatch":
        value = event["inputs"]["pr"]
        if re.fullmatch(r"[1-9][0-9]*", value):
            number = int(value)
    elif name == "workflow_run" and event["action"] == "completed":
        run = event["workflow_run"]
        if run["event"] not in {"pull_request", "workflow_dispatch"}:
            print("Non-PR CI completion; skipping CI assistance.")
            return
        candidates = {p["number"] for p in run["pull_requests"]}
        plan = re.fullmatch(
            r"CI plan #([1-9][0-9]*) \| [0-9a-f]{40} \| [0-9a-f]{40}",
            run["display_title"],
        )
        if plan:
            candidates.add(int(plan[1]))
        if not candidates:
            match = re.match(r"(?:Slurm|K8s) ([0-9a-f]{40}) \|", run["display_title"])
            source = match[1] if match else run["head_sha"]
            # Dispatch head_sha identifies main's controller, not the tested
            # commit. Its associated merged PR must never enter this lookup.
            candidates.update(
                p["number"]
                for p in pages(f"commits/{source}/pulls", None)
                if p["state"] == "open"
                and p["head"]["repo"]
                and p["head"]["repo"]["full_name"] == REPO
                and p["base"]["ref"] == "main"
            )
            if match:
                for branch in pages(f"commits/{source}/branches-where-head", None):
                    ref = re.fullmatch(
                        r"bot/pr-ci-assist-([1-9][0-9]*)-[1-9][0-9]*", branch["name"]
                    )
                    if ref:
                        candidates.add(int(ref[1]))
        if len(candidates) == 1:
            number = candidates.pop()
    if number:
        pr = api(f"pulls/{number}")
        if name == "workflow_run" and pr["state"] != "open":
            print("PR is closed or merged; skipping CI assistance.")
            return
        _require_open_pr(pr)
        if name == "workflow_run":
            comments = pages(f"issues/{number}/comments", None)
            state = load_state(comments, pr)
            comment = latest_command(comments)
            initial = bool(comment and (not state or comment["id"] > state["command"]))
            if not initial and (not state or state["phase"] in FINISHED_PHASES):
                print("No active watch/fix command; skipping CI assistance.")
                return
        output("pr", str(number))


def plan_source():
    """Resolve an open PR at its requested source; skip obsolete events."""
    output("active", "false")
    name = os.environ["GITHUB_EVENT_NAME"]
    if name not in {"pull_request", "workflow_dispatch"}:
        print("CI plans require a PR event or a refresh for a specific PR; skipping.")
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    number = (
        int(event["inputs"]["pr"])
        if name == "workflow_dispatch"
        else event["pull_request"]["number"]
    )
    pr = api(f"pulls/{number}")
    # Fast merges can close the PR before its queued planner starts.
    if pr["state"] != "open":
        print("PR is closed or merged; skipping the CI plan.")
        return
    _require_open_pr(pr)
    if name == "pull_request" and (
        event["pull_request"]["head"]["sha"] != pr["head"]["sha"]
        or event["pull_request"]["base"]["sha"] != pr["base"]["sha"]
    ):
        print("PR event source changed; skipping the obsolete CI plan.")
        return
    if name == "workflow_dispatch" and (
        event["inputs"]["head"] != pr["head"]["sha"]
        or event["inputs"]["base"] != pr["base"]["sha"]
    ):
        print("Plan request source changed; skipping the obsolete CI plan.")
        return
    output("pr", str(number))
    output("head", pr["head"]["sha"])
    output("base", pr["base"]["sha"])
    output("active", "true")


def public_gate():
    command("gh", "auth", "status")
    if json.loads(command("gh", "api", "user"))["login"] != BOT:
        raise ValueError("Bot authentication required.")
    if (
        command(
            "gh", "repo", "view", REPO, "--json", "visibility", "--jq", ".visibility"
        )
        != "PUBLIC"
    ):
        raise ValueError("Public destination verification failed.")


def publish(state: dict, message: str):
    public_gate()
    if (
        record(
            {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)},
            "assist",
        )
        != state
    ):
        raise ValueError("Invalid outbound state.")
    # All editable text here is fixed, identifiers were validated against the
    # public catalog; do not copy API errors, task logs or model prose.
    body = f"**CI {state['action']}** · `{state['head'][:8]}`\n\n{message}\n"
    if state.get("native_checks") or state["statuses"]:
        body += "\n| Check | Source | Result |\n|---|---|---|\n"
        for check in state.get("native_checks", []):
            label = NATIVE_CHECKS[check["workflow"]]["label"]
            result = check["status"]
            if check["run"]:
                result = (
                    f"[{result}](https://github.com/{REPO}/actions/runs/{check['run']})"
                )
            body += f"| {label} | PR `{state['head'][:8]}` | {result} |\n"
        sha = state.get("candidate", {}).get("validation", state["head"])
        for task, status in zip(state["tasks"], state["statuses"]):
            link = f"https://github.com/{REPO}/blob/{sha}/{quote(task['config'], safe='/')}"
            run = state["run_ids"].get(task_key(task))
            result = (
                f"[{status}](https://github.com/{REPO}/actions/runs/{run})"
                if run
                else status
            )
            body += (
                f"| [{Path(task['config']).stem}]({link}) | `{sha[:8]}` | {result} |\n"
            )
    body += marker("assist", state)
    scanned = re.sub(
        rf"https://github\.com/{REPO}/(?:blob/[0-9a-f]{{40}}/[A-Za-z0-9_./%-]+|actions/runs/[0-9]+)",
        "PUBLIC_SOURCE",
        body,
    )
    if re.search(
        r"https?://|\bwww\.|(?:sk-|ghp_|github_pat_)|/(?:home|tmp|root)/", scanned
    ):
        raise ValueError("Public output rejected.")
    WORK.mkdir(parents=True, exist_ok=True)
    file = WORK / "comment.md"
    file.write_text(body)
    url = command(
        "gh",
        "pr",
        "comment",
        str(state["pr"]),
        "--repo",
        REPO,
        "--body-file",
        str(file),
    )
    live = api(f"issues/comments/{url.rsplit('issuecomment-', 1)[-1]}")
    if live["body"].rstrip() != body.rstrip() or record(live, "assist") != state:
        raise ValueError("Published state differs from reviewed content.")


def checkout(head: str, base: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    target = WORK / "source"
    for sha in (head, base):
        if not SHA.fullmatch(sha):
            raise ValueError("Invalid source SHA.")
        command("git", "fetch", "origin", sha)
    command("git", "worktree", "add", "--detach", str(target), head)
    return target


def validate_plan(plan: dict, data: dict) -> list[dict]:
    if any(plan[k] != data[k] for k in ("repository", "pr", "head", "base")):
        raise ValueError("Plan source changed.")
    if (
        not isinstance(plan["tasks"], list)
        or not isinstance(plan["tests"], list)
        or len(plan["tasks"]) > 8
    ):
        raise ValueError("Invalid selected task set.")
    catalog = {t["config"]: t for t in data["catalog"]}
    tasks = []
    for selected in plan["tasks"]:
        if set(selected) != {"config", "runner", "cluster"} or not all(
            isinstance(v, str) for v in selected.values()
        ):
            raise ValueError("Invalid task record.")
        task = catalog[selected["config"]]
        cluster, runner = selected["cluster"], selected["runner"]
        runners = task["runners"] if cluster == "" else task["slurm_runners"][cluster]
        if runner not in runners or (
            cluster == ""
            and (
                not runner.startswith("amd-")
                or not re.search(r"(?:^|-)[1-9][0-9]*gpu(?:-|$)", runner)
            )
        ):
            raise ValueError("Task has no supported assistance route.")
        if any(
            len(v) > 180 or not re.fullmatch(r"[a-zA-Z0-9_./-]*", v)
            for v in selected.values()
        ):
            raise ValueError("Unsafe task identifier.")
        tasks.append(
            {
                **selected,
                "name": task["name"],
                "type": task["type"],
                "native_runners": task["runners"],
                "triggers": task["triggers"],
            }
        )
    if len({task_key(t) for t in tasks}) != len(tasks) or any(
        t not in data["test_files"] for t in plan["tests"]
    ):
        raise ValueError("Invalid selected coverage.")
    validate_test_coverage(plan["tests"], tasks, data["catalog"], data.get("paths", []))
    return tasks


def effective_runner(task: dict, cluster: str) -> str:
    runner = task["runner"]
    if cluster != "gb300":
        return runner
    prefix = "slurm-" if runner.startswith("slurm-") else ""
    return f"{prefix}gb300-{runner.removeprefix(prefix).split('-', 1)[1]}"


def run_title(task: dict, sha: str, cluster: str) -> str:
    if cluster:
        return f"Slurm {sha} | {task['config']} | {task['runner']} | {cluster}"
    return f"K8s {sha} | {task['config']} | {task['runner']}"


def dispatch(task: dict, sha: str, cluster: str):
    public_gate()
    fields = dict(
        commit=sha,
        yaml="off" if cluster else "all",
        match=task["config"],
        task_types=task["type"],
        trigger="all",
        include_mmlu="true",
    )
    if cluster:
        fields.update(cluster=cluster, runners=task["runner"], require_idle="true")
    else:
        fields.update(runner_pool="amd", runner=task["runner"])
    args = [
        "gh",
        "workflow",
        "run",
        "slurm-dispatch.yml" if cluster else "k8s-dispatch.yml",
        "--repo",
        REPO,
        "--ref",
        "main",
    ]
    for key, value in fields.items():
        args += ["-f", f"{key}={value}"]
    command(*args)


def runs_for(state: dict) -> list[dict]:
    since = api(f"issues/comments/{state['since']}")["created_at"]
    runs = pages(f"actions/runs?head_sha={state['head']}", "workflow_runs")
    for workflow in ("slurm-dispatch.yml", "k8s-dispatch.yml"):
        runs += pages(
            f"actions/workflows/{workflow}/runs?event=workflow_dispatch&created=>={since}",
            "workflow_runs",
        )
    return sorted(
        {r["id"]: r for r in runs}.values(), key=lambda r: r["id"], reverse=True
    )


def native_check(check: dict, state: dict, runs: list[dict]) -> dict:
    result = dict(workflow=check["workflow"], status="waiting", run=0)
    for run in runs:
        if (
            run["event"] != "pull_request"
            or run["head_sha"] != state["head"]
            or run["path"] != f".github/workflows/{check['workflow']}"
            or not any(
                p["number"] == state["pr"]
                and p["head"]["sha"] == state["head"]
                and p["base"]["sha"] == state["base"]
                and p["base"]["ref"] == "main"
                for p in run["pull_requests"]
            )
        ):
            continue
        result["run"] = run["id"]
        if run["status"] != "completed":
            return result
        if run["conclusion"] == "failure":
            return {**result, "status": "failed"}
        jobs = pages(f"actions/runs/{run['id']}/jobs?filter=latest", "jobs")
        jobs = [j for j in jobs if j["name"] == check["job"]]
        if (
            run["conclusion"] == "success"
            and len(jobs) == 1
            and jobs[0]["conclusion"] == "skipped"
        ):
            return result
        if (
            run["conclusion"] == "success"
            and len(jobs) == 1
            and jobs[0]["status"] == "completed"
            and jobs[0]["conclusion"] == "success"
            and any(
                step["name"] == check["step"]
                and step["status"] == "completed"
                and step["conclusion"] == "success"
                for step in jobs[0]["steps"]
            )
        ):
            return {**result, "status": "passed"}
        return {**result, "status": "missing"}
    return result


def download(run: dict, name: str, target: Path):
    command(
        "gh",
        "run",
        "download",
        str(run["id"]),
        "--repo",
        REPO,
        "--name",
        name,
        "--dir",
        str(target),
    )


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


def report(run: dict, task: dict, sha: str, cluster: str) -> str:
    if run["status"] != "completed":
        created = datetime.datetime.fromisoformat(
            run["created_at"].replace("Z", "+00:00")
        )
        return (
            "blocked"
            if datetime.datetime.now(datetime.timezone.utc) - created
            > datetime.timedelta(hours=12)
            else "waiting"
        )
    artifacts = pages(f"actions/runs/{run['id']}/artifacts", "artifacts")
    with tempfile.TemporaryDirectory(dir=WORK) as directory:
        target = Path(directory)
        if cluster:
            name = f"slurm-{run['id']}-{run['run_attempt']}"
            if not any(a["name"] == name and not a["expired"] for a in artifacts):
                return "missing"
            download(run, name, target)
            source = json.loads((target / "source.json").read_text())
            if source.get("source_sha") != sha:
                return "missing"
            availability = target / "availability.json"
            if availability.is_file() and json.loads(availability.read_text()) == {
                "source_sha": sha,
                "availability": "unavailable",
            }:
                return "unavailable"
            manifest = json.loads((target / "manifest.json").read_text())
            rows = [
                r
                for r in manifest
                if r["task"]["config"] == task["config"]
                and r["task"]["runner"] == effective_runner(task, cluster)
            ]
            if len(rows) != 1 or not re.fullmatch(r"[0-9]+", rows[0]["job_id"]):
                return "missing"
            row = rows[0]
            result = {
                "source_sha": sha,
                "config": task["config"],
                **json.loads((target / f"{row['job_id']}-result.json").read_text()),
            }
            status = result_status(result, task, sha, effective_runner(task, cluster))
            if (
                status == "passed"
                and row["state"] == "COMPLETED"
                and row["exit_code"] == "0:0"
            ):
                return "passed"
            return "failed" if status == "failed" else "missing"
        name = (
            f"pr-test-{task['name']}-{task['runner']}-{run['id']}-{run['run_attempt']}"
        )
        if not any(a["name"] == name and not a["expired"] for a in artifacts):
            return "missing"
        download(run, name, target)
        results = list(target.rglob("result.json"))
        if len(results) != 1:
            return "missing"
        raw = json.loads(results[0].read_text())
        proofs = list(target.rglob("source.json"))
        if len(proofs) != 1 or json.loads(proofs[0].read_text()) != {
            "source_sha": sha,
            "config": task["config"],
            "runner": task["runner"],
        }:
            return "missing"
        result = {"source_sha": sha, "config": task["config"], **raw}
        status = result_status(result, task, sha, task["runner"])
        jobs = pages(f"actions/runs/{run['id']}/jobs?filter=latest", "jobs")
        jobs = [
            j for j in jobs if j["name"].endswith(f"{task['name']} ({task['runner']})")
        ]
        if status == "passed" and len(jobs) == 1 and jobs[0]["conclusion"] == "success":
            return "passed"
        return "failed" if status == "failed" else "missing"


def source_matches(sha: str, state: dict) -> bool:
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        return False
    if sha == state["head"]:
        return True
    parents = [p["sha"] for p in api(f"commits/{sha}")["parents"]]
    return parents == [state["base"], state["head"]]


def native_result(run: dict, task: dict, state: dict, job: dict) -> str:
    artifacts = pages(f"actions/runs/{run['id']}/artifacts", "artifacts")
    attempt = run["run_attempt"]
    with tempfile.TemporaryDirectory(dir=WORK) as directory:
        target = Path(directory)
        names = {
            f"pr-test-{task['name']}-{r}-{run['id']}-{attempt}": r
            for r in task["native_runners"]
        }
        if run["name"] in {"NVIDIA GB200 Tests", "NVIDIA GB300 Tests"}:
            prefix = "gb200" if run["name"] == "NVIDIA GB200 Tests" else "gb300"
            name = f"{prefix}-slurm-{task['name']}-{run['id']}-{attempt}"
            if not any(a["name"] == name and not a["expired"] for a in artifacts):
                return "missing"
            download(run, name, target)
            manifest = json.loads((target / "manifest.json").read_text())
            source = json.loads((target / "source.json").read_text())["source_sha"]
            if not source_matches(source, state):
                return "missing"
            rows = [r for r in manifest if r["task"]["config"] == task["config"]]
            if len(rows) != 1 or not re.fullmatch(r"[0-9]+", rows[0]["job_id"]):
                return "missing"
            row = rows[0]
            result = {
                "source_sha": source,
                "config": task["config"],
                **json.loads((target / f"{row['job_id']}-result.json").read_text()),
            }
            status = result_status(result, task, source, row["task"]["runner"])
            return (
                "passed"
                if status == "passed"
                and row["state"] == "COMPLETED"
                and row["exit_code"] == "0:0"
                else "missing"
            )
        for artifact in artifacts:
            runner = names.get(artifact["name"])
            if (
                not runner
                or artifact["expired"]
                or not job["name"].endswith(f"({runner})")
            ):
                continue
            download(run, artifact["name"], target)
            results = list(target.rglob("result.json"))
            if len(results) != 1:
                return "missing"
            proofs = list(target.rglob("source.json"))
            if len(proofs) != 1:
                return "missing"
            proof = json.loads(proofs[0].read_text())
            if proof.get("config") != task["config"] or proof.get("runner") != runner:
                return "missing"
            result = {**proof, **json.loads(results[0].read_text())}
            if not source_matches(result.get("source_sha"), state):
                return "missing"
            return result_status(result, task, result["source_sha"], runner)
    return "missing"


def original_status(runs: list[dict], task: dict, state: dict) -> str:
    # Native PR matrices may use a different physical NVIDIA label; match the
    # declared task name only within a workflow tied to this exact PR source.
    pending = "absent"
    for run in runs:
        if (
            run["event"] != "pull_request"
            or run["head_sha"] != state["head"]
            or run["name"]
            not in {
                "NVIDIA B200 Tests",
                "NVIDIA GB200 Tests",
                "NVIDIA GB300 Tests",
                "PR Test NVIDIA ARM",
                "AMD Tests",
            }
        ):
            continue
        jobs = pages(f"actions/runs/{run['id']}/jobs?filter=latest", "jobs")
        dispatch_queued = bool(task["cluster"]) and run["name"] not in {
            "NVIDIA GB200 Tests",
            "NVIDIA GB300 Tests",
        }
        labels = {
            r
            for r in task["native_runners"]
            if r.startswith("amd-") == task["runner"].startswith("amd-")
        }
        matches = [
            j
            for j in jobs
            if any(j["name"].endswith(f"{task['name']} ({label})") for label in labels)
            or (
                task["cluster"]
                and run["name"] in {"NVIDIA GB200 Tests", "NVIDIA GB300 Tests"}
                and j["name"] == task["name"]
            )
        ]
        if not matches:
            if "per-commit" in task["triggers"] and any(
                j["name"] == "scan" and j["status"] != "completed" for j in jobs
            ):
                if dispatch_queued:
                    pending = "queued"
                    continue
                return "waiting"
            continue
        job = matches[0]
        if job["status"] == "queued" and dispatch_queued:
            pending = "queued"
            continue
        state["run_ids"][task_key(task)] = run["id"]
        if job["status"] != "completed":
            return "waiting"
        if job["conclusion"] == "failure":
            return "failed"
        if job["conclusion"] == "success":
            try:
                return native_result(run, task, state, job)
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.CalledProcessError,
            ):
                return "missing"
        return "missing"
    return pending


def task_status(task: dict, state: dict, runs: list[dict], *, submit: bool) -> str:
    sha = state.get("candidate", {}).get("validation", state["head"])
    clusters = [task["cluster"]]
    if task["cluster"] == "gb200":
        clusters.append("gb300")
    for cluster in clusters:
        matches = [
            r
            for r in runs
            if r["event"] == "workflow_dispatch"
            and r["head_branch"] == "main"
            and r["actor"]["login"] == BOT
            and r["display_title"] == run_title(task, sha, cluster)
        ]
        if matches:
            state["run_ids"][task_key(task)] = matches[0]["id"]
            try:
                status = report(matches[0], task, sha, cluster)
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.CalledProcessError,
            ):
                status = "missing"
            if status == "unavailable":
                continue
            return status
        title = run_title(task, sha, cluster)
        if title in state["submitted"]:
            # The dispatch owns this check, even before its run becomes visible.
            return "waiting"
        if "candidate" not in state and cluster == clusters[0]:
            status = original_status(runs, task, state)
            if status == "failed" and state["action"] == "fix":
                return status
            if status not in {"failed", "absent", "missing", "queued"} or not submit:
                return status
        if submit:
            state["submitted"].append(title)
            publish(
                state,
                "Selected task dispatch requested; existing allocations are reused.",
            )
            try:
                dispatch(task, sha, cluster)
            except subprocess.CalledProcessError:
                state["phase"] = "manual"
                publish(
                    state,
                    "Dispatch could not be confirmed. Human intervention required; no blind retry.",
                )
                raise
        return "waiting"
    return "blocked"


def load_state(comments: list[dict], pr: dict) -> dict | None:
    for comment in reversed(comments):
        state = record(comment, "assist")
        if state and state["pr"] == pr["number"]:
            command_comment = api(f"issues/comments/{state['command']}")
            if (
                command_comment["issue_url"].rsplit("/", 1)[-1] != str(pr["number"])
                or permitted(command_comment) != state["action"]
            ):
                raise ValueError("Command is no longer authorized.")
            return state
    return None


def refresh_plan(state: dict, message: str):
    state["phase"] = "waiting-plan"
    publish(state, message)
    command(
        "gh",
        "workflow",
        "run",
        "pr-ci-plan.yml",
        "--repo",
        REPO,
        "--ref",
        "main",
        "-f",
        f"pr={state['pr']}",
        "-f",
        f"head={state['head']}",
        "-f",
        f"base={state['base']}",
    )


def control(number: int):
    public_gate()
    pr = pull(number)
    comments = pages(f"issues/{number}/comments", None)
    state = load_state(comments, pr)
    comment = latest_command(comments)
    initial = bool(comment and (not state or comment["id"] > state["command"]))
    if initial:
        action = permitted(comment)
        prior = state
        state = dict(
            version=1,
            repository=REPO,
            pr=number,
            head=pr["head"]["sha"],
            base=pr["base"]["sha"],
            command=comment["id"],
            action=action,
            phase="watching",
            tasks=[],
            statuses=[],
            run_ids={},
            conflicts=pr["mergeable"] is False,
            since=comment["id"],
            submitted=[],
        )
        if prior and (prior["head"], prior["base"]) == (state["head"], state["base"]):
            state["submitted"] = [
                t
                for t in prior["submitted"]
                if t.startswith((f"Slurm {state['head']} |", f"K8s {state['head']} |"))
            ]
            state["since"] = prior["since"]
    if not state or state["phase"] in FINISHED_PHASES:
        return
    if state["head"] != pr["head"]["sha"] or state["base"] != pr["base"]["sha"]:
        state["phase"] = "stale"
        publish(state, "PR or main changed. Request a new plan and command.")
        return
    if "plan_refresh" in state and os.environ["GITHUB_EVENT_NAME"] == "workflow_run":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        completed = event["workflow_run"]
        if (
            completed["display_title"]
            == f"CI plan #{number} | {state['head']} | {state['base']}"
            and completed["id"] > state["plan_refresh"]
            and completed["conclusion"] != "success"
        ):
            completed = api(f"actions/runs/{completed['id']}")
            if (
                completed["path"] == ".github/workflows/pr-ci-plan.yml"
                and completed["event"] == "workflow_dispatch"
                and completed["head_branch"] == "main"
                and completed["actor"]["login"] == BOT
                and completed["conclusion"] != "success"
            ):
                state["phase"] = "manual"
                publish(state, "CI plan refresh failed. Human intervention required.")
                return
    source = checkout(state["head"], state["base"])
    os.environ.update(PR_NUMBER=str(number), GITHUB_REPOSITORY=REPO)
    data = context(source, state["head"], state["base"])
    plans = [
        p
        for c in reversed(comments)
        if (p := record(c, "plan"))
        and p["pr"] == number
        and p["head"] == state["head"]
        and p["base"] == state["base"]
    ]
    if not plans:
        if initial:
            refresh_plan(
                state,
                "Refreshing the CI plan for this head and base; only its selected tasks will be watched.",
            )
        return
    plan = plans[0]
    plan_run = api(f"actions/runs/{plan['run']}")
    if (
        plan_run["path"] != ".github/workflows/pr-ci-plan.yml"
        or plan_run["conclusion"] != "success"
        or plan_run["display_title"]
        != f"CI plan #{number} | {state['head']} | {state['base']}"
    ):
        if initial:
            state["phase"] = "waiting-plan"
            publish(state, "Waiting for the current CI plan to finish.")
        return
    try:
        tasks = validate_plan(plan, data)
    except CoverageError:
        if "plan_refresh" not in state:
            state["plan_refresh"] = plan["run"]
            refresh_plan(
                state, "CI plan omits required validation; refreshing coverage."
            )
        elif plan["run"] > state["plan_refresh"]:
            state["phase"] = "manual"
            publish(
                state,
                "Refreshed CI plan still omits required validation. Human intervention required.",
            )
        return
    except ValueError:
        state["phase"] = "manual"
        publish(
            state,
            "Some planned tasks have no supported GPU route. Human intervention required.",
        )
        return
    replanned = state.pop("plan_refresh", None) is not None
    if not tasks and (state["action"] != "watch" or not data.get("native_checks")):
        state["phase"] = "manual"
        publish(
            state,
            "No supported GPU task in this plan. Select relevant coverage before automatic repair.",
        )
        return
    state["tasks"] = [{k: t[k] for k in ("config", "runner", "cluster")} for t in tasks]
    sources = [state["head"]] + (
        [state["candidate"]["validation"]] if "candidate" in state else []
    )
    titles = {
        run_title(t, sha, cluster)
        for t in tasks
        for sha in sources
        for cluster in (
            ["gb200", "gb300"] if t["cluster"] == "gb200" else [t["cluster"]]
        )
    }
    state["submitted"] = [t for t in state["submitted"] if t in titles]
    if state["phase"] == "repairing":
        owner = api(f"actions/runs/{state['repair_run']}")
        if owner["status"] == "completed":
            state["phase"] = "manual"
            publish(
                state,
                "Repair workflow ended before staging. Human intervention required; request a fresh fix.",
            )
        return
    runs = runs_for(state)
    previous_checks = state.get("native_checks", [])
    state["native_checks"] = [
        native_check(check, state, runs)
        for check in data.get("native_checks", [])
        if state["action"] == "watch"
    ]
    native_statuses = [c["status"] for c in state["native_checks"]]
    if any(s in {"failed", "missing", "blocked"} for s in native_statuses):
        state["phase"] = "manual"
        publish(
            state,
            "Native checks need human intervention; no dispatch retry or PR update.",
        )
        return
    requested_fix = state["action"] == "fix" and "candidate" not in state
    if requested_fix and pr["mergeable"] is False:
        state["conflicts"] = True
        statuses = ["waiting"] * len(tasks)
    else:
        statuses = [task_status(t, state, runs, submit=True) for t in tasks]
    previous = state["statuses"]
    state["statuses"] = statuses
    if requested_fix and (pr["mergeable"] is False or "failed" in statuses):
        state["phase"] = "repairing"
        state["repair_run"] = int(os.environ["GITHUB_RUN_ID"])
        publish(
            state,
            (
                "Resolving conflicts first."
                if pr["mergeable"] is False
                else "Preparing a focused repair; validation precedes PR updates."
            ),
        )
        WORK.joinpath("request.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "plan": plan,
                    "data": data,
                    "conflicts": pr["mergeable"] is False,
                }
            )
        )
        output("repair", "true")
        return
    combined = native_statuses + statuses
    if "candidate" in state and all(s == "passed" for s in combined):
        from pr_ci_repair import promote

        promote(state)
        state["phase"] = "promoted"
        publish(
            state,
            "Validated repair cherry-picked to the PR. Required CI remains in effect.",
        )
        return
    if any(s in {"failed", "missing", "blocked"} for s in combined):
        state["phase"] = "manual"
    elif all(s == "passed" for s in combined):
        state["phase"] = "done"
    else:
        state["phase"] = "validating" if "candidate" in state else "watching"
    if (
        initial
        or replanned
        or previous != statuses
        or previous_checks != state["native_checks"]
    ):
        counts = ", ".join(
            f"{combined.count(s)} {s}"
            for s in ("passed", "waiting", "failed", "missing", "blocked")
            if s in combined
        )
        message = f"{counts}."
        if state["phase"] == "manual":
            message += " Human intervention required; no further retry or PR update."
        publish(state, message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("resolve", "control", "plan-source"))
    parser.add_argument("--pr", type=int)
    args = parser.parse_args()
    try:
        {
            "resolve": resolve,
            "control": lambda: control(args.pr),
            "plan-source": plan_source,
        }[args.stage]()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        raise SystemExit("CI assistance stopped; raw diagnostics withheld.") from None
