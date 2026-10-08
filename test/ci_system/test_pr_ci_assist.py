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
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts"))
import pr_ci_assist as assist
import pr_ci_repair as repair
from ci_result_source import write_source
from pr_ci_state import BOT, BOT_ID, NATIVE_CHECKS, REPO, marker, record


@pytest.fixture
def selected():
    task = dict(
        config="test/ci/ut/ut-runtime-1gpu.yaml",
        runner="b200-1gpu",
        cluster="gb200",
        name="runtime",
        type="ut",
        native_runners=["b200v2-1gpu", "gb200-1gpu"],
        triggers=["per-commit"],
    )
    state = dict(
        version=1,
        repository=REPO,
        pr=123,
        head="a" * 40,
        base="b" * 40,
        action="watch",
        command=42,
        phase="watching",
        tasks=[{k: task[k] for k in ("config", "runner", "cluster")}],
        statuses=[],
        run_ids={},
        conflicts=False,
        since=42,
        submitted=[],
    )
    return task, state


def test_only_actual_writer_can_issue_command(monkeypatch):
    comment = dict(body="@lightseek-bot fix", user={"login": "example"})
    monkeypatch.setattr(assist, "api", lambda path: {"permission": "read"})
    assert assist.permitted(comment) is None
    monkeypatch.setattr(assist, "api", lambda path: {"permission": "write"})
    assert assist.permitted(comment) == "fix"
    comment["body"] += " and run every CI"
    assert assist.permitted(comment) is None
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: {
            "state": "open",
            "head": {"repo": {"full_name": "untrusted-source"}},
            "base": {"ref": "main"},
        },
    )
    with pytest.raises(ValueError, match="same-repository"):
        assist.pull(123)


def test_plan_source_only_activates_for_current_open_pr(monkeypatch, tmp_path):
    pr = dict(
        number=123,
        state="open",
        head={"sha": "a" * 40, "repo": {"full_name": REPO}},
        base={"sha": "b" * 40, "ref": "main"},
    )
    event = tmp_path / "event.json"
    result = tmp_path / "output"
    event.write_text(json.dumps({"pull_request": pr}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_OUTPUT", str(result))
    monkeypatch.setattr(assist, "api", lambda path: pr)
    assist.plan_source()
    assert dict(line.split("=", 1) for line in result.read_text().splitlines()) == {
        "active": "true",
        "pr": "123",
        "head": "a" * 40,
        "base": "b" * 40,
    }
    result.write_text("")
    pr["state"] = "closed"
    assist.plan_source()
    assert result.read_text() == "active=false\n"
    result.write_text("")
    pr.update(state="open", head={**pr["head"], "sha": "c" * 40})
    assist.plan_source()
    assert result.read_text() == "active=false\n"
    result.write_text("")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    event.write_text(json.dumps({"ref": "refs/heads/main"}))
    monkeypatch.setattr(
        assist, "api", lambda path: pytest.fail("Main pushes must not resolve a PR")
    )
    assist.plan_source()
    assert result.read_text() == "active=false\n"
    workflow = yaml.safe_load(
        (assist.ROOT / ".github/workflows/pr-ci-plan.yml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    steps = workflow["jobs"]["plan"]["steps"]
    source = next(i for i, step in enumerate(steps) if step.get("id") == "source")
    assert all(
        step["if"] == "steps.source.outputs.active == 'true'"
        for step in steps[source + 1 :]
    )


def test_main_ci_completion_skips_before_resolving_pr(monkeypatch, tmp_path):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            dict(
                action="completed",
                workflow_run=dict(
                    event="push",
                    head_branch="main",
                    head_sha="a" * 40,
                    pull_requests=[],
                    display_title="Main CI",
                ),
            )
        )
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setattr(
        assist, "api", lambda *args: pytest.fail("Main CI looked up a PR")
    )
    monkeypatch.setattr(
        assist, "pages", lambda *args: pytest.fail("Main CI looked up commits")
    )
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    assist.resolve()
    assert not emitted
    workflow = yaml.safe_load(
        (assist.ROOT / ".github/workflows/pr-ci-assist.yml").read_text()
    )
    assert "NVIDIA Kernel Library Tests" in workflow[True]["workflow_run"]["workflows"]
    assert workflow[True]["workflow_run"]["branches-ignore"] == ["main"]
    assert "workflow_call" in workflow[True]
    dispatcher = yaml.safe_load(
        (assist.ROOT / ".github/workflows/pr-ci-assist-dispatch.yml").read_text()
    )
    assert dispatcher[True]["workflow_run"]["branches"] == ["main"]
    assert dispatcher[True]["workflow_run"]["workflows"] == [
        "PR CI Plan",
        "Slurm Dispatch",
        "K8s Dispatch",
    ]
    assert (
        dispatcher["jobs"]["assist"]["uses"] == "./.github/workflows/pr-ci-assist.yml"
    )
    assert dispatcher["jobs"]["assist"]["secrets"] == "inherit"
    condition = (
        workflow["jobs"]["resolve"]["if"]
        .replace("\n", " ")
        .replace("&&", "and")
        .replace("||", "or")
    )
    github = SimpleNamespace(
        repository=REPO,
        event_name="workflow_run",
        event=SimpleNamespace(
            workflow_run=SimpleNamespace(event="push"),
            issue=SimpleNamespace(pull_request=True),
            comment=SimpleNamespace(body="Ordinary PR comment"),
        ),
    )
    context = {
        "github": github,
        "contains": lambda text, part: part.lower() in text.lower(),
    }
    assert not eval(condition, {"__builtins__": {}}, context)
    github.event.workflow_run.event = "pull_request"
    assert eval(condition, {"__builtins__": {}}, context)
    github.event.workflow_run.event = "workflow_dispatch"
    assert eval(condition, {"__builtins__": {}}, context)
    github.event_name = "issue_comment"
    assert not eval(condition, {"__builtins__": {}}, context)
    github.event.comment.body = "@LIGHTSEEK-BOT\tWATCH"
    assert eval(condition, {"__builtins__": {}}, context)
    github.event.issue.pull_request = False
    assert not eval(condition, {"__builtins__": {}}, context)


def test_state_is_typed_and_bot_owned(selected):
    _, state = selected
    comment = {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)}
    assert record(comment, "assist") == state
    comment["user"]["id"] = 1
    assert record(comment, "assist") is None
    comment["user"]["id"] = BOT_ID
    state["extra"] = "arbitrary embedded content"
    comment["body"] = marker("assist", state)
    assert record(comment, "assist") is None


@pytest.mark.parametrize(
    "workflow", ["scheduler-cpp-test.yml", "nvidia-kernel-library-tests.yml"]
)
def test_native_result_needs_current_source_and_executed_test(
    monkeypatch, selected, workflow
):
    _, state = selected
    check = {"workflow": workflow, **NATIVE_CHECKS[workflow]}
    run = dict(
        id=101,
        event="pull_request",
        head_sha=state["head"],
        path=f".github/workflows/{workflow}",
        pull_requests=[
            dict(
                number=state["pr"],
                head={"sha": state["head"]},
                base={"sha": state["base"], "ref": "main"},
            )
        ],
        status="completed",
        conclusion="success",
    )
    step = dict(name=check["step"], status="completed", conclusion="success")
    job = dict(
        name=check["job"], status="completed", conclusion="success", steps=[step]
    )
    monkeypatch.setattr(assist, "pages", lambda path, field: [job])
    assert assist.native_check(check, state, [run])["status"] == "passed"
    job["conclusion"] = "skipped"
    assert assist.native_check(check, state, [run])["status"] == "waiting"
    job["conclusion"] = "success"
    step["conclusion"] = "skipped"
    assert assist.native_check(check, state, [run])["status"] == "missing"
    newer = {**run, "id": 102, "status": "in_progress"}
    assert assist.native_check(check, state, [newer, run]) == dict(
        workflow=workflow, status="waiting", run=102
    )
    newer.update(status="completed", conclusion="failure")
    assert assist.native_check(check, state, [newer, run])["status"] == "failed"
    run["pull_requests"][0]["base"]["sha"] = "c" * 40
    assert assist.native_check(check, state, [run])["run"] == 0
    run["head_sha"] = "d" * 40
    assert assist.native_check(check, state, [run])["run"] == 0


def test_watch_waits_for_cpu_and_hands_off_failure_without_gpu_retry(
    monkeypatch, tmp_path, selected
):
    task, state = selected
    workflow = "scheduler-cpp-test.yml"
    check = {"workflow": workflow, **NATIVE_CHECKS[workflow]}
    plan = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    plan.update(run=55, tasks=state["tasks"], tests=[])
    comments = [{"user": {"login": BOT, "id": BOT_ID}, "body": marker("plan", plan)}]
    pr = dict(
        number=state["pr"],
        head={"sha": state["head"]},
        base={"sha": state["base"]},
        mergeable=True,
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda *args: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "load_state", lambda *args: state)
    monkeypatch.setattr(assist, "latest_command", lambda *args: None)
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(assist, "context", lambda *args: {"native_checks": [check]})
    tasks = [task]
    monkeypatch.setattr(assist, "validate_plan", lambda *args: tasks)
    monkeypatch.setattr(assist, "runs_for", lambda *args: [])
    monkeypatch.setattr(
        assist,
        "api",
        lambda *args: dict(
            path=".github/workflows/pr-ci-plan.yml",
            conclusion="success",
            display_title=f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
        ),
    )
    monkeypatch.setattr(assist, "task_status", lambda *args, **kwargs: "passed")
    monkeypatch.setattr(
        assist, "dispatch", lambda *args: pytest.fail("CPU retried on GPU")
    )
    cpu = dict(workflow=workflow, status="waiting", run=101)
    monkeypatch.setattr(assist, "native_check", lambda *args: dict(cpu))
    messages = []
    monkeypatch.setattr(assist, "publish", lambda *args: messages.append(args[1]))
    assist.control(state["pr"])
    assert state["phase"] == "watching" and "1 waiting" in messages[-1]
    cpu["status"] = "failed"
    assist.control(state["pr"])
    assert state["phase"] == "manual" and "human intervention" in messages[-1]
    # A fresh CPU-only watch can finish without inventing a GPU task.
    state.update(phase="watching", statuses=[])
    tasks.clear()
    cpu["status"] = "passed"
    assist.control(state["pr"])
    assert state["phase"] == "done" and state["tasks"] == []
    comment = {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)}
    assert record(comment, "assist") == state
    state["native_checks"][0]["workflow"] = "arbitrary.yml"
    comment["body"] = marker("assist", state)
    assert record(comment, "assist") is None


def test_failed_task_retries_once_and_falls_back_only_before_submission(
    monkeypatch, selected
):
    task, state = selected
    monkeypatch.setattr(assist, "publish", lambda *args: None)
    dispatched = []
    monkeypatch.setattr(assist, "original_status", lambda *args: "failed")
    monkeypatch.setattr(assist, "dispatch", lambda *args: dispatched.append(args))
    assert assist.task_status(task, state, [], submit=False) == "failed"
    assert assist.task_status(task, state, [], submit=True) == "waiting"
    assert dispatched == [(task, state["head"], "gb200")]
    assert assist.task_status(task, state, [], submit=True) == "waiting"
    assert len(dispatched) == 1  # event races before the run becomes visible
    run = {
        "id": 101,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "actor": {"login": BOT},
        "display_title": assist.run_title(task, state["head"], "gb200"),
    }
    dispatched.clear()
    monkeypatch.setattr(assist, "report", lambda *args: "waiting")
    assert assist.task_status(task, state, [run], submit=True) == "waiting"
    assert not dispatched  # queued work never creates a second allocation
    monkeypatch.setattr(assist, "report", lambda *args: "unavailable")
    assert assist.task_status(task, state, [run], submit=True) == "waiting"
    assert dispatched == [(task, state["head"], "gb300")]
    dispatched.clear()
    monkeypatch.setattr(assist, "report", lambda *args: "failed")
    assert assist.task_status(task, state, [run], submit=True) == "failed"
    assert not dispatched


def test_queued_nvidia_uses_slurm_once_and_keeps_dispatch_authoritative(
    monkeypatch, selected
):
    task, state = selected
    run = dict(
        id=100,
        event="pull_request",
        head_sha=state["head"],
        name="NVIDIA B200 Tests",
    )
    job = dict(name="unit-test / runtime (b200v2-1gpu)", status="queued")
    monkeypatch.setattr(assist, "pages", lambda *args: [job])
    monkeypatch.setattr(assist, "publish", lambda *args: None)
    dispatched = []
    monkeypatch.setattr(assist, "dispatch", lambda *args: dispatched.append(args))
    assert assist.task_status(task, state, [run], submit=True) == "waiting"
    assert dispatched == [(task, state["head"], "gb200")]
    job.update(status="completed", conclusion="success")
    monkeypatch.setattr(
        assist, "native_result", lambda *args: pytest.fail("dispatch lost ownership")
    )
    assert assist.task_status(task, state, [run], submit=True) == "waiting"
    assert len(dispatched) == 1


def test_active_native_work_and_queued_slurm_or_amd_are_reused(monkeypatch, selected):
    task, state = selected
    run = dict(
        id=100,
        event="pull_request",
        head_sha=state["head"],
        name="NVIDIA B200 Tests",
    )
    queued = dict(name="unit-test / runtime (b200v2-1gpu)", status="queued")
    active = {**queued, "status": "in_progress"}
    monkeypatch.setattr(
        assist, "pages", lambda path, field: [queued if "/100/" in path else active]
    )
    monkeypatch.setattr(
        assist, "dispatch", lambda *args: pytest.fail("existing work duplicated")
    )
    runs = [run, {**run, "id": 101}]
    assert assist.task_status(task, state, runs, submit=True) == "waiting"
    assert state["run_ids"][assist.task_key(task)] == 101
    monkeypatch.setattr(assist, "pages", lambda *args: [queued])
    run["name"] = "NVIDIA GB200 Tests"
    assert assist.task_status(task, state, [run], submit=True) == "waiting"
    task.update(cluster="", runner="amd-1gpu", native_runners=["amd-1gpu"])
    run["name"] = "AMD Tests"
    queued["name"] = "unit-test / runtime (amd-1gpu)"
    assert assist.task_status(task, state, [run], submit=True) == "waiting"


def test_incomplete_old_plan_refreshes_once_and_failed_refresh_requests_help(
    monkeypatch, tmp_path, selected
):
    task, state = selected
    test = "test/runtime/test_multimodal_encoded_offload.py"
    data = {
        **{k: state[k] for k in ("repository", "pr", "head", "base")},
        "test_files": [test],
        "catalog": [
            {
                **task,
                "runners": task["native_runners"],
                "slurm_runners": {"gb200": [task["runner"]]},
                "targets": {"test_files": [test]},
            },
            {
                **task,
                "config": "test/ci/ut/other.yaml",
                "runners": task["native_runners"],
                "slurm_runners": {"gb200": [task["runner"]]},
                "targets": {"test_files": []},
            },
        ],
    }
    plan = {
        **{k: state[k] for k in ("version", "repository", "pr", "head", "base")},
        "run": 55,
        "tests": [test],
        "tasks": [
            dict(config="test/ci/ut/other.yaml", runner=task["runner"], cluster="gb200")
        ],
    }
    with pytest.raises(assist.CoverageError):
        assist.validate_plan(plan, data)
    data["paths"] = ["tokenspeed-mla/python/tokenspeed_mla/mla_decode_fp8.py"]
    plan["tests"] = []
    with pytest.raises(assist.CoverageError, match="serving CI task"):
        assist.validate_plan(plan, data)
    data["paths"] = []
    plan["tests"] = [test]
    comments = [{"user": {"login": BOT, "id": BOT_ID}, "body": marker("plan", plan)}]
    pr = dict(
        number=state["pr"],
        state="open",
        head={"sha": state["head"], "repo": {"full_name": REPO}},
        base={"sha": state["base"], "ref": "main"},
    )
    event = tmp_path / "event.json"
    event.write_text("{}")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda *args: pr)
    monkeypatch.setattr(assist, "load_state", lambda *args: state)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(assist, "context", lambda *args: data)
    title = f"CI plan #{state['pr']} | {state['head']} | {state['base']}"
    refresh = dict(
        path=".github/workflows/other.yml",
        event="workflow_dispatch",
        head_branch="main",
        actor={"login": BOT},
        conclusion="failure",
    )
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: (
            pr
            if path == f"pulls/{state['pr']}"
            else (
                refresh
                if path.endswith("/56")
                else dict(
                    path=".github/workflows/pr-ci-plan.yml",
                    conclusion="success",
                    display_title=title,
                )
            )
        ),
    )
    monkeypatch.setattr(
        assist, "dispatch", lambda *args: pytest.fail("incomplete coverage dispatched")
    )
    published, commands = [], []
    monkeypatch.setattr(assist, "publish", lambda *args: published.append(args))
    monkeypatch.setattr(assist, "command", lambda *args: commands.append(args))
    assist.control(state["pr"])
    assert state["phase"] == "waiting-plan" and state["plan_refresh"] == 55
    assert len(commands) == 1
    assert (
        f"head={state['head']}" in commands[0]
        and f"base={state['base']}" in commands[0]
    )
    assert (
        record(
            {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)},
            "assist",
        )
        == state
    )
    assist.control(state["pr"])
    assert len(commands) == 1 and len(published) == 1
    run = dict(
        id=56,
        event="workflow_dispatch",
        name=title,
        display_title=title,
        conclusion="failure",
        pull_requests=[],
        head_sha=state["base"],
    )
    event.write_text(json.dumps(dict(action="completed", workflow_run=run)))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    resolved = []
    monkeypatch.setattr(assist, "output", lambda *args: resolved.append(args))
    assist.resolve()
    assert resolved == [("pr", str(state["pr"]))]
    assist.control(state["pr"])
    assert state["phase"] == "waiting-plan" and len(commands) == 1
    refresh["path"] = ".github/workflows/pr-ci-plan.yml"
    assist.control(state["pr"])
    assert state["phase"] == "manual" and len(commands) == 1


def test_promotion_needs_nonempty_matching_task_result(selected):
    task, state = selected
    result = dict(
        source_sha=state["head"],
        config=task["config"],
        task=task["name"],
        runner=task["runner"],
        ok=True,
        executed_stages=["install"],
    )
    assert (
        assist.result_status(result, task, state["head"], task["runner"]) == "missing"
    )

    plan = {k: state[k] for k in ("repository", "pr", "head", "base")}
    plan.update(
        tests=[],
        tasks=[dict(config=task["config"], runner="amd-mi45x-cpu-test", cluster="")],
    )
    data = {
        **plan,
        "catalog": [{**task, "runners": ["amd-mi45x-cpu-test"]}],
        "test_files": [],
    }
    with pytest.raises(ValueError, match="supported assistance route"):
        assist.validate_plan(plan, data)
    result["executed_stages"].append("ut")
    assert assist.result_status(result, task, state["head"], task["runner"]) == "passed"
    result["source_sha"] = "c" * 40
    assert (
        assist.result_status(result, task, state["head"], task["runner"]) == "missing"
    )


def test_conflict_patch_preserves_main_and_can_be_cherry_picked(tmp_path):
    def git(*args):
        return assist.command(
            "git", "-c", "core.hooksPath=/dev/null", *args, cwd=tmp_path
        )

    git("init", "-b", "main")
    repair.identity(tmp_path)
    file = tmp_path / "model.py"
    file.write_text("value = 1\n")
    git("add", ".")
    git("commit", "-s", "-m", "initial")
    common = git("rev-parse", "HEAD")
    file.write_text("value = 2\n")
    other = tmp_path / "base.py"
    other.write_text("base_only = True\n")
    git("add", ".")
    git("commit", "-s", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-b", "bot/test", common)
    file.write_text("value = 3\n")
    git("add", ".")
    git("commit", "-s", "-m", "head")
    head = git("rev-parse", "HEAD")
    assert repair.merge(tmp_path, base, commit=False) == ["model.py"]
    file.write_text("value = 4\n")
    repair.restore_patch(tmp_path, head, {"model.py"})
    assert not other.exists()  # no wholesale main changes in the PR patch
    repair.guard(tmp_path, head, {"model.py"})
    git("add", ".")
    git("commit", "-s", "-m", "repair")
    patch = git("rev-parse", "HEAD")
    tree = repair.effective_merge(tmp_path, base, head)
    assert other.read_text() == "base_only = True\n"
    assert file.read_text() == "value = 4\n"
    repair.commit_merge(tmp_path, "validation")
    assert git("rev-parse", "HEAD^{tree}") == tree
    git("checkout", "--detach", head)
    git("cherry-pick", "--signoff", patch)
    assert git("rev-parse", "HEAD^1") == head
    assert repair.effective_merge(tmp_path, base, head) == tree
    repair.commit_merge(tmp_path, "reconcile main")
    assert git("rev-parse", "HEAD^2") == base
    assert git("rev-parse", "HEAD^{tree}") == tree
    write_source(tmp_path, git("rev-parse", "HEAD"), "test/ci/example.yaml", "amd-1gpu")
    proof = json.loads(tmp_path.joinpath(".ci-artifacts/source.json").read_text())
    assert proof["source_sha"] == git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="selected commit"):
        write_source(tmp_path, "f" * 40, "test/ci/example.yaml", "amd-1gpu")


def test_source_change_stops_before_dispatch(monkeypatch, tmp_path, selected):
    _, state = selected
    pr = {
        "number": state["pr"],
        "head": {"sha": "c" * 40},
        "base": {"sha": state["base"]},
    }
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda n: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: [])
    monkeypatch.setattr(assist, "load_state", lambda *args: state)
    event = tmp_path / "event.json"
    event.write_text("{}")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    published = []
    monkeypatch.setattr(assist, "publish", lambda *args: published.append(args))
    monkeypatch.setattr(
        assist, "dispatch", lambda *args: pytest.fail("stale source dispatched")
    )
    assist.control(state["pr"])
    assert state["phase"] == "stale" and len(published) == 1


def test_watch_failure_then_authorized_fix_waits_for_candidate_validation(
    monkeypatch, tmp_path, selected
):
    task, state = selected
    plain = {k: task[k] for k in ("config", "runner", "cluster")}
    data = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    data.update(
        paths=["model.py"],
        test_files=[],
        catalog=[
            {
                "config": task["config"],
                "name": task["name"],
                "type": task["type"],
                "runners": task["native_runners"],
                "triggers": task["triggers"],
                "slurm_runners": {"gb200": [task["runner"]], "gb300": [task["runner"]]},
            }
        ],
    )
    plan = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    plan.update(run=55, tests=[], tasks=[plain])
    comments = [{"user": {"login": BOT, "id": BOT_ID}, "body": marker("plan", plan)}]
    pr = {
        "number": state["pr"],
        "head": {"sha": state["head"]},
        "base": {"sha": state["base"]},
        "mergeable": True,
    }
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"comment": {"id": 42}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setattr(assist, "WORK", tmp_path)
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda n: pr)
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(assist, "context", lambda *args: data)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    live = [None]
    monkeypatch.setattr(assist, "load_state", lambda *args: copy.deepcopy(live[0]))
    monkeypatch.setattr(assist, "latest_command", lambda *args: author)

    def publish(s, message):
        live[0] = copy.deepcopy(s)

    monkeypatch.setattr(assist, "publish", publish)
    author = {"id": 42, "body": "watch"}
    monkeypatch.setattr(assist, "permitted", lambda c: c["body"])
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: (
            {
                "name": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
                "path": ".github/workflows/pr-ci-plan.yml",
                "conclusion": "success",
                "display_title": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
            }
            if "actions/runs" in path
            else author
        ),
    )
    runs = []
    monkeypatch.setattr(assist, "runs_for", lambda *args: runs)
    monkeypatch.setattr(assist, "original_status", lambda *args: "failed")
    dispatched = []
    monkeypatch.setattr(assist, "dispatch", lambda *args: dispatched.append(args))
    assist.control(state["pr"])
    assert len(dispatched) == 1 and live[0]["phase"] == "watching"
    runs.append(
        {
            "id": 101,
            "event": "workflow_dispatch",
            "head_branch": "main",
            "actor": {"login": BOT},
            "display_title": assist.run_title(task, state["head"], "gb200"),
        }
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setattr(assist, "report", lambda *args: "failed")
    assist.control(state["pr"])
    assert live[0]["phase"] == "manual" and len(dispatched) == 1
    author.update(id=43, body="fix")
    event.write_text(json.dumps({"comment": {"id": 43}}))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    assist.control(state["pr"])
    assert live[0]["phase"] == "repairing" and emitted == [("repair", "true")]
    assert tmp_path.joinpath("request.json").is_file()
    assert live[0]["repair_run"] == 200
    # The validation branch has a different immutable source; an old head pass
    # must not authorize promotion of that candidate.
    live[0].update(
        phase="validating",
        candidate={
            "patch": "c" * 40,
            "validation": "d" * 40,
            "tree": "e" * 40,
            "branch": "bot/pr-ci-assist-123-43",
        },
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    promoted = []
    monkeypatch.setattr(repair, "promote", lambda s: promoted.append(copy.deepcopy(s)))
    monkeypatch.setattr(assist, "report", lambda *args: "passed")
    assist.control(state["pr"])
    assert not promoted and len(dispatched) == 2
    runs.insert(
        0,
        {
            **runs[0],
            "id": 102,
            "display_title": assist.run_title(task, "d" * 40, "gb200"),
        },
    )
    # Native PR CPU checks must not gate conflict repair: a conflicted PR
    # cannot start those workflows. Candidate GPU validation remains mandatory.
    workflow = "scheduler-cpp-test.yml"
    data["native_checks"] = [{"workflow": workflow, **NATIVE_CHECKS[workflow]}]
    monkeypatch.setattr(
        assist,
        "native_check",
        lambda *args: pytest.fail("PR CPU result used for candidate"),
    )
    assist.control(state["pr"])
    assert len(promoted) == 1 and live[0]["phase"] == "promoted"


def test_cancelled_repair_is_recovered_on_next_reconciliation(
    monkeypatch, tmp_path, selected
):
    task, state = selected
    state.update(action="fix", phase="repairing", repair_run=201)
    plan = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    plan.update(run=55, tests=[], tasks=state["tasks"])
    comments = [{"user": {"login": BOT, "id": BOT_ID}, "body": marker("plan", plan)}]
    pr = {
        "number": state["pr"],
        "head": {"sha": state["head"]},
        "base": {"sha": state["base"]},
    }
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda n: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "load_state", lambda *args: state)
    monkeypatch.setattr(assist, "latest_command", lambda *args: None)
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(assist, "context", lambda *args: {})
    monkeypatch.setattr(assist, "validate_plan", lambda *args: [task])
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: (
            {"status": "completed", "conclusion": "cancelled"}
            if path.endswith("/201")
            else {
                "name": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
                "path": ".github/workflows/pr-ci-plan.yml",
                "conclusion": "success",
                "display_title": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
            }
        ),
    )
    messages = []
    monkeypatch.setattr(assist, "publish", lambda *args: messages.append(args))
    monkeypatch.setattr(
        assist, "dispatch", lambda *args: pytest.fail("cancelled repair dispatched")
    )
    assist.control(state["pr"])
    assert state["phase"] == "manual" and len(messages) == 1


def test_dispatch_event_uses_tested_source_not_main_controller(
    monkeypatch, tmp_path, selected
):
    event = tmp_path / "event.json"
    tested, controller = "c" * 40, "b" * 40
    event.write_text(
        json.dumps(
            {
                "action": "completed",
                "workflow_run": {
                    "event": "workflow_dispatch",
                    "pull_requests": [],
                    "head_sha": controller,
                    "display_title": f"Slurm {tested} | test/ci/ut/example.yaml | b200-1gpu | gb200",
                },
            }
        )
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    _, state = selected
    state["head"] = tested
    command_comment = dict(
        id=state["command"],
        body="@lightseek-bot watch",
        user={"login": "example"},
        issue_url=f"https://api.github.com/repos/{REPO}/issues/123",
    )
    comments = []

    def pages(path, field):
        if path == "issues/123/comments":
            return comments
        if path == f"commits/{tested}/pulls":
            return [
                {"number": 999, "state": "closed"},
                {
                    "number": 123,
                    "state": "open",
                    "head": {"repo": {"full_name": REPO}},
                    "base": {"ref": "main"},
                },
                {
                    "number": 124,
                    "state": "open",
                    "head": {"repo": {"full_name": "untrusted-source"}},
                    "base": {"ref": "main"},
                },
            ]
        if path == f"commits/{tested}/branches-where-head":
            return []
        pytest.fail("controller commit confused with tested source")

    monkeypatch.setattr(assist, "pages", pages)
    resolved = []
    pr = dict(
        number=123,
        state="open",
        head={"repo": {"full_name": REPO}},
        base={"ref": "main"},
    )

    def api(path):
        if path == "collaborators/example/permission":
            return {"permission": "write"}
        if path == f"issues/comments/{state['command']}":
            return command_comment
        assert path == "pulls/123"
        resolved.append(123)
        return pr

    monkeypatch.setattr(assist, "api", api)
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    assist.resolve()
    assert resolved == [123] and not emitted
    comments.append(command_comment)
    assist.resolve()
    assert resolved == [123, 123] and emitted == [("pr", "123")]
    comments.append(
        {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)}
    )
    emitted.clear()
    assist.resolve()
    assert emitted == [("pr", "123")]
    state["phase"] = "done"
    comments[-1]["body"] = marker("assist", state)
    emitted.clear()
    assist.resolve()
    assert not emitted
    comments.append({**command_comment, "id": state["command"] + 2})
    assist.resolve()
    assert emitted == [("pr", "123")]
    pr["state"] = "closed"
    emitted.clear()
    assist.resolve()
    assert not emitted
    with pytest.raises(ValueError, match="open same-repository"):
        assist.pull(123)
