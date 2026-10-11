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
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts"))
import pr_ci_assist as assist
import pr_ci_common
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


def test_targeted_fix_command_keeps_authorization_and_pr_binding(monkeypatch):
    url = f"https://github.com/{REPO}/actions/runs/101/job/201?pr=123"
    comment = dict(body=f"@lightseek-bot fix {url}", user={"login": "example"})
    monkeypatch.setattr(assist, "api", lambda path: {"permission": "read"})
    assert assist.permitted(comment) is None
    monkeypatch.setattr(assist, "api", lambda path: {"permission": "write"})
    assert assist.permitted(comment) == "fix"
    assert assist.command_target(comment, 123) == {"run": 101, "job": 201}
    with pytest.raises(ValueError, match="another PR"):
        assist.command_target(comment, 124)
    comment["body"] = f"@lightseek-bot watch {url}"
    assert assist.permitted(comment) is None
    comment["body"] = (
        f"@lightseek-bot fix {url.replace(REPO, 'pre-commit/pre-commit-hooks')}"
    )
    assert assist.permitted(comment) is None


@pytest.fixture
def targeted(selected):
    task, state = selected
    state.update(action="fix", target={"run": 101, "job": 201})
    amd = dict(config=task["config"], runner="amd-mi35x-1gpu-test", cluster="")
    b200 = {**amd, "runner": "b200v2-1gpu"}
    data = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    data.update(
        paths=["model.py", "test/test_model.py"],
        test_files=[],
        catalog=[
            dict(
                config=task["config"],
                name=task["name"],
                type=task["type"],
                runners=[amd["runner"], b200["runner"]],
                slurm_runners={},
                triggers=task["triggers"],
            )
        ],
    )
    plan = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    plan.update(run=55, tests=[], tasks=[amd])
    job = dict(
        id=201,
        run_id=101,
        run_attempt=1,
        status="completed",
        conclusion="failure",
        head_sha=state["head"],
        name=f"unit-test / {task['name']} ({b200['runner']})",
    )
    run = dict(
        id=101,
        run_attempt=1,
        status="in_progress",
        conclusion=None,
        event="pull_request",
        head_sha=state["head"],
        head_repository={"full_name": REPO},
        path=".github/workflows/nvidia-b200-tests.yml",
        pull_requests=[dict(number=state["pr"], head={"sha": state["head"]})],
    )
    return state, data, plan, job, run, b200


def test_failed_job_in_running_workflow_adds_exact_backend(monkeypatch, targeted):
    state, data, plan, job, run, b200 = targeted
    with pytest.raises(ValueError, match="supported assistance route"):
        assist.validate_plan({**plan, "tasks": [b200]}, data)
    monkeypatch.setattr(assist, "api", lambda path: job if "/jobs/" in path else run)
    updated = assist.targeted_plan(plan, data, state)
    assert updated["tasks"] == [*plan["tasks"], b200]
    assert assist.targeted_plan(updated, data, state) == updated
    assert (
        record(
            {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)},
            "assist",
        )
        == state
    )
    sent = []
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "command", lambda *args: sent.append(args))
    assist.dispatch(assist.validate_plan(updated, data)[-1], "c" * 40, "")
    assert "runner_pool=b200v2" in sent[0] and "runner=b200v2-1gpu" in sent[0]
    job["head_sha"] = "d" * 40
    with pytest.raises(ValueError, match="current PR commit"):
        assist.targeted_plan(plan, data, state)
    job["head_sha"] = state["head"]
    job["run_id"] = 102
    with pytest.raises(ValueError, match="current PR commit"):
        assist.targeted_plan(plan, data, state)
    job["run_id"] = 101
    run["run_attempt"] = 2
    with pytest.raises(ValueError, match="current PR commit"):
        assist.targeted_plan(plan, data, state)


def test_targeted_fix_enters_repair_and_waits_for_target_validation(
    monkeypatch, tmp_path, targeted
):
    state, data, plan, job, run, b200 = targeted
    author = dict(
        id=43,
        body=f"@lightseek-bot fix https://github.com/{REPO}/actions/runs/101/job/201",
    )
    pr = dict(
        number=state["pr"],
        head={"sha": state["head"]},
        base={"sha": state["base"]},
        mergeable=True,
    )
    comments = [{"user": {"login": BOT, "id": BOT_ID}, "body": marker("plan", plan)}]
    live = [None]

    def api(path):
        if path == "actions/jobs/201":
            return job
        if path == "actions/runs/101":
            return run
        if path == "git/ref/heads/main":
            return {"object": {"sha": state["base"]}}
        return dict(
            path=".github/workflows/pr-ci-plan.yml",
            conclusion="success",
            status="completed",
            display_title=f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
            run_started_at="2030-01-01T00:00:00Z",
        )

    monkeypatch.setattr(assist, "api", api)
    monkeypatch.setattr(assist, "WORK", tmp_path)
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda n: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "load_state", lambda *args: copy.deepcopy(live[0]))
    monkeypatch.setattr(assist, "latest_command", lambda *args: author)
    monkeypatch.setattr(assist, "permitted", lambda c: "fix")
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(assist, "context", lambda *args: copy.deepcopy(data))
    monkeypatch.setattr(assist, "runs_for", lambda *args: [])
    monkeypatch.setattr(
        assist, "publish", lambda s, m: live.__setitem__(0, copy.deepcopy(s))
    )
    monkeypatch.setattr(
        assist,
        "task_status",
        lambda *args, **kw: pytest.fail("Original jobs need not finish before repair"),
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    assist.control(state["pr"])
    request = json.loads((tmp_path / "request.json").read_text())
    assert live[0]["phase"] == "repairing"
    assert request["state"]["target"] == state["target"]
    assert request["plan"]["tasks"][-1] == b200
    assert repair.allowed_paths(request) == {"model.py", "test/test_model.py"}
    monkeypatch.setattr(assist, "repair_plan", lambda _: request["plan"])
    candidate = dict(
        patch="c" * 40,
        validation="d" * 40,
        tree="e" * 40,
        branch=f"bot/pr-ci-assist-{state['pr']}-43-200",
    )
    live[0].update(phase="validating", candidate=candidate)
    monkeypatch.setattr(assist, "main_merge_clean", lambda *args, **kwargs: True)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setattr(
        assist,
        "task_status",
        lambda t, *args, **kw: "waiting" if t["runner"] == b200["runner"] else "passed",
    )
    promoted = []
    monkeypatch.setattr(repair, "promote", lambda s, *, deadline: promoted.append(s))
    assist.control(state["pr"])
    assert live[0]["phase"] == "validating" and not promoted
    monkeypatch.setattr(assist, "task_status", lambda *args, **kw: "passed")
    assist.control(state["pr"])
    assert live[0]["phase"] == "promoted" and len(promoted) == 1


def test_targeted_diagnostics_use_completed_job_log_endpoint(
    monkeypatch, tmp_path, targeted
):
    state, data, plan, job, run, _ = targeted
    request = dict(state=state, data=data, plan=plan)
    (tmp_path / "request.json").write_text(json.dumps(request))
    monkeypatch.setattr(repair, "WORK", tmp_path)
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path / "provider"))
    monkeypatch.setattr(repair, "api", lambda path: job)
    monkeypatch.setattr(
        repair,
        "org_variables",
        lambda *args: {"KIMI_API_URL": "provider", "KIMI_MODEL": "planner"},
    )
    calls = []

    def command(*args):
        calls.append(args)
        assert "--allow-escape-sequences" in args
        return "\x1b[31mtarget job failure evidence\x1b[0m"

    monkeypatch.setattr(repair, "command", command)
    repair.configure()
    assert calls == [
        ("gh", "api", "--allow-escape-sequences", f"repos/{REPO}/actions/jobs/201/logs")
    ]
    text = (tmp_path / "model/diagnostics.txt").read_text()
    assert job["name"] in text and "target job failure evidence" in text


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
    assert triggers["pull_request"]["branches"] == ["main"]
    assert "concurrency" not in workflow
    assert workflow["jobs"]["plan"]["concurrency"]["cancel-in-progress"] is True
    condition = (
        workflow["jobs"]["plan"]["if"]
        .replace("\n", " ")
        .replace("&&", "and")
        .replace("||", "or")
    )
    github = SimpleNamespace(
        repository=REPO,
        event_name="pull_request",
        actor="author",
        event=SimpleNamespace(
            action="edited",
            changes=SimpleNamespace(base=None),
            pull_request=SimpleNamespace(
                head=SimpleNamespace(repo=SimpleNamespace(full_name=REPO))
            ),
        ),
    )
    assert not eval(condition, {"__builtins__": {}}, {"github": github})
    github.event.changes.base = {"ref": {"from": "feature"}}
    assert eval(condition, {"__builtins__": {}}, {"github": github})
    github.event.action = "synchronize"
    github.event.changes.base = None
    assert eval(condition, {"__builtins__": {}}, {"github": github})
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
            workflow_run=SimpleNamespace(
                event="push",
                name="AMD Tests",
                path=".github/workflows/amd-tests.yml",
                conclusion="skipped",
            ),
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
    github.event.workflow_run.name = "CI plan #123 | head | base"
    github.event.workflow_run.path = ".github/workflows/pr-ci-plan.yml"
    assert not eval(condition, {"__builtins__": {}}, context)
    github.event.workflow_run.conclusion = "success"
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


@pytest.fixture
def published_comments(monkeypatch, tmp_path):
    comments = []

    def get_comment(path):
        return next(c for c in comments if path == f"issues/comments/{c['id']}")

    def command(*args):
        if args == ("gh", "auth", "token", "--hostname", "github.com"):
            return "test-token"
        assert args[:3] == ("gh", "pr", "comment")
        assert args[4:6] == ("--repo", REPO)
        comment = dict(
            id=len(comments) + 1,
            user={"login": BOT, "id": BOT_ID},
            body=args[-1],
        )
        comments.append(comment)
        return f"https://github.com/{REPO}/pull/123#issuecomment-{comment['id']}"

    def patch(request, *, timeout):
        assert request.method == "PATCH" and timeout == 30
        assert request.get_header("Authorization") == "Bearer test-token"
        prefix = f"https://api.github.com/repos/{REPO}/"
        assert request.full_url.startswith(prefix)
        get_comment(request.full_url.removeprefix(prefix))["body"] = json.loads(
            request.data
        )["body"]
        return io.BytesIO()

    monkeypatch.setattr(assist, "WORK", tmp_path)
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "api", get_comment)
    monkeypatch.setattr(assist, "command", command)
    monkeypatch.setattr(pr_ci_common, "urlopen", patch)
    return comments


def test_publish_saves_progress_without_new_comments(selected, published_comments):
    task, state = selected
    assist.publish(state, "Watching selected tasks.")
    initial = published_comments[0]["body"].split("<!--")[0]
    # A newer bot comment must not redirect edits away from the state record.
    unrelated = dict(id=2, user={"login": BOT, "id": BOT_ID}, body="Other update.")
    published_comments.append(unrelated)

    state["submitted"].append(assist.run_title(task, state["head"], "gb200"))
    assist.publish(state, "Selected task dispatch requested.")
    state["statuses"] = ["waiting"]
    state["run_ids"][assist.task_key(task)] = 101
    assist.publish(state, "1 waiting.")
    assert len(published_comments) == 2
    assert unrelated["body"] == "Other update."
    assert published_comments[0]["body"].split("<!--")[0] == initial
    assert record(published_comments[0], "assist") == state

    state.update(phase="done", statuses=["passed"])
    assist.publish(state, "1 passed.")
    assert len(published_comments) == 3
    assert "1 passed." in published_comments[-1]["body"]
    assert "| Check | Source | Result |" in published_comments[-1]["body"]
    assist.publish(state, "1 passed.")
    assert len(published_comments) == 3


def test_publish_keeps_latest_record_readable(selected, published_comments):
    _, state = selected
    assist.publish(state, "Started.")
    first_id = published_comments[0]["id"]
    # A newer command updates the same record instead of adding a comment.
    assist.publish({**state, "command": 43}, "Started.")
    assert len(published_comments) == 1
    # A new push also updates the same record without notifying subscribers.
    assist.publish({**state, "head": "b" * 40}, "Started.")
    assert len(published_comments) == 1
    state["statuses"] = ["waiting"]
    assist.publish(state, "1 waiting.")
    assert len(published_comments) == 1
    latest = assist.latest_state_comment(published_comments, state["pr"])
    assert latest["id"] == first_id and record(latest, "assist") == state
    # Terminal outcomes notify once and their records are never edited.
    state.update(phase="done", statuses=["passed"])
    assist.publish(state, "1 passed.")
    assert len(published_comments) == 2
    assist.publish({**state, "command": 43, "phase": "watching"}, "Started.")
    assert len(published_comments) == 3
    assert "1 passed." in published_comments[-2]["body"]


def test_publish_verifies_edited_state(monkeypatch, selected, published_comments):
    _, state = selected
    assist.publish(state, "Started.")
    stale = copy.deepcopy(published_comments[0])
    state["statuses"] = ["waiting"]
    monkeypatch.setattr(assist, "api", lambda path: stale)
    with pytest.raises(ValueError, match="Published state differs"):
        assist.publish(state, "1 waiting.")


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
    state["action"] = "fix"
    run["conclusion"] = "failure"
    assert assist.native_check(check, state, [run])["status"] == "failed"
    run["conclusion"] = "success"
    assert assist.native_check(check, state, [run])["run"] == 0
    run["head_sha"] = "d" * 40
    assert assist.native_check(check, state, [run])["run"] == 0


@pytest.mark.parametrize(
    ("action", "workflow"),
    [
        ("watch", "scheduler-cpp-test.yml"),
        ("fix", "nvidia-kernel-library-tests.yml"),
    ],
)
def test_native_check_waits_and_hands_off_failure_without_gpu_retry(
    monkeypatch, tmp_path, selected, action, workflow
):
    task, state = selected
    state["action"] = action
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
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setattr(assist, "WORK", tmp_path)
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda *args: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "load_state", lambda *args: state)
    monkeypatch.setattr(assist, "latest_command", lambda *args: None)
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(
        assist, "context", lambda *args: {"native_checks": [check], "paths": []}
    )
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
            run_started_at="2030-01-01T00:00:00Z",
            object={"sha": "c" * 40},
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
    if action == "fix":
        pr["mergeable"] = False
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    assist.control(state["pr"])
    if action == "fix":
        assert state["phase"] == "repairing" and emitted == [("repair", "true")]
        request = json.loads(tmp_path.joinpath("request.json").read_text())
        assert request["state"]["native_checks"] == [cpu]
        assert request["state"]["statuses"] == ["waiting"]
        assert request["state"]["validation_base"] == "c" * 40
        assert repair.NATIVE_CONFIG in repair.allowed_paths(request)
        assert (
            record(
                {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)},
                "assist",
            )
            == state
        )
        return
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


def test_candidate_native_run_requires_matching_source_and_executed_ut(
    monkeypatch, tmp_path, selected
):
    _, state = selected
    workflow = "nvidia-kernel-library-tests.yml"
    check = dict(workflow=workflow, **NATIVE_CHECKS[workflow])
    state["candidate"] = dict(
        patch="c" * 40,
        validation="d" * 40,
        tree="e" * 40,
        branch="bot/pr-ci-assist-123-42",
    )
    run = dict(
        id=102,
        path=f".github/workflows/{workflow}",
        event="workflow_dispatch",
        head_sha="d" * 40,
        head_branch=state["candidate"]["branch"],
        actor={"login": BOT},
        status="completed",
        conclusion="success",
    )
    job = dict(
        name=check["job"],
        status="completed",
        conclusion="success",
        steps=[dict(name=check["step"], status="completed", conclusion="success")],
    )
    monkeypatch.setattr(assist, "WORK", tmp_path)
    monkeypatch.setattr(
        assist,
        "pages",
        lambda path, field: (
            [job] if "/jobs" in path else [dict(name=check["artifact"], expired=False)]
        ),
    )
    proof = dict(
        source_sha="d" * 40,
        config=check["config"],
        runner=check["runner"],
        ok=True,
        executed_stages=["install", "ut"],
    )

    def download(run, name, target):
        target.joinpath("source.json").write_text(
            json.dumps({"source_sha": proof["source_sha"]})
        )
        target.joinpath("manifest.json").write_text(
            json.dumps(
                [
                    dict(
                        job_id="123",
                        task=dict(config=check["config"], runner=check["runner"]),
                        state="COMPLETED",
                        exit_code="0:0",
                    ),
                ]
            )
        )
        target.joinpath("123-result.json").write_text(json.dumps(proof))

    monkeypatch.setattr(assist, "download", download)
    assert assist.native_check(check, state, [run])["status"] == "passed"
    proof["executed_stages"] = ["install"]
    assert assist.native_check(check, state, [run])["status"] == "missing"
    proof["source_sha"] = state["head"]
    assert assist.native_check(check, state, [run])["status"] == "missing"
    for wrong in (
        dict(head_sha=state["head"]),
        dict(head_branch="main"),
        dict(actor={"login": "example"}),
        dict(event="pull_request"),
    ):
        assert assist.native_check(check, state, [{**run, **wrong}])["run"] == 0
    run.update(status="in_progress", conclusion=None)
    assert assist.native_check(check, state, [run])["status"] == "waiting"


def test_native_dispatch_reservation_prevents_duplicate_submission(
    monkeypatch, selected
):
    _, state = selected
    workflow = "nvidia-kernel-library-tests.yml"
    state.update(
        candidate=dict(
            patch="c" * 40,
            validation="d" * 40,
            tree="e" * 40,
            branch="bot/pr-ci-assist-123-42",
        ),
        native_checks=[dict(workflow=workflow, status="waiting", run=0)],
    )
    monkeypatch.setattr(repair, "guard_native_dispatch", lambda state: None)
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "api", lambda path: {"object": {"sha": "d" * 40}})
    published, dispatched = [], []
    monkeypatch.setattr(
        assist, "publish", lambda state, message: published.append(copy.deepcopy(state))
    )
    monkeypatch.setattr(assist, "command", lambda *args: dispatched.append(args))
    assist.dispatch_native_checks(state)
    assist.dispatch_native_checks(state)
    assert len(dispatched) == 1
    assert dispatched[0][-2:] == ("--ref", state["candidate"]["branch"])
    assert published[0]["native_submitted"] == [workflow]
    comment = dict(user=dict(login=BOT, id=BOT_ID), body=marker("assist", state))
    assert record(comment, "assist") == state
    state["repair_run"] = 201
    state["candidate"]["branch"] += "-201"
    comment["body"] = marker("assist", state)
    assert record(comment, "assist") == state
    state["repair_run"] = 202
    comment["body"] = marker("assist", state)
    assert record(comment, "assist") is None


def test_repair_progress_withholds_model_text_arguments_and_errors(capsys):
    seen = set()
    repair.repair_progress(
        json.dumps(
            dict(
                type="turn.step.retrying",
                error_name="APIConnectionError",
                error_message="OUTBOUND_SENTINEL",
            )
        ),
        seen,
    )
    repair.repair_progress(
        json.dumps(
            dict(
                role="assistant",
                content="OUTBOUND_SENTINEL",
                tool_calls=[
                    dict(function=dict(name="Read", arguments="OUTBOUND_SENTINEL"))
                ],
            )
        ),
        seen,
    )
    repair.repair_progress(
        json.dumps(
            dict(
                role="tool",
                content=json.dumps(dict(type="error", message="OUTBOUND_SENTINEL")),
            )
        ),
        seen,
    )
    output = capsys.readouterr().out
    assert "OUTBOUND_SENTINEL" not in output
    assert "APIConnectionError" in output and "tool requested: Read" in output


def test_native_task_repair_preserves_commands_and_protected_controls(tmp_path):
    def git(*args):
        return assist.command(
            "git", "-c", "core.hooksPath=/dev/null", *args, cwd=tmp_path
        )

    git("init", "-b", "main")
    repair.identity(tmp_path)
    task = tmp_path / repair.NATIVE_CONFIG
    task.parent.mkdir(parents=True)
    prefix = 'PYTHONPATH="python:tokenspeed-kernel/python"'
    suffix = "$" + "{PYTHONPATH:+:$PYTHONPATH}"
    original = (
        assist.ROOT.joinpath(repair.NATIVE_CONFIG)
        .read_text()
        .replace(prefix[:-1] + suffix + '"', prefix)
        .replace(
            'PYTHONPATH="tokenspeed-mla/python' + suffix + '" ',
            "PYTHONPATH=tokenspeed-mla/python ",
        )
    )
    task.write_text(original)
    git("add", ".")
    git("commit", "-s", "-m", "initial")
    head = git("rev-parse", "HEAD")
    request = dict(
        data=dict(
            paths=["model.py", repair.NATIVE_CONFIG, ".github/workflows/test.yml"]
        ),
        state=dict(
            native_checks=[
                dict(
                    workflow="nvidia-kernel-library-tests.yml", status="failed", run=101
                ),
            ]
        ),
    )
    allowed = repair.allowed_paths(request)
    assert allowed == {"model.py", repair.NATIVE_CONFIG}
    candidate = original.replace(prefix, prefix[:-1] + suffix + '"')
    task.write_text(candidate)
    assert repair.guard(tmp_path, head, allowed)
    assert repair.public_source_diff(tmp_path, head, {repair.NATIVE_CONFIG}) == ""
    task.write_text(
        candidate.replace(" -v --junitxml", " --collect-only -v --junitxml")
    )
    with pytest.raises(ValueError, match="retain the original tests"):
        repair.guard(tmp_path, head, allowed)
    # Main may update another command while this older PR needs its import fix.
    main_task = original.replace(
        "PYTHONPATH=tokenspeed-mla/python ",
        'PYTHONPATH="tokenspeed-mla/python${PYTHONPATH:+:$PYTHONPATH}" ',
    )
    task.write_text(main_task)
    assist.command("git", "add", ".", cwd=tmp_path)
    assist.command(
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-s",
        "-m",
        "base",
        cwd=tmp_path,
    )
    base = assist.command("git", "rev-parse", "HEAD", cwd=tmp_path)
    task.write_text(main_task.replace(prefix, prefix[:-1] + suffix + '"'))
    assert repair.guard(tmp_path, head, allowed, validation_base=base)
    task.write_text(candidate)
    with pytest.raises(ValueError, match="retain the original tests"):
        repair.guard(tmp_path, head, allowed, validation_base=base)
    task.write_text(candidate.replace(suffix, ":$PYTHONPATH"))
    with pytest.raises(ValueError, match="retain the original tests"):
        repair.guard(tmp_path, head, allowed)
    request["state"]["native_checks"][0]["status"] = "passed"
    assert repair.NATIVE_CONFIG not in repair.allowed_paths(request)


def test_runtime_lint_returns_unused_import_to_corrective_turn(monkeypatch, tmp_path):
    import time

    path = "python/tokenspeed/runtime/model.py"
    source = tmp_path / path
    source.parent.mkdir(parents=True)
    source.write_text("import math\n")
    # Repository files must not replace the installed checker in the model runner.
    tmp_path.joinpath("ruff.py").write_text('raise RuntimeError("untrusted checker")\n')
    request = dict(deadline=int(time.time()) + 60, data=dict(paths=[path]))
    feedback = tmp_path / "feedback.json"
    turns = []

    def edit(attempt):
        turns.append(attempt)
        if attempt:
            issue = json.loads(feedback.read_text())
            assert issue["category"] == "runtime-lint"
            assert issue["path"] == path
            assert issue["details"][0]["code"] == "F401"
            source.write_text("VALUE = 1\n")

    def proposal():
        repair.runtime_lint(tmp_path, request)
        return source.read_text()

    assert (
        repair.repair_with_feedback(request, edit, proposal, feedback) == "VALUE = 1\n"
    )
    assert turns == [0, 1]


def test_conflicted_test_resolution_preserves_both_sides_assertions(tmp_path, selected):
    def git(*args):
        return assist.command(
            "git", "-c", "core.hooksPath=/dev/null", *args, cwd=tmp_path
        )

    git("init", "-b", "main")
    repair.identity(tmp_path)
    path = "test/runtime/test_model.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("def test_result():\n    assert value == 0\n")
    other_path = "test/runtime/test_other.py"
    other = tmp_path / other_path
    other.write_text("assert other_value == 3\n")
    git("add", ".")
    git("commit", "-s", "-m", "initial")
    ancestor = git("rev-parse", "HEAD")
    file.write_text(
        "def test_result():\n    assert value == 0\n    assert value == 1\n"
    )
    git("add", ".")
    git("commit", "-s", "-m", "head")
    head = git("rev-parse", "HEAD")
    git("checkout", "--detach", ancestor)
    file.write_text(
        "def test_result():\n    torch.testing.assert_close(actual, expected, atol=0)\n"
    )
    other.write_text("assert other_value == 3\nassert other_value == 5\n")
    git("add", ".")
    git("commit", "-s", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "--detach", head)
    _, state = selected
    state.update(action="fix", validation_base=base, repair_run=201)
    request = dict(
        state=state,
        data=dict(paths=[path, "test/runtime/test_other.py"]),
        conflicts=True,
    )
    assert path not in repair.allowed_paths(request)
    request["conflicted_tests"] = [path]
    assert repair.allowed_paths(request) == {path}
    candidate = "def test_result():\n    assert value == 1\n    torch.testing.assert_close(actual, expected, atol=0)\n"
    file.write_text(candidate)
    repair.guard_test_assertions(tmp_path, head, base, path)
    request["state"].update(head=head, target=dict(run=101, job=201))
    repair.guard_targeted_tests(tmp_path, request)
    file.write_text(
        "import pytest\npytest.skip('disabled', allow_module_level=True)\n" + candidate
    )
    with pytest.raises(repair.RepairRejected, match="test assertions"):
        repair.guard_targeted_tests(tmp_path, request)
    file.write_text(
        candidate.replace("def test_result():\n", "def test_result():\n    return\n")
    )
    with pytest.raises(repair.RepairRejected, match="test assertions"):
        repair.guard_targeted_tests(tmp_path, request)
    file.write_text(candidate.replace("atol=0", "atol=1"))
    with pytest.raises(ValueError, match="test assertions"):
        repair.guard_test_assertions(tmp_path, head, base, path)
    file.write_text(candidate.replace("    assert value == 1\n", ""))
    with pytest.raises(repair.RepairRejected, match="test assertions") as rejection:
        repair.guard_test_assertions(tmp_path, head, base, path)
    assert rejection.value.feedback["details"] == ["assert value == 1"]
    other.write_text("assert other_value == 3\n")
    with pytest.raises(repair.RepairRejected) as rejection:
        repair.guard(tmp_path, head, {path, other_path}, validation_base=base)
    issues = rejection.value.feedback["issues"]
    assert {issue["path"] for issue in issues} == {path, other_path}
    assert issues[1]["details"] == ["assert other_value == 5"]


def test_source_screen_retains_public_parents_and_rejects_new_private_text(tmp_path):
    def git(*args):
        return assist.command(
            "git", "-c", "core.hooksPath=/dev/null", *args, cwd=tmp_path
        )

    git("init", "-b", "main")
    repair.identity(tmp_path)
    file = tmp_path / "model.py"
    file.write_text("# head https://example.com/v1\nvalue = 1\n")
    git("add", ".")
    git("commit", "-s", "-m", "initial")
    head = git("rev-parse", "HEAD")
    file.write_text("# main https://example.com/v1\nvalue = 2\n")
    git("add", ".")
    git("commit", "-s", "-m", "base")
    base = git("rev-parse", "HEAD")
    file.write_text("# main https://example.com/v1\nvalue = 3\n")
    assert repair.guard(tmp_path, head, {"model.py"}, validation_base=base)
    assert repair.new_source_text(tmp_path, head, base, {"model.py"}) == "value = 3"
    file.write_text("# new https://example.com/v1\nvalue = 3\n")
    with pytest.raises(repair.RepairRejected, match="public-output"):
        repair.guard(tmp_path, head, {"model.py"}, validation_base=base)


def test_native_dispatch_rejects_changed_workflow_controls(
    monkeypatch, tmp_path, selected
):
    _, state = selected
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return assist.command(
            "git", "-c", "core.hooksPath=/dev/null", *args, cwd=source
        )

    git("init", "-b", "main")
    repair.identity(source)
    control = source / ".github/workflows/native.yml"
    control.parent.mkdir(parents=True)
    control.write_text("trusted\n")
    git("add", ".")
    git("commit", "-s", "-m", "initial")
    state["base"] = git("rev-parse", "HEAD")
    state["candidate"] = dict(validation=state["base"])
    pr = dict(draft=False, head=dict(repo=dict(full_name=REPO)))
    monkeypatch.setattr(repair, "WORK", tmp_path)
    monkeypatch.setattr(repair, "pull", lambda number: pr)
    command = repair.command
    monkeypatch.setattr(
        repair,
        "command",
        lambda *args, **kwargs: (
            "" if args[:2] == ("git", "fetch") else command(*args, **kwargs)
        ),
    )
    repair.guard_native_dispatch(state)
    control.write_text("changed\n")
    git("add", ".")
    git("commit", "-s", "-m", "change controls")
    state["candidate"]["validation"] = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="trusted native workflow"):
        repair.guard_native_dispatch(state)
    pr["draft"] = True
    with pytest.raises(ValueError, match="active same-repository"):
        repair.guard_native_dispatch(state)


def test_native_failure_diagnostics_include_slurm_artifact(monkeypatch, tmp_path):
    check = NATIVE_CHECKS["nvidia-kernel-library-tests.yml"]
    request = dict(
        state=dict(
            run_ids={},
            native_checks=[
                dict(
                    workflow="nvidia-kernel-library-tests.yml", status="failed", run=101
                ),
            ],
        ),
        plan=dict(tasks=[]),
        data={},
    )
    tmp_path.joinpath("request.json").write_text(json.dumps(request))
    monkeypatch.setattr(repair, "WORK", tmp_path)
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path / "provider"))
    monkeypatch.setattr(repair, "validate_plan", lambda *args: [])
    monkeypatch.setattr(repair, "api", lambda path: dict(run_attempt=1))

    def pages(path, field):
        if "/jobs" in path:
            return [dict(name=check["job"], conclusion="failure", id=201)]
        return [dict(name=check["artifact"], expired=False)]

    def command(*args):
        if "download" in args:
            target = Path(args[-1])
            target.joinpath("manifest.json").write_text(
                json.dumps(
                    [
                        dict(job_id="123", task=dict(config=check["config"])),
                    ]
                )
            )
            target.joinpath("123.log").write_text("native Slurm failure evidence")
            return ""
        return "native job failure evidence"

    monkeypatch.setattr(repair, "pages", pages)
    monkeypatch.setattr(repair, "command", command)
    monkeypatch.setattr(
        repair,
        "org_variables",
        lambda *args: {"KIMI_API_URL": "provider", "KIMI_MODEL": "planner"},
    )
    repair.configure()
    diagnostics = tmp_path.joinpath("model/diagnostics.txt").read_text()
    assert "native job failure evidence" in diagnostics
    assert "native Slurm failure evidence" in diagnostics
    request["state"]["target"] = dict(run=101, job=201)
    tmp_path.joinpath("request.json").write_text(json.dumps(request))
    monkeypatch.setattr(
        repair,
        "api",
        lambda path: (
            dict(id=201, run_id=101, name=check["job"], head_sha="a" * 40)
            if "/jobs/" in path
            else dict(run_attempt=1)
        ),
    )
    repair.configure()
    diagnostics = tmp_path.joinpath("model/diagnostics.txt").read_text()
    assert "native job failure evidence" in diagnostics
    assert "native Slurm failure evidence" in diagnostics


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
    run["name"] = "AMD Kernel Benchmark"
    queued["name"] = "kernel-benchmark / runtime (amd-1gpu)"
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
    unchanged = tmp_path / "context.py"
    unchanged.write_text("context = 1\n")
    test_path = "test/test_model.py"
    resolved = tmp_path / test_path
    resolved.parent.mkdir()
    resolved.write_text("assert value == 1\n")
    legacy = tmp_path / "test/test_legacy.py"
    legacy.write_text("# preserved\n" * 8 + "assert value == 1\n")
    renamed_path = "test/test_retained.py"
    retained = tmp_path / renamed_path
    git("add", ".")
    git("commit", "-s", "-m", "initial")
    common = git("rev-parse", "HEAD")
    file.write_text("value = 2\n")
    unchanged.write_text("context = 2\n")
    resolved.unlink()
    legacy.rename(retained)
    retained.write_text(retained.read_text() + "assert value == 2\n")
    other = tmp_path / "base.py"
    other.write_text("base_only = True\n")
    git("add", ".")
    git("commit", "-s", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-b", "bot/test", common)
    file.write_text("value = 3\n")
    unchanged.write_text("context = 3\n")
    resolved.write_text("assert value == 1\nassert value == 3\n")
    legacy.unlink()
    git("add", ".")
    git("commit", "-s", "-m", "head")
    head = git("rev-parse", "HEAD")
    conflicts = {"model.py", "context.py", test_path, renamed_path}
    assert set(repair.merge(tmp_path, base, commit=False)) == conflicts
    file.write_text("value = 4\n")
    repair.restore_patch(tmp_path, head, {"model.py", renamed_path})
    assert not other.exists()  # no wholesale main changes in the PR patch
    file.write_text("value =\n")
    with pytest.raises(repair.RepairRejected, match="not valid Python"):
        repair.guard(tmp_path, head, {"model.py", renamed_path}, validation_base=base)
    file.write_text("value = 4\n")
    allowed = conflicts
    diff = repair.guard(tmp_path, head, allowed, validation_base=base)
    assert f"b/{renamed_path}" in diff
    retained.chmod(0o755)
    with pytest.raises(repair.RepairRejected, match="file type or mode"):
        repair.guard(tmp_path, head, allowed, validation_base=base)
    retained.chmod(0o644)
    repair.restore_patch(tmp_path, head, allowed)
    assert file.read_text() == "value = 4\n"
    assert retained.read_text().endswith("assert value == 2\n")
    repair.guard(tmp_path, head, allowed, validation_base=base)
    git("add", ".")
    git("commit", "-s", "-m", "repair")
    patch = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="beyond the reviewed patch"):
        repair.effective_merge(tmp_path, base, head, resolved_paths=set())
    git("merge", "--abort")
    tree = repair.effective_merge(tmp_path, base, head, resolved_paths=conflicts)
    assert other.read_text() == "base_only = True\n"
    assert file.read_text() == "value = 4\n"
    assert unchanged.read_text() == "context = 3\n"
    assert resolved.read_text() == "assert value == 1\nassert value == 3\n"
    repair.commit_merge(tmp_path, "validation")
    assert git("rev-parse", "HEAD^{tree}") == tree
    git("checkout", "--detach", head)
    git("cherry-pick", "--signoff", patch)
    assert git("rev-parse", "HEAD^1") == head
    assert (
        repair.effective_merge(tmp_path, base, head, resolved_paths=conflicts) == tree
    )
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


@pytest.fixture
def main_advance_repo(tmp_path, selected):
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return assist.command(
            "git", "-c", "core.hooksPath=/dev/null", *args, cwd=source
        )

    def commit(path, text):
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        git("add", "--all")
        git("commit", "-s", "-m", "test input")
        return git("rev-parse", "HEAD")

    git("init", "-b", "main")
    repair.identity(source)
    path = "tokenspeed-kernel-amd/python/transform.py"
    commit(path, "value = 1\n")
    shared = "tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/cuda.py"
    base = commit(shared, "is_amd = True\n")
    git("checkout", "-b", "pr")
    head = commit(path, "value = 2\n")
    patch = commit(path, "value = 3\n")
    _, state = selected
    state.update(
        head=head,
        base=base,
        validation_base=base,
        repair_run=200,
        action="fix",
        phase="validating",
        native_checks=[],
        candidate=dict(
            patch=patch,
            validation=patch,
            tree=git("rev-parse", "HEAD^{tree}"),
            branch="bot/pr-ci-assist-123-42-200",
        ),
    )
    state["tasks"][0].update(runner="amd-mi35x-1gpu-test", cluster="")
    git("checkout", "main")
    return source, git, commit, state, path, shared


def test_pr_source_matches_allows_main_advance_within_validation_base(
    main_advance_repo, monkeypatch
):
    source, _, commit, state, _, _ = main_advance_repo
    docs = commit("docs/guide.md", "Documentation\n")
    other_vendor = commit("tokenspeed-mla/python/kernel.py", "value = 1\n")
    pr = dict(head=dict(sha=state["head"]), base=dict(sha=docs))
    monkeypatch.setattr(assist, "api", lambda _: {"object": {"sha": other_vendor}})
    assert assist.pr_source_matches(state, pr, source=source)
    pr["head"]["sha"] = docs
    assert not assist.pr_source_matches(state, pr, source=source)


def test_promotion_reuses_original_tree_but_rechecks_latest_main(
    main_advance_repo, monkeypatch, tmp_path
):
    source, git, commit, state, path, _ = main_advance_repo
    main = commit("docs/guide.md", "Documentation\n")
    changed = commit(path, "value = 4\n")
    git("remote", "add", "origin", str(source))
    git("checkout", "--detach", state["head"])
    pr = dict(head=dict(sha=state["head"], ref="pr"), base=dict(sha=state["base"]))
    monkeypatch.setattr(repair, "ROOT", source)
    monkeypatch.setattr(repair, "WORK", tmp_path)
    monkeypatch.setattr(repair, "public_gate", lambda: None)
    monkeypatch.setattr(repair, "pull", lambda _: pr)
    monkeypatch.setattr(repair, "pages", lambda *args: [])
    monkeypatch.setattr(repair, "latest_command", lambda _: {"id": state["command"]})
    monkeypatch.setattr(repair.time, "time", lambda: 50)
    mains = [main, main]

    def api(endpoint):
        return {
            "object": {
                "sha": (
                    mains.pop(0)
                    if endpoint == "git/ref/heads/main"
                    else state["candidate"]["validation"]
                )
            }
        }

    monkeypatch.setattr(repair, "api", api)
    pushed = []
    monkeypatch.setattr(
        repair, "push", lambda *args, **kwargs: pushed.append(git("rev-parse", "HEAD"))
    )
    repair.promote(state, deadline=100)
    assert len(pushed) == 1
    assert git("rev-parse", "HEAD^") == state["head"]
    assert git("rev-parse", "HEAD^{tree}") == state["candidate"]["tree"]
    assert not source.joinpath("docs/guide.md").exists()
    # A conflicting change arriving during promotion must stop the push, even
    # if the earlier documentation edit merged cleanly with the candidate.
    git("checkout", "--detach", state["head"])
    mains[:] = [main, changed]
    with pytest.raises(ValueError, match="conflicts with current main"):
        repair.promote(state, deadline=100)
    assert len(pushed) == 1


def test_repair_plan_requires_the_original_authorized_source(monkeypatch, selected):
    _, state = selected
    state.update(action="fix", repair_run=201)
    request = dict(state=copy.deepcopy(state), plan={"run": 55})
    run = dict(id=201, path=".github/workflows/pr-ci-assist.yml", event="issue_comment")
    monkeypatch.setattr(assist, "api", lambda _: run)

    def download(owner, name, target):
        assert owner == run and name == "repair-request"
        target.joinpath("request.json").write_text(json.dumps(request))

    monkeypatch.setattr(assist, "download", download)
    assert assist.repair_plan(state) == request["plan"]
    run.update(path=".github/workflows/pr-ci-assist-dispatch.yml", event="workflow_run")
    assert assist.repair_plan(state) == request["plan"]
    run["event"] = "issue_comment"
    with pytest.raises(ValueError, match="Unexpected repair workflow"):
        assist.repair_plan(state)
    run["event"] = "workflow_run"
    request["state"]["command"] += 1
    with pytest.raises(ValueError, match="another authorized source"):
        assist.repair_plan(state)
    run["path"] = ".github/workflows/pr-ci-plan.yml"
    with pytest.raises(ValueError, match="Unexpected repair workflow"):
        assist.repair_plan(state)


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

    def download(owner, name, target):
        request = json.loads(tmp_path.joinpath("request.json").read_text())
        assert owner["id"] == request["state"]["repair_run"]
        assert name == "repair-request"
        target.joinpath("request.json").write_text(json.dumps(request))

    monkeypatch.setattr(assist, "download", download)

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
                "id": int(path.rsplit("/", 1)[-1]),
                "event": "workflow_dispatch",
                "path": (
                    ".github/workflows/pr-ci-plan.yml"
                    if path == "actions/runs/55"
                    else ".github/workflows/pr-ci-assist.yml"
                ),
                "conclusion": "success",
                "display_title": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
                "status": "completed",
                "run_started_at": "2030-01-01T00:00:00Z",
                "object": {"sha": state["base"]},
            }
            if "actions/runs" in path
            else (
                {"object": {"sha": state["base"]}}
                if path == "git/ref/heads/main"
                else author
            )
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
    # Lint alone must enter repair even when the selected GPU task passed.
    runs.append(
        dict(
            id=102,
            path=".github/workflows/lint.yml",
            event="pull_request",
            head_sha=state["head"],
            status="completed",
            conclusion="failure",
            pull_requests=[dict(number=state["pr"], head=dict(sha=state["head"]))],
        )
    )
    monkeypatch.setattr(assist, "report", lambda *args: "passed")
    event.write_text(json.dumps({"comment": {"id": 43}}))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    assist.control(state["pr"])
    assert live[0]["phase"] == "repairing" and emitted == [("repair", "true")]
    assert tmp_path.joinpath("request.json").is_file()
    assert live[0]["repair_run"] == 200
    assert json.loads(tmp_path.joinpath("request.json").read_text())["lint_run"] == 102
    # A legacy monitor can leave an unowned manual state after cancellation of
    # the redundant original-head native run. Explicit dispatch still repairs
    # the current lint failure; the fresh candidate must run its native checks.
    workflow = "nvidia-kernel-library-tests.yml"
    data["native_checks"] = [{"workflow": workflow, **NATIVE_CHECKS[workflow]}]
    monkeypatch.setattr(
        assist, "native_check", lambda *args: dict(workflow=workflow, status="missing")
    )
    live[0]["phase"] = "manual"
    del live[0]["repair_run"]
    emitted.clear()
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    assist.control(state["pr"])
    assert live[0]["phase"] == "repairing" and emitted == [("repair", "true")]
    assert live[0]["repair_run"] == 200
    assert json.loads(tmp_path.joinpath("request.json").read_text())["lint_run"] == 102
    del data["native_checks"]
    runs.pop()
    monkeypatch.setattr(assist, "report", lambda *args: "failed")
    assert (
        json.loads(tmp_path.joinpath("request.json").read_text())["deadline"]
        == 1893459600
    )
    # A failed repair waits for an explicit dispatch before trying again with
    # the existing authorized fix; completion events must not create retries.
    live[0]["phase"] = "manual"
    emitted.clear()
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    assist.control(state["pr"])
    assert live[0]["phase"] == "manual" and not emitted
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_RUN_ID", "201")
    pr["head"]["sha"] = "f" * 40
    assist.control(state["pr"])
    assert not emitted and live[0]["phase"] == "manual"
    pr["head"]["sha"] = state["head"]
    # An explicit retry starts a fresh repair from the retained authorized fix.
    live[0].update(phase="stale", validation_base="f" * 40)
    assist.control(state["pr"])
    assert live[0]["phase"] == "repairing" and emitted == [("repair", "true")]
    assert live[0]["command"] == 43 and live[0]["repair_run"] == 201
    assert live[0]["validation_base"] == state["base"]
    request = json.loads(tmp_path.joinpath("request.json").read_text())
    assert set(request) == {
        "state",
        "plan",
        "data",
        "conflicts",
        "deadline",
        "lint_run",
    }
    # A newer plan for the same PR source must not change an existing candidate's
    # checks, even when a manual retry recovers state written by an old monitor.
    comments.append(
        {
            "user": {"login": BOT, "id": BOT_ID},
            "body": marker("plan", {**plan, "run": 56, "tasks": []}),
        }
    )
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
    monkeypatch.setattr(assist, "main_merge_clean", lambda *args, **kwargs: True)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    promoted = []
    promotion_deadlines = []

    def promote(s, *, deadline):
        promoted.append(copy.deepcopy(s))
        promotion_deadlines.append(deadline)

    monkeypatch.setattr(repair, "promote", promote)
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
    # Candidate native results must also pass; old PR results cannot promote it.
    workflow = "scheduler-cpp-test.yml"
    data["native_checks"] = [{"workflow": workflow, **NATIVE_CHECKS[workflow]}]
    monkeypatch.setattr(
        assist,
        "native_check",
        lambda *args: dict(workflow=workflow, status="waiting", run=103),
    )
    assist.control(state["pr"])
    assert not promoted and live[0]["phase"] == "validating"
    monkeypatch.setattr(
        assist,
        "native_check",
        lambda *args: dict(workflow=workflow, status="passed", run=103),
    )
    # Completed checks cannot authorize a late promotion after queueing consumed
    # the shared hour; the same candidate can promote inside its original budget.
    monkeypatch.setattr(assist.time, "time", lambda: 1893459600)
    assist.control(state["pr"])
    assert not promoted and live[0]["phase"] == "manual"
    live[0]["phase"] = "validating"
    monkeypatch.setattr(assist.time, "time", lambda: 1893459599)
    assist.control(state["pr"])
    assert len(promoted) == 1 and live[0]["phase"] == "promoted"
    # A manual rerun with a retained candidate harvests its existing results on
    # a 15-minute budget anchored at the new run, dispatching nothing; a later
    # rerun promotes the same candidate once every check has passed.
    live[0]["phase"] = "manual"
    assist.control(state["pr"])
    assert len(promoted) == 1 and live[0]["phase"] == "manual"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    original_api = assist.api
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: (
            {**original_api(path), "run_started_at": "2030-01-01T01:00:00Z"}
            if path == "actions/runs/999"
            else original_api(path)
        ),
    )
    monkeypatch.setattr(assist.time, "time", lambda: 1893459600)
    native_status = ["waiting"]
    monkeypatch.setattr(
        assist,
        "native_check",
        lambda *args: dict(workflow=workflow, status=native_status[0], run=103),
    )
    emitted.clear()
    assist.control(state["pr"])
    assert live[0]["phase"] == "manual" and live[0]["repair_run"] == 201
    assert len(promoted) == 1 and len(dispatched) == 2 and not emitted
    native_status[0] = "passed"
    assist.control(state["pr"])
    assert len(promoted) == 2 and live[0]["phase"] == "promoted"
    assert promotion_deadlines[-1] == 1893460500
    assert promoted[-1]["candidate"] == promoted[0]["candidate"]
    assert promoted[-1]["tasks"] == [plain]
    assert promoted[-1]["repair_run"] == 201
    assert promoted[-1]["validation_base"] == promoted[0]["validation_base"]
    assert len(dispatched) == 2 and not emitted


def test_owned_request_recovers_only_an_unowned_monitor_update(monkeypatch, selected):
    _, state = selected
    state.update(action="fix", phase="repairing", repair_run=201)
    request = dict(state=state, deadline=100)
    pr = dict(head=dict(sha=state["head"]), base=dict(sha=state["base"]))
    live = copy.deepcopy(state)
    live["phase"] = "manual"
    del live["repair_run"]
    monkeypatch.setenv("GITHUB_RUN_ID", "201")
    monkeypatch.setattr(repair.time, "time", lambda: 50)
    monkeypatch.setattr(repair, "repair_deadline", lambda _: 100)
    monkeypatch.setattr(repair, "pull", lambda _: pr)
    monkeypatch.setattr(repair, "pages", lambda *args: [])
    monkeypatch.setattr(repair, "load_state", lambda *args: live)
    monkeypatch.setattr(repair, "latest_command", lambda _: dict(id=state["command"]))
    published = []
    monkeypatch.setattr(
        repair, "publish", lambda s, _: published.append(copy.deepcopy(s))
    )
    assert repair.current_request(request) == pr
    assert published == [state] and request["deadline"] == 100
    # A different owner must never be overwritten by the previous repair.
    live["repair_run"] = 202
    with pytest.raises(ValueError, match="authorization or source changed"):
        repair.current_request(request)
    assert published == [state]


def test_cancelled_repair_run_requests_help_on_the_next_event(
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
            {
                "status": "completed",
                "conclusion": "cancelled",
                "run_started_at": "2030-01-01T00:00:00Z",
            }
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


def test_queued_validation_expires_without_a_completion_event(
    monkeypatch, tmp_path, selected
):
    """An expired budget turns manual on the next event; nothing polls for it."""
    _, state = selected
    state.update(action="fix", phase="validating", repair_run=201)
    pr = {
        "number": state["pr"],
        "head": {"sha": state["head"]},
        "base": {"sha": state["base"]},
    }
    event = tmp_path / "event.json"
    event.write_text("{}")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda n: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: [])
    monkeypatch.setattr(assist, "load_state", lambda *args: state)
    monkeypatch.setattr(assist, "latest_command", lambda *args: None)
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: {
            "status": "completed",
            "run_started_at": "2020-01-01T00:00:00Z",
        },
    )
    messages = []
    monkeypatch.setattr(assist, "publish", lambda *args: messages.append(args[1]))
    monkeypatch.setattr(
        assist, "dispatch", lambda *args: pytest.fail("expired budget dispatched")
    )
    assist.control(state["pr"])
    assert state["phase"] == "manual" and "one-hour" in messages[0]


def test_manual_dispatch_retry_revalidates_candidate_or_repairs_afresh(
    monkeypatch, tmp_path, selected
):
    task, state = selected
    state.update(action="fix", phase="manual", repair_run=201)
    plan = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    plan.update(run=55, tests=[], tasks=state["tasks"])
    comments = [{"user": {"login": BOT, "id": BOT_ID}, "body": marker("plan", plan)}]
    pr = {
        "number": state["pr"],
        "head": {"sha": state["head"]},
        "base": {"sha": state["base"]},
        "mergeable": True,
    }
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_RUN_ID", "202")
    monkeypatch.setattr(assist, "WORK", tmp_path)
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda n: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "load_state", lambda *args: state)
    monkeypatch.setattr(
        assist,
        "latest_command",
        lambda *args: {"id": state["command"], "body": "@lightseek-bot fix"},
    )
    monkeypatch.setattr(assist, "permitted", lambda *args: "fix")
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(assist, "context", lambda *args: {})
    monkeypatch.setattr(assist, "validate_plan", lambda *args: [task])
    monkeypatch.setattr(assist, "runs_for", lambda *args: [])
    monkeypatch.setattr(assist, "main_merge_clean", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: (
            {"status": "completed", "run_started_at": "2030-01-01T00:00:00Z"}
            if path == "actions/runs/201"
            else (
                {
                    "name": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
                    "path": ".github/workflows/pr-ci-plan.yml",
                    "conclusion": "success",
                    "display_title": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
                    "run_started_at": "2030-01-01T00:00:00Z",
                }
                if "actions/runs" in path
                else {"object": {"sha": state["base"]}}
            )
        ),
    )
    published = []
    monkeypatch.setattr(assist, "publish", lambda s, m: published.append(s["phase"]))
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    # Without a candidate the retry starts a fresh repair, even when the
    # original failure evidence no longer fails right now.
    monkeypatch.setattr(assist, "task_status", lambda *args, **kwargs: "passed")
    assist.control(state["pr"])
    assert published == ["repairing"] and emitted == [("repair", "true")]
    request = json.loads(tmp_path.joinpath("request.json").read_text())
    assert request["state"]["repair_run"] == 202
    assert set(request) == {
        "state",
        "plan",
        "data",
        "conflicts",
        "deadline",
        "lint_run",
    }
    # With a retained candidate the retry harvests existing results within a
    # 15-minute window: no new repair, no new dispatch, original repair run.
    state.update(
        phase="manual",
        repair_run=201,
        candidate=dict(
            patch="c" * 40,
            validation="d" * 40,
            tree="e" * 40,
            branch="bot/pr-ci-assist-123-42-201",
        ),
    )
    published.clear()
    emitted.clear()
    monkeypatch.setenv("GITHUB_RUN_ID", "203")
    monkeypatch.setattr(assist, "repair_plan", lambda _: plan)
    monkeypatch.setattr(assist, "task_status", lambda *args, **kwargs: "waiting")
    monkeypatch.setattr(
        assist, "dispatch", lambda *args: pytest.fail("retry duplicated validation")
    )
    assist.control(state["pr"])
    assert state["phase"] == "manual" and state["repair_run"] == 201
    assert published == ["manual"] and not emitted


def test_rejected_patch_receives_feedback_within_the_original_budget(
    monkeypatch, tmp_path, capsys
):
    request = {"deadline": 100, "data": {"paths": ["test/example.py"]}}
    now = [40]
    monkeypatch.setattr(repair.time, "time", lambda: now[0])
    feedback = tmp_path / "feedback.json"
    turns = []

    def run_model(attempt):
        turns.append(repair.remaining_time(request))
        if attempt:
            assert (
                json.loads(feedback.read_text())["reason"]
                == repair.REPAIR_FEEDBACK["test-assertions"]
            )
        now[0] += 10

    def proposal():
        if len(turns) < 3:
            raise repair.RepairRejected(
                "test-assertions",
                path="test/example.py",
                details=["private diagnostic"],
            )
        return "accepted patch"

    assert (
        repair.repair_with_feedback(request, run_model, proposal, feedback)
        == "accepted patch"
    )
    assert turns == [60, 50, 40] and request["deadline"] == 100
    output = capsys.readouterr().out
    assert "test/example.py" in output and "private diagnostic" not in output
    turns.clear()

    def rejected():
        raise repair.RepairRejected("test-syntax")

    with pytest.raises(repair.RepairRejected):
        repair.repair_with_feedback(
            request, lambda attempt: turns.append(attempt), rejected, feedback
        )
    assert turns == list(range(3))
    now[0] = request["deadline"]
    with pytest.raises(ValueError, match="budget expired"):
        repair.repair_with_feedback(
            request,
            lambda attempt: pytest.fail("Started an expired turn"),
            rejected,
            feedback,
        )


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


def test_promotion_accepts_only_owned_validation_branch(
    monkeypatch, tmp_path, selected
):
    _, state = selected
    state.update(action="fix", phase="validating", repair_run=200)
    state["candidate"] = dict(
        patch="c" * 40,
        validation="d" * 40,
        tree="e" * 40,
        branch="bot/pr-ci-assist-123-42-200",
    )
    monkeypatch.setattr(repair, "WORK", tmp_path)
    monkeypatch.setattr(repair, "public_gate", lambda: None)
    monkeypatch.setattr(repair, "main_merge_clean", lambda *args, **kwargs: True)
    monkeypatch.setattr(repair.time, "time", lambda: 1893459599)
    monkeypatch.setattr(
        repair,
        "pull",
        lambda n: dict(head={"sha": state["head"]}, base={"sha": state["base"]}),
    )
    monkeypatch.setattr(repair, "pages", lambda *a: [])
    monkeypatch.setattr(repair, "latest_command", lambda c: {"id": state["command"]})
    monkeypatch.setattr(repair, "api", lambda p: {"object": {"sha": "d" * 40}})

    class FetchReached(Exception):
        pass

    def command(*args, **kwargs):
        assert args == ("git", "fetch", "origin", "d" * 40)
        raise FetchReached

    monkeypatch.setattr(repair, "command", command)
    with pytest.raises(FetchReached):
        repair.promote(state, deadline=1893459600)
    state["candidate"]["branch"] = "bot/pr-ci-assist-123-42-199"
    with pytest.raises(ValueError, match="Invalid candidate record"):
        repair.promote(state, deadline=1893459600)


def test_rerun_command_requires_a_job_url(monkeypatch, selected):
    _, state = selected
    monkeypatch.setattr(assist, "api", lambda path: {"permission": "write"})
    url = f"https://github.com/{REPO}/actions/runs/101/job/201"
    author = {"user": {"login": "someone"}}
    assert (
        assist.permitted({**author, "body": f"@lightseek-bot rerun {url}"}) == "rerun"
    )
    assert assist.permitted({**author, "body": "@lightseek-bot rerun"}) is None
    assert assist.permitted({**author, "body": f"@lightseek-bot watch {url}"}) is None
    # The rerun action and its target round-trip through the state record.
    state.update(action="rerun", target={"run": 101, "job": 201})
    comment = {"user": {"login": BOT, "id": BOT_ID}, "body": marker("assist", state)}
    assert record(comment, "assist") == state


def test_rerun_redispatches_and_failure_escalates_to_repair(
    monkeypatch, tmp_path, selected
):
    task, state = selected
    plan = {k: state[k] for k in ("version", "repository", "pr", "head", "base")}
    plan.update(run=55, tests=[], tasks=state["tasks"])
    comments = [{"user": {"login": BOT, "id": BOT_ID}, "body": marker("plan", plan)}]
    command_comment = {
        "id": 43,
        "body": (
            f"@lightseek-bot rerun "
            f"https://github.com/{REPO}/actions/runs/101/job/201"
        ),
    }
    pr = {
        "number": state["pr"],
        "head": {"sha": state["head"]},
        "base": {"sha": state["base"]},
        "mergeable": True,
    }
    live = [None]
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setattr(assist, "WORK", tmp_path)
    monkeypatch.setattr(assist, "public_gate", lambda: None)
    monkeypatch.setattr(assist, "pull", lambda n: pr)
    monkeypatch.setattr(assist, "pages", lambda *args: comments)
    monkeypatch.setattr(assist, "load_state", lambda *args: copy.deepcopy(live[0]))
    monkeypatch.setattr(assist, "latest_command", lambda *args: command_comment)
    monkeypatch.setattr(assist, "permitted", lambda *args: "rerun")
    monkeypatch.setattr(assist, "checkout", lambda *args: tmp_path)
    monkeypatch.setattr(assist, "context", lambda *args: {})
    monkeypatch.setattr(assist, "targeted_plan", lambda plan, *args: plan)
    monkeypatch.setattr(assist, "validate_plan", lambda *args: [task])
    monkeypatch.setattr(assist, "main_merge_clean", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        assist,
        "api",
        lambda path: (
            {
                "path": ".github/workflows/pr-ci-plan.yml",
                "conclusion": "success",
                "display_title": f"CI plan #{state['pr']} | {state['head']} | {state['base']}",
                "run_started_at": "2030-01-01T00:00:00Z",
            }
            if path == "actions/runs/55"
            else (
                {"run_started_at": "2030-01-01T00:00:00Z"}
                if "actions/runs" in path
                else {"object": {"sha": "c" * 40}}
            )
        ),
    )
    published = []

    def publish(s, message):
        live[0] = copy.deepcopy(s)
        published.append(s["phase"])

    monkeypatch.setattr(assist, "publish", publish)
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    dispatched = []
    monkeypatch.setattr(assist, "dispatch", lambda *args: dispatched.append(args))
    title = assist.run_title(task, state["head"], "gb200")
    bot_run = {
        "id": 999,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "actor": {"login": BOT},
        "display_title": title,
    }
    runs = [[]]
    monkeypatch.setattr(assist, "runs_for", lambda *args: runs[0])
    report = ["waiting"]
    monkeypatch.setattr(assist, "report", lambda *args: report[0])

    # The rerun command re-dispatches the target's validation task; the
    # inherited submission from an earlier watch must not suppress it.
    live[0] = {**state, "submitted": [title]}
    assist.control(state["pr"])
    assert dispatched == [(task, state["head"], "gb200")]
    assert live[0]["phase"] == "watching"
    assert live[0]["action"] == "rerun" and live[0]["since"] == 43
    assert live[0]["target"] == {"run": 101, "job": 201}
    assert published == ["watching", "watching"] and not emitted

    # A passed rerun closes the request without touching the repair flow.
    runs[0] = [bot_run]
    report[0] = "passed"
    published.clear()
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    assist.control(state["pr"])
    assert live[0]["phase"] == "done"
    assert published == ["done"] and not emitted

    # A failed rerun escalates into the repair flow with the target diagnostics.
    report[0] = "failed"
    published.clear()
    live[0].update(phase="watching")
    monkeypatch.setenv("GITHUB_RUN_ID", "202")
    assist.control(state["pr"])
    assert live[0]["phase"] == "repairing" and live[0]["repair_run"] == 202
    assert emitted == [("repair", "true")]
    request = json.loads(tmp_path.joinpath("request.json").read_text())
    assert request["state"]["action"] == "rerun"
    assert request["state"]["target"] == {"run": 101, "job": 201}
    assert set(request) == {
        "state",
        "plan",
        "data",
        "conflicts",
        "deadline",
        "lint_run",
    }


def test_sweep_picks_the_oldest_active_authorized_pr(monkeypatch, tmp_path):
    prs = [
        {
            "number": 1,
            "state": "open",
            "head": {"repo": {"full_name": REPO}, "sha": "a" * 40},
            "base": {"ref": "main"},
        },
        {
            "number": 2,
            "state": "open",
            "head": {"repo": {"full_name": REPO}, "sha": "b" * 40},
            "base": {"ref": "main"},
        },
        {
            "number": 3,
            "state": "open",
            "head": {"repo": None},
            "base": {"ref": "main"},
        },
        {
            "number": 4,
            "state": "open",
            "head": {"repo": {"full_name": REPO}, "sha": "c" * 40},
            "base": {"ref": "dev"},
        },
    ]
    comments = {
        1: [{"id": 11, "updated_at": "2026-10-10T00:00:01Z"}],
        2: [{"id": 22, "updated_at": "2026-10-10T00:00:02Z"}],
        5: [{"id": 55, "updated_at": "2026-10-10T00:00:03Z"}],
    }
    states = {
        1: {"phase": "done"},
        2: {"phase": "watching"},
        5: ValueError("revoked"),
    }

    def pages(path, field):
        if path == "pulls?state=open":
            return prs
        number = int(path.split("/")[1])
        return comments.get(number, [])

    monkeypatch.setattr(assist, "pages", pages)
    monkeypatch.setattr(
        assist,
        "latest_state_comment",
        lambda cs, number: (cs or [None])[-1],
    )

    def load_state(cs, pr):
        state = states[pr["number"]]
        if isinstance(state, Exception):
            raise state
        return state

    monkeypatch.setattr(assist, "load_state", load_state)
    assert assist.sweep() == 2
    states[2] = {"phase": "manual"}
    assert assist.sweep() is None


def test_resolve_schedule_outputs_the_swept_pr(monkeypatch, tmp_path):
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "event.json"))
    tmp_path.joinpath("event.json").write_text("{}")
    pr = {
        "number": 2,
        "state": "open",
        "head": {"repo": {"full_name": REPO}, "sha": "b" * 40},
        "base": {"ref": "main"},
    }
    monkeypatch.setattr(assist, "sweep", lambda: 2)
    monkeypatch.setattr(assist, "api", lambda path: pr)
    assist.resolve()
    assert output.read_text().strip() == "pr=2"
    monkeypatch.setattr(assist, "sweep", lambda: None)
    output.unlink()
    assist.resolve()
    assert not output.exists()


def test_assist_workflow_wakes_only_for_owned_dispatches_and_sweeps():
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/pr-ci-assist.yml"
        ).read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    workflows = set(triggers["workflow_run"]["workflows"])
    assert workflows == {"PR CI Plan", "Slurm Dispatch", "K8s Dispatch"}
    assert triggers["schedule"] == [{"cron": "13,33,53 * * * *"}]
