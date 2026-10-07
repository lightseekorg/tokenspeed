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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts"))
import pr_ci_assist as assist
import pr_ci_repair as repair
from ci_result_source import write_source
from pr_ci_state import BOT, BOT_ID, REPO, marker, record


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
                "name": "PR CI Plan",
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
                "name": "PR CI Plan",
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


def test_dispatch_event_uses_tested_source_not_main_controller(monkeypatch, tmp_path):
    event = tmp_path / "event.json"
    tested, controller = "c" * 40, "b" * 40
    event.write_text(
        json.dumps(
            {
                "action": "completed",
                "workflow_run": {
                    "pull_requests": [],
                    "head_sha": controller,
                    "display_title": f"Slurm {tested} | test/ci/ut/example.yaml | b200-1gpu | gb200",
                },
            }
        )
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")

    def pages(path, field):
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
    monkeypatch.setattr(assist, "pull", lambda number: resolved.append(number))
    emitted = []
    monkeypatch.setattr(assist, "output", lambda *args: emitted.append(args))
    assist.resolve()
    assert resolved == [123] and emitted == [("pr", "123")]
