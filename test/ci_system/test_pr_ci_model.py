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

"""Keep decoded proposal text behind the publication guard."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_escaped_sensitive_summary_cannot_be_published(tmp_path, monkeypatch):
    scripts = REPO_ROOT / ".github/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "pr_ci_model", scripts / "pr-ci-model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    home = tmp_path / "home"
    home.mkdir()
    home.joinpath("config.toml").write_text(
        '[providers.planner]\nbase_url = "https://example.com/v1"\n'
        '[models.planner]\nmodel = "example-model"\n'
    )
    monkeypatch.setenv("KIMI_CODE_HOME", str(home))
    monkeypatch.setenv("KIMI_API_KEY", "token-q7z-key-only")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.chdir(source)
    data = {
        "version": 1,
        "repository": "lightseekorg/tokenspeed",
        "pr": 1,
        "head": "a" * 40,
        "base": "b" * 40,
        "paths": [],
        "catalog": [],
        "test_files": [],
    }
    tmp_path.joinpath("context.json").write_text(json.dumps(data))
    raw = r'{"summary":"h\u0074tps:\/\/ex\u0061mple.com/v1","tests":[],"tasks":[],"conflicts":""}'
    # The wire representation hides the URL; the decoded comment must be screened.
    module._check_public_output(raw, tmp_path)

    def run(*args, stdout, **kwargs):
        assert f"Source root: {source}." in args[0][-1]
        stdout.write(json.dumps({"role": "assistant", "content": raw}) + "\n")
        stdout.flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(SystemExit, match="public-output check"):
        module.plan(tmp_path)
    assert not tmp_path.joinpath("comment.md").exists()
    data["catalog"] = [{"config": "test/ci/example.yaml", "name": "example.com"}]
    tmp_path.joinpath("context.json").write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="public-output check"):
        module._check_public_output("example.com", tmp_path)
    link = module.source_url(data, data["catalog"][0]["config"])
    with pytest.raises(SystemExit, match="public-output check"):
        module._check_public_output(link, tmp_path)
    module._check_public_output(f"[CI]({link})", tmp_path, source_links=True)
    data["native_checks"] = [{"workflow": "scheduler-cpp-test.yml"}]
    tmp_path.joinpath("context.json").write_text(json.dumps(data))
    module._check_public_output(
        module.source_url(data, ".github/workflows/scheduler-cpp-test.yml"),
        tmp_path,
        source_links=True,
    )
    for text in (f"[CI]({link}) example.com", f"[CI]({link}?extra=1)"):
        with pytest.raises(SystemExit, match="public-output check"):
            module._check_public_output(text, tmp_path, source_links=True)
    with pytest.raises(SystemExit, match="comment size limit"):
        module._check_public_output("x" * 60001, tmp_path)
    module._check_public_output("x" * 60001, tmp_path, max_length=200000)
    with pytest.raises(SystemExit, match="public-output check"):
        module._check_public_output("example.com", tmp_path, max_length=200000)


def test_invalid_proposal_gets_one_bounded_correction(tmp_path, monkeypatch):
    scripts = REPO_ROOT / ".github/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "pr_ci_model", scripts / "pr-ci-model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_RUN_ID", "55")
    data = dict(
        version=1,
        repository="lightseekorg/tokenspeed",
        pr=123,
        head="a" * 40,
        base="b" * 40,
        catalog=[],
        test_files=[],
    )
    tmp_path.joinpath("context.json").write_text(json.dumps(data))
    responses = [
        json.dumps(dict(summary="x" * 201, conflicts="", tests=[], tasks=[])),
        json.dumps(
            dict(summary="Prioritized CI checks", conflicts="", tests=[], tasks=[])
        ),
    ]
    corrections = []

    def generate(root, source, correction=""):
        corrections.append(correction)
        return responses.pop(0)

    monkeypatch.setattr(module, "_generate", generate)
    monkeypatch.setattr(module, "_check_public_output", lambda *args, **kwargs: None)
    module.plan(tmp_path)
    assert len(corrections) == 2 and "Invalid proposal summary" in corrections[1]
    assert "Prioritized CI checks" in tmp_path.joinpath("comment.md").read_text()
    # A second invalid response stops rather than publishing or looping.
    tmp_path.joinpath("comment.md").unlink()
    responses.extend(
        [json.dumps(dict(summary="x" * 201, conflicts="", tests=[], tasks=[]))] * 2
    )
    corrections.clear()
    with pytest.raises(SystemExit, match="Invalid CI proposal"):
        module.plan(tmp_path)
    assert len(corrections) == 2 and not tmp_path.joinpath("comment.md").exists()


def test_plan_publish_updates_the_existing_plan_comment(tmp_path, monkeypatch):
    scripts = REPO_ROOT / ".github/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "pr_ci_model", scripts / "pr-ci-model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import io

    import pr_ci_common
    from pr_ci_state import BOT, BOT_ID, marker

    repo = "lightseekorg/tokenspeed"
    head = "a" * 40
    base = "b" * 40
    metadata = dict(
        version=1,
        repository=repo,
        pr=123,
        head=head,
        base=base,
        run=55,
        tests=[],
        tasks=[],
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", repo)
    monkeypatch.setenv("PR_NUMBER", "123")
    monkeypatch.setenv("PR_HEAD_SHA", head)
    tmp_path.joinpath("context.json").write_text(json.dumps({"base": base}))
    body = "CI plan body\n" + marker("plan", metadata)
    tmp_path.joinpath("comment.md").write_text(body)
    comments = []
    created = []
    patched = []

    def command(*args):
        if args == ("gh", "auth", "status"):
            return ""
        if args == ("gh", "api", "user"):
            return json.dumps({"login": BOT})
        if args[:3] == ("gh", "repo", "view"):
            return "PUBLIC"
        if args[:3] == ("gh", "pr", "view"):
            return head
        if args == ("gh", "api", f"repos/{repo}/pulls/123"):
            return json.dumps({"base": {"sha": base}})
        if args == (
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/issues/123/comments?per_page=100",
        ):
            return json.dumps([comments])
        if args[:3] == ("gh", "pr", "comment"):
            created.append(args)
            comment = dict(
                id=len(comments) + 1,
                user={"login": BOT, "id": BOT_ID},
                body=args[-1],
            )
            comments.append(comment)
            return f"https://github.com/{repo}/pull/123#issuecomment-{comment['id']}"
        if args == ("gh", "auth", "token", "--hostname", "github.com"):
            return "test-token\n"
        if args[:2] == ("gh", "api") and "/issues/comments/" in args[-1]:
            comment_id = int(args[-1].rsplit("/", 1)[-1])
            return json.dumps(next(c for c in comments if c["id"] == comment_id))
        raise AssertionError(f"unexpected command: {args}")

    def patch(request, *, timeout):
        assert request.method == "PATCH" and timeout == 30
        assert request.get_header("Authorization") == "Bearer test-token"
        patched.append(request.full_url)
        comment_id = int(request.full_url.rsplit("/", 1)[-1])
        next(c for c in comments if c["id"] == comment_id)["body"] = json.loads(
            request.data
        )["body"]
        return io.BytesIO()

    monkeypatch.setattr(module, "_command", command)
    monkeypatch.setattr(pr_ci_common, "urlopen", patch)
    module.publish(tmp_path)
    assert len(created) == 1 and not patched
    assert tmp_path.joinpath("published.md").read_text() == body
    # A later plan for the same PR updates the same comment without notifying.
    body_v2 = "CI plan body v2\n" + marker("plan", {**metadata, "run": 56})
    tmp_path.joinpath("comment.md").write_text(body_v2)
    module.publish(tmp_path)
    assert len(created) == 1 and len(patched) == 1
    assert patched[0].endswith("/issues/comments/1")
    assert tmp_path.joinpath("published.md").read_text() == body_v2
