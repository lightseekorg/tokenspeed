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
