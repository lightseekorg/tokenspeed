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

import base64
import importlib.util
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

ROOT = Path(__file__).parents[2]


@pytest.fixture
def release_module():
    spec = importlib.util.spec_from_file_location(
        "weekly_release", ROOT / ".github/scripts/weekly-release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controller(release_module, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    return release_module.Release(tmp_path / "state.json", "amd")


def test_dispatch_failure_resumes_exact_child_without_dispatching_again(
    controller, release_module, monkeypatch
):
    calls = []
    run = {
        "head_sha": "a" * 40,
        "head_branch": "release/0.1.4",
        "event": "workflow_dispatch",
        "path": ".github/workflows/release-tokenspeed-kernel-amd.yml",
        "status": "completed",
        "conclusion": "failure",
        "html_url": "https://github.com/lightseekorg/tokenspeed/actions/runs/456",
    }

    def api(path, *, data=None):
        calls.append((path, data))
        return {"workflow_run_id": 456} if data is not None else run

    monkeypatch.setattr(release_module, "api", api)
    workflow = "release-tokenspeed-kernel-amd.yml"
    with pytest.raises(RuntimeError, match="Child workflow failed"):
        controller.child(
            workflow, "a" * 40, "release/0.1.4", {}, event="workflow_dispatch"
        )
    assert json.loads(controller.path.read_text())["runs"][workflow]["id"] == 456
    resumed = release_module.Release(controller.path, "amd")
    run["conclusion"] = "success"
    assert (
        resumed.child(
            workflow, "a" * 40, "release/0.1.4", {}, event="workflow_dispatch"
        )
        == 456
    )
    assert sum(data is not None for _, data in calls) == 1


def test_child_success_for_wrong_commit_cannot_advance(
    controller, release_module, monkeypatch
):
    workflow = "release-tokenspeed-kernel-amd.yml"
    controller.state["runs"][workflow] = {
        "id": 456,
        "sha": "a" * 40,
        "ref": "release/0.1.4",
        "event": "workflow_dispatch",
    }
    monkeypatch.setattr(release_module, "api", lambda *a, **kw: {"head_sha": "b" * 40})
    with pytest.raises(RuntimeError, match="source mismatch"):
        controller.child(
            workflow, "a" * 40, "release/0.1.4", {}, event="workflow_dispatch"
        )
    assert not controller.phase.get("complete")


def test_ambiguous_dispatch_response_recovers_by_source_without_reposting(
    controller, release_module, monkeypatch
):
    workflow = "release-tokenspeed-kernel-amd.yml"
    controller.state["runs"][workflow] = {
        "sha": "a" * 40,
        "ref": "release/0.1.4",
        "event": "workflow_dispatch",
        "dispatch_started": True,
    }
    monkeypatch.setattr(controller, "find_run", lambda *a: 456)

    def api(path, *, data=None):
        assert data is None
        return {
            "head_sha": "a" * 40,
            "head_branch": "release/0.1.4",
            "event": "workflow_dispatch",
            "path": f".github/workflows/{workflow}",
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://github.com/lightseekorg/tokenspeed/actions/runs/456",
        }

    monkeypatch.setattr(release_module, "api", api)
    assert (
        controller.child(
            workflow, "a" * 40, "release/0.1.4", {}, event="workflow_dispatch"
        )
        == 456
    )


def test_push_run_lookup_requires_exact_source_and_actor(
    controller, release_module, monkeypatch
):
    sha = "a" * 40

    def api(path, *, data=None):
        assert "head_sha=" + sha in path and "event=push" in path
        common = {
            "head_branch": "main",
            "event": "push",
            "actor": {"login": "lightseek-bot"},
        }
        return {
            "workflow_runs": [
                dict(common, id=1, head_sha="b" * 40),
                dict(common, id=2, head_sha=sha),
            ]
        }

    monkeypatch.setattr(release_module, "api", api)
    assert controller.find_run("release-pypi.yml", sha, "main", "push") == 2


def test_version_pr_requires_registered_lint_and_pending_checks(release_module):
    pr = {"statusCheckRollup": [], "reviewDecision": "", "mergeStateStatus": "CLEAN"}
    assert not release_module.checks_ready(pr)
    pr["statusCheckRollup"] = [
        {"name": "lint", "conclusion": "SUCCESS"},
        {"name": "build", "status": "IN_PROGRESS"},
    ]
    assert not release_module.checks_ready(pr)
    pr["statusCheckRollup"][1]["conclusion"] = "FAILURE"
    with pytest.raises(RuntimeError, match="PR check failed"):
        release_module.checks_ready(pr)
    pr["statusCheckRollup"][1]["conclusion"] = "SKIPPED"
    assert release_module.checks_ready(pr)


def test_metadata_updates_both_versions_and_keeps_kernel_boundary(
    release_module, tmp_path, monkeypatch
):
    files = set(sum(release_module.PR_FILES.values(), []))
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    monkeypatch.chdir(tmp_path)
    versions = {
        package: release_module.next_version(
            release_module.read_version(package),
            release_module.read_version(package),
            "",
        )
        for package in release_module.PACKAGES.values()
    }
    release_module.update_metadata("amd", versions)
    release_module.update_metadata("kernel", versions)
    release_module.update_metadata("tokenspeed", versions)
    assert (
        release_module.read_version("tokenspeed-kernel")
        == versions["tokenspeed-kernel"]
    )
    assert release_module.read_version("tokenspeed") == versions["tokenspeed"]
    assert (
        f'__version__ = "{versions["tokenspeed"]}"'
        in Path("python/tokenspeed/version.py").read_text()
    )
    deps = release_module.requirements("python/pyproject.toml")
    assert (
        str(deps["tokenspeed-kernel"].specifier) == f'>={versions["tokenspeed-kernel"]}'
    )
    assert "tokenspeed-kernel-amd" not in deps and "tokenspeed-mla" not in deps
    assert (
        f'tokenspeed-kernel-amd>={versions["tokenspeed-kernel-amd"]}'
        in Path(release_module.PR_FILES["kernel"][1]).read_text()
    )
    assert release_module.next_version("0.1.3", "0.1.3", "") == "0.1.4"
    with pytest.raises(ValueError, match="reuse or downgrade"):
        release_module.next_version("0.1.3", "0.1.3", "0.1.3")


def test_pypi_source_rejects_mixed_artifact_commits(release_module, monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "fixture")])
    files = [
        {
            "filename": f"scheduler-{i}.whl",
            "digests": {"sha256": "c" * 64},
            "yanked": False,
        }
        for i in range(2)
    ]
    monkeypatch.setattr(release_module, "pypi", lambda *a: {"urls": files})

    def request(url, *, github, data=None):
        i = int(url.split("scheduler-")[1][0])
        builder = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(1)
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=1))
        )
        for oid, value in (
            (1, "https://token.actions.githubusercontent.com"),
            (3, "ab"[i] * 40),
            (5, release_module.REPO),
        ):
            builder = builder.add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(f"1.3.6.1.4.1.57264.1.{oid}"), value.encode()
                ),
                critical=False,
            )
        certificate = builder.sign(key, hashes.SHA256()).public_bytes(
            serialization.Encoding.DER
        )
        statement = {
            "subject": [{"name": files[i]["filename"], "digest": files[i]["digests"]}]
        }
        return {
            "attestation_bundles": [
                {
                    "publisher": {
                        "repository": release_module.REPO,
                        "workflow": "release-tokenspeed-scheduler.yml",
                    },
                    "attestations": [
                        {
                            "envelope": {
                                "statement": base64.b64encode(
                                    json.dumps(statement).encode()
                                ).decode()
                            },
                            "verification_material": {
                                "certificate": base64.b64encode(certificate).decode()
                            },
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(release_module, "request", request)
    with pytest.raises(RuntimeError, match="different commits"):
        release_module.source_sha(
            "tokenspeed-scheduler", "0.1.25", "release-tokenspeed-scheduler.yml"
        )


def test_stable_index_preserves_history_and_rejects_replaced_wheels(
    release_module, tmp_path
):
    name = "tokenspeed-0.1.1-py3-none-any.whl"
    release = {
        "assets": [
            {
                "name": name,
                "browser_download_url": f"https://github.com/lightseekorg/whl/releases/download/tokenspeed-v0.1.1/{name}",
                "digest": "sha256:" + "a" * 64,
            }
        ]
    }
    index = tmp_path / "cu130/tokenspeed/index.html"
    index.parent.mkdir(parents=True)
    index.write_text("previous releases\n")
    release_module.index_release(tmp_path, "cu130", "tokenspeed", release)
    contents = index.read_text()
    release_module.index_release(tmp_path, "cu130", "tokenspeed", release)
    assert index.read_text() == contents and contents.startswith("previous releases\n")
    assert not (tmp_path / "nightly").exists()
    release["assets"][0]["digest"] = "sha256:" + "b" * 64
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        release_module.index_release(tmp_path, "cu130", "tokenspeed", release)


def test_weekly_schedule_and_failure_resume_contract():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/weekly-release.yml").read_text()
    )
    events = workflow.get("on", workflow.get(True))
    assert events["schedule"] == [
        {"cron": "0 20 * * 0", "timezone": "America/Los_Angeles"}
    ]
    assert workflow["concurrency"]["cancel-in-progress"] is False
    for previous, stage in zip(
        ("plan", "amd", "kernel", "tokenspeed", "index", "docker"),
        ("amd", "kernel", "tokenspeed", "index", "docker", "release"),
    ):
        assert workflow["jobs"][stage]["needs"] == previous
    reusable = yaml.safe_load(
        (ROOT / ".github/workflows/weekly-release-stage.yml").read_text()
    )
    upload = reusable["jobs"]["stage"]["steps"][-1]
    assert upload["if"] == "always()" and upload["with"]["overwrite"] is True
    assert reusable["jobs"]["stage"]["steps"][0]["with"]["persist-credentials"] is False


def test_source_tree_check_catches_unreleased_changes(
    release_module, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    git = lambda *args: release_module.command("git", *args)
    git("init", "-q", "-b", "main")
    git("config", "user.name", "lightseek-bot")
    git("config", "user.email", release_module.IDENTITY)
    (tmp_path / "tokenspeed-mla").mkdir()
    source = tmp_path / "tokenspeed-mla/source.py"
    source.write_text("released = True\n")
    git("add", ".")
    git("commit", "-q", "-s", "-m", "fixture release")
    sha = git("rev-parse", "HEAD")
    git("remote", "add", "origin", str(tmp_path))
    (tmp_path / "unrelated.txt").write_text("unrelated change\n")
    git("add", ".")
    git("commit", "-q", "-s", "-m", "fixture unrelated change")
    release_module.check_tree(sha, "tokenspeed-mla")
    source.write_text("released = False\n")
    git("add", ".")
    git("commit", "-q", "-s", "-m", "fixture unreleased change")
    with pytest.raises(RuntimeError, match="Unreleased changes"):
        release_module.check_tree(sha, "tokenspeed-mla")


def test_rocm_release_checks_updated_amd_requirement(tmp_path, monkeypatch):
    import zipfile

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-tokenspeed-kernel-rocm.yml").read_text()
    )
    build = workflow["jobs"]["build-wheel"]["steps"]
    script = next(
        step["run"]
        for step in build
        if "AMD_NIGHTLY_VERSION" in step.get("run", "") and "BytesParser" in step["run"]
    )
    code = script.split("import os\nimport sys\nimport zipfile", 1)[1].split("\nPY", 1)[
        0
    ]
    code = "import os\nimport sys\nimport zipfile" + code
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/rocm-thirdparty.txt").write_text(
        "tokenspeed-kernel-amd>=0.1.4\n"
    )
    wheel = tmp_path / "kernel.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "kernel.dist-info/METADATA",
            "Name: tokenspeed-kernel\nVersion: 0.1.4\nRequires-Dist: tokenspeed-kernel-amd>=0.1.4\n\n",
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AMD_NIGHTLY_VERSION", "")
    monkeypatch.setattr("sys.argv", ["check", str(wheel)])
    exec(compile(code, "rocm-release-metadata", "exec"), {})


def test_stable_index_retries_after_concurrent_nightly_push(
    controller, release_module, tmp_path, monkeypatch
):
    import subprocess

    real_command = release_module.command
    bare = tmp_path / "remote.git"
    real_command("git", "init", "--bare", "-q", str(bare))
    writer = tmp_path / "writer"
    real_command("git", "clone", str(bare), str(writer))
    real_command("git", "switch", "-c", "gh-pages", cwd=writer)
    real_command("git", "config", "user.name", "lightseek-bot", cwd=writer)
    real_command("git", "config", "user.email", release_module.IDENTITY, cwd=writer)
    (writer / "index.html").write_text("<!DOCTYPE html>\n")
    real_command("git", "add", ".", cwd=writer)
    real_command("git", "commit", "-q", "-s", "-m", "fixture index", cwd=writer)
    real_command("git", "push", "origin", "HEAD:gh-pages", cwd=writer)
    controller.state["versions"] = {
        p: "0.1.4" for p in release_module.PACKAGES.values()
    }
    controller.state["stages"].update(
        {s: {"sha": "a" * 40} for s in release_module.PACKAGES}
    )

    def wheelhouse(tag, sha, count):
        name = f"{tag}-py3-none-any.whl"
        return {
            "assets": [
                {
                    "name": name,
                    "browser_download_url": f"https://github.com/lightseekorg/whl/releases/download/{tag}/{name}",
                    "digest": "sha256:" + "a" * 64,
                }
            ]
        }

    monkeypatch.setattr(controller, "wheelhouse", wheelhouse)

    def command(*args, cwd=None, env=None):
        if args[:2] == ("git", "clone"):
            args = (*args[:5], str(bare), args[-1])
        if args[:3] == ("git", "remote", "get-url"):
            return "https://github.com/lightseekorg/whl.git"
        if args[:2] == ("gh", "api"):
            path = args[2].split("/contents/", 1)[1].split("?", 1)[0]
            return base64.b64encode(
                real_command("git", "show", f"gh-pages:{path}", cwd=bare).encode()
                + b"\n"
            ).decode()
        return real_command(*args, cwd=cwd, env=env)

    real_run = subprocess.run
    pushed = []

    def run(args, **kwargs):
        if args[:2] == ["git", "push"]:
            if not pushed:
                (writer / "nightly.txt").write_text("concurrent nightly\n")
                real_command("git", "add", ".", cwd=writer)
                real_command(
                    "git", "commit", "-q", "-s", "-m", "fixture nightly", cwd=writer
                )
                real_command("git", "push", "origin", "HEAD:gh-pages", cwd=writer)
            result = real_run(args, **kwargs)
            pushed.append(result.returncode)
            return result
        return real_run(args, **kwargs)

    monkeypatch.setattr(release_module, "command", command)
    monkeypatch.setattr(release_module.subprocess, "run", run)
    controller.index()
    assert pushed == [1, 0]
    assert (
        real_command("git", "show", "gh-pages:nightly.txt", cwd=bare)
        == "concurrent nightly"
    )
    assert "tokenspeed-v0.1.4" in real_command(
        "git", "show", "gh-pages:cu130/tokenspeed/index.html", cwd=bare
    )
