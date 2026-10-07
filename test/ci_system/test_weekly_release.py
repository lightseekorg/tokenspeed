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
import hashlib
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


def test_version_pr_does_not_wait_for_repeated_ci(release_module):
    pr = {
        "isDraft": False,
        "reviewDecision": "REVIEW_REQUIRED",
        "mergeable": "MERGEABLE",
    }
    assert release_module.version_pr_ready(pr)
    pr["statusCheckRollup"] = [
        {"name": "lint", "conclusion": "SUCCESS"},
        {"name": "build", "status": "IN_PROGRESS"},
    ]
    assert release_module.version_pr_ready(pr)
    pr["statusCheckRollup"][1]["conclusion"] = "FAILURE"
    assert release_module.version_pr_ready(pr)
    pr["mergeable"] = "UNKNOWN"
    assert not release_module.version_pr_ready(pr)


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
        str(deps["tokenspeed-kernel"].specifier)
        == f'>={versions["tokenspeed-kernel"]}.dev0'
    )
    assert (
        f'{versions["tokenspeed-kernel"]}.dev20260101+gitabcdef12'
        in deps["tokenspeed-kernel"].specifier
    )
    assert "tokenspeed-kernel-amd" not in deps and "tokenspeed-mla" not in deps
    assert (
        f'tokenspeed-kernel-amd>={versions["tokenspeed-kernel-amd"]}'
        in Path(release_module.PR_FILES["kernel"][1]).read_text()
    )
    assert release_module.next_version("0.1.3", "0.1.3", "") == "0.1.4"
    assert release_module.next_version("0.1.4", "0.1.3", "") == "0.1.5"
    with pytest.raises(ValueError, match="reuse or downgrade"):
        release_module.next_version("0.1.4", "0.1.3", "0.1.4")
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
    assert workflow["jobs"]["plan"]["if"] == "needs.prepare.outputs.mode == 'normal'"
    assert (
        workflow["jobs"]["resume"]["if"] == "needs.prepare.outputs.mode == 'recovery'"
    )
    assert workflow["jobs"]["resume"]["with"]["stage"] == "release"
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
    for component in ("scheduler", "mla"):
        pipeline = yaml.safe_load(
            (ROOT / f".github/workflows/{component}-release.yml").read_text()
        )
        assert set(pipeline.get("on") or pipeline.get(True)) == {"workflow_dispatch"}
        assert pipeline["concurrency"] == workflow["concurrency"]
        for job, previous in (
            ("version", ""),
            ("publish", f"{component}-version"),
            ("dependency", f"{component}-publish"),
        ):
            stage = pipeline["jobs"][job]
            assert stage["uses"] == "./.github/workflows/weekly-release-stage.yml"
            assert stage["with"]["stage"] == f"{component}-{job}"
            assert stage["with"]["previous"] == previous
        assert pipeline["jobs"]["publish"]["needs"] == "version"
        assert pipeline["jobs"]["dependency"]["needs"] == "publish"


@pytest.fixture
def publication_state(release_module):
    versions = {
        package: "0.1.4" for package in (*release_module.PROJECTS, "tokenspeed-kernel")
    }
    stages = {stage: {"complete": True} for stage in release_module.STAGES[:-1]}
    for stage, letter in zip(release_module.PACKAGES, "abc"):
        stages[stage]["sha"] = letter * 40
    stages["docker"]["image"] = "lightseekorg/tokenspeed:0.1.4"
    runs = {}
    for i, (workflow, stage, event) in enumerate(
        (
            ("release-tokenspeed-kernel-amd.yml", "amd", "workflow_dispatch"),
            ("release-tokenspeed-kernel.yml", "kernel", "workflow_dispatch"),
            ("release-tokenspeed-kernel-rocm.yml", "kernel", "workflow_dispatch"),
            ("release-pypi.yml", "tokenspeed", "push"),
            ("publish-release-docker.yml", "tokenspeed", "workflow_dispatch"),
        ),
        1,
    ):
        runs[workflow] = {
            "id": i,
            "sha": stages[stage]["sha"],
            "ref": (
                "main"
                if event == "push"
                else release_module.Release.branch(stage, "0.1.4")
            ),
            "event": event,
            "complete": True,
        }
    return {"run_id": "123", "versions": versions, "stages": stages, "runs": runs}


def test_oversized_generated_notes_publish_bounded_body_with_all_links(
    release_module, publication_state, tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(publication_state))
    release = release_module.Release(path, "release")
    generated = (
        "- Change 中文: https://github.com/lightseekorg/tokenspeed/pull/1\n" * 3000
    )
    body = []
    monkeypatch.setattr(release, "release_exists", lambda *args: False)

    def api(path, *, data=None):
        if data is None:
            return [{"tag_name": "v0.1.3", "draft": False, "prerelease": False}]
        assert (
            path == "releases/generate-notes" and data["previous_tag_name"] == "v0.1.3"
        )
        assert data["target_commitish"] == "c" * 40
        return {"body": generated}

    def command(*args, **kwargs):
        if args[:2] == ("git", "ls-remote"):
            return "c" * 40 + "\t" + args[-1]
        assert args[:3] == ("gh", "release", "create")
        text = Path(args[args.index("--notes-file") + 1]).read_text()
        # Model the actual API limit, including gh's server-generated notes path.
        submitted = text + (generated if "--generate-notes" in args else "")
        assert len(submitted) <= 125000
        body.append(submitted)
        return ""

    monkeypatch.setattr(release_module, "api", api)
    monkeypatch.setattr(release_module, "command", command)
    monkeypatch.setattr(
        release_module,
        "request",
        lambda *a, **kw: {"body": body[-1], "draft": False, "prerelease": False},
    )
    release.release()
    assert len(body[0].encode()) <= 100000
    assert "Biweekly component versions" in body[0]
    for package, version in publication_state["versions"].items():
        assert f"https://pypi.org/project/{package}/{version}/" in body[0]
    assert "tokenspeed-kernel-v0.1.4-rocm72" in body[0]
    assert "https://lightseek.org/whl/cu130/" in body[0]
    assert "actions/runs/5" in body[0]
    assert body[0].endswith(
        "**Full Changelog**: https://github.com/lightseekorg/tokenspeed/compare/v0.1.3...v0.1.4\n"
    )


def test_release_only_recovery_validates_origin_and_never_republishes(
    release_module, publication_state, tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(publication_state))
    with pytest.raises(RuntimeError, match="different weekly run"):
        release_module.Release(path, "release")
    with pytest.raises(RuntimeError, match="release stage"):
        release_module.Release(path, "amd", resume_run_id="123")
    with pytest.raises(RuntimeError, match="different weekly run"):
        release_module.Release(path, "release", resume_run_id="456")
    release = release_module.Release(path, "release", resume_run_id="123")
    source = {
        "status": "completed",
        "path": ".github/workflows/weekly-release.yml",
        "head_branch": "main",
        "head_repository": {"full_name": release_module.REPO},
        "event": "workflow_dispatch",
    }
    jobs = [
        {"name": f"{stage} / stage", "conclusion": "success"}
        for stage in release_module.STAGES[:-1]
    ]

    def api(path, *, data=None):
        assert data is None
        if path == "actions/runs/123":
            return source
        record = next(
            r
            for r in release.state["runs"].values()
            if path == f"actions/runs/{r['id']}"
        )
        workflow = next(w for w, r in release.state["runs"].items() if r is record)
        return {
            "status": "completed",
            "conclusion": "success",
            "head_sha": record["sha"],
            "head_branch": record["ref"],
            "event": record["event"],
            "actor": {"login": "lightseek-bot"},
            "path": f".github/workflows/{workflow}",
            "head_repository": {"full_name": release_module.REPO},
            "html_url": f"https://github.com/{release_module.REPO}/actions/runs/{record['id']}",
        }

    manifest = {
        "manifests": [
            {"platform": {"os": "linux", "architecture": a}} for a in ("amd64", "arm64")
        ]
    }

    def command(*args, **kwargs):
        if args[:2] == ("docker", "buildx"):
            return json.dumps(manifest)
        assert args[:2] == ("gh", "api") and "--paginate" in args
        return json.dumps([{"jobs": jobs}])

    monkeypatch.setattr(release_module, "api", api)
    monkeypatch.setattr(release_module, "command", command)
    monkeypatch.setattr(
        release_module, "source_sha", lambda p, v, w: release.state["runs"][w]["sha"]
    )
    monkeypatch.setattr(release, "wheelhouse", lambda *a: {})
    monkeypatch.setattr(release, "guard", lambda: None)
    monkeypatch.setattr(
        release, "release", lambda: setattr(release, "page_verified", True)
    )
    monkeypatch.setattr(release, "cleanup", lambda: None)
    release.run("")
    assert release.publications_verified and release.phase["complete"]
    assert release.state["versions"] == publication_state["versions"]
    assert release.state["resumed_by_run_id"] == "999"
    source["status"] = "in_progress"
    with pytest.raises(RuntimeError, match="completed release run"):
        release.validate_recovery()
    source["status"] = "completed"
    jobs[-1]["conclusion"] = "failure"
    with pytest.raises(RuntimeError, match="all publication stages"):
        release.validate_recovery()
    jobs[-1]["conclusion"] = "success"
    release.state["runs"]["release-pypi.yml"]["sha"] = "d" * 40
    with pytest.raises(RuntimeError, match="source mismatch"):
        release.verify_publications()


def test_cleanup_deletes_only_expected_refs_and_refuses_moved_branches(
    release_module, publication_state, tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(publication_state))
    release = release_module.Release(path, "release")
    refs = {
        f"refs/heads/{release.branch(s, '0.1.4')}": publication_state["stages"][s][
            "sha"
        ]
        for s in release_module.PACKAGES
    }
    deleted = []

    def command(*args, **kwargs):
        if args[:2] == ("git", "ls-remote"):
            return f"{refs[args[-1]]}\t{args[-1]}" if args[-1] in refs else ""
        ref = args[-1][1:]
        assert (
            args[:2] == ("git", "push")
            and args[2] == f"--force-with-lease={ref}:{refs[ref]}"
        )
        deleted.append(ref)
        del refs[ref]
        return ""

    monkeypatch.setattr(release_module, "command", command)
    with pytest.raises(RuntimeError, match="verified publications"):
        release.cleanup()
    release.publications_verified = release.page_verified = True
    moved = "refs/heads/release/kernel-0.1.4"
    refs[moved] = "d" * 40
    with pytest.raises(RuntimeError, match="branch that moved"):
        release.cleanup()
    assert not deleted
    refs[moved] = "b" * 40
    release.cleanup()
    release.cleanup()
    assert len(deleted) == 3 and not refs and release.phase["branches_cleaned"]


def workflow_python(job):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/weekly-release.yml").read_text()
    )
    script = workflow["jobs"][job]["steps"][0]["run"]
    return script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def test_biweekly_calendar_and_off_week_result(tmp_path, monkeypatch):
    import datetime as dt

    class Clock(dt.datetime):
        current = dt.datetime(2026, 10, 18, 20)

        @classmethod
        def now(cls, tz):
            return cls.current.replace(tzinfo=tz)

    monkeypatch.setattr(dt, "datetime", Clock)
    monkeypatch.setenv("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("RESUME_RUN_ID", "")
    monkeypatch.setenv("VERSION", "")
    monkeypatch.setenv("EVENT", "schedule")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    script = workflow_python("prepare")
    exec(compile(script, "release-mode", "exec"), {})
    assert (tmp_path / "output").read_text().endswith("mode=normal\n")
    Clock.current = dt.datetime(2026, 10, 25, 20)
    exec(compile(script, "release-mode", "exec"), {})
    assert (tmp_path / "output").read_text().endswith("mode=skip\n")
    # The next release falls after the fall-back transition, still at local 20:00.
    Clock.current = dt.datetime(2026, 11, 1, 20)
    exec(compile(script, "release-mode", "exec"), {})
    assert (tmp_path / "output").read_text().endswith("mode=normal\n")
    monkeypatch.setenv("EVENT", "workflow_dispatch")
    monkeypatch.setenv("RESUME_RUN_ID", "123")
    exec(compile(script, "release-mode", "exec"), {})
    assert (tmp_path / "output").read_text().endswith("mode=recovery\n")
    results = {
        stage: {"result": "skipped"}
        for stage in (
            "plan",
            "amd",
            "kernel",
            "tokenspeed",
            "index",
            "docker",
            "release",
            "resume",
        )
    }
    results["prepare"] = {"result": "success"}
    monkeypatch.setenv("MODE", "skip")
    monkeypatch.setenv("RESULTS", json.dumps(results))
    exec(compile(workflow_python("result"), "release-result", "exec"), {})
    monkeypatch.setenv("MODE", "recovery")
    results["resume"]["result"] = "success"
    monkeypatch.setenv("RESULTS", json.dumps(results))
    exec(compile(workflow_python("result"), "release-result", "exec"), {})


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


def test_version_pr_refuses_requested_changes_and_conflicts(release_module):
    pr = {
        "isDraft": False,
        "reviewDecision": "CHANGES_REQUESTED",
        "mergeable": "MERGEABLE",
    }
    with pytest.raises(RuntimeError, match="manual review"):
        release_module.version_pr_ready(pr)
    pr["reviewDecision"] = ""
    pr["mergeable"] = "CONFLICTING"
    with pytest.raises(RuntimeError, match="merge conflicts"):
        release_module.version_pr_ready(pr)


def test_merge_policy_requires_explicit_existing_bot_exemption(
    release_module, monkeypatch
):
    monkeypatch.setattr(release_module, "command", lambda *a: "123")
    actors = [{"actor_type": "User", "actor_id": 123, "bypass_mode": "always"}]

    def api(path, *, data=None):
        assert data is None
        if path == "rules/branches/main":
            return [
                {
                    "type": "required_status_checks",
                    "parameters": {"required_status_checks": [{"context": "finish"}]},
                    "ruleset_source": release_module.REPO,
                    "ruleset_source_type": "Repository",
                    "ruleset_id": 2,
                },
                {
                    "type": "pull_request",
                    "ruleset_source": release_module.REPO,
                    "ruleset_source_type": "Repository",
                    "ruleset_id": 1,
                },
            ]
        return {"bypass_actors": actors if path == "rulesets/1" else ci_actors}

    ci_actors = list(actors)
    monkeypatch.setattr(release_module, "api", api)
    assert release_module.merge_policy()
    ci_actors = []
    assert not release_module.merge_policy()
    ci_actors = list(actors)
    actors[0]["bypass_mode"] = "pull_request"
    assert not release_module.merge_policy()


@pytest.fixture
def version_repository(release_module, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in {
        *sum(release_module.PR_FILES.values(), []),
        release_module.PROJECTS["tokenspeed-scheduler"],
        release_module.PROJECTS["tokenspeed-mla"],
        "tokenspeed-kernel/python/requirements/cuda-thirdparty.txt",
    }:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    monkeypatch.chdir(source)
    command = release_module.command
    command("git", "init", "-b", "main")
    command("git", "config", "user.name", "lightseek-bot")
    command("git", "config", "user.email", release_module.IDENTITY)
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "initial metadata")
    return source


def test_main_ci_inherits_only_unchanged_inputs_and_retries_once(
    controller, release_module, version_repository, monkeypatch, tmp_path
):
    command = release_module.command
    workflows = version_repository / ".github/workflows"
    workflows.mkdir(parents=True)
    names = ("lint.yml", "kernel.yml", "old.yml", "release-pypi.yml")
    for name in names:
        push = {"branches": ["main"]}
        if name == "kernel.yml":
            push["paths"] = [
                "tokenspeed-kernel/test/**",
                "!tokenspeed-kernel/test/amd/**",
            ]
        (workflows / name).write_text(yaml.safe_dump({"on": {"push": push}}))
    kernel = version_repository / "tokenspeed-kernel/test/test_kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("# kernel test input\n")
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "CI inputs")
    tested = command("git", "rev-parse", "HEAD")
    ignored = kernel.parent / "amd/test_amd.py"
    ignored.parent.mkdir()
    ignored.write_text("# excluded input\n")
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "AMD input")
    (version_repository / "README.md").write_text("Documentation\n")
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "docs")
    head = command("git", "rev-parse", "HEAD")
    bare = tmp_path / "origin.git"
    command("git", "clone", "--bare", str(version_repository), str(bare))
    command("git", "remote", "add", "origin", str(bare))
    catalog = [
        {
            "id": i + 1,
            "path": f".github/workflows/{name}",
            "state": "disabled_manually" if name == "old.yml" else "active",
        }
        for i, name in enumerate(names)
    ]
    runs = {}
    reruns = []

    def pages(path, key):
        if key == "workflows":
            return catalog
        workflow_id = int(path.split("/")[2])
        sha = head if workflow_id == 1 else tested
        run = {
            "id": workflow_id,
            "head_sha": sha,
            "head_branch": "main",
            "event": "push",
            "path": catalog[workflow_id - 1]["path"]
            + ("@refs/heads/main" if workflow_id == 1 else ""),
            "status": "completed",
            "conclusion": "success" if workflow_id == 1 else "failure",
            "run_attempt": 1,
        }
        runs[workflow_id] = run
        return [run]

    def retry_command(*args, **kwargs):
        if args[:3] == ("gh", "run", "rerun"):
            run_id = int(args[3])
            saved = json.loads(controller.path.read_text())["main_ci"]["workflows"]
            assert saved[runs[run_id]["path"]]["retry"]["requested"] is False
            assert args[-3:] == ("--repo", release_module.REPO, "--failed")
            reruns.append(run_id)
            return ""
        return command(*args, **kwargs)

    def finish_retry():
        runs[2].update(run_attempt=2, conclusion="success")

    monkeypatch.setattr(release_module, "pages", pages)
    monkeypatch.setattr(
        release_module, "api", lambda path: runs[int(path.split("/")[-1])]
    )
    monkeypatch.setattr(release_module, "command", retry_command)
    monkeypatch.setattr(controller, "pause", finish_retry)
    controller.check_main_ci()
    ci = controller.state["main_ci"]
    assert ci["validated"] and ci["sha"] == head
    assert set(ci["workflows"]) == {catalog[0]["path"], catalog[1]["path"]}
    assert ci["workflows"][catalog[1]["path"]]["sha"] == tested
    assert reruns == [2]
    runs[2]["conclusion"] = "failure"
    resumed = release_module.Release(controller.path, "amd")
    with pytest.raises(RuntimeError, match="Main CI failed after retry"):
        resumed.check_main_ci()
    assert reruns == [2]


def test_version_diff_rejects_non_version_changes_in_metadata_file(
    controller, release_module, version_repository
):
    command = release_module.command
    base = command("git", "rev-parse", "HEAD")
    versions = {
        p: release_module.next_version(
            release_module.read_version(p), release_module.read_version(p), ""
        )
        for p in release_module.PACKAGES.values()
    }
    controller.state["versions"] = versions
    release_module.update_metadata("amd", versions)
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "version metadata")
    head = command("git", "rev-parse", "HEAD")
    controller.verify_version_diff("amd", base, head)
    command("git", "checkout", "--detach", head)
    metadata = Path(release_module.PR_FILES["amd"][0])
    metadata.write_text(metadata.read_text() + "\n# extra non-version change\n")
    command("git", "add", ".")
    command("git", "commit", "-s", "--amend", "--no-edit")
    with pytest.raises(RuntimeError, match="outside the expected metadata"):
        controller.verify_version_diff("amd", base, command("git", "rev-parse", "HEAD"))


def test_version_fast_forward_refuses_main_race(
    controller, release_module, version_repository, tmp_path, monkeypatch
):
    command = release_module.command
    base = command("git", "rev-parse", "HEAD")
    versions = {
        p: release_module.next_version(
            release_module.read_version(p), release_module.read_version(p), ""
        )
        for p in release_module.PACKAGES.values()
    }
    controller.state.update(versions=versions, main_ci={"sha": base, "validated": True})
    bare = tmp_path / "origin.git"
    command("git", "clone", "--bare", str(version_repository), str(bare))
    command("git", "remote", "add", "origin", str(bare))
    release_module.update_metadata("amd", versions)
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "version metadata")
    head = command("git", "rev-parse", "HEAD")
    command("git", "push", "origin", f"{head}:refs/heads/bot/version")
    command("git", "checkout", "--detach", base)
    Path("README.md").write_text("Concurrent source change\n")
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "source update")
    other = command("git", "rev-parse", "HEAD")
    controller.phase.update(base=base, head=head, pr=1)
    pr = {
        "state": "OPEN",
        "headRefOid": head,
        "baseRefOid": base,
        "isDraft": False,
        "reviewDecision": "REVIEW_REQUIRED",
        "mergeable": "MERGEABLE",
    }

    def racing_command(*args, **kwargs):
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps(pr)
        if args[:2] == ("git", "push"):
            assert f"--force-with-lease=refs/heads/main:{base}" in args
            command("git", "push", "origin", f"{other}:refs/heads/main")
        return command(*args, **kwargs)

    monkeypatch.setattr(release_module, "command", racing_command)
    monkeypatch.setattr(release_module, "merge_policy", lambda: True)
    with pytest.raises(release_module.subprocess.CalledProcessError):
        controller.fast_forward_version("amd", pr)
    assert command("git", "ls-remote", "origin", "refs/heads/main").split()[0] == other


@pytest.mark.parametrize("component", ["scheduler", "mla"])
def test_component_metadata_prs_replay_exact_diffs_without_downstream_version_bump(
    release_module, version_repository, tmp_path, monkeypatch, component
):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    command = release_module.command
    base = command("git", "rev-parse", "HEAD")
    controller_type = {
        "scheduler": release_module.SchedulerRelease,
        "mla": release_module.MLARelease,
    }[component]
    package = f"tokenspeed-{component}"
    current = release_module.read_version(package)
    version = release_module.next_version(current, current, "")
    runtime = release_module.read_version("tokenspeed")
    kernel = release_module.read_version("tokenspeed-kernel")
    controller = controller_type(tmp_path / "state.json", f"{component}-version")
    dependency_path = controller.pr_files[f"{component}-dependency"][0]
    controller.state.update(
        versions={package: version},
        initial_versions={package: current},
        initial_dependency=str(
            release_module.requirements(dependency_path)[package].specifier
        ),
    )
    controller.phase["base"] = base
    controller.update_metadata(controller.stage)
    assert command("git", "diff", "--name-only").splitlines() == [
        release_module.PROJECTS[package]
    ]
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "component version")
    head = command("git", "rev-parse", "HEAD")
    controller.verify_version_diff(controller.stage, base, head)
    command("git", "checkout", "--detach", head)
    controller.save()
    dependency = controller_type(controller.path, f"{component}-dependency")
    dependency.phase["base"] = head
    dependency.update_metadata(dependency.stage)
    assert command("git", "diff", "--name-only").splitlines() == [dependency_path]
    assert (
        str(release_module.requirements(dependency_path)[package].specifier)
        == f"{'>=' if component == 'scheduler' else '=='}{version}"
    )
    assert release_module.read_version("tokenspeed") == runtime
    assert release_module.read_version("tokenspeed-kernel") == kernel
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "component dependency")
    dependency.verify_version_diff(
        dependency.stage, head, command("git", "rev-parse", "HEAD")
    )
    if component == "mla":
        path = Path(dependency_path)
        # Verification restores the base, whose pin can lag the package version.
        path.write_text(
            path.read_text().replace(
                f"tokenspeed-mla{dependency.state['initial_dependency']}",
                "tokenspeed-mla==0.0.0",
            )
        )
        assert (
            str(release_module.requirements(dependency_path)[package].specifier)
            == "==0.0.0"
        )
        with pytest.raises(RuntimeError, match="dependency changed outside"):
            dependency.update_metadata(dependency.stage)


@pytest.mark.parametrize("component", ["scheduler", "mla"])
def test_component_failed_publication_blocks_dependency_and_resumes_same_run(
    release_module, version_repository, tmp_path, monkeypatch, component
):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    command = release_module.command
    sha = command("git", "rev-parse", "HEAD")
    controller_type = {
        "scheduler": release_module.SchedulerRelease,
        "mla": release_module.MLARelease,
    }[component]
    package = f"tokenspeed-{component}"
    version = release_module.read_version(package)
    controller = controller_type(tmp_path / "state.json", f"{component}-publish")
    controller.state.update(
        versions={package: version},
        initial_versions={package: version},
    )
    controller.state["stages"][f"{component}-version"] = {"complete": True, "sha": sha}
    controller.save()
    files = {
        f"scheduler-cp{python}-{arch}.whl": "a" * 64
        for python in range(310, 314)
        for arch in ("x86_64", "aarch64")
    }
    files["scheduler.tar.gz"] = "b" * 64
    wheel = b"MLA universal wheel"
    if component == "mla":
        files = {
            f"tokenspeed_mla-{version}-py3-none-any.whl": hashlib.sha256(
                wheel
            ).hexdigest()
        }
    ref = controller.branch(f"{component}-version", version)
    workflow = f"release-{package}.yml"
    run = {
        "head_sha": sha,
        "head_branch": ref,
        "event": "workflow_dispatch",
        "path": f".github/workflows/{workflow}",
        "status": "in_progress",
        "html_url": f"https://github.com/{release_module.REPO}/actions/runs/456",
    }
    dispatched = []
    paused = []
    indexed = False
    artifact_downloads = []
    corrupt_artifact = True

    def artifact_command(*args, **kwargs):
        if args[:3] == ("gh", "run", "download"):
            assert (
                args[3] == "456"
                and args[args.index("--repo") + 1] == release_module.REPO
            )
            dist = Path(args[args.index("--dir") + 1])
            dist.mkdir(parents=True, exist_ok=True)
            (dist / next(iter(files))).write_bytes(
                b"corrupt" if corrupt_artifact else wheel
            )
            artifact_downloads.append(args)
            return ""
        return command(*args, **kwargs)

    def api(path, *, data=None):
        if data is not None:
            dispatched.append(data)
            return {"workflow_run_id": 456}
        return run

    def pause(self):
        nonlocal indexed
        paused.append(self.stage)
        if run["status"] != "completed":
            run.update(status="completed", conclusion="failure")
        else:
            indexed = True

    def request(url, *, github, **kwargs):
        if github:
            return {
                "draft": False,
                "prerelease": False,
                "assets": [
                    {"name": name, "digest": f"sha256:{digest}"}
                    for name, digest in files.items()
                ],
            }
        assert kwargs["accept"] == "application/vnd.pypi.simple.v1+json"
        return {
            "files": (
                [
                    {"filename": name, "hashes": {"sha256": digest}, "yanked": False}
                    for name, digest in files.items()
                ]
                if indexed
                else []
            )
        }

    monkeypatch.setattr(controller_type, "guard", lambda self: None)
    monkeypatch.setattr(
        controller_type,
        "immutable_ref",
        lambda self, stage, source: ref,
    )
    monkeypatch.setattr(controller_type, "available", lambda self, value: None)
    monkeypatch.setattr(controller_type, "pause", pause)
    monkeypatch.setattr(release_module, "command", artifact_command)
    monkeypatch.setattr(release_module, "api", api)
    monkeypatch.setattr(release_module, "request", request)
    monkeypatch.setattr(
        release_module,
        "pypi",
        lambda *args: {
            "urls": [
                {"filename": name, "digests": {"sha256": digest}, "yanked": False}
                for name, digest in files.items()
            ]
        },
    )
    monkeypatch.setattr(release_module, "source_sha", lambda *args: sha)
    with pytest.raises(RuntimeError, match="Child workflow failed"):
        controller.run("")
    assert not controller.phase.get("complete")
    dependency = controller_type(controller.path, f"{component}-dependency")
    with pytest.raises(RuntimeError, match=f"Previous {component} stage"):
        dependency.run("")
    run["conclusion"] = "success"
    resumed = controller_type(controller.path, f"{component}-publish")
    if component == "mla":
        with pytest.raises(RuntimeError, match="artifact differs"):
            resumed.run("")
        assert "publication_files" not in resumed.state
    corrupt_artifact = False
    resumed.run("")
    assert resumed.phase["complete"] and resumed.state["publication_files"] == files
    assert len(dispatched) == 1 and paused == [
        f"{component}-publish",
        f"{component}-publish",
    ]
    assert dispatched[0]["ref"] == ref and resumed.state["runs"][workflow]["id"] == 456
    if component == "mla":
        assert dispatched[0]["inputs"] == {} and len(artifact_downloads) == 2
        resumed.run("")
        assert len(dispatched) == 1 and len(artifact_downloads) == 2
    bare = tmp_path / "origin.git"
    command("git", "clone", "--bare", str(version_repository), str(bare))
    command("git", "remote", "add", "origin", str(bare))
    dependency = controller_type(controller.path, f"{component}-dependency")
    dependency.gate()
    if component == "mla":
        publisher = yaml.safe_load(
            (ROOT / ".github/workflows/release-tokenspeed-mla.yml").read_text()
        )
        source_gate = next(
            step["run"]
            for step in publisher["jobs"]["build"]["steps"]
            if step.get("name") == "Require main or a pinned release"
        )
        gate_env = dict(release_module.os.environ, SOURCE_REF=f"refs/heads/{ref}")
        assert (
            release_module.subprocess.run(
                ["bash", "-e", "-c", source_gate], env=gate_env, capture_output=True
            ).returncode
            == 0
        )
    source_path = Path(release_module.PROJECTS[package])
    source_path.write_text(
        source_path.read_text() + "\n# unpublished component change\n"
    )
    command("git", "add", ".")
    command("git", "commit", "-s", "-m", "component source update")
    with pytest.raises(RuntimeError, match="Unreleased changes"):
        dependency.gate()
    if component == "mla":
        assert (
            release_module.subprocess.run(
                ["bash", "-e", "-c", source_gate], env=gate_env, capture_output=True
            ).returncode
            != 0
        )


@pytest.mark.parametrize("component", ["scheduler", "mla"])
def test_component_version_rerun_keeps_reservation(
    release_module, version_repository, tmp_path, monkeypatch, component
):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    command = release_module.command
    bare = tmp_path / "origin.git"
    command("git", "clone", "--bare", str(version_repository), str(bare))
    command("git", "remote", "add", "origin", str(bare))
    controller_type = {
        "scheduler": release_module.SchedulerRelease,
        "mla": release_module.MLARelease,
    }[component]
    package = f"tokenspeed-{component}"
    current = release_module.read_version(package)
    expected = release_module.next_version(current, current, "")
    reserved = []
    monkeypatch.setattr(controller_type, "guard", lambda self: None)
    monkeypatch.setattr(
        controller_type,
        "available",
        lambda self, value: reserved.append(value),
    )
    monkeypatch.setattr(
        controller_type,
        "pr",
        lambda self, stage: self.state["versions"][package],
    )
    monkeypatch.setattr(release_module, "latest_version", lambda package: current)
    controller = controller_type(tmp_path / "state.json", f"{component}-version")
    controller.run("")
    resumed = controller_type(controller.path, f"{component}-version")
    resumed.run("")
    assert reserved == [expected] and resumed.state["versions"] == {package: expected}
