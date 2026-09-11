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

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from dataclasses import asdict

import pytest
import slurm_submit as slurm

COMMIT = "a" * 40
IMAGE = "ghcr.io/lightseekorg/tokenspeed-runner:test@sha256:" + "b" * 64


@pytest.fixture
def report(tmp_path):
    root = tmp_path / "coordinator"
    for folder in ("snapshots", "scripts", "logs", "runs"):
        (root / folder).mkdir(parents=True)
    source = root / "snapshots" / f"{COMMIT}.tar"
    with tarfile.open(
        source, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": COMMIT}
    ) as archive:
        payload = b"original source, not the current checkout"
        member = tarfile.TarInfo("source.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    report_dir = tmp_path / "download"
    report_dir.mkdir()
    (report_dir / "summary.md").write_text("## Slurm validation\n")
    manifest = report_dir / "manifest.json"
    rows = []

    def add(job_id, runner, state, ok, modern):
        declared = "b200-4gpu" if runner == "gb300-4gpu" else None
        task = slurm.Task(
            f"test/ci/eval/case-{job_id}.yaml",
            f"case-{job_id}",
            "eval",
            runner,
            4,
            1,
            declared,
        )
        script = slurm.render_script(task, source, root / "runs", root / "cache", IMAGE)
        stem = f"{task.name}-{COMMIT[:12]}-{job_id}000"
        script_path = root / "scripts" / f"{stem}.sbatch"
        script_path.write_text(script)
        log = root / "logs" / f"{stem}-{job_id}.out"
        data = asdict(task)
        data.pop("declared_runner")  # The original example report had no alias field.
        row = {
            "job_id": job_id,
            "task": data,
            "log": str(log),
            "state": state,
            "exit_code": "0:0" if state == "COMPLETED" else "1:0",
        }
        if modern:
            args = argparse.Namespace(
                partition="original-partition",
                time="01:23:45",
                nodelist="node-[01-02]",
                container_image=IMAGE,
            )
            context = slurm.replay_context(source, args, COMMIT, "42")
            context.update(
                script=str(script_path),
                script_sha256=hashlib.sha256(script.encode()).hexdigest(),
            )
            context["environment"]["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] = "1"
            row["replay"] = context
            row["task"]["declared_runner"] = declared
        rows.append(row)
        manifest.write_text(json.dumps(rows))
        if ok is not None:
            (report_dir / f"{job_id}-result.json").write_text(json.dumps({"ok": ok}))
        return row, script_path

    return root, source, manifest, rows, add


def test_select_only_unsuccessful_cases(report):
    _, _, manifest, _, add = report
    add("1", "b200-4gpu", "COMPLETED", True, False)
    add("2", "b200-4gpu", "FAILED", False, False)
    add("3", "gb200-4gpu", "COMPLETED", True, False)
    assert [row["job_id"] for row in slurm.failed_report_rows(manifest)] == ["2"]


@pytest.mark.parametrize(
    "state,ok",
    [
        ("CANCELLED", None),
        ("TIMEOUT", None),
        ("UNKNOWN", None),
        ("COMPLETED", None),
        ("COMPLETED", False),
    ],
)
def test_nonpassing_or_missing_results_are_retryable(report, state, ok):
    _, _, manifest, _, add = report
    add("1", "b200-4gpu", state, ok, False)
    assert len(slurm.failed_report_rows(manifest)) == 1


def test_nonzero_exit_and_malformed_result_are_retryable(report):
    _, _, manifest, rows, add = report
    row, _ = add("1", "b200-4gpu", "COMPLETED", True, False)
    row["exit_code"] = "1:0"
    manifest.write_text(json.dumps(rows))
    assert len(slurm.failed_report_rows(manifest)) == 1
    row["exit_code"] = "0:0"
    manifest.write_text(json.dumps(rows))
    (manifest.parent / "1-result.json").write_text("broken JSON")
    assert len(slurm.failed_report_rows(manifest)) == 1


def test_legacy_replay_preserves_old_source_image_and_script(report, monkeypatch):
    root, source, manifest, _, add = report
    add("1", "b200-4gpu", "COMPLETED", True, False)
    _, retained = add("2", "gb200-4gpu", "FAILED", False, False)
    monkeypatch.setattr(
        slurm,
        "git",
        lambda *args: pytest.fail("replay must not resolve the current branch"),
    )
    monkeypatch.setattr(
        slurm,
        "snapshot",
        lambda *args: pytest.fail("replay must not overwrite the original snapshot"),
    )
    prepared = slurm.prepare_replay(manifest, root, "slurm-dispatch")
    assert len(prepared) == 1
    task, script, context = prepared[0]
    assert (task.config, task.runner) == ("test/ci/eval/case-2.yaml", "gb200-4gpu")
    assert script == retained.read_text()
    assert context["commit"] == COMMIT
    assert context["snapshot"] == str(source)
    assert context["container_image"] == IMAGE
    assert context["environment"]["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] == "0"


@pytest.mark.parametrize(
    "summary",
    [
        "**Target PR:** [#42](https://github.com/lightseekorg/tokenspeed/pull/42)\n",
        "**Target PR:** #42\n",
    ],
)
def test_legacy_pr_install_flag_and_gb300_alias_survive(report, summary):
    root, _, manifest, _, add = report
    add("1", "gb300-4gpu", "FAILED", False, False)
    (manifest.parent / "summary.md").write_text(summary)
    task, _, context = slurm.prepare_replay(manifest, root, "slurm-dispatch-gb300")[0]
    assert task.runner == "gb300-4gpu"
    assert task.declared_runner == "b200-4gpu"
    assert context["source_pr"] == "42"
    assert context["environment"]["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] == "1"


def test_modern_replay_preserves_scheduler_options_and_install_mode(
    report, monkeypatch, tmp_path
):
    root, _, manifest, _, add = report
    _, retained = add("1", "gb300-4gpu", "FAILED", False, True)
    prepared = slurm.prepare_replay(manifest, root, "slurm-dispatch-gb300")
    calls = []

    def command(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv[0] == "sbatch"
        return subprocess.CompletedProcess(argv, 0, stdout="987\n", stderr="")

    monkeypatch.setattr(slurm.subprocess, "run", command)
    args = argparse.Namespace(
        wait=False,
        render=False,
        follow=False,
        report_dir=str(tmp_path / "retry"),
        partition="new-default",
        time="00:01:00",
        nodelist=None,
    )
    assert slurm.submit_prepared(prepared, args, root) == 0
    argv, options = calls[0]
    assert "--partition=original-partition" in argv
    assert "--time=01:23:45" in argv
    assert "--nodelist=node-[01-02]" in argv
    assert options["env"]["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] == "1"
    new_scripts = [
        path for path in (root / "scripts").glob("*.sbatch") if path != retained
    ]
    assert len(new_scripts) == 1
    assert new_scripts[0].read_bytes() == retained.read_bytes()


@pytest.mark.parametrize("original_value", [None, "", "/original/models"])
def test_submission_preserves_environment_presence(report, monkeypatch, original_value):
    root, source, manifest, rows, add = report
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("INSTALL_TOKENSPEED_MLA_FROM_SOURCE", raising=False)
    if original_value is None:
        monkeypatch.delenv("TS_CI_LOCAL_MODEL_ROOT", raising=False)
    else:
        monkeypatch.setenv("TS_CI_LOCAL_MODEL_ROOT", original_value)
    row, retained = add("1", "b200-4gpu", "FAILED", False, True)
    args = argparse.Namespace(
        partition="batch",
        time="12:00:00",
        nodelist=None,
        container_image=IMAGE,
        render=False,
    )
    context = slurm.replay_context(source, args, COMMIT, None)
    assert context["environment"] == {
        "INSTALL_TOKENSPEED_MLA_FROM_SOURCE": None,
        "TS_CI_LOCAL_MODEL_ROOT": original_value,
    }
    calls = []

    def command(argv, **kwargs):
        calls.append(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout="987\n", stderr="")

    monkeypatch.setattr(slurm.subprocess, "run", command)
    task = slurm.Task(**row["task"])
    slurm.submit(task, retained.read_text(), root, args, COMMIT, context)
    row["replay"]["environment"] = context["environment"]
    manifest.write_text(json.dumps(rows))
    # A retry must also clear new coordinator overrides that were absent originally.
    monkeypatch.setenv("INSTALL_TOKENSPEED_MLA_FROM_SOURCE", "1")
    monkeypatch.setenv("TS_CI_LOCAL_MODEL_ROOT", "/new/models")
    task, script, metadata = slurm.prepare_replay(manifest, root, "slurm-dispatch")[0]
    slurm.submit(task, script, root, args, COMMIT, metadata)
    for environment in calls:
        assert "INSTALL_TOKENSPEED_MLA_FROM_SOURCE" not in environment
        if original_value is None:
            assert "TS_CI_LOCAL_MODEL_ROOT" not in environment
        else:
            assert environment["TS_CI_LOCAL_MODEL_ROOT"] == original_value


@pytest.mark.parametrize(
    "state",
    ["PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "COMPLETED"],
)
def test_old_job_check_handles_mixed_purged_and_live_ids(monkeypatch, state):
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if "--jobs=1" in argv:
            raise subprocess.CalledProcessError(
                1,
                argv,
                output="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        assert "--jobs=2" in argv
        return subprocess.CompletedProcess(argv, 0, stdout=f"2|{state}|00:01|node01\n")

    monkeypatch.setattr(slurm.subprocess, "run", command)
    assert set(slurm.active_replay_jobs(["1", "2"])) == (
        set() if state == "COMPLETED" else {"2"}
    )
    assert len(calls) == 2


def test_old_job_check_does_not_hide_scheduler_outage(monkeypatch):
    def command(argv, **kwargs):
        raise subprocess.CalledProcessError(
            1, argv, output="", stderr="Unable to contact slurm controller"
        )

    monkeypatch.setattr(slurm.subprocess, "run", command)
    with pytest.raises(subprocess.CalledProcessError):
        slurm.active_replay_jobs(["1"])


@pytest.mark.parametrize(
    "corruption,error",
    [
        ("missing_script", "No such file"),
        ("missing_snapshot", "No such file"),
        ("wrong_commit", "commit"),
        ("wrong_image", "immutable"),
        ("wrong_config", "config"),
        ("wrong_topology", "topology"),
        ("outside_root", "artifact root"),
        ("wrong_cluster", "coordinator"),
    ],
)
def test_replay_validates_all_cases_before_submitting(
    report, monkeypatch, corruption, error
):
    root, source, manifest, rows, add = report
    add("1", "b200-4gpu", "FAILED", False, False)
    row, script = add("2", "b200-4gpu", "FAILED", False, False)
    coordinator = "slurm-dispatch"
    if corruption == "missing_script":
        script.unlink()
    elif corruption == "missing_snapshot":
        source.unlink()
    elif corruption == "wrong_commit":
        with tarfile.open(
            source, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "c" * 40}
        ) as archive:
            archive.addfile(tarfile.TarInfo("source.txt"))
    elif corruption == "wrong_image":
        script.write_text(
            script.read_text().replace(IMAGE, "registry.invalid/unpinned:latest")
        )
    elif corruption == "wrong_config":
        script.write_text(script.read_text().replace("case-2.yaml", "different.yaml"))
    elif corruption == "wrong_topology":
        script.write_text(script.read_text().replace("--nodes=1", "--nodes=2"))
    elif corruption == "outside_root":
        row["log"] = "/other/root/logs/case.out"
    elif corruption == "wrong_cluster":
        coordinator = "slurm-dispatch-gb300"
    manifest.write_text(json.dumps(rows))
    monkeypatch.setattr(
        slurm,
        "submit",
        lambda *args: pytest.fail(
            "no job may be submitted before all preflight checks pass"
        ),
    )
    with pytest.raises((ValueError, OSError), match=error):
        slurm.prepare_replay(manifest, root, coordinator)


@pytest.mark.parametrize(
    "kind",
    [
        "script",
        "snapshot",
        "environment",
        "environment_list",
        "source_pr",
        "source_pr_type",
    ],
)
def test_modern_metadata_detects_changed_inputs(report, kind):
    root, source, manifest, rows, add = report
    row, script = add("1", "b200-4gpu", "FAILED", False, True)
    if kind == "script":
        script.write_text(script.read_text() + "\n# changed\n")
    elif kind == "snapshot":
        with source.open("ab") as stream:
            stream.write(b"changed")
    elif kind == "environment":
        row["replay"]["environment"]["BASH_ENV"] = "/untrusted/file"
    elif kind == "environment_list":
        row["replay"]["environment"] = ["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"]
    elif kind == "source_pr":
        row["replay"]["source_pr"] = "not-a-pr"
    else:
        row["replay"]["source_pr"] = {"number": 42}
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="checksum|environment|PR|pull request"):
        slurm.prepare_replay(manifest, root, "slurm-dispatch")


def test_all_passed_is_noop_without_retained_files(report):
    root, source, manifest, _, add = report
    _, script = add("1", "b200-4gpu", "COMPLETED", True, False)
    script.unlink()
    source.unlink()
    assert slurm.prepare_replay(manifest, root, "slurm-dispatch") == []


def test_multinode_replay_keeps_allocation(report):
    root, source, manifest, rows, add = report
    row, script = add("1", "slurm-gb300-4gpu", "FAILED", False, False)
    row["task"]["nodes"] = 2
    task = slurm.Task(**row["task"])
    script.write_text(
        slurm.render_script(task, source, root / "runs", root / "cache", IMAGE)
    )
    manifest.write_text(json.dumps(rows))
    replayed, text, _ = slurm.prepare_replay(manifest, root, "slurm-dispatch-gb300")[0]
    assert (replayed.nodes, replayed.gpus, replayed.runner) == (
        2,
        4,
        "slurm-gb300-4gpu",
    )
    assert text == script.read_text()


@pytest.mark.parametrize("still_active", [False, True])
def test_replay_cli_never_reads_current_source_and_report_is_retryable(
    report, monkeypatch, tmp_path, still_active
):
    root, _, manifest, _, add = report
    add("1", "b200-4gpu", "FAILED", False, True)
    retry_report = tmp_path / "retry"
    calls = []

    def forbidden(*args):
        pytest.fail("replay must not read or archive the current checkout")

    for name in ("git", "snapshot", "select_tasks"):
        monkeypatch.setattr(slurm, name, forbidden)
    monkeypatch.setattr(
        slurm,
        "slurm_states",
        lambda ids: {
            "987": {"state": "FAILED", "elapsed": "00:01", "exit_code": "1:0"}
        },
    )

    def command(argv, **kwargs):
        if argv[0] == "squeue":
            if "--jobs=987" in argv:
                return subprocess.CompletedProcess(argv, 0, stdout="")
            assert "--jobs=1" in argv
            if still_active:
                return subprocess.CompletedProcess(
                    argv, 0, stdout="1|RUNNING|00:01|node01\n"
                )
            raise subprocess.CalledProcessError(
                1,
                argv,
                output="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        calls.append(argv)
        assert argv[0] == "sbatch"
        result_dir = root / "runs" / "987"
        result_dir.mkdir()
        (result_dir / "result.json").write_text('{"ok": false}')
        return subprocess.CompletedProcess(argv, 0, stdout="987\n", stderr="")

    monkeypatch.setattr(slurm.subprocess, "run", command)
    result = slurm.main(
        [
            "--replay-manifest",
            str(manifest),
            "--replay-coordinator",
            "slurm-dispatch",
            "--repo-root",
            str(tmp_path / "unrelated-checkout"),
            "--artifact-root",
            str(root),
            "--cache-dir",
            str(tmp_path / "new-cache"),
            "--container-image",
            "ignored-new-image",
            "--wait",
            "--report-dir",
            str(retry_report),
        ]
    )
    if still_active:
        assert result == 2
        assert calls == []
    else:
        assert result == 1
        assert len(calls) == 1
        task, _, context = slurm.prepare_replay(
            retry_report / "manifest.json", root, "slurm-dispatch"
        )[0]
        assert task.config == "test/ci/eval/case-1.yaml"
        assert context["commit"] == COMMIT
        assert context["container_image"] == IMAGE
        assert context["source_job_id"] == "987"


@pytest.mark.parametrize("during_accounting_gap", [False, True])
def test_interrupted_wait_keeps_original_context(
    report, monkeypatch, tmp_path, during_accounting_gap
):
    import signal

    root, _, manifest, _, add = report
    add("1", "b200-4gpu", "FAILED", False, True)
    task, _, metadata = slurm.prepare_replay(manifest, root, "slurm-dispatch")[0]
    submission = slurm.Submission(task, "987", root / "logs" / "987.out", metadata)
    calls = []

    def interrupt(ids):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    def command(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(
        slurm,
        "queued_states",
        (lambda ids: {"987": {}}) if during_accounting_gap else interrupt,
    )
    monkeypatch.setattr(slurm, "slurm_states", lambda ids: {})
    monkeypatch.setattr(slurm.time, "sleep", interrupt)
    monkeypatch.setattr(slurm.subprocess, "run", command)
    retry_report = tmp_path / "interrupted"
    with pytest.raises(SystemExit) as exc:
        slurm.wait_all([submission], root / "runs", retry_report, source_pr=None)
    assert exc.value.code == 128 + signal.SIGTERM
    assert calls == [["scancel", "987"]]
    saved = json.loads((retry_report / "manifest.json").read_text())
    assert saved[0]["replay"]["commit"] == COMMIT
    assert saved[0]["state"] == "UNKNOWN"


def test_report_retains_replay_context_after_later_submission_fails(
    report, monkeypatch, tmp_path
):
    root, _, manifest, _, add = report
    add("1", "b200-4gpu", "FAILED", False, True)
    add("2", "b200-4gpu", "FAILED", False, True)
    prepared = slurm.prepare_replay(manifest, root, "slurm-dispatch")
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "scancel":
            return subprocess.CompletedProcess(argv, 0)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 0, stdout="987\n", stderr="")
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(slurm.subprocess, "run", command)
    retry_report = tmp_path / "retry"
    args = argparse.Namespace(
        wait=True, render=False, follow=False, report_dir=str(retry_report)
    )
    with pytest.raises(subprocess.CalledProcessError):
        slurm.submit_prepared(prepared, args, root)
    saved = json.loads((retry_report / "manifest.json").read_text())
    assert saved[0]["job_id"] == "987"
    assert saved[0]["replay"]["commit"] == COMMIT
    assert saved[0]["replay"]["partition"] == "original-partition"
    assert calls[-1] == ["scancel", "987"]
