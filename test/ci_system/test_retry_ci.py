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

import json
import subprocess
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest
import retry_ci
import slurm_submit as slurm

REPOSITORY = "lightseekorg/tokenspeed"
COMMIT = "a" * 40
IMAGE = "ghcr.io/lightseekorg/tokenspeed-runner:test@sha256:" + "b" * 64


@pytest.mark.parametrize(
    "value", ["123", f"https://github.com/{REPOSITORY}/actions/runs/123"]
)
def test_run_id_and_url(value):
    assert retry_ci.parse_run_id(value, REPOSITORY) == 123


@pytest.mark.parametrize(
    "value", ["123; echo unsafe", "0", "https://github.com/other/repo/actions/runs/123"]
)
def test_invalid_run(value):
    with pytest.raises(ValueError):
        retry_ci.parse_run_id(value, REPOSITORY)


@pytest.mark.parametrize(
    "invalid", [None, "workflow", "event", "running", "artifact", "attempt"]
)
def test_resolve_completed_attempt(monkeypatch, invalid):
    reads = 0

    def api(endpoint):
        nonlocal reads
        if endpoint.endswith("/123"):
            reads += 1
            return {
                "path": (
                    ".github/workflows/other.yml"
                    if invalid == "workflow"
                    else ".github/workflows/slurm-dispatch.yml"
                ),
                "event": "pull_request" if invalid == "event" else "workflow_dispatch",
                "status": "in_progress" if invalid == "running" else "completed",
                "run_attempt": 3 if invalid == "attempt" and reads == 2 else 2,
            }
        if "/attempts/2/jobs?" in endpoint:
            return {"jobs": [{"labels": ["slurm-dispatch-gb300"]}]}
        assert "/artifacts?" in endpoint
        return {
            "artifacts": [
                {"name": "slurm-123-1", "id": 11, "expired": False},
                {"name": "slurm-123-2", "id": 12, "expired": invalid == "artifact"},
            ]
        }

    monkeypatch.setattr(retry_ci, "gh_json", api)
    if invalid:
        with pytest.raises(ValueError):
            retry_ci.resolve_run("123", REPOSITORY)
    else:
        result = retry_ci.resolve_run("123", REPOSITORY)
        assert result["artifact_id"] == "12"
        assert result["source_attempt"] == "2"
        assert result["coordinator"] == "slurm-dispatch-gb300"


@pytest.fixture
def report(tmp_path):
    root = tmp_path / "coordinator"
    for name in ("scripts", "snapshots", "logs", "runs"):
        (root / name).mkdir(parents=True)
    source = root / "snapshots" / f"{COMMIT}.tar"
    with tarfile.open(
        source, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": COMMIT}
    ) as archive:
        archive.addfile(tarfile.TarInfo("source.txt"))
    original = tmp_path / "original"
    original.mkdir()
    (original / "summary.md").write_text("**Target PR:** #42\n")
    rows, scripts = [], {}
    for number in (1, 2, 3):
        task = slurm.Task(
            f"test/ci/eval/case-{number}.yaml",
            f"case-{number}",
            "eval",
            "b200-4gpu",
            4,
            2 if number == 3 else 1,
        )
        stem = f"{task.name}-{COMMIT[:12]}-{number}000"
        script = slurm.render_script(task, source, root / "runs", root / "cache", IMAGE)
        scripts[task.name] = script
        (root / "scripts" / f"{stem}.sbatch").write_text(script)
        rows.append(
            {
                "job_id": str(number),
                "task": asdict(task),
                "log": str(root / "logs" / f"{stem}-{number}.out"),
                "state": "FAILED" if number == 2 else "COMPLETED",
                "exit_code": "1:0" if number == 2 else "0:0",
            }
        )
    (original / "1-result.json").write_text('{"ok": true}')
    (original / "2-result.json").write_text('{"ok": false}')
    manifest = original / "manifest.json"
    manifest.write_text(json.dumps(rows))
    return root, manifest, scripts


@pytest.mark.parametrize("problem", [None, "active", "missing", "snapshot", "outside"])
def test_replay_preserves_original_scripts_and_commit(
    report, monkeypatch, tmp_path, problem
):
    root, manifest, scripts = report
    calls = []
    if problem == "missing":
        next((root / "scripts").glob("case-3-*")).unlink()
    elif problem == "snapshot":
        next((root / "snapshots").iterdir()).unlink()
    elif problem == "outside":
        rows = json.loads(manifest.read_text())
        rows[2]["log"] = "/elsewhere/case-3.out"
        manifest.write_text(json.dumps(rows))

    def command(argv, **kwargs):
        if argv[0] == "squeue":
            assert argv == ["squeue", "--noheader", "--format=%i"]
            return subprocess.CompletedProcess(
                argv, 0, stdout="2\n" if problem == "active" else ""
            )
        assert argv[0] == "sbatch"
        assert retry_ci.os.environ["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] == "1"
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=str(100 + len(calls)))

    def wait(submissions, run_root, output, source_pr):
        assert source_pr == "42"
        assert [s.task.name for s in submissions] == ["case-2", "case-3"]
        return True

    monkeypatch.setattr(retry_ci.subprocess, "run", command)
    monkeypatch.setattr(slurm, "wait_all", wait)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("INSTALL_TOKENSPEED_MLA_FROM_SOURCE", "0")
    output = tmp_path / "retry"
    args = [
        "replay",
        "--manifest",
        str(manifest),
        "--artifact-root",
        str(root),
        "--report-dir",
        str(output),
        "--coordinator",
        "slurm-dispatch",
    ]
    assert retry_ci.main(args) == (2 if problem else 0)
    if problem:
        assert calls == []
    else:
        assert len(calls) == 2
        assert "--nodes=2" in calls[1]
        for row in json.loads((output / "manifest.json").read_text()):
            stem = Path(row["log"]).name.removesuffix(f"-{row['job_id']}.out")
            script = (root / "scripts" / f"{stem}.sbatch").read_text()
            assert script == scripts[row["task"]["name"]]
            assert COMMIT in script and IMAGE in script
        assert "#42" in (output / "summary.md").read_text()


def test_all_passed_needs_no_retained_files_or_slurm(report, monkeypatch, tmp_path):
    root, manifest, _ = report
    rows = json.loads(manifest.read_text())
    for row in rows:
        row.update(state="COMPLETED", exit_code="0:0")
        (manifest.parent / f"{row['job_id']}-result.json").write_text('{"ok": true}')
    manifest.write_text(json.dumps(rows))
    for path in (root / "scripts").iterdir():
        path.unlink()
    monkeypatch.setattr(
        retry_ci.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("no Slurm calls expected"),
    )
    monkeypatch.setenv("INSTALL_TOKENSPEED_MLA_FROM_SOURCE", "0")
    assert retry_ci.replay(manifest, root, tmp_path / "retry", "slurm-dispatch") == 0
