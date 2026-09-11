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

"""Resolve a Slurm run and retry unsuccessful cases using its retained scripts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

SUPPORTED_WORKFLOWS = {
    ".github/workflows/slurm-dispatch.yml",
    ".github/workflows/retry-failed-ci-cases.yml",
}
COORDINATORS = {"slurm-dispatch", "slurm-dispatch-gb300"}


def parse_run_id(value: str, repository: str) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository")
    value = value.strip()
    if re.fullmatch(r"[1-9]\d*", value):
        return int(value)
    match = re.fullmatch(
        rf"https://github\.com/{re.escape(repository)}/actions/runs/([1-9]\d*)/?",
        value,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError("source_run must be a run ID or a run URL in this repository")
    return int(match[1])


def gh_json(endpoint: str) -> dict:
    result = subprocess.run(
        ["gh", "api", endpoint], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def gh_items(endpoint: str, key: str) -> list[dict]:
    items = []
    page = 1
    while True:
        batch = gh_json(f"{endpoint}?per_page=100&page={page}")[key]
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def resolve_run(source_run: str, repository: str) -> dict[str, str]:
    run_id = parse_run_id(source_run, repository)
    endpoint = f"repos/{repository}/actions/runs/{run_id}"
    run = gh_json(endpoint)
    if run["status"] != "completed":
        raise ValueError("wait for the source run to finish before retrying its cases")
    if run["path"].split("@", 1)[0] not in SUPPORTED_WORKFLOWS:
        raise ValueError(
            "this workflow accepts Slurm Dispatch or Retry Failed CI Cases runs; "
            "use GitHub's Re-run failed jobs for CI with one case per job"
        )
    if run["event"] != "workflow_dispatch":
        raise ValueError("only manually dispatched Slurm reports are supported")
    attempt = run["run_attempt"]
    jobs = gh_items(f"{endpoint}/attempts/{attempt}/jobs", "jobs")
    coordinators = [
        COORDINATORS.intersection(job.get("labels", []))
        for job in jobs
        if COORDINATORS.intersection(job.get("labels", []))
    ]
    if len(coordinators) != 1 or len(coordinators[0]) != 1:
        raise ValueError("source attempt has no unique Slurm coordinator job")
    name = f"slurm-{run_id}-{attempt}"
    artifacts = [
        artifact
        for artifact in gh_items(f"{endpoint}/artifacts", "artifacts")
        if artifact["name"] == name
    ]
    if len(artifacts) != 1 or artifacts[0]["expired"]:
        raise ValueError(f"source attempt report {name} is missing or expired")
    # Do not mix a report from one attempt with a concurrently started retry.
    current = gh_json(endpoint)
    if current["run_attempt"] != attempt or current["status"] != "completed":
        raise ValueError(
            "source run changed while resolving it; try again after it finishes"
        )
    return {
        "source_run_id": str(run_id),
        "source_attempt": str(attempt),
        "artifact_name": name,
        "artifact_id": str(artifacts[0]["id"]),
        "coordinator": next(iter(coordinators[0])),
    }


def failed_cases(manifest: Path) -> list[dict]:
    failed = []
    for row in json.loads(manifest.read_text()):
        try:
            result = json.loads(
                (manifest.parent / f"{row['job_id']}-result.json").read_text()
            )
        except (OSError, ValueError):
            result = {}
        if not (
            row["state"] == "COMPLETED"
            and row["exit_code"] in ("0", "0:0")
            and isinstance(result, dict)
            and result.get("ok") is True
        ):
            failed.append(row)
    return failed


def replay(manifest: Path, root: Path, report: Path, coordinator: str) -> int:
    import slurm_submit as slurm

    root = root.resolve()
    rows = failed_cases(manifest)
    prepared = []
    for row in rows:
        task = slurm.Task(**row["task"])
        gb300 = task.runner.startswith(("gb300-", "slurm-gb300-"))
        if gb300 != (coordinator == "slurm-dispatch-gb300"):
            raise ValueError("case belongs to a different Slurm coordinator")
        log = Path(row["log"])
        suffix = f"-{row['job_id']}.out"
        stem = log.name.removesuffix(suffix)
        if (
            log.parent.resolve() != root / "logs"
            or not log.name.endswith(suffix)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+-[0-9a-f]{12}-[0-9]+", stem)
        ):
            raise ValueError("invalid retained script path")
        retained = (root / "scripts" / f"{stem}.sbatch").resolve()
        if retained.parent != root / "scripts":
            raise ValueError("retained script is outside the artifact root")
        script = retained.read_text()
        archives = re.findall(r"^source_archive=(.+)$", script, re.M)
        if len(archives) != 1 or len(shlex.split(archives[0])) != 1:
            raise ValueError("retained script has no unique source snapshot")
        source = Path(shlex.split(archives[0])[0]).resolve()
        if source.parent != root / "snapshots" or not re.fullmatch(
            r"[0-9a-f]{40}\.tar", source.name
        ):
            raise ValueError("invalid retained snapshot path")
        with tarfile.open(source) as archive:
            if archive.pax_headers.get("comment") != source.stem:
                raise ValueError("snapshot does not match its original commit")
        prepared.append((task, script, source.stem))
    if len({commit for _, _, commit in prepared}) > 1:
        raise ValueError("source cases do not share one commit")

    summary = (manifest.parent / "summary.md").read_text()
    pr = re.search(r"^\*\*Target PR:\*\* \[?#([0-9]+)", summary, re.M)
    source_pr = pr[1] if pr else None
    if rows:
        queued = subprocess.run(
            ["squeue", "--noheader", "--format=%i"],
            check=True,
            capture_output=True,
            text=True,
        )
        if {row["job_id"] for row in rows}.intersection(queued.stdout.split()):
            raise ValueError("original Slurm cases are still active")
    os.environ["INSTALL_TOKENSPEED_MLA_FROM_SOURCE"] = "1" if source_pr else "0"
    options = argparse.Namespace(
        partition="batch", time="12:00:00", nodelist=None, render=False
    )
    submitted = []
    try:
        for task, script, commit in prepared:
            print(f"Retry {task.name} at {commit}", flush=True)
            submitted.append(slurm.submit(task, script, root, options, commit))
        slurm.write_report(submitted, {}, root / "runs", report, source_pr)
        return (
            0
            if not submitted
            or slurm.wait_all(submitted, root / "runs", report, source_pr)
            else 1
        )
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *[s.job_id for s in submitted]], check=False)
            slurm.write_report(submitted, {}, root / "runs", report, source_pr)
        raise


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--source-run", required=True)
    resolve.add_argument("--repository", required=True)
    resolve.add_argument("--github-output", type=Path, required=True)
    retry = commands.add_parser("replay")
    retry.add_argument("--manifest", type=Path, required=True)
    retry.add_argument("--artifact-root", type=Path, required=True)
    retry.add_argument("--report-dir", type=Path, required=True)
    retry.add_argument("--coordinator", choices=sorted(COORDINATORS), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            return replay(
                args.manifest, args.artifact_root, args.report_dir, args.coordinator
            )
        outputs = resolve_run(args.source_run, args.repository)
        with args.github_output.open("a") as output:
            for key, value in outputs.items():
                output.write(f"{key}={value}\n")
        print(json.dumps(outputs, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Cannot retry source run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
