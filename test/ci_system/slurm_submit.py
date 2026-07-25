#!/usr/bin/env python3
"""Select and submit existing TokenSpeed CI YAML tasks through Slurm/Pyxis."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pipeline import build_matrix, normalize_task

GPU_RE = re.compile(r"(?:^|-)([1-9]\d*)gpu(?:-|$)")
TASK_TYPES = {"ut", "server_smoke", "eval", "perf"}
DEFAULT_TASK_TYPES = {"eval", "perf"}
PR_RE = re.compile(r"^(?:https://github\.com/[^/]+/[^/]+/pull/)?(\d+)(?:/)?$")


@dataclass(frozen=True)
class Task:
    config: str
    name: str
    task_type: str
    runner: str
    gpus: int


@dataclass(frozen=True)
class Submission:
    task: Task
    job_id: str
    log: Path


def gpu_count(runner: str) -> int:
    matches = GPU_RE.findall(runner)
    if len(matches) != 1:
        raise ValueError(f"runner {runner!r} must contain one '<N>gpu' segment")
    return int(matches[0])


def load_task(repo: Path, config: str, runner: str | None = None) -> Task:
    path = (repo / config).resolve()
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ValueError("--config must be inside the repository") from exc
    data = normalize_task(path, repo)
    if data["type"] not in TASK_TYPES:
        raise ValueError(f"unsupported task type {data['type']!r}")
    labels = list(data["runner"]["labels"])
    if runner is None:
        if len(labels) != 1:
            raise ValueError("task has multiple runners; pass --runner")
        runner = labels[0]
    if runner not in labels:
        raise ValueError(f"{runner!r} is not declared by {relative}")
    return Task(
        relative, str(data["name"]), str(data["type"]), runner, gpu_count(runner)
    )


def task_matches(repo: Path, task: Task, patterns: list[str]) -> bool:
    if not patterns:
        return True
    data = normalize_task(repo / task.config, repo)
    commands = [
        str(data.get(stage, {}).get("command", ""))
        for stage in ("server", "ut", "server_smoke", "eval", "perf")
    ]
    haystack = "\n".join([task.config, task.name, *commands]).lower()
    return any(pattern.lower() in haystack for pattern in patterns)


def select_tasks(args: argparse.Namespace, repo: Path) -> list[Task]:
    runners = args.runner or []
    task_types = set(args.task_types or DEFAULT_TASK_TYPES)
    patterns = args.match or []
    excluded_patterns = getattr(args, "exclude_match", None) or []

    def matches_filters(task: Task) -> bool:
        return task_matches(repo, task, patterns) and (
            not excluded_patterns or not task_matches(repo, task, excluded_patterns)
        )

    if args.config:
        tasks = (
            [load_task(repo, args.config, runner) for runner in runners]
            if runners
            else [load_task(repo, args.config)]
        )
        tasks = [
            task
            for task in tasks
            if task.task_type in task_types and matches_filters(task)
        ]
        if not tasks:
            raise ValueError("the selected config does not match the task filters")
        return tasks
    if not runners:
        raise ValueError("--all requires --runner")
    matrix = build_matrix(repo / "test/ci", repo, args.trigger)
    tasks = [
        load_task(repo, item["config"], item["runner"])
        for item in matrix["include"]
        if item["type"] in task_types and item["runner"] in runners
    ]
    tasks = [task for task in tasks if matches_filters(task)]
    if not tasks:
        raise ValueError("no tasks match the requested runners and filters")
    return tasks


def parse_pr_number(value: str) -> int:
    match = PR_RE.fullmatch(value)
    if not match:
        raise ValueError(
            "--pr must be a pull request number or GitHub pull request URL"
        )
    return int(match.group(1))


@contextlib.contextmanager
def pr_worktree(repo: Path, value: str):
    number = parse_pr_number(value)
    if git(repo, "rev-parse", "--is-shallow-repository") == "true":
        raise ValueError(
            "cannot merge a pull request from a shallow checkout; "
            "use actions/checkout with fetch-depth: 0 or unshallow the repository"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f"tokenspeed-pr-{number}-"))
    reference = f"refs/tokenspeed-slurm/pr-{number}-{os.getpid()}"
    try:
        git(repo, "fetch", "origin", f"refs/pull/{number}/head:{reference}")
        git(repo, "worktree", "add", "--detach", str(temporary), "HEAD")
        environment = os.environ.copy()
        environment.setdefault("GIT_AUTHOR_NAME", "TokenSpeed Slurm")
        environment.setdefault("GIT_AUTHOR_EMAIL", "slurm@tokenspeed.local")
        environment.setdefault("GIT_COMMITTER_NAME", environment["GIT_AUTHOR_NAME"])
        environment.setdefault("GIT_COMMITTER_EMAIL", environment["GIT_AUTHOR_EMAIL"])
        subprocess.run(
            ["git", "-C", str(temporary), "merge", "--no-ff", "--no-edit", reference],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        yield temporary
    finally:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(temporary)],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "update-ref", "-d", reference],
            capture_output=True,
            text=True,
        )
        shutil.rmtree(temporary, ignore_errors=True)


def print_tasks(tasks: list[Task]) -> None:
    print("TYPE\tRUNNER\tGPUS\tCONFIG\tNAME")
    for task in tasks:
        print(
            f"{task.task_type}\t{task.runner}\t{task.gpus}\t"
            f"{task.config}\t{task.name}"
        )


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def snapshot(repo: Path, artifact_root: Path, commit: str) -> Path:
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit tracked changes before submitting")
    target = artifact_root / "snapshots" / f"{commit}.tar"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(
            ["git", "-C", str(repo), "archive", f"--output={temporary}", commit],
            check=True,
        )
        temporary.replace(target)
    return target


def shell_array(name: str, values: list[str]) -> str:
    body = "\n".join(f"  {shlex.quote(value)}" for value in values)
    return f"{name}=(\n{body}\n)"


def render_script(
    task: Task,
    source: Path,
    run_root: Path,
    cache: Path,
    image: str,
) -> str:
    pipeline = [
        "python3",
        "test/ci_system/pipeline.py",
        "execute",
        f"--config={task.config}",
        f"--runner={task.runner}",
        "--work-dir=/workspace",
        "--setup-mode=slurm",
        "--print-plan",
        "--result-json=/workspace/.ci-artifacts/result.json",
    ]
    bootstrap = (
        'export LD_LIBRARY_PATH="/opt/tokenspeed-cuda-runtime'
        '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; '
        'if ! python3 -c "import yaml" >/dev/null 2>&1; then '
        "python3 -m pip install --no-cache-dir "
        '--target=/tmp/tokenspeed-ci-python "PyYAML>=6,<7"; '
        'export PYTHONPATH="/tmp/tokenspeed-ci-python'
        '${PYTHONPATH:+:${PYTHONPATH}}"; fi; exec "$@"'
    )
    srun = [
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{task.gpus}",
        "--unbuffered",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        f"--container-image={image}",
        "--container-workdir=/workspace",
        "--no-container-entrypoint",
        "--no-container-mount-home",
        "--container-writable",
        "--container-remap-root",
        "--container-env=SLURM_JOB_ID,RUNNER_NAME,HF_TOKEN,"
        "HUGGING_FACE_HUB_TOKEN,HF_HOME,XDG_CACHE_HOME",
    ]
    command = ["bash", "-c", bootstrap, "tokenspeed-pipeline", *pipeline]
    return f"""#!/usr/bin/env bash
set -euo pipefail

export RUNNER_NAME="slurm-${{SLURM_JOB_ID}}"
export HF_HOME=/home/runner/.cache/huggingface
export XDG_CACHE_HOME=/home/runner/.cache
unset GITHUB_STEP_SUMMARY GITHUB_OUTPUT GITHUB_ENV GITHUB_PATH GITHUB_STATE \
  GITHUB_EVENT_PATH

scratch="${{SLURM_TMPDIR:-/tmp}}/tokenspeed-${{SLURM_JOB_ID}}"
src="$scratch/src"
tmp="$scratch/tmp"
run={shlex.quote(str(run_root))}/"${{SLURM_JOB_ID}}"
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$src/.ci-artifacts" "$tmp" "$run"
tar -xf {shlex.quote(str(source))} -C "$src"

mounts=(
  "$src:/workspace"
  "$tmp:/tmp"
  "$run:/workspace/.ci-artifacts"
  {shlex.quote(str(cache) + ":/home/runner/.cache")}
)

# This cluster's Pyxis hook exposes devices but omits driver libraries/tools.
driver_dir=""
for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
  link="$(ldconfig -p 2>/dev/null | awk -v name="$lib" \
    '$1 == name && !found {{ print $NF; found = 1 }}')"
  if [ -n "$link" ] && [ -e "$link" ]; then
    path="$(readlink -f "$link")"
    [ -n "$driver_dir" ] || driver_dir="$(dirname "$path")"
    mounts+=("$path:$driver_dir/$lib:ro")
  fi
done

cudart="$(ldconfig -p 2>/dev/null | awk \
  '$1 == "libcudart.so.13" && !found {{ print $NF; found = 1 }}')"
if [ -n "$cudart" ] && [ -e "$cudart" ]; then
  mounts+=("$(dirname "$(readlink -f "$cudart")"):/opt/tokenspeed-cuda-runtime:ro")
fi
nvidia_smi="$(command -v nvidia-smi 2>/dev/null || true)"
[ -z "$nvidia_smi" ] || mounts+=("$nvidia_smi:/usr/bin/nvidia-smi:ro")

container_mounts="$(IFS=,; printf '%s' "${{mounts[*]}}")"
{shell_array("srun_args", srun)}
srun_args+=(--container-mounts="$container_mounts")
{shell_array("container_command", command)}
srun "${{srun_args[@]}}" "${{container_command[@]}}"
"""


def submit(
    task: Task,
    script: str,
    artifact_root: Path,
    args: argparse.Namespace,
    commit: str,
) -> Submission:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", task.name).strip("-")
    unique = f"{commit[:12]}-{time.time_ns()}"
    script_path = artifact_root / "scripts" / f"{stem}-{unique}.sbatch"
    log_pattern = artifact_root / "logs" / f"{stem}-{unique}-%j.out"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    log_pattern.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    script_path.chmod(0o755)

    command = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--job-name=ts-{stem}",
        f"--partition={args.partition}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{task.gpus}",
        "--exclusive",
        "--mem=0",
        f"--output={log_pattern}",
        "--chdir=/tmp",
    ]
    if args.time:
        command.append(f"--time={args.time}")
    if args.nodelist:
        command.append(f"--nodelist={args.nodelist}")
    command.append(f"--wrap={shlex.join(['bash', str(script_path)])}")

    if args.render:
        print(f"$ {shlex.join(command)}")
        print(script, end="")
        return Submission(task, "", log_pattern)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    log = Path(str(log_pattern).replace("%j", job_id))
    print(f"Submitted {job_id}: {task.config}")
    print(f"Log: {log}")
    return Submission(task, job_id, log)


def follow(job_id: str, log: Path) -> None:
    tail: subprocess.Popen[str] | None = None
    try:
        while True:
            if tail is None and log.exists():
                tail = subprocess.Popen(["tail", "-n", "+1", "-F", str(log)])
            queued = subprocess.run(
                ["squeue", "-h", "-j", job_id], capture_output=True, text=True
            )
            if not queued.stdout.strip():
                return
            time.sleep(2)
    finally:
        if tail is not None:
            tail.terminate()


def slurm_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,Elapsed,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] in job_ids:
            states[fields[0]] = {
                "state": fields[1].split("+", 1)[0],
                "elapsed": fields[2],
                "exit_code": fields[3],
            }
    return states


def result_detail(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "invalid result.json"
    if data.get("eval_score_check"):
        check = data["eval_score_check"]
        return f"score={check['score']:g}, threshold={check['threshold']}"
    if data.get("perf_reference_check"):
        check = data["perf_reference_check"]
        return f"perf={'pass' if check['passed'] else 'fail'}"
    command_results = data.get("command_results", [])
    if command_results and command_results[-1].get("pytest_summary"):
        return str(command_results[-1]["pytest_summary"])
    return str(data.get("error", ""))


def write_report(
    submissions: list[Submission],
    states: dict[str, dict[str, str]],
    run_root: Path,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = [
        "## Slurm validation",
        "",
        "| Job | Type | Runner | Task | State | Elapsed | Result |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for submission in submissions:
        state = states.get(
            submission.job_id,
            {"state": "UNKNOWN", "elapsed": "", "exit_code": ""},
        )
        result_path = run_root / submission.job_id / "result.json"
        detail = result_detail(result_path)
        row = {
            "job_id": submission.job_id,
            "task": submission.task.__dict__,
            "log": str(submission.log),
            "result": str(result_path),
            **state,
            "detail": detail,
        }
        rows.append(row)
        summary.append(
            f"| {submission.job_id} | {submission.task.task_type} | "
            f"{submission.task.runner} | {submission.task.name} | "
            f"{state['state']} | {state['elapsed']} | {detail} |"
        )
        if submission.log.exists():
            shutil.copy2(submission.log, report_dir / f"{submission.job_id}.log")
        if result_path.exists():
            shutil.copy2(result_path, report_dir / f"{submission.job_id}-result.json")
    (report_dir / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    (report_dir / "summary.md").write_text("\n".join(summary) + "\n")


def wait_all(submissions: list[Submission], run_root: Path, report_dir: Path) -> bool:
    job_ids = [submission.job_id for submission in submissions]
    previous_handlers = {}

    def cancel_jobs(signum, _frame):
        subprocess.run(["scancel", *job_ids], check=False)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, cancel_jobs)
    try:
        while True:
            queued = subprocess.run(
                ["squeue", "-h", "-j", ",".join(job_ids)],
                check=True,
                capture_output=True,
                text=True,
            )
            if not queued.stdout.strip():
                break
            time.sleep(10)
        states = {}
        for _ in range(6):
            states = slurm_states(job_ids)
            if len(states) == len(job_ids):
                break
            time.sleep(2)
        write_report(submissions, states, run_root, report_dir)
        return all(item.get("state") == "COMPLETED" for item in states.values()) and (
            len(states) == len(job_ids)
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--all", action="store_true")
    parser.add_argument("--runner", action="append")
    parser.add_argument(
        "--type", dest="task_types", action="append", choices=sorted(TASK_TYPES)
    )
    parser.add_argument(
        "--match",
        action="append",
        help="Case-insensitive task/config/model substring; repeat for OR matching.",
    )
    parser.add_argument(
        "--exclude-match",
        action="append",
        help="Exclude a case-insensitive task/config/model substring; repeat for OR.",
    )
    parser.add_argument("--pr", help="PR number or GitHub pull request URL to merge.")
    parser.add_argument("--list", action="store_true", help="List matching tasks only.")
    parser.add_argument(
        "--trigger", choices=("per-commit", "manual", "nightly", "debug")
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--nodelist")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument(
        "--wait", action="store_true", help="Wait for every submitted job."
    )
    parser.add_argument(
        "--report-dir", help="Write aggregate manifest, summary, logs, and results."
    )
    parser.add_argument("--render", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        repo = Path(args.repo_root).resolve()
        artifact_root = Path(args.artifact_root).expanduser().resolve()
        cache = Path(args.cache_dir).expanduser().resolve()
        if args.pr:
            with pr_worktree(repo, args.pr) as checkout:
                return run(args, checkout, artifact_root, cache)
        return run(args, repo, artifact_root, cache)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace, repo: Path, artifact_root: Path, cache: Path) -> int:
    tasks = select_tasks(args, repo)
    if args.list:
        print_tasks(tasks)
        return 0
    commit = git(repo, "rev-parse", "HEAD")
    source = artifact_root / "snapshots" / f"{commit}.tar"
    if not args.render:
        source = snapshot(repo, artifact_root, commit)
        cache.mkdir(parents=True, exist_ok=True)
    submitted = [
        submit(
            task,
            render_script(
                task, source, artifact_root / "runs", cache, args.container_image
            ),
            artifact_root,
            args,
            commit,
        )
        for task in tasks
    ]
    if args.wait:
        if args.render:
            raise ValueError("--wait cannot be combined with --render")
        report_dir = (
            Path(args.report_dir).expanduser().resolve()
            if args.report_dir
            else artifact_root / "reports" / f"{commit[:12]}-{time.time_ns()}"
        )
        completed = wait_all(submitted, artifact_root / "runs", report_dir)
        print(f"Report: {report_dir}")
        return 0 if completed else 1
    if args.follow:
        for submission in submitted:
            if submission.job_id:
                follow(submission.job_id, submission.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
