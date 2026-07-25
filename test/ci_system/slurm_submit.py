#!/usr/bin/env python3
"""Submit existing TokenSpeed eval/perf YAML tasks to Slurm through Pyxis."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pipeline import build_matrix, normalize_task

GPU_RE = re.compile(r"(?:^|-)([1-9]\d*)gpu(?:-|$)")
TASK_TYPES = {"eval", "perf"}


@dataclass(frozen=True)
class Task:
    config: str
    name: str
    runner: str
    gpus: int


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
        raise ValueError("only eval and perf YAML tasks can run through Slurm")
    labels = list(data["runner"]["labels"])
    if runner is None:
        if len(labels) != 1:
            raise ValueError("task has multiple runners; pass --runner")
        runner = labels[0]
    if runner not in labels:
        raise ValueError(f"{runner!r} is not declared by {relative}")
    return Task(relative, str(data["name"]), runner, gpu_count(runner))


def select_tasks(args: argparse.Namespace, repo: Path) -> list[Task]:
    if args.config:
        return [load_task(repo, args.config, args.runner)]
    if not args.runner:
        raise ValueError("--all requires --runner")
    matrix = build_matrix(repo / "test/ci", repo, args.trigger)
    tasks = [
        load_task(repo, item["config"], item["runner"])
        for item in matrix["include"]
        if item["type"] in TASK_TYPES and item["runner"] == args.runner
    ]
    if not tasks:
        raise ValueError(f"no eval/perf tasks declare runner {args.runner!r}")
    return tasks


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
) -> tuple[str, Path]:
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
        return "", log_pattern
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    log = Path(str(log_pattern).replace("%j", job_id))
    print(f"Submitted {job_id}: {task.config}")
    print(f"Log: {log}")
    return job_id, log


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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--all", action="store_true")
    parser.add_argument("--runner")
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
    parser.add_argument("--render", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        repo = Path(args.repo_root).resolve()
        artifact_root = Path(args.artifact_root).expanduser().resolve()
        cache = Path(args.cache_dir).expanduser().resolve()
        tasks = select_tasks(args, repo)
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
        if args.follow:
            for job_id, log in submitted:
                if job_id:
                    follow(job_id, log)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
