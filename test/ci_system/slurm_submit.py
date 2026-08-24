#!/usr/bin/env python3
"""Select and submit existing TokenSpeed CI YAML tasks through Slurm/Pyxis."""

from __future__ import annotations

import argparse
import contextlib
import html
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
from typing import Iterable, TextIO

from pipeline import build_matrix, normalize_task

GPU_RE = re.compile(r"(?:^|-)([1-9]\d*)gpu(?:-|$)")
TASK_TYPES = {"ut", "server_smoke", "eval", "perf"}
DEFAULT_TASK_TYPES = {"eval", "perf"}
PR_RE = re.compile(
    r"^(?:https://github\.com/"
    r"(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+)/pull/)?"
    r"(?P<number>\d+)/?$"
)
REPOSITORY_RE = re.compile(r"^(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+)$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)


@dataclass(frozen=True)
class Task:
    config: str
    name: str
    task_type: str
    runner: str
    gpus: int
    nodes: int = 1
    client: str = "compute"


@dataclass(frozen=True)
class Submission:
    task: Task
    job_id: str
    log: Path


@dataclass
class CoordinatorEval:
    submission: Submission
    process: subprocess.Popen[str]
    output: TextIO
    done: Path


def gpu_count(runner: str) -> int:
    matches = GPU_RE.findall(runner)
    if len(matches) != 1:
        raise ValueError(f"runner {runner!r} must contain one '<N>gpu' segment")
    return int(matches[0])


def load_task(
    repo: Path,
    config: str,
    runner: str | None = None,
) -> Task:
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
    gpus = gpu_count(runner)
    slurm = data.get("slurm", {})
    nodes = int(slurm.get("nodes", 1))
    gpus_per_node = int(slurm.get("gpus_per_node", gpus))
    client = str(slurm.get("client", "compute"))
    if gpus_per_node != gpus:
        raise ValueError(
            f"{relative}: slurm.gpus_per_node={gpus_per_node} does not match "
            f"runner {runner!r} ({gpus} GPUs)"
        )
    return Task(
        relative,
        str(data["name"]),
        str(data["type"]),
        runner,
        gpus,
        nodes,
        client,
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
    matrix = build_matrix(repo / "test/ci", repo, args.trigger, "all", None, "all")
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
    return int(match.group("number"))


def source_pr_url(value: str) -> str | None:
    match = PR_RE.fullmatch(value)
    if not match:
        raise ValueError(
            "--pr must be a pull request number or GitHub pull request URL"
        )
    owner = match.group("owner")
    repo = match.group("repo")
    if owner is None or repo is None:
        repository_match = REPOSITORY_RE.fullmatch(
            os.environ.get("GITHUB_REPOSITORY", "")
        )
        if repository_match:
            owner = repository_match.group("owner")
            repo = repository_match.group("repo")
    if owner is not None and repo is not None:
        return f"https://github.com/{owner}/{repo}/pull/{match.group('number')}"
    return None


def source_pr_summary(value: str) -> str:
    number = parse_pr_number(value)
    url = source_pr_url(value)
    if url is not None:
        return f"**Target PR:** [#{number}]({url})"
    return f"**Target PR:** #{number}"


def print_target(repo: Path, source_pr: str | None, test_commit: str) -> None:
    if source_pr is None:
        print("Target: latest main", flush=True)
        print(f"Target commit: {test_commit}", flush=True)
        return

    number = parse_pr_number(source_pr)
    print(f"Target: PR #{number}", flush=True)
    url = source_pr_url(source_pr)
    if url is not None:
        print(f"Link: {url}", flush=True)
    try:
        target_commit = git(repo, "rev-parse", "HEAD^2")
        base_commit = git(repo, "rev-parse", "HEAD^1")
    except subprocess.CalledProcessError:
        print(f"Target commit: {test_commit}", flush=True)
        return
    print(f"Target commit: {target_commit}", flush=True)
    print(f"Merged test commit: {test_commit}", flush=True)
    print(f"Base commit: {base_commit}", flush=True)


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
    print("TYPE\tRUNNER\tNODES\tGPUS/NODE\tCLIENT\tCONFIG\tNAME")
    for task in tasks:
        print(
            f"{task.task_type}\t{task.runner}\t{task.nodes}\t{task.gpus}\t"
            f"{task.client}\t"
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
    handle, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{commit}.", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
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
    container_args = [
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
        "HUGGING_FACE_HUB_TOKEN,HF_HOME,XDG_CACHE_HOME,"
        "SLURM_STEP_ID,SLURM_STEP_NUM_NODES,SLURM_STEP_NODELIST,SLURM_NODEID,"
        "SLURM_PROCID,SLURM_LOCALID",
    ]
    gpu_device_mounts = ""
    local_model_mounts = ""
    if task.runner.startswith(("gb300-", "slurm-gb300-")):
        local_model_mounts = r"""
# GB300 nodes keep large model snapshots on their local RAID.  Keep the
# source configurable for other coordinators while exposing one stable path
# to server containers.
local_model_root="${TS_CI_LOCAL_MODEL_ROOT:-/scratch/${USER}-models}"
if [ -d "$local_model_root" ]; then
  model_mounts+=("$local_model_root:/models:ro")
fi
"""
        gpu_device_mounts = r"""
# The GB300 Pyxis hook does not expose allocated device nodes.
gpu_ids="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
IFS=',' read -ra visible_gpus <<< "$gpu_ids"
mounted_gpu_count=0
for token in "${visible_gpus[@]}"; do
  if [[ "$token" =~ ^[0-9]+$ ]]; then
    first_gpu=$((10#$token))
    last_gpu=$first_gpu
  elif [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    first_gpu=$((10#${BASH_REMATCH[1]}))
    last_gpu=$((10#${BASH_REMATCH[2]}))
  else
    continue
  fi
  for ((gpu = first_gpu; gpu <= last_gpu; gpu++)); do
    device="/dev/nvidia$gpu"
    if [ -e "$device" ]; then
      gpu_mounts+=("$device:$device")
      mounted_gpu_count=$((mounted_gpu_count + 1))
    fi
  done
done
if ((mounted_gpu_count == 0)); then
  echo "No NVIDIA device nodes matched Slurm GPU allocation: ${gpu_ids:-<unset>}" >&2
  exit 2
fi
for device in \
  /dev/nvidiactl \
  /dev/nvidia-uvm \
  /dev/nvidia-uvm-tools \
  /dev/nvidia-nvswitchctl \
  /dev/nvidia-caps \
  /dev/nvidia-caps-imex-channels; do
  [ ! -e "$device" ] || gpu_mounts+=("$device:$device")
done
"""
    common = f"""#!/usr/bin/env bash
set -euo pipefail

export RUNNER_NAME="slurm-${{SLURM_JOB_ID}}"
export HF_HOME=/home/runner/.cache/huggingface
export XDG_CACHE_HOME=/home/runner/.cache
unset GITHUB_STEP_SUMMARY GITHUB_OUTPUT GITHUB_ENV GITHUB_PATH GITHUB_STATE \
  GITHUB_EVENT_PATH

scratch={"/tmp" if task.nodes > 1 else '"${SLURM_TMPDIR:-/tmp}"'}/tokenspeed-${{SLURM_JOB_ID}}
src="$scratch/src"
tmp="$scratch/tmp"
run={shlex.quote(str(run_root))}/"${{SLURM_JOB_ID}}"
source_archive={shlex.quote(str(source))}
mkdir -p "$run"

mounts=(
  "$run:/workspace/.ci-artifacts"
  {shlex.quote(str(cache) + ":/home/runner/.cache")}
)
model_mounts=()
gpu_mounts=()

{local_model_mounts}
{gpu_device_mounts}

# The cluster Pyxis hooks omit driver libraries and tools.
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
"""
    if task.nodes == 1:
        srun = [
            "--nodes=1",
            "--ntasks=1",
            f"--gres=gpu:{task.gpus}",
            *container_args,
        ]
        command = [
            "bash",
            "-c",
            bootstrap,
            "tokenspeed-pipeline",
            *pipeline,
            "--result-json=/workspace/.ci-artifacts/result.json",
        ]
        return common + f"""
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$src/.ci-artifacts" "$tmp"
tar -xf "$source_archive" -C "$src"
mounts=("$src:/workspace" "$tmp:/tmp" "${{gpu_mounts[@]}}" "${{model_mounts[@]}}" "${{mounts[@]}}")
container_mounts="$(IFS=,; printf '%s' "${{mounts[*]}}")"
{shell_array("srun_args", srun)}
srun_args+=(--container-mounts="$container_mounts")
{shell_array("container_command", command)}
srun "${{srun_args[@]}}" "${{container_command[@]}}"
"""

    prepare_args = [
        "--overlap",
        f"--nodes={task.nodes}",
        f"--ntasks={task.nodes}",
        "--ntasks-per-node=1",
        "--gres=none",
        "--unbuffered",
        "--kill-on-bad-exit=1",
    ]
    client_prepare_args = [
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--gres=none",
        "--unbuffered",
        "--kill-on-bad-exit=1",
    ]
    image_prepare_args = [
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--gres=none",
        "--unbuffered",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        f"--container-image={image}",
        "--no-container-entrypoint",
        "--no-container-mount-home",
        "--container-remap-root",
    ]
    server_srun = [
        "--overlap",
        "--label",
        f"--nodes={task.nodes}",
        f"--ntasks={task.nodes}",
        "--ntasks-per-node=1",
        f"--gres=gpu:{task.gpus}",
        *container_args,
    ]
    client_srun = [
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--gres=none",
        *container_args,
    ]
    cleanup_args = [
        "--overlap",
        f"--nodes={task.nodes}",
        f"--ntasks={task.nodes}",
        "--ntasks-per-node=1",
        "--gres=none",
    ]
    server_command = [
        "bash",
        "-c",
        bootstrap,
        "tokenspeed-server",
        *pipeline,
        "--serve-only",
    ]
    if task.client == "coordinator":
        return common + f"""
{shell_array("prepare_args", prepare_args)}
{shell_array("image_prepare_args", image_prepare_args)}
{shell_array("cleanup_args", cleanup_args)}

server_src="$scratch/server-src"
server_tmp="$scratch/server-tmp"
server_mounts=("$server_src:/workspace" "$server_tmp:/tmp" "${{gpu_mounts[@]}}" "${{model_mounts[@]}}" "${{mounts[@]}}")
server_container_mounts="$(IFS=,; printf '%s' "${{server_mounts[*]}}")"
head_node="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
image_prepare_args+=(--nodelist="$head_node")

server_step_pid=""
cleanup() {{
  status=$?
  trap - EXIT INT TERM
  if [ -n "$server_step_pid" ] && kill -0 "$server_step_pid" 2>/dev/null; then
    kill -TERM "$server_step_pid" 2>/dev/null || true
  fi
  [ -z "$server_step_pid" ] || wait "$server_step_pid" 2>/dev/null || true
  srun "${{cleanup_args[@]}}" bash -c 'rm -rf -- "$1"' tokenspeed-cleanup "$scratch" || true
  exit "$status"
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

srun "${{image_prepare_args[@]}}" true
srun "${{prepare_args[@]}}" \
  bash -c 'set -euo pipefail; mkdir -p "$1/.ci-artifacts" "$2"; tar -xf "$3" -C "$1"' \
  tokenspeed-prepare "$server_src" "$server_tmp" "$source_archive"

{shell_array("server_srun_args", server_srun)}
server_srun_args+=(--container-mounts="$server_container_mounts")
{shell_array("server_command", server_command)}
srun "${{server_srun_args[@]}}" "${{server_command[@]}}" &
server_step_pid=$!

printf '%s\\n' "$head_node" > "$run/server-host.tmp"
mv "$run/server-host.tmp" "$run/server-host"

while kill -0 "$server_step_pid" 2>/dev/null && [ ! -e "$run/coordinator.done" ]; do
  sleep 2
done

if [ ! -e "$run/coordinator.done" ]; then
  set +e
  wait "$server_step_pid"
  status=$?
  set -e
  [ "$status" -ne 0 ] || status=1
  echo "multi-node server step exited before coordinator evaluation" >&2
  exit "$status"
fi

python3 - "$run/result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
raise SystemExit(0 if result.get("ok") else 1)
PY
"""
    client_command = [
        "bash",
        "-c",
        bootstrap,
        "tokenspeed-client",
        *pipeline,
        "--external-server",
        "--result-json=/workspace/.ci-artifacts/result.json",
    ]
    return common + f"""
{shell_array("prepare_args", prepare_args)}
{shell_array("client_prepare_args", client_prepare_args)}
{shell_array("image_prepare_args", image_prepare_args)}
{shell_array("cleanup_args", cleanup_args)}

server_src="$scratch/server-src"
client_src="$scratch/client-src"
server_tmp="$scratch/server-tmp"
client_tmp="$scratch/client-tmp"
# Full-node server allocations use the same GPU device IDs on every node.
server_mounts=("$server_src:/workspace" "$server_tmp:/tmp" "${{gpu_mounts[@]}}" "${{model_mounts[@]}}" "${{mounts[@]}}")
client_mounts=("$client_src:/workspace" "$client_tmp:/tmp" "${{mounts[@]}}")
server_container_mounts="$(IFS=,; printf '%s' "${{server_mounts[*]}}")"
client_container_mounts="$(IFS=,; printf '%s' "${{client_mounts[*]}}")"
head_node="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
client_prepare_args+=(--nodelist="$head_node")
image_prepare_args+=(--nodelist="$head_node")

server_step_pid=""
client_step_pid=""
cleanup() {{
  status=$?
  trap - EXIT INT TERM
  for pid in "$client_step_pid" "$server_step_pid"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  [ -z "$client_step_pid" ] || wait "$client_step_pid" 2>/dev/null || true
  [ -z "$server_step_pid" ] || wait "$server_step_pid" 2>/dev/null || true
  srun "${{cleanup_args[@]}}" bash -c 'rm -rf -- "$1"' tokenspeed-cleanup "$scratch" || true
  exit "$status"
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Import a new image digest once before every node opens the shared Enroot
# cache. Concurrent first-use imports race on the cache's final rename.
srun "${{image_prepare_args[@]}}" true

srun "${{prepare_args[@]}}" \
  bash -c 'set -euo pipefail; mkdir -p "$1/.ci-artifacts" "$2"; tar -xf "$3" -C "$1"' \
  tokenspeed-prepare "$server_src" "$server_tmp" "$source_archive"
srun "${{client_prepare_args[@]}}" \
  bash -c 'set -euo pipefail; mkdir -p "$1/.ci-artifacts" "$2"; tar -xf "$3" -C "$1"' \
  tokenspeed-client-prepare "$client_src" "$client_tmp" "$source_archive"

{shell_array("server_srun_args", server_srun)}
server_srun_args+=(--container-mounts="$server_container_mounts")
{shell_array("server_command", server_command)}
srun "${{server_srun_args[@]}}" "${{server_command[@]}}" &
server_step_pid=$!

{shell_array("client_srun_args", client_srun)}
client_srun_args+=(--nodelist="$head_node")
client_srun_args+=(--container-mounts="$client_container_mounts")
{shell_array("client_command", client_command)}
srun "${{client_srun_args[@]}}" "${{client_command[@]}}" &
client_step_pid=$!

while kill -0 "$server_step_pid" 2>/dev/null && \
      kill -0 "$client_step_pid" 2>/dev/null; do
  sleep 2
done

set +e
if ! kill -0 "$server_step_pid" 2>/dev/null; then
  wait "$server_step_pid"
  status=$?
  [ "$status" -ne 0 ] || status=1
  echo "multi-node server step exited before the client step" >&2
else
  wait "$client_step_pid"
  status=$?
fi
set -e
exit "$status"
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
        f"--nodes={task.nodes}",
        f"--ntasks={task.nodes}",
        "--ntasks-per-node=1",
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
    print(f"Submitted {job_id}: {task.config}", flush=True)
    print(f"Log: {log}", flush=True)
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


def scontrol_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    states = {}
    for job_id in job_ids:
        result = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            continue

        def field(name: str) -> str | None:
            match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", result.stdout)
            return match.group(1) if match else None

        observed_id = field("JobId")
        raw_state = field("JobState")
        elapsed = field("RunTime")
        exit_code = field("ExitCode")
        if None in {observed_id, raw_state, elapsed, exit_code}:
            continue
        if observed_id.split("+", 1)[0] != job_id:
            continue
        state = raw_state.split("+", 1)[0].upper()
        if state not in TERMINAL_STATES:
            continue
        states[job_id] = {
            "state": state,
            "elapsed": elapsed,
            "exit_code": exit_code,
        }
    return states


def queued_states(job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Return live state for only the explicitly submitted Slurm jobs."""
    requested = set(job_ids)
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            f"--jobs={','.join(job_ids)}",
            "--format=%i|%T|%M|%R",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|", 3)
        if len(fields) != 4 or fields[0] not in requested:
            continue
        job_id, state, elapsed, reason = fields
        states[job_id] = {
            "state": state.upper(),
            "elapsed": elapsed,
            "exit_code": "",
            "reason": reason if state.upper() == "PENDING" else "",
        }
    return states


def print_progress(
    submissions: list[Submission],
    states: dict[str, dict[str, str]],
) -> None:
    print(f"\nSlurm progress ({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())})")
    for submission in submissions:
        state = states.get(submission.job_id) or {
            "state": "UNKNOWN",
            "elapsed": "",
            "reason": "",
        }
        status = {
            "COMPLETED": "PASS",
            "FAILED": "FAIL",
            "RUNNING": "RUN ",
            "PENDING": "WAIT",
        }.get(state["state"], state["state"])
        suffix = f" ({state['reason']})" if state.get("reason") else ""
        print(
            f"[{status:4}] {submission.job_id:>8} "
            f"{submission.task.runner:<12} {submission.task.name} "
            f"{state.get('elapsed', '')}{suffix}",
        )
    print(flush=True)


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
    for command_result in reversed(command_results):
        if not isinstance(command_result, dict):
            continue
        if command_result.get("stage") != "eval":
            continue
        for key in ("evalscope_score", "inspect_score"):
            score = command_result.get(key)
            if score is not None:
                try:
                    return f"score={float(score):g}"
                except (TypeError, ValueError):
                    continue
    if command_results and command_results[-1].get("pytest_summary"):
        return str(command_results[-1]["pytest_summary"])
    return str(data.get("error", ""))


def clean_report_text(value: object) -> str:
    return CONTROL_RE.sub("", ANSI_ESCAPE_RE.sub("", str(value)))


def table_cell(value: object, limit: int = 120) -> str:
    text = clean_report_text(value)
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if len(line) > limit:
        line = line[: limit - 3].rstrip() + "..."
    return line.replace("\\", "\\\\").replace("|", "\\|")


def detail_block(job_id: str, task_name: str, detail: str) -> list[str]:
    text = clean_report_text(detail)
    lines = text.splitlines()[:40]
    body = "\n".join(lines)
    truncated = len(text.splitlines()) > 40 or len(body) > 4000
    if len(body) > 4000:
        body = body[:4000].rstrip()
    if truncated:
        body += f"\n[truncated; see {job_id}.log in the artifact]"
    longest_tilde_run = max((len(run) for run in re.findall(r"~+", body)), default=0)
    fence = "~" * max(3, longest_tilde_run + 1)
    title = html.escape(f"Job {job_id} — {task_name}")
    return [
        "<details>",
        f"<summary>{title}</summary>",
        "",
        f"{fence}text",
        body,
        fence,
        "</details>",
        "",
    ]


def write_report(
    submissions: list[Submission],
    states: dict[str, dict[str, str]],
    run_root: Path,
    report_dir: Path,
    source_pr: str | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = [
        "## Slurm validation",
        "",
    ]
    if source_pr:
        summary.extend([source_pr_summary(source_pr), ""])
    summary.extend(
        [
            "| Job | Type | Runner | Task | State | Elapsed | Result |",
            "|---:|---|---|---|---|---:|---|",
        ]
    )
    details = []
    for submission in submissions:
        state = states.get(
            submission.job_id,
            {"state": "UNKNOWN", "elapsed": "", "exit_code": ""},
        )
        result_path = run_root / submission.job_id / "result.json"
        detail = result_detail(result_path)
        row = {
            "job_id": submission.job_id,
            "task": {
                "config": submission.task.config,
                "name": submission.task.name,
                "task_type": submission.task.task_type,
                "runner": submission.task.runner,
                "gpus": submission.task.gpus,
                "nodes": submission.task.nodes,
            },
            "log": str(submission.log),
            "result": str(result_path),
            **state,
            "detail": detail,
        }
        rows.append(row)
        display_state = {
            "COMPLETED": "✅",
            "FAILED": "❌",
        }.get(state["state"], state["state"])
        cells = [
            submission.job_id,
            submission.task.task_type,
            submission.task.runner,
            submission.task.name,
            display_state,
            state["elapsed"],
            detail,
        ]
        summary.append("| " + " | ".join(table_cell(cell) for cell in cells) + " |")
        if table_cell(detail) != detail:
            details.extend(
                detail_block(submission.job_id, submission.task.name, detail)
            )
        if submission.log.exists():
            shutil.copy2(submission.log, report_dir / f"{submission.job_id}.log")
        if result_path.exists():
            shutil.copy2(result_path, report_dir / f"{submission.job_id}-result.json")
    (report_dir / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    if details:
        summary.extend(["", "## Details", "", *details])
    (report_dir / "summary.md").write_text("\n".join(summary) + "\n")


def check_coordinator_runtime() -> None:
    if shutil.which("docker") is None:
        raise ValueError("coordinator evaluation requires Docker on the dispatcher")
    daemon = subprocess.run(
        ["docker", "info"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if daemon.returncode:
        raise ValueError("coordinator evaluation requires a working Docker daemon")
    compose = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if compose.returncode:
        raise ValueError("coordinator evaluation requires Docker Compose")


def wait_for_server_host(submission: Submission, run_root: Path) -> str:
    host_path = run_root / submission.job_id / "server-host"
    deadline = time.monotonic() + 12 * 60 * 60
    last_update = 0.0
    while time.monotonic() < deadline:
        if host_path.exists():
            host = host_path.read_text().strip()
            if not re.fullmatch(r"[A-Za-z0-9._-]+", host):
                raise ValueError(
                    f"Slurm job {submission.job_id} wrote an invalid server host"
                )
            return host
        active = queued_states([submission.job_id])
        if submission.job_id not in active:
            state = slurm_states([submission.job_id]).get(submission.job_id)
            if state is None:
                state = scontrol_states([submission.job_id]).get(submission.job_id)
            if state is not None:
                raise RuntimeError(
                    f"Slurm job {submission.job_id} reached {state['state']} "
                    "before publishing its server host"
                )
        now = time.monotonic()
        if now - last_update >= 60:
            state = active.get(submission.job_id, {})
            print(
                f"Waiting for coordinator endpoint from {submission.job_id}: "
                f"{state.get('state', 'UNKNOWN')}",
                flush=True,
            )
            last_update = now
        time.sleep(2)
    raise TimeoutError(
        f"timed out waiting for Slurm job {submission.job_id} server host"
    )


def start_coordinator_evals(
    submissions: list[Submission], repo: Path, run_root: Path
) -> list[CoordinatorEval]:
    evaluations = []
    for submission in submissions:
        if submission.task.client != "coordinator":
            continue
        host = wait_for_server_host(submission, run_root)
        run_dir = run_root / submission.job_id
        result = run_dir / "result.json"
        done = run_dir / "coordinator.done"
        env = os.environ.copy()
        env.update(
            {
                "SLURM_JOB_ID": f"coordinator-{submission.job_id}",
                "TS_CI_SERVER_HOST": host,
                "TS_CI_SERVER_BASE_URL": f"http://{host}:8000/v1",
                "TS_CI_SERVER_READY_URL": f"http://{host}:8000/readiness",
            }
        )
        command = [
            sys.executable,
            str(repo / "test/ci_system/pipeline.py"),
            "execute",
            f"--config={submission.task.config}",
            f"--runner={submission.task.runner}",
            f"--work-dir={repo}",
            "--setup-mode=slurm",
            "--external-server",
            "--skip-stage=install",
            "--print-plan",
            f"--result-json={result}",
        ]
        output = submission.log.open("a", encoding="utf-8")
        print(
            f"Starting coordinator evaluation for {submission.job_id} "
            f"against {host}:8000",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        evaluations.append(CoordinatorEval(submission, process, output, done))
    return evaluations


def finish_coordinator_eval(evaluation: CoordinatorEval, run_root: Path) -> None:
    if evaluation.done.exists():
        return
    returncode = evaluation.process.poll()
    if returncode is None:
        return
    evaluation.output.close()
    result = run_root / evaluation.submission.job_id / "result.json"
    if not result.exists():
        result.write_text(
            json.dumps(
                {
                    "ok": False,
                    "task": evaluation.submission.task.name,
                    "type": evaluation.submission.task.task_type,
                    "runner": evaluation.submission.task.runner,
                    "error": f"coordinator pipeline exited with status {returncode}",
                },
                indent=2,
            )
            + "\n"
        )
    evaluation.done.touch()


def stop_coordinator_evals(evaluations: list[CoordinatorEval], run_root: Path) -> None:
    for evaluation in evaluations:
        if evaluation.process.poll() is None:
            try:
                os.killpg(evaluation.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for evaluation in evaluations:
        if evaluation.process.poll() is None:
            try:
                evaluation.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(evaluation.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                evaluation.process.wait(timeout=5)
        finish_coordinator_eval(evaluation, run_root)


def wait_all(
    submissions: list[Submission],
    run_root: Path,
    report_dir: Path,
    source_pr: str | None = None,
    coordinator_evals: list[CoordinatorEval] | None = None,
) -> bool:
    job_ids = [submission.job_id for submission in submissions]
    coordinator_evals = coordinator_evals or []
    previous_handlers = {}

    def cancel_jobs(signum, _frame):
        stop_coordinator_evals(coordinator_evals, run_root)
        subprocess.run(["scancel", *job_ids], check=False)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, cancel_jobs)
    try:
        previous_snapshot = None
        last_update = 0.0
        while True:
            for evaluation in coordinator_evals:
                finish_coordinator_eval(evaluation, run_root)
            active = queued_states(job_ids)
            accounting = slurm_states(job_ids)
            states = {
                job_id: active.get(job_id, accounting.get(job_id, {}))
                for job_id in job_ids
            }
            snapshot = tuple(
                (
                    job_id,
                    state.get("state", "UNKNOWN"),
                    state.get("reason", ""),
                )
                for job_id, state in states.items()
            )
            now = time.monotonic()
            if snapshot != previous_snapshot or now - last_update >= 60:
                print_progress(submissions, states)
                previous_snapshot = snapshot
                last_update = now
            if not active:
                break
            time.sleep(10)
        stop_coordinator_evals(coordinator_evals, run_root)
        states = {}
        for _ in range(6):
            states = slurm_states(job_ids)
            missing = [job_id for job_id in job_ids if job_id not in states]
            states.update(scontrol_states(missing))
            if len(states) == len(job_ids):
                break
            time.sleep(2)
        write_report(submissions, states, run_root, report_dir, source_pr)
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
    pull_request = parser.add_mutually_exclusive_group()
    pull_request.add_argument(
        "--pr", help="PR number or GitHub pull request URL to merge."
    )
    pull_request.add_argument(
        "--source-pr",
        help="PR number or GitHub pull request URL for the current checkout.",
    )
    parser.add_argument("--list", action="store_true", help="List matching tasks only.")
    parser.add_argument(
        "--trigger", choices=("per-commit", "manual", "nightly", "debug", "slurm")
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
    coordinator_tasks = [task for task in tasks if task.client == "coordinator"]
    if coordinator_tasks and not args.render:
        if not args.wait:
            raise ValueError("coordinator evaluation requires --wait")
        if len(coordinator_tasks) > 1:
            raise ValueError("submit one coordinator evaluation at a time")
        check_coordinator_runtime()
    commit = git(repo, "rev-parse", "HEAD")
    source_pr = args.pr or args.source_pr
    print_target(repo, source_pr, commit)
    print(f"Selected tasks: {len(tasks)}", flush=True)
    for task in tasks:
        print(
            f"- {task.runner:<12} {task.task_type:<12} {task.name} " f"({task.config})",
            flush=True,
        )
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
        try:
            coordinator_evals = start_coordinator_evals(
                submitted, repo, artifact_root / "runs"
            )
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
            subprocess.run(
                ["scancel", *[item.job_id for item in submitted]], check=False
            )
            raise
        completed = wait_all(
            submitted,
            artifact_root / "runs",
            report_dir,
            source_pr=source_pr,
            coordinator_evals=coordinator_evals,
        )
        print(f"Report: {report_dir}", flush=True)
        return 0 if completed else 1
    if args.follow:
        for submission in submitted:
            if submission.job_id:
                follow(submission.job_id, submission.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
