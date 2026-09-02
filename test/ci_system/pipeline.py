#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import glob
import io
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from process_group_manager import ProcessGroupManager, make_manager

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run test/ci_system/pipeline.py") from exc


SUPPORTED_TYPES = {"ut", "server_smoke", "eval", "perf"}
SUPPORTED_TRIGGERS = {"per-commit", "manual", "nightly", "debug", "slurm"}
WORKFLOW_STAGE_TYPES = {
    "unit-test": {"ut", "server_smoke"},
    "model-test": {"eval", "perf"},
}
SUPPORTED_WORKFLOW_STAGES = tuple(WORKFLOW_STAGE_TYPES)
SUPPORTED_SETUP_MODES = ("ci", "slurm")
# Lower sort key = dispatched earlier. GitHub Actions starts matrix jobs in
# include-list order, so `high` entries reach runner pools first when several
# jobs in the same workflow stage contend for the same hardware.
SUPPORTED_PRIORITIES = ("high", "normal", "low")
DEFAULT_PRIORITY = "normal"
SUPPORTED_RUNNER_GROUPS = ("all", "amd", "nvidia", "nvidia-arm", "nvidia-x86")
SUPPORTED_MULTI_NODE_FILTERS = ("all", "only", "exclude")
_PRIORITY_ORDER = {value: index for index, value in enumerate(SUPPORTED_PRIORITIES)}
B200_RUNNER_LABEL_ENV = "TOKENSPEED_B200_RUNNER_LABEL"
EXCLUDED_RUNNER_LABELS_ENV = "TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS"
STALE_PROCESS_PATTERNS = [
    r"ts serve",
    r"python.*-m\s+smg(\s|\.launch|$)",
    # smg's launch_router rewrites its cmdline to `smg::router` via
    # setproctitle, so the python pattern above stops matching once
    # the router is fully up.
    r"smg::",
    r"smg_grpc_servicer\.tokenspeed",
    r"run_ci_suite",
]
JIT_CACHE_ENV_SUBDIRS = {
    "TRITON_CACHE_DIR": "triton",
    "CUTE_DSL_CACHE_DIR": "cute_dsl",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}
RUNNER_SM_PREFIXES = (
    (("h100", "h200"), "sm90"),
    (("b200", "gb200", "slurm-gb200"), "sm100"),
    (("b300", "gb300", "slurm-b300", "slurm-gb300"), "sm103"),
)

AMD_RUNNER_PREFIXES = ("amd-mi35x-", "amd-mi355-", "amd-mi350-", "amd-mi450-")
NVIDIA_ARM_RUNNER_PREFIXES = (
    "gb200",
    "gb300",
    "slurm-b300",
    "slurm-gb200",
    "slurm-gb300",
)
GB200_RUNNER_PREFIXES = ("gb200", "slurm-gb200")
ISOLATED_JIT_CACHE_RUNNER_PREFIXES = (
    *GB200_RUNNER_PREFIXES,
    "slurm-b300",
    "slurm-gb300",
)
NVIDIA_GPU_CLEANUP_RUNNER_PREFIXES = ("gb200", "b300")
PERF_DIAGNOSTIC_RUNNERS = ("b300-4gpu",)


def is_amd_runner(runner: str) -> bool:
    return runner.startswith(AMD_RUNNER_PREFIXES)


def is_nvidia_arm_runner(runner: str) -> bool:
    return runner.startswith(NVIDIA_ARM_RUNNER_PREFIXES)


def runner_matches_group(runner: str, runner_group: str) -> bool:
    if runner_group == "all":
        return True
    if runner_group == "amd":
        return is_amd_runner(runner)
    if runner_group == "nvidia":
        return not is_amd_runner(runner)
    if runner_group == "nvidia-arm":
        return is_nvidia_arm_runner(runner)
    if runner_group == "nvidia-x86":
        return not is_amd_runner(runner) and not is_nvidia_arm_runner(runner)
    raise ValueError(f"unsupported runner group: {runner_group!r}")


def is_gb200_runner(runner: str) -> bool:
    return runner.startswith(GB200_RUNNER_PREFIXES)


def uses_isolated_jit_cache(runner: str) -> bool:
    return runner.startswith(ISOLATED_JIT_CACHE_RUNNER_PREFIXES)


def should_run_nvidia_gpu_cleanup(runner: str) -> bool:
    return runner.startswith(NVIDIA_GPU_CLEANUP_RUNNER_PREFIXES)


def should_run_perf_diagnostics(task: Dict[str, Any], runner: str) -> bool:
    return task["type"] == "perf" and runner in PERF_DIAGNOSTIC_RUNNERS


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML document must be a mapping")
    return data


def validate_task(data: Dict[str, Any], path: Path) -> None:
    required = {"api_version", "name", "type", "triggers", "runner"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    if data["api_version"] != "ci.tokenspeed.io/v1":
        raise ValueError(f"{path}: unsupported api_version {data['api_version']!r}")
    if data["type"] not in SUPPORTED_TYPES:
        raise ValueError(f"{path}: unsupported type {data['type']!r}")
    workflow_stage = data.get("workflow_stage")
    if workflow_stage is not None:
        if workflow_stage not in WORKFLOW_STAGE_TYPES:
            raise ValueError(f"{path}: unsupported workflow_stage {workflow_stage!r}")
        if data["type"] not in WORKFLOW_STAGE_TYPES[workflow_stage]:
            raise ValueError(
                f"{path}: type {data['type']!r} is incompatible with "
                f"workflow_stage {workflow_stage!r}"
            )
    triggers = data["triggers"]
    if not isinstance(triggers, list) or not triggers:
        raise ValueError(f"{path}: triggers must be a non-empty list")
    bad_triggers = [
        trigger for trigger in triggers if trigger not in SUPPORTED_TRIGGERS
    ]
    if bad_triggers:
        raise ValueError(f"{path}: unsupported triggers: {bad_triggers}")
    runner = data["runner"]
    if not isinstance(runner, dict):
        raise ValueError(f"{path}: runner must be a mapping")
    labels = runner.get("labels")
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label for label in labels)
    ):
        raise ValueError(f"{path}: runner.labels must be a non-empty string list")
    runner_env = runner.get("env", {})
    if runner_env:
        if not isinstance(runner_env, dict):
            raise ValueError(f"{path}: runner.env must be a mapping")
        unknown_labels = sorted(set(runner_env) - set(labels))
        if unknown_labels:
            raise ValueError(
                f"{path}: runner.env contains unknown labels: {unknown_labels}"
            )
        for label, env in runner_env.items():
            if not isinstance(env, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in env.items()
            ):
                raise ValueError(
                    f"{path}: runner.env[{label!r}] must be a string mapping"
                )
    if "priority" in data:
        priority = data["priority"]
        if isinstance(priority, str):
            if priority not in SUPPORTED_PRIORITIES:
                raise ValueError(
                    f"{path}: priority must be one of "
                    f"{sorted(SUPPORTED_PRIORITIES)}; got {priority!r}"
                )
        elif isinstance(priority, dict):
            unknown_labels = sorted(set(priority) - set(labels))
            if unknown_labels:
                raise ValueError(
                    f"{path}: priority contains unknown labels: {unknown_labels}"
                )
            bad_values = sorted(
                {
                    value
                    for value in priority.values()
                    if value not in SUPPORTED_PRIORITIES
                }
            )
            if bad_values:
                raise ValueError(
                    f"{path}: priority values must each be one of "
                    f"{sorted(SUPPORTED_PRIORITIES)}; got {bad_values}"
                )
        else:
            raise ValueError(
                f"{path}: priority must be a string or a per-label mapping; "
                f"got {type(priority).__name__}"
            )
    if "optional" in data:
        optional = data["optional"]
        if isinstance(optional, bool):
            pass
        elif isinstance(optional, dict):
            unknown_labels = sorted(set(optional) - set(labels))
            if unknown_labels:
                raise ValueError(
                    f"{path}: optional contains unknown labels: {unknown_labels}"
                )
            bad_values = sorted(
                label
                for label, value in optional.items()
                if not isinstance(value, bool)
            )
            if bad_values:
                raise ValueError(
                    f"{path}: optional values must be booleans for labels: "
                    f"{bad_values}"
                )
        else:
            raise ValueError(
                f"{path}: optional must be a boolean or a per-label mapping; "
                f"got {type(optional).__name__}"
            )
    if "retries" in data:
        retries = data["retries"]
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError(f"{path}: retries must be a non-negative integer")
        if retries > 0 and data["type"] not in {"eval", "perf"}:
            raise ValueError(
                f"{path}: retries is only supported for eval and perf tasks"
            )
    if "slurm" in data:
        slurm = data["slurm"]
        if not isinstance(slurm, dict):
            raise ValueError(f"{path}: slurm must be a mapping")
        unknown = sorted(set(slurm) - {"nodes", "gpus_per_node"})
        if unknown:
            raise ValueError(f"{path}: unsupported slurm keys: {unknown}")
        missing_slurm = sorted({"nodes", "gpus_per_node"} - set(slurm))
        if missing_slurm:
            raise ValueError(f"{path}: missing slurm keys: {missing_slurm}")
        for key in ("nodes", "gpus_per_node"):
            value = slurm[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{path}: slurm.{key} must be a positive integer")
        if slurm["nodes"] > 1:
            if data["type"] not in {"eval", "perf"}:
                raise ValueError(
                    f"{path}: multi-node Slurm tasks must have type 'eval' or 'perf'"
                )
            if len(data["triggers"]) != 1 or data["triggers"][0] not in {
                "nightly",
                "per-commit",
                "slurm",
            }:
                raise ValueError(
                    f"{path}: multi-node Slurm tasks must use exactly one of the "
                    "'nightly', 'per-commit', or 'slurm' triggers"
                )
            if not all(label.startswith("slurm-") for label in labels):
                raise ValueError(
                    f"{path}: multi-node Slurm runner labels must start with 'slurm-'"
                )
            server = data.get("server", {})
            if not isinstance(server, dict) or not server.get("command"):
                raise ValueError(
                    f"{path}: multi-node Slurm tasks require server.command"
                )
            ready = server.get("ready", {})
            if not isinstance(ready, dict) or not ready.get("url"):
                raise ValueError(
                    f"{path}: multi-node Slurm tasks require server.ready.url"
                )


def normalize_task(path: Path, repo_root: Path) -> Dict[str, Any]:
    data = load_yaml(path)
    validate_task(data, path)
    data["_source_path"] = path.relative_to(repo_root).as_posix()
    return data


def get_b200_runner_label_override() -> str:
    return os.environ.get(B200_RUNNER_LABEL_ENV, "").strip()


def get_excluded_runner_labels() -> List[str]:
    return [
        term.strip().lower()
        for term in os.environ.get(EXCLUDED_RUNNER_LABELS_ENV, "").split(",")
        if term.strip()
    ]


def runner_is_excluded(runner: str, excluded_labels: Iterable[str]) -> bool:
    normalized_runner = runner.lower()
    return any(label in normalized_runner for label in excluded_labels)


def resolve_runner_label(label: str) -> str:
    override = get_b200_runner_label_override()
    if not override:
        return label

    match = re.fullmatch(r"b200-(\d+gpu)", label)
    if not match:
        return label
    return f"{override.rstrip('-')}-{match.group(1)}"


def resolve_runner_labels(labels: Iterable[str]) -> List[str]:
    return [resolve_runner_label(label) for label in labels]


def find_task_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.yaml"))


def resolve_priority_for_label(priority: Any, label: str) -> str:
    """Pick the effective priority for ``label`` from the task's ``priority``.

    - Missing / ``None`` -> ``DEFAULT_PRIORITY``.
    - String -> applies to every label.
    - Mapping -> only the listed labels are overridden; everything else
      stays at ``DEFAULT_PRIORITY``. ``validate_task`` is the source of
      truth for accepted keys and values.
    """
    if priority is None:
        return DEFAULT_PRIORITY
    if isinstance(priority, str):
        return priority
    if isinstance(priority, dict):
        return priority.get(label, DEFAULT_PRIORITY)
    return DEFAULT_PRIORITY


def resolve_optional_for_label(optional: Any, label: str) -> bool:
    """Pick whether ``label`` is allowed to fail from task ``optional`` data."""
    if optional is None:
        return False
    if isinstance(optional, bool):
        return optional
    if isinstance(optional, dict):
        return optional.get(label, False)
    return False


def build_matrix(
    root: Path,
    repo_root: Path,
    trigger: str | None,
    runner_group: str = "all",
    workflow_stage: str | None = None,
    multi_node: str = "exclude",
) -> Dict[str, Any]:
    if multi_node not in SUPPORTED_MULTI_NODE_FILTERS:
        raise ValueError(f"unsupported multi-node filter: {multi_node!r}")
    include = []
    excluded_runner_labels = get_excluded_runner_labels()
    for path in find_task_files(root):
        task = normalize_task(path, repo_root)
        if trigger and trigger not in task["triggers"]:
            continue
        is_multi_node = task.get("slurm", {}).get("nodes", 1) > 1
        if multi_node == "only" and not is_multi_node:
            continue
        if multi_node == "exclude" and is_multi_node:
            continue
        task_workflow_stage = task.get("workflow_stage")
        if workflow_stage:
            if task_workflow_stage is None:
                raise ValueError(
                    f"{path}: workflow_stage is required when filtering by stage"
                )
            if workflow_stage != task_workflow_stage:
                continue
        priority = task.get("priority")
        optional = task.get("optional")
        for label in task["runner"]["labels"]:
            # `priority` keys are the labels as written in YAML, so look
            # up per-label metadata before `resolve_runner_label` rewrites
            # b200 to b200v2.
            effective = resolve_priority_for_label(priority, label)
            is_optional = resolve_optional_for_label(optional, label)
            runner = resolve_runner_label(label)
            if runner_is_excluded(runner, excluded_runner_labels):
                continue
            if not runner_matches_group(runner, runner_group):
                continue
            entry = {
                "name": task["name"],
                "type": task["type"],
                "config": task["_source_path"],
                "runner": runner,
                "priority": effective,
                "optional": is_optional,
            }
            if task_workflow_stage is not None:
                entry["workflow_stage"] = task_workflow_stage
            include.append(entry)
    # Stable sort: tasks at the same priority keep their file-path / label
    # order, so tasks that omit `priority` see no change from the previous
    # behaviour.
    include.sort(key=lambda entry: _PRIORITY_ORDER[entry["priority"]])
    return {"include": include}


def shell_run(
    command: str,
    *,
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
    check: bool = True,
) -> Dict[str, Any]:
    print(f"$ {command}", flush=True)
    if dry_run:
        return {"command": command, "returncode": 0, "output": "", "dry_run": True}
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="ignore",
    )
    output_lines: List[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    process.wait()
    completed = process
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command}"
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": "".join(output_lines),
    }


def merge_env(task_env: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in task_env.items()})
    return env


def get_default_runner_env(runner: str) -> Dict[str, str]:
    for prefixes, sm in RUNNER_SM_PREFIXES:
        if runner.startswith(prefixes):
            return {"SM": sm}
    return {}


def get_runner_specific_env(task: Dict[str, Any], runner: str) -> Dict[str, str]:
    runner_env = task["runner"].get("env", {})
    if runner in runner_env:
        return dict(runner_env[runner])

    for label in task["runner"]["labels"]:
        if resolve_runner_label(label) == runner:
            return dict(runner_env.get(label, {}))

    return {}


def validate_gb300_runner_alias(declared_runner: str, effective_runner: str) -> str:
    """Validate a B200/GB200-to-GB300 alias with an unchanged topology."""
    declared_slurm = declared_runner.startswith("slurm-")
    effective_slurm = effective_runner.startswith("slurm-")
    declared_base = declared_runner.removeprefix("slurm-")
    effective_base = effective_runner.removeprefix("slurm-")

    if declared_runner == effective_runner and effective_base.startswith("gb300-"):
        return effective_runner
    if declared_slurm != effective_slurm:
        raise ValueError("runner alias must preserve the slurm- prefix")

    declared_family = next(
        (
            family
            for family in ("b200", "gb200")
            if declared_base.startswith(f"{family}-")
        ),
        None,
    )
    if declared_family is None or not effective_base.startswith("gb300-"):
        raise ValueError(
            "runner alias must map [slurm-]b200-* or [slurm-]gb200-* "
            "to [slurm-]gb300-*"
        )

    declared_suffix = declared_base.removeprefix(f"{declared_family}-")
    effective_suffix = effective_base.removeprefix("gb300-")
    gpu_pattern = re.compile(r"(?:^|-)([1-9]\d*)gpu(?:-|$)")
    declared_gpus = gpu_pattern.findall(declared_suffix)
    effective_gpus = gpu_pattern.findall(effective_suffix)
    if len(declared_gpus) != 1 or declared_gpus != effective_gpus:
        raise ValueError("runner alias GPU counts must match")
    if declared_suffix != effective_suffix:
        raise ValueError("runner alias suffixes must match")
    return effective_runner


def apply_slurm_runner_override(
    declared_runner: str,
    runner_override: str | None,
    setup_mode: str,
) -> str:
    if runner_override is None:
        return declared_runner
    if setup_mode != "slurm":
        raise ValueError("--runner-override requires --setup-mode=slurm")
    try:
        return validate_gb300_runner_alias(declared_runner, runner_override)
    except ValueError as exc:
        raise ValueError(str(exc).replace("runner alias", "runner override")) from exc


def create_ci_venv_name(runner_name: str | None = None) -> str:
    if runner_name:
        # Fixed path per runner so flashinfer JIT cache (which embeds the
        # venv path in build.ninja) stays valid across CI runs.
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", runner_name)
        return f"/tmp/ci-env-{safe_name}"
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"/tmp/ci-env-{run_id}-{run_attempt}-{os.getpid()}"


def create_ci_jit_cache_root(runner_name: str | None = None) -> str:
    if runner_name:
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", runner_name)
        return f"/tmp/ci-jit-cache-{safe_name}"
    return "/tmp/ci-jit-cache-local"


def get_jit_cache_env(env: Dict[str, str]) -> Dict[str, str]:
    runner_name = (
        env.get("RUNNER_NAME") or env.get("CI_RUNNER_NAME") or env.get("HOSTNAME")
    )
    root = create_ci_jit_cache_root(runner_name)
    return {
        variable: os.path.join(root, subdir)
        for variable, subdir in JIT_CACHE_ENV_SUBDIRS.items()
        if not env.get(variable)
    }


def remove_stale_torch_extension_builds(env: Dict[str, str], dry_run: bool) -> None:
    cache_root = env.get("TORCH_EXTENSIONS_DIR")
    if not cache_root:
        return

    # Setup runs before new work on this runner, so any existing lock is stale.
    for lock_path in sorted(Path(cache_root).glob("*/lock")):
        extension_dir = lock_path.parent
        if dry_run:
            print(
                f"[dry-run] remove stale torch extension build: {extension_dir}",
                flush=True,
            )
            continue
        print(
            f"removing stale torch extension build: {extension_dir}",
            flush=True,
        )
        shutil.rmtree(extension_dir, ignore_errors=True)
        if lock_path.exists():
            raise RuntimeError(
                f"failed to remove stale torch extension build: {extension_dir}"
            )


def _pkill(
    pattern: str,
    signal_name: str,
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
) -> None:
    shell_run(
        f'pkill -{signal_name} -f "{pattern}" 2>/dev/null || true',
        env=env,
        cwd=cwd,
        dry_run=dry_run,
        check=False,
    )


def kill_stale_processes(
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
) -> None:
    for pattern in STALE_PROCESS_PATTERNS:
        _pkill(pattern, "TERM", env, cwd, dry_run)
    if not dry_run:
        time.sleep(5)
    for pattern in STALE_PROCESS_PATTERNS:
        _pkill(pattern, "KILL", env, cwd, dry_run)


def get_ready_port(ready: Dict[str, Any]) -> int | None:
    parsed = urlparse(str(ready["url"]))
    return parsed.port


def configure_slurm_server_command(command: str, timeout: int) -> str:
    """Keep the TokenSpeed engine deadline aligned with Slurm readiness."""
    tokens = shlex.split(command)
    if tokens[:2] != ["ts", "serve"] or "--engine-startup-timeout" in tokens:
        return command
    return f"{command} --engine-startup-timeout {timeout}"


def kill_port_listeners(
    port: int,
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
) -> None:
    command = (
        "if ! command -v lsof >/dev/null 2>&1; then "
        "sudo apt-get update -q && sudo apt-get install -y lsof; "
        "fi; "
        f"pids=$(lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null || true); "
        'if [ -n "$pids" ]; then '
        f'echo "Killing stale listener(s) on port {port}: $pids"; '
        "kill -TERM $pids 2>/dev/null || true; "
        "sleep 2; "
        f"pids=$(lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null || true); "
        'if [ -n "$pids" ]; then kill -KILL $pids 2>/dev/null || true; fi; '
        "fi"
    )
    shell_run(command, env=env, cwd=cwd, dry_run=dry_run)


def kill_ready_port_listener(
    ready: Dict[str, Any],
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
) -> None:
    port = get_ready_port(ready)
    if port is None:
        return
    kill_port_listeners(port, env, cwd, dry_run)


def ensure_ready_port_available(ready: Dict[str, Any], dry_run: bool) -> None:
    """Fail without mutating the host when a task's readiness port is occupied."""
    port = get_ready_port(ready)
    if port is None:
        return
    host = urlparse(str(ready["url"])).hostname or "127.0.0.1"
    if dry_run:
        print(f"[dry-run] check readiness port is available: {host}:{port}", flush=True)
        return

    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        return
    raise RuntimeError(
        f"readiness port {host}:{port} is already in use; "
        "Slurm mode will not terminate another job's listener"
    )


def setup_runner(
    runner: str,
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
    reuse_state: bool = False,
    setup_mode: str = "ci",
) -> tuple[Dict[str, str], Optional[ProcessGroupManager]]:
    local_env = dict(env)
    pgm: Optional[ProcessGroupManager] = None

    if setup_mode == "slurm":
        job_id = local_env.get("SLURM_JOB_ID")
        if not job_id:
            raise RuntimeError(
                "SLURM_JOB_ID is required when --setup-mode=slurm is used"
            )
        runner_id = f"slurm-{job_id}"
        step_id = local_env.get("SLURM_STEP_ID")
        proc_id = local_env.get("SLURM_PROCID")
        if step_id:
            runner_id += f"-step-{step_id}"
        if proc_id:
            runner_id += f"-proc-{proc_id}"
        pgm = ProcessGroupManager(runner_id=runner_id)
        local_env["CI_RUNNER_ID"] = pgm.runner_id
        print(f"[slurm] runner_id={pgm.runner_id}", flush=True)
        return local_env, pgm
    if setup_mode != "ci":
        raise ValueError(f"unsupported setup mode: {setup_mode!r}")

    if is_gb200_runner(runner):
        pgm = make_manager()
        local_env["CI_RUNNER_ID"] = pgm.runner_id
        print(f"[gb200] runner_id={pgm.runner_id}", flush=True)

        # Kill stale processes from previous run
        pgm.cleanup_stale(dry_run=dry_run)
        remove_stale_torch_extension_builds(local_env, dry_run)
        shell_run(
            "bash test/ci_system/cleanup_nvidia_gpu_state.sh",
            env=local_env,
            cwd=cwd,
            dry_run=dry_run,
            check=False,
        )

        venv_path = create_ci_venv_name(runner_name=pgm.runner_id)

        # Per-runner HOME isolates flashinfer JIT cache between runners
        runner_home = f"/mnt/workspace/ts-ci-homes/{pgm.runner_id}"
        Path(runner_home).mkdir(parents=True, exist_ok=True)
        local_env["HOME"] = runner_home

        # Recreate venv at fixed path unless a previous CI step already did it.
        if Path(venv_path).exists() and not dry_run and not reuse_state:
            print(f"[gb200] removing old venv: {venv_path}", flush=True)
            shutil.rmtree(venv_path, ignore_errors=True)
        if not reuse_state or not Path(venv_path).exists():
            shell_run(
                f"python3 -m venv --system-site-packages {venv_path}",
                env=local_env,
                cwd=cwd,
                dry_run=dry_run,
            )
        local_env["CI_VENV_PATH"] = venv_path
        local_env["PATH"] = f"{venv_path}/bin:{local_env.get('PATH', '')}"
        local_env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
        local_env["PIP_TRUSTED_HOST"] = (
            "pypi.org pypi.python.org files.pythonhosted.org github.com "
            "objects.githubusercontent.com"
        )
        local_env["REQUESTS_CA_BUNDLE"] = ""
        local_env["CURL_CA_BUNDLE"] = ""
        shell_run(
            "git config --global http.sslVerify false",
            env=local_env,
            cwd=cwd,
            dry_run=dry_run,
        )
        return local_env, pgm

    kill_stale_processes(local_env, cwd, dry_run)
    if should_run_nvidia_gpu_cleanup(runner):
        shell_run(
            "bash test/ci_system/cleanup_nvidia_gpu_state.sh",
            env=local_env,
            cwd=cwd,
            dry_run=dry_run,
            check=False,
        )
    if runner.startswith("b200v2-"):
        # b200v2 nodes do not have a usable IPv6 default route. Persist the
        # setting for every apt invocation in this runner, including the
        # later install_deps.sh step.
        shell_run(
            "echo 'Acquire::ForceIPv4 \"true\";' | "
            "sudo tee /etc/apt/apt.conf.d/99tokenspeed-force-ipv4 >/dev/null",
            env=local_env,
            cwd=cwd,
            dry_run=dry_run,
        )
        # The b200v2 network path intermittently replaces Ubuntu HTTP
        # InRelease responses with a short non-clearsigned payload, which
        # apt reports as NOSPLIT. HTTPS bypasses that HTTP interception.
        shell_run(
            "find /etc/apt -maxdepth 2 -type f "
            "\\( -path /etc/apt/sources.list -o -name '*.sources' \\) "
            "-exec sudo sed -i "
            "'s|http://archive.ubuntu.com/ubuntu|"
            "https://archive.ubuntu.com/ubuntu|g; "
            "s|http://security.ubuntu.com/ubuntu|"
            "https://security.ubuntu.com/ubuntu|g' "
            "{} +",
            env=local_env,
            cwd=cwd,
            dry_run=dry_run,
        )
    shell_run(
        "sudo apt-get -o Acquire::Retries=5 update -q",
        env=local_env,
        cwd=cwd,
        dry_run=dry_run,
    )
    shell_run(
        "sudo apt-get install -y ninja-build",
        env=local_env,
        cwd=cwd,
        dry_run=dry_run,
    )

    # nvidia-cusparseLt is a CUDA-only dependency; skip on AMD/ROCm runners.
    if not is_amd_runner(runner):
        shell_run(
            "pip3 install --break-system-packages -q nvidia-cusparselt-cu13",
            env=local_env,
            cwd=cwd,
            dry_run=dry_run,
        )

    if is_amd_runner(runner):
        if runner != "amd-mi450-sim":
            # Best-effort: kill any GPU-holding processes left over by a
            # previous pod scheduled on the same node. Cluster admins flagged
            # a known race where the device plugin releases a GPU back to the
            # pool before the previous pod's processes have actually
            # relinquished VRAM, so we can land in a pod with ~0 GiB free
            # VRAM. Cleanup script never fails the task.
            shell_run(
                "bash test/ci_system/cleanup_amd_gpu_state.sh",
                env=local_env,
                cwd=cwd,
                dry_run=dry_run,
                check=False,
            )
        return local_env, pgm
    if dry_run:
        return local_env, pgm

    lookup = subprocess.run(
        "ldconfig -p 2>/dev/null | grep libcusparseLt.so | head -1",
        shell=True,
        cwd=cwd,
        env=local_env,
        capture_output=True,
        text=True,
        check=False,
    )
    lib_path = lookup.stdout.strip().split(" => ")[-1] if lookup.stdout.strip() else ""
    if lib_path:
        local_env["LD_LIBRARY_PATH"] = (
            f"{Path(lib_path).parent}:{local_env.get('LD_LIBRARY_PATH', '')}"
        ).strip(":")
    return local_env, pgm


def run_perf_diagnostics(
    label: str,
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
) -> None:
    shell_run(
        f"bash test/ci_system/diagnose_nvidia_state.sh {shlex.quote(label)}",
        env=env,
        cwd=cwd,
        dry_run=dry_run,
        check=False,
    )


def _read_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(), filename=str(path))


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_suite_name(command: str) -> str | None:
    match = re.search(r"--suite\s+([A-Za-z0-9._-]+)", command)
    return match.group(1) if match else None


def _find_registered_runtime_tests(repo_root: Path, suite: str) -> List[str]:
    base_dir = repo_root / "test" / "runtime"
    matches: List[str] = []
    for name in glob.glob(str(base_dir / "**" / "*.py"), recursive=True):
        if (
            name.endswith("/conftest.py")
            or name.endswith("/__init__.py")
            or name.endswith("/run_ci_suite.py")
        ):
            continue
        path = Path(name)
        tree = _read_ast(path)
        for stmt in getattr(tree, "body", []):
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            if (
                not isinstance(call.func, ast.Name)
                or call.func.id != "register_cuda_ci"
            ):
                continue
            for keyword in call.keywords:
                if keyword.arg == "suite" and _const_str(keyword.value) == suite:
                    matches.append(path.relative_to(repo_root).as_posix())
                    break
            else:
                if len(call.args) >= 2 and _const_str(call.args[1]) == suite:
                    matches.append(path.relative_to(repo_root).as_posix())
                    break
    return sorted(matches)


def _extract_ci_models_from_test_file(path: Path) -> List[str]:
    tree = _read_ast(path)
    models: List[str] = []
    for stmt in getattr(tree, "body", []):
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "CI_MODELS"
            for target in stmt.targets
        ):
            continue
        if not isinstance(stmt.value, ast.List):
            continue
        for elt in stmt.value.elts:
            if not isinstance(elt, ast.Call):
                continue
            if not isinstance(elt.func, ast.Name) or elt.func.id != "ModelCase":
                continue
            if elt.args:
                model = _const_str(elt.args[0])
                if model:
                    models.append(model)
                    continue
            for keyword in elt.keywords:
                if keyword.arg == "model_path":
                    model = _const_str(keyword.value)
                    if model:
                        models.append(model)
        break
    return sorted(dict.fromkeys(models))


def summarize_ut_targets(task: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    commands = task.get("ut", {}).get("commands", [])
    summary: Dict[str, Any] = {
        "test_files": [],
        "models": [],
        "suites": [],
        "commands": commands,
    }
    test_files: List[str] = []
    model_names: List[str] = []
    suites: List[str] = []
    for command in commands:
        suite = _extract_suite_name(command)
        if suite:
            suites.append(suite)
            suite_files = _find_registered_runtime_tests(repo_root, suite)
            test_files.extend(suite_files)
            for rel_path in suite_files:
                model_names.extend(
                    _extract_ci_models_from_test_file(repo_root / rel_path)
                )
        if "pytest tokenspeed-kernel/test/" in command:
            summary["kernel_pytest"] = "tokenspeed-kernel/test/"
    summary["test_files"] = sorted(dict.fromkeys(test_files))
    summary["models"] = sorted(dict.fromkeys(model_names))
    summary["suites"] = sorted(dict.fromkeys(suites))
    return summary


def summarize_task_targets(task: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    if task["type"] == "ut":
        return summarize_ut_targets(task, repo_root)
    return {}


def summarize_command_output(command: str, output: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    models_run = sorted(
        dict.fromkeys(re.findall(r"\[RTRunner\] model=([^\n]+)", output))
    )
    if models_run:
        result["models_run"] = models_run

    models_skipped = sorted(dict.fromkeys(re.findall(r"Skipping ([^:\n]+):", output)))
    if models_skipped:
        result["models_skipped"] = models_skipped

    suite_match = re.search(r"Test Summary:\s+(\d+)/(\d+)\s+passed", output)
    if suite_match:
        result["suite_passed"] = int(suite_match.group(1))
        result["suite_total"] = int(suite_match.group(2))

    pytest_match = re.search(r"=+\s+(.+?)\s+in [0-9.]+s\s+=+", output)
    if pytest_match:
        result["pytest_summary"] = pytest_match.group(1).strip()

    passed_files = re.findall(r"^✓ PASSED:\n((?:  .+\n)+)", output, flags=re.MULTILINE)
    if passed_files:
        result["passed_files"] = [
            line.strip() for line in passed_files[0].splitlines() if line.strip()
        ]

    failed_files = re.findall(r"^✗ FAILED:\n((?:  .+\n)+)", output, flags=re.MULTILINE)
    if failed_files:
        result["failed_files"] = [
            line.strip() for line in failed_files[0].splitlines() if line.strip()
        ]

    evalscope_report_table = extract_evalscope_table(output, "Overall report table:")
    if evalscope_report_table:
        result["evalscope_report_table"] = evalscope_report_table
        evalscope_score = extract_evalscope_score(evalscope_report_table)
        if evalscope_score is not None:
            result["evalscope_score"] = evalscope_score

    evalscope_perf_table = extract_evalscope_table(output, "Overall perf table:")
    if evalscope_perf_table:
        result["evalscope_perf_table"] = evalscope_perf_table

    inspect_score = extract_inspect_score(output)
    if inspect_score is not None:
        result["inspect_score"] = inspect_score

    perf_summary_rows = extract_perf_summary_rows(output)
    if perf_summary_rows:
        result["perf_summary_rows"] = perf_summary_rows

    return result


_ACCEPT_RATE_RE = re.compile(r"\baccept_rate:\s*([0-9]+(?:\.[0-9]+)?)")


def extract_accept_rates(output: str) -> List[float]:
    return [float(value) for value in _ACCEPT_RATE_RE.findall(output)]


def parse_evalscope_score_cell(cell: str) -> float | None:
    """Read one Score cell as a fraction of 1, or None when it is not a score.

    evalscope reports the score as a bare fraction ("0.9667") up to 1.10 and
    as a percentage ("96.67%") from 1.11. Both mean the same thing, and the
    thresholds in the task configs are fractions, so normalize percentages
    back to fractions here rather than teaching every caller two formats.
    """
    text = cell.strip()
    if not text:
        return None
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return value / 100.0 if percent else value


def extract_evalscope_score(report_table: str) -> float | None:
    score_index: int | None = None
    score: float | None = None

    for line in report_table.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("|", "│")):
            continue
        separator = "│" if stripped.startswith("│") else "|"
        cells = [cell.strip() for cell in stripped.strip(separator).split(separator)]
        if not cells:
            continue
        if all(set(cell) <= {"=", "-"} for cell in cells if cell):
            continue
        normalized = [cell.lower() for cell in cells]
        if "score" in normalized:
            score_index = normalized.index("score")
            continue
        if score_index is None or len(cells) <= score_index:
            continue
        parsed = parse_evalscope_score_cell(cells[score_index])
        if parsed is not None:
            score = parsed

    return score


def extract_inspect_score(output: str) -> float | None:
    score: float | None = None
    for line in output.splitlines():
        match = re.fullmatch(
            r"\s*accuracy\s+([0-9]+(?:\.[0-9]+)?)\s*",
            clean_log_line(line),
        )
        if match:
            score = float(match.group(1))
    return score


def summarize_eval_accept_rate(
    task: Dict[str, Any],
    command_results: List[Dict[str, Any]],
    stages_run: List[str],
    server_log_path: Path | None,
) -> Dict[str, Any] | None:
    if task["type"] != "eval" or "eval" not in stages_run:
        return None

    accept_rates: List[float] = []
    for result in command_results:
        if result.get("stage") == "eval":
            accept_rates.extend(extract_accept_rates(str(result.get("output", ""))))

    if server_log_path is not None and server_log_path.exists():
        accept_rates.extend(extract_accept_rates(server_log_path.read_text()))

    if not accept_rates:
        print("[eval-accept-rate] no accept_rate found in eval logs", flush=True)
        return None

    average = sum(accept_rates) / len(accept_rates)
    print(
        f"Eval accept rate: {average:g} (samples={len(accept_rates)})",
        flush=True,
    )
    return {
        "accept_rate": average,
        "samples": len(accept_rates),
    }


def parse_eval_score_threshold(threshold: Any) -> tuple[float, float | None]:
    if isinstance(threshold, (int, float)):
        return float(threshold), None
    if isinstance(threshold, str):
        return float(threshold), None
    if isinstance(threshold, list) and len(threshold) == 2:
        return float(threshold[0]), float(threshold[1])
    raise ValueError(
        "eval.score_threshold must be a number, a two-item [min, max] range, "
        "or a mapping of runner label to one of those values"
    )


def resolve_score_threshold_for_runner(threshold: Any, runner: str) -> Any:
    if isinstance(threshold, dict):
        return threshold.get(runner)
    return threshold


def format_eval_score_threshold(min_score: float, max_score: float | None) -> str:
    if max_score is None:
        return f">= {min_score:g}"
    return f"[{min_score:g}, {max_score:g}]"


def check_eval_score_threshold(
    task: Dict[str, Any],
    command_results: List[Dict[str, Any]],
    stages_run: List[str],
    runner: str,
) -> Dict[str, Any] | None:
    threshold = task.get("score_threshold")
    if threshold is None:
        threshold = task.get("eval", {}).get("score_threshold")
    if threshold is None:
        print("[eval-score] no score_threshold configured", flush=True)
        return None
    threshold = resolve_score_threshold_for_runner(threshold, runner)
    if threshold is None:
        print(
            "[eval-score] no score_threshold configured for runner " f"{runner!r}",
            flush=True,
        )
        return None
    if "eval" not in stages_run:
        print(
            "[eval-score] score_threshold configured but eval stage was not "
            f"executed; stages={stages_run}",
            flush=True,
        )
        return None

    scores = []
    for result in command_results:
        if result.get("stage") != "eval":
            continue
        for key in ("evalscope_score", "inspect_score"):
            if key in result:
                scores.append(float(result[key]))
                break
    if not scores:
        print("[eval-score] no eval score found in command output", flush=True)
        raise ValueError("eval.score_threshold is configured but no score was found")

    score = scores[-1]
    min_score, max_score = parse_eval_score_threshold(threshold)
    passed = score >= min_score and (max_score is None or score <= max_score)
    threshold_text = format_eval_score_threshold(min_score, max_score)
    status = "passed" if passed else "failed"
    print(
        f"[eval-score] score={score:g}, threshold={threshold_text}, status={status}",
        flush=True,
    )
    return {
        "score": score,
        "min": min_score,
        "max": max_score,
        "passed": passed,
        "threshold": threshold_text,
    }


PERF_CSV_HEADER = (
    "config,Conc.,Latency (tps/user),Throughput (tps/gpu),"
    "Approx Cache Hit,Decoded Tok/Iter"
)
PERF_REFERENCE_METRICS = ("Latency (tps/user)", "Throughput (tps/gpu)")


def extract_perf_summary_rows(output: str) -> List[Dict[str, str]] | None:
    idx = output.find(PERF_CSV_HEADER)
    if idx < 0:
        return None
    block = output[idx:]
    lines: List[str] = []
    for raw in block.splitlines():
        line = clean_log_line(raw)
        if not line.strip():
            if lines:
                break
            continue
        if lines and re.match(r"^\d{4}-\d{2}-\d{2} .* - .* - ", line):
            break
        lines.append(line)
    if len(lines) < 2:
        return None
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def check_perf_reference(
    task: Dict[str, Any],
    command_results: List[Dict[str, Any]],
    stages_run: List[str],
) -> Dict[str, Any] | None:
    reference = task.get("perf_reference")
    if reference is None:
        print("[perf-ref] no perf_reference configured", flush=True)
        return None
    if "perf" not in stages_run:
        print(
            "[perf-ref] perf_reference configured but perf stage was not "
            f"executed; stages={stages_run}",
            flush=True,
        )
        return None
    if not isinstance(reference, dict) or not reference:
        raise ValueError("perf_reference must be a non-empty dict")

    threshold = float(task.get("perf_threshold", 0.9))

    rows: List[Dict[str, str]] | None = None
    for result in command_results:
        if "perf_summary_rows" in result:
            rows = result["perf_summary_rows"]
    if not rows:
        print("[perf-ref] no perf summary rows found in command output", flush=True)
        raise ValueError(
            "perf_reference is configured but no perf summary rows were found"
        )

    failures: List[str] = []
    checks: List[Dict[str, Any]] = []
    for conc_key, ref_pair in reference.items():
        try:
            conc = int(conc_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"perf_reference key must be an int concurrency: got {conc_key!r}"
            ) from exc
        if not isinstance(ref_pair, list) or len(ref_pair) != 2:
            raise ValueError(
                f"perf_reference[{conc_key}] must be [tps_user, tps_gpu]; "
                f"got {ref_pair!r}"
            )
        match = next((r for r in rows if int(r["Conc."]) == conc), None)
        if match is None:
            failures.append(f"conc={conc}: no matching row in perf summary")
            continue
        entry: Dict[str, Any] = {"conc": conc}
        for metric, ref in zip(PERF_REFERENCE_METRICS, ref_pair):
            ref_v = float(ref)
            actual = float(match[metric])
            floor = ref_v * threshold
            passed_metric = actual >= floor
            entry[metric] = {
                "actual": actual,
                "ref": ref_v,
                "floor": floor,
                "passed": passed_metric,
            }
            if not passed_metric:
                failures.append(
                    f"conc={conc}: {metric} {actual:g} < {floor:g} "
                    f"({threshold * 100:g}% of {ref_v:g})"
                )
        checks.append(entry)

    passed = not failures
    status = "passed" if passed else "failed"
    print(f"[perf-ref] threshold={threshold:g}, status={status}", flush=True)
    for line in format_perf_reference_table(checks):
        print(f"[perf-ref]   {line}", flush=True)
    for line in failures:
        print(f"[perf-ref]   {line}", flush=True)
    return {
        "passed": passed,
        "threshold": threshold,
        "checks": checks,
        "failures": failures,
    }


def _ratio_pct(actual: float, ref: float) -> str:
    """Format actual/ref as a percentage, e.g. ``105.5%``."""
    if ref == 0:
        return "n/a"
    return f"{actual / ref * 100:.1f}%"


def format_perf_reference_table(checks: List[Dict[str, Any]]) -> List[str]:
    """Render the per-concurrency actual-vs-reference comparison as a
    monospace text table. Each metric shows four columns: ``actual``, the
    raw ``ref`` (un-thresholded), the ``floor`` (``ref * perf_threshold`` —
    the value an actual must clear to pass), and ``actual/ref`` (the raw
    percentage against ref, ``perf_threshold`` is NOT applied). Returns a
    list of lines without any prefix so the caller can decorate (e.g.
    ``[perf-ref]`` for stdout). Empty when ``checks`` has no entries."""
    if not checks:
        return []
    header = (
        f"{'Conc':>4}  "
        f"{'Lat actual':>10} {'Lat ref':>9} {'Lat floor':>10} {'Lat actual/ref':>14}  "
        f"{'Thru actual':>12} {'Thru ref':>10} {'Thru floor':>11} {'Thru actual/ref':>15}"
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for entry in checks:
        lat = entry.get("Latency (tps/user)") or {}
        thru = entry.get("Throughput (tps/gpu)") or {}
        lines.append(
            f"{entry['conc']:>4}  "
            f"{lat.get('actual', 0):>10.2f} "
            f"{lat.get('ref', 0):>9.2f} "
            f"{lat.get('floor', 0):>10.2f} "
            f"{_ratio_pct(lat.get('actual', 0), lat.get('ref', 0)):>14}  "
            f"{thru.get('actual', 0):>12.2f} "
            f"{thru.get('ref', 0):>10.2f} "
            f"{thru.get('floor', 0):>11.2f} "
            f"{_ratio_pct(thru.get('actual', 0), thru.get('ref', 0)):>15}"
        )
    return lines


def format_perf_reference_markdown_table(checks: List[Dict[str, Any]]) -> List[str]:
    """Markdown-table variant of ``format_perf_reference_table`` for the
    GitHub Step Summary. Same four columns per metric: ``actual``, raw
    ``ref``, the threshold-adjusted ``floor``, and ``actual/ref`` (raw
    percentage against ref). Empty when ``checks`` has no entries."""
    if not checks:
        return []
    lines = [
        "| Conc | Lat actual | Lat ref | Lat floor | Lat actual/ref "
        "| Thru actual | Thru ref | Thru floor | Thru actual/ref |",
        "|-----:|-----------:|--------:|----------:|---------------:"
        "|------------:|---------:|-----------:|----------------:|",
    ]
    for entry in checks:
        lat = entry.get("Latency (tps/user)") or {}
        thru = entry.get("Throughput (tps/gpu)") or {}
        lines.append(
            f"| {entry['conc']} "
            f"| {lat.get('actual', 0):.2f} "
            f"| {lat.get('ref', 0):.2f} "
            f"| {lat.get('floor', 0):.2f} "
            f"| {_ratio_pct(lat.get('actual', 0), lat.get('ref', 0))} "
            f"| {thru.get('actual', 0):.2f} "
            f"| {thru.get('ref', 0):.2f} "
            f"| {thru.get('floor', 0):.2f} "
            f"| {_ratio_pct(thru.get('actual', 0), thru.get('ref', 0))} |"
        )
    return lines


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_GITHUB_LOG_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z ?")


def clean_log_line(line: str) -> str:
    line = _ANSI_ESCAPE_RE.sub("", line)
    line = _GITHUB_LOG_TIMESTAMP_RE.sub("", line)
    return line.rstrip()


def extract_evalscope_table(output: str, marker: str) -> str | None:
    marker_index = output.find(marker)
    if marker_index < 0:
        return None

    lines = output[marker_index + len(marker) :].splitlines()
    table_lines: List[str] = []
    for line in lines:
        line = clean_log_line(line)
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        if table_lines and re.match(r"^\d{4}-\d{2}-\d{2} .* - .* - ", line):
            break
        table_lines.append(line.rstrip())

    if not table_lines:
        return None
    return "\n".join(table_lines)


def build_step_summary_lines(result: Dict[str, Any]) -> List[str]:
    lines = [
        f"## CI Task `{result['task']}`",
        "",
        f"- Runner: `{result['runner']}`",
        f"- Status: `{'success' if result['ok'] else 'failure'}`",
        f"- Executed stages: `{', '.join(result['executed_stages']) if result['executed_stages'] else 'none'}`",
    ]
    targets = result.get("targets", {})
    if targets.get("suites"):
        lines.append(f"- Suites: `{', '.join(targets['suites'])}`")
    if targets.get("test_files"):
        lines.append(f"- Test files: `{len(targets['test_files'])}`")
    if targets.get("models"):
        lines.append(f"- Planned models: `{', '.join(targets['models'])}`")
    if targets.get("kernel_pytest"):
        lines.append(f"- Kernel pytest: `{targets['kernel_pytest']}`")
    if result.get("eval_score_check"):
        check = result["eval_score_check"]
        status = "pass" if check["passed"] else "fail"
        lines.append(
            f"- Eval score: `{check['score']:g}` "
            f"(threshold `{check['threshold']}`, {status})"
        )
    if result.get("perf_reference_check"):
        check = result["perf_reference_check"]
        status = "pass" if check["passed"] else "fail"
        lines.append(
            f"- Perf reference: `{status}` "
            f"(threshold `{check['threshold']:g}`, "
            f"{len(check['checks'])} concurrency levels)"
        )
        md_table = format_perf_reference_markdown_table(check["checks"])
        if md_table:
            lines.extend(["", *md_table, ""])
        if not check["passed"]:
            lines.extend([f"  - {failure}" for failure in check["failures"]])
    if result.get("eval_accept_rate"):
        accept_rate = result["eval_accept_rate"]
        lines.append(
            f"- Eval accept rate: `{accept_rate['accept_rate']:g}` "
            f"({accept_rate['samples']} samples)"
        )

    command_results = result.get("command_results", [])
    if command_results:
        lines.extend(["", "### Command Results", ""])
        for item in command_results:
            lines.append(f"- Command: `{item['command']}`")
            if "pytest_summary" in item:
                lines.append(f"  pytest: `{item['pytest_summary']}`")
            if "suite_total" in item:
                lines.append(
                    f"  suite: `{item['suite_passed']}/{item['suite_total']} passed`"
                )
            if item.get("models_run"):
                lines.append(f"  ran models: `{', '.join(item['models_run'])}`")
            if item.get("models_skipped"):
                lines.append(f"  skipped models: `{', '.join(item['models_skipped'])}`")
            if item.get("failed_files"):
                lines.append(f"  failed files: `{', '.join(item['failed_files'])}`")
            if "evalscope_score" in item:
                lines.append(f"  evalscope score: `{item['evalscope_score']:g}`")
            if "inspect_score" in item:
                lines.append(f"  inspect score: `{item['inspect_score']:g}`")
            if item.get("evalscope_report_table"):
                lines.extend(
                    [
                        "  evalscope overall report:",
                        "  ```text",
                        *[
                            f"  {line}"
                            for line in item["evalscope_report_table"].splitlines()
                        ],
                        "  ```",
                    ]
                )
            if item.get("evalscope_perf_table"):
                lines.extend(
                    [
                        "  evalscope overall perf:",
                        "  ```text",
                        *[
                            f"  {line}"
                            for line in item["evalscope_perf_table"].splitlines()
                        ],
                        "  ```",
                    ]
                )
    if result.get("error"):
        lines.extend(["", f"- Error: `{result['error']}`"])
    lines.append("")
    return lines


def write_detailed_step_summary(result: Dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a") as handle:
            handle.write("\n".join(build_step_summary_lines(result)))
    except OSError as exc:
        print(
            f"warning: could not write GitHub step summary: {exc}",
            file=sys.stderr,
        )


def write_result(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n")


def server_exit_error(
    process: subprocess.Popen[str], log_path: Path | None
) -> str | None:
    returncode = process.poll()
    if returncode is None:
        return None
    detail = ""
    if log_path is not None and log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        if lines:
            detail = "\nLast server log lines:\n" + "\n".join(lines[-20:])
    return f"server exited before readiness with exit code {returncode}{detail}"


def poll_readiness(
    ready: Dict[str, Any],
    dry_run: bool,
    process: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
) -> None:
    url = str(ready["url"])
    timeout_seconds = int(ready.get("timeout", 600))
    interval_seconds = int(ready.get("interval", 10))
    expected_status = int(ready.get("expected_status", 200))

    if dry_run:
        print(f"[dry-run] wait for readiness: {url} -> {expected_status}", flush=True)
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process is not None:
            error = server_exit_error(process, log_path)
            if error is not None:
                raise RuntimeError(error)
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == expected_status:
                    return
        except URLError:
            pass
        sleep_deadline = min(deadline, time.time() + interval_seconds)
        while time.time() < sleep_deadline:
            if process is not None:
                error = server_exit_error(process, log_path)
                if error is not None:
                    raise RuntimeError(error)
            time.sleep(min(0.5, sleep_deadline - time.time()))

    raise RuntimeError(f"server readiness probe timed out: {url}")


def start_server(
    command: str, env: Dict[str, str], cwd: Path, dry_run: bool
) -> subprocess.Popen[str] | None:
    print(f"$ {command}", flush=True)
    if dry_run:
        return None
    return subprocess.Popen(
        command, shell=True, cwd=cwd, env=env, start_new_session=True
    )


def wrap_command_with_log(
    command: str, log_path: Path, *, login_shell: bool = True
) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = (
        "set -o pipefail; "
        f"{{ {command}; }} 2>&1 | tee -a {shlex.quote(str(log_path))}"
    )
    flag = "-lc" if login_shell else "-c"
    return f"bash {flag} {shlex.quote(wrapped)}"


def stop_server(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def get_stage_commands(task: Dict[str, Any]) -> List[tuple[str, Any]]:
    stages: List[tuple[str, Any]] = []
    install = task.get("install", [])
    if install:
        if not isinstance(install, list) or not all(
            isinstance(item, str) for item in install
        ):
            raise ValueError("install must be a string list")
        stages.append(("install", install))

    task_type = task["type"]
    if task_type == "ut":
        commands = task.get("ut", {}).get("commands", [])
        if not isinstance(commands, list) or not all(
            isinstance(item, str) for item in commands
        ):
            raise ValueError("ut.commands must be a string list")
        stages.append(("ut", commands))
        return stages

    if task_type == "server_smoke":
        smoke = task.get("smoke", {})
        if smoke.get("command"):
            stages.append(("smoke", [smoke["command"]]))
        return stages

    if task_type in {"eval", "perf"}:
        server = task.get("server", {})
        if server.get("command"):
            stages.append(("server", server))
        section = task.get(task_type, {})
        section_install = section.get("install", [])
        if section_install:
            if not isinstance(section_install, list) or not all(
                isinstance(item, str) for item in section_install
            ):
                raise ValueError(f"{task_type}.install must be a string list")
            stages.append((f"{task_type}.install", section_install))
        if section.get("command"):
            stages.append((task_type, [section["command"]]))
        return stages

    raise ValueError(f"unsupported task type: {task_type}")


def filter_stage_commands(
    stages: List[tuple[str, Any]],
    *,
    only_stages: set[str] | None,
    skip_stages: set[str],
) -> List[tuple[str, Any]]:
    if only_stages is not None:
        stages = [(name, payload) for name, payload in stages if name in only_stages]
    if skip_stages:
        stages = [
            (name, payload) for name, payload in stages if name not in skip_stages
        ]
    return stages


def cleanup_runner(
    env: Dict[str, str], cwd: Path, dry_run: bool, pgm: ProcessGroupManager = None
) -> None:
    if pgm is None:
        kill_stale_processes(env, cwd, dry_run)

    venv_path = env.get("CI_VENV_PATH")
    if venv_path and Path(venv_path).exists():
        if dry_run:
            print(f"[dry-run] remove {venv_path}", flush=True)
        else:
            shutil.rmtree(venv_path, ignore_errors=True)


def execute_task(
    *,
    config: str,
    runner: str,
    runner_override: str | None = None,
    work_dir: str,
    dry_run: bool,
    print_plan: bool,
    result_json: str | None,
    only_stages: set[str] | None = None,
    skip_stages: set[str] | None = None,
    keep_runner_state: bool = False,
    reuse_runner_state: bool = False,
    setup_mode: str = "ci",
    serve_only: bool = False,
    external_server: bool = False,
) -> int:
    if setup_mode not in SUPPORTED_SETUP_MODES:
        raise ValueError(f"unsupported setup mode: {setup_mode!r}")
    if setup_mode == "slurm" and (keep_runner_state or reuse_runner_state):
        raise ValueError(
            "--keep-runner-state and --reuse-runner-state are not supported "
            "with --setup-mode=slurm"
        )
    if serve_only and external_server:
        raise ValueError("--serve-only and --external-server cannot be combined")
    if (serve_only or external_server) and setup_mode != "slurm":
        raise ValueError(
            "--serve-only and --external-server require --setup-mode=slurm"
        )

    repo_root = Path(work_dir).resolve()
    task = normalize_task(repo_root / config, repo_root)
    if runner not in resolve_runner_labels(task["runner"]["labels"]):
        raise ValueError(
            f"{config}: runner {runner!r} is not declared in runner.labels"
        )
    declared_runner = runner
    runner = apply_slurm_runner_override(declared_runner, runner_override, setup_mode)
    targets = summarize_task_targets(task, repo_root)

    env = merge_env(task.get("env", {}))
    env["CI_TASK_NAME"] = str(task["name"])
    env["CI_TASK_TYPE"] = str(task["type"])
    env["CI_RUNNER_LABEL"] = runner
    env.update(get_default_runner_env(runner))
    env.update(get_runner_specific_env(task, declared_runner))

    jit_cache_env = get_jit_cache_env(env) if uses_isolated_jit_cache(runner) else {}
    env.update(jit_cache_env)
    if not dry_run:
        for cache_dir in jit_cache_env.values():
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    task_stages = get_stage_commands(task)
    if serve_only or external_server:
        if task["type"] not in {"eval", "perf"} or not any(
            name == "server" for name, _ in task_stages
        ):
            raise ValueError(
                "--serve-only and --external-server require an eval/perf task "
                "with a server stage"
            )
    if serve_only:
        task_stages = [
            (name, payload)
            for name, payload in task_stages
            if name in {"install", "server"}
        ]
    stages = filter_stage_commands(
        task_stages,
        only_stages=only_stages,
        skip_stages=skip_stages or set(),
    )

    if print_plan:
        print(
            json.dumps(
                {
                    "name": task["name"],
                    "type": task["type"],
                    "config": config,
                    "runner": runner,
                    "setup_mode": setup_mode,
                    "server_mode": (
                        "serve-only"
                        if serve_only
                        else "external" if external_server else "managed"
                    ),
                    "stages": [name for name, _ in stages],
                    "targets": targets,
                },
                indent=2,
            )
        )
        if dry_run:
            return 0

    runner_env, pgm = setup_runner(
        runner,
        env,
        repo_root,
        dry_run,
        reuse_state=reuse_runner_state,
        setup_mode=setup_mode,
    )
    enable_perf_diagnostics = should_run_perf_diagnostics(task, runner)
    stages_run: List[str] = []
    command_results: List[Dict[str, Any]] = []
    eval_score_check: Dict[str, Any] | None = None
    eval_accept_rate: Dict[str, Any] | None = None
    perf_reference_check: Dict[str, Any] | None = None
    server_process = None
    server_log_path: Path | None = None
    error: str | None = None
    error_reported = False
    max_attempts = 1 + int(task.get("retries") or 0)

    def _stop_managed_server() -> None:
        nonlocal server_process
        if pgm is not None:
            pgm.terminate_all(dry_run=dry_run)
        else:
            stop_server(server_process)
        server_process = None

    def _run_task_stages() -> None:
        nonlocal server_process, server_log_path
        nonlocal eval_score_check, eval_accept_rate, stages_run, command_results
        for stage_name, stage_payload in stages:
            stages_run.append(stage_name)
            if stage_name == "server":
                ready = dict(stage_payload["ready"])
                server_command = stage_payload["command"]
                if setup_mode == "slurm":
                    ready["timeout"] = max(int(ready.get("timeout", 600)), 7200)
                    server_command = configure_slurm_server_command(
                        server_command, int(ready["timeout"])
                    )
                if serve_only:
                    if pgm is not None:
                        command_result = pgm.run(
                            server_command,
                            cwd=repo_root,
                            env=runner_env,
                            dry_run=dry_run,
                        )
                    else:
                        command_result = shell_run(
                            server_command,
                            env=runner_env,
                            cwd=repo_root,
                            dry_run=dry_run,
                        )
                    command_result["stage"] = stage_name
                    command_results.append(command_result)
                    continue
                if external_server:
                    poll_readiness(ready, dry_run)
                    continue
                if enable_perf_diagnostics:
                    run_perf_diagnostics(
                        "before server", runner_env, repo_root, dry_run
                    )
                server_log_path = repo_root / ".ci-artifacts" / "server.log"
                server_log_path.parent.mkdir(parents=True, exist_ok=True)
                if not dry_run:
                    server_log_path.write_text("")
                if setup_mode == "slurm":
                    ensure_ready_port_available(ready, dry_run)
                else:
                    kill_ready_port_listener(ready, runner_env, repo_root, dry_run)
                if pgm is not None:
                    server_process = pgm.start(
                        wrap_command_with_log(
                            server_command,
                            server_log_path,
                            login_shell=False,
                        ),
                        cwd=repo_root,
                        env=runner_env,
                        dry_run=dry_run,
                    )
                else:
                    server_process = start_server(
                        wrap_command_with_log(server_command, server_log_path),
                        runner_env,
                        repo_root,
                        dry_run,
                    )
                poll_readiness(
                    ready,
                    dry_run,
                    process=server_process,
                    log_path=server_log_path,
                )
                if enable_perf_diagnostics:
                    run_perf_diagnostics(
                        "after server ready", runner_env, repo_root, dry_run
                    )
                continue
            for command in stage_payload:
                if enable_perf_diagnostics and stage_name == "perf":
                    run_perf_diagnostics(
                        "before perf command", runner_env, repo_root, dry_run
                    )
                if pgm is not None:
                    command_result = pgm.run(
                        command,
                        cwd=repo_root,
                        env=runner_env,
                        dry_run=dry_run,
                    )
                else:
                    command_result = shell_run(
                        command, env=runner_env, cwd=repo_root, dry_run=dry_run
                    )
                command_result["stage"] = stage_name
                command_result.update(
                    summarize_command_output(
                        command, str(command_result.get("output", ""))
                    )
                )
                command_results.append(command_result)
                if enable_perf_diagnostics and stage_name == "perf":
                    run_perf_diagnostics(
                        "after perf command", runner_env, repo_root, dry_run
                    )

        eval_accept_rate = summarize_eval_accept_rate(
            task, command_results, stages_run, server_log_path
        )
        eval_score_check = check_eval_score_threshold(
            task, command_results, stages_run, declared_runner
        )
        if eval_score_check is not None and not eval_score_check["passed"]:
            raise RuntimeError(
                f"eval score {eval_score_check['score']:g} does not satisfy "
                f"threshold {eval_score_check['threshold']}"
            )

    try:
        if enable_perf_diagnostics:
            run_perf_diagnostics("before stages", runner_env, repo_root, dry_run)
        for attempt in range(1, max_attempts + 1):
            stages_run = []
            command_results = []
            eval_score_check = None
            eval_accept_rate = None
            try:
                _run_task_stages()
                error = None
                break
            except Exception as exc:
                error = str(exc)
                if attempt >= max_attempts:
                    break
                print(
                    f"[CI Retry] {task['name']} failed "
                    f"(attempt {attempt}/{max_attempts}): {error}",
                    flush=True,
                )
            finally:
                _stop_managed_server()
    finally:
        if enable_perf_diagnostics:
            run_perf_diagnostics("before cleanup", runner_env, repo_root, dry_run)
        _stop_managed_server()
        if setup_mode == "ci" and not keep_runner_state:
            cleanup_runner(runner_env, repo_root, dry_run, pgm)
        if enable_perf_diagnostics:
            run_perf_diagnostics("after cleanup", runner_env, repo_root, dry_run)

    if error is None:
        try:
            perf_reference_check = check_perf_reference(
                task, command_results, stages_run
            )
            if perf_reference_check is not None and not perf_reference_check["passed"]:
                error = "perf_reference check failed: " + "; ".join(
                    perf_reference_check["failures"]
                )
                error_reported = True
        except Exception as exc:
            error = str(exc)

    result = {
        "ok": error is None,
        "task": task["name"],
        "type": task["type"],
        "runner": runner,
        "setup_mode": setup_mode,
        "server_mode": (
            "serve-only" if serve_only else "external" if external_server else "managed"
        ),
        "executed_stages": stages_run,
        "targets": targets,
        "command_results": command_results,
    }
    if error is not None:
        result["error"] = error
    if eval_score_check is not None:
        result["eval_score_check"] = eval_score_check
    if perf_reference_check is not None:
        result["perf_reference_check"] = perf_reference_check
    if eval_accept_rate is not None:
        result["eval_accept_rate"] = eval_accept_rate
    write_result(result_json, result)
    if task.get("report", {}).get("github_step_summary"):
        write_detailed_step_summary(result)
    if error is not None:
        if not error_reported:
            print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenSpeed CI pipeline helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan CI task specs into a matrix")
    scan_parser.add_argument("--root", default="test/ci", help="Task root directory")
    scan_parser.add_argument(
        "--trigger",
        choices=sorted(SUPPORTED_TRIGGERS),
        default=None,
        help="Optional trigger filter",
    )
    scan_parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to emit relative config paths",
    )
    scan_parser.add_argument(
        "--runner-group",
        choices=SUPPORTED_RUNNER_GROUPS,
        default="all",
        help="Optional runner group filter",
    )
    scan_parser.add_argument(
        "--workflow-stage",
        choices=SUPPORTED_WORKFLOW_STAGES,
        default=None,
        help="Optional workflow stage filter",
    )
    scan_parser.add_argument(
        "--multi-node",
        choices=SUPPORTED_MULTI_NODE_FILTERS,
        default="exclude",
        help="Optionally select only or exclude multi-node Slurm tasks",
    )

    execute_parser = subparsers.add_parser("execute", help="Execute one CI task")
    execute_parser.add_argument(
        "--config", required=True, help="Task config path relative to repo root"
    )
    execute_parser.add_argument(
        "--runner", required=True, help="Runner label selected by the matrix"
    )
    execute_parser.add_argument(
        "--runner-override",
        help="Slurm-only effective GB300 runner for a declared B200/GB200 runner.",
    )
    execute_parser.add_argument(
        "--work-dir", default=".", help="Repository work directory"
    )
    execute_parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without executing"
    )
    execute_parser.add_argument(
        "--print-plan", action="store_true", help="Print normalized execution plan"
    )
    execute_parser.add_argument("--result-json", help="Optional JSON result path")
    execute_parser.add_argument(
        "--only-stage",
        action="append",
        default=None,
        help="Only execute this stage. May be passed multiple times.",
    )
    execute_parser.add_argument(
        "--skip-stage",
        action="append",
        default=[],
        help="Skip this stage. May be passed multiple times.",
    )
    execute_parser.add_argument(
        "--keep-runner-state",
        action="store_true",
        help="Leave runner setup state in place for a later execute invocation.",
    )
    execute_parser.add_argument(
        "--reuse-runner-state",
        action="store_true",
        help="Reuse runner setup state left by an earlier execute invocation.",
    )
    execute_parser.add_argument(
        "--setup-mode",
        choices=SUPPORTED_SETUP_MODES,
        default="ci",
        help="Runner setup policy. Slurm mode avoids host-wide CI mutations.",
    )
    server_mode = execute_parser.add_mutually_exclusive_group()
    server_mode.add_argument(
        "--serve-only",
        action="store_true",
        help="Run install and the server in the foreground for a Slurm server step.",
    )
    server_mode.add_argument(
        "--external-server",
        action="store_true",
        help="Wait for an externally managed server before running eval or perf.",
    )

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "scan":
        repo_root = Path(args.repo_root).resolve()
        root = (repo_root / args.root).resolve()
        matrix = build_matrix(
            root,
            repo_root,
            args.trigger,
            args.runner_group,
            args.workflow_stage,
            args.multi_node,
        )
        print(json.dumps(matrix, separators=(",", ":")))
        return 0

    if args.command == "execute":
        return execute_task(
            config=args.config,
            runner=args.runner,
            runner_override=args.runner_override,
            work_dir=args.work_dir,
            dry_run=args.dry_run,
            print_plan=args.print_plan,
            result_json=args.result_json,
            only_stages=set(args.only_stage) if args.only_stage else None,
            skip_stages=set(args.skip_stage),
            keep_runner_state=args.keep_runner_state,
            reuse_runner_state=args.reuse_runner_state,
            setup_mode=args.setup_mode,
            serve_only=args.serve_only,
            external_server=args.external_server,
        )

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
