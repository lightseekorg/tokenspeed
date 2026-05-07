#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import shlex
import shutil
import signal
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
SUPPORTED_TRIGGERS = {"per-commit", "manual", "nightly"}
B200_RUNNER_LABEL_ENV = "TOKENSPEED_B200_RUNNER_LABEL"
STALE_PROCESS_PATTERNS = [
    r"python.*tokenspeed\.api_server",
    r"tokenspeed-serve",
    r"run_ci_suite",
]
RUNNER_SM_PREFIXES = (
    (("h100", "h200"), "sm90"),
    (("b200", "gb200"), "sm100"),
    (("b300", "gb300"), "sm103"),
)

AMD_RUNNER_PREFIXES = ("linux-mi355",)


def is_amd_runner(runner: str) -> bool:
    return runner.startswith(AMD_RUNNER_PREFIXES)


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


def normalize_task(path: Path, repo_root: Path) -> Dict[str, Any]:
    data = load_yaml(path)
    validate_task(data, path)
    data["_source_path"] = path.relative_to(repo_root).as_posix()
    return data


def get_b200_runner_label_override() -> str:
    return os.environ.get(B200_RUNNER_LABEL_ENV, "").strip()


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


def build_matrix(root: Path, repo_root: Path, trigger: str | None) -> Dict[str, Any]:
    include = []
    for path in find_task_files(root):
        task = normalize_task(path, repo_root)
        if trigger and trigger not in task["triggers"]:
            continue
        for label in resolve_runner_labels(task["runner"]["labels"]):
            include.append(
                {
                    "name": task["name"],
                    "type": task["type"],
                    "config": task["_source_path"],
                    "runner": label,
                }
            )
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


def create_ci_venv_name(runner_name: str | None = None) -> str:
    if runner_name:
        # Fixed path per runner so flashinfer JIT cache (which embeds the
        # venv path in build.ninja) stays valid across CI runs.
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", runner_name)
        return f"/tmp/ci-env-{safe_name}"
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"/tmp/ci-env-{run_id}-{run_attempt}-{os.getpid()}"


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


def setup_runner(
    runner: str,
    env: Dict[str, str],
    cwd: Path,
    dry_run: bool,
    reuse_state: bool = False,
) -> Dict[str, str]:
    local_env = dict(env)
    pgm: Optional[ProcessGroupManager] = None

    if runner.startswith("gb200"):
        pgm = make_manager()
        local_env["CI_RUNNER_ID"] = pgm.runner_id
        print(f"[gb200] runner_id={pgm.runner_id}", flush=True)

        # Kill stale processes from previous run
        pgm.cleanup_stale(dry_run=dry_run)

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
    shell_run("sudo apt-get update -q", env=local_env, cwd=cwd, dry_run=dry_run)
    shell_run(
        "sudo apt-get install -y ninja-build",
        env=local_env,
        cwd=cwd,
        dry_run=dry_run,
    )
    shell_run(
        "sudo apt-get install -y libspdlog-dev || "
        "(git clone --depth 1 https://github.com/gabime/spdlog.git /tmp/spdlog && "
        "sudo cp -r /tmp/spdlog/include/spdlog /usr/local/include/)",
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

    return result


_ACCEPT_RATE_RE = re.compile(r"\baccept_rate:\s*([0-9]+(?:\.[0-9]+)?)")


def extract_accept_rates(output: str) -> List[float]:
    return [float(value) for value in _ACCEPT_RATE_RE.findall(output)]


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
        if any(set(cell) <= {"=", "-"} for cell in cells if cell):
            continue
        normalized = [cell.lower() for cell in cells]
        if "score" in normalized:
            score_index = normalized.index("score")
            continue
        if score_index is None or len(cells) <= score_index:
            continue
        try:
            score = float(cells[score_index])
        except ValueError:
            continue

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
        "eval.score_threshold must be a number or a two-item [min, max] range"
    )


def format_eval_score_threshold(min_score: float, max_score: float | None) -> str:
    if max_score is None:
        return f">= {min_score:g}"
    return f"[{min_score:g}, {max_score:g}]"


def check_eval_score_threshold(
    task: Dict[str, Any],
    command_results: List[Dict[str, Any]],
    stages_run: List[str],
) -> Dict[str, Any] | None:
    threshold = task.get("score_threshold")
    if threshold is None:
        threshold = task.get("eval", {}).get("score_threshold")
    if threshold is None:
        print("[eval-score] no score_threshold configured", flush=True)
        return None
    if "eval" not in stages_run:
        print(
            "[eval-score] score_threshold configured but eval stage was not "
            f"executed; stages={stages_run}",
            flush=True,
        )
        return None

    scores = [
        float(result["evalscope_score"])
        for result in command_results
        if "evalscope_score" in result
    ]
    if not scores:
        print("[eval-score] no evalscope score found in command output", flush=True)
        raise ValueError(
            "eval.score_threshold is configured but no evalscope score was found"
        )

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
    with open(summary_path, "a") as handle:
        handle.write("\n".join(build_step_summary_lines(result)))


def write_result(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n")


def poll_readiness(ready: Dict[str, Any], dry_run: bool) -> None:
    url = str(ready["url"])
    timeout_seconds = int(ready.get("timeout", 600))
    interval_seconds = int(ready.get("interval", 10))
    expected_status = int(ready.get("expected_status", 200))

    if dry_run:
        print(f"[dry-run] wait for readiness: {url} -> {expected_status}", flush=True)
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == expected_status:
                    return
        except URLError:
            pass
        time.sleep(interval_seconds)

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


def wrap_command_with_log(command: str, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = f"{{ {command}; }} 2>&1 | tee -a {shlex.quote(str(log_path))}"
    return f"bash -lc {shlex.quote(wrapped)}"


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
    work_dir: str,
    dry_run: bool,
    print_plan: bool,
    result_json: str | None,
    only_stages: set[str] | None = None,
    skip_stages: set[str] | None = None,
    keep_runner_state: bool = False,
    reuse_runner_state: bool = False,
) -> int:
    repo_root = Path(work_dir).resolve()
    task = normalize_task(repo_root / config, repo_root)
    if runner not in resolve_runner_labels(task["runner"]["labels"]):
        raise ValueError(
            f"{config}: runner {runner!r} is not declared in runner.labels"
        )
    targets = summarize_task_targets(task, repo_root)

    env = merge_env(task.get("env", {}))
    env["CI_TASK_NAME"] = str(task["name"])
    env["CI_TASK_TYPE"] = str(task["type"])
    env["CI_RUNNER_LABEL"] = runner
    env.update(get_default_runner_env(runner))
    env.update(task["runner"].get("env", {}).get(runner, {}))

    stages = filter_stage_commands(
        get_stage_commands(task),
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
                    "stages": [name for name, _ in stages],
                    "targets": targets,
                },
                indent=2,
            )
        )
        if dry_run:
            return 0

    runner_env, pgm = setup_runner(
        runner, env, repo_root, dry_run, reuse_state=reuse_runner_state
    )
    stages_run: List[str] = []
    command_results: List[Dict[str, Any]] = []
    eval_score_check: Dict[str, Any] | None = None
    eval_accept_rate: Dict[str, Any] | None = None
    server_process = None
    server_log_path: Path | None = None

    try:
        for stage_name, stage_payload in stages:
            stages_run.append(stage_name)
            if stage_name == "server":
                server_log_path = repo_root / ".ci-artifacts" / "server.log"
                server_log_path.parent.mkdir(parents=True, exist_ok=True)
                if not dry_run:
                    server_log_path.write_text("")
                kill_ready_port_listener(
                    stage_payload["ready"], runner_env, repo_root, dry_run
                )
                if pgm is not None:
                    server_process = pgm.start(
                        wrap_command_with_log(
                            stage_payload["command"], server_log_path
                        ),
                        cwd=repo_root,
                        env=runner_env,
                        dry_run=dry_run,
                    )
                else:
                    server_process = start_server(
                        wrap_command_with_log(
                            stage_payload["command"], server_log_path
                        ),
                        runner_env,
                        repo_root,
                        dry_run,
                    )
                poll_readiness(stage_payload["ready"], dry_run)
                continue
            for command in stage_payload:
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

        eval_accept_rate = summarize_eval_accept_rate(
            task, command_results, stages_run, server_log_path
        )
        eval_score_check = check_eval_score_threshold(task, command_results, stages_run)
        if eval_score_check is not None and not eval_score_check["passed"]:
            raise RuntimeError(
                f"eval score {eval_score_check['score']:g} does not satisfy "
                f"threshold {eval_score_check['threshold']}"
            )

        result = {
            "ok": True,
            "task": task["name"],
            "type": task["type"],
            "runner": runner,
            "executed_stages": stages_run,
            "targets": targets,
            "command_results": command_results,
        }
        if eval_score_check is not None:
            result["eval_score_check"] = eval_score_check
        if eval_accept_rate is not None:
            result["eval_accept_rate"] = eval_accept_rate
        if task.get("report", {}).get("github_step_summary"):
            write_detailed_step_summary(result)
        write_result(result_json, result)
        return 0
    except Exception as exc:
        result = {
            "ok": False,
            "task": task["name"],
            "type": task["type"],
            "runner": runner,
            "executed_stages": stages_run,
            "targets": targets,
            "command_results": command_results,
            "error": str(exc),
        }
        if eval_score_check is not None:
            result["eval_score_check"] = eval_score_check
        if eval_accept_rate is not None:
            result["eval_accept_rate"] = eval_accept_rate
        if task.get("report", {}).get("github_step_summary"):
            write_detailed_step_summary(result)
        write_result(result_json, result)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if pgm is not None:
            pgm.terminate_all(dry_run=dry_run)
        else:
            stop_server(server_process)
        if not keep_runner_state:
            cleanup_runner(runner_env, repo_root, dry_run, pgm)


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

    execute_parser = subparsers.add_parser("execute", help="Execute one CI task")
    execute_parser.add_argument(
        "--config", required=True, help="Task config path relative to repo root"
    )
    execute_parser.add_argument(
        "--runner", required=True, help="Runner label selected by the matrix"
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

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "scan":
        repo_root = Path(args.repo_root).resolve()
        root = (repo_root / args.root).resolve()
        matrix = build_matrix(root, repo_root, args.trigger)
        print(json.dumps(matrix, separators=(",", ":")))
        return 0

    if args.command == "execute":
        return execute_task(
            config=args.config,
            runner=args.runner,
            work_dir=args.work_dir,
            dry_run=args.dry_run,
            print_plan=args.print_plan,
            result_json=args.result_json,
            only_stages=set(args.only_stage) if args.only_stage else None,
            skip_stages=set(args.skip_stage),
            keep_runner_state=args.keep_runner_state,
            reuse_runner_state=args.reuse_runner_state,
        )

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
