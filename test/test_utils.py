"""Common utilities for testing and benchmarking."""

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.utils import get_bool_env_var, get_device
from tokenspeed.runtime.utils.process import kill_process_tree


def is_in_ci():
    """Return whether it is in CI runner."""
    return get_bool_env_var("CI") or get_bool_env_var("GITHUB_ACTIONS")


def is_in_amd_ci():
    """Return whether it is in CI on an AMD runner."""
    return is_in_ci() and current_platform().is_amd


def is_blackwell_system():
    """Return whether it is running on a Blackwell system."""
    return current_platform().is_blackwell


DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600
if is_in_amd_ci():
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600
if is_blackwell_system():
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3000

if is_in_ci():
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
        10000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 2000
    )
else:
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
        20000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 1000
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"


def auto_config_device() -> str:
    return get_device()


def _serve_process(
    command: List[str],
    env: dict,
    return_stdout_stderr: Optional[tuple],
) -> subprocess.Popen:
    if return_stdout_stderr:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env.copy(),
            text=True,
            bufsize=1,
        )

        def _dump(src, sinks):
            for line in iter(src.readline, ""):
                for sink in sinks:
                    sink.write(line)
                    sink.flush()
            src.close()

        threading.Thread(
            target=_dump,
            args=(proc.stdout, [return_stdout_stderr[0], sys.stdout]),
            daemon=True,
        ).start()
        threading.Thread(
            target=_dump,
            args=(proc.stderr, [return_stdout_stderr[1], sys.stderr]),
            daemon=True,
        ).start()
    else:
        proc = subprocess.Popen(command, stdout=None, stderr=None, env=env.copy())

    return proc


def _wait_for_server_health(
    proc: subprocess.Popen,
    base_url: str,
    api_key: Optional[str],
    timeout_duration: float,
) -> Tuple[bool, Optional[str]]:
    start_time = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout_duration:
            return_code = proc.poll()
            if return_code is not None:
                return False, f"Server process exited with code {return_code}"

            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {api_key}",
                }
                response = session.get(
                    f"{base_url}/health",
                    headers=headers,
                    timeout=5,
                )
                if response.status_code == 200:
                    return True, None
            except requests.RequestException:
                pass

            return_code = proc.poll()
            if return_code is not None:
                return False, f"Server unexpectedly exited (return_code={return_code})"

            time.sleep(10)

    return False, "Server failed to start within the timeout period"


def popen_serve_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: Optional[str] = None,
    other_args: Optional[list[str]] = None,
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
    device: str = "auto",
):
    other_args = other_args or []

    if device == "auto":
        device = auto_config_device()
        other_args = list(other_args)
        other_args += ["--device", str(device)]

    if env is None:
        env = os.environ.copy()
    else:
        merged = os.environ.copy()
        merged.update(env)
        env = merged

    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "tokenspeed",
        "serve",
        "--model",
        model,
        *[str(x) for x in other_args],
        "--host",
        host,
        "--port",
        port,
    ]

    if api_key:
        command += ["--api-key", api_key]

    print(f"command={shlex.join(command)}")

    process = _serve_process(command, env, return_stdout_stderr)
    success, error_msg = _wait_for_server_health(process, base_url, api_key, timeout)

    if success:
        return process

    try:
        kill_process_tree(process.pid)
    except Exception as e:
        print(f"Error killing process after launch failure: {e}")

    if "exited" in error_msg:
        raise Exception(error_msg + ". Check server logs for errors.")
    raise TimeoutError(error_msg)


def lcs(x, y):
    m = len(x)
    n = len(y)
    table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                table[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    return table[m][n]


def calculate_rouge_l(output_strs_list1, output_strs_list2):
    rouge_l_scores = []

    for s1, s2 in zip(output_strs_list1, output_strs_list2):
        lcs_len = lcs(s1, s2)
        precision = lcs_len / len(s1) if len(s1) > 0 else 0
        recall = lcs_len / len(s2) if len(s2) > 0 else 0
        if precision + recall > 0:
            fmeasure = (2 * precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0
        rouge_l_scores.append(fmeasure)

    return rouge_l_scores


def _evalscope_executable() -> str:
    evalscope_bin = os.environ.get("EVALSCOPE_BIN")
    if evalscope_bin:
        return evalscope_bin

    evalscope_bin = shutil.which("evalscope")
    if evalscope_bin:
        return evalscope_bin

    venv_dir = "/tmp/evalscope-perf"
    evalscope_bin = os.path.join(venv_dir, "bin", "evalscope")
    if os.path.exists(evalscope_bin):
        return evalscope_bin

    subprocess.run(
        ["python3", "-m", "uv", "venv", "--seed", "--clear", venv_dir],
        check=True,
    )
    subprocess.run(
        [
            "python3",
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            os.path.join(venv_dir, "bin", "python"),
            "evalscope[perf]",
        ],
        check=True,
    )
    return evalscope_bin


def _iter_evalscope_scores(value):
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {
                "score",
                "accuracy",
                "acc",
                "averageaccuracy",
            } and isinstance(item, (int, float)):
                yield float(item)
            else:
                yield from _iter_evalscope_scores(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_evalscope_scores(item)


def _parse_evalscope_stdout(output: str) -> Optional[float]:
    scores = []
    for line in output.splitlines():
        if "|" not in line or "Score" in line or "===" in line or "---" in line:
            continue
        columns = [col.strip() for col in line.strip().strip("|").split("|")]
        if not columns:
            continue
        try:
            scores.append(float(columns[-1]))
        except ValueError:
            continue
    return sum(scores) / len(scores) if scores else None


def _load_evalscope_score(work_dir: str, output: str) -> float:
    scores = []
    reports_dir = os.path.join(work_dir, "reports")
    if os.path.isdir(reports_dir):
        for path in Path(reports_dir).rglob("*.json"):
            try:
                with open(path) as f:
                    scores.extend(_iter_evalscope_scores(json.load(f)))
            except (OSError, json.JSONDecodeError):
                continue

    if scores:
        return sum(scores) / len(scores)

    score = _parse_evalscope_stdout(output)
    if score is not None:
        return score

    raise RuntimeError(f"Unable to parse evalscope score from {work_dir}")


def run_evalscope(
    *,
    base_url: str,
    model: str,
    dataset: str,
    limit: Optional[int] = None,
    eval_batch_size: int = 16,
    generation_config: Optional[dict] = None,
    dataset_args: Optional[dict] = None,
) -> dict:
    work_dir = tempfile.mkdtemp(prefix=f"evalscope-{dataset}-", dir="/tmp")
    api_url = base_url.rstrip("/")
    if not api_url.endswith("/v1"):
        api_url += "/v1"
    cmd = [
        _evalscope_executable(),
        "eval",
        "--model",
        model,
        "--api-url",
        api_url,
        "--api-key",
        "EMPTY_TOKEN",
        "--datasets",
        dataset,
        "--eval-batch-size",
        str(eval_batch_size),
        "--work-dir",
        work_dir,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if generation_config:
        cmd.extend(["--generation-config", json.dumps(generation_config)])
    if dataset_args:
        cmd.extend(["--dataset-args", json.dumps(dataset_args)])

    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)
    score = _load_evalscope_score(work_dir, result.stdout)
    return {"score": score, "accuracy": score, "work_dir": work_dir}


def write_github_step_summary(content):
    if not os.environ.get("GITHUB_STEP_SUMMARY"):
        logging.warning("GITHUB_STEP_SUMMARY environment variable not set")
        return

    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(content)
