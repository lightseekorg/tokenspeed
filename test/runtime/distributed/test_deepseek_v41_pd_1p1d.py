"""
DeepSeek V4.1 Flash PD (1-prefill-1-decode) disaggregation smoke test.

Launches the PD serve script with DSpark on both roles and validates output
quality via /v1/chat/completions with known prompts, plus that the decode
node's draft proposals are being accepted (its DSpark context arrived with
the transferred KV).

Usage:
    pytest test/runtime/distributed/test_deepseek_v41_pd_1p1d.py -v
"""

import os
import subprocess
import sys
import threading
import time

import pytest
import requests

SERVE_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "ci_system",
    "serve_deepseek_v41_flash_pd_1p1d.sh",
)
LB_PORT = int(os.environ.get("LB_PORT", "18345"))
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL)
STARTUP_TIMEOUT = int(os.environ.get("PD_STARTUP_TIMEOUT", "2400"))
LOG_DIR = os.environ.get("PD_CI_LOG_DIR", ".ci-artifacts/pd-deepseek-v41-flash-1p1d")
ENABLE_DSPARK = os.environ.get("ENABLE_DSPARK", "1") == "1"

QUALITY_CHECKS = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Reply in one word.",
            }
        ],
        "expected": "Paris",
        "max_tokens": 64,
    },
    {
        "messages": [
            {"role": "user", "content": "What is 17 * 23? Reply with just the number."}
        ],
        "expected": "391",
        "max_tokens": 64,
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Name the largest planet in our solar system in one word.",
            }
        ],
        "expected": "Jupiter",
        "max_tokens": 64,
    },
    {
        # Long enough for several DSpark blocks; every draft token the decode
        # node accepts came from context rows it received from the prefill node.
        "messages": [
            {
                "role": "user",
                "content": "List the first twelve prime numbers, separated by commas.",
            }
        ],
        "expected": "37",
        "max_tokens": 128,
    },
]


def _wait_for_server(proc: subprocess.Popen, port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            resp = requests.get(url, timeout=5)
            # Require a real model list so a port squatter cannot fake readiness.
            if resp.status_code == 200 and "data" in resp.json():
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def _chat(port: int, messages, max_tokens=64, temperature=0):
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"thinking": False},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _tail_file(path: str, stop_event: threading.Event, prefix: str = "[decode] "):
    while not os.path.exists(path):
        if stop_event.wait(1):
            return
    with open(path) as f:
        while not stop_event.is_set():
            line = f.readline()
            if line:
                sys.stdout.write(f"{prefix}{line}")
                sys.stdout.flush()
            else:
                stop_event.wait(0.5)


@pytest.fixture(scope="session")
def pd_server():
    env = os.environ.copy()
    env.setdefault("LB_PORT", str(LB_PORT))
    proc = subprocess.Popen(
        ["bash", os.path.abspath(SERVE_SCRIPT)],
        env=env,
    )
    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=_tail_file,
        args=(os.path.join(LOG_DIR, "decode.log"), stop_event),
        daemon=True,
    )
    tail_thread.start()
    try:
        if not _wait_for_server(proc, LB_PORT, STARTUP_TIMEOUT):
            rc = proc.poll()
            pytest.fail(
                f"PD server did not become ready within {STARTUP_TIMEOUT}s"
                f" (exit_code={rc})"
            )
        time.sleep(5)
        yield proc
    finally:
        from tokenspeed.runtime.utils.process import kill_process_tree

        kill_process_tree(proc.pid)
        stop_event.set()
        tail_thread.join(timeout=5)


def test_quality(pd_server):
    assert pd_server.poll() is None, "PD server exited unexpectedly"
    failures = []
    accepted = 0
    for i, check in enumerate(QUALITY_CHECKS):
        data = _chat(LB_PORT, check["messages"], max_tokens=check["max_tokens"])
        content = data["choices"][0]["message"]["content"]
        prompt = check["messages"][0]["content"]
        print(f"\n[Q{i}] {prompt}")
        print(f"[A{i}] {content}")
        print(f"[U{i}] {data.get('usage')}")
        if check["expected"] not in content:
            failures.append(
                f"  check[{i}]: expected {check['expected']!r} in {content!r}"
            )
        details = (data.get("usage") or {}).get("completion_tokens_details") or {}
        accepted += int(details.get("accepted_prediction_tokens") or 0)
    assert not failures, "Quality check failures:\n" + "\n".join(failures)
    if ENABLE_DSPARK:
        assert accepted > 0, "DSpark accepted no draft tokens on the decode node"
