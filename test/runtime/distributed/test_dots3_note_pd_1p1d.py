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

"""Same-host dots3-note 1P1D integration smokes, not a model accuracy benchmark.

Run from the repository root with the runtime venv and source PYTHONPATH active:
    MODEL_PATH=/path/to/checkpoint SPECULATIVE_ALGORITHM=NONE \
        python -m pytest --noconftest -v -s \
        test/runtime/distributed/test_dots3_note_pd_1p1d.py
    MODEL_PATH=/path/to/checkpoint SPECULATIVE_ALGORITHM=MTP SPECULATIVE_NUM_STEPS=3 \
        python -m pytest --noconftest -v -s \
        test/runtime/distributed/test_dots3_note_pd_1p1d.py

SPECULATIVE_ALGORITHM=NONE|MTP is required, with no default. MTP requires explicit
SPECULATIVE_NUM_STEPS >= 1 (use 1 or 3 for MTP1/MTP3). Both workers use verify
width steps+1, top-k 1, the same checkpoint and FP8 draft quantization. NONE
rejects steps; other SPECULATIVE_* settings and launcher CLI args are rejected.
No checkpoint is downloaded; an unset MODEL_PATH skips only the live smokes.
The launcher uses TP4 on GPUs 0-3 and TP4 on GPUs 4-7 by default. Its environment
overrides are preserved. Set CHUNKED_PREFILL_SIZE=1024 explicitly for multi-chunk
coverage; the default 8192 matches the serving baseline. All live tests share
one server boot. MTP prime-list responses must report positive
usage.completion_tokens_details.accepted_prediction_tokens through SMG; missing
telemetry fails, not skips. This proves acceptance, not an acceptance-rate target.

Offline checks (no servers, sockets or GPU probes, even with MODEL_PATH set):
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
        python -m pytest --noconftest -q -p no:cacheprovider \
        test/runtime/distributed/test_dots3_note_pd_1p1d.py -k offline
--noconftest avoids the parent conftest's CUDA availability probe.

Budgets: PD_STARTUP_TIMEOUT=2400 + GATEWAY_STARTUP_TIMEOUT=600 for readiness;
PD_REQUEST_TIMEOUT=600 per complete HTTP request; WORKER_SHUTDOWN_TIMEOUT=30
for graceful group shutdown. The three live tests make 2, 2 and 3 requests, with the
last three concurrent. A passing long-prompt smoke proves a response to a
>2048-token input, not kernel dispatch, prefix-hit rate or bytewise KV correctness.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import os
import re
import signal
import subprocess
from collections import deque
from pathlib import Path

import aiohttp
import psutil
import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVE_SCRIPT = ROOT / "test/ci_system/serve_dots3_note_pd_1p1d.sh"
MODEL_PATH = os.environ.get("MODEL_PATH")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "dots3-note")
BASE_URL = f"http://127.0.0.1:{int(os.environ.get('LB_PORT', '18345'))}"
STARTUP_TIMEOUT = float(os.environ.get("PD_STARTUP_TIMEOUT", "2400"))
GATEWAY_TIMEOUT = float(os.environ.get("GATEWAY_STARTUP_TIMEOUT", "600"))
REQUEST_TIMEOUT = float(os.environ.get("PD_REQUEST_TIMEOUT", "600"))
SHUTDOWN_TIMEOUT = float(os.environ.get("WORKER_SHUTDOWN_TIMEOUT", "30"))
PRIMES_PROMPT = (
    "List the first twelve prime numbers. Reply only with the comma-separated numbers."
)
PRIMES = ["2", "3", "5", "7", "11", "13", "17", "19", "23", "29", "31", "37"]


async def _wait_for_server(proc):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
        while True:
            assert proc.poll() is None, f"PD launcher exited with {proc.returncode}"
            try:
                async with client.get(f"{BASE_URL}/readiness") as response:
                    ready = response.status == 200
                if ready:
                    async with client.get(f"{BASE_URL}/v1/models") as response:
                        if response.status == 200:
                            data = await response.json()
                            if any(
                                model.get("id") == SERVED_MODEL_NAME
                                for model in data.get("data", [])
                            ):
                                return
            except (aiohttp.ClientError, TimeoutError, ValueError):
                pass
            await asyncio.sleep(1)


async def _wait_for_startup(proc):
    await asyncio.wait_for(
        _wait_for_server(proc), timeout=STARTUP_TIMEOUT + GATEWAY_TIMEOUT + 10
    )


def _stop_launcher(proc):
    # Snapshot owned sessions before TERM; preserve their IDs even if a parent
    # exits before its TP children. Never signal the pytest runner's group.
    groups = {proc.pid}
    with contextlib.suppress(psutil.Error):
        for child in psutil.Process(proc.pid).children(recursive=True):
            with contextlib.suppress(ProcessLookupError):
                if os.getpgid(child.pid) == child.pid:
                    groups.add(child.pid)
    try:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=SHUTDOWN_TIMEOUT + 30)
    finally:
        for group in groups:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
        proc.wait(timeout=5)


@pytest.fixture(scope="module")
def pd_server():
    if not MODEL_PATH:
        pytest.skip("Set MODEL_PATH to a local dots3-note checkpoint")
    model = Path(MODEL_PATH).resolve()
    assert (model / "config.json").is_file(), "MODEL_PATH must contain config.json"
    env = {**os.environ, "MODEL_PATH": str(model)}
    log_dir = Path(env.get("PD_CI_LOG_DIR", ".ci-artifacts/pd-dots3-note-1p1d"))
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    env["PD_CI_LOG_DIR"] = str(log_dir)

    proc = None
    pending_signal = None

    def terminate(signum, _frame):
        nonlocal pending_signal
        pending_signal = signum
        # Do not interrupt Popen before its child has been recorded.
        if proc is not None:
            raise SystemExit(128 + signum)

    previous_handlers = {
        signum: signal.signal(signum, terminate)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        proc = subprocess.Popen(
            ["bash", str(SERVE_SCRIPT)], cwd=ROOT, env=env, start_new_session=True
        )
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal)
        asyncio.run(_wait_for_startup(proc))
        yield proc
    finally:
        # Repeated cancellation must not interrupt group cleanup.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if proc is not None:
                _stop_launcher(proc)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        for name in ("prefill", "decode", "lb"):
            path = log_dir / f"{name}.log"
            if path.is_file():
                with path.open(errors="replace") as stream:
                    print(f"\n[{path}]\n{''.join(deque(stream, maxlen=30))}")


async def _chat(proc, prompt: str, *, max_tokens: int):
    assert proc.poll() is None, f"PD launcher exited with {proc.returncode}"
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    ) as client:
        async with client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ) as response:
            assert response.status == 200, await response.text()
            data = await response.json()
    assert proc.poll() is None, f"PD launcher exited with {proc.returncode}"
    assert not data.get("error"), data
    choice = data["choices"][0]
    content = choice["message"]["content"]
    assert isinstance(content, str) and content.strip(), data
    assert choice["finish_reason"] == "stop", data
    assert 0 < data["usage"]["completion_tokens"] <= max_tokens, data
    print(f"\n[content] {content}\n[usage] {data['usage']}")
    return data


def _assert_primes(data):
    assert re.findall(r"\d+", data["choices"][0]["message"]["content"]) == PRIMES, data
    # A single bootstrap token from P is not sufficient evidence of decode work.
    assert data["usage"]["completion_tokens"] > 1, data
    if os.environ["SPECULATIVE_ALGORITHM"] == "MTP":
        details = data["usage"].get("completion_tokens_details") or {}
        accepted = details.get("accepted_prediction_tokens")
        assert (
            type(accepted) is int and 0 < accepted < data["usage"]["completion_tokens"]
        ), f"MTP requires positive accepted_prediction_tokens from SMG: {data['usage']}"


def test_short_smokes(pd_server):
    _assert_primes(asyncio.run(_chat(pd_server, PRIMES_PROMPT, max_tokens=128)))
    data = asyncio.run(
        _chat(pd_server, "What is 17 + 25? Give only the answer.", max_tokens=32)
    )
    assert re.findall(r"\d+", data["choices"][0]["message"]["content"]) == ["42"], data


def test_repeated_long_prefix(pd_server):
    prompt = (
        "Ignore the filler below and answer the final question.\n"
        + "This sentence is only filler text.\n" * 400
        + f"\n{PRIMES_PROMPT}"
    )
    for _ in range(2):
        data = asyncio.run(_chat(pd_server, prompt, max_tokens=128))
        assert 2048 < data["usage"]["prompt_tokens"] <= 8192 - 128, data["usage"]
        _assert_primes(data)


def test_concurrent_requests(pd_server):
    async def run_batch():
        return await asyncio.gather(
            *(
                _chat(pd_server, f"Request {label}. {PRIMES_PROMPT}", max_tokens=128)
                for label in ("Red", "Green", "Blue")
            )
        )

    for data in asyncio.run(run_batch()):
        _assert_primes(data)


def test_launcher_contract_offline():
    # Extract only the pure argument builder; never execute the launcher main.
    source = SERVE_SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_speculative_args"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[builder], type_ignores=[]), str(SERVE_SCRIPT), "exec"),
        namespace,
    )
    args = namespace["_speculative_args"]
    model = str(SERVE_SCRIPT)  # A file cannot be mistaken for a checkpoint directory.
    assert args({"SPECULATIVE_ALGORITHM": "NONE"}, model) == []
    for steps in (1, 3):
        assert args(
            {"SPECULATIVE_ALGORITHM": "MTP", "SPECULATIVE_NUM_STEPS": str(steps)}, model
        ) == [
            "--speculative-algorithm",
            "MTP",
            "--speculative-num-steps",
            str(steps),
            "--speculative-num-draft-tokens",
            str(steps + 1),
            "--speculative-eagle-topk",
            "1",
            "--speculative-draft-model-path",
            model,
            "--speculative-draft-model-quantization",
            "fp8",
        ]
    invalid = [({}, "SPECULATIVE_ALGORITHM")]
    invalid += [
        ({"SPECULATIVE_ALGORITHM": algorithm}, "SPECULATIVE_ALGORITHM")
        for algorithm in ("", "mtp", "EAGLE3")
    ]
    invalid += [({"SPECULATIVE_ALGORITHM": "MTP"}, "SPECULATIVE_NUM_STEPS")]
    invalid += [
        (
            {"SPECULATIVE_ALGORITHM": "MTP", "SPECULATIVE_NUM_STEPS": steps},
            "SPECULATIVE_NUM_STEPS",
        )
        for steps in ("", "0", "-1", "1.5", "abc")
    ]
    invalid += [
        (
            {"SPECULATIVE_ALGORITHM": "NONE", "SPECULATIVE_NUM_STEPS": steps},
            "SPECULATIVE_NUM_STEPS",
        )
        for steps in ("", "1")
    ]
    invalid += [
        (
            {"SPECULATIVE_ALGORITHM": "MTP", "SPECULATIVE_NUM_STEPS": "3", key: value},
            "Unsupported speculative settings",
        )
        for key, value in (
            ("SPECULATIVE_NUM_DRAFT_TOKENS", "4"),
            ("SPECULATIVE_EAGLE_TOPK", "1"),
            ("SPECULATIVE_DRAFT_MODEL_PATH", model),
            ("SPECULATIVE_DRAFT_MODEL_QUANTIZATION", "fp8"),
            ("SPECULATIVE_CONFIG", ""),
        )
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SPECULATIVE_")
    }
    env.update(MODEL_PATH=model, PYTHONDONTWRITEBYTECODE="1")
    for config, message in invalid:
        with pytest.raises(ValueError, match=message):
            args(config, model)
        result = subprocess.run(
            ["bash", str(SERVE_SCRIPT)],
            env={**env, **config},
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0 and message in result.stderr, result.stderr
    result = subprocess.run(
        ["bash", str(SERVE_SCRIPT), "--enforce-eager"],
        env={**env, "SPECULATIVE_ALGORITHM": "NONE"},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2 and "not positional arguments" in result.stderr


def test_mtp_acceptance_offline(monkeypatch):
    data = {
        "choices": [{"message": {"content": ", ".join(PRIMES)}}],
        "usage": {"completion_tokens": 43},
    }
    monkeypatch.setenv("SPECULATIVE_ALGORITHM", "NONE")
    _assert_primes(data)
    monkeypatch.setenv("SPECULATIVE_ALGORITHM", "MTP")
    with pytest.raises(AssertionError, match="accepted_prediction_tokens"):
        _assert_primes(data)
    for accepted in (None, 0, -1, 43, "1", True):
        data["usage"]["completion_tokens_details"] = {
            "accepted_prediction_tokens": accepted
        }
        with pytest.raises(AssertionError, match="accepted_prediction_tokens"):
            _assert_primes(data)
    data["usage"]["completion_tokens_details"]["accepted_prediction_tokens"] = 1
    _assert_primes(data)
