"""
GPT-OSS tests — pure TP, DP+EP, COMBINE modes, and Eagle3 speculative.

Launches a tokenspeed server per config and validates output quality
via the /v1/chat/completions API with known prompts and expected content.

Usage:
    cd test/runtime
    python3 -m unittest models.test_gpt_oss -v
    python3 -m unittest models.test_gpt_oss.TestGptOss.test_pure_tp -v

Environment (all optional):
    GPT_OSS_MODEL       model path  (default: openai/gpt-oss-120b)
    GPT_OSS_WORLD_SIZE  num GPUs    (default: 4)
"""

import dataclasses
import os
import subprocess
import time
import unittest
from typing import Optional

import requests

from tokenspeed.runtime.utils.process import kill_process_tree

MODEL = os.environ.get("GPT_OSS_MODEL", "openai/gpt-oss-120b")
WORLD_SIZE = int(os.environ.get("GPT_OSS_WORLD_SIZE", "4"))
TIMEOUT = 600

_server_port = 21000
_dist_port = 5000


def _next_server_port() -> int:
    global _server_port
    port = _server_port
    _server_port += 1
    return port


def _next_dist_port() -> int:
    global _dist_port
    port = _dist_port
    _dist_port += 100
    return port


# ── Server lifecycle ─────────────────────────────────────────────────


def _api_server(port: int, extra_args=()) -> subprocess.Popen:
    cmd = [
        "tokenspeed-serve",
        "--model",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--world-size",
        str(WORLD_SIZE),
        "--moe-backend",
        "flashinfer_mxfp4",
    ] + list(extra_args)
    return subprocess.Popen(cmd, env=os.environ.copy())


def _wait_for_server(port: int, timeout: int = TIMEOUT) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _chat(port: int, messages, max_tokens=32, temperature=0):
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Quality prompts ──────────────────────────────────────────────────

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
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ],
        "expected": "4",
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
]


# ── Mesh configs ─────────────────────────────────────────────────────


@dataclasses.dataclass
class MeshCase:
    name: str
    extra_args: tuple = ()


MESH_CASES = {
    "pure_tp": MeshCase("pure_tp"),
    "dp_ep": MeshCase(
        "dp_ep",
        (
            "--data-parallel-size",
            "4",
            "--ep-size",
            "4",
            "--dist-init-addr",
            f"127.0.0.1:{_next_dist_port()}",
        ),
    ),
    "combine_dp2_tp2": MeshCase(
        "combine_dp2_tp2",
        (
            "--data-parallel-size",
            "2",
            "--ep-size",
            "4",
            "--dist-init-addr",
            f"127.0.0.1:{_next_dist_port()}",
        ),
    ),
    "combine_dense_moe": MeshCase(
        "combine_dense_moe",
        (
            "--data-parallel-size",
            "2",
            "--dist-init-addr",
            f"127.0.0.1:{_next_dist_port()}",
        ),
    ),
    "eagle3": MeshCase(
        "eagle3",
        (
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            os.environ.get(
                "GPT_OSS_DRAFT_MODEL", "nvidia/gpt-oss-120b-Eagle3-long-context"
            ),
            "--speculative-num-steps",
            "3",
        ),
    ),
}


# ── Tests ────────────────────────────────────────────────────────────


class TestGptOss(unittest.TestCase):

    def _run_quality_checks(self, case: MeshCase):
        port = _next_server_port()
        proc = _api_server(port, case.extra_args)
        try:
            if not _wait_for_server(port):
                self.fail(f"[{case.name}] Server did not start within {TIMEOUT}s")

            for i, q in enumerate(QUALITY_CHECKS):
                data = _chat(port, q["messages"], max_tokens=q["max_tokens"])
                content = data["choices"][0]["message"]["content"]
                self.assertIn(
                    q["expected"],
                    content,
                    f"[{case.name}] check {i}: "
                    f'expected {q["expected"]!r} in {content!r}',
                )
        finally:
            kill_process_tree(proc.pid)

    def test_pure_tp(self):
        self._run_quality_checks(MESH_CASES["pure_tp"])

    def test_dp_ep(self):
        self._run_quality_checks(MESH_CASES["dp_ep"])

    def test_combine_dp2_tp2(self):
        self._run_quality_checks(MESH_CASES["combine_dp2_tp2"])

    def test_combine_dense_moe(self):
        self._run_quality_checks(MESH_CASES["combine_dense_moe"])

    def test_eagle3(self):
        self._run_quality_checks(MESH_CASES["eagle3"])


if __name__ == "__main__":
    unittest.main()
