"""
Kimi K2.5 tests — base and EAGLE3 speculative.

Launches a tokenspeed server per config and validates output quality
via the /v1/chat/completions API with known prompts and expected content.

Usage:
    cd test/runtime
    python3 -m unittest models.test_kimi_models -v
    python3 -m unittest models.test_kimi_models.TestKimiK25.test_base -v
    python3 -m unittest models.test_kimi_models.TestKimiK25.test_tsckpt_eagle3 -v
    python3 -m unittest models.test_kimi_models.TestKimiK25.test_nvckpt_eagle3 -v

Environment (all optional):
    KIMI_K25_MODEL            HF model id or path (default: nvidia/Kimi-K2.5-NVFP4)
    KIMI_K25_WORLD_SIZE       GPU count (default: 4)
    KIMI_K25_DRAFT_MODEL      EAGLE3 draft repo (default: lightseekorg/kimi-k2.5-eagle3)
    KIMI_K25_MLA_DRAFT_MODEL  MLA EAGLE3 draft repo (default: nvidia/Kimi-K2.5-Thinking-Eagle3)
"""

import dataclasses
import os
import subprocess
import sys
import time
import unittest

import requests

from tokenspeed.runtime.utils.process import kill_process_tree

MODEL = os.environ.get("KIMI_K25_MODEL", "nvidia/Kimi-K2.5-NVFP4")
WORLD_SIZE = int(os.environ.get("KIMI_K25_WORLD_SIZE", "4"))
DRAFT_MODEL = os.environ.get("KIMI_K25_DRAFT_MODEL", "lightseekorg/kimi-k2.5-eagle3")
MLA_DRAFT_MODEL = os.environ.get(
    "KIMI_K25_MLA_DRAFT_MODEL", "nvidia/Kimi-K2.5-Thinking-Eagle3"
)
TIMEOUT = 600

_server_port = 22000


def _next_server_port() -> int:
    global _server_port
    port = _server_port
    _server_port += 1
    return port


# ── Server lifecycle ─────────────────────────────────────────────────


def _serve_server(port: int, extra_args=()) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "tokenspeed.cli",
        "serve",
        "--model",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--world-size",
        str(WORLD_SIZE),
        "--trust-remote-code",
        "--max-model-len",
        "81920",
        "--quantization",
        "nvfp4",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "8",
        "--max-cudagraph-capture-size",
        "8",
        "--attn-tp-size",
        str(WORLD_SIZE),
        "--moe-tp-size",
        str(WORLD_SIZE),
        "--dense-tp-size",
        str(WORLD_SIZE),
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
    "base": MeshCase(
        "base",
        (
            "--attention-backend",
            "trtllm_mla",
        ),
    ),
    "tsckpt_eagle3": MeshCase(
        "tsckpt_eagle3",
        (
            "--attention-backend",
            "trtllm_mla",  # use trtllm_mla for eagle3 cases
            "--moe-backend",
            "flashinfer_trtllm",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
            "--speculative-num-steps",
            "3",
            "--drafter-attention-backend",
            "trtllm",
        ),
    ),
    "nvckpt_eagle3": MeshCase(
        "nvckpt_eagle3",
        (
            "--attention-backend",
            "trtllm_mla",
            "--moe-backend",
            "flashinfer_trtllm",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--max-prefill-tokens",
            "8192",
            "--chunked-prefill-size",
            "8192",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            MLA_DRAFT_MODEL,
            "--speculative-num-steps",
            "3",
            "--drafter-attention-backend",
            "trtllm_mla",
        ),
    ),
}


# ── Tests ────────────────────────────────────────────────────────────


class TestKimiK25(unittest.TestCase):

    def _run_quality_checks(self, case: MeshCase):
        port = _next_server_port()
        proc = _serve_server(port, case.extra_args)
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

    def test_base(self):
        """Kimi K2.5 with NVFP4 quantization."""
        self._run_quality_checks(MESH_CASES["base"])

    def test_tsckpt_eagle3(self):
        """Kimi K2.5 with EAGLE3 draft."""
        self._run_quality_checks(MESH_CASES["tsckpt_eagle3"])

    def test_nvckpt_eagle3(self):
        """Kimi K2.5 with MLA EAGLE3 draft (trtllm_mla drafter + FP8 KV cache)."""
        self._run_quality_checks(MESH_CASES["nvckpt_eagle3"])


if __name__ == "__main__":
    unittest.main()
