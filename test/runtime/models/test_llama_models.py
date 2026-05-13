"""Dense Llama tests.

Covers the ``LlamaForCausalLM`` architecture (Llama-2 / 3 / 3.1 / 3.2 dense
checkpoints) registered by ``tokenspeed.runtime.models.llama``. The sibling
``LlamaForCausalLMMoE`` and ``LlamaForCausalLMEagle3`` variants have their
own test coverage elsewhere.

Launches one tokenspeed server per config and validates output quality via
``/v1/chat/completions`` with known prompts and expected content substrings
— the same pattern used by ``test_kimi_models.py``.

Usage:
    cd test/runtime
    python3 -m unittest models.test_llama_models -v
    python3 -m unittest models.test_llama_models.TestLlamaDense.test_base -v

Environment (all optional):
    LLAMA_DENSE_MODEL         HF model id or local path; default is the
                              ungated ``unsloth/Llama-3.2-1B-Instruct``
                              so the test works without an HF gated-repo token.
    LLAMA_DENSE_WORLD_SIZE    GPU count (default: 1)
"""

# CI Registration (parsed via AST, runtime no-op)
import os
import sys
import time
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="runtime-1gpu")

import subprocess

import requests

from tokenspeed.runtime.utils.process import kill_process_tree

MODEL = os.environ.get("LLAMA_DENSE_MODEL", "unsloth/Llama-3.2-1B-Instruct")
WORLD_SIZE = int(os.environ.get("LLAMA_DENSE_WORLD_SIZE", "1"))
TIMEOUT = 600

_server_port = 23100


def _next_server_port() -> int:
    global _server_port
    port = _server_port
    _server_port += 1
    return port


# ── Server lifecycle ─────────────────────────────────────────────────


def _serve_server(port: int, extra_args=()) -> subprocess.Popen:
    # Use ``python -m tokenspeed.cli serve`` instead of the ``ts`` console
    # script — the CI runner doesn't always have the entrypoint on PATH
    # (e.g. when tests are executed against a source tree rather than a
    # wheel install), and the module form works unconditionally.
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
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.5",
        "--max-total-tokens",
        "8192",
    ] + list(extra_args)
    return subprocess.Popen(cmd, env=os.environ.copy())


def _wait_for_server(port: int, timeout: int = TIMEOUT) -> bool:
    url = f"http://127.0.0.1:{port}/readiness"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _chat(port: int, messages, max_tokens=64, temperature=0):
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
# A 1B model answers these reliably at temperature=0. We only check for
# the expected substring — exact wording varies by decoding budget.

QUALITY_CHECKS = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Reply in one word.",
            }
        ],
        "expected": "Paris",
        "max_tokens": 32,
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ],
        "expected": "4",
        "max_tokens": 32,
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Name the largest planet in our solar system in one word.",
            }
        ],
        "expected": "Jupiter",
        "max_tokens": 32,
    },
]


# ── Tests ────────────────────────────────────────────────────────────


class TestLlamaRegistry(unittest.TestCase):
    """Cheap, no-GPU sanity check that the dense Llama class is wired up."""

    def test_registered(self):
        from tokenspeed.runtime.models.registry import ModelRegistry

        supported = ModelRegistry.get_supported_archs()
        self.assertIn(
            "LlamaForCausalLM",
            supported,
            "Dense LlamaForCausalLM should be in the model registry alongside "
            "LlamaForCausalLMMoE and LlamaForCausalLMEagle3.",
        )

    def test_resolves_to_dense_class(self):
        from tokenspeed.runtime.models.llama import LlamaForCausalLM
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["LlamaForCausalLM"])
        self.assertIs(
            cls,
            LlamaForCausalLM,
            "LlamaForCausalLM should resolve to tokenspeed.runtime.models.llama."
            "LlamaForCausalLM, not the MoE or Eagle3 variants.",
        )
        self.assertEqual(arch, "LlamaForCausalLM")


class TestLlamaDense(unittest.TestCase):
    """Quality checks against a live server loading a dense Llama checkpoint."""

    def _run_quality_checks(self, extra_args=()):
        port = _next_server_port()
        proc = _serve_server(port, extra_args)
        try:
            if not _wait_for_server(port):
                self.fail(f"Server did not start within {TIMEOUT}s")

            for i, q in enumerate(QUALITY_CHECKS):
                data = _chat(port, q["messages"], max_tokens=q["max_tokens"])
                content = data["choices"][0]["message"]["content"] or ""
                self.assertIn(
                    q["expected"],
                    content,
                    f"check {i}: expected {q['expected']!r} in {content!r}",
                )
        finally:
            kill_process_tree(proc.pid)

    def test_base(self):
        """Dense Llama-3.2-1B-Instruct with default attention backend."""
        self._run_quality_checks()


if __name__ == "__main__":
    unittest.main()
