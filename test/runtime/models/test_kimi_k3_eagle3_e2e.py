"""Opt-in TP16 Kimi-K3 + EAGLE3 acceptance smoke.

This is intentionally opt-in because it needs the real K3 target and a
compatible serving-format Eagle3 draft.  It validates the production K3 chat
path and fails if speculative decoding never accepts a draft token.

Run on a 16-GPU PP1 node:

    KIMI_K3_EAGLE3_E2E=1 \
    KIMI_K3_MODEL=/models/Kimi-K3 \
    KIMI_K3_EAGLE3_DRAFT=/models/kimi-k3-eagle3 \
    python3 -m unittest models.test_kimi_k3_eagle3_e2e -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import requests

from tokenspeed.runtime.utils.process import kill_process_tree

ENABLED = os.environ.get("KIMI_K3_EAGLE3_E2E") == "1"
MODEL = os.environ.get("KIMI_K3_MODEL", "moonshotai/Kimi-K3")
DRAFT = os.environ.get("KIMI_K3_EAGLE3_DRAFT")
WORLD_SIZE = int(os.environ.get("KIMI_K3_EAGLE3_WORLD_SIZE", "16"))
PORT = int(os.environ.get("KIMI_K3_EAGLE3_PORT", "22080"))
TIMEOUT = int(os.environ.get("KIMI_K3_EAGLE3_TIMEOUT", "3600"))


@unittest.skipUnless(
    ENABLED and DRAFT,
    "set KIMI_K3_EAGLE3_E2E=1 and KIMI_K3_EAGLE3_DRAFT on a TP16 K3 host",
)
class TestKimiK3Eagle3E2E(unittest.TestCase):
    def _wait_for_ready(self) -> None:
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            try:
                response = requests.get(f"http://127.0.0.1:{PORT}/readiness", timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        self.fail("K3 EAGLE3 server did not become ready")

    def test_k3_eagle3_accepts_nonzero_drafts(self):
        with tempfile.TemporaryDirectory(prefix="tokenspeed-k3-eagle3-") as temp_dir:
            log_path = Path(temp_dir) / "server.log"
            with log_path.open("w") as log_file:
                command = [
                    sys.executable,
                    "-m",
                    "tokenspeed.cli",
                    "serve",
                    "--model",
                    MODEL,
                    "--served-model-name",
                    "kimi-k3",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(PORT),
                    "--world-size",
                    str(WORLD_SIZE),
                    "--tensor-parallel-size",
                    str(WORLD_SIZE),
                    "--dense-tp-size",
                    str(WORLD_SIZE),
                    "--moe-tp-size",
                    str(WORLD_SIZE),
                    "--data-parallel-size",
                    "1",
                    "--trust-remote-code",
                    "--max-model-len",
                    "32768",
                    "--kv-cache-dtype",
                    "fp8",
                    "--gpu-memory-utilization",
                    "0.94",
                    "--max-num-seqs",
                    "32",
                    "--mm-encoder-tp-mode",
                    "data",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model-path",
                    DRAFT,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--speculative-eagle-topk",
                    "1",
                    "--eagle3-layers-to-capture",
                    "2,46,90",
                    "--reasoning-parser",
                    "kimi_k3",
                    "--default-chat-template-kwargs",
                    '{"thinking":true,"thinking_effort":"max"}',
                    "--enable-log-request-stats",
                    "--disable-kvstore",
                    "--force-deterministic-rsag",
                    "--enforce-eager",
                ]
                proc = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                try:
                    self._wait_for_ready()
                    response = requests.post(
                        f"http://127.0.0.1:{PORT}/v1/chat/completions",
                        json={
                            "model": "kimi-k3",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant.",
                                },
                                {
                                    "role": "user",
                                    "content": "Compute 37 * 29 and verify it.",
                                },
                            ],
                            "chat_template_kwargs": {
                                "thinking": True,
                                "thinking_effort": "max",
                            },
                            "temperature": 0,
                            "max_tokens": 512,
                        },
                        timeout=TIMEOUT,
                    )
                    response.raise_for_status()
                    message = response.json()["choices"][0]["message"]
                    content = message.get("content", "")
                    self.assertIn("1073", content)
                    self.assertFalse(
                        any(
                            tag in content
                            for tag in ("<|open|>", "<|sep|>", "<|close|>")
                        )
                    )

                    deadline = time.time() + 60
                    while time.time() < deadline:
                        log_file.flush()
                        rates = [
                            float(rate)
                            for rate in re.findall(
                                r"acc_rate=([0-9]+(?:\.[0-9]+)?)",
                                log_path.read_text(errors="replace"),
                            )
                        ]
                        if rates:
                            break
                        time.sleep(1)
                    self.assertTrue(
                        rates, "no EAGLE3 acceptance statistic in server log"
                    )
                    self.assertGreater(
                        max(rates), 0.0, "EAGLE3 accepted no draft tokens"
                    )
                finally:
                    kill_process_tree(proc.pid)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
