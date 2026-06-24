"""End-to-end tests for Qwen3.5 VLM image requests."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

# /test on sys.path so "ci_system.ci_register" resolves from test/ci_system/.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=1200,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*", "h100-*"],
    disabled_on_runners_reason="Qwen3.5 FP8 VLM requires NVIDIA Blackwell.",
)

MODEL = os.environ.get("QWEN35_VLM_E2E_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8")
PORT = int(os.environ.get("QWEN35_VLM_E2E_PORT", "23220"))
LOG_DIR = Path(os.environ.get("QWEN35_VLM_E2E_LOG_DIR", ".ci-artifacts/qwen35-vlm-e2e"))
SERVED_MODEL_NAME = "qwen35-vlm-e2e"
SERVER_LAUNCH_TIMEOUT = 900
REQUEST_TIMEOUT = 300


def _image_data_url() -> str:
    from PIL import Image

    image = Image.new("RGB", (256, 256), color=(220, 20, 20))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _serve_server(log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "tokenspeed.cli",
        "serve",
        "--gateway-startup-timeout",
        "300",
        "--model",
        MODEL,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--world-size",
        "1",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "1",
        "--max-prefill-tokens",
        "4096",
        "--chunked-prefill-size",
        "4096",
        "--gpu-memory-utilization",
        "0.85",
        "--attention-backend",
        "trtllm",
        "--moe-backend",
        "flashinfer_trtllm",
        "--sampling-backend",
        "greedy",
        "--quantization",
        "fp8",
        "--mm-attention-backend",
        "fa4",
        "--policy",
        "random",
        "--disable-kvstore",
        "--disable-prefill-graph",
        "--trust-remote-code",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    proc._log_file = log_f
    return proc


def _wait_for_server(timeout: int = SERVER_LAUNCH_TIMEOUT) -> bool:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if (
                requests.get(
                    f"http://127.0.0.1:{PORT}/readiness", timeout=5
                ).status_code
                == 200
            ):
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


class TestQwen35VlmE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tokenspeed_kernel.platform import current_platform
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"TokenSpeed kernel deps unavailable: {exc}")

        if not current_platform().is_blackwell:
            raise unittest.SkipTest("Qwen3.5 FP8 VLM requires NVIDIA Blackwell")

    def test_image_chat_completion(self):
        import requests

        log_path = LOG_DIR / "server.log"
        proc = _serve_server(log_path)
        try:
            if not _wait_for_server():
                tail = (
                    log_path.read_text(errors="replace")[-4000:]
                    if log_path.exists()
                    else ""
                )
                self.fail(f"server did not become ready; log tail:\n{tail}")

            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in one short sentence.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_url()},
                            },
                        ],
                    }
                ],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": False,
            }
            deadline = time.time() + REQUEST_TIMEOUT
            resp = None
            while time.time() < deadline:
                resp = requests.post(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    json=payload,
                    timeout=60,
                )
                if resp.status_code < 500:
                    break
                time.sleep(5)
            assert resp is not None
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content") or ""
            self.assertTrue(content.strip(), f"empty VLM response: {data}")
            self.assertGreater(data.get("usage", {}).get("prompt_tokens", 0), 0)
        finally:
            from tokenspeed.runtime.utils.process import kill_process_tree

            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            proc._log_file.close()
            time.sleep(5)


if __name__ == "__main__":
    unittest.main()
