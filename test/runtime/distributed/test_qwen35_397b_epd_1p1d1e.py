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

"""
Qwen3.5-397B EPD (1-encode-1-prefill-1-decode) disaggregation smoke test.

Launches the EPD serve script (encode + prefill + decode workers behind the SMG
gateway) and validates that the vision path works end to end via the
/v1/chat/completions API: it posts a self-contained image (rendered with PIL, no
network fixture) plus a prompt and asserts the model reads it correctly. This is
the EPD analogue of test_qwen35_pd_1p1d.py -- a functional gate that the
encode->prefill->decode embedding transfer does not silently corrupt the image.

Usage:
    pytest test/runtime/distributed/test_qwen35_397b_epd_1p1d1e.py -v -s
"""

import base64
import io
import os
import subprocess
import sys
import threading
import time

import pytest
import requests
from PIL import Image, ImageDraw, ImageFont

SERVE_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "ci_system",
    "serve_qwen35_397b_nvfp4_epd_1p1d1e.sh",
)
LB_PORT = int(os.environ.get("LB_PORT", "12345"))
MODEL = os.environ.get("MODEL", "nvidia/Qwen3.5-397B-A17B-NVFP4")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL)
STARTUP_TIMEOUT = int(os.environ.get("EPD_STARTUP_TIMEOUT", "2400"))
LOG_DIR = os.environ.get("EPD_CI_LOG_DIR", ".ci-artifacts/epd-qwen35-397b-1p1d1e")


def _truetype_font(size: int):
    """A large TrueType face if the runner ships one, else None (the caller then
    renders with the bitmap default and upscales so glyphs stay legible)."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return None


def _image_data_uri(text: str, *, bg: str, fg: str = "black") -> str:
    """Render ``text`` centered on a solid ``bg`` canvas and return it as an
    OpenAI ``data:image/png;base64,...`` URI. Big, high-contrast glyphs so the
    smoke test exercises the real vision path without depending on a hosted
    fixture (CI has no guaranteed network)."""
    w, h = 512, 256
    font = _truetype_font(160)
    if font is not None:
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text(
            ((w - tw) / 2 - box[0], (h - th) / 2 - box[1]), text, fill=fg, font=font
        )
    else:
        # No TrueType face: draw with the tiny bitmap default, then upscale 8x
        # with nearest-neighbor so the glyphs are big and blocky but legible.
        small = Image.new("RGB", (w // 8, h // 8), bg)
        ImageDraw.Draw(small).text((4, 6), text, fill=fg)
        img = small.resize((w, h), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# (image, prompt, expected-substring) -- expected is matched case-insensitively.
QUALITY_CHECKS = [
    {
        "image": _image_data_uri("STOP", bg="red", fg="white"),
        "prompt": "What is the background color of this image? Answer in one word.",
        "expected": "red",
        "max_tokens": 32,
    },
    {
        "image": _image_data_uri("42", bg="white", fg="black"),
        "prompt": "What number is written in this image? Reply with digits only.",
        "expected": "42",
        "max_tokens": 32,
    },
    {
        "image": _image_data_uri("CAT", bg="white", fg="black"),
        "prompt": "What single word is written in this image? Reply with just the word.",
        "expected": "cat",
        "max_tokens": 32,
    },
]


def _wait_for_server(proc: subprocess.Popen, port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def _chat(port: int, image: str, prompt: str, max_tokens=32, temperature=0):
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _tail_file(path: str, stop_event: threading.Event, prefix: str = "[encode] "):
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
def epd_server():
    env = os.environ.copy()
    env.setdefault("LB_PORT", str(LB_PORT))
    proc = subprocess.Popen(
        ["bash", os.path.abspath(SERVE_SCRIPT)],
        env=env,
    )
    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=_tail_file,
        args=(os.path.join(LOG_DIR, "encode.log"), stop_event),
        daemon=True,
    )
    tail_thread.start()
    try:
        if not _wait_for_server(proc, LB_PORT, STARTUP_TIMEOUT):
            rc = proc.poll()
            pytest.fail(
                f"EPD server did not become ready within {STARTUP_TIMEOUT}s"
                f" (exit_code={rc})"
            )
        time.sleep(5)
        yield proc
    finally:
        from tokenspeed.runtime.utils.process import kill_process_tree

        kill_process_tree(proc.pid)
        stop_event.set()
        tail_thread.join(timeout=5)


def test_quality(epd_server):
    assert epd_server.poll() is None, "EPD server exited unexpectedly"
    failures = []
    for i, check in enumerate(QUALITY_CHECKS):
        data = _chat(
            LB_PORT, check["image"], check["prompt"], max_tokens=check["max_tokens"]
        )
        content = data["choices"][0]["message"]["content"]
        print(f"\n[Q{i}] {check['prompt']}")
        print(f"[A{i}] {content}")
        if check["expected"].lower() not in content.lower():
            failures.append(
                f"  check[{i}]: expected {check['expected']!r} in {content!r}"
            )
    assert not failures, "Quality check failures:\n" + "\n".join(failures)
