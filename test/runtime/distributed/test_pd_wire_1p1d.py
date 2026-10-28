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

"""Prefill/decode disaggregation (one prefill, one decode engine) end to end.

The test drives both engines over their wire (``WireDriver``) as a PD router
does: each request goes to the prefill engine for one token, with the
rendezvous of its KV (the prefill engine's bootstrap server and a room), and
to the decode engine for the rest, at once. The decode engine's answers to a
few chat questions must hold the expected words.

Environment:
    MODEL               the model (default Qwen/Qwen3-0.6B)
    WORLD_SIZE          ranks per engine (default 1)
    PREFILL_GPUS        the prefill engine's GPUs (default 0)
    DECODE_GPUS         the decode engine's GPUs (default 1)
    PD_ENGINE_ARGS      more of both engines' command lines
    CHAT_TEMPLATE_KWARGS  JSON options for the chat template
                        (default {"enable_thinking": false})
    PD_STARTUP_TIMEOUT  seconds an engine may take to start (default 2400)

Usage:
    pytest test/runtime/distributed/test_pd_wire_1p1d.py -v -s
"""

import json
import os
import random
import shlex
from concurrent.futures import ThreadPoolExecutor

import pytest

from tokenspeed.runtime.entrypoints.wire_driver import WireDriver, generate_request
from tokenspeed.runtime.utils.network import get_free_port

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
WORLD_SIZE = os.environ.get("WORLD_SIZE", "1")
PREFILL_GPUS = os.environ.get("PREFILL_GPUS", "0")
DECODE_GPUS = os.environ.get("DECODE_GPUS", "1")
ENGINE_ARGS = shlex.split(os.environ.get("PD_ENGINE_ARGS", ""))
CHAT_TEMPLATE_KWARGS = json.loads(
    os.environ.get("CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
)
STARTUP_TIMEOUT = float(os.environ.get("PD_STARTUP_TIMEOUT", "2400"))

QUALITY_CHECKS = [
    ("What is the capital of France? Reply in one word.", "Paris"),
    ("What is 2+2? Reply with just the number.", "4"),
    ("Name the largest planet in our solar system in one word.", "Jupiter"),
]
MAX_NEW_TOKENS = 64


def _engine(role: str, gpus: str, args: list[str]) -> WireDriver:
    return WireDriver(
        [
            "--model",
            MODEL,
            "--world-size",
            WORLD_SIZE,
            "--trust-remote-code",
            "--disable-kvstore",
            "--disaggregation-mode",
            role,
            "--disaggregation-transfer-backend",
            "mooncake",
            "--port",
            str(get_free_port()),
            "--dist-init-addr",
            f"127.0.0.1:{get_free_port()}",
            *args,
            *ENGINE_ARGS,
        ],
        num_engines=1,
        routing="bootstrap_room",
        startup_timeout_s=STARTUP_TIMEOUT,
        env={"CUDA_VISIBLE_DEVICES": gpus},
    )


@pytest.fixture(scope="module")
def pd_pair():
    bootstrap_port = get_free_port()
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Both engines load at once; each driver waits on its own.
        prefill = pool.submit(
            _engine,
            "prefill",
            PREFILL_GPUS,
            ["--disaggregation-bootstrap-port", str(bootstrap_port)],
        )
        decode = pool.submit(_engine, "decode", DECODE_GPUS, [])
        engines = []
        try:
            for future in (prefill, decode):
                engines.append(future.result())
            yield engines[0], engines[1], bootstrap_port
        finally:
            for engine in engines:
                engine.shutdown()


def _disaggregated(prefill, decode, bootstrap_port, input_ids):
    """The decode engine's completion of one request served by both."""
    bootstrap = {
        "bootstrap_host": "127.0.0.1",
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": random.getrandbits(63),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        # The prefill engine samples the first token and ships it with the
        # KV; the decode engine generates the rest.
        first = pool.submit(
            prefill.generate,
            [
                generate_request(
                    input_ids, {"max_new_tokens": 1, "temperature": 0}, **bootstrap
                )
            ],
        )
        rest = pool.submit(
            decode.generate,
            [
                generate_request(
                    input_ids,
                    {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0},
                    **bootstrap,
                )
            ],
        )
        [prefill_done] = first.result()
        [completion] = rest.result()
    assert prefill_done.finish_reason != "abort", prefill_done.finish_details
    assert completion.finish_reason != "abort", completion.finish_details
    return completion


def test_quality(pd_pair):
    from transformers import AutoTokenizer

    prefill, decode, bootstrap_port = pd_pair
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    failures = []
    for question, expected in QUALITY_CHECKS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion = _disaggregated(prefill, decode, bootstrap_port, input_ids)
        answer = tokenizer.decode(completion.output_ids, skip_special_tokens=True)
        print(f"\n[Q] {question}\n[A] {answer}")
        if expected not in answer:
            failures.append(f"expected {expected!r} in {answer!r}")
    assert not failures, "Quality check failures:\n" + "\n".join(failures)
