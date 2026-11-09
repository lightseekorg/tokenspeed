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

"""GPU integration test of the development forward reload (``--allow-reload-forward``)."""

import os
import sys
import uuid

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=240, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import ReloadForwardReqInput  # noqa: E402
from tokenspeed.runtime.entrypoints.wire_driver import (  # noqa: E402
    WireDriver,
    generate_request,
)

MODEL = os.environ.get("FORWARD_RELOAD_MODEL", "Qwen/Qwen3-0.6B")
MODEL_MODULE = os.environ.get(
    "FORWARD_RELOAD_MODULE", "tokenspeed.runtime.models.qwen3"
)
PROMPT = "The capital of France is"
GREEDY = {"temperature": 0.0, "max_new_tokens": 16}


class Engine:
    """A headless engine with the forward reload enabled, driven over its wire."""

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.driver = WireDriver(
            [
                "--model",
                MODEL,
                "--allow-reload-forward",
                "--max-model-len",
                "2048",
                "--gpu-memory-utilization",
                "0.3",
                "--cudagraph-capture-sizes",
                "1",
                "2",
                "4",
            ],
            num_engines=1,
            routing="round_robin",
            startup_timeout_s=3600,
        )

    def generate(self) -> list[int]:
        [completion] = self.driver.generate(
            [generate_request(self.tokenizer.encode(PROMPT), GREEDY)]
        )
        return completion.output_ids

    def reload(self, **fields):
        [reply] = self.driver.control(
            ReloadForwardReqInput(id=uuid.uuid4().hex, **fields)
        )
        return reply


@pytest.fixture(scope="module")
def engine():
    engine = Engine()
    yield engine
    engine.driver.shutdown()


@pytest.fixture(scope="module")
def baseline(engine):
    return engine.generate()


def test_a_reload_without_edits_recaptures_and_keeps_greedy_outputs(engine, baseline):
    reply = engine.reload()
    assert reply.success, reply.message
    assert "recaptured" in reply.message
    assert engine.generate() == baseline


def test_a_forced_reload_of_the_model_module_keeps_greedy_outputs(engine, baseline):
    reply = engine.reload(modules=[MODEL_MODULE])
    assert reply.success, reply.message
    assert f"reloaded {MODEL_MODULE}" in reply.message
    assert engine.generate() == baseline


def test_a_refused_reload_leaves_the_engine_serving(engine, baseline):
    reply = engine.reload(modules=["tokenspeed.runtime.models.not_a_module"])
    assert not reply.success
    assert "not loaded" in reply.message
    assert engine.generate() == baseline


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
