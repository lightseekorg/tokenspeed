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

"""When a decode graph replay cleans DeepEP's low-latency workspace."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.execution.forward_step import DeepEPCudaGraphRunnerAdapter

deep_ep = pytest.importorskip("tokenspeed_kernel.ops.communication.deep_ep")
DeepEPBuffer = deep_ep.DeepEPBuffer


class _Buffer:
    """Stand-in for DeepEP's GPU ``Buffer``: counts workspace cleans."""

    low_latency_mode = True

    def __init__(self):
        self.cleans = 0

    def clean_low_latency_buffer(self, tokens, hidden, experts):
        self.cleans += 1


@pytest.fixture
def buffer(monkeypatch):
    fake = _Buffer()
    monkeypatch.setattr(DeepEPBuffer, "_buffer", fake)
    monkeypatch.setattr(DeepEPBuffer, "_dispatch_mode", None)
    monkeypatch.setattr(DeepEPBuffer, "_num_max_dispatch_tokens_per_rank", 384)
    monkeypatch.setattr(DeepEPBuffer, "_hidden_size", 3072)
    monkeypatch.setattr(DeepEPBuffer, "_num_experts", 256)
    return fake


def test_consecutive_replays_do_not_clean(buffer):
    adapter = DeepEPCudaGraphRunnerAdapter()
    adapter.capture()
    for _ in range(4):
        adapter.replay()
    assert buffer.cleans == 0


def test_replay_after_a_normal_forward_cleans_once(buffer):
    adapter = DeepEPCudaGraphRunnerAdapter()
    adapter.capture()
    adapter.replay()
    # An extend-shaped forward took the normal legs in between.
    DeepEPBuffer.set_dispatch_mode_as_normal()
    adapter.replay()
    adapter.replay()
    assert buffer.cleans == 1
