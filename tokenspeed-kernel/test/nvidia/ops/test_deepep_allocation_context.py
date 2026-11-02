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

"""CPU checks for persistent DeepEP preparation allocation ownership."""

from contextlib import contextmanager

import pytest
import torch
from tokenspeed_kernel.ops import moe
from tokenspeed_kernel.ops.communication import deep_ep
from torch import nn


class CpuAllocationTracker:
    """Simulate lost managed pages without allocating device memory."""

    def __init__(self):
        self.managed = True
        self.allocations = []

    @contextmanager
    def resident(self):
        previous = self.managed
        self.managed = False
        try:
            yield
        finally:
            self.managed = previous

    def allocate(self, value):
        tensor = torch.tensor([value], dtype=torch.int32)
        self.allocations.append((self.managed, tensor))
        return tensor

    def discard(self):
        for managed, tensor in self.allocations:
            if managed:
                tensor.zero_()


class Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 16
        self.num_experts = 4
        self.register_buffer("prepared", None)


class FalseyInitializer:
    def __init__(self, tracker):
        self.tracker = tracker

    def __bool__(self):
        return False

    def __call__(self, operation):
        with self.tracker.resident():
            operation()


@pytest.mark.parametrize("context_kind", ["resident", "falsey", "none"])
def test_only_communication_allocations_escape_weight_discard(
    monkeypatch, context_kind
):
    tracker = CpuAllocationTracker()
    weights = Weights()
    communication = []

    def preprocess(*, plan, w):
        del plan
        w.prepared = tracker.allocate(11)
        return w.prepared

    def prepare(**kwargs):
        del kwargs
        # Device/NVSHMEM allocation is the only missing hardware operation.
        communication.append(tracker.allocate(17))

    monkeypatch.setattr(deep_ep, "prepare_deepep_buffer", prepare)
    plan = dict(
        a2a_backend="deepep",
        weight_preprocessor=preprocess,
        process_group=object(),
        deepep_mode="low_latency",
        deepep_low_latency_max_num_tokens_per_gpu=32,
    )

    def initialize(operation):
        with tracker.resident():
            operation()

    context = {
        "resident": initialize,
        "falsey": FalseyInitializer(tracker),
        "none": None,
    }[context_kind]
    prepared = moe.moe_process_weights(plan, weights, communication_initializer=context)
    assert prepared is weights.prepared
    later_weight = tracker.allocate(23)
    tracker.discard()
    assert weights.prepared.item() == 0
    assert later_weight.item() == 0
    assert communication[0].item() == (0 if context_kind == "none" else 17)


def test_native_preparation_failure_restores_weight_allocator(monkeypatch):
    tracker = CpuAllocationTracker()

    def prepare(**kwargs):
        del kwargs
        raise RuntimeError("native preparation failed")

    monkeypatch.setattr(deep_ep, "prepare_deepep_buffer", prepare)
    plan = dict(
        a2a_backend="deepep",
        weight_preprocessor=None,
        process_group=object(),
        deepep_mode="low_latency",
        deepep_low_latency_max_num_tokens_per_gpu=32,
    )

    def initialize(operation):
        with tracker.resident():
            operation()

    with pytest.raises(RuntimeError, match="native preparation failed"):
        moe.moe_process_weights(plan, Weights(), communication_initializer=initialize)
    weight = tracker.allocate(29)
    tracker.discard()
    assert weight.item() == 0


def test_preparation_can_be_deferred_until_weights_finish_loading(monkeypatch):
    pending = []
    prepared = []
    monkeypatch.setattr(
        deep_ep, "prepare_deepep_buffer", lambda **kwargs: prepared.append(kwargs)
    )
    plan = dict(
        a2a_backend="deepep",
        weight_preprocessor=None,
        process_group=object(),
        deepep_mode="normal",
        deepep_low_latency_max_num_tokens_per_gpu=32,
    )
    moe.moe_process_weights(plan, Weights(), communication_initializer=pending.append)
    assert prepared == [] and len(pending) == 1
    pending[0]()
    assert prepared[0]["hidden_size"] == 16
    assert prepared[0]["num_experts"] == 4
