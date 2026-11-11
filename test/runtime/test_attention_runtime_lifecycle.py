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

"""Native attention state must outlive every captured consumer."""

import weakref

import pytest
import torch
from torch import nn

from tokenspeed.runtime.execution import workspace
from tokenspeed.runtime.execution.device import DeviceHandle
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.model_runner import ModelRunner


class CapturedConsumers:
    """Stand in for the unavailable accelerator graphs, retaining real storage."""

    def __init__(self, tensors):
        self.tensors = tensors

    def release_graphs(self):
        self.tensors.clear()


class PreparedAttention(nn.Module):
    def __init__(self, consumers):
        super().__init__()
        self.consumer_refs = tuple(weakref.ref(tensor) for tensor in consumers)
        self.native_state = torch.ones(4)

    def release_attention_runtime(self):
        if any(ref() is not None for ref in self.consumer_refs):
            raise RuntimeError("captured consumers still retain native attention state")
        self.native_state = None


@pytest.mark.parametrize("with_draft", [False, True])
def test_shutdown_releases_captured_consumers_before_native_attention(
    monkeypatch, with_draft
):
    captured = [torch.ones(3), torch.ones(5)]
    models = [PreparedAttention(captured) for _ in range(2 if with_draft else 1)]
    # Construction loads models and initializes device state. Supply only that
    # absent setup; execute the actual executor/runner/device lifecycle methods.
    runners = []
    for model in models:
        runner = object.__new__(ModelRunner)
        runner.model = model
        runners.append(runner)
    executor = object.__new__(ModelExecutor)
    executor.device = "cpu"
    executor.forward_step = CapturedConsumers(captured)
    executor.prefill_graph = CapturedConsumers([])
    executor.drafter = None
    executor.model_runner = runners[0]
    executor._draft_model_runner = runners[1] if with_draft else None
    monkeypatch.setitem(
        workspace._pools, torch.device("cpu"), workspace.WorkspacePool("cpu", 0)
    )
    handle = DeviceHandle(executor)
    handle.shutdown_attention()
    assert all(model.native_state is None for model in models)
    handle.shutdown_attention()
