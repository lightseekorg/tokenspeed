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

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.model_executor import ModelExecutor


class _RuntimeStates:
    def __init__(self):
        self.valid_cache_lengths = torch.arange(20, dtype=torch.int32)

    def reset_states(self, req_pool_indices, prefix_lens):
        self.valid_cache_lengths[req_pool_indices] = prefix_lens


class _ExecutionStream:
    def wait_stream(self, _):
        return None


def test_mixed_batch_resets_only_prefill_lengths(monkeypatch):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cpu"
    executor.execution_stream = _ExecutionStream()
    executor.runtime_states = _RuntimeStates()

    forward_op = SimpleNamespace(
        request_pool_indices=[2, 3, 4],
        extend_prefix_lens=[10],
        num_extends=lambda: 1,
    )

    torch_tensor = torch.tensor

    def tensor_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor_without_pinning)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    executor.reset_valid_cache_length(forward_op)

    assert executor.runtime_states.valid_cache_lengths[2].item() == 10
    assert executor.runtime_states.valid_cache_lengths[3].item() == 3
    assert executor.runtime_states.valid_cache_lengths[4].item() == 4
