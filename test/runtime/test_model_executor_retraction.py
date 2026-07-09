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

import pytest
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


class _RecordingAttentionBackend:
    def __init__(self):
        self.reset_calls = []

    def reset_current_inputs(self, req_pool_indices, mamba_pool_indices):
        self.reset_calls.append(
            (req_pool_indices.tolist(), mamba_pool_indices.tolist())
        )


def test_mixed_batch_resets_prefill_and_retracted_decode_lengths(monkeypatch):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.device = "cpu"
    executor.execution_stream = _ExecutionStream()
    executor.runtime_states = _RuntimeStates()

    forward_op = SimpleNamespace(
        request_pool_indices=[2, 3, 4],
        extend_prefix_lens=[10],
        # hist_token_lens contains decode rows only: one normal decode and one
        # recovery row following the prefill row.
        hist_token_lens=[-1, 7],
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
    assert executor.runtime_states.valid_cache_lengths[4].item() == 7


@pytest.mark.parametrize(
    ("mamba_cow_src", "skipped_layerwise_cow_mask"),
    [
        ([-1, -1, 77], None),
        ([-1, -1, -1], [False, False, True]),
    ],
)
def test_mixed_batch_resets_prefill_and_retracted_mamba_inputs(
    mamba_cow_src,
    skipped_layerwise_cow_mask,
):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.attn_backend = _RecordingAttentionBackend()
    executor.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([10, 11, 12], dtype=torch.int32)
    )
    forward_op = SimpleNamespace(hist_token_lens=[-1, 7])

    executor._reset_mamba_current_inputs(
        num_extends=1,
        bs=3,
        has_retract=executor._contains_retracted_decode(forward_op),
        mamba_pool_indices=torch.tensor([20, 21, 22], dtype=torch.int32),
        mamba_cow_src=torch.tensor(mamba_cow_src, dtype=torch.int32),
        skipped_layerwise_cow_mask=(
            torch.tensor(skipped_layerwise_cow_mask, dtype=torch.bool)
            if skipped_layerwise_cow_mask is not None
            else None
        ),
    )

    assert executor.attn_backend.reset_calls == [
        ([10], [20]),
        ([12], [22]),
    ]
