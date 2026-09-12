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

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.runtime_states import RuntimeStates

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [[], [2], [1, 3]])
def test_reset_states_preserves_unselected_rows(device, rows):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    states = RuntimeStates(4, 32, 2, device)
    states.valid_cache_lengths.fill_(7)
    states.remote_spec_candidate_ready.fill_(True)
    states.future_input_map.fill_(11)
    indices = torch.tensor(rows, dtype=torch.int64, device=device)
    lengths = torch.full((len(rows),), 100, dtype=torch.int32, device=device)

    states.reset_states(indices, lengths)

    assert states.valid_cache_lengths.tolist() == [
        100 if i in rows else 7 for i in range(5)
    ]
    assert states.remote_spec_candidate_ready.tolist() == [
        i not in rows for i in range(5)
    ]
    assert states.future_input_map.tolist() == [[11, 11]] * 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reset_states_does_not_synchronize_cuda():
    states = RuntimeStates(4, 32, 2, "cuda")
    indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    lengths = torch.tensor([128, 256], dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        states.reset_states(indices, lengths)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.cuda.synchronize()
    assert states.valid_cache_lengths.tolist() == [0, 128, 0, 256, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reset_states_graph_replay_uses_current_inputs():
    states = RuntimeStates(4, 32, 2, "cuda")
    indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    lengths = torch.tensor([128, 256], dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        states.reset_states(indices, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        states.reset_states(indices, lengths)

    for offset in (0, 1):
        indices.copy_(torch.tensor([1 + offset, 3 + offset], device="cuda"))
        lengths.fill_(512 + offset)
        states.valid_cache_lengths.zero_()
        states.remote_spec_candidate_ready.fill_(True)
        graph.replay()
        selected = (1 + offset, 3 + offset)
        assert states.valid_cache_lengths.tolist() == [
            512 + offset if i in selected else 0 for i in range(5)
        ]
        assert states.remote_spec_candidate_ready.tolist() == [
            i not in selected for i in range(5)
        ]
