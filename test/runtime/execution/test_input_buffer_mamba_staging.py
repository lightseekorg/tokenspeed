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

from __future__ import annotations

from types import SimpleNamespace

import torch

import tokenspeed.runtime.execution.input_buffer as input_buffer_module
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_executor import ModelExecutor


def test_mamba_decode_inputs_use_per_step_bulk_staging(monkeypatch):
    buffers = InputBuffers(
        max_bs=4,
        max_num_tokens=16,
        page_size=64,
        dummy_kv_slot=0,
        state_write_padding_pool_index=9,
        device="cpu",
        has_mamba=True,
    )
    assert not hasattr(buffers, "_mamba_pool_indices_cpu")
    assert not hasattr(buffers, "_mamba_cow_src_indices_cpu")

    bulk_calls = []

    def fake_bulk(*specs):
        bulk_calls.append(specs)
        return [torch.empty(numel, dtype=dtype) for numel, dtype in specs]

    def fake_decode_input_prep(
        *,
        out_cache_loc_ptr,
        positions_ptr,
        seq_lens_out_ptr,
        req_pool_indices,
        valid_cache_lengths,
        uniform_input_length,
        req_to_pages,
        page_size,
    ):
        del req_to_pages, page_size
        for row, req_pool_index in enumerate(req_pool_indices.tolist()):
            start = int(valid_cache_lengths[req_pool_index].item())
            base = row * uniform_input_length
            positions_ptr[base : base + uniform_input_length] = torch.arange(
                start,
                start + uniform_input_length,
                dtype=positions_ptr.dtype,
            )
            out_cache_loc_ptr[base : base + uniform_input_length].fill_(base)
            seq_lens_out_ptr[row] = start + uniform_input_length

    monkeypatch.setattr(buffers, "_bulk_pinned", fake_bulk)
    monkeypatch.setattr(
        input_buffer_module, "fused_decode_input_prep", fake_decode_input_prep
    )

    runtime_states = SimpleNamespace(
        valid_cache_lengths=torch.zeros(10, dtype=torch.int32),
        future_input_map=torch.arange(40, dtype=torch.int32).reshape(10, 4),
        remote_spec_candidate_ready=torch.zeros(10, dtype=torch.bool),
        vocab_size=1000,
    )
    forward_op = SimpleNamespace(
        request_pool_indices=[2, 3],
        request_ids=["a", "b"],
        input_lengths=[4, 4],
        decode_input_ids=None,
        mamba_pool_indices=[20, 21],
        mamba_cow_src_indices=[-1, 7],
        mamba_branching_seqlens=[64, 65],
        mamba_track_pool_indices=[30, 31],
        num_extends=lambda: 0,
    )

    buffers.fill_input_buffers(
        forward_op=forward_op,
        runtime_states=runtime_states,
        req_to_page=torch.zeros((10, 1), dtype=torch.int32),
        total_tokens=8,
    )

    assert bulk_calls == [
        ((2, torch.int64), (2, torch.int32)),
        (
            (2, torch.int32),
            (2, torch.int32),
            (2, torch.int32),
            (2, torch.int32),
        ),
    ]
    assert buffers.mamba_pool_indices_buf[:2].tolist() == [20, 21]
    assert buffers.mamba_cow_src_indices_buf[:2].tolist() == [-1, 7]
    assert buffers.mamba_branching_seqlens_buf[:2].tolist() == [64, 65]
    assert buffers.mamba_track_pool_indices_buf[:2].tolist() == [30, 31]


def test_layerwise_mamba_cow_uses_per_step_bulk_staging():
    executor = ModelExecutor.__new__(ModelExecutor)
    bulk_calls = []

    def fake_bulk(*specs):
        bulk_calls.append(specs)
        return [torch.empty(numel, dtype=dtype) for numel, dtype in specs]

    executor.input_buffers = SimpleNamespace(
        has_mamba=True,
        _bulk_pinned=fake_bulk,
        mamba_cow_src_indices_buf=torch.full((4,), 99, dtype=torch.int32),
    )
    executor._layerwise_mamba_cow_done = {77: {22}}
    forward_op = SimpleNamespace(
        mamba_cow_src_indices=[-1, 77, 77],
        mamba_pool_indices=[20, 21, 22],
    )

    skipped = executor._skip_completed_layerwise_mamba_cow(forward_op, bs=3)

    assert bulk_calls == [((3, torch.int32),)]
    assert skipped.tolist() == [False, False, True]
    assert executor.input_buffers.mamba_cow_src_indices_buf[:3].tolist() == [
        -1,
        77,
        -1,
    ]
