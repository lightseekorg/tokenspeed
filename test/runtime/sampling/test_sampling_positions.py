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

"""Absolute sampling positions are independent of chunking and cache hits."""

import sys
from pathlib import Path

import pytest
import torch

# CI discovers this registration without importing GPU dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_kernel.ops.sampling import prepare_sampling_offsets

from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        assert torch.cuda.is_available(), "Position kernel acceptance requires CUDA"
    return request.param


@pytest.mark.parametrize("prefix", [0, 130, 514, 1026, 2048])
@pytest.mark.parametrize("chunk_size", [128, 1024, 4096])
def test_prefill_and_decode_predict_at_the_same_absolute_position(
    device, prefix, chunk_size
):
    frontier = torch.tensor([0, prefix, 17], dtype=torch.int32, device=device)
    slots = torch.tensor([1], dtype=torch.int64, device=device)
    offsets = torch.full((3,), -99, dtype=torch.int64, device=device)
    while prefix < 2049:
        chunk = min(chunk_size, 2049 - prefix)
        before = frontier.clone()
        prepare_sampling_offsets(
            frontier,
            slots,
            torch.tensor([chunk], dtype=torch.int32, device=device),
            out=offsets,
        )
        assert torch.equal(frontier, before)
        assert offsets[1].item() == prefix + chunk
        prefix += chunk
        frontier[1] = prefix
    expected = offsets.clone()
    frontier[1] = 2048
    prepare_sampling_offsets(frontier, slots, None, out=offsets)
    assert torch.equal(offsets, expected)
    assert offsets[[0, 2]].tolist() == [-99, -99]


def test_mixed_views_and_padding_do_not_write_unowned_slots(device):
    frontier = torch.tensor([0, 2048, 130, 514, 900], dtype=torch.int32, device=device)
    slots = torch.tensor([2, 3, 1, 0, 0], dtype=torch.int64, device=device)
    lengths = torch.tensor([895, 511], dtype=torch.int32, device=device)
    info = SamplingBatchInfo(
        req_pool_indices=slots, valid_cache_lengths=frontier, extend_lengths=lengths
    )
    offsets = torch.full((5,), -99, dtype=torch.int64, device=device)
    for view in (info[:2], info[2:]):
        prepare_sampling_offsets(
            view.valid_cache_lengths,
            view.req_pool_indices,
            view.extend_lengths,
            out=offsets,
        )
    assert offsets.tolist() == [-99, 2049, 1025, 1025, -99]
    assert info[1:].extend_lengths.tolist() == [511]
    assert info[2:].extend_lengths is None
    assert info[-3:].extend_lengths is None
    assert torch.equal(
        frontier,
        torch.tensor([0, 2048, 130, 514, 900], dtype=torch.int32, device=device),
    )


def test_reused_slot_recomputes_position_and_empty_call_does_not_write(device):
    frontier = torch.tensor([0, 130, 514], dtype=torch.int32, device=device)
    slots = torch.tensor([1, 2], device=device)
    lengths = torch.tensor([1919, 1535], dtype=torch.int32, device=device)
    offsets = torch.zeros(3, dtype=torch.int64, device=device)
    address = offsets.data_ptr()
    prepare_sampling_offsets(frontier, slots, lengths, out=offsets)
    assert offsets.tolist() == [0, 2049, 2049]
    frontier.copy_(torch.tensor([0, 2049, 1024], dtype=torch.int32, device=device))
    prepare_sampling_offsets(frontier, slots.flip(0), None, out=offsets)
    assert offsets.tolist() == [0, 2050, 1025]
    before = offsets.clone()
    prepare_sampling_offsets(frontier, slots[:0], None, out=offsets)
    assert torch.equal(offsets, before)
    assert offsets.data_ptr() == address
