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

"""Convolution program maps: exact indices, per-forward ownership, and graphs."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    build_causal_conv1d_prefill_metadata,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    return request.param


def _boundaries(lengths, device, dtype):
    values = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    return torch.tensor(values, device=device, dtype=dtype)


def _reference(lengths, block_m):
    counts = [(length + block_m - 1) // block_m for length in lengths]
    return (
        [row for row, count in enumerate(counts) for _ in range(count)],
        [chunk for count in counts for chunk in range(count)],
    )


@pytest.mark.parametrize(
    "lengths",
    [
        [],
        [0],
        [1],
        [7, 8, 9, 0],
        [868],
        [8193, 17],
        [0, 17, 8199, 0],
        [1] * 129,
    ],
)
@pytest.mark.parametrize("block_m", [8, 16])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_metadata_matches_reference(device, lengths, block_m, dtype):
    metadata = build_causal_conv1d_prefill_metadata(
        _boundaries(lengths, device, dtype), torch.tensor(lengths, dtype=dtype), block_m
    )
    rows, chunks = _reference(lengths, block_m)
    assert metadata.batch_indices.tolist() == rows
    assert metadata.chunk_offsets.tolist() == chunks
    assert metadata.batch_indices.dtype == metadata.chunk_offsets.dtype == torch.int32
    assert metadata.block_m == block_m
    assert metadata.batch_indices.device.type == device


def test_metadata_forward_storage_is_independent(device):
    previous = build_causal_conv1d_prefill_metadata(
        _boundaries([9, 7], device, torch.int32), torch.tensor([9, 7]), 8
    )
    saved_rows = previous.batch_indices.clone()
    saved_chunks = previous.chunk_offsets.clone()
    current = build_causal_conv1d_prefill_metadata(
        _boundaries([7, 9], device, torch.int32), torch.tensor([7, 9]), 8
    )
    assert previous.batch_indices.data_ptr() != current.batch_indices.data_ptr()
    assert torch.equal(previous.batch_indices, saved_rows)
    assert torch.equal(previous.chunk_offsets, saved_chunks)
    assert current.batch_indices.tolist() == [0, 1, 1]
    assert current.chunk_offsets.tolist() == [0, 0, 1]


@pytest.mark.parametrize(
    "lengths,block_m,error",
    [
        ([-1], 8, "nonnegative"),
        ([8], 0, "positive"),
    ],
)
def test_invalid_lengths(lengths, block_m, error):
    with pytest.raises(ValueError, match=error):
        build_causal_conv1d_prefill_metadata(
            _boundaries(lengths, "cpu", torch.int32), torch.tensor(lengths), block_m
        )


def test_invalid_boundary_count():
    with pytest.raises(ValueError, match="boundary"):
        build_causal_conv1d_prefill_metadata(
            torch.tensor([0, 1]), torch.tensor([1, 2]), 8
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_metadata_cuda_graph_refresh():
    lengths = torch.tensor([9, 7], dtype=torch.int32)
    boundaries = _boundaries(lengths.tolist(), "cuda", torch.int32)
    for _ in range(3):
        build_causal_conv1d_prefill_metadata(boundaries, lengths, 8)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = build_causal_conv1d_prefill_metadata(boundaries, lengths, 8)
    pointer = metadata.batch_indices.data_ptr()
    for values in ([7, 9], [9, 7], [0, 24]):
        # Same output capacity, different live request-to-program mapping.
        lengths.copy_(torch.tensor(values, dtype=torch.int32))
        boundaries.copy_(_boundaries(values, "cuda", torch.int32))
        graph.replay()
        rows, chunks = _reference(values, 8)
        assert metadata.batch_indices.tolist() == rows
        assert metadata.chunk_offsets.tolist() == chunks
        assert metadata.batch_indices.data_ptr() == pointer
