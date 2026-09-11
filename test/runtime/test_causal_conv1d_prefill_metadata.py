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

"""Conv consumers reuse shared maps; compare to the former host-built maps."""

import os
import sys

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    CausalConv1dPrefillMetadata,
    build_causal_conv1d_prefill_metadata,
)

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import causal_conv1d_fn

# CI registration is parsed via AST; standalone execution needs test/ on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _reference_metadata(lengths, device):
    counts = [(length + 7) // 8 for length in lengths]
    rows = [row for row, count in enumerate(counts) for _ in range(count)]
    offsets = [offset for count in counts for offset in range(count)]
    return CausalConv1dPrefillMetadata(
        batch_indices=torch.tensor(rows, dtype=torch.int32, device=device),
        chunk_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
        block_m=CAUSAL_CONV1D_BLOCK_M,
    )


def _conv(x, weight, state, boundaries, indices, history, metadata):
    return causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=None,
        conv_states=state,
        query_start_loc=boundaries,
        cache_indices=indices,
        has_initial_state=history,
        activation="silu",
        pad_slot_id=-1,
        validate_data=False,
        prefill_metadata=metadata,
    )


@pytest.mark.parametrize("lengths", [[1], [7, 8, 9], [868], [8193, 17]])
@pytest.mark.parametrize("channels", [16, 257])
def test_conv_matches_host_maps_without_layer_uploads(lengths, channels, monkeypatch):
    torch.manual_seed(42)
    lens = torch.tensor(lengths, dtype=torch.int32)
    bounds = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0)]).to(
        device="cuda", dtype=torch.int32
    )
    metadata = build_causal_conv1d_prefill_metadata(bounds, lens, CAUSAL_CONV1D_BLOCK_M)
    reference = _reference_metadata(lengths, "cuda")
    indices = torch.arange(1, len(lengths) + 1, dtype=torch.int32, device="cuda")
    history = indices % 2 == 0
    x = torch.randn(sum(lengths), channels, dtype=torch.bfloat16, device="cuda").T
    pointers = (metadata.batch_indices.data_ptr(), metadata.chunk_offsets.data_ptr())

    def unexpected_upload(*args, **kwargs):
        raise AssertionError("the conv consumer must not prepare or upload metadata")

    # Two different layers consume the identical schedule. Their state and
    # weights remain layer-local; only request/chunk geometry is shared.
    for _ in range(2):
        weight = torch.randn(channels, 4, dtype=torch.bfloat16, device="cuda") * 0.1
        state = torch.randn(
            len(lengths) + 1, channels, 3, dtype=torch.bfloat16, device="cuda"
        )
        reference_state = state.clone()
        expected = _conv(
            x, weight, reference_state, bounds, indices, history, reference
        )
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "pin_memory", unexpected_upload)
            patch.setattr(torch.Tensor, "copy_", unexpected_upload)
            patch.setattr(torch, "full", unexpected_upload)
            actual = _conv(x, weight, state, bounds, indices, history, metadata)
        assert torch.equal(actual, expected)
        assert torch.equal(state, reference_state)
        assert pointers == (
            metadata.batch_indices.data_ptr(),
            metadata.chunk_offsets.data_ptr(),
        )


def test_conv_metadata_and_consumer_cuda_graph():
    lengths = torch.tensor([9, 7], dtype=torch.int32)
    boundaries = torch.tensor([0, 9, 16], dtype=torch.int32, device="cuda")
    x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16).T
    weight = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    initial = torch.randn(3, 16, 3, device="cuda", dtype=torch.bfloat16)
    state = initial.clone()
    indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    history = torch.tensor([False, True], device="cuda")
    for _ in range(3):
        metadata = build_causal_conv1d_prefill_metadata(boundaries, lengths, 8)
        _conv(x, weight, state, boundaries, indices, history, metadata)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = build_causal_conv1d_prefill_metadata(boundaries, lengths, 8)
        actual = _conv(x, weight, state, boundaries, indices, history, metadata)
    for sizes in ([9, 7], [7, 9]):
        lengths.copy_(torch.tensor(sizes, dtype=torch.int32))
        boundaries.copy_(
            torch.tensor([0, sizes[0], sum(sizes)], device="cuda", dtype=torch.int32)
        )
        reference_state = initial.clone()
        reference = _reference_metadata(sizes, "cuda")
        expected = _conv(
            x, weight, reference_state, boundaries, indices, history, reference
        )
        state.copy_(initial)
        graph.replay()
        assert torch.equal(actual, expected)
        assert torch.equal(state, reference_state)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
