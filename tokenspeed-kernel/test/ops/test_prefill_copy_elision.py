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

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn.triton import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn
from tokenspeed_kernel.ops.residual import attn_res_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@pytest.mark.parametrize("lengths", [[1, 2, 3], [7, 8, 9], [129, 17], [8192]])
@pytest.mark.parametrize("channels", [96, 768])
@pytest.mark.parametrize("capture", [False, True])
def test_strided_projection_convolution(lengths, channels, capture):
    torch.manual_seed(123)
    total = sum(lengths)
    projection = torch.randn(total, channels + 128, device="cuda", dtype=torch.bfloat16)
    x = projection[:, :channels].T
    lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
    boundaries = torch.tensor(
        [0, *lengths_cpu.cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    metadata = build_causal_conv1d_prefill_metadata(boundaries, lengths_cpu, 8)
    indices = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
    history = indices % 2 == 0
    weight = torch.randn(channels, 4, device="cuda", dtype=x.dtype) * 0.1
    original = torch.randn(len(lengths) + 2, channels, 3, device="cuda", dtype=x.dtype)
    actual_state = original.clone()
    reference_state = original.clone()

    def run(value, state):
        return causal_conv1d_fn(
            x=value,
            weight=weight,
            bias=None,
            conv_states=state,
            query_start_loc=boundaries,
            cache_indices=indices,
            has_initial_state=history,
            activation="silu",
            pad_slot_id=-1,
            validate_data=True,
            prefill_metadata=metadata,
        )

    actual = run(x, actual_state)
    if capture:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(x, actual_state)
        projection.normal_()
        actual_state.copy_(original)
        graph.replay()
    expected = run(x.T.contiguous().T, reference_state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=0, atol=0)
    assert actual.T.is_contiguous()
    assert torch.equal(actual_state[0], original[0])
    assert torch.equal(actual_state[-1], original[-1])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("write", [False, True])
def test_empty_attnres_preserves_storage(normalize, write):
    hidden = 7168
    layer_storage = torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16)
    block_storage = torch.randn(3, 2, hidden, device="cuda", dtype=torch.bfloat16)
    original_layer = layer_storage.clone()
    original_blocks = block_storage.clone()
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

    def run():
        return attn_res_fwd(
            layer_storage[:0],
            block_storage[:, :0],
            weight,
            weight,
            out_norm_weight=weight if normalize else None,
            delta=layer_storage[:0],
            num_valid_blocks=2,
            block_write_idx=2 if write else -1,
        )

    result = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    graph.replay()
    assert result.shape == (0, hidden)
    assert result.dtype == layer_storage.dtype
    torch.testing.assert_close(layer_storage, original_layer, rtol=0, atol=0)
    torch.testing.assert_close(block_storage, original_blocks, rtol=0, atol=0)
