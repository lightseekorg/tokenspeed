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

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn.triton import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn
from tokenspeed_kernel.ops.conv._triton.causal_conv1d import _causal_conv1d_fwd_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@pytest.mark.parametrize("lengths", [[1, 2, 3], [7, 8, 9], [129, 17], [8192]])
@pytest.mark.parametrize("channels", [96, 1536])
@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_partitioned_conv_preserves_values_and_state(
    lengths, channels, partitions, width
):
    total = sum(lengths)
    torch.manual_seed(total + channels)
    projected = torch.randn(
        total + 5, channels + 128, device="cuda", dtype=torch.bfloat16
    )
    x = projected[:, :channels].T
    weight = torch.randn(channels, width, device="cuda", dtype=x.dtype) * 0.1
    bias = torch.randn(channels, device="cuda", dtype=x.dtype)
    original = torch.randn(
        len(lengths) + 2, channels, width - 1, device="cuda", dtype=x.dtype
    )
    state, reference_state = original.clone(), original.clone()
    slots = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
    sizes = torch.tensor(lengths, dtype=torch.int32)
    bounds = torch.tensor(
        [0, *sizes.cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    metadata = build_causal_conv1d_prefill_metadata(bounds, sizes, 8)
    output = torch.full(
        (partitions, total + 5, channels // partitions),
        13,
        device="cuda",
        dtype=x.dtype,
    )
    kwargs = dict(
        weight=weight,
        bias=bias,
        query_start_loc=bounds,
        cache_indices=slots,
        has_initial_state=slots % 2 == 0,
        activation="silu",
        prefill_metadata=metadata,
    )
    actual = causal_conv1d_fn(x, conv_states=state, out=output, **kwargs)
    expected = causal_conv1d_fn(x, conv_states=reference_state, **kwargs).T[:total]
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(
        actual[:, :total],
        expected.view(total, partitions, -1).permute(1, 0, 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
    assert bool((output[:, total:] == 13).all())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = causal_conv1d_fn(x, conv_states=state, out=output, **kwargs)
    projected.normal_()
    state.copy_(original)
    reference_state.copy_(original)
    graph.replay()
    expected = causal_conv1d_fn(x, conv_states=reference_state, **kwargs).T[:total]
    torch.testing.assert_close(
        actual[:, :total],
        expected.view(total, partitions, -1).permute(1, 0, 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
    torch.testing.assert_close(state[0], original[0], atol=0, rtol=0)
    torch.testing.assert_close(state[-1], original[-1], atol=0, rtol=0)


def test_partitioned_conv_reuses_compilation():
    channels = 384
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(2, channels, 3, device="cuda", dtype=weight.dtype)
    slots = torch.ones(1, device="cuda", dtype=torch.int32)
    history = torch.zeros(1, device="cuda", dtype=torch.bool)

    def run(tokens):
        x = torch.randn(tokens, channels + 128, device="cuda", dtype=weight.dtype)[
            :, :channels
        ].T
        bounds = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        sizes = torch.tensor([tokens], dtype=torch.int32)
        metadata = build_causal_conv1d_prefill_metadata(bounds, sizes, 8)
        output = torch.empty(3, tokens, channels // 3, device="cuda", dtype=x.dtype)
        return causal_conv1d_fn(
            x,
            weight,
            None,
            state,
            bounds,
            slots,
            history,
            prefill_metadata=metadata,
            out=output,
        )

    run(32)
    with assert_no_triton_compile(_causal_conv1d_fwd_kernel):
        for tokens in (1, 3, 17, 33, 65, 129, 257):
            run(tokens)
