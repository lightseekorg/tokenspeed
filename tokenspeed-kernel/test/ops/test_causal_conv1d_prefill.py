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

"""The Triton prefill causal convolution on a column slice of a fused
projection: the same bits as on its contiguous copy, written as packed token
rows."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _conv(x, weight, states, query_start_loc, lengths, cache_indices):
    metadata = build_causal_conv1d_prefill_metadata(
        query_start_loc,
        torch.tensor(lengths, dtype=torch.int64),
        CAUSAL_CONV1D_BLOCK_M,
    )
    return causal_conv1d_fn(
        x.transpose(0, 1),
        weight,
        None,
        conv_states=states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=torch.ones(len(lengths), dtype=torch.bool, device=x.device),
        activation="silu",
        prefill_metadata=metadata,
    ).transpose(0, 1)


@pytest.mark.parametrize("lengths", [[8192], [1, 15, 848, 64]])
def test_row_strided_input_matches_its_contiguous_copy(device, lengths):
    # GLM-5.3-Flash's q|k|v columns of the fused KDA input projection.
    channels, width = 6144, 6416
    generator = torch.Generator(device=device).manual_seed(5)
    tokens = sum(lengths)
    projection = torch.randn(
        tokens, width, device=device, generator=generator
    ).bfloat16()
    weight = (
        0.5 * torch.randn(channels, 4, device=device, generator=generator)
    ).bfloat16()
    states = torch.randn(
        len(lengths) + 1, channels, 3, device=device, generator=generator
    ).bfloat16()
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_indices = torch.arange(1, len(lengths) + 1, dtype=torch.int32, device=device)
    sliced = projection[:, :channels]
    expected_states, states_from_slice = states.clone(), states.clone()

    expected = _conv(
        sliced.contiguous(),
        weight,
        expected_states,
        query_start_loc,
        lengths,
        cache_indices,
    )
    output = _conv(
        sliced, weight, states_from_slice, query_start_loc, lengths, cache_indices
    )

    assert output.is_contiguous()
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(
        states_from_slice.view(torch.int16), expected_states.view(torch.int16)
    )
