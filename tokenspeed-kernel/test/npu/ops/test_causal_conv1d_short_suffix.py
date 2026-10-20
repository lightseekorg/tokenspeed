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

"""Masked short-suffix lanes must not read beyond the input allocation."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn.triton import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv.triton import triton_causal_conv1d_prefill


@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_short_suffix_masked_addresses(has_initial_state, length, activation):
    if not torch.npu.is_available():
        pytest.skip("requires NPU")
    channels = 256
    host = (
        (torch.arange(length * channels).view(length, channels) % 7 - 3) / 8
    ).bfloat16()
    backing = host.npu()
    x = backing.T
    if length == 1:
        # The only token is still at offset zero. Inactive token offsets point
        # far outside storage, so a read-before-mask cannot hide in allocator padding.
        x = backing.as_strided((channels, 1), (1, 1 << 40))
    weights = ((torch.arange(channels * 4).view(channels, 4) % 5 - 2) / 8).bfloat16()
    state_host = (
        (torch.arange(4 * channels * 3).view(4, channels, 3) % 9 - 4) / 8
    ).bfloat16()
    state = state_host.npu()
    boundaries = torch.tensor([0, length], dtype=torch.int32, device="npu")
    metadata = build_causal_conv1d_prefill_metadata(
        boundaries, torch.tensor([length]), 8
    )
    out = triton_causal_conv1d_prefill(
        x,
        weights.npu(),
        None,
        state,
        boundaries,
        cache_indices=torch.tensor([2], dtype=torch.int32, device="npu"),
        has_initial_state=torch.tensor([has_initial_state], device="npu"),
        activation=activation,
        prefill_metadata=metadata,
    )
    torch.npu.synchronize()
    history = (
        state_host[2].T
        if has_initial_state
        else torch.zeros(3, channels, dtype=torch.bfloat16)
    )
    sequence = torch.cat((history, host))
    expected = torch.zeros(length, channels, dtype=torch.float32)
    for tap in range(4):
        expected += sequence[tap : tap + length].float() * weights[:, tap].float()
    if activation == "silu":
        expected = torch.nn.functional.silu(expected)
    torch.testing.assert_close(
        out.T.cpu(),
        expected.bfloat16(),
        rtol=0 if activation is None else 0.0078125,
        atol=0 if activation is None else 1e-4,
    )
    state_host[2] = sequence[-3:].T
    torch.testing.assert_close(state.cpu(), state_host, rtol=0, atol=0)
