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

"""Causal convolution output, history, and independent cache page updates."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn.triton import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn, causal_conv1d_update

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Convolution tests require an Ascend NPU"
)


def reference(x, weight, history):
    window = history.double()
    rows = []
    for token in x.double():
        taps = torch.cat((window, token[:, None]), dim=1)
        rows.append(F.silu((taps * weight.double()).sum(1)))
        window = taps[:, 1:]
    return torch.stack(rows), window


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 63, 64, 65, 129])
@pytest.mark.parametrize("channels", [17, 6144])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill_history_matches_reference(length, channels, dtype):
    torch.manual_seed(37)
    x = torch.randn(length, channels).to(dtype)
    w = (torch.randn(channels, 4) * 0.1).to(dtype)
    history = torch.randn(channels, 3).to(dtype)
    expected, final = reference(x, w, history)
    pool = torch.randn(5, channels, 3, device="npu").to(dtype)
    pool[2] = history.npu()
    before = pool.clone()
    bounds = torch.tensor([0, length], dtype=torch.int32, device="npu")
    metadata = build_causal_conv1d_prefill_metadata(bounds, torch.tensor([length]), 64)
    actual = causal_conv1d_fn(
        x.npu().T,
        w.npu(),
        None,
        pool,
        bounds,
        cache_indices=torch.tensor([2], dtype=torch.int32, device="npu"),
        has_initial_state=torch.tensor([True], device="npu"),
        activation="silu",
        prefill_metadata=metadata,
    )
    torch.testing.assert_close(actual.T.cpu().double(), expected, atol=4e-3, rtol=4e-3)
    torch.testing.assert_close(pool[2].cpu().double(), final, atol=0, rtol=0)
    torch.testing.assert_close(pool[[0, 1, 3, 4]], before[[0, 1, 3, 4]], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_remapping_and_reinitialized_request(dtype):
    torch.manual_seed(41)
    channels, length = 6144, 7
    x = torch.randn(length, channels).to(dtype)
    w = (torch.randn(channels, 4) * 0.1).to(dtype)
    pool = torch.randn(6, channels, 3, device="npu").to(dtype)
    old = pool.clone()
    expected, final = reference(x, w, pool[1].cpu())
    wn = w.npu()
    read = torch.tensor([1], device="npu", dtype=torch.int32)
    write = torch.tensor([[3]], device="npu", dtype=torch.int32)
    for t in range(length):
        actual = causal_conv1d_update(
            x[t : t + 1].npu(),
            pool,
            wn,
            activation="silu",
            conv_state_indices=read,
            output_state_indices=write,
            parent_indices=None,
        )
        torch.testing.assert_close(
            actual.cpu().double(), expected[t : t + 1], atol=4e-3, rtol=4e-3
        )
        read = write[:, 0]
    torch.testing.assert_close(pool[3].cpu().double(), final, atol=0, rtol=0)
    torch.testing.assert_close(
        pool[[0, 1, 2, 4, 5]], old[[0, 1, 2, 4, 5]], atol=0, rtol=0
    )
    # Reuse the slot after zero initialization, as the allocator does.
    pool[3].zero_()
    fresh_expected, _ = reference(x[:1], w, torch.zeros(channels, 3))
    actual = causal_conv1d_update(
        x[:1].npu(),
        pool,
        wn,
        activation="silu",
        conv_state_indices=read,
        output_state_indices=write,
        parent_indices=None,
    )
    torch.testing.assert_close(
        actual.cpu().double(), fresh_expected, atol=4e-3, rtol=4e-3
    )


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_half_convolution_keeps_products_in_fp32(mode):
    # Cancellation exposes rounding a half product before the FP32 sum.
    channels = 17
    a = 1 + 2**-9
    weight = torch.tensor([a, -1, 0, 0], dtype=torch.float16).repeat(channels, 1)
    history = torch.tensor([a, 1, 0], dtype=torch.float16).repeat(channels, 1)
    x = torch.zeros(1, channels, dtype=torch.float16, device="npu")
    pool = history.npu().unsqueeze(0).clone()
    indices = torch.tensor([0], device="npu", dtype=torch.int32)
    if mode == "decode":
        actual = causal_conv1d_update(
            x,
            pool,
            weight.npu(),
            activation=None,
            conv_state_indices=indices,
            parent_indices=None,
        )
    else:
        bounds = torch.tensor([0, 1], device="npu", dtype=torch.int32)
        metadata = build_causal_conv1d_prefill_metadata(bounds, torch.tensor([1]), 64)
        actual = causal_conv1d_fn(
            x.T,
            weight.npu(),
            None,
            pool,
            bounds,
            cache_indices=indices,
            has_initial_state=torch.tensor([True], device="npu"),
            activation=None,
            prefill_metadata=metadata,
        ).T
    expected = torch.full((1, channels), a * a - 1, dtype=torch.float64).half()
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
