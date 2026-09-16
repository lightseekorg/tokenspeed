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

"""State staging compared bitwise with the previous PyTorch operations."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn._triton.prefill_state_inputs import (
    prepare_prefill_state_inputs,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    return request.param


def _reference(conv, state, src, dst):
    history = src > 0
    safe = torch.where(history, src, dst).long()
    conv[dst.long()] = conv[safe]
    recurrent = state[safe]
    recurrent.masked_fill_(~history[:, None, None, None], 0)
    return recurrent, history


def _inputs(device, dtype, strided, shape):
    heads, key, value = shape
    if strided:
        conv = torch.randn(13, 15, 8, dtype=dtype, device=device)[1::2, ::2, 1::2]
        state = torch.randn(
            13, heads * 2, value * 2, key * 2, dtype=dtype, device=device
        )
        state = state[1::2, ::2, 1::2, ::2].transpose(-1, -2)
    else:
        conv = torch.randn(6, 8, 4, dtype=dtype, device=device)
        state = torch.randn(6, heads, key, value, dtype=dtype, device=device)
    state[0].fill_(float("nan"))
    state[1].fill_(float("nan"))
    conv[0].fill_(float("nan"))
    return conv, state


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("shape", [(2, 7, 9), (3, 32, 64)])
def test_state_preparation_exact(device, dtype, strided, shape):
    torch.manual_seed(42)
    conv, state = _inputs(device, dtype, strided, shape)
    conv_before, state_before = conv.clone(), state.clone()
    # Fresh, two readers of a shared snapshot, and private in-place evolution.
    src = torch.tensor([0, 99, 2, 99, 2, 99, 5, 99], dtype=torch.int32, device=device)[
        ::2
    ]
    dst = torch.tensor([1, 99, 3, 99, 4, 99, 5, 99], dtype=torch.int64, device=device)[
        ::2
    ]
    expected_conv = conv.clone()
    expected, expected_history = _reference(expected_conv, state, src, dst)
    actual, history = prepare_prefill_state_inputs(conv, state, src, dst)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(history, expected_history)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(state, state_before, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(conv[2], conv_before[2], rtol=0, atol=0)
    assert torch.equal(conv[1], conv_before[1])


def test_empty_state_batch(device):
    conv = torch.empty(3, 8, 3, device=device)
    state = torch.empty(3, 2, 4, 4, device=device)
    indices = torch.empty(0, dtype=torch.int32, device=device)
    out, history = prepare_prefill_state_inputs(conv, state, indices, indices)
    assert out.shape == (0, 2, 4, 4)
    assert history.shape == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_state_preparation_cuda_graph():
    conv, state = _inputs("cuda", torch.bfloat16, True, (3, 7, 9))
    state = state.float()
    src = torch.tensor([0, 2, 2], dtype=torch.int32, device="cuda")
    dst = torch.tensor([1, 3, 4], dtype=torch.int32, device="cuda")
    for _ in range(3):
        prepare_prefill_state_inputs(conv, state, src, dst)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, history = prepare_prefill_state_inputs(conv, state, src, dst)
    for sources in ([2, 0, 2], [0, 2, 0]):
        src.copy_(torch.tensor(sources, dtype=torch.int32, device="cuda"))
        expected_conv = conv.clone()
        expected, expected_history = _reference(expected_conv, state, src, dst)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(history, expected_history)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0, equal_nan=True)
