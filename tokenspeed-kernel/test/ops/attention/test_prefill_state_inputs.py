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
from tokenspeed_kernel.numerics.torch.attention.gdn import (
    torch_prepare_prefill_state_inputs,
)
from tokenspeed_kernel.ops.attention.gdn._triton.prefill_state_inputs import (
    _prepare_prefill_state_inputs_kernel,
    prepare_prefill_state_inputs,
)
from utils import assert_warmup_covers


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    return request.param


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


@pytest.mark.parametrize(
    ("dtype", "output_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("shape", [(2, 7, 9), (3, 32, 64)])
def test_state_preparation_exact(device, dtype, output_dtype, strided, shape):
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
    expected, expected_history = torch_prepare_prefill_state_inputs(
        expected_conv, state, src, dst
    )
    expected = expected.to(output_dtype)
    actual, history = prepare_prefill_state_inputs(
        conv, state, src, dst, dtype=output_dtype
    )
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
    out, history = prepare_prefill_state_inputs(
        conv, state, indices, indices, dtype=state.dtype
    )
    assert out.shape == (0, 2, 4, 4)
    assert history.shape == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_state_preparation_cuda_graph():
    conv, state = _inputs("cuda", torch.bfloat16, True, (3, 7, 9))
    src = torch.tensor([0, 2, 2], dtype=torch.int32, device="cuda")
    dst = torch.tensor([1, 3, 4], dtype=torch.int32, device="cuda")
    for _ in range(3):
        prepare_prefill_state_inputs(conv, state, src, dst, dtype=torch.float32)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, history = prepare_prefill_state_inputs(
            conv, state, src, dst, dtype=torch.float32
        )
    for sources in ([2, 0, 2], [0, 2, 0]):
        src.copy_(torch.tensor(sources, dtype=torch.int32, device="cuda"))
        expected_conv = conv.clone()
        expected, expected_history = torch_prepare_prefill_state_inputs(
            expected_conv, state, src, dst
        )
        expected = expected.float()
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(history, expected_history)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0, equal_nan=True)


def _assert_bitwise(actual, expected):
    """Compare raw bytes so NaN payloads and signed zeros must match too."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _guarded_pools(device, conv_blocks, ssm_blocks):
    """Pools viewed from the middle of larger backings: the page before block 0
    and the page after the last block are real memory an out-of-range id would
    reach, and each pool's null block holds a NaN sentinel."""
    conv_backing = torch.randn(
        conv_blocks + 2, 8, 4, dtype=torch.bfloat16, device=device
    )
    ssm_backing = torch.randn(
        ssm_blocks + 2, 2, 7, 9, dtype=torch.bfloat16, device=device
    )
    conv_backing[1].fill_(float("nan"))
    ssm_backing[1].fill_(float("nan"))
    return conv_backing, ssm_backing


def _guard_rows(conv_blocks, ssm_blocks):
    """(source, destination) rows mixing valid requests with every kind of id
    outside ``(0, num_blocks)``. Only the first and possibly the ``ssm_blocks``
    row write conv, to distinct blocks that no row reads."""
    return [
        (2, 3),  # resumed: copy conv, gather state
        (4, 4),  # private in-place evolution
        (0, 1),  # fresh
        (2, -1),  # resumed partial row without a destination
        (2, 0),  # null destination
        (2, conv_blocks),  # destination one past the end
        (2, conv_blocks + 4),  # destination far past the end
        (4, -7),  # negative destination
        (-1, 5),  # negative source
        (-7, 5),  # negative source
        (conv_blocks, 1),  # source one past the conv pool
        (conv_blocks + 3, 1),  # source far past the conv pool
        (ssm_blocks, 1),  # source one past the recurrent pool
        (ssm_blocks + 3, 5),  # source far past the recurrent pool
    ]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("conv_blocks", "ssm_blocks"), [(6, 6), (6, 8), (8, 6)])
def test_out_of_range_ids_leave_guard_pages(
    device, index_dtype, conv_blocks, ssm_blocks
):
    torch.manual_seed(conv_blocks * 10 + ssm_blocks)
    conv_backing, ssm_backing = _guarded_pools(device, conv_blocks, ssm_blocks)
    conv_before, ssm_before = conv_backing.clone(), ssm_backing.clone()
    rows = _guard_rows(conv_blocks, ssm_blocks)
    src = torch.tensor([s for s, _ in rows], dtype=index_dtype, device=device)
    dst = torch.tensor([d for _, d in rows], dtype=index_dtype, device=device)

    # Expected results written out per row, independent of either implementation.
    expected_conv = conv_before.clone()
    expected_state = torch.zeros(
        len(rows), 2, 7, 9, dtype=torch.bfloat16, device=device
    )
    expected_history = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for row, (source, destination) in enumerate(rows):
        if 0 < source < ssm_blocks:
            expected_state[row] = ssm_before[1 + source]
            expected_history[row] = True
        if 0 < source < conv_blocks and 0 < destination < conv_blocks:
            if source != destination:
                expected_conv[1 + destination] = conv_before[1 + source]
    assert expected_history.any() and not expected_history.all()

    reference_conv = conv_before.clone()
    reference, reference_history = torch_prepare_prefill_state_inputs(
        reference_conv[1:-1], ssm_before[1:-1], src, dst
    )
    actual, history = prepare_prefill_state_inputs(
        conv_backing[1:-1], ssm_backing[1:-1], src, dst, dtype=torch.bfloat16
    )
    for conv, state, flags in (
        (reference_conv, reference, reference_history),
        (conv_backing, actual, history),
    ):
        _assert_bitwise(state, expected_state)
        assert torch.equal(flags, expected_history)
        # Covers both guard pages and the NaN null block of the conv pool.
        _assert_bitwise(conv, expected_conv)
        assert conv[1].isnan().all()
    _assert_bitwise(ssm_backing, ssm_before)
    assert ssm_backing[1].isnan().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_no_recompile_across_pool_sizes_and_index_offsets():
    def launch(num_blocks, offset, index_dtype):
        conv = torch.randn(num_blocks, 8, 4, dtype=torch.bfloat16, device="cuda")
        state = torch.randn(num_blocks, 2, 7, 9, dtype=torch.bfloat16, device="cuda")
        # Offset slices of larger index buffers, as the runtime hands them over.
        src = torch.zeros(8, dtype=index_dtype, device="cuda")[offset : offset + 4]
        dst = torch.zeros(8, dtype=index_dtype, device="cuda")[offset : offset + 4]
        src.copy_(torch.tensor([2, 0, num_blocks - 1, num_blocks]))
        dst.copy_(torch.tensor([3, 1, -1, num_blocks + 1]))
        expected_conv = conv.clone()
        expected, expected_history = torch_prepare_prefill_state_inputs(
            expected_conv, state, src, dst
        )
        actual, history = prepare_prefill_state_inputs(
            conv, state, src, dst, dtype=torch.bfloat16
        )
        _assert_bitwise(actual, expected)
        assert torch.equal(history, expected_history)
        _assert_bitwise(conv, expected_conv)

    index_dtypes = (torch.int32, torch.int64)

    def warm():
        for index_dtype in index_dtypes:
            launch(8, 0, index_dtype)
        torch.cuda.synchronize()

    assert_warmup_covers(
        [_prepare_prefill_state_inputs_kernel],
        warm,
        lambda case: launch(*case),
        [
            (num_blocks, offset, index_dtype)
            for num_blocks in (16, 17, 32, 65)
            for offset in (1, 2, 3)
            for index_dtype in index_dtypes
        ],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_large_page_offsets():
    if torch.cuda.mem_get_info()[0] < 10 * 1024**3:
        pytest.skip("large-offset test requires 10 GiB free")
    # Block 2 starts past 2^31 elements, so page offsets need int64. Only the
    # three blocks are touched; the rest of each backing stays uninitialized.
    block_stride = 2**30 + 64
    conv_backing = torch.empty(
        2 * block_stride + 32, dtype=torch.bfloat16, device="cuda"
    )
    conv = conv_backing.as_strided((3, 8, 4), (block_stride, 4, 1))
    ssm_backing = torch.empty(
        2 * block_stride + 126, dtype=torch.bfloat16, device="cuda"
    )
    ssm = ssm_backing.as_strided((3, 2, 7, 9), (block_stride, 63, 9, 1))
    for block in range(3):
        conv[block].copy_(torch.randn(8, 4, dtype=torch.bfloat16))
        ssm[block].copy_(torch.randn(2, 7, 9, dtype=torch.bfloat16))
    conv[0].fill_(float("nan"))
    ssm[0].fill_(float("nan"))
    conv_before, ssm_before = conv.clone(), ssm.clone()

    def run(source, destination):
        src = torch.tensor([source], dtype=torch.int32, device="cuda")
        dst = torch.tensor([destination], dtype=torch.int32, device="cuda")
        return prepare_prefill_state_inputs(conv, ssm, src, dst, dtype=torch.bfloat16)

    # Read the far source page.
    actual, history = run(2, 1)
    _assert_bitwise(actual[0], ssm_before[2])
    assert history.tolist() == [True]
    _assert_bitwise(conv[1], conv_before[2])
    # Write the far destination page.
    conv[1].copy_(conv_before[1])
    actual, history = run(1, 2)
    _assert_bitwise(actual[0], ssm_before[1])
    _assert_bitwise(conv[2], conv_before[1])
    _assert_bitwise(conv[0], conv_before[0])
    _assert_bitwise(ssm, ssm_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_large_row_offsets():
    if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        pytest.skip("large-offset test requires 6 GiB free")
    conv = torch.randn(3, 8, 4, dtype=torch.bfloat16, device="cuda")
    state = torch.randn(3, 4, 128, 128, dtype=torch.bfloat16, device="cuda")
    state[0].fill_(float("nan"))
    # The last row of the recurrent output starts at 2^15 * 2^16 = 2^31
    # elements, so the row offset needs int64.
    rows = 2**15 + 1
    src = torch.zeros(rows, dtype=torch.int32, device="cuda")
    src[-1] = 2
    dst = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
    conv_before = conv.clone()
    actual, history = prepare_prefill_state_inputs(
        conv, state, src, dst, dtype=torch.bfloat16
    )
    _assert_bitwise(actual[-1], state[2])
    _assert_bitwise(actual[-2], torch.zeros_like(state[2]))
    _assert_bitwise(actual[0], torch.zeros_like(state[2]))
    assert history[-1].item() and not history[:-1].any().item()
    _assert_bitwise(conv, conv_before)
