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

"""Masked row gathers preserve values and do not specialize on row counts."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import verify_query_prefixes
from tokenspeed_kernel.ops.metadata.query_rows import (
    _gather_query_rows_kernel,
    gather_query_rows,
)
from utils import assert_no_triton_compile


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_gather_aligned_rows_zeros_invalid_without_reading_them(device, dtype):
    values = (
        torch.arange(48, dtype=dtype, device=device).reshape(6, 8),
        torch.arange(6 * 3 * 4, dtype=dtype, device=device).reshape(6, 3, 4),
    )
    indices = torch.tensor([4, 1000, 1, -1000], device=device)
    valid = torch.tensor([True, False, True, False], device=device)
    outputs = gather_query_rows(values, indices, valid)
    for value, output in zip(values, outputs):
        torch.testing.assert_close(output[0], value[4], rtol=0, atol=0)
        torch.testing.assert_close(output[2], value[1], rtol=0, atol=0)
        assert torch.count_nonzero(output[1]) == 0
        assert torch.count_nonzero(output[3]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_query_row_count_and_source_stride_do_not_compile():
    def launch(rows, selected, padding):
        values = tuple(
            torch.randn(rows, width + padding, device="cuda")[:, :width]
            for width in (8, 256, 1024)
        )
        indices = torch.arange(selected, device="cuda") % rows
        valid = torch.ones(selected, device="cuda", dtype=torch.bool)
        actual = gather_query_rows(values, indices, valid)
        for source, gathered in zip(values, actual):
            torch.testing.assert_close(gathered, source[indices], rtol=0, atol=0)

    launch(8, 4, 0)
    with assert_no_triton_compile(_gather_query_rows_kernel):
        for rows, selected, padding in ((9, 5, 3), (16, 16, 16), (32, 7, 7)):
            launch(rows, selected, padding)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_packed_probabilities_replay_masks_and_limit_prefixes():
    rates = torch.tensor(
        [[float("nan"), 1, 1], [1, 1, 1], [1, 1, 1], [0.5, float("inf"), 1]],
        device="cuda",
    )
    active = torch.tensor([True, False, True, True], device="cuda")
    ready = torch.tensor([True, True, False, True], device="cuda")
    rows = torch.empty(7, device="cuda", dtype=torch.int64)
    offsets = torch.empty(5, device="cuda", dtype=torch.int32)
    lengths = torch.empty(4, device="cuda", dtype=torch.int32)

    def run():
        verify_query_prefixes(
            rates,
            active,
            ready,
            rows_out=rows,
            offsets_out=offsets,
            lengths_out=lengths,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    assert lengths.tolist() == [4, 0, 1, 2]
    assert offsets.tolist() == [0, 4, 4, 5, 7]
    assert rows.tolist() == [0, 1, 2, 3, 8, 12, 13]

    active.copy_(torch.tensor([False, True, False, True], device="cuda"))
    ready[1] = False
    graph.replay()
    assert lengths.tolist() == [0, 1, 0, 4]
    assert offsets.tolist() == [0, 0, 1, 1, 5]
    assert rows.tolist() == [4, 12, 13, 14, 15, -1, -1]

    active.zero_()
    graph.replay()
    assert lengths.tolist() == [0] * 4
    assert offsets.tolist() == [0] * 5
    assert rows.tolist() == [-1] * 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_packed_layout_runtime_capacities_do_not_compile():
    from tokenspeed_kernel.ops.metadata.query_rows import _verify_layout_kernel
    from utils import assert_no_triton_compile

    rates = torch.full((32, 5), 0.5, device="cuda")
    active = torch.ones(32, dtype=torch.bool, device="cuda")

    def run(batch, capacity):
        rows = torch.empty(capacity, device="cuda", dtype=torch.int64)
        offsets = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
        lengths = torch.empty(batch, device="cuda", dtype=torch.int32)
        verify_query_prefixes(
            rates[:batch],
            active[:batch],
            active[:batch],
            rows_out=rows,
            offsets_out=offsets,
            lengths_out=lengths,
        )
        assert offsets[-1].item() == capacity

    run(32, 128)
    with assert_no_triton_compile(_verify_layout_kernel):
        run(24, 72)
        run(25, 100)
