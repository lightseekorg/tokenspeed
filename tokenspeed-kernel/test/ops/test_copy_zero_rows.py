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
from tokenspeed_kernel.ops.copy import zero_rows_after
from tokenspeed_kernel.ops.copy.triton import _zero_rows_after_kernel
from utils import assert_no_triton_compile


def _check(tensor: torch.Tensor, live: int) -> None:
    before = tensor.clone()
    zero_rows_after(tensor, torch.tensor([live], device=tensor.device))
    torch.testing.assert_close(tensor[:live], before[:live], rtol=0, atol=0)
    assert tensor[live:].eq(0).all()


@pytest.mark.parametrize("live", [0, 1, 15, 16, 37, 64])
@pytest.mark.parametrize(
    "make",
    [
        lambda d: torch.randn(64, 300, device=d, dtype=torch.bfloat16),
        lambda d: torch.randn(64, device=d),
        lambda d: torch.randn(64, 3, 5, device=d, dtype=torch.float16),
        # A column slice of a wider projection, and a transposed view: the
        # rows are strided, so a reshape would zero a copy.
        lambda d: torch.randn(64, 700, device=d, dtype=torch.bfloat16)[:, 128:640],
        lambda d: torch.randn(300, 64, device=d, dtype=torch.bfloat16).t(),
    ],
)
def test_zeroes_rows_past_the_count_and_keeps_live_rows(device, make, live) -> None:
    _check(make(device), live)


def test_rejects_trailing_dims_that_are_not_one_view(device) -> None:
    tensor = torch.randn(8, 4, 6, device=device).transpose(1, 2)
    with pytest.raises(RuntimeError):
        zero_rows_after(tensor, torch.tensor([2], device=device))


def test_captured_graph_reads_the_count_at_replay(device) -> None:
    rows = torch.ones(48, 96, device=device, dtype=torch.bfloat16)
    live = torch.tensor([40], dtype=torch.int32, device=device)
    zero_rows_after(rows, live)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        zero_rows_after(rows, live)
    for count in (5, 33, 48):
        rows.fill_(1)
        live.fill_(count)
        graph.replay()
        assert rows[:count].eq(1).all() and rows[count:].eq(0).all()


def test_varying_shapes_and_counts_reuse_one_compile(device) -> None:
    for rows in (16, 17):
        zero_rows_after(
            torch.ones(rows, 32, device=device), torch.tensor([3], device=device)
        )
    with assert_no_triton_compile(_zero_rows_after_kernel):
        for rows, cols, live in ((352, 6144, 300), (8192, 128, 8191), (1792, 37, 0)):
            _check(torch.ones(rows, cols, device=device), live)
        _check(torch.ones(512, 1024, device=device)[:, :999].t(), 7)
