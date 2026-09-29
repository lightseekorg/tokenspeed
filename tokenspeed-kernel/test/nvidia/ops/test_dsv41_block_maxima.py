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
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as impl
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _block_maxima_kernel,
    block_maxima,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)


def reference(logits, visible):
    rows, width = logits.shape
    if width % 8:
        logits = torch.nn.functional.pad(logits, (0, (-width) % 8), value=-float("inf"))
    result = logits.reshape(rows, (width + 7) // 8, 8).amax(-1)
    newest = ((visible.to(torch.int64) - 1) // 8).clamp(0, result.shape[1] - 1)
    row = torch.arange(rows, device=logits.device)
    current = result[row, newest]
    result[row, newest] = torch.where(
        current > -float("inf"), torch.full_like(current, float("inf")), current
    )
    return result


@pytest.mark.parametrize("width", [1, 7, 8, 9, 1023, 1024, 1025, 16384, 65537])
@pytest.mark.parametrize("strided", [False, True])
def test_block_maxima(width, strided):
    torch.manual_seed(37)
    x = torch.randn((9, width * (2 if strided else 1)), device="cuda")
    if strided:
        x = x[:, ::2]
    visible = torch.tensor(
        [0, 1, 7, 8, 9, width, width + 8, -1, 0], device="cuda", dtype=torch.int32
    )
    x[0].fill_(-float("inf"))
    x[1, 0] = float("nan")
    x[2, -1] = float("inf")
    x[3, -1] = float("nan")
    x[4].fill_(-float("inf"))
    torch.testing.assert_close(
        block_maxima(x, visible), reference(x, visible), atol=0, rtol=0, equal_nan=True
    )


def test_graph_refresh():
    x = torch.randn((8, 1025), device="cuda")
    visible = torch.full((8,), 1025, device="cuda", dtype=torch.int32)
    block_maxima(x, visible)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = block_maxima(x, visible)
    for length in [0, 1, 8, 1025]:
        x.normal_()
        visible.fill_(length)
        graph.replay()
        torch.testing.assert_close(output, reference(x, visible), atol=0, rtol=0)


def test_empty_rows():
    x = torch.empty((0, 17), device="cuda")
    visible = torch.empty((0,), device="cuda", dtype=torch.int32)
    assert block_maxima(x, visible).shape == (0, 3)


def test_capacity_does_not_recompile():
    def run(rows, width):
        x = torch.randn((rows, width), device="cuda")
        visible = torch.full((rows,), width, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(
            block_maxima(x, visible), reference(x, visible), atol=0, rtol=0
        )

    for width in [1024, 1025, 1032]:
        run(8, width)
    with assert_no_triton_compile(_block_maxima_kernel):
        for rows, width in [(1, 2048), (17, 2049), (32, 2056), (8, 16384), (3, 16385)]:
            run(rows, width)


def select(logits, visible, bounded):
    maxima, block_ends = impl.block_maxima_with_lengths(logits, visible)
    k = min(2048, maxima.shape[1])
    ids = torch.empty((logits.shape[0], k), device="cuda", dtype=torch.int32)
    lengths = torch.empty((logits.shape[0],), device="cuda", dtype=torch.int32)
    ends = block_ends if bounded else None
    impl._select(maxima, k, 2048, ids, lengths, ends=ends)
    return ids, lengths


def canonical(ids):
    return ids.sort(dim=1).values


@pytest.mark.parametrize(
    "visible_values",
    [[0, 1, 1001, 1330], [7, 8, 9, 16384], [16385, 20000, 65535, 65536]],
)
def test_candidate_sets_and_downstream(visible_values):
    torch.manual_seed(37)
    width = 1048640
    visible = torch.tensor(visible_values, device="cuda", dtype=torch.int32)
    logits = torch.randn((4, width), device="cuda")
    logits.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= visible[:, None], -float("inf")
    )
    # Invalid pages can leave holes inside the visible prefix.
    logits[:, 64:128] = -float("inf")
    old, old_lengths = select(logits, visible, False)
    new, new_lengths = select(logits, visible, True)
    torch.testing.assert_close(old_lengths, new_lengths, atol=0, rtol=0)
    torch.testing.assert_close(canonical(old), canonical(new), atol=0, rtol=0)
    results = []
    # Feed both candidate orders through the actual candidate gather and TopK.
    for candidates in (old, new):
        scores = impl.candidate_scores(logits, candidates)
        ids = torch.empty((4, 512), device="cuda", dtype=torch.int32)
        lengths = torch.empty((4,), device="cuda", dtype=torch.int32)
        impl._select(scores, 512, 512, ids, lengths, candidates)
        results.append((canonical(ids), lengths))
    for x, y in zip(*results):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_bounded_selection_graph_refresh():
    width = 1048640
    torch.manual_seed(73)
    original = torch.randn((2, width), device="cuda")
    logits = original.clone()
    visible = torch.tensor([0, 1001], device="cuda", dtype=torch.int32)
    columns = torch.arange(width, device="cuda")[None, :]
    logits.masked_fill_(columns >= visible[:, None], -float("inf"))
    select(logits, visible, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = select(logits, visible, True)
    for values in [[1330, 8], [0, 0], [20000, 16385]]:
        visible.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        logits.copy_(original).masked_fill_(columns >= visible[:, None], -float("inf"))
        graph.replay()
        expected = select(logits, visible, False)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        torch.testing.assert_close(
            canonical(actual[0]), canonical(expected[0]), atol=0, rtol=0
        )
