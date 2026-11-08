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
from tokenspeed_kernel.ops.attention.dsv41 import verify_layout


def outputs(scores, capacity):
    bs = scores.shape[0]
    return dict(
        rows_out=torch.empty(capacity, dtype=torch.int64, device=scores.device),
        indptr_out=torch.empty(bs + 1, dtype=torch.int64, device=scores.device),
        counts_out=torch.empty(bs, dtype=torch.int64, device=scores.device),
        valid_out=torch.empty((), dtype=torch.bool, device=scores.device),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "bs,draft_width,extra", [(1, 5, 0), (4, 3, 5), (32, 5, 120), (128, 5, 640)]
)
def test_layout_matches_stable_prefix_reference(device, bs, draft_width, extra):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    scores = torch.randn(
        bs, draft_width, generator=torch.Generator().manual_seed(42)
    ).to(device)
    capacity = bs + extra
    out = outputs(scores, capacity)
    verify_layout(scores, query_capacity=capacity, **out)
    probs = torch.sigmoid(scores.cpu().clamp(-60, 60)).cumprod(1).tolist()
    ordered = sorted(
        ((p, r, d) for r, row in enumerate(probs) for d, p in enumerate(row)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    counts = [1] * bs
    for _, row, _ in ordered[:extra]:
        counts[row] += 1
    rows = [
        r * (draft_width + 1) + d
        for r, count in enumerate(counts)
        for d in range(count)
    ]
    assert out["valid_out"].item()
    assert out["counts_out"].tolist() == counts
    assert out["rows_out"].tolist() == rows
    assert out["indptr_out"][-1].item() == capacity


def test_unknown_scores_require_full_prefix_and_ties_are_stable():
    scores = torch.tensor([[float("nan"), 0.0], [0.0, 0.0]])
    out = outputs(scores, 4)
    verify_layout(scores, query_capacity=4, **out)
    assert out["valid_out"].item()
    assert out["counts_out"].tolist() == [3, 1]
    out = outputs(scores, 3)
    verify_layout(scores, query_capacity=3, **out)
    assert not out["valid_out"].item()
    assert out["counts_out"].tolist() == [1, 1]
    assert out["rows_out"].tolist() == [0, 3, 0]
    scores.fill_(-1000)
    out = outputs(scores, 4)
    verify_layout(scores, query_capacity=4, **out)
    assert out["counts_out"].tolist() == [2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_graph_replay_reads_changed_scores_and_keeps_output_storage():
    scores = torch.zeros((4, 5), device="cuda")
    out = outputs(scores, 12)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            verify_layout(scores, query_capacity=12, **out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify_layout(scores, query_capacity=12, **out)
    for row in range(4):
        scores.fill_(-10)
        scores[row].fill_(10)
        graph.replay()
        expected = outputs(scores, 12)
        verify_layout(scores, query_capacity=12, **expected)
        for name, value in out.items():
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    graph.reset()
