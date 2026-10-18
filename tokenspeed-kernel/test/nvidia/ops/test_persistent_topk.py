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
from tokenspeed_kernel.thirdparty.cuda.dsv4_attention import (
    has_persistent_topk,
    persistent_topk,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)

# Cover both register sizes, single-CTA streaming, and every cluster variant.
SHAPES = [
    (1, 8192),
    (1, 16384),
    (33, 32768),
    (1, 65536),
    (1, 131072),
    (1, 262144),
    (1, 262148),
]


@pytest.fixture(autouse=True)
def require_persistent_topk():
    if not has_persistent_topk():
        pytest.skip("persistent CUDA top-k is unavailable")


def assert_selection(scores, lengths, output):
    topk = output.shape[1]
    for row, length in enumerate(lengths.cpu().tolist()):
        selected = output[row]
        live = min(length, topk)
        if length <= topk:
            expected = torch.full_like(selected, -1)
            expected[:live] = torch.arange(live, device="cuda", dtype=torch.int32)
            torch.testing.assert_close(selected, expected, atol=0, rtol=0)
            continue
        assert bool(((selected >= 0) & (selected < length)).all())
        assert selected.unique().numel() == topk
        actual = scores[row, selected.long()].sort().values
        expected = scores[row, :length].topk(topk).values.sort().values
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows,width", SHAPES)
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    "distribution", ["positive", "negative", "fp16_overflow", "subnormal"]
)
def test_threshold_bucket_overflow(rows, width, topk, enable_pdl, distribution):
    columns = torch.arange(width, device="cuda", dtype=torch.float32)
    if distribution == "subnormal":
        row = (columns.to(torch.int32) * 16 + 1).view(torch.float32)
    elif distribution == "fp16_overflow":
        row = 1.0e6 + columns * 0.125
    else:
        row = (1.0 if distribution == "positive" else -1.0) + columns * 2**-23
    scores = row.expand(rows, -1).contiguous()
    # Include already accepted scores above the overflowing threshold bucket.
    scores[:, : topk // 2 + 1] = float("inf")
    lengths = torch.full((rows,), width - 1, device="cuda", dtype=torch.int32)
    scores[:, -1] = float("inf")  # A poisoned tail must never be selected.
    output = torch.full((rows, topk), -77, device="cuda", dtype=torch.int32)
    persistent_topk(
        scores,
        lengths,
        output,
        topk,
        width,
        q_len_per_req=1,
        enable_pdl=enable_pdl,
    )
    assert_selection(scores, lengths, output)


@pytest.mark.parametrize(
    "rows,width,hist_bits",
    [(1, 8192, 12), (33, 32768, 12), (1, 65536, 9), (1, 131072, 10)],
)
def test_negative_coarse_bucket_midpoint(rows, width, hist_bits):
    # At a negative bucket boundary, nearest-even rounds the exact midpoint
    # into the lower bucket. Counting it as strictly above can hide higher scores.
    half_bits = 0xBC00 + (1 << (16 - hist_bits)) - 1
    neighbors = torch.tensor([half_bits, half_bits + 1], dtype=torch.uint16)
    midpoint = neighbors.view(torch.float16).float().mean().item()
    topk = 512
    scores = torch.full((rows, width), -4.0, device="cuda")
    scores[:, :1024] = midpoint
    scores[:, -topk // 2 :] = -0.5
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    output = torch.full((rows, topk), -77, device="cuda", dtype=torch.int32)
    persistent_topk(
        scores,
        lengths,
        output,
        topk,
        width,
        q_len_per_req=1,
        enable_pdl=False,
    )
    assert_selection(scores, lengths, output)


@pytest.mark.parametrize("rows,width", SHAPES)
@pytest.mark.parametrize("value", [0.0, float("inf"), -float("inf")])
@pytest.mark.parametrize("topk", [512, 2048])
def test_exact_value_ties(rows, width, value, topk):
    scores = torch.full((rows, width), value, device="cuda")
    if value == 0.0:
        scores[:, ::2] = -0.0
    lengths = torch.full((rows,), width - 1, device="cuda", dtype=torch.int32)
    output = torch.full((rows, topk), -77, device="cuda", dtype=torch.int32)
    persistent_topk(
        scores,
        lengths,
        output,
        topk,
        width,
        q_len_per_req=1,
        enable_pdl=False,
    )
    assert_selection(scores, lengths, output)


@pytest.mark.parametrize("num_ties", [33, 65, 127])
def test_small_negative_infinity_ties(num_ties):
    topk, width = 512, 8192
    length = topk - 1 + num_ties
    scores = torch.full((1, width), float("inf"), device="cuda")
    scores[:, : topk - 1] = 1.0
    scores[:, topk - 1 : length] = -float("inf")
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    output = torch.full((1, topk), -77, device="cuda", dtype=torch.int32)
    persistent_topk(
        scores,
        lengths,
        output,
        topk,
        width,
        q_len_per_req=1,
        enable_pdl=False,
    )
    assert_selection(scores, lengths, output)


def test_overflow_grouped_query_causality():
    topk, width, q_len_per_req = 512, 65536, 4
    request_lengths = torch.tensor([8192, width], device="cuda", dtype=torch.int32)
    lengths = (
        request_lengths[:, None]
        - q_len_per_req
        + 1
        + torch.arange(q_len_per_req, device="cuda")
    ).flatten()
    columns = torch.arange(width, device="cuda")
    scores = (1.0 + columns * 2**-23).expand(lengths.numel(), -1).contiguous()
    scores.masked_fill_(columns[None, :] >= lengths[:, None], float("inf"))
    output = torch.full((lengths.numel(), topk), -77, device="cuda", dtype=torch.int32)
    persistent_topk(
        scores,
        request_lengths,
        output,
        topk,
        width,
        q_len_per_req=q_len_per_req,
        enable_pdl=True,
    )
    assert_selection(scores, lengths, output)


@pytest.mark.parametrize("rows,width", [(1, 8192), (1, 131072)])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_overflow_graph_refresh(rows, width, enable_pdl):
    topk = 2048
    scores = (1.0 + torch.arange(width, device="cuda") * 2**-23).unsqueeze(0)
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    output = torch.full((rows, topk), -77, device="cuda", dtype=torch.int32)

    def select():
        persistent_topk(
            scores,
            lengths,
            output,
            topk,
            width,
            q_len_per_req=1,
            enable_pdl=enable_pdl,
        )

    select()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    for length in [width, 0, 17, topk, width - 1, 4096, width]:
        lengths.fill_(length)
        output.fill_(-77)
        graph.replay()
        assert_selection(scores, lengths, output)
