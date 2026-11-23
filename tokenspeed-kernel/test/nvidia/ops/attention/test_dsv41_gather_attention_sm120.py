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
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41.flashinfer import selected_attention


@pytest.fixture(autouse=True)
def use_gather_attention(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_SELECTED_ATTENTION", "gather")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires SM120",
)


def make_case(tokens, heads, extra, sink):
    torch.manual_seed(927)
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    caches = []
    for fmt, width in (("swa", 528), ("global", 288)):
        backing = torch.zeros(12, 64 * width + 512, dtype=torch.uint8, device="cuda")
        cache = backing[:, : 64 * width].view(12, 64, width)
        values = torch.randn(768, 512, device="cuda", dtype=torch.bfloat16)
        dsv41.cache_scatter(values, cache, torch.arange(768, device="cuda"), fmt)
        caches.append(cache)
    ss = torch.randint(0, 768, (tokens, 128), device="cuda", dtype=torch.int32)
    gs = torch.randint(0, 768, (tokens, extra), device="cuda", dtype=torch.int32)
    ss[:, ::11] = -1
    ss[:, 4] = 900
    sl = torch.full((tokens,), 119, device="cuda", dtype=torch.int32)
    gl = torch.full_like(sl, max(0, extra - 13))
    if tokens > 1:
        sl[0] = 0
        gl[0] = 0
    bias = torch.linspace(-2, 2, heads, device="cuda") if sink else None
    return (
        q,
        caches[0],
        ss,
        sl,
        caches[1] if extra else None,
        gs if extra else None,
        gl if extra else None,
        bias,
    )


def reference(case):
    q, swa, ss, sl, glob, gs, gl, sink = case
    values, masks = [], []
    for cache, slots, lens, fmt in [(swa, ss, sl, "swa")] + (
        [] if glob is None else [(glob, gs, gl, "global")]
    ):
        values.append(dsv41.cache_gather(cache, slots, fmt, None).float())
        masks.append(
            (slots >= 0)
            & (slots < cache.shape[0] * 64)
            & (torch.arange(slots.shape[1], device="cuda") < lens[:, None])
        )
    kv, mask = torch.cat(values, 1), torch.cat(masks, 1)
    scores = torch.bmm(q.float(), kv.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~mask[:, None], -torch.inf)
    if sink is not None:
        scores = torch.cat([scores, sink[None, :, None].expand(q.shape[0], -1, -1)], -1)
    lse = scores.logsumexp(-1)
    prob = scores.softmax(-1).nan_to_num(0)
    if sink is not None:
        prob = prob[..., :-1]
    return torch.bmm(prob, kv).bfloat16(), lse


def run(case, schedule):
    q = case[0]
    lse = (
        torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
        if case[-1] is None
        else None
    )
    out = selected_attention(*case, 512**-0.5, None, 256, schedule, None, None, lse=lse)
    return out, lse


@pytest.mark.parametrize(
    "tokens,heads,extra",
    [(1, 2, 0), (1, 8, 512), (17, 8, 512), (4, 64, 512), (65, 8, 512), (2, 16, 192)],
)
@pytest.mark.parametrize("sink", [False, True])
def test_paged_attention(tokens, heads, extra, sink):
    case = make_case(tokens, heads, extra, sink)
    expected, expected_lse = reference(case)
    actual, lse = run(case, dsv41.new_attention_schedule())
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.003)
    if tokens > 1:
        assert not actual[0].count_nonzero().item()
    if lse is not None:
        torch.testing.assert_close(lse, expected_lse, rtol=1e-4, atol=1e-3)


def test_graph_refresh_with_warmed_schedule():
    case = make_case(8, 8, 512, False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    schedule = dsv41.new_attention_schedule()
    with torch.cuda.stream(stream):
        run(case, schedule)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    schedule = dsv41.new_attention_schedule(previous=schedule)
    with torch.cuda.graph(graph):
        out, lse = run(case, schedule)
    for length in [128, 1, 0, 77]:
        case[0].normal_()
        case[3].fill_(length)
        case[6].fill_(length * 3)
        case[2].copy_(case[2].roll(1, 1))
        graph.replay()
        expected, expected_lse = reference(case)
        torch.testing.assert_close(out, expected, rtol=0.015, atol=0.003)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-4, atol=1e-3)
