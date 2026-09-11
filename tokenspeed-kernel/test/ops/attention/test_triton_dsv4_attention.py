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
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_decode, dsv4_prefill


def _cache(pages: int, page_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    nope = torch.randn(pages, page_size, 448, device="cuda").to(torch.float8_e4m3fn)
    rope = torch.randn(pages, page_size, 64, device="cuda", dtype=torch.bfloat16)
    # V4 pages store all token payloads, then all eight-byte scale records.
    cache = torch.full((pages, page_size * 584), 127, device="cuda", dtype=torch.uint8)
    payload = torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
    cache[:, : page_size * 576] = payload.reshape(pages, -1)
    reference = torch.cat((nope.to(torch.bfloat16), rope), dim=-1).reshape(-1, 512)
    return cache, reference


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(q)
    for token in range(q.shape[0]):
        selected = slots[token, : int(lens[token])].long()
        selected = selected[(selected >= 0) & (selected < kv.shape[0])]
        keys = kv[selected].float()
        logits = q[token].float() @ keys.T / (512**0.5)
        probabilities = torch.cat((logits, sink[:, None]), dim=1).softmax(dim=1)[:, :-1]
        result[token] = (probabilities @ keys).to(q.dtype)
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("heads", [16, 64, 128])
@pytest.mark.parametrize("extra", [False, True])
def test_selected_attention_and_graph_replay(heads: int, extra: bool) -> None:
    torch.manual_seed(74)
    cache, kv = _cache(3, 64)
    compressed, extra_kv = _cache(2, 64)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    slots = torch.arange(128, device="cuda", dtype=torch.int32).repeat(3, 1)
    slots[0, 10] = -1
    lens = torch.tensor([91, 1, 0], device="cuda", dtype=torch.int32)
    extra_slots = torch.arange(64, device="cuda", dtype=torch.int32).repeat(3, 1)
    extra_lens = torch.tensor([60, 0, 0], device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)

    def run():
        return dsv4_decode(
            q,
            cache,
            slots,
            lens,
            64,
            sink,
            512**-0.5,
            extra_kv_cache=compressed if extra else None,
            extra_slots=extra_slots if extra else None,
            extra_lens=extra_lens if extra else None,
            extra_page_size=64 if extra else None,
            out=output,
            override=None,
            solution="triton",
        )

    selected = slots.clone()
    selected.masked_fill_(
        torch.arange(128, device="cuda")[None, :] >= lens[:, None], -1
    )
    if extra:
        appended = extra_slots + kv.shape[0]
        appended.masked_fill_(
            torch.arange(64, device="cuda")[None, :] >= extra_lens[:, None], -1
        )
        selected = torch.cat((selected, appended), dim=1)
        kv = torch.cat((kv, extra_kv), dim=0)
    selected_lens = torch.full(
        (3,), selected.shape[1], device="cuda", dtype=torch.int32
    )
    expected = _reference(q, kv, selected, selected_lens, sink)
    actual = run()
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)
    prefill = dsv4_prefill(
        q,
        kv,
        selected,
        selected_lens,
        sink,
        512**-0.5,
        out=None,
        override=None,
        solution="triton",
    )
    torch.testing.assert_close(prefill, expected, atol=8e-3, rtol=8e-3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(0.5)
    graph.replay()
    expected = _reference(q, kv, selected, selected_lens, sink)
    torch.testing.assert_close(output, expected, atol=8e-3, rtol=8e-3)
