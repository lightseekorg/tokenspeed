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

"""trtllm-gen MHA decode through the FlashInfer solution: long decodes split
the KV across CTAs, whose semaphores every call shares."""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(
        not platform.is_blackwell, reason="trtllm-gen attention requires SM10x"
    ),
]

PAGE = 64
HEAD_DIM = 128
HEADS_Q = 16
HEADS_KV = 2


def _inputs(seq_lens: list[int], q_len: int, seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [math.ceil(n / PAGE) for n in seq_lens]
    shape = (sum(pages), PAGE, HEADS_KV, HEAD_DIM)
    # Each request's pages in reverse order, so no table is the identity.
    table = torch.zeros(len(seq_lens), max(pages), dtype=torch.int32, device="cuda")
    first = 0
    for row, n in enumerate(pages):
        table[row, :n] = torch.arange(first + n - 1, first - 1, -1, device="cuda")
        first += n
    q_shape = (len(seq_lens) * q_len, HEADS_Q, HEAD_DIM)
    return dict(
        q=torch.randn(q_shape, device="cuda", generator=g).bfloat16(),
        k_cache=torch.randn(shape, device="cuda", generator=g).bfloat16(),
        v_cache=torch.randn(shape, device="cuda", generator=g).bfloat16(),
        page_table=table,
        cache_seqlens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        max_seqlen_k=max(seq_lens),
        max_seqlen_q=q_len,
    )


def _reference(inputs: dict) -> torch.Tensor:
    """FP32 causal attention per request: query row i sees keys up to
    seq_len - q_len + i."""
    q_len = inputs["max_seqlen_q"]
    outs = []
    for row, seq_len in enumerate(inputs["cache_seqlens"].tolist()):
        q = inputs["q"][row * q_len : (row + 1) * q_len].float()
        pages = inputs["page_table"][row].long()
        k = inputs["k_cache"][pages].flatten(0, 1)[:seq_len].float()
        v = inputs["v_cache"][pages].flatten(0, 1)[:seq_len].float()
        k = k.repeat_interleave(HEADS_Q // HEADS_KV, dim=1)
        v = v.repeat_interleave(HEADS_Q // HEADS_KV, dim=1)
        logits = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(HEAD_DIM)
        keys = torch.arange(seq_len, device="cuda")
        last = seq_len - q_len + torch.arange(q_len, device="cuda")
        logits = logits.masked_fill(keys[None, :] > last[:, None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), v))
    return torch.cat(outs)


@pytest.mark.parametrize(
    "seq_lens, q_len",
    [
        ([16384], 1),
        ([16384], 4),
        ([16384, 3000, 9000, 700, 16384, 64, 5000, 12000], 4),
    ],
)
def test_repeated_and_replayed_decodes_agree(seq_lens: list[int], q_len: int):
    """Decodes that split their KV across CTAs, run three times eagerly and
    replayed from a CUDA graph: every result is the first one, bit for bit,
    and matches the reference."""
    from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

    inputs = _inputs(seq_lens, q_len, seed=len(seq_lens) * q_len)

    def decode() -> torch.Tensor:
        return mha_decode_with_kvcache(solution="flashinfer", **inputs)

    first = decode()
    torch.testing.assert_close(first.float(), _reference(inputs), atol=2e-2, rtol=2e-2)
    for _ in range(2):
        assert torch.equal(decode(), first)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = decode()
    for _ in range(2):
        replayed.zero_()
        graph.replay()
        assert torch.equal(replayed, first)
    # An eager call after the replays still sees reset counters.
    assert torch.equal(decode(), first)
