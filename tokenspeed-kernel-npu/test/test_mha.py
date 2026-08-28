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

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend MHA tests require an NPU"
)


def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        q_positions = (
            k.shape[0] - q.shape[0] + torch.arange(q.shape[0], device=q.device)
        )
        k_positions = torch.arange(k.shape[0], device=q.device)
        scores.masked_fill_(
            ~(k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)).unsqueeze(0),
            torch.finfo(q.dtype).min,
        )
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


def _paged_rows(
    cache: torch.Tensor, table: torch.Tensor, batch: int, length: int
) -> torch.Tensor:
    pages = cache[table[batch].long()].reshape(-1, cache.shape[-2], cache.shape[-1])
    return pages[:length]


def test_mha_prefill() -> None:
    lengths = [3, 5]
    cumulative = torch.tensor([0, 3, 8], dtype=torch.int32, device="npu")
    q = torch.randn(8, 4, 64, dtype=torch.bfloat16, device="npu")
    k = torch.randn(8, 2, 64, dtype=torch.bfloat16, device="npu")
    v = torch.randn_like(k)

    output = mha_prefill(q, k, v, cumulative, [0, 3, 8], 5)
    expected = torch.cat(
        [
            _attention(q[:3], k[:3], v[:3], causal=True),
            _attention(q[3:], k[3:], v[3:], causal=True),
        ]
    )

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


def test_mha_extend_with_paged_cache() -> None:
    page_size = 128
    q_lengths = [2, 3]
    cache_lengths = torch.tensor([5, 70], dtype=torch.int32, device="npu")
    cumulative_q = torch.tensor([0, 2, 5], dtype=torch.int32, device="npu")
    cumulative_kv = torch.tensor([0, 5, 75], dtype=torch.int32, device="npu")
    q = torch.randn(5, 4, 64, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(4, page_size, 2, 64, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32, device="npu")

    output = mha_extend_with_kvcache(
        q,
        cumulative_q,
        cumulative_kv,
        k_cache,
        v_cache,
        page_table,
        cache_lengths,
        max(q_lengths),
        128,
        is_causal=True,
    )
    expected = []
    start = 0
    for batch, q_length in enumerate(q_lengths):
        length = int(cache_lengths[batch])
        expected.append(
            _attention(
                q[start : start + q_length],
                _paged_rows(k_cache, page_table, batch, length),
                _paged_rows(v_cache, page_table, batch, length),
                causal=True,
            )
        )
        start += q_length

    torch.testing.assert_close(output, torch.cat(expected), atol=2e-2, rtol=2e-2)


def test_mha_decode_with_paged_cache() -> None:
    page_size = 128
    q = torch.randn(2, 4, 64, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(4, page_size, 2, 64, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32, device="npu")
    cache_lengths = torch.tensor([5, 70], dtype=torch.int32, device="npu")

    output = mha_decode_with_kvcache(
        q, k_cache, v_cache, page_table, cache_lengths, max_seqlen_k=128
    )
    expected = torch.stack(
        [
            _attention(
                q[batch : batch + 1],
                _paged_rows(k_cache, page_table, batch, int(length)),
                _paged_rows(v_cache, page_table, batch, int(length)),
                causal=False,
            )[0]
            for batch, length in enumerate(cache_lengths)
        ]
    )

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


def test_mha_decode_npugraph_updates_sequence_lengths() -> None:
    page_size = 128
    q = torch.randn(2, 4, 64, dtype=torch.bfloat16, device="npu")
    k_cache = torch.randn(4, page_size, 2, 64, dtype=torch.bfloat16, device="npu")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32, device="npu")
    capture_lengths = torch.ones(2, dtype=torch.int32, device="npu")

    for _ in range(3):
        mha_decode_with_kvcache(
            q, k_cache, v_cache, page_table, capture_lengths, max_seqlen_k=128
        )
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        output = mha_decode_with_kvcache(
            q, k_cache, v_cache, page_table, capture_lengths, max_seqlen_k=128
        )
    torch.npu.synchronize()

    for lengths in ([5, 70], [12, 80]):
        graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
        graph.replay()
        torch.npu.synchronize()

        expected = torch.stack(
            [
                _attention(
                    q[batch : batch + 1],
                    _paged_rows(k_cache, page_table, batch, length),
                    _paged_rows(v_cache, page_table, batch, length),
                    causal=False,
                )[0]
                for batch, length in enumerate(lengths)
            ]
        )
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
