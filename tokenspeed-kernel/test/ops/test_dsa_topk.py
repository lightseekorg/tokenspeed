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
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    dsa_decode_topk,
    dsa_prefill_topk,
)
from tokenspeed_kernel.platform import current_platform


def _dsa_scores(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    k = index_k.index_select(0, slots.long()).float()
    return (torch.einsum("hd,nd->nh", q.float(), k) * weights.float()).sum(
        -1
    ) * softmax_scale


def test_dsa_decode_topk_matches_reference(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("DSA Triton top-k is currently targeted at AMD CDNA4")

    torch.manual_seed(1)
    num_tokens = 3
    num_heads = 4
    head_dim = 128
    page_size = 8
    num_pages = 5
    topk = 8
    softmax_scale = 0.3
    num_slots = num_pages * page_size

    q = torch.randn(
        (num_tokens, num_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    index_k = torch.randn((num_slots, head_dim), device=device, dtype=torch.bfloat16)
    weights = torch.randn((num_tokens, num_heads), device=device, dtype=torch.float32)
    seq_lens = torch.tensor([31, 19, 27], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [
            [4, 1, 3, 0, 2],
            [2, 0, 4, 1, 3],
            [1, 3, 0, 4, 2],
        ],
        device=device,
        dtype=torch.int32,
    )

    actual_slots, actual_lens = dsa_decode_topk(
        q,
        index_k,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    expected_slots = torch.empty_like(actual_slots)
    expected_lens = torch.minimum(seq_lens, torch.full_like(seq_lens, topk))
    for token in range(num_tokens):
        local = torch.arange(int(seq_lens[token].item()), device=device)
        pages = block_table[token].index_select(0, local // page_size)
        slots = pages * page_size + local % page_size
        scores = _dsa_scores(q[token], index_k, weights[token], slots, softmax_scale)
        top_local = torch.topk(scores, k=topk, sorted=True).indices
        expected_slots[token] = slots.index_select(0, top_local).to(torch.int32)

    torch.testing.assert_close(actual_slots, expected_slots, atol=0, rtol=0)
    torch.testing.assert_close(actual_lens, expected_lens, atol=0, rtol=0)


def test_dsa_decode_topk_streams_multiple_blocks(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("DSA Triton top-k is currently targeted at AMD CDNA4")

    torch.manual_seed(3)
    num_tokens = 1
    num_heads = 2
    head_dim = 128
    page_size = 64
    num_pages = 36
    topk = 64
    softmax_scale = 0.2
    num_slots = num_pages * page_size

    q = torch.randn(
        (num_tokens, num_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    index_k = torch.randn((num_slots, head_dim), device=device, dtype=torch.bfloat16)
    weights = torch.randn((num_tokens, num_heads), device=device, dtype=torch.float32)
    seq_lens = torch.tensor([num_slots], device=device, dtype=torch.int32)
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(1, -1)

    actual_slots, actual_lens = dsa_decode_topk(
        q,
        index_k,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    slots = torch.arange(num_slots, device=device)
    scores = _dsa_scores(q[0], index_k, weights[0], slots, softmax_scale)
    expected_slots = torch.topk(scores, k=topk, sorted=True).indices.to(torch.int32)

    torch.testing.assert_close(actual_slots[0], expected_slots, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_lens, seq_lens.new_full((1,), topk), atol=0, rtol=0
    )


def test_dsa_prefill_topk_matches_reference(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("DSA Triton top-k is currently targeted at AMD CDNA4")

    torch.manual_seed(2)
    num_tokens = 4
    num_heads = 3
    head_dim = 128
    topk = 4
    softmax_scale = 0.4
    num_slots = 40
    seq_len_sum = 23

    q = torch.randn(
        (num_tokens, num_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    index_k = torch.randn((num_slots, head_dim), device=device, dtype=torch.bfloat16)
    weights = torch.randn((num_tokens, num_heads), device=device, dtype=torch.float32)
    kv_workspace_slots = torch.tensor(
        [
            7,
            2,
            9,
            11,
            4,
            8,
            12,
            1,
            5,
            6,
            3,
            10,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ],
        device=device,
        dtype=torch.int64,
    )
    assert kv_workspace_slots.numel() == seq_len_sum
    row_starts = torch.tensor([0, 3, 7, 11], device=device, dtype=torch.int32)
    row_ends = torch.tensor([6, 12, 18, 23], device=device, dtype=torch.int32)

    actual_indices, actual_lens = dsa_prefill_topk(
        q,
        index_k,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    expected_indices = torch.empty_like(actual_indices)
    expected_lens = torch.minimum(
        row_ends - row_starts, torch.full_like(row_ends, topk)
    )
    for token in range(num_tokens):
        rows = torch.arange(
            int(row_starts[token].item()),
            int(row_ends[token].item()),
            device=device,
        )
        slots = kv_workspace_slots.index_select(0, rows)
        scores = _dsa_scores(q[token], index_k, weights[token], slots, softmax_scale)
        top_local = torch.topk(scores, k=topk, sorted=True).indices
        expected_indices[token] = rows.index_select(0, top_local).to(torch.int32)

    torch.testing.assert_close(actual_indices, expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(actual_lens, expected_lens, atol=0, rtol=0)
