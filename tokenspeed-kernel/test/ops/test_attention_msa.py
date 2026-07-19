# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Numerical tests for the Triton MSA kernels."""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.minimax_indexer import minimax_indexer
from tokenspeed_kernel.ops.attention.triton.minimax_sparse_attention import (
    minimax_sparse_attention,
)

_BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOPK = 16

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MSA Triton kernels require a GPU",
)


def _reference_selected_blocks(
    query: torch.Tensor,
    keys: torch.Tensor,
    query_position: int,
) -> torch.Tensor:
    visible_keys = keys[: query_position + 1]
    scores = query.float() @ visible_keys.float().T
    scores *= _HEAD_DIM**-0.5
    num_blocks = math.ceil((query_position + 1) / _BLOCK_SIZE)
    scores = torch.nn.functional.pad(
        scores,
        (0, num_blocks * _BLOCK_SIZE - scores.numel()),
        value=-torch.inf,
    )
    block_scores = scores.view(num_blocks, _BLOCK_SIZE).amax(dim=-1)
    block_scores[query_position // _BLOCK_SIZE] = torch.inf
    return block_scores.topk(min(_TOPK, num_blocks)).indices


def _reference_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_table: torch.Tensor,
    query_position: int,
) -> torch.Tensor:
    blocks = selected_blocks.long()
    keys = torch.cat(
        [key_cache[block_table[block].long(), 0] for block in blocks], dim=0
    )
    values = torch.cat(
        [value_cache[block_table[block].long(), 0] for block in blocks], dim=0
    )
    key_positions = (
        blocks[:, None] * _BLOCK_SIZE
        + torch.arange(_BLOCK_SIZE, device=query.device)[None]
    ).flatten()
    visible = key_positions <= query_position
    probabilities = torch.softmax(
        query.float() @ keys[visible].float().T * (_HEAD_DIM**-0.5),
        dim=-1,
    )
    return probabilities @ values[visible].float()


@requires_cuda
def test_msa_prefill_and_decode_after_2048() -> None:
    torch.manual_seed(20260714)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        prefill_len = 2305
        num_blocks = math.ceil(prefill_len / _BLOCK_SIZE)
        num_pages = num_blocks + 1  # Physical page zero is the dummy page.
        block_table = torch.arange(
            1,
            num_pages,
            dtype=torch.int32,
            device="cuda",
        )[None]
        positions = torch.arange(prefill_len, device="cuda")
        slot_mapping = (
            (positions // _BLOCK_SIZE + 1) * _BLOCK_SIZE + positions % _BLOCK_SIZE
        ).to(torch.int32)
        index_query = torch.randn(
            prefill_len,
            1,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        index_key = torch.randn(
            prefill_len,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        index_key_cache = torch.zeros(
            num_pages * _BLOCK_SIZE,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device="cuda")
        cu_seqlens = torch.tensor([0, prefill_len], dtype=torch.int32, device="cuda")
        prefix_lens = torch.zeros(1, dtype=torch.int32, device="cuda")

        selected = minimax_indexer(
            index_query,
            index_key,
            index_key_cache,
            slot_mapping,
            block_table,
            seq_lens,
            topk=_TOPK,
            scale=_HEAD_DIM**-0.5,
            init_blocks=0,
            local_blocks=1,
            cu_seqlens_q=cu_seqlens,
            prefix_lens=prefix_lens,
            max_query_len=prefill_len,
            max_blocks=num_blocks,
        )

        for query_position in (0, 127, 128, 2047, 2048, prefill_len - 1):
            expected = _reference_selected_blocks(
                index_query[query_position, 0],
                index_key,
                query_position,
            )
            actual = selected[query_position, 0, : expected.numel()]
            assert set(actual.cpu().tolist()) == set(expected.cpu().tolist())

        key_cache = torch.randn(
            num_pages,
            1,
            _BLOCK_SIZE,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        value_cache = torch.randn_like(key_cache)
        query = torch.randn(
            1,
            16,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        prefill_output = minimax_sparse_attention(
            query,
            key_cache,
            value_cache,
            selected[-1:].contiguous(),
            block_table,
            seq_lens,
            scale=_HEAD_DIM**-0.5,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            prefix_lens=torch.tensor(
                [prefill_len - 1], dtype=torch.int32, device="cuda"
            ),
            max_query_len=1,
        )
        prefill_reference = _reference_sparse_attention(
            query,
            key_cache,
            value_cache,
            selected[-1, 0],
            block_table[0],
            prefill_len - 1,
        )
        torch.testing.assert_close(
            prefill_output.float(),
            prefill_reference,
            atol=2e-3,
            rtol=2e-2,
        )

        decode_position = prefill_len
        decode_index_query = torch.randn(
            1, 1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        decode_index_key = torch.randn(
            1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        decode_slot = torch.tensor(
            [num_blocks * _BLOCK_SIZE + 1],
            dtype=torch.int32,
            device="cuda",
        )
        decode_seq_lens = torch.tensor(
            [prefill_len + 1], dtype=torch.int32, device="cuda"
        )
        decode_selected = minimax_indexer(
            decode_index_query,
            decode_index_key,
            index_key_cache,
            decode_slot,
            block_table,
            decode_seq_lens,
            topk=_TOPK,
            scale=_HEAD_DIM**-0.5,
            init_blocks=0,
            local_blocks=1,
            decode_query_len=1,
            max_blocks=num_blocks,
        )
        all_index_keys = torch.cat([index_key, decode_index_key], dim=0)
        decode_expected = _reference_selected_blocks(
            decode_index_query[0, 0],
            all_index_keys,
            decode_position,
        )
        assert set(decode_selected[0, 0].cpu().tolist()) == set(
            decode_expected.cpu().tolist()
        )

        decode_query = torch.randn(
            1,
            16,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        decode_output = minimax_sparse_attention(
            decode_query,
            key_cache,
            value_cache,
            decode_selected,
            block_table,
            decode_seq_lens,
            scale=_HEAD_DIM**-0.5,
            decode_query_len=1,
        )
        decode_reference = _reference_sparse_attention(
            decode_query,
            key_cache,
            value_cache,
            decode_selected[0, 0],
            block_table[0],
            decode_position,
        )
        torch.testing.assert_close(
            decode_output.float(),
            decode_reference,
            atol=2e-3,
            rtol=2e-2,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
