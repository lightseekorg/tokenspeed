# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Numerical tests for the Triton MSA kernels."""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention import (
    msa_decode_with_kvcache,
    msa_extend_with_kvcache,
)
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


@requires_cuda
@pytest.mark.parametrize(
    "kv_cache_dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8_e4m3"],
)
def test_msa_decode_qlen4_verify_matches_per_token_decode(
    kv_cache_dtype: torch.dtype,
) -> None:
    """Multi-query verify decode (q_len=4) must equal per-token decode.

    Verify token j at total length L is positioned like a plain decode step
    at seq_len L-4+1+j, so the batched call must reproduce four single-token
    calls: same per-token block selection and matching attention output.
    The per-token calls run after the batched call, with all four draft
    index keys already in the cache, so they also assert that token j never
    sees the in-cache future tokens j+1..3.
    """
    torch.manual_seed(20260721)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        num_verify = 4
        num_heads = 16
        prefix_lens = [2305, 2497]
        total_lens = [prefix + num_verify for prefix in prefix_lens]
        num_blocks = [math.ceil(total / _BLOCK_SIZE) for total in total_lens]
        max_blocks = max(num_blocks)
        num_pages = 1 + sum(num_blocks)  # Physical page zero is the dummy page.

        block_table = torch.zeros((2, max_blocks), dtype=torch.int32, device="cuda")
        next_page = 1
        for request, blocks in enumerate(num_blocks):
            block_table[request, :blocks] = torch.arange(
                next_page, next_page + blocks, dtype=torch.int32, device="cuda"
            )
            next_page += blocks

        def slots_for(request: int, positions: torch.Tensor) -> torch.Tensor:
            pages = block_table[request, positions // _BLOCK_SIZE].to(torch.int64)
            return (pages * _BLOCK_SIZE + positions % _BLOCK_SIZE).to(torch.int32)

        key_cache = torch.randn(
            num_pages, 1, _BLOCK_SIZE, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        value_cache = torch.randn_like(key_cache)
        if kv_cache_dtype is torch.float8_e4m3fn:
            # Identity-scale quantization; e4m3 -> bf16 is exact, so the
            # reference sees the same representable values as the kernel.
            key_cache = key_cache.to(kv_cache_dtype)
            value_cache = value_cache.to(kv_cache_dtype)
            ref_key_cache = key_cache.to(torch.bfloat16)
            ref_value_cache = value_cache.to(torch.bfloat16)
        else:
            ref_key_cache = key_cache
            ref_value_cache = value_cache
        index_key_cache = torch.zeros(
            num_pages * _BLOCK_SIZE, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        index_keys = []
        for request, (prefix, total) in enumerate(zip(prefix_lens, total_lens)):
            keys = torch.randn(total, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            index_keys.append(keys)
            prefix_positions = torch.arange(prefix, device="cuda")
            index_key_cache[slots_for(request, prefix_positions).long()] = keys[:prefix]

        query = torch.randn(
            2 * num_verify, num_heads, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        index_query = torch.randn(
            2 * num_verify, 1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        verify_index_key = torch.cat([keys[-num_verify:] for keys in index_keys], dim=0)
        verify_positions = [
            torch.arange(total - num_verify, total, device="cuda")
            for total in total_lens
        ]
        slot_mapping = torch.cat(
            [slots_for(request, pos) for request, pos in enumerate(verify_positions)]
        )
        seq_lens = torch.tensor(total_lens, dtype=torch.int32, device="cuda")

        batched_selected = minimax_indexer(
            index_query,
            verify_index_key,
            index_key_cache,
            slot_mapping,
            block_table,
            seq_lens,
            topk=_TOPK,
            scale=_HEAD_DIM**-0.5,
            init_blocks=0,
            local_blocks=1,
            decode_query_len=num_verify,
            max_blocks=max_blocks,
        )
        batched_output = minimax_sparse_attention(
            query,
            key_cache,
            value_cache,
            batched_selected,
            block_table,
            seq_lens,
            scale=_HEAD_DIM**-0.5,
            decode_query_len=num_verify,
        )

        for offset in range(num_verify):
            token_ids = torch.tensor([offset, num_verify + offset], device="cuda")
            token_seq_lens = (seq_lens - num_verify + 1 + offset).contiguous()
            single_selected = minimax_indexer(
                index_query[token_ids].contiguous(),
                verify_index_key[token_ids].contiguous(),
                index_key_cache,
                slot_mapping[token_ids].contiguous(),
                block_table,
                token_seq_lens,
                topk=_TOPK,
                scale=_HEAD_DIM**-0.5,
                init_blocks=0,
                local_blocks=1,
                decode_query_len=1,
                max_blocks=max_blocks,
            )
            single_output = minimax_sparse_attention(
                query[token_ids].contiguous(),
                key_cache,
                value_cache,
                single_selected,
                block_table,
                token_seq_lens,
                scale=_HEAD_DIM**-0.5,
                decode_query_len=1,
            )
            for request in range(2):
                token = request * num_verify + offset
                query_position = total_lens[request] - num_verify + offset
                # All verify positions see >= topk blocks, so the full
                # selected list is consumed and set comparison is exact.
                assert query_position + 1 >= _TOPK * _BLOCK_SIZE
                batched_blocks = set(batched_selected[token, 0].cpu().tolist())
                single_blocks = set(single_selected[request, 0].cpu().tolist())
                expected_blocks = set(
                    _reference_selected_blocks(
                        index_query[token, 0],
                        index_keys[request],
                        query_position,
                    )
                    .cpu()
                    .tolist()
                )
                assert batched_blocks == single_blocks == expected_blocks
                reference = _reference_sparse_attention(
                    query[token : token + 1],
                    ref_key_cache,
                    ref_value_cache,
                    batched_selected[token, 0],
                    block_table[request],
                    query_position,
                )
                torch.testing.assert_close(
                    batched_output[token].float(),
                    single_output[request].float(),
                    atol=2e-3,
                    rtol=2e-2,
                )
                torch.testing.assert_close(
                    batched_output[token : token + 1].float(),
                    reference,
                    atol=2e-3,
                    rtol=2e-2,
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@requires_cuda
@pytest.mark.parametrize("phase", ["decode", "extend"])
def test_msa_fp8_kv_descale_matches_dequant_reference(phase: str) -> None:
    """FP8 K/V + descales must match the BF16 kernel on the dequantized cache.

    The cache is quantized with known non-unit scales (K divided by 0.25, V by
    0.5 before conversion, mirroring the write-side convention), so the run
    without descales is a negative control: it must NOT match, proving the
    scales are actually applied. Runs through the public dispatch ops, which
    also proves the FP8 signature is selectable.
    """
    torch.manual_seed(20260722)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        k_scale, v_scale = 0.25, 0.5
        prefix_len = 2305
        new_tokens = 1 if phase == "decode" else 8
        total_len = prefix_len + new_tokens
        num_blocks = math.ceil(total_len / _BLOCK_SIZE)
        num_pages = num_blocks + 1  # Physical page zero is the dummy page.
        block_table = torch.arange(1, num_pages, dtype=torch.int32, device="cuda")[None]

        key_ref = torch.randn(
            num_pages, 1, _BLOCK_SIZE, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        value_ref = torch.randn_like(key_ref)
        key_fp8 = (key_ref.float() / k_scale).to(torch.float8_e4m3fn)
        value_fp8 = (value_ref.float() / v_scale).to(torch.float8_e4m3fn)
        key_dequant = (key_fp8.float() * k_scale).to(torch.bfloat16)
        value_dequant = (value_fp8.float() * v_scale).to(torch.bfloat16)

        index_key_cache = torch.zeros(
            num_pages * _BLOCK_SIZE, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        prefix_positions = torch.arange(prefix_len, device="cuda")
        prefix_slots = (
            prefix_positions // _BLOCK_SIZE + 1
        ) * _BLOCK_SIZE + prefix_positions % _BLOCK_SIZE
        index_key_cache[prefix_slots] = torch.randn(
            prefix_len, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )

        new_positions = torch.arange(prefix_len, total_len, device="cuda")
        slot_mapping = (
            (new_positions // _BLOCK_SIZE + 1) * _BLOCK_SIZE
            + new_positions % _BLOCK_SIZE
        ).to(torch.int32)
        seq_lens = torch.tensor([total_len], dtype=torch.int32, device="cuda")
        query = torch.randn(
            new_tokens, 16, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        index_query = torch.randn(
            new_tokens, 1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        index_key = torch.randn(
            new_tokens, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )

        common = dict(
            index_q=index_query,
            index_k=index_key,
            index_k_cache=index_key_cache,
            slot_mapping=slot_mapping,
            page_table=block_table,
            cache_seqlens=seq_lens,
            topk=_TOPK,
            page_size=_BLOCK_SIZE,
            index_scale=_HEAD_DIM**-0.5,
            attention_scale=_HEAD_DIM**-0.5,
            init_blocks=0,
            local_blocks=1,
        )
        if phase == "decode":
            op = msa_decode_with_kvcache
            common.update(max_seqlen_q=1, max_seqlen_k=total_len)
        else:
            op = msa_extend_with_kvcache
            common.update(
                cu_seqlens_q=torch.tensor(
                    [0, new_tokens], dtype=torch.int32, device="cuda"
                ),
                prefix_lens=torch.tensor(
                    [prefix_len], dtype=torch.int32, device="cuda"
                ),
                max_seqlen_q=new_tokens,
                max_seqlen_k=total_len,
            )

        got = op(
            q=query,
            k_cache=key_fp8,
            v_cache=value_fp8,
            k_scale=k_scale,
            v_scale=v_scale,
            **common,
        )
        reference = op(q=query, k_cache=key_dequant, v_cache=value_dequant, **common)
        unscaled = op(q=query, k_cache=key_fp8, v_cache=value_fp8, **common)

        torch.testing.assert_close(got, reference, atol=2e-2, rtol=2e-2)
        assert not torch.allclose(unscaled, reference, atol=2e-2, rtol=2e-2)

        with pytest.raises(ValueError, match="only valid with an FP8 KV cache"):
            minimax_sparse_attention(
                query,
                key_dequant,
                value_dequant,
                torch.zeros((new_tokens, 1, _TOPK), dtype=torch.int32, device="cuda"),
                block_table,
                seq_lens,
                scale=_HEAD_DIM**-0.5,
                decode_query_len=new_tokens,
                k_scale=k_scale,
                v_scale=v_scale,
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
