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


def _msa_cute_registered() -> bool:
    import tokenspeed_kernel.ops.attention.msa as msa_mod

    return hasattr(msa_mod, "msa_minimax_extend_with_kvcache")


requires_msa_cute = pytest.mark.skipif(
    not torch.cuda.is_available() or not _msa_cute_registered(),
    reason="the msa CuTe attend requires SM100 with cutlass-dsl and quack",
)


def _two_request_extend_case(kv_cache_dtype: torch.dtype):
    """Two ragged chunked-prefill requests sharing one paged cache."""
    prefix_lens = [2305, 517]
    new_tokens = [7, 129]
    total_lens = [p + n for p, n in zip(prefix_lens, new_tokens)]
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
        key_cache = key_cache.to(kv_cache_dtype)
        value_cache = value_cache.to(kv_cache_dtype)

    index_key_cache = torch.zeros(
        num_pages * _BLOCK_SIZE, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    for request, prefix in enumerate(prefix_lens):
        prefix_positions = torch.arange(prefix, device="cuda")
        index_key_cache[slots_for(request, prefix_positions).long()] = torch.randn(
            prefix, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )

    total_q = sum(new_tokens)
    query = torch.randn(total_q, 16, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    index_query = torch.randn(
        total_q, 1, _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    index_key = torch.randn(total_q, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.cat(
        [
            slots_for(request, torch.arange(prefix, total, device="cuda"))
            for request, (prefix, total) in enumerate(zip(prefix_lens, total_lens))
        ]
    )
    kwargs = dict(
        q=query,
        index_q=index_query,
        index_k=index_key,
        k_cache=key_cache,
        v_cache=value_cache,
        index_k_cache=index_key_cache,
        slot_mapping=slot_mapping,
        page_table=block_table,
        cache_seqlens=torch.tensor(total_lens, dtype=torch.int32, device="cuda"),
        cu_seqlens_q=torch.tensor(
            [0, new_tokens[0], total_q], dtype=torch.int32, device="cuda"
        ),
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
        max_seqlen_q=max(new_tokens),
        max_seqlen_k=max(total_lens),
        topk=_TOPK,
        page_size=_BLOCK_SIZE,
        index_scale=_HEAD_DIM**-0.5,
        attention_scale=_HEAD_DIM**-0.5,
        init_blocks=0,
        local_blocks=1,
    )
    return kwargs


@requires_msa_cute
@pytest.mark.parametrize(
    "kv_cache_dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8_e4m3"],
)
def test_msa_cute_extend_matches_triton(kv_cache_dtype: torch.dtype) -> None:
    """The CuTe attend must match the Triton attend on ragged extend batches.

    Both solutions run through the public dispatch op with a solution pin, on
    the same caches and the same (deterministic) indexer selection; only the
    attend kernel differs. The FP8 case exercises the CuTe kernel's BF16-Q +
    E4M3-KV staging mode at identity scale.
    """
    torch.manual_seed(20260722)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        kwargs = _two_request_extend_case(kv_cache_dtype)
        # Each solution's indexer pass rewrites the same index_k_cache slots
        # with identical values, so back-to-back calls stay comparable.
        out_cute = msa_extend_with_kvcache(solution="msa", **kwargs)
        out_triton = msa_extend_with_kvcache(solution="triton", **kwargs)
        assert out_cute.dtype == torch.bfloat16
        torch.testing.assert_close(out_cute, out_triton, atol=2e-2, rtol=2e-2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@requires_msa_cute
def test_msa_cute_extend_wins_selection_and_decode_stays_triton() -> None:
    """On SM100 the SPECIALIZED CuTe solution must win extend dispatch, decode
    must keep resolving to Triton (no fmha_sm100 decode registration), and the
    descale fallback must keep non-identity FP8 scales on the Triton attend."""
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    traits = {
        "head_dim": _HEAD_DIM,
        "index_head_dim": _HEAD_DIM,
        "page_size": _BLOCK_SIZE,
        "topk": _TOPK,
    }
    for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        signature = format_signature(
            q=dense_tensor_format(torch.bfloat16),
            index_q=dense_tensor_format(torch.bfloat16),
            index_k=dense_tensor_format(torch.bfloat16),
            k_cache=dense_tensor_format(kv_dtype),
            v_cache=dense_tensor_format(kv_dtype),
            index_k_cache=dense_tensor_format(torch.bfloat16),
        )
        extend = select_kernel(
            "attention", "msa_extend_with_kvcache", signature, traits=traits
        )
        assert extend.name == "msa_minimax_extend_with_kvcache"
        decode = select_kernel(
            "attention", "msa_decode_with_kvcache", signature, traits=traits
        )
        assert decode.name == "triton_minimax_msa_decode_with_kvcache"

    # Descaled FP8 extend goes through the dispatcher-selected CuTe wrapper
    # but must produce the Triton-descaled result (internal fallback).
    torch.manual_seed(20260722)
    kwargs = _two_request_extend_case(torch.float8_e4m3fn)
    out_default = msa_extend_with_kvcache(k_scale=0.25, v_scale=0.5, **kwargs)
    out_triton = msa_extend_with_kvcache(
        solution="triton", k_scale=0.25, v_scale=0.5, **kwargs
    )
    torch.testing.assert_close(out_default, out_triton, atol=0.0, rtol=0.0)


def _cutedsl_decode_score_available() -> bool:
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_blackwell:
        return False
    try:
        from tokenspeed_kernel.ops.attention.cute_dsl import (  # noqa: F401
            minimax_index_decode_score,
        )
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _cutedsl_decode_score_available(),
    reason="CuteDSL index decode score requires SM100 and cutlass-dsl",
)
@pytest.mark.parametrize(
    "decode_query_len,seq_list",
    [
        (1, [300, 1000, 129]),
        (4, [300, 1000, 132]),
        (1, [70000]),
    ],
)
def test_cutedsl_decode_score_matches_triton(
    decode_query_len: int, seq_list: list[int]
) -> None:
    import tokenspeed_kernel.ops.attention.triton.minimax_indexer as mi
    import triton
    from tokenspeed_kernel.ops.attention.cute_dsl.minimax_index_decode_score import (
        decode_score_supported,
        minimax_index_decode_score,
    )

    torch.manual_seed(0)
    device = "cuda"
    heads, head_dim, block_k = 4, 128, 128
    scale = head_dim**-0.5
    init_blocks, local_blocks = 1, 2

    requests = len(seq_list)
    tokens = requests * decode_query_len
    seq_lens = torch.tensor(seq_list, device=device, dtype=torch.int32)
    max_blocks = ((max(seq_list) + block_k - 1) // block_k + 3) // 4 * 4
    pages = requests * max_blocks
    cache_pages = torch.randn(
        pages, block_k, head_dim, device=device, dtype=torch.bfloat16
    )
    block_table = torch.arange(pages, device=device, dtype=torch.int32).view(
        requests, max_blocks
    )
    index_q = torch.randn(tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    assert decode_score_supported(index_q, cache_pages, decode_query_len, max_blocks)

    scores_cute = torch.full(
        (tokens, heads, max_blocks),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    minimax_index_decode_score(
        index_q,
        cache_pages,
        scores_cute,
        block_table,
        seq_lens,
        scale=scale,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
    )

    scores_triton = torch.full_like(scores_cute, -float("inf"))
    num_chunks = 64
    mi._decode_block_score_kernel[(requests, num_chunks)](
        index_q,
        cache_pages,
        scores_triton,
        block_table,
        seq_lens,
        num_index_heads=heads,
        scale=scale,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
        max_blocks=max_blocks,
        num_chunks=num_chunks,
        stride_q_n=index_q.stride(0),
        stride_q_h=index_q.stride(1),
        stride_q_d=index_q.stride(2),
        stride_k_page=cache_pages.stride(0),
        stride_k_pos=cache_pages.stride(1),
        stride_k_d=cache_pages.stride(2),
        stride_s_n=scores_triton.stride(0),
        stride_s_h=scores_triton.stride(1),
        stride_s_b=scores_triton.stride(2),
        stride_bt_b=block_table.stride(0),
        head_dim=head_dim,
        BLOCK_Q=triton.next_power_of_2(decode_query_len),
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    # Forced (+inf) and unwritten (-inf) positions must agree exactly; finite
    # scores are the same fp32 MMA math on both paths.
    assert torch.equal(torch.isinf(scores_cute), torch.isinf(scores_triton))
    assert torch.equal(scores_cute == float("inf"), scores_triton == float("inf"))
    finite = torch.isfinite(scores_triton)
    assert finite.any()
    torch.testing.assert_close(
        scores_cute[finite], scores_triton[finite], atol=2e-3, rtol=2e-3
    )


@pytest.mark.skipif(
    not _cutedsl_decode_score_available(),
    reason="CuteDSL index decode score requires SM100 and cutlass-dsl",
)
def test_cutedsl_decode_score_gates() -> None:
    from tokenspeed_kernel.ops.attention.cute_dsl.minimax_index_decode_score import (
        decode_score_supported,
    )

    device = "cuda"
    q = torch.randn(4, 4, 128, device=device, dtype=torch.bfloat16)
    pages = torch.randn(8, 128, 128, device=device, dtype=torch.bfloat16)
    assert decode_score_supported(q, pages, 1, 8)
    # Flattened Q tile too wide (4 heads x 16 queries > 32).
    assert not decode_score_supported(q, pages, 16, 8)
    # Non-4-aligned score tile dim.
    assert not decode_score_supported(q, pages, 1, 7)
    # Dtype mismatch between queries and cache.
    assert not decode_score_supported(q, pages.to(torch.float8_e4m3fn), 1, 8)


def _fmha_prefill_score_available() -> bool:
    from tokenspeed_kernel.platform import current_platform

    if not torch.cuda.is_available() or not current_platform().is_blackwell:
        return False
    try:
        import tokenspeed_kernel.thirdparty.msa.jit  # noqa: F401
    except ImportError:
        return False
    return True


requires_fmha_prefill_score = pytest.mark.skipif(
    not _fmha_prefill_score_available(),
    reason="fmha OnlyScore prefill scorer requires SM100 and the vendored "
    "msa JIT stack",
)


def _prefill_indexer_case(
    qo_lens: list[int],
    kv_lens: list[int],
    query_lens_cpu: list[int] | None,
    seq_lens_cpu: list[int] | None,
) -> torch.Tensor:
    device = "cuda"
    heads, head_dim, block_k = 4, 128, 128
    requests = len(qo_lens)
    tokens = sum(qo_lens)
    max_blocks = (max(kv_lens) + block_k - 1) // block_k
    pages_per_req = [(k + block_k - 1) // block_k for k in kv_lens]
    total_pages = sum(pages_per_req)
    generator = torch.Generator(device=device).manual_seed(7)
    perm = torch.randperm(total_pages, device=device, generator=generator).to(
        torch.int32
    )
    block_table = torch.zeros(requests, max_blocks, device=device, dtype=torch.int32)
    offset = 0
    for request, num_pages in enumerate(pages_per_req):
        block_table[request, :num_pages] = perm[offset : offset + num_pages]
        offset += num_pages
    index_k_cache = torch.randn(
        total_pages * block_k,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    index_q = torch.randn(
        tokens,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    index_k = torch.randn(
        tokens, head_dim, device=device, dtype=torch.bfloat16, generator=generator
    )
    slots = []
    for request, (query_len, kv_len) in enumerate(zip(qo_lens, kv_lens)):
        for position in range(kv_len - query_len, kv_len):
            page = int(block_table[request, position // block_k].item())
            slots.append(page * block_k + position % block_k)
    slot_mapping = torch.tensor(slots, device=device, dtype=torch.int32)
    seq_lens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(requests + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.cumsum(torch.tensor(qo_lens, device=device), 0)
    prefix_lens = torch.tensor(
        [k - q for q, k in zip(qo_lens, kv_lens)], device=device, dtype=torch.int32
    )
    return minimax_indexer(
        index_q,
        index_k,
        index_k_cache,
        slot_mapping,
        block_table,
        seq_lens,
        topk=16,
        scale=head_dim**-0.5,
        init_blocks=1,
        local_blocks=2,
        cu_seqlens_q=cu_seqlens_q,
        prefix_lens=prefix_lens,
        max_query_len=max(qo_lens),
        max_blocks=max_blocks,
        query_lens_cpu=query_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
    )


@requires_fmha_prefill_score
@pytest.mark.parametrize(
    "qo_lens,kv_lens",
    [
        ([600], [20000]),
        ([1024, 512], [50000, 30000]),
        ([40], [70000]),
    ],
)
def test_fmha_prefill_score_matches_triton(
    qo_lens: list[int], kv_lens: list[int]
) -> None:
    from tokenspeed_kernel.ops.attention import msa_score

    if not msa_score.ensure_prefill_score_ready(None):
        pytest.skip("fmha OnlyScore JIT compilation failed (nvcc unavailable?)")

    torch.manual_seed(0)
    plan_cache_before = msa_score._plan_for_batch.cache_info().currsize
    fmha_selected = _prefill_indexer_case(qo_lens, kv_lens, qo_lens, kv_lens)
    assert (
        msa_score._plan_for_batch.cache_info().currsize > plan_cache_before
        or msa_score._plan_for_batch.cache_info().hits > 0
    ), "fmha OnlyScore path was not taken"
    triton_selected = _prefill_indexer_case(qo_lens, kv_lens, None, None)

    # Selections are compared as sets: fmha/sparse_topk_select emits ascending
    # block ids with -1 padding, the Triton top-k emits score order.
    assert torch.equal(
        fmha_selected.sort(dim=-1).values, triton_selected.sort(dim=-1).values
    )


@requires_fmha_prefill_score
def test_fmha_prefill_score_gates() -> None:
    from tokenspeed_kernel.ops.attention.msa_score import prefill_score_supported

    device = "cuda"
    index_q = torch.randn(8, 4, 128, device=device, dtype=torch.bfloat16)
    pages = torch.randn(256, 128, 128, device=device, dtype=torch.bfloat16)
    # Host lens are required for sync-free planning.
    assert not prefill_score_supported(index_q, pages, 16, 256, None, None)
    # Upstream top-k only supports topk == 16.
    assert not prefill_score_supported(index_q, pages, 32, 256, [8], [32000])
    # Below the measured crossover the Triton pair is faster.
    assert not prefill_score_supported(index_q, pages, 16, 64, [8], [8192])
    # FP8 index cache is stage 3.
    assert not prefill_score_supported(
        index_q, pages.to(torch.float8_e4m3fn), 16, 256, [8], [32000]
    )
