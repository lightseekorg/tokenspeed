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

from dataclasses import replace

import pytest
import tokenspeed_kernel.ops.attention.qsa as qsa_api
import tokenspeed_kernel.ops.attention.qsa.triton as qsa_ops
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.numerics.torch.attention.qsa import (
    qsa_logits_topk_ids,
    qsa_stream_topk_ids,
    torch_qsa_paged_block_scores,
    torch_qwen4_exp_qsa_pool_compress,
    torch_qwen4_exp_qsa_recent_rows,
    torch_qwen4_exp_qsa_rope,
)
from tokenspeed_kernel.ops.attention.qsa import qwen4_exp_qsa_block_topk
from tokenspeed_kernel.ops.attention.qsa.triton import (
    _qwen4_exp_qsa_stream_block_topk_kernel,
    qwen4_exp_qsa_commit_verify_layers,
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_prepare_metadata,
    qwen4_exp_qsa_recent_write,
    qwen4_exp_qsa_selected_slots,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Qwen4-Exp QSA kernels require CUDA or ROCm",
)


@pytest.mark.parametrize("query_lengths", [4, [3, 4, 5]])
def test_qwen4_exp_qsa_prepare_metadata_matches_torch(
    device: str, query_lengths: int | list[int]
) -> None:
    seq_lens = torch.tensor([3, 14, 23], device=device, dtype=torch.int32)
    if not isinstance(query_lengths, int):
        query_lengths = torch.tensor(query_lengths, device=device, dtype=torch.int64)
    total_tokens = (
        query_lengths * seq_lens.shape[0]
        if isinstance(query_lengths, int)
        else int(query_lengths.sum())
    )
    ratio = 4
    qsa_page_size = 8
    recent_page_size = 4
    qsa_table = torch.tensor(
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]], device=device, dtype=torch.int32
    )
    recent_table = torch.arange(2, 2 + 3 * 6, device=device, dtype=torch.int32).reshape(
        3, 6
    )
    draft_tags = torch.arange(3 * ratio, device=device, dtype=torch.int64).reshape(
        3, ratio
    )

    actual = qwen4_exp_qsa_prepare_metadata(
        seq_lens,
        query_lengths,
        total_tokens,
        qsa_table,
        qsa_page_size,
        recent_table,
        recent_page_size,
        ratio,
        enable_pdl=False,
        draft_logical_positions=draft_tags,
    )
    lengths = (
        torch.full_like(seq_lens, query_lengths, dtype=torch.int64)
        if isinstance(query_lengths, int)
        else query_lengths
    )
    request_ids = torch.arange(seq_lens.shape[0], device=device)
    expected_requests = torch.repeat_interleave(
        request_ids, lengths, output_size=total_tokens
    )
    cumulative = torch.cumsum(lengths, dim=0)
    row_starts = torch.repeat_interleave(
        cumulative - lengths, lengths, output_size=total_tokens
    )
    row_offsets = torch.arange(total_tokens, device=device) - row_starts
    expected_positions = (seq_lens.long() - lengths)[expected_requests] + row_offsets
    safe_positions = expected_positions.clamp_min(0)
    qsa_pages = qsa_table[expected_requests, safe_positions // qsa_page_size].long()
    expected_qsa = qsa_pages * qsa_page_size + safe_positions % qsa_page_size
    expected_qsa = torch.where(
        (expected_positions >= 0) & (qsa_pages > 0), expected_qsa, 0
    ).to(torch.int32)
    recent_pages = recent_table[
        expected_requests, safe_positions // recent_page_size
    ].long()
    expected_recent = (
        recent_pages * recent_page_size + safe_positions % recent_page_size
    )
    expected_recent = torch.where(
        (expected_positions >= 0) & (recent_pages > 0), expected_recent, 0
    ).to(torch.int32)
    expected_blocks = ((expected_positions + 1) // ratio).to(torch.int32)

    references = (
        expected_positions,
        expected_requests,
        expected_qsa,
        expected_recent,
        expected_blocks,
    )
    for value, reference in zip(actual, references, strict=True):
        torch.testing.assert_close(value, reference)
    torch.testing.assert_close(
        draft_tags,
        torch.full_like(draft_tags, torch.iinfo(torch.int64).min),
    )


def _ref_compress_pool(
    token_k,
    logical,
    requests,
    recent_locs,
    raw,
    position_values,
    position_cache,
    recent_page_size,
    ratio,
):
    return torch_qwen4_exp_qsa_pool_compress(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        recent_page_size=recent_page_size,
        ratio=ratio,
    )


def _ref_rope(tensor, positions, cos_sin_cache, rotary_dim, sections=None):
    return torch_qwen4_exp_qsa_rope(
        tensor, positions, cos_sin_cache, rotary_dim, sections
    )


def test_qwen4_exp_qsa_compress_and_store_matches_torch(device: str) -> None:
    torch.manual_seed(5)
    ratio, head_dim, rotary_dim, recent_page_size = 4, 16, 8, 64
    compressed_token_page_size, rows_per_page = 256, 64
    # Rows 1 and 2 end compression groups; row 1's group spans the batch
    # start (cached fallback plus cached group-start positions), row 2 has
    # no valid compressed slot and must not write.
    logical = torch.tensor([10, 11, 15], device=device)
    requests = torch.tensor([0, 0, 0], device=device)
    token_k = torch.randn(3, 1, head_dim, device=device, dtype=torch.bfloat16)
    raw = torch.randn(4, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_values = torch.randint(0, 48, (3, 3), device=device, dtype=torch.int64)
    position_cache = torch.randint(0, 48, (4, 3), device=device, dtype=torch.int64)
    recent_locs = torch.tensor([130, 131, 134], device=device, dtype=torch.int32)
    qsa_locs = torch.tensor([0, 263, 0], device=device, dtype=torch.int32)
    norm_weight = torch.rand(head_dim, device=device) + 0.5
    epsilon = 1e-6
    cos_sin_cache = torch.randn(64, rotary_dim, device=device)
    compressed = torch.zeros(
        4, rows_per_page, 1, head_dim, device=device, dtype=torch.bfloat16
    )

    qwen4_exp_qsa_compress_and_store(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        norm_weight,
        epsilon,
        cos_sin_cache,
        qsa_locs,
        compressed,
        recent_page_size,
        ratio,
        compressed_token_page_size,
        stage_verify_buffers=None,
        stage_draft=False,
    )

    pooled, first_positions = _ref_compress_pool(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        recent_page_size,
        ratio,
    )
    pooled = pooled.reshape(3, head_dim).float()
    normed = (
        pooled
        * torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        * norm_weight
    )
    rotated = _ref_rope(
        normed.view(3, 1, head_dim),
        first_positions.T,
        cos_sin_cache,
        rotary_dim,
    )
    expected = torch.zeros_like(compressed).view(-1, 1, head_dim)
    boundaries = ((logical + 1) % ratio == 0) & (qsa_locs > 0) & (recent_locs > 0)
    indices = qsa_locs.long().clamp_min(0)
    locs = (indices // compressed_token_page_size) * rows_per_page + (
        indices % compressed_token_page_size
    ) // ratio
    locs = torch.where(boundaries, locs, torch.zeros_like(locs))
    expected.index_copy_(
        0,
        locs,
        torch.where(
            boundaries.view(-1, 1, 1), rotated.to(compressed.dtype), expected[0]
        ),
    )

    torch.testing.assert_close(
        compressed, expected.view_as(compressed), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    ("heads", "head_dim", "rotary_dim"),
    [(3, 16, 8), (6, 256, 64)],
)
def test_qwen4_exp_qsa_fused_query_and_verify_staging_matches_separate(
    device: str,
    heads: int,
    head_dim: int,
    rotary_dim: int,
) -> None:
    torch.manual_seed(7)
    rows, ratio = 6, 4
    section = rotary_dim // 6
    sections = (section, section, rotary_dim // 2 - 2 * section)
    wide = torch.randn(
        rows,
        heads * head_dim + head_dim + 3,
        device=device,
        dtype=torch.bfloat16,
    )
    query = wide[:, : heads * head_dim]
    token_k = wide[:, heads * head_dim : heads * head_dim + head_dim].reshape(
        rows, 1, head_dim
    )
    assert not query.is_contiguous()
    assert not token_k.is_contiguous()
    logical = torch.arange(10, 10 + rows, device=device, dtype=torch.int64)
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    recent_locs = (64 + logical).to(torch.int32)
    qsa_locs = (256 + logical).to(torch.int32)
    positions = torch.randint(0, 32, (3, rows), device=device, dtype=torch.int32)
    position_values = positions.T
    raw = torch.randn(4, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_cache = torch.randint(0, 32, (4, 3), device=device, dtype=torch.int64)
    q_weight = torch.rand(head_dim, device=device) + 0.5
    k_weight = torch.rand(head_dim, device=device) + 0.5
    cos_sin_cache = torch.randn(64, rotary_dim, device=device)
    expected_compressed = torch.zeros(
        4, 64, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    actual_compressed = torch.zeros_like(expected_compressed)

    query_heads = query.float().reshape(rows, heads, head_dim)
    normalized_query = (
        query_heads
        * torch.rsqrt(query_heads.square().mean(dim=-1, keepdim=True) + 1e-6)
        * q_weight
    )
    expected_query = _ref_rope(
        normalized_query,
        positions,
        cos_sin_cache,
        rotary_dim,
        sections,
    ).to(query.dtype)
    qwen4_exp_qsa_compress_and_store(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        k_weight,
        1e-6,
        cos_sin_cache,
        qsa_locs,
        expected_compressed,
        64,
        ratio,
        256,
        sections=sections,
        stage_verify_buffers=None,
        stage_draft=False,
    )
    staged = (
        token_k.new_empty((1, 4, 1, head_dim)),
        position_values.new_empty((1, 4, 3)),
        logical.new_empty((1, 4)),
        recent_locs.new_empty((1, 4)),
    )

    actual_query = qwen4_exp_qsa_compress_and_store(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        k_weight,
        1e-6,
        cos_sin_cache,
        qsa_locs,
        actual_compressed,
        64,
        ratio,
        256,
        sections=sections,
        query=query,
        query_norm_weight=q_weight,
        query_norm_epsilon=1e-6,
        num_query_heads=heads,
        stage_verify_buffers=staged,
        stage_draft=False,
    )

    torch.testing.assert_close(actual_query, expected_query, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        actual_compressed, expected_compressed, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(staged[0], token_k[-4:].reshape_as(staged[0]))
    torch.testing.assert_close(staged[1], position_values[-4:].reshape_as(staged[1]))
    torch.testing.assert_close(staged[2], logical[-4:].reshape_as(staged[2]))
    torch.testing.assert_close(staged[3], recent_locs[-4:].reshape_as(staged[3]))


def test_qwen4_exp_qsa_fused_draft_staging_reads_old_ring_first(
    device: str,
) -> None:
    torch.manual_seed(11)
    rows, ratio, head_dim, rotary_dim = 2, 4, 16, 8
    logical = torch.tensor([3, 7], device=device, dtype=torch.int64)
    requests = torch.tensor([0, 1], device=device, dtype=torch.int64)
    recent_locs = torch.tensor([67, 135], device=device, dtype=torch.int32)
    qsa_locs = torch.tensor([259, 519], device=device, dtype=torch.int32)
    token_k = torch.randn(rows, 1, head_dim, device=device, dtype=torch.bfloat16)
    query = torch.randn(rows, head_dim, device=device, dtype=torch.bfloat16)
    position_values = logical[:, None].expand(-1, 3).clone()
    raw = torch.randn(4, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_cache = torch.zeros(4, 3, device=device, dtype=torch.int64)
    k_weight = torch.rand(head_dim, device=device) + 0.5
    q_weight = torch.rand(head_dim, device=device) + 0.5
    cos_sin_cache = torch.randn(32, rotary_dim, device=device)
    draft_raw = torch.randn(2, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
    draft_positions = torch.tensor(
        [[0, 0, 0], [4, 4, 4]], device=device, dtype=torch.int64
    )
    draft_tags = torch.tensor(
        [[0, 1, 2, -1], [4, 5, 6, -1]], device=device, dtype=torch.int64
    )
    expected_scratch = (
        draft_raw.clone(),
        draft_positions.clone(),
        draft_tags.clone(),
    )
    actual_scratch = tuple(value.clone() for value in expected_scratch)
    expected_compressed = torch.zeros(
        4, 64, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    actual_compressed = torch.zeros_like(expected_compressed)

    qwen4_exp_qsa_compress_and_store(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        k_weight,
        1e-6,
        cos_sin_cache,
        qsa_locs,
        expected_compressed,
        64,
        ratio,
        256,
        draft_raw_cache=expected_scratch[0],
        draft_position_cache=expected_scratch[1],
        draft_logical_positions=expected_scratch[2],
        stage_verify_buffers=None,
        stage_draft=False,
    )
    scratch_slots = torch.remainder(logical, ratio).long()
    request_rows = requests.long()
    expected_scratch[0][request_rows, scratch_slots] = token_k
    expected_scratch[2][request_rows, scratch_slots] = logical
    starts = scratch_slots == 0
    expected_scratch[1][request_rows[starts]] = position_values[starts]
    qwen4_exp_qsa_compress_and_store(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        k_weight,
        1e-6,
        cos_sin_cache,
        qsa_locs,
        actual_compressed,
        64,
        ratio,
        256,
        draft_raw_cache=actual_scratch[0],
        draft_position_cache=actual_scratch[1],
        draft_logical_positions=actual_scratch[2],
        query=query,
        query_norm_weight=q_weight,
        query_norm_epsilon=1e-6,
        num_query_heads=1,
        stage_draft=True,
        stage_verify_buffers=None,
    )

    torch.testing.assert_close(
        actual_compressed, expected_compressed, rtol=2e-2, atol=2e-2
    )
    for actual, expected in zip(actual_scratch, expected_scratch, strict=True):
        torch.testing.assert_close(actual, expected)


def test_qwen4_exp_qsa_ignores_negative_draft_scratch_tags(device: str) -> None:
    ratio, head_dim, rotary_dim, recent_page_size = 4, 16, 8, 64
    logical = torch.tensor([2], device=device)
    requests = torch.zeros(1, device=device, dtype=torch.long)
    token_k = torch.randn(1, 1, head_dim, device=device, dtype=torch.bfloat16)
    recent_locs = torch.tensor([64], device=device, dtype=torch.int32)
    raw = torch.zeros(2, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_values = logical[:, None].expand(-1, 3)
    position_cache = torch.zeros(2, 3, device=device, dtype=torch.int64)
    norm_weight = torch.ones(head_dim, device=device)
    cos_sin_cache = torch.zeros(16, rotary_dim, device=device)
    cos_sin_cache[:, : rotary_dim // 2] = 1
    qsa_locs = torch.tensor([258], device=device, dtype=torch.int32)
    compressed = torch.zeros(2, 64, 1, head_dim, device=device, dtype=torch.bfloat16)
    draft_raw = torch.full(
        (1, ratio, 1, head_dim),
        torch.nan,
        device=device,
        dtype=torch.bfloat16,
    )
    # A legacy -1 empty tag collides with the group head expected at logical
    # position 2. Poisoning its uninitialized RoPE header makes an accidental
    # scratch hit address beyond the cos/sin table.
    draft_logical = torch.full((1, ratio), -1, device=device, dtype=torch.int64)
    draft_positions = torch.full(
        (1, 3), cos_sin_cache.shape[0] + 1024, device=device, dtype=torch.int64
    )

    qwen4_exp_qsa_compress_and_store(
        token_k,
        logical,
        requests,
        recent_locs,
        raw,
        position_values,
        position_cache,
        norm_weight,
        1e-6,
        cos_sin_cache,
        qsa_locs,
        compressed,
        recent_page_size,
        ratio,
        256,
        draft_raw_cache=draft_raw,
        draft_logical_positions=draft_logical,
        draft_position_cache=draft_positions,
        enable_pdl=True,
        stage_verify_buffers=None,
        stage_draft=False,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(compressed, torch.zeros_like(compressed))


def _ref_recent_write(
    token_k,
    logical,
    requests,
    recent_locs,
    position_values,
    raw,
    position_cache,
    recent_page_size,
    ratio,
    write_mask=None,
    request_limit=None,
):
    return torch_qwen4_exp_qsa_recent_rows(
        token_k,
        logical,
        requests,
        recent_locs,
        position_values,
        raw,
        position_cache,
        recent_page_size=recent_page_size,
        ratio=ratio,
        write_mask=write_mask,
        request_limit=request_limit,
    )


def test_qwen4_exp_qsa_recent_write_matches_torch(device: str) -> None:
    torch.manual_seed(11)
    ratio, head_dim, recent_page_size = 4, 8, 64
    # One request whose eight tokens reuse the four-slot ring twice; row 5 is
    # invalid so dedup must keep its shadowed row 1 writer.
    logical = torch.arange(8, device=device)
    requests = torch.zeros(8, device=device, dtype=torch.long)
    token_k = torch.randn(8, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_values = torch.randint(0, 64, (8, 3), device=device, dtype=torch.int64)
    recent_locs = torch.tensor(
        [64, 65, 66, 67, 68, 0, 70, 71], device=device, dtype=torch.int32
    )

    for kwargs in ({}, {"request_limit": 1}, {"request_limit": 0}):
        raw = torch.zeros(3, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
        position_cache = torch.zeros(3, 3, device=device, dtype=torch.int64)
        qwen4_exp_qsa_recent_write(
            token_k,
            logical,
            requests,
            recent_locs,
            position_values,
            raw,
            position_cache,
            recent_page_size,
            ratio,
            **kwargs,
        )
        expected_raw, expected_positions = _ref_recent_write(
            token_k,
            logical,
            requests,
            recent_locs,
            position_values,
            torch.zeros_like(raw),
            torch.zeros_like(position_cache),
            recent_page_size,
            ratio,
            **kwargs,
        )
        torch.testing.assert_close(raw, expected_raw)
        torch.testing.assert_close(position_cache, expected_positions)


def test_qwen4_exp_qsa_recent_write_honors_explicit_mask(device: str) -> None:
    ratio, head_dim, recent_page_size = 4, 8, 64
    logical = torch.tensor([0, 1], device=device)
    requests = torch.tensor([0, 0], device=device)
    token_k = torch.ones(2, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_values = torch.full((2, 3), 7, device=device, dtype=torch.int64)
    recent_locs = torch.tensor([64, 65], device=device, dtype=torch.int32)
    write_mask = torch.tensor([False, True], device=device)
    raw = torch.zeros(2, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_cache = torch.zeros(2, 3, device=device, dtype=torch.int64)

    qwen4_exp_qsa_recent_write(
        token_k,
        logical,
        requests,
        recent_locs,
        position_values,
        raw,
        position_cache,
        recent_page_size,
        ratio,
        write_mask=write_mask,
    )

    assert raw[1, 0].abs().sum() == 0
    torch.testing.assert_close(
        raw[1, 1, 0], torch.ones(head_dim, device=device, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        position_cache[1], torch.zeros(3, device=device, dtype=torch.int64)
    )


def test_qwen4_exp_qsa_block_topk_matches_torch(device: str) -> None:
    torch.manual_seed(29)
    rows, heads, head_dim, page_size = 3, 4, 16, 64
    block_topk = 64
    pages_per_request = 6
    num_blocks = pages_per_request * page_size
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        4 * num_blocks, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randint(
        1,
        4 * pages_per_request,
        (rows, pages_per_request),
        device=device,
        dtype=torch.int32,
    )
    # Request 1 owns fewer blocks than block_topk, exercising the -1 padding.
    requests = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    complete_blocks = torch.tensor([200, 40, 384], device=device, dtype=torch.int32)

    actual = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=1,
        max_partial_bytes=32 * 1024 * 1024,
        path="stream",
        max_logits_bytes=256 << 20,
        solution=None,
        override=None,
        enable_pdl=False,
    )

    block_ids = torch.arange(num_blocks, device=device)
    columns = block_ids // page_size
    offsets = block_ids % page_size
    for row in range(rows):
        pages = page_table[requests[row]][columns].long()
        slots = pages * page_size + offsets
        gathered = key_cache[slots, 0]
        scores = torch.relu(
            torch.einsum("hd,nd->hn", query[row].float(), gathered.float())
        ).sum(dim=0)
        valid = block_ids < complete_blocks[row]
        scores = torch.where(valid, scores, torch.tensor(-float("inf"), device=device))
        expected = set(
            torch.topk(
                scores,
                min(block_topk, int(complete_blocks[row]), num_blocks),
            ).indices.tolist()
        )
        got = [int(value) for value in actual[row] if value >= 0]
        assert len(got) == len(expected)
        assert set(got) == expected


def test_qwen4_exp_qsa_block_topk_long_stream_matches_torch(device: str) -> None:
    # A long stream retains one exact running set across every key tile.
    torch.manual_seed(31)
    rows, heads, head_dim, page_size = 2, 4, 16, 64
    block_topk = 512
    pages_per_request = 256
    num_blocks = pages_per_request * page_size
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        num_blocks, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randint(
        1,
        pages_per_request,
        (rows, pages_per_request),
        device=device,
        dtype=torch.int32,
    )
    requests = torch.tensor([0, 1], device=device, dtype=torch.long)
    complete_blocks = torch.tensor([num_blocks, 300], device=device, dtype=torch.int32)

    actual = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=1,
        max_partial_bytes=32 * 1024 * 1024,
        path="stream",
        max_logits_bytes=256 << 20,
        solution=None,
        override=None,
        enable_pdl=False,
    )

    block_ids = torch.arange(num_blocks, device=device)
    columns = block_ids // page_size
    offsets = block_ids % page_size
    for row in range(rows):
        pages = page_table[requests[row]][columns].long()
        slots = pages * page_size + offsets
        gathered = key_cache[slots, 0]
        scores = torch.relu(
            torch.einsum("hd,nd->hn", query[row].float(), gathered.float())
        ).sum(dim=0)
        valid = block_ids < complete_blocks[row]
        scores = torch.where(valid, scores, torch.tensor(-float("inf"), device=device))
        expected = torch.topk(
            scores,
            min(block_topk, int(complete_blocks[row]), num_blocks),
        )
        got = [int(value) for value in actual[row] if value >= 0]
        assert len(got) == len(expected.indices)
        assert len(set(got)) == len(got)
        assert all(index < int(complete_blocks[row]) for index in got)
        torch.testing.assert_close(
            scores[got].sort(descending=True).values, expected.values, rtol=0, atol=0
        )


def _block_topk_reference_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    requests: torch.Tensor,
    complete_blocks: torch.Tensor,
    page_size: int,
    num_blocks: int,
) -> list[torch.Tensor]:
    """Per-row FP32 block scores with ``-inf`` outside ``complete_blocks``."""
    table = page_table[:, : -(-num_blocks // page_size)]
    scores = torch_qsa_paged_block_scores(
        query, key_cache, table, requests, complete_blocks, page_size=page_size
    )
    return list(scores[:, :num_blocks].unbind(0))


def _expected_logits_ids(scores: torch.Tensor, k: int, n_cols_padded: int) -> list[int]:
    """Reference ids under the DSA selection kernel's tie-breaking (the
    padded width only fixes the kernel's packing)."""
    return qsa_logits_topk_ids(scores, k)


@pytest.mark.parametrize("query_group_size", [1, 4])
@pytest.mark.parametrize("tile_programs", [1, 2, 5])
@pytest.mark.parametrize("group_programs", [1, 4])
def test_qwen4_exp_qsa_scores_leave_invalid_blocks_unwritten(
    device: str, query_group_size: int, tile_programs: int, group_programs: int
) -> None:
    rows, heads, dim, page_size, columns = 4, 3, 32, 64, 9
    num_blocks = page_size * columns
    query = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
    keys = torch.randn(
        (columns + 1) * page_size, 1, dim, device=device, dtype=query.dtype
    )
    keys[:page_size].fill_(float("nan"))
    table = torch.arange(1, columns + 1, device=device, dtype=torch.int32)[None, :]
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    complete = torch.tensor([0, 1, 129, 513], device=device, dtype=torch.int32)
    logits = torch.full((rows, num_blocks), float("nan"), device=device)
    qsa_ops._qwen4_exp_qsa_score_blocks_kernel[
        (min(rows // query_group_size, group_programs), tile_programs)
    ](
        query,
        keys,
        table,
        requests,
        complete,
        logits,
        heads,
        dim,
        num_blocks,
        rows // query_group_size,
        page_size,
        *query.stride(),
        keys.stride(0),
        keys.stride(2),
        table.stride(0),
        logits.stride(0),
        BLOCK_N=128,
        BLOCK_H=4,
        BLOCK_D=32,
        QUERY_GROUP_SIZE=query_group_size,
        LOOP_QUERY_GROUPS=group_programs < rows // query_group_size,
        ENABLE_PDL=False,
        num_warps=4,
        num_stages=2,
    )
    expected = torch.stack(
        _block_topk_reference_scores(
            query, keys, table, requests, complete, page_size, num_blocks
        )
    )
    valid = torch.arange(num_blocks, device=device)[None, :] < complete[:, None]
    torch.testing.assert_close(logits[valid], expected[valid], rtol=1e-5, atol=1e-4)
    assert torch.isnan(logits[~valid]).all()


def test_qwen4_exp_qsa_score_grid_refreshes_without_recompile(device: str) -> None:
    torch.manual_seed(149)
    batch, width, heads, dim, page_size = 3, 4, 4, 128, 64
    rows, max_columns = batch * width, 19
    query = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
    keys = torch.randn(
        (batch * max_columns + 1) * page_size,
        1,
        dim,
        device=device,
        dtype=query.dtype,
    )
    keys[:page_size].fill_(float("nan"))
    requests = torch.arange(batch, device=device).repeat_interleave(width)
    complete = torch.zeros(rows, device=device, dtype=torch.int32)
    kernel = qsa_ops._qwen4_exp_qsa_score_blocks_kernel

    def launch(table, logits, tile_programs):
        kernel[(batch, tile_programs)](
            query,
            keys,
            table,
            requests,
            complete,
            logits,
            heads,
            dim,
            table.shape[1] * page_size,
            batch,
            page_size,
            *query.stride(),
            keys.stride(0),
            keys.stride(2),
            table.stride(0),
            logits.stride(0),
            BLOCK_N=128,
            BLOCK_H=heads,
            BLOCK_D=dim,
            QUERY_GROUP_SIZE=width,
            LOOP_QUERY_GROUPS=False,
            ENABLE_PDL=False,
            num_warps=4,
            num_stages=2,
        )

    full_table = torch.arange(
        1, batch * max_columns + 1, device=device, dtype=torch.int32
    ).reshape(batch, max_columns)
    for columns, tile_programs in ((9, 1), (16, 2), (19, 3)):
        num_blocks = columns * page_size
        # Serving shortens the view while preserving the allocation's row stride.
        table = full_table[:, :columns]
        logits = torch.empty(rows, num_blocks, device=device)
        # Width, row stride and grid changes must reuse one compiled kernel.
        if columns == 9:
            launch(table, logits, tile_programs)
            torch.cuda.synchronize()
        with assert_no_triton_compile(kernel):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(table, logits, tile_programs)
            for frontiers in (
                [num_blocks] * rows,
                [0, 1, 127, 128, 129, 511, 513, num_blocks, 0, 0, 0, 0],
                [num_blocks] * rows,
                [0] * rows,
            ):
                query.normal_()
                keys[page_size:].normal_()
                complete.copy_(
                    torch.tensor(frontiers, device=device, dtype=torch.int32)
                )
                logits.fill_(float("nan"))
                graph.replay()
                expected = torch.stack(
                    _block_topk_reference_scores(
                        query, keys, table, requests, complete, page_size, num_blocks
                    )
                )
                valid = (
                    torch.arange(num_blocks, device=device)[None, :] < complete[:, None]
                )
                torch.testing.assert_close(
                    logits[valid], expected[valid], rtol=1e-5, atol=1e-4
                )
                assert torch.isnan(logits[~valid]).all()


@pytest.mark.parametrize("queries_per_request", [None, 1, 2, 3, 4, 6, 8])
@pytest.mark.parametrize(("heads", "head_dim"), [(3, 32), (4, 128), (6, 96)])
def test_qwen4_exp_qsa_grouped_scores_match_independent_queries(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    queries_per_request: int | None,
    heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(113)
    widths = [1, 3, 5] if queries_per_request is None else [queries_per_request] * 3
    rows, page_size, columns, block_topk = sum(widths), 64, 9, 64
    num_blocks = page_size * columns
    # Keep the projection's strided Q view and a non-contiguous page table.
    query = torch.randn(rows, heads + 1, head_dim, device=device, dtype=torch.bfloat16)[
        :, :heads
    ]
    key_cache = torch.randn(
        (3 * columns + 1) * page_size,
        1,
        head_dim,
        device=device,
        dtype=query.dtype,
    )
    key_cache[:page_size].fill_(float("nan"))
    table = torch.empty((3, columns + 2), device=device, dtype=torch.int32)
    table[:, :columns] = (
        torch.randperm(3 * columns, device=device).reshape(3, columns) + 1
    )
    table = table[:, :columns]
    # Owning request IDs need not match the order of the uniform query runs.
    requests = torch.tensor(
        [request for request, width in zip((2, 0, 1), widths) for _ in range(width)],
        device=device,
        dtype=torch.int64,
    )
    frontiers = [0, 1, 63, 64, 127, 128, 129, 511, 513, num_blocks]
    complete = torch.tensor(
        [frontiers[row % len(frontiers)] for row in range(rows)],
        device=device,
        dtype=torch.int32,
    )
    # A padded request must select no blocks without reading page 0.
    table[1].zero_()
    complete[-widths[-1] :].zero_()
    scores = []
    select = qsa_api.qsa_block_topk_select

    def record_scores(logits, topk, **kwargs):
        assert kwargs["lengths"] is complete
        valid = torch.arange(num_blocks, device=device)[None, :] < complete[:, None]
        scores.append(logits.masked_fill(~valid, -float("inf")))
        return select(logits, topk, **kwargs)

    monkeypatch.setattr(qsa_api, "qsa_block_topk_select", record_scores)
    selected = []
    for width in (queries_per_request, None):
        selected.append(
            qwen4_exp_qsa_block_topk(
                query,
                key_cache,
                table,
                requests,
                complete,
                page_size=page_size,
                block_topk=block_topk,
                queries_per_request=width,
                max_partial_bytes=32 * 1024 * 1024,
                path="logits",
                max_logits_bytes=256 << 20,
                solution=None,
                override=None,
                enable_pdl=False,
            )
        )
    expected = torch.stack(
        _block_topk_reference_scores(
            query, key_cache, table, requests, complete, page_size, num_blocks
        )
    )
    # Grouping changes the dot/reduction layout, so FP32 scores may round
    # differently even when both layouts agree with the reference.
    torch.testing.assert_close(scores[0], scores[1], rtol=1e-5, atol=1e-4)
    n_cols_padded = 1 << (max(num_blocks, block_topk) - 1).bit_length()
    for logits, indices in zip(scores, selected):
        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-4)
        assert torch.isneginf(logits[-widths[-1] :]).all()
        assert (indices[-widths[-1] :] == -1).all()
        # Near-ties can change membership across layouts. Each selection
        # must still match its own scores, including tie-breaking and padding.
        for row in range(rows):
            expected_ids = _expected_logits_ids(logits[row], block_topk, n_cols_padded)
            expected_ids += [-1] * (block_topk - len(expected_ids))
            assert sorted(indices[row].tolist()) == sorted(expected_ids)


def test_qwen4_exp_qsa_grouped_scores_refresh_during_graph_replay(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(127)
    # Force a bounded production grid so replay must walk multiple tiles per CTA.
    platform = replace(qsa_ops.current_platform(), sm_count=1)
    monkeypatch.setattr(qsa_ops, "current_platform", lambda: platform)
    batch, width, heads, dim, page_size, columns = 3, 4, 4, 128, 64, 19
    rows = batch * width
    query = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
    keys = torch.randn(
        (batch * columns + 1) * page_size,
        1,
        dim,
        device=device,
        dtype=query.dtype,
    )
    table = torch.arange(
        1, batch * columns + 1, device=device, dtype=torch.int32
    ).reshape(batch, columns)
    requests = torch.arange(batch, device=device).repeat_interleave(width)
    complete = torch.full((rows,), 1216, device=device, dtype=torch.int32)

    def forward(queries_per_request):
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=page_size,
            block_topk=64,
            queries_per_request=queries_per_request,
            max_partial_bytes=32 * 1024 * 1024,
            path="logits",
            max_logits_bytes=256 << 20,
            solution=None,
            override=None,
            enable_pdl=False,
        )

    forward(width)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward(width)
    for frontiers in (
        [0, 1, 127, 128, 129, 511, 513, 1216, 0, 0, 0, 0],
        [1216, 513, 511, 129, 128, 127, 1, 0, 0, 0, 0, 0],
        [0] * rows,
    ):
        query.normal_()
        keys.normal_()
        complete.copy_(torch.tensor(frontiers, device=device, dtype=torch.int32))
        requests.copy_(torch.tensor([2, 0, 1], device=device).repeat_interleave(width))
        # Check replay refresh against the same layout; cross-layout score
        # agreement is covered separately without requiring identical top-k.
        expected = forward(width)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [1024, 1056])
def test_qwen4_exp_qsa_long_query_groups_match_independent_rows(
    device: str, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    torch.manual_seed(137)
    batch, heads, dim, page_size, columns = 2, 4, 128, 64, 9
    rows = batch * width
    query = torch.randn(rows, heads + 1, dim, device=device, dtype=torch.bfloat16)[
        :, :heads
    ]
    keys = torch.randn(
        (batch * columns + 1) * page_size,
        1,
        dim,
        device=device,
        dtype=query.dtype,
    )
    keys[:page_size].fill_(float("nan"))
    table = torch.arange(
        1, batch * columns + 1, device=device, dtype=torch.int32
    ).reshape(batch, columns)
    requests = torch.tensor([1, 0], device=device).repeat_interleave(width)
    complete = torch.arange(rows, device=device, dtype=torch.int32) % 577
    complete[width : width + 64] = 0
    scores = []
    selector = qsa_api.qsa_block_topk_select

    def record(logits, topk, **kwargs):
        assert kwargs["lengths"] is complete
        valid = (
            torch.arange(columns * page_size, device=device)[None, :]
            < complete[:, None]
        )
        scores.append(logits.masked_fill(~valid, -float("inf")))
        return selector(logits, topk, **kwargs)

    monkeypatch.setattr(qsa_api, "qsa_block_topk_select", record)
    selected = []
    for query_width in (width, None):
        selected.append(
            qwen4_exp_qsa_block_topk(
                query,
                keys,
                table,
                requests,
                complete,
                page_size=page_size,
                block_topk=64,
                queries_per_request=query_width,
                max_partial_bytes=32 * 1024 * 1024,
                path="logits",
                max_logits_bytes=256 << 20,
                solution=None,
                override=None,
                enable_pdl=False,
            )
        )
    expected = torch.stack(
        _block_topk_reference_scores(
            query, keys, table, requests, complete, page_size, columns * page_size
        )
    )
    torch.testing.assert_close(scores[0], scores[1], rtol=1e-5, atol=1e-4)
    for logits, indices in zip(scores, selected):
        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-4)
        for row in range(rows):
            expected_ids = _expected_logits_ids(logits[row], 64, 1024)
            expected_ids += [-1] * (64 - len(expected_ids))
            assert sorted(indices[row].tolist()) == sorted(expected_ids)
    assert torch.isneginf(scores[0][width : width + 64]).all()
    assert (selected[0][width : width + 64] == -1).all()


@pytest.mark.parametrize("queries_per_request", [0, -1, 5])
def test_qwen4_exp_qsa_rejects_invalid_query_width(
    device: str, queries_per_request: int
) -> None:
    query = torch.empty((12, 4, 16), device=device, dtype=torch.bfloat16)
    keys = torch.empty((64, 1, 16), device=device, dtype=query.dtype)
    table = torch.zeros((3, 1), device=device, dtype=torch.int32)
    requests = torch.zeros(12, device=device, dtype=torch.int64)
    complete = torch.zeros(12, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="queries_per_request"):
        qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=64,
            queries_per_request=queries_per_request,
            max_partial_bytes=32 * 1024 * 1024,
            path="logits",
            max_logits_bytes=256 << 20,
            solution=None,
            override=None,
            enable_pdl=False,
        )


def test_qwen4_exp_qsa_block_topk_logits_matches_stream(device: str) -> None:
    torch.manual_seed(41)
    rows, heads, head_dim, page_size = 3, 4, 16, 64
    block_topk = 2048
    pages_per_request = 192
    num_blocks = pages_per_request * page_size
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        num_blocks, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randint(
        1,
        pages_per_request,
        (rows, pages_per_request),
        device=device,
        dtype=torch.int32,
    )
    requests = torch.arange(rows, device=device, dtype=torch.long)
    complete_blocks = torch.tensor(
        [num_blocks, 9000, 300], device=device, dtype=torch.int32
    )

    kwargs = dict(
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=1,
        max_partial_bytes=32 * 1024 * 1024,
        enable_pdl=False,
    )
    stream = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        path="stream",
        max_logits_bytes=256 << 20,
        solution=None,
        override=None,
        **kwargs,
    )
    logits = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        path="logits",
        max_logits_bytes=256 << 20,
        solution=None,
        override=None,
        **kwargs,
    )

    # The two solutions break fp32 score ties in opposite directions, so
    # each output is checked against its own packed-key reference.
    scores_per_row = _block_topk_reference_scores(
        query, key_cache, page_table, requests, complete_blocks, page_size, num_blocks
    )
    n_cols_padded = 1 << (max(num_blocks, block_topk) - 1).bit_length()
    for row in range(rows):
        expected_len = min(block_topk, int(complete_blocks[row]), num_blocks)
        scores = scores_per_row[row]
        got_stream = sorted(int(v) for v in stream[row] if v >= 0)
        got_logits = sorted(int(v) for v in logits[row] if v >= 0)
        assert len(got_stream) == expected_len
        assert len(got_logits) == expected_len
        assert got_stream == sorted(qsa_stream_topk_ids(scores, expected_len))
        assert got_logits == sorted(
            _expected_logits_ids(scores, expected_len, n_cols_padded)
        )


@pytest.mark.parametrize(("cols", "topk"), [(4099, 512), (65537, 512), (65537, 2048)])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_qwen4_exp_qsa_registered_selection_reads_strided_scores_and_lengths(
    device: str, cols: int, topk: int, length_dtype: torch.dtype
) -> None:
    rows = 5
    scores = torch.empty((rows, cols * 2), device=device)[:, ::2]
    lengths = torch.empty(rows * 2, device=device, dtype=length_dtype)[::2]
    lengths.copy_(torch.tensor([-1, 1, topk - 1, cols - 1, cols + 7], device=device))
    columns = torch.arange(cols, device=device)
    valid = columns[None, :] < lengths.clamp(0, cols)[:, None]
    # Repeated scores check that threshold ties retain the smaller column id.
    values = columns.remainder(257).float().expand(rows, -1)
    scores.copy_(torch.where(valid, values, float("nan")))
    selected = qsa_api.qsa_block_topk_select(
        scores,
        topk,
        lengths=lengths,
        enable_pdl=False,
        override=None,
        solution="triton",
    )
    reference = scores.masked_fill(~valid, -float("inf"))
    expected = torch.argsort(reference, dim=1, descending=True, stable=True)[:, :topk]
    expected = torch.where(torch.isfinite(reference.gather(1, expected)), expected, -1)
    torch.testing.assert_close(
        selected.sort(dim=1).values.long(), expected.sort(dim=1).values
    )


@pytest.mark.parametrize("topk", [64, 512])
def test_qwen4_exp_qsa_registered_selection_ignores_uninitialized_tails(
    device: str, topk: int
) -> None:
    rows, heads, dim, page_size, columns = 4, 4, 32, 64, 9
    query = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
    keys = torch.randn(
        (columns + 1) * page_size, 1, dim, device=device, dtype=query.dtype
    )
    table = torch.arange(1, columns + 1, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    complete = torch.tensor([0, 1, 129, 513], device=device, dtype=torch.int32)
    scores = qsa_api.qsa_block_scores(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=page_size,
        queries_per_request=rows,
        enable_pdl=False,
        override=None,
        solution="triton",
    )
    valid = torch.arange(scores.shape[1], device=device)[None, :] < complete[:, None]
    scores.masked_fill_(~valid, float("nan"))
    selected = qsa_api.qsa_block_topk_select(
        scores,
        topk,
        lengths=complete,
        enable_pdl=False,
        override=None,
        solution="triton",
    )
    reference = scores.masked_fill(~valid, -float("inf"))
    for row, count in enumerate(complete.tolist()):
        expected = torch.argsort(reference[row], descending=True, stable=True)[
            : min(topk, count)
        ]
        actual = selected[row][selected[row] >= 0]
        torch.testing.assert_close(actual.sort().values.long(), expected.sort().values)
        assert (selected[row] < 0).sum().item() == max(0, topk - count)


@pytest.mark.parametrize("columns", [1, 2, 8])
def test_qwen4_exp_qsa_compact_logits_shorter_than_topk(
    device: str, columns: int
) -> None:
    torch.manual_seed(139)
    rows, heads, dim, page_size, topk = 4, 4, 128, 64, 512
    query = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
    keys = torch.randn(
        (columns + 1) * page_size, 1, dim, device=device, dtype=query.dtype
    )
    table = torch.arange(1, columns + 1, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    complete = torch.tensor(
        [0, 1, 31, columns * page_size], device=device, dtype=torch.int32
    )
    selected = qwen4_exp_qsa_block_topk(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=page_size,
        block_topk=topk,
        queries_per_request=rows,
        max_partial_bytes=32 * 1024 * 1024,
        path="logits",
        max_logits_bytes=256 << 20,
        solution=None,
        override=None,
        enable_pdl=False,
    )
    for row, count in enumerate(complete.tolist()):
        ids = selected[row][selected[row] >= 0].sort().values
        torch.testing.assert_close(
            ids, torch.arange(count, device=device, dtype=ids.dtype)
        )
        assert int((selected[row] == -1).sum()) == topk - count


def test_qwen4_exp_qsa_block_topk_logits_long_rows_match_stream(
    device: str,
) -> None:
    # 70400 blocks exercise the Triton radix selection path. The second
    # row is shorter than top-k and verifies ragged -1 padding.
    torch.manual_seed(43)
    rows, heads, head_dim, page_size = 2, 4, 16, 64
    block_topk = 512
    pages_per_request = 1100
    num_blocks = pages_per_request * page_size
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        num_blocks, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randint(
        1,
        pages_per_request,
        (rows, pages_per_request),
        device=device,
        dtype=torch.int32,
    )
    requests = torch.arange(rows, device=device, dtype=torch.long)
    complete_blocks = torch.tensor([num_blocks, 300], device=device, dtype=torch.int32)

    kwargs = dict(
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=1,
        max_partial_bytes=32 * 1024 * 1024,
        enable_pdl=False,
    )
    stream = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        path="stream",
        max_logits_bytes=256 << 20,
        solution=None,
        override=None,
        **kwargs,
    )
    logits = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        path="logits",
        max_logits_bytes=256 << 20,
        solution=None,
        override=None,
        **kwargs,
    )

    scores_per_row = _block_topk_reference_scores(
        query, key_cache, page_table, requests, complete_blocks, page_size, num_blocks
    )
    for row in range(rows):
        expected_len = min(block_topk, int(complete_blocks[row]), num_blocks)
        scores = scores_per_row[row]
        got_stream = sorted(int(v) for v in stream[row] if v >= 0)
        got_logits = sorted(int(v) for v in logits[row] if v >= 0)
        assert len(got_stream) == expected_len
        assert len(got_logits) == expected_len
        # The streaming and Triton selectors may break equal-score ties
        # differently, so compare the selected score multisets.
        got_scores = sorted((float(scores[int(i)]) for i in got_stream), reverse=True)
        ref_scores = sorted(
            (float(v) for v in torch.topk(scores, expected_len).values.tolist()),
            reverse=True,
        )
        assert got_scores == ref_scores
        got_logits_scores = sorted(
            (float(scores[int(i)]) for i in got_logits), reverse=True
        )
        assert got_logits_scores == ref_scores


def test_qwen4_exp_qsa_selected_slots_matches_torch(device: str) -> None:
    ratio, block_topk = 4, 8
    token_topk = block_topk * ratio
    selected_blocks = torch.tensor(
        [[0, 5, 2, 9, 1, 3, 4, 6], [2, 0, 0, 0, 0, 0, 0, 0]],
        device=device,
        dtype=torch.int64,
    )
    selected_blocks[0, 5:] = -1
    complete_blocks = torch.tensor([6, 3], device=device, dtype=torch.int32)
    logical = torch.tensor([25, 10], device=device, dtype=torch.long)
    requests = torch.tensor([0, 1], device=device, dtype=torch.long)
    page_size = 8
    page_table = torch.tensor(
        [[2, 4, 0, 7], [9, 1, 6, 3]], device=device, dtype=torch.int32
    )

    actual = qwen4_exp_qsa_selected_slots(
        selected_blocks,
        complete_blocks,
        logical,
        requests,
        page_table,
        page_size,
        ratio,
        token_topk,
        enable_pdl=False,
    )

    blocks = torch.where(
        selected_blocks < complete_blocks.unsqueeze(1), selected_blocks, -1
    )
    offsets = torch.arange(ratio, device=device)
    tokens = (blocks.unsqueeze(-1) * ratio + offsets).reshape(2, token_topk)
    tokens = torch.where(blocks.repeat_interleave(ratio, dim=1) >= 0, tokens, -1)
    suffix_offsets = torch.arange(ratio - 1, device=device)
    suffix = complete_blocks.long().unsqueeze(1) * ratio + suffix_offsets
    suffix = torch.where(suffix <= logical.unsqueeze(1), suffix, -1)
    selected = torch.cat((tokens, suffix), dim=1)
    valid = (selected >= 0) & (selected <= logical.unsqueeze(1))
    safe = selected.clamp_min(0)
    columns = safe // page_size
    pages = page_table[requests.unsqueeze(1).expand_as(columns), columns].long()
    expected = pages * page_size + safe % page_size
    expected = torch.where(valid & (pages > 0), expected, -1).to(torch.int32)

    torch.testing.assert_close(actual, expected)


def test_qwen4_exp_qsa_block_topk_reads_strided_inputs(device: str) -> None:
    torch.manual_seed(53)
    rows, heads, head_dim, page_size = 3, 4, 16, 64
    block_topk = 64
    pages_per_request = 6
    num_blocks = pages_per_request * page_size
    wide = torch.randn(rows, 2 * heads, head_dim, device=device, dtype=torch.bfloat16)
    query = wide[:, :heads]
    key_cache = torch.randn(
        4 * num_blocks, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.randint(
        1, 4 * pages_per_request, (rows, pages_per_request + 2), device=device
    )[:, :pages_per_request]
    requests = torch.arange(rows, device=device)
    complete_blocks = torch.tensor([num_blocks, 40, 200], device=device)

    for algorithm in ("stream", "logits"):
        strided = qwen4_exp_qsa_block_topk(
            query,
            key_cache,
            page_table,
            requests,
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            queries_per_request=1,
            max_partial_bytes=32 * 1024 * 1024,
            path=algorithm,
            max_logits_bytes=256 << 20,
            solution=None,
            override=None,
            enable_pdl=False,
        )
        packed = qwen4_exp_qsa_block_topk(
            query.contiguous(),
            key_cache,
            page_table.contiguous(),
            requests.to(torch.int32),
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            queries_per_request=1,
            max_partial_bytes=32 * 1024 * 1024,
            path=algorithm,
            max_logits_bytes=256 << 20,
            solution=None,
            override=None,
            enable_pdl=False,
        )
        torch.testing.assert_close(strided, packed)


def test_qwen4_exp_qsa_compress_and_store_reads_strided_token_k(device: str) -> None:
    torch.manual_seed(59)
    ratio, head_dim, rotary_dim, recent_page_size = 4, 16, 8, 64
    compressed_token_page_size, rows_per_page = 256, 64
    rows = 6
    logical = torch.arange(rows, device=device) * ratio + (ratio - 1)
    requests = torch.zeros(rows, device=device, dtype=torch.long)
    # Keys sliced out of a wider QK projection view: row stride 3 * head_dim.
    qk = torch.randn(rows, 3 * head_dim, device=device, dtype=torch.bfloat16)
    token_k = qk[:, -head_dim:].reshape(rows, 1, head_dim)
    assert not token_k.is_contiguous()
    raw = torch.randn(4, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
    position_values = torch.randint(0, 48, (rows, 3), device=device, dtype=torch.int64)
    position_cache = torch.randint(0, 48, (4, 3), device=device, dtype=torch.int64)
    recent_locs = torch.arange(128, 128 + rows, device=device, dtype=torch.int32)
    qsa_locs = (torch.arange(rows, device=device, dtype=torch.int32) + 1) * ratio
    norm_weight = torch.rand(head_dim, device=device) + 0.5
    cos_sin_cache = torch.randn(64, rotary_dim, device=device)

    def run(keys: torch.Tensor, pdl: bool) -> torch.Tensor:
        compressed = torch.zeros(
            4, rows_per_page, 1, head_dim, device=device, dtype=torch.bfloat16
        )
        qwen4_exp_qsa_compress_and_store(
            keys,
            logical,
            requests,
            recent_locs,
            raw,
            position_values,
            position_cache,
            norm_weight,
            1e-6,
            cos_sin_cache,
            qsa_locs,
            compressed,
            recent_page_size,
            ratio,
            compressed_token_page_size,
            enable_pdl=pdl,
            stage_verify_buffers=None,
            stage_draft=False,
        )
        return compressed

    expected = run(token_k.contiguous(), False)
    assert expected.view(-1, head_dim).abs().sum() > 0
    torch.testing.assert_close(run(token_k, False), expected)
    # The PDL-chained launch must drain to the same compressed cache.
    torch.testing.assert_close(run(token_k, True), expected)


def test_qwen4_exp_qsa_recent_write_reads_strided_token_k(device: str) -> None:
    torch.manual_seed(61)
    ratio, head_dim, recent_page_size = 4, 8, 64
    rows = 8
    logical = torch.arange(rows, device=device)
    requests = torch.zeros(rows, device=device, dtype=torch.long)
    qk = torch.randn(rows, 2 * head_dim, device=device, dtype=torch.bfloat16)
    token_k = qk[:, -head_dim:].reshape(rows, 1, head_dim)
    assert not token_k.is_contiguous()
    position_values = torch.randint(0, 64, (rows, 3), device=device, dtype=torch.int32)
    recent_locs = torch.arange(64, 64 + rows, device=device, dtype=torch.int32)

    def run(keys: torch.Tensor, pdl: bool):
        raw = torch.zeros(3, ratio, 1, head_dim, device=device, dtype=torch.bfloat16)
        position_cache = torch.zeros(3, 3, device=device, dtype=torch.int64)
        qwen4_exp_qsa_recent_write(
            keys,
            logical,
            requests,
            recent_locs,
            position_values,
            raw,
            position_cache,
            recent_page_size,
            ratio,
            enable_pdl=pdl,
        )
        return raw, position_cache

    expected_raw, expected_positions = run(token_k.contiguous(), False)
    assert expected_raw.abs().sum() > 0
    raw, positions = run(token_k, False)
    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(positions, expected_positions)
    raw, positions = run(token_k, True)
    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(positions, expected_positions)


@pytest.mark.parametrize("ratio, width", [(1, 4), (4, 3), (4, 4), (4, 5), (4, 9)])
@pytest.mark.parametrize("null_pages", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_qwen4_exp_qsa_commit_verify_layers_matches_torch(
    device: str, ratio: int, width: int, null_pages: bool, dtype: torch.dtype
) -> None:
    head_dim, recent_page_size, num_layers = 8, 64, 3
    counts = [-1, 0, 1, width - 1, width, width + 1]
    bs = len(counts)
    rows = bs * width
    logical = torch.arange(40, 40 + rows, device=device, dtype=torch.int64)
    requests = torch.arange(bs, device=device).repeat_interleave(width)
    recent_locs = ((requests + 1) * recent_page_size + logical % recent_page_size).int()
    if null_pages:
        recent_locs.zero_()
    positions = torch.randint(1, 64, (rows, 3), device=device, dtype=torch.int64)
    # A spare request per layer exercises the layer stride above the live batch.
    staged = torch.randn(
        num_layers, bs + 1, width, 1, head_dim, device=device, dtype=dtype
    )
    raws = [
        torch.full(
            (bs + 1, ratio, 1, head_dim), -3.0, device=device, dtype=torch.bfloat16
        )
        for _ in range(num_layers)
    ]
    caches = [
        torch.full((bs + 1, 3), -3, device=device, dtype=torch.int64)
        for _ in range(num_layers)
    ]
    expected_raws = [raw.cpu() for raw in raws]
    expected_positions = [cache.cpu() for cache in caches]
    keys_cpu, positions_cpu = staged.cpu(), positions.cpu()
    if not null_pages:
        # Sequential accepted writes provide an independent ring-buffer reference.
        for layer in range(num_layers):
            for request, count in enumerate(counts):
                for step in range(max(0, min(count, width))):
                    row = request * width + step
                    slot = (40 + row) % ratio
                    expected_raws[layer][request + 1, slot] = keys_cpu[
                        layer, request, step
                    ]
                    if slot == 0:
                        expected_positions[layer][request + 1] = positions_cpu[row]

    qwen4_exp_qsa_commit_verify_layers(
        torch.tensor(
            [raw.data_ptr() for raw in raws], device=device, dtype=torch.uint64
        ),
        torch.tensor(
            [cache.data_ptr() for cache in caches], device=device, dtype=torch.uint64
        ),
        staged,
        logical,
        recent_locs,
        positions,
        torch.tensor(counts, device=device, dtype=torch.int64),
        raws[0],
        caches[0],
        recent_page_size,
        ratio,
        verify_width=width,
    )
    for layer in range(num_layers):
        torch.testing.assert_close(
            raws[layer].cpu(), expected_raws[layer], atol=0, rtol=0
        )
        torch.testing.assert_close(
            caches[layer].cpu(), expected_positions[layer], atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "argument, replace, error",
    [
        ("position_addresses", lambda t: t[:1], "one address per layer"),
        ("raw_addresses", lambda t: t.long(), "torch.uint64"),
        ("staged_k", lambda t: t[:1], "one layer block per address"),
        ("verify_width", lambda width: width - 1, "positive multiple of verify_width"),
        ("accepted_lengths", lambda t: t[:1], "one accepted length per request"),
        ("staged_k", lambda t: t[..., ::2], "must be contiguous"),
        ("raw_cache", lambda t: t.float(), "must be bfloat16"),
        ("position_cache", lambda t: t.int(), "must be int64"),
        ("staged_k", lambda t: t[:, :1].contiguous(), "bucket covers fewer rows"),
    ],
)
def test_qwen4_exp_qsa_commit_verify_layers_rejects_bad_args(
    device: str, argument: str, replace, error: str
) -> None:
    ratio, head_dim, recent_page_size = 4, 8, 64
    num_layers, bs, width = 2, 2, 4
    rows = bs * width
    args = dict(
        raw_addresses=torch.zeros(num_layers, device=device, dtype=torch.uint64),
        position_addresses=torch.zeros(num_layers, device=device, dtype=torch.uint64),
        staged_k=torch.zeros(
            num_layers, bs, width, 1, head_dim, device=device, dtype=torch.bfloat16
        ),
        logical_positions=torch.arange(rows, device=device, dtype=torch.int64),
        recent_locs=torch.arange(1, rows + 1, device=device, dtype=torch.int32),
        position_values=torch.zeros(rows, 3, device=device, dtype=torch.int64),
        accepted_lengths=torch.tensor([2, 3], device=device, dtype=torch.int64),
        raw_cache=torch.zeros(
            bs + 1, ratio, 1, head_dim, device=device, dtype=torch.bfloat16
        ),
        position_cache=torch.zeros(bs + 1, 3, device=device, dtype=torch.int64),
        recent_page_size=recent_page_size,
        compress_ratio=ratio,
        verify_width=width,
    )
    args[argument] = replace(args[argument])
    with pytest.raises(ValueError, match=error):
        qwen4_exp_qsa_commit_verify_layers(**args)


@pytest.mark.parametrize("budget", [0, 1 << 20])
@pytest.mark.parametrize("topk", [64, 512, 2048])
@pytest.mark.parametrize("queries_per_request", [None, 4])
def test_portable_stream_allocates_only_output_and_refreshes_graph(
    device, monkeypatch, budget, topk, queries_per_request
):
    torch.manual_seed(311)
    rows, blocks = 4, 4096
    query = torch.randn(rows, 5, 32, device=device, dtype=torch.bfloat16)[:, :4]
    keys = torch.randn(blocks, 1, 32, device=device, dtype=query.dtype)
    table = torch.arange(blocks // 64, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    complete = torch.tensor([0, 50, 513, blocks], device=device, dtype=torch.int32)

    def invoke():
        return qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=topk,
            queries_per_request=queries_per_request,
            max_partial_bytes=budget,
            max_logits_bytes=256 << 20,
            path="stream",
            enable_pdl=False,
            solution="triton",
            override=None,
        )

    invoke()
    allocations = []
    empty = torch.empty

    def record_empty(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        allocations.append(tensor.numel() * tensor.element_size())
        return tensor

    monkeypatch.setattr(torch, "empty", record_empty)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = invoke()
    assert allocations == [actual.numel() * actual.element_size()]
    for frontiers in ([0, 50, 513, blocks], [blocks, 513, 50, 0]):
        complete.copy_(torch.tensor(frontiers, device=device, dtype=torch.int32))
        query.normal_()
        keys.normal_()
        graph.replay()
        scores = _block_topk_reference_scores(
            query, keys, table, requests, complete, 64, blocks
        )
        for row, count in enumerate(frontiers):
            expected = qsa_stream_topk_ids(scores[row], min(count, topk))
            assert sorted(actual[row][actual[row] >= 0].tolist()) == sorted(expected)


@pytest.mark.parametrize("queries_per_request", [None, 4])
def test_portable_stream_reuses_compile_across_rows_capacities_and_strides(
    device, queries_per_request
):
    torch.manual_seed(317)
    query_storage = torch.randn(12, 5, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(1024, 1, 128, device=device, dtype=query_storage.dtype)
    table_storage = torch.arange(16, device=device, dtype=torch.int32).repeat(12, 1)
    requests = torch.zeros(12, device=device, dtype=torch.int32)
    complete = torch.full_like(requests, 512)

    def invoke(rows, pages, feature_stride, width):
        return qwen4_exp_qsa_block_topk(
            (
                query_storage[:rows, :4, :64]
                if feature_stride == 1
                else query_storage[:rows, :4, :128:2]
            ),
            keys[:, :, :64] if feature_stride == 1 else keys[:, :, ::2],
            table_storage[:rows, :pages],
            requests[:rows],
            complete[:rows],
            page_size=64,
            block_topk=64,
            queries_per_request=width,
            max_partial_bytes=0,
            max_logits_bytes=256 << 20,
            path="stream",
            enable_pdl=False,
            solution="triton",
            override=None,
        )

    # Triton distinguishes unit and non-unit runtime strides; warm each class.
    first_rows = 3 if queries_per_request is None else 4
    invoke(first_rows, 8, 1, queries_per_request)
    invoke(first_rows, 8, 2, queries_per_request)
    with assert_no_triton_compile(_qwen4_exp_qsa_stream_block_topk_kernel):
        shapes = (
            ((7, 11), (9, 13), (12, 16))
            if queries_per_request is None
            else ((8, 11), (12, 13), (12, 16))
        )
        for rows, pages in shapes:
            for feature_stride in (1, 2):
                width = None if queries_per_request is None else rows
                selected = invoke(rows, pages, feature_stride, width)
                assert bool(((selected >= 0) & (selected < 512)).all())


@pytest.mark.parametrize("queries_per_request", [None, 3, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_portable_stream_request_groups_and_frontiers(
    device, queries_per_request, dtype
):
    torch.manual_seed(331)
    width = queries_per_request or 8
    rows, blocks, topk = 3 * width, 4096, 512
    query = torch.randn(rows, 5, 128, device=device, dtype=dtype)[:, :4]
    keys = torch.randn(3 * blocks, 1, 128, device=device, dtype=dtype)
    # Persistent row strides and permuted physical pages exercise address
    # calculation independently of each group's request id.
    table_storage = torch.full((3, 81), -1, device=device, dtype=torch.int32)
    table = table_storage[:, : blocks // 64]
    table.copy_(torch.randperm(3 * blocks // 64, device=device).reshape_as(table))
    requests = torch.tensor([2, 0, 1], device=device, dtype=torch.int32)
    requests = requests.repeat_interleave(width)
    if queries_per_request is None:
        requests = requests[torch.randperm(rows, device=device)]
    complete = torch.tensor(
        [0, 64, 512, 575, 2049, blocks], device=device, dtype=torch.int32
    ).repeat((rows + 5) // 6)[:rows]
    actual = qwen4_exp_qsa_block_topk(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=64,
        block_topk=topk,
        queries_per_request=queries_per_request,
        max_partial_bytes=0,
        max_logits_bytes=0,
        path="stream",
        enable_pdl=False,
        solution="triton",
        override=None,
    )
    scores = _block_topk_reference_scores(
        query, keys, table, requests, complete, 64, blocks
    )
    for row in range(rows):
        ids = actual[row]
        count = min(int(complete[row]), topk)
        assert int((ids >= 0).sum()) == count
        assert torch.unique(ids[ids >= 0]).numel() == count
        selected_scores = scores[row][ids.clamp_min(0).long()].masked_fill(
            ids < 0, -float("inf")
        )
        torch.testing.assert_close(
            selected_scores.sort(descending=True).values,
            scores[row].topk(topk).values,
            rtol=1e-5,
            atol=1e-4,
        )


def test_portable_stream_preserves_ties_and_ids_above_65535(device):
    rows, blocks, topk = 8, 65664, 512
    query = torch.zeros(rows, 4, 32, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(blocks, 1, 32, device=device, dtype=query.dtype)
    table = torch.arange(blocks // 64, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    frontiers = [0, 512, 513, 65535, 65536, 65537, blocks - 1, blocks]
    complete = torch.tensor(frontiers, device=device, dtype=torch.int32)
    actual = qwen4_exp_qsa_block_topk(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=64,
        block_topk=topk,
        queries_per_request=8,
        max_partial_bytes=0,
        max_logits_bytes=0,
        path="stream",
        enable_pdl=False,
        solution="triton",
        override=None,
    )
    for row, frontier in enumerate(frontiers):
        count = min(frontier, topk)
        assert int((actual[row] >= 0).sum()) == count
        expected = torch.arange(frontier - count, frontier, device=device)
        torch.testing.assert_close(
            actual[row][actual[row] >= 0].long().sort().values, expected
        )


def test_portable_stream_keeps_fp32_boundary_precision(device):
    rows, blocks, topk = 4, 4096, 512
    query = torch.zeros(rows, 5, 32, device=device)[:, :4]
    query[:, 0, 0] = 1
    keys = torch.zeros(blocks, 1, 32, device=device)
    # These distinct scores share FP16/TF32 bins around the selection boundary.
    keys[:, 0, 0] = 1 + torch.arange(blocks, device=device) * 2**-22
    table = torch.arange(blocks // 64, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    frontiers = [600, 1024, 3073, blocks]
    complete = torch.tensor(frontiers, device=device, dtype=torch.int32)
    actual = qwen4_exp_qsa_block_topk(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=64,
        block_topk=topk,
        queries_per_request=4,
        max_partial_bytes=0,
        max_logits_bytes=0,
        path="stream",
        enable_pdl=False,
        solution="triton",
        override=None,
    )
    for row, frontier in enumerate(frontiers):
        expected = torch.arange(frontier - topk, frontier, device=device)
        torch.testing.assert_close(actual[row].long().sort().values, expected)
