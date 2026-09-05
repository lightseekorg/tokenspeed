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
import tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa as qsa_ops
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.cuda.dsa_topk import has_ragged_decode_topk
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    _qwen4_exp_qsa_merge_block_topk_kernel,
    _qwen4_exp_qsa_stream_block_topk_kernel,
    qwen4_exp_qsa_block_topk,
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_prepare_metadata,
    qwen4_exp_qsa_recent_write,
    qwen4_exp_qsa_selected_slots,
)

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
    qsa_expansion = 2
    recent_page_size = 4
    qsa_logical = torch.tensor(
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]], device=device, dtype=torch.int32
    )
    qsa_table = qsa_logical.repeat_interleave(qsa_expansion, dim=1)
    qsa_table = qsa_table * qsa_expansion + (
        torch.arange(qsa_table.shape[1], device=device) % qsa_expansion
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
        qsa_expansion,
        qsa_page_size,
        recent_table,
        1,
        recent_page_size,
        ratio,
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
    qsa_pages = (
        qsa_table[
            expected_requests,
            (safe_positions // qsa_page_size) * qsa_expansion,
        ].long()
        // qsa_expansion
    )
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
    rows = logical.shape[0]
    raw_pages = recent_locs.long().clamp_min(0) // recent_page_size
    row_ids = torch.arange(rows, device=logical.device)
    offsets = torch.arange(ratio - 1, -1, -1, device=logical.device)
    source_rows = row_ids.unsqueeze(1) - offsets
    safe_source = source_rows.clamp(min=0, max=max(rows - 1, 0))
    expected = logical.unsqueeze(1) - offsets
    from_current = source_rows >= 0
    if rows:
        from_current &= requests[safe_source] == requests.unsqueeze(1)
        from_current &= logical[safe_source] == expected
    raw_slots = torch.remainder(expected, ratio).long()
    cached = raw[raw_pages.unsqueeze(1), raw_slots]
    current = token_k[safe_source].to(raw.dtype)
    group = torch.where(from_current.view(rows, ratio, 1, 1), current, cached)
    pooled = group.float().mean(dim=1).to(raw.dtype)
    cached_first = position_cache[raw_pages]
    current_first = position_values[safe_source[:, 0]]
    first = torch.where(from_current[:, :1], current_first, cached_first)
    return pooled, first


def _ref_rope(tensor, positions, cos_sin_cache, rotary_dim, sections=None):
    if positions.ndim == 2 and sections is None:
        positions = positions[0]
    cos, sin = cos_sin_cache[positions.long()].chunk(2, dim=-1)
    if positions.ndim == 2:
        cos = torch.cat(
            [part[axis] for axis, part in enumerate(cos.split(sections, -1))], dim=-1
        )
        sin = torch.cat(
            [part[axis] for axis, part in enumerate(sin.split(sections, -1))], dim=-1
        )
    first, second = tensor[..., :rotary_dim].chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(tensor.dtype)
    sin = sin.unsqueeze(-2).to(tensor.dtype)
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


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
    raw = raw.clone()
    position_cache = position_cache.clone()
    rows = logical.shape[0]
    if write_mask is None:
        mask = recent_locs > 0
    else:
        mask = write_mask.clone()
    if rows:
        row_ids = torch.arange(rows, device=logical.device)
        future = row_ids + ratio
        has_future = future < rows
        safe_future = future.clamp(max=rows - 1)
        has_future &= mask[safe_future]
        has_future &= requests[safe_future] == requests
        has_future &= logical[safe_future] == logical + ratio
        mask = mask & ~has_future
    if request_limit is not None:
        mask &= requests < request_limit
    pages = recent_locs.long().clamp_min(0) // recent_page_size
    slots = torch.remainder(logical.long(), ratio)
    for row in range(rows):
        if not bool(mask[row]):
            continue
        raw[pages[row], slots[row]] = token_k[row].to(raw.dtype)
        if slots[row] == 0:
            position_cache[pages[row]] = position_values[row]
    return raw, position_cache


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


@pytest.mark.parametrize("splits", [1, 3, 13, 16])
def test_qwen4_exp_qsa_merge_tree_matches_flat_topk(device: str, splits: int) -> None:
    """Merge bitonic partial rows exactly, including padded split rows."""

    torch.manual_seed(37)
    rows, block_topk = 3, 64
    shape = (2, rows, splits, block_topk)
    score_bits = torch.randint(1, 1 << 20, shape, dtype=torch.int64)
    block_ids = torch.arange(1, 1 + 2 * rows * splits * block_topk).reshape(shape)
    packed = ((score_bits + 1) << 32) | block_ids
    ascending = packed[0].sort(dim=-1).values
    descending = packed[1].sort(dim=-1, descending=True).values
    partial = torch.maximum(ascending, descending).to(device)
    valid_splits = torch.tensor(
        [splits, max(splits - 1, 0), 0], device=device, dtype=torch.int32
    )
    blocks_per_split = block_topk
    complete_blocks = valid_splits * blocks_per_split

    actual = torch.empty((rows, block_topk), dtype=torch.int32, device=device)
    _qwen4_exp_qsa_merge_block_topk_kernel[(rows,)](
        partial,
        actual,
        complete_blocks,
        blocks_per_split,
        partial.stride(0),
        partial.stride(1),
        actual.stride(0),
        BLOCK_TOPK=block_topk,
        SPLITS=splits,
        POW2_SPLITS=triton.next_power_of_2(splits),
        ENABLE_PDL=False,
        num_warps=8,
    )
    expected = torch.full_like(actual, -1)
    for row, count in enumerate(valid_splits.tolist()):
        if count:
            expected_keys = torch.topk(
                partial[row, :count].flatten(), block_topk, sorted=True
            ).values
            expected[row] = (expected_keys & 0xFFFFFFFF).to(torch.int32)
    torch.testing.assert_close(actual, expected)


def test_qwen4_exp_qsa_stream_skips_empty_split_writes(device: str) -> None:
    """A split beyond a row's complete range leaves its partials untouched."""

    torch.manual_seed(39)
    rows, heads, head_dim, page_size = 2, 4, 16, 64
    block_topk, splits, blocks_per_split = 64, 4, 96
    num_blocks = splits * blocks_per_split
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        num_blocks, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.arange(
        num_blocks // page_size, device=device, dtype=torch.int32
    ).repeat(rows, 1)
    requests = torch.arange(rows, device=device, dtype=torch.int64)
    complete_blocks = torch.tensor([num_blocks, 40], device=device, dtype=torch.int32)
    sentinel = 0x123456789ABCDEF
    partial = torch.full(
        (rows, splits, block_topk), sentinel, device=device, dtype=torch.int64
    )

    _qwen4_exp_qsa_stream_block_topk_kernel[(rows, splits)](
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        partial,
        heads,
        head_dim,
        num_blocks,
        page_size,
        1,
        blocks_per_split,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        partial.stride(0),
        partial.stride(1),
        BLOCK_TOPK=block_topk,
        BLOCK_N=block_topk,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_D=triton.next_power_of_2(head_dim),
        ENABLE_PDL=False,
        num_warps=8,
        num_stages=2,
    )

    assert torch.all(partial[1, 1:] == sentinel)
    assert torch.all(partial[1, 0] != sentinel)


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

    # The same selection must come out of a consumer-granularity page table
    # whose entries are expanded 2x.
    expanded_pt = page_table.repeat_interleave(2, dim=1) * 2
    actual_expanded = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        expanded_pt,
        requests,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        page_expansion=2,
    )
    torch.testing.assert_close(actual_expanded, actual)


def test_qwen4_exp_qsa_block_topk_two_stage_merge_matches_torch(device: str) -> None:
    # block_topk=512 with 16k blocks forces 32 streaming splits, so the
    # merge runs in two stages (16 splits per chunk program).
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

    block_ids = torch.arange(num_blocks, device=query.device)
    columns = block_ids // page_size
    offsets = block_ids % page_size
    scores_per_row = []
    for row in range(query.shape[0]):
        pages = page_table[requests[row]][columns].long()
        slots = pages * page_size + offsets
        gathered = key_cache[slots, 0]
        scores = torch.relu(
            torch.einsum("hd,nd->hn", query[row].float(), gathered.float())
        ).sum(dim=0)
        valid = block_ids < complete_blocks[row]
        scores = torch.where(
            valid, scores, torch.tensor(float("-inf"), device=query.device)
        )
        scores_per_row.append(scores)
    return scores_per_row


def _expected_stream_ids(scores: torch.Tensor, k: int) -> list[int]:
    """Reference ids under the stream path's tie-breaking.

    Packed keys are ``(score_bits + 1) << 32 | block_id`` with zero reserved
    for invalid rows, so equal fp32 bits prefer the larger block id.
    """

    ids = torch.arange(scores.shape[0], device=scores.device)
    bits = scores.contiguous().view(torch.int32).to(torch.int64)
    packed = ((bits + 1) << 32) | ids.to(torch.int64)
    packed = torch.where(torch.isfinite(scores), packed, torch.zeros_like(packed))
    top = torch.topk(packed, k).values
    return [int(v & 0xFFFFFFFF) for v in top.tolist() if v != 0]


def _expected_logits_ids(scores: torch.Tensor, k: int, n_cols_padded: int) -> list[int]:
    """Reference ids under the DSA selection kernel's tie-breaking.

    The selection kernel packs ``ordered_key << 32 | (n_cols_padded -
    offset)``, so equal scores prefer the smaller block id and ``-inf``
    never wins; a stable descending sort is the same ordering.
    """

    order = torch.argsort(scores, descending=True, stable=True)
    return [int(i) for i in order[:k].tolist() if torch.isfinite(scores[int(i)])]


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

    kwargs = dict(page_size=page_size, block_topk=block_topk)
    stream = qwen4_exp_qsa_block_topk(
        query, key_cache, page_table, requests, complete_blocks, **kwargs
    )
    logits = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        solution="logits",
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
        assert got_stream == sorted(_expected_stream_ids(scores, expected_len))
        assert got_logits == sorted(
            _expected_logits_ids(scores, expected_len, n_cols_padded)
        )


def test_qwen4_exp_qsa_block_topk_logits_dispatches_persistent_radix(
    device: str, monkeypatch
) -> None:
    rows, heads, head_dim, page_size = 2, 4, 16, 64
    block_topk = 512
    pages_per_request = 2
    num_blocks = pages_per_request * page_size
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        3 * page_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    page_table = torch.tensor([[1, 2], [2, 1]], device=device, dtype=torch.int32)
    requests = torch.arange(rows, device=device, dtype=torch.long)
    complete_blocks = torch.tensor([100, 40], device=device, dtype=torch.int32)
    workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)
    calls = {}

    def fake_radix_topk(logits, out, topk, *, lengths, workspace, max_seq_len):
        calls.update(
            logits=logits,
            topk=topk,
            lengths=lengths,
            workspace=workspace,
            max_seq_len=max_seq_len,
        )
        out.fill_(7)

    monkeypatch.setattr(qsa_ops, "_is_nvidia", True)
    monkeypatch.setattr(qsa_ops, "has_ragged_decode_topk", lambda: True)
    monkeypatch.setattr(qsa_ops, "ragged_decode_topk", fake_radix_topk)
    monkeypatch.setattr(
        qsa_ops,
        "triton_topk_from_logits",
        lambda *args, **kwargs: pytest.fail("unexpected Triton top-k fallback"),
    )

    actual = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        solution="logits",
        persistent_topk_workspace=workspace,
        enable_pdl=False,
    )

    assert torch.equal(actual, torch.full_like(actual, 7))
    assert calls["logits"].shape == (rows, num_blocks)
    assert calls["topk"] == block_topk
    assert calls["lengths"] is complete_blocks
    assert calls["workspace"] is workspace
    assert calls["max_seq_len"] == num_blocks


def test_qwen4_exp_qsa_block_topk_logits_persistent_radix_matches_stream(
    device: str,
) -> None:
    if not has_ragged_decode_topk():
        pytest.skip("persistent radix top-k is unavailable")
    # 70400 blocks exercise the long-row persistent radix path. The second
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
    workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)

    kwargs = dict(page_size=page_size, block_topk=block_topk)
    stream = qwen4_exp_qsa_block_topk(
        query, key_cache, page_table, requests, complete_blocks, **kwargs
    )
    logits = qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        requests,
        complete_blocks,
        solution="logits",
        persistent_topk_workspace=workspace,
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
        # The streaming and persistent selectors may break equal-score ties
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


def test_qwen4_exp_qsa_block_topk_reads_strided_query(device: str) -> None:
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
        1, 4 * pages_per_request, (rows, pages_per_request), device=device
    )
    requests = torch.arange(rows, device=device)
    complete_blocks = torch.tensor([num_blocks, 40, 200], device=device)

    for kwargs in ({}, {"solution": "logits"}):
        strided = qwen4_exp_qsa_block_topk(
            query,
            key_cache,
            page_table,
            requests,
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            **kwargs,
        )
        packed = qwen4_exp_qsa_block_topk(
            query.contiguous(),
            key_cache,
            page_table,
            requests.to(torch.int32),
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            **kwargs,
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
