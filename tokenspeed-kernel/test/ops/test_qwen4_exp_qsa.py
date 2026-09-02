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
    qwen4_exp_qsa_complete_blocks,
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_group_cache_locs,
    qwen4_exp_qsa_logical_layout,
    qwen4_exp_qsa_mqa_scores,
    qwen4_exp_qsa_norm_rope,
    qwen4_exp_qsa_recent_write,
    qwen4_exp_qsa_rope,
    qwen4_exp_qsa_selected_tokens,
    qwen4_exp_qsa_sparse_attention,
    qwen4_exp_qsa_sparse_slots,
    qwen4_exp_qsa_stage_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Qwen4-Exp QSA kernels require CUDA or ROCm",
)


def test_qwen4_exp_qsa_mqa_scores_matches_torch(device: str) -> None:
    torch.manual_seed(17)
    rows, heads, head_dim, keys = 3, 4, 32, 40
    query = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(64, 1, head_dim, device=device, dtype=torch.bfloat16)
    key_slots = torch.randint(1, 64, (rows, keys), device=device, dtype=torch.int32)
    valid_counts = torch.tensor([5, 23, keys], device=device, dtype=torch.int32)

    actual = qwen4_exp_qsa_mqa_scores(query, key_cache, key_slots, valid_counts)
    gathered = key_cache[key_slots.long(), 0]
    expected = torch.relu(
        torch.einsum("mhd,mnd->mhn", query.float(), gathered.float())
    ).sum(dim=1)
    columns = torch.arange(keys, device=device).unsqueeze(0)
    expected.masked_fill_(columns >= valid_counts.unsqueeze(1), -float("inf"))

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qwen4_exp_qsa_sparse_attention_matches_torch(device: str) -> None:
    torch.manual_seed(23)
    rows, q_heads, kv_heads, head_dim, budget = 4, 8, 2, 64, 37
    query = torch.randn(rows, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(96, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    selected = torch.randint(1, 96, (rows, budget), device=device, dtype=torch.int32)
    selected[0, 13:] = -1
    scale = head_dim**-0.5

    actual = qwen4_exp_qsa_sparse_attention(
        query,
        key_cache,
        value_cache,
        selected,
        scale=scale,
    )
    expected = torch.zeros_like(actual)
    group_size = q_heads // kv_heads
    for row in range(rows):
        slots = selected[row][selected[row] > 0].long()
        for head in range(q_heads):
            kv_head = head // group_size
            scores = query[row, head].float() @ key_cache[slots, kv_head].float().T
            probabilities = torch.softmax(scores * scale, dim=-1)
            expected[row, head] = (
                probabilities.to(value_cache.dtype) @ value_cache[slots, kv_head]
            )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_qwen4_exp_qsa_sparse_attention_ignores_empty_selection_blocks(
    device: str,
) -> None:
    query = torch.randn(2, 4, 16, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(96, 1, 16, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn(96, 1, 16, device=device, dtype=torch.bfloat16)
    selected = torch.full((2, 64), -1, device=device, dtype=torch.int32)
    selected[0, 40:44] = torch.tensor([3, 9, 17, 25], device=device)

    actual = qwen4_exp_qsa_sparse_attention(
        query,
        key_cache,
        value_cache,
        selected,
        scale=16**-0.5,
    )
    slots = selected[0, 40:44].long()
    scores = torch.einsum("hd,kd->hk", query[0].float(), key_cache[slots, 0].float())
    probabilities = torch.softmax(scores * (16**-0.5), dim=-1)
    expected_first = probabilities @ value_cache[slots, 0].float()

    torch.testing.assert_close(actual[0].float(), expected_first, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]))


def test_qwen4_exp_qsa_logical_layout_matches_torch(device: str) -> None:
    seq_lens = torch.tensor([8, 12, 5], device=device, dtype=torch.int32)
    query_lengths = torch.tensor([4, 3, 5], device=device, dtype=torch.long)
    total_tokens = int(query_lengths.sum())

    positions, requests = qwen4_exp_qsa_logical_layout(
        seq_lens, query_lengths, total_tokens
    )

    request_ids = torch.arange(3, device=device)
    expected_requests = torch.repeat_interleave(
        request_ids, query_lengths, output_size=total_tokens
    )
    cumulative = torch.cumsum(query_lengths, dim=0)
    row_starts = torch.repeat_interleave(
        cumulative - query_lengths, query_lengths, output_size=total_tokens
    )
    offsets = torch.arange(total_tokens, device=device) - row_starts
    expected_positions = (seq_lens.long() - query_lengths)[expected_requests] + offsets

    torch.testing.assert_close(positions, expected_positions)
    torch.testing.assert_close(requests, expected_requests)

    # Uniform scalar lengths must reproduce the same layout without any
    # lengths tensor (the decode fast path).
    uniform_seq_lens = torch.tensor([8, 12], device=device, dtype=torch.int32)
    uniform_positions, uniform_requests = qwen4_exp_qsa_logical_layout(
        uniform_seq_lens, 4, 8
    )
    torch.testing.assert_close(
        uniform_positions, torch.tensor([4, 5, 6, 7, 8, 9, 10, 11], device=device)
    )
    torch.testing.assert_close(
        uniform_requests, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    )


def test_qwen4_exp_qsa_group_cache_locs_matches_torch(device: str) -> None:
    positions = torch.tensor([3, 16, 5, -1, 20, 21], device=device, dtype=torch.long)
    requests = torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.long)
    # QSA table at consumer granularity with a 2x expansion: logical pages
    # [[2, 0, 7], [5, 9, 1]] stored as page id * expansion + sub-page index.
    qsa_logical = torch.tensor([[2, 0, 7], [5, 9, 1]], device=device, dtype=torch.int32)
    qsa_expanded = (qsa_logical.repeat_interleave(2, dim=1) * 2) + torch.arange(
        6, device=device
    ) % 2
    recent_table = torch.tensor(
        [[3, 1, 6, 2, 8, 4], [6, 0, 2, 9, 3, 1]], device=device, dtype=torch.int32
    )

    qsa_locs, recent_locs, complete_blocks = qwen4_exp_qsa_group_cache_locs(
        positions, requests, qsa_expanded, 2, 8, recent_table, 1, 4, 4
    )

    safe = positions.clamp_min(0)
    qsa_pages = qsa_logical[requests, safe // 8].long()
    expected_qsa = qsa_pages * 8 + safe % 8
    expected_qsa = torch.where((positions >= 0) & (qsa_pages > 0), expected_qsa, 0)
    recent_pages = recent_table[requests, safe // 4].long()
    expected_recent = recent_pages * 4 + safe % 4
    expected_recent = torch.where(
        (positions >= 0) & (recent_pages > 0), expected_recent, 0
    )

    torch.testing.assert_close(qsa_locs, expected_qsa.to(torch.int32))
    torch.testing.assert_close(recent_locs, expected_recent.to(torch.int32))
    torch.testing.assert_close(complete_blocks, ((positions + 1) // 4).to(torch.int32))


def test_qwen4_exp_qsa_complete_blocks_matches_torch(device: str) -> None:
    positions = torch.tensor([0, 3, 4, 17], device=device, dtype=torch.long)

    actual = qwen4_exp_qsa_complete_blocks(positions, 4)
    expected = ((positions + 1) // 4).to(torch.int32)

    torch.testing.assert_close(actual, expected)


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


def test_qwen4_exp_qsa_stage_verify_matches_copies(device: str) -> None:
    torch.manual_seed(41)
    bs, width, head_dim = 3, 8, 128
    rows = bs * width
    # Trailing slices of larger buffers, exactly like the indexer's verify
    # path; the positions view is the transposed mrope layout with strides
    # (1, total_rows), so the fused copy must gather, not memcpy.
    token_k = torch.randn(rows + 5, 1, head_dim, device=device, dtype=torch.bfloat16)[
        -rows:
    ]
    rope_positions = torch.randint(
        0, 4096, (3, rows + 5), device=device, dtype=torch.int64
    )
    position_values = rope_positions.T[-rows:]
    logical_positions = torch.arange(
        1000, 1000 + rows + 5, device=device, dtype=torch.int64
    )[-rows:]
    recent_locs = torch.randint(1, 4096, (rows + 5,), device=device, dtype=torch.int32)[
        -rows:
    ]
    staged = (
        token_k.new_empty((bs, width, 1, head_dim)),
        position_values.new_empty((bs, width, 3)),
        logical_positions.new_empty((bs, width)),
        recent_locs.new_empty((bs, width)),
    )
    expected = tuple(tensor.clone() for tensor in staged)

    qwen4_exp_qsa_stage_verify(
        token_k, position_values, logical_positions, recent_locs, *staged
    )

    expected[0].copy_(token_k.view_as(expected[0]))
    expected[1].copy_(position_values.view_as(expected[1]))
    expected[2].copy_(logical_positions.view_as(expected[2]))
    expected[3].copy_(recent_locs.view_as(expected[3]))
    for actual, reference in zip(staged, expected):
        torch.testing.assert_close(actual, reference)


def test_qwen4_exp_qsa_stage_verify_handles_expanded_positions(
    device: str,
) -> None:
    torch.manual_seed(43)
    bs, width, head_dim = 2, 4, 64
    rows = bs * width
    # One-dimensional scheduler positions expand to a stride-zero
    # ``[rows, 3]`` view; all three axes must land in the staged buffer.
    token_k = torch.randn(rows, 1, head_dim, device=device, dtype=torch.float16)
    positions = torch.arange(rows, device=device, dtype=torch.int64)
    position_values = positions.unsqueeze(0).expand(3, -1).reshape(3, -1).T
    logical_positions = torch.arange(512, 512 + rows, device=device, dtype=torch.int64)
    recent_locs = torch.randint(1, 1024, (rows,), device=device, dtype=torch.int32)
    staged = (
        token_k.new_empty((bs, width, 1, head_dim)),
        position_values.new_empty((bs, width, 3)),
        logical_positions.new_empty((bs, width)),
        recent_locs.new_empty((bs, width)),
    )

    qwen4_exp_qsa_stage_verify(
        token_k, position_values, logical_positions, recent_locs, *staged
    )

    torch.testing.assert_close(staged[0], token_k.reshape(bs, width, 1, head_dim))
    torch.testing.assert_close(
        staged[1],
        positions.unsqueeze(-1).expand(rows, 3).reshape(bs, width, 3),
    )
    torch.testing.assert_close(staged[2], logical_positions.reshape(bs, width))
    torch.testing.assert_close(staged[3], recent_locs.reshape(bs, width))


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


def test_qwen4_exp_qsa_selected_tokens_matches_torch(device: str) -> None:
    ratio, block_topk = 4, 8
    token_topk = block_topk * ratio
    selected_blocks = torch.tensor(
        [[0, 5, 2, 9, 1, 3, 4, 6], [7, 0, 0, 0, 0, 0, 0, 0]],
        device=device,
        dtype=torch.int64,
    )
    selected_blocks[0, 5:] = -1
    complete_blocks = torch.tensor([6, 3], device=device, dtype=torch.int32)
    logical = torch.tensor([25, 10], device=device, dtype=torch.long)

    actual = qwen4_exp_qsa_selected_tokens(
        selected_blocks, complete_blocks, logical, ratio, token_topk
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
    expected = torch.cat((tokens, suffix), dim=1).to(torch.int32)

    torch.testing.assert_close(actual, expected)


def test_qwen4_exp_qsa_sparse_slots_matches_torch(device: str) -> None:
    page_size = 16
    selected = torch.tensor(
        [[0, 15, 16, 40, -1], [3, 7, 8, 2, 5]], device=device, dtype=torch.int32
    )
    logical = torch.tensor([20, 7], device=device, dtype=torch.long)
    requests = torch.tensor([0, 1], device=device, dtype=torch.long)
    page_table = torch.tensor([[2, 4, 0], [9, 1, 6]], device=device, dtype=torch.int32)

    actual = qwen4_exp_qsa_sparse_slots(
        selected, logical, requests, page_table, page_size
    )

    valid = (selected.long() >= 0) & (selected.long() <= logical.unsqueeze(1))
    safe = selected.long().clamp_min(0)
    columns = safe // page_size
    pages = page_table[requests.unsqueeze(1).expand_as(columns), columns].long()
    expected = pages * page_size + safe % page_size
    expected = torch.where(valid & (pages > 0), expected, -1).to(torch.int32)

    torch.testing.assert_close(actual, expected)


def _ref_rope(tensor, positions, cos_sin_cache, rotary_dim, sections=None):
    if positions.ndim == 2 and sections is None:
        positions = positions[0]
    cos_sin = cos_sin_cache[positions.long()]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        cos = torch.cat(
            [part[axis] for axis, part in enumerate(cos.split(sections, -1))], dim=-1
        )
        sin = torch.cat(
            [part[axis] for axis, part in enumerate(sin.split(sections, -1))], dim=-1
        )
    rotary = tensor[..., :rotary_dim]
    passthrough = tensor[..., rotary_dim:]
    cos_t = cos.unsqueeze(-2).to(tensor.dtype)
    sin_t = sin.unsqueeze(-2).to(tensor.dtype)
    first, second = torch.chunk(rotary, 2, dim=-1)
    rotated = torch.cat(
        (first * cos_t - second * sin_t, second * cos_t + first * sin_t), dim=-1
    )
    return torch.cat((rotated, passthrough), dim=-1)


def test_qwen4_exp_qsa_rope_matches_torch(device: str) -> None:
    torch.manual_seed(7)
    rotary_dim, head_dim, heads = 32, 40, 3
    cos_sin_cache = torch.randn(64, rotary_dim, device=device, dtype=torch.float32)
    tensor = torch.randn(5, heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 64, (5,), device=device)

    actual = qwen4_exp_qsa_rope(tensor, positions, cos_sin_cache)
    expected = _ref_rope(tensor, positions, cos_sin_cache, rotary_dim)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qwen4_exp_qsa_rope_matches_mrope_sections(device: str) -> None:
    torch.manual_seed(13)
    rotary_dim, head_dim, heads = 32, 32, 2
    sections = (4, 6, 6)
    cos_sin_cache = torch.randn(48, rotary_dim, device=device, dtype=torch.float32)
    tensor = torch.randn(4, heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 48, (3, 4), device=device)

    actual = qwen4_exp_qsa_rope(tensor, positions, cos_sin_cache, sections=sections)
    expected = _ref_rope(tensor, positions, cos_sin_cache, rotary_dim, sections)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    axis0 = qwen4_exp_qsa_rope(tensor, positions, cos_sin_cache)
    expected_axis0 = _ref_rope(tensor, positions, cos_sin_cache, rotary_dim)

    torch.testing.assert_close(axis0, expected_axis0, rtol=2e-2, atol=2e-2)


def test_qwen4_exp_qsa_mqa_scores_reads_strided_inputs(device: str) -> None:
    torch.manual_seed(31)
    rows, heads, head_dim, keys = 3, 4, 32, 40
    wide = torch.randn(rows, heads, 2 * head_dim, device=device, dtype=torch.bfloat16)
    query = wide[:, :, :head_dim]
    key_cache = torch.randn(64, 1, head_dim, device=device, dtype=torch.bfloat16)
    # Int64 row slices of wider grids exercise strided slot reads, in-kernel
    # casts, and the strided valid-counts load without any host-side copy.
    slot_grid = torch.randint(1, 64, (rows, 2, keys), device=device, dtype=torch.int64)
    key_slots = slot_grid[:, 1]
    counts_grid = torch.randint(
        1, keys + 1, (rows, 2), device=device, dtype=torch.int64
    )
    valid_counts = counts_grid[:, 1]

    actual = qwen4_exp_qsa_mqa_scores(query, key_cache, key_slots, valid_counts)
    gathered = key_cache[key_slots, 0]
    expected = torch.relu(
        torch.einsum("mhd,mnd->mhn", query.float(), gathered.float())
    ).sum(dim=1)
    columns = torch.arange(keys, device=device).unsqueeze(0)
    expected.masked_fill_(columns >= valid_counts.unsqueeze(1), -float("inf"))

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qwen4_exp_qsa_sparse_attention_reads_strided_inputs(device: str) -> None:
    torch.manual_seed(47)
    rows, q_heads, kv_heads, head_dim, budget = 4, 8, 2, 64, 29
    wide = torch.randn(rows, 2 * q_heads, head_dim, device=device, dtype=torch.bfloat16)
    query = wide[:, :q_heads]
    key_cache = torch.randn(96, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    slot_grid = torch.randint(
        1, 96, (rows, 2, budget), device=device, dtype=torch.int64
    )
    selected = slot_grid[:, 1]
    selected[0, 17:] = -1
    scale = head_dim**-0.5

    actual = qwen4_exp_qsa_sparse_attention(
        query,
        key_cache,
        value_cache,
        selected,
        scale=scale,
    )
    expected = torch.zeros_like(actual)
    group_size = q_heads // kv_heads
    for row in range(rows):
        slots = selected[row][selected[row] > 0].long()
        for head in range(q_heads):
            kv_head = head // group_size
            scores = query[row, head].float() @ key_cache[slots, kv_head].float().T
            probabilities = torch.softmax(scores * scale, dim=-1)
            expected[row, head] = (
                probabilities.to(value_cache.dtype) @ value_cache[slots, kv_head]
            )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


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


def test_qwen4_exp_qsa_norm_rope_matches_torch(device: str) -> None:
    torch.manual_seed(19)
    tokens, heads, head_dim, rotary_dim = 5, 3, 16, 8
    norm_weight = torch.rand(head_dim, device=device) + 0.5
    epsilon = 1e-6
    cos_sin_cache = torch.randn(32, rotary_dim, device=device)
    positions = torch.randint(0, 32, (tokens,), device=device)
    wide = torch.randn(
        tokens, heads * head_dim + 4, device=device, dtype=torch.bfloat16
    )

    def reference(region):
        x = region.float().view(tokens, heads, head_dim)
        normed = (
            x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon) * norm_weight
        )
        return _ref_rope(normed, positions, cos_sin_cache, rotary_dim)

    contiguous = wide[:, : heads * head_dim].contiguous()
    actual = qwen4_exp_qsa_norm_rope(
        contiguous, positions, norm_weight, epsilon, cos_sin_cache, num_heads=heads
    )
    torch.testing.assert_close(
        actual.float(), reference(contiguous), rtol=2e-2, atol=2e-2
    )

    # The same result must come out of the strided GEMM-view split.
    strided = wide[:, : heads * head_dim]
    actual_strided = qwen4_exp_qsa_norm_rope(
        strided, positions, norm_weight, epsilon, cos_sin_cache, num_heads=heads
    )
    torch.testing.assert_close(
        actual_strided.float(), reference(strided), rtol=2e-2, atol=2e-2
    )


def test_qwen4_exp_qsa_norm_rope_reads_strided_positions(device: str) -> None:
    torch.manual_seed(67)
    tokens, heads, head_dim, rotary_dim = 4, 3, 16, 8
    norm_weight = torch.rand(head_dim, device=device) + 0.5
    cos_sin_cache = torch.randn(32, rotary_dim, device=device)
    inputs = torch.randn(tokens, heads * head_dim, device=device, dtype=torch.bfloat16)
    # Strided int32 positions: every other element of a wider grid.
    grid = torch.randint(0, 32, (tokens, 2), device=device, dtype=torch.int32)
    positions = grid[:, 1]
    assert not positions.is_contiguous()

    actual = qwen4_exp_qsa_norm_rope(
        inputs, positions, norm_weight, 1e-6, cos_sin_cache, num_heads=heads
    )
    packed = qwen4_exp_qsa_norm_rope(
        inputs,
        positions.contiguous().long(),
        norm_weight,
        1e-6,
        cos_sin_cache,
        num_heads=heads,
    )
    torch.testing.assert_close(actual, packed, rtol=2e-2, atol=2e-2)

    # Strided ``[3, tokens]`` positions exercise the axis stride path too.
    mrope = torch.randint(0, 32, (3, 2 * tokens), device=device, dtype=torch.int32)[
        :, 1::2
    ]
    actual_mrope = qwen4_exp_qsa_norm_rope(
        inputs, mrope, norm_weight, 1e-6, cos_sin_cache, num_heads=heads
    )
    packed_mrope = qwen4_exp_qsa_norm_rope(
        inputs,
        mrope.contiguous().long(),
        norm_weight,
        1e-6,
        cos_sin_cache,
        num_heads=heads,
    )
    torch.testing.assert_close(actual_mrope, packed_mrope, rtol=2e-2, atol=2e-2)


def test_qwen4_exp_qsa_rope_reads_strided_inputs(device: str) -> None:
    torch.manual_seed(71)
    rotary_dim, head_dim, heads = 32, 40, 2
    cos_sin_cache = torch.randn(64, rotary_dim, device=device, dtype=torch.float32)
    # ``[tokens, heads, 2 * head_dim]`` sliced on the last dim keeps a
    # non-unit element stride across every access in the kernel.
    wide = torch.randn(5, heads, 2 * head_dim, device=device, dtype=torch.bfloat16)
    tensor = wide[:, :, :head_dim]
    assert not tensor.is_contiguous()
    grid = torch.randint(0, 64, (2, 5, 2), device=device, dtype=torch.int32)
    positions = grid[:, :, 1]
    assert not positions.is_contiguous()

    actual = qwen4_exp_qsa_rope(tensor, positions, cos_sin_cache)
    packed = qwen4_exp_qsa_rope(
        tensor.contiguous(), positions.contiguous().long(), cos_sin_cache
    )
    torch.testing.assert_close(actual, packed, rtol=2e-2, atol=2e-2)
