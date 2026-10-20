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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.qsa``: the block
scorer and attention over the selected cache slots."""

from __future__ import annotations

import torch
from tokenspeed_kernel.numerics.torch.attention.mha import torch_dense_attention
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "qsa_selected_attention",
    name="torch_qsa_selected_attention",
    solution="torch",
    signatures=format_signatures(
        "q", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_qsa_selected_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    scale: float,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    """Attention of each query row over its selected cache slots.

    Slots ``<= 0`` are padding; a row with no slots returns zeros. Cache
    values dequantize by the per-tensor ``k_scale``/``v_scale``.

    Args:
        q: ``[rows, heads, D]``.
        k_cache, v_cache: ``[slots, kv_heads, D]``.
        selected_slots: ``[rows, topk]``.
        scale: Logit scale.

    Returns:
        ``[rows, heads, Dv]`` in ``q``'s dtype.
    """
    output = torch.zeros(
        (q.shape[0], q.shape[1], v_cache.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    for row in range(q.shape[0]):
        slots = selected_slots[row][selected_slots[row] > 0].long()
        if slots.numel() == 0:
            continue
        output[row], _ = torch_dense_attention(
            q[row, None],
            k_cache[slots].float() * k_scale,
            v_cache[slots].float() * v_scale,
            scale=scale,
            causal_offset=None,
            window_left=None,
            sinks=None,
        )
    return output.to(q.dtype)


@register_kernel(
    "attention",
    "qsa_paged_block_scores",
    name="torch_qsa_paged_block_scores",
    solution="torch",
    signatures=format_signatures("query", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_qsa_paged_block_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    complete: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """``sum_h relu(q_h . k_b)`` of every block key in each row's request.

    Block ``b`` of a request reads ``key_cache[table[r, b // page_size] *
    page_size + b % page_size, 0]``; blocks past ``complete`` score ``-inf``.

    Returns:
        FP32 ``[rows, table_width * page_size]``.
    """
    num_blocks = table.shape[1] * page_size
    blocks = torch.arange(num_blocks, device=query.device)
    slots = table[requests].long()[:, blocks // page_size] * page_size + (
        blocks % page_size
    )
    keys = key_cache[slots, 0].float()
    scores = torch.relu(torch.einsum("rhd,rnd->rhn", query.float(), keys)).sum(1)
    return torch.where(blocks[None, :] < complete[:, None], scores, -torch.inf)


def qsa_stream_topk_ids(scores: torch.Tensor, k: int) -> list[int]:
    """Top-k block ids under the stream path's tie-breaking.

    Packed keys are ``(score_bits + 1) << 32 | block_id`` with zero reserved
    for non-finite scores, so equal FP32 bits prefer the larger block id.
    """
    ids = torch.arange(scores.shape[0], device=scores.device)
    bits = scores.contiguous().view(torch.int32).to(torch.int64)
    packed = ((bits + 1) << 32) | ids.to(torch.int64)
    packed = torch.where(torch.isfinite(scores), packed, torch.zeros_like(packed))
    top = torch.topk(packed, k).values
    return [int(v & 0xFFFFFFFF) for v in top.tolist() if v != 0]


def qsa_logits_topk_ids(scores: torch.Tensor, k: int) -> list[int]:
    """Top-k block ids under the logits selection's tie-breaking: equal scores
    prefer the smaller id and non-finite scores never win."""
    order = torch.argsort(scores, descending=True, stable=True)
    return [int(i) for i in order[:k].tolist() if torch.isfinite(scores[int(i)])]


@register_kernel(
    "attention",
    "qwen4_exp_qsa_pool_compress",
    name="torch_qwen4_exp_qsa_pool_compress",
    solution="torch",
    signatures=format_signatures("token_k", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_qwen4_exp_qsa_pool_compress(
    token_k: torch.Tensor,
    logical: torch.Tensor,
    requests: torch.Tensor,
    recent_locs: torch.Tensor,
    raw: torch.Tensor,
    position_values: torch.Tensor,
    position_cache: torch.Tensor,
    *,
    recent_page_size: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool each row's ``ratio`` logical positions ending at it.

    A pooled position comes from the current batch when an earlier row of
    the same request holds it, otherwise from the request's raw recent page
    at slot ``position % ratio``. The pool's first position value likewise
    comes from the batch or the page's position cache.

    Returns:
        ``(pooled [rows, ...] in raw's dtype, first position values)``.
    """
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


@register_kernel(
    "attention",
    "qwen4_exp_qsa_rope",
    name="torch_qwen4_exp_qsa_rope",
    solution="torch",
    signatures=format_signatures("tensor", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_qwen4_exp_qsa_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    sections: list[int] | None,
) -> torch.Tensor:
    """NeoX RoPE of the first ``rotary_dim`` features in ``tensor``'s dtype.

    2-D ``positions`` with ``sections`` take each section's cos/sin from its
    own position row (M-RoPE); without ``sections`` only row 0 is used.
    """
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


@register_kernel(
    "attention",
    "qwen4_exp_qsa_recent_rows",
    name="torch_qwen4_exp_qsa_recent_rows",
    solution="torch",
    signatures=format_signatures("token_k", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_qwen4_exp_qsa_recent_rows(
    token_k: torch.Tensor,
    logical: torch.Tensor,
    requests: torch.Tensor,
    recent_locs: torch.Tensor,
    position_values: torch.Tensor,
    raw: torch.Tensor,
    position_cache: torch.Tensor,
    *,
    recent_page_size: int,
    ratio: int,
    write_mask: torch.Tensor | None,
    request_limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw recent pages after writing each row that stays recent.

    A row writes its key to slot ``position % ratio`` of its page (and its
    position value when the slot is 0) unless masked out (``write_mask``,
    default ``recent_locs > 0``), the same request's row ``ratio`` later in
    the batch supersedes it, or its request is past ``request_limit``.

    Returns:
        Updated copies of ``(raw, position_cache)``.
    """
    raw = raw.clone()
    position_cache = position_cache.clone()
    rows = logical.shape[0]
    mask = recent_locs > 0 if write_mask is None else write_mask.clone()
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
