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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import torch


def build_prefill_kv_workspace_slots(
    *,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    device: torch.device,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """Flatten a kernel page table into per-token KV cache slot ids.

    Args:
        page_table: ``[num_reqs, num_pages]`` kernel page table whose entries
            index pages of ``page_size`` tokens.
        seq_lens: ``[num_reqs]`` number of valid tokens per request.
        max_seq_len: Upper bound of ``seq_lens``; fixes the gather width.
        page_size: Tokens covered by one kernel page.
        device: Device the slot tensor is built on.
        num_tokens: Exact sum of ``seq_lens`` when already known on the host.
            Providing it avoids the device synchronization otherwise required by
            boolean indexing to determine the packed output size.

    Returns:
        1-D int64 tensor of cache slot ids, request-major, covering exactly
        the first ``seq_lens[i]`` positions of each request.
    """
    if num_tokens is not None:
        seq_lens_i64 = seq_lens.to(device=device, dtype=torch.int64)
        request_ids = torch.repeat_interleave(
            torch.arange(seq_lens.numel(), dtype=torch.int64, device=device),
            seq_lens_i64,
            output_size=int(num_tokens),
        )
        seq_starts = torch.zeros(
            seq_lens.numel() + 1,
            dtype=torch.int64,
            device=device,
        )
        torch.cumsum(seq_lens_i64, dim=0, out=seq_starts[1:])
        local_offsets = torch.arange(
            int(num_tokens), dtype=torch.int64, device=device
        ) - seq_starts.index_select(0, request_ids)
        pages = page_table.to(device=device, dtype=torch.int64)[
            request_ids,
            torch.div(local_offsets, int(page_size), rounding_mode="floor"),
        ]
        return (pages * int(page_size) + local_offsets % int(page_size)).contiguous()

    local_offsets = torch.arange(
        int(max_seq_len),
        dtype=torch.int64,
        device=device,
    )
    page_offsets = torch.div(
        local_offsets,
        int(page_size),
        rounding_mode="floor",
    )
    block_offsets = local_offsets % int(page_size)
    pages = page_table.to(device=device, dtype=torch.int64).index_select(
        1,
        page_offsets,
    )
    slots = pages * int(page_size) + block_offsets
    valid = local_offsets.unsqueeze(0) < seq_lens.to(
        device=device,
        dtype=torch.int64,
    ).unsqueeze(1)
    return slots[valid].contiguous()


def safe_page_ids(
    block_table: torch.Tensor,
    req_indices: torch.Tensor,
    page_indices: torch.Tensor,
) -> torch.Tensor:
    """``block_table[req, page]`` with every out-of-range coordinate resolved
    to the ``-1`` sentinel instead of raising or clamping into a live row."""
    req_i64 = req_indices.to(torch.int64)
    page_i64 = page_indices.to(torch.int64)
    sentinel = torch.full_like(page_i64, -1, dtype=torch.int64)
    rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
    cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if rows <= 0 or cols <= 0:
        return sentinel
    valid = (req_i64 >= 0) & (req_i64 < rows) & (page_i64 >= 0) & (page_i64 < cols)
    safe_req = req_i64.clamp(0, rows - 1)
    safe_page = page_i64.clamp(0, cols - 1)
    page_ids = block_table[safe_req, safe_page].to(torch.int64)
    return torch.where(valid, page_ids, sentinel)


def expand_group_values_for_tokens(
    values: torch.Tensor,
    num_tokens: int,
    name: str,
) -> torch.Tensor:
    """Broadcast per-request values to per-token rows (packed multi-token
    decode expands each request's value across its uniform token count)."""
    if values.numel() == num_tokens:
        return values
    if values.numel() <= 0 or num_tokens % values.numel() != 0:
        raise RuntimeError(
            f"{name} has incompatible shape for packed tokens: "
            f"{values.numel()} entries for {num_tokens} tokens"
        )
    return values.repeat_interleave(num_tokens // values.numel())


def group_slot_mapping_from_raw(
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    rows_per_page: int,
    entry_stride_tokens: int = 1,
) -> torch.Tensor:
    """Per-token write slots over one group's raw table — the same
    ``table[req, pos // P] * P + pos % P`` invariant as the router's stacked
    math (``backends/write_locations.py``), generalized for arbitrary
    positions and entry strides (one entry per ``entry_stride_tokens``,
    compressed groups). Invalid coordinates yield the ``-1`` sentinel — the
    masked-scatter cache-insert kernels skip them — rather than slot 0,
    because these writes target group buffers with no reserved dummy page.
    """
    if rows_per_page <= 0:
        raise ValueError(f"rows_per_page must be > 0, got {rows_per_page}")
    if entry_stride_tokens <= 0:
        raise ValueError(f"entry_stride_tokens must be > 0, got {entry_stride_tokens}")
    pos_i64 = positions.to(torch.int64)
    logical_row = torch.div(pos_i64, entry_stride_tokens, rounding_mode="floor")
    logical_page = torch.div(logical_row, rows_per_page, rounding_mode="floor")
    offsets = logical_row % rows_per_page
    req_indices = expand_group_values_for_tokens(
        req_indices,
        positions.numel(),
        "request indices",
    )
    page_ids = safe_page_ids(block_table, req_indices, logical_page)
    slots = page_ids * rows_per_page + offsets
    return torch.where(page_ids >= 0, slots, torch.full_like(slots, -1))


def mask_invalid_graph_tokens(
    slot_mapping: torch.Tensor,
    is_valid_token: torch.Tensor | None,
) -> torch.Tensor:
    """Mask CUDA-graph padding rows to the ``-1`` sentinel (per-token or
    per-request validity, expanded like the request indices)."""
    if is_valid_token is None:
        return slot_mapping
    valid = expand_group_values_for_tokens(
        is_valid_token,
        slot_mapping.numel(),
        "slot validity mask",
    ).to(
        device=slot_mapping.device,
        dtype=torch.bool,
    )
    return torch.where(valid, slot_mapping, torch.full_like(slot_mapping, -1))
