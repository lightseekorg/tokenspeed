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
from tokenspeed_kernel.ops.attention.triton.page_table import (
    expand_page_table as fused_expand_page_table,
)


def _page_ratio(block_granularity: int, kernel_page_size: int) -> int:
    if (
        block_granularity <= 0
        or kernel_page_size <= 0
        or block_granularity % kernel_page_size
    ):
        raise ValueError(
            "block_granularity must be a positive multiple of kernel_page_size, "
            f"got {block_granularity} and {kernel_page_size}"
        )
    return block_granularity // kernel_page_size


def expand_page_table(
    page_table: torch.Tensor,
    *,
    block_granularity: int,
    kernel_page_size: int,
    max_kernel_pages: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand scheduler page IDs into the smaller pages consumed by a kernel."""
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be 2-D, got shape {tuple(page_table.shape)}")
    ratio = _page_ratio(block_granularity, kernel_page_size)
    if max_kernel_pages is None:
        max_kernel_pages = page_table.shape[1] * ratio
    if max_kernel_pages < 0:
        raise ValueError(
            f"max_kernel_pages must be non-negative, got {max_kernel_pages}"
        )

    rows = page_table.shape[0]
    if ratio == 1 and out is None and max_kernel_pages <= page_table.shape[1]:
        return page_table[:, :max_kernel_pages]

    # The fused kernel writes every column it is handed, zeros included.
    fused = ratio != 1 and rows > 0 and page_table.is_cuda
    if out is None:
        alloc = torch.empty if fused else torch.zeros
        out = alloc(
            (rows, max_kernel_pages),
            dtype=page_table.dtype,
            device=page_table.device,
        )
    else:
        if out.ndim != 2 or out.shape[0] < rows or out.shape[1] < max_kernel_pages:
            raise ValueError(
                f"out shape {tuple(out.shape)} cannot hold ({rows}, {max_kernel_pages})"
            )
        if out.dtype != page_table.dtype or out.device != page_table.device:
            raise ValueError("out must have the same dtype and device as page_table")
        if not fused:
            out[:rows].zero_()

    result = out[:rows, :max_kernel_pages]
    if fused:
        return fused_expand_page_table(
            page_table, out, ratio=ratio, max_kernel_pages=max_kernel_pages
        )
    if ratio == 1:
        copy_columns = min(max_kernel_pages, page_table.shape[1])
        result[:, :copy_columns].copy_(page_table[:, :copy_columns])
        return result

    logical_columns = min(
        page_table.shape[1],
        (max_kernel_pages + ratio - 1) // ratio,
    )
    if logical_columns == 0:
        return result

    kernel_offsets = torch.arange(
        ratio,
        dtype=page_table.dtype,
        device=page_table.device,
    )
    expanded = (
        page_table[:, :logical_columns].clamp_min(0).unsqueeze(-1) * ratio
        + kernel_offsets
    ).reshape(rows, logical_columns * ratio)
    copy_columns = min(max_kernel_pages, expanded.shape[1])
    result[:, :copy_columns].copy_(expanded[:, :copy_columns])
    return result


def build_prefill_kv_workspace_slots(
    *,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Flatten a kernel page table into per-token KV cache slot ids.

    Args:
        page_table: ``[num_reqs, num_pages]`` kernel page table whose entries
            index pages of ``page_size`` tokens.
        seq_lens: ``[num_reqs]`` number of valid tokens per request.
        max_seq_len: Upper bound of ``seq_lens``; fixes the gather width.
        page_size: Tokens covered by one kernel page.
        device: Device the slot tensor is built on.

    Returns:
        1-D int64 tensor of cache slot ids, request-major, covering exactly
        the first ``seq_lens[i]`` positions of each request.
    """
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
