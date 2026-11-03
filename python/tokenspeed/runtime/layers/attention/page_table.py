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
