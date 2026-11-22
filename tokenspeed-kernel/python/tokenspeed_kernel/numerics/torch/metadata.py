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

"""PyTorch references for ``tokenspeed_kernel.ops.metadata``."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "metadata",
    "advance_accepted_frontier",
    name="torch_advance_accepted_frontier",
    solution="torch",
    signatures=format_signatures(
        "req_pool_indices", "dense", {torch.int32, torch.int64}
    ),
    priority=Priority.REFERENCE,
)
def torch_advance_accepted_frontier(
    req_pool_indices: torch.Tensor,
    input_lengths: torch.Tensor,
    accept_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    num_extends: int,
    padding_index: int,
) -> None:
    """Advance each request's valid cache length, in place: the first
    ``num_extends`` requests by their input length, the rest by their
    accepted length; padding rows add nothing."""
    deltas = torch.cat([input_lengths[:num_extends], accept_lengths[num_extends:]])
    deltas = torch.where(req_pool_indices != padding_index, deltas, 0).to(torch.int32)
    valid_cache_lengths.index_add_(0, req_pool_indices, deltas)


@register_kernel(
    "metadata",
    "expand_block_table_rows",
    name="torch_expand_block_table_rows",
    solution="torch",
    signatures=format_signatures("rows", "dense", {torch.int32}),
    priority=Priority.REFERENCE,
)
def torch_expand_block_table_rows(
    rows: torch.Tensor, values: torch.Tensor, out: torch.Tensor
) -> None:
    """Fill each block-table row of ``out`` from its span, one row at a time:
    null holes (0) before the span, its ``values``, then padding (-1)."""
    for offset, width, start, end, source in rows.tolist():
        row = out[offset : offset + width]
        row[:start] = 0
        row[start:end] = values[source : source + end - start]
        row[end:] = -1


@register_kernel(
    "metadata",
    "prepare_decode_tables",
    name="torch_prepare_decode_tables",
    solution="torch",
    signatures=format_signatures("tables", "dense", {torch.int32}),
    priority=Priority.REFERENCE,
)
def torch_prepare_decode_tables(
    sources: tuple[torch.Tensor, ...],
    tables: torch.Tensor,
    locations: torch.Tensor,
    lengths: torch.Tensor,
    actual_bs: int,
    tokens_per_req: int,
    ratios: tuple[int, ...],
    atom_pages_per_block: tuple[int, ...],
    pages: tuple[int, ...],
    page_sizes: tuple[int, ...],
    targets: tuple[tuple[torch.Tensor, torch.Tensor, int] | None, ...],
) -> None:
    """Torch implementation of ``prepare_decode_tables``' in-place contract."""
    bs = lengths.numel()
    if bs == 0:
        return
    steps = torch.arange(tokens_per_req, device=tables.device)
    positions = (lengths.to(torch.int64)[:, None] - tokens_per_req + steps).clamp_min(0)
    requests = torch.arange(bs, device=tables.device)[:, None]
    for g, source in enumerate(sources):
        dst = tables[g, :bs]
        if actual_bs:
            num_blocks = min(source.shape[1], -(-pages[g] // ratios[g]))
            blocks = source[:actual_bs, :num_blocks].to(torch.int64).unsqueeze(-1)
            offsets = torch.arange(ratios[g], device=blocks.device)
            live = torch.where(
                blocks > 0, blocks * atom_pages_per_block[g] + offsets, 0
            ).reshape(actual_bs, -1)
            num_pages = min(live.shape[1], pages[g])
            dst[:actual_bs, :num_pages].copy_(live[:, :num_pages])
            dst[:actual_bs, num_pages:].zero_()
        dst[actual_bs:].zero_()
        page_index = positions // page_sizes[g]
        page = dst[requests, page_index.clamp_max(dst.shape[1] - 1)].to(torch.int64)
        slots = page * page_sizes[g] + positions % page_sizes[g]
        slots = torch.where((page_index >= dst.shape[1]) | (page <= 0), 0, slots)
        locations[g, : bs * tokens_per_req].copy_(slots.flatten())
    copies = [(g, target) for g, target in enumerate(targets) if target is not None]
    if copies:
        floored = {
            floor: lengths if floor <= 1 else lengths.clamp_min(floor)
            for floor in {target[2] for _, target in copies}
        }
        torch._foreach_copy_(
            [buf for _, target in copies for buf in target[:2]],
            [
                src
                for g, target in copies
                for src in (floored[target[2]], tables[g, :bs, : pages[g]])
            ],
        )
