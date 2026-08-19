# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pure cyclic sequence-layout math used by DCP metadata planners."""

from __future__ import annotations

import math

import torch


def _validate_rank(dcp_rank: int, dcp_size: int) -> None:
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")


def dcp_block_span(
    configured_prefix_granularity: int,
    kernel_page_size: int,
    dcp_size: int,
) -> int:
    """Global scheduler block span for fixed-size local kernel pages."""
    if configured_prefix_granularity <= 0 or kernel_page_size <= 0:
        raise ValueError("prefix granularity and kernel page size must be positive")
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    return math.lcm(configured_prefix_granularity, kernel_page_size * dcp_size)


def owner_rank(global_position: int, dcp_size: int) -> int:
    if global_position < 0:
        raise ValueError(f"global_position must be non-negative, got {global_position}")
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    return global_position % dcp_size


def local_position(global_position: int, dcp_size: int) -> int:
    if global_position < 0:
        raise ValueError(f"global_position must be non-negative, got {global_position}")
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    return global_position // dcp_size


def local_length(global_length: int, dcp_rank: int, dcp_size: int) -> int:
    """Rows owned by one rank in global positions ``[0, global_length)``."""
    _validate_rank(dcp_rank, dcp_size)
    if global_length < 0:
        raise ValueError(f"global_length must be non-negative, got {global_length}")
    return global_length // dcp_size + int(dcp_rank < global_length % dcp_size)


def local_lengths(
    global_lengths: torch.Tensor,
    dcp_rank: int,
    dcp_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized, graph-safe :func:`local_length` with optional stable output."""
    _validate_rank(dcp_rank, dcp_size)
    if (
        not torch.is_floating_point(global_lengths)
        and global_lengths.dtype != torch.bool
    ):
        result = torch.div(global_lengths, dcp_size, rounding_mode="floor")
        result = result + (dcp_rank < torch.remainder(global_lengths, dcp_size))
    else:
        raise TypeError(
            f"global_lengths must have an integer dtype, got {global_lengths.dtype}"
        )
    result = result.clamp_min(0).to(global_lengths.dtype)
    if out is not None:
        if out.shape != global_lengths.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != input shape {tuple(global_lengths.shape)}"
            )
        out.copy_(result)
        return out
    return result


def visible_local_lengths(
    final_global_lengths: torch.Tensor,
    query_width: int,
    dcp_rank: int,
    dcp_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Visible local rows for each trailing multi-token decode query.

    ``final_global_lengths`` includes all ``query_width`` trailing tokens.
    Query column ``j`` therefore has the global exclusive causal bound
    ``L - query_width + 1 + j``.
    """
    _validate_rank(dcp_rank, dcp_size)
    if query_width <= 0:
        raise ValueError(f"query_width must be positive, got {query_width}")
    steps = torch.arange(
        1 - query_width,
        1,
        dtype=final_global_lengths.dtype,
        device=final_global_lengths.device,
    )
    bounds = (final_global_lengths.unsqueeze(-1) + steps).clamp_min(0)
    result = local_lengths(bounds, dcp_rank, dcp_size)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != result shape {tuple(result.shape)}"
            )
        out.copy_(result)
        return out
    return result


def local_page_table(
    logical_page_table: torch.Tensor,
    *,
    global_block_span: int,
    kernel_page_size: int,
    dcp_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand logical DCP blocks into rank-local kernel pages.

    The scheduler page ID is shared across ranks. One scheduler block contains
    ``global_block_span / dcp_size`` local rows, which may span multiple fixed
    kernel pages. Holes (negative IDs) are preserved as holes.
    """
    if global_block_span <= 0 or kernel_page_size <= 0 or dcp_size <= 0:
        raise ValueError("block span, kernel page size, and dcp size must be positive")
    if global_block_span % (kernel_page_size * dcp_size):
        raise ValueError(
            f"global_block_span={global_block_span} must be divisible by "
            f"kernel_page_size*dcp_size={kernel_page_size * dcp_size}"
        )
    ratio = global_block_span // (kernel_page_size * dcp_size)
    offsets = torch.arange(
        ratio, dtype=logical_page_table.dtype, device=logical_page_table.device
    )
    expanded = logical_page_table.unsqueeze(-1) * ratio + offsets
    expanded = torch.where(
        logical_page_table.unsqueeze(-1) >= 0,
        expanded,
        torch.full_like(expanded, -1),
    ).flatten(-2)
    if out is not None:
        if out.shape != expanded.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != result shape {tuple(expanded.shape)}"
            )
        out.copy_(expanded)
        return out
    return expanded


def local_cache_slots(
    global_positions: torch.Tensor,
    logical_page_table: torch.Tensor,
    *,
    global_block_span: int,
    dcp_rank: int,
    dcp_size: int,
    out: torch.Tensor | None = None,
    owned_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve fixed-shape local write slots and their ownership mask.

    ``global_positions`` and ``logical_page_table`` are batch aligned. The
    result uses slot zero for non-owned/padded positions; callers must predicate
    stores with the returned mask so the reserved null page remains untouched.
    """
    _validate_rank(dcp_rank, dcp_size)
    if global_block_span <= 0 or global_block_span % dcp_size:
        raise ValueError(
            f"global_block_span={global_block_span} must be positive and divisible "
            f"by dcp_size={dcp_size}"
        )
    if global_positions.ndim != 2 or logical_page_table.ndim != 2:
        raise ValueError("global_positions and logical_page_table must both be 2-D")
    if global_positions.shape[0] != logical_page_table.shape[0]:
        raise ValueError("global positions and page table batch dimensions differ")

    valid = global_positions >= 0
    safe_positions = global_positions.clamp_min(0).to(torch.int64)
    block_columns = torch.div(safe_positions, global_block_span, rounding_mode="floor")
    in_bounds = block_columns < logical_page_table.shape[1]
    safe_columns = block_columns.clamp_max(max(logical_page_table.shape[1] - 1, 0))
    pages = logical_page_table.gather(1, safe_columns)
    owned = (
        valid
        & in_bounds
        & (torch.remainder(safe_positions, dcp_size) == dcp_rank)
        & (pages > 0)
    )
    local_rows_per_block = global_block_span // dcp_size
    row_in_block = torch.div(
        torch.remainder(safe_positions, global_block_span),
        dcp_size,
        rounding_mode="floor",
    )
    slots = pages.to(torch.int64) * local_rows_per_block + row_in_block
    slots = torch.where(owned, slots, torch.zeros_like(slots))
    if out is not None:
        if out.shape != slots.shape:
            raise ValueError(f"out shape {tuple(out.shape)} != {tuple(slots.shape)}")
        out.copy_(slots)
        slots = out
    if owned_out is not None:
        if owned_out.shape != owned.shape:
            raise ValueError(
                f"owned_out shape {tuple(owned_out.shape)} != {tuple(owned.shape)}"
            )
        owned_out.copy_(owned)
        owned = owned_out
    return slots, owned
