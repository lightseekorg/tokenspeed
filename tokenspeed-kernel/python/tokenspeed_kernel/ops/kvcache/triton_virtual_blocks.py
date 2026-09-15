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

"""Translate whole virtual cache blocks into safe rank-local addresses."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def virtual_block_to_local(block, DEGREE: tl.constexpr, RANK: tl.constexpr):
    """Return a safe physical page and its explicit owner mask."""
    positive = tl.maximum(block - 1, 0)
    owned = (block > 0) & (positive % DEGREE == RANK)
    local = positive // DEGREE + 1
    return tl.where(owned, local, 0), owned


@triton.jit
def _translate_virtual_slots(
    source,
    destination,
    owner_mask,
    count,
    ROWS: tl.constexpr,
    VIRTUAL_COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < count
    raw = tl.load(source + offsets, mask=active, other=0).to(tl.int64)
    safe = tl.maximum(raw, 0)
    local, owner = virtual_block_to_local(safe // ROWS, DEGREE, RANK)
    owner = owner & (raw >= ROWS) & (raw < VIRTUAL_COUNT * ROWS)
    slot = tl.where(owner, local * ROWS + safe % ROWS, 0)
    tl.store(destination + offsets, slot, mask=active)
    tl.store(owner_mask + offsets, owner, mask=active)


def virtual_slots_to_local(
    slots: torch.Tensor,
    *,
    rows_per_page: int,
    virtual_block_count: int,
    degree: int,
    rank: int,
    out: torch.Tensor | None = None,
    owner_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate scheduler slots, with page 0 and nonowners explicitly invalid.

    Args:
        slots: Integer virtual slots; negative entries are also invalid.
        rows_per_page: Unchanged physical rows per owned cache block. Use 1
            to translate a block table instead of row slots.
        virtual_block_count: Scheduler capacity including null block 0.
        degree: Number of owners, or 1 for a replicated group.
        rank: Owner index in [0, degree); replicated groups use 0.
        out: Optional contiguous output tensor with the same shape/dtype.
        owner_mask: Optional contiguous boolean output of the same shape.

    Returns:
        Safe local slots and a boolean mask. Every false-mask slot is 0;
        callers must use the mask to suppress all payload and scale stores.
    """
    if rows_per_page <= 0 or virtual_block_count <= 1 or degree <= 0:
        raise ValueError(
            "virtual slot geometry must be positive and include usable pages"
        )
    if not 0 <= rank < degree:
        raise ValueError("virtual slot owner rank is out of range")
    if slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("virtual slots must be int32 or int64")
    if out is None:
        out = torch.empty_like(slots, memory_format=torch.contiguous_format)
    if owner_mask is None:
        owner_mask = torch.empty(slots.shape, dtype=torch.bool, device=slots.device)
    for value, dtype in ((out, slots.dtype), (owner_mask, torch.bool)):
        if (
            value.shape != slots.shape
            or value.dtype != dtype
            or value.device != slots.device
            or not value.is_contiguous()
        ):
            raise ValueError("virtual slot outputs must match shape, device and dtype")
    if slots.numel() == 0:
        return out, owner_mask
    if slots.is_cuda:
        source = slots.contiguous()
        _translate_virtual_slots[(triton.cdiv(slots.numel(), 256),)](
            source,
            out,
            owner_mask,
            slots.numel(),
            ROWS=rows_per_page,
            VIRTUAL_COUNT=virtual_block_count,
            DEGREE=degree,
            RANK=rank,
            BLOCK=256,
        )
    else:
        safe = slots.to(torch.int64).clamp_min(0)
        block = safe // rows_per_page
        positive = (block - 1).clamp_min(0)
        owned = (
            (slots >= rows_per_page)
            & (slots < virtual_block_count * rows_per_page)
            & (positive % degree == rank)
        )
        local = (positive // degree + 1) * rows_per_page + safe % rows_per_page
        out.copy_(torch.where(owned, local, 0))
        owner_mask.copy_(owned)
    return out, owner_mask
