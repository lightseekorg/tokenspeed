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

"""Scheduler (virtual) cache block IDs to this rank's local pages, on the CPU.

The scheduler addresses every cache group by virtual block ID; a group with
``shard_count`` D assigns those IDs cyclically to D owners, and a replicated
group (D == 1) owns every ID itself. The same translation serves both, so the
zeroing path never asks which case it is in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from tokenspeed_kernel.ops.kvcache.triton_virtual_blocks import virtual_slots_to_local

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
    require_positive_int,
)


def local_pages(
    virtual_blocks: Sequence[int],
    *,
    shard_count: int,
    rank: int,
    virtual_block_count: int,
) -> list[int]:
    """Translate a batch of scheduler blocks to owned local pages on the CPU.

    Args:
        virtual_blocks: Scheduler block IDs, including reserved null ID 0.
        shard_count: Cyclic owner count from the group's spec; 1 is replicated.
        rank: This process's rank in the DCP subgroup.
        virtual_block_count: Exclusive bound from the arena's runtime contract.

    Returns:
        Owned local page IDs in input order, preserving duplicates and
        excluding null and remote blocks.

    Raises:
        IndexError: If any virtual block ID is outside the contract's bounds.
        ValueError: If the shard count or rank is invalid.
    """
    require_positive_int("shard_count", shard_count)
    if rank < 0 or (shard_count > 1 and rank >= shard_count):
        raise ValueError("DCP rank is out of range")
    blocks = torch.tensor(virtual_blocks, dtype=torch.int64, device="cpu")
    if ((blocks < 0) | (blocks >= virtual_block_count)).any():
        raise IndexError("virtual cache block ID is out of range")
    local, owned = virtual_slots_to_local(
        blocks,
        rows_per_page=1,
        virtual_block_count=virtual_block_count,
        degree=shard_count,
        rank=rank if shard_count > 1 else 0,
    )
    return local[owned].tolist()


def local_pages_by_group(
    virtual_blocks_by_group: Mapping[str, Sequence[int]],
    *,
    contract: CacheRuntimeContract,
    rank: int,
) -> dict[str, list[int]]:
    """Translate every group's scheduler blocks through :func:`local_pages`.

    Args:
        virtual_blocks_by_group: Scheduler block IDs keyed by cache group id.
        contract: The bound arena's runtime contract; supplies each group's
            shard count and virtual block bound.
        rank: This process's rank in the DCP subgroup.

    Returns:
        Owned local page IDs keyed by the same group ids.
    """
    shard_counts = {spec.group_id: spec.shard_count for spec in contract.group_specs}
    counts = contract.virtual_block_counts
    return {
        group_id: local_pages(
            block_ids,
            shard_count=shard_counts[group_id],
            rank=rank,
            virtual_block_count=counts[group_id],
        )
        for group_id, block_ids in virtual_blocks_by_group.items()
    }
