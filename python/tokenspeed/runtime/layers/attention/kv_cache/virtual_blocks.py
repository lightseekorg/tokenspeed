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

import numpy as np

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
    require_positive_int,
)


def owned_local_pages(
    virtual_blocks: Sequence[int] | np.ndarray,
    *,
    shard_count: int,
    rank: int,
    virtual_block_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate scheduler blocks to owned local pages, keeping the owner mask.

    Args:
        virtual_blocks: Scheduler block IDs, including reserved null ID 0; a
            sequence or an integer array (the scheduler's zero-copy export).
        shard_count: Cyclic owner count from the group's spec; 1 is replicated.
        rank: This process's rank in the KVP subgroup.
        virtual_block_count: Exclusive bound from the arena's runtime contract.

    Returns:
        ``(owned, local)``: a boolean mask over the input marking the blocks
        this rank owns (the null block is never owned), and the local page ID
        of every owned block as an int64 array in input order, preserving
        duplicates. The mask lets a caller that pairs each source block with
        a destination (the PD sender) keep the matching destination
        subsequence. No Python int is built per page: a long prompt's
        admission is thousands of them.

    Raises:
        IndexError: If any virtual block ID is outside the contract's bounds.
        ValueError: If the shard count or rank is invalid.
    """
    require_positive_int("shard_count", shard_count)
    if rank < 0 or (shard_count > 1 and rank >= shard_count):
        raise ValueError("KVP rank is out of range")
    blocks = np.asarray(virtual_blocks, dtype=np.int64).reshape(-1)
    if blocks.size == 0:
        return np.zeros(0, dtype=bool), blocks
    if blocks.min() < 0 or blocks.max() >= virtual_block_count:
        raise IndexError("virtual cache block ID is out of range")
    # Same placement as the device kernels' virtual_block_to_local: block 0 is
    # the null page, the rest are dealt cyclically to the shard_count owners.
    positive = blocks - 1
    owned = blocks > 0
    if shard_count > 1:
        owned &= positive % shard_count == rank
    local = positive // shard_count + 1
    return owned, local[owned]


def local_pages(
    virtual_blocks: Sequence[int] | np.ndarray,
    *,
    shard_count: int,
    rank: int,
    virtual_block_count: int,
) -> np.ndarray:
    """Translate a batch of scheduler blocks to owned local pages on the CPU.

    The same translation as :func:`owned_local_pages` without the mask, for
    callers such as zeroing that only need the pages this rank holds.

    Returns:
        Owned local page IDs as an int64 array in input order, preserving
        duplicates and excluding null and remote blocks.
    """
    _, local = owned_local_pages(
        virtual_blocks,
        shard_count=shard_count,
        rank=rank,
        virtual_block_count=virtual_block_count,
    )
    return local


def local_pages_by_group(
    virtual_blocks_by_group: Mapping[str, Sequence[int] | np.ndarray],
    *,
    contract: CacheRuntimeContract,
    rank: int,
) -> dict[str, np.ndarray]:
    """Translate every group's scheduler blocks through :func:`local_pages`.

    Args:
        virtual_blocks_by_group: Scheduler block IDs keyed by cache group id.
        contract: The bound arena's runtime contract; supplies each group's
            shard count and virtual block bound.
        rank: This process's rank in the KVP subgroup.

    Returns:
        Owned local page ID arrays keyed by the same group ids.
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
