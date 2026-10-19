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

"""One owner translation for both ends of a Host copy.

The scheduler addresses the Device arena and every Host pool by virtual block
id. A group with ``shard_count`` D deals its virtual blocks cyclically to D
owners -- block ``v > 0`` belongs to rank ``(v - 1) % D`` -- and the allocator
places a Host block in the same residue class as the Device block it mirrors,
so one rank owns both ends of every copy it performs. This module applies the
same ``owned_local_pages`` placement the zeroing and PD paths use to a copy's
Device end and Host end at once: the rows this rank owns, in local ids, and
an error if the two ends disagree. The L2 prefix tier and the retraction
snapshot pool each hold one translation (same Device bound, their own Host
bound); a replicated group (D == 1) translates to the identity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tokenspeed.runtime.layers.attention.kv_cache.virtual_blocks import (
    owned_local_pages,
)


class BlockOwnerTranslation:
    """Scheduler block ids of a copy's two ends to this rank's local ids."""

    def __init__(
        self,
        *,
        shard_counts: Sequence[int],
        device_virtual_counts: Sequence[int],
        host_virtual_counts: Sequence[int],
        rank: int,
    ) -> None:
        """
        Args:
            shard_counts: Each layout group's cyclic owner count; 1 is
                replicated.
            device_virtual_counts: Each group's Device virtual block bound
                (the arena contract's ``virtual_block_counts``).
            host_virtual_counts: Each group's bound in the Host pool this
                translation addresses: ``1 + host_lcm_blocks * packing *
                shard_count``.
            rank: This rank in the owner subgroup.
        """
        if not (
            len(shard_counts) == len(device_virtual_counts) == len(host_virtual_counts)
        ):
            raise ValueError("owner translation tables must cover the same groups")
        self.shard_counts = tuple(int(count) for count in shard_counts)
        self.device_virtual_counts = tuple(
            int(count) for count in device_virtual_counts
        )
        self.host_virtual_counts = tuple(int(count) for count in host_virtual_counts)
        self.rank = int(rank)

    @classmethod
    def for_host_pool(
        cls, layout, contract, *, num_host_lcm_blocks: int, rank: int
    ) -> BlockOwnerTranslation:
        """The translation between the arena and a Host pool of ``num_host_lcm_blocks``.

        Args:
            layout: The executor's ``CacheTransferLayout``; its group order is
                the translation's.
            contract: The arena's ``CacheRuntimeContract`` (shard counts and
                Device bounds).
            num_host_lcm_blocks: The Host pool's LCM blocks, null excluded.
            rank: This rank in the owner subgroup.
        """
        shard_by_id = {
            spec.group_id: int(spec.shard_count) for spec in contract.group_specs
        }
        device_counts = contract.virtual_block_counts
        shard_counts = [shard_by_id[group.group_id] for group in layout.groups]
        return cls(
            shard_counts=shard_counts,
            device_virtual_counts=[
                device_counts[group.group_id] for group in layout.groups
            ],
            host_virtual_counts=[
                1 + num_host_lcm_blocks * group.cache_blocks_per_lcm_block * shard
                for group, shard in zip(layout.groups, shard_counts)
            ],
            rank=rank,
        )

    @property
    def num_groups(self) -> int:
        return len(self.shard_counts)

    def owned_positions(
        self, rows: Sequence[tuple[int, int, int]]
    ) -> list[tuple[int, tuple[int, int, int]]]:
        """This rank's rows of ``(group_index, device_block, host_block)`` as
        ``(position, local_row)`` pairs, ``position`` indexing ``rows``.

        Groups rows by group, translates each end through
        ``owned_local_pages`` against its own bound, and requires both ends
        to agree on the owner. Input order is kept within a group; groups
        come out in index order (the transfer workspace buckets by group
        anyway). The position lets a caller carry per-row side data -- the
        L3 key of an L2 row -- for exactly the rows this rank copies.

        Raises:
            IndexError: A group index or block id is out of range.
            ValueError: A null block, or a pair owned by two different ranks.
        """
        positions: list[list[int]] = [[] for _ in self.shard_counts]
        device_ids: list[list[int]] = [[] for _ in self.shard_counts]
        host_ids: list[list[int]] = [[] for _ in self.shard_counts]
        for position, (group, device_block, host_block) in enumerate(rows):
            group = int(group)
            if not 0 <= group < self.num_groups:
                raise IndexError(f"cache transfer names unknown group {group}")
            device_block = int(device_block)
            host_block = int(host_block)
            if device_block <= 0 or host_block <= 0:
                raise ValueError("a cache transfer cannot name the null block 0")
            positions[group].append(position)
            device_ids[group].append(device_block)
            host_ids[group].append(host_block)
        owned: list[tuple[int, tuple[int, int, int]]] = []
        for group, shard in enumerate(self.shard_counts):
            if not device_ids[group]:
                continue
            device_owned, device_local = owned_local_pages(
                device_ids[group],
                shard_count=shard,
                rank=self.rank,
                virtual_block_count=self.device_virtual_counts[group],
            )
            host_owned, host_local = owned_local_pages(
                host_ids[group],
                shard_count=shard,
                rank=self.rank,
                virtual_block_count=self.host_virtual_counts[group],
            )
            if not np.array_equal(device_owned, host_owned):
                raise ValueError(
                    f"cache transfer of group {group} pairs blocks owned by "
                    "different ranks; a Host block must sit in its Device "
                    "block's residue class"
                )
            owned_positions = [
                position
                for position, is_owned in zip(positions[group], device_owned.tolist())
                if is_owned
            ]
            owned.extend(
                (position, (group, int(device_block), int(host_block)))
                for position, device_block, host_block in zip(
                    owned_positions, device_local.tolist(), host_local.tolist()
                )
            )
        return owned

    def owned_host_positions(
        self, rows: Sequence[tuple[int, int]]
    ) -> list[tuple[int, tuple[int, int]]]:
        """This rank's rows of ``(group_index, host_block)`` as ``(position,
        (group_index, local_host_block))`` pairs.

        The Host-only form of :meth:`owned_positions`, for a copy with no
        Device end: an L3 prefetch fills Host pages before any Device page
        exists for the request. Same ownership rule, same local ids.

        Raises:
            IndexError: A group index or block id is out of range.
            ValueError: A null block.
        """
        positions: list[list[int]] = [[] for _ in self.shard_counts]
        host_ids: list[list[int]] = [[] for _ in self.shard_counts]
        for position, (group, host_block) in enumerate(rows):
            group = int(group)
            if not 0 <= group < self.num_groups:
                raise IndexError(f"cache transfer names unknown group {group}")
            host_block = int(host_block)
            if host_block <= 0:
                raise ValueError("a cache transfer cannot name the null block 0")
            positions[group].append(position)
            host_ids[group].append(host_block)
        owned: list[tuple[int, tuple[int, int]]] = []
        for group, shard in enumerate(self.shard_counts):
            if not host_ids[group]:
                continue
            host_owned, host_local = owned_local_pages(
                host_ids[group],
                shard_count=shard,
                rank=self.rank,
                virtual_block_count=self.host_virtual_counts[group],
            )
            # ``host_local`` lists the owned pages only, in input order.
            owned_positions = [
                position
                for position, is_owned in zip(positions[group], host_owned.tolist())
                if is_owned
            ]
            owned.extend(
                (position, (group, int(local)))
                for position, local in zip(owned_positions, host_local.tolist())
            )
        owned.sort(key=lambda item: item[0])
        return owned
