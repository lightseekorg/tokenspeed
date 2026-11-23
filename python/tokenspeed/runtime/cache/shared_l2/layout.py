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

"""Where a rank's Device cache blocks live, and how a pool page holds one.

A pool page holds one scheduler CacheBlock as a Host L2 block does: the pool
is laid out as ``HostCacheStorage`` lays out a rank's Host tier (one parent
size, each group packed into it, a block's fields back to back). Only the
parents are spread over the providers. The scheduler's view of each group
travels to the owner as the rank's geometry, so the owner builds the cache
index the rank's scheduler uses.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.cache.host_storage import _compute_host_layout
from tokenspeed.runtime.cache.shared_l2.client import SharedL2PoolView
from tokenspeed.runtime.cache.transfer.layout import (
    CacheTransferLayout,
    combine_cache_transfer_layouts,
)


@dataclass(frozen=True, slots=True)
class SharedL2Layout:
    """Startup-only facts, in the scheduler's group order.

    Attributes:
        transfer: The rank's Device cache fields and buffers.
        view_layers: The layers of each cache pool view, the target's and
            then the draft's when it has its own: together the flat layers of
            ``transfer.consumers``.
        block_bytes: Bytes of one pool page per group.
        field_offsets: Per group, each field's offset inside a pool page.
        lcm_bytes: Bytes of one pool LCM block.
        host_packing: Pool pages per LCM block, per group.
        geometry: What the owner builds its pool from and checks every rank
            against (``owner``'s module docstring).
    """

    transfer: CacheTransferLayout
    view_layers: tuple[int, ...]
    block_bytes: tuple[int, ...]
    field_offsets: tuple[tuple[int, ...], ...]
    lcm_bytes: int
    host_packing: tuple[int, ...]
    geometry: dict[str, Any]

    @classmethod
    def build(
        cls,
        transfer: CacheTransferLayout,
        *,
        view_layers: tuple[int, ...],
        group_specs: Sequence[Any],
        cache_groups: Sequence[Any],
        prefix_granularity: int,
    ) -> SharedL2Layout:
        """
        Args:
            transfer: The rank's transfer layout, target and draft fields,
                groups in the scheduler's order.
            view_layers: The layers of each cache pool view.
            group_specs: The arena's ``CacheGroupSpec`` of each group.
            cache_groups: The scheduler's ``CacheGroupConfig`` of each group
                (``pool_to_cache_groups``).
            prefix_granularity: The scheduler's prefix page, in tokens.
        """
        if any(spec.shard_count != 1 for spec in group_specs):
            raise ValueError("a shared L2 pool holds replicated cache groups only")
        block_bytes, field_offsets, lcm_bytes, host_packing = _compute_host_layout(
            transfer, group_specs=group_specs
        )
        geometry = {
            "prefix_granularity": int(prefix_granularity),
            "lcm_bytes": lcm_bytes,
            "groups": [
                {
                    "group_id": config.group_id,
                    "attn_kind": config.attn_kind.name,
                    "block_granularity": config.block_granularity,
                    "cache_blocks_per_lcm_block": config.cache_blocks_per_lcm_block,
                    "host_packing": packing,
                    "shard_count": config.shard_count,
                    "sliding_window_tokens": config.sliding_window_tokens,
                    "sink_tokens": config.sink_tokens,
                    "replayable": config.replayable,
                    "checkpoint_granularity": config.checkpoint_granularity,
                }
                for config, packing in zip(cache_groups, host_packing, strict=True)
            ],
        }
        return cls(
            transfer=transfer,
            view_layers=view_layers,
            block_bytes=block_bytes,
            field_offsets=field_offsets,
            lcm_bytes=lcm_bytes,
            host_packing=host_packing,
            geometry=geometry,
        )

    def registration_regions(self) -> tuple[tuple[int, int], ...]:
        """``(address, bytes)`` of every Device buffer a transfer touches."""
        return tuple(
            (int(buffer.data_ptr()), int(buffer.nbytes))
            for buffer in self.transfer.buffers
        )

    def device_fields(self, group: int) -> tuple[tuple[int, int, int], ...]:
        """Every field of a group's Device CacheBlocks, in the order a pool
        page holds them: ``(address of block zero, bytes from one block to
        the next, payload bytes)``."""
        return tuple(
            (
                int(self.transfer.buffers[field.device_buffer_index].data_ptr())
                + field.device_block_zero_offset_bytes,
                field.block_stride_bytes,
                field.payload_bytes,
            )
            for field in self.transfer.groups[group].fields
        )

    def page_address(
        self, pool: SharedL2PoolView, group: int, bucket: int, page: int
    ) -> tuple[str, int]:
        """The provider session and the remote address of a group's pool page.

        A page lies in its LCM block as ``HostCacheStorage.host_block_offset``
        places a Host block; the LCM block lies on the provider that holds it.
        """
        if bucket != 0:
            raise ValueError("a shared L2 pool holds bucket 0 only")
        lcm, slot = divmod(page - 1, self.host_packing[group])
        starts = [provider.start for provider in pool.providers]
        provider = pool.providers[max(bisect.bisect_right(starts, lcm) - 1, 0)]
        if page <= 0 or not 0 <= lcm - provider.start < provider.lcm_blocks:
            raise IndexError(f"shared L2 page {page} of group {group} is out of range")
        return provider.session, (
            provider.base
            + (lcm - provider.start) * self.lcm_bytes
            + slot * self.block_bytes[group]
        )


def build_shared_l2_layout(
    device_pool: Any,
    draft_pool: Any | None,
    *,
    cache_groups: Sequence[Any],
    prefix_granularity: int,
) -> SharedL2Layout:
    """Read the layout from the cache pools, as the Host cache executor reads
    its own.

    Args:
        device_pool: The target cache pool; its arena publishes the transfer
            layout and the groups' specs.
        draft_pool: The draft pool sharing the arena, or None.
        cache_groups: The scheduler's ``CacheGroupConfig`` of each group
            (``pool_to_cache_groups`` of the target pool).
        prefix_granularity: The scheduler's prefix page, in tokens.
    """
    contract = device_pool.arena.runtime_contract
    target = device_pool.cache_transfer_layout()
    draft = draft_pool.cache_transfer_layout() if draft_pool is not None else None
    combined = combine_cache_transfer_layouts(
        target,
        draft,
        group_ids=tuple(spec.group_id for spec in contract.group_specs) or None,
    )
    return SharedL2Layout.build(
        combined,
        # A draft whose view is the target's has no layers of its own.
        view_layers=(
            (len(target.consumers),)
            if combined is target
            else (len(target.consumers), len(draft.consumers))
        ),
        group_specs=contract.group_specs,
        cache_groups=cache_groups,
        prefix_granularity=prefix_granularity,
    )
