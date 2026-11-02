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

"""Where a rank's Device cache blocks live, and how a pool slot holds one.

A pool slot holds one scheduler CacheBlock as a Host L2 block does: its
fields' payloads in field order, at ``slot × (LCM bytes / packing)`` inside its
LCM block. The scheduler's view of each group travels to the owner, which
builds the same cache index the rank's scheduler uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FieldLayout:
    field_id: str
    address: int
    page_stride_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class GroupLayout:
    group_id: str
    packing: int
    fields: tuple[FieldLayout, ...]
    # The scheduler's view: its AttnKind name, the tokens one block covers
    # (0 for State), its window and whether replay rebuilds it.
    attn_kind: str
    block_granularity: int
    sliding_window_tokens: int | None
    replayable: bool

    @property
    def block_bytes(self) -> int:
        return sum(field.payload_bytes for field in self.fields)


@dataclass(frozen=True, slots=True)
class SharedL2Layout:
    """Startup-only Device cache facts, in the scheduler's group order."""

    groups: tuple[GroupLayout, ...]
    registration_regions: tuple[tuple[int, int], ...]
    prefix_granularity: int

    @property
    def lcm_bytes(self) -> int:
        return max(group.packing * group.block_bytes for group in self.groups)

    def slot_stride(self, group: int) -> int:
        return self.lcm_bytes // self.groups[group].packing

    def block_segments(self, group: int, block_id: int) -> tuple[tuple[int, int], ...]:
        """``(address, bytes)`` of every field of one Device CacheBlock."""
        if block_id <= 0:
            raise IndexError(f"block {block_id} is the null or an invalid block")
        return tuple(
            (field.address + block_id * field.page_stride_bytes, field.payload_bytes)
            for field in self.groups[group].fields
        )

    def geometry(self) -> dict[str, Any]:
        """What the owner checks every client against and builds its pool from."""
        return {
            "prefix_granularity": self.prefix_granularity,
            "lcm_bytes": self.lcm_bytes,
            "groups": [
                {
                    "group_id": group.group_id,
                    "attn_kind": group.attn_kind,
                    "block_granularity": group.block_granularity,
                    "cache_blocks_per_lcm_block": group.packing,
                    "sliding_window_tokens": group.sliding_window_tokens,
                    "replayable": group.replayable,
                }
                for group in self.groups
            ],
        }


def build_shared_l2_layout(arena: Any) -> SharedL2Layout:
    """Read the layout from a CacheArena; holding it keeps nothing alive."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
        cache_group_config,
    )

    if arena.offload is not None:
        raise ValueError("shared L2 does not support KV offloading")
    contract = arena.runtime_contract
    groups = []
    for spec in contract.group_specs:
        if spec.shard_count != 1:
            raise ValueError("shared L2 supports replicated cache groups only")
        packing = int(contract.virtual_packing[spec.group_id])
        scheduled = cache_group_config(
            spec, total_pages=0, cache_blocks_per_lcm_block=packing
        )
        fields = sorted(
            (field for field in arena.plan.fields if field.group_id == spec.group_id),
            key=lambda field: field.field_id,
        )
        groups.append(
            GroupLayout(
                group_id=spec.group_id,
                packing=packing,
                fields=tuple(
                    FieldLayout(
                        field_id=str(field.field_id),
                        address=int(arena.field_address(field.field_id)),
                        page_stride_bytes=int(field.page_stride_bytes),
                        payload_bytes=int(field.payload_bytes),
                    )
                    for field in fields
                ),
                attn_kind=scheduled.attn_kind.name,
                block_granularity=int(scheduled.block_granularity),
                sliding_window_tokens=scheduled.sliding_window_tokens,
                replayable=bool(scheduled.replayable),
            )
        )
    return SharedL2Layout(
        groups=tuple(groups),
        registration_regions=tuple(
            (int(address), int(size)) for address, size in arena.registration_regions()
        ),
        prefix_granularity=int(contract.prefix_granularity),
    )
