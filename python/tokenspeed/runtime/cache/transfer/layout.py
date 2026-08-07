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

"""Model-neutral byte geometry for moving CacheBlocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.plan import CacheMemoryPlan


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class CacheField:
    """One cache field stored as block rows in a device buffer."""

    field_id: str
    device_buffer_index: int
    device_block_zero_offset_bytes: int
    block_stride_bytes: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("field_id must be non-empty")
        if self.device_buffer_index < 0:
            raise ValueError("device_buffer_index must be non-negative")
        if self.device_block_zero_offset_bytes < 0:
            raise ValueError("device_block_zero_offset_bytes must be non-negative")
        _positive("block_stride_bytes", self.block_stride_bytes)
        _positive("payload_bytes", self.payload_bytes)
        if self.payload_bytes > self.block_stride_bytes:
            raise ValueError("payload_bytes cannot exceed block_stride_bytes")


@dataclass(frozen=True, slots=True)
class CacheGroupLayout:
    """Cache fields and two-level packing for one scheduler cache group."""

    group_id: str
    cache_blocks_per_lcm_block: int
    fields: tuple[CacheField, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        _positive("cache_blocks_per_lcm_block", self.cache_blocks_per_lcm_block)
        if not self.fields:
            raise ValueError("cache group must contain at least one field")
        field_ids = tuple(field.field_id for field in self.fields)
        if len(field_ids) != len(set(field_ids)):
            raise ValueError(f"group {self.group_id!r} contains a duplicate field")


@dataclass(frozen=True, slots=True)
class CacheTransferLayout:
    """Complete local contract for cache transfer and layer consumption."""

    num_lcm_blocks: int
    groups: tuple[CacheGroupLayout, ...]
    buffers: tuple[object, ...]
    consumers: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _positive("num_lcm_blocks", self.num_lcm_blocks)
        if not self.groups:
            raise ValueError("layout must contain at least one cache group")
        if not self.buffers:
            raise ValueError("layout must contain at least one device buffer")

        group_ids = tuple(group.group_id for group in self.groups)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("layout contains a duplicate group")

        fields = tuple(field for group in self.groups for field in group.fields)
        field_ids = tuple(field.field_id for field in fields)
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("layout contains a duplicate field")
        if any(field.device_buffer_index >= len(self.buffers) for field in fields):
            raise ValueError("field device_buffer_index is outside the buffer tuple")

        known_fields = set(field_ids)
        consumed_fields = []
        for consumer in self.consumers:
            if len(consumer) != len(set(consumer)):
                raise ValueError("consumer contains a duplicate field")
            unknown = set(consumer) - known_fields
            if unknown:
                raise ValueError(f"consumer references unknown field {sorted(unknown)}")
            consumed_fields.extend(consumer)
        if len(consumed_fields) != len(set(consumed_fields)):
            raise ValueError("a cache field cannot belong to multiple consumers")
        missing = known_fields - set(consumed_fields)
        if missing:
            raise ValueError(f"cache fields have no consumer {sorted(missing)}")


def layout_from_lcm_plan(
    plan: CacheMemoryPlan,
    buffer: object,
    *,
    consumers: tuple[tuple[str, ...], ...],
    group_ids: tuple[str, ...] | None = None,
) -> CacheTransferLayout:
    """Derive transfer rows directly from an LCM arena plan."""

    planes = {plane.plane_id: plane for plane in plan.planes}
    fields_by_group = {
        group.group_id: tuple(
            field for field in plan.fields if field.group_id == group.group_id
        )
        for group in plan.groups
    }
    groups_by_id = {group.group_id: group for group in plan.groups}
    if group_ids is None:
        ordered_groups = plan.groups
    else:
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("scheduler cache groups contain a duplicate group")
        if set(group_ids) != set(groups_by_id):
            raise ValueError("scheduler and transfer cache groups do not match")
        ordered_groups = tuple(groups_by_id[group_id] for group_id in group_ids)
    groups = []
    for group in ordered_groups:
        fields = []
        for field in fields_by_group[group.group_id]:
            plane = planes[field.plane_id]
            fields.append(
                CacheField(
                    field_id=field.field_id,
                    device_buffer_index=0,
                    device_block_zero_offset_bytes=(
                        plane.arena_offset_bytes
                        + plane.bytes_per_lcm_block
                        - field.page_stride_bytes
                        + field.field_offset_bytes
                    ),
                    block_stride_bytes=field.page_stride_bytes,
                    payload_bytes=field.payload_bytes,
                )
            )
        groups.append(
            CacheGroupLayout(
                group_id=group.group_id,
                cache_blocks_per_lcm_block=group.cache_blocks_per_lcm_block,
                fields=tuple(fields),
            )
        )
    return CacheTransferLayout(
        num_lcm_blocks=plan.num_lcm_blocks,
        groups=tuple(groups),
        buffers=(buffer,),
        consumers=consumers,
    )


def combine_cache_transfer_layouts(
    target: CacheTransferLayout,
    draft: CacheTransferLayout | None,
) -> CacheTransferLayout:
    """Combine target and draft arenas that share scheduler CacheBlock IDs."""

    if draft is None:
        return target
    if draft.num_lcm_blocks != target.num_lcm_blocks:
        raise ValueError("target and draft cache layouts use different geometry")

    target_groups = {group.group_id: group for group in target.groups}
    draft_groups = {group.group_id: group for group in draft.groups}
    unknown_groups = set(draft_groups) - set(target_groups)
    if unknown_groups:
        raise ValueError(
            f"draft cache groups are absent from target: {sorted(unknown_groups)}"
        )
    for group_id, draft_group in draft_groups.items():
        target_group = target_groups[group_id]
        if (
            draft_group.cache_blocks_per_lcm_block
            != target_group.cache_blocks_per_lcm_block
        ):
            raise ValueError(
                f"target and draft cache group {group_id!r} use different geometry"
            )

    draft_buffer_base = len(target.buffers)

    def namespaced_field(
        field: CacheField, namespace: str, buffer_base: int
    ) -> CacheField:
        return CacheField(
            field_id=f"{namespace}:{field.field_id}",
            device_buffer_index=buffer_base + field.device_buffer_index,
            device_block_zero_offset_bytes=field.device_block_zero_offset_bytes,
            block_stride_bytes=field.block_stride_bytes,
            payload_bytes=field.payload_bytes,
        )

    groups = []
    for target_group in target.groups:
        fields = tuple(
            namespaced_field(field, "target", 0) for field in target_group.fields
        )
        if draft_group := draft_groups.get(target_group.group_id):
            fields += tuple(
                namespaced_field(field, "draft", draft_buffer_base)
                for field in draft_group.fields
            )
        groups.append(
            CacheGroupLayout(
                group_id=target_group.group_id,
                cache_blocks_per_lcm_block=target_group.cache_blocks_per_lcm_block,
                fields=fields,
            )
        )

    consumers = tuple(
        tuple(f"target:{field_id}" for field_id in consumer)
        for consumer in target.consumers
    ) + tuple(
        tuple(f"draft:{field_id}" for field_id in consumer)
        for consumer in draft.consumers
    )
    return CacheTransferLayout(
        num_lcm_blocks=target.num_lcm_blocks,
        groups=tuple(groups),
        buffers=target.buffers + draft.buffers,
        consumers=consumers,
    )
