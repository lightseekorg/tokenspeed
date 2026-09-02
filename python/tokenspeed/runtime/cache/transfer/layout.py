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

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheMemoryPlan,
    )


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def select_layer_fields(
    fields: tuple[object, ...],
    *,
    first_layer: int,
    num_layers: int,
) -> tuple[frozenset[str], tuple[tuple[str, ...], ...]]:
    """Select one compute view's fields and map global to local layer IDs.

    Args:
        fields: Fields from the merged target/draft memory plan.
        first_layer: First global layer owned by this compute view.
        num_layers: Number of local layer consumers in this view.

    Returns:
        Selected field IDs and one field tuple per local layer.
    """
    if first_layer < 0 or num_layers < 0:
        raise ValueError("cache layer view bounds must be non-negative")
    last_layer = first_layer + num_layers
    selected = set()
    consumers = [[] for _ in range(num_layers)]
    for field in fields:
        layer_id = cache_field_layer_id(field.field_id)
        if first_layer <= layer_id < last_layer:
            selected.add(field.field_id)
            consumers[layer_id - first_layer].append(field.field_id)
    return frozenset(selected), tuple(tuple(consumer) for consumer in consumers)


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
    field_ids: frozenset[str] | None = None,
) -> CacheTransferLayout:
    """Derive transfer rows for selected fields from an LCM arena plan.

    Args:
        plan: Shared physical cache geometry.
        buffer: Device buffer that stores the plan.
        consumers: Selected field IDs grouped by local layer.
        group_ids: Optional scheduler order for the selected groups.
        field_ids: Optional field subset owned by this compute view.

    Returns:
        A transfer layout over the selected view.
    """

    planes = {plane.plane_id: plane for plane in plan.planes}
    known_field_ids = {field.field_id for field in plan.fields}
    if field_ids is None:
        field_ids = frozenset(known_field_ids)
    else:
        unknown_fields = field_ids - known_field_ids
        if unknown_fields:
            raise ValueError(
                f"selected cache fields are absent from the plan: {sorted(unknown_fields)}"
            )
    fields_by_group = {
        group.group_id: tuple(
            field
            for field in plan.fields
            if field.group_id == group.group_id and field.field_id in field_ids
        )
        for group in plan.groups
    }
    groups_by_id = {group.group_id: group for group in plan.groups}
    selected_group_ids = {
        group_id for group_id, fields in fields_by_group.items() if fields
    }
    if group_ids is None:
        ordered_groups = tuple(
            group for group in plan.groups if group.group_id in selected_group_ids
        )
    else:
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("scheduler cache groups contain a duplicate group")
        if set(group_ids) != selected_group_ids:
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
    *,
    group_ids: tuple[str, ...] | None = None,
) -> CacheTransferLayout:
    """Combine target and draft views that share scheduler CacheBlock IDs.

    Args:
        target: Target model's local transfer view.
        draft: Optional draft model view into the target's arena.
        group_ids: Optional scheduler order for the merged groups.

    Returns:
        One transfer layout containing every target and draft field.
    """

    if draft is None:
        return target
    if draft.num_lcm_blocks != target.num_lcm_blocks:
        raise ValueError("target and draft cache layouts use different geometry")
    if (
        target.groups == draft.groups
        and target.consumers == draft.consumers
        and len(target.buffers) == len(draft.buffers)
        and all(
            target_buffer is draft_buffer
            for target_buffer, draft_buffer in zip(target.buffers, draft.buffers)
        )
    ):
        return target

    target_groups = {group.group_id: group for group in target.groups}
    draft_groups = {group.group_id: group for group in draft.groups}
    for group_id in set(target_groups) & set(draft_groups):
        target_group = target_groups[group_id]
        draft_group = draft_groups[group_id]
        if (
            draft_group.cache_blocks_per_lcm_block
            != target_group.cache_blocks_per_lcm_block
        ):
            raise ValueError(
                f"target and draft cache group {group_id!r} use different geometry"
            )

    all_group_ids = set(target_groups) | set(draft_groups)
    if group_ids is None:
        ordered_group_ids = tuple(target_groups) + tuple(
            group_id for group_id in draft_groups if group_id not in target_groups
        )
    else:
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("scheduler cache groups contain a duplicate group")
        if set(group_ids) != all_group_ids:
            raise ValueError("scheduler and transfer cache groups do not match")
        ordered_group_ids = group_ids

    shared_arena = (
        len(target.buffers) == 1
        and len(draft.buffers) == 1
        and target.buffers[0] is draft.buffers[0]
    )
    if not shared_arena:
        raise ValueError("target and draft cache views must share one arena")
    target_fields = {
        field.field_id for group in target.groups for field in group.fields
    }
    draft_fields = {field.field_id for group in draft.groups for field in group.fields}
    aliased_fields = target_fields & draft_fields
    if aliased_fields:
        # A strict draft subset is one physical layout with two compute views.
        if not draft_fields < target_fields:
            raise ValueError(
                "shared target/draft arena views have ambiguous overlapping "
                f"fields {sorted(aliased_fields)}"
            )

        target_locations = {
            field.field_id: (group.group_id, field)
            for group in target.groups
            for field in group.fields
        }
        conflicts = sorted(
            field.field_id
            for group in draft.groups
            for field in group.fields
            if target_locations[field.field_id] != (group.group_id, field)
        )
        if conflicts:
            raise ValueError(
                "shared target/draft arena fields use conflicting layouts "
                f"{conflicts}"
            )
    if any(
        field.device_buffer_index != 0
        for layout in (target, draft)
        for group in layout.groups
        for field in group.fields
    ):
        raise ValueError("shared arena fields must use device buffer zero")
    groups = []
    for group_id in ordered_group_ids:
        target_group = target_groups.get(group_id)
        draft_group = draft_groups.get(group_id)
        fields = ()
        if target_group is not None:
            fields += target_group.fields
        if draft_group is not None:
            fields += tuple(
                field
                for field in draft_group.fields
                if field.field_id not in aliased_fields
            )
        geometry = target_group if target_group is not None else draft_group
        groups.append(
            CacheGroupLayout(
                group_id=group_id,
                cache_blocks_per_lcm_block=geometry.cache_blocks_per_lcm_block,
                fields=fields,
            )
        )
    draft_consumers = tuple(
        tuple(field_id for field_id in consumer if field_id not in aliased_fields)
        for consumer in draft.consumers
    )
    return CacheTransferLayout(
        num_lcm_blocks=target.num_lcm_blocks,
        groups=tuple(groups),
        buffers=target.buffers,
        consumers=target.consumers + draft_consumers,
    )
