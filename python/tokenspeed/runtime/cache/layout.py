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

"""Model-neutral byte geometry for moving paged cache entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.lcm_memory_plan import LcmMemoryPlan


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True, slots=True)
class CacheSegment:
    """One payload row inside a device cache buffer."""

    segment_id: str
    buffer_index: int
    page_zero_offset: int
    page_stride_bytes: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must be non-empty")
        if self.buffer_index < 0:
            raise ValueError("buffer_index must be non-negative")
        if self.page_zero_offset < 0:
            raise ValueError("page_zero_offset must be non-negative")
        _positive("page_stride_bytes", self.page_stride_bytes)
        _positive("payload_bytes", self.payload_bytes)
        if self.payload_bytes > self.page_stride_bytes:
            raise ValueError("payload_bytes cannot exceed page_stride_bytes")


@dataclass(frozen=True, slots=True)
class CacheGroupLayout:
    """Transfer rows and two-level packing for one scheduler cache group."""

    group_id: str
    cache_blocks_per_lcm_block: int
    page_count: int
    segments: tuple[CacheSegment, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        packing = _positive(
            "cache_blocks_per_lcm_block", self.cache_blocks_per_lcm_block
        )
        pages = _positive("page_count", self.page_count)
        if pages <= 1 or (pages - 1) % packing != 0:
            raise ValueError(
                "page_count must contain page 0 plus complete LCM parents"
            )
        if not self.segments:
            raise ValueError("cache group must contain at least one segment")
        segment_ids = tuple(segment.segment_id for segment in self.segments)
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError(f"group {self.group_id!r} contains a duplicate segment")

    @property
    def num_lcm_blocks(self) -> int:
        return (self.page_count - 1) // self.cache_blocks_per_lcm_block


@dataclass(frozen=True, slots=True)
class PackedCacheLayout:
    """Compact Host offsets derived from a CacheTransferLayout."""

    child_bytes: tuple[int, ...]
    segment_offsets: tuple[tuple[int, ...], ...]
    parent_bytes: int


@dataclass(frozen=True, slots=True)
class CacheTransferLayout:
    """Complete local contract for cache transfer and layer consumption."""

    logical_block_tokens: int
    groups: tuple[CacheGroupLayout, ...]
    buffers: tuple[object, ...]
    consumers: tuple[tuple[str, ...], ...]
    lcm_block_count: int | None = None

    def __post_init__(self) -> None:
        _positive("logical_block_tokens", self.logical_block_tokens)
        if not self.groups:
            raise ValueError("layout must contain at least one cache group")
        if not self.buffers:
            raise ValueError("layout must contain at least one device buffer")

        group_ids = tuple(group.group_id for group in self.groups)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("layout contains a duplicate group")
        parent_counts = {group.num_lcm_blocks for group in self.groups}
        if len(parent_counts) != 1:
            raise ValueError("cache groups must describe the same LCM parent count")
        if self.lcm_block_count is not None:
            configured_count = _positive("lcm_block_count", self.lcm_block_count)
            if parent_counts != {configured_count}:
                raise ValueError(
                    "lcm_block_count disagrees with cache group page geometry"
                )

        segments = tuple(
            segment for group in self.groups for segment in group.segments
        )
        segment_ids = tuple(segment.segment_id for segment in segments)
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("layout contains a duplicate segment")
        if any(segment.buffer_index >= len(self.buffers) for segment in segments):
            raise ValueError("segment buffer_index is outside the buffer tuple")

        known_segments = set(segment_ids)
        consumed_segments = []
        for consumer in self.consumers:
            if len(consumer) != len(set(consumer)):
                raise ValueError("consumer contains a duplicate segment")
            unknown = set(consumer) - known_segments
            if unknown:
                raise ValueError(f"consumer references unknown segment {sorted(unknown)}")
            consumed_segments.extend(consumer)
        if len(consumed_segments) != len(set(consumed_segments)):
            raise ValueError("a cache segment cannot belong to multiple consumers")
        missing = known_segments - set(consumed_segments)
        if missing:
            raise ValueError(f"cache segments have no consumer {sorted(missing)}")

    @property
    def num_lcm_blocks(self) -> int:
        return self.lcm_block_count or self.groups[0].num_lcm_blocks

    def pack(self, alignment: int = 16) -> PackedCacheLayout:
        """Derive compact child bundles without copying device padding."""

        alignment = _positive("alignment", alignment)
        child_bytes = []
        segment_offsets = []
        parent_bytes = 0
        for group in self.groups:
            offsets = []
            cursor = 0
            for segment in group.segments:
                cursor = _align_up(cursor, alignment)
                offsets.append(cursor)
                cursor += segment.payload_bytes
            packed_child_bytes = _align_up(cursor, alignment)
            child_bytes.append(packed_child_bytes)
            segment_offsets.append(tuple(offsets))
            parent_bytes = max(
                parent_bytes,
                group.cache_blocks_per_lcm_block * packed_child_bytes,
            )
        return PackedCacheLayout(
            child_bytes=tuple(child_bytes),
            segment_offsets=tuple(segment_offsets),
            parent_bytes=parent_bytes,
        )


def layout_from_lcm_plan(
    plan: LcmMemoryPlan,
    backing: object,
    *,
    consumers: tuple[tuple[str, ...], ...],
) -> CacheTransferLayout:
    """Derive transfer rows directly from an LCM arena plan."""

    planes = {plane.plane_id: plane for plane in plan.planes}
    fields_by_group = {
        group.group_id: tuple(
            field for field in plan.fields if field.group_id == group.group_id
        )
        for group in plan.groups
    }
    groups = []
    for group in plan.groups:
        segments = []
        for field in fields_by_group[group.group_id]:
            plane = planes[field.plane_id]
            segments.append(
                CacheSegment(
                    segment_id=field.field_id,
                    buffer_index=0,
                    page_zero_offset=(
                        plane.arena_offset_bytes
                        + plane.bytes_per_lcm_block
                        - field.page_stride_bytes
                        + field.field_offset_bytes
                    ),
                    page_stride_bytes=field.page_stride_bytes,
                    payload_bytes=field.payload_bytes,
                )
            )
        groups.append(
            CacheGroupLayout(
                group_id=group.group_id,
                cache_blocks_per_lcm_block=group.cache_blocks_per_lcm_block,
                page_count=group.page_count,
                segments=tuple(segments),
            )
        )
    return CacheTransferLayout(
        logical_block_tokens=plan.logical_block_tokens,
        groups=tuple(groups),
        buffers=(backing,),
        consumers=consumers,
        lcm_block_count=plan.num_lcm_blocks,
    )


def combine_cache_transfer_layouts(
    target: CacheTransferLayout,
    draft: CacheTransferLayout | None,
) -> CacheTransferLayout:
    """Combine independent target/draft arenas that share scheduler page IDs."""

    if draft is None:
        return target
    if (
        draft.logical_block_tokens != target.logical_block_tokens
        or draft.num_lcm_blocks != target.num_lcm_blocks
    ):
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
            or draft_group.page_count != target_group.page_count
        ):
            raise ValueError(
                f"target and draft cache group {group_id!r} use different geometry"
            )

    draft_buffer_base = len(target.buffers)

    def namespaced_segment(
        segment: CacheSegment, namespace: str, buffer_base: int
    ) -> CacheSegment:
        return CacheSegment(
            segment_id=f"{namespace}:{segment.segment_id}",
            buffer_index=buffer_base + segment.buffer_index,
            page_zero_offset=segment.page_zero_offset,
            page_stride_bytes=segment.page_stride_bytes,
            payload_bytes=segment.payload_bytes,
        )

    groups = []
    for target_group in target.groups:
        segments = tuple(
            namespaced_segment(segment, "target", 0)
            for segment in target_group.segments
        )
        if draft_group := draft_groups.get(target_group.group_id):
            segments += tuple(
                namespaced_segment(segment, "draft", draft_buffer_base)
                for segment in draft_group.segments
            )
        groups.append(
            CacheGroupLayout(
                group_id=target_group.group_id,
                cache_blocks_per_lcm_block=target_group.cache_blocks_per_lcm_block,
                page_count=target_group.page_count,
                segments=segments,
            )
        )

    consumers = tuple(
        tuple(f"target:{segment_id}" for segment_id in consumer)
        for consumer in target.consumers
    ) + tuple(
        tuple(f"draft:{segment_id}" for segment_id in consumer)
        for consumer in draft.consumers
    )
    return CacheTransferLayout(
        logical_block_tokens=target.logical_block_tokens,
        groups=tuple(groups),
        buffers=target.buffers + draft.buffers,
        consumers=consumers,
        lcm_block_count=target.num_lcm_blocks,
    )
