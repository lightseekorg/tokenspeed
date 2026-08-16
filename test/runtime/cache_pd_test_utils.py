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

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from types import SimpleNamespace

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldLayout,
    CacheGroupLayout,
    CacheMemoryPlan,
    CachePlaneLayout,
    cache_dtype_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheFieldPartition,
    CacheFieldTransferSpec,
    CachePDGroupPages,
    CachePDPageManifest,
    CacheProducerSchedule,
    CacheTransferContract,
    CacheTransferSchema,
)


@dataclass(frozen=True, slots=True)
class _TestCacheField:
    """Unplanned field input used by the compact PD test builders."""

    field_id: str
    dtype: str
    page_zero_offset: int
    page_stride_bytes: int
    shape: tuple[int, ...]
    partition: CacheFieldPartition | None
    producer_step: int | None


@dataclass(frozen=True, slots=True)
class _TestCacheGroup:
    """Semantic group input awaiting the shared memory plan."""

    group_id: str
    family: str
    prefix_granularity: int
    retention: str
    sliding_window_tokens: int | None
    fields: tuple[_TestCacheField, ...]
    cache_blocks_per_lcm_block: int

    @property
    def transfer_policy(self) -> str:
        return (
            "latest_snapshot"
            if self.family == "state" and self.retention == "full_history"
            else "full_suffix"
        )


def segment(
    field_id: str,
    *,
    shape: tuple[int, ...] = (1,),
    dtype: str = "uint8",
    offset: int = 0,
    stride: int | None = None,
    axis: int | None = None,
    extent: int | None = None,
    parts: tuple[int, ...] = (),
    producer_step: int | None = None,
) -> _TestCacheField:
    payload = prod(shape) * cache_dtype_bytes(dtype)
    partition = (
        None
        if extent is None
        else CacheFieldPartition(
            axis=0 if axis is None else axis,
            global_extent=extent,
            global_parts=parts,
        )
    )
    return _TestCacheField(
        field_id,
        dtype,
        offset,
        payload if stride is None else stride,
        shape,
        partition,
        producer_step,
    )


def group(
    group_id: str,
    *segments: _TestCacheField,
    family: str = "history",
    q: int = 2,
    retention: str = "full_history",
    window: int | None = None,
    packing: int = 1,
) -> _TestCacheGroup:
    return _TestCacheGroup(group_id, family, q, retention, window, segments, packing)


def layout(
    *groups: _TestCacheGroup,
    block_size: int = 2,
    capacity: int = 16,
    page_bytes: int = 32,
) -> CacheTransferContract:
    plan_groups = tuple(
        CacheGroupLayout(
            group_id=entry.group_id,
            cache_blocks_per_lcm_block=entry.cache_blocks_per_lcm_block,
            page_count=(1 + (capacity - 1) * entry.cache_blocks_per_lcm_block),
        )
        for entry in groups
    )
    specs = tuple(
        (
            CacheGroupSpec(
                group_id=entry.group_id,
                retention=entry.retention,
                sliding_window_tokens=entry.sliding_window_tokens,
                family=entry.family,
                transfer_policy=entry.transfer_policy,
                checkpoint_granularity=entry.prefix_granularity,
            )
            if entry.family == "state" and entry.retention == "full_history"
            else CacheGroupSpec(
                group_id=entry.group_id,
                retention=entry.retention,
                rows_per_page=entry.prefix_granularity,
                entry_stride_tokens=1,
                sliding_window_tokens=entry.sliding_window_tokens,
                family=entry.family,
                transfer_policy=entry.transfer_policy,
            )
        )
        for entry in groups
    )

    fields: list[CacheFieldLayout] = []
    planes: list[CachePlaneLayout] = []
    for group_position, entry in enumerate(groups):
        for field_position, field in enumerate(entry.fields):
            plane_id = f"test.{group_position}.{field_position}.{field.field_id}"
            planes.append(
                CachePlaneLayout(
                    plane_id=plane_id,
                    bytes_per_lcm_block=field.page_stride_bytes,
                    arena_offset_bytes=field.page_zero_offset,
                )
            )
            fields.append(
                CacheFieldLayout(
                    group_id=entry.group_id,
                    field_id=field.field_id,
                    plane_id=plane_id,
                    shape=field.shape,
                    dtype=field.dtype,
                    field_offset_bytes=0,
                    page_stride_bytes=field.page_stride_bytes,
                )
            )

    return CacheTransferContract(
        plan=CacheMemoryPlan(
            prefix_granularity=block_size,
            lcm_block_bytes=page_bytes,
            num_lcm_blocks=capacity - 1,
            groups=plan_groups,
            planes=tuple(planes),
            fields=tuple(fields),
        ),
        group_specs=specs,
        transfer_schema=CacheTransferSchema(
            tuple(
                CacheFieldTransferSpec(field.field_id, field.partition)
                for entry in groups
                for field in entry.fields
                if field.partition is not None
            )
        ),
    )


def producer_schedule(
    *groups: _TestCacheGroup,
    steps: int,
) -> CacheProducerSchedule:
    fields_by_step: list[list[str]] = [[] for _ in range(steps)]
    for entry in groups:
        for field in entry.fields:
            assert field.producer_step is not None
            fields_by_step[field.producer_step].append(field.field_id)
    return CacheProducerSchedule(
        tuple(tuple(step_fields) for step_fields in fields_by_step)
    )


def manifest(
    *groups: tuple[str, tuple[int, ...]],
    prefix: int = 0,
    prompt: int = 2,
) -> CachePDPageManifest:
    return CachePDPageManifest(
        groups=tuple(CachePDGroupPages(*entry) for entry in groups),
        prefix_len=prefix,
        prompt_len=prompt,
    )


def operation(tables: dict[str, np.ndarray], **attrs) -> SimpleNamespace:
    return SimpleNamespace(block_tables_arrays=lambda: tables, **attrs)
