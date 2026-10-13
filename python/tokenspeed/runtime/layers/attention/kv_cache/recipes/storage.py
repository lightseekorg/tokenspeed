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

"""Physical cache regions and workspace, bound from the logical memory plan.

All capacities are bytes of actual regions, including null parents. Host
placement describes authoritative storage and never creates an L2 owner.
"""

from dataclasses import dataclass, replace
from math import lcm, prod
from typing import Literal

from tokenspeed.runtime.layers.attention.kv_cache.offload_config import KVOffloadConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheLayout,
    CacheMemoryPlan,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


@dataclass(frozen=True)
class StorageRegion:
    region_id: str
    placement: Literal["device", "host"]
    logical_offset: int
    size_bytes: int


@dataclass(frozen=True)
class FieldStorageBinding:
    field_id: str
    region_id: str
    offset_bytes: int


@dataclass(frozen=True)
class OffloadFieldWorkspace:
    field_id: str
    row_shape: tuple[int, ...]
    dtype: str
    row_bytes: int
    device_rows: int
    metadata: tuple[tuple[str, int], ...]
    temporary_bytes: int

    @property
    def persistent_bytes(self) -> int:
        return self.device_rows * self.row_bytes + sum(n * 4 for _, n in self.metadata)


@dataclass(frozen=True)
class CacheStoragePlan:
    """One physical build result shared by allocation, budgets and consumers."""

    regions: tuple[StorageRegion, ...]
    fields: tuple[FieldStorageBinding, ...]
    workspaces: tuple[OffloadFieldWorkspace, ...]
    offload: KVOffloadConfig | None

    @property
    def device_history_bytes(self) -> int:
        return sum(r.size_bytes for r in self.regions if r.placement == "device")

    @property
    def host_bytes(self) -> int:
        return sum(r.size_bytes for r in self.regions if r.placement == "host")

    @property
    def persistent_workspace_bytes(self) -> int:
        return sum(w.persistent_bytes for w in self.workspaces)

    @property
    def temporary_workspace_bytes(self) -> int:
        # Selection producers can prefetch another field concurrently.
        # Reserving every field's peak also bounds graph-private workspaces.
        return sum(w.temporary_bytes for w in self.workspaces)

    @property
    def fixed_device_bytes(self) -> int:
        return self.persistent_workspace_bytes + self.temporary_workspace_bytes

    @property
    def device_budget_bytes(self) -> int:
        return self.device_history_bytes + self.fixed_device_bytes

    @property
    def monolithic(self) -> bool:
        return len(self.regions) == 1 and self.regions[0].region_id == "arena"

    def split_ranges(
        self, ranges: list[tuple[int, int]]
    ) -> dict[str, list[tuple[int, int]]]:
        """Translate plan byte ranges into region-relative ranges."""
        result = {r.region_id: [] for r in self.regions}
        for offset, size in ranges:
            covered = 0
            for region in self.regions:
                start = max(offset, region.logical_offset)
                end = min(offset + size, region.logical_offset + region.size_bytes)
                if start < end:
                    result[region.region_id].append(
                        (start - region.logical_offset, end - start)
                    )
                    covered += end - start
            if covered != size:
                raise ValueError("cache byte range is outside the physical binding")
        return result


def compute_offload_capacity(
    layout: CacheLayout,
    group_specs: tuple[CacheGroupSpec, ...],
    *,
    offload: KVOffloadConfig,
    device_budget_bytes: int,
    max_lcm_blocks: int,
    probe_lcm_blocks: int | None,
) -> tuple[int, KVOffloadConfig]:
    """Compute history capacity after reserving every configured hot partition.

    Request-slot ownership grants a complete hot partition, including during
    PD landing. Insufficient memory is a startup error, never a lower implicit
    concurrency limit. Probe and final rebind use the same fixed hot geometry.

    Returns the history LCM block count (excluding the null parent) and the
    offload configuration with page-aligned device rows. No buffers are allocated.
    """
    unit = plan_cache_storage(layout.bind(1), group_specs, offload=offload)
    specs = {s.group_id: s for s in group_specs}
    alignment = lcm(
        *(
            specs[f.group_id].rows_per_page
            for f in layout.bind(1).fields
            if f.field_id in offload.field_ids
        )
    )
    row_bytes = sum(w.row_bytes for w in unit.workspaces)
    overhead = unit.temporary_workspace_bytes + sum(
        sum(n * 4 for _, n in w.metadata) for w in unit.workspaces
    )
    device_parent = unit.device_history_bytes // 2
    host_parent = unit.host_bytes // 2
    rows = ((offload.minimum_device_rows + alignment - 1) // alignment) * alignment
    if rows >= 2**31:
        raise ValueError("KV offload fixed hot row IDs must fit int32")
    history_budget = device_budget_bytes - overhead - rows * row_bytes
    if probe_lcm_blocks is None:
        limits = [max_lcm_blocks, offload.host_budget_bytes // host_parent - 1]
        if history_budget < 0:
            raise ValueError("configured fixed hot pool exceeds cache budget")
        if device_parent:
            limits.append(history_budget // device_parent - 1)
        parents = min(limits)
    else:
        # A zero budget is the existing geometry-only profiling contract.
        parents = probe_lcm_blocks
        if device_budget_bytes > 0 and history_budget < (parents + 1) * device_parent:
            raise ValueError(
                "fixed hot pool leaves insufficient probe history capacity"
            )
    if parents < 1:
        raise ValueError("fixed hot pool leaves insufficient history capacity")
    return parents, replace(offload, device_rows=rows)


def plan_cache_storage(
    plan: CacheMemoryPlan,
    group_specs: tuple[CacheGroupSpec, ...],
    *,
    offload: KVOffloadConfig | None,
) -> CacheStoragePlan:
    """Bind placements and exact persistent allocations; validate row geometry."""
    if offload is None:
        return CacheStoragePlan(
            (StorageRegion("arena", "device", 0, plan.arena_bytes),),
            tuple(
                FieldStorageBinding(
                    f.field_id, "arena", plan.field_page_byte_offset(f.field_id, 0)
                )
                for f in plan.fields
            ),
            (),
            None,
        )
    selected = frozenset(offload.field_ids)
    fields = {f.field_id: f for f in plan.fields}
    if not selected <= fields.keys():
        raise ValueError("offload policy names an absent cache field")
    specs = {s.group_id: s for s in group_specs}
    for producer, consumer in offload.selection_consumers:
        if producer not in fields or consumer not in fields:
            raise ValueError("selection dependency names an absent cache field")
        if fields[producer].group_id != fields[consumer].group_id:
            raise ValueError("shared selections must use the same history group")
    regions, bindings, workspaces = [], [], []
    for plane in plan.planes:
        members = [f for f in plan.fields if f.plane_id == plane.plane_id]
        host = any(f.field_id in selected for f in members)
        if host and (
            len(members) != 1
            or members[0].page_stride_bytes != members[0].payload_bytes
        ):
            raise ValueError("offloaded fields require contiguous independent planes")
        regions.append(
            StorageRegion(
                plane.plane_id,
                "host" if host else "device",
                plane.arena_offset_bytes,
                plane.bytes_per_lcm_block * (plan.num_lcm_blocks + 1),
            )
        )
        for f in members:
            bindings.append(
                FieldStorageBinding(
                    f.field_id,
                    plane.plane_id,
                    plan.field_page_byte_offset(f.field_id, 0)
                    - plane.arena_offset_bytes,
                )
            )
            if not host:
                continue
            spec = specs[f.group_id]
            if (
                spec.family != "history"
                or spec.entry_stride_tokens != 1
                or spec.retention != "full_history"
                or spec.shard_count != 1
                or f.shape[0] != spec.rows_per_page
            ):
                raise ValueError(
                    "sparse residency requires unsharded full-history token rows"
                )
            row_shape = f.shape[1:]
            workspaces.append(
                OffloadFieldWorkspace(
                    f.field_id,
                    row_shape,
                    f.dtype,
                    prod(row_shape) * f.element_size,
                    (
                        (offload.device_rows + spec.rows_per_page - 1)
                        // spec.rows_per_page
                    )
                    * spec.rows_per_page,
                    tuple(offload.metadata_counts().items()),
                    offload.temporary_bytes(),
                )
            )
            if plan.group(f.group_id).page_count * spec.rows_per_page >= 2**31:
                raise ValueError("KV offload history row IDs must fit int32")
    result = CacheStoragePlan(
        tuple(regions), tuple(bindings), tuple(workspaces), offload
    )
    if result.host_bytes > offload.host_budget_bytes:
        raise ValueError("offload host allocation exceeds configured budget")
    return result
