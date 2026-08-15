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

"""Pure integer geometry for one shared LCM cache arena."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise

# Planner/runtime limits, not model layout inputs.
_MAX_LCM_BLOCK_BYTES = (1 << 63) - 1
_MAX_KERNEL_PAGE_ID = (1 << 31) - 1


def cache_field_layer_id(field_id: str) -> int:
    """Return the owning model layer encoded in a cache field ID."""
    parts = field_id.split(".", 2)
    if len(parts) != 3 or parts[0] != "layer":
        raise ValueError(f"cache field {field_id!r} is not owned by a model layer")
    try:
        layer_id = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"cache field {field_id!r} has an invalid layer id") from exc
    if layer_id < 0:
        raise ValueError(f"cache field {field_id!r} has an invalid layer id")
    return layer_id


@dataclass(frozen=True)
class CacheGroupLayout:
    group_id: str
    cache_blocks_per_lcm_block: int
    page_count: int


@dataclass(frozen=True)
class CachePlaneLayout:
    plane_id: str
    bytes_per_lcm_block: int
    arena_offset_bytes: int


@dataclass(frozen=True)
class CacheFieldLayout:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    field_offset_bytes: int
    page_stride_bytes: int

    @property
    def payload_bytes(self) -> int:
        return math.prod(self.shape) * self.element_size


@dataclass(frozen=True)
class CacheMemoryPlan:
    """Static byte geometry for one shared physical LCM arena.

    ``num_lcm_blocks`` excludes the null parent. ``arena_bytes`` includes it.
    """

    prefix_granularity: int
    lcm_block_bytes: int
    num_lcm_blocks: int
    groups: tuple[CacheGroupLayout, ...]
    planes: tuple[CachePlaneLayout, ...] = ()
    fields: tuple[CacheFieldLayout, ...] = ()

    @property
    def arena_bytes(self) -> int:
        return (self.num_lcm_blocks + 1) * self.lcm_block_bytes

    def group(self, group_id: str) -> CacheGroupLayout:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(group_id)

    def field(self, field_id: str) -> CacheFieldLayout:
        for field in self.fields:
            if field.field_id == field_id:
                return field
        raise KeyError(field_id)

    def plane(self, plane_id: str) -> CachePlaneLayout:
        for plane in self.planes:
            if plane.plane_id == plane_id:
                return plane
        raise KeyError(plane_id)

    def field_page_byte_offset(self, field_id: str, page_id: int) -> int:
        """Return one field page's byte offset in the shared cache arena."""
        field = self.field(field_id)
        group = self.group(field.group_id)
        if (
            isinstance(page_id, bool)
            or not isinstance(page_id, int)
            or page_id < 0
            or page_id >= group.page_count
        ):
            raise IndexError(
                f"page_id {page_id} outside [0, {group.page_count}) for "
                f"group {group.group_id!r}"
            )
        plane = self.plane(field.plane_id)
        return (
            plane.arena_offset_bytes
            + plane.bytes_per_lcm_block
            - field.page_stride_bytes
            + page_id * field.page_stride_bytes
            + field.field_offset_bytes
        )

    def capacity_report(
        self,
        *,
        window_tokens: Mapping[str, int] | None = None,
        per_request_blocks: Mapping[str, int] | None = None,
        max_num_seqs: int | None = None,
    ) -> dict:
        """Per-group capacity in its own consumption unit, plus dead bytes.

        Heterogeneous groups consume in different units — full-attention
        per token, state (KDA/Mamba) per request, sliding-window bounded by
        the window — so a single token-denominated capacity misstates two
        of the three. Callers name the special groups:

        Args:
            window_tokens: retention-bounded groups (sliding window): active
                demand per request is at most the window, so capacity beyond
                ``max_num_seqs × window`` is dead rows (stranded by the
                static slab split).
            per_request_blocks: per-request-constant groups (state): the
                admission they support is ``page_count / blocks_per_req``.
            max_num_seqs: admission bound used for dead-row estimates.

        Returns:
            ``{group_id: {"unit", "capacity", "supported_requests",
            "dead_bytes", "binding_utilization"}}`` with
            ``supported_requests`` None when unknown. Binding admission =
            min over non-None supported_requests. ``binding_utilization``
            is the fraction of a parent a binding of this group actually
            uses (aliased slabs are sized by their widest tenant; a
            narrower group's binding leaves the rest dead — the
            binding hole).
        """
        window_tokens = dict(window_tokens or {})
        per_request_blocks = dict(per_request_blocks or {})
        group_bytes_per_block: dict[str, int] = {}
        for field in self.fields:
            group_bytes_per_block[field.group_id] = (
                group_bytes_per_block.get(field.group_id, 0) + field.payload_bytes
            )
        report: dict[str, dict] = {}
        for group in self.groups:
            usable_pages = self.num_lcm_blocks * group.cache_blocks_per_lcm_block
            block_bytes = group_bytes_per_block.get(group.group_id, 0)
            binding_utilization = (
                group.cache_blocks_per_lcm_block * block_bytes / self.lcm_block_bytes
                if self.lcm_block_bytes
                else 0.0
            )
            if group.group_id in per_request_blocks:
                blocks_per_req = max(1, per_request_blocks[group.group_id])
                supported = usable_pages // blocks_per_req
                demand_pages = max_num_seqs * blocks_per_req if max_num_seqs else None
                report[group.group_id] = {
                    "unit": "requests",
                    "capacity": supported,
                    "supported_requests": supported,
                    "dead_bytes": (
                        max(0, usable_pages - demand_pages) * block_bytes
                        if demand_pages is not None
                        else None
                    ),
                    "binding_utilization": binding_utilization,
                }
                continue
            token_capacity = usable_pages * self.prefix_granularity
            if group.group_id in window_tokens:
                window = max(1, window_tokens[group.group_id])
                demand_tokens = max_num_seqs * window if max_num_seqs else None
                supported = token_capacity // window
                report[group.group_id] = {
                    "unit": "tokens",
                    "capacity": token_capacity,
                    "supported_requests": supported,
                    "dead_bytes": (
                        max(0, token_capacity - demand_tokens)
                        // self.prefix_granularity
                        * block_bytes
                        if demand_tokens is not None
                        else None
                    ),
                    "binding_utilization": binding_utilization,
                }
                continue
            report[group.group_id] = {
                "unit": "tokens",
                "capacity": token_capacity,
                # Depends on per-request context length; unknown here.
                "supported_requests": None,
                "dead_bytes": 0,
                "binding_utilization": binding_utilization,
            }
        return report


@dataclass(frozen=True)
class CacheFieldSpec:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    # True when the field's kernel walks pages by an implicit payload-sized
    # stride. False when the kernel consumes the tensor's runtime stride.
    exact_page_stride: bool = True
    # Some kernels accept padded pages but still require the runtime page
    # stride to satisfy an alignment constraint (for example, a TMA row
    # stride). The planner applies this in bytes after group packing.
    page_stride_alignment_bytes: int = 1

    @property
    def payload_bytes(self) -> int:
        return math.prod(self.shape) * self.element_size


@dataclass(frozen=True)
class CacheLayout:
    """Capacity-independent byte geometry for one LCM block."""

    prefix_granularity: int
    lcm_block_bytes: int
    group_packing: tuple[tuple[str, int], ...]
    plane_bytes: tuple[tuple[str, int], ...]
    fields: tuple[CacheFieldLayout, ...]

    def with_num_lcm_blocks(self, num_lcm_blocks: int) -> CacheMemoryPlan:
        if (
            isinstance(num_lcm_blocks, bool)
            or not isinstance(num_lcm_blocks, int)
            or num_lcm_blocks < 1
        ):
            raise ValueError("num_lcm_blocks must be a positive integer")

        groups = []
        for group_id, count in self.group_packing:
            if num_lcm_blocks * count > _MAX_KERNEL_PAGE_ID:
                raise ValueError(
                    f"cache group {group_id!r}: kernel page id exceeds "
                    f"{_MAX_KERNEL_PAGE_ID}"
                )
            groups.append(
                CacheGroupLayout(
                    group_id=group_id,
                    cache_blocks_per_lcm_block=count,
                    page_count=1 + num_lcm_blocks * count,
                )
            )

        planes = []
        arena_offset = 0
        for plane_id, bytes_per_lcm_block in self.plane_bytes:
            planes.append(
                CachePlaneLayout(
                    plane_id=plane_id,
                    bytes_per_lcm_block=bytes_per_lcm_block,
                    arena_offset_bytes=arena_offset,
                )
            )
            arena_offset += (num_lcm_blocks + 1) * bytes_per_lcm_block

        return CacheMemoryPlan(
            prefix_granularity=self.prefix_granularity,
            lcm_block_bytes=self.lcm_block_bytes,
            num_lcm_blocks=num_lcm_blocks,
            groups=tuple(groups),
            planes=tuple(planes),
            fields=self.fields,
        )


def merge_continuation_layers(
    *,
    fields,
    layer_types: tuple[str, ...],
    group_ids: tuple[str, ...],
    layer_kv_head_counts: tuple[int, ...] | None = None,
    draft_fields=None,
    draft_layer_types: tuple[str, ...] = (),
    draft_group_ids: tuple[str, ...] = (),
    draft_layer_kv_head_counts: tuple[int, ...] | None = None,
) -> tuple:
    """Merge a draft model's per-layer vectors after the target's.

    One big model: the draft's fields renumber via
    :func:`continue_layer_fields`; every other per-layer vector is plain
    concatenation. Returns ``(fields, layer_types, group_ids,
    layer_kv_head_counts, num_draft_layers)`` — all target-shaped, so
    downstream builders stay draft-oblivious.
    """
    merged_fields = tuple(fields)
    merged_layer_types = tuple(layer_types)
    merged_group_ids = tuple(group_ids)
    merged_head_counts = layer_kv_head_counts
    num_draft_layers = 0
    if draft_fields is not None:
        num_target_layers = len(merged_group_ids)
        num_draft_layers = len(draft_group_ids)
        if len(draft_layer_types) != num_draft_layers:
            raise ValueError(
                f"draft layer_types has {len(draft_layer_types)} entries but "
                f"draft group_ids has {num_draft_layers}"
            )
        merged_fields += continue_layer_fields(
            draft_fields, first_layer_id=num_target_layers
        )
        merged_layer_types += tuple(draft_layer_types)
        merged_group_ids += tuple(draft_group_ids)
        if bool(layer_kv_head_counts) != bool(draft_layer_kv_head_counts):
            raise ValueError(
                "layer_kv_head_counts must be supplied for both sides or neither"
            )
        if layer_kv_head_counts is not None:
            merged_head_counts = tuple(layer_kv_head_counts) + tuple(
                draft_layer_kv_head_counts
            )
    return (
        merged_fields,
        merged_layer_types,
        merged_group_ids,
        merged_head_counts,
        num_draft_layers,
    )


def continue_layer_fields(
    fields,
    *,
    first_layer_id: int,
) -> tuple[CacheFieldSpec, ...]:
    """Renumber per-layer fields as continuation layers of one big model.

    Draft layers join the target's ``solve_cache_layout`` as ordinary
    layers of the ONE merged model:
    a draft model's local ``layer.{i}...`` field/plane ids become the
    global ``layer.{first_layer_id + i}...``. Group ids are untouched: a
    draft layer in a target group shares its page-id space and packing by
    construction. No draft-specific namespace exists — the merged plan is
    simply a model with more layers.
    """
    renumber = re.compile(r"^(?P<head>layer|unit|slot)\.(?P<idx>\d+)")

    def _shift(identifier: str) -> str:
        return renumber.sub(
            lambda m: f"{m.group('head')}.{int(m.group('idx')) + first_layer_id}",
            identifier,
        )

    return tuple(
        replace(
            field,
            field_id=_shift(field.field_id),
            plane_id=_shift(field.plane_id),
        )
        for field in fields
    )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _solve_packing(fields):
    """Derive per-group cache blocks per LCM block from exact field ratios."""
    exact_by_plane: dict[str, dict[str, int]] = {}
    groups = set()
    for field in fields:
        groups.add(field.group_id)
        if field.exact_page_stride:
            plane = exact_by_plane.setdefault(field.plane_id, {})
            plane[field.group_id] = plane.get(field.group_id, 0) + field.payload_bytes

    # ratio[g] = (num, den) expressing K_g as a multiple of its component root.
    root = {group_id: group_id for group_id in groups}
    ratio = {group_id: (1, 1) for group_id in groups}

    def resolve(group_id):
        num, den = 1, 1
        while root[group_id] != group_id:
            r_num, r_den = ratio[group_id]
            num, den = num * r_num, den * r_den
            group_id = root[group_id]
        return group_id, (num, den)

    for plane_id in sorted(exact_by_plane):
        tenants = sorted(exact_by_plane[plane_id].items())
        for (first_id, first_bytes), (other_id, other_bytes) in pairwise(tenants):
            first_root, (f_num, f_den) = resolve(first_id)
            other_root, (o_num, o_den) = resolve(other_id)
            if first_root == other_root:
                if f_num * o_den * first_bytes != o_num * f_den * other_bytes:
                    return None
                continue
            num = f_num * o_den * first_bytes
            den = o_num * f_den * other_bytes
            common = math.gcd(num, den)
            root[other_root] = first_root
            ratio[other_root] = (num // common, den // common)

    members: dict[str, list] = {}
    for group_id in sorted(groups):
        component, fraction = resolve(group_id)
        members.setdefault(component, []).append((group_id, fraction))

    packing: dict[str, int] = {}
    for component_members in members.values():
        if len(component_members) == 1:
            continue
        scale = 1
        for _, (_, den) in component_members:
            scale = scale * den // math.gcd(scale, den)
        counts = {
            group_id: num * scale // den for group_id, (num, den) in component_members
        }
        smallest = 0
        for count in counts.values():
            smallest = math.gcd(smallest, count)
        packing.update(
            {group_id: count // smallest for group_id, count in counts.items()}
        )

    return packing


def _packing_by_group_ratio(raw_by_group):
    largest_payload = max(raw_by_group.values())
    return {
        group_id: max(1, largest_payload // raw_by_group[group_id])
        for group_id in raw_by_group
    }


def _check_exact_page_strides(fields, plane_bytes, packing):
    for field in fields:
        if not field.exact_page_stride:
            continue
        stride = plane_bytes[field.plane_id] // packing[field.group_id]
        if stride != field.payload_bytes:
            widest = max(
                (
                    (other.payload_bytes * packing[other.group_id], other.field_id)
                    for other in fields
                    if other.plane_id == field.plane_id
                ),
                default=(0, ""),
            )
            raise ValueError(
                f"cache field {field.field_id!r} needs page stride "
                f"{field.payload_bytes} but plane {field.plane_id!r} gives "
                f"{stride}; its kernel indexes pages by an implicit "
                f"payload-sized stride and would read the wrong rows. The "
                f"plane is sized by {widest[1]!r}; move one of them to another "
                "plane or align their payloads"
            )


def solve_cache_layout(
    fields,
    *,
    prefix_granularity,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
    alignment=1,
    max_padding_fraction=0.25,
):
    """Solve one capacity-independent, plane-major LCM block layout."""
    if prefix_granularity <= 0:
        raise ValueError("prefix_granularity must be > 0")
    if alignment <= 0:
        raise ValueError("alignment must be > 0")
    if max_padding_fraction < 0:
        raise ValueError("max_padding_fraction must be >= 0")

    ordered_fields = tuple(
        sorted(
            fields,
            key=lambda field: (field.plane_id, field.group_id, field.field_id),
        )
    )
    if not ordered_fields:
        raise ValueError("at least one cache field is required")
    if len({field.field_id for field in ordered_fields}) != len(ordered_fields):
        raise ValueError("cache field ids must be unique")

    raw_by_group: dict[str, int] = {}
    for field in ordered_fields:
        if not field.group_id or not field.field_id or not field.plane_id:
            raise ValueError(
                "cache field group_id, field_id and plane_id must be non-empty"
            )
        if (
            field.element_size <= 0
            or not field.shape
            or any(extent <= 0 for extent in field.shape)
            or isinstance(field.page_stride_alignment_bytes, bool)
            or not isinstance(field.page_stride_alignment_bytes, int)
            or field.page_stride_alignment_bytes <= 0
        ):
            raise ValueError(f"cache field {field.field_id!r} has invalid geometry")
        raw_by_group[field.group_id] = (
            raw_by_group.get(field.group_id, 0) + field.payload_bytes
        )

    ordered_group_ids = tuple(sorted(raw_by_group))
    has_explicit_packing = cache_blocks_per_lcm_block is not None
    if has_explicit_packing:
        unknown = set(cache_blocks_per_lcm_block) - set(ordered_group_ids)
        if unknown:
            raise ValueError(
                "cache_blocks_per_lcm_block names groups outside the plan: "
                f"{sorted(unknown)}"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in cache_blocks_per_lcm_block.values()
        ):
            raise ValueError("cache group packing must be a positive integer")
        # Pinned groups take the caller's count; groups the caller leaves
        # unpinned pack by their byte ratio as in the unpinned solve.
        packing = _packing_by_group_ratio(raw_by_group)
        constrained = _solve_packing(ordered_fields)
        if constrained is not None:
            packing.update(constrained)
        packing.update(cache_blocks_per_lcm_block)
    else:
        packing = _packing_by_group_ratio(raw_by_group)
        constrained = _solve_packing(ordered_fields)
        if constrained is not None:
            packing.update(constrained)

    plane_fields: dict[str, dict[str, list[CacheFieldSpec]]] = {}
    for field in ordered_fields:
        plane_fields.setdefault(field.plane_id, {}).setdefault(
            field.group_id, []
        ).append(field)

    field_offsets: dict[str, int] = {}
    group_plane_bytes: dict[str, dict[str, int]] = defaultdict(dict)
    for plane_id, by_group in plane_fields.items():
        for group_id, group_fields in by_group.items():
            offset = 0
            for field in group_fields:
                offset = _align_up(offset, field.element_size)
                field_offsets[field.field_id] = offset
                offset += field.payload_bytes
            group_plane_bytes[group_id][plane_id] = offset

    # Flexible fields consume explicit strides and can use slack left by exact
    # fields, but their K must divide every plane they occupy.
    exact_groups = {
        field.group_id for field in ordered_fields if field.exact_page_stride
    }
    flexible_groups = set(ordered_group_ids) - exact_groups
    if flexible_groups and not has_explicit_packing:
        fixed_plane_bytes = {}
        for plane_id, by_group in plane_fields.items():
            fixed_alignment = alignment
            fixed_required = 0
            for group_id, group_fields in by_group.items():
                if group_id in flexible_groups:
                    continue
                fixed_required = max(
                    fixed_required,
                    packing[group_id] * group_plane_bytes[group_id][plane_id],
                )
                for field in group_fields:
                    required = packing[group_id] * field.element_size
                    fixed_alignment = (
                        fixed_alignment
                        // math.gcd(fixed_alignment, required)
                        * required
                    )
            if fixed_required:
                fixed_plane_bytes[plane_id] = _align_up(
                    fixed_required,
                    fixed_alignment,
                )

        for group_id in sorted(flexible_groups):
            occupied_planes = group_plane_bytes[group_id]
            if not all(plane_id in fixed_plane_bytes for plane_id in occupied_planes):
                continue
            upper = min(
                fixed_plane_bytes[plane_id] // payload_bytes
                for plane_id, payload_bytes in occupied_planes.items()
            )
            element_alignment_by_plane = {}
            for plane_id, group_fields in plane_fields.items():
                fields_for_group = group_fields.get(group_id, ())
                element_alignment = 1
                for field in fields_for_group:
                    element_alignment = (
                        element_alignment
                        // math.gcd(element_alignment, field.element_size)
                        * field.element_size
                    )
                if fields_for_group:
                    element_alignment_by_plane[plane_id] = element_alignment
            for count in range(upper, 0, -1):
                if all(
                    fixed_plane_bytes[plane_id]
                    % (count * element_alignment_by_plane[plane_id])
                    == 0
                    for plane_id in occupied_planes
                ):
                    packing[group_id] = count
                    break

    plane_bytes: dict[str, int] = {}
    for plane_id, by_group in plane_fields.items():
        plane_alignment = alignment
        required_bytes = 0
        for group_id, group_fields in by_group.items():
            required_bytes = max(
                required_bytes,
                packing[group_id] * group_plane_bytes[group_id][plane_id],
            )
            for field in group_fields:
                required = packing[group_id] * field.element_size
                plane_alignment = (
                    plane_alignment // math.gcd(plane_alignment, required) * required
                )
                stride_required = packing[group_id] * field.page_stride_alignment_bytes
                plane_alignment = (
                    plane_alignment
                    // math.gcd(plane_alignment, stride_required)
                    * stride_required
                )
                if plane_alignment > _MAX_LCM_BLOCK_BYTES:
                    raise ValueError(
                        f"LCM block size alignment {plane_alignment} exceeds "
                        f"limit {_MAX_LCM_BLOCK_BYTES}"
                    )
        plane_bytes[plane_id] = _align_up(required_bytes, plane_alignment)

    _check_exact_page_strides(ordered_fields, plane_bytes, packing)

    parent_alignment = alignment
    for count in packing.values():
        parent_alignment = parent_alignment // math.gcd(parent_alignment, count) * count
    lcm_block_bytes = _align_up(sum(plane_bytes.values()), parent_alignment)
    if lcm_block_bytes > _MAX_LCM_BLOCK_BYTES:
        raise ValueError(
            f"LCM block size {lcm_block_bytes} exceeds limit {_MAX_LCM_BLOCK_BYTES}"
        )

    for group_id in ordered_group_ids:
        count = packing[group_id]
        if lcm_block_bytes % count:
            raise ValueError(
                f"cache group {group_id!r}: packing {count} does not partition "
                f"LCM block size {lcm_block_bytes}"
            )
        stride = lcm_block_bytes // count
        padding_fraction = (stride - raw_by_group[group_id]) / raw_by_group[group_id]
        if padding_fraction > max_padding_fraction:
            raise ValueError(
                f"cache group {group_id!r}: padding fraction "
                f"{padding_fraction:.6f} exceeds limit {max_padding_fraction:.6f}"
            )

    field_layouts = []
    for field in ordered_fields:
        field_layouts.append(
            CacheFieldLayout(
                group_id=field.group_id,
                field_id=field.field_id,
                plane_id=field.plane_id,
                shape=field.shape,
                element_size=field.element_size,
                field_offset_bytes=field_offsets[field.field_id],
                page_stride_bytes=plane_bytes[field.plane_id]
                // packing[field.group_id],
            )
        )

    return CacheLayout(
        prefix_granularity=prefix_granularity,
        lcm_block_bytes=lcm_block_bytes,
        group_packing=tuple(
            (group_id, packing[group_id]) for group_id in ordered_group_ids
        ),
        plane_bytes=tuple(
            (plane_id, plane_bytes[plane_id]) for plane_id in sorted(plane_bytes)
        ),
        fields=tuple(field_layouts),
    )
