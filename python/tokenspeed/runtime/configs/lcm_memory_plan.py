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
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class LcmGroupLayout:
    group_id: str
    raw_cache_block_bytes: int
    slot_stride_bytes: int
    cache_blocks_per_lcm_block: int
    page_count: int

    @property
    def padding_bytes_per_slot(self) -> int:
        return self.slot_stride_bytes - self.raw_cache_block_bytes


@dataclass(frozen=True)
class LcmFieldSpec:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    # True when the field's kernel walks pages by an implicit payload-sized
    # stride. False when the kernel consumes the tensor's runtime stride.
    exact_page_stride: bool = True

    @property
    def payload_bytes(self) -> int:
        return math.prod(self.shape) * self.element_size


@dataclass(frozen=True)
class LcmPlaneLayout:
    plane_id: str
    bytes_per_lcm_block: int
    arena_offset_bytes: int


@dataclass(frozen=True)
class LcmFieldLayout:
    group_id: str
    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    element_size: int
    payload_bytes: int
    field_offset_bytes: int
    page_stride_bytes: int


@dataclass(frozen=True)
class LcmMemoryPlan:
    """Static byte geometry for one shared physical LCM arena.

    ``num_lcm_blocks`` excludes the null parent. ``arena_bytes`` includes it.
    """

    logical_block_tokens: int
    lcm_block_bytes: int
    num_lcm_blocks: int
    arena_bytes: int
    groups: tuple[LcmGroupLayout, ...]
    planes: tuple[LcmPlaneLayout, ...] = ()
    fields: tuple[LcmFieldLayout, ...] = ()

    def group(self, group_id: str) -> LcmGroupLayout:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(group_id)

    def field(self, field_id: str) -> LcmFieldLayout:
        for field in self.fields:
            if field.field_id == field_id:
                return field
        raise KeyError(field_id)

    def plane(self, plane_id: str) -> LcmPlaneLayout:
        for plane in self.planes:
            if plane.plane_id == plane_id:
                return plane
        raise KeyError(plane_id)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _solve_packing(fields, *, max_packing):
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
        for (first_id, first_bytes), (other_id, other_bytes) in zip(
            tenants, tenants[1:]
        ):
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

    if any(count > max_packing for count in packing.values()):
        return None
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


def plan_lcm_fields(
    fields,
    *,
    logical_block_tokens,
    budget_bytes,
    alignment=1,
    max_lcm_block_bytes=(1 << 63) - 1,
    max_kernel_page_id=(1 << 31) - 1,
    max_padding_fraction=0.25,
    max_packing=128,
):
    """Plan a plane-major arena whose fields overlay across cache groups."""
    if logical_block_tokens <= 0:
        raise ValueError("logical_block_tokens must be > 0")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be >= 0")
    if alignment <= 0:
        raise ValueError("alignment must be > 0")
    if max_lcm_block_bytes <= 0:
        raise ValueError("max_lcm_block_bytes must be > 0")
    if max_padding_fraction < 0:
        raise ValueError("max_padding_fraction must be >= 0")
    if max_packing <= 0:
        raise ValueError("max_packing must be > 0")

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
        ):
            raise ValueError(f"cache field {field.field_id!r} has invalid geometry")
        raw_by_group[field.group_id] = (
            raw_by_group.get(field.group_id, 0) + field.payload_bytes
        )

    ordered_group_ids = tuple(sorted(raw_by_group))
    packing = _packing_by_group_ratio(raw_by_group)
    constrained = _solve_packing(ordered_fields, max_packing=max_packing)
    if constrained is not None:
        packing.update(constrained)

    plane_fields: dict[str, dict[str, list[LcmFieldSpec]]] = {}
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
    if flexible_groups:
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
                max_packing,
                *(
                    fixed_plane_bytes[plane_id] // payload_bytes
                    for plane_id, payload_bytes in occupied_planes.items()
                ),
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
                if plane_alignment > max_lcm_block_bytes:
                    raise ValueError(
                        f"LCM block size alignment {plane_alignment} exceeds "
                        f"limit {max_lcm_block_bytes}"
                    )
        plane_bytes[plane_id] = _align_up(required_bytes, plane_alignment)

    _check_exact_page_strides(ordered_fields, plane_bytes, packing)

    parent_alignment = alignment
    for count in packing.values():
        parent_alignment = parent_alignment // math.gcd(parent_alignment, count) * count
    lcm_block_bytes = _align_up(sum(plane_bytes.values()), parent_alignment)
    if lcm_block_bytes > max_lcm_block_bytes:
        raise ValueError(
            f"LCM block size {lcm_block_bytes} exceeds limit {max_lcm_block_bytes}"
        )

    # Parent 0 backs kernel page 0 and is never schedulable.
    num_lcm_blocks = budget_bytes // lcm_block_bytes - 1
    if num_lcm_blocks < 1:
        raise ValueError("budget must hold a null parent and one usable LCM block")

    groups = []
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
        if num_lcm_blocks * count > max_kernel_page_id:
            raise ValueError(
                f"cache group {group_id!r}: kernel page id exceeds "
                f"{max_kernel_page_id}"
            )
        groups.append(
            LcmGroupLayout(
                group_id=group_id,
                raw_cache_block_bytes=raw_by_group[group_id],
                slot_stride_bytes=stride,
                cache_blocks_per_lcm_block=count,
                page_count=1 + num_lcm_blocks * count,
            )
        )

    planes = []
    arena_offset = 0
    for plane_id in sorted(plane_bytes):
        nbytes = plane_bytes[plane_id]
        planes.append(LcmPlaneLayout(plane_id, nbytes, arena_offset))
        arena_offset += (num_lcm_blocks + 1) * nbytes

    field_layouts = []
    for field in ordered_fields:
        field_layouts.append(
            LcmFieldLayout(
                group_id=field.group_id,
                field_id=field.field_id,
                plane_id=field.plane_id,
                shape=field.shape,
                element_size=field.element_size,
                payload_bytes=field.payload_bytes,
                field_offset_bytes=field_offsets[field.field_id],
                page_stride_bytes=plane_bytes[field.plane_id]
                // packing[field.group_id],
            )
        )

    return LcmMemoryPlan(
        logical_block_tokens=logical_block_tokens,
        lcm_block_bytes=lcm_block_bytes,
        num_lcm_blocks=num_lcm_blocks,
        arena_bytes=(num_lcm_blocks + 1) * lcm_block_bytes,
        groups=tuple(groups),
        planes=tuple(planes),
        fields=tuple(field_layouts),
    )
