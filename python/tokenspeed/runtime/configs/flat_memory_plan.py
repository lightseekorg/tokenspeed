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

"""Flat KV-cache memory plans: pure sizing/binding decisions, no torch.

The LCM planner overlays group-local CacheBlocks on one physical parent while
keeping each kernel field in a stable, strided tensor view. The legacy planner
below retains the earlier equalized-row layout for non-LCM paths.

Legacy components declare per-block bytes as a function of P (block_size):
linear components scale (bytes_per_slot > 0), constant components do not
(const_bytes > 0, mamba state snapshots). Same-(group, layer) components
pack into one page row ([conv|ssm|pad], the vLLM hybrid layout). One
equalizer move: constant rows inflate P until the widest linear row
covers them (vLLM align). plan_tensors then pairs physical slot j with
the j-th layer of every group over a single page-id space and sizes each
slab by its own packed row from the budget.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace

# Labels whose group is state-family (recurrent state rows, not KV history).
# Deliberate one-line duplicate of paged_cache_spec.STATE_LAYER_TYPES: both
# modules are direct-loaded standalone by their tests (importlib, no package
# context), so a cross-module import would break either loader. Keep in sync.
STATE_LAYER_TYPES = frozenset({"linear_attention"})


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
    # stride, so the planner must hand it page_stride == payload_bytes:
    # history K/V (the pool flattens them to [N, H, D]) and GDN SSM state
    # (the decode kernel hardcodes idx * HV * V * K). False for fields whose
    # kernel reads stride(0) at launch, e.g. the GDN conv window.
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


class LcmArena:
    """One fixed backing allocation interpreted through per-group page views."""

    def __init__(self, plan: LcmMemoryPlan, backing):
        if int(backing.nbytes) != plan.arena_bytes:
            raise ValueError(
                f"backing has {int(backing.nbytes)} bytes; plan requires "
                f"{plan.arena_bytes}"
            )
        self.plan = plan
        self.backing = backing

    @classmethod
    def allocate(cls, plan: LcmMemoryPlan, device: str):
        """Allocate the arena once. All group/field tensors are views of it."""
        import torch

        backing = torch.zeros(plan.arena_bytes, dtype=torch.uint8, device=device)
        return cls(plan, backing)

    def field_page_byte_offset(self, field_id: str, page_id: int) -> int:
        field = self.plan.field(field_id)
        group = self.plan.group(field.group_id)
        if page_id < 0 or page_id >= group.page_count:
            raise IndexError(
                f"page_id {page_id} outside [0, {group.page_count}) for "
                f"group {group.group_id!r}"
            )
        plane = self.plan.plane(field.plane_id)
        return (
            plane.arena_offset_bytes
            + plane.bytes_per_lcm_block
            - field.page_stride_bytes
            + page_id * field.page_stride_bytes
            + field.field_offset_bytes
        )

    def page_byte_segments(self, group_id: str, page_ids) -> list[tuple[int, int]]:
        """Return backing byte ranges occupied by the group's selected pages."""
        self.plan.group(group_id)
        fields = [field for field in self.plan.fields if field.group_id == group_id]
        return [
            (
                self.field_page_byte_offset(field.field_id, page_id),
                field.payload_bytes,
            )
            for page_id in page_ids
            for field in fields
        ]

    def field_tensor(self, field_id: str, dtype):
        """Return a stable typed tensor view for one planned field."""
        import torch

        field = self.plan.field(field_id)
        if torch.empty((), dtype=dtype).element_size() != field.element_size:
            raise ValueError(f"field {field_id!r}: dtype itemsize does not match plan")
        group = self.plan.group(field.group_id)
        base_bytes = self.field_page_byte_offset(field_id, 0)
        typed = self.backing.view(dtype)
        element_strides = []
        stride = 1
        for extent in reversed(field.shape):
            element_strides.append(stride)
            stride *= extent
        return typed.as_strided(
            (group.page_count, *field.shape),
            (
                field.page_stride_bytes // field.element_size,
                *reversed(element_strides),
            ),
            base_bytes // field.element_size,
        )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _solve_packing(fields, *, max_packing):
    """Cache blocks per LCM block for each group, from per-field ratios.

    A plane's byte size is ``max(K_g * payload_g)`` over its tenants, so a
    tenant's page stride is ``plane_bytes / K_g``. Fields marked
    ``exact_page_stride`` need that to land exactly on their payload, which
    pins the ratio between every pair of exact tenants sharing a plane::

        K_g * payload_g == K_h * payload_h   =>   K_g : K_h = payload_h : payload_g

    A plane holding a single exact tenant constrains nothing (its stride is
    ``K * payload / K`` for any K), so only pairs of exact tenants pin a ratio.
    The ratios form a graph over groups; each connected component is solved
    independently and scaled down to the smallest integers. Groups no constraint
    touches are left out for the caller to size.

    Tolerant fields are ignored here, but they are not free: a plane takes a max
    over its tenants, so a tolerant field wider than ``K * payload`` of an exact
    co-tenant inflates the plane and overshoots that tenant's stride. Scaling
    cannot fix it (both sides grow together) -- it means the field-to-plane
    assignment is wrong, and ``_check_exact_page_strides`` says so.

    Sizing K from the whole-group byte ratio instead (the earlier rule) breaks
    as soon as one group's fields span several planes: on Qwen3.5 it halved the
    history packing and doubled the KV footprint.

    Near-coprime payloads make the exact solution explode (1000 and 2048 on one
    plane want K=256 and K=125), so this returns ``None`` above ``max_packing``
    and lets the caller keep its heuristic. Whether a field that truly needed an
    exact stride went unserved is settled afterwards by
    ``_check_exact_page_strides``, against the strides actually planned.

    Args:
        fields: ``LcmFieldSpec`` values, already validated.
        max_packing: Give up above this many cache blocks per LCM block.

    Returns:
        dict[str, int] | None: group id -> cache blocks per LCM block, covering
        only groups a ratio actually constrains, or None when no solution
        within ``max_packing`` exists.
    """
    exact_by_plane: dict[str, dict[str, int]] = {}
    groups = set()
    for field in fields:
        groups.add(field.group_id)
        if field.exact_page_stride:
            plane = exact_by_plane.setdefault(field.plane_id, {})
            plane[field.group_id] = plane.get(field.group_id, 0) + field.payload_bytes

    # ratio[g] = (num, den) expressing K_g as a multiple of its component's root.
    root = {group_id: group_id for group_id in groups}
    ratio = {group_id: (1, 1) for group_id in groups}

    def resolve(group_id):
        """Return (component root, K_group / K_root as an exact fraction)."""
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
                    return None  # contradictory ratios; no exact solution exists
                continue
            # Link the components so K_first * first_bytes == K_other * other_bytes.
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
            continue  # unconstrained; the caller keeps its density heuristic
        scale = 1
        for _, (_, den) in component_members:
            scale = scale * den // math.gcd(scale, den)
        counts = {
            group_id: num * scale // den for group_id, (num, den) in component_members
        }
        smallest = 0
        for count in counts.values():
            smallest = math.gcd(smallest, count)
        packing.update({g: c // smallest for g, c in counts.items()})

    if any(count > max_packing for count in packing.values()):
        return None
    return packing


def _packing_by_group_ratio(raw_by_group):
    """Fallback packing: scale each group against the widest one.

    Used when no exact-stride solution fits within the packing limit. It keeps
    every group at or below the widest group's block size, which bounds the LCM
    block, at the cost of padding narrow groups.
    """
    largest_payload = max(raw_by_group.values())
    return {
        group_id: max(1, largest_payload // raw_by_group[group_id])
        for group_id in raw_by_group
    }


def _check_exact_page_strides(fields, plane_bytes, packing):
    """Fail closed when a field that needs an exact page stride did not get one.

    Kernels for these fields walk pages by an implicit payload-sized stride, so
    a gap does not crash -- it silently reads the wrong rows. Two things cause
    it: field payloads whose ratio has no affordable packing, and a co-tenant on
    the same plane (often a tolerant one) wide enough to set ``plane_bytes``
    itself. Both mean the field-to-plane assignment needs revisiting.
    """
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
    """Plan a plane-major arena whose fields overlay across cache groups.

    A plane is a stable tensor address domain. Fields from different groups may
    share one plane because a physical parent is bound to only one group at a
    time; fields of the same group are packed side by side within its child
    slot.
    """
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
            fields, key=lambda field: (field.plane_id, field.group_id, field.field_id)
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
    # Groups tied together by an exact-stride ratio must use that ratio; the
    # rest keep the density heuristic (scale against the widest group).
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

    # A group containing only explicit-stride fields does not pin a ratio.
    # Fit as many of its CacheBlocks as possible into the planes established
    # by exact groups, but only at a K that partitions every occupied plane
    # cleanly. This keeps child slots inside their LCM parent instead of
    # letting integer division drift them across parent boundaries.
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
                fixed_plane_bytes[plane_id] = _align_up(fixed_required, fixed_alignment)

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

    # C++ placement divides one physical parent into K_g equal logical slots.
    # Independent planes can otherwise produce a total byte size that is not
    # divisible by every K_g. Keep field strides unchanged and reserve only the
    # smallest aligned tail needed to make that parent geometry integral.
    parent_alignment = alignment
    for count in packing.values():
        parent_alignment = parent_alignment // math.gcd(parent_alignment, count) * count
    lcm_block_bytes = _align_up(sum(plane_bytes.values()), parent_alignment)
    if lcm_block_bytes > max_lcm_block_bytes:
        raise ValueError(
            f"LCM block size {lcm_block_bytes} exceeds limit " f"{max_lcm_block_bytes}"
        )
    # Parent 0 backs kernel page 0 and is never schedulable. It is still a real
    # allocation, so the byte budget must cover it as well as every usable
    # parent.
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


@dataclass(frozen=True)
class ComponentSpec:
    group_id: str
    layer: int
    component: str
    bytes_per_slot: int  # linear in P; 0 for constant components
    const_bytes: int  # constant in P; 0 for linear components


@dataclass(frozen=True)
class BlockGeometry:
    block_size: int
    block_bytes: int
    num_blocks: int = 0  # filled by the planners from the memory budget


def occurrence_index(labels):
    """Within-label occurrence index per position.

    Args:
        labels: Iterable of hashable labels (e.g. per-layer type strings).

    Returns:
        list[int]: out[i] == number of earlier positions carrying the same
        label as position i — the slab pairing order shared by
        components_from_layers and the KV pool's slab layout.
    """
    counts: dict = {}
    out: list[int] = []
    for label in labels:
        idx = counts.get(label, 0)
        counts[label] = idx + 1
        out.append(idx)
    return out


def state_const_bytes(conv_shape, conv_dtype, ssm_shape, ssm_dtype):
    """Constant per-page state row bytes of one GDN/mamba2 state layer.

    Args:
        conv_shape / ssm_shape: Per-layer state tensor shapes (the configs'
            mamba2_cache_params conv and temporal shapes).
        conv_dtype / ssm_dtype: Matching dtypes (anything with ``itemsize``).

    Returns:
        dict[str, int]: {"conv": bytes, "ssm": bytes} — the exact
        ``state_const_bytes`` mapping components_from_layers /
        equalized_block_size consume (insertion order = row_offset order).
    """
    return {
        "conv": math.prod(conv_shape) * conv_dtype.itemsize,
        "ssm": math.prod(ssm_shape) * ssm_dtype.itemsize,
    }


def hybrid_lcm_field_specs(
    *,
    layer_types,
    layer_group_ids,
    logical_block_tokens,
    kv_shape,
    kv_element_size,
    conv_shape,
    conv_element_size,
    ssm_shape,
    ssm_element_size,
):
    """Build the per-unit planes of a repeated hybrid layer unit.

    Two planes per unit. History K shares one with recurrent SSM, history V the
    other with recurrent conv. Both K/V and SSM are addressed by an implicit
    payload-sized page stride, and pairing K with SSM lets the planner size the
    history packing off their byte ratio, so all three land on exact strides.
    Conv is the only tolerant field (its kernel reads stride(0) at launch), so
    it rides the V plane and absorbs whatever slack is left.
    """
    if len(layer_types) != len(layer_group_ids):
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but layer_group_ids "
            f"has {len(layer_group_ids)}"
        )
    if tuple(kv_shape)[0] != logical_block_tokens:
        raise ValueError("kv_shape must start with logical_block_tokens")

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, (label, group_id) in enumerate(zip(layer_types, layer_group_ids)):
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        if label in STATE_LAYER_TYPES:
            fields.extend(
                (
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.ssm",
                        f"unit.{unit}.a",
                        tuple(ssm_shape),
                        ssm_element_size,
                    ),
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.conv",
                        f"unit.{unit}.b",
                        tuple(conv_shape),
                        conv_element_size,
                        # causal_conv1d reads stride(0) at launch, so conv can
                        # absorb the slack left by history V.
                        exact_page_stride=False,
                    ),
                )
            )
        else:
            fields.extend(
                (
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k",
                        f"unit.{unit}.a",
                        tuple(kv_shape),
                        kv_element_size,
                    ),
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v",
                        f"unit.{unit}.b",
                        tuple(kv_shape),
                        kv_element_size,
                    ),
                )
            )
    return tuple(fields)


def inkling_lcm_field_specs(
    *,
    layer_group_ids,
    logical_block_tokens,
    layer_kv_heads,
    head_dim,
    kv_element_size,
    hidden_size,
    checkpoint_rows,
    kvconv_element_size,
    hiddenconv_element_size,
    kv_scale_block_size=0,
    kv_scale_element_size=0,
):
    """Describe Inkling attention pages and ShortConv boundary checkpoints.

    One occurrence of every attention group shares the same physical unit.
    K-side fields share one plane in that unit and V-side fields share the
    other. The two checkpoint groups aggregate every layer's K/V and
    ATTN/MLP streams respectively. FP8 hidden checkpoints use slack in those
    attention planes; BF16 checkpoints use dedicated planes so they cannot
    change the exact page stride required by attention kernels.

    Args:
        layer_group_ids: Per-layer attention cache group.
        logical_block_tokens: Shared scheduler/hash CacheBlock size.
        layer_kv_heads: Served KV heads per rank for every layer.
        head_dim: Width of one KV head.
        kv_element_size: Bytes per stored K/V element.
        hidden_size: ATTN/MLP ShortConv channel width.
        checkpoint_rows: Rows saved at each boundary, normally ``W - 1``.
        kvconv_element_size: Bytes per K/V ShortConv checkpoint element.
        hiddenconv_element_size: Bytes per ATTN/MLP checkpoint element.
        kv_scale_block_size: Data elements covered by one KV scale value, or
            zero when the cache has no scale sidecar.
        kv_scale_element_size: Bytes per scale value, or zero with no sidecar.

    Returns:
        Tuple of :class:`LcmFieldSpec` values for the shared LCM planner.
    """
    if len(layer_group_ids) != len(layer_kv_heads):
        raise ValueError(
            f"layer_group_ids has {len(layer_group_ids)} entries but "
            f"layer_kv_heads has {len(layer_kv_heads)}"
        )
    if bool(kv_scale_block_size) != bool(kv_scale_element_size):
        raise ValueError(
            "kv_scale_block_size and kv_scale_element_size must both be zero "
            "or both be positive"
        )
    if kv_scale_block_size and head_dim % kv_scale_block_size:
        raise ValueError("head_dim must be divisible by kv_scale_block_size")
    if kv_scale_block_size and (
        logical_block_tokens % 128 or head_dim != 128 or kv_scale_block_size != 32
    ):
        raise ValueError(
            "Inkling MXFP8 scale fields require P divisible by 128, "
            "head_dim 128 and scale block size 32"
        )

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, (group_id, kv_heads) in enumerate(
        zip(layer_group_ids, layer_kv_heads)
    ):
        unit = occurrences.get(group_id, 0)
        occurrences[group_id] = unit + 1
        k_plane = f"unit.{unit}.k"
        v_plane = f"unit.{unit}.v"
        if hiddenconv_element_size > 1:
            hidden_k_plane = f"unit.{unit}.hidden_k"
            hidden_v_plane = f"unit.{unit}.hidden_v"
        else:
            hidden_k_plane = k_plane
            hidden_v_plane = v_plane
        kv_shape = (logical_block_tokens, kv_heads, head_dim)
        kvconv_shape = (checkpoint_rows, kv_heads * head_dim)
        hiddenconv_shape = (checkpoint_rows, hidden_size)
        fields.extend(
            (
                LcmFieldSpec(
                    group_id,
                    f"layer.{layer_id}.k",
                    k_plane,
                    kv_shape,
                    kv_element_size,
                ),
                LcmFieldSpec(
                    group_id,
                    f"layer.{layer_id}.v",
                    v_plane,
                    kv_shape,
                    kv_element_size,
                ),
                LcmFieldSpec(
                    "kvconv",
                    f"layer.{layer_id}.kvconv_k",
                    k_plane,
                    kvconv_shape,
                    kvconv_element_size,
                    exact_page_stride=False,
                ),
                LcmFieldSpec(
                    "kvconv",
                    f"layer.{layer_id}.kvconv_v",
                    v_plane,
                    kvconv_shape,
                    kvconv_element_size,
                    exact_page_stride=False,
                ),
                LcmFieldSpec(
                    "hiddenconv",
                    f"layer.{layer_id}.attnconv",
                    hidden_k_plane,
                    hiddenconv_shape,
                    hiddenconv_element_size,
                    exact_page_stride=False,
                ),
                LcmFieldSpec(
                    "hiddenconv",
                    f"layer.{layer_id}.mlpconv",
                    hidden_v_plane,
                    hiddenconv_shape,
                    hiddenconv_element_size,
                    exact_page_stride=False,
                ),
            )
        )
        if kv_scale_block_size:
            scale_dim = head_dim // kv_scale_block_size
            scale_shape = (
                kv_heads,
                logical_block_tokens // 128,
                32,
                scale_dim,
                scale_dim,
            )
            fields.extend(
                (
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.k_scale",
                        f"unit.{unit}.k_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                    LcmFieldSpec(
                        group_id,
                        f"layer.{layer_id}.v_scale",
                        f"unit.{unit}.v_scale",
                        scale_shape,
                        kv_scale_element_size,
                    ),
                )
            )
    return tuple(fields)


def components_from_layers(*, layer_types, kv_bytes_per_slot, state_const_bytes):
    """Per-layer ComponentSpecs: history layers carry one linear kv component;
    state layers one constant component per state tensor. Layer index is the
    within-group occurrence count (the slab pairing order). State component
    order (hence row_offset order downstream) follows state_const_bytes
    insertion order."""
    comps: list[ComponentSpec] = []
    for label, idx in zip(layer_types, occurrence_index(layer_types)):
        if label in STATE_LAYER_TYPES:
            for name, nbytes in state_const_bytes.items():
                comps.append(ComponentSpec(label, idx, name, 0, nbytes))
        else:
            comps.append(ComponentSpec(label, idx, "kv", kv_bytes_per_slot, 0))
    return comps


def _row_demands(components):
    """Per-(group, layer) row: (linear bytes-per-slot sum, constant bytes sum)."""
    rows = defaultdict(lambda: [0, 0])
    for c in components:
        row = rows[(c.group_id, c.layer)]
        row[0] += c.bytes_per_slot
        row[1] += c.const_bytes
    return rows


def solve_page_geometry(components, *, block_size, alignment):
    """Smallest P >= block_size (multiple of `alignment` when inflated)
    such that the widest linear row covers the widest constant row."""
    rows = _row_demands(components).values()
    # NOTE: a row mixing linear and constant components is not needed by any
    # known model; reject it so the math stays honest.
    for lin, const in rows:
        if lin > 0 and const > 0:
            raise ValueError("a row must be all-linear or all-constant")
    max_linear = max((lin for lin, _ in rows), default=0)
    max_const = max((const for _, const in rows), default=0)
    if max_const > 0:
        if max_linear == 0:
            raise ValueError("constant components need a linear row to size P against")
        needed = -(-max_const // max_linear)  # exact integer ceil
        if needed > block_size:
            block_size = alignment * math.ceil(needed / alignment)
    block_bytes = max(max_linear * block_size, max_const)
    return BlockGeometry(block_size=block_size, block_bytes=block_bytes)


def equalized_block_size(
    *,
    layer_types,
    kv_bytes_per_slot,
    state_const_bytes,
    block_size,
    alignment=None,
):
    """Effective P for a state-hybrid profile: `block_size` when the
    widest KV row already covers the widest constant state row, else the
    smallest multiple of `alignment` that does. `alignment` defaults to the
    original `block_size` (the attention backend's page granularity —
    no backend declares a finer one), so the inflated P stays a multiple of
    the configured block size. Pure wrapper over components_from_layers +
    solve_page_geometry so the config-level equalization decision and its
    tests share one implementation."""
    comps = components_from_layers(
        layer_types=layer_types,
        kv_bytes_per_slot=kv_bytes_per_slot,
        state_const_bytes=state_const_bytes,
    )
    geo = solve_page_geometry(
        comps,
        block_size=block_size,
        alignment=alignment if alignment is not None else block_size,
    )
    return geo.block_size


@dataclass(frozen=True)
class LayerBinding:
    slot: int
    group_id: str
    layer: int
    component: str
    nbytes_per_block: int
    row_offset: int  # byte offset of this component within its (group, layer) page row


@dataclass(frozen=True)
class TensorPlan:
    name: str
    nbytes: int
    bindings: tuple[LayerBinding, ...]


@dataclass(frozen=True)
class FlatMemoryPlan:
    geometry: BlockGeometry
    tensors: tuple[TensorPlan, ...]


def plan_component_tensors(
    components, *, block_size, budget_bytes, reserved_bytes_per_block=0
):
    """One tensor per ComponentSpec, honestly sized: row bytes = that
    component's per-block bytes, num_blocks = budget // (sum of all rows +
    reserved_bytes_per_block). No cross-component packing, no padding —
    every tensor keeps today's standalone-slab shape, so kernels, CUDA
    graphs and the host mirror stay untouched. reserved_bytes_per_block
    carries co-resident rows outside these components (the MTP draft
    pool's KV rows ride the same block-id space). Under this planner each
    component is its own slot, in input order."""
    row_bytes = [c.bytes_per_slot * block_size + c.const_bytes for c in components]
    per_block = sum(row_bytes) + reserved_bytes_per_block
    num_blocks = budget_bytes // per_block
    if num_blocks <= 1:
        raise ValueError("budget too small for one usable block")
    geo = BlockGeometry(
        block_size=block_size, block_bytes=per_block, num_blocks=num_blocks
    )
    tensors = tuple(
        TensorPlan(
            name=f"flat_{c.group_id}_{c.layer}_{c.component}",
            nbytes=num_blocks * nbytes,
            bindings=(LayerBinding(i, c.group_id, c.layer, c.component, nbytes, 0),),
        )
        for i, (c, nbytes) in enumerate(zip(components, row_bytes))
    )
    return FlatMemoryPlan(geometry=geo, tensors=tensors)


def plan_tensors(components, *, block_size, alignment, budget_bytes):
    """Pair slot j with the j-th layer of every group over one page-id space.
    Each slot tensor is sized by its own packed row (the sum of its bindings'
    per-block bytes); geometry.block_bytes accounts one block's total across
    all slots."""
    geo = solve_page_geometry(components, block_size=block_size, alignment=alignment)
    layers_by_group: dict[str, list[int]] = {}
    for c in components:
        layers = layers_by_group.setdefault(c.group_id, [])
        if c.layer not in layers:
            layers.append(c.layer)
    num_slots = max(len(v) for v in layers_by_group.values())

    slot_bindings: list[tuple[LayerBinding, ...]] = []
    for slot in range(num_slots):
        bindings = []
        for gid, layers in layers_by_group.items():
            if slot >= len(layers):
                continue
            layer = layers[slot]
            row_offset = 0
            for c in components:
                if c.group_id != gid or c.layer != layer:
                    continue
                nbytes = c.bytes_per_slot * geo.block_size + c.const_bytes
                bindings.append(
                    LayerBinding(slot, gid, layer, c.component, nbytes, row_offset)
                )
                row_offset += nbytes
        slot_bindings.append(tuple(bindings))
    slot_rows = [sum(b.nbytes_per_block for b in bs) for bs in slot_bindings]

    num_blocks = budget_bytes // sum(slot_rows)
    if num_blocks <= 1:
        raise ValueError("budget too small for one usable block per slot")
    geo = replace(geo, block_bytes=sum(slot_rows), num_blocks=num_blocks)

    tensors = tuple(
        TensorPlan(
            name=f"flat_slab_{slot}",
            nbytes=num_blocks * slot_rows[slot],
            bindings=bindings,
        )
        for slot, bindings in enumerate(slot_bindings)
    )
    return FlatMemoryPlan(geometry=geo, tensors=tensors)
