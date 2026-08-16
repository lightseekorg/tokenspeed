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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        CacheGroupSpec,
    )

# Planner/runtime limits, not model layout inputs.
_MAX_LCM_BLOCK_BYTES = (1 << 63) - 1
_MAX_KERNEL_PAGE_ID = (1 << 31) - 1

# Byte width per cache dtype name. This module stays torch-free (pure integer
# geometry, and the plan travels the PD wire as JSON), so dtypes are named by
# string and their widths live in this table.
_CACHE_DTYPE_BYTES = {
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8_e8m0fnu": 1,
    "bfloat16": 2,
    "float16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}

# Elementwise scatter (Tensor.index_put) has no fp8 kernel, so pools that
# write KV that way view the bytes as uint8 instead. Pools whose writes go
# through a dtype-aware kernel (MXFP8, via store_sf_interleaved and
# quantize_store_kv_mxfp8) keep the fp8 view.
_INDEX_PUT_UNSUPPORTED = ("float8_e5m2", "float8_e4m3fn")


def cache_dtype_name(dtype) -> str:
    """Return the plan-facing name of the dtype the arena views bytes as.

    Args:
        dtype: A ``torch.dtype`` or an already-normalized dtype name.

    Returns:
        The dtype name to record in the plan.

    Raises:
        ValueError: The dtype has no cache representation.
    """
    name = str(dtype).removeprefix("torch.")
    if name not in _CACHE_DTYPE_BYTES:
        raise ValueError(f"unsupported cache dtype {name!r}")
    return name


def scatter_stored_dtype_name(dtype) -> str:
    """Plan dtype for a field whose pool writes it with elementwise scatter.

    fp8 collapses to ``uint8`` because ``index_put`` has no fp8 kernel --
    the same substitution the pools applied before the plan owned dtypes,
    so the PD wire string is unchanged.

    Args:
        dtype: A ``torch.dtype`` or an already-normalized dtype name.

    Returns:
        The dtype name to record in the plan.
    """
    name = cache_dtype_name(dtype)
    return "uint8" if name in _INDEX_PUT_UNSUPPORTED else name


def cache_dtype_bytes(dtype_name: str) -> int:
    """Return the byte width of one element of a plan dtype name.

    Args:
        dtype_name: A name produced by :func:`cache_dtype_name`.

    Returns:
        Bytes per element.

    Raises:
        ValueError: The name is not a known cache dtype.
    """
    try:
        return _CACHE_DTYPE_BYTES[dtype_name]
    except KeyError:
        raise ValueError(f"unsupported cache dtype {dtype_name!r}") from None


# Token span of one interleaved mxfp8 KV-scale tile, and the head-dim block a
# single e8m0 scale covers. Both are properties of the fused FP8 attention
# kernels, so the shape they imply is built here once.
MXFP8_KV_SCALE_TILE_TOKENS = 128
MXFP8_SCALE_BLOCK_SIZE = 32


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
    # The dtype the arena views these bytes as -- the plan is the single
    # source, so no consumer has to supply it a second time.
    dtype: str
    field_offset_bytes: int
    page_stride_bytes: int

    def __post_init__(self) -> None:
        cache_dtype_bytes(self.dtype)

    @property
    def element_size(self) -> int:
        return cache_dtype_bytes(self.dtype)

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
    """One field a cache group declares, without naming that group.

    A field belongs to whichever group lists it in the ``{spec: fields}``
    mapping handed to :func:`pack`, so a group id is written
    once -- in its spec -- instead of once per field.
    """

    field_id: str
    plane_id: str
    shape: tuple[int, ...]
    # The dtype the arena views these bytes as. Recipes know this when they
    # build the spec, so it travels with the geometry instead of being
    # re-supplied at bind time.
    dtype: str
    # True when the field's kernel walks pages by an implicit payload-sized
    # stride. False when the kernel consumes the tensor's runtime stride.
    exact_page_stride: bool = True
    # Some kernels accept padded pages but still require the runtime page
    # stride to satisfy an alignment constraint (for example, a TMA row
    # stride). The planner applies this in bytes after group packing.
    page_stride_alignment_bytes: int = 1

    def __post_init__(self) -> None:
        cache_dtype_bytes(self.dtype)

    @property
    def element_size(self) -> int:
        return cache_dtype_bytes(self.dtype)

    @property
    def payload_bytes(self) -> int:
        return math.prod(self.shape) * self.element_size


def mxfp8_kv_scale_fields(
    *,
    layer_id: int,
    occurrence: int,
    kv_heads: int,
    head_dim: int,
    prefix_granularity: int,
) -> tuple[CacheFieldSpec, ...]:
    """The k_scale/v_scale planes of one mxfp8 KV layer.

    One kernel layout, one definition: the fused FP8 attention kernels read
    scales as ``(num_ids, heads, tiles, 32, sf, sf)`` where a tile spans
    ``MXFP8_KV_SCALE_TILE_TOKENS`` tokens, so the per-page shape a recipe
    declares is fixed by the page's token span and the head count. Recipes
    differ in how they arrive at those two numbers, never in the shape.
    """
    if prefix_granularity % MXFP8_KV_SCALE_TILE_TOKENS or kv_heads <= 0:
        raise ValueError(
            "mxfp8 KV scales need a page span that is a positive multiple of "
            f"{MXFP8_KV_SCALE_TILE_TOKENS} and at least one KV head, got "
            f"{prefix_granularity} and {kv_heads}"
        )
    if head_dim % MXFP8_SCALE_BLOCK_SIZE:
        raise ValueError(
            f"mxfp8 head_dim must be a multiple of {MXFP8_SCALE_BLOCK_SIZE}, "
            f"got {head_dim}"
        )
    scale_dim = head_dim // MXFP8_SCALE_BLOCK_SIZE
    shape = (
        kv_heads,
        prefix_granularity // MXFP8_KV_SCALE_TILE_TOKENS,
        32,
        scale_dim,
        scale_dim,
    )
    dtype = cache_dtype_name(torch.float8_e8m0fnu)
    return tuple(
        CacheFieldSpec(
            f"layer.{layer_id}.{plane}_scale",
            f"unit.{occurrence}.{plane}_scale",
            shape,
            dtype,
        )
        for plane in ("k", "v")
    )


@dataclass(frozen=True)
class CacheLayout:
    """Capacity-independent byte geometry for one LCM block."""

    prefix_granularity: int
    lcm_block_bytes: int
    group_packing: tuple[tuple[str, int], ...]
    plane_bytes: tuple[tuple[str, int], ...]
    fields: tuple[CacheFieldLayout, ...]

    def bind(self, num_lcm_blocks: int) -> CacheMemoryPlan:
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


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _solve_packing(fields):
    """Derive per-group cache blocks per LCM block from exact field ratios.

    ``fields`` are ``(group_id, CacheFieldSpec)`` pairs.
    """
    exact_by_plane: dict[str, dict[str, int]] = {}
    groups = set()
    for group_id, field in fields:
        groups.add(group_id)
        if field.exact_page_stride:
            plane = exact_by_plane.setdefault(field.plane_id, {})
            plane[group_id] = plane.get(group_id, 0) + field.payload_bytes

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
    """``fields`` are ``(group_id, CacheFieldSpec)`` pairs."""
    for group_id, field in fields:
        if not field.exact_page_stride:
            continue
        stride = plane_bytes[field.plane_id] // packing[group_id]
        if stride != field.payload_bytes:
            widest = max(
                (
                    (other.payload_bytes * packing[other_group], other.field_id)
                    for other_group, other in fields
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


def pack(
    groups: Sequence[tuple[CacheGroupSpec, tuple[CacheFieldSpec, ...]]],
    *,
    prefix_granularity,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
    alignment=1,
    max_padding_fraction=0.25,
):
    """Pack declared cache groups into one capacity-independent LCM block.

    The second stage of the pipeline: ``group`` says which groups exist and
    what bytes they need, this says how those bytes sit inside one physical
    parent (plane sizes, per-field offsets and page strides), and
    :meth:`CacheLayout.bind` later multiplies it out by a parent count.

    Args:
        groups: ``(group spec, its declared fields)`` pairs. The pairing is
            the whole point: a group id is spelled once, in its spec, so the
            planned group set equals the declared one by construction.
        prefix_granularity: Scheduler-wide identity grain in tokens.
        cache_blocks_per_lcm_block: How many of each group's CacheBlocks share
            one parent. A recipe that owns this policy pins every group and
            nothing is derived here; groups left unpinned are derived from
            byte ratios plus the exact-page-stride constraints their fields
            impose.
        alignment: Byte alignment for plane sizes.
        max_padding_fraction: Padding budget per group.

    Returns:
        The packed :class:`CacheLayout`.

    Raises:
        ValueError: No group declared, duplicate field ids, invalid geometry,
            or packing pinned for a group outside the declaration.
    """
    if prefix_granularity <= 0:
        raise ValueError("prefix_granularity must be > 0")
    if alignment <= 0:
        raise ValueError("alignment must be > 0")
    if max_padding_fraction < 0:
        raise ValueError("max_padding_fraction must be >= 0")

    groups = tuple(groups)
    declared_ids = [spec.group_id for spec, _ in groups]
    if len(declared_ids) != len(set(declared_ids)):
        raise ValueError(f"cache group declared more than once: {declared_ids}")
    empty = sorted(spec.group_id for spec, fields in groups if not fields)
    if empty:
        raise ValueError(
            f"cache groups {empty} declare no fields; a group with no bytes "
            "cannot be addressed"
        )
    # Each field carries its declaring group id from here on: the planner
    # works in (group_id, field) pairs so nothing can rebind a field.
    ordered_fields = tuple(
        sorted(
            ((spec.group_id, field) for spec, fields in groups for field in fields),
            key=lambda entry: (entry[1].plane_id, entry[0], entry[1].field_id),
        )
    )
    if not ordered_fields:
        raise ValueError("at least one cache group is required")
    if len({field.field_id for _, field in ordered_fields}) != len(ordered_fields):
        raise ValueError("cache field ids must be unique")

    raw_by_group: dict[str, int] = {}
    for group_id, field in ordered_fields:
        if not group_id or not field.field_id or not field.plane_id:
            raise ValueError("cache group id, field id and plane id must be non-empty")
        if (
            not field.shape
            or any(extent <= 0 for extent in field.shape)
            or isinstance(field.page_stride_alignment_bytes, bool)
            or not isinstance(field.page_stride_alignment_bytes, int)
            or field.page_stride_alignment_bytes <= 0
        ):
            raise ValueError(f"cache field {field.field_id!r} has invalid geometry")
        raw_by_group[group_id] = raw_by_group.get(group_id, 0) + field.payload_bytes

    ordered_group_ids = tuple(sorted(raw_by_group))
    pinned = dict(cache_blocks_per_lcm_block or {})
    unknown = set(pinned) - set(ordered_group_ids)
    if unknown:
        raise ValueError(
            "cache_blocks_per_lcm_block names groups outside the plan: "
            f"{sorted(unknown)}"
        )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in pinned.values()
    ):
        raise ValueError("cache group packing must be a positive integer")
    if set(pinned) == set(ordered_group_ids):
        # The recipe owns the policy for every group; deriving one here only
        # to overwrite it would be wasted work.
        packing = pinned
    else:
        packing = _packing_by_group_ratio(raw_by_group)
        constrained = _solve_packing(ordered_fields)
        if constrained is not None:
            packing.update(constrained)
        packing.update(pinned)

    plane_fields: dict[str, dict[str, list[CacheFieldSpec]]] = {}
    for group_id, field in ordered_fields:
        plane_fields.setdefault(field.plane_id, {}).setdefault(group_id, []).append(
            field
        )

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
        group_id for group_id, field in ordered_fields if field.exact_page_stride
    }
    flexible_groups = set(ordered_group_ids) - exact_groups
    if flexible_groups and not pinned:
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
    for group_id, field in ordered_fields:
        field_layouts.append(
            CacheFieldLayout(
                group_id=group_id,
                field_id=field.field_id,
                plane_id=field.plane_id,
                shape=field.shape,
                dtype=field.dtype,
                field_offset_bytes=field_offsets[field.field_id],
                page_stride_bytes=plane_bytes[field.plane_id] // packing[group_id],
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
