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

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from tokenspeed.runtime.configs.flat_cache_runtime import require_positive_int

CacheFamily = Literal["history", "state"]
CacheRetention = Literal["full_history", "sliding_window"]
CacheTransferPolicy = Literal["full_suffix", "latest_snapshot"]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class CacheComponentSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    bytes_per_token: int
    constant_bytes: int
    alignment: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("component name must be non-empty")
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x <= 0
                for x in self.shape
            )
        ):
            raise ValueError(
                f"component {self.name}: shape must contain positive integers"
            )
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(f"component {self.name}: dtype must be torch.dtype")
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x < 0
            for x in (self.bytes_per_token, self.constant_bytes)
        ):
            raise ValueError(
                f"component {self.name}: byte counts must be non-negative integers"
            )
        linear, constant = self.bytes_per_token > 0, self.constant_bytes > 0
        if linear == constant:
            raise ValueError(
                f"component {self.name}: exactly one of bytes_per_token and constant_bytes must be positive"
            )
        expected = math.prod(self.shape) * self.dtype.itemsize
        actual = self.bytes_per_token if linear else self.constant_bytes
        if actual != expected:
            raise ValueError(
                f"component {self.name}: declared bytes {actual} do not match {expected}"
            )
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or not _is_power_of_two(self.alignment)
        ):
            raise ValueError(f"component {self.name}: alignment must be a power of two")
        if self.alignment < self.dtype.itemsize or self.alignment % self.dtype.itemsize:
            raise ValueError(
                f"component {self.name}: alignment {self.alignment} is incompatible with dtype item size {self.dtype.itemsize}"
            )


@dataclass(frozen=True)
class LayerCacheSpec:
    layer_id: int
    family: CacheFamily
    retention: CacheRetention
    transfer_policy: CacheTransferPolicy
    group_id_prefix: str
    group_order: int
    compatibility_key: Hashable
    preferred_block_size: int
    kernel_alignment: int
    components: tuple[CacheComponentSpec, ...]
    sliding_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_id, bool)
            or not isinstance(self.layer_id, int)
            or self.layer_id < 0
        ):
            raise ValueError(
                f"layer_id must be a non-negative integer, got {self.layer_id!r}"
            )
        if self.family not in ("history", "state"):
            raise ValueError(
                f"layer {self.layer_id}: unsupported family {self.family!r}"
            )
        if self.retention not in ("full_history", "sliding_window"):
            raise ValueError(
                f"layer {self.layer_id}: unsupported retention {self.retention!r}"
            )
        if self.transfer_policy not in ("full_suffix", "latest_snapshot"):
            raise ValueError(
                f"layer {self.layer_id}: unsupported transfer_policy "
                f"{self.transfer_policy!r}"
            )
        window = self.sliding_window_tokens
        if self.retention == "full_history":
            if window is not None:
                raise ValueError(
                    f"full_history layer {self.layer_id} requires sliding_window_tokens=None"
                )
        elif isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError(
                f"sliding_window layer {self.layer_id} requires positive sliding_window_tokens"
            )
        if not self.group_id_prefix:
            raise ValueError(
                f"layer {self.layer_id}: group_id_prefix must be non-empty"
            )
        if (
            isinstance(self.group_order, bool)
            or not isinstance(self.group_order, int)
            or self.group_order < 0
        ):
            raise ValueError(
                f"group_order must be a non-negative integer, got {self.group_order!r}"
            )
        if self.compatibility_key is None:
            raise ValueError(
                f"layer {self.layer_id}: compatibility_key must not be None"
            )
        try:
            hash(self.compatibility_key)
        except TypeError as exc:
            raise ValueError(
                f"layer {self.layer_id}: compatibility_key must be hashable"
            ) from exc
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0
            for x in (self.preferred_block_size, self.kernel_alignment)
        ):
            raise ValueError(
                f"layer {self.layer_id}: preferred_block_size and kernel_alignment must be positive"
            )
        if not self.components:
            raise ValueError(f"layer {self.layer_id}: components must be non-empty")
        if any(
            not isinstance(component, CacheComponentSpec)
            for component in self.components
        ):
            raise ValueError(
                f"layer {self.layer_id}: components must be CacheComponentSpec instances"
            )
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError(
                f"layer {self.layer_id}: duplicate component names {names}"
            )
        if self.family == "history" and any(
            component.bytes_per_token <= 0 for component in self.components
        ):
            raise ValueError(
                f"history layer {self.layer_id} requires bytes_per_token components"
            )
        if self.family == "state" and any(
            component.constant_bytes <= 0 for component in self.components
        ):
            raise ValueError(
                f"state layer {self.layer_id} requires constant_bytes components"
            )


@dataclass(frozen=True)
class _GroupDraft:
    group_id: str
    family: CacheFamily
    retention: CacheRetention
    transfer_policy: CacheTransferPolicy
    layer_ids: tuple[int, ...]
    sliding_window_tokens: int | None


def _partition_signature(spec: LayerCacheSpec) -> tuple:
    return (
        spec.family,
        spec.retention,
        spec.transfer_policy,
        spec.sliding_window_tokens,
        spec.group_id_prefix,
        spec.group_order,
        spec.preferred_block_size,
        spec.kernel_alignment,
        spec.components,
    )


def _build_group_drafts(
    layer_specs: Sequence[LayerCacheSpec],
) -> tuple[_GroupDraft, ...]:
    if not layer_specs:
        raise ValueError("layer_specs must be non-empty")
    ordered = sorted(layer_specs, key=lambda spec: spec.layer_id)
    seen_layers: set[int] = set()
    partitions: dict[Hashable, list[LayerCacheSpec]] = {}
    signatures: dict[Hashable, tuple] = {}
    for spec in ordered:
        if spec.layer_id in seen_layers:
            raise ValueError(f"duplicate layer_id {spec.layer_id}")
        seen_layers.add(spec.layer_id)
        signature = _partition_signature(spec)
        if (
            spec.compatibility_key in signatures
            and signatures[spec.compatibility_key] != signature
        ):
            raise ValueError(
                f"compatibility partition {spec.compatibility_key!r} has "
                "conflicting family, retention, transfer_policy, prefix, "
                "group_order, or geometry"
            )
        signatures.setdefault(spec.compatibility_key, signature)
        partitions.setdefault(spec.compatibility_key, []).append(spec)
    ordered_partitions = sorted(
        partitions.values(), key=lambda rows: (rows[0].group_order, rows[0].layer_id)
    )
    target = min(len(rows) for rows in ordered_partitions)
    drafts: list[_GroupDraft] = []
    used_group_ids: set[str] = set()
    for rows in ordered_partitions:
        subgroup_count = (len(rows) + target - 1) // target
        base, remainder = divmod(len(rows), subgroup_count)
        sizes = [base + int(index < remainder) for index in range(subgroup_count)]
        cursor = 0
        for subgroup_index, size in enumerate(sizes):
            group_id = (
                rows[0].group_id_prefix
                if subgroup_count == 1
                else f"{rows[0].group_id_prefix}_{subgroup_index}"
            )
            if group_id in used_group_ids:
                raise ValueError(f"generated group id collision: {group_id!r}")
            used_group_ids.add(group_id)
            subgroup = rows[cursor : cursor + size]
            cursor += size
            drafts.append(
                _GroupDraft(
                    group_id,
                    rows[0].family,
                    rows[0].retention,
                    rows[0].transfer_policy,
                    tuple(spec.layer_id for spec in subgroup),
                    rows[0].sliding_window_tokens,
                )
            )
    return tuple(drafts)


@dataclass(frozen=True)
class ComponentBinding:
    name: str
    byte_offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class LayerBinding:
    layer_id: int
    group_id: str
    physical_slot: int
    components: tuple[ComponentBinding, ...]


@dataclass(frozen=True)
class CacheGroupPlan:
    group_id: str
    family: CacheFamily
    retention: CacheRetention
    transfer_policy: CacheTransferPolicy
    block_size: int
    slot_layer_ids: tuple[int | None, ...]
    sliding_window_tokens: int | None = None


@dataclass(frozen=True)
class PhysicalSlotPlan:
    physical_slot: int
    physical_page_bytes: int
    bindings: tuple[LayerBinding, ...]


@dataclass(frozen=True)
class _LayoutDraft:
    configured_block_size: int
    block_size: int
    physical_page_bytes: int
    groups: tuple[CacheGroupPlan, ...]
    physical_slots: tuple[PhysicalSlotPlan, ...]
    layer_bindings: tuple[LayerBinding, ...]


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _pack_layer(
    spec: LayerCacheSpec, block_size: int
) -> tuple[tuple[ComponentBinding, ...], int]:
    offset = 0
    bindings: list[ComponentBinding] = []
    for component in spec.components:
        offset = _align_up(offset, component.alignment)
        nbytes = (
            component.bytes_per_token * block_size
            if component.bytes_per_token
            else component.constant_bytes
        )
        bindings.append(
            ComponentBinding(
                component.name, offset, nbytes, component.shape, component.dtype
            )
        )
        offset += nbytes
    return tuple(bindings), offset


def _build_layout(
    layer_specs: Sequence[LayerCacheSpec], group_drafts: Sequence[_GroupDraft]
) -> _LayoutDraft:
    specs_by_id = {spec.layer_id: spec for spec in layer_specs}
    history_specs = [spec for spec in layer_specs if spec.family == "history"]
    if not history_specs:
        raise ValueError(
            "hybrid cache plan needs a token-history row to define the page"
        )
    configured = max(spec.preferred_block_size for spec in layer_specs)
    kernel_alignment = math.lcm(*(spec.kernel_alignment for spec in layer_specs))
    block_size = _align_up(configured, kernel_alignment)
    max_component_alignment = max(
        component.alignment for spec in layer_specs for component in spec.components
    )
    state_row_bytes = max(
        (
            _pack_layer(spec, block_size)[1]
            for spec in layer_specs
            if spec.family == "state"
        ),
        default=0,
    )
    while True:
        history_row_bytes = max(
            _pack_layer(spec, block_size)[1] for spec in history_specs
        )
        physical_page_bytes = _align_up(history_row_bytes, max_component_alignment)
        if physical_page_bytes >= state_row_bytes:
            break
        block_size += kernel_alignment

    slot_count = max(len(draft.layer_ids) for draft in group_drafts)
    groups = tuple(
        CacheGroupPlan(
            draft.group_id,
            draft.family,
            draft.retention,
            draft.transfer_policy,
            block_size,
            draft.layer_ids + (None,) * (slot_count - len(draft.layer_ids)),
            sliding_window_tokens=draft.sliding_window_tokens,
        )
        for draft in group_drafts
    )
    layer_bindings: list[LayerBinding] = []
    bindings_by_slot: list[list[LayerBinding]] = [[] for _ in range(slot_count)]
    for group in groups:
        for physical_slot, layer_id in enumerate(group.slot_layer_ids):
            if layer_id is None:
                continue
            components, row_bytes = _pack_layer(specs_by_id[layer_id], block_size)
            if row_bytes > physical_page_bytes:
                raise ValueError(
                    f"layer {layer_id} row bytes {row_bytes} exceed physical page bytes {physical_page_bytes}"
                )
            binding = LayerBinding(layer_id, group.group_id, physical_slot, components)
            layer_bindings.append(binding)
            bindings_by_slot[physical_slot].append(binding)
    physical_slots = tuple(
        PhysicalSlotPlan(slot, physical_page_bytes, tuple(bindings))
        for slot, bindings in enumerate(bindings_by_slot)
    )
    return _LayoutDraft(
        configured,
        block_size,
        physical_page_bytes,
        groups,
        physical_slots,
        tuple(sorted(layer_bindings, key=lambda binding: binding.layer_id)),
    )


@dataclass(frozen=True)
class FlatHybridCacheDiagnostics:
    configured_block_size: int
    effective_block_size: int
    physical_page_bytes: int
    bytes_per_page_set: int
    usable_pages: int
    null_pages: int
    physical_slot_count: int
    padding_binding_count: int
    total_allocated_bytes: int
    unused_budget_bytes: int
    group_layer_counts: tuple[tuple[str, int], ...]
    group_slot_counts: tuple[tuple[str, int], ...]
    theoretical_capacity_tokens: tuple[tuple[str, int], ...]
    component_padding_bytes: tuple[tuple[str, int], ...]
    component_padding_ratio: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FlatHybridCachePlan:
    block_size: int
    physical_page_bytes: int
    usable_pages: int
    token_capacity: int
    groups: tuple[CacheGroupPlan, ...]
    physical_slots: tuple[PhysicalSlotPlan, ...]
    layer_bindings: tuple[LayerBinding, ...]
    diagnostics: FlatHybridCacheDiagnostics


def _binding_row_bytes(binding: LayerBinding) -> int:
    return max(
        component.byte_offset + component.nbytes for component in binding.components
    )


def plan_flat_hybrid_cache(
    layer_specs: Sequence[LayerCacheSpec],
    *,
    cache_budget_bytes: int,
    minimum_usable_pages: int,
) -> FlatHybridCachePlan:
    require_positive_int("cache_budget_bytes", cache_budget_bytes)
    require_positive_int("minimum_usable_pages", minimum_usable_pages)

    drafts = _build_group_drafts(layer_specs)
    layout = _build_layout(layer_specs, drafts)
    bytes_per_page_set = len(layout.physical_slots) * layout.physical_page_bytes
    usable_pages = cache_budget_bytes // bytes_per_page_set - 1
    if usable_pages < minimum_usable_pages:
        raise ValueError(
            f"cache budget yields usable_pages={usable_pages}, below "
            f"minimum_usable_pages={minimum_usable_pages}; "
            f"bytes_per_page_set={bytes_per_page_set}"
        )

    allocated_bytes = (usable_pages + 1) * bytes_per_page_set
    bindings_by_group = {
        group.group_id: tuple(
            binding
            for binding in layout.layer_bindings
            if binding.group_id == group.group_id
        )
        for group in layout.groups
    }
    row_bytes_by_group = {
        group_id: max(_binding_row_bytes(binding) for binding in bindings)
        for group_id, bindings in bindings_by_group.items()
    }
    padding = tuple(
        (
            group.group_id,
            layout.physical_page_bytes - row_bytes_by_group[group.group_id],
        )
        for group in layout.groups
    )
    diagnostics = FlatHybridCacheDiagnostics(
        configured_block_size=layout.configured_block_size,
        effective_block_size=layout.block_size,
        physical_page_bytes=layout.physical_page_bytes,
        bytes_per_page_set=bytes_per_page_set,
        usable_pages=usable_pages,
        null_pages=1,
        physical_slot_count=len(layout.physical_slots),
        padding_binding_count=sum(
            layer_id is None
            for group in layout.groups
            for layer_id in group.slot_layer_ids
        ),
        total_allocated_bytes=allocated_bytes,
        unused_budget_bytes=cache_budget_bytes - allocated_bytes,
        group_layer_counts=tuple(
            (
                group.group_id,
                sum(layer_id is not None for layer_id in group.slot_layer_ids),
            )
            for group in layout.groups
        ),
        group_slot_counts=tuple(
            (group.group_id, len(group.slot_layer_ids)) for group in layout.groups
        ),
        theoretical_capacity_tokens=tuple(
            (group.group_id, usable_pages * group.block_size) for group in layout.groups
        ),
        component_padding_bytes=padding,
        component_padding_ratio=tuple(
            (group_id, pad_bytes / layout.physical_page_bytes)
            for group_id, pad_bytes in padding
        ),
    )
    return FlatHybridCachePlan(
        block_size=layout.block_size,
        physical_page_bytes=layout.physical_page_bytes,
        usable_pages=usable_pages,
        token_capacity=usable_pages * layout.block_size,
        groups=layout.groups,
        physical_slots=layout.physical_slots,
        layer_bindings=layout.layer_bindings,
        diagnostics=diagnostics,
    )


def format_flat_hybrid_cache_plan(plan: FlatHybridCachePlan) -> str:
    diagnostics = plan.diagnostics
    groups = ", ".join(
        f"{group_id}:{layer_count}/{dict(diagnostics.group_slot_counts)[group_id]}"
        for group_id, layer_count in diagnostics.group_layer_counts
    )
    padding = ", ".join(
        f"{group_id}:{pad_bytes}B/"
        f"{dict(diagnostics.component_padding_ratio)[group_id]:.6f}"
        for group_id, pad_bytes in diagnostics.component_padding_bytes
    )
    capacity = ", ".join(
        f"{group_id}:{tokens}"
        for group_id, tokens in diagnostics.theoretical_capacity_tokens
    )
    return (
        "FlatHybridCachePlan("
        f"configured_block_size={diagnostics.configured_block_size}, "
        f"effective_block_size={diagnostics.effective_block_size}, "
        f"physical_page_bytes={diagnostics.physical_page_bytes}, "
        f"page_set_bytes={diagnostics.bytes_per_page_set}, "
        f"physical_slots={diagnostics.physical_slot_count}, "
        f"padding_bindings={diagnostics.padding_binding_count}, "
        f"usable_pages={diagnostics.usable_pages}, "
        f"token_capacity={plan.token_capacity}, "
        f"null_pages={diagnostics.null_pages}, "
        f"allocated_bytes={diagnostics.total_allocated_bytes}, "
        f"unused_budget_bytes={diagnostics.unused_budget_bytes}, "
        f"groups=[{groups}], capacity=[{capacity}], padding=[{padding}])"
    )
