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

import hashlib
import json
import math
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

import torch

from tokenspeed.runtime.configs.flat_cache_runtime import (
    FlatPagedCacheRuntimeContract,
    flat_cache_debug_enabled,
    require_positive_int,
)
from tokenspeed.runtime.configs.hybrid_cache_plan import (
    ComponentBinding,
    FlatHybridCachePlan,
)
from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec
from tokenspeed.runtime.pd.flatkv import (
    FLATKV_PD_PROTOCOL_VERSION,
    FlatKVPDGroup,
    FlatKVPDLayout,
    FlatKVPDSLabRegistration,
    validate_flatkv_slab_registrations,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
    TorchMemorySaverAdapter,
)

_MATERIALIZABLE_DTYPES = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }
)

_FLAT_PD_LAYOUT_SCHEMA_VERSION = 1
_FLAT_PD_RAW_SLAB_BUFFER_PREFIX = "flat_hybrid.raw_slab"
_FLAT_PD_TRANSFER_POLICIES = frozenset(("full_suffix", "latest_snapshot"))


def _require_layer_id(layer_id: object) -> None:
    if isinstance(layer_id, bool) or not isinstance(layer_id, int):
        raise TypeError("layer_id must be an integer")


def _require_non_negative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _require_materializable_dtype(context: str, dtype: object) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{context}: dtype must be torch.dtype")
    if dtype not in _MATERIALIZABLE_DTYPES:
        raise ValueError(f"{context}: dtype {dtype} is not materializable")
    return dtype


def _resolve_pool_device(device: object) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"device must be a real CPU or CUDA device, got {device!r}"
        ) from exc
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError(f"device must be a real CPU or CUDA device, got {resolved}")
    return resolved


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    running = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(running)
        running *= dimension
    return tuple(reversed(result))


def _component_view(
    *,
    raw: torch.Tensor,
    component: ComponentBinding,
    family: str,
    num_pages: int,
    block_size: int,
    physical_page_bytes: int,
    layer_id: int,
) -> torch.Tensor:
    itemsize = component.dtype.itemsize
    typed_base = torch.empty(0, dtype=component.dtype, device=raw.device)
    typed_base.set_(raw.untyped_storage())
    inner_strides = _contiguous_strides(component.shape)
    if family == "history":
        size = (num_pages, block_size, *component.shape)
        stride = (
            physical_page_bytes // itemsize,
            math.prod(component.shape),
            *inner_strides,
        )
    else:
        size = (num_pages, *component.shape)
        stride = (physical_page_bytes // itemsize, *inner_strides)
    view = torch.as_strided(
        typed_base,
        size=size,
        stride=stride,
        storage_offset=component.byte_offset // itemsize,
    )

    raw_storage = raw.untyped_storage()
    view_storage = view.untyped_storage()
    context = f"layer {layer_id} component {component.name!r}"
    if (
        view_storage.data_ptr() != raw_storage.data_ptr()
        or view_storage.nbytes() != raw_storage.nbytes()
    ):
        raise ValueError(f"{context}: component view does not share raw storage")
    expected_pointer = raw.data_ptr() + component.byte_offset
    if view.data_ptr() != expected_pointer:
        raise ValueError(
            f"{context}: component view pointer disagrees with plan offset"
        )
    return view


def _validate_plan(plan: object) -> None:
    if not isinstance(plan, FlatHybridCachePlan):
        raise ValueError("plan must be a FlatHybridCachePlan")

    require_positive_int("block_size", plan.block_size)
    require_positive_int("physical_page_bytes", plan.physical_page_bytes)
    require_positive_int("usable_pages", plan.usable_pages)
    require_positive_int("token_capacity", plan.token_capacity)
    if plan.token_capacity > plan.usable_pages * plan.block_size:
        raise ValueError("token_capacity exceeds raw physical page capacity")

    if not plan.groups:
        raise ValueError("groups must be non-empty")

    physical_slot_count = len(plan.physical_slots)
    if not physical_slot_count:
        raise ValueError("physical_slots must be non-empty")

    groups_by_id = {}
    expected_by_layer = {}
    for group in plan.groups:
        if not isinstance(group.group_id, str) or not group.group_id:
            raise ValueError("group_id must be a non-empty string")
        if group.group_id in groups_by_id:
            raise ValueError("duplicate group_id in cache plan")
        groups_by_id[group.group_id] = group
        if group.family not in ("history", "state"):
            raise ValueError(
                f"group {group.group_id}: unsupported family {group.family!r}"
            )
        if group.retention not in ("full_history", "sliding_window"):
            raise ValueError(
                f"group {group.group_id}: unsupported retention {group.retention!r}"
            )
        if group.transfer_policy not in _FLAT_PD_TRANSFER_POLICIES:
            raise ValueError(
                f"group {group.group_id}: unsupported transfer_policy "
                f"{group.transfer_policy!r}"
            )
        if group.retention == "full_history":
            if group.sliding_window_tokens is not None:
                raise ValueError(
                    f"group {group.group_id}: full_history requires no sliding window"
                )
        elif (
            isinstance(group.sliding_window_tokens, bool)
            or not isinstance(group.sliding_window_tokens, int)
            or group.sliding_window_tokens <= 0
        ):
            raise ValueError(
                f"group {group.group_id}: sliding window must be a positive integer"
            )
        require_positive_int(f"group {group.group_id}: block_size", group.block_size)
        if group.block_size != plan.block_size:
            raise ValueError(
                f"group {group.group_id}: block_size does not match plan block_size"
            )
        if len(group.slot_layer_ids) != physical_slot_count:
            raise ValueError("groups have inconsistent padded slot count")
        for physical_slot, layer_id in enumerate(group.slot_layer_ids):
            if layer_id is None:
                continue
            _require_non_negative_integer(f"group {group.group_id}: layer_id", layer_id)
            if layer_id in expected_by_layer:
                raise ValueError(f"duplicate layer_id in group slots: {layer_id}")
            expected_by_layer[layer_id] = (group.group_id, physical_slot)

    actual_by_layer = {}
    canonical_by_layer = {}
    for binding in plan.layer_bindings:
        _require_non_negative_integer("layer binding: layer_id", binding.layer_id)
        _require_non_negative_integer(
            f"layer {binding.layer_id}: physical_slot", binding.physical_slot
        )
        if binding.layer_id in actual_by_layer:
            raise ValueError(f"duplicate layer binding for layer {binding.layer_id}")
        actual_by_layer[binding.layer_id] = (
            binding.group_id,
            binding.physical_slot,
        )
        canonical_by_layer[binding.layer_id] = binding
    if expected_by_layer != actual_by_layer:
        raise ValueError("group slots and layer bindings disagree")

    physical_by_layer = {}
    for position, slot in enumerate(plan.physical_slots):
        if slot.physical_slot != position:
            raise ValueError(
                f"physical_slot {slot.physical_slot} is not at tuple position {position}"
            )
        if slot.physical_page_bytes != plan.physical_page_bytes:
            raise ValueError(
                f"physical slot {position}: physical_page_bytes is not uniform"
            )
        for binding in slot.bindings:
            if binding.physical_slot != position:
                raise ValueError("physical slot bindings and layer bindings disagree")
            if binding.layer_id in physical_by_layer:
                raise ValueError(
                    f"physical slot contains duplicate layer binding {binding.layer_id}"
                )
            physical_by_layer[binding.layer_id] = binding
    if physical_by_layer != canonical_by_layer:
        raise ValueError("physical slot bindings and layer bindings disagree")

    for binding in plan.layer_bindings:
        group = groups_by_id[binding.group_id]
        if not binding.components:
            raise ValueError(f"layer {binding.layer_id}: components must be non-empty")

        names = set()
        previous_end = 0
        for component in binding.components:
            context = f"layer {binding.layer_id} component {component.name!r}"
            if not isinstance(component.name, str) or not component.name:
                raise ValueError(
                    f"layer {binding.layer_id}: component name must be non-empty"
                )
            if component.name in names:
                raise ValueError(
                    f"layer {binding.layer_id}: duplicate component "
                    f"name {component.name!r}"
                )
            names.add(component.name)
            if (
                not isinstance(component.shape, tuple)
                or not component.shape
                or any(
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension <= 0
                    for dimension in component.shape
                )
            ):
                raise ValueError(f"{context}: shape must contain positive integers")
            dtype = _require_materializable_dtype(context, component.dtype)
            if plan.physical_page_bytes % dtype.itemsize:
                raise ValueError(
                    f"physical_page_bytes is not aligned to dtype for {context}"
                )
            _require_non_negative_integer(
                f"{context}: byte_offset", component.byte_offset
            )
            require_positive_int(f"{context}: nbytes", component.nbytes)
            if component.byte_offset < previous_end:
                raise ValueError(
                    f"{context}: byte range overlaps or precedes previous component"
                )
            if component.byte_offset % dtype.itemsize:
                raise ValueError(f"{context}: byte_offset is not dtype aligned")

            logical_nbytes = math.prod(component.shape) * dtype.itemsize
            expected_nbytes = (
                plan.block_size * logical_nbytes
                if group.family == "history"
                else logical_nbytes
            )
            if component.nbytes != expected_nbytes:
                raise ValueError(
                    f"{context}: {group.family} nbytes {component.nbytes} "
                    f"does not match expected {expected_nbytes}"
                )
            if component.byte_offset + component.nbytes > plan.physical_page_bytes:
                raise ValueError(f"{context} crosses the physical page boundary")
            previous_end = component.byte_offset + component.nbytes

    if (
        isinstance(plan.diagnostics.null_pages, bool)
        or not isinstance(plan.diagnostics.null_pages, int)
        or plan.diagnostics.null_pages != 1
    ):
        raise ValueError("diagnostics.null_pages must equal 1")


def paged_cache_group_specs_from_plan(
    plan: FlatHybridCachePlan,
) -> tuple[PagedCacheGroupSpec, ...]:
    _validate_plan(plan)
    return _paged_cache_group_specs_from_validated_plan(plan)


def _paged_cache_group_specs_from_validated_plan(
    plan: FlatHybridCachePlan,
) -> tuple[PagedCacheGroupSpec, ...]:
    return tuple(
        PagedCacheGroupSpec(
            group_id=group.group_id,
            retention=group.retention,
            rows_per_page=plan.block_size,
            entry_stride_tokens=1,
            sliding_window_tokens=group.sliding_window_tokens,
            family=group.family,
            block_size=plan.block_size,
            transfer_policy=group.transfer_policy,
        )
        for group in plan.groups
    )


def _genuine_scheduler_page_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"scheduler {name} pages must be a genuine non-boolean integer, "
            f"got {value!r}"
        )
    return value


# Component name of the MLA latent KV cache in history-family bindings.
_MLA_LATENT_COMPONENT = "latent_kv"


def _flat_pd_buffer_id(physical_slot: int) -> str:
    _require_non_negative_integer("physical_slot", physical_slot)
    return f"{_FLAT_PD_RAW_SLAB_BUFFER_PREFIX}.{physical_slot}"


def _flat_pd_dtype_name(dtype: torch.dtype) -> str:
    _require_materializable_dtype("Flat PD component", dtype)
    name = str(dtype)
    if not name.startswith("torch."):
        raise ValueError(f"Flat PD component dtype has no stable torch name: {dtype}")
    return name


def _flat_pd_semantic_payload(plan: FlatHybridCachePlan) -> dict[str, Any]:
    """Return the stable semantic fields committed by the peer fingerprint."""
    group_order = {group.group_id: order for order, group in enumerate(plan.groups)}
    slots = []
    for slot in plan.physical_slots:
        ordered_bindings = tuple(
            sorted(
                slot.bindings,
                key=lambda binding: (
                    group_order[binding.group_id],
                    binding.layer_id,
                ),
            )
        )
        bindings = [
            {
                "group_id": binding.group_id,
                "layer_id": binding.layer_id,
                "components": [
                    {
                        "name": component.name,
                        "byte_offset": component.byte_offset,
                        "nbytes": component.nbytes,
                        "shape": list(component.shape),
                        "dtype": _flat_pd_dtype_name(component.dtype),
                    }
                    for component in binding.components
                ],
            }
            for binding in ordered_bindings
        ]
        slots.append(
            {
                "physical_slot": slot.physical_slot,
                "buffer_id": _flat_pd_buffer_id(slot.physical_slot),
                "bound_group_ids": [binding["group_id"] for binding in bindings],
                "bindings": bindings,
            }
        )
    return {
        "schema_version": _FLAT_PD_LAYOUT_SCHEMA_VERSION,
        "block_size": plan.block_size,
        "physical_page_bytes": plan.physical_page_bytes,
        "groups": [
            {
                "order": order,
                "group_id": group.group_id,
                "family": group.family,
                "transfer_policy": group.transfer_policy,
                "retention": group.retention,
                "sliding_window_tokens": group.sliding_window_tokens,
                "block_size": group.block_size,
            }
            for order, group in enumerate(plan.groups)
        ],
        "physical_slots": slots,
    }


def _flat_pd_semantic_fingerprint(payload: dict[str, Any]) -> str:
    """Hash stable wire semantics without peer-local allocation details."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _build_flat_pd_layout(
    plan: FlatHybridCachePlan, num_pages_with_null: int
) -> FlatKVPDLayout:
    slots_by_group = {group.group_id: [] for group in plan.groups}
    for slot in plan.physical_slots:
        for group_id in dict.fromkeys(binding.group_id for binding in slot.bindings):
            slots_by_group[group_id].append(slot.physical_slot)
    return FlatKVPDLayout(
        version=FLATKV_PD_PROTOCOL_VERSION,
        layout_fingerprint=_flat_pd_semantic_fingerprint(
            _flat_pd_semantic_payload(plan)
        ),
        block_size=plan.block_size,
        num_pages_with_null=num_pages_with_null,
        physical_buffer_ids=tuple(
            _flat_pd_buffer_id(slot.physical_slot) for slot in plan.physical_slots
        ),
        physical_page_bytes=plan.physical_page_bytes,
        groups=tuple(
            FlatKVPDGroup(
                group.group_id,
                group.family,
                group.transfer_policy,
                tuple(slots_by_group[group.group_id]),
            )
            for group in plan.groups
        ),
    )


def _build_flat_pd_contract(
    *,
    plan: FlatHybridCachePlan,
    raw_slabs: tuple[torch.Tensor, ...],
    num_pages_with_null: int,
) -> tuple[FlatKVPDLayout, tuple[FlatKVPDSLabRegistration, ...]]:
    if len(raw_slabs) != len(plan.physical_slots):
        raise ValueError("Flat PD raw slab count does not match physical slots")
    require_positive_int("Flat PD num_pages_with_null", num_pages_with_null)
    expected_num_pages_with_null = plan.usable_pages + plan.diagnostics.null_pages
    if num_pages_with_null != expected_num_pages_with_null:
        raise ValueError("Flat PD num_pages_with_null does not match the plan capacity")

    layout = _build_flat_pd_layout(plan, num_pages_with_null)
    registrations = []
    for physical_slot, raw in enumerate(raw_slabs):
        context = f"Flat PD physical slot {physical_slot}"
        if not isinstance(raw, torch.Tensor):
            raise ValueError(f"{context}: raw slab must be a tensor")
        if raw.dtype != torch.uint8:
            raise ValueError(f"{context}: raw slab must use torch.uint8")
        if raw.shape != (num_pages_with_null, plan.physical_page_bytes):
            raise ValueError(f"{context}: raw slab shape is inconsistent")
        if not raw.is_contiguous() or raw.storage_offset() != 0:
            raise ValueError(f"{context}: raw slab must be a contiguous owner")
        if raw.data_ptr() != raw.untyped_storage().data_ptr():
            raise ValueError(f"{context}: raw slab is not its storage owner")
        registrations.append(
            FlatKVPDSLabRegistration(
                physical_slot=physical_slot,
                buffer_id=layout.physical_buffer_ids[physical_slot],
                base_addr=raw.data_ptr(),
                length=raw.nbytes,
            )
        )
    registrations = validate_flatkv_slab_registrations(
        tuple(registrations), layout=layout, peer="local"
    )
    return layout, registrations


class FlatHybridCachePool:
    supports_hierarchical_kv_cache = False
    # Aliases recurrent-state and KV bytes in one slab; reused pages must be
    # sanitized (see ``zero_pages``) to avoid poisoned tails.
    flat_kv_requires_page_zeroing = True

    def __init__(
        self,
        *,
        plan: FlatHybridCachePlan,
        device: str | torch.device,
        enable_memory_saver: bool = False,
        mla_kv_lora_rank: int | None = None,
        mla_qk_rope_head_dim: int | None = None,
    ) -> None:
        """Allocate the planned raw slabs and expose no-copy component views.

        Args:
            plan: Validated flat hybrid cache plan.
            device: Real CPU or CUDA device the slabs live on.
            enable_memory_saver: Wrap allocation in the torch memory saver.
            mla_kv_lora_rank: Latent (nope) width of the MLA ``latent_kv``
                component. Required for the MLA read adapters
                (``get_value_buffer`` / ``get_mla_kv_buffer``); together with
                ``mla_qk_rope_head_dim`` it must sum to the component's last
                dimension.
            mla_qk_rope_head_dim: RoPE width of the MLA ``latent_kv``
                component; see ``mla_kv_lora_rank``.
        """
        _validate_plan(plan)
        if (mla_kv_lora_rank is None) != (mla_qk_rope_head_dim is None):
            raise ValueError(
                "mla_kv_lora_rank and mla_qk_rope_head_dim must be provided together"
            )
        if mla_kv_lora_rank is not None:
            require_positive_int("mla_kv_lora_rank", mla_kv_lora_rank)
            require_positive_int("mla_qk_rope_head_dim", mla_qk_rope_head_dim)
        paged_cache_group_specs = _paged_cache_group_specs_from_validated_plan(plan)
        resolved_device = _resolve_pool_device(device)
        memory_saver = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        num_pages = plan.usable_pages + 1
        group_ids_by_layer = MappingProxyType(
            {binding.layer_id: binding.group_id for binding in plan.layer_bindings}
        )
        physical_slots_by_layer = MappingProxyType(
            {binding.layer_id: binding.physical_slot for binding in plan.layer_bindings}
        )

        with memory_saver.region(tag="kv_cache", enable_cpu_backup=False):
            raw_slabs = tuple(
                torch.zeros(
                    (num_pages, slot.physical_page_bytes),
                    dtype=torch.uint8,
                    device=resolved_device,
                )
                for slot in plan.physical_slots
            )

        zero_slab_addresses = None
        if resolved_device.type == "cuda":
            zero_slab_addresses = torch.tensor(
                [raw_slab.data_ptr() for raw_slab in raw_slabs],
                dtype=torch.uint64,
                device=resolved_device,
            )

        allocated_bytes = sum(raw_slab.nbytes for raw_slab in raw_slabs)
        if allocated_bytes != plan.diagnostics.total_allocated_bytes:
            raise ValueError("raw slab allocation bytes disagree with plan diagnostics")

        groups_by_id = {group.group_id: group for group in plan.groups}
        component_views_by_key: dict[tuple[int, str], torch.Tensor] = {}
        for binding in plan.layer_bindings:
            raw = raw_slabs[binding.physical_slot]
            group = groups_by_id[binding.group_id]
            for component in binding.components:
                key = (binding.layer_id, component.name)
                component_views_by_key[key] = _component_view(
                    raw=raw,
                    component=component,
                    family=group.family,
                    num_pages=num_pages,
                    block_size=plan.block_size,
                    physical_page_bytes=plan.physical_page_bytes,
                    layer_id=binding.layer_id,
                )
        component_views = MappingProxyType(component_views_by_key)

        mla_flat_views_by_layer: dict[int, torch.Tensor] = {}
        for binding in plan.layer_bindings:
            key = (binding.layer_id, _MLA_LATENT_COMPONENT)
            view = component_views_by_key.get(key)
            if view is None:
                continue
            group = groups_by_id[binding.group_id]
            component = next(
                component
                for component in binding.components
                if component.name == _MLA_LATENT_COMPONENT
            )
            context = f"layer {binding.layer_id} component {_MLA_LATENT_COMPONENT!r}"
            if group.family != "history":
                raise ValueError(f"{context}: MLA latent must be history-family")
            if (
                component.byte_offset != 0
                or component.nbytes != plan.physical_page_bytes
            ):
                # MLA requires the full physical row; a partial binding would
                # make the flattened absolute-token view non-contiguous and
                # break every existing MLA writer.
                raise ValueError(
                    f"{context}: MLA latent must fill the physical page "
                    f"(offset {component.byte_offset}, nbytes {component.nbytes}, "
                    f"page bytes {plan.physical_page_bytes})"
                )
            if not view.is_contiguous():
                raise ValueError(f"{context}: MLA latent view is not contiguous")
            flat_view = view.view(num_pages * plan.block_size, *component.shape)
            if flat_view.data_ptr() != view.data_ptr():
                raise ValueError(f"{context}: flattened MLA view is not zero-copy")
            if (
                mla_kv_lora_rank is not None
                and mla_qk_rope_head_dim is not None
                and component.shape[-1] != mla_kv_lora_rank + mla_qk_rope_head_dim
            ):
                raise ValueError(
                    f"{context}: latent width {component.shape[-1]} does not "
                    f"match mla_kv_lora_rank {mla_kv_lora_rank} + "
                    f"mla_qk_rope_head_dim {mla_qk_rope_head_dim}"
                )
            mla_flat_views_by_layer[binding.layer_id] = flat_view

        runtime_contract = FlatPagedCacheRuntimeContract(
            block_size=plan.block_size,
            usable_pages=plan.usable_pages,
            num_device_pages_with_null=num_pages,
            token_capacity=plan.token_capacity,
            group_specs=paged_cache_group_specs,
            group_page_counts={
                spec.group_id: num_pages for spec in paged_cache_group_specs
            },
        )
        pd_layout, pd_slab_registrations = _build_flat_pd_contract(
            plan=plan,
            raw_slabs=raw_slabs,
            num_pages_with_null=num_pages,
        )

        self._plan = plan
        self._paged_cache_group_specs = paged_cache_group_specs
        self._num_device_pages_with_null = num_pages
        self._runtime_contract = runtime_contract
        self._raw_slabs = raw_slabs
        self._group_ids_by_layer = group_ids_by_layer
        self._physical_slots_by_layer = physical_slots_by_layer
        self._component_views = component_views
        self._allocated_bytes = allocated_bytes
        self._mla_flat_views_by_layer = MappingProxyType(mla_flat_views_by_layer)
        self._mla_kv_lora_rank = mla_kv_lora_rank
        self._mla_qk_rope_head_dim = mla_qk_rope_head_dim
        self._zero_slab_addresses = zero_slab_addresses
        self._pd_layout = pd_layout
        self._pd_slab_registrations = pd_slab_registrations

    @property
    def plan(self) -> FlatHybridCachePlan:
        return self._plan

    @property
    def paged_cache_group_specs(self) -> tuple[PagedCacheGroupSpec, ...]:
        return self._paged_cache_group_specs

    @property
    def num_device_pages_with_null(self) -> int:
        return self._num_device_pages_with_null

    @property
    def runtime_contract(self) -> FlatPagedCacheRuntimeContract:
        return self._runtime_contract

    @property
    def page_size(self) -> int:
        return self._runtime_contract.block_size

    @property
    def size(self) -> int:
        return self._runtime_contract.token_capacity

    @property
    def num_usable_pages(self) -> int:
        return self._runtime_contract.usable_pages

    @property
    def paged_cache_group_page_counts(self):
        return self._runtime_contract.group_page_counts

    @property
    def prefix_cache_required_group_ids(self) -> None:
        return None

    @property
    def supports_disaggregation(self) -> bool:
        return True

    def get_flatkv_pd_contract(
        self,
    ) -> tuple[FlatKVPDLayout, tuple[FlatKVPDSLabRegistration, ...]]:
        return self._pd_layout, self._pd_slab_registrations

    def bind_paged_cache_scheduler(self, scheduler: object) -> None:
        available_fn = getattr(scheduler, "available_kv_pages", None)
        active_fn = getattr(scheduler, "active_kv_pages", None)
        if not callable(available_fn) or not callable(active_fn):
            raise ValueError("scheduler must expose flat page introspection")
        available = _genuine_scheduler_page_count("available", available_fn())
        active = _genuine_scheduler_page_count("active", active_fn())
        if available != self.num_usable_pages:
            raise ValueError(
                f"scheduler available pages {available} disagree with usable_pages "
                f"{self.num_usable_pages}"
            )
        if active != 0:
            raise ValueError(
                f"scheduler must bind while idle, got active pages {active}"
            )

    def raw_slab(self, physical_slot: int) -> torch.Tensor:
        if isinstance(physical_slot, bool) or not isinstance(physical_slot, int):
            raise TypeError("physical_slot must be an integer")
        if physical_slot < 0 or physical_slot >= len(self._raw_slabs):
            raise IndexError(f"physical_slot {physical_slot} is out of range")
        return self._raw_slabs[physical_slot]

    def group_id_for_layer(self, layer_id: int) -> str:
        _require_layer_id(layer_id)
        try:
            return self._group_ids_by_layer[layer_id]
        except KeyError:
            raise KeyError(f"unknown layer_id {layer_id}") from None

    def physical_slot_for_layer(self, layer_id: int) -> int:
        _require_layer_id(layer_id)
        try:
            return self._physical_slots_by_layer[layer_id]
        except KeyError:
            raise KeyError(f"unknown layer_id {layer_id}") from None

    def get_component(self, layer_id: int, component_name: str) -> torch.Tensor:
        _require_layer_id(layer_id)
        if not isinstance(component_name, str):
            raise TypeError("component_name must be a string")
        if layer_id not in self._group_ids_by_layer:
            raise KeyError(f"unknown layer_id {layer_id}")
        try:
            return self._component_views[(layer_id, component_name)]
        except KeyError:
            raise KeyError(
                f"unknown component {component_name!r} for layer_id {layer_id}"
            ) from None

    # ------------------------------------------------------------------
    # MLA adapter surface: thin wrappers over the generic
    # "latent_kv" component view, mirroring the MLATokenToKVPool methods the
    # MLA backend and model-owned writers call. All views are no-copy; the
    # absolute token location formula ``page_id * block_size + offset`` is
    # preserved by the page-filling latent component.
    # ------------------------------------------------------------------

    def _require_mla_flat_view(self, layer_id: int) -> torch.Tensor:
        _require_layer_id(layer_id)
        view = self._mla_flat_views_by_layer.get(layer_id)
        if view is None:
            raise KeyError(
                f"layer_id {layer_id} has no MLA {_MLA_LATENT_COMPONENT!r} "
                "component binding"
            )
        return view

    def _require_mla_geometry(self) -> tuple[int, int]:
        if self._mla_kv_lora_rank is None or self._mla_qk_rope_head_dim is None:
            raise RuntimeError(
                "MLA latent geometry is not configured; construct the pool "
                "with mla_kv_lora_rank and mla_qk_rope_head_dim"
            )
        return self._mla_kv_lora_rank, self._mla_qk_rope_head_dim

    def _check_mla_locations(self, loc: torch.Tensor, what: str) -> None:
        """Structural checks on MLA token locations.

        Host-side shape/dtype checks always run. Value checks (write to the
        null page 0, location outside the planned capacity) need a GPU sync,
        so they run only under ``TOKENSPEED_FLAT_DEBUG=1``.
        """
        if not isinstance(loc, torch.Tensor) or loc.dim() != 1:
            raise ValueError(f"{what}: loc must be a 1-D tensor")
        if loc.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{what}: loc must be int32 or int64")
        if not flat_cache_debug_enabled() or loc.numel() == 0:
            return
        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            get_is_cuda_graph_phase,
        )

        if get_is_cuda_graph_phase() or torch.cuda.is_current_stream_capturing():
            # CUDA graph warmup + capture run with placeholder locations that
            # resolve to the null page 0 by design (padded-row dummy-page
            # protection); a .item() sync would also be illegal mid-capture.
            # Real rows are validated host-side at every replay refresh.
            return
        block_size = self._plan.block_size
        capacity = self._num_device_pages_with_null * block_size
        loc_min = int(loc.min().item())
        loc_max = int(loc.max().item())
        if loc_min < block_size:
            raise ValueError(
                f"{what}: token location {loc_min} lands in the null page 0 "
                f"(block_size {block_size})"
            )
        if loc_max >= capacity:
            raise ValueError(
                f"{what}: token location {loc_max} is outside the planned "
                f"capacity {capacity}"
            )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Flattened latent view ``[(num_pages) * block_size, 1, D]``."""
        return self._require_mla_flat_view(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Latent (nope) slice of the flattened view, ``[..., :kv_lora_rank]``."""
        kv_lora_rank, _ = self._require_mla_geometry()
        return self._require_mla_flat_view(layer_id)[..., :kv_lora_rank]

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_mla_kv_buffer(
        self,
        layer: object,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        sanitize: bool = True,
    ) -> None:
        """Write latent KV rows at absolute token locations.

        Args:
            layer: Attention layer (only ``layer.layer_id`` is consulted).
            loc: 1-D int tensor of absolute token locations
                (``page_id * block_size + offset``).
            cache_k_nope: ``[T, 1, kv_lora_rank]`` latent rows.
            cache_k_rope: ``[T, 1, qk_rope_head_dim]`` rope rows.
            sanitize: Replace NaN/Inf in rows being written. This defaults on
                because the production FlatKV path bypasses
                ``LayerMappedKVPool``.

        Note: pages are recycled from the shared BlockPool, so the unwritten
        remainder of a sequence's final 64-token kernel block can hold a
        previous tenant's bytes (fp8 NaN encodings included). The MLA decode
        kernel sanitizes those V-tail rows itself (mla_decode_fp8: zero
        invalid V rows post-TMA-load, pre-PV-MMA) — do not rely on cache
        content beyond the written range being zero.
        """
        view = self._require_mla_flat_view(int(layer.layer_id))
        self._check_mla_locations(loc, "set_mla_kv_buffer")
        if (
            cache_k_nope.shape[0] != loc.shape[0]
            or cache_k_rope.shape[0] != loc.shape[0]
        ):
            raise ValueError(
                "set_mla_kv_buffer: cache rows must match loc count, got "
                f"{cache_k_nope.shape[0]}/{cache_k_rope.shape[0]} vs {loc.shape[0]}"
            )
        if view.device.type == "cuda":
            from tokenspeed.runtime.cache.utils import set_mla_kv_buffer_triton
            from tokenspeed.runtime.utils.pdl import pdl_enabled

            # The write kernel casts to the buffer dtype on store; bf16 sources go in directly.
            set_mla_kv_buffer_triton(
                view,
                loc,
                cache_k_nope,
                cache_k_rope,
                enable_pdl=pdl_enabled(),
                sanitize=sanitize,
            )
        else:
            if sanitize:
                cache_k_nope = torch.nan_to_num(
                    cache_k_nope.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).to(view.dtype)
                cache_k_rope = torch.nan_to_num(
                    cache_k_rope.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).to(view.dtype)
            else:
                cache_k_nope = cache_k_nope.to(view.dtype)
                cache_k_rope = cache_k_rope.to(view.dtype)
            nope_dim = cache_k_nope.shape[-1]
            view[loc.long(), :, :nope_dim] = cache_k_nope
            view[loc.long(), :, nope_dim:] = cache_k_rope

    def get_mla_kv_buffer(
        self,
        layer: object,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather latent KV rows at absolute token locations.

        Args:
            layer: Attention layer (only ``layer.layer_id`` is consulted).
            loc: 1-D int tensor of absolute token locations.
            dst_dtype: Output dtype; defaults to the stored latent dtype.

        Returns:
            ``(cache_k_nope, cache_k_rope)`` shaped ``[T, 1, kv_lora_rank]``
            and ``[T, 1, qk_rope_head_dim]``.
        """
        kv_lora_rank, qk_rope_head_dim = self._require_mla_geometry()
        view = self._require_mla_flat_view(int(layer.layer_id))
        self._check_mla_locations(loc, "get_mla_kv_buffer")
        dst_dtype = dst_dtype or view.dtype
        if view.device.type == "cuda":
            from tokenspeed.runtime.cache.utils import get_mla_kv_buffer_triton
            from tokenspeed.runtime.utils.pdl import pdl_enabled

            cache_k_nope = torch.empty(
                (loc.shape[0], 1, kv_lora_rank), dtype=dst_dtype, device=view.device
            )
            cache_k_rope = torch.empty(
                (loc.shape[0], 1, qk_rope_head_dim),
                dtype=dst_dtype,
                device=view.device,
            )
            get_mla_kv_buffer_triton(
                view, loc, cache_k_nope, cache_k_rope, enable_pdl=pdl_enabled()
            )
            return cache_k_nope, cache_k_rope
        rows = view[loc.long()]
        return (
            rows[..., :kv_lora_rank].to(dst_dtype).contiguous(),
            rows[..., kv_lora_rank:].to(dst_dtype).contiguous(),
        )

    def maybe_log_paged_cache_group_pages(self) -> None:
        """Decode-stats hook (BaseTokenToKVPool surface); the scheduler's page
        counters already cover the shared page-id space, so nothing to log."""
        return None

    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    def zero_pages(self, page_ids: Sequence[int]) -> None:
        """Sanitize newly-owned physical pages across every aliased slab.

        Page 0 is the permanent null page and is never a legal ownership
        target. Prefix-cache hits are not passed here; their bytes remain
        intact. Calls run on the current stream so the event loop can order
        this write before host loadback and model execution without a host
        synchronization.
        """
        if not isinstance(page_ids, Sequence):
            raise TypeError("page_ids must be a sequence of integers")
        if not page_ids:
            return
        validated_page_ids: set[int] = set()
        for page_id in page_ids:
            if (
                isinstance(page_id, bool)
                or not isinstance(page_id, int)
                or page_id <= 0
                or page_id >= self._num_device_pages_with_null
            ):
                raise ValueError(
                    f"page id {page_id!r} is outside the usable range "
                    f"[1, {self._num_device_pages_with_null - 1}]"
                )
            validated_page_ids.add(page_id)
        unique_page_ids = sorted(validated_page_ids)

        if self._raw_slabs[0].device.type == "cpu":
            indices = torch.tensor(unique_page_ids, dtype=torch.int64)
            for raw_slab in self._raw_slabs:
                raw_slab.index_fill_(0, indices, 0)
            return

        assert self._zero_slab_addresses is not None
        # Give each launch independent storage. Reusing a pinned host staging
        # buffer would let the overlap loop mutate it before the prior async
        # H2D copy has necessarily completed.
        page_ids_device = torch.tensor(
            unique_page_ids,
            dtype=torch.int32,
            device=self._raw_slabs[0].device,
        )

        from tokenspeed_kernel.ops.kvcache.triton import zero_flat_cache_pages

        zero_flat_cache_pages(
            self._zero_slab_addresses,
            page_ids_device,
            page_size_bytes=self._plan.physical_page_bytes,
        )

    def clear(self) -> None:
        for raw_slab in self._raw_slabs:
            raw_slab.zero_()

    def clear_kv_buffers(self) -> None:
        self.clear()
