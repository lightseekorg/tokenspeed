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

"""Flat KV-cache memory plan: pure sizing/binding decisions, no torch.

Components declare per-block bytes as a function of P (block_size):
linear components scale (bytes_per_slot > 0), constant components do not
(const_bytes > 0, mamba state snapshots). Same-(group, layer) components
pack into one page row ([conv|ssm|pad], the vLLM hybrid layout). One
equalizer move: constant rows inflate P until the widest linear row
covers them (vLLM align). plan_tensors then pairs physical slot j with
the j-th layer of every group over a single page-id space and sizes each
slab by its own packed row from the budget.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from tokenspeed.runtime.configs.flat_kv_contract import (
    CACHE_OWNER_DRAFT,
    CACHE_OWNER_TARGET,
    STATE_LAYER_TYPES,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec


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


FlatCacheOwner = Literal["target", "draft"]

V4_FLAT_GRAPH_METADATA_ELEMENT_BYTES = 4

# Conservative 64-bit host-container estimates.  They remain explicit plan
# inputs so measured C++ high-water values can replace them without changing
# the device payload or table-width contracts.
V4_CPU_BLOCK_REF_BYTES_ESTIMATE = 16
V4_CPU_EXPORT_GROUP_HEADER_BYTES_ESTIMATE = 64
V4_CPU_POOL_FIXED_BYTES_ESTIMATE = 256
V4_CPU_POOL_METADATA_BYTES_PER_BLOCK_ESTIMATE = 192


@dataclass(frozen=True)
class FlatComponentTensorPlan:
    """Immutable per-block storage schema for one owner-namespaced plane.

    ``shape_per_block`` excludes the pool's leading page dimension.  The arena
    allocation shape is therefore ``(pool.total_blocks, *shape_per_block)``.
    ``stride_bytes`` describes that trailing per-block view; owner and the
    enclosing pool namespace the component identity.
    """

    owner: FlatCacheOwner
    group_id: str
    layer: int
    component: str
    dtype: str
    shape_per_block: tuple[int, ...]
    stride_bytes: tuple[int, ...]
    alignment_bytes: int
    bytes_per_block: int

    def __post_init__(self) -> None:
        if self.owner not in ("target", "draft"):
            raise ValueError(f"unsupported flat component owner {self.owner!r}")
        if not self.group_id:
            raise ValueError("flat component group_id must be non-empty")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise ValueError(f"flat component layer must be an int, got {self.layer!r}")
        if self.layer < 0:
            raise ValueError(f"flat component layer must be >= 0, got {self.layer}")
        if not self.component:
            raise ValueError("flat component name must be non-empty")
        if not self.dtype:
            raise ValueError("flat component dtype must be non-empty")

        shape = tuple(self.shape_per_block)
        strides = tuple(self.stride_bytes)
        object.__setattr__(self, "shape_per_block", shape)
        object.__setattr__(self, "stride_bytes", strides)
        if not shape or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in shape
        ):
            raise ValueError(
                "flat component shape_per_block must contain positive integers"
            )
        if len(strides) != len(shape) or any(
            isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0
            for stride in strides
        ):
            raise ValueError(
                "flat component stride_bytes must contain one positive integer "
                "per shape dimension"
            )
        if (
            isinstance(self.alignment_bytes, bool)
            or not isinstance(self.alignment_bytes, int)
            or self.alignment_bytes <= 0
        ):
            raise ValueError(
                "flat component alignment_bytes must be a positive integer"
            )
        if (
            isinstance(self.bytes_per_block, bool)
            or not isinstance(self.bytes_per_block, int)
            or self.bytes_per_block <= 0
        ):
            raise ValueError("flat component bytes_per_block must be positive")


@dataclass(frozen=True)
class FlatBlockPoolPlan:
    """Canonical physical capacity and tensor planes for one flat pool."""

    pool_id: str
    total_blocks: int
    bytes_per_block: int
    storage_schema_hash: str
    tensors: tuple[FlatComponentTensorPlan, ...]

    def __post_init__(self) -> None:
        tensors = tuple(self.tensors)
        object.__setattr__(self, "tensors", tensors)
        if not self.pool_id:
            raise ValueError("flat block pool_id must be non-empty")
        if (
            isinstance(self.total_blocks, bool)
            or not isinstance(self.total_blocks, int)
            or self.total_blocks < 2
        ):
            raise ValueError("flat block pool total_blocks must be >= 2")
        if (
            isinstance(self.bytes_per_block, bool)
            or not isinstance(self.bytes_per_block, int)
            or self.bytes_per_block <= 0
        ):
            raise ValueError("flat block pool bytes_per_block must be positive")
        if not tensors:
            raise ValueError("flat block pool must contain at least one tensor")
        if len(self.storage_schema_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.storage_schema_hash
        ):
            raise ValueError("storage_schema_hash must be a lowercase SHA-256 hex")


@dataclass(frozen=True)
class FlatGroupTablePlan:
    """Canonical capture/export bounds for one scheduler-union group."""

    group_id: str
    target_capture_cols: int
    draft_capture_cols: int
    max_export_cols: int
    max_live_descriptor_cols: int

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("flat group table plan group_id must be non-empty")
        for name, value in (
            ("target_capture_cols", self.target_capture_cols),
            ("draft_capture_cols", self.draft_capture_cols),
            ("max_export_cols", self.max_export_cols),
            ("max_live_descriptor_cols", self.max_live_descriptor_cols),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        if self.target_capture_cols == 0 and self.draft_capture_cols == 0:
            raise ValueError("flat group table plan must belong to target and/or draft")
        if self.max_export_cols <= 0:
            raise ValueError("flat group max_export_cols must be positive")
        if self.max_live_descriptor_cols < self.max_export_cols:
            raise ValueError(
                "flat group max_live_descriptor_cols must cover max_export_cols"
            )

    def capture_cols(self, owner: FlatCacheOwner) -> int:
        if owner == "target":
            return self.target_capture_cols
        if owner == "draft":
            return self.draft_capture_cols
        raise ValueError(f"unsupported flat cache owner {owner!r}")


def v4_flat_graph_owner_metadata_bytes(
    group_table_plans: Sequence[FlatGroupTablePlan],
    *,
    owner: FlatCacheOwner,
    batch_rows: int,
) -> int:
    """Exact int32 table plus base-buffer bytes for one graph owner."""
    if owner not in ("target", "draft"):
        raise ValueError(f"unsupported flat cache owner {owner!r}")
    if (
        isinstance(batch_rows, bool)
        or not isinstance(batch_rows, int)
        or batch_rows < 0
    ):
        raise ValueError("flat graph batch_rows must be an integer >= 0")
    return (
        V4_FLAT_GRAPH_METADATA_ELEMENT_BYTES
        * batch_rows
        * sum(
            plan.capture_cols(owner) + 1
            for plan in group_table_plans
            if plan.capture_cols(owner) > 0
        )
    )


def validate_v4_flat_graph_owner_allocation(
    *,
    owner: FlatCacheOwner,
    capture_cols_by_group: Mapping[str, int],
    batch_rows: int,
    table_shapes: Mapping[str, Sequence[int]],
    base_shapes: Mapping[str, Sequence[int]],
    table_nbytes: Mapping[str, int],
    base_nbytes: Mapping[str, int],
) -> int:
    """Validate runtime graph tensors against one owner-local plan view."""
    expected_groups = set(capture_cols_by_group)
    observed_maps = {
        "tables": table_shapes,
        "bases": base_shapes,
        "table_nbytes": table_nbytes,
        "base_nbytes": base_nbytes,
    }
    for label, observed in observed_maps.items():
        if set(observed) != expected_groups:
            raise RuntimeError(
                f"DeepSeek V4 {owner} CUDA graph {label} groups disagree with "
                f"flat plan: actual={sorted(observed)}, "
                f"expected={sorted(expected_groups)}"
            )

    actual_total = 0
    for group_id, cols in capture_cols_by_group.items():
        expected_table_shape = (batch_rows, cols)
        expected_base_shape = (batch_rows,)
        actual_table_shape = tuple(int(dim) for dim in table_shapes[group_id])
        actual_base_shape = tuple(int(dim) for dim in base_shapes[group_id])
        if (
            actual_table_shape != expected_table_shape
            or actual_base_shape != expected_base_shape
        ):
            raise RuntimeError(
                f"DeepSeek V4 {owner} CUDA graph shape disagrees with flat plan "
                f"for {group_id!r}: table={actual_table_shape}, "
                f"expected_table={expected_table_shape}, base={actual_base_shape}, "
                f"expected_base={expected_base_shape}"
            )
        expected_table_nbytes = V4_FLAT_GRAPH_METADATA_ELEMENT_BYTES * batch_rows * cols
        expected_base_nbytes = V4_FLAT_GRAPH_METADATA_ELEMENT_BYTES * batch_rows
        actual_table_nbytes = table_nbytes[group_id]
        actual_base_nbytes = base_nbytes[group_id]
        if (
            actual_table_nbytes != expected_table_nbytes
            or actual_base_nbytes != expected_base_nbytes
        ):
            raise RuntimeError(
                f"DeepSeek V4 {owner} CUDA graph bytes disagree with flat plan "
                f"for {group_id!r}: table={actual_table_nbytes}, "
                f"expected_table={expected_table_nbytes}, "
                f"base={actual_base_nbytes}, expected_base={expected_base_nbytes}"
            )
        actual_total += actual_table_nbytes + actual_base_nbytes
    return actual_total


@dataclass(frozen=True)
class V4FlatMetadataAccounting:
    """Metadata inputs flattened into the final V4 memory plan."""

    group_table_plans: tuple[FlatGroupTablePlan, ...] = ()
    graph_metadata_bytes: int = 0
    forward_input_bytes: int = 0
    cpu_forward_export_bytes: int = 0
    cpu_forward_staging_bytes: int = 0
    cpu_request_metadata_bytes_estimate: int = 0
    forward_buffer_depth: int = 1
    target_graph_batch_rows: int = 0
    draft_graph_batch_rows: int = 0
    max_scheduled_batch_rows: int = 0
    cpu_pool_metadata_bytes_per_block_estimate: int = 0
    cpu_pool_fixed_bytes_estimate: int = 0

    def __post_init__(self) -> None:
        plans = tuple(sorted(self.group_table_plans, key=lambda item: item.group_id))
        object.__setattr__(self, "group_table_plans", plans)
        group_ids = [plan.group_id for plan in plans]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("flat metadata accounting has duplicate group plans")
        for name in (
            "graph_metadata_bytes",
            "forward_input_bytes",
            "cpu_forward_export_bytes",
            "cpu_forward_staging_bytes",
            "cpu_request_metadata_bytes_estimate",
            "target_graph_batch_rows",
            "draft_graph_batch_rows",
            "max_scheduled_batch_rows",
            "cpu_pool_metadata_bytes_per_block_estimate",
            "cpu_pool_fixed_bytes_estimate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        if (
            isinstance(self.forward_buffer_depth, bool)
            or not isinstance(self.forward_buffer_depth, int)
            or self.forward_buffer_depth <= 0
        ):
            raise ValueError("forward_buffer_depth must be a positive integer")
        if self.cpu_forward_staging_bytes != self.forward_input_bytes:
            raise ValueError(
                "CPU forward staging and GPU forward inputs must share one "
                "table/base shape and depth"
            )
        if self.cpu_forward_export_bytes < self.cpu_forward_staging_bytes:
            raise ValueError("CPU forward export must cover staging bytes and headers")
        max_graph_batch_rows = max(
            self.target_graph_batch_rows,
            self.draft_graph_batch_rows,
        )
        if max_graph_batch_rows > self.max_scheduled_batch_rows:
            raise ValueError(
                "flat metadata graph batch rows cannot exceed scheduled rows: "
                f"graph={max_graph_batch_rows}, "
                f"scheduled={self.max_scheduled_batch_rows}"
            )
        if not plans and any(
            (
                self.graph_metadata_bytes,
                self.forward_input_bytes,
                self.cpu_forward_export_bytes,
                self.cpu_forward_staging_bytes,
                self.cpu_request_metadata_bytes_estimate,
            )
        ):
            raise ValueError(
                "non-zero flat metadata accounting requires group table plans"
            )
        expected_graph_metadata_bytes = v4_flat_graph_owner_metadata_bytes(
            plans,
            owner="target",
            batch_rows=self.target_graph_batch_rows,
        ) + v4_flat_graph_owner_metadata_bytes(
            plans,
            owner="draft",
            batch_rows=self.draft_graph_batch_rows,
        )
        if self.graph_metadata_bytes != expected_graph_metadata_bytes:
            raise ValueError(
                "graph_metadata_bytes must equal the canonical target/draft "
                "int32 table plus base shapes: "
                f"got {self.graph_metadata_bytes}, "
                f"expected {expected_graph_metadata_bytes}"
            )


@dataclass(frozen=True)
class V4FlatMemoryPlan:
    """Canonical device-side V4 pool plan and scheduler/owner group views."""

    max_total_tokens: int
    pools: tuple[FlatBlockPoolPlan, ...]
    scheduler_group_specs: tuple[PagedCacheGroupSpec, ...]
    target_owner_group_specs: tuple[PagedCacheGroupSpec, ...]
    draft_owner_group_specs: tuple[PagedCacheGroupSpec, ...]
    group_table_plans: tuple[FlatGroupTablePlan, ...]
    payload_bytes: int
    graph_metadata_bytes: int
    forward_input_bytes: int
    cpu_forward_export_bytes: int
    cpu_forward_staging_bytes: int
    cpu_pool_metadata_bytes_estimate: int
    cpu_request_metadata_bytes_estimate: int
    forward_buffer_depth: int
    target_graph_batch_rows: int
    draft_graph_batch_rows: int
    max_scheduled_batch_rows: int
    device_cache_total_bytes: int
    plan_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pools", tuple(self.pools))
        object.__setattr__(
            self, "scheduler_group_specs", tuple(self.scheduler_group_specs)
        )
        object.__setattr__(
            self, "target_owner_group_specs", tuple(self.target_owner_group_specs)
        )
        object.__setattr__(
            self, "draft_owner_group_specs", tuple(self.draft_owner_group_specs)
        )
        object.__setattr__(self, "group_table_plans", tuple(self.group_table_plans))
        if (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
            or self.max_total_tokens < 0
        ):
            raise ValueError("V4 flat memory max_total_tokens must be >= 0")
        if not self.pools:
            raise ValueError("V4 flat memory plan must contain at least one pool")
        if (
            isinstance(self.payload_bytes, bool)
            or not isinstance(self.payload_bytes, int)
            or self.payload_bytes <= 0
        ):
            raise ValueError("V4 flat memory payload_bytes must be positive")
        for name in (
            "graph_metadata_bytes",
            "forward_input_bytes",
            "cpu_forward_export_bytes",
            "cpu_forward_staging_bytes",
            "cpu_pool_metadata_bytes_estimate",
            "cpu_request_metadata_bytes_estimate",
            "target_graph_batch_rows",
            "draft_graph_batch_rows",
            "max_scheduled_batch_rows",
            "device_cache_total_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        if (
            isinstance(self.forward_buffer_depth, bool)
            or not isinstance(self.forward_buffer_depth, int)
            or self.forward_buffer_depth <= 0
        ):
            raise ValueError("forward_buffer_depth must be a positive integer")
        if self.device_cache_total_bytes <= 0:
            raise ValueError("device_cache_total_bytes must be positive")
        if self.cpu_forward_staging_bytes != self.forward_input_bytes:
            raise ValueError(
                "CPU forward staging and GPU forward inputs must share one "
                "table/base shape and depth"
            )
        if self.cpu_forward_export_bytes < self.cpu_forward_staging_bytes:
            raise ValueError("CPU forward export must cover staging bytes and headers")
        expected_device_total = (
            self.payload_bytes + self.graph_metadata_bytes + self.forward_input_bytes
        )
        if self.device_cache_total_bytes != expected_device_total:
            raise ValueError(
                "device_cache_total_bytes must equal payload + graph metadata "
                f"+ forward inputs, got {self.device_cache_total_bytes} and "
                f"expected {expected_device_total}"
            )
        scheduler_group_ids = {spec.group_id for spec in self.scheduler_group_specs}
        table_group_id_list = [plan.group_id for plan in self.group_table_plans]
        table_group_ids = set(table_group_id_list)
        if len(table_group_id_list) != len(table_group_ids):
            raise ValueError("group_table_plans contains duplicate group ids")
        if table_group_ids and table_group_ids != scheduler_group_ids:
            raise ValueError(
                "group_table_plans must cover the scheduler group union exactly"
            )
        expected_graph_metadata_bytes = self.graph_metadata_bytes_for_owner(
            "target"
        ) + self.graph_metadata_bytes_for_owner("draft")
        if self.graph_metadata_bytes != expected_graph_metadata_bytes:
            raise ValueError(
                "graph_metadata_bytes must equal the canonical target/draft "
                "int32 table plus base shapes: "
                f"got {self.graph_metadata_bytes}, "
                f"expected {expected_graph_metadata_bytes}"
            )
        if len(self.plan_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.plan_fingerprint
        ):
            raise ValueError("plan_fingerprint must be a lowercase SHA-256 hex")

    @property
    def scheduler_group_specs_by_id(self) -> dict[str, PagedCacheGroupSpec]:
        return {spec.group_id: spec for spec in self.scheduler_group_specs}

    @property
    def group_table_plans_by_id(self) -> dict[str, FlatGroupTablePlan]:
        return {plan.group_id: plan for plan in self.group_table_plans}

    def graph_batch_rows(self, owner: FlatCacheOwner) -> int:
        if owner == "target":
            return self.target_graph_batch_rows
        if owner == "draft":
            return self.draft_graph_batch_rows
        raise ValueError(f"unsupported flat cache owner {owner!r}")

    def graph_capture_cols_by_group(self, owner: FlatCacheOwner) -> dict[str, int]:
        return {
            plan.group_id: plan.capture_cols(owner)
            for plan in self.group_table_plans
            if plan.capture_cols(owner) > 0
        }

    def graph_metadata_bytes_for_owner(self, owner: FlatCacheOwner) -> int:
        return v4_flat_graph_owner_metadata_bytes(
            self.group_table_plans,
            owner=owner,
            batch_rows=self.graph_batch_rows(owner),
        )

    def validate_graph_metadata_allocation(
        self,
        *,
        target_actual_bytes: int,
        draft_actual_bytes: int,
    ) -> None:
        """Fail closed when graph table/base tensors drift from this plan."""
        actual_by_owner = {
            "target": target_actual_bytes,
            "draft": draft_actual_bytes,
        }
        for owner, actual in actual_by_owner.items():
            if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
                raise ValueError(
                    f"{owner} graph metadata actual bytes must be an integer >= 0"
                )
            expected = self.graph_metadata_bytes_for_owner(owner)
            if actual != expected:
                raise RuntimeError(
                    f"DeepSeek V4 {owner} CUDA graph metadata allocation "
                    f"disagrees with flat plan: actual={actual}, expected={expected}"
                )
        actual_total = target_actual_bytes + draft_actual_bytes
        if actual_total != self.graph_metadata_bytes:
            raise RuntimeError(
                "DeepSeek V4 CUDA graph metadata allocation disagrees with "
                f"flat plan total: actual={actual_total}, "
                f"expected={self.graph_metadata_bytes}"
            )

    @property
    def cpu_cache_metadata_total_bytes(self) -> int:
        return (
            self.cpu_pool_metadata_bytes_estimate
            + self.cpu_request_metadata_bytes_estimate
            + self.cpu_forward_export_bytes
            + self.cpu_forward_staging_bytes
        )


@dataclass(frozen=True)
class V4FlatPlanAgreementRecord:
    """Rank-local canonical plan exported for attention-TP agreement."""

    rank: int
    plan_fingerprint: str
    canonical_plan: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise ValueError(f"rank must be an integer >= 0, got {self.rank!r}")
        if len(self.plan_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.plan_fingerprint
        ):
            raise ValueError("plan_fingerprint must be a lowercase SHA-256 hex")


_SCHEDULING_SCHEMA_FIELDS = (
    "pool_id",
    "block_size_tokens",
    "rows_per_page",
    "entry_stride_tokens",
    "retention",
    "family",
    "sliding_window_tokens",
    "prefix_role",
    "table_layout",
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_component(component: FlatComponentTensorPlan) -> dict[str, object]:
    return {
        "owner": component.owner,
        "group_id": component.group_id,
        "layer": component.layer,
        "component": component.component,
        "dtype": component.dtype,
        "shape_per_block": list(component.shape_per_block),
        "stride_bytes": list(component.stride_bytes),
        "alignment_bytes": component.alignment_bytes,
        "bytes_per_block": component.bytes_per_block,
    }


def _canonical_group(spec: PagedCacheGroupSpec) -> dict[str, object]:
    return {
        "group_id": spec.group_id,
        "pool_id": spec.pool_id,
        "block_size_tokens": spec.block_size_tokens,
        "rows_per_page": spec.rows_per_page,
        "entry_stride_tokens": spec.entry_stride_tokens,
        "retention": spec.retention,
        "family": spec.family,
        "sliding_window_tokens": spec.sliding_window_tokens,
        "prefix_role": spec.prefix_role,
        "table_layout": spec.table_layout,
        "required_producer_domain_mask": spec.required_producer_domain_mask,
        "owner_mask": spec.owner_mask,
    }


def _canonical_v4_flat_memory_plan(
    *,
    max_total_tokens: int,
    pools: Sequence[FlatBlockPoolPlan],
    scheduler_group_specs: Sequence[PagedCacheGroupSpec],
    target_owner_group_specs: Sequence[PagedCacheGroupSpec],
    draft_owner_group_specs: Sequence[PagedCacheGroupSpec],
    group_table_plans: Sequence[FlatGroupTablePlan],
    payload_bytes: int,
    graph_metadata_bytes: int,
    forward_input_bytes: int,
    cpu_forward_export_bytes: int,
    cpu_forward_staging_bytes: int,
    cpu_pool_metadata_bytes_estimate: int,
    cpu_request_metadata_bytes_estimate: int,
    forward_buffer_depth: int,
    target_graph_batch_rows: int,
    draft_graph_batch_rows: int,
    max_scheduled_batch_rows: int,
    device_cache_total_bytes: int,
) -> dict[str, object]:
    return {
        "max_total_tokens": max_total_tokens,
        "pools": [
            {
                "pool_id": pool.pool_id,
                "total_blocks": pool.total_blocks,
                "bytes_per_block": pool.bytes_per_block,
                "storage_schema_hash": pool.storage_schema_hash,
                "tensors": [_canonical_component(tensor) for tensor in pool.tensors],
            }
            for pool in pools
        ],
        "scheduler_group_specs": [
            _canonical_group(spec) for spec in scheduler_group_specs
        ],
        "target_owner_group_specs": [
            _canonical_group(spec) for spec in target_owner_group_specs
        ],
        "draft_owner_group_specs": [
            _canonical_group(spec) for spec in draft_owner_group_specs
        ],
        "group_table_plans": [
            {
                "group_id": plan.group_id,
                "target_capture_cols": plan.target_capture_cols,
                "draft_capture_cols": plan.draft_capture_cols,
                "max_export_cols": plan.max_export_cols,
                "max_live_descriptor_cols": plan.max_live_descriptor_cols,
            }
            for plan in group_table_plans
        ],
        "payload_bytes": payload_bytes,
        "graph_metadata_bytes": graph_metadata_bytes,
        "forward_input_bytes": forward_input_bytes,
        "cpu_forward_export_bytes": cpu_forward_export_bytes,
        "cpu_forward_staging_bytes": cpu_forward_staging_bytes,
        "cpu_pool_metadata_bytes_estimate": cpu_pool_metadata_bytes_estimate,
        "cpu_request_metadata_bytes_estimate": cpu_request_metadata_bytes_estimate,
        "forward_buffer_depth": forward_buffer_depth,
        "target_graph_batch_rows": target_graph_batch_rows,
        "draft_graph_batch_rows": draft_graph_batch_rows,
        "max_scheduled_batch_rows": max_scheduled_batch_rows,
        "device_cache_total_bytes": device_cache_total_bytes,
    }


def canonical_v4_flat_memory_plan(plan: V4FlatMemoryPlan) -> dict[str, object]:
    """Return the JSON-safe canonical fields covered by ``plan_fingerprint``."""

    return _canonical_v4_flat_memory_plan(
        max_total_tokens=plan.max_total_tokens,
        pools=plan.pools,
        scheduler_group_specs=plan.scheduler_group_specs,
        target_owner_group_specs=plan.target_owner_group_specs,
        draft_owner_group_specs=plan.draft_owner_group_specs,
        group_table_plans=plan.group_table_plans,
        payload_bytes=plan.payload_bytes,
        graph_metadata_bytes=plan.graph_metadata_bytes,
        forward_input_bytes=plan.forward_input_bytes,
        cpu_forward_export_bytes=plan.cpu_forward_export_bytes,
        cpu_forward_staging_bytes=plan.cpu_forward_staging_bytes,
        cpu_pool_metadata_bytes_estimate=plan.cpu_pool_metadata_bytes_estimate,
        cpu_request_metadata_bytes_estimate=plan.cpu_request_metadata_bytes_estimate,
        forward_buffer_depth=plan.forward_buffer_depth,
        target_graph_batch_rows=plan.target_graph_batch_rows,
        draft_graph_batch_rows=plan.draft_graph_batch_rows,
        max_scheduled_batch_rows=plan.max_scheduled_batch_rows,
        device_cache_total_bytes=plan.device_cache_total_bytes,
    )


def make_v4_flat_plan_agreement_record(
    plan: V4FlatMemoryPlan,
    *,
    rank: int,
) -> V4FlatPlanAgreementRecord:
    """Build the CPU-serializable record gathered by attention-TP peers."""

    return V4FlatPlanAgreementRecord(
        rank=rank,
        plan_fingerprint=plan.plan_fingerprint,
        canonical_plan=canonical_v4_flat_memory_plan(plan),
    )


def _first_canonical_difference(
    reference: object,
    candidate: object,
    *,
    path: str = "",
) -> tuple[str, object, object] | None:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        reference_keys = list(reference)
        candidate_extra_keys = sorted(set(candidate) - set(reference))
        for key in (*reference_keys, *candidate_extra_keys):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in reference:
                return child_path, "<missing>", candidate[key]
            if key not in candidate:
                return child_path, reference[key], "<missing>"
            difference = _first_canonical_difference(
                reference[key], candidate[key], path=child_path
            )
            if difference is not None:
                return difference
        return None

    if isinstance(reference, list) and isinstance(candidate, list):
        for index, (reference_item, candidate_item) in enumerate(
            zip(reference, candidate)
        ):
            difference = _first_canonical_difference(
                reference_item,
                candidate_item,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        if len(reference) != len(candidate):
            return f"{path}.length", len(reference), len(candidate)
        return None

    if reference != candidate:
        return path or "<root>", reference, candidate
    return None


def assert_v4_flat_plan_agreement(
    records: Sequence[V4FlatPlanAgreementRecord],
) -> None:
    """Fail on the first canonical V4 plan difference across TP ranks.

    A one-rank group is intentionally a no-op. Multi-rank callers must pass
    records in process-group rank order, as returned by ``all_gather_object``.
    """

    if len(records) <= 1:
        return

    for record in records:
        canonical_fingerprint = _canonical_hash(record.canonical_plan)
        if canonical_fingerprint != record.plan_fingerprint:
            raise RuntimeError(
                "DeepSeek V4 flat plan fingerprint does not cover its canonical "
                f"fields on rank {record.rank}: declared={record.plan_fingerprint}, "
                f"canonical={canonical_fingerprint}"
            )

    reference = records[0]
    for candidate in records[1:]:
        difference = _first_canonical_difference(
            reference.canonical_plan,
            candidate.canonical_plan,
        )
        if difference is None:
            continue
        field, reference_value, candidate_value = difference
        raise RuntimeError(
            "DeepSeek V4 flat plan differs across attention-TP ranks: "
            f"rank {candidate.rank} diverges from rank {reference.rank} at "
            f"canonical field {field}: rank {reference.rank}={reference_value!r}, "
            f"rank {candidate.rank}={candidate_value!r}; "
            f"fingerprints rank {reference.rank}={reference.plan_fingerprint}, "
            f"rank {candidate.rank}={candidate.plan_fingerprint}"
        )


def _normalize_owner_specs(
    specs: Sequence[PagedCacheGroupSpec],
    *,
    owner: FlatCacheOwner,
    owner_bit: int,
) -> tuple[PagedCacheGroupSpec, ...]:
    by_group: dict[str, PagedCacheGroupSpec] = {}
    for spec in specs:
        if not spec.group_id:
            raise ValueError(f"{owner} group_id must be non-empty")
        if spec.block_size_tokens is None or spec.block_size_tokens <= 0:
            raise ValueError(
                f"{owner} group {spec.group_id}: block_size_tokens must be positive"
            )
        if not spec.pool_id:
            raise ValueError(
                f"{owner} group {spec.group_id}: pool_id must be non-empty"
            )
        if spec.owner_mask not in (0, owner_bit):
            raise ValueError(
                f"{owner} group {spec.group_id}: owner_mask must be 0 or "
                f"{owner_bit}, got {spec.owner_mask}"
            )
        normalized = replace(spec, owner_mask=owner_bit)
        previous = by_group.get(spec.group_id)
        if previous is not None and previous != normalized:
            raise ValueError(
                f"{owner} group {spec.group_id}: duplicate group schema drift"
            )
        by_group[spec.group_id] = normalized
    return tuple(by_group[group_id] for group_id in sorted(by_group))


def _normalize_page_counts(
    specs: Sequence[PagedCacheGroupSpec],
    counts: Mapping[str, int] | None,
    *,
    owner: FlatCacheOwner,
) -> dict[str, int]:
    expected = {spec.group_id for spec in specs}
    if counts is None:
        if expected:
            raise ValueError(f"{owner}_group_page_counts are required")
        return {}
    actual = set(counts)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{owner}_group_page_counts keys do not match specs: "
            f"missing={missing}, extra={extra}"
        )
    normalized: dict[str, int] = {}
    for group_id in sorted(expected):
        count = counts[group_id]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(
                f"{owner} group {group_id}: total block count must be >= 2, "
                f"got {count!r}"
            )
        normalized[group_id] = count
    return normalized


def _merge_scheduler_specs(
    target_specs: Sequence[PagedCacheGroupSpec],
    draft_specs: Sequence[PagedCacheGroupSpec],
) -> tuple[PagedCacheGroupSpec, ...]:
    target_by_group = {spec.group_id: spec for spec in target_specs}
    draft_by_group = {spec.group_id: spec for spec in draft_specs}
    merged: list[PagedCacheGroupSpec] = []
    for group_id in sorted(target_by_group.keys() | draft_by_group.keys()):
        target = target_by_group.get(group_id)
        draft = draft_by_group.get(group_id)
        if target is not None and draft is not None:
            for field in _SCHEDULING_SCHEMA_FIELDS:
                target_value = getattr(target, field)
                draft_value = getattr(draft, field)
                if target_value != draft_value:
                    raise ValueError(
                        f"group {group_id} scheduling schema mismatch at {field}: "
                        f"target={target_value!r}, draft={draft_value!r}"
                    )
        base = target if target is not None else draft
        assert base is not None
        merged.append(
            replace(
                base,
                owner_mask=(target.owner_mask if target is not None else 0)
                | (draft.owner_mask if draft is not None else 0),
                required_producer_domain_mask=(
                    target.required_producer_domain_mask if target is not None else 0
                )
                | (draft.required_producer_domain_mask if draft is not None else 0),
            )
        )
    return tuple(merged)


def _normalize_components(
    target_components: Sequence[FlatComponentTensorPlan],
    draft_components: Sequence[FlatComponentTensorPlan],
    *,
    target_specs: Sequence[PagedCacheGroupSpec],
    draft_specs: Sequence[PagedCacheGroupSpec],
) -> tuple[tuple[str, FlatComponentTensorPlan], ...]:
    owner_specs = {
        "target": {spec.group_id: spec for spec in target_specs},
        "draft": {spec.group_id: spec for spec in draft_specs},
    }
    by_identity: dict[
        tuple[str, FlatCacheOwner, int, str],
        tuple[str, FlatComponentTensorPlan],
    ] = {}
    groups_with_components: dict[FlatCacheOwner, set[str]] = {
        "target": set(),
        "draft": set(),
    }
    for expected_owner, components in (
        ("target", target_components),
        ("draft", draft_components),
    ):
        for component in components:
            if component.owner != expected_owner:
                raise ValueError(
                    f"{expected_owner}_components contains {component.owner!r} "
                    f"component {component.component!r}"
                )
            spec = owner_specs[expected_owner].get(component.group_id)
            if spec is None:
                raise ValueError(
                    f"{expected_owner} component {component.component!r} refers "
                    f"to unknown group {component.group_id!r}"
                )
            groups_with_components[expected_owner].add(component.group_id)
            identity = (
                spec.pool_id,
                component.owner,
                component.layer,
                component.component,
            )
            previous = by_identity.get(identity)
            current = (spec.pool_id, component)
            if previous is not None and previous != current:
                raise ValueError(
                    "duplicate component identity has storage schema drift: "
                    f"pool={spec.pool_id!r}, owner={component.owner!r}, "
                    f"layer={component.layer}, component={component.component!r}"
                )
            by_identity[identity] = current

    for owner, specs in owner_specs.items():
        missing = sorted(set(specs) - groups_with_components[owner])
        if missing:
            raise ValueError(f"{owner} groups missing component tensors: {missing}")

    owner_order = {"target": 0, "draft": 1}
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (
                item[0],
                owner_order[item[1].owner],
                item[1].layer,
                item[1].component,
                item[1].group_id,
                item[1].dtype,
                item[1].shape_per_block,
                item[1].stride_bytes,
                item[1].alignment_bytes,
                item[1].bytes_per_block,
            ),
        )
    )


def build_v4_flat_memory_plan(
    *,
    max_total_tokens: int = 0,
    target_group_specs: Sequence[PagedCacheGroupSpec],
    target_group_page_counts: Mapping[str, int],
    target_components: Sequence[FlatComponentTensorPlan],
    draft_group_specs: Sequence[PagedCacheGroupSpec] = (),
    draft_group_page_counts: Mapping[str, int] | None = None,
    draft_components: Sequence[FlatComponentTensorPlan] = (),
    metadata_accounting: V4FlatMetadataAccounting | None = None,
) -> V4FlatMemoryPlan:
    """Build one canonical physical plan from target/draft owner-local inputs.

    Args:
        max_total_tokens: Token capacity represented by the plan.
        target_group_specs: Target model's owner-local logical group specs.
        target_group_page_counts: Target capacity demand including null page 0.
        target_components: Target component-plane storage schemas.
        draft_group_specs: Optional draft model owner-local group specs.
        draft_group_page_counts: Draft capacity demand including null page 0.
        draft_components: Draft component-plane storage schemas.
        metadata_accounting: Optional exact table/buffer accounting inputs.

    Returns:
        An immutable plan whose scheduler specs are the canonical target/draft
        union.  A shared group's capacity is the maximum owner demand while its
        bytes per block include every owner-namespaced component plane.

    Raises:
        ValueError: If logical scheduling schemas, counts, owner membership, or
        duplicate component storage schemas disagree.
    """
    if (
        isinstance(max_total_tokens, bool)
        or not isinstance(max_total_tokens, int)
        or max_total_tokens < 0
    ):
        raise ValueError("max_total_tokens must be an integer >= 0")
    if metadata_accounting is None:
        metadata_accounting = V4FlatMetadataAccounting()
    elif not isinstance(metadata_accounting, V4FlatMetadataAccounting):
        raise TypeError("metadata_accounting must be V4FlatMetadataAccounting or None")
    target_specs = _normalize_owner_specs(
        target_group_specs,
        owner="target",
        owner_bit=CACHE_OWNER_TARGET,
    )
    draft_specs = _normalize_owner_specs(
        draft_group_specs,
        owner="draft",
        owner_bit=CACHE_OWNER_DRAFT,
    )
    scheduler_specs = _merge_scheduler_specs(target_specs, draft_specs)
    if not scheduler_specs:
        raise ValueError("V4 flat memory plan requires at least one cache group")
    metadata_group_ids = {
        plan.group_id for plan in metadata_accounting.group_table_plans
    }
    scheduler_group_ids = {spec.group_id for spec in scheduler_specs}
    if metadata_group_ids and metadata_group_ids != scheduler_group_ids:
        missing = sorted(scheduler_group_ids - metadata_group_ids)
        extra = sorted(metadata_group_ids - scheduler_group_ids)
        raise ValueError(
            "metadata group plans do not match scheduler group union: "
            f"missing={missing}, extra={extra}"
        )
    if metadata_group_ids:
        target_group_ids = {spec.group_id for spec in target_specs}
        draft_group_ids = {spec.group_id for spec in draft_specs}
        for table_plan in metadata_accounting.group_table_plans:
            target_has_capture = table_plan.target_capture_cols > 0
            draft_has_capture = table_plan.draft_capture_cols > 0
            target_owns_group = table_plan.group_id in target_group_ids
            draft_owns_group = table_plan.group_id in draft_group_ids
            if (
                target_has_capture != target_owns_group
                or draft_has_capture != draft_owns_group
            ):
                raise ValueError(
                    "flat table capture owner membership does not match group "
                    f"specs for {table_plan.group_id!r}: "
                    f"capture=(target={target_has_capture}, "
                    f"draft={draft_has_capture}), "
                    f"owners=(target={target_owns_group}, "
                    f"draft={draft_owns_group})"
                )
    groups_by_pool: dict[str, str] = {}
    for spec in scheduler_specs:
        previous_group = groups_by_pool.setdefault(spec.pool_id, spec.group_id)
        if previous_group != spec.group_id:
            raise ValueError(
                f"V4 flat pool {spec.pool_id!r} must bind exactly one group, "
                f"got {previous_group!r} and {spec.group_id!r}"
            )

    target_counts = _normalize_page_counts(
        target_specs,
        target_group_page_counts,
        owner="target",
    )
    draft_counts = _normalize_page_counts(
        draft_specs,
        draft_group_page_counts,
        owner="draft",
    )
    components_with_pools = _normalize_components(
        target_components,
        draft_components,
        target_specs=target_specs,
        draft_specs=draft_specs,
    )

    pool_ids = sorted({spec.pool_id for spec in scheduler_specs})
    pools: list[FlatBlockPoolPlan] = []
    for pool_id in pool_ids:
        group_ids = {
            spec.group_id for spec in scheduler_specs if spec.pool_id == pool_id
        }
        capacity_demands = [
            counts[group_id]
            for counts in (target_counts, draft_counts)
            for group_id in group_ids
            if group_id in counts
        ]
        if not capacity_demands:
            raise ValueError(f"pool {pool_id!r} has no capacity demand")
        tensors = tuple(
            component
            for component_pool_id, component in components_with_pools
            if component_pool_id == pool_id
        )
        if not tensors:
            raise ValueError(f"pool {pool_id!r} has no component tensors")
        bytes_per_block = sum(tensor.bytes_per_block for tensor in tensors)
        storage_schema_hash = _canonical_hash(
            {
                "pool_id": pool_id,
                "tensors": [_canonical_component(tensor) for tensor in tensors],
            }
        )
        pools.append(
            FlatBlockPoolPlan(
                pool_id=pool_id,
                total_blocks=max(capacity_demands),
                bytes_per_block=bytes_per_block,
                storage_schema_hash=storage_schema_hash,
                tensors=tensors,
            )
        )

    payload_bytes = sum(pool.total_blocks * pool.bytes_per_block for pool in pools)
    cpu_pool_metadata_bytes_estimate = sum(
        pool.total_blocks
        * metadata_accounting.cpu_pool_metadata_bytes_per_block_estimate
        + metadata_accounting.cpu_pool_fixed_bytes_estimate
        for pool in pools
    )
    device_cache_total_bytes = (
        payload_bytes
        + metadata_accounting.graph_metadata_bytes
        + metadata_accounting.forward_input_bytes
    )
    canonical_plan = _canonical_v4_flat_memory_plan(
        max_total_tokens=max_total_tokens,
        pools=pools,
        scheduler_group_specs=scheduler_specs,
        target_owner_group_specs=target_specs,
        draft_owner_group_specs=draft_specs,
        group_table_plans=metadata_accounting.group_table_plans,
        payload_bytes=payload_bytes,
        graph_metadata_bytes=metadata_accounting.graph_metadata_bytes,
        forward_input_bytes=metadata_accounting.forward_input_bytes,
        cpu_forward_export_bytes=metadata_accounting.cpu_forward_export_bytes,
        cpu_forward_staging_bytes=metadata_accounting.cpu_forward_staging_bytes,
        cpu_pool_metadata_bytes_estimate=cpu_pool_metadata_bytes_estimate,
        cpu_request_metadata_bytes_estimate=(
            metadata_accounting.cpu_request_metadata_bytes_estimate
        ),
        forward_buffer_depth=metadata_accounting.forward_buffer_depth,
        target_graph_batch_rows=metadata_accounting.target_graph_batch_rows,
        draft_graph_batch_rows=metadata_accounting.draft_graph_batch_rows,
        max_scheduled_batch_rows=metadata_accounting.max_scheduled_batch_rows,
        device_cache_total_bytes=device_cache_total_bytes,
    )
    return V4FlatMemoryPlan(
        max_total_tokens=max_total_tokens,
        pools=tuple(pools),
        scheduler_group_specs=scheduler_specs,
        target_owner_group_specs=target_specs,
        draft_owner_group_specs=draft_specs,
        group_table_plans=metadata_accounting.group_table_plans,
        payload_bytes=payload_bytes,
        graph_metadata_bytes=metadata_accounting.graph_metadata_bytes,
        forward_input_bytes=metadata_accounting.forward_input_bytes,
        cpu_forward_export_bytes=metadata_accounting.cpu_forward_export_bytes,
        cpu_forward_staging_bytes=metadata_accounting.cpu_forward_staging_bytes,
        cpu_pool_metadata_bytes_estimate=cpu_pool_metadata_bytes_estimate,
        cpu_request_metadata_bytes_estimate=(
            metadata_accounting.cpu_request_metadata_bytes_estimate
        ),
        forward_buffer_depth=metadata_accounting.forward_buffer_depth,
        target_graph_batch_rows=metadata_accounting.target_graph_batch_rows,
        draft_graph_batch_rows=metadata_accounting.draft_graph_batch_rows,
        max_scheduled_batch_rows=metadata_accounting.max_scheduled_batch_rows,
        device_cache_total_bytes=device_cache_total_bytes,
        plan_fingerprint=_canonical_hash(canonical_plan),
    )
