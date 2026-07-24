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

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from tokenspeed.runtime.configs.flat_kv_contract import STATE_LAYER_TYPES


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

FLAT_GRAPH_METADATA_ELEMENT_BYTES = 4
# One row describes table source offset/width, table destination offset/width,
# and source/destination base offsets for one owner/group.
FLAT_PACKED_UNPACK_META_FIELDS = 6


def require_sha256_hexdigest(value: str, *, field_name: str) -> None:
    """Validate the canonical lowercase SHA-256 representation."""
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex")


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
        require_sha256_hexdigest(
            self.storage_schema_hash,
            field_name="storage_schema_hash",
        )


@dataclass(frozen=True)
class FlatGroupTablePlan:
    """Canonical capture/export bounds for one scheduler-union group."""

    group_id: str
    target_capture_cols: int
    draft_capture_cols: int
    max_export_cols: int

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("flat group table plan group_id must be non-empty")
        for name, value in (
            ("target_capture_cols", self.target_capture_cols),
            ("draft_capture_cols", self.draft_capture_cols),
            ("max_export_cols", self.max_export_cols),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        if self.target_capture_cols == 0 and self.draft_capture_cols == 0:
            raise ValueError("flat group table plan must belong to target and/or draft")
        if self.max_export_cols <= 0:
            raise ValueError("flat group max_export_cols must be positive")
        if (
            max(self.target_capture_cols, self.draft_capture_cols)
            > self.max_export_cols
        ):
            raise ValueError("flat group capture columns cannot exceed max_export_cols")
        if (
            self.target_capture_cols > 0
            and self.draft_capture_cols > 0
            and self.target_capture_cols != self.draft_capture_cols
        ):
            raise ValueError(
                "a shared flat group must use one target/draft capture width"
            )

    def capture_cols(self, owner: FlatCacheOwner) -> int:
        if owner == "target":
            return self.target_capture_cols
        if owner == "draft":
            return self.draft_capture_cols
        raise ValueError(f"unsupported flat cache owner {owner!r}")


def _flat_graph_owner_metadata_bytes(
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
        FLAT_GRAPH_METADATA_ELEMENT_BYTES
        * batch_rows
        * sum(
            plan.capture_cols(owner) + 1
            for plan in group_table_plans
            if plan.capture_cols(owner) > 0
        )
    )


def _flat_forward_input_metadata_bytes(
    group_table_plans: Sequence[FlatGroupTablePlan],
    *,
    buffer_depth: int,
    batch_rows: int,
    graph_batch_rows: int,
) -> int:
    """Exact packed table/base payload plus graph-unpack header bytes."""
    for name, value in (("buffer_depth", buffer_depth), ("batch_rows", batch_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"flat {name} must be an integer >= 0")
    if (
        isinstance(graph_batch_rows, bool)
        or not isinstance(graph_batch_rows, int)
        or graph_batch_rows < 0
    ):
        raise ValueError("flat graph_batch_rows must be an integer >= 0")
    if buffer_depth == 0:
        raise ValueError("flat buffer_depth must be positive")
    payload_bytes = (
        FLAT_GRAPH_METADATA_ELEMENT_BYTES
        * buffer_depth
        * batch_rows
        * sum(plan.max_export_cols + 1 for plan in group_table_plans)
    )
    if graph_batch_rows == 0:
        return payload_bytes
    owner_group_count = sum(
        int(plan.target_capture_cols > 0) + int(plan.draft_capture_cols > 0)
        for plan in group_table_plans
    )
    header_bytes = (
        FLAT_GRAPH_METADATA_ELEMENT_BYTES
        * buffer_depth
        * owner_group_count
        * FLAT_PACKED_UNPACK_META_FIELDS
    )
    return payload_bytes + header_bytes


@dataclass(frozen=True)
class FlatRuntimeMetadataPlan:
    """Canonical table geometry and exact device metadata accounting."""

    group_table_plans: tuple[FlatGroupTablePlan, ...]
    forward_buffer_depth: int
    graph_batch_rows: int
    max_scheduled_batch_rows: int

    def __post_init__(self) -> None:
        plans = tuple(sorted(self.group_table_plans, key=lambda item: item.group_id))
        object.__setattr__(self, "group_table_plans", plans)
        group_ids = [plan.group_id for plan in plans]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("flat runtime metadata has duplicate group plans")
        if not plans:
            raise ValueError("flat runtime metadata requires group table plans")
        for name in (
            "graph_batch_rows",
            "max_scheduled_batch_rows",
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
        if self.graph_batch_rows > self.max_scheduled_batch_rows:
            raise ValueError(
                "flat metadata graph batch rows cannot exceed scheduled rows: "
                f"graph={self.graph_batch_rows}, "
                f"scheduled={self.max_scheduled_batch_rows}"
            )

    @property
    def graph_metadata_bytes(self) -> int:
        """Exact target plus draft CUDA-graph table/base device bytes."""

        return self.graph_metadata_bytes_for_owner(
            "target"
        ) + self.graph_metadata_bytes_for_owner("draft")

    @property
    def forward_input_bytes(self) -> int:
        """Exact ring-buffered eager table/base device bytes."""

        return _flat_forward_input_metadata_bytes(
            self.group_table_plans,
            buffer_depth=self.forward_buffer_depth,
            batch_rows=self.max_scheduled_batch_rows,
            graph_batch_rows=self.graph_batch_rows,
        )

    def graph_capture_cols_by_group(self, owner: FlatCacheOwner) -> dict[str, int]:
        return {
            plan.group_id: plan.capture_cols(owner)
            for plan in self.group_table_plans
            if plan.capture_cols(owner) > 0
        }

    def graph_metadata_bytes_for_owner(self, owner: FlatCacheOwner) -> int:
        return _flat_graph_owner_metadata_bytes(
            self.group_table_plans,
            owner=owner,
            batch_rows=self.graph_batch_rows,
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
                    f"{owner} CUDA graph metadata allocation disagrees with "
                    f"flat plan: actual={actual}, expected={expected}"
                )
        actual_total = target_actual_bytes + draft_actual_bytes
        if actual_total != self.graph_metadata_bytes:
            raise RuntimeError(
                "CUDA graph metadata allocation disagrees with flat plan total: "
                f"actual={actual_total}, expected={self.graph_metadata_bytes}"
            )


def validate_flat_graph_owner_allocation(
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
                f"{owner} CUDA graph {label} groups disagree with "
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
                f"{owner} CUDA graph shape disagrees with flat plan "
                f"for {group_id!r}: table={actual_table_shape}, "
                f"expected_table={expected_table_shape}, base={actual_base_shape}, "
                f"expected_base={expected_base_shape}"
            )
        expected_table_nbytes = FLAT_GRAPH_METADATA_ELEMENT_BYTES * batch_rows * cols
        expected_base_nbytes = FLAT_GRAPH_METADATA_ELEMENT_BYTES * batch_rows
        actual_table_nbytes = table_nbytes[group_id]
        actual_base_nbytes = base_nbytes[group_id]
        if (
            actual_table_nbytes != expected_table_nbytes
            or actual_base_nbytes != expected_base_nbytes
        ):
            raise RuntimeError(
                f"{owner} CUDA graph bytes disagree with flat plan "
                f"for {group_id!r}: table={actual_table_nbytes}, "
                f"expected_table={expected_table_nbytes}, "
                f"base={actual_base_nbytes}, expected_base={expected_base_nbytes}"
            )
        actual_total += actual_table_nbytes + actual_base_nbytes
    return actual_total


__all__ = [
    "BlockGeometry",
    "ComponentSpec",
    "FLAT_GRAPH_METADATA_ELEMENT_BYTES",
    "FLAT_PACKED_UNPACK_META_FIELDS",
    "FlatBlockPoolPlan",
    "FlatCacheOwner",
    "FlatComponentTensorPlan",
    "FlatGroupTablePlan",
    "FlatMemoryPlan",
    "FlatRuntimeMetadataPlan",
    "LayerBinding",
    "TensorPlan",
    "components_from_layers",
    "equalized_block_size",
    "occurrence_index",
    "plan_component_tensors",
    "plan_tensors",
    "require_sha256_hexdigest",
    "solve_page_geometry",
    "state_const_bytes",
    "validate_flat_graph_owner_allocation",
]
