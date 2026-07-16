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
from dataclasses import dataclass, replace

# Labels whose group is state-family (recurrent state rows, not KV history).
# Deliberate one-line duplicate of paged_cache_spec.STATE_LAYER_TYPES: both
# modules are direct-loaded standalone by their tests (importlib, no package
# context), so a cross-module import would break either loader. Keep in sync.
STATE_LAYER_TYPES = frozenset({"linear_attention"})


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
