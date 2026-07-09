"""Flat KV-cache memory plan: pure sizing/binding decisions, no torch.

Components declare per-page bytes as a function of P (page_size_tokens):
linear components scale (bytes_per_slot > 0), constant components do not
(const_bytes > 0, mamba state snapshots). Same-(group, layer) components
pack into one page row ([conv|ssm|pad], the vLLM hybrid layout). Two
equalizer moves: constant rows inflate P until the widest linear row
covers them (vLLM align); linear rows pad to the widest at binding time.
plan_tensors then pairs physical slot j with the j-th layer of every
group over a single page-id space and sizes each slab from the budget.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


# Labels whose group is state-family (recurrent state rows, not KV history).
# Deliberate one-line duplicate of paged_cache_spec._STATE_LAYER_TYPES: both
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
class PageGeometry:
    page_size_tokens: int
    page_bytes: int
    num_pages: int = 0  # filled by plan_tensors from the memory budget


def components_from_layers(*, layer_types, kv_bytes_per_slot, state_const_bytes):
    """Per-layer ComponentSpecs: history layers carry one linear kv component;
    state layers one constant component per state tensor. Layer index is the
    within-group occurrence count (the slab pairing order)."""
    counts: dict[str, int] = {}
    comps: list[ComponentSpec] = []
    for label in layer_types:
        idx = counts.get(label, 0)
        counts[label] = idx + 1
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


def solve_page_geometry(components, *, page_size_tokens, alignment):
    """Smallest P >= page_size_tokens (multiple of `alignment` when inflated)
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
        if needed > page_size_tokens:
            page_size_tokens = alignment * math.ceil(needed / alignment)
    page_bytes = max(max_linear * page_size_tokens, max_const)
    return PageGeometry(page_size_tokens=page_size_tokens, page_bytes=page_bytes)


@dataclass(frozen=True)
class LayerBinding:
    slot: int
    group_id: str
    layer: int
    component: str
    nbytes_per_page: int
    row_offset: int  # byte offset of this component within its (group, layer) page row


@dataclass(frozen=True)
class TensorPlan:
    name: str
    nbytes: int
    bindings: tuple[LayerBinding, ...]


@dataclass(frozen=True)
class FlatMemoryPlan:
    geometry: PageGeometry
    tensors: tuple[TensorPlan, ...]


def plan_tensors(components, *, page_size_tokens, alignment, budget_bytes):
    """Pair slot j with the j-th layer of every group over one page-id space."""
    geo = solve_page_geometry(
        components, page_size_tokens=page_size_tokens, alignment=alignment
    )
    layers_by_group: dict[str, list[int]] = {}
    for c in components:
        layers = layers_by_group.setdefault(c.group_id, [])
        if c.layer not in layers:
            layers.append(c.layer)
    num_slots = max(len(v) for v in layers_by_group.values())
    num_pages = budget_bytes // (num_slots * geo.page_bytes)
    if num_pages <= 1:
        raise ValueError("budget too small for one usable page per slot")
    geo = PageGeometry(geo.page_size_tokens, geo.page_bytes, num_pages)

    tensors = []
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
                nbytes = c.bytes_per_slot * geo.page_size_tokens + c.const_bytes
                bindings.append(
                    LayerBinding(slot, gid, layer, c.component, nbytes, row_offset)
                )
                row_offset += nbytes
        tensors.append(
            TensorPlan(
                name=f"flat_slab_{slot}",
                nbytes=num_pages * geo.page_bytes,
                bindings=tuple(bindings),
            )
        )
    return FlatMemoryPlan(geometry=geo, tensors=tuple(tensors))
