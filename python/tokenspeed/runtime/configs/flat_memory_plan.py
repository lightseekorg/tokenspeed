"""Flat KV-cache memory plan: pure sizing/binding decisions, no torch.

Components declare per-page bytes as a function of P (page_size_tokens):
linear components scale (bytes_per_slot > 0), constant components do not
(const_bytes > 0, mamba state snapshots). Same-(group, layer) components
pack into one page row ([conv|ssm|pad], the vLLM hybrid layout). Two
equalizer moves: constant rows inflate P until the widest linear row
covers them (vLLM align); linear rows pad to the widest at binding time.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


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
        needed = math.ceil(max_const / max_linear)
        if needed > page_size_tokens:
            page_size_tokens = alignment * math.ceil(needed / alignment)
    page_bytes = max(max_linear * page_size_tokens, max_const)
    return PageGeometry(page_size_tokens=page_size_tokens, page_bytes=page_bytes)
