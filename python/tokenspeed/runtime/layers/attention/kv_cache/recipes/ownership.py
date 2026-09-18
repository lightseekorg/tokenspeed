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

"""Cache-layer ownership for resident fields and producer readiness.

Callers supply stage windows in the cache-layer namespace. Independent draft
cache layers belong to one producer on the final stage. Ownership is pure so
model construction and PD need no allocation or communication dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheLayerOwnership:
    """One stage's target producers and optional trailing draft producer.

    Windows and counts use the merged cache-layer namespace: target cache
    layers first, then independent draft cache layers. The final stage's
    resident window includes the draft fields, which become ready together
    at one final producer barrier.
    """

    num_target_cache_layers: int
    num_draft_cache_layers: int
    target_cache_window: tuple[int, int]

    def __post_init__(self) -> None:
        if self.num_target_cache_layers < 1 or self.num_draft_cache_layers < 0:
            raise ValueError("cache layer counts must be non-negative, target >= 1")
        start, end = self.target_cache_window
        if not 0 <= start < end <= self.num_target_cache_layers:
            raise ValueError(
                "cache layer window is outside the target cache layer range"
            )

    @property
    def owns_draft_cache(self) -> bool:
        return (
            self.num_draft_cache_layers > 0
            and self.target_cache_window[1] == self.num_target_cache_layers
        )

    @property
    def resident_cache_window(self) -> tuple[int, int]:
        start, end = self.target_cache_window
        return start, end + (
            self.num_draft_cache_layers if self.owns_draft_cache else 0
        )

    @property
    def producer_cache_layers(self) -> tuple[tuple[int, ...], ...]:
        start, end = self.target_cache_window
        steps = tuple((layer,) for layer in range(start, end))
        if self.owns_draft_cache:
            steps += (tuple(range(end, end + self.num_draft_cache_layers)),)
        return steps


def pipeline_cache_ownership(
    num_target_cache_layers: int,
    num_draft_cache_layers: int,
    target_cache_windows: Sequence[tuple[int, int]],
) -> tuple[CacheLayerOwnership, ...]:
    """Return one cache owner per stage, with draft fields only on the last.

    Args:
        num_target_cache_layers: Leading cache layers declared by the target recipe.
        num_draft_cache_layers: Trailing independent draft cache layers.
        target_cache_windows: Stage windows already mapped to target cache-layer
            IDs, covering the target namespace once in order. Execution-layer
            partitioning belongs to the caller.

    Returns:
        Owners in stage order, sharing the declared cache-layer counts.
    """
    owners = tuple(
        CacheLayerOwnership(num_target_cache_layers, num_draft_cache_layers, window)
        for window in target_cache_windows
    )
    if (
        not owners
        or owners[0].target_cache_window[0] != 0
        or owners[-1].target_cache_window[1] != num_target_cache_layers
        or any(
            left.target_cache_window[1] != right.target_cache_window[0]
            for left, right in zip(owners, owners[1:])
        )
    ):
        raise ValueError("stage cache windows must cover the target cache layers once")
    return owners


def cache_field_placement(plan, owners: tuple[CacheLayerOwnership, ...]):
    """Resolve model ownership once into field sets and local readiness steps.

    Args:
        plan: The complete logical cache memory plan.
        owners: Cache construction's stage owners, in pipeline order.

    Returns:
        A pair of stage-indexed tuples: resident field IDs and field IDs grouped
        by producer step. Transfer code consumes these without model identities.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        cache_field_layer_id,
    )

    by_layer: dict[int, list[str]] = {}
    for field in plan.fields:
        by_layer.setdefault(cache_field_layer_id(field.field_id), []).append(
            field.field_id
        )
    schedules = tuple(
        tuple(
            tuple(field for layer in layers for field in by_layer.get(layer, ()))
            for layers in owner.producer_cache_layers
        )
        for owner in owners
    )
    resident = tuple(
        tuple(field for step in schedule for field in step) for schedule in schedules
    )
    flattened = [field for fields in resident for field in fields]
    if len(flattened) != len(set(flattened)) or set(flattened) != {
        field.field_id for field in plan.fields
    }:
        raise ValueError("cache ownership must cover every field exactly once")
    return resident, schedules
