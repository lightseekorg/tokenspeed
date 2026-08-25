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

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

import torch

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker

logger = get_colorful_logger(__name__)

_LAYER_FIELD = re.compile(r"^layer\.(\d+)\.(.+)$")


def _layer_plane(
    field_id: str, first_layer: int, num_layers: int
) -> tuple[int, str] | None:
    """Split a planned field id into this view's local layer id and plane.

    Returns None for fields outside the view's layer window, and for fields
    that are not per-layer at all.
    """
    match = _LAYER_FIELD.match(field_id)
    if match is None:
        return None
    global_layer = int(match.group(1))
    local_layer = global_layer - first_layer
    if not 0 <= local_layer < num_layers:
        return None
    return local_layer, match.group(2)


def derive_state_groups_by_layer(
    arena: CacheArena,
    *,
    first_layer: int,
    num_layers: int,
    state_layer_ids: Iterable[int],
) -> dict[int, str]:
    """Map each recurrent layer to the state-family group holding its fields.

    The memory plan is the single record of which group a layer's fields were
    declared in, so the mapping is read back from the planned fields rather
    than carried as a parallel per-layer tuple.

    Args:
        arena: The cache arena whose plan and group specs to read.
        first_layer: This view's first layer in the merged plan.
        num_layers: Number of layers in this view's window.
        state_layer_ids: View-local ids of the recurrent (state) layers.

    Returns:
        View-local layer id -> state-family group id, one entry per state
        layer whose fields the plan declares inside this view's window.

    Raises:
        ValueError: a state layer's fields span more than one state group.
    """
    state_groups = {
        spec.group_id for spec in arena.cache_group_specs if spec.family == "state"
    }
    wanted = set(state_layer_ids)
    mapping: dict[int, str] = {}
    for field in arena.plan.fields:
        located = _layer_plane(field.field_id, first_layer, num_layers)
        if located is None:
            continue
        layer_id, _ = located
        if layer_id not in wanted or field.group_id not in state_groups:
            continue
        existing = mapping.setdefault(layer_id, field.group_id)
        if existing != field.group_id:
            raise ValueError(
                f"layer {layer_id} has state fields in more than one cache "
                f"group: {existing!r} and {field.group_id!r}"
            )
    return mapping


class CachePool(ABC):
    """One model's typed layer window onto a shared cache arena.

    A pool owns no memory and no geometry: ``self.arena`` owns the
    allocation, the field views, the plan and the scheduler contract, and
    callers that want any of those ask the arena directly. What a pool
    adds is per-view: the dtype its kernels read these bytes as, where its
    layer window starts in the merged plan, and the per-layer buffers its
    kernels index. Target and draft are therefore two pools over one
    arena -- and may read it as different dtypes.
    """

    # Pools that alias recurrent-state bytes and KV in one buffer must
    # zero physical pages on reuse to avoid poisoned tails. Pure-attention
    # pools do not alias state, so reused pages need no sanitization.
    requires_page_zeroing: bool = False

    def __init__(
        self,
        arena: CacheArena,
        dtype: torch.dtype,
        rank: int,
        *,
        field_layer_offset: int = 0,
    ):
        self.arena = arena
        self.dtype = dtype
        self.rank = rank
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            #  Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self._field_layer_offset = int(field_layer_offset)
        if self._field_layer_offset < 0:
            raise ValueError("field_layer_offset must be non-negative")
        # default state for optional layer-wise transfer control
        self.layerwise_load_tracker = None
        logger.info(
            "Initialized cache view over %d slots as %s, layers from %d, rank %d",
            arena.size,
            dtype,
            self._field_layer_offset,
            rank,
        )

    def _field_layer_id(self, layer_id: int) -> int:
        """Map this compute view's local layer id into the merged plan.

        Callers pass the id their own model numbers the layer with, which for a
        draft view means ``0..num_draft_layers-1``. Reject anything outside the
        window: a global id offset a second time would silently address another
        model's planes.
        """
        if not 0 <= layer_id < self.layer_num:
            raise ValueError(
                f"layer {layer_id} is outside this cache view's window of "
                f"{self.layer_num} layers (ids are local to the view)"
            )
        return self._field_layer_offset + layer_id

    # Per-layer plane name -> the attribute holding its per-layer list.
    # Subclasses declare only the planes their kernels read; a plane they do
    # not name is not this view's concern.
    layer_plane_bindings: ClassVar[dict[str, str]] = {}

    def _bind_layer_planes(self) -> None:
        """Arrange this view's planned per-layer fields into kernel buffers.

        The plan names every field, its dtype, its shape and which layer it
        belongs to, and the arena already materialized every view in the
        shape it is addressed by. Walk the plan once, keep the fields inside
        this view's layer window, and file each under the attribute its
        kernels read.
        """
        lists = {
            attribute: [None] * self.layer_num
            for attribute in self.layer_plane_bindings.values()
        }
        for field in self.arena.plan.fields:
            located = _layer_plane(
                field.field_id, self._field_layer_offset, self.layer_num
            )
            if located is None:
                continue
            layer_id, plane = located
            attribute = self.layer_plane_bindings.get(plane)
            if attribute is None:
                continue
            lists[attribute][layer_id] = self.arena.field(field.field_id)
        for attribute, values in lists.items():
            setattr(self, attribute, values)

    def register_layerwise_load_tracker(
        self, layerwise_load_tracker: LayerwiseLoadTracker
    ) -> None:
        self.layerwise_load_tracker = layerwise_load_tracker

    def cache_transfer_layout(self):
        """Return the transfer layout consumed by this compute view."""
        from tokenspeed.runtime.cache.transfer.layout import (
            select_layer_fields,
        )

        try:
            field_ids, consumers = select_layer_fields(
                self.arena.plan.fields,
                first_layer=self._field_layer_offset,
                num_layers=self.layer_num,
            )
        except (AttributeError, IndexError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        return self._build_cache_transfer_layout(field_ids, consumers)

    def _build_cache_transfer_layout(self, field_ids, consumers):
        from tokenspeed.runtime.cache.transfer.layout import layout_from_lcm_plan

        local_group_ids = {
            field.group_id
            for field in self.arena.plan.fields
            if field.field_id in field_ids
        }
        scheduler_group_ids = tuple(
            spec.group_id
            for spec in self.arena.cache_group_specs
            if spec.group_id in local_group_ids
        )
        return layout_from_lcm_plan(
            self.arena.plan,
            self.arena.buffer,
            consumers=consumers,
            group_ids=scheduler_group_ids or None,
            field_ids=field_ids,
        )

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        """Zero the shared cache arena after sleep/wake remaps its storage."""
        # The event loop visits both target and draft pools; both name the
        # same arena, and zeroing it twice is harmless.
        self.arena.clear()

    # ------------------------------------------------------------------
    # What every cache view owes its kernels. Abstract, so a subclass that
    # forgets one fails at construction instead of at the first write.
    # ------------------------------------------------------------------

    @abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """This layer's K plane, in the shape its kernels read."""

    @abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """This layer's V plane, in the shape its kernels read."""

    @abstractmethod
    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Both of this layer's planes at once."""

    @abstractmethod
    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Scatter one forward pass's K/V into this layer's planes."""
