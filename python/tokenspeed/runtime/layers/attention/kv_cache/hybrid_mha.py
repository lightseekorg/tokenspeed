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

"""Paged MHA-history and recurrent-state cache."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

import torch

from tokenspeed.runtime.layers.attention.kv_cache.base import (
    derive_state_groups_by_layer,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import (
    MHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_LAYER_TYPES,
)


class HybridMHATokenToKVPool(MHATokenToKVPool):
    """MHA compute interface whose history and state share one buffer."""

    def __init__(self, *, layer_types: tuple[str, ...], **kwargs):
        layer_types = tuple(layer_types)
        self._state_layer_ids = tuple(
            layer_id
            for layer_id, label in enumerate(layer_types)
            if label in STATE_LAYER_TYPES
        )
        self._state_buffers_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.requires_page_zeroing = True

        if len(layer_types) != kwargs["layer_num"]:
            raise ValueError("cache layer types must cover every model layer")

        super().__init__(**kwargs)

    # A state layer has no k/v field planned and an attention layer has no
    # conv/ssm, so the plan's field list decides which planes a layer has.
    # State planes keep their planned shape: the GDN decode ABI reads them as
    # the plan lays them out.
    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MHATokenToKVPool.layer_plane_bindings,
        "conv": "_conv_state",
        "ssm": "_ssm_state",
    }

    def _bind_layer_planes(self) -> None:
        super()._bind_layer_planes()
        self._state_buffers_by_layer = {
            layer_id: (self._conv_state[layer_id], self._ssm_state[layer_id])
            for layer_id in self._state_layer_ids
        }

    @cached_property
    def state_group_by_layer(self) -> dict[int, str]:
        """View-local state layer id -> its state-family cache group id."""
        return derive_state_groups_by_layer(
            self.arena,
            first_layer=self._field_layer_offset,
            num_layers=self.layer_num,
            state_layer_ids=self._state_layer_ids,
            state_field_suffixes=("conv", "ssm"),
        )

    def get_state_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id not in self._state_layer_ids:
            raise ValueError(f"layer {layer_id} is not a state layer")
        try:
            return self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no bound state fields") from exc

    def get_component(self, layer_id: int, component_name: str) -> torch.Tensor:
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        conv, recurrent = self.get_state_buffers(layer_id)
        if component_name == "conv_state":
            return conv
        if component_name == "recurrent_state":
            return recurrent
        raise ValueError(f"unknown state component {component_name!r}")

    def zero_new_blocks(self, new_page_ids: dict[str, list[int]]) -> None:
        if new_page_ids:
            self.arena.zero_blocks(new_page_ids)


class HybridMHATokenToKVPoolMXFP8(
    HybridMHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
):
    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **HybridMHATokenToKVPool.layer_plane_bindings,
        **MHATokenToKVPoolMXFP8.layer_plane_bindings,
    }
