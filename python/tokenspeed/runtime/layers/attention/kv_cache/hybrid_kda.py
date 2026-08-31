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

"""Paged latent-KV and KDA-state cache."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

import torch

from tokenspeed.runtime.layers.attention.kda_geometry import (
    kda_conv_state_channel_axis,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import (
    derive_state_groups_by_layer,
)
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_LAYER_TYPES,
)


class HybridKDATokenToKVPool(MLATokenToKVPool):
    """MLA compute interface whose latent KV and KDA state share one buffer."""

    #: Sanitizing latent write: under the prefill graph a padded row's NaN
    #: reaches a live row's softmax through the shared dummy slot, because the
    #: paged MLA decode computes ``q·k`` before the causal mask.
    latent_write_sanitizes: ClassVar[bool] = True

    def __init__(
        self,
        *,
        layer_types: tuple[str, ...],
        **kwargs,
    ):
        self._layer_types = tuple(layer_types)
        self._state_buffers_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.requires_page_zeroing = True

        if len(self._layer_types) != kwargs["layer_num"]:
            raise ValueError("cache layer types must cover every model layer")

        super().__init__(**kwargs)

    # A KDA layer has conv/recurrent planes planned and an MLA layer has a
    # latent plane; the plan's field list decides per layer. Latent-page
    # contiguity is a plan invariant (exact_page_stride).
    # Recurrent planes keep their planned shape. A history-major convolution
    # plane is exposed as a zero-copy sequence-major compute view below.
    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MLATokenToKVPool.layer_plane_bindings,
        "conv_state": "_conv_state",
        "recurrent_state": "_recurrent_state",
    }

    def _bind_layer_planes(self) -> None:
        if self.quant_method == "per_token_head":
            raise ValueError("KDA cache does not support per-token-head KV")
        super()._bind_layer_planes()
        # A state layer with no planned planes belongs to another pipeline
        # stage (the PP-narrowed plan drops its fields); this view never
        # executes it, so it gets no entry.
        self._state_buffers_by_layer = {}
        for layer_id, label in enumerate(self._layer_types):
            physical_conv = self._conv_state[layer_id]
            if label not in STATE_LAYER_TYPES or physical_conv is None:
                continue
            if physical_conv.ndim != 3:
                raise RuntimeError(
                    "KDA convolution state must have three dimensions, "
                    f"got {tuple(physical_conv.shape)}"
                )
            recurrent = self._recurrent_state[layer_id]
            if recurrent is None:
                raise RuntimeError("KDA convolution state has no recurrent peer")
            channels = 3 * recurrent.shape[1] * recurrent.shape[-1]
            channel_axis = kda_conv_state_channel_axis(
                tuple(physical_conv.shape[1:]), channels=channels
            )
            conv = physical_conv if channel_axis == 0 else physical_conv.transpose(1, 2)
            channels, history = conv.shape[1:]
            supported_strides = {(history, 1), (1, channels)}
            if conv.stride()[1:] not in supported_strides:
                raise RuntimeError(
                    "KDA convolution state must use a dense supported layout, "
                    f"got {tuple(conv.stride())}"
                )
            self._state_buffers_by_layer[layer_id] = (
                conv,
                recurrent,
            )

    @property
    def num_lcm_blocks(self) -> int:
        return self.arena.plan.num_lcm_blocks

    @property
    def state_slabs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(self._state_buffers_by_layer.values())

    @cached_property
    def state_group_by_layer(self) -> dict[int, str]:
        """View-local state layer id -> its state-family cache group id."""
        return derive_state_groups_by_layer(
            self.arena,
            first_layer=self._field_layer_offset,
            num_layers=self.layer_num,
            state_layer_ids=(
                layer_id
                for layer_id, label in enumerate(self._layer_types)
                if label in STATE_LAYER_TYPES
            ),
            state_field_suffixes=("conv_state", "recurrent_state"),
        )

    def get_component(self, layer_id: int, component_name: str) -> torch.Tensor:
        """Return one KDA state plane. Latent KV is read via ``kv_buffer``."""
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        try:
            conv, recurrent = self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no KDA state") from exc
        if component_name == "conv_state":
            return conv
        if component_name == "recurrent_state":
            return recurrent
        raise ValueError(f"unknown KDA component {component_name!r}")

    def get_state_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no KDA state") from exc

    def zero_new_blocks(self, new_page_ids: dict[str, list[int]]) -> None:
        if new_page_ids:
            self.arena.zero_blocks(new_page_ids)

    def get_kv_size_bytes(self):
        return self.arena.buffer.nbytes
