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

"""Paged latent-KV and KDA-state cache, with Kimi-K3.1's DSA index-K planes."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from tokenspeed.runtime.layers.attention.kv_cache.base import (
    derive_state_groups_by_layer,
)
from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_LAYER_TYPES,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.layernorm import LayerNorm


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
    # State planes keep their planned shape: the KDA decode ABI reads them
    # as the plan lays them out.
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
        self._state_buffers_by_layer = {
            layer_id: (self._conv_state[layer_id], self._recurrent_state[layer_id])
            for layer_id, label in enumerate(self._layer_types)
            if label in STATE_LAYER_TYPES and self._conv_state[layer_id] is not None
        }

    @cached_property
    def state_group_by_layer(self) -> dict[int, str]:
        """View-local state layer id -> its state cache group id."""
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

    def zero_new_blocks(self, new_page_ids: Mapping[str, np.ndarray]) -> None:
        if new_page_ids:
            self.arena.zero_blocks(new_page_ids)

    def get_kv_size_bytes(self):
        return self.arena.buffer.nbytes


class HybridKDADSATokenToKVPool(HybridKDATokenToKVPool, DSATokenToKVPool):
    """Kimi-K3.1: the K3 latent-KV/KDA-state view plus one index-K plane per
    DSA layer.

    The index-K rows are companion fields of the MLA layer's full-attention
    pages (``KimiK3Recipe``), so they are allocated, prefix-matched,
    transferred and freed with the latent rows. The write, the packed-row
    gather and the top-k scorers address the plane in DSA kernel pages
    (``index_k_page_size``), which subdivide K3's 128-token cache blocks the
    same way the router subdivides the latent pages. KDA layers and draft
    MLA layers have no index-K plane.
    """

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **HybridKDATokenToKVPool.layer_plane_bindings,
        **DSATokenToKVPool.layer_plane_bindings,
    }

    def __init__(self, *, index_k_page_size: int, **kwargs):
        self._index_k_page_size = int(index_k_page_size)
        super().__init__(**kwargs)
        page = self._index_k_page_size
        if page <= 0 or self.arena.kv_page_size % page:
            raise ValueError(
                f"index-K page size {page} must divide the {self.arena.kv_page_size}"
                "-token cache page"
            )

    @property
    def index_k_page_size(self) -> int:
        return self._index_k_page_size

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        buffer = super().get_index_k_buffer(layer_id)
        if buffer is None:
            raise ValueError(f"layer {layer_id} has no DSA index-K plane")
        return buffer

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        *,
        write_mask: torch.Tensor | None,
        key_norm: LayerNorm | None = None,
    ) -> None:
        if self.index_k_buffer[layer_id] is None:
            raise ValueError(f"layer {layer_id} has no DSA index-K plane")
        super().set_index_k_buffer(
            layer_id, loc, index_k, write_mask=write_mask, key_norm=key_norm
        )
