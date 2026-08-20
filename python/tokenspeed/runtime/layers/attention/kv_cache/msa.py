# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Paged cache storage for MiniMax sparse attention."""

from __future__ import annotations

from typing import ClassVar

import torch

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool


class MSATokenToKVPool(MHATokenToKVPool):
    """MHA K/V cache plus a key-only sparse-index side cache."""

    def __init__(
        self,
        *,
        arena: CacheArena,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        rank: int,
        index_head_dim: int,
        index_dtype: torch.dtype,
        indexed_layer_ids: frozenset[int],
        layer_types: tuple[str, ...] = (),
        field_layer_offset: int = 0,
    ) -> None:
        self.index_head_dim = index_head_dim
        self.index_dtype = index_dtype
        self.indexed_layer_ids = frozenset(indexed_layer_ids)
        super().__init__(
            arena,
            dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            rank=rank,
            layer_types=layer_types,
            field_layer_offset=field_layer_offset,
        )

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MHATokenToKVPool.layer_plane_bindings,
        "index_k": "_index_k",
    }

    def _bind_layer_planes(self) -> None:
        super()._bind_layer_planes()
        # Only sparse layers plan an index plane, so the list is holey; the
        # kernel-facing surface is a dict keyed by the layers that have one.
        self.index_k_buffer = {
            layer_id: plane
            for layer_id, plane in enumerate(self._index_k)
            if plane is not None
        }

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        if layer_id not in self.index_k_buffer:
            raise RuntimeError(f"Layer {layer_id} has no index-key cache.")
        return self.index_k_buffer[layer_id]

    def get_kv_size_bytes(self) -> tuple[int, int]:
        key_bytes, value_bytes = super().get_kv_size_bytes()
        index_bytes = sum(cache.nbytes for cache in self.index_k_buffer.values())
        return key_bytes + index_bytes, value_bytes
