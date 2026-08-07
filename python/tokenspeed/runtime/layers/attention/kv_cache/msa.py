# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Paged cache storage for MiniMax sparse attention."""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan


class MSATokenToKVPool(MHATokenToKVPool):
    """MHA K/V cache plus a key-only sparse-index side cache."""

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        page_size: int,
        rank: int,
        index_head_dim: int,
        index_dtype: torch.dtype,
        indexed_layer_ids: frozenset[int],
        memory_plan: CacheMemoryPlan,
        paged_cache_group_specs: tuple = (),
        token_capacity: int | None = None,
        layer_types: tuple[str, ...] = (),
        layer_group_ids: tuple[str, ...] = (),
        pd_disaggregation_enabled: bool = False,
    ) -> None:
        self.index_head_dim = index_head_dim
        self.index_dtype = index_dtype
        self.indexed_layer_ids = frozenset(indexed_layer_ids)
        self.index_k_buffer: dict[int, torch.Tensor] = {}
        super().__init__(
            size=size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            page_size=page_size,
            rank=rank,
            layer_types=layer_types,
            layer_group_ids=layer_group_ids,
            pd_disaggregation_enabled=pd_disaggregation_enabled,
            memory_plan=memory_plan,
            paged_cache_group_specs=paged_cache_group_specs,
            token_capacity=token_capacity,
        )
        with self.memory_saver_adapter.region(
            tag="kv_cache",
            enable_cpu_backup=False,
        ):
            self.index_k_buffer = {
                layer_id: self.field(
                    f"layer.{layer_id}.index_k", self.index_dtype
                ).view(-1, self.index_head_dim)
                for layer_id in sorted(self.indexed_layer_ids)
            }

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        if layer_id not in self.index_k_buffer:
            raise RuntimeError(f"Layer {layer_id} has no index-key cache.")
        return self.index_k_buffer[layer_id]

    def get_kv_size_bytes(self) -> tuple[int, int]:
        key_bytes, value_bytes = super().get_kv_size_bytes()
        index_bytes = sum(cache.nbytes for cache in self.index_k_buffer.values())
        return key_bytes + index_bytes, value_bytes

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "MiniMax sparse cache transfer requires index-key side-cache support."
        )
