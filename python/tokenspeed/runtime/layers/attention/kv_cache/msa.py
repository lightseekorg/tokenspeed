# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Paged cache storage for MiniMax sparse attention."""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool


class MSATokenToKVPool(MHATokenToKVPool):
    """MHA K/V cache plus a key-only sparse-index side cache."""

    supports_hierarchical_kv_cache = False

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
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        index_head_dim: int,
        index_dtype: torch.dtype,
        indexed_layer_ids: frozenset[int],
        layer_types: tuple[str, ...] = (),
        sliding_window_tokens: int | tuple[int | None, ...] | None = None,
        max_scheduled_tokens: int = 0,
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
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
            layer_types=layer_types,
            sliding_window_tokens=sliding_window_tokens,
            max_scheduled_tokens=max_scheduled_tokens,
            pd_disaggregation_enabled=pd_disaggregation_enabled,
        )
        with self.memory_saver_adapter.region(
            tag="kv_cache",
            enable_cpu_backup=False,
        ):
            self.index_k_buffer = {
                layer_id: torch.zeros(
                    (self.size + self.page_size, self.index_head_dim),
                    dtype=self.index_dtype,
                    device=self.device,
                )
                for layer_id in sorted(self.indexed_layer_ids)
            }

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        if layer_id not in self.index_k_buffer:
            raise RuntimeError(f"Layer {layer_id} has no index-key cache.")
        return self.index_k_buffer[layer_id]

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        super().move_kv_cache(tgt_loc, src_loc)
        if tgt_loc.numel() == 0:
            return
        target = tgt_loc.view(-1).long()
        source = src_loc.view(-1).long()
        for cache in self.index_k_buffer.values():
            cache[target] = cache[source]

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        super().clear_kv_buffers()
        for cache in self.index_k_buffer.values():
            cache.zero_()

    def get_kv_size_bytes(self) -> tuple[int, int]:
        key_bytes, value_bytes = super().get_kv_size_bytes()
        index_bytes = sum(cache.nbytes for cache in self.index_k_buffer.values())
        return key_bytes + index_bytes, value_bytes

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "MiniMax sparse cache transfer requires index-key side-cache support."
        )
