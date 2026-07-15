# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""MHA KV cache with a per-token key-only side cache."""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool


class IndexedMHATokenToKVPool(MHATokenToKVPool):
    """Standard paged K/V plus one matching-dtype index-key vector per token."""

    def __init__(
        self,
        *args,
        index_head_dim: int,
        index_dtype: torch.dtype,
        **kwargs,
    ) -> None:
        self.index_head_dim = int(index_head_dim)
        self.index_dtype = index_dtype
        self.index_store_dtype = (
            torch.uint8
            if index_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else index_dtype
        )
        self.index_k_buffer: list[torch.Tensor] = []
        super().__init__(*args, **kwargs)
        # Radix-cache host offload and PD transfer currently describe only the
        # main K/V tensors; exposing them without the side cache would be wrong.
        self.supports_hierarchical_kv_cache = False

    def _create_buffers(self) -> None:
        """Allocate main K/V and index K before the parent reports pool size."""

        super()._create_buffers()
        with self.memory_saver_adapter.region(
            tag="kv_cache",
            enable_cpu_backup=False,
        ):
            self.index_k_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.index_head_dim),
                    dtype=self.index_store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the key-only side cache for ``layer_id``."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        buffer = self.index_k_buffer[layer_id]
        if self.index_store_dtype != self.index_dtype:
            return buffer.view(self.index_dtype)
        return buffer

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        super().move_kv_cache(tgt_loc, src_loc)
        if tgt_loc.numel() == 0:
            return
        target = tgt_loc.view(-1).long()
        source = src_loc.view(-1).long()
        for cache in self.index_k_buffer:
            cache[target] = cache[source]

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        """Clear both the standard K/V tensors and the index-key cache."""
        super().clear_kv_buffers()
        for cache in self.index_k_buffer:
            cache.zero_()

    def get_kv_size_bytes(self) -> tuple[int, int]:
        key_bytes, value_bytes = super().get_kv_size_bytes()
        index_bytes = sum(cache.nbytes for cache in self.index_k_buffer)
        return key_bytes + index_bytes, value_bytes

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "Indexed MHA cache transfer requires index-key side-cache support."
        )


__all__ = ["IndexedMHATokenToKVPool"]
