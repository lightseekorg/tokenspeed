from __future__ import annotations

import torch

from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter
from tokenspeed.runtime.cache.mamba_cache_host import MambaPoolHost
from tokenspeed.runtime.cache.transfer.types import CacheKind
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    SimpleMambaPool,
)


class MambaCachePool:
    kind = CacheKind.MAMBA

    def __init__(
        self,
        device_pool: SimpleMambaPool,
        host_pool: MambaPoolHost,
        io_backend: str,
    ):
        self.device_pool = device_pool
        self.host_pool = host_pool
        self.io_backend = io_backend
        self._counter = LayerDoneCounter(self.num_layers())
        device_pool.register_layer_transfer_counter(self._counter)

    @property
    def device(self):
        return self.device_pool.device

    @property
    def host_layout(self) -> str:
        return self.host_pool.layout

    def page_size(self) -> int:
        return 1

    def num_layers(self) -> int:
        return int(self.device_pool.conv_state.shape[0])

    def supports_layerwise_loadback(self) -> bool:
        return True

    def get_layer_done_counter(self) -> LayerDoneCounter:
        return self._counter

    def local_layer_idx(self, global_layer_id: int) -> int:
        return self.device_pool.mamba_map[global_layer_id]

    def writeback(
        self,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        block_quota: int | None = None,
    ) -> None:
        self.host_pool.backup_from_device_all_layer(
            self.device_pool,
            host_indices=dst_indices,
            device_indices=src_indices,
            io_backend=self.io_backend,
            block_quota=block_quota,
        )

    def loadback(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor, layer_idx: int
    ) -> None:
        self.host_pool.load_to_device_per_layer(
            self.device_pool,
            host_indices=src_indices,
            device_indices=dst_indices,
            layer_idx=layer_idx,
            io_backend=self.io_backend,
        )

    def copy_layer(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor, layer_idx: int
    ) -> None:
        if src_indices.numel() == 0:
            return
        src_indices = src_indices.to(
            device=self.device, dtype=torch.int64, non_blocking=True
        )
        dst_indices = dst_indices.to(
            device=self.device, dtype=torch.int64, non_blocking=True
        )
        for cache in self.device_pool.mamba_cache:
            layer = cache[layer_idx]
            layer.index_copy_(0, dst_indices, layer.index_select(0, src_indices))

    def alloc_host(self, n: int):
        return self.host_pool.alloc(n)

    def free_host(self, indices: torch.Tensor) -> None:
        self.host_pool.free(indices)

    def host_available(self) -> int:
        return self.host_pool.available_size()
