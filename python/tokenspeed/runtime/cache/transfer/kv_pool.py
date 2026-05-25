from __future__ import annotations

import torch

from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter
from tokenspeed.runtime.cache.transfer.types import CacheKind


class KVCachePool:
    kind = CacheKind.KV

    def __init__(
        self,
        device_pool,
        host_pool,
        io_backend: str,
        layer_num: int,
        draft_device_pool=None,
        draft_host_pool=None,
        draft_layer_num: int = 0,
    ):
        self.device_pool = device_pool
        self.host_pool = host_pool
        self.io_backend = io_backend
        self.layer_num = layer_num
        self.draft_device_pool = draft_device_pool
        self.draft_host_pool = draft_host_pool
        self.draft_layer_num = draft_layer_num
        self._counter = LayerDoneCounter(max(layer_num, draft_layer_num, 1))
        device_pool.register_layer_transfer_counter(self._counter)

    @property
    def device(self):
        return self.device_pool.device

    @property
    def host_layout(self) -> str:
        return self.host_pool.layout

    def page_size(self) -> int:
        return self.host_pool.page_size

    def num_layers(self) -> int:
        return max(self.layer_num, self.draft_layer_num)

    def supports_layerwise_loadback(self) -> bool:
        return True

    def get_layer_done_counter(self) -> LayerDoneCounter:
        return self._counter

    def local_layer_idx(self, global_layer_id: int) -> int:
        return global_layer_id

    def writeback(
        self,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        block_quota: int | None = None,
    ) -> None:
        self.host_pool.backup_from_device_all_layer(
            self.device_pool,
            dst_indices,
            src_indices,
            self.io_backend,
            block_quota=block_quota,
        )
        if self.draft_host_pool is not None:
            self.draft_host_pool.backup_from_device_all_layer(
                self.draft_device_pool,
                dst_indices,
                src_indices,
                self.io_backend,
                block_quota=block_quota,
            )

    def loadback(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor, layer_idx: int
    ) -> None:
        if layer_idx < self.layer_num:
            self.host_pool.load_to_device_per_layer(
                self.device_pool,
                src_indices,
                dst_indices,
                layer_idx,
                self.io_backend,
            )
        if self.draft_host_pool is not None and layer_idx < self.draft_layer_num:
            self.draft_host_pool.load_to_device_per_layer(
                self.draft_device_pool,
                src_indices,
                dst_indices,
                layer_idx,
                self.io_backend,
            )

    def alloc_host(self, n: int):
        return self.host_pool.alloc(n)

    def free_host(self, indices: torch.Tensor) -> None:
        self.host_pool.free(indices)

    def host_available(self) -> int:
        return self.host_pool.available_size()
