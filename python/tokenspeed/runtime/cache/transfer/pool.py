from __future__ import annotations

from typing import Optional, Protocol

import torch

from tokenspeed.runtime.cache.transfer.types import CacheKind


class CachePool(Protocol):
    kind: CacheKind
    device: torch.device | str
    host_layout: str

    def page_size(self) -> int: ...

    def num_layers(self) -> int: ...

    def supports_layerwise_loadback(self) -> bool: ...

    def writeback(
        self,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        block_quota: int | None = None,
    ) -> None: ...

    def loadback(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor, layer_idx: int
    ) -> None: ...

    def copy_layer(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor, layer_idx: int
    ) -> None: ...

    def get_layer_done_counter(self): ...

    def local_layer_idx(self, global_layer_id: int) -> int: ...

    def alloc_host(self, n: int) -> Optional[torch.Tensor]: ...

    def free_host(self, indices: torch.Tensor) -> None: ...

    def host_available(self) -> int: ...
