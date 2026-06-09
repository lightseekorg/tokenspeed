from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

PAGED_CACHE_KIND = "paged_cache"


class CacheKind(str, Enum):
    KV = "kv"
    MAMBA = "mamba"


class Location(str, Enum):
    DEVICE = "device"
    HOST = "host"
    STORAGE = "storage"


@dataclass(slots=True)
class TransferUnit:
    kind: CacheKind
    src_loc: Location
    dst_loc: Location
    src_indices: torch.Tensor
    dst_indices: torch.Tensor
    op_id: int
    is_retract: bool = False
    layerwise_cow_src_indices: torch.Tensor | None = None
    layerwise_cow_dst_indices: torch.Tensor | None = None

    @property
    def direction(self) -> tuple[Location, Location]:
        return (self.src_loc, self.dst_loc)


@dataclass(slots=True)
class TransferBatch:
    units: list[TransferUnit]
    op_ids: list[int]


@dataclass(slots=True)
class PagedCacheTransferUnit:
    op_id: int
    transfers: list[Any]
    is_retract: bool = False
