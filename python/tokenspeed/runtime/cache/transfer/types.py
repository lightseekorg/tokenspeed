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
