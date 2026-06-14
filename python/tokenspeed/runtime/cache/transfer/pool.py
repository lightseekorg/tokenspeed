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

from typing import Optional, Protocol

import torch

from tokenspeed.runtime.cache.transfer.types import CacheKind


class CachePool(Protocol):
    kind: CacheKind  # cache type (KV / Mamba)
    is_draft: bool  # target vs draft model role
    pool_id: str  # executor key, e.g. "kv", "kv.draft", "mamba"
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
