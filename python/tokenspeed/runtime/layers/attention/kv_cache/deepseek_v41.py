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

"""Strided V4.1 FlatKV views over the scheduler-owned shared arena."""

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.paged_attention import PagedAttention


class DeepseekV41CachePool(CachePool):
    """Own no cache memory; expose only the fields the LCM recipe allocated.

    Field pages are not contiguous across owners or groups. In particular,
    flattening page and row with reshape can copy an entire cache. Kernels
    must preserve the page stride and interpret each page as all value rows
    followed by all scale rows. The field shape carries its byte budget, not
    separately addressable packed rows.
    """

    requires_page_zeroing = True
    layer_plane_bindings = {"swa": "swa_buffers"}

    def __init__(
        self,
        arena: CacheArena,
        *,
        layer_num: int,
        rank: int,
        field_layer_offset: int,
    ) -> None:
        super().__init__(
            arena=arena,
            dtype=torch.uint8,
            rank=rank,
            field_layer_offset=field_layer_offset,
        )
        self.layer_num = layer_num
        self._bind_layer_planes()

    def _field(self, layer: int, name: str) -> torch.Tensor:
        if not 0 <= layer < self.layer_num:
            raise ValueError(f"V4.1 layer {layer} is outside this cache pool")
        field_id = f"layer.{self._field_layer_id(layer)}.{name}"
        view = self.arena.field(field_id)
        if self.layerwise_load_tracker is not None:
            # A consumer waits on the source owner, never its own nonexistent
            # global field. Fence before either reading OR writing these bytes.
            self.layerwise_load_tracker.wait_for_layer(layer)
        return view

    def swa(self, layer: int) -> torch.Tensor:
        """Return layer's uint8 [pages, 64, 528] page-planar FP8 values/E8M0 scales."""
        return self._field(layer, "swa")

    def global_kv(self, owner: int) -> torch.Tensor:
        """Return owner's uint8 [pages, 64, 288] page-planar FP4 values/E4M3 scales.

        Pass the KV source layer, not a Reuse/Reindex consumer layer.
        """
        return self._field(owner, "global_kv")

    def index_k(self, owner: int) -> torch.Tensor:
        """Return owner's uint8 [pages, 64, 68] page-planar FP4 values/E8M0 scales.

        Main KV and index K share logical rows and page ids, not byte offsets.
        """
        return self._field(owner, "index_k")

    def compressor_tail(self, owner: int) -> torch.Tensor:
        """Return ratio-2 owner's FP32 [pages, 2, 2, 512] projected input.

        Axes are page, raw-token row, content/score (in that order), channel.
        Ratio-1 owners have no tail; requesting one fails instead of allocating.
        """
        return self._field(owner, "compressor_tail")

    @override
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa(layer_id)

    @override
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa(layer_id)

    @override
    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.swa(layer_id)
        return latent, latent

    @override
    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "V4.1 requires its quantized cache writer with swa(layer), not a generic K/V scatter"
        )
