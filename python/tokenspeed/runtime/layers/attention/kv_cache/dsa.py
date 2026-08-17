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

from typing import ClassVar

import torch
from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

from tokenspeed.runtime.layers.attention.configs.dsa import dsa_index_k_row_bytes
from tokenspeed.runtime.layers.attention.kv_cache.mla import (
    MLATokenToKVPool,
    _get_tensor_size_bytes,
)

_INDEX_K_FP8_GROUP_SIZE = 128


class DSATokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        *args,
        index_head_dim: int,
        **kwargs,
    ):
        self.index_head_dim = int(index_head_dim)
        self.index_k_row_bytes = dsa_index_k_row_bytes(self.index_head_dim)
        super().__init__(*args, **kwargs)

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MLATokenToKVPool.layer_plane_bindings,
        "index_k": "index_k_buffer",
    }

    def get_kv_size_bytes(self):
        return super().get_kv_size_bytes() + _get_tensor_size_bytes(self.index_k_buffer)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self.index_k_buffer[layer_id]

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        if index_k.dtype != self.model_dtype:
            index_k = index_k.to(self.model_dtype)
        index_k = index_k.view(-1, self.index_head_dim)
        self._set_index_k_buffer(layer_id, loc, index_k)

    def _set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        buf = self.index_k_buffer[layer_id]
        index_k_fp8, index_k_scale = quantize_fp8_with_scale(
            index_k,
            granularity="token_group",
            group_size=_INDEX_K_FP8_GROUP_SIZE,
            scale_encoding="float32",
        )

        # Fused scatter; (page, slot_in_page) is derived from loc in-kernel.
        index_k_block_split_scatter(
            buf,
            index_k_fp8,
            index_k_scale,
            loc,
            page_size=self.arena.kv_page_size,
            head_dim=self.index_head_dim,
            group_size=_INDEX_K_FP8_GROUP_SIZE,
        )
