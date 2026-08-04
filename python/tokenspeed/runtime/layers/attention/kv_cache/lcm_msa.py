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

"""MiniMax sparse-attention (MSA) cache on the shared LCM arena."""

from __future__ import annotations

import numpy as np
import torch

from tokenspeed.runtime.layers.attention.kv_cache.lcm_mha import LcmMHATokenToKVPool


class LcmMSATokenToKVPool(LcmMHATokenToKVPool):
    """MHA K/V history plus a per-sparse-layer index-K side cache, one arena.

    Extends the MHA LCM interface with the MiniMax sparse index-K cache. Each
    sparse layer's ``layer.{i}.index_k`` field lives in the same LCM arena as
    that layer's K/V (see :func:`msa_index_k_lcm_fields`), so it packs into the
    same parent block and is addressed by the same page ids. ``_create_lcm_buffers``
    binds those fields, reshaped to the flat ``[num_slots, index_head_dim]``
    layout the MSA sparse kernels read.
    """

    supports_hierarchical_kv_cache = False

    def __init__(
        self,
        *,
        index_head_dim: int,
        index_dtype: torch.dtype,
        indexed_layer_ids: frozenset[int],
        **kwargs,
    ) -> None:
        self.index_head_dim = int(index_head_dim)
        self.index_dtype = index_dtype
        self.indexed_layer_ids = frozenset(indexed_layer_ids)
        self.index_k_buffer: dict[int, torch.Tensor] = {}
        super().__init__(**kwargs)

    def _create_lcm_buffers(self) -> None:
        super()._create_lcm_buffers()
        assert self.lcm_pool is not None
        self.index_k_buffer = {}
        for layer_id in sorted(self.indexed_layer_ids):
            field = self.get_lcm_field(f"layer.{layer_id}.index_k", self.index_dtype)
            # field is [page_count, page_size, index_head_dim]; the MSA kernels
            # want a flat [num_slots, index_head_dim] buffer. Pages are kept
            # contiguous by LcmCachePool, so a plain reshape aliases the same
            # storage.
            page_elements = int(np.prod(field.shape[1:]))
            if field.stride(0) != page_elements:
                raise ValueError(
                    f"layer {layer_id} index-K pages have padding between pages"
                )
            self.index_k_buffer[layer_id] = field.reshape(-1, self.index_head_dim)

    def has_index_k_buffer(self) -> bool:
        return True

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        if layer_id not in self.index_k_buffer:
            raise RuntimeError(f"Layer {layer_id} has no index-key cache.")
        return self.index_k_buffer[layer_id]
