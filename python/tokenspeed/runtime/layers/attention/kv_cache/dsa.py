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
_INDEX_K_SCALE_BYTES = torch._utils._element_size(torch.float32)


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

    def has_index_k_buffer(self) -> bool:
        return True

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

    def _index_k_block_views(
        self, buf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-page block-split views into a packed FP8 index-K buffer.

        DeepGEMM's ``fp8_paged_mqa_logits`` expects each page of ``page_size``
        tokens to be laid out as ``[page_size * head_dim FP8 values]`` followed
        by ``[page_size * num_groups FP32 scales]`` (block-split), NOT a
        per-token ``[fp8 | scale]`` interleave. The two ``as_strided`` views
        below alias the same storage as ``buf`` so writes land in place.

        Args:
            buf: Packed FP8 index-K buffer of shape ``[num_slots, row_bytes]``
                and dtype ``uint8``, where ``row_bytes == head_dim +
                scale_bytes`` and ``num_slots`` is a multiple of
                ``page_size``.

        Returns:
            ``(fp8_view, scale_view)`` where ``fp8_view`` has shape
            ``[num_pages, page_size, head_dim]`` (FP8 e4m3) and ``scale_view``
            has shape ``[num_pages, page_size, num_groups]`` (float32), both
            indexed as ``view[page, slot_in_page]``.
        """
        # DSA requires kv_page_size == DSA_SPARSE_PAGE_SIZE (64), the only
        # implemented sparse layout.
        ps = self.arena.kv_page_size
        hd = self.index_head_dim
        ng = hd // _INDEX_K_FP8_GROUP_SIZE
        row = hd + ng * _INDEX_K_SCALE_BYTES
        num_pages = buf.shape[0] // ps
        page_bytes = ps * row
        flat = buf.reshape(-1)
        fp8_view = torch.as_strided(
            flat.view(torch.float8_e4m3fn),
            (num_pages, ps, hd),
            (page_bytes, hd, 1),
        )
        scale_view = torch.as_strided(
            flat.view(torch.float32),
            (num_pages, ps, ng),
            (page_bytes // 4, ng, 1),
            (ps * hd) // 4,
        )
        return fp8_view, scale_view

    def gather_index_k(
        self, layer_id: int, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather per-token FP8 index-K values and scales from the cache.

        The packed FP8 index-K buffer is stored block-split per page (see
        :meth:`_index_k_block_views`), so the non-paged prefill
        scoring kernel (``fp8_mqa_logits``), which consumes contiguous
        ``(k_fp8, k_scale)`` tensors, must gather token rows through the
        block-split views rather than indexing raw rows.

        Args:
            layer_id: Layer whose index-K cache to read.
            slots: 1D int tensor of global token slot indices to gather.

        Returns:
            ``(k_fp8, k_scale)`` where ``k_fp8`` has shape
            ``[num_slots, head_dim]`` (FP8 e4m3) and ``k_scale`` has shape
            ``[num_slots, num_groups]`` (float32).
        """
        buf = self.get_index_k_buffer(layer_id)
        fp8_view, scale_view = self._index_k_block_views(buf)
        slots = slots.to(torch.long)
        page = slots // self.arena.kv_page_size
        slot_in_page = slots % self.arena.kv_page_size
        k_fp8 = fp8_view[page, slot_in_page]
        k_scale = scale_view[page, slot_in_page]
        return k_fp8, k_scale

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
