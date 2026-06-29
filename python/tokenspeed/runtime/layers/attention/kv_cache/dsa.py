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

import torch
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

from tokenspeed.runtime.layers.attention.configs.dsa import dsa_index_k_row_bytes
from tokenspeed.runtime.layers.attention.kv_cache.mla import (
    MLATokenToKVPool,
    _get_tensor_size_bytes,
)

_INDEX_K_FP8_GROUP_SIZE = 128
_INDEX_K_SCALE_BYTES = torch._utils._element_size(torch.float32)


class DSATokenToKVPool(MLATokenToKVPool):
    # KVStore currently transfers only the base MLA KV rows. DSA also needs
    # sparse/indexer cache rows to stay coherent with the reused KV pages.
    supports_hierarchical_kv_cache = False

    def __init__(
        self,
        *args,
        index_head_dim: int,
        **kwargs,
    ):
        self.index_head_dim = int(index_head_dim)
        self.index_k_row_bytes = dsa_index_k_row_bytes(
            self.index_head_dim,
        )
        self._index_k_available = self.index_k_row_bytes > 0
        super().__init__(*args, **kwargs)
        with self.memory_saver_adapter.region():
            self.index_k_buffer = [
                (
                    torch.zeros(
                        (
                            self.size + self.page_size,
                            self.index_k_row_bytes,
                        ),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    if self.index_k_row_bytes > 0
                    else None
                )
                for _ in range(self.layer_num)
            ]

    def _get_page_size_bytes(self):
        index_size_bytes = self.index_k_row_bytes
        return (
            super()._get_page_size_bytes()
            + self.page_size * self.layer_num * index_size_bytes
        )

    def get_kv_size_bytes(self):
        return super().get_kv_size_bytes() + _get_tensor_size_bytes(
            [buf for buf in self.index_k_buffer if buf is not None]
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        super().move_kv_cache(tgt_loc, src_loc)
        if tgt_loc.numel() == 0:
            return
        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for buf in self.index_k_buffer:
            if buf is not None:
                # Packed FP8 index-K is block-split per page, so a single
                # token's bytes are NOT a contiguous row; move the FP8 values
                # and FP32 scales through their block-split views instead.
                fp8_view, scale_view = self._index_k_block_views(buf)
                ps = self.page_size
                tgt_page = tgt_loc_flat // ps
                tgt_slot = tgt_loc_flat % ps
                src_page = src_loc_flat // ps
                src_slot = src_loc_flat % ps
                fp8_view[tgt_page, tgt_slot] = fp8_view[src_page, src_slot]
                scale_view[tgt_page, tgt_slot] = scale_view[src_page, src_slot]

    def has_index_k_buffer(self) -> bool:
        return self._index_k_available

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        buf = self.index_k_buffer[layer_id]
        if buf is None or not self._index_k_available:
            raise RuntimeError("GLM DSA FP8 index K cache is unavailable")
        return buf

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
        ps = self.page_size
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
        page = slots // self.page_size
        slot_in_page = slots % self.page_size
        k_fp8 = fp8_view[page, slot_in_page]
        k_scale = scale_view[page, slot_in_page]
        return k_fp8, k_scale

    def _set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        if not self._index_k_available:
            return
        buf = self.index_k_buffer[layer_id]
        if buf is None:
            raise RuntimeError("DSA FP8 index K cache is unavailable")
        index_k_fp8, index_k_scale = quantize_fp8_with_scale(
            index_k,
            granularity="token_group",
            group_size=_INDEX_K_FP8_GROUP_SIZE,
            scale_encoding="float32",
        )

        fp8_view, scale_view = self._index_k_block_views(buf)
        loc = loc.to(torch.long)
        page = loc // self.page_size
        slot_in_page = loc % self.page_size
        fp8_view[page, slot_in_page] = index_k_fp8.view(-1, self.index_head_dim)
        scale_view[page, slot_in_page] = index_k_scale.view(
            -1, self.index_head_dim // _INDEX_K_FP8_GROUP_SIZE
        )

    def get_contiguous_buf_infos(self):
        data_ptrs, data_lens, item_lens = super().get_contiguous_buf_infos()
        data_ptrs = list(data_ptrs)
        data_lens = list(data_lens)
        item_lens = list(item_lens)
        for buf in self.index_k_buffer:
            if buf is not None:
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes * self.page_size)
        return data_ptrs, data_lens, item_lens

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        offsets = super().get_layerwise_buf_info_offsets(start_idx)
        if self.quant_method == "per_token_head":
            base_count = 3 * self.layer_num
        else:
            base_count = self.layer_num
        return [
            layer_offsets
            + (
                [start_idx + base_count + layer_id]
                if self.index_k_row_bytes > 0
                else []
            )
            for layer_id, layer_offsets in enumerate(offsets)
        ]

    def get_cpu_copy(self, token_indices: list[int]) -> torch.Tensor:
        del token_indices
        raise NotImplementedError(
            "DSA KV cache offload is not implemented; sparse/indexer cache "
            "buffers require page-aware layout handling."
        )

    def load_cpu_copy(
        self, kv_cache_cpu: torch.Tensor, token_indices: list[int]
    ) -> None:
        del kv_cache_cpu, token_indices
        raise NotImplementedError(
            "DSA KV cache reload is not implemented; sparse/indexer cache "
            "buffers require page-aware layout handling."
        )
