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

import logging

import torch
from tokenspeed_kernel.ops.attention.triton.dsa import (
    GLM_DSA_SPARSE_DECODE_ROW_BYTES,
    glm_dsa_pack_sparse_decode_kv,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

from tokenspeed.runtime.layers.attention.configs.dsa import (
    glm_dsa_sparse_decode_row_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.mla import (
    MLATokenToKVPool,
    _get_tensor_size_bytes,
)

logger = logging.getLogger(__name__)


class DSATokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        *args,
        index_head_dim: int,
        **kwargs,
    ):
        self.index_head_dim = int(index_head_dim)
        kv_lora_rank = kwargs.get("kv_lora_rank", args[4] if len(args) > 4 else None)
        qk_rope_head_dim = kwargs.get(
            "qk_rope_head_dim",
            args[5] if len(args) > 5 else None,
        )
        self.sparse_decode_kv_row_bytes = glm_dsa_sparse_decode_row_bytes(
            kv_lora_rank,
            qk_rope_head_dim,
        )
        if self.index_head_dim % 128 == 0:
            self.index_k_with_scale_row_bytes = (
                self.index_head_dim + self.index_head_dim // 128 * 4
            )
        else:
            self.index_k_with_scale_row_bytes = 0
        self._index_k_with_scale_available = self.index_k_with_scale_row_bytes > 0
        self._index_k_with_scale_warning_emitted = False
        super().__init__(*args, **kwargs)
        if self.sparse_decode_kv_row_bytes != GLM_DSA_SPARSE_DECODE_ROW_BYTES:
            raise ValueError(
                "GLM DSA sparse decode FlashMLA layout requires "
                f"kv_lora_rank=512 and qk_rope_head_dim=64, got "
                f"kv_lora_rank={self.kv_lora_rank}, "
                f"qk_rope_head_dim={self.qk_rope_head_dim}"
            )
        with self.memory_saver_adapter.region():
            self.sparse_decode_kv_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.sparse_decode_kv_row_bytes),
                    dtype=torch.uint8,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.index_k_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.index_head_dim),
                    dtype=self.model_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.index_k_with_scale_buffer = [
                (
                    torch.zeros(
                        (
                            self.size + self.page_size,
                            self.index_k_with_scale_row_bytes,
                        ),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    if self.index_k_with_scale_row_bytes > 0
                    else None
                )
                for _ in range(self.layer_num)
            ]

    def _get_page_size_bytes(self):
        sparse_decode_size_bytes = self.sparse_decode_kv_row_bytes
        index_size_bytes = self.index_head_dim * torch._utils._element_size(
            self.model_dtype
        )
        index_with_scale_size_bytes = self.index_k_with_scale_row_bytes
        return (
            super()._get_page_size_bytes()
            + self.page_size * self.layer_num * sparse_decode_size_bytes
            + self.page_size * self.layer_num * index_size_bytes
            + self.page_size * self.layer_num * index_with_scale_size_bytes
        )

    def get_kv_size_bytes(self):
        return (
            super().get_kv_size_bytes()
            + _get_tensor_size_bytes(self.sparse_decode_kv_buffer)
            + _get_tensor_size_bytes(self.index_k_buffer)
            + _get_tensor_size_bytes(
                [buf for buf in self.index_k_with_scale_buffer if buf is not None]
            )
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        super().move_kv_cache(tgt_loc, src_loc)
        if tgt_loc.numel() == 0:
            return
        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for buf in self.sparse_decode_kv_buffer:
            buf[tgt_loc_flat] = buf[src_loc_flat]
        for buf in self.index_k_buffer:
            buf[tgt_loc_flat] = buf[src_loc_flat]
        for buf in self.index_k_with_scale_buffer:
            if buf is not None:
                # Packed FP8 index-K is block-split per page, so a single
                # token's bytes are NOT a contiguous row; move the FP8 values
                # and FP32 scales through their block-split views instead.
                fp8_view, scale_view = self._index_k_with_scale_block_views(buf)
                ps = self.page_size
                tgt_page = tgt_loc_flat // ps
                tgt_slot = tgt_loc_flat % ps
                src_page = src_loc_flat // ps
                src_slot = src_loc_flat % ps
                fp8_view[tgt_page, tgt_slot] = fp8_view[src_page, src_slot]
                scale_view[tgt_page, tgt_slot] = scale_view[src_page, src_slot]

    def set_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        if self.quant_method != "per_token_head" and (
            cache_k_nope.dtype in fp8_dtypes or cache_k_rope.dtype in fp8_dtypes
        ):
            raise RuntimeError(
                "GLM DSA sparse decode cache requires BF16 MLA writes; use "
                "--kv-cache-dtype auto or bfloat16 for GLM DSA."
            )
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
        if self.quant_method == "per_token_head":
            return
        if cache_k_nope.dtype != torch.bfloat16:
            cache_k_nope = cache_k_nope.to(torch.bfloat16)
            cache_k_rope = cache_k_rope.to(torch.bfloat16)
        glm_dsa_pack_sparse_decode_kv(
            out=self.sparse_decode_kv_buffer[layer.layer_id],
            loc=loc,
            cache_k_nope=cache_k_nope,
            cache_k_rope=cache_k_rope,
        )

    def get_sparse_decode_kv_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.sparse_decode_kv_buffer[layer_id]

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.index_k_buffer[layer_id]

    def has_index_k_with_scale_buffer(self) -> bool:
        return self._index_k_with_scale_available

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        buf = self.index_k_with_scale_buffer[layer_id]
        if buf is None or not self._index_k_with_scale_available:
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
        self.index_k_buffer[layer_id][loc] = index_k
        self._set_index_k_with_scale_buffer(layer_id, loc, index_k)

    def _index_k_with_scale_block_views(
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
                num_groups * 4`` and ``num_slots`` is a multiple of
                ``page_size``.

        Returns:
            ``(fp8_view, scale_view)`` where ``fp8_view`` has shape
            ``[num_pages, page_size, head_dim]`` (FP8 e4m3) and ``scale_view``
            has shape ``[num_pages, page_size, num_groups]`` (float32), both
            indexed as ``view[page, slot_in_page]``.
        """
        ps = self.page_size
        hd = self.index_head_dim
        ng = hd // 128
        row = hd + ng * 4
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

    def gather_index_k_with_scale(
        self, layer_id: int, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather per-token FP8 index-K values and scales from the cache.

        The packed FP8 index-K buffer is stored block-split per page (see
        :meth:`_index_k_with_scale_block_views`), so the non-paged prefill
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
        buf = self.get_index_k_with_scale_buffer(layer_id)
        fp8_view, scale_view = self._index_k_with_scale_block_views(buf)
        slots = slots.to(torch.long)
        page = slots // self.page_size
        slot_in_page = slots % self.page_size
        k_fp8 = fp8_view[page, slot_in_page]
        k_scale = scale_view[page, slot_in_page]
        return k_fp8, k_scale

    def _set_index_k_with_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        if not self._index_k_with_scale_available:
            return
        buf = self.index_k_with_scale_buffer[layer_id]
        if buf is None:
            self._index_k_with_scale_available = False
            return
        try:
            index_k_fp8, index_k_scale = quantize_fp8_with_scale(
                index_k,
                granularity="token_group",
                group_size=128,
                scale_encoding="float32",
            )
        except Exception as exc:
            self._index_k_with_scale_available = False
            if not self._index_k_with_scale_warning_emitted:
                logger.warning(
                    "GLM DSA FP8 index K cache is disabled; falling back to "
                    "BF16 indexer top-k. Reason: %s",
                    exc,
                )
                self._index_k_with_scale_warning_emitted = True
            return

        fp8_view, scale_view = self._index_k_with_scale_block_views(buf)
        loc = loc.to(torch.long)
        page = loc // self.page_size
        slot_in_page = loc % self.page_size
        fp8_view[page, slot_in_page] = index_k_fp8.view(-1, self.index_head_dim)
        scale_view[page, slot_in_page] = index_k_scale.view(
            -1, self.index_head_dim // 128
        )

    def get_contiguous_buf_infos(self):
        data_ptrs, data_lens, item_lens = super().get_contiguous_buf_infos()
        data_ptrs = list(data_ptrs)
        data_lens = list(data_lens)
        item_lens = list(item_lens)
        for buf in self.sparse_decode_kv_buffer:
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes * self.page_size)
        for buf in self.index_k_buffer:
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes * self.page_size)
        for buf in self.index_k_with_scale_buffer:
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
            + [
                start_idx + base_count + layer_id,
                start_idx + base_count + self.layer_num + layer_id,
                *(
                    [start_idx + base_count + 2 * self.layer_num + layer_id]
                    if self.index_k_with_scale_row_bytes > 0
                    else []
                ),
            ]
            for layer_id, layer_offsets in enumerate(offsets)
        ]

    def get_cpu_copy(self, token_indices: list[int]) -> torch.Tensor:
        kv_cache_cpu = super().get_cpu_copy(token_indices)
        torch.cuda.synchronize()
        for layer_id in range(self.layer_num):
            for i in range(0, len(token_indices), self.offload_chunk_page_num):
                chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                kv_cache_cpu[layer_id][i // self.offload_chunk_page_num].append(
                    self.sparse_decode_kv_buffer[layer_id][chunk_indices].to(
                        "cpu", non_blocking=True
                    )
                )
                kv_cache_cpu[layer_id][i // self.offload_chunk_page_num].append(
                    self.index_k_buffer[layer_id][chunk_indices].to(
                        "cpu", non_blocking=True
                    )
                )
                buf = self.index_k_with_scale_buffer[layer_id]
                if buf is not None:
                    kv_cache_cpu[layer_id][i // self.offload_chunk_page_num].append(
                        buf[chunk_indices].to("cpu", non_blocking=True)
                    )
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu: torch.Tensor, token_indices: list[int]
    ) -> None:
        super().load_cpu_copy(kv_cache_cpu, token_indices)
        torch.cuda.synchronize()
        sparse_offset = 3 if self.quant_method == "per_token_head" else 1
        index_offset = sparse_offset + 1
        index_with_scale_offset = index_offset + 1
        for layer_id in range(self.layer_num):
            for i in range(0, len(token_indices), self.offload_chunk_page_num):
                chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                chunk = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num]
                sparse_decode_cpu = kv_cache_cpu[layer_id][
                    i // self.offload_chunk_page_num
                ][sparse_offset]
                assert sparse_decode_cpu.shape[0] == len(chunk_indices)
                self.sparse_decode_kv_buffer[layer_id][chunk_indices] = (
                    sparse_decode_cpu.to(
                        self.sparse_decode_kv_buffer[layer_id].device,
                        non_blocking=True,
                    )
                )
                index_k_cpu = chunk[index_offset]
                assert index_k_cpu.shape[0] == len(chunk_indices)
                self.index_k_buffer[layer_id][chunk_indices] = index_k_cpu.to(
                    self.index_k_buffer[layer_id].device,
                    non_blocking=True,
                )
                buf = self.index_k_with_scale_buffer[layer_id]
                if buf is not None and len(chunk) > index_with_scale_offset:
                    index_k_with_scale_cpu = chunk[index_with_scale_offset]
                    assert index_k_with_scale_cpu.shape[0] == len(chunk_indices)
                    buf[chunk_indices] = index_k_with_scale_cpu.to(
                        buf.device,
                        non_blocking=True,
                    )
        torch.cuda.synchronize()
