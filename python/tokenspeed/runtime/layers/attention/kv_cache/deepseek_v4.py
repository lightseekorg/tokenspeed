# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
    DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    deepseek_v4_compressed_slot_mapping,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class DeepseekV4CacheLayout:
    layer_ratio: tuple[int, ...]
    head_dim: int
    page_size: int
    use_fp4_indexer_cache: bool
    index_head_dim: int = 128

    @property
    def swa_row_bytes(self) -> int:
        return DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM

    def swa_block_bytes(self, page_size: int | None = None) -> int:
        if page_size is None:
            page_size = self.page_size
        block_bytes = page_size * self.swa_row_bytes
        alignment = DEEPSEEK_V4_SWA_TOKEN_STRIDE
        return ((block_bytes + alignment - 1) // alignment) * alignment

    def swa_cell_bytes(self) -> int:
        block_bytes = self.swa_block_bytes()
        return (block_bytes + self.page_size - 1) // self.page_size

    def storage_block_size(self, compress_ratio: int) -> int:
        if compress_ratio > 1 and self.page_size == 256:
            return max(1, self.page_size // compress_ratio)
        return self.page_size

    def compressed_cell_bytes(self, compress_ratio: int) -> int:
        block_bytes = self.swa_block_bytes(self.storage_block_size(compress_ratio))
        return (block_bytes + self.page_size - 1) // self.page_size

    @property
    def indexer_row_bytes(self) -> int:
        if self.use_fp4_indexer_cache:
            return (
                DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
                + DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
            )
        return self.index_head_dim + (self.index_head_dim // 128) * 4

    def state_width(self, layer_id: int, *, indexer: bool = False) -> int:
        if indexer:
            return self.index_head_dim * 2
        return self.head_dim * (2 if self.layer_ratio[layer_id] == 4 else 1)

    def cache_cell_size(self, layer_num: int | None = None) -> int:
        """Return bytes per token for the current V4 cache allocation layout."""
        if layer_num is None:
            layer_num = len(self.layer_ratio)
        if layer_num > len(self.layer_ratio):
            raise ValueError(
                "DeepSeek V4 cache layout has fewer layer ratios "
                f"({len(self.layer_ratio)}) than requested layers ({layer_num})"
            )

        fp32_size = torch._utils._element_size(torch.float32)
        cell_size = 0
        for layer_id in range(layer_num):
            ratio = self.layer_ratio[layer_id]
            cell_size += self.swa_cell_bytes()
            if ratio > 1:
                cell_size += self.compressed_cell_bytes(ratio)
                cell_size += self.state_width(layer_id) * 2 * fp32_size
            if ratio == 4:
                indexer_block_bytes = (
                    self.storage_block_size(ratio) * self.indexer_row_bytes
                )
                cell_size += (
                    indexer_block_bytes + self.page_size - 1
                ) // self.page_size
                cell_size += self.state_width(layer_id, indexer=True) * 2 * fp32_size
        return cell_size


@dataclass
class DeepseekV4ForwardMetadata:
    page_size: int
    req_pool_indices: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req_indices: torch.Tensor
    forward_mode: object = None
    decode_swa_indices: torch.Tensor | None = None
    decode_swa_lens: torch.Tensor | None = None
    decode_swa_window_size: int = 0
    decode_swa_block_size: int = 0
    decode_compressed_slot_mappings: dict[tuple[int, int], torch.Tensor] = field(
        default_factory=dict
    )
    decode_indexer_schedule_metadata: dict[tuple[int, int, int], torch.Tensor] = field(
        default_factory=dict
    )

    def _use_decode_compressed_slot_cache(self, positions: torch.Tensor) -> bool:
        return (
            self.forward_mode is not None
            and self.forward_mode.is_decode()
            and positions.is_cuda
            and self.block_table.is_cuda
        )

    def _update_decode_compressed_slot_mapping(
        self,
        compress_ratio: int,
        kv_cache_block_size: int,
    ) -> torch.Tensor:
        num_tokens = self.token_to_req_indices.shape[0]
        key = (compress_ratio, kv_cache_block_size)
        out = self.decode_compressed_slot_mappings.get(key)
        if (
            out is None
            or out.shape[0] < num_tokens
            or out.device != self.seq_lens.device
        ):
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 compressed slot metadata must be allocated before "
                    "CUDA graph capture"
                )
            with torch.inference_mode(False):
                out = torch.empty(
                    num_tokens, dtype=torch.int64, device=self.seq_lens.device
                )
            self.decode_compressed_slot_mappings[key] = out

        return deepseek_v4_compressed_slot_mapping(
            num_tokens=num_tokens,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            block_table=self.block_table,
            block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            out=out,
        )

    def refresh_decode_compressed_slot_mappings(self) -> None:
        if self.forward_mode is None or not self.forward_mode.is_decode():
            return
        for compress_ratio, kv_cache_block_size in list(
            self.decode_compressed_slot_mappings
        ):
            self._update_decode_compressed_slot_mapping(
                compress_ratio,
                kv_cache_block_size,
            )

    def compressed_slot_mapping(
        self,
        positions: torch.Tensor,
        compress_ratio: int,
        kv_cache_block_size: int | None = None,
    ) -> torch.Tensor:
        if kv_cache_block_size is None:
            kv_cache_block_size = self.page_size
        if self._use_decode_compressed_slot_cache(positions):
            cached = self.decode_compressed_slot_mappings.get(
                (compress_ratio, kv_cache_block_size)
            )
            if (
                cached is not None
                and cached.shape[0] >= positions.numel()
                and cached.device == self.seq_lens.device
            ):
                return cached[: positions.numel()]
            mapping = self._update_decode_compressed_slot_mapping(
                compress_ratio,
                kv_cache_block_size,
            )
            return mapping[: positions.numel()]
        compressed_pos = torch.div(
            positions.to(torch.int64), compress_ratio, rounding_mode="floor"
        )
        page_indices = torch.div(
            compressed_pos, kv_cache_block_size, rounding_mode="floor"
        )
        offsets = compressed_pos % kv_cache_block_size
        page_ids = self.block_table[
            self.token_to_req_indices[: positions.numel()].long(),
            page_indices.long(),
        ]
        return (page_ids.to(torch.int64) * kv_cache_block_size + offsets).to(
            torch.int64
        )


def deepseek_v4_cache_layout_from_config(
    hf_config,
    page_size: int,
    use_fp4_indexer_cache: bool,
) -> DeepseekV4CacheLayout:
    return DeepseekV4CacheLayout(
        layer_ratio=tuple(max(1, int(x)) for x in hf_config.compress_ratios),
        head_dim=int(hf_config.head_dim),
        page_size=page_size,
        use_fp4_indexer_cache=use_fp4_indexer_cache,
        index_head_dim=int(getattr(hf_config, "index_head_dim", 128)),
    )


class DeepseekV4TokenToKVPool(BaseTokenToKVPool):
    """DeepSeek V4 fp8_ds_mla cache pool.

    TokenSpeed keeps the SWA, compressed, compressor-state, and CSA indexer
    caches in one V4-only pool so ordinary MLA models keep their existing cache
    contract untouched. Compressed caches currently reuse the request page table;
    this is correctness-first and leaves ratio-specific allocation for the
    optimized follow-up.
    """

    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        layout: DeepseekV4CacheLayout,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
    ) -> None:
        if size <= 0:
            raise ValueError(f"DeepSeek V4 KV pool size must be positive, got {size}")
        super().__init__(
            size=size,
            dtype=torch.uint8,
            device=device,
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
        )
        del enable_memory_saver
        self.model_dtype = model_dtype
        self.layout = layout
        self.layer_num = layer_num
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        self.num_pages = (size + page_size - 1) // page_size + 1
        self.swa_block_size = page_size
        self.state_block_size = page_size
        self.swa_block_bytes = layout.swa_block_bytes(page_size)
        self.compressed_block_sizes = tuple(
            layout.storage_block_size(ratio) if ratio > 1 else page_size
            for ratio in layout.layer_ratio
        )
        self.compressed_block_size = (
            self.compressed_block_sizes[0] if self.compressed_block_sizes else page_size
        )

        self.swa_kv_buffer = [
            torch.zeros(
                (self.num_pages, self.swa_block_bytes),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(layer_num)
        ]
        self.compressed_kv_buffer: list[torch.Tensor | None] = []
        self.compressor_state_buffer: list[torch.Tensor | None] = []
        self.indexer_kv_buffer: list[torch.Tensor | None] = []
        self.indexer_state_buffer: list[torch.Tensor | None] = []
        for layer_id, ratio in enumerate(layout.layer_ratio):
            has_compressed = ratio > 1
            has_indexer = ratio == 4
            compressed_block_size = self.compressed_block_sizes[layer_id]
            self.compressed_kv_buffer.append(
                torch.zeros(
                    (self.num_pages, layout.swa_block_bytes(compressed_block_size)),
                    dtype=torch.uint8,
                    device=device,
                )
                if has_compressed
                else None
            )
            self.compressor_state_buffer.append(
                torch.empty(
                    (
                        self.num_pages,
                        page_size,
                        layout.state_width(layer_id) * 2,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                if has_compressed
                else None
            )
            self.indexer_kv_buffer.append(
                torch.zeros(
                    (self.num_pages, compressed_block_size * layout.indexer_row_bytes),
                    dtype=torch.uint8,
                    device=device,
                )
                if has_indexer
                else None
            )
            self.indexer_state_buffer.append(
                torch.empty(
                    (
                        self.num_pages,
                        page_size,
                        layout.state_width(layer_id, indexer=True) * 2,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                if has_indexer
                else None
            )

        logger.info(
            "Initialized DeepSeek V4 KV pool: %d pages, %d layers, fp4 indexer=%s, compressed block sizes=%s",
            self.num_pages,
            layer_num,
            layout.use_fp4_indexer_cache,
            self.compressed_block_sizes,
        )

    def _require(
        self, buffers: list[torch.Tensor | None], layer_id: int, name: str
    ) -> torch.Tensor:
        buf = buffers[layer_id]
        if buf is None:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no {name} cache")
        return buf

    def get_swa_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_buffer[layer_id]

    def get_compressed_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.compressed_kv_buffer, layer_id, "compressed KV")

    def get_compressed_block_size(self, layer_id: int) -> int:
        return self.compressed_block_sizes[layer_id]

    def get_indexer_block_size(self, layer_id: int) -> int:
        return self.compressed_block_sizes[layer_id]

    def get_compressor_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.compressor_state_buffer, layer_id, "compressor state")

    def get_indexer_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_kv_buffer, layer_id, "indexer KV")

    def get_indexer_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_state_buffer, layer_id, "indexer state")

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_swa_kv_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_swa_kv_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        buf = self.get_swa_kv_buffer(layer_id)
        return buf, buf

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DeepSeek V4 writes KV cache through V4 attention helpers"
        )

    def _move_fp8_ds_mla_rows(
        self,
        buf: torch.Tensor,
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        block_size: int,
    ) -> None:
        if tgt_loc.numel() == 0:
            return
        flat = buf.reshape(-1)
        tgt = tgt_loc.to(torch.int64)
        src = src_loc.to(torch.int64)
        tgt_page = torch.div(tgt, block_size, rounding_mode="floor")
        src_page = torch.div(src, block_size, rounding_mode="floor")
        tgt_pos = tgt % block_size
        src_pos = src % block_size
        block_stride = buf.stride(0)

        value_offsets = torch.arange(
            DEEPSEEK_V4_SWA_TOKEN_STRIDE,
            dtype=torch.int64,
            device=buf.device,
        )
        tgt_value = (
            tgt_page[:, None] * block_stride
            + tgt_pos[:, None] * DEEPSEEK_V4_SWA_TOKEN_STRIDE
            + value_offsets[None, :]
        )
        src_value = (
            src_page[:, None] * block_stride
            + src_pos[:, None] * DEEPSEEK_V4_SWA_TOKEN_STRIDE
            + value_offsets[None, :]
        )
        value_rows = flat[src_value].clone()
        flat[tgt_value] = value_rows

        scale_offsets = torch.arange(
            DEEPSEEK_V4_SWA_SCALE_DIM,
            dtype=torch.int64,
            device=buf.device,
        )
        scale_base = block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        tgt_scale = (
            tgt_page[:, None] * block_stride
            + scale_base
            + tgt_pos[:, None] * DEEPSEEK_V4_SWA_SCALE_DIM
            + scale_offsets[None, :]
        )
        src_scale = (
            src_page[:, None] * block_stride
            + scale_base
            + src_pos[:, None] * DEEPSEEK_V4_SWA_SCALE_DIM
            + scale_offsets[None, :]
        )
        scale_rows = flat[src_scale].clone()
        flat[tgt_scale] = scale_rows

    def _move_rows(
        self,
        buf: torch.Tensor,
        row_bytes: int,
        tgt_loc: torch.Tensor,
        src_loc: torch.Tensor,
        block_size: int,
    ) -> None:
        rows = buf.view(-1, block_size, row_bytes).reshape(-1, row_bytes)
        rows[tgt_loc.long()] = rows[src_loc.long()]

    def _compressed_locs_from_token_locs(
        self,
        loc: torch.Tensor,
        *,
        ratio: int,
        block_size: int,
    ) -> torch.Tensor:
        page = torch.div(loc.to(torch.int64), self.page_size, rounding_mode="floor")
        pos = loc.to(torch.int64) % self.page_size
        return page * block_size + torch.div(pos, ratio, rounding_mode="floor")

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        for layer_id in range(self.layer_num):
            self._move_fp8_ds_mla_rows(
                self.swa_kv_buffer[layer_id],
                tgt_loc,
                src_loc,
                self.swa_block_size,
            )
            buf = self.compressed_kv_buffer[layer_id]
            if buf is not None:
                ratio = self.layout.layer_ratio[layer_id]
                block_size = self.get_compressed_block_size(layer_id)
                self._move_fp8_ds_mla_rows(
                    buf,
                    self._compressed_locs_from_token_locs(
                        tgt_loc, ratio=ratio, block_size=block_size
                    ),
                    self._compressed_locs_from_token_locs(
                        src_loc, ratio=ratio, block_size=block_size
                    ),
                    block_size,
                )
            for buffers, row_bytes in (
                (self.indexer_kv_buffer, self.layout.indexer_row_bytes),
            ):
                buf = buffers[layer_id]
                if buf is not None:
                    ratio = self.layout.layer_ratio[layer_id]
                    block_size = self.get_indexer_block_size(layer_id)
                    self._move_rows(
                        buf,
                        row_bytes,
                        self._compressed_locs_from_token_locs(
                            tgt_loc, ratio=ratio, block_size=block_size
                        ),
                        self._compressed_locs_from_token_locs(
                            src_loc, ratio=ratio, block_size=block_size
                        ),
                        block_size,
                    )
            for buffers in (self.compressor_state_buffer, self.indexer_state_buffer):
                buf = buffers[layer_id]
                if buf is not None:
                    rows = buf.view(-1, buf.shape[-1])
                    rows[tgt_loc.long()] = rows[src_loc.long()]

    def _all_buffers(self) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for layer_id in range(self.layer_num):
            out.append(self.swa_kv_buffer[layer_id])
            for buffers in (
                self.compressed_kv_buffer,
                self.compressor_state_buffer,
                self.indexer_kv_buffer,
                self.indexer_state_buffer,
            ):
                buf = buffers[layer_id]
                if buf is not None:
                    out.append(buf)
        return out

    def get_kv_size_bytes(self) -> int:
        return int(
            sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self._all_buffers())
        )

    def get_contiguous_buf_infos(self):
        buffers = self._all_buffers()
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        offsets = []
        cursor = start_idx
        for layer_id in range(self.layer_num):
            layer_offsets = [cursor]
            cursor += 1
            for buffers in (
                self.compressed_kv_buffer,
                self.compressor_state_buffer,
                self.indexer_kv_buffer,
                self.indexer_state_buffer,
            ):
                if buffers[layer_id] is not None:
                    layer_offsets.append(cursor)
                    cursor += 1
            offsets.append(layer_offsets)
        return offsets

    def get_cpu_copy(self, token_indices: list[int]) -> list[torch.Tensor]:
        del token_indices
        raise NotImplementedError(
            "DeepSeek V4 KV cache offload is not implemented; the compressed-MQA "
            "and indexer buffers are page-shaped and require page-aware indexing."
        )

    def load_cpu_copy(self, kv_cache_cpu, token_indices: list[int]) -> None:
        del kv_cache_cpu, token_indices
        raise NotImplementedError(
            "DeepSeek V4 KV cache reload is not implemented; the compressed-MQA "
            "and indexer buffers are page-shaped and require page-aware indexing."
        )
