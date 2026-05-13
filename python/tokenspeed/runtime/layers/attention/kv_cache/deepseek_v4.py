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
from typing import Any

import numpy as np
import torch

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_KERNEL_BLOCK_ROWS,
    V4_SWA_KV_GROUP_ID,
    build_v4_cache_specs,
    parse_v4_compressor_state_group_id,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.configs.paged_cache_spec import (
    compute_paged_cache_group_page_counts,
)
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

DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE = 256


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
        if compress_ratio > 1:
            return max(1, DEEPSEEK_V4_COMPRESSED_LOGICAL_BLOCK_SIZE // compress_ratio)
        return self.page_size

    def compressor_state_block_size(self, compress_ratio: int) -> int:
        if compress_ratio == 4:
            return 4
        if compress_ratio == 128:
            return 8
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


def _split_paged_cache_block_tables_into_v4_metadata(
    paged_cache_block_tables: dict[str, torch.Tensor],
    paged_cache_block_table_base_offsets: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
]:
    """Split paged-cache dict into V4-named tables + per-sliding-group offsets.

    Returns (swa, {ratio: compressor_state}, indexer_state, swa_base,
    {ratio: compressor_state_base}, indexer_state_base). Unknown group ids
    are ignored. Base offsets are None / missing when the input lacks them.
    """
    offsets = paged_cache_block_table_base_offsets or {}
    swa = paged_cache_block_tables.get(V4_SWA_KV_GROUP_ID)
    indexer_state = paged_cache_block_tables.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    swa_base = offsets.get(V4_SWA_KV_GROUP_ID)
    indexer_state_base = offsets.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    compressor_state: dict[int, torch.Tensor] = {}
    compressor_state_base: dict[int, torch.Tensor] = {}
    for gid, table in paged_cache_block_tables.items():
        ratio = parse_v4_compressor_state_group_id(gid)
        if ratio is None:
            continue
        compressor_state[ratio] = table
        base = offsets.get(gid)
        if base is not None:
            compressor_state_base[ratio] = base
    return (
        swa,
        compressor_state,
        indexer_state,
        swa_base,
        compressor_state_base,
        indexer_state_base,
    )


def _safe_page_ids(
    block_table: torch.Tensor,
    req_indices: torch.Tensor,
    page_indices: torch.Tensor,
) -> torch.Tensor:
    req_i64 = req_indices.to(torch.int64)
    page_i64 = page_indices.to(torch.int64)
    sentinel = torch.full_like(page_i64, -1, dtype=torch.int64)
    rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
    cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    if rows <= 0 or cols <= 0:
        return sentinel
    valid = (req_i64 >= 0) & (req_i64 < rows) & (page_i64 >= 0) & (page_i64 < cols)
    safe_req = req_i64.clamp(0, rows - 1)
    safe_page = page_i64.clamp(0, cols - 1)
    page_ids = block_table[safe_req, safe_page].to(torch.int64)
    return torch.where(valid, page_ids, sentinel)


def _group_slot_mapping_from_raw(
    positions: torch.Tensor,
    req_indices: torch.Tensor,
    block_table: torch.Tensor,
    rows_per_page: int,
    entry_stride_tokens: int = 1,
    base_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    if rows_per_page <= 0:
        raise ValueError(f"rows_per_page must be > 0, got {rows_per_page}")
    if entry_stride_tokens <= 0:
        raise ValueError(f"entry_stride_tokens must be > 0, got {entry_stride_tokens}")
    pos_i64 = positions.to(torch.int64)
    logical_row = torch.div(pos_i64, entry_stride_tokens, rounding_mode="floor")
    logical_page = torch.div(logical_row, rows_per_page, rounding_mode="floor")
    offsets = logical_row % rows_per_page
    table_page = logical_page
    if base_offsets is not None:
        req_i64 = req_indices.to(torch.int64)
        rows = int(base_offsets.shape[0])
        if rows <= 0:
            table_page = logical_page.new_full(logical_page.shape, -1)
        else:
            valid_req = (req_i64 >= 0) & (req_i64 < rows)
            safe_req = req_i64.clamp(0, rows - 1)
            base = base_offsets.to(
                device=logical_page.device,
                dtype=torch.int64,
            )[safe_req]
            table_page = torch.where(valid_req, logical_page - base, -1)
    page_ids = _safe_page_ids(block_table, req_indices, table_page)
    slots = page_ids * rows_per_page + offsets
    return torch.where(page_ids >= 0, slots, torch.full_like(slots, -1))


@dataclass
class DeepseekV4ForwardMetadata:
    page_size: int
    req_pool_indices: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req_indices: torch.Tensor
    # Padding mask for CUDA graph replay rows; this is not mixed-batch state.
    is_valid_token: Optional[torch.Tensor] = None
    # CPU lens are retained for sparse prefill/indexer planning without
    # forcing another device-to-host sync in the model path.
    seq_lens_cpu: Optional[torch.Tensor] = None
    query_lens_cpu: Optional[torch.Tensor] = None
    # Cached split boundary derived from scheduler num_extends/query_lens.
    num_prefill_reqs: int = 0
    num_prefill_tokens: int = 0
    forward_mode: object = None
    decode_swa_indices: torch.Tensor | None = None
    decode_swa_lens: torch.Tensor | None = None
    decode_swa_window_size: int = 0
    decode_swa_block_size: int = 0
    paged_cache_block_tables: dict[str, torch.Tensor] = field(default_factory=dict)
    # Per-sliding-group [num_reqs] int32 base logical-page offset that
    # accompanies each compact block table. Consumers index sliding tables as
    # logical_page - base_offset; full-history groups omit the key (base 0).
    paged_cache_block_table_base_offsets: dict[str, torch.Tensor] = field(
        default_factory=dict
    )
    swa_block_table: torch.Tensor | None = None
    swa_base_logical_page: torch.Tensor | None = None
    compressor_state_block_tables: dict[int, torch.Tensor] = field(default_factory=dict)
    compressor_state_base_logical_pages: dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    indexer_state_block_table: torch.Tensor | None = None
    indexer_state_base_logical_page: torch.Tensor | None = None
    decode_compressed_slot_mappings: dict[tuple[int, int], torch.Tensor] = field(
        default_factory=dict
    )
    # Cache for dense compressed decode attention indices/lens. CSA decode uses
    # dynamic top-k indices and does not populate this cache.
    decode_dense_compressed_indices_cache: dict[
        tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = field(default_factory=dict)
    decode_dense_compressed_indices_capture_safe_keys: set[
        tuple[int, int, int, int]
    ] = field(default_factory=set)
    decode_indexer_schedule_metadata: dict[tuple[int, int, int], torch.Tensor] = field(
        default_factory=dict
    )
    decode_indexer_plan_cache: dict[tuple[int, int, int], Any] = field(
        default_factory=dict
    )
    decode_indexer_plan_refreshed_keys: set[tuple[int, int, int]] = field(
        default_factory=set
    )
    prefill_indexer_plan_cache: dict[tuple[int, int, int], Any] = field(
        default_factory=dict
    )

    def decode_req_count(self) -> int:
        return max(0, int(self.req_pool_indices.shape[0]) - int(self.num_prefill_reqs))

    def decode_token_count(self) -> int:
        return max(
            0,
            int(self.token_to_req_indices.shape[0]) - int(self.num_prefill_tokens),
        )

    def _use_decode_compressed_slot_cache(self, positions: torch.Tensor) -> bool:
        return (
            self.forward_mode is not None
            and self.forward_mode.is_decode()
            and positions.is_cuda
            and (
                self.compressed_block_table(1, self.page_size).is_cuda
                or self.block_table.is_cuda
            )
        )

    def compressed_block_table(
        self,
        compress_ratio: int,
        kv_cache_block_size: int | None = None,
    ) -> torch.Tensor:
        del kv_cache_block_size
        table = self.paged_cache_block_tables.get(
            v4_compressed_kv_group_id(compress_ratio)
        )
        return table if table is not None else self.block_table

    @staticmethod
    def safe_page_ids(
        block_table: torch.Tensor,
        req_indices: torch.Tensor,
        page_indices: torch.Tensor,
    ) -> torch.Tensor:
        return _safe_page_ids(block_table, req_indices, page_indices)

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

        block_table = self.compressed_block_table(compress_ratio, kv_cache_block_size)
        if block_table is not self.block_table:
            req_idx = self.token_to_req_indices[:num_tokens].to(torch.int64)
            positions = self.seq_lens[req_idx].to(torch.int64) - 1
            compressed_pos = torch.div(
                positions,
                compress_ratio,
                rounding_mode="floor",
            )
            page_indices = torch.div(
                compressed_pos,
                kv_cache_block_size,
                rounding_mode="floor",
            )
            offsets = compressed_pos % kv_cache_block_size
            page_ids = _safe_page_ids(block_table, req_idx, page_indices)
            out.copy_(
                torch.where(
                    page_ids >= 0,
                    page_ids * kv_cache_block_size + offsets,
                    torch.full_like(page_ids, -1),
                )
            )
            return out

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
        block_table = self.compressed_block_table(compress_ratio, kv_cache_block_size)
        req_idx = self.token_to_req_indices[: positions.numel()].long()
        if block_table is self.block_table:
            page_ids = block_table[req_idx, page_indices.long()].to(torch.int64)
        else:
            page_ids = _safe_page_ids(block_table, req_idx, page_indices.long())
        slots = page_ids.to(torch.int64) * kv_cache_block_size + offsets
        return torch.where(
            page_ids >= 0,
            slots,
            torch.full_like(slots, -1),
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

    TokenSpeed keeps SWA, compressed, compressor-state, and CSA indexer caches
    in dedicated per-group paged pools (see PagedCacheGroup* on the scheduler
    side and ``build_v4_cache_specs`` here), keeping ordinary MLA models on
    their existing single-pool contract. The ``indexer_kv_buffer`` shares its
    page table and page-count budget with the ``v4.c{ratio}a.compressed_kv``
    group rather than owning a separate group of its own.
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
        hf_config: Any,
        max_scheduled_tokens: int,
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
        self.paged_cache_group_specs = tuple(
            build_v4_cache_specs(hf_config, layer_ratio=layout.layer_ratio)
        )
        self.paged_cache_group_page_counts = compute_paged_cache_group_page_counts(
            self.paged_cache_group_specs,
            max_live_requests=max_batch_size,
            max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
            max_total_tokens=size,
            max_context_len=max_context_len,
        )

        self._paged_cache_group_specs_by_id = {
            spec.group_id: spec for spec in self.paged_cache_group_specs
        }

        def _group_rows(group_id: str, default: int) -> int:
            spec = self._paged_cache_group_specs_by_id.get(group_id)
            return int(spec.rows_per_page) if spec is not None else int(default)

        def _group_pages(group_id: str, default: int) -> int:
            return int(self.paged_cache_group_page_counts.get(group_id, default))

        self.swa_block_size = _group_rows(V4_SWA_KV_GROUP_ID, V4_KERNEL_BLOCK_ROWS)
        self.state_block_size = page_size
        self.swa_block_bytes = layout.swa_block_bytes(self.swa_block_size)
        self.compressed_block_sizes = tuple(
            layout.storage_block_size(ratio) if ratio > 1 else page_size
            for ratio in layout.layer_ratio
        )
        self.indexer_block_sizes = tuple(
            (
                max(V4_KERNEL_BLOCK_ROWS, self.compressed_block_sizes[layer_id])
                if ratio == 4
                else 0
            )
            for layer_id, ratio in enumerate(layout.layer_ratio)
        )
        self.compressor_state_block_sizes = tuple(
            (
                _group_rows(v4_compressor_state_group_id(ratio), page_size)
                if ratio > 1
                else page_size
            )
            for ratio in layout.layer_ratio
        )
        self.indexer_state_block_sizes = tuple(
            (
                _group_rows(
                    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
                    layout.compressor_state_block_size(ratio),
                )
                if ratio == 4
                else 0
            )
            for ratio in layout.layer_ratio
        )
        self.compressed_block_size = (
            self.compressed_block_sizes[0] if self.compressed_block_sizes else page_size
        )

        swa_pages = _group_pages(V4_SWA_KV_GROUP_ID, self.num_pages)
        self.swa_kv_buffer = [
            torch.zeros(
                (swa_pages, self.swa_block_bytes),
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
            compressed_pages = (
                _group_pages(v4_compressed_kv_group_id(ratio), self.num_pages)
                if has_compressed
                else self.num_pages
            )
            self.compressed_kv_buffer.append(
                torch.zeros(
                    (
                        compressed_pages,
                        layout.swa_block_bytes(compressed_block_size),
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                if has_compressed
                else None
            )
            compressor_state_block_size = self.compressor_state_block_sizes[layer_id]
            compressor_state_pages = (
                _group_pages(
                    v4_compressor_state_group_id(ratio),
                    self.num_pages,
                )
                if has_compressed
                else self.num_pages
            )
            self.compressor_state_buffer.append(
                torch.empty(
                    (
                        compressor_state_pages,
                        compressor_state_block_size,
                        layout.state_width(layer_id) * 2,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                if has_compressed
                else None
            )
            indexer_block_size = self.indexer_block_sizes[layer_id]
            self.indexer_kv_buffer.append(
                torch.zeros(
                    (
                        compressed_pages,
                        indexer_block_size * layout.indexer_row_bytes,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                if has_indexer
                else None
            )
            indexer_state_block_size = self.indexer_state_block_sizes[layer_id]
            indexer_state_pages = (
                _group_pages(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID, self.num_pages)
                if has_indexer
                else self.num_pages
            )
            self.indexer_state_buffer.append(
                torch.empty(
                    (
                        indexer_state_pages,
                        indexer_state_block_size,
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

    @property
    def prefix_cache_required_group_ids(self) -> tuple[str, ...]:
        """All V4 paged-cache groups must be present for a snapshot to be complete."""
        return tuple(str(spec.group_id) for spec in self.paged_cache_group_specs)

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
        block_size = self.indexer_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no indexer cache")
        return block_size

    def get_compressor_state_block_size(self, layer_id: int) -> int:
        block_size = self.compressor_state_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(
                f"DeepSeek V4 layer {layer_id} has no compressor state cache"
            )
        return block_size

    def get_compressor_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.compressor_state_buffer, layer_id, "compressor state")

    def get_compressor_state_view(self, layer_id: int) -> torch.Tensor:
        buf = self.get_compressor_state_buffer(layer_id)
        block_size = self.get_compressor_state_block_size(layer_id)
        return buf.view(-1, block_size, buf.shape[-1])

    def get_indexer_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_kv_buffer, layer_id, "indexer KV")

    def get_indexer_state_block_size(self, layer_id: int) -> int:
        block_size = self.indexer_state_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no indexer state cache")
        return block_size

    def get_indexer_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_state_buffer, layer_id, "indexer state")

    def get_indexer_state_view(self, layer_id: int) -> torch.Tensor:
        buf = self.get_indexer_state_buffer(layer_id)
        block_size = self.get_indexer_state_block_size(layer_id)
        return buf.view(-1, block_size, buf.shape[-1])

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
