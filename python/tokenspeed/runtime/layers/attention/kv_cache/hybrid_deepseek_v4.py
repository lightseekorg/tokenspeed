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
from typing import ClassVar

import torch
from tokenspeed_kernel.ops.attention.triton.dsv4 import dsv4_compressed_slot_mapping

from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_KERNEL_BLOCK_ROWS,
    V4_SWA_KV_GROUP_ID,
    DeepseekV4CacheLayout,
    parse_v4_compressor_state_group_id,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _split_block_tables_into_v4_metadata(
    block_tables: dict[str, torch.Tensor],
    block_table_base_offsets: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
    torch.Tensor | None,
    dict[int, torch.Tensor],
    torch.Tensor | None,
]:
    """Split the cache-group dict into V4-named tables + per-sliding-group offsets.

    Returns (swa, {ratio: compressor_state}, indexer_state, swa_base,
    {ratio: compressor_state_base}, indexer_state_base). Unknown group ids
    are ignored. Base offsets are None / missing when the input lacks them.
    """
    offsets = block_table_base_offsets or {}
    swa = block_tables.get(V4_SWA_KV_GROUP_ID)
    indexer_state = block_tables.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    swa_base = offsets.get(V4_SWA_KV_GROUP_ID)
    indexer_state_base = offsets.get(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID)
    compressor_state: dict[int, torch.Tensor] = {}
    compressor_state_base: dict[int, torch.Tensor] = {}
    for gid, table in block_tables.items():
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


def _expand_group_values_for_tokens(
    values: torch.Tensor,
    num_tokens: int,
    name: str,
) -> torch.Tensor:
    if values.numel() == num_tokens:
        return values
    if values.numel() <= 0 or num_tokens % values.numel() != 0:
        raise RuntimeError(
            f"DeepSeek V4 {name} has incompatible shape for packed tokens: "
            f"{values.numel()} entries for {num_tokens} tokens"
        )
    return values.repeat_interleave(num_tokens // values.numel())


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
    req_indices = _expand_group_values_for_tokens(
        req_indices,
        positions.numel(),
        "request indices",
    )
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


def _mask_invalid_graph_tokens(
    slot_mapping: torch.Tensor,
    is_valid_token: torch.Tensor | None,
) -> torch.Tensor:
    if is_valid_token is None:
        return slot_mapping
    valid = _expand_group_values_for_tokens(
        is_valid_token,
        slot_mapping.numel(),
        "slot validity mask",
    ).to(
        device=slot_mapping.device,
        dtype=torch.bool,
    )
    return torch.where(valid, slot_mapping, torch.full_like(slot_mapping, -1))


def _compressed_boundary_mask(
    positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio <= 1:
        return torch.ones_like(positions, dtype=torch.bool)
    return ((positions.to(torch.int64) + 1) % compress_ratio) == 0


@dataclass
class DeepseekV4CacheMetadata:
    page_size: int
    page_table: torch.Tensor
    block_tables: dict[str, torch.Tensor] = field(default_factory=dict)
    # Per-sliding-group [num_reqs] int32 base logical-page offset that
    # accompanies each compact per-group table. Consumers index sliding tables as
    # logical_page - base_offset; full-history groups omit the key (base 0).
    block_table_base_offsets: dict[str, torch.Tensor] = field(default_factory=dict)
    swa_page_table: torch.Tensor | None = None
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

    def compressed_page_table(
        self,
        compress_ratio: int,
        kv_cache_block_size: int | None = None,
    ) -> torch.Tensor:
        del kv_cache_block_size
        if compress_ratio <= 1:
            return self.page_table
        table = self.block_tables.get(v4_compressed_kv_group_id(compress_ratio))
        if table is None:
            raise RuntimeError(
                "DeepSeek V4 missing cache-group block table for compressed "
                f"KV group {v4_compressed_kv_group_id(compress_ratio)!r}"
            )
        return table

    @staticmethod
    def safe_page_ids(
        block_table: torch.Tensor,
        req_indices: torch.Tensor,
        page_indices: torch.Tensor,
    ) -> torch.Tensor:
        return _safe_page_ids(block_table, req_indices, page_indices)

    def _update_decode_compressed_slot_mapping(
        self,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        compress_ratio: int,
        kv_cache_block_size: int,
        is_valid_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = token_to_req_indices.shape[0]
        key = (compress_ratio, kv_cache_block_size)
        out = self.decode_compressed_slot_mappings.get(key)
        if out is None or out.shape[0] < num_tokens or out.device != seq_lens.device:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 compressed slot metadata must be allocated before "
                    "CUDA graph capture"
                )
            with torch.inference_mode(False):
                out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
            self.decode_compressed_slot_mappings[key] = out

        page_table = self.compressed_page_table(compress_ratio, kv_cache_block_size)
        if page_table is not self.page_table:
            req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
            query_starts = query_start_loc[req_idx].to(torch.int64)
            query_lens = query_start_loc[req_idx + 1].to(torch.int64) - query_starts
            seq_lens_for_token = seq_lens[req_idx].to(torch.int64)
            token_offsets = torch.arange(
                num_tokens,
                dtype=torch.int64,
                device=seq_lens.device,
            )
            positions = seq_lens_for_token - query_lens + token_offsets - query_starts
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
            base_offsets = self.block_table_base_offsets.get(
                v4_compressed_kv_group_id(compress_ratio)
            )
            if base_offsets is not None:
                page_indices = (
                    page_indices
                    - base_offsets.to(
                        device=page_indices.device,
                        dtype=torch.int64,
                    )[req_idx]
                )
            page_ids = _safe_page_ids(page_table, req_idx, page_indices)
            valid_slots = (page_ids >= 0) & _compressed_boundary_mask(
                positions,
                compress_ratio,
            )
            slot_mapping = torch.where(
                valid_slots,
                page_ids * kv_cache_block_size + offsets,
                torch.full_like(page_ids, -1),
            )
            out.copy_(_mask_invalid_graph_tokens(slot_mapping, is_valid_token))
            return out

        mapping = dsv4_compressed_slot_mapping(
            num_tokens=num_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=self.page_table,
            block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            out=out,
        )
        if is_valid_token is not None:
            mapping.copy_(_mask_invalid_graph_tokens(mapping, is_valid_token))
        return mapping

    def refresh_decode_compressed_slot_mappings(
        self,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        is_valid_token: torch.Tensor | None = None,
    ) -> None:
        for compress_ratio, kv_cache_block_size in list(
            self.decode_compressed_slot_mappings
        ):
            self._update_decode_compressed_slot_mapping(
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                compress_ratio=compress_ratio,
                kv_cache_block_size=kv_cache_block_size,
                is_valid_token=is_valid_token,
            )

    def compressed_slot_mapping(
        self,
        positions: torch.Tensor,
        compress_ratio: int,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache_block_size: int | None = None,
        use_decode_cache: bool = False,
        is_valid_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache_block_size is None:
            kv_cache_block_size = self.page_size
        page_table = self.compressed_page_table(compress_ratio, kv_cache_block_size)
        if (
            use_decode_cache
            and positions.is_cuda
            and (page_table.is_cuda or self.page_table.is_cuda)
        ):
            cached = self.decode_compressed_slot_mappings.get(
                (compress_ratio, kv_cache_block_size)
            )
            if (
                cached is not None
                and cached.shape[0] >= positions.numel()
                and cached.device == seq_lens.device
            ):
                return cached[: positions.numel()]
            mapping = self._update_decode_compressed_slot_mapping(
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                compress_ratio=compress_ratio,
                kv_cache_block_size=kv_cache_block_size,
                is_valid_token=is_valid_token,
            )
            return mapping[: positions.numel()]
        compressed_pos = torch.div(
            positions.to(torch.int64), compress_ratio, rounding_mode="floor"
        )
        page_indices = torch.div(
            compressed_pos, kv_cache_block_size, rounding_mode="floor"
        )
        offsets = compressed_pos % kv_cache_block_size
        req_idx = token_to_req_indices[: positions.numel()].long()
        if page_table is self.page_table:
            page_ids = page_table[req_idx, page_indices.long()].to(torch.int64)
        else:
            base_offsets = self.block_table_base_offsets.get(
                v4_compressed_kv_group_id(compress_ratio)
            )
            if base_offsets is not None:
                page_indices = (
                    page_indices
                    - base_offsets.to(
                        device=page_indices.device,
                        dtype=torch.int64,
                    )[req_idx]
                )
            page_ids = _safe_page_ids(page_table, req_idx, page_indices.long())
        slots = page_ids.to(torch.int64) * kv_cache_block_size + offsets
        valid_slots = (page_ids >= 0) & _compressed_boundary_mask(
            positions,
            compress_ratio,
        )
        slot_mapping = torch.where(
            valid_slots,
            slots,
            torch.full_like(slots, -1),
        )
        return _mask_invalid_graph_tokens(slot_mapping, is_valid_token)


class HybridDeepseekV4TokenToKVPool(CachePool):
    """DeepSeek V4 fp8_ds_mla cache pool.

    TokenSpeed keeps SWA, compressed, compressor-state, and CSA indexer caches
    in dedicated per-group paged pools (see CacheGroup* on the scheduler
    side and the V4 recipe here), keeping ordinary MLA models on
    their existing single-pool contract. The ``indexer_kv_buffer`` shares its
    page table and page-count budget with the ``v4.c{ratio}a.compressed_kv``
    group rather than owning a separate group of its own.
    """

    def __init__(
        self,
        arena: CacheArena,
        model_dtype: torch.dtype,
        layout: DeepseekV4CacheLayout,
        layer_num: int,
        rank: int,
        field_layer_offset: int = 0,
    ) -> None:
        # The layout is this view's own window (a draft view carries only its
        # continuation layers' ratios), so it is indexed by local layer id.
        if layer_num != len(layout.layer_ratio):
            raise ValueError(
                "DeepSeek V4 KV pool layer_num must match cache layout ratios: "
                f"layer_num={layer_num}, ratios={len(layout.layer_ratio)}"
            )
        super().__init__(
            arena,
            torch.uint8,
            rank,
            field_layer_offset=field_layer_offset,
        )
        plan = self.arena.plan
        prefix_granularity = self.arena.prefix_granularity
        self.model_dtype = model_dtype
        self.layout = layout
        self.layer_num = layer_num
        self._cache_group_specs_by_id = {
            spec.group_id: spec for spec in self.arena.cache_group_specs
        }
        self.requires_page_zeroing = True

        def _group_rows(group_id: str, default: int) -> int:
            spec = self._cache_group_specs_by_id.get(group_id)
            return int(spec.rows_per_page) if spec is not None else int(default)

        self.swa_block_size = _group_rows(
            V4_SWA_KV_GROUP_ID,
            V4_KERNEL_BLOCK_ROWS,
        )
        self.swa_block_bytes = layout.swa_block_bytes(self.swa_block_size)
        self.compressed_block_sizes = tuple(
            layout.storage_block_size(ratio) if ratio > 1 else prefix_granularity
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
                _group_rows(v4_compressor_state_group_id(ratio), prefix_granularity)
                if ratio > 1
                else prefix_granularity
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
        self._bind_layer_planes()

        logger.info(
            "Initialized DeepSeek V4 cache pool: %d parents, P=%d, %d layers, "
            "fp4 indexer=%s, compressed block sizes=%s",
            plan.num_lcm_blocks,
            prefix_granularity,
            layer_num,
            layout.use_fp4_indexer_cache,
            self.compressed_block_sizes,
        )

    # A ratio-1 layer plans no compressed/state planes and only ratio-4 plans
    # indexer planes, so the plan's field list decides which planes a layer
    # has, and each is read with the shape the plan gives it.
    layer_plane_bindings: ClassVar[dict[str, str]] = {
        "swa": "swa_kv_buffer",
        "compressed_kv": "compressed_kv_buffer",
        "compressor_state": "compressor_state_buffer",
        "indexer_kv": "indexer_kv_buffer",
        "indexer_state": "indexer_state_buffer",
    }

    def _require(
        self, buffers: list[torch.Tensor | None], layer_id: int, name: str
    ) -> torch.Tensor:
        buf = buffers[layer_id]
        if buf is None:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no {name} cache")
        return buf

    def get_swa_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_buffer[layer_id]

    @property
    def swa_capacity_slots(self) -> int:
        """Writable SWA cache capacity shared by every layer, in token slots.

        Every layer's SWA buffer is allocated with the same page count, so a
        single capacity (pages * tokens per block) bounds the write-slot
        mapping shared across layers. Returns 0 when no SWA buffers exist;
        callers must then mask all slots rather than skip the bounds check.
        """
        # Under a pipeline-parallel layer window only this stage's layers'
        # buffers are bound (the rest stay None), so probe the first bound
        # one — every layer's SWA plane shares the same page count.
        for buffer in self.swa_kv_buffer or ():
            if buffer is not None:
                return int(buffer.shape[0]) * int(self.swa_block_size)
        return 0

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

    def get_indexer_kv_buffer_2d(self, layer_id: int) -> torch.Tensor:
        return self._require(self.indexer_kv_buffer, layer_id, "indexer KV")

    def get_indexer_state_block_size(self, layer_id: int) -> int:
        block_size = self.indexer_state_block_sizes[layer_id]
        if block_size <= 0:
            raise ValueError(f"DeepSeek V4 layer {layer_id} has no indexer state cache")
        return block_size

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

    def get_kv_size_bytes(self) -> int:
        return int(self.arena.buffer.nbytes)

    def zero_new_blocks(self, new_page_ids: dict[str, list[int]]) -> None:
        self.arena.zero_blocks(new_page_ids)
