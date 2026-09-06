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
from tokenspeed_kernel.ops.attention.triton.dsv4 import (
    dsv4_compact_compressed_slot_mapping,
    dsv4_compressed_slot_mapping,
)

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
from tokenspeed.runtime.layers.attention.page_table import (
    mask_invalid_graph_tokens as _mask_invalid_graph_tokens,
)
from tokenspeed.runtime.layers.attention.page_table import (
    safe_page_ids as _safe_page_ids,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


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
    swa_page_table: torch.Tensor | None = None
    compressor_state_block_tables: dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_state_block_table: torch.Tensor | None = None
    decode_compressed_slot_mappings: dict[tuple[int, int], torch.Tensor] = field(
        default_factory=dict
    )

    @classmethod
    def from_group_tables(
        cls,
        *,
        page_size: int,
        page_table: torch.Tensor,
        block_tables: dict[str, torch.Tensor],
    ) -> "DeepseekV4CacheMetadata":
        """Bind the cache-group tables and name the V4-specific ones.

        Args:
            page_size: Kernel page size of ``page_table``.
            page_table: Batch-ordered base full-history table.
            block_tables: Cache-group tables keyed by group id; the SWA,
                per-ratio compressor-state and indexer-state groups are
                also exposed under their V4 names. Unknown ids ride along.

        Returns:
            The metadata over exactly these tensors (no copies).
        """
        compressor_state: dict[int, torch.Tensor] = {}
        for gid, table in block_tables.items():
            ratio = parse_v4_compressor_state_group_id(gid)
            if ratio is not None:
                compressor_state[ratio] = table
        return cls(
            page_size=page_size,
            page_table=page_table,
            block_tables=block_tables,
            swa_page_table=block_tables.get(V4_SWA_KV_GROUP_ID),
            compressor_state_block_tables=compressor_state,
            indexer_state_block_table=block_tables.get(
                V4_INDEXER_COMPRESSOR_STATE_GROUP_ID
            ),
        )

    def compressed_page_table(self, compress_ratio: int) -> torch.Tensor:
        if compress_ratio <= 1:
            return self.page_table
        table = self.block_tables.get(v4_compressed_kv_group_id(compress_ratio))
        if table is None:
            raise RuntimeError(
                "DeepSeek V4 missing cache-group block table for compressed "
                f"KV group {v4_compressed_kv_group_id(compress_ratio)!r}"
            )
        return table

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

        page_table = self.compressed_page_table(compress_ratio)
        if page_table is not self.page_table:
            return dsv4_compact_compressed_slot_mapping(
                num_tokens=num_tokens,
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=page_table,
                block_size=kv_cache_block_size,
                compress_ratio=compress_ratio,
                is_valid_token=is_valid_token,
                out=out,
            )

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
        page_table = self.compressed_page_table(compress_ratio)
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
    """DeepSeek V4 fp8_ds_mla cache pool: one layer window over the arena.

    The SWA, compressed-KV, compressor-state and CSA indexer-state caches are
    each a cache group of the one shared arena (the V4 recipe declares them;
    the scheduler addresses them as ``CacheGroup``s), and this view binds
    their planes per layer. The ``indexer_kv_buffer`` shares its page table
    and page-count budget with the ``v4.c{ratio}a.compressed_kv`` group
    rather than owning a separate group of its own.
    """

    def __init__(
        self,
        arena: CacheArena,
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
        self.layer_num = layer_num
        self._cache_group_specs_by_id = {
            spec.group_id: spec for spec in self.arena.cache_group_specs
        }
        self.requires_page_zeroing = True

        def _group_rows(group_id: str) -> int:
            spec = self._cache_group_specs_by_id.get(group_id)
            if spec is None:
                raise RuntimeError(
                    f"DeepSeek V4 cache pool: the arena publishes no {group_id!r} "
                    f"group (published: {sorted(self._cache_group_specs_by_id)})"
                )
            return int(spec.rows_per_page)

        self.swa_block_size = _group_rows(V4_SWA_KV_GROUP_ID)
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
                _group_rows(v4_compressor_state_group_id(ratio))
                if ratio > 1
                else prefix_granularity
            )
            for ratio in layout.layer_ratio
        )
        self.indexer_state_block_sizes = tuple(
            _group_rows(V4_INDEXER_COMPRESSOR_STATE_GROUP_ID) if ratio == 4 else 0
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
