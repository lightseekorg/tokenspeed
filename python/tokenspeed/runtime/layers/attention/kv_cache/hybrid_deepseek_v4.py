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

from dataclasses import dataclass, field
from typing import ClassVar

import torch
from tokenspeed_kernel.ops.attention.dsv4.triton import (
    dsv4_compact_compressed_slot_mapping,
    dsv4_compressed_slot_mapping,
)
from tokenspeed_kernel.ops.kvcache.triton_virtual_blocks import virtual_slots_to_local
from typing_extensions import override

from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_INDEXER_KV_GROUP_ID,
    V4_KERNEL_BLOCK_ROWS,
    V4_SWA_KV_GROUP_ID,
    DeepseekV4CacheLayout,
    parse_v4_compressed_kv_group_id,
    parse_v4_compressor_state_group_id,
    v4_compressed_kv_group_id,
    v4_compressed_rows_per_page,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
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
    """Scheduler block tables of one batch plus their rank-local read views.

    ``block_tables`` hold the scheduler's virtual block IDs. Writers translate
    through :meth:`local_compressed_write_slots`; attention reads the tables
    :meth:`refresh_page_tables` derives from them, in which the null
    block and every page another DCP rank owns are ``-1``. A replicated group
    (``shard_count == 1``) translates to itself, so the same path serves every
    DCP size.
    """

    page_size: int
    page_table: torch.Tensor
    dcp_size: int
    dcp_rank: int
    runtime_contract: CacheRuntimeContract
    block_tables: dict[str, torch.Tensor] = field(default_factory=dict)
    swa_page_table: torch.Tensor | None = None
    compressor_state_block_tables: dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_state_block_table: torch.Tensor | None = None
    # Keyed by (compress_ratio, indexer, kv_cache_block_size).
    decode_compressed_slot_mappings: dict[tuple[int, bool, int], torch.Tensor] = field(
        default_factory=dict
    )
    # Local read tables per compressed group; refreshed in place so CUDA
    # graphs keep their captured pointers.
    compressed_page_tables: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_group_tables(
        cls,
        *,
        page_size: int,
        page_table: torch.Tensor,
        block_tables: dict[str, torch.Tensor],
        dcp_size: int,
        dcp_rank: int,
        runtime_contract: CacheRuntimeContract,
    ) -> "DeepseekV4CacheMetadata":
        """Bind the cache-group tables and name the V4-specific ones.

        Args:
            page_size: Kernel page size of ``page_table``.
            page_table: Batch-ordered base full-history table.
            block_tables: Cache-group tables keyed by group id; the SWA,
                per-ratio compressor-state and indexer-state groups are
                also exposed under their V4 names. Unknown ids ride along.
            dcp_size: Owners of each sharded group's virtual blocks.
            dcp_rank: This process's position among those owners.
            runtime_contract: The bound arena's contract; its virtual block
                counts bound every translation.

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
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            runtime_contract=runtime_contract,
            block_tables=block_tables,
            swa_page_table=block_tables.get(V4_SWA_KV_GROUP_ID),
            compressor_state_block_tables=compressor_state,
            indexer_state_block_table=block_tables.get(
                V4_INDEXER_COMPRESSOR_STATE_GROUP_ID
            ),
        )

    def slice_requests(self, start: int, end: int) -> "DeepseekV4CacheMetadata":
        """Views over request rows ``[start, end)`` of every table, read views included."""
        sliced = DeepseekV4CacheMetadata.from_group_tables(
            page_size=self.page_size,
            page_table=self.page_table[start:end],
            block_tables={
                key: table[start:end] for key, table in self.block_tables.items()
            },
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            runtime_contract=self.runtime_contract,
        )
        sliced.compressed_page_tables = {
            group_id: table[start:end]
            for group_id, table in self.compressed_page_tables.items()
        }
        return sliced

    def refresh_page_tables(self) -> None:
        """Translate every compressed group's table to local pages, in place.

        Pages this rank does not own and the virtual null block become ``-1``.
        Output buffers are reused whenever their geometry matches, so a table
        captured into a CUDA graph is refreshed rather than replaced.
        """
        for group_id, table in self.block_tables.items():
            if parse_v4_compressed_kv_group_id(group_id) is None:
                continue
            out = self.compressed_page_tables.get(group_id)
            if out is not None and (
                out.shape != table.shape
                or out.dtype != table.dtype
                or out.device != table.device
            ):
                out = None
            if out is None:
                # Replay setup may run outside the warmup inference context.
                with torch.inference_mode(False):
                    out = torch.empty_like(table, memory_format=torch.contiguous_format)
            local, owned = virtual_slots_to_local(
                table,
                rows_per_page=1,
                virtual_block_count=self.runtime_contract.virtual_block_counts[
                    group_id
                ],
                degree=self.dcp_size,
                rank=self.dcp_rank,
                out=out,
            )
            local.masked_fill_(~owned, -1)
            self.compressed_page_tables[group_id] = local

    def compressed_page_table(self, compress_ratio: int) -> torch.Tensor:
        """The local read table :meth:`refresh_page_tables` prepared."""
        return self.compressed_page_tables[v4_compressed_kv_group_id(compress_ratio)]

    def compressed_block_table(self, compress_ratio: int) -> torch.Tensor:
        """The scheduler's virtual table for one compressed KV group."""
        if compress_ratio <= 1:
            return self.page_table
        return self._group_table(v4_compressed_kv_group_id(compress_ratio))

    def indexer_block_table(self) -> torch.Tensor:
        """The scheduler's table for the replicated indexer K group."""
        return self._group_table(V4_INDEXER_KV_GROUP_ID)

    def _group_table(self, group_id: str) -> torch.Tensor:
        table = self.block_tables.get(group_id)
        if table is None:
            raise RuntimeError(
                f"DeepSeek V4 missing cache-group block table for group {group_id!r}"
            )
        return table

    def _slot_table(self, compress_ratio: int, *, indexer: bool) -> torch.Tensor:
        """The virtual table slot mappings for ``compress_ratio`` derive from."""
        if indexer:
            return self.indexer_block_table()
        return self.compressed_block_table(compress_ratio)

    def local_compressed_write_slots(
        self, slots: torch.Tensor, compress_ratio: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate virtual compressed-KV slots to local slots and a write mask.

        Args:
            slots: Virtual slots from :meth:`compressed_slot_mapping`; ``-1``
                marks a token that writes nothing.
            compress_ratio: The compressed KV group the slots address.

        Returns:
            Local slots, with unowned and masked tokens parked on slot 0, and
            the mask that is True only where this rank owns the row.
        """
        group_id = v4_compressed_kv_group_id(compress_ratio)
        return virtual_slots_to_local(
            slots,
            rows_per_page=v4_compressed_rows_per_page(compress_ratio),
            virtual_block_count=self.runtime_contract.virtual_block_counts[group_id],
            degree=self.dcp_size,
            rank=self.dcp_rank,
        )

    def _update_decode_compressed_slot_mapping(
        self,
        *,
        token_to_req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        compress_ratio: int,
        indexer: bool,
        kv_cache_block_size: int,
        is_valid_token: torch.Tensor | None,
    ) -> torch.Tensor:
        num_tokens = token_to_req_indices.shape[0]
        key = (compress_ratio, indexer, kv_cache_block_size)
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

        page_table = self._slot_table(compress_ratio, indexer=indexer)
        if page_table is not self.page_table:
            mapping = dsv4_compact_compressed_slot_mapping(
                num_tokens=num_tokens,
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=page_table,
                block_size=kv_cache_block_size,
                compress_ratio=compress_ratio,
                block_table_base_offsets=None,
                is_valid_token=is_valid_token,
                out=out,
            )
            # The fused mapper permits page 0 during warmup; the virtual null
            # block has no owner and must never be written.
            mapping.masked_fill_(mapping < kv_cache_block_size, -1)
            return mapping

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
        for compress_ratio, indexer, kv_cache_block_size in list(
            self.decode_compressed_slot_mappings
        ):
            self._update_decode_compressed_slot_mapping(
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                compress_ratio=compress_ratio,
                indexer=indexer,
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
        indexer: bool,
        kv_cache_block_size: int | None = None,
        use_decode_cache: bool = False,
        is_valid_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Virtual slots each token writes in a compressed group, ``-1`` for none.

        Args:
            positions: Token positions in their requests.
            compress_ratio: Compression ratio of the group being written.
            token_to_req_indices: Request row of each token.
            query_start_loc: Cumulative query lengths per request.
            seq_lens: Sequence length per request.
            indexer: Address the replicated indexer K group instead of the
                compressed KV group of ``compress_ratio``.
            kv_cache_block_size: Rows per page of the addressed group.
            use_decode_cache: Serve decode batches from the persistent
                per-group mapping buffers.
            is_valid_token: Graph padding mask; padded tokens write nothing.

        Returns:
            Int64 virtual slots, one per token. Compressed KV callers pass them
            through :meth:`local_compressed_write_slots` before writing.
        """
        if kv_cache_block_size is None:
            kv_cache_block_size = self.page_size
        page_table = self._slot_table(compress_ratio, indexer=indexer)
        if (
            use_decode_cache
            and positions.is_cuda
            and (page_table.is_cuda or self.page_table.is_cuda)
        ):
            cached = self.decode_compressed_slot_mappings.get(
                (compress_ratio, indexer, kv_cache_block_size)
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
                indexer=indexer,
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
        # Page 0 is the null block: never a write target.
        valid_slots = (page_ids >= 1) & _compressed_boundary_mask(
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

    The SWA, compressed-KV, compressor-state, CSA indexer K and indexer-state
    caches are each a cache group of the one shared arena (the V4 recipe
    declares them; the scheduler addresses them as ``CacheGroup``s), and this
    view binds their planes per layer. Only the compressed-KV groups may be
    sharded across DCP ranks; the indexer reads every rank's rows, so its K
    stays a replicated group of its own.
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
            f"Initialized DeepSeek V4 cache pool: {plan.num_lcm_blocks:d} parents, P="
            f"{prefix_granularity:d}, {layer_num:d} layers, "
            f"fp4 indexer={layout.use_fp4_indexer_cache!s}, compressed block sizes="
            f"{self.compressed_block_sizes!s}",
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

    # A V4 layer's fused attention reads several history groups at once (the
    # SWA window beside its compressed chain and the compressor tails), so no
    # layer rides one group through ``PagedAttention`` and no router leaf
    # serves this view: the V4 backend takes every group's table by id.
    @property
    @override
    def paged_group_ids(self) -> tuple[str, ...]:
        return ()

    @override
    def history_group_by_layer(self) -> dict[int, str]:
        return {}

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
