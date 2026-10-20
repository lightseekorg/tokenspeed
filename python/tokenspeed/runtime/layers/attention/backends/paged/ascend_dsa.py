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

"""Ascend implementation of the paged DSA backend."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.dsa.ascend import (
    ascend_dsa_cp_metadata,
    ascend_dsa_kernels,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import register_backend

# Fixed page geometry of Ascend SparseFlashAttentionDecode.
ASCEND_SFAD_PAGE_SIZE = 128


@dataclass(frozen=True)
class _AscendCachePlacement(CachePlacement):
    # The placement owner prepares native DSA inputs before the first Prolog.
    prepare_decode_metadata: Callable | None = None


def prepare_ascend_dsa_cp_metadata(
    locations, placement, *, initial_tokens, window_tokens
):
    """Prepare the leaf's CP metadata before Prolog through its placement owner."""
    if (
        not isinstance(placement, _AscendCachePlacement)
        or placement.prepare_decode_metadata is None
    ):
        raise ValueError("Native DSA requires Ascend CP cache placement")
    slots = placement.prepare_decode_metadata(
        locations,
        init_tokens=initial_tokens,
        local_tokens=window_tokens,
    )
    return slots


@dataclass(frozen=True)
class _AscendSparseSelection:
    query: torch.Tensor
    indices: torch.Tensor
    valid_chunks: torch.Tensor
    q_ends: torch.Tensor
    kv_lengths: torch.Tensor
    table: torch.Tensor
    sparse_mode: int
    context_parallel: bool


@dataclass(frozen=True)
class _AscendDCPKernelMetadata:
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    init_counts: torch.Tensor
    local_counts: torch.Tensor


@dataclass(frozen=True)
class _AscendIndexerMetadata:
    q_ends: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    init_counts: torch.Tensor
    local_counts: torch.Tensor
    visible_lengths: torch.Tensor | None


class _AscendDSAContextParallel:
    """Local DSA page metadata for one context-parallel group."""

    def __init__(
        self,
        *,
        degree: int,
        rank: int,
        ranks: tuple[int, ...],
        virtual_block_count: int | None,
    ) -> None:
        if degree < 1 or len(ranks) != degree or len(set(ranks)) != degree:
            raise ValueError("invalid DCP group")
        if not 0 <= rank < degree:
            raise ValueError("invalid DCP rank")
        self.degree, self.rank, self.ranks = degree, rank, ranks
        self.virtual_block_count = virtual_block_count
        self.page_table: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.init_counts: torch.Tensor | None = None
        self.local_counts: torch.Tensor | None = None
        self.query_ends: torch.Tensor | None = None
        self.write_slots: torch.Tensor | None = None
        self._metadata_buffers: dict[str, torch.Tensor] = {}
        self._decode_rows_capacity: int | None = None
        self._prepared_metadata_key: tuple | None = None
        self._metadata_sources: tuple[torch.Tensor, ...] = ()
        self._indexer_metadata: _AscendIndexerMetadata | None = None
        self._query_group_size: int | None = None

    def clear_forward_metadata(self) -> None:
        """Invalidate layer-shared results before the next forward/capture."""
        self._prepared_metadata_key = None
        self._metadata_sources = ()
        self._indexer_metadata = None
        self._query_group_size = None

    def prepare_metadata(
        self,
        *,
        seq_lens,
        page_table,
        page_size,
        block_granularity,
        init_tokens,
        local_tokens,
        query_width,
        query_group_size,
        slots,
    ) -> None:
        sources = (seq_lens, page_table) + (() if slots is None else (slots,))
        key = (
            tuple(
                (t.data_ptr(), t.shape, t.stride(), t.dtype, t.device) for t in sources
            ),
            page_size,
            block_granularity,
            init_tokens,
            local_tokens,
            query_width,
            query_group_size,
        )
        if key == self._prepared_metadata_key:
            return
        self.refresh_metadata(
            seq_lens=seq_lens,
            page_table=page_table,
            page_size=page_size,
            block_granularity=block_granularity,
            init_tokens=init_tokens,
            local_tokens=local_tokens,
            query_width=query_width,
            query_group_size=query_group_size,
            slots=slots,
        )
        self._metadata_sources = sources
        self._prepared_metadata_key = key

    def bind_virtual_block_count(self, virtual_block_count: int) -> None:
        """Bind the scheduler capacity published by the cache arena."""
        if virtual_block_count <= 1:
            raise ValueError("DSA CP cache must contain usable virtual blocks")
        self.clear_forward_metadata()
        if self.virtual_block_count != virtual_block_count:
            self.virtual_block_count = virtual_block_count
            self.page_table = None
            self.seq_lens = None
            self.init_counts = None
            self.local_counts = None

    def allocate_decode_buffers(self, rows: int, columns: int, device) -> None:
        """Keep all decode metadata at stable addresses across batch sizes."""
        self.clear_forward_metadata()
        self._metadata_buffers = {
            name: torch.empty(shape, dtype=torch.int32, device=device)
            for name, shape in (
                ("query_ends", (rows,)),
                ("page_table", (rows, columns)),
                ("seq_lens", (rows,)),
                ("init_counts", (rows,)),
                ("local_counts", (rows,)),
                ("index_q_ends", (rows,)),
                ("index_page_table", (rows, columns)),
                ("index_seq_lens", (rows,)),
                ("index_init_counts", (rows,)),
            )
        }
        self._metadata_buffers["write_slots"] = torch.empty(
            rows, dtype=torch.int64, device=device
        )
        self._decode_rows_capacity = rows

    def _buffer(self, name, shape, device, dtype=torch.int32):
        target = self._metadata_buffers.get(name)
        if (
            target is None
            or target.ndim != len(shape)
            or target.shape[1:] != shape[1:]
            or target.shape[0] < shape[0]
            or target.dtype != dtype
            or target.device != device
        ):
            if self._decode_rows_capacity is not None:
                raise RuntimeError(f"DSA {name} exceeds preallocated decode geometry")
            target = torch.empty(shape, dtype=dtype, device=device)
            self._metadata_buffers[name] = target
        return target[: shape[0]]

    def refresh_metadata(
        self,
        *,
        seq_lens,
        page_table,
        page_size,
        block_granularity,
        init_tokens,
        local_tokens,
        query_width,
        query_group_size,
        slots,
    ) -> None:
        self.clear_forward_metadata()
        if self.virtual_block_count is None:
            raise RuntimeError("DSA CP cache metadata was not bound")
        rows, columns = page_table.shape[0] * query_width, page_table.shape[1]
        if query_width < 1 or query_group_size < 1 or rows % query_group_size:
            raise ValueError("DSA query rows must form complete Indexer groups")
        groups = rows // query_group_size
        device = page_table.device
        # Prefill can exceed the decode ladder without replacing captured storage.
        separate = self._decode_rows_capacity is not None and (
            rows > self._decode_rows_capacity
            or columns != self._metadata_buffers["page_table"].shape[1]
        )

        def buffer(name, shape, dtype=torch.int32):
            if separate:
                return torch.empty(shape, dtype=dtype, device=device)
            return self._buffer(name, shape, device, dtype)

        self.page_table = buffer("page_table", (rows, columns))
        self.seq_lens = buffer("seq_lens", (rows,))
        self.init_counts = buffer("init_counts", (rows,))
        self.local_counts = buffer("local_counts", (rows,))
        self.query_ends = buffer("query_ends", (rows,))
        self.write_slots = (
            None if slots is None else buffer("write_slots", slots.shape, torch.int64)
        )
        self._query_group_size = query_group_size
        self._indexer_metadata = _AscendIndexerMetadata(
            buffer("index_q_ends", (groups,)),
            buffer("index_page_table", (groups, columns)),
            buffer("index_seq_lens", (groups,)),
            buffer("index_init_counts", (groups,)),
            self.local_counts,
            self.seq_lens if query_group_size > 1 else None,
        )
        index = self._indexer_metadata
        ascend_dsa_cp_metadata(
            page_table,
            seq_lens,
            page_size=page_size,
            block_granularity=block_granularity,
            virtual_block_count=self.virtual_block_count,
            degree=self.degree,
            rank=self.rank,
            initial_tokens=init_tokens,
            window_tokens=local_tokens,
            query_width=query_width,
            query_group_size=query_group_size,
            slots=slots,
            out=self.page_table,
            local_lengths=self.seq_lens,
            initial_counts=self.init_counts,
            window_counts=self.local_counts,
            query_ends=self.query_ends,
            index_table=index.page_table,
            index_lengths=index.seq_lens,
            index_initial_counts=index.init_counts,
            index_query_ends=index.q_ends,
            local_slots=self.write_slots,
        )

    def metadata(self, *, start: int, end: int) -> _AscendDCPKernelMetadata:
        values = (self.page_table, self.seq_lens, self.init_counts, self.local_counts)
        if any(value is None for value in values):
            raise RuntimeError("DSA CP metadata was not refreshed")
        page_table, seq_lens, init_counts, local_counts = values
        assert page_table is not None
        assert seq_lens is not None
        assert init_counts is not None
        assert local_counts is not None
        return _AscendDCPKernelMetadata(
            page_table=page_table[start:end],
            seq_lens=seq_lens[start:end],
            init_counts=init_counts[start:end],
            local_counts=local_counts[start:end],
        )

    def indexer_metadata(self, q_ends, local, *, query_group_size: int):
        """Return layouts already written by the metadata launch, without copies."""
        if (
            query_group_size != self._query_group_size
            or self._indexer_metadata is None
            or q_ends.numel() != local.seq_lens.numel()
        ):
            raise ValueError("Indexer query layout differs from prepared DSA metadata")
        return self._indexer_metadata


class AscendDSABackend(DSABackend):
    """Ascend DSA execution backed by Ascend indexer and sparse-attention ops."""

    # The decode path owns its DCP page placement and query exchange; the
    # generic MLA gate must accept this backend when attn_tp == dcp_size.
    supports_mla_dcp = True
    default_kernel_page_size = ASCEND_SFAD_PAGE_SIZE

    @classmethod
    def resolve_kernel_page_size(cls, config, block_granularity: int) -> int:
        del block_granularity
        if config.kernel_page_size not in (None, ASCEND_SFAD_PAGE_SIZE):
            raise ValueError(
                "Ascend SparseFlashAttentionDecode requires "
                f"kernel_page_size={ASCEND_SFAD_PAGE_SIZE}, got "
                f"{config.kernel_page_size}"
            )
        return ASCEND_SFAD_PAGE_SIZE

    @staticmethod
    def _make_dense_leaf(config, spec, platform, kernel_page_size):
        del platform
        dense_spec = dataclasses.replace(spec, backend_name=None)
        return MLAAttnBackend(config, dense_spec, kernel_page_size=kernel_page_size)

    def __init__(
        self,
        config,
        spec,
        *,
        kernel_page_size: int,
    ) -> None:
        if not (spec.is_dsa and spec.index_k_format == "bf16"):
            raise NotImplementedError("Ascend DSA currently requires BF16 index keys")
        if kernel_page_size != ASCEND_SFAD_PAGE_SIZE:
            raise ValueError(
                "Ascend SparseFlashAttentionDecode requires "
                f"kernel_page_size={ASCEND_SFAD_PAGE_SIZE}, got {kernel_page_size}"
            )
        if config.kv_cache_dtype != torch.bfloat16:
            raise ValueError("Ascend DSA requires BF16 KV cache")
        if config.kv_cache_quant_method not in (None, "none"):
            raise ValueError("Ascend DSA cache format is selected by kv_cache_dtype")
        dcp = _AscendDSAContextParallel(
            degree=int(config.kvp_size),
            rank=int(config.kvp_rank),
            ranks=tuple(config.kvp_group),
            virtual_block_count=None,
        )
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        if self.is_draft:
            raise ValueError("Ascend DSA draft attention is disabled")
        if self.spec_num_tokens > 8:
            raise ValueError("Ascend DSA verify supports at most 8 tokens")
        self._indexer_spec = spec
        self._indexer_kernels = ascend_dsa_kernels()
        self._fused_group: str | None = None
        self._fused_attention_outputs: dict[int, tuple] = {}
        self._fused_peer_inputs: dict[int, torch.Tensor] = {}
        self._ascend_placement: _AscendCachePlacement | None = None
        if dcp.degree > 1:
            if spec.attn_tp_size != dcp.degree:
                raise ValueError(
                    "DSA requires DCP to span the complete head-TP group, got "
                    f"dcp={dcp.degree}, head_tp={spec.attn_tp_size}"
                )
            if self.num_local_heads * dcp.degree != spec.num_attention_heads:
                raise ValueError("DSA DCP does not reconstruct all query heads")
            self._indexer_kernels.require_fused_context_parallel(
                rank_size=dcp.degree, num_heads=spec.num_attention_heads
            )
        self._dcp = dcp

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        self._ascend_placement = None
        self._dcp.clear_forward_metadata()
        super()._publish_cache_pool(cache_pool)
        if self._dcp.degree > 1:
            counts = cache_pool.arena.runtime_contract.virtual_block_counts
            if FULL_ATTENTION not in counts:
                raise ValueError("DSA CP requires a full-attention cache group")
            self._dcp.bind_virtual_block_count(counts[FULL_ATTENTION])

    def cache_placement(self, layer) -> CachePlacement | None:
        if self._ascend_placement is None:
            placement = super().cache_placement(layer)
            if placement is not None:
                self._ascend_placement = _AscendCachePlacement(
                    placement.block_granularity,
                    placement.virtual_block_count,
                    placement.group,
                    placement.rank,
                    self.prepare_prolog_metadata,
                )
        return self._ascend_placement

    def _clear_forward_scratch(self) -> None:
        self._dcp.clear_forward_metadata()

    def init_cuda_graph_state(self, max_bs: int) -> None:
        self._clear_forward_scratch()
        super().init_cuda_graph_state(max_bs)
        self._dcp.allocate_decode_buffers(
            max_bs * self.spec_num_tokens, self.max_num_pages, self.device
        )

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        self._clear_forward_scratch()
        self._dense_backend.refresh_decode_metadata(
            bs,
            actual_bs,
            seq_lens,
            page_table,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
        )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ):
        self._clear_forward_scratch()
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "DSA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        self._dense_backend.init_forward_metadata(
            bs,
            num_extends,
            seq_lens,
            page_table,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_with_prefix=extend_with_prefix,
            **kwargs,
        )
        self._prefill_page_table = None
        if num_extends > 0 and forward_mode.is_extend_or_mixed():
            cmeta = self._dense_backend.chunked_prefill_metadata
            if cmeta is not None:
                self._prefill_page_table = page_table[:num_extends]
                cmeta.page_table = self._prefill_page_table

    def prepare_prolog_metadata(self, locations, *, init_tokens, local_tokens):
        """Prepare slots and attention metadata once before the first Prolog."""
        metadata = self.forward_decode_metadata
        placement = self._ascend_placement
        assert placement is not None
        requests = metadata.seq_lens.shape[0] - metadata.num_extends
        if requests <= 0 or locations.numel() % requests:
            raise ValueError("DSA write slots must cover complete query groups")
        width = locations.numel() // requests
        self._dcp.prepare_metadata(
            seq_lens=metadata.seq_lens[metadata.num_extends :],
            page_table=metadata.page_table[metadata.num_extends :],
            page_size=self.kernel_page_size,
            block_granularity=placement.block_granularity,
            init_tokens=init_tokens,
            local_tokens=local_tokens,
            query_width=width,
            query_group_size=width,
            slots=locations,
        )
        slots = self._dcp.write_slots
        assert slots is not None
        return slots

    def configure_fused_decode(
        self,
        *,
        group: str,
        attention_outputs: dict[int, tuple],
        peer_inputs: dict[int, torch.Tensor],
    ) -> None:
        """Bind CP resources and stable producer outputs outside capture."""
        if self._dcp.degree <= 1 or not group:
            raise ValueError("DSA fusion requires a context-parallel communicator")
        if not attention_outputs or set(attention_outputs) != set(peer_inputs):
            raise ValueError("Fused DSA requires IPC-registered SFA outputs")
        if any(table.shape != (self._dcp.degree,) for table in peer_inputs.values()):
            raise ValueError("Fused DSA IPC tables must cover the CP group")
        self._fused_group = group
        self._fused_attention_outputs = attention_outputs
        self._fused_peer_inputs = peer_inputs

    def _select_indexed(
        self,
        q,
        k,
        layer,
        out_cache_loc,
        pool,
        q_ends,
        kv_lengths,
        table,
        kwargs,
        *,
        query_group_size: int,
        context_parallel,
        query_width,
    ):
        """Write the index cache and select request-local sparse indices.

        Query ends delimit the TND request spans; page identities and write
        slots belong to the router. The MLA prologue already wrote KV.
        """
        del k
        if kwargs.get("packed_query") is not None and self._fused_group is None:
            raise RuntimeError(
                "Initialize fused DSA resources before using packed queries"
            )
        spec = self._indexer_spec
        init_tokens = kwargs["index_init_tokens"]
        local_tokens = kwargs["index_local_tokens"]
        if init_tokens < 0 or local_tokens < 0:
            raise ValueError("DSA initial/local token counts must be nonnegative")
        if init_tokens + local_tokens > spec.index_topk:
            raise ValueError("DSA initial/local tokens exceed index_topk")
        kernels = self._indexer_kernels
        index_cache = pool.get_component(layer.layer_id, "dsa_index_k")
        page = self.kernel_page_size
        index_cache = index_cache.view(-1, page, 1, spec.index_head_dim)
        placement = self.cache_placement(layer)
        if context_parallel or q_ends is None:
            if self._dcp.degree > 1 and self._fused_group is None:
                raise RuntimeError(
                    "Initialize fused DSA resources before context-parallel execution"
                )
            # Without sharding, physical kernel pages are already local. Reuse
            # the same query expansion with identity ownership and slot mapping.
            if placement is None:
                if self._dcp.degree > 1:
                    raise RuntimeError("DSA CP cache placement was not configured")
                capacity = index_cache.shape[0]
                if self._dcp.virtual_block_count != capacity:
                    self._dcp.bind_virtual_block_count(capacity)
            # Model-owned selection parameters arrive with the indexer inputs.
            # Prepare precisely these query rows, including each verify candidate.
            self._dcp.prepare_metadata(
                seq_lens=kv_lengths,
                page_table=table,
                page_size=page,
                block_granularity=(
                    page if placement is None else placement.block_granularity
                ),
                init_tokens=init_tokens,
                local_tokens=local_tokens,
                query_width=query_width,
                query_group_size=query_group_size,
                slots=out_cache_loc,
            )
            local = self._dcp.metadata(start=0, end=table.shape[0] * query_width)
            q_ends = self._dcp.query_ends
            assert q_ends is not None
            index_slots = self._dcp.write_slots
            if self._dcp.degree == 1:
                kv_lengths, table = local.seq_lens, local.page_table
        else:
            if placement is None:
                index_slots = out_cache_loc
            else:
                local_slots, owned = resolve_cache_slots(out_cache_loc, placement)
                index_slots = torch.where(owned, local_slots, -1)
        if not kwargs.get("index_cache_written", False):
            kernels.scatter(kwargs["index_key"].contiguous(), index_cache, index_slots)
        if self.step_counter is not None:
            self.step_counter.record_cache()
        if not context_parallel or self._dcp.degree == 1:
            indices, valid_chunks = kernels.index(
                kwargs["index_query"],
                index_cache,
                kwargs["index_weights"],
                q_ends,
                kv_lengths,
                table,
                spec.index_topk,
                init_tokens,
                local_tokens,
            )
            return _AscendSparseSelection(
                q,
                indices,
                valid_chunks,
                q_ends,
                kv_lengths,
                table,
                3,
                False,
            )

        if self._fused_group is None:
            raise RuntimeError(
                "Initialize fused DSA resources before context-parallel execution"
            )
        # Keep each candidate's local causal end while sharing the request's
        # page table and K tiles across its verify queries.
        sparse_mode = 3
        index_metadata = self._dcp.indexer_metadata(
            q_ends, local, query_group_size=query_group_size
        )
        local_indices, local_values = kernels.index_partial(
            kwargs["index_query"],
            index_cache,
            kwargs["index_weights"],
            index_metadata.q_ends,
            index_metadata.seq_lens,
            index_metadata.page_table,
            spec.index_topk,
            index_metadata.init_counts,
            index_metadata.local_counts,
            sparse_mode=sparse_mode,
            visible_lengths=index_metadata.visible_lengths,
        )

        if local_values.shape[
            0
        ] not in self._fused_attention_outputs or local_values.shape[1:] != (
            1,
            2048,
        ):
            raise ValueError("Fused DSA TopK requires a prepared [T,1,2048] shape")
        packed_query = kwargs.get("packed_query")
        if packed_query is None:
            raise ValueError("Fused DSA requires the fused Prolog query")
        indices, valid_chunks = kernels.merge_topk_fused(
            local_indices,
            local_values,
            group=self._fused_group,
            rank_size=self._dcp.degree,
        )
        return _AscendSparseSelection(
            packed_query,
            indices,
            valid_chunks,
            q_ends,
            local.seq_lens,
            local.page_table,
            sparse_mode,
            True,
        )

    def _run_indexed_sparse_attention(
        self,
        selection,
        layer,
        token_to_kv_pool,
        *,
        head_major_output,
    ):
        spec = self._indexer_spec
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            1,
            spec.kv_cache_dim,
        )
        expected_width = spec.kv_lora_rank + spec.qk_rope_head_dim
        if selection.query.shape[-1] != expected_width:
            raise ValueError(
                f"Ascend DSA query width must be {expected_width}, got "
                f"{selection.query.shape[-1]}"
            )
        kernels = self._indexer_kernels
        if not selection.context_parallel:
            # TND accepts 32/64 query heads. TP can leave fewer Q heads on
            # each rank; repeat those heads locally and keep only the first
            # copy of the result. MLA's single latent KV head stays intact.
            live_heads = selection.query.shape[1]
            if live_heads < 1:
                raise ValueError("Ascend DSA needs at least one query head")
            if live_heads > 64:
                raise ValueError("Ascend DSA supports at most 64 local query heads")
            padded_heads = 32 if live_heads <= 32 else 64
            query = selection.query
            if live_heads != padded_heads:
                copies = (padded_heads + live_heads - 1) // live_heads
                query = query.repeat(1, copies, 1)[:, :padded_heads]
            result = kernels.attention(
                query[..., : spec.kv_lora_rank].contiguous(),
                query[..., spec.kv_lora_rank :].contiguous(),
                kv_cache,
                selection.indices,
                selection.valid_chunks,
                selection.q_ends,
                selection.kv_lengths,
                selection.table,
                layer.scaling,
            )
            result = result[:, :live_heads].contiguous()
            return (
                result.transpose(0, 1).contiguous()
                if head_major_output
                else result.flatten(1)
            )

        if self._fused_group is None:
            raise RuntimeError(
                "Initialize fused DSA resources before context-parallel execution"
            )
        rows = selection.query.shape[1]
        if (
            rows not in self._fused_attention_outputs
            or rows not in self._fused_peer_inputs
        ):
            raise ValueError("Fused DSA LSE requires a prepared [64,T,512] SFA shape")
        output, softmax_max, softmax_sum = kernels.attention_partial(
            selection.query,
            kv_cache,
            selection.indices,
            selection.valid_chunks,
            selection.q_ends,
            selection.kv_lengths,
            selection.table,
            layer.scaling,
            sparse_mode=selection.sparse_mode,
            out=self._fused_attention_outputs[rows],
        )
        if output.shape != (64, rows, 512):
            raise ValueError("Fused DSA LSE requires a prepared [64,T,512] SFA shape")
        result = kernels.merge_lse_direct(
            output,
            softmax_max,
            softmax_sum,
            self._fused_peer_inputs[rows],
            group=self._fused_group,
            rank_size=self._dcp.degree,
            setup_mode=0,
        )
        return (
            result
            if head_major_output
            else result.transpose(0, 1).contiguous().flatten(1)
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        metadata = self.forward_prefill_metadata
        head_major_output = kwargs.pop("head_major_output", False)
        q_ends = metadata.cum_extend_seq_lens[1:].to(torch.int32)
        kv_lengths = metadata.seq_lens.to(torch.int32)
        table = metadata.page_table
        context_parallel = self._dcp.degree > 1
        if context_parallel:
            # Compact history drops foreign pages. Translate each prefill
            # query's global causal endpoint, as verify does; local KV lengths
            # cannot be used to infer the global Q offsets.
            starts = metadata.cum_extend_seq_lens[:-1]
            requests = torch.repeat_interleave(
                torch.arange(bs, device=q.device),
                q_ends - starts,
                output_size=q.shape[0],
            )
            positions = torch.arange(q.shape[0], device=q.device)
            kv_lengths = (kv_lengths[requests] - q_ends[requests] + positions + 1).to(
                torch.int32
            )
            table = table[requests]
            q_ends = torch.arange(1, q.shape[0] + 1, device=q.device, dtype=torch.int32)
        selection = self._select_indexed(
            q,
            k,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            q_ends,
            kv_lengths,
            table,
            kwargs,
            query_group_size=1,
            context_parallel=context_parallel,
            query_width=1,
        )
        return self._run_indexed_sparse_attention(
            selection,
            layer,
            token_to_kv_pool,
            head_major_output=head_major_output,
        )

    def forward_extend_chunked(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Ascend DSA does not implement chunked prefix-replay prefill"
        )

    def forward_sparse_prefill(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Ascend DSA does not accept externally selected sparse prefill"
        )

    def forward_sparse_decode(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Ascend DSA does not accept externally selected sparse decode"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if bs <= 0 or q.shape[0] % bs:
            raise ValueError("Ascend DSA decode tokens must divide by requests")
        tokens_per_request = q.shape[0] // bs
        if tokens_per_request not in (1, self.spec_num_tokens):
            raise ValueError(
                "Ascend DSA decode requires one token or a complete verify "
                "window per request"
            )
        if topk_indices is not None or topk_lens is not None:
            raise ValueError(
                "Ascend DSA expects indexer projections, not global-slot TopK"
            )
        metadata = self.forward_decode_metadata
        start = metadata.num_extends
        head_major_output = kwargs.pop("head_major_output", False)
        query_lengths = metadata.seq_lens[start : start + bs]
        query_table = metadata.page_table[start : start + bs]
        selection = self._select_indexed(
            q,
            k,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            None,
            query_lengths,
            query_table,
            kwargs,
            query_group_size=tokens_per_request,
            context_parallel=self._dcp.degree > 1,
            query_width=tokens_per_request,
        )
        return self._run_indexed_sparse_attention(
            selection,
            layer,
            token_to_kv_pool,
            head_major_output=head_major_output,
        )


if current_platform().is_npu:
    register_backend(
        "dsa",
        {AttentionArch.DSA},
        AscendDSABackend,
    )
