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
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.dsa.ascend import ascend_dsa_kernels
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.dcp.ascend import AscendDCPContext
from tokenspeed.runtime.layers.attention.dcp.metadata import (
    CompactDCPLayout,
    CompactDCPMetadata,
    refresh_dcp_page_table_metadata,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.device_stream import (
    StreamFork,
    limit_stream_cores,
    new_device_stream,
)

# Fixed page geometry of Ascend SparseFlashAttentionDecode.
ASCEND_SFAD_PAGE_SIZE = 128


@dataclass(frozen=True)
class _AscendCachePlacement(CachePlacement):
    # Per-forward scratch, not persistent cache state. Keep the source alive
    # so an allocator cannot reuse its address during the same forward.
    write_slots: dict = dataclasses.field(default_factory=dict, compare=False)


def ascend_cache_write_slots(locations, placement):
    """Share one local-slot translation across MLA/indexer writes and layers.

    The Ascend leaf clears this scratch at each metadata refresh and pool
    rebind. During capture the first use records the translation; later
    layers reuse its output, which is recomputed by every graph replay.
    Only this Ascend ABI uses -1 for non-owner writes; the common placement
    helper retains its zero-address plus ownership-mask contract.
    """
    if placement is None:
        return locations
    key = (locations.data_ptr(), locations.shape, locations.stride(), locations.dtype)
    if isinstance(placement, _AscendCachePlacement) and key in placement.write_slots:
        return placement.write_slots[key][1]
    local, owned = resolve_cache_slots(locations, placement)
    local = torch.where(owned, local, -1)
    if isinstance(placement, _AscendCachePlacement):
        placement.write_slots[key] = (locations, local)
    return local


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


class _AscendDSAContextParallel(AscendDCPContext):
    """Ascend DSA metadata and collectives for one context-parallel group."""

    def __init__(
        self,
        *,
        degree: int,
        rank: int,
        ranks: tuple[int, ...],
        auxiliary_namespace: str,
        virtual_block_count: int | None,
    ) -> None:
        super().__init__(
            degree=degree,
            rank=rank,
            ranks=ranks,
            auxiliary_namespace=auxiliary_namespace,
        )
        self.virtual_block_count = virtual_block_count
        self.page_placement: CompactDCPMetadata | None = None
        self.page_table: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.init_counts: torch.Tensor | None = None
        self.local_counts: torch.Tensor | None = None
        self._metadata_buffers: dict[str, torch.Tensor] = {}
        self._decode_rows_capacity: int | None = None
        self._prepared_metadata_key: tuple | None = None
        self._metadata_sources: tuple[torch.Tensor, ...] = ()

    def clear_forward_metadata(self) -> None:
        """Invalidate layer-shared results before the next forward/capture."""
        self._prepared_metadata_key = None
        self._metadata_sources = ()

    def prepare_metadata(
        self,
        *,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        page_size: int,
        block_granularity: int,
        init_tokens: int,
        local_tokens: int,
    ) -> None:
        # New views of the same rows share results, but different row spans or
        # selection windows must refresh. Sources stay alive until invalidation.
        key = (
            tuple(
                (t.data_ptr(), t.shape, t.stride(), t.dtype, t.device)
                for t in (seq_lens, page_table)
            ),
            page_size,
            block_granularity,
            init_tokens,
            local_tokens,
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
        )
        self._metadata_sources = (seq_lens, page_table)
        self._prepared_metadata_key = key

    def bind_virtual_block_count(self, virtual_block_count: int) -> None:
        """Bind the scheduler capacity published by the cache arena."""
        if virtual_block_count <= 1:
            raise ValueError("DSA CP cache must contain usable virtual blocks")
        self.clear_forward_metadata()
        if self.virtual_block_count != virtual_block_count:
            self.virtual_block_count = virtual_block_count
            self.page_placement = None
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
                ("query_page_table", (rows, columns)),
                ("query_seq_lens", (rows,)),
                ("query_ends", (rows,)),
                ("page_table", (rows, columns)),
                ("seq_lens", (rows,)),
                ("init_counts", (rows,)),
                ("local_counts", (rows,)),
            )
        }
        self._decode_rows_capacity = rows

    def _copy_buffer(self, name: str, value: torch.Tensor) -> torch.Tensor:
        target = self._metadata_buffers.get(name)
        if (
            target is None
            or target.ndim != value.ndim
            or target.shape[1:] != value.shape[1:]
            or target.shape[0] < value.shape[0]
            or target.dtype != value.dtype
            or target.device != value.device
        ):
            if self._decode_rows_capacity is not None:
                raise RuntimeError(f"DSA {name} exceeds preallocated decode geometry")
            target = torch.empty_like(value)
            self._metadata_buffers[name] = target
        view = target[: value.shape[0]]
        view.copy_(value)
        return view

    def refresh_metadata(
        self,
        *,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        page_size: int,
        block_granularity: int,
        init_tokens: int,
        local_tokens: int,
    ) -> None:
        self.clear_forward_metadata()
        virtual_block_count = self.virtual_block_count
        if virtual_block_count is None:
            raise RuntimeError("DSA CP cache metadata was not bound")
        # Translate global selection windows through the common owned-page
        # prefix, including blocks containing multiple kernel pages.
        visible = torch.stack(
            (
                seq_lens.clamp_max(init_tokens),
                (seq_lens - local_tokens).clamp_min(0),
                seq_lens,
            ),
            dim=1,
        )
        previous = self.page_placement
        if previous is not None and previous.local_page_table.shape != page_table.shape:
            previous = None
        self.page_placement = refresh_dcp_page_table_metadata(
            page_table=page_table,
            virtual_block_count=virtual_block_count,
            degree=self.degree,
            rank=self.rank,
            layout=CompactDCPLayout(
                seq_lens=seq_lens,
                page_size=page_size,
                block_granularity=block_granularity,
                visible_lens=visible,
            ),
            previous=previous,
        )
        metadata = self.page_placement
        endpoints = metadata.local_visible_lens
        assert endpoints is not None
        if self._decode_rows_capacity is not None and (
            seq_lens.shape[0] > self._decode_rows_capacity
            or page_table.shape[1:] != self._metadata_buffers["page_table"].shape[1:]
        ):
            # Prefill can have a different shape from the decode graph ladder.
            # Use this call's common metadata without replacing decode buffers.
            self.page_table = metadata.local_page_table
            self.seq_lens = metadata.local_seq_lens
            self.init_counts = endpoints[:, 0].contiguous()
            self.local_counts = endpoints[:, 2] - endpoints[:, 1]
            return
        self.page_table = self._copy_buffer("page_table", metadata.local_page_table)
        self.seq_lens = self._copy_buffer("seq_lens", metadata.local_seq_lens)
        self.init_counts = self._copy_buffer("init_counts", endpoints[:, 0])
        self.local_counts = self._copy_buffer(
            "local_counts", endpoints[:, 2] - endpoints[:, 1]
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
            raise ValueError("Ascend DSA currently supports BF16 KV only")
        if config.kv_cache_quant_method not in (None, "none"):
            raise ValueError("Ascend DSA requires unquantized BF16 cache planes")
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        if self.is_draft:
            raise ValueError("Ascend DSA draft attention is disabled")
        if self.spec_num_tokens > 8:
            raise ValueError("Ascend DSA verify supports at most 8 tokens")
        self._indexer_spec = spec
        self._indexer_kernels = ascend_dsa_kernels()
        self._stream_fork = StreamFork(new_device_stream())
        self._decode_query_page_table: torch.Tensor | None = None
        self._decode_query_seq_lens: torch.Tensor | None = None
        self._decode_query_ends: torch.Tensor | None = None
        self._ascend_placement: _AscendCachePlacement | None = None
        dcp_size = int(config.kvp_size)
        if dcp_size > 1:
            self._indexer_kernels.require_context_parallel()
            if spec.attn_tp_size != dcp_size:
                raise ValueError(
                    "DSA requires DCP to span the complete head-TP group, got "
                    f"dcp={dcp_size}, head_tp={spec.attn_tp_size}"
                )
            if self.num_local_heads * dcp_size != spec.num_attention_heads:
                raise ValueError("DSA DCP does not reconstruct all query heads")
        self._dcp = _AscendDSAContextParallel(
            degree=dcp_size,
            rank=int(config.kvp_rank),
            ranks=tuple(config.kvp_group),
            auxiliary_namespace="dsa_cp_aux",
            virtual_block_count=None,
        )

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
                )
        return self._ascend_placement

    def _clear_forward_scratch(self) -> None:
        if self._ascend_placement is not None:
            self._ascend_placement.write_slots.clear()
        self._dcp.clear_forward_metadata()

    def init_cuda_graph_state(self, max_bs: int) -> None:
        self._clear_forward_scratch()
        super().init_cuda_graph_state(max_bs)
        self._decode_query_page_table = None
        self._decode_query_seq_lens = None
        self._decode_query_ends = None
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
        metadata = self.forward_decode_metadata
        width = self.spec_num_tokens
        final_lengths = metadata.seq_lens[:bs]
        offsets = torch.arange(
            1 - width, 1, dtype=final_lengths.dtype, device=final_lengths.device
        )
        query_lengths = final_lengths.unsqueeze(1).add(offsets).clamp_min(1).reshape(-1)
        query_table = metadata.page_table[:bs].repeat_interleave(width, dim=0)
        self._decode_query_seq_lens = self._dcp._copy_buffer(
            "query_seq_lens", query_lengths
        )
        self._decode_query_page_table = self._dcp._copy_buffer(
            "query_page_table", query_table
        )
        self._decode_query_ends = self._dcp._copy_buffer(
            "query_ends",
            torch.arange(1, bs * width + 1, dtype=torch.int32, device=seq_lens.device),
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
        self._decode_query_ends = torch.arange(
            1, bs * self.spec_num_tokens + 1, dtype=torch.int32, device=seq_lens.device
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
        context_parallel=False,
    ):
        """Write the index cache and select request-local sparse indices.

        Query ends delimit the TND request spans; page identities and write
        slots belong to the router. The MLA prologue already wrote KV.
        """
        del k
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
        index_slots = ascend_cache_write_slots(out_cache_loc, placement)
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

        if not self._dcp.ready:
            raise RuntimeError("Ascend DSA CP process group was not configured")
        # Model-owned selection parameters arrive with the indexer inputs.
        # Prepare precisely these query rows, including each verify candidate.
        self._dcp.prepare_metadata(
            seq_lens=kv_lengths,
            page_table=table,
            page_size=page,
            block_granularity=placement.block_granularity,
            init_tokens=init_tokens,
            local_tokens=local_tokens,
        )
        local = self._dcp.metadata(start=0, end=table.shape[0])
        # Each verify candidate is a one-row query with its own visible
        # length. Cyclic page ownership stays unchanged; sparse mode 3 scans
        # only that candidate's local prefix on each rank.
        sparse_mode = 3
        local_indices, local_values = kernels.index_partial(
            kwargs["index_query"],
            index_cache,
            kwargs["index_weights"],
            q_ends,
            local.seq_lens,
            local.page_table,
            spec.index_topk,
            local.init_counts,
            local.local_counts,
            sparse_mode=sparse_mode,
        )

        local_query = q.contiguous()
        tokens = local_values.shape[0]
        candidates = local_values.shape[2]
        degree = self._dcp.degree
        padded_tokens = (tokens + degree - 1) // degree * degree
        if padded_tokens != tokens:
            local_values = torch.cat(
                (
                    local_values,
                    torch.full(
                        (padded_tokens - tokens, 1, candidates),
                        float("-inf"),
                        dtype=local_values.dtype,
                        device=local_values.device,
                    ),
                )
            )
        tokens_per_rank = padded_tokens // degree
        received_values = torch.empty(
            degree * tokens_per_rank * candidates,
            dtype=local_values.dtype,
            device=local_values.device,
        )
        graph_phase = get_is_cuda_graph_phase()
        capture_mode = get_is_capture_mode()
        overlap_fork = self._stream_fork
        device_module = overlap_fork.device_module
        main_stream = (
            device_module.current_stream()
            if graph_phase and device_module is not None
            else None
        )

        def gather_query():
            return self._dcp.all_gather(local_query, auxiliary=True).view(
                degree, *local_query.shape
            )

        can_overlap_query = graph_phase and self._dcp.has_auxiliary
        if can_overlap_query:
            with (
                limit_stream_cores(
                    main_stream, cube_num=12, vector_num=24, enable=True
                ),
                limit_stream_cores(
                    overlap_fork.aux_stream,
                    cube_num=12,
                    vector_num=24,
                    enable=True,
                ),
                overlap_fork.scope(enable=True, overlap=capture_mode) as fork,
            ):
                with fork.branch():
                    packed_query = gather_query()
                self._dcp.all_to_all(
                    received_values,
                    local_values.squeeze(1).contiguous().view(-1),
                    auxiliary=False,
                )
        else:
            self._dcp.all_to_all(
                received_values,
                local_values.squeeze(1).contiguous().view(-1),
                auxiliary=False,
            )
            packed_query = gather_query()

        with limit_stream_cores(
            main_stream, cube_num=16, vector_num=32, enable=graph_phase
        ):
            global_values = (
                received_values.view(degree, tokens_per_rank, candidates)
                .transpose(0, 1)
                .contiguous()
                .view(tokens_per_rank, degree * candidates)
            )
            _, global_positions = global_values.topk(spec.index_topk, dim=1)
            global_positions = self._dcp.all_gather(
                global_positions.to(torch.int32).contiguous(),
                auxiliary=False,
            )[:tokens]
            indices, valid_chunks = kernels.select_local(
                local_indices,
                global_positions,
                self._dcp.rank,
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
        )
        tokens = selection.indices.shape[0]
        heads = spec.num_attention_heads
        degree = self._dcp.degree
        heads_per_rank = heads // degree
        graph_phase = get_is_cuda_graph_phase()
        capture_mode = get_is_capture_mode()
        overlap_fork = self._stream_fork
        device_module = overlap_fork.device_module
        main_stream = (
            device_module.current_stream()
            if graph_phase and device_module is not None
            else None
        )

        def update_output():
            output_send = output.view(
                degree, heads_per_rank, tokens, spec.kv_lora_rank
            ).contiguous()
            output_recv = torch.empty_like(output_send)
            self._dcp.all_to_all(
                output_recv,
                output_send,
                auxiliary=False,
            )
            return output_recv

        def update_lse():
            lse_send = (
                (softmax_max + torch.log(softmax_sum))
                .squeeze(0)
                .transpose(0, 1)
                .reshape(degree, heads_per_rank, tokens)
                .contiguous()
            )
            lse_recv = torch.empty_like(lse_send)
            self._dcp.all_to_all(
                lse_recv,
                lse_send,
                auxiliary=True,
            )
            return lse_recv

        if self._dcp.has_auxiliary:
            with (
                limit_stream_cores(
                    main_stream,
                    cube_num=12,
                    vector_num=24,
                    enable=graph_phase,
                ),
                limit_stream_cores(
                    overlap_fork.aux_stream,
                    cube_num=12,
                    vector_num=24,
                    enable=graph_phase,
                ),
                overlap_fork.scope(
                    enable=graph_phase,
                    overlap=capture_mode,
                ) as fork,
            ):
                with fork.branch():
                    lse_recv = update_lse()
                output_recv = update_output()
        else:
            output_recv = update_output()
            lse_recv = update_lse()

        result = kernels.merge_partials(
            output_recv.reshape(
                degree,
                heads_per_rank * tokens,
                spec.kv_lora_rank,
            ).contiguous(),
            lse_recv.reshape(degree, heads_per_rank * tokens).contiguous(),
        ).view(heads_per_rank, tokens, spec.kv_lora_rank)
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
            context_parallel=context_parallel,
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
        if tokens_per_request == self.spec_num_tokens:
            query_start = start * tokens_per_request
            query_end = query_start + q.shape[0]
            if (
                self._decode_query_seq_lens is None
                or self._decode_query_page_table is None
                or query_end > self._decode_query_seq_lens.shape[0]
            ):
                raise RuntimeError("DSA decode query metadata was not refreshed")
            query_lengths = self._decode_query_seq_lens[query_start:query_end]
            query_table = self._decode_query_page_table[query_start:query_end]
        else:
            # Mixed rounds may carry one decode row even when the configured
            # target verify width is larger.
            query_lengths = metadata.seq_lens[start : start + bs]
            query_table = metadata.page_table[start : start + bs]
        selection = self._select_indexed(
            q,
            k,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            self._decode_query_ends[: q.shape[0]],
            query_lengths,
            query_table,
            kwargs,
            context_parallel=True,
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
