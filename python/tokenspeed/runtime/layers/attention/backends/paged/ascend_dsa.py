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
from tokenspeed_kernel.ops.attention import merge_normalized_partials
from tokenspeed_kernel.ops.attention.ascend_prefill import (
    finish_prefill_workspace,
    gather_prefill_current,
    gather_prefill_history,
    require_dsa_prefill_ops,
    snapshot_prefill_write_slots,
    store_gathered_index_keys,
    validate_prefill_cache,
)
from tokenspeed_kernel.ops.attention.dsa import (
    dsa_gathered_prefill_attention,
    dsa_gathered_prefill_topk,
    dsa_local_topk,
    dsa_local_topk_candidates,
    dsa_select_local_topk,
    dsa_sparse_decode,
    dsa_sparse_decode_partial,
    merge_dsa_lse,
    merge_dsa_topk,
    store_index_keys,
)
from tokenspeed_kernel.ops.attention.dsa.ascend import validate_dsa_fused_decode
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.layers.attention.dcp.ascend import (
    AscendDCPContext,
    AscendDCPMetadata,
    AscendPrefillCache,
    DSAPrefillQueryMetadata,
    PrefillCacheMetadata,
    ascend_prefill_store,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import register_backend

# Fixed page geometry of Ascend SparseFlashAttentionDecode.
ASCEND_SFAD_PAGE_SIZE = 128


def ascend_dsa_prefill_backend(backend):
    """Find the DSA leaf without adding model hooks to shared router nodes."""
    if isinstance(backend, AscendDSABackend):
        return backend
    leaves = []
    pending = list(backend.child_backends())
    while pending:
        node = pending.pop()
        if isinstance(node, AscendDSABackend):
            leaves.append(node)
        else:
            pending.extend(node.child_backends())
    if len(leaves) != 1:
        raise ValueError("DSA Prefill prefetch needs exactly one DSA cache leaf")
    return leaves[0]


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
    fused: bool


class AscendDSABackend(PagedAttentionBackend):
    """Ascend DSA execution backed by Ascend indexer and sparse-attention ops."""

    # The decode path owns its DCP page placement and query exchange; the
    # generic MLA gate must accept this backend when attn_tp == dcp_size.
    supports_mla_dcp = True
    default_kernel_page_size = ASCEND_SFAD_PAGE_SIZE
    cuda_graph_support = CudaGraphSupport(prefill_graph=False)

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
        dcp = AscendDCPMetadata(
            degree=int(config.kvp_size),
            rank=int(config.kvp_rank),
            ranks=tuple(config.kvp_group),
            virtual_block_count=None,
        )
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size
        self._dense_backend = self._make_dense_leaf(
            config, spec, current_platform(), kernel_page_size
        )
        self._block_granularity: int | None = None
        self._virtual_block_count: int | None = None
        if self.is_draft:
            raise ValueError("Ascend DSA draft attention is disabled")
        if self.spec_num_tokens > 8:
            raise ValueError("Ascend DSA verify supports at most 8 tokens")
        self._indexer_spec = spec
        self._fused_group: str | None = None
        self._fused_attention_outputs: dict[int, tuple] = {}
        self._collective_context: AscendDCPContext | None = None
        self._ascend_placement: _AscendCachePlacement | None = None
        if dcp.degree > 1:
            if spec.attn_tp_size != dcp.degree:
                raise ValueError(
                    "DSA requires DCP to span the complete head-TP group, got "
                    f"dcp={dcp.degree}, head_tp={spec.attn_tp_size}"
                )
            if self.num_local_heads * dcp.degree != spec.num_attention_heads:
                raise ValueError("DSA DCP does not reconstruct all query heads")
        self._dcp = dcp
        self._prefill_cache_metadata: PrefillCacheMetadata | None = None
        self._prefill_query_metadata: DSAPrefillQueryMetadata | None = None
        self._prefill_workspace = None
        self._prefill_group = None
        self._prefill_options = None

    @property
    def prefill_enabled(self):
        return self._prefill_options is not None

    @property
    def prefill_cache_metadata(self) -> PrefillCacheMetadata:
        if self._prefill_cache_metadata is None:
            raise RuntimeError("Prepare native Prefill metadata before forward")
        return self._prefill_cache_metadata

    @property
    def prefill_needs_host_page_table(self):
        return self.prefill_enabled

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        self.reset_prefill_workspace()
        self._ascend_placement = None
        self._block_granularity = None
        self._virtual_block_count = None
        self._dcp.clear_forward_metadata()
        super()._publish_cache_pool(cache_pool)
        self._prefill_options = (
            cache_pool.prefill_options
            if isinstance(cache_pool, AscendPrefillCache)
            else None
        )
        self._prefill_cache_metadata = None
        self._prefill_query_metadata = None
        if self._dcp.degree > 1:
            counts = cache_pool.arena.runtime_contract.virtual_block_counts
            if FULL_ATTENTION not in counts:
                raise ValueError("DSA CP requires a full-attention cache group")
            self._dcp.bind_virtual_block_count(counts[FULL_ATTENTION])
        # Every world rank creates the cache-owner groups before any forward.
        if self.prefill_enabled and self._dcp.degree > 1:
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager,
            )

            self._prefill_group = process_group_manager.get_dedicated_device_group(
                self._dcp.ranks, "prefill-history"
            )
        else:
            self._prefill_group = None

        if self.prefill_enabled:
            layer_id = min(self.cache_pool.history_group_by_layer())
            validate_prefill_cache(
                self.cache_pool.get_key_buffer(layer_id),
                self.cache_pool.get_component(layer_id, "dsa_index_k"),
                ranks=self._dcp.ranks,
                history_group=self._prefill_group,
            )
            require_dsa_prefill_ops(torch.ops.custom)

    def prefetch_prefill_layer(self, layer_id):
        """Submit history at KDA, or ensure the same layer at attention entry."""
        plan = self.prefill_cache_metadata
        if not self.prefill_enabled:
            raise RuntimeError("Prepare DSA Prefill page metadata first")
        if sum(plan.projection_row_counts) > self._prefill_options.max_current_rows:
            raise ValueError("Prefill rows exceed the prepared token capacity")
        workspace = self._prefill_workspace
        if workspace is not None:
            if workspace.layer_id != layer_id or workspace.batch_key is not plan:
                raise RuntimeError("Prefill history from another batch/layer is active")
            return
        self._prefill_workspace = gather_prefill_history(
            layer_id,
            self.cache_pool.get_key_buffer(layer_id),
            self.cache_pool.get_component(layer_id, "dsa_index_k"),
            ranks=self._dcp.ranks,
            history_group=self._prefill_group,
            batch_key=plan,
            total_rows=sum(plan.projection_row_counts),
            history_null_rows=self._prefill_options.kv_page_size,
            has_history=plan.has_history,
            history_slots=plan.history_gather.send_slots,
            history_counts=plan.history_gather.row_counts,
        )

    def prepare_prefill_current(self, layer_id, latent, index_key, slots):
        self.prefetch_prefill_layer(layer_id)
        workspace = self._prefill_workspace
        gather_prefill_current(
            workspace,
            latent,
            index_key,
            counts=self.prefill_cache_metadata.projection_row_counts,
        )
        snapshot_prefill_write_slots(workspace, slots)

    def finish_prefill_layer(self, layer_id):
        workspace = self._prefill_workspace
        if workspace is None or workspace.layer_id != layer_id:
            raise RuntimeError("Prefill completion differs from its active layer")
        finish_prefill_workspace(workspace)
        self._prefill_workspace = None

    def reset_prefill_workspace(self) -> None:
        """Drop active-layer references after tracking their queued consumers."""
        if self._prefill_workspace is not None:
            finish_prefill_workspace(self._prefill_workspace)
        self._prefill_workspace = None

    @property
    def forward_decode_metadata(self):
        return self._dense_backend.forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._dense_backend.forward_prefill_metadata

    @property
    def chunked_prefill_metadata(self):
        return self._dense_backend.chunked_prefill_metadata

    @property
    def decode_seq_lens_buffer(self):
        return self._dense_backend.decode_seq_lens_buffer

    def child_backends(self):
        return (self._dense_backend,)

    def register_step_counter(self, step_counter):
        self.step_counter = step_counter
        self._dense_backend.step_counter = step_counter

    def override_num_extends(self, num_extends):
        return self._dense_backend.override_num_extends(num_extends)

    def configure_runtime(
        self, *, block_granularity, virtual_block_count, shard_count, **kwargs
    ):
        super().configure_runtime(**kwargs)
        if shard_count != self._dcp.degree or block_granularity % self.kernel_page_size:
            raise ValueError("DSA cache geometry does not match the KVP topology")
        self._block_granularity = block_granularity
        self._virtual_block_count = virtual_block_count
        self._ascend_placement = None

    def cache_placement(self, layer) -> CachePlacement | None:
        if self._dcp.degree == 1:
            return None
        if self._ascend_placement is None:
            if self._block_granularity is None or self._virtual_block_count is None:
                raise RuntimeError("DSA KVP cache geometry is not configured")
            self._ascend_placement = _AscendCachePlacement(
                self._block_granularity,
                self._virtual_block_count,
                self._dcp.ranks,
                self._dcp.rank,
                self.prepare_prolog_metadata,
            )
        return self._ascend_placement

    @staticmethod
    def _validate_logit_cap(value):
        if value and value > 0:
            raise NotImplementedError("Ascend DSA does not support logit caps")

    def _clear_forward_scratch(self) -> None:
        self._dcp.clear_forward_metadata()
        self._prefill_cache_metadata = None
        self._prefill_query_metadata = None

    def init_cuda_graph_state(self, max_bs: int) -> None:
        self._clear_forward_scratch()
        self._dense_backend.init_cuda_graph_state(max_bs)
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
        self._prefill_cache_metadata = None
        self._prefill_query_metadata = None
        if num_extends > 0 and self.prefill_enabled:
            layer_id = min(self.cache_pool.history_group_by_layer())
            local_rows = self.cache_pool.get_key_buffer(layer_id).numel() // 576
            self._prefill_cache_metadata = PrefillCacheMetadata.build(
                self.forward_prefill_metadata,
                self.cache_placement(None),
                degree=self._dcp.degree,
                rank=self._dcp.rank,
                page_size=self.kernel_page_size,
                local_cache_pages=local_rows // self.kernel_page_size,
                page_table_cpu=kwargs.get("page_table_cpu"),
                options=self._prefill_options,
            )
            self._prefill_query_metadata = DSAPrefillQueryMetadata.build(
                self.forward_prefill_metadata,
                self._prefill_cache_metadata,
                topk=self._indexer_spec.index_topk,
            )

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
        ranks: tuple[int, ...],
        attention_outputs: dict[int, tuple],
    ) -> None:
        """Bind CP resources and stable producer outputs outside capture."""
        if self._dcp.degree <= 1 or not group:
            raise ValueError("DSA fusion requires a context-parallel communicator")
        if tuple(ranks) != tuple(self._dcp.ranks):
            raise ValueError("Fused DSA resources must cover the CP group ranks")
        if not attention_outputs:
            raise ValueError("Fused DSA requires prepared SFA producer outputs")
        validate_dsa_fused_decode(
            rank_size=self._dcp.degree,
            num_heads=self._indexer_spec.num_attention_heads,
        )
        self._fused_group = group
        self._fused_attention_outputs = attention_outputs

    def _context_collectives(self) -> AscendDCPContext:
        """Use the existing device group for shapes without native fusion."""
        if self._collective_context is None:
            self._collective_context = AscendDCPContext(
                degree=self._dcp.degree,
                rank=self._dcp.rank,
                ranks=self._dcp.ranks,
                auxiliary_namespace=None,
            )
        if not self._collective_context.ready:
            raise RuntimeError("Ascend DSA CP process group was not configured")
        return self._collective_context

    def _select_collective_topk(self, q, local_indices, local_values):
        collective = self._context_collectives()
        degree = self._dcp.degree
        tokens, _, candidates = local_values.shape
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
        rows = padded_tokens // degree
        received = torch.empty(
            degree * rows * candidates,
            dtype=local_values.dtype,
            device=local_values.device,
        )
        collective.all_to_all(
            received, local_values.squeeze(1).contiguous().view(-1), auxiliary=False
        )
        packed_query = collective.all_gather(q.contiguous(), auxiliary=False).view(
            degree, *q.shape
        )
        values = received.view(degree, rows, candidates).transpose(0, 1)
        _, positions = values.reshape(rows, degree * candidates).topk(
            self._indexer_spec.index_topk, dim=1
        )
        positions = collective.all_gather(
            positions.to(torch.int32).contiguous(), auxiliary=False
        )[:tokens]
        indices, valid_chunks = dsa_select_local_topk(
            local_indices, positions, self._dcp.rank
        )
        return packed_query, indices, valid_chunks

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
        fused: bool,
        query_group_size: int,
        context_parallel,
        query_width,
    ):
        """Write the index cache and select request-local sparse indices.

        Query ends delimit the TND request spans; page identities and write
        slots belong to the router. The MLA prologue already wrote KV.
        """
        del k
        packed_query = kwargs.get("packed_query")
        if fused and packed_query is None:
            raise ValueError("Fused DSA requires the fused Prolog query")
        if not fused and packed_query is not None:
            raise ValueError("Ordinary DSA does not accept a fused Prolog query")
        if fused and self._fused_group is None:
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
        index_cache = pool.get_component(layer.layer_id, "dsa_index_k")
        page = self.kernel_page_size
        index_cache = index_cache.view(-1, page, 1, spec.index_head_dim)
        placement = self.cache_placement(layer)
        if context_parallel or q_ends is None:
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
            store_index_keys(kwargs["index_key"].contiguous(), index_cache, index_slots)
        if not context_parallel or self._dcp.degree == 1:
            indices, valid_chunks = dsa_local_topk(
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
                False,
            )

        # Keep each candidate's local causal end while sharing the request's
        # page table and K tiles across its verify queries.
        sparse_mode = 3
        index_metadata = self._dcp.indexer_metadata(
            q_ends, local, query_group_size=query_group_size
        )
        local_indices, local_values = dsa_local_topk_candidates(
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

        if not fused:
            packed_query, indices, valid_chunks = self._select_collective_topk(
                q, local_indices, local_values
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
                False,
            )

        if local_values.shape[
            0
        ] not in self._fused_attention_outputs or local_values.shape[1:] != (
            1,
            2048,
        ):
            raise ValueError("Fused DSA TopK requires a prepared [T,1,2048] shape")
        indices, valid_chunks = merge_dsa_topk(
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
            True,
        )

    def _run_collective_attention(self, selection, kv_cache, layer):
        output, maximum, total = dsa_sparse_decode_partial(
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
        lse = (maximum + total.log()).squeeze(0)
        return self._context_collectives().combine(
            output.transpose(0, 1), lse, merge=merge_normalized_partials
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
            result = dsa_sparse_decode(
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

        if not selection.fused:
            result = self._run_collective_attention(selection, kv_cache, layer)
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
        if rows not in self._fused_attention_outputs:
            raise ValueError("Fused DSA LSE requires a prepared [64,T,512] SFA shape")
        output, softmax_max, softmax_sum = dsa_sparse_decode_partial(
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
        result = merge_dsa_lse(
            output,
            softmax_max,
            softmax_sum,
            group=self._fused_group,
            rank_size=self._dcp.degree,
        )
        return (
            result
            if head_major_output
            else result.transpose(0, 1).contiguous().flatten(1)
        )

    def prefill_store(self, layer):
        return ascend_prefill_store(
            self._prefill_workspace,
            self.prefill_cache_metadata,
            self.cache_placement(layer),
            kernel_page_size=self.kernel_page_size,
        )

    def select_prefill_indices(self, query, weights, layer, *, initial, window):
        spec = self._indexer_spec
        if initial < 0 or window < 0 or initial + window > spec.index_topk:
            raise ValueError("Prefill init/local token counts must fit index_topk")
        plan, workspace = self.prefill_cache_metadata, self._prefill_workspace
        placement = self.cache_placement(layer)
        local_rows = workspace.source_kv.numel() // spec.kv_cache_dim
        store_gathered_index_keys(
            workspace,
            plan.current_direct,
            plan.patch_slots,
            block_rows=(
                self.kernel_page_size
                if placement is None
                else placement.block_granularity
            ),
            virtual_blocks=(
                local_rows // self.kernel_page_size
                if placement is None
                else placement.virtual_block_count
            ),
        )
        if not query.shape[0]:
            return torch.empty(
                (0, spec.index_topk), dtype=torch.int32, device=query.device
            )
        return dsa_gathered_prefill_topk(
            query,
            workspace.index.view(-1, self.kernel_page_size, 1, spec.index_head_dim),
            weights,
            self._prefill_query_metadata.query_cu,
            self._prefill_query_metadata.key_cu,
            plan.page_table,
            spec.index_topk,
            initial,
            window,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        bs,
        *,
        fused: bool,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if fused:
            raise ValueError("Native fused decode queries are not Prefill inputs")
        plan, workspace = self.prefill_cache_metadata, self._prefill_workspace
        if workspace is None or workspace.layer_id != layer.layer_id:
            raise RuntimeError("Prepare Prefill current rows before attention")
        spec = self._indexer_spec
        if q.shape[:2] != (
            plan.projection_row_counts[self._dcp.rank],
            spec.num_attention_heads,
        ):
            raise ValueError(
                "Prefill needs token-local queries with all attention heads"
            )
        if self.step_counter is not None:
            self.step_counter.record_cache()
        if q.shape[0]:
            output = dsa_gathered_prefill_attention(
                q,
                workspace.kv.view(-1, self.kernel_page_size, spec.kv_cache_dim),
                kwargs["prefill_indices"],
                self._prefill_query_metadata.scan_count,
                plan.page_table,
                self._prefill_query_metadata.query_seq,
                layer.scaling,
            )
        else:
            output = q.new_empty((q.shape[1], 0, spec.kv_lora_rank))
        self.finish_prefill_layer(layer.layer_id)
        return (
            output
            if kwargs.get("head_major_output", False)
            else output.transpose(0, 1).contiguous().flatten(1)
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
        *,
        fused: bool,
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
            fused=fused,
            query_group_size=tokens_per_request,
            context_parallel=self._dcp.degree > 1,
            query_width=tokens_per_request,
        )
        if self.step_counter is not None:
            self.step_counter.record_cache()
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
