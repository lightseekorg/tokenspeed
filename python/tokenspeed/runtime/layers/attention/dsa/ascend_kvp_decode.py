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

"""DSA Decode over sharded KV: query/TopK exchange and partial-output merge."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.dsa.ascend import validate_dsa_context_parallel
from tokenspeed_kernel.ops.attention.dsa.paged import (
    select_paged_dsa_attention,
    select_paged_dsa_indexer,
)

from tokenspeed.runtime.layers.attention.dcp.ascend import (
    AscendDCPContext,
    AscendDCPMetadata,
)


@dataclass(frozen=True)
class _AscendSparseSelection:
    query: torch.Tensor
    indices: torch.Tensor
    valid_chunks: torch.Tensor
    q_ends: torch.Tensor
    kv_lengths: torch.Tensor
    table: torch.Tensor
    sparse_mode: int
    fused: bool


@dataclass(frozen=True)
class _FusedDecodeBinding:
    """Borrowed producer views; the model runtime owns their native registration."""

    group: str
    attention_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class AscendDSATPKVPDecode:
    def __init__(self, config, spec, *, page_size):
        self._indexer_spec = spec
        self.kernel_page_size = page_size
        self._dcp = AscendDCPMetadata(
            degree=config.kvp_size,
            rank=config.kvp_rank,
            ranks=tuple(config.kvp_group),
            virtual_block_count=None,
        )
        self._decode_indexer = select_paged_dsa_indexer()
        self._decode_attention = select_paged_dsa_attention()
        self._collective_context = None
        self._fused_decode = None
        self.placement = None

    def bind_cache(self, placement):
        self.placement = placement
        self._dcp.bind_virtual_block_count(placement.virtual_block_count)

    def refresh(self, metadata):
        self._dcp.clear_forward_metadata()

    def allocate_decode_buffers(self, rows, columns, device):
        self._dcp.allocate_decode_buffers(rows, columns, device)

    def prepare_prolog_metadata(
        self, locations, metadata, *, init_tokens, local_tokens
    ):
        start = metadata.num_extends
        width = locations.numel() // (metadata.seq_lens.numel() - start)
        self._prepare_decode_metadata(
            locations,
            metadata.seq_lens[start:],
            metadata.page_table[start:],
            width=width,
            initial=init_tokens,
            window=local_tokens,
        )
        return self._dcp.write_slots

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
        spec = self._indexer_spec
        for rows, outputs in attention_outputs.items():
            expected = (
                (spec.num_attention_heads, rows, spec.kv_lora_rank),
                (1, rows, spec.num_attention_heads),
                (1, rows, spec.num_attention_heads),
            )
            if len(outputs) != 3 or any(
                value.shape != shape or value.dtype != torch.float32
                for value, shape in zip(outputs, expected)
            ):
                raise ValueError(
                    "Fused DSA requires prepared FP32 SFA producer outputs"
                )
        validate_dsa_context_parallel(
            rank_size=self._dcp.degree, num_heads=spec.num_attention_heads
        )
        self._fused_decode = _FusedDecodeBinding(group, attention_outputs)

    def clear_fused_decode(self) -> None:
        """Drop borrowed views after the owner has drained/destroyed its graphs."""
        self._fused_decode = None

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
        indices, valid_chunks = self._decode_indexer.select_local_indices(
            local_indices, positions, self._dcp.rank
        )
        return packed_query, indices, valid_chunks

    def _prepare_decode_metadata(
        self, locations, lengths, table, *, width, initial, window
    ):
        """Expand causal query rows and resolve owner-local Index-K write slots.

        The fused prologue prepares this same metadata before writing both
        caches. Storage-view reuse makes subsequent layers share one result;
        eager and graph replay refresh through the same DCP metadata owner.
        """
        placement = self.placement
        page = self.kernel_page_size
        self._dcp.prepare_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=page,
            block_granularity=(placement.block_granularity),
            init_tokens=initial,
            local_tokens=window,
            query_width=width,
            query_group_size=width,
            slots=locations,
        )
        return self._dcp.metadata(start=0, end=table.shape[0] * width)

    def _select_decode_indices(
        self,
        query,
        index_query,
        index_cache,
        weights,
        local,
        *,
        initial,
        window,
        query_width,
        packed_query,
        fused: bool,
    ):
        """Select owner-local indices; exchange only queries and TopK candidates."""
        kernels = self._decode_indexer
        q_ends = self._dcp.query_ends
        # Verify candidates share page/K tiles but retain their causal ends.
        index_metadata = self._dcp.indexer_metadata(
            q_ends, local, query_group_size=query_width
        )
        local_indices, local_values = kernels.select_candidates(
            index_query,
            index_cache,
            weights,
            index_metadata.q_ends,
            index_metadata.seq_lens,
            index_metadata.page_table,
            self._indexer_spec.index_topk,
            index_metadata.init_counts,
            index_metadata.local_counts,
            sparse_mode=3,
            visible_lengths=index_metadata.visible_lengths,
        )
        if fused:
            if local_values.shape[1:] != (1, 2048):
                raise ValueError("Fused DSA TopK requires a prepared [T,1,2048] shape")
            indices, valid_chunks = kernels.merge_topk(
                local_indices,
                local_values,
                group=self._fused_decode.group,
                rank_size=self._dcp.degree,
            )
        else:
            packed_query, indices, valid_chunks = self._select_collective_topk(
                query, local_indices, local_values
            )
        return _AscendSparseSelection(
            packed_query,
            indices,
            valid_chunks,
            q_ends,
            local.seq_lens,
            local.page_table,
            3,
            fused,
        )

    def _run_collective_attention(self, selection, kv_cache, layer, kernels):
        output, maximum, total = kernels.attention_partial(
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
            output.transpose(0, 1), lse, merge=kernels.merge_partials
        )

    def _run_fused_attention(self, selection, kv_cache, scale, kernels):
        """Write registered SFA producers and return merged local heads [H,T,D]."""
        binding = self._fused_decode
        if binding is None:
            raise RuntimeError(
                "Initialize fused DSA resources before context-parallel execution"
            )
        rows = selection.query.shape[1]
        if rows not in binding.attention_outputs:
            raise ValueError("Fused DSA LSE requires prepared SFA producer outputs")
        output, maximum, total = kernels.attention_partial(
            selection.query,
            kv_cache,
            selection.indices,
            selection.valid_chunks,
            selection.q_ends,
            selection.kv_lengths,
            selection.table,
            scale,
            sparse_mode=selection.sparse_mode,
            out=binding.attention_outputs[rows],
        )
        return kernels.merge_lse_direct(
            output,
            maximum,
            total,
            group=binding.group,
            rank_size=self._dcp.degree,
        )

    def _run_indexed_sparse_attention(
        self,
        selection,
        layer,
        token_to_kv_pool,
        *,
        head_major_output,
    ):
        """Consume owner-local KV and return this rank's attention-head partition."""
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
        kernels = self._decode_attention
        if selection.fused:
            result = self._run_fused_attention(
                selection, kv_cache, layer.scaling, kernels
            )
            return (
                result
                if head_major_output
                else result.transpose(0, 1).contiguous().flatten(1)
            )
        result = self._run_collective_attention(selection, kv_cache, layer, kernels)
        return (
            result.transpose(0, 1).contiguous()
            if head_major_output
            else result.flatten(1)
        )

    def forward(
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
        tokens_per_request = q.shape[0] // bs
        initial, window = kwargs["index_init_tokens"], kwargs["index_local_tokens"]
        packed_query = kwargs.get("packed_query")
        if fused:
            if packed_query is None:
                raise ValueError("Fused DSA requires the fused Prolog query")
            if self._fused_decode is None:
                raise RuntimeError(
                    "Initialize fused DSA resources before using packed queries"
                )
            if q.shape[0] not in self._fused_decode.attention_outputs:
                raise ValueError("Fused DSA requires prepared SFA producer outputs")
        elif packed_query is not None:
            raise ValueError("Ordinary DSA does not accept a fused Prolog query")

        metadata = kwargs["metadata"]
        start = metadata.num_extends
        index_cache = token_to_kv_pool.get_component(
            layer.layer_id, "dsa_index_k"
        ).view(-1, self.kernel_page_size, 1, self._indexer_spec.index_head_dim)
        local = self._prepare_decode_metadata(
            out_cache_loc,
            metadata.seq_lens[start : start + bs],
            metadata.page_table[start : start + bs],
            width=tokens_per_request,
            initial=initial,
            window=window,
        )
        # Main KV was written by MLA prologue. Fused prologue also writes Index-K;
        # ordinary projections hand Index-K to this one owner-local store.
        if not kwargs.get("index_cache_written", False):
            self._decode_indexer.store_index_keys(
                kwargs["index_key"].contiguous(), index_cache, self._dcp.write_slots
            )
        selection = self._select_decode_indices(
            q,
            kwargs["index_query"],
            index_cache,
            kwargs["index_weights"],
            local,
            initial=initial,
            window=window,
            query_width=tokens_per_request,
            packed_query=packed_query,
            fused=fused,
        )
        step_counter = kwargs.get("step_counter")
        if step_counter is not None:
            step_counter.record_cache()
        return self._run_indexed_sparse_attention(
            selection,
            layer,
            token_to_kv_pool,
            head_major_output=kwargs.get("head_major_output", False),
        )
