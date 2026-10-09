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

"""QSA paged attention leaf: cache writes and sparse kernel dispatch."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.qsa import QSAPrefillMetadata, qsa_sparse_attention

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.qsa.metadata import (
    QSALayout,
    QSASelection,
    decode_query_lengths,
)
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


class QSAAttnBackend(MHAAttnBackend):
    """Sparse MHA leaf over the ordinary router's resolved pages and slots.

    MHA supplies the unified metadata path. The indexer owns cross-group
    indexing; the root backend's side state owns verify commits outside this leaf.
    """

    def __init__(
        self, config: AttnConfig, spec: MHAConfig, *, kernel_page_size: int
    ) -> None:
        super().__init__(
            config,
            dataclasses.replace(spec, backend_name="mha"),
            kernel_page_size=kernel_page_size,
        )
        self._metadata_capacity_rows = 0

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        self._metadata_capacity_rows = max_bs * self.spec_num_tokens

    def _sparse_attention(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool: CachePool,
        topk_indices: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        if self.is_mxfp8:
            raise NotImplementedError(
                "QSA sparse attention does not support MXFP8 KV cache"
            )
        num_real = current_valid_rows()
        if num_real is not None:
            q, topk_indices = slice_to_real_tokens(num_real, q, topk_indices)
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        max_seqlen_q = decode_query_lengths(
            ctx,
            q.shape[0],
            force_uniform=ctx.draft_narrowing is not None,
        )
        prefill_metadata = None
        if max_seqlen_q is None:
            selection = ctx.attn_backend.sparse_topk.prefill
            if selection is not None:
                layout = ctx.attn_backend.sparse_topk.qsa_metadata
                metadata = self.forward_extend_metadata
                if (
                    not isinstance(selection, QSASelection)
                    or not isinstance(layout, QSALayout)
                    or metadata is None
                ):
                    raise RuntimeError(
                        "QSA prefill requires the current indexer and extend metadata"
                    )
                # Graph breaks may copy physical slots into a stable handoff
                # buffer. The per-forward memo owns the logical selection.
                rows = q.shape[0]
                prefill_metadata = QSAPrefillMetadata(
                    selected_blocks=selection.selected_blocks[:rows],
                    block_table=layout.full_page_table,
                    token_to_request=layout.request_indices[:rows].to(torch.int32),
                    query_positions=layout.logical_positions[:rows],
                    query_start_loc=tuple(metadata.cu_extend_seq_lens_cpu),
                    page_size=layout.full_kernel_page_size,
                    block_size=selection.block_size,
                )
        output = qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            topk_indices,
            scale=layer.scaling,
            max_seqlen_q=max_seqlen_q,
            metadata_capacity_rows=max(q.shape[0], self._metadata_capacity_rows),
            k_scale=1.0 if k_cache.dtype == torch.float8_e4m3fn else None,
            v_scale=1.0 if v_cache.dtype == torch.float8_e4m3fn else None,
            override=None,
            solution=None,
            prefill_metadata=prefill_metadata,
        )
        return output.reshape(q.shape[0], -1)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        bs,
        *,
        # Both are required; explicit topk_indices=None selects dense MHA.
        topk_indices: torch.Tensor | None,
        ctx: ForwardContext,
        **kwargs,
    ):
        if topk_indices is None:
            return super().forward_decode(
                q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
            )
        return self._sparse_attention(q, layer, token_to_kv_pool, topk_indices, ctx)

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
        # Both are required; explicit topk_indices=None selects dense MHA.
        topk_indices: torch.Tensor | None,
        ctx: ForwardContext,
        **kwargs,
    ):
        if topk_indices is None:
            return super().forward_extend(
                q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
            )
        return self._sparse_attention(q, layer, token_to_kv_pool, topk_indices, ctx)


register_backend("qsa", {AttentionArch.MHA}, QSAAttnBackend)
