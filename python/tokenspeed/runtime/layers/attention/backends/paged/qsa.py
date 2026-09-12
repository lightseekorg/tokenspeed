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
from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.qsa.metadata import decode_query_lengths
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

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        self._metadata_capacity_rows = max_bs * self.spec_num_tokens

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
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
            q, k, v, out_cache_loc, topk_indices = slice_to_real_tokens(
                num_real, q, k, v, out_cache_loc, topk_indices
            )
        full_locs = out_cache_loc[: k.shape[0]]
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if self.is_fp8 and (k.dtype == k_cache.dtype or v.dtype == v_cache.dtype):
            # Already-quantized inputs must not be scaled a second time.
            token_to_kv_pool.set_kv_buffer(
                layer,
                full_locs,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        else:
            self._save_kv_cache(layer, full_locs, token_to_kv_pool, k, v)
        max_seqlen_q = decode_query_lengths(
            ctx,
            q.shape[0],
            force_uniform=ctx.draft_narrowing is not None,
        )
        output = qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            topk_indices,
            scale=layer.scaling,
            max_seqlen_q=max_seqlen_q if max_seqlen_q is not None else 1,
            metadata_capacity_rows=max(q.shape[0], self._metadata_capacity_rows),
            k_scale=(
                (1.0 if layer.k_scale is None else layer.k_scale)
                if k_cache.dtype == torch.float8_e4m3fn
                else None
            ),
            v_scale=(
                (1.0 if layer.v_scale is None else layer.v_scale)
                if v_cache.dtype == torch.float8_e4m3fn
                else None
            ),
            override=None,
            solution=None,
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
        save_kv_cache: bool,
        *,
        # Both are required; explicit topk_indices=None selects dense MHA.
        topk_indices: torch.Tensor | None,
        ctx: ForwardContext,
        **kwargs,
    ):
        if topk_indices is None:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        assert save_kv_cache, "QSA sparse attention requires save_kv_cache=True"
        return self._sparse_attention(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, topk_indices, ctx
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
        save_kv_cache: bool,
        *,
        # Both are required; explicit topk_indices=None selects dense MHA.
        topk_indices: torch.Tensor | None,
        ctx: ForwardContext,
        **kwargs,
    ):
        if topk_indices is None:
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        assert save_kv_cache, "QSA sparse attention requires save_kv_cache=True"
        return self._sparse_attention(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, topk_indices, ctx
        )


register_backend("qsa", {AttentionArch.MHA}, QSAAttnBackend)
