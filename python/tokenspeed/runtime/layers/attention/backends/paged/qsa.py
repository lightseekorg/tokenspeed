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
from tokenspeed_kernel.ops.attention.qsa import (
    QSASparseMetadata,
    QSAWorkspace,
    qsa_sparse_attention,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.support import TreeSupport
from tokenspeed.runtime.layers.attention.qsa.bf16_scratch import (
    QSABf16PrefillScratch,
    QSABf16ScratchPlan,
    QSAExtendLengths,
    extend_lengths,
)
from tokenspeed.runtime.layers.attention.qsa.metadata import (
    QSALayout,
    QSASelection,
    decode_query_lengths,
)
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan
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
        self._qsa_workspace = QSAWorkspace()
        # The target's BF16 copy of large FP8 prefill calls, allocated from
        # the plan the cache recipe reserved (preallocate_bf16_scratch).
        self._bf16_scratch: QSABf16PrefillScratch | None = None
        # Host lengths of the current forward while the scratch is in use.
        self._extend_lengths: QSAExtendLengths | None = None

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self._qsa_workspace.clear()
        self._extend_lengths = None

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        self._metadata_capacity_rows = max_bs * self.spec_num_tokens

    def preallocate_bf16_scratch(self, plan: QSABf16ScratchPlan) -> int:
        """Allocate the BF16 prefill scratch the cache recipe reserved.

        Every attention-component build hands the same plan to the same leaf,
        which allocates it once.

        Args:
            plan: The recipe's ``qsa_bf16_scratch_plan``.

        Returns:
            The bytes allocated, which the caller checks against the plan.
        """
        if self.is_draft:
            raise RuntimeError("the QSA BF16 prefill scratch serves the target only")
        if self.kv_cache_dtype is not torch.float8_e4m3fn or self.is_mxfp8:
            raise ValueError(
                "--enable-qsa-prefill-bf16-scratch needs an FP8 E4M3 KV cache"
            )
        planned = (plan.page_size, plan.kv_heads, plan.head_dim, plan.q_heads)
        served = (
            self.kernel_page_size,
            self.tp_kv_head_num,
            self.head_dim,
            self.tp_q_head_num,
        )
        if planned != served:
            raise RuntimeError(
                f"QSA BF16 scratch plan (page, KV heads, head dim, query heads) "
                f"{planned} does not match the attention leaf's {served}"
            )
        if self._bf16_scratch is None:
            self._bf16_scratch = QSABf16PrefillScratch(plan, device=self.device)
        elif self._bf16_scratch.plan != plan:
            raise RuntimeError("the QSA BF16 prefill scratch was planned differently")
        return self._bf16_scratch.nbytes

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
        query_shard: QueryShardPlan | None,
        page_table_cpu: torch.Tensor | None,
        **kwargs,
    ) -> None:
        super().init_forward_metadata(
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
            query_shard=query_shard,
            page_table_cpu=page_table_cpu,
            **kwargs,
        )
        # The scratch sizes a call from host mirrors, never a device read.
        self._extend_lengths = (
            extend_lengths(
                bs,
                num_extends,
                forward_mode.is_extend(),
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
            )
            if self._bf16_scratch is not None
            else None
        )

    def tree_support(self) -> TreeSupport:
        # The sparse layers do not read the tree mask; keep the base blockers
        # instead of the MHA tree cascade.
        return PagedAttentionBackend.tree_support(self)

    def reads_fp8_kv_scales(self, layer: PagedAttention) -> bool:
        # The sparse kernel reads an FP8 cache at unit scale.
        return False

    def query_dtype(
        self, layer: PagedAttention, forward_mode: ForwardMode, activation: torch.dtype
    ) -> torch.dtype:
        # The sparse kernel reads activation-dtype queries.
        return activation

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
        sparse_metadata = None
        selection = ctx.attn_backend.sparse_topk.qsa_selection
        if selection is not None:
            if not isinstance(selection, QSASelection):
                raise RuntimeError("invalid QSA logical selection memo")
            rows = q.shape[0]
            # Draft step zero may narrow target-aligned physical slots without
            # carrying their logical rows. Direct-slot kernels handle that case.
            if selection.selected_slots.shape[0] == rows:
                layout = ctx.attn_backend.sparse_topk.qsa_metadata
                if not isinstance(layout, QSALayout):
                    raise RuntimeError("QSA selection requires the current row layout")
                if (
                    self._bf16_scratch is not None
                    and max_seqlen_q is None
                    and ctx.draft_narrowing is None
                ):
                    # A large target extend attends a BF16 copy of its pages;
                    # None keeps the FP8 path below.
                    output = self._bf16_scratch.attention(
                        q,
                        k_cache,
                        v_cache,
                        topk_indices,
                        page_table=layout.full_page_table,
                        seq_lens=layout.seq_lens,
                        lengths=self._extend_lengths,
                        scale=layer.scaling,
                    )
                    if output is not None:
                        return output.reshape(rows, -1)
                if max_seqlen_q is None:
                    metadata = self.forward_extend_metadata
                    if metadata is None:
                        raise RuntimeError("QSA ragged queries require extend metadata")
                    offsets = tuple(metadata.cu_extend_seq_lens_cpu)
                else:
                    offsets = tuple(range(0, rows + 1, max_seqlen_q))
                # Graph breaks may copy physical slots into a stable buffer;
                # logical candidates stay in the same per-forward memo.
                sparse_metadata = QSASparseMetadata(
                    selected_blocks=selection.selected_blocks[:rows],
                    block_table=layout.full_page_table,
                    token_to_request=layout.request_indices[:rows].to(torch.int32),
                    query_positions=layout.logical_positions[:rows],
                    query_start_loc=offsets,
                    page_size=layout.full_kernel_page_size,
                    block_size=selection.block_size,
                    workspace=self._qsa_workspace,
                    seq_lens=layout.seq_lens,
                )
            elif ctx.draft_narrowing is None:
                raise RuntimeError("QSA selection must have one row per query")
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
            sparse_metadata=sparse_metadata,
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
