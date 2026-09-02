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

"""MiniMax sparse attention leaf and the dense/sparse hybrid composite."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    msa_decode_with_kvcache,
    msa_extend_with_kvcache,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.mha import trim_kv_to_locs
from tokenspeed.runtime.layers.attention.backends.paged import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
)
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    MSA_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.registry import (
    register_backend,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MSAExtendMetadata:
    # Device-side metadata:
    # - seq_lens: total length after this step
    # - extend_seq_lens: length of new tokens
    #   cu_extend_seq_lens: the cumsum version of extend_seq_lens
    #   cu_seqlens_kv: the cumsum version of seq_lens
    # - extend_prefix_lens: length of the cached prefix tokens
    # seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_extend_seq_lens: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens_cpu: list[int]
    cu_extend_seq_lens_cpu: list[int]
    # Per-request total lengths (prefix + new tokens) on the host, so kernels
    # can plan host-side without a device sync.
    seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int = 0


@dataclass(kw_only=True)
class MSADecodeMetadata:
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    # Per-forward view of the leaf's shared decode score buffer, pre-filled
    # with -inf and reused by every sparse layer.
    score_out: torch.Tensor | None = None


class MSAAttnBackend(PagedAttentionBackend):
    """MiniMax sparse attention leaf routed through tokenspeed_kernel APIs."""

    default_kernel_page_size = MSA_PAGE_SIZE

    def __init__(self, config: AttnConfig, spec: MSAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)

        self.tp_q_head_num = max(spec.num_attention_heads // spec.attn_tp_size, 1)
        self.tp_kv_head_num = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_fp8 = self.kv_cache_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )

        # Sparse attention parameters
        self.sparse_layer_ids = spec.sparse_layer_ids
        self.index_head_dim = spec.index_head_dim
        self.index_topk_blocks = spec.index_topk_blocks
        self.index_init_blocks = spec.index_init_blocks
        self.index_local_blocks = spec.index_local_blocks

        # DFLASH draft: expand decode metadata to spec_num_tokens rows/request
        # (whole block in one decode forward), with uniform non-causal seq_lens.
        self.draft_block_decode = bool(config.draft_block_decode)

        self.forward_decode_metadata: MSADecodeMetadata | None = None
        self.forward_extend_metadata: MSAExtendMetadata | None = None
        self.page_table_buf: torch.Tensor | None = None
        self.seq_lens_buf: torch.Tensor | None = None
        self._decode_views_by_bs: dict[int, MSADecodeMetadata] = {}

        # Persistent decode index-score buffer, shared across sparse layers so
        # the indexer's -inf tail is reset once per forward instead of a
        # per-layer torch.full. Full page width == max_blocks for decode
        # (max_seqlen_k == context_len).
        self.decode_score_buffer = torch.empty(
            (
                config.max_bs * self.tokens_per_req,
                self.tp_kv_head_num,
                self.max_num_pages,
            ),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def tokens_per_req(self) -> int:
        return 1 if self.is_draft else self.spec_num_tokens

    # ------------------------------------------------------------------
    # Metadata initialization
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_mixed(), "MSA backend does not support mixed batch"
        if not forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                "MSA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend ({forward_mode})"
            )
        assert extend_seq_lens is not None
        assert extend_seq_lens_cpu is not None
        assert extend_prefix_lens is not None
        assert extend_prefix_lens_cpu is not None

        seq_lens = seq_lens[:bs]

        # Create cumulative sum of the sequence lengths for Q and KV.
        extend_seq_lens = extend_seq_lens[:bs]
        extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:bs].tolist()]
        cu_extend_seq_lens = torch.nn.functional.pad(
            torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        cu_extend_seq_lens_cpu = [0]
        for length in extend_seq_lens_cpu:
            cu_extend_seq_lens_cpu.append(cu_extend_seq_lens_cpu[-1] + length)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        extend_prefix_lens = extend_prefix_lens[:bs]
        max_extend_seq_len = max(extend_seq_lens_cpu)
        prefix_lens_cpu = [int(x) for x in extend_prefix_lens_cpu[:bs].tolist()]
        max_extend_prefix_len = max(prefix_lens_cpu)
        seq_lens_cpu = [p + e for p, e in zip(prefix_lens_cpu, extend_seq_lens_cpu)]

        self.forward_extend_metadata = MSAExtendMetadata(
            page_table=page_table[:bs],
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            cu_extend_seq_lens=cu_extend_seq_lens,
            cu_seqlens_kv=cu_seqlens_kv,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
        )

    def init_cuda_graph_state(self, max_bs: int) -> None:
        rows = max_bs * (self.spec_num_tokens if self.block_decode_active else 1)
        self.page_table_buf = torch.zeros(
            (rows, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        # Own the cache-seqlens buffer; under block decode the drafter fills
        # it in-graph, so seed a valid baseline.
        self.seq_lens_buf = torch.full(
            (rows,),
            self.spec_num_tokens if self.block_decode_active else 0,
            dtype=torch.int32,
            device=self.device,
        )
        self._decode_views_by_bs = {}

    @property
    def decode_seq_lens_buffer(self) -> torch.Tensor:
        return self.seq_lens_buf

    def _decode_views(self, bs: int) -> MSADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers; one
        builder for capture and refresh, cached per bs — pointer-stable."""
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is None:
            if self.block_decode_active:
                rows = bs * self.spec_num_tokens
                metadata = MSADecodeMetadata(
                    page_table=self.page_table_buf[:rows],
                    seq_lens=self.seq_lens_buf[:rows],
                )
            else:
                metadata = MSADecodeMetadata(
                    page_table=self.page_table_buf[:bs],
                    seq_lens=self.seq_lens_buf[:bs],
                    score_out=self.decode_score_buffer[: bs * self.tokens_per_req],
                )
            self._decode_views_by_bs[bs] = metadata
        return metadata

    # Capture is inherited (the leaf default: idle refresh over the same
    # buffers). It relies on the runner seeding capture seq_lens >= the
    # verify floor: the refresh below copies them without a clamp.

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
        del num_extends
        if self.block_decode_active:
            # DFLASH draft: replicate each request's page table to its
            # spec_num_tokens block rows. Under replay the block-end seq_lens
            # are re-derived inside the captured graph
            # (fill_block_decode_seq_lens); eager fills them here.
            spec = self.spec_num_tokens
            self.page_table_buf[: bs * spec].view(bs, spec, self.max_num_pages).copy_(
                page_table[:bs, None, :]
            )
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
            self.forward_decode_metadata = self._decode_views(bs)
            return

        self.seq_lens_buf[:bs].copy_(seq_lens[:bs])
        self.page_table_buf[:bs].copy_(page_table[:bs])
        self.forward_decode_metadata = self._decode_views(bs)
        # Reset the shared score buffer to -inf before the forward; the
        # score kernels overwrite only visible blocks, leaving the tail.
        self.forward_decode_metadata.score_out.fill_(-float("inf"))

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        index_q: torch.Tensor | None = None,
        index_k: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run sparse decode and update the standard and index-key caches."""
        del bs, kwargs
        metadata = self.forward_decode_metadata
        assert (
            metadata is not None
        ), "MSA decode requires initialized paged-KV metadata."
        assert (
            index_q is not None and index_k is not None
        ), "MSA requires index_q and index_k from the model layer."
        assert save_kv_cache, (
            "MSA does not support KV-cache prewrite because its "
            "index-key side cache is backend-owned."
        )
        assert k is not None and v is not None, "MSA requires K/V inputs on every call."
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        k_cache, v_cache, index_k_cache = self._get_sparse_caches(
            layer, token_to_kv_pool
        )

        num_requests = metadata.seq_lens.shape[0]
        if num_requests == 0 or q.shape[0] % num_requests:
            raise RuntimeError("MSA decode requires a uniform query count per request.")
        decode_query_len = q.shape[0] // num_requests
        output = msa_decode_with_kvcache(
            q=q,
            index_q=index_q,
            index_k=index_k,
            k_cache=k_cache,
            v_cache=v_cache,
            index_k_cache=index_k_cache,
            slot_mapping=out_cache_loc,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            topk=self.index_topk_blocks,
            page_size=self.kernel_page_size,
            index_scale=self.index_head_dim**-0.5,
            attention_scale=layer.scaling,
            init_blocks=self.index_init_blocks,
            local_blocks=self.index_local_blocks,
            max_seqlen_q=decode_query_len,
            max_seqlen_k=self.max_context_len,
            k_scale=layer.k_scale if self.is_fp8 else None,
            v_scale=layer.v_scale if self.is_fp8 else None,
            score_out=metadata.score_out,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        index_q: torch.Tensor | None = None,
        index_k: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run sparse extend/prefill and update both cache components."""
        del bs, kwargs
        metadata = self.forward_extend_metadata
        assert (
            metadata is not None
        ), "MSA prefill requires initialized paged-KV metadata."
        assert (
            index_q is not None and index_k is not None
        ), "MSA requires index_q and index_k from the model layer."
        assert save_kv_cache, (
            "MSA does not support KV-cache prewrite because its "
            "index-key side cache is backend-owned."
        )
        assert k is not None and v is not None, "MSA requires K/V inputs on every call."
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        total_tokens = q.shape[0]
        real_tokens = int(metadata.cu_extend_seq_lens_cpu[-1])
        q = q[:real_tokens]
        k = k[:real_tokens]
        v = v[:real_tokens]
        index_q = index_q[:real_tokens]
        index_k = index_k[:real_tokens]

        out_cache_loc = out_cache_loc[:real_tokens]
        self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        k_cache, v_cache, index_k_cache = self._get_sparse_caches(
            layer, token_to_kv_pool
        )

        max_seq_len = metadata.max_extend_prefix_len + metadata.max_extend_seq_len
        output = msa_extend_with_kvcache(
            q=q,
            index_q=index_q,
            index_k=index_k,
            k_cache=k_cache,
            v_cache=v_cache,
            index_k_cache=index_k_cache,
            slot_mapping=out_cache_loc,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            prefix_lens=metadata.extend_prefix_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=max_seq_len,
            topk=self.index_topk_blocks,
            page_size=self.kernel_page_size,
            index_scale=self.index_head_dim**-0.5,
            attention_scale=layer.scaling,
            init_blocks=self.index_init_blocks,
            local_blocks=self.index_local_blocks,
            k_scale=layer.k_scale if self.is_fp8 else None,
            v_scale=layer.v_scale if self.is_fp8 else None,
            query_lens_cpu=metadata.extend_seq_lens_cpu,
            seq_lens_cpu=metadata.seq_lens_cpu,
        )
        return self._reshape_and_pad_output(output, total_tokens, layer)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _save_kv_cache(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
    ) -> None:
        if k is None:
            return
        k, v = trim_kv_to_locs(out_cache_loc, k, v)

        if (
            self.kv_cache_dtype == torch.float8_e4m3fn
            and k.dtype != torch.float8_e4m3fn
        ):
            k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            fused_fp8_set_kv_buffer(
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self.kernel_page_size,
            )
        else:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

    def _get_kv_cache(self, layer: PagedAttention, token_to_kv_pool):
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )
        return k_cache, v_cache

    def _get_sparse_caches(self, layer: PagedAttention, token_to_kv_pool):
        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        k_cache = k_cache.permute(0, 2, 1, 3)
        v_cache = v_cache.permute(0, 2, 1, 3)
        return k_cache, v_cache, token_to_kv_pool.get_index_k_buffer(layer.layer_id)

    @staticmethod
    def _reshape_and_pad_output(
        output: torch.Tensor,
        total_tokens: int,
        layer: PagedAttention,
    ) -> torch.Tensor:
        output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        if output.shape[0] == total_tokens:
            return output
        padded = output.new_zeros((total_tokens, output.shape[1]))
        padded[: output.shape[0]].copy_(output)
        return padded


class MSAHybridAttnBackend(AttentionBackend):
    """MiniMax hybrid composite: one router of dense leaves and one of sparse
    leaves over the same history groups, dispatched per layer id."""

    def __init__(self, config: AttnConfig, spec: MSAConfig) -> None:
        from tokenspeed.runtime.layers.attention.registry import (
            create_paged_router,
        )

        super().__init__(config, spec)
        self.full_router = create_paged_router(
            config,
            AttentionArch.MHA,
            backend_name=spec.full_attn_backend_name,
        )
        self.sparse_router = create_paged_router(
            config,
            AttentionArch.MSA,
            backend_name="msa_leaf",
        )
        self.sparse_layer_ids = spec.sparse_layer_ids
        logger.info(
            "Created MiniMax hybrid attention backend: %d dense layers, "
            "%d sparse layers",
            len(spec.compute_layer_types) - len(spec.sparse_layer_ids),
            len(spec.sparse_layer_ids),
        )

    def _router_for_layer(self, layer_id: int):
        if layer_id in self.sparse_layer_ids:
            return self.sparse_router
        return self.full_router

    def child_backends(self) -> tuple[AttentionBackend, ...]:
        return (self.full_router, self.sparse_router)

    def set_cache_pool(self, cache_pool) -> None:
        self.cache_pool = cache_pool
        self.full_router.set_cache_pool(cache_pool)
        self.sparse_router.set_cache_pool(cache_pool)

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        # A single model-wide answer must be safe for sparse layers too.
        del forward_mode
        return False

    def write_locations(self, layer, forward_mode):
        return self._router_for_layer(layer.layer_id).write_locations(
            layer, forward_mode
        )

    def init_forward_metadata(self, *args, **kwargs):
        self.full_router.init_forward_metadata(*args, **kwargs)
        self.sparse_router.init_forward_metadata(*args, **kwargs)

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        self.full_router.init_cuda_graph_state(max_bs, **kwargs)
        self.sparse_router.init_cuda_graph_state(max_bs, **kwargs)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self.full_router.init_forward_metadata_capture_cuda_graph(*args, **kwargs)
        self.sparse_router.init_forward_metadata_capture_cuda_graph(*args, **kwargs)

    def refresh_decode_metadata(self, *args, **kwargs) -> None:
        self.full_router.refresh_decode_metadata(*args, **kwargs)
        self.sparse_router.refresh_decode_metadata(*args, **kwargs)

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        self.full_router.advance_draft_forward_metadata(seq_lens)
        self.sparse_router.advance_draft_forward_metadata(seq_lens)

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        self.full_router.update_draft_forward_metadata(frontier)
        self.sparse_router.update_draft_forward_metadata(frontier)

    def draft_history_view(self):
        return self.full_router.draft_history_view()

    def draft_write_locations_uniform(self, out, cache_start, num_tokens):
        return self.full_router.draft_write_locations_uniform(
            out, cache_start, num_tokens
        )

    def publish_draft_step_locations(self, cache_start, num_tokens):
        # Both routers must present the step window: sparse layers write KV too.
        self.sparse_router.publish_draft_step_locations(cache_start, num_tokens)
        return self.full_router.publish_draft_step_locations(cache_start, num_tokens)

    def decode_window_locations(self):
        return self.full_router.decode_window_locations()

    def extend_span_locations(self):
        return self.full_router.extend_span_locations()

    def configure_runtime(self, **kwargs) -> None:
        self.full_router.configure_runtime(**kwargs)
        self.sparse_router.configure_runtime(**kwargs)

    def register_step_counter(self, step_counter) -> None:
        # The hybrid backend records exactly one step per model layer.
        self.step_counter = step_counter

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch at the CUDA-graph break point using the live forward mode."""
        ambient = current_forward_ctx()
        if ambient is not None:
            forward_mode = ambient.forward_mode
            bs = ambient.bs

        if forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        router = self._router_for_layer(layer.layer_id)
        out_cache_loc = router.write_locations(layer, forward_mode)
        leaf = router._leaf_for(layer)
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                return leaf.forward_decode(
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
            return leaf.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                forward_mode=forward_mode,
                **kwargs,
            )


register_backend("msa_leaf", {AttentionArch.MSA}, MSAAttnBackend)
register_backend(
    "msa",
    {AttentionArch.MSA},
    MSAHybridAttnBackend,
)
