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

"""MiniMax sparse attention backend and dense/sparse router."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    learn_cache_group_geometry,
)
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
)
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    MSA_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.registry import (
    register_backend,
)
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div

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
    # page_table is None on the cache path (per-group page_tables route reads).
    page_table: torch.Tensor | None
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
    # Per-group page tables (group_id -> [num_reqs, max_pages]); None on
    # the cache path (DFLASH block-decode drafts still use it).
    page_tables: dict[str, torch.Tensor] | None = None
    # Per-group KV write locations (group_id -> [num_tokens] int32),
    # built with page_tables — same groups, same lifecycle.
    out_cache_locs: dict[str, torch.Tensor] | None = None


@dataclass(kw_only=True)
class MSADecodeMetadata:
    # page_table is None on the cache path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    # Per-group tables/write-locs; see MSAExtendMetadata.
    page_tables: dict[str, torch.Tensor] | None = None
    out_cache_locs: dict[str, torch.Tensor] | None = None
    # Per-forward view of the backend's shared decode score buffer, pre-filled
    # with -inf and reused by every sparse layer. None on paths that keep the
    # per-layer allocation (draft/DFLASH, or before the buffer is allocated).
    score_out: torch.Tensor | None = None


class MSAAttnBackend(AttentionBackend):
    """MiniMax sparse attention backend that routes through tokenspeed_kernel attention APIs."""

    # The refresh nulls its own dummy table rows, so the wrapper must pass
    # UNPADDED tables (no per-step F.pad).
    tables_self_padding = True

    def bind_decode_views(self, bs: int, cache_group_ids: tuple[str, ...] = ()) -> None:
        """Pre-build the per-bs views with the capture-time group set pinned,
        so the base default capture records the exact per-group views the
        refresh repoints at."""
        if cache_group_ids:
            # Verify keeps [bs]-row tables plus [bs*N] location views.
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "cache_group_ids is unsupported with DFLASH block decode"
        self._decode_views(bs, cache_group_ids=cache_group_ids)

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Per-group write locations for out-of-backend KV writers (fused
        RoPE prewrite): the write must land in the pages this layer's group
        reads, never the scheduler's single-table locations. Draft chains own
        their per-step locations, so they must keep the caller-provided
        tensor."""
        metadata = self._prewrite_metadata(forward_mode)
        if metadata is None or metadata.out_cache_locs is None:
            return out_cache_loc
        return self._select_out_cache_loc(
            layer, metadata, out_cache_loc, prefer_caller=self.is_draft
        )

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Re-anchor the k-row decode metadata to the committed frontier:
        seq_lens becomes ``frontier`` and the grouped write locs cover
        positions ``frontier-k..frontier-1``. Accept-dependent, so pure
        tensor ops recomputed per graph replay; the next metadata init
        resets."""
        md = self.forward_decode_metadata
        fields = {"seq_lens": frontier}
        if md.out_cache_locs is not None:
            fields["out_cache_locs"] = self._compute_decode_group_out_cache_locs(
                md.page_tables,
                frontier,
                self.kernel_page_size,
                self.spec_num_tokens,
            )
        self.forward_decode_metadata = replace(md, **fields)

    # Unconditional: safety comes from the publication rule
    # (kv_cache.recipes.publish) plus the replay
    # stale-table guard. drop the flag.

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return False

    def __init__(self, config: MSAConfig) -> None:
        super().__init__(config)

        # Static information needed for metadata construction and kernel dispatch
        self.max_context_len = config.context_len
        self.kernel_page_size = (
            config.kernel_page_size
            if config.kernel_page_size is not None
            else MSA_PAGE_SIZE
        )
        self.max_num_pages = ceil_div(self.max_context_len, self.kernel_page_size)
        self.tp_q_head_num = max(config.num_attention_heads // config.attn_tp_size, 1)
        self.tp_kv_head_num = max(config.num_kv_heads // config.attn_tp_size, 1)
        self.head_dim = config.head_dim
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_fp8 = self.kv_cache_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )

        # Sparse attention parameters
        self.sparse_layer_ids = config.sparse_layer_ids
        self.index_head_dim = config.index_head_dim
        self.index_topk_blocks = config.index_topk_blocks
        self.index_init_blocks = config.index_init_blocks
        self.index_local_blocks = config.index_local_blocks

        # DFLASH draft: expand decode metadata to spec_num_tokens rows/request
        # (whole block in one decode forward), with uniform non-causal seq_lens.
        self.draft_block_decode = bool(config.draft_block_decode)

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MSADecodeMetadata | None = None
        self.forward_extend_metadata: MSAExtendMetadata | None = None

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
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        # Warmup/placeholder callers omit the extend arrays, so they must be
        # optional.
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        block_tables: dict[str, torch.Tensor] | None = None,
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

        group_page_tables = self._shed_state_groups(block_tables)
        group_out_cache_locs = None
        if group_page_tables:
            # The cache path routes every read/write through the per-group
            # tables; a shared single page_table would be dead work.
            page_table = None
            group_out_cache_locs = self._compute_extend_group_out_cache_locs(
                group_page_tables,
                extend_prefix_lens_cpu[:bs],
                extend_seq_lens_cpu[:bs],
                self.kernel_page_size,
            )
            self._maybe_check_group_write_locs(
                group_page_tables, group_out_cache_locs, self.kernel_page_size
            )
            group_page_tables = self._kernel_page_tables(group_page_tables)
        else:
            page_table = build_page_table(
                req_pool_indices[:bs],
                page_table,
                self.kernel_page_size,
                self.max_context_len,
            )

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
            page_table=page_table,
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
            page_tables=group_page_tables,
            out_cache_locs=group_out_cache_locs,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        cache_group_specs: Sequence = (),
        **kwargs,
    ):
        # State-family groups (GDN/mamba pages) belong to the mamba backend;
        # learn their ids from the pool's specs so every table/location path
        # here (eager, capture, replay) sheds them.
        self._geometry = learn_cache_group_geometry(
            cache_group_specs, default_granularity=self.kernel_page_size
        )

        self.cuda_graph_decode_metadata = {}
        # Per-group persistent buffers, lazily allocated at first
        # capture. parallels cuda_graph_page_table.
        # Initialized before the DFLASH early return: replay reads the dict
        # unconditionally for the stale-table guard.
        self._init_group_graph_buffers(max_bs)
        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: expand to spec_num_tokens decode rows per
            # request (one row per block position), so max_seqlen_q == 1 per row
            # and every block query attends over the whole block (non-causal).
            self.cuda_graph_page_table, self.cuda_graph_seq_lens = (
                self._make_spec_metadata_buffers(max_bs, self.device)
            )
            self.cuda_graph_page_table.zero_()
            # seq_lens are filled from the live draft length inside the captured
            # graph; seed a valid baseline so any pre-broadcast read stays in range.
            self.cuda_graph_seq_lens.fill_(self.spec_num_tokens)
            return
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        # Own the cache-seqlens buffer; replay copies the live lengths in.
        self.cuda_graph_seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )

    def _decode_views(
        self,
        bs: int,
        cache_group_ids: tuple[str, ...] | None = None,
    ) -> MSADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated. ``cache_group_ids`` pins the group set at
        capture time; lazy callers reuse it.
        """
        if cache_group_ids is not None:
            self._decode_view_group_ids = cache_group_ids
        metadata = self.cuda_graph_decode_metadata.get(bs)
        if metadata is not None:
            return metadata
        if self.draft_block_decode and self.spec_num_tokens > 1:
            expanded_bs = bs * self.spec_num_tokens
            metadata = MSADecodeMetadata(
                page_table=self.cuda_graph_page_table[:expanded_bs, :],
                seq_lens=self.cuda_graph_seq_lens[:expanded_bs],
            )
        else:
            gids = getattr(self, "_decode_view_group_ids", None)
            if gids is None:
                gids = tuple(self.cuda_graph_page_tables)
            page_tables, out_cache_locs = self._capture_group_views(
                bs,
                gids,
                tokens_per_req=self.tokens_per_req,
            )
            metadata = MSADecodeMetadata(
                # Grouped captures route reads through the per-group tables;
                # the shared single page_table is never filled on that path.
                page_table=(
                    None
                    if page_tables is not None
                    else self.cuda_graph_page_table[:bs, :]
                ),
                seq_lens=self.cuda_graph_seq_lens[:bs],
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
                score_out=self.decode_score_buffer[: bs * self.tokens_per_req],
            )
        self.cuda_graph_decode_metadata[bs] = metadata
        return metadata

    # Capture is inherited (base default: bind_decode_views + idle refresh).
    # It relies on the runner seeding capture seq_lens >= the verify floor:
    # the refresh below copies them without a clamp.

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        page_table: torch.Tensor | None = None,
        num_extends: int = 0,
        for_graph_replay: bool = False,
        block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        assert not forward_mode.is_extend_or_mixed()

        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft: replicate each request's page table to its
            # spec_num_tokens block rows. The staged table carries raw
            # scheduler pages — expand into kernel pages first. Under replay
            # the block-end seq_lens are re-derived inside the captured graph
            # (fill_block_decode_seq_lens); eager fills them here.
            base_page_table = self._expand_history_table(page_table[:bs])
            self.cuda_graph_page_table[: bs * self.spec_num_tokens, :].view(
                bs, self.spec_num_tokens, self.max_num_pages
            ).copy_(base_page_table[:, None, :])
            # Eager has no in-graph writer; the capture default's idle
            # refresh (actual_bs == 0) must seed the same safe baseline the
            # recorded fill_block_decode_seq_lens overwrites on replay.
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
            self.forward_decode_metadata = self._decode_views(bs)
            return

        # Every pool publishes at least one history group now, so the
        # per-group capture buffers always exist; the pre-LCM single-table
        # gather has no remaining producer. The actual_bs == 0 arm (capture
        # seeding / idle) computes nothing over live rows, so group-less unit
        # fixtures may pass through it.
        if not self.cuda_graph_page_tables and actual_bs > 0:
            raise RuntimeError(
                "MSA decode without per-group capture buffers: the pool "
                "published no cache groups, which the LCM contract forbids"
            )
        self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])

        if block_tables:
            self._fill_group_graph_buffers(
                bs,
                block_tables,
                self.cuda_graph_seq_lens,
                tokens_per_req=self.tokens_per_req,
            )

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

        out_cache_loc = self._select_out_cache_loc(
            layer,
            metadata,
            out_cache_loc,
        )
        page_table = self._select_page_table(layer, metadata)
        if page_table is None:
            raise RuntimeError("MSA decode requires a page table.")
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
            page_table=page_table,
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

        out_cache_loc = self._select_out_cache_loc(layer, metadata, out_cache_loc)
        out_cache_loc = out_cache_loc[:real_tokens]
        page_table = self._select_page_table(layer, metadata)
        if page_table is None:
            raise RuntimeError("MSA prefill requires a page table.")
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
            page_table=page_table,
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
        k, v = self._trim_kv_to_locs(out_cache_loc, k, v)

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

    def _make_spec_metadata_buffers(
        self,
        bs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_bs = bs * self.spec_num_tokens
        cuda_graph_page_table = torch.empty(
            (expanded_bs, self.max_num_pages),
            dtype=torch.int32,
            device=device,
        )
        cuda_graph_seq_lens = torch.empty(
            (expanded_bs,),
            dtype=torch.int32,
            device=device,
        )
        return (cuda_graph_page_table, cuda_graph_seq_lens)

    def _fill_spec_metadata_uniform(
        self,
        expanded_page_table: torch.Tensor,
        expanded_seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Expand spec metadata with a uniform (non-causal) seq_len per row.

        Replicates the full seq_len to all spec_num_tokens rows of a request so
        each row decodes with max_seqlen_q == 1 over the whole block. Used by the
        DFLASH drafter so every block query attends over the entire block
        (non-causal block-diffusion drafting), as opposed to the target's
        unexpanded causal multi-query verify path.
        """
        bs = seq_lens.shape[0]
        spec_num_tokens = self.spec_num_tokens
        expanded_page_table = expanded_page_table.view(
            bs, spec_num_tokens, self.max_num_pages
        )
        expanded_page_table.copy_(page_table[:, None, :])
        # Clamp to max_context_len so the draft decode never asks the attention
        # kernel for more than max_num_pages worth of page-table columns. The
        # block-end length is prefix + spec_num_tokens, which can exceed
        # max_context_len for a request near the context limit; without the
        # clamp the kernel reads page_table[:, >= max_num_pages] out of bounds
        # (CUDA illegal memory access). Mirrors fill_block_decode_seq_lens on the
        # cuda-graph path (this eager path is taken by mixed prefill+decode
        # batches even when cuda graphs are enabled).
        expanded_seq_lens.view(bs, spec_num_tokens).copy_(
            seq_lens.clamp(spec_num_tokens, self.max_context_len)[:, None]
        )

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
    """Minimax hybrid attention backend that dispatches to either a dense or sparse backend per layer."""

    def __init__(self, config: MSAConfig) -> None:
        from tokenspeed.runtime.layers.attention.registry import (
            _create_attn_backend_with_name,
        )

        super().__init__(config)
        full_attn_backend = _create_attn_backend_with_name(
            config.full_attn_backend_name,
            AttentionArch.MHA,
            config,
        )
        sparse_attn_backend = MSAAttnBackend(config)
        self.full_attn_backend = full_attn_backend
        self.sparse_attn_backend = sparse_attn_backend
        self.sparse_layer_ids = sparse_attn_backend.sparse_layer_ids
        logger.info(
            "Created MiniMax hybrid attention backend: %d dense layers, "
            "%d sparse layers (dense=%s, sparse=%s)",
            len(config.compute_layer_types) - len(config.sparse_layer_ids),
            len(config.sparse_layer_ids),
            type(full_attn_backend).__name__,
            type(sparse_attn_backend).__name__,
        )

    def _backend_for_layer(self, layer_id: int) -> AttentionBackend:
        if layer_id in self.sparse_layer_ids:
            return self.sparse_attn_backend
        return self.full_attn_backend

    def child_backends(self) -> tuple[AttentionBackend, ...]:
        return (self.full_attn_backend, self.sparse_attn_backend)

    @property
    def cache_consumer_families(self) -> frozenset[str]:
        """Cache families consumed by the dense and sparse children."""
        return frozenset(self.full_attn_backend.cache_consumer_families) | frozenset(
            self.sparse_attn_backend.cache_consumer_families
        )

    def set_cache_pool(self, cache_pool) -> None:
        self.cache_pool = cache_pool
        self.full_attn_backend.set_cache_pool(cache_pool)
        self.sparse_attn_backend.set_cache_pool(cache_pool)

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        # A single model-wide answer must be safe for sparse layers too.
        del forward_mode
        return False

    def init_forward_metadata(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata(*args, **kwargs)
        self.sparse_attn_backend.init_forward_metadata(*args, **kwargs)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        **kwargs,
    ) -> None:
        self.full_attn_backend.init_cuda_graph_state(max_bs, **kwargs)
        self.sparse_attn_backend.init_cuda_graph_state(max_bs, **kwargs)

    # Capture is inherited: the base default binds and refreshes through the
    # two fan-outs below, and both children are default-capture themselves.

    def bind_decode_views(self, bs: int, cache_group_ids: tuple[str, ...] = ()) -> None:
        self.full_attn_backend.bind_decode_views(bs, cache_group_ids)
        self.sparse_attn_backend.bind_decode_views(bs, cache_group_ids)

    def refresh_decode_metadata(self, *args, **kwargs) -> None:
        self.full_attn_backend.refresh_decode_metadata(*args, **kwargs)
        self.sparse_attn_backend.refresh_decode_metadata(*args, **kwargs)

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        self.full_attn_backend.advance_draft_forward_metadata(seq_lens)
        self.sparse_attn_backend.advance_draft_forward_metadata(seq_lens)

    def configure_runtime(self, **kwargs) -> None:
        self.full_attn_backend.configure_runtime(**kwargs)
        self.sparse_attn_backend.configure_runtime(**kwargs)

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
        out_cache_loc: torch.Tensor,
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

        backend = self._backend_for_layer(layer.layer_id)
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                return backend.forward_decode(
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
            return backend.forward_extend(
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

    def forward_decode(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        return self._backend_for_layer(layer.layer_id).forward_decode(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        return self._backend_for_layer(layer.layer_id).forward_extend(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )


register_backend(
    "msa",
    {AttentionArch.MSA},
    MSAHybridAttnBackend,
)
