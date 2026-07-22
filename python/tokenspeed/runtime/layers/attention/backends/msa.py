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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    msa_decode_with_kvcache,
    msa_extend_with_kvcache,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    gather_page_table_with_padding,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    init_backend_cuda_graph_state,
)
from tokenspeed.runtime.layers.attention.backends.flat_groups import (
    FlatCacheGroupsMixin,
)
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
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
    # page_table is None on the flat path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_extend_seq_lens: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens_cpu: list[int]
    cu_extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int = 0
    # Flat per-group page tables (group_id -> [num_reqs, max_pages]); None on
    # the single-table path. TODO(radix-removal): drop the single page_table.
    page_tables: dict[str, torch.Tensor] | None = None
    # Flat per-group KV write locations (group_id -> [num_tokens] int32),
    # built with page_tables — same groups, same lifecycle.
    out_cache_locs: dict[str, torch.Tensor] | None = None


@dataclass(kw_only=True)
class MSADecodeMetadata:
    # page_table is None on the flat path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    # Flat per-group tables/write-locs; see MSAExtendMetadata.
    page_tables: dict[str, torch.Tensor] | None = None
    out_cache_locs: dict[str, torch.Tensor] | None = None


class MSAAttnBackend(FlatCacheGroupsMixin, AttentionBackend):
    """MiniMax sparse attention backend that routes through tokenspeed_kernel attention APIs."""

    # Unconditional: safety comes from the publication rule
    # (paged_cache_spec.publish_paged_cache_groups) plus the replay
    # stale-table guard. TODO(radix-removal): drop the flag.
    uses_flat_cache_groups: bool = True

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return False

    def __init__(self, config: MSAConfig) -> None:
        super().__init__(config)

        # Static information needed for metadata construction and kernel dispatch
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = ceil_div(self.max_context_len, self.page_size)
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
        self.draft_block_decode = bool(getattr(config, "draft_block_decode", False))

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MSADecodeMetadata | None = None
        self.forward_extend_metadata: MSAExtendMetadata | None = None

    # ------------------------------------------------------------------
    # Metadata initialization
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        # Only consumed on the extend/mixed path; decode callers (e.g. the
        # DFLASH draft and the cuda-graph wrapper's draft decode init) omit
        # them, so they must be optional.
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_mixed(), "MSA backend does not support mixed batch"

        seq_lens = seq_lens[:bs]

        flat_page_tables = self._shed_state_groups(flat_block_tables)
        flat_out_cache_locs = None
        if flat_page_tables:
            # Verify keeps [bs]-row tables; only DFLASH expands rows. TODO(flat+dflash).
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "flat cache groups are unsupported with DFLASH block decode"
            # The flat path routes every read/write through the per-group
            # tables; the radix single table would be dead work.
            page_table = None
            if forward_mode.is_extend_or_mixed():
                assert extend_prefix_lens_cpu is not None
                assert extend_seq_lens_cpu is not None
                flat_out_cache_locs = self._compute_flat_extend_out_cache_locs(
                    flat_page_tables,
                    extend_prefix_lens_cpu[:bs],
                    extend_seq_lens_cpu[:bs],
                    self.page_size,
                )
            else:
                verify_tokens = (
                    self.spec_num_tokens
                    if self.spec_num_tokens > 1 and not self.is_draft
                    else 1
                )
                flat_out_cache_locs = self._compute_flat_decode_out_cache_locs(
                    flat_page_tables,
                    seq_lens,
                    self.page_size,
                    verify_tokens,
                )
            self._maybe_check_flat_write_locs(
                flat_page_tables, flat_out_cache_locs, self.page_size
            )
        else:
            page_table = build_page_table(
                req_pool_indices[:bs],
                req_to_page,
                self.page_size,
                self.max_context_len,
            )

        if forward_mode.is_extend_or_mixed():
            assert extend_seq_lens is not None
            assert extend_seq_lens_cpu is not None
            assert extend_prefix_lens is not None
            assert extend_prefix_lens_cpu is not None

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
            max_extend_prefix_len = int(extend_prefix_lens_cpu[:bs].max().item())

            self.forward_extend_metadata = MSAExtendMetadata(
                page_table=page_table,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                cu_extend_seq_lens=cu_extend_seq_lens,
                cu_seqlens_kv=cu_seqlens_kv,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
                max_extend_seq_len=max_extend_seq_len,
                max_extend_prefix_len=max_extend_prefix_len,
                page_tables=flat_page_tables,
                out_cache_locs=flat_out_cache_locs,
            )

            # Drafter step 1+ decodes under an EXTEND/MIXED target; seq_lens
            # aliases the drafter's live buffer (pre-written by the wrapper).
            if self.is_draft:
                self.forward_decode_metadata = MSADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )
        else:
            if self.draft_block_decode and self.spec_num_tokens > 1:
                # DFLASH drafts a whole block in one decode forward; the decode
                # kernel keys masking off max_seqlen_q, so expand each request
                # into spec_num_tokens rows with the SAME full seq_len. That
                # makes max_seqlen_q == 1 per row, so every block query attends
                # over the entire block (non-causal block-diffusion drafting).
                # Target verify keeps the unexpanded multi-query decode path.
                expanded_page_table, expanded_seq_lens = (
                    self._make_spec_metadata_buffers(
                        bs,
                        page_table.device,
                    )
                )
                self._fill_spec_metadata_uniform(
                    expanded_page_table,
                    expanded_seq_lens,
                    page_table,
                    seq_lens,
                )
                self.forward_decode_metadata = MSADecodeMetadata(
                    page_table=expanded_page_table,
                    seq_lens=expanded_seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )
            else:
                self.forward_decode_metadata = MSADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        seq_lens_buf: torch.Tensor,
        paged_cache_group_specs: Sequence = (),
        **kwargs,
    ):
        # State-family groups (GDN/mamba pages) belong to the mamba backend;
        # learn their ids from the pool's specs so every flat table/loc path
        # here (eager, capture, replay) sheds them.
        self._learn_flat_state_groups(paged_cache_group_specs)
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )

        self.cuda_graph_decode_metadata = {}
        # Flat per-group persistent buffers, lazily allocated at first
        # capture. TODO(radix-removal): parallels cuda_graph_page_table.
        # Initialized before the DFLASH early return: replay reads the dict
        # unconditionally for the stale-table guard.
        self._init_flat_graph_buffers(max_bs)
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
        if self.spec_num_tokens > 1 and not self.is_draft:
            self.cuda_graph_seq_lens = torch.empty(
                (max_bs,), dtype=torch.int32, device=self.device
            )
        else:
            # Alias controller's seq_lens_buf — backend never mutates it.
            self.cuda_graph_seq_lens = seq_lens_buf

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        flat_cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # Real tables only arrive at replay: capture lazily allocates
        # persistent per-group buffers and records metadata views into them,
        # so replay can copy_ fresh data to the graph-recorded addresses.
        if flat_cache_group_ids:
            # Verify keeps [bs]-row tables + [bs*N] loc views. TODO(flat+dflash).
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "flat_cache_group_ids is unsupported with DFLASH block decode"
        page_tables, out_cache_locs = self._flat_capture_group_views(
            bs,
            flat_cache_group_ids,
            tokens_per_req=(
                self.spec_num_tokens
                if self.spec_num_tokens > 1 and not self.is_draft
                else 1
            ),
        )

        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: spec_num_tokens decode rows per request.
            expanded_bs = bs * self.spec_num_tokens
            metadata = MSADecodeMetadata(
                page_table=self.cuda_graph_page_table[:expanded_bs, :],
                seq_lens=self.cuda_graph_seq_lens[:expanded_bs],
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
            )
            # Uniform non-causal seq_lens are written by the drafter inside the
            # captured graph (see fill_block_decode_seq_lens); seed a safe
            # baseline for the capture run before that op records.
            metadata.seq_lens.fill_(self.spec_num_tokens)
        else:
            metadata = MSADecodeMetadata(
                # Flat captures route reads through the per-group tables and
                # replay never fills the radix single table, so mirror the
                # eager flat path: page_table=None instead of a slice of the
                # never-filled zero buffer.
                page_table=(
                    None
                    if page_tables is not None
                    else self.cuda_graph_page_table[:bs, :]
                ),
                seq_lens=self.cuda_graph_seq_lens[:bs],
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
            )
            if self.spec_num_tokens > 1 and not self.is_draft:
                metadata.seq_lens.copy_(seq_lens[:bs].clamp_min(self.spec_num_tokens))
        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # Fail loudly instead of replaying over stale/zero page tables.
        self._flat_replay_stale_guard(bs, flat_block_tables)

        # Flat captures read only the per-group buffers; the radix single
        # table (cuda_graph_page_table) would be dead work there.
        if not self.cuda_graph_flat_page_tables:
            gather_page_table_with_padding(
                req_to_page=req_to_page,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out=self.cuda_graph_page_table,
                bs=bs,
                max_num_pages=self.max_num_pages,
                page_size=self.page_size,
                dummy_slot=0,
            )
        if self.spec_num_tokens > 1 and not self.is_draft:
            self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])
        elif self.draft_block_decode:
            # DFLASH draft: replicate each request's page table to its
            # spec_num_tokens block rows. The block-end seq_lens are filled by
            # the drafter inside the captured graph, so they are not touched
            # here (they re-derive from the live draft length on every replay).
            base_page_table = req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            self.cuda_graph_page_table[: bs * self.spec_num_tokens, :].view(
                bs, self.spec_num_tokens, self.max_num_pages
            ).copy_(base_page_table[:, None, :])

        # cuda_graph_seq_lens is filled by input prep / the spec copy above.
        if flat_block_tables:
            self._flat_replay_fill(
                bs,
                flat_block_tables,
                self.cuda_graph_seq_lens,
                tokens_per_req=(
                    self.spec_num_tokens
                    if self.spec_num_tokens > 1 and not self.is_draft
                    else 1
                ),
            )

        if bs in self.cuda_graph_decode_metadata:
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast each request's block-end length to its
        spec_num_tokens cuda-graph decode rows (uniform, non-causal).

        Called by the drafter inside the captured graph so that on every replay
        the expanded seq_lens re-derive from the live draft length (which is
        recomputed in-graph from the target's accept lengths).

        Args:
            bs: Number of draft requests.
            block_seq_lens: ``[bs]`` per-request block-end lengths
                (prefix + spec_num_tokens).
        """
        spec = self.spec_num_tokens
        self.cuda_graph_seq_lens[: bs * spec].view(bs, spec).copy_(
            block_seq_lens[:bs]
            .clamp(self.spec_num_tokens, self.max_context_len)
            .unsqueeze(1)
        )

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
            page_size=self.page_size,
            index_scale=self.index_head_dim**-0.5,
            attention_scale=layer.scaling,
            init_blocks=self.index_init_blocks,
            local_blocks=self.index_local_blocks,
            max_seqlen_q=decode_query_len,
            max_seqlen_k=self.max_context_len,
            k_scale=layer.k_scale if self.is_fp8 else None,
            v_scale=layer.v_scale if self.is_fp8 else None,
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
            page_size=self.page_size,
            index_scale=self.index_head_dim**-0.5,
            attention_scale=layer.scaling,
            init_blocks=self.index_init_blocks,
            local_blocks=self.index_local_blocks,
            k_scale=layer.k_scale if self.is_fp8 else None,
            v_scale=layer.v_scale if self.is_fp8 else None,
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
                page_size=self.page_size,
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
            self.page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            self.page_size,
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

    uses_flat_cache_groups: bool = True

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

    @property
    def sinks_dtype(self) -> torch.dtype:
        return self.full_attn_backend.sinks_dtype

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
        seq_lens_buf: torch.Tensor,
        **kwargs,
    ) -> None:
        init_backend_cuda_graph_state(
            self.full_attn_backend, max_bs, seq_lens_buf, **kwargs
        )
        init_backend_cuda_graph_state(
            self.sparse_attn_backend, max_bs, seq_lens_buf, **kwargs
        )

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(*args, **kwargs)
        self.sparse_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **kwargs
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata_replay_cuda_graph(*args, **kwargs)
        self.sparse_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **kwargs
        )

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
