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

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_plan,
    mha_prefill,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    gather_page_table_with_padding,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import scrub_padding_tail
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


def _scrub_extend_padding(metadata, q, k, v) -> None:
    """Zero the q/k/v rows beyond the real (unpadded) token count under a prefill graph.

    Reads the count from the pinned CPU cu-seqlens mirror (sync-free) and delegates the
    zeroing to the shared prefill-graph padding helper. No-op on normal unpadded forwards.
    """
    scrub_padding_tail(metadata.cu_extend_seq_lens_cpu[-1], q, k, v)


@dataclass(kw_only=True)
class MHAExtendMetadata:
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
class MHADecodeMetadata:
    # page_table is None on the flat path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    # Flat per-group tables/write-locs; see MHAExtendMetadata.
    page_tables: dict[str, torch.Tensor] | None = None
    out_cache_locs: dict[str, torch.Tensor] | None = None


class MHAAttnBackend(AttentionBackend):
    """Standard MHA backend that routes through tokenspeed_kernel attention APIs."""

    # Unconditional: safety comes from the publication rule
    # (paged_cache_spec.publish_paged_cache_groups) plus the replay
    # stale-table guard. TODO(radix-removal): drop the flag.
    uses_flat_cache_groups: bool = True

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return forward_mode is not None and forward_mode.is_decode()

    def __init__(self, config: MHAConfig):
        super().__init__(config)
        # Map the selected backend to the corresponding kernel solution string.
        backend_name = config.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]

        # Static information needed for metadata construction and kernel dispatch
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = ceil_div(self.max_context_len, self.page_size)
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_kv_heads
        self.tp_q_head_num = max(num_q_heads // config.attn_tp_size, 1)
        self.tp_kv_head_num = max(num_kv_heads // config.attn_tp_size, 1)
        self.head_dim = config.head_dim
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_fp8 = self.kv_cache_dtype == torch.float8_e4m3fn
        self.plan = partial(
            mha_plan,
            dtype=torch.float8_e4m3fn if self.is_fp8 else self.qkv_dtype,
            head_dim=self.head_dim,
            return_lse=False,
            solution=self.kernel_solution,
        )
        # DFLASH draft: expand decode metadata to spec_num_tokens rows/request
        # (whole block in one decode forward), with uniform non-causal seq_lens.
        self.draft_block_decode = bool(getattr(config, "draft_block_decode", False))

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_extend_metadata: MHAExtendMetadata | None = None

        # family="state" group ids (GDN/mamba state pages) learned from the
        # pool's specs in init_cuda_graph_state; this backend sheds them.
        self.flat_state_group_ids: frozenset[str] = frozenset()

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
        assert not forward_mode.is_mixed(), "mha backend does not support mixed batch"

        seq_lens = seq_lens[:bs]

        flat_page_tables = self._shed_state_groups(flat_block_tables)
        flat_out_cache_locs = None
        if flat_page_tables:
            # Mirrors the capture-time assert: flat + spec is unsupported.
            assert not (
                self.spec_num_tokens > 1
                and (not self.is_draft or self.draft_block_decode)
            ), "flat cache groups are unsupported with spec_num_tokens > 1"
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
                flat_out_cache_locs = self._compute_flat_decode_out_cache_locs(
                    flat_page_tables,
                    seq_lens,
                    self.page_size,
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

            self.forward_extend_metadata = MHAExtendMetadata(
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
                self.forward_decode_metadata = MHADecodeMetadata(
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
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=expanded_page_table,
                    seq_lens=expanded_seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )
            else:
                self.forward_decode_metadata = MHADecodeMetadata(
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
        self.flat_state_group_ids = frozenset(
            str(spec.group_id)
            for spec in paged_cache_group_specs
            if spec.family == "state"
        )
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
        self.cuda_graph_flat_page_tables: dict[str, torch.Tensor] = {}
        self.cuda_graph_flat_out_cache_locs: dict[str, torch.Tensor] = {}
        self._cuda_graph_max_bs = max_bs
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
        page_tables = None
        out_cache_locs = None
        if flat_cache_group_ids:
            # Per-group views are bs rows; spec buffers expand to
            # bs * spec_num_tokens rows. TODO(flat+spec).
            assert not (
                self.spec_num_tokens > 1
                and (not self.is_draft or self.draft_block_decode)
            ), "flat_cache_group_ids is unsupported with spec_num_tokens > 1"
            page_tables = {}
            out_cache_locs = {}
            for gid in flat_cache_group_ids:
                if gid in self.flat_state_group_ids:
                    # State pages ride to the mamba backend; no MHA buffers.
                    continue
                buf = self.cuda_graph_flat_page_tables.get(gid)
                if buf is None:
                    buf = torch.zeros(
                        (self._cuda_graph_max_bs, self.max_num_pages),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.cuda_graph_flat_page_tables[gid] = buf
                loc_buf = self.cuda_graph_flat_out_cache_locs.get(gid)
                if loc_buf is None:
                    loc_buf = torch.zeros(
                        (self._cuda_graph_max_bs,),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.cuda_graph_flat_out_cache_locs[gid] = loc_buf
                page_tables[gid] = buf[:bs, :]
                out_cache_locs[gid] = loc_buf[:bs]
            if not page_tables:
                # Only state groups delivered: nothing for this backend.
                page_tables = None
                out_cache_locs = None

        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: spec_num_tokens decode rows per request.
            expanded_bs = bs * self.spec_num_tokens
            metadata = MHADecodeMetadata(
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
            metadata = MHADecodeMetadata(
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
        # bs == 0 may skip: col-0 buffer entries stay valid (never -1),
        # outputs are discarded, and only unit tests reach it.
        if self.cuda_graph_flat_page_tables and bs > 0:
            if not flat_block_tables:
                raise RuntimeError(
                    "MHAAttnBackend replay: flat per-group CUDA-graph buffers "
                    f"exist for groups "
                    f"{sorted(self.cuda_graph_flat_page_tables)} "
                    f"but flat_block_tables is missing/empty at bs={bs}; the "
                    "captured graph would read stale page tables."
                )
            missing = set(self.cuda_graph_flat_page_tables) - set(flat_block_tables)
            if missing:
                raise RuntimeError(
                    "MHAAttnBackend replay: flat_block_tables at bs="
                    f"{bs} is missing captured groups {sorted(missing)} "
                    f"(delivered: {sorted(flat_block_tables)}); the captured "
                    "graph would read stale page tables for those groups."
                )

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

        # Padding contract (canonical; bs is the padded bs): dummy ROWS pad
        # with 0 — replayed at seq_lens=1 they dereference exactly col 0,
        # the zero-init dummy page. Column tails pad with -1, never read
        # past cache_seqlens.
        if flat_block_tables:
            for gid, src in flat_block_tables.items():
                if gid in self.flat_state_group_ids:
                    # State group: the mamba backend consumes it directly.
                    continue
                buf = self.cuda_graph_flat_page_tables[gid]
                cols = src.shape[1]
                # cols >= 1: a zero-width table would leave dummy rows'
                # col 0 unwritten.
                assert 1 <= cols <= buf.shape[1], (
                    f"flat table for group {gid!r}: {cols} cols outside"
                    f" [1, {buf.shape[1]}] (CUDA-graph buffer width)"
                )
                assert src.shape[0] >= bs, (
                    f"flat table for group {gid!r} has {src.shape[0]} rows"
                    f" < padded bs {bs}"
                )
                buf[:bs, :cols].copy_(src[:bs, :])
                if cols < buf.shape[1]:
                    buf[:bs, cols:].fill_(-1)

            # cuda_graph_seq_lens aliases the controller's seq_lens_buf,
            # which input prep fills (current lens + padding 1s) BEFORE this
            # call, so [:bs] is current when recomputing write locs.
            locs = self._compute_flat_decode_out_cache_locs(
                {
                    gid: self.cuda_graph_flat_page_tables[gid][:bs, :]
                    for gid in flat_block_tables
                    if gid not in self.flat_state_group_ids
                },
                self.cuda_graph_seq_lens[:bs],
                self.page_size,
            )
            for gid, val in locs.items():
                self.cuda_graph_flat_out_cache_locs[gid][:bs].copy_(val)

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        has_kv = k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if has_kv:
            k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        sinks = kwargs.get("sinks")

        out_cache_loc = self._select_out_cache_loc(
            layer, self.forward_decode_metadata, out_cache_loc
        )

        return self._forward_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            self.forward_decode_metadata,
            save_kv_cache=save_kv_cache,
            sinks=sinks,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        assert k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        metadata = self.forward_extend_metadata
        sinks = kwargs.get("sinks")
        out_cache_loc = self._select_out_cache_loc(layer, metadata, out_cache_loc)
        plan = self.plan(
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
        )

        extend_mode = plan.get("extend_mode", "prewrite")
        if metadata.max_extend_prefix_len == 0 and extend_mode == "postwrite":
            return self._forward_prefill(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
            )
        else:
            return self._forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
            )

    @staticmethod
    def _select_group_entry(layer, mapping, what: str):
        """Pick this layer's entry from a flat per-group dict (page tables or
        write locs): the layer's group entry, or the sole entry when the
        layer carries no/unknown group id. TODO(radix-removal): collapses to
        `mapping[layer.group_id]` once flat is the only path.
        """
        group_id = getattr(layer, "group_id", "")
        if not group_id or group_id not in mapping:
            if len(mapping) == 1:
                return next(iter(mapping.values()))
            raise KeyError(
                f"{what}: layer group_id={group_id!r} not in flat group "
                f"keys {sorted(mapping)}"
            )
        return mapping[group_id]

    def _select_page_table(self, layer, metadata):
        if metadata.page_tables is None:
            return metadata.page_table
        return self._select_group_entry(layer, metadata.page_tables, "page table")

    def _select_out_cache_loc(self, layer, metadata, out_cache_loc):
        # KV writes must land in the pages the layer's group reads (M-W1).
        if metadata.out_cache_locs is None:
            return out_cache_loc
        return self._select_group_entry(
            layer, metadata.out_cache_locs, "flat write locs"
        )

    def select_out_cache_loc(self, layer, out_cache_loc):
        """Per-group write locations for out-of-backend KV writers (fused
        RoPE prewrite); prewrite is decode-only, so reads decode metadata.
        """
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.out_cache_locs is None:
            return out_cache_loc
        return self._select_out_cache_loc(layer, metadata, out_cache_loc)

    @staticmethod
    def _compute_flat_decode_out_cache_locs(page_tables, seq_lens, page_size):
        """Per-group decode write locs: one token per request at seq_len-1,
        gathered from the group's own read table (M-W1). The tail page is
        never a hole (SWA holes sit only at the window front).
        """
        pos = (seq_lens - 1).to(torch.int64)
        page_idx = pos // page_size
        off = (pos % page_size).to(torch.int32)
        out = {}
        for gid, table in page_tables.items():
            pages = table.gather(1, page_idx.unsqueeze(1)).squeeze(1)
            out[gid] = pages * page_size + off
        return out

    def _shed_state_groups(self, tables):
        """Drop family="state" groups (GDN/mamba state pages, consumed by the
        mamba backend): computing write locs / capture buffers over the
        hole-heavy state table writes the dummy page and trips
        TOKENSPEED_FLAT_DEBUG. Returns None when nothing is left.
        """
        if not tables:
            return None
        if self.flat_state_group_ids:
            tables = {
                gid: table
                for gid, table in tables.items()
                if gid not in self.flat_state_group_ids
            }
        return tables or None

    @staticmethod
    def _compute_flat_extend_out_cache_locs(
        page_tables, extend_prefix_lens_cpu, extend_seq_lens_cpu, page_size
    ):
        """Per-group extend write locs: positions [prefix_len, seq_len) per
        request, flattened in q/k/v token order (cu_extend_seq_lens). Bounds
        come from the CPU mirrors — no per-request GPU sync.
        TODO(flat-perf): batch the per-request loop via repeat_interleave.
        """
        device = next(iter(page_tables.values())).device
        prefix_lens = [int(x) for x in extend_prefix_lens_cpu.tolist()]
        extend_lens = [int(x) for x in extend_seq_lens_cpu.tolist()]
        out = {gid: [] for gid in page_tables}
        for i, (start, num_new) in enumerate(zip(prefix_lens, extend_lens)):
            pos = torch.arange(start, start + num_new, dtype=torch.int64, device=device)
            page_idx = pos // page_size
            off = (pos % page_size).to(torch.int32)
            for gid, table in page_tables.items():
                pages = table[i].gather(0, page_idx)
                out[gid].append(pages * page_size + off)
        return {
            gid: (
                torch.cat(chunks)
                if chunks
                else torch.empty(0, dtype=torch.int32, device=device)
            )
            for gid, chunks in out.items()
        }

    @staticmethod
    def _maybe_check_flat_write_locs(page_tables, out_cache_locs, page_size):
        """TOKENSPEED_FLAT_DEBUG=1 (eager only, GPU sync): write pages must
        be real and inside the group's table. Not for graph-padded batches —
        dummy rows would trip the non-hole assert (see the padding contract
        in init_forward_metadata_replay_cuda_graph).
        """
        if os.environ.get("TOKENSPEED_FLAT_DEBUG") != "1":
            return
        for gid, locs in out_cache_locs.items():
            pages = (locs // page_size).to(torch.int32)
            table = page_tables[gid]
            assert (
                pages != 0
            ).all(), f"flat write loc in null page 0 for group {gid!r}"
            real = table[table > 0]
            assert torch.isin(
                pages, real
            ).all(), f"flat write pages escape group {gid!r}'s table"

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        _scrub_extend_padding(metadata, q, k, v)
        # TODO: use a custom kernel to do downcast
        if self.is_fp8:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        output = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_extend_seq_lens,
            cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
            max_seqlen=metadata.max_extend_seq_len,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        return output

    def _forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        _scrub_extend_padding(metadata, q, k, v)
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        if self.is_fp8:
            q = q.to(torch.float8_e4m3fn)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        output = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            cu_seqlens_kv=metadata.cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=self.max_context_len,
            # DFLASH marks its draft attention non-causal so the draft block's
            # query positions attend bidirectionally. Every other layer leaves
            # the attribute unset, so this stays causal by default.
            is_causal=not bool(getattr(layer, "non_causal", False)),
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHADecodeMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        if self.is_fp8:
            q = q.to(torch.float8_e4m3fn)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        max_seqlen_q = q.shape[0] // metadata.seq_lens.shape[0]
        output = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_k=self.max_context_len,
            max_seqlen_q=max_seqlen_q,
            solution=self.kernel_solution,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

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

        if self.is_fp8 and k.dtype != torch.float8_e4m3fn:
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


for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
