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

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp8

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import slice_to_real_tokens
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    learn_cache_group_geometry,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    MHA_PAGE_SIZE,
)
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


def _slice_extend_inputs(metadata, q, k, v):
    """Remove prefill-graph padding rows before calling an attention kernel.

    The live cu-seqlens describe only real tokens. Some kernels tolerate extra
    zero rows, but others still derive work from the tensor shape. Use the
    pinned CPU mirror (sync-free) so every solution receives the same exact-row
    contract. No-op on normal unpadded forwards.
    """
    return slice_to_real_tokens(metadata.cu_extend_seq_lens_cpu[-1], q, k, v)


@dataclass(kw_only=True)
class MHAExtendMetadata:
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
    max_extend_seq_len: int
    max_extend_prefix_len: int = 0
    # Per-group page tables (group_id -> [num_reqs, max_pages]); None on
    # the cache path (DFLASH block-decode drafts still use it).
    page_tables: dict[str, torch.Tensor] | None = None
    # Per-group KV write locations (group_id -> [num_tokens] int32),
    # built with page_tables — same groups, same lifecycle.
    out_cache_locs: dict[str, torch.Tensor] | None = None


@dataclass(kw_only=True)
class MHADecodeMetadata:
    # page_table is None on the cache path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    # Per-group tables/write-locs; see MHAExtendMetadata.
    page_tables: dict[str, torch.Tensor] | None = None
    out_cache_locs: dict[str, torch.Tensor] | None = None


class MHAAttnBackend(AttentionBackend):
    """Standard MHA backend that routes through tokenspeed_kernel attention APIs."""

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
            fields["out_cache_locs"] = self._compute_decode_out_cache_locs(
                md.page_tables,
                frontier,
                self.spec_num_tokens,
            )
        self.forward_decode_metadata = replace(md, **fields)

    # Unconditional: safety comes from the publication rule
    # (kv_cache.recipes.publish) plus the replay
    # stale-table guard. drop the flag.

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return forward_mode is not None and forward_mode.is_decode()

    def __init__(self, config: AttnConfig, spec: MHAConfig):
        super().__init__(config, spec)
        # Map the selected backend to the corresponding kernel solution string.
        backend_name = spec.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]

        # Static information needed for metadata construction and kernel dispatch
        self.max_context_len = config.context_len
        self.kernel_page_size = (
            config.kernel_page_size
            if config.kernel_page_size is not None
            else MHA_PAGE_SIZE
        )
        self.max_num_pages = ceil_div(self.max_context_len, self.kernel_page_size)
        num_q_heads = spec.num_attention_heads
        num_kv_heads = spec.num_kv_heads
        self.tp_q_head_num = max(num_q_heads // spec.attn_tp_size, 1)
        self.tp_kv_head_num = max(num_kv_heads // spec.attn_tp_size, 1)
        self.head_dim = spec.head_dim
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_mxfp8 = bool(getattr(config, "kv_cache_mxfp8", False))
        # mxfp8 shares the fp8 storage dtype but uses block scales; keep it off the per-tensor casts
        self.is_fp8 = (
            self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not self.is_mxfp8
        )
        self.plan = partial(
            mha_plan,
            dtype=(
                torch.float8_e4m3fn
                if self.is_mxfp8
                else (self.kv_cache_dtype if self.is_fp8 else self.qkv_dtype)
            ),
            head_dim=self.head_dim,
            return_lse=False,
            solution=self.kernel_solution,
        )
        # DFLASH draft: expand decode metadata to spec_num_tokens rows/request
        # (whole block in one decode forward), with uniform non-causal seq_lens.
        self.draft_block_decode = bool(config.draft_block_decode)

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_extend_metadata: MHAExtendMetadata | None = None

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
        assert not forward_mode.is_mixed(), "mha backend does not support mixed batch"
        if not forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                "MHA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend ({forward_mode})"
            )
        assert extend_seq_lens is not None
        assert extend_seq_lens_cpu is not None
        assert extend_prefix_lens is not None
        assert extend_prefix_lens_cpu is not None

        seq_lens = seq_lens[:bs]

        page_tables = self._consumed_group_tables(block_tables)
        out_cache_locs = None
        if page_tables:
            # The cache path routes every read/write through the per-group
            # tables; a shared single page_table would be dead work.
            page_table = None
            out_cache_locs = self._compute_extend_out_cache_locs(
                page_tables,
                extend_prefix_lens_cpu[:bs],
                extend_seq_lens_cpu[:bs],
            )
            self._maybe_check_group_write_locs(page_tables, out_cache_locs)
            page_tables = self._kernel_page_tables(page_tables)
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
            page_tables=page_tables,
            out_cache_locs=out_cache_locs,
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
        # Per-group persistent buffers, parallels cuda_graph_page_table.
        # Initialized before the DFLASH early return: the base view builder
        # and the LCM published-groups contract check read the dict.
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
        # Own the cache-seqlens buffer in all non-DFLASH cases; replay copies the
        # live lengths in (verify clamps, plain decode/draft copies), so graph
        # state does not depend on the controller mutating a shared tensor.
        self.cuda_graph_seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )

    @property
    def _verify_floor(self) -> int:
        """Minimum per-row seq_len the decode metadata must present.

        The target's verify window spans seq-N..seq-1, so its rows need
        seq_len >= N; every other decode (plain, drafts) needs >= 1, where
        the clamp is the identity.
        """
        return self.spec_num_tokens if not self.is_draft else 1

    def _decode_views(
        self,
        bs: int,
        cache_group_ids: tuple[str, ...] | None = None,
    ) -> MHADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh: capture records these views into
        the graph, refresh (eager or replay) repoints forward_decode_metadata
        at them. Cached per bs — pointer-stable, no storage allocated.
        ``cache_group_ids`` pins the group set at capture time (a draft may
        consume a family subset of its buffers); lazy callers reuse it.
        """
        if cache_group_ids is not None:
            self._decode_view_group_ids = cache_group_ids
        metadata = self.cuda_graph_decode_metadata.get(bs)
        if metadata is not None:
            return metadata
        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: spec_num_tokens decode rows per request.
            expanded_bs = bs * self.spec_num_tokens
            metadata = MHADecodeMetadata(
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
                # Multi-token window locs for target verify AND window-mode
                # drafts alike (a chaining draft ignores them via
                # prefer_caller in the base forward_decode).
                tokens_per_req=self.spec_num_tokens,
            )
            metadata = MHADecodeMetadata(
                # Grouped captures route reads through the per-group tables;
                # the shared single page_table is never filled on that path,
                # so leave it None instead of a stale zero-buffer slice.
                page_table=(
                    None
                    if page_tables is not None
                    else self.cuda_graph_page_table[:bs, :]
                ),
                seq_lens=self.cuda_graph_seq_lens[:bs],
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
            )
        self.cuda_graph_decode_metadata[bs] = metadata
        return metadata

    # Capture is inherited (base default: bind_decode_views + idle refresh).

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
            # spec_num_tokens block rows. The drafter's page table is
            # batch-ordered (row i == batch position i, raw scheduler pages —
            # expand into kernel pages first). Under replay the block-end
            # seq_lens are re-derived inside the captured graph from the live
            # draft length (fill_block_decode_seq_lens); eager has no
            # in-graph writer, so fill them here the same way.
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
                "MHA decode without per-group capture buffers: the pool "
                "published no cache groups, which the LCM contract forbids"
            )
        # Clamp short rows (padded rows replay at seq_len 1) to the verify
        # floor: verify derives row lengths as seq - N + t + 1, which must
        # stay positive. Plain decode and drafts have floor 1 (identity);
        # drafters republish their in-loop edits through
        # advance_draft_forward_metadata each step.
        torch.clamp_min(
            seq_lens[:bs],
            self._verify_floor,
            out=self.cuda_graph_seq_lens[:bs],
        )
        if block_tables:
            self._fill_group_graph_buffers(
                bs,
                block_tables,
                self.cuda_graph_seq_lens,
                tokens_per_req=self.spec_num_tokens,
            )

        self.forward_decode_metadata = self._decode_views(bs)

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
            layer,
            self.forward_decode_metadata,
            out_cache_loc,
            prefer_caller=self.is_draft,
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
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
        # TODO: use a custom kernel to do downcast
        if self.is_fp8:
            q = q.to(self.kv_cache_dtype)
            k = k.to(self.kv_cache_dtype)
            v = v.to(self.kv_cache_dtype)

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
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
        if save_kv_cache:
            # KV store (incl. the mxfp8 quantize-on-store path) lives solely
            # in _save_kv_cache.
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            q = q.to(self.kv_cache_dtype)

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
            **scale_kwargs,
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
            # KV store (incl. the mxfp8 quantize-on-store path) lives solely
            # in _save_kv_cache.
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            q = q.to(self.kv_cache_dtype)

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
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _store_kv_mxfp8(self, layer, loc, token_to_kv_pool, k, v) -> None:
        """MXFP8 quantize-on-store: one fused launch when the pool supports
        it (bit-identical, PDL-chained), else the split 5-launch path.
        The sole caller (_save_kv_cache) has already trimmed k/v to loc."""
        fused = getattr(token_to_kv_pool, "quantize_and_set_kv_buffer", None)
        if fused is not None and fused(layer, loc, k, v):
            return
        k_q, k_sf = self._quantize_mxfp8_tokens(k)
        v_q, v_sf = self._quantize_mxfp8_tokens(v)
        token_to_kv_pool.set_kv_buffer(layer, loc, k_q, v_q, k_scale=k_sf, v_scale=v_sf)

    def _quantize_mxfp8_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token MXFP8: (fp8-e4m3 [T, H, D], UE8M0 scales [T, H, D // 32]).

        Accepts [T, H, D] or [T, H * D]; H is inferred so the same helper
        serves q (tp_q_head_num) and k/v (tp_kv_head_num).
        """
        t, d = x.shape[0], self.head_dim
        h = x.numel() // (t * d)
        # (A PDL triton variant measured 0.07 ms slower e2e at decode Q shapes; flashinfer stays)
        data, sf = quantize_mxfp8(x.reshape(t * h, d))
        return (
            data.view(t, h, d),
            sf.view(torch.float8_e8m0fnu).view(t, h, d // 32),
        )

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

        if self.is_mxfp8:
            # Quantize-on-store: fp8 data + per-token e8m0 scales into the paged interleaved layout
            self._store_kv_mxfp8(layer, out_cache_loc, token_to_kv_pool, k, v)
        elif self.is_fp8:
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
        # Hetero block sizes: page ids are in the LAYER'S group page-size units, so view at that size
        page_size = self._layer_page_size(layer)
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )
        return k_cache, v_cache

    def _consumer_page_size(self, group_id: str) -> int:
        return self._group_block_granularity(group_id)

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


for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
