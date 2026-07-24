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

"""
MHA attention backend for TokenSpeed scheduling.
Uses fused kernels optimized for SM100 (Blackwell).
Supports sliding window, attention sinks, and FP8 KV cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_context_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    gather_page_table_with_padding,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    is_breakable_capture_active,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.flat_groups import (
    FlatCacheGroupsMixin,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.common import fp8_cast_contiguous
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = get_colorful_logger(__name__)

# Workspace buffer shared across all trtllm_mha wrappers.
_global_workspace_buffer: torch.Tensor | None = None
TRTLLM_MHA_WORKSPACE = 512 * 1024 * 1024


def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """Adjust degenerate strides for a tensor, make it canonical.

    When a dimension has size=1, PyTorch may use the same stride as the next dim.
    This causes TMA desc validation failures in the trtllm_mha backend.
    See: https://github.com/flashinfer-ai/flashinfer/issues/2232
    """
    sizes = tensor.size()
    strides = tensor.stride()
    ndim = tensor.dim()

    need_fix = any(
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)
    )

    if not need_fix:
        return tensor

    new_strides = [0] * ndim
    new_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]

    return tensor.as_strided(sizes, new_strides)


@dataclass
class TRTLLMMHAMetadata:
    cache_seqlens_int32: torch.Tensor = None
    max_seq_len_q: int = 1
    max_seq_len_k: int = 0
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    # page_table is None on the flat path (per-group page_tables route reads).
    page_table: torch.Tensor = None
    # Flat per-group tables/write-locs, keyed by group id (see flat_groups).
    page_tables: dict[str, torch.Tensor] | None = None
    block_table_base_offsets: dict[str, torch.Tensor] | None = None
    out_cache_locs: dict[str, torch.Tensor] | None = None


class TRTLLMMHAAttnBackend(FlatCacheGroupsMixin, AttentionBackend):
    """trtllm_mha attention backend optimized for SM100 (Blackwell)."""

    # Per-group flat tables: reads and writes route by layer.group_id, so
    # slab-aliased pools (e.g. gpt-oss sliding+full pairing) are safe.
    uses_flat_cache_groups: bool = True
    # Graph-buffer column tails pad with the zero-init dummy page, matching
    # the radix replay contract (gather_page_table_with_padding dummy_slot=0).
    flat_table_tail_pad: int = 0

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        # Under a breakable prefill-graph capture the prewrite would bake this
        # forward's write locations into the graph (stale on every replay;
        # dummy-page locs on the flat path) -- bake the non-prewrite branch
        # instead: the eager attention break writes KV from fresh metadata.
        if is_breakable_capture_active():
            return False
        return True

    def _prewrite_metadata(self, forward_mode):
        # Prewrite fires on extend too (unlike MHA): route it to the slot
        # init_forward_metadata filled for this forward. Target verify is
        # DECODE mode but its multi-token metadata lives in the prefill slot.
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            return self.forward_prefill_metadata
        if self.spec_num_tokens > 1 and not self.is_draft:
            return self.forward_prefill_metadata
        return self.forward_decode_metadata

    @property
    def sinks_dtype(self) -> torch.dtype:
        return torch.float32

    def __init__(self, config: MHAConfig):
        super().__init__(config)

        self.page_size = config.page_size
        self.max_context_len = config.context_len
        self.kv_cache_dtype = config.kv_cache_dtype
        max_bs = config.max_bs

        # Shared workspace buffer (allocated once per process).
        global _global_workspace_buffer
        if _global_workspace_buffer is None:
            _global_workspace_buffer = torch.zeros(
                TRTLLM_MHA_WORKSPACE,
                dtype=torch.uint8,
                device=config.device,
            )
        self.workspace_buffer = _global_workspace_buffer

        # Max pages per request.
        self.max_num_pages = (config.context_len + self.page_size - 1) // self.page_size

        # Persistent buffers for page table construction.
        self.page_table_buf = torch.zeros(
            (max_bs, self.max_num_pages),
            dtype=torch.int32,
            device=config.device,
        )
        self.cache_seqlens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=config.device
        )
        # KV seqlens clamped to >= spec_num_tokens for the MTP verify path.
        # Padded decode rows have seq_len=1 (InputBuffer); with q_len=spec_num_tokens
        # they'd hit an empty causal span and the kernel returns NaN. Mirrors mha.py.
        self.spec_cache_seqlens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=config.device
        )
        self.cu_seqlens_q_buf = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )
        self.cu_seqlens_k_buf = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )

        # Separate slots for prefill-kernel vs decode-kernel forward paths.
        # forward_extend reads prefill; forward_decode reads decode.
        self.forward_prefill_metadata: TRTLLMMHAMetadata | None = None
        self.forward_decode_metadata: TRTLLMMHAMetadata | None = None

        # CUDA graph state — per-slot dicts.
        self.cuda_graph_prefill_metadata: dict[int, TRTLLMMHAMetadata] = {}
        self.cuda_graph_decode_metadata: dict[int, TRTLLMMHAMetadata] = {}

        # DFLASH draft: the drafter predicts a whole block of spec_num_tokens
        # per decode forward and needs non-causal (block-diffusion) attention.
        # Instead of a non-causal mask, expand each request into spec_num_tokens
        # single-query rows sharing the SAME block-end seq_len, so each row
        # attends over the whole block. Mirrors the MHA draft_block_decode path;
        # target verify and ordinary trtllm decode are untouched.
        self.draft_block_decode = bool(getattr(config, "draft_block_decode", False))

    # ------------------------------------------------------------------
    # Page table helpers
    # ------------------------------------------------------------------

    def _build_page_table(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        bs: int,
        req_to_page: torch.Tensor,
        page_table_buf: torch.Tensor,
    ) -> torch.Tensor:
        """Build page table in [bs, max_pages] format from req_to_page.

        req_to_page is [req_pool_size+1, max_pages] containing page IDs.
        """
        page_table_buf[:bs].copy_(
            req_to_page[req_pool_indices[:bs], : self.max_num_pages]
        )
        return page_table_buf[:bs]

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def _get_kv_cache_permuted(self, layer: PagedAttention, token_to_kv_pool):
        """Get KV cache in [num_pages, num_kv_heads, page_size, head_dim] layout."""
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        return k_cache, v_cache

    def _compute_scales(self, layer: PagedAttention):
        """Compute bmm1/bmm2 scales for the fused kernel."""
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0
        return bmm1_scale, bmm2_scale

    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k) -> bool:
        return (
            save_kv_cache
            and k is not None
            and self.kv_cache_dtype == torch.float8_e4m3fn
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _save_kv_and_prepare_q(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, save_kv_cache
    ):
        k, v = self._trim_kv_to_locs(out_cache_loc, k, v)
        if self._should_use_fused_fp8_path(save_kv_cache, k):
            k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            fused_fp8_set_kv_buffer(
                k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v=v.view(-1, layer.tp_k_head_num, layer.head_dim),
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self.page_size,
            )
        elif save_kv_cache and k is not None:
            token_to_kv_pool.set_kv_buffer(
                layer, out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        if self.kv_cache_dtype == torch.float8_e4m3fn:
            q = fp8_cast_contiguous(q)
        else:
            q = q.contiguous()

        return q.view(-1, layer.tp_q_head_num, layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: metadata is expanded to bs*spec_num_tokens
            # single-query rows, so use the decode slot directly. Inferring
            # q_len_per_req from q.shape[0]//bs would be spec_num_tokens and
            # wrongly pick the prefill slot.
            metadata = self.forward_decode_metadata
        else:
            # Multi-token decode (q_len > 1) reads the prefill slot's
            # uniform-stride metadata; plain decode reads the single-token slot.
            q_len_per_req = q.shape[0] // bs if bs > 0 else 1
            metadata = (
                self.forward_prefill_metadata
                if q_len_per_req > 1
                else self.forward_decode_metadata
            )

        out_cache_loc = self._select_out_cache_loc(
            layer, metadata, out_cache_loc, prefer_caller=self.is_draft
        )
        q = self._save_kv_and_prepare_q(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, save_kv_cache
        )
        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        bmm1_scale, bmm2_scale = self._compute_scales(layer)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        o = trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=self._select_page_table(layer, metadata),
            seq_lens=metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            out_dtype=self.dtype,
            q_len_per_req=metadata.max_seq_len_q,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        metadata = self.forward_prefill_metadata
        out_cache_loc = self._select_out_cache_loc(layer, metadata, out_cache_loc)
        q = self._save_kv_and_prepare_q(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, save_kv_cache
        )
        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        bmm1_scale, bmm2_scale = self._compute_scales(layer)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        o = trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=self._select_page_table(layer, metadata),
            seq_lens=metadata.cache_seqlens_int32,
            max_q_len=metadata.max_seq_len_q,
            max_kv_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=metadata.cu_seqlens_q.shape[0] - 1,
            cum_seq_lens_q=metadata.cu_seqlens_q,
            cum_seq_lens_kv=metadata.cu_seqlens_k,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            out_dtype=self.dtype,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    # ------------------------------------------------------------------
    # Metadata initialisation
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        use_cuda_graph: bool = False,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        flat_block_table_base_offsets: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        flat_page_tables, flat_base_offsets = self._shed_state_group_inputs(
            flat_block_tables, flat_block_table_base_offsets
        )
        flat_out_cache_locs = None
        if flat_page_tables:
            # Verify keeps [bs]-row tables; only DFLASH expands rows. TODO(flat+dflash).
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "flat cache groups are unsupported with DFLASH block decode"
            if forward_mode.is_extend_or_mixed():
                assert extend_prefix_lens_cpu is not None
                assert extend_seq_lens_cpu is not None
                flat_out_cache_locs = self._compute_flat_extend_out_cache_locs(
                    flat_page_tables,
                    flat_base_offsets,
                    extend_prefix_lens_cpu[:bs],
                    extend_seq_lens_cpu[:bs],
                    self.page_size,
                )
            else:
                flat_out_cache_locs = self._compute_flat_decode_out_cache_locs(
                    flat_page_tables,
                    flat_base_offsets,
                    seq_lens[:bs],
                    self.page_size,
                    self._flat_verify_tokens(),
                )
            self._maybe_check_flat_write_locs(
                flat_page_tables, flat_out_cache_locs, self.page_size
            )

        if forward_mode.is_extend_or_mixed():
            self._init_extend_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                req_to_page,
                extend_with_prefix=extend_with_prefix,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                flat_page_tables=flat_page_tables,
                flat_base_offsets=flat_base_offsets,
                flat_out_cache_locs=flat_out_cache_locs,
            )
            # Drafter: also fill decode_metadata so step 1+ multi-step has
            # metadata under EXTEND/MIXED target. seq_lens is the drafter's
            # live alias buffer (wrapper pre-writes before this call).
            if self.is_draft:
                self._init_decode_metadata(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    req_to_page,
                    flat_page_tables=flat_page_tables,
                    flat_base_offsets=flat_base_offsets,
                    flat_out_cache_locs=flat_out_cache_locs,
                )
            return

        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block (eager): expand to spec_num_tokens single-query
            # rows per request; seq_lens is the block-end length the drafter
            # already wrote.
            self._init_block_decode_metadata(
                bs, req_pool_indices, seq_lens, req_to_page
            )
            return

        if self.spec_num_tokens > 1:
            self._init_multi_token_metadata(
                bs,
                self.spec_num_tokens,
                req_pool_indices,
                seq_lens,
                req_to_page,
                flat_page_tables=flat_page_tables,
                flat_base_offsets=flat_base_offsets,
                flat_out_cache_locs=flat_out_cache_locs,
            )
            if self.is_draft:
                # Drafter's N-1 single-token steps after the first.
                self._init_decode_metadata(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    req_to_page,
                    flat_page_tables=flat_page_tables,
                    flat_base_offsets=flat_base_offsets,
                    flat_out_cache_locs=flat_out_cache_locs,
                )
        else:
            self._init_decode_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                req_to_page,
                flat_page_tables=flat_page_tables,
                flat_base_offsets=flat_base_offsets,
                flat_out_cache_locs=flat_out_cache_locs,
            )

    def _init_decode_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        flat_page_tables: dict[str, torch.Tensor] | None = None,
        flat_base_offsets: dict[str, torch.Tensor] | None = None,
        flat_out_cache_locs: dict[str, torch.Tensor] | None = None,
    ):
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        device = seq_lens.device
        # Alias seq_lens (no copy, no mutation). cu_seqlens_k omitted:
        # the decode kernel doesn't read it. On the flat path the per-group
        # tables route every read; the radix single table would be dead work.
        self.forward_decode_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=seq_lens[:bs],
            max_seq_len_q=1,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(0, bs + 1, dtype=torch.int32, device=device),
            page_table=(
                None
                if flat_page_tables
                else self._build_page_table(
                    req_pool_indices, seq_lens, bs, req_to_page, self.page_table_buf
                )
            ),
            page_tables=flat_page_tables,
            block_table_base_offsets=flat_base_offsets,
            out_cache_locs=flat_out_cache_locs,
        )

    def _replicate_block_page_table(
        self,
        out: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int,
        req_to_page: torch.Tensor,
    ) -> None:
        """Replicate each request's page table to its spec_num_tokens block rows.

        ``out`` is the [bs*spec_num_tokens, max_num_pages] destination. All block
        rows of a request share its pages (decode only reads KV), so a single
        broadcast copy suffices.
        """
        spec = self.spec_num_tokens
        base_page_table = req_to_page[req_pool_indices[:bs], : self.max_num_pages]
        out[: bs * spec, :].view(bs, spec, self.max_num_pages).copy_(
            base_page_table[:, None, :]
        )

    def _init_block_decode_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
    ):
        """Eager DFLASH draft-block metadata: spec_num_tokens single-query rows
        per request, all carrying the block-end seq_len (prefix + spec_num_tokens)
        so each query attends over the whole block. Allocates fresh buffers (the
        cuda-graph path uses persistent ones), mirroring the MHA backend.
        """
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        spec = self.spec_num_tokens
        device = seq_lens.device
        expanded_bs = bs * spec

        page_table = torch.empty(
            (expanded_bs, self.max_num_pages), dtype=torch.int32, device=device
        )
        self._replicate_block_page_table(page_table, req_pool_indices, bs, req_to_page)

        # Clamp the block-end length so the decode never asks for more page-table
        # columns than exist (prefix + spec_num_tokens can exceed max_context_len).
        cache_seqlens = (
            seq_lens[:bs]
            .clamp(spec, self.max_context_len)
            .unsqueeze(1)
            .expand(bs, spec)
            .reshape(expanded_bs)
            .contiguous()
        )

        self.forward_decode_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=cache_seqlens,
            max_seq_len_q=1,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(
                0, expanded_bs + 1, dtype=torch.int32, device=device
            ),
            page_table=page_table,
        )

    def _clamped_spec_seqlens(
        self, seq_lens: torch.Tensor, bs: int, spec_num_tokens: int
    ) -> torch.Tensor:
        """Return KV seqlens clamped to >= spec_num_tokens for the MTP verify path.

        Writes into the persistent spec_cache_seqlens_buf (CUDA-graph safe)
        to avoid NaN from empty causal spans on padded rows (seq_len=1).
        """
        dst = self.spec_cache_seqlens_buf[:bs]
        torch.clamp_min(seq_lens[:bs], spec_num_tokens, out=dst)
        return dst

    def _flat_verify_tokens(self) -> int:
        # Write locs per request on the flat path: N for target verify
        # ([bs*N], token-major), 1 elsewhere (draft chains use caller locs).
        return (
            self.spec_num_tokens
            if self.spec_num_tokens > 1 and not self.is_draft
            else 1
        )

    def _init_multi_token_metadata(
        self,
        bs: int,
        spec_num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        flat_page_tables: dict[str, torch.Tensor] | None = None,
        flat_base_offsets: dict[str, torch.Tensor] | None = None,
        flat_out_cache_locs: dict[str, torch.Tensor] | None = None,
    ):
        """Prefill-slot metadata for multi-token decode (uniform q_len per
        request). Routes through the decode kernel via q_len_per_req; the
        kernel doesn't read cu_seqlens_k."""
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        device = seq_lens.device
        self.forward_prefill_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=self._clamped_spec_seqlens(
                seq_lens, bs, spec_num_tokens
            ),
            max_seq_len_q=spec_num_tokens,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(
                0,
                bs * spec_num_tokens + 1,
                spec_num_tokens,
                dtype=torch.int32,
                device=device,
            ),
            page_table=(
                None
                if flat_page_tables
                else self._build_page_table(
                    req_pool_indices, seq_lens, bs, req_to_page, self.page_table_buf
                )
            ),
            page_tables=flat_page_tables,
            block_table_base_offsets=flat_base_offsets,
            out_cache_locs=flat_out_cache_locs,
        )

    def _init_extend_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu=None,
        extend_seq_lens_cpu=None,
        flat_page_tables: dict[str, torch.Tensor] | None = None,
        flat_base_offsets: dict[str, torch.Tensor] | None = None,
        flat_out_cache_locs: dict[str, torch.Tensor] | None = None,
    ):
        """Populate prefill slot for regular EXTEND (ragged query)."""
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        assert (
            extend_seq_lens_cpu is not None
        ), "trtllm extend requires extend_seq_lens_cpu (pinned-CPU mirror) to avoid GPU sync"
        cache_seqlens_int32 = seq_lens[:bs]
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        # Flat path: per-group tables route every read (see _init_decode_metadata).
        page_table = (
            None
            if flat_page_tables
            else self._build_page_table(
                req_pool_indices, seq_lens, bs, req_to_page, self.page_table_buf
            )
        )

        # Read the max from the pinned-CPU mirror — avoids a per-iter
        # GPU->CPU sync that would block the host on the previous step's
        # forward and erase prefill/decode overlap. Both branches want
        # max(new tokens per request); for a no-prefix extend that's
        # seq_lens, for a prefix-cached extend it's seq_lens-prefix_lens —
        # extend_seq_lens_cpu holds those new-token counts in either case.
        max_seq_len_q = int(extend_seq_lens_cpu[:bs].max().item())

        if extend_with_prefix and (
            (extend_prefix_lens_cpu is not None and any(extend_prefix_lens_cpu))
            or (extend_prefix_lens is not None and any(extend_prefix_lens.tolist()))
        ):
            if extend_prefix_lens is None:
                raise RuntimeError(
                    "TRTLLMMHAAttnBackend requires extend_prefix_lens tensor "
                    "when extend_with_prefix is true."
                )
            extend_seq_lens = seq_lens - extend_prefix_lens
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
        else:
            cu_seqlens_q = cu_seqlens_k

        self.forward_prefill_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table,
            page_tables=flat_page_tables,
            block_table_base_offsets=flat_base_offsets,
            out_cache_locs=flat_out_cache_locs,
        )

    # ------------------------------------------------------------------
    # CUDA graph support
    # ------------------------------------------------------------------

    def init_cuda_graph_state(
        self,
        max_bs: int,
        seq_lens_buf: torch.Tensor,
        paged_cache_group_specs: Sequence = (),
        **kwargs,
    ):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )
        self.cuda_graph_prefill_metadata = {}
        self.cuda_graph_decode_metadata = {}
        # Flat per-group persistent buffers + state-group shed; before the
        # DFLASH early return (replay reads the dict for the stale guard).
        self._learn_flat_state_groups(paged_cache_group_specs)
        self._init_flat_graph_buffers(max_bs)
        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: spec_num_tokens decode rows per request. Unlike
            # the plain path, cache_seqlens is a dedicated buffer (NOT aliasing
            # seq_lens_buf): it is filled in-graph by fill_block_decode_seq_lens.
            self.cuda_graph_page_table = torch.zeros(
                (max_bs * self.spec_num_tokens, self.max_num_pages),
                dtype=torch.int32,
                device=self.device,
            )
            self.cuda_graph_cache_seqlens = torch.full(
                (max_bs * self.spec_num_tokens,),
                self.spec_num_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            return
        # Alias controller's seq_lens_buf — backend never mutates it.
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cache_seqlens = seq_lens_buf

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        flat_cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"trtllm CUDA graph capture not supported for {forward_mode}"
            )

        # Real tables only arrive at replay; capture records metadata views
        # into the persistent per-group buffers.
        if flat_cache_group_ids:
            # Verify keeps [bs]-row tables + [bs*N] loc views. TODO(flat+dflash).
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "flat_cache_group_ids is unsupported with DFLASH block decode"
        page_tables, block_table_base_offsets, out_cache_locs = (
            self._flat_capture_group_views(
                bs,
                flat_cache_group_ids,
                tokens_per_req=self._flat_verify_tokens(),
            )
        )

        if self.draft_block_decode and self.spec_num_tokens > 1:
            self._init_block_decode_metadata_capture(bs)
            return

        if self.spec_num_tokens > 1:
            self._init_multi_token_metadata_capture(
                bs,
                self.spec_num_tokens,
                page_tables,
                block_table_base_offsets,
                out_cache_locs,
            )
            if self.is_draft:
                self._init_decode_metadata_capture(
                    bs,
                    seq_lens,
                    page_tables,
                    block_table_base_offsets,
                    out_cache_locs,
                )
        else:
            self._init_decode_metadata_capture(
                bs,
                seq_lens,
                page_tables,
                block_table_base_offsets,
                out_cache_locs,
            )

    def _init_block_decode_metadata_capture(self, bs: int):
        """DFLASH draft block (cuda-graph capture): spec_num_tokens single-query
        rows per request over the persistent expanded buffers. seq_lens are
        filled in-graph by fill_block_decode_seq_lens; seed a safe baseline here
        so the capture run stays in range before that op records."""
        expanded_bs = bs * self.spec_num_tokens
        self.cuda_graph_cache_seqlens[:expanded_bs].fill_(self.spec_num_tokens)
        metadata = self.cuda_graph_decode_metadata.get(bs)
        if metadata is None:
            metadata = TRTLLMMHAMetadata(
                cache_seqlens_int32=self.cuda_graph_cache_seqlens[:expanded_bs],
                max_seq_len_q=1,
                max_seq_len_k=self.max_context_len,
                cu_seqlens_q=torch.arange(
                    0, expanded_bs + 1, dtype=torch.int32, device=self.device
                ),
                page_table=self.cuda_graph_page_table[:expanded_bs, :],
            )
        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _init_decode_metadata_capture(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        page_tables: dict[str, torch.Tensor] | None = None,
        block_table_base_offsets: dict[str, torch.Tensor] | None = None,
        out_cache_locs: dict[str, torch.Tensor] | None = None,
    ):
        # cache_seqlens aliases seq_lens_buf (set in init_cuda_graph_state).
        # Flat captures route reads through the per-group buffer views and
        # replay never fills the radix single table, so record page_table=None
        # instead of a slice of the never-filled zero buffer.
        cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
        page_table = (
            None if page_tables is not None else self.cuda_graph_page_table[:bs, :]
        )
        metadata = self.cuda_graph_decode_metadata.get(bs)
        if metadata is None:
            metadata = TRTLLMMHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                max_seq_len_q=1,
                max_seq_len_k=self.max_context_len,
                cu_seqlens_q=torch.arange(
                    0, bs + 1, dtype=torch.int32, device=self.device
                ),
                page_table=page_table,
                page_tables=page_tables,
                block_table_base_offsets=block_table_base_offsets,
                out_cache_locs=out_cache_locs,
            )
        else:
            metadata.cache_seqlens_int32 = cache_seqlens
            metadata.page_table = page_table
            metadata.page_tables = page_tables
            metadata.block_table_base_offsets = block_table_base_offsets
            metadata.out_cache_locs = out_cache_locs
        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _init_multi_token_metadata_capture(
        self,
        bs: int,
        spec_num_tokens: int,
        page_tables: dict[str, torch.Tensor] | None = None,
        block_table_base_offsets: dict[str, torch.Tensor] | None = None,
        out_cache_locs: dict[str, torch.Tensor] | None = None,
    ):
        # Multi-token decode: seed spec_cache_seqlens_buf (clamped to >=
        # spec_num_tokens) at capture so padded rows (seq_len=1) avoid NaN.
        # The replay path refreshes it each step.
        cache_seqlens = self._clamped_spec_seqlens(
            self.cuda_graph_cache_seqlens, bs, spec_num_tokens
        )
        page_table = (
            None if page_tables is not None else self.cuda_graph_page_table[:bs, :]
        )
        metadata = self.cuda_graph_prefill_metadata.get(bs)
        if metadata is None:
            metadata = TRTLLMMHAMetadata(
                cache_seqlens_int32=cache_seqlens,
                max_seq_len_q=spec_num_tokens,
                max_seq_len_k=self.max_context_len,
                cu_seqlens_q=torch.arange(
                    0,
                    bs * spec_num_tokens + 1,
                    spec_num_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                page_table=page_table,
                page_tables=page_tables,
                block_table_base_offsets=block_table_base_offsets,
                out_cache_locs=out_cache_locs,
            )
        else:
            metadata.cache_seqlens_int32 = cache_seqlens
            metadata.page_table = page_table
            metadata.page_tables = page_tables
            metadata.block_table_base_offsets = block_table_base_offsets
            metadata.out_cache_locs = out_cache_locs
        self.cuda_graph_prefill_metadata[bs] = metadata
        self.forward_prefill_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor = None,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        flat_block_table_base_offsets: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"trtllm CUDA graph replay not supported for {forward_mode}"
            )

        # Fail loudly instead of replaying over stale/zero page tables.
        validated_flat_group_ids = self._flat_replay_stale_guard(
            bs, flat_block_tables, flat_block_table_base_offsets
        )

        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: replicate the page table to each request's
            # block rows. seq_lens are re-derived in-graph, so not touched here.
            if req_to_page is not None:
                self._replicate_block_page_table(
                    self.cuda_graph_page_table, req_pool_indices, bs, req_to_page
                )
            if bs in self.cuda_graph_decode_metadata:
                self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]
            return

        # cache_seqlens aliases seq_lens_buf; only page tables need refresh.
        # Flat captures read only the per-group buffers; the radix single
        # table would be dead work there (and req_to_page is unpopulated on
        # a flat scheduler build).
        if not self.cuda_graph_flat_page_tables and req_to_page is not None:
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
        if flat_block_tables:
            # cuda_graph_cache_seqlens aliases the controller's seq_lens_buf,
            # filled by input prep BEFORE this call.
            self._flat_replay_fill(
                bs,
                flat_block_tables,
                flat_block_table_base_offsets,
                self.cuda_graph_cache_seqlens,
                tokens_per_req=self._flat_verify_tokens(),
                validated_group_ids=validated_flat_group_ids,
            )

        # Refresh for both verify and draft: draft step 1 is multi-token
        # and reads spec_cache_seqlens_buf; later single-token steps don't.
        if self.spec_num_tokens > 1:
            self._clamped_spec_seqlens(seq_lens, bs, self.spec_num_tokens)

        if bs in self.cuda_graph_prefill_metadata:
            self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]
        if bs in self.cuda_graph_decode_metadata:
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast each request's block-end length to its
        spec_num_tokens cuda-graph decode rows.

        Called by the drafter inside the captured graph so every replay
        re-derives cache_seqlens from the live draft length. Mirrors the MHA
        backend method of the same name.

        Args:
            bs: Number of draft requests.
            block_seq_lens: ``[bs]`` per-request block-end lengths.
        """
        spec = self.spec_num_tokens
        self.cuda_graph_cache_seqlens[: bs * spec].view(bs, spec).copy_(
            block_seq_lens[:bs].clamp(spec, self.max_context_len).unsqueeze(1)
        )


register_backend("trtllm", {AttentionArch.MHA}, TRTLLMMHAAttnBackend)
