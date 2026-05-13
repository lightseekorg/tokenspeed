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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_context_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.trtllm_fp8_kv_kernel import (
    fused_fp8_set_kv_buffer,
)
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
    page_table: torch.Tensor = None


class TRTLLMMHAAttnBackend(AttentionBackend):
    """trtllm_mha attention backend optimized for SM100 (Blackwell)."""

    @property
    def support_kv_cache_prewrite(self) -> bool:
        return True

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

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
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

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        bmm1_scale, bmm2_scale = self._compute_scales(layer)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        metadata = self.forward_decode_metadata
        o = trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            out_dtype=self.dtype,
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
        forward_mode: ForwardMode,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
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

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        bmm1_scale, bmm2_scale = self._compute_scales(layer)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        # target_verify and draft_extend both produce uniform spec-length
        # query batches (init_forward_metadata calls _init_target_verify_metadata
        # for both), so route both through the latency-tagged decode kernel.
        use_decode_kernel = (
            forward_mode.is_target_verify() or forward_mode.is_draft_extend()
        )

        metadata = self.forward_prefill_metadata

        if use_decode_kernel:

            o = trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=(k_cache, v_cache),
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens_int32,
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=layer.sliding_window_size,
                sinks=attention_sink,
                out_dtype=self.dtype,
                q_len_per_req=metadata.max_seq_len_q,
            )

        else:

            o = trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=(k_cache, v_cache),
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
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
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        spec_info=None,
        use_cuda_graph: bool = False,
        **kwargs,
    ):
        if forward_mode.is_decode():
            self._init_decode_metadata(bs, req_pool_indices, seq_lens, req_to_page)
        elif forward_mode.is_target_verify():
            spec_num_tokens = num_tokens // bs if bs > 0 else 1
            self._init_target_verify_metadata(
                bs, spec_num_tokens, req_pool_indices, seq_lens, req_to_page
            )
        elif forward_mode.is_draft_extend():
            # Compound: draft's step 0 uses prefill kernel (multi-token),
            # steps 1..N-1 use decode kernel (single-token). Populate both
            # slots in one call.
            spec_num_tokens = num_tokens // bs if bs > 0 else 1
            self._init_target_verify_metadata(
                bs, spec_num_tokens, req_pool_indices, seq_lens, req_to_page
            )
            self._init_decode_metadata(bs, req_pool_indices, seq_lens, req_to_page)
        elif forward_mode.is_extend():
            self._init_extend_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                req_to_page,
                extend_with_prefix=extend_with_prefix,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
            )
        else:
            raise NotImplementedError(f"Unsupported forward mode: {forward_mode}")

    def _init_decode_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
    ):
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        device = seq_lens.device
        # Alias seq_lens (no copy, no mutation). cu_seqlens_k omitted:
        # the decode kernel doesn't read it.
        self.forward_decode_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=seq_lens[:bs],
            max_seq_len_q=1,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(0, bs + 1, dtype=torch.int32, device=device),
            page_table=self._build_page_table(
                req_pool_indices, seq_lens, bs, req_to_page, self.page_table_buf
            ),
        )

    def _init_target_verify_metadata(
        self,
        bs: int,
        spec_num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
    ):
        """Prefill slot for TARGET_VERIFY / DRAFT_EXTEND. Routes through the
        decode kernel (q_len_per_req), which doesn't read cu_seqlens_k."""
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        device = seq_lens.device
        self.forward_prefill_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=seq_lens[:bs],
            max_seq_len_q=spec_num_tokens,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(
                0,
                bs * spec_num_tokens + 1,
                spec_num_tokens,
                dtype=torch.int32,
                device=device,
            ),
            page_table=self._build_page_table(
                req_pool_indices, seq_lens, bs, req_to_page, self.page_table_buf
            ),
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
        page_table = self._build_page_table(
            req_pool_indices, seq_lens, bs, req_to_page, self.page_table_buf
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
        )

    # ------------------------------------------------------------------
    # CUDA graph support
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
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
        # Alias controller's seq_lens_buf — backend never mutates it.
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cache_seqlens = seq_lens_buf

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        if forward_mode.is_decode():
            self._init_decode_metadata_capture(bs, seq_lens)
        elif forward_mode.is_target_verify():
            spec_num_tokens = num_tokens // bs if bs > 0 else 1
            self._init_target_verify_metadata_capture(bs, spec_num_tokens, seq_lens)
        elif forward_mode.is_draft_extend():
            # Compound: capture both prefill (step 0) and decode (steps 1..N-1).
            spec_num_tokens = num_tokens // bs if bs > 0 else 1
            self._init_target_verify_metadata_capture(bs, spec_num_tokens, seq_lens)
            self._init_decode_metadata_capture(bs, seq_lens)
        else:
            raise NotImplementedError(
                f"trtllm CUDA graph capture not supported for {forward_mode}"
            )

    def _init_decode_metadata_capture(self, bs: int, seq_lens: torch.Tensor):
        # cache_seqlens aliases seq_lens_buf (set in init_cuda_graph_state).
        metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=self.cuda_graph_cache_seqlens[:bs],
            max_seq_len_q=1,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(0, bs + 1, dtype=torch.int32, device=self.device),
            page_table=self.cuda_graph_page_table[:bs, :],
        )
        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _init_target_verify_metadata_capture(
        self, bs: int, spec_num_tokens: int, seq_lens: torch.Tensor
    ):
        # cache_seqlens aliases seq_lens_buf; routes through the decode kernel.
        metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=self.cuda_graph_cache_seqlens[:bs],
            max_seq_len_q=spec_num_tokens,
            max_seq_len_k=self.max_context_len,
            cu_seqlens_q=torch.arange(
                0,
                bs * spec_num_tokens + 1,
                spec_num_tokens,
                dtype=torch.int32,
                device=self.device,
            ),
            page_table=self.cuda_graph_page_table[:bs, :],
        )
        self.cuda_graph_prefill_metadata[bs] = metadata
        self.forward_prefill_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        # cache_seqlens aliases seq_lens_buf; only page_table needs refresh.
        if req_to_page is not None:
            torch.index_select(
                req_to_page[:, : self.max_num_pages],
                0,
                req_pool_indices[:bs].long(),
                out=self.cuda_graph_page_table[:bs, : self.max_num_pages],
            )

        if forward_mode.is_decode():
            self._init_decode_metadata_replay(bs)
        elif forward_mode.is_target_verify():
            self._init_target_verify_metadata_replay(bs)
        elif forward_mode.is_draft_extend():
            self._init_target_verify_metadata_replay(bs)
            self._init_decode_metadata_replay(bs)
        else:
            raise NotImplementedError(
                f"trtllm CUDA graph replay not supported for {forward_mode}"
            )

    def _init_decode_metadata_replay(self, bs: int):
        self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]

    def _init_target_verify_metadata_replay(self, bs: int):
        self.forward_prefill_metadata = self.cuda_graph_prefill_metadata[bs]


register_backend("trtllm", {AttentionArch.MHA}, TRTLLMMHAAttnBackend)
