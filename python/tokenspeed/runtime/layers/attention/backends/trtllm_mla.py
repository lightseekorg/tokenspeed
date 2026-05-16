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
MLA attention backend for TokenSpeed scheduling.

Uses fused kernels optimized for SM100 (Blackwell) GPUs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_decode_with_kv_cache_mla,
    trtllm_ragged_attention_deepseek,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = logging.getLogger(__name__)

# Block constraint from flashinfer: block_num % (128 / page_size) == 0
TRTLLM_BLOCK_CONSTRAINT = 128

# Shared workspace buffer for fused kernels (150 MB, zero-initialized).
# Zero-init is required for the kernel's internal semaphore mechanism.
_trtllm_workspace_buffer = None


def get_trtllm_workspace_buffer(device):
    """Get or create the shared fused-kernel workspace buffer."""
    global _trtllm_workspace_buffer
    if _trtllm_workspace_buffer is None:
        _trtllm_workspace_buffer = torch.zeros(
            150 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    return _trtllm_workspace_buffer


@dataclass
class TRTLLMMLAPrefillMetadata:
    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class TRTLLMMLAChunkedPrefillMetadata:
    extend_prefix_lens: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    cum_extend_seq_lens: torch.Tensor  # cumsum prefix-padded, sized num_extends+1
    max_extend_seq_len: int
    # Per-prefix-chunk arrays for non-causal cross-attention (built once per
    # iteration in _init_prefill_metadata, indexed by loop_idx in the model).
    chunked_loop_num: int
    chunk_kv_indices_list: list  # List[torch.Tensor], one per loop_idx
    chunked_seq_len: torch.Tensor  # (chunked_loop_num, num_extends) int32 GPU
    cu_chunked_seq_len: torch.Tensor  # (chunked_loop_num, num_extends+1) int32 GPU
    max_chunk_len_per_loop: list  # List[int], one per loop_idx


@dataclass
class TRTLLMMLADecodeMetadata:
    block_kv_indices: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None


class TRTLLMMLABackend(AttentionBackend):
    """trtllm_mla attention backend using fused kernels."""

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        self.max_context_len = config.context_len
        self.page_size = config.page_size

        # MLA dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype

        # Workspace zero-initialized for the fused kernel semaphore.
        self.trtllm_workspace = get_trtllm_workspace_buffer(config.device)

        # Validate page_size
        if self.page_size not in (32, 64):
            raise ValueError(
                f"trtllm_mla backend requires page_size 32 or 64, got {self.page_size}"
            )

        self.num_local_heads = config.num_attention_heads // config.attn_tp_size

        # Metadata
        self.forward_decode_metadata: TRTLLMMLADecodeMetadata | None = None
        self.forward_prefill_metadata: TRTLLMMLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, TRTLLMMLADecodeMetadata] = {}
        self.decode_cuda_graph_kv_indices = None
        self.chunked_prefill_metadata: TRTLLMMLAChunkedPrefillMetadata | None = None

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """Calculate block count padded to satisfy the fused-kernel constraint."""
        blocks = triton.cdiv(max_seq_len, self.page_size)
        constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        if blocks % constraint != 0:
            blocks = triton.cdiv(blocks, constraint) * constraint
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        block_kv_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build page-table from req_to_page using vectorized tensor indexing."""
        if block_kv_indices is None:
            block_kv_indices = torch.zeros(
                (batch_size, max_blocks), dtype=torch.int32, device=self.device
            )

        copy_len = min(max_blocks, req_to_page.shape[1])

        # Vectorized: gather all rows at once, no Python loop.
        # Pages beyond actual seq_len are 0 (from req_to_page init); the kernel
        # uses seq_lens to bound access so these padding entries are never read.
        block_kv_indices[:batch_size, :copy_len] = req_to_page[
            req_pool_indices[:batch_size], :copy_len
        ]

        return block_kv_indices

    # ---- Metadata initialization ----

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        spec_info=None,
        **kwargs,
    ):
        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_page=req_to_page,
                extend_prefix_lens=kwargs.pop("extend_prefix_lens"),
                extend_prefix_lens_cpu=kwargs.pop("extend_prefix_lens_cpu"),
                extend_seq_lens=kwargs.pop("extend_seq_lens"),
                extend_seq_lens_cpu=kwargs.pop("extend_seq_lens_cpu"),
            )
        else:
            self._init_decode_metadata(
                bs, req_pool_indices, seq_lens, forward_mode, req_to_page, spec_info
            )

    def _init_decode_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        spec_info=None,
    ):
        # For target_verify, the draft tokens have already been written to the KV
        # cache. The seq_lens passed in should already reflect the full context.
        # Use max_context_len to avoid GPU->CPU sync from seq_lens.max().item()
        max_blocks = self._calc_padded_blocks(self.max_context_len)

        block_kv_indices = self._create_block_kv_indices(
            bs, max_blocks, req_pool_indices, seq_lens, req_to_page
        )

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_decode_metadata = TRTLLMMLADecodeMetadata(
            block_kv_indices=block_kv_indices,
            max_seq_len_k=self.max_context_len,
            seq_lens_k=seq_lens[:bs],
        )

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor | None = None,
        req_to_page: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
    ):
        max_seq_len = self.max_context_len
        cum_seq_lens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
            max_seq_len=max_seq_len,
            cum_seq_lens=cum_seq_lens,
            seq_lens=seq_lens,
        )
        num_extends = extend_seq_lens.shape[0]
        cum_extend_seq_lens = torch.zeros(
            num_extends + 1, device=self.device, dtype=torch.int32
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])
        max_extend_seq_len = extend_seq_lens_cpu.max().item()
        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            req_to_page,
            req_pool_indices,
            self.page_size,
        )
        self.chunked_prefill_metadata = TRTLLMMLAChunkedPrefillMetadata(
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            cum_extend_seq_lens=cum_extend_seq_lens,
            max_extend_seq_len=max_extend_seq_len,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
        )

    # ---- CUDA Graph ----

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )
        # Alias controller's seq_lens_buf — backend never mutates it.
        self.cuda_graph_seq_lens_buf = seq_lens_buf
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        self.decode_cuda_graph_kv_indices = torch.zeros(
            (max_bs, max_blocks), dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"trtllm_mla CUDA graph capture not supported for {forward_mode}"
            )

        metadata = TRTLLMMLADecodeMetadata()
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks]

        # For capture we don't have req_to_page yet; just zero-fill the block indices.
        # The actual indices will be filled on replay.
        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len
        # seq_lens_k aliases seq_lens_buf (set in init_cuda_graph_state).
        metadata.seq_lens_k = self.cuda_graph_seq_lens_buf[:bs]

        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"trtllm_mla CUDA graph replay not supported for {forward_mode}"
            )

        metadata = self.decode_cuda_graph_metadata[bs]

        # seq_lens_k aliases seq_lens_buf; only block indices need refresh.
        if req_to_page is not None:
            self._create_block_kv_indices(
                bs,
                metadata.block_kv_indices.shape[1],
                req_pool_indices[:bs],
                seq_lens[:bs],
                req_to_page,
                metadata.block_kv_indices,
            )

        self.forward_decode_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- Forward: Decode ----

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
        # q is whole Q [T, H, head_dim]; k is whole latent [T, 1, head_dim].
        if save_kv_cache:
            assert k is not None
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        metadata = self.forward_decode_metadata
        spec_num_tokens = q.shape[0] // bs if bs > 0 else 1

        if spec_num_tokens > 1 and self.is_draft:
            # First draft step catching up its KV after verify: one query entry per token;
            # per-token seq_lens advance by 1 so each successive token sees its own KV write.
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
            block_tables = metadata.block_kv_indices.repeat_interleave(
                spec_num_tokens, dim=0
            )
            base_lens = metadata.seq_lens_k.repeat_interleave(spec_num_tokens)
            offsets = torch.arange(
                spec_num_tokens, device=base_lens.device, dtype=base_lens.dtype
            ).repeat(bs)
            seq_lens = base_lens + offsets
            max_seq_len = metadata.max_seq_len_k + spec_num_tokens
        else:
            # Plain decode (q_len=1) or bs-grouped multi-token decode.
            query = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            block_tables = metadata.block_kv_indices
            seq_lens = metadata.seq_lens_k
            max_seq_len = metadata.max_seq_len_k

        if self.data_type == torch.float8_e4m3fn:
            query = query.to(self.data_type)
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            bmm1_scale = k_scale * layer.scaling
        else:
            bmm1_scale = layer.scaling

        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        raw_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.trtllm_workspace,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
        )

        return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap,
        *,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        seq_lens,
        batch_size,
        causal,
    ):
        if causal:
            step_counter = getattr(self, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()

        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = q.reshape(-1, self.num_local_heads, head_dim)
        k = k.reshape(-1, self.num_local_heads, head_dim)
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim)

        # FP8 prefill: if Q is already FP8 (model decided to use FP8 prefill),
        # ensure K/V match. If Q is BF16, respect the model's decision.
        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        # The ragged path does not support FP8 output.
        out_dtype = self.q_data_type
        if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            out_dtype = torch.bfloat16

        out = torch.empty(
            q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=out_dtype,
        )

        result = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.trtllm_workspace,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=scaling,
            bmm2_scale=1.0,
            o_sf_scale=-1.0,
            batch_size=batch_size,
            window_left=-1,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            enable_pdl=pdl_enabled(),
            is_causal=causal,
            return_lse=True,
            out=out,
        )

        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None


register_backend("trtllm_mla", {AttentionArch.MLA}, TRTLLMMLABackend)
