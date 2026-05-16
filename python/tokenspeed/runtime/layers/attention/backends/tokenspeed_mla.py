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
CuteDSL MLA attention backend for TokenSpeed scheduling.

Uses CuTe DSL JIT-compiled kernels for MLA decode and prefill on Blackwell SM100 GPUs:
- tokenspeed_mla_decode for decode/verify
- tokenspeed_mla_prefill for prefill
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
from tokenspeed_kernel.ops.attention.tokenspeed_mla import (
    get_num_sm,
    tokenspeed_mla_decode,
    tokenspeed_mla_prefill,
    warmup_compile_prefill,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import (
    TRTLLM_BLOCK_CONSTRAINT,
    TRTLLMMLAChunkedPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = logging.getLogger(__name__)

# CuteDSL decode workspace. The kernel's own `get_workspace_size` formula is
#   B * H * q_len * split_kv * (D + 1) * (acc_dtype.width // 8)   (bytes)
# and `B * split_kv <= num_SMs`, so a closed-form upper bound (Float32 acc) is
#   num_SMs * H * q_len * (D + 1) * 4
# Buffer is per-device and does NOT need zero-init.
_cutedsl_workspace_buffer: dict[torch.device, torch.Tensor] = {}

# Initial q_len capacity for the per-device decode workspace. The buffer grows
# on demand before each launch, so larger verify/draft batches are supported.
_CUTEDSL_INITIAL_Q_LEN_CAPACITY = 8


def get_cutedsl_workspace_buffer(
    device: torch.device,
    num_heads_per_tp: int,
    kv_lora_rank: int,
    q_len_capacity: int = _CUTEDSL_INITIAL_Q_LEN_CAPACITY,
) -> torch.Tensor:
    """Get or grow the per-device CuteDSL workspace buffer."""
    num_sms = get_num_sm(device)
    required = num_sms * num_heads_per_tp * q_len_capacity * (kv_lora_rank + 1) * 4

    existing = _cutedsl_workspace_buffer.get(device)
    if existing is None or existing.numel() < required:
        _cutedsl_workspace_buffer[device] = torch.empty(
            required, dtype=torch.int8, device=device
        )
    return _cutedsl_workspace_buffer[device]


@dataclass
class CuteDSLMLAPrefillMetadata:
    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class CuteDSLMLADecodeMetadata:
    block_kv_indices: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None


class CuteDSLMLABackend(AttentionBackend):
    """CuteDSL MLA attention backend for Blackwell SM100 GPUs.

    Decode uses CuTe DSL JIT-compiled kernels via tokenspeed_mla_decode().
    Prefill uses CuTe DSL FMHA kernel via tokenspeed_mla_prefill().
    """

    _logged_decode = False
    _logged_prefill = False

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

        # Workspace buffers — sized from config's num_heads / kv_lora_rank.
        num_heads_per_tp = config.num_attention_heads // config.attn_tp_size
        self.cutedsl_workspace = get_cutedsl_workspace_buffer(
            config.device, num_heads_per_tp, self.kv_lora_rank
        )

        # Pre-compile prefill kernel variants so JIT doesn't run during serving.
        # The backend may be constructed once per attention layer (60x for
        # Kimi-K2.5), but `warmup_compile_prefill` is idempotent: each config
        # is only JIT'd once and cached in a module-global dict.
        # tokenspeed_mla requires --kv-cache-dtype fp8_e4m3, so tokenspeed's
        # FP8 prefill path (deepseek_v3.py:946 `use_fp8_prefill`) is always
        # on and feeds fp8_e4m3fn q/k/v to the kernel — bf16 is unreachable
        # for this backend.
        d_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        warmup_compile_prefill(
            q_dtype=torch.float8_e4m3fn,
            d_qk=d_qk,
            d_v=self.v_head_dim,
            enable_pdl=pdl_enabled(),
        )

        # Validate page_size
        if self.page_size not in (32, 64):
            raise ValueError(
                f"tokenspeed_mla backend requires page_size 32 or 64, got {self.page_size}"
            )

        # tokenspeed_mla's CuTe DSL kernel only supports fp8_e4m3 KV cache; check
        # at startup so misconfiguration surfaces here, not in the first forward.
        kv_cache_dtype = global_server_args_dict.get("kv_cache_dtype", "auto")
        if kv_cache_dtype != "fp8_e4m3":
            raise NotImplementedError(
                f"tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got {kv_cache_dtype!r}."
            )

        self.num_local_heads = num_heads_per_tp

        # Metadata
        self.forward_decode_metadata: CuteDSLMLADecodeMetadata | None = None
        self.forward_prefill_metadata: CuteDSLMLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, CuteDSLMLADecodeMetadata] = {}
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
        seq_lens_cpu: torch.Tensor | None = None,
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
        max_blocks = self._calc_padded_blocks(self.max_context_len)

        block_kv_indices = self._create_block_kv_indices(
            bs, max_blocks, req_pool_indices, seq_lens, req_to_page
        )

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_decode_metadata = CuteDSLMLADecodeMetadata(
            block_kv_indices=block_kv_indices,
            max_seq_len_k=self.max_context_len,
            seq_lens_k=seq_lens,
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
        # Worst-case bound to avoid GPU->CPU sync from seq_lens.max().item().
        # TODO: track a loose CPU upper bound (advance by chunked_prefill_size /
        # accept_lengths.max(); correct when accurate values land) for tighter
        # kernel-grid sizing without syncing.
        max_seq_len = self.max_context_len
        cum_seq_lens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        self.forward_prefill_metadata = CuteDSLMLAPrefillMetadata(
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
                f"tokenspeed_mla CUDA graph capture not supported for {forward_mode}"
            )

        metadata = CuteDSLMLADecodeMetadata()
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks]

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
                f"tokenspeed_mla CUDA graph replay not supported for {forward_mode}"
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

        spec_num_tokens = q.shape[0] // bs if bs > 0 else 1
        query = q.view(bs, spec_num_tokens, layer.tp_q_head_num, layer.head_dim)

        softmax_scale = layer.scaling
        if self.data_type == torch.float8_e4m3fn:
            query = query.to(self.data_type)
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            softmax_scale = k_scale * layer.scaling

        # Prepare KV cache: [num_pages, page_size, kv_cache_dim] (3D for CuteDSL)
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim)

        metadata = self.forward_decode_metadata

        if not CuteDSLMLABackend._logged_decode:
            logger.info(
                "CuteDSL MLA decode kernel invoked (tokenspeed_mla_decode, query_dtype=%s, kv_dtype=%s)",
                query.dtype,
                kv_cache.dtype,
            )
            CuteDSLMLABackend._logged_decode = True

        self.cutedsl_workspace = get_cutedsl_workspace_buffer(
            query.device, layer.tp_q_head_num, self.kv_lora_rank, query.shape[1]
        )

        raw_out = tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.cutedsl_workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices,
            seq_lens=metadata.seq_lens_k,
            max_seq_len=metadata.max_seq_len_k,
            softmax_scale=softmax_scale,
            enable_pdl=pdl_enabled(),
        )

        return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ---- Forward: Extend/Prefill ----

    def forward_extend_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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

        # CuteDSL FMHA MLA: if Q is FP8, ensure K/V match. `.to()` is a no-op
        # when the source dtype already matches.
        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        if not CuteDSLMLABackend._logged_prefill:
            logger.info("CuteDSL MLA prefill kernel invoked (tokenspeed_mla_prefill)")
            CuteDSLMLABackend._logged_prefill = True

        result = tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens,
            cum_seq_lens=cum_seq_lens_kv,
            max_seq_len=max_kv_len,
            batch_size=batch_size,
            softmax_scale=scaling,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=max_q_len,
            enable_pdl=pdl_enabled(),
        )

        if isinstance(result, tuple):
            out, lse = result[0], result[1]
        else:
            out, lse = result, None

        if out.dtype != self.q_data_type:
            out = out.to(self.q_data_type)

        return out, lse


register_backend("tokenspeed_mla", {AttentionArch.MLA}, CuteDSLMLABackend)
