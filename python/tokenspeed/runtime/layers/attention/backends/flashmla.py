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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.flash_attn import flash_attn_varlen_func
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.ops.attention.flashinfer import (
    BatchMLAPagedAttentionWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
)
from tokenspeed.runtime.spec_decode.eagle import (
    EagleDraftInput,
    generate_attn_arg_prefill,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.flashinfer_config import get_flashinfer_workspace_size

PAGE_SIZE = 64

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class FlashMLADecodeMetadata:
    flashmla_metadata: tuple | None = None
    num_splits: torch.Tensor | None = None
    block_table: torch.Tensor | None = None

    def __init__(
        self,
        flashmla_metadata=None,
        num_splits=None,
        block_table=None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_table = block_table


@dataclass
class _PrefillMetadata:
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    use_ragged: bool


@dataclass
class _ChunkedPrefillMetadata:
    extend_prefix_lens: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    max_extend_seq_len: int
    chunked_loop_num: int
    chunk_kv_indices_list: list
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list


# Shared across all flashinfer prefill wrappers used by FlashMLABackend.
_global_workspace_buffer = None


class FlashMLABackend(AttentionBackend):
    """FlashMLA attention backend for TokenSpeed scheduling.

    Uses the FlashMLA kernel for DECODE, TARGET_VERIFY, and DRAFT_EXTEND;
    uses FlashInfer's MLA prefill wrappers for the EXTEND path.
    """

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        # Parse constants
        self.max_context_len = config.context_len
        self.kv_cache_quant_method = config.kv_cache_quant_method
        self.cache_dtype = config.kv_cache_dtype

        # MLA-specific dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        self.scaling = config.scaling
        self.softmax_scale = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.num_q_heads = config.num_attention_heads // config.attn_tp_size

        # FlashMLA-specific
        self.draft_token_num = 0

        if self.kv_cache_quant_method == "per_token_head":
            raise NotImplementedError(
                "FlashMLABackend no longer supports "
                "kv_cache_quant_method='per_token_head'."
            )
        if self.cache_dtype == torch.float8_e4m3fn:
            raise NotImplementedError(
                "FlashMLABackend no longer supports dense FP8 KV cache. "
                "Use a non-FP8 KV cache."
            )

        # Workspace buffer + flashinfer prefill wrappers (EXTEND path only).
        global _global_workspace_buffer
        if _global_workspace_buffer is None:
            _global_workspace_buffer = torch.empty(
                get_flashinfer_workspace_size(),
                dtype=torch.uint8,
                device=config.device,
            )
        self.workspace_buffer = _global_workspace_buffer

        max_bs = config.max_bs
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer,
            backend="auto",
        )
        self.indices_updater_prefill = _PrefillIndicesUpdater(config, self)

        # Metadata state
        self.forward_metadata: FlashMLADecodeMetadata | _PrefillMetadata | None = None
        self.chunked_prefill_metadata: _ChunkedPrefillMetadata | None = None
        self.last_seq_lens_sum: int | None = None

    # ------------------------------------------------------------------
    # Metadata init
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor = None,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        spec_info=None,
        **kwargs,
    ):
        if req_to_page is not None:
            block_table = req_to_page[req_pool_indices]
        else:
            block_table = None

        if forward_mode.is_decode_or_idle():
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_q_heads,
                1,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_table,
            )
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            seq_lens = seq_lens + self.draft_token_num
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.draft_token_num * self.num_q_heads,
                1,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_table,
            )
        else:
            # EXTEND path — flashinfer ragged/paged prefill.
            if extend_prefix_lens is None:
                raise RuntimeError(
                    "FlashMLABackend.init_forward_metadata requires "
                    "extend_prefix_lens in extend mode."
                )
            seq_lens_cpu = seq_lens.cpu()
            seq_lens_sum = seq_lens_cpu.sum().item()
            self.last_seq_lens_sum = seq_lens_sum

            extend_no_prefix = not extend_with_prefix
            use_ragged = (
                not global_server_args_dict["mla_disable_ragged"] and extend_no_prefix
            )

            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                extend_prefix_lens,
                req_to_page=req_to_page,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=use_ragged,
            )
            self.forward_metadata = _PrefillMetadata(
                self.prefill_wrapper_paged, use_ragged
            )

            extend_seq_lens = kwargs.pop("extend_seq_lens")
            extend_seq_lens_cpu = kwargs.pop("extend_seq_lens_cpu")
            extend_prefix_lens_cpu = kwargs.pop("extend_prefix_lens_cpu")
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
                PAGE_SIZE,
            )
            self.chunked_prefill_metadata = _ChunkedPrefillMetadata(
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

    # ------------------------------------------------------------------
    # CUDA graph (DECODE / TARGET_VERIFY / DRAFT_EXTEND only)
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        del seq_lens_buf  # flashmla allocates its own buffers.
        max_context_len = self.max_context_len + PAGE_SIZE - 1
        # 4 PAGES are reserved for speculation
        cuda_graph_kv_indices = torch.full(
            (max_bs, (max_context_len + 4 * PAGE_SIZE) // PAGE_SIZE),
            1,
            dtype=torch.int32,
            device="cuda",
        )

        if self.draft_token_num:
            (
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits,
            ) = get_mla_metadata(
                torch.ones(
                    max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                ),
                self.draft_token_num * self.num_q_heads,
                1,
            )
        else:
            (
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits,
            ) = get_mla_metadata(
                torch.ones(
                    max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                ),
                self.num_q_heads,
                1,
            )
        self.cuda_graph_kv_indices = cuda_graph_kv_indices

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        block_table = self.cuda_graph_kv_indices[:bs]
        if forward_mode.is_decode_or_idle():
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :],
            )
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            seq_lens = seq_lens + self.draft_token_num
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.draft_token_num * self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs],
            )
        else:
            raise RuntimeError(f"Not supported forward mode: {forward_mode}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        req_pool_indices = req_pool_indices[:bs]
        if req_to_page is not None:
            block_table = req_to_page[req_pool_indices]
        else:
            block_table = self.cuda_graph_kv_indices[:bs]
        seq_lens = seq_lens[:bs]

        if forward_mode is not None and forward_mode.is_decode_or_idle():
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_table = self.cuda_graph_kv_indices[:bs]
        elif forward_mode is not None and (
            forward_mode.is_target_verify() or forward_mode.is_draft_extend()
        ):
            seq_lens = seq_lens + self.draft_token_num
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.draft_token_num * self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_table = self.cuda_graph_kv_indices[:bs]
        else:
            raise RuntimeError(f"Not supported forward mode: {forward_mode}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
        seq_lens: torch.Tensor | None = None,
        forward_mode: ForwardMode | None = None,
        **kwargs,
    ):
        if forward_mode is None or forward_mode == ForwardMode.EXTEND:
            # Prefill: dispatch to ragged (MHA-style) or absorbed (MQA) path.
            if self.forward_metadata.use_ragged:
                return self._forward_normal_extend(q, k, v, layer, save_kv_cache)
            else:
                return self._forward_absorbed_extend(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    save_kv_cache,
                )

        assert forward_mode.is_target_verify() or forward_mode.is_draft_extend()
        if k is not None:
            assert v is not None
            if save_kv_cache:
                token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

        bs = (
            q.shape[0]
            if forward_mode.is_draft_extend()
            else self.forward_metadata.block_table.shape[0]
        )
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)

        assert (
            layer.tp_q_head_num == self.num_q_heads
        ), f"{layer.tp_q_head_num=} != {self.num_q_heads=}"
        reshape_q = q.view(bs, -1, self.num_q_heads, layer.head_dim)

        o, _ = flash_mla_with_kvcache(
            q=reshape_q,
            k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
            block_table=self.forward_metadata.block_table[:bs],
            cache_seqlens=seq_lens.to(torch.int32) + self.draft_token_num,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
            num_splits=self.forward_metadata.num_splits,
            softmax_scale=layer.scaling,
            causal=True,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap=None,
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
        output, lse, *_ = flash_attn_varlen_func(
            q=q.view(-1, self.num_local_heads, head_dim),
            k=k.view(-1, self.num_local_heads, head_dim).to(q.dtype),
            v=v.view(-1, self.num_local_heads, self.v_head_dim).to(q.dtype),
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_k=cum_seq_lens_kv,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scaling,
            causal=causal,
            return_attn_probs=True,
        )
        # lse must be transposed when using fa3.
        return output, lse.T.contiguous()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
        seq_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                token_to_kv_pool.set_kv_buffer(
                    layer,
                    out_cache_loc,
                    k,
                    v,
                )
        bs = q.shape[0]
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        assert (
            layer.tp_q_head_num == self.num_q_heads
        ), f"{layer.tp_q_head_num=} != {self.num_q_heads=}"
        reshape_q = q.view(bs, -1, self.num_q_heads, layer.head_dim)
        cache_lens = seq_lens

        o, _ = flash_mla_with_kvcache(
            q=reshape_q,
            k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
            block_table=self.forward_metadata.block_table[:bs],
            cache_seqlens=cache_lens.to(torch.int32),
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
            num_splits=self.forward_metadata.num_splits,
            softmax_scale=layer.scaling,
            causal=True,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # EXTEND prefill helpers
    # ------------------------------------------------------------------

    def _forward_normal_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        save_kv_cache: bool = True,
    ):
        assert not save_kv_cache

        o = self.prefill_wrapper_ragged.forward(
            q,
            k.view(-1, layer.tp_k_head_num, layer.head_dim),
            v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_absorbed_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
    ):
        # q is whole Q [T, H, head_dim]; k is whole latent [T, 1, head_dim].
        # flashinfer prefill_wrapper.run() requires q_nope / q_pe split, so
        # slice views here (free) before handing off to the kernel.
        assert k is not None

        if save_kv_cache:
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : layer.v_head_dim],
                k[..., layer.v_head_dim :],
            )

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        q_nope = q[..., : layer.v_head_dim]
        q_pe = q[..., layer.v_head_dim :]
        o = q_nope.new_empty(q_nope.shape)

        k_buf = token_to_kv_pool.get_key_buffer(layer.layer_id).to(q_nope.dtype)
        o = self.forward_metadata.prefill_wrapper.run(
            q_nope,
            q_pe,
            k_buf[:, :, : layer.v_head_dim],
            k_buf[:, :, layer.v_head_dim :],
            out=o,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class _PrefillIndicesUpdater:
    """Plans FlashInfer MLA prefill wrappers for the EXTEND path."""

    def __init__(self, config: MLAConfig, attn_backend: FlashMLABackend):
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.kv_cache_quant_method = config.kv_cache_quant_method
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.attn_backend = attn_backend

        self.kv_indptr = attn_backend.kv_indptr
        self.qo_indptr = attn_backend.qo_indptr
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        req_to_page: torch.Tensor = None,
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper = None,
        use_ragged: bool = False,
        spec_info: EagleDraftInput | None = None,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = 0
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self._call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
            req_to_page=req_to_page,
            spec_info=spec_info,
        )

    def _call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        req_to_page: torch.Tensor = None,
        spec_info: EagleDraftInput | None = None,
    ):
        bs = len(seq_lens)
        sm_scale = self.scaling

        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            torch.cumsum(paged_kernel_lens, dim=0, out=kv_indptr[1 : bs + 1])
            kv_indptr = kv_indptr[: bs + 1]
            if wrapper_paged._use_cuda_graph:
                kv_indices = wrapper_paged._kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum,
                    dtype=torch.int32,
                    device=req_pool_indices.device,
                )
            if req_to_page is not None:
                create_flashinfer_kv_indices_triton[(bs,)](
                    req_to_page,
                    req_pool_indices,
                    paged_kernel_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    req_to_page.shape[1],
                )
            torch.cumsum(seq_lens - prefix_lens, dim=0, out=qo_indptr[1 : bs + 1])
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, _ = generate_attn_arg_prefill(
                spec_info.draft_token_num,
                req_pool_indices,
                paged_kernel_lens,
                req_to_page,
            )

        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
            )
        else:
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )


register_backend("flashmla", {AttentionArch.MLA}, FlashMLABackend)
