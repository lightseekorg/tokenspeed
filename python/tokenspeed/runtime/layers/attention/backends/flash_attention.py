# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# Upstream lineage includes ModelTC/lightllm, vllm-project/vllm,
# and sgl-project/sglang. See python/THIRDPARTYNOTICES.
# Licensed under the Apache License, Version 2.0
#
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
FlashAttention (FA3) backend for TokenSpeed scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import triton
import triton.language as tl

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import (
    build_page_table,
    token_indices_from_pages,
    update_page_table_inplace,
)
from tokenspeed.runtime.spec_decode.eagle import EagleDraftInput, EagleVerifyInput
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

from tokenspeed_kernel.ops.attention.flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from tokenspeed_kernel.thirdparty.cuda.merge_state import merge_state

# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None
    # FA3 AOT scheduler metadata, shared by all attention layers in a decode step.
    scheduler_metadata: torch.Tensor = None
    max_num_splits: int = 0

    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
        local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
        local_block_table: torch.Tensor = None  # block table for local attention
        local_max_query_len: int = 0  # max query length for local attention
        local_max_seq_len: int = 0  # max sequence length for local attention

    local_attn_metadata: LocalAttentionMetadata | None = None

    # For sliding window attention topk>1 spec decoding
    swa_spec_metadata: FlashAttentionMetadata | None = None


# ---------------------------------------------------------------------------
# Local-attention virtual-batch helper
# ---------------------------------------------------------------------------


def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
    local attention blocks, where each block is passed to the attention kernel
    as an independent local ("virtual") batch item.
    """
    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    if effective_chunk_size < page_size:
        effective_chunk_size = page_size
    attn_chunk_size = effective_chunk_size

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1

    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)

    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    block_starts = k_seqstarts_absolute // page_size

    assert attn_chunk_size % page_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by page_size {page_size}"
    )
    pages_per_local_batch = attn_chunk_size // page_size

    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )
    block_table_local = block_table[batch_indices, block_indices].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def round_up(a: int, b: int) -> int:
    return cdiv(a, b) * b


# FA3 CUDA graph decode needs a fixed split count so the captured workspace and
# scheduler metadata shape stay stable across graph replay. Keep this as an
# internal backend constant instead of using the dynamic split heuristic.
FA3_CUDA_GRAPH_SCHEDULER_NUM_SPLITS = 32


# ---------------------------------------------------------------------------
# FlashAttentionBackend
# ---------------------------------------------------------------------------


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation for TokenSpeed scheduling.

    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(self, config: MHAConfig):
        super().__init__(config)

        # Pattern A: separate prefill/decode metadata slots so a compound
        # DRAFT_EXTEND capture can populate both once (step 0 = prefill,
        # steps 1..N-1 = decode) and the captured graph reads from fixed
        # addresses.
        self.forward_prefill_metadata: FlashAttentionMetadata = None
        self.forward_decode_metadata: FlashAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = config.context_len
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.kv_cache_dtype = config.kv_cache_dtype
        self.kv_cache_dtype_str = (
            "auto"
            if config.kv_cache_dtype in (torch.bfloat16, torch.float16)
            else "fp8_e4m3" if config.kv_cache_dtype == torch.float8_e4m3fn else "auto"
        )
        self.page_size = config.page_size
        self.use_mla = False  # MHA backend — MLA is handled separately

        # Speculative decoding settings
        self.topk = 0
        self.speculative_num_steps = getattr(config, "speculative_num_steps", 0)
        self.speculative_num_draft_tokens = getattr(
            config, "speculative_num_draft_tokens", 0
        )
        self.speculative_step_id = 0

        self.fa_impl_ver = 3

        # Local attention settings — may be overridden via kwargs.
        self.attention_chunk_size = None

        # Sliding-window settings
        self.sliding_window_size = None
        self.has_swa = False
        self.is_hybrid = False

        # If num_splits == 0, we use a heuristic to automatically determine the number of splits.
        self.num_splits = 0
        self.cuda_graph_scheduler_num_splits = FA3_CUDA_GRAPH_SCHEDULER_NUM_SPLITS

    @property
    def support_kv_cache_prewrite(self) -> bool:
        return True

    def configure_runtime(self, **kwargs) -> None:
        sliding_window_size = kwargs.get("sliding_window_size", None)
        self.sliding_window_size = sliding_window_size
        self.has_swa = sliding_window_size is not None and sliding_window_size > 0

    def _scheduler_metadata_size(self, bs: int) -> int:
        # FA3 stores one semaphore plus four scheduling vectors per batch item.
        return 1 + round_up(bs, 4) * 4

    def _can_use_decode_scheduler_metadata(self) -> bool:
        # A single FA3 AOT schedule is only valid when all decode attention
        # layers share the same plain MHA shape. Local/SWA/spec/MLA paths keep
        # their existing per-call scheduler behavior.
        return (
            self.fa_impl_ver == 3
            and not self.use_mla
            and self.attention_chunk_size is None
            and not self.has_swa
            and self.topk <= 1
        )

    def _build_decode_scheduler_metadata(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
        num_splits: int,
    ) -> Optional[torch.Tensor]:
        if not self._can_use_decode_scheduler_metadata():
            return None
        if (
            metadata.cache_seqlens_int32 is None
            or metadata.cu_seqlens_q is None
            or metadata.max_seq_len_k <= 0
        ):
            return None

        return get_scheduler_metadata(
            batch_size=bs,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_k,
            num_heads_q=self.num_qo_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_dim,
            cache_seqlens=metadata.cache_seqlens_int32,
            qkv_dtype=self.kv_cache_dtype,
            cu_seqlens_q=metadata.cu_seqlens_q,
            page_size=self.page_size,
            causal=True,
            window_size=(-1, -1),
            num_splits=num_splits,
        )

    def _init_decode_cuda_graph_scheduler_metadata(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
    ) -> None:
        if "scheduler_metadata" not in self.decode_cuda_graph_metadata:
            return
        scheduler_metadata = self._build_decode_scheduler_metadata(
            metadata,
            bs,
            self.cuda_graph_scheduler_num_splits,
        )
        if scheduler_metadata is None:
            return
        max_size = self._scheduler_metadata_size(bs)
        scheduler_metadata_buf = self.decode_cuda_graph_metadata["scheduler_metadata"][
            :max_size
        ]
        n = scheduler_metadata.shape[0]
        if n > scheduler_metadata_buf.shape[0]:
            raise RuntimeError(
                f"FA3 scheduler metadata is larger than the graph buffer: "
                f"{n} > {scheduler_metadata_buf.shape[0]}"
            )
        scheduler_metadata_buf[:n].copy_(scheduler_metadata)
        scheduler_metadata_buf[n:].zero_()
        metadata.scheduler_metadata = scheduler_metadata_buf[:n]
        metadata.max_num_splits = self.cuda_graph_scheduler_num_splits

    def _update_decode_cuda_graph_scheduler_metadata(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
    ) -> None:
        if metadata.scheduler_metadata is None:
            return
        scheduler_metadata = self._build_decode_scheduler_metadata(
            metadata,
            bs,
            metadata.max_num_splits,
        )
        if scheduler_metadata is None:
            return
        n = scheduler_metadata.shape[0]
        if n > metadata.scheduler_metadata.shape[0]:
            raise RuntimeError(
                f"FA3 scheduler metadata changed size across graph replay: "
                f"{n} > {metadata.scheduler_metadata.shape[0]}"
            )
        metadata.scheduler_metadata[:n].copy_(scheduler_metadata)
        metadata.scheduler_metadata[n:].zero_()

    # ------------------------------------------------------------------
    # Draft decode metadata
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # init_forward_metadata
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        req_to_page: torch.Tensor = None,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu=None,
        spec_info=None,
        use_cuda_graph: bool = False,
        out_cache_loc: torch.Tensor | None = None,
        token_to_kv_pool=None,
        **kwargs,
    ):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = seq_lens
        batch_size = bs
        device = seqlens_in_batch.device
        max_context_len = self.max_context_len

        assert req_to_page is not None, "req_to_page must be provided"

        # Use max_context_len as worst-case max_seq_len_k — avoids GPU sync (.item()).
        # The actual per-request lengths are in cache_seqlens_int32.
        page_table = build_page_table(
            req_pool_indices, req_to_page, self.page_size, max_context_len
        )

        if forward_mode.is_decode_or_idle():
            # Draft Decode
            if spec_info is not None:
                if self.topk <= 1:
                    # seqlens_in_batch is the controller's seq_lens_buf
                    # (post-write cache length); use it directly.
                    assert seqlens_in_batch.dtype == torch.int32
                    metadata.cache_seqlens_int32 = seqlens_in_batch
                    metadata.max_seq_len_k = max_context_len
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.page_table = page_table
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = max_context_len
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.page_table = page_table

                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = out_cache_loc.view(-1, self.speculative_num_steps)
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = max_context_len
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.page_table = page_table
            self._init_local_attn_metadata(
                metadata,
                device,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cache_seqlens_int32=metadata.cache_seqlens_int32,
                page_table=page_table,
            )
        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                # seq_lens = valid_cache_lengths + speculative_num_draft_tokens
                # (the controller writes the post-write cache length); the
                # decode kernel reads cache_seqlens as the AFTER-write length.
                assert seq_lens.dtype == torch.int32
                metadata.cache_seqlens_int32 = seq_lens
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = max_context_len
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.page_table = page_table

                self._init_local_attn_metadata(
                    metadata,
                    device,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cache_seqlens_int32=metadata.cache_seqlens_int32,
                    page_table=page_table,
                )
            else:
                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = max_context_len
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.page_table = page_table

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    seq_lens.numel() * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(0)
                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )

                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask,
                    col_indices,
                    col_indices + self.speculative_num_draft_tokens,
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = token_indices_from_pages(
                    req_pool_indices, cols, req_to_page, self.page_size
                ).repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                self.forward_metadata_spec_decode_expand = metadata_expand

                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_mode.is_extend():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = max_context_len
            metadata.page_table = page_table

            if extend_with_prefix and extend_prefix_lens is not None:
                extend_seq_lens = seq_lens - extend_prefix_lens
                # The FA3 workspace is sized from max_seq_len_q.  For prefix
                # chunks, num_tokens is only the sum over the batch and can be
                # much larger than any single query sequence.
                extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
                if extend_seq_lens_cpu is not None:
                    metadata.max_seq_len_q = int(
                        extend_seq_lens_cpu[:batch_size].max().item()
                    )
                else:
                    metadata.max_seq_len_q = num_tokens
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            elif (
                forward_mode == ForwardMode.DRAFT_EXTEND
                and kwargs.get("extend_seq_lens") is not None
            ):
                metadata.max_seq_len_q = num_tokens
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(kwargs["extend_seq_lens"], dim=0, dtype=torch.int32),
                    (1, 0),
                )
            else:
                # No prefix / no per-request extend lens — Q and K cumsums are
                # the same (full sequence prefill), so derive cu_seqlens_q
                # straight from seqlens_in_batch.
                metadata.max_seq_len_q = max_context_len
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )

            # Setup local attention if enabled
            if forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(
                    metadata,
                    device,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cache_seqlens_int32=metadata.cache_seqlens_int32,
                    page_table=page_table,
                )

        # Route to prefill/decode slot. DRAFT_EXTEND is compound: step 0
        # uses the prefill slot (multi-token query), steps 1..N-1 use the
        # decode slot (single-token query, advanced per step).
        if forward_mode.is_decode_or_idle():
            self.forward_decode_metadata = metadata
        elif forward_mode == ForwardMode.DRAFT_EXTEND:
            self.forward_prefill_metadata = metadata
            decode_metadata = FlashAttentionMetadata()
            decode_metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            decode_metadata.max_seq_len_k = max_context_len
            decode_metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            decode_metadata.page_table = page_table
            self.forward_decode_metadata = decode_metadata
        else:
            self.forward_prefill_metadata = metadata

    # ------------------------------------------------------------------
    # forward_extend
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
        # For multi-head latent attention
        q_rope: torch.Tensor | None = None,
        k_rope: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
        forward_mode: ForwardMode = None,
        spec_info=None,
        batch_size: int = None,
        **kwargs,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = out_cache_loc

                if not self.use_mla:
                    token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_prefill_metadata

        # Calculate window size
        is_swa = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa else (-1, -1)
        k_descale, v_descale = None, None
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                bs_for_descale = batch_size if batch_size is not None else q.shape[0]
                descale_shape = (bs_for_descale, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None

        # Check if we should use local attention
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        use_cascade_attn = (
            forward_mode is not None
            and forward_mode.is_target_verify()
            and self.topk > 1
            and not is_swa
        )

        # Only pass ``ver`` when talking to a non-default FlashAttention
        # interface version.
        fa_kwargs = {}
        if self.fa_impl_ver != 3:
            fa_kwargs["ver"] = self.fa_impl_ver
        if sinks is not None:
            fa_kwargs["sinks"] = sinks

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_swa and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
        else:
            page_table = metadata.page_table
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q

        # Use Flash Attention for prefill
        if not self.use_mla:
            assert self.fa_impl_ver in [3], "Only FA3 support here"
            # Do multi-head attention
            key_cache, value_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            # cu_seqlens_k_new=None: KV is written separately via set_kv_buffer
            # before this call, so no k_new tensor is passed and the kernel
            # has no use for the cumulative new-K offsets.
            result = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=not use_cascade_attn,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
                num_splits=self.num_splits,
                **fa_kwargs,
            )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                    **fa_kwargs,
                )
                o, _ = merge_state(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                    enable_pdl=pdl_enabled(),
                )
            else:
                o = result
        else:
            # MLA path
            attn_attend_prefix_cache = kwargs.get("attn_attend_prefix_cache", None)
            mha_return_lse = kwargs.get("mha_return_lse", False)
            prefix_chunk_idx = kwargs.get("prefix_chunk_idx", None)
            prefix_chunk_cu_seq_lens = kwargs.get("prefix_chunk_cu_seq_lens", None)
            prefix_chunk_max_seq_lens = kwargs.get("prefix_chunk_max_seq_lens", None)

            if (
                attn_attend_prefix_cache is not None
                and forward_mode is not None
                and not forward_mode.is_target_verify()
                and not forward_mode.is_draft_extend()
            ):
                # Do multi-head attention with chunked prefix cache
                if attn_attend_prefix_cache:
                    assert prefix_chunk_idx is not None
                    assert prefix_chunk_cu_seq_lens is not None
                    assert prefix_chunk_max_seq_lens is not None

                    chunk_idx = prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                        **fa_kwargs,
                    )
                else:
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=mha_return_lse,
                        **fa_kwargs,
                    )
                if mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                assert self.fa_impl_ver in [3], "Only FA3 support here"
                # Do absorbed multi-latent attention
                kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=not use_cascade_attn,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=None,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            num_splits=self.num_splits,
                        )
                    )
                    o, _ = merge_state(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                        enable_pdl=pdl_enabled(),
                    )
                else:
                    o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # forward_decode
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
        # For multi-head latent attention
        q_rope: torch.Tensor | None = None,
        k_rope: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
        forward_mode: ForwardMode = None,
        spec_info=None,
        batch_size: int = None,
        **kwargs,
    ) -> torch.Tensor:
        assert self.fa_impl_ver in [3], "Only FA3 support decoding"
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = out_cache_loc
                if not self.use_mla:
                    token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_decode_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don't need cascade attention, spec_info will be none in this case
        use_cascade_attn = spec_info is not None and self.topk > 1

        # Calculate window size
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )

        # Only pass ``ver`` when talking to a non-default FlashAttention
        # interface version.
        fa_kwargs = {}
        if self.fa_impl_ver != 3:
            fa_kwargs["ver"] = self.fa_impl_ver
        if sinks is not None:
            fa_kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                bs_for_descale = batch_size if batch_size is not None else q.shape[0]
                descale_shape = (bs_for_descale, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            if use_local_attn:
                # Use chunked (local) attention batching for self-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=self.num_splits,
                    **fa_kwargs,
                )
            else:
                page_table = metadata.page_table
                cache_seqlens = metadata.cache_seqlens_int32
                max_seqlen_q = metadata.max_seq_len_q
                scheduler_metadata = (
                    None if use_cascade_attn else metadata.scheduler_metadata
                )
                num_splits = (
                    metadata.max_num_splits
                    if scheduler_metadata is not None
                    else self.num_splits
                )
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # Default: single-token self-attention
                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=not use_cascade_attn,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    **fa_kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=None,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            num_splits=self.num_splits,
                            **fa_kwargs,
                        )
                    )
                    o, _ = merge_state(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                        enable_pdl=pdl_enabled(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=not use_cascade_attn,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
                num_splits=self.num_splits,
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                )
                o, _ = merge_state(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                    enable_pdl=pdl_enabled(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # CUDA graph support
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        # cache_seqlens aliases the controller's seq_lens_buf — backend
        # never mutates it.
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": seq_lens_buf,
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "scheduler_metadata": torch.zeros(
                self._scheduler_metadata_size(max_bs),
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }
        # Only allocate local attention buffers if local attention is enabled
        if self.attention_chunk_size is not None:
            max_seq_len = self.max_context_len
            page_size = self.page_size or 1
            attn_chunk_size = self.attention_chunk_size
            max_virtual_batches = max_bs * (
                (max_seq_len + attn_chunk_size - 1) // attn_chunk_size
            )
            max_pages_per_block = (attn_chunk_size + page_size - 1) // page_size

            self.decode_cuda_graph_local_attn_metadata = {
                "local_query_start_loc": torch.zeros(
                    max_virtual_batches + 1, dtype=torch.int32, device=self.device
                ),
                "local_seqused_k": torch.zeros(
                    max_virtual_batches, dtype=torch.int32, device=self.device
                ),
                "local_block_table": torch.zeros(
                    max_virtual_batches,
                    max_pages_per_block,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        # This is used by draft decode's first half of metadata when topk > 1
        if self.topk > 1:
            self.draft_decode_metadata_topk_normal = {
                "cache_seqlens": seq_lens_buf,
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    step=self.topk,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            # This is used by draft decode's second half of metadata when topk > 1
            decode_length = self.speculative_step_id + 1
            self.draft_decode_metadata_topk_expand = {
                "cache_seqlens": torch.full(
                    (max_bs * self.topk,),
                    decode_length,
                    device=self.device,
                    dtype=torch.int32,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.topk,
                    decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            # "page_table_draft_decode" will be set only when spec decoding enabled to save memory
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )

            self.target_verify_metadata = {
                "cache_seqlens": seq_lens_buf,
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            self.draft_extend_metadata = {
                "cache_seqlens": seq_lens_buf,
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {
                "cache_seqlens": seq_lens_buf,
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            self.target_verify_metadata_topk_expand = {
                "cache_seqlens": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            if self.has_swa:
                self.target_verify_metadata_topk_swa = {
                    "cache_seqlens": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "cu_seqlens_q": torch.arange(
                        0,
                        max_bs * self.speculative_num_draft_tokens + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "page_table": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        self.max_context_len,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: EagleDraftInput | EagleVerifyInput | None = None,
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = FlashAttentionMetadata()

        # metadata_expand is needed for Spec Decoding when top k > 1
        metadata_expand = FlashAttentionMetadata()

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                        "cache_seqlens"
                    ][:bs]
                    metadata.max_seq_len_k = self.max_context_len
                    metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.page_table = self.decode_cuda_graph_metadata[
                        "page_table_draft_decode"
                    ][:bs, :]
                    self.decode_cuda_graph_metadata[bs] = metadata
                else:
                    metadata.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_normal["cache_seqlens"][:bs]
                    )
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = seq_lens.max().item()
                    metadata.cu_seqlens_q = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.page_table = self.draft_decode_metadata_topk_normal[
                        "page_table"
                    ][:bs, :]

                    metadata_expand.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_expand["cache_seqlens"][
                            : bs * self.topk
                        ]
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_q"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.page_table = self.draft_decode_metadata_topk_expand[
                        "page_table"
                    ][: bs * self.topk]
                    self.draft_decode_metadata_topk_normal[bs] = metadata
                    self.draft_decode_metadata_topk_expand[bs] = metadata_expand
            else:
                # Normal Decode — cache_seqlens aliases seq_lens_buf.
                assert seq_lens.dtype == torch.int32
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                batch_size = len(seq_lens)
                device = seq_lens.device
                metadata.max_seq_len_k = self.max_context_len
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                self._init_decode_cuda_graph_scheduler_metadata(metadata, bs)
                self.decode_cuda_graph_metadata[bs] = metadata

                if self.attention_chunk_size is not None:
                    self._update_local_attn_metadata_for_capture(metadata, batch_size)

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                # cache_seqlens aliases seq_lens_buf (= valid_cache_lengths
                # + speculative_num_draft_tokens for TARGET_VERIFY); the
                # controller has already written the right values.
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]

                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = self.max_context_len

                metadata.cu_seqlens_q = torch.arange(
                    0,
                    bs * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )

                metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]

                self.target_verify_metadata[bs] = metadata
            else:
                metadata.cache_seqlens_int32 = self.target_verify_metadata_topk_normal[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.cu_seqlens_q = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_q"
                ][: bs + 1]
                metadata.page_table = self.target_verify_metadata_topk_normal[
                    "page_table"
                ][:bs, :]

                metadata_expand.cache_seqlens_int32 = (
                    self.target_verify_metadata_topk_expand["cache_seqlens"][
                        : bs * self.speculative_num_draft_tokens
                    ]
                )
                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_q"
                ][: bs * self.speculative_num_draft_tokens + 1]

                metadata_expand.page_table = self.target_verify_metadata_topk_expand[
                    "page_table"
                ][: bs * self.speculative_num_draft_tokens]

                self.target_verify_metadata_topk_normal[bs] = metadata
                self.target_verify_metadata_topk_expand[bs] = metadata_expand

                if self.has_swa:
                    metadata_swa = FlashAttentionMetadata()
                    metadata_swa.cache_seqlens_int32 = (
                        self.target_verify_metadata_topk_swa["cache_seqlens"][
                            : bs * self.speculative_num_draft_tokens
                        ]
                    )
                    metadata_swa.max_seq_len_q = 1
                    metadata_swa.cu_seqlens_q = self.target_verify_metadata_topk_swa[
                        "cu_seqlens_q"
                    ][: bs * self.speculative_num_draft_tokens + 1]

                    metadata_swa.page_table = self.target_verify_metadata_topk_swa[
                        "page_table"
                    ][: bs * self.speculative_num_draft_tokens]
                    self.target_verify_metadata_topk_swa[bs] = metadata_swa
                    metadata.swa_spec_metadata = metadata_swa

        elif forward_mode.is_draft_extend():
            # Compound DRAFT_EXTEND: step 0 uses prefill slot (multi-token
            # query over spec_num_tokens), steps 1..N-1 use decode slot
            # (single-token query). Both slots' cache_seqlens alias
            # seq_lens_buf — controller-written, no copy.
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]

            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.max_seq_len_k = self.max_context_len

            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )

            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]

            self.draft_extend_metadata[bs] = metadata

            # Decode slot for steps 1..N-1 (single-token per request).
            decode_metadata = FlashAttentionMetadata()
            decode_metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                "cache_seqlens"
            ][:bs]
            decode_metadata.max_seq_len_k = self.max_context_len
            decode_metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                "cu_seqlens_q"
            ][: bs + 1]
            decode_metadata.page_table = self.decode_cuda_graph_metadata[
                "page_table_draft_decode"
            ][:bs, :]
            self.decode_cuda_graph_metadata[bs] = decode_metadata

        # Route to prefill/decode slots. DRAFT_EXTEND populates both.
        if forward_mode.is_decode_or_idle():
            self.forward_decode_metadata = metadata
        elif forward_mode.is_target_verify():
            self.forward_prefill_metadata = metadata
        elif forward_mode.is_draft_extend():
            self.forward_prefill_metadata = metadata
            self.forward_decode_metadata = decode_metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        spec_info: EagleDraftInput | EagleVerifyInput | None = None,
        out_cache_loc: torch.Tensor | None = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None
        metadata_expand = None

        assert req_to_page is not None, "req_to_page must be provided"
        max_context_len = self.max_context_len

        if forward_mode.is_decode_or_idle():

            if spec_info is not None:
                # Draft Decode — cache_seqlens aliases seq_lens_buf.
                if self.topk <= 1:
                    metadata = self.decode_cuda_graph_metadata[bs]
                    metadata.max_seq_len_k = max_context_len
                    update_page_table_inplace(
                        metadata.page_table,
                        req_pool_indices,
                        req_to_page,
                        self.page_size,
                        max_context_len,
                    )

                else:
                    metadata = self.draft_decode_metadata_topk_normal[bs]
                    metadata.max_seq_len_k = max_context_len
                    update_page_table_inplace(
                        metadata.page_table,
                        req_pool_indices,
                        req_to_page,
                        self.page_size,
                        max_context_len,
                    )

                    metadata_expand = self.draft_decode_metadata_topk_expand[bs]
                    decode_length = self.speculative_step_id + 1
                    cache_loc = out_cache_loc.view(-1, self.speculative_num_steps)
                    metadata_expand.page_table[: cache_loc.shape[0]].copy_(
                        cache_loc[:, :decode_length]
                    )
            else:
                # Normal Decode — cache_seqlens aliases seq_lens_buf.
                metadata = self.decode_cuda_graph_metadata[bs]
                metadata.max_seq_len_k = max_context_len
                update_page_table_inplace(
                    metadata.page_table,
                    req_pool_indices,
                    req_to_page,
                    self.page_size,
                    max_context_len,
                )
                self._update_decode_cuda_graph_scheduler_metadata(metadata, bs)

                self._update_local_attn_metadata_for_replay(
                    metadata,
                    bs,
                )
        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                # cache_seqlens aliases seq_lens_buf; the controller already
                # wrote vc + speculative_num_draft_tokens via fill_input_buffers.
                metadata = self.target_verify_metadata[bs]
                metadata.max_seq_len_k = max_context_len
                update_page_table_inplace(
                    metadata.page_table,
                    req_pool_indices,
                    req_to_page,
                    self.page_size,
                    max_context_len,
                )
            else:
                metadata = self.target_verify_metadata_topk_normal[bs]
                metadata.max_seq_len_k = max_context_len
                update_page_table_inplace(
                    metadata.page_table,
                    req_pool_indices,
                    req_to_page,
                    self.page_size,
                    max_context_len,
                )

                metadata_expand = self.target_verify_metadata_topk_expand[bs]

                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(0)

                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                # avoid extracting padded seq indices which will be out of boundary
                mask_extraction_indices[
                    :,
                    spec_info.positions.numel() * self.speculative_num_draft_tokens :,
                ].fill_(0)
                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )

                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask,
                    col_indices,
                    col_indices + self.speculative_num_draft_tokens,
                )
                _, sort_order = torch.sort(keys, dim=1)

                non_masked_page_table = token_indices_from_pages(
                    req_pool_indices, cols, req_to_page, self.page_size
                ).repeat_interleave(self.speculative_num_draft_tokens, dim=0)

                metadata_expand.page_table.copy_(
                    non_masked_page_table.gather(1, sort_order)
                )
                metadata_expand.cache_seqlens_int32.copy_(mask.sum(dim=1))

                if self.has_swa:
                    metadata_swa = self.target_verify_metadata_topk_swa[bs]
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand, metadata_swa
                    )

        elif forward_mode.is_draft_extend():
            # Compound DRAFT_EXTEND: refresh both prefill slot (step 0,
            # multi-token query) and decode slot (steps 1..N-1, single
            # token query). cache_seqlens aliases seq_lens_buf.
            metadata = self.draft_extend_metadata[bs]
            metadata.max_seq_len_k = max_context_len
            update_page_table_inplace(
                metadata.page_table,
                req_pool_indices,
                req_to_page,
                self.page_size,
                max_context_len,
            )

            decode_metadata = self.decode_cuda_graph_metadata[bs]
            decode_metadata.max_seq_len_k = max_context_len
            update_page_table_inplace(
                decode_metadata.page_table,
                req_pool_indices,
                req_to_page,
                self.page_size,
                max_context_len,
            )

        # Route to prefill/decode slots. DRAFT_EXTEND populates both.
        if forward_mode.is_decode_or_idle():
            self.forward_decode_metadata = metadata
        elif forward_mode.is_target_verify():
            self.forward_prefill_metadata = metadata
        elif forward_mode.is_draft_extend():
            self.forward_prefill_metadata = metadata
            self.forward_decode_metadata = decode_metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    # ------------------------------------------------------------------
    # Local attention helpers
    # ------------------------------------------------------------------

    def _init_local_attn_metadata(
        self,
        metadata: FlashAttentionMetadata,
        device,
        cu_seqlens_q=None,
        cache_seqlens_int32=None,
        page_table=None,
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        if cu_seqlens_q is None:
            cu_seqlens_q = metadata.cu_seqlens_q
        if cache_seqlens_int32 is None:
            cache_seqlens_int32 = metadata.cache_seqlens_int32
        if page_table is None:
            page_table = metadata.page_table

        if self.is_hybrid:
            page_table = self.full_to_swa_index_mapping[page_table].to(torch.int32)

        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _update_local_attn_metadata_for_capture(
        self, metadata: FlashAttentionMetadata, bs: int
    ):
        """Update local attention metadata during CUDA graph capture phase."""
        seq_lens_capture = metadata.cache_seqlens_int32
        max_seq_len = int(seq_lens_capture.max().item())
        page_table_capture = metadata.page_table

        cu_seqlens_q_np = metadata.cu_seqlens_q.cpu().numpy()
        seqlens_np = seq_lens_capture.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local_np,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            page_table_capture,
            self.page_size,
        )

        q_len = len(cu_seqlens_q_local_np)
        k_len = len(seqlens_k_local_np)
        b0 = block_table_local_np.shape[0] if block_table_local_np.shape[0] > 0 else bs
        b1 = block_table_local_np.shape[1] if block_table_local_np.shape[1] > 0 else 1

        local_query_start_loc = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ][:q_len]

        local_seqused_k = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"][
            :k_len
        ]

        local_block_table = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ][:b0, :b1]

        metadata.local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seqused_k=local_seqused_k,
            local_block_table=local_block_table,
            local_max_query_len=1,
            local_max_seq_len=max_seq_len,
        )

    def _update_local_attn_metadata_for_replay(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
    ):
        """Update preallocated local attention metadata in-place before CUDA graph replay."""
        if self.attention_chunk_size is None:
            return

        local_q_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ]
        local_k_buf = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"]
        local_block_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ]
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"]

        cu_seqlens_q = torch.arange(
            bs + 1, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        seqlens = metadata.cache_seqlens_int32[:bs]
        max_seq_len = int(seqlens.max().item())
        if self.is_hybrid:
            sliced_page_table = self.full_to_swa_index_mapping[
                metadata.page_table[:bs, :max_seq_len]
            ].to(torch.int32)
        else:
            sliced_page_table = metadata.page_table[:bs, :max_seq_len]

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seqlens_np = seqlens.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            sliced_page_table,
            self.page_size,
        )

        device = local_q_buf.device
        cu_seqlens_q_local = torch.from_numpy(cu_seqlens_q_local_np).to(device)
        seqlens_k_local = torch.from_numpy(seqlens_k_local_np).to(device)
        block_table_local = block_table_local.to(device)
        q_len = cu_seqlens_q_local.shape[0]
        k_len = seqlens_k_local.shape[0]
        b0, b1 = block_table_local.shape

        local_q_buf[:q_len].copy_(cu_seqlens_q_local)
        local_q_buf[q_len:].fill_(0)
        local_k_buf[:k_len].copy_(seqlens_k_local)
        local_k_buf[k_len:].fill_(0)
        local_block_buf[:b0, :b1].copy_(block_table_local)
        local_block_buf[b0:, :].fill_(0)
        local_block_buf[:b0, b1:].fill_(0)

        if metadata.local_attn_metadata is not None:
            lam = metadata.local_attn_metadata
            lam.local_max_query_len = int(seqlens_q_local_np.max())
            lam.local_max_seq_len = int(seqlens_k_local_np.max())

    # ------------------------------------------------------------------
    # Sliding window attention helpers for speculative decoding
    # ------------------------------------------------------------------

    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: FlashAttentionMetadata | None = None,
    ):
        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        cache_seqlens_int32 = (
            metadata.cache_seqlens_int32.repeat_interleave(
                self.speculative_num_draft_tokens
            )
            + metadata_expand.cache_seqlens_int32
        )
        bs = cache_seqlens_int32.shape[0]
        page_table = (
            metadata.page_table.new_zeros(
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])
            )
            if metadata_swa is None
            else metadata_swa.page_table
        )

        prepare_swa_spec_page_table_triton(
            page_table,
            metadata.page_table,
            metadata_expand.page_table,
            metadata.cache_seqlens_int32,
            metadata_expand.cache_seqlens_int32,
            self.speculative_num_draft_tokens,
        )

        if metadata_swa is None:
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.page_table = page_table
        else:
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)

        metadata.swa_spec_metadata = metadata_swa


# ---------------------------------------------------------------------------
# Triton kernel for SWA spec page table preparation
# ---------------------------------------------------------------------------


@triton.jit
def _prepare_swa_spec_page_table_kernel(
    dst_ptr,
    src_a_ptr,
    src_b_ptr,
    seq_len_a_ptr,
    seq_len_b_ptr,
    dst_stride_m,
    dst_stride_n,
    a_stride_m,
    a_stride_n,
    b_stride_m,
    b_stride_n,
    LEN_A: tl.constexpr,
    LEN_B: tl.constexpr,
    REPEAT_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx_a = pid_m // REPEAT_STEP
    idx_b = pid_m
    seq_len_a = tl.load(seq_len_a_ptr + idx_a)
    seq_len_b = tl.load(seq_len_b_ptr + idx_b)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_len = seq_len_a + seq_len_b

    if pid_n * BLOCK_N >= total_len:
        return

    mask = offs_n < total_len
    dst = dst_ptr + pid_m * dst_stride_m + offs_n * dst_stride_n

    if (pid_n + 1) * BLOCK_N < seq_len_a:
        a_ptr = src_a_ptr + idx_a * a_stride_m + offs_n * a_stride_n
        a_mask = mask & (offs_n < LEN_A)
        val = tl.load(a_ptr, mask=a_mask, other=0)
        tl.store(dst, val, mask=mask)
    elif pid_n * BLOCK_N >= seq_len_a:
        offs_b = offs_n - seq_len_a
        b_ptr = src_b_ptr + idx_b * b_stride_m + offs_b * b_stride_n
        b_mask = mask & (offs_b < LEN_B)
        val = tl.load(b_ptr, mask=b_mask, other=0)
        tl.store(dst, val, mask=mask)
    else:
        # mixed part
        a_offs = offs_n
        a_mask = (a_offs < seq_len_a) & (a_offs < LEN_A)
        a_ptr = src_a_ptr + idx_a * a_stride_m + a_offs * a_stride_n
        a_val = tl.load(a_ptr, mask=a_mask, other=0)

        b_offs = offs_n - seq_len_a
        b_mask = (b_offs >= 0) & (b_offs < seq_len_b) & (b_offs < LEN_B)
        b_ptr = src_b_ptr + idx_b * b_stride_m + b_offs * b_stride_n
        b_val = tl.load(b_ptr, mask=b_mask, other=0)

        result = tl.where(offs_n < seq_len_a, a_val, b_val)
        tl.store(dst, result, mask=mask)


def prepare_swa_spec_page_table_triton(
    page_table_dst: torch.Tensor,
    page_table_a: torch.Tensor,
    page_table_b: torch.Tensor,  # expand page table
    seq_len_a: torch.Tensor,
    seq_len_b: torch.Tensor,  # expand seq lens
    speculative_num_draft_tokens: int,
):
    # concat page_table and expand page_table by kv seq length
    bs = seq_len_a.numel()
    bs_expand = seq_len_b.numel()
    assert bs_expand == bs * speculative_num_draft_tokens

    LEN_A = page_table_a.shape[1]
    LEN_B = page_table_b.shape[1]
    LEN_OUT = LEN_A + LEN_B
    REPEAT_STEP = speculative_num_draft_tokens
    BLOCK_N = 256

    grid = (bs_expand, triton.cdiv(LEN_OUT, BLOCK_N))
    _prepare_swa_spec_page_table_kernel[grid](
        page_table_dst,
        page_table_a,
        page_table_b,
        seq_len_a,
        seq_len_b,
        page_table_dst.stride(0),
        page_table_dst.stride(1),
        page_table_a.stride(0),
        page_table_a.stride(1),
        page_table_b.stride(0),
        page_table_b.stride(1),
        LEN_A=LEN_A,
        LEN_B=LEN_B,
        REPEAT_STEP=REPEAT_STEP,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_backend("fa3", {AttentionArch.MHA}, FlashAttentionBackend)
