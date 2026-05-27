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

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import ErrorClass, Priority, error_fn, register_kernel

platform = current_platform()

BatchDecodeWithPagedKVCacheWrapper = ErrorClass
BatchMLAPagedAttentionWrapper = ErrorClass
BatchPrefillWithPagedKVCacheWrapper = ErrorClass
BatchPrefillWithRaggedKVCacheWrapper = ErrorClass
cudnn_batch_prefill_with_kv_cache = error_fn
trtllm_batch_context_with_kv_cache = error_fn
trtllm_batch_decode_with_kv_cache = error_fn
trtllm_batch_decode_with_kv_cache_mla = error_fn
trtllm_ragged_attention_deepseek = error_fn

if platform.is_nvidia:
    try:
        from flashinfer.decode import (
            BatchDecodeWithPagedKVCacheWrapper,
            trtllm_batch_decode_with_kv_cache,
            trtllm_batch_decode_with_kv_cache_mla,
        )
    except ImportError:
        pass

    try:
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache
    except ImportError:
        pass

    try:
        from flashinfer.prefill import (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
            trtllm_batch_context_with_kv_cache,
            trtllm_ragged_attention_deepseek,
        )
    except ImportError:
        pass

if platform.is_nvidia and platform.is_blackwell:
    try:
        from flashinfer.mla import (
            BatchMLAPagedAttentionWrapper,
            trtllm_batch_decode_with_kv_cache_mla,
        )
    except ImportError:
        pass


# ------------------------------------------------------------------------------
# Kernel registration
# ------------------------------------------------------------------------------

_FLASHINFER_LOG2_E = math.log2(math.e)
_workspace_buffer: torch.Tensor | None = None
_ragged_prefill_workspaces: dict[torch.device, torch.Tensor] = {}
_ragged_prefill_wrappers: dict[torch.device, BatchPrefillWithRaggedKVCacheWrapper] = {}
_paged_prefill_workspaces: dict[torch.device, torch.Tensor] = {}
_paged_prefill_wrappers: dict[torch.device, BatchPrefillWithPagedKVCacheWrapper] = {}
_paged_prefill_metadata_buffers: dict[
    tuple[torch.device, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


@triton.jit
def _build_paged_prefill_metadata_kernel(
    page_table,
    cache_seqlens,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    page_table_stride_b: tl.constexpr,
    page_size: tl.constexpr,
    batch_size: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    cache_seqlen = tl.load(cache_seqlens + batch_idx)
    num_pages = tl.cdiv(cache_seqlen, page_size)
    last_page_len = tl.where(cache_seqlen > 0, ((cache_seqlen - 1) % page_size) + 1, 0)

    page_offset = 0
    for prev_batch_idx in range(0, batch_size):
        prev_seqlen = tl.load(cache_seqlens + prev_batch_idx)
        prev_num_pages = tl.cdiv(prev_seqlen, page_size)
        page_offset += tl.where(prev_batch_idx < batch_idx, prev_num_pages, 0)

    tl.store(paged_kv_indptr, 0, mask=batch_idx == 0)
    tl.store(paged_kv_indptr + batch_idx + 1, page_offset + num_pages)
    tl.store(paged_kv_last_page_len + batch_idx, last_page_len)

    for page_start in range(0, max_pages_per_seq, BLOCK_P):
        page_offsets = page_start + tl.arange(0, BLOCK_P)
        mask = page_offsets < num_pages
        page_ids = tl.load(
            page_table + batch_idx * page_table_stride_b + page_offsets,
            mask=mask,
            other=0,
        )
        tl.store(paged_kv_indices + page_offset + page_offsets, page_ids, mask=mask)


def _get_ragged_prefill_wrapper(
    device: torch.device,
) -> BatchPrefillWithRaggedKVCacheWrapper:
    wrapper = _ragged_prefill_wrappers.get(device)
    if wrapper is None:
        workspace = torch.empty(
            256 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        _ragged_prefill_workspaces[device] = workspace
        _ragged_prefill_wrappers[device] = wrapper
    return wrapper


def _get_paged_prefill_wrapper(
    device: torch.device,
) -> BatchPrefillWithPagedKVCacheWrapper:
    wrapper = _paged_prefill_wrappers.get(device)
    if wrapper is None:
        workspace = torch.empty(
            256 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        _paged_prefill_workspaces[device] = workspace
        _paged_prefill_wrappers[device] = wrapper
    return wrapper


if platform.is_nvidia and platform.is_hopper_plus:

    @register_kernel(
        "attention",
        "mha_prefill",
        name="flashinfer_mha_prefill",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={torch.float16, torch.bfloat16},
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
        tags={"throughput"},
    )
    def flashinfer_mha_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if sinks is not None:
            raise NotImplementedError(
                "FlashInfer ragged prefill does not support sinks"
            )
        wrapper = _get_ragged_prefill_wrapper(q.device)
        wrapper.plan(
            cu_seqlens_q,
            cu_seqlens_q,
            q.shape[1],
            k.shape[1],
            q.shape[-1],
            head_dim_vo=v.shape[-1],
            causal=True,
            window_left=window_left,
            logits_soft_cap=(logit_cap if logit_cap != 0.0 else None),
            sm_scale=(
                softmax_scale
                if softmax_scale is not None
                else 1.0 / math.sqrt(q.shape[-1])
            ),
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            o_data_type=q.dtype,
        )
        result = wrapper.run(q, k, v, return_lse=return_lse)
        if return_lse:
            out, lse = result
            return out, lse / _FLASHINFER_LOG2_E
        return result

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="flashinfer_mha_extend_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={torch.float16, torch.bfloat16},
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128, 256}),
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({True}),
        },
        tags={"throughput"},
    )
    def flashinfer_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        is_causal: bool = False,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        wrapper = _get_paged_prefill_wrapper(q.device)
        page_size = k_cache.shape[1]
        batch_size = cache_seqlens.shape[0]
        max_pages_per_seq = page_table.shape[1]
        buffers_key = (page_table.device, batch_size, max_pages_per_seq)
        buffers = _paged_prefill_metadata_buffers.get(buffers_key)
        if buffers is None:
            buffers = (
                torch.empty(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=page_table.device,
                ),
                torch.empty(
                    max(1, batch_size * max_pages_per_seq),
                    dtype=torch.int32,
                    device=page_table.device,
                ),
                torch.empty(
                    batch_size,
                    dtype=torch.int32,
                    device=page_table.device,
                ),
            )
            _paged_prefill_metadata_buffers[buffers_key] = buffers
        paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = buffers
        _build_paged_prefill_metadata_kernel[(batch_size,)](
            page_table,
            cache_seqlens,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            page_table.stride(0),
            page_size,
            batch_size,
            max_pages_per_seq,
            BLOCK_P=min(1024, 1 << (max_pages_per_seq - 1).bit_length()),
        )
        wrapper.plan(
            cu_seqlens_q,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            q.shape[1],
            k_cache.shape[2],
            q.shape[-1],
            page_size,
            head_dim_vo=v_cache.shape[-1],
            causal=is_causal,
            sm_scale=(
                softmax_scale
                if softmax_scale is not None
                else 1.0 / math.sqrt(q.shape[-1])
            ),
            window_left=window_left,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            o_data_type=q.dtype,
            seq_lens=cache_seqlens,
            max_token_per_sequence=max_seqlen_q,
        )
        result = wrapper.run(
            q,
            (k_cache, v_cache),
            return_lse=return_lse,
            window_left=window_left,
            sinks=sinks,
        )

        out, lse = result
        return out, lse / _FLASHINFER_LOG2_E

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="flashinfer_trtllm_mha_extend_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={torch.float16, torch.bfloat16},
        priority=Priority.SPECIALIZED,
        traits={
            "is_causal": frozenset({False, True}),
            "head_dim": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        tags={"throughput"},
    )
    def flashinfer_trtllm_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        is_causal: bool = False,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        global _workspace_buffer
        if _workspace_buffer is None:
            _workspace_buffer = torch.zeros(
                512 * 1024 * 1024,
                dtype=torch.uint8,
                device=q.device,
            )
        cum_seq_lens_kv = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        # TRTLLM kernels require fp32 sinks.
        if sinks is not None and sinks.dtype != torch.float32:
            sinks = sinks.to(torch.float32)
        return trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(
                k_cache.permute(0, 2, 1, 3),
                v_cache.permute(0, 2, 1, 3),
            ),
            workspace_buffer=_workspace_buffer,
            block_tables=page_table,
            seq_lens=cache_seqlens,
            max_q_len=max_seqlen_q,
            max_kv_len=max_seqlen_k,
            bmm1_scale=(
                softmax_scale
                if softmax_scale is not None
                else 1.0 / math.sqrt(q.shape[-1])
            ),
            bmm2_scale=1.0,
            batch_size=cache_seqlens.shape[0],
            cum_seq_lens_q=cu_seqlens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            window_left=window_left,
            sinks=sinks,
            out_dtype=q.dtype,
            causal=is_causal,
        )

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="flashinfer_trtllm_mha_decode_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={torch.float16, torch.bfloat16},
        priority=Priority.SPECIALIZED,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        tags={"latency"},
    )
    def flashinfer_trtllm_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        global _workspace_buffer
        if _workspace_buffer is None:
            _workspace_buffer = torch.zeros(
                512 * 1024 * 1024,
                dtype=torch.uint8,
                device=q.device,
            )

        # TRTLLM kernels require fp32 sinks
        if sinks is not None and sinks.dtype != torch.float32:
            sinks = sinks.to(torch.float32)
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(
                k_cache.permute(0, 2, 1, 3),
                v_cache.permute(0, 2, 1, 3),
            ),
            workspace_buffer=_workspace_buffer,
            block_tables=page_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seqlen_k,
            bmm1_scale=(
                softmax_scale
                if softmax_scale is not None
                else 1.0 / math.sqrt(q.shape[-1])
            ),
            bmm2_scale=1.0,
            window_left=window_left,
            sinks=sinks,
            out_dtype=q.dtype,
        )


# ------------------------------------------------------------------------------
# Direct export
# ------------------------------------------------------------------------------

__all__ = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchMLAPagedAttentionWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
    "cudnn_batch_prefill_with_kv_cache",
    "trtllm_batch_context_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache_mla",
    "trtllm_ragged_attention_deepseek",
]
