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

if platform.is_nvidia and platform.is_blackwell:
    try:
        from flashinfer.mla import (
            BatchMLAPagedAttentionWrapper,
            trtllm_batch_decode_with_kv_cache_mla,
        )
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


# ------------------------------------------------------------------------------
# Kernel registration
# ------------------------------------------------------------------------------

_workspace_buffer: torch.Tensor | None = None
_ragged_prefill_workspaces: dict[torch.device, torch.Tensor] = {}
_ragged_prefill_wrappers: dict[torch.device, BatchPrefillWithRaggedKVCacheWrapper] = {}


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


if platform.is_nvidia and platform.is_blackwell:

    @register_kernel(
        "attention",
        "mha_prefill",
        name="flashinfer_mha_prefill",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={torch.float16, torch.bfloat16},
        priority=Priority.SPECIALIZED + 1,
        traits={
            "head_dim": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({True, False}),
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
        is_causal: bool = True,
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
            causal=is_causal,
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
        return wrapper.run(q, k, v, return_lse=return_lse)

    @register_kernel(
        "attention",
        "mha_prefill_with_kvcache",
        name="flashinfer_mha_prefill_with_kvcache_cached",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={torch.float16, torch.bfloat16},
        priority=Priority.SPECIALIZED + 2,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "prewritten_kv": frozenset({True}),
            "return_lse": frozenset({False}),
        },
        tags={"throughput"},
    )
    def flashinfer_mha_prefill_with_kvcache(
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        is_causal: bool = True,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        if k is not None or v is not None:
            raise ValueError("FlashInfer cached prefill requires prewritten KV cache")
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
        # TRTLLM kernels require fp32 sinks
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
        )

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="flashinfer_mha_decode_with_kvcache_cached",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        dtypes={torch.float16, torch.bfloat16},
        priority=Priority.SPECIALIZED + 2,
        traits={
            "query_len": frozenset({1}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        tags={"latency"},
    )
    def flashinfer_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        is_causal: bool = True,
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
    "trtllm_batch_context_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache_mla",
    "trtllm_ragged_attention_deepseek",
]
