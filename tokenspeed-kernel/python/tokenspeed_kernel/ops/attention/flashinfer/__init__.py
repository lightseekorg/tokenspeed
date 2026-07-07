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

import tokenspeed_kernel.ops.attention.flashinfer.gated_delta_rule  # noqa: F401
import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import ErrorClass, Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

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
    from flashinfer.decode import (
        BatchDecodeWithPagedKVCacheWrapper,
        trtllm_batch_decode_with_kv_cache,
        trtllm_batch_decode_with_kv_cache_mla,
    )
    from flashinfer.prefill import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        cudnn_batch_prefill_with_kv_cache,
        trtllm_batch_context_with_kv_cache,
        trtllm_ragged_attention_deepseek,
    )

if platform.is_blackwell:
    from flashinfer.mla import (
        BatchMLAPagedAttentionWrapper,
        trtllm_batch_decode_with_kv_cache_mla,
    )


# ------------------------------------------------------------------------------
# Kernel registration
# ------------------------------------------------------------------------------

_workspace_buffer: torch.Tensor | None = None
_dsa_sparse_workspace_buffers: dict[torch.device, torch.Tensor] = {}
_DSA_SPARSE_WORKSPACE_BYTES = 384 * 1024 * 1024


def _get_dsa_sparse_workspace(device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    workspace = _dsa_sparse_workspace_buffers.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _DSA_SPARSE_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        _dsa_sparse_workspace_buffers[device] = workspace
    return workspace


def _flashinfer_trtllm_mla_kv_cache(
    kv_cache: torch.Tensor,
    page_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kv_cache.dtype != dtype:
        kv_cache = kv_cache.to(dtype)
    if kv_cache.dim() == 2:
        return kv_cache.view(-1, int(page_size), kv_cache.shape[-1]).unsqueeze(1)
    if kv_cache.dim() == 3 and kv_cache.shape[1] == 1:
        return (
            kv_cache.squeeze(1)
            .view(-1, int(page_size), kv_cache.shape[-1])
            .unsqueeze(1)
        )
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] == int(page_size) and kv_cache.shape[2] == 1:
            return kv_cache.permute(0, 2, 1, 3).contiguous()
        if kv_cache.shape[1] == 1 and kv_cache.shape[2] == int(page_size):
            return kv_cache.contiguous()
    raise ValueError(
        "kv_cache must be flat [slots, dim], flat [slots, 1, dim], or paged "
        f"[pages, page_size, 1, dim] for FlashInfer/TRTLLM sparse MLA, got {tuple(kv_cache.shape)}"
    )


def _topk_lens_or_count(
    topk_slots: torch.Tensor, topk_lens: torch.Tensor | None
) -> torch.Tensor:
    if topk_lens is not None:
        return topk_lens.to(device=topk_slots.device, dtype=torch.int32).contiguous()
    return (topk_slots >= 0).sum(dim=-1, dtype=torch.int32).contiguous()


if platform.is_nvidia and platform.is_hopper_plus:

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="flashinfer_trtllm_mha_extend_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "is_causal": frozenset({False, True}),
            "head_dim": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
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
            bmm1_scale=1.0 / math.sqrt(q.shape[-1]),
            bmm2_scale=1.0,
            batch_size=cache_seqlens.shape[0],
            cum_seq_lens_q=cu_seqlens_q,
            cum_seq_lens_kv=cu_seqlens_kv,
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
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        max_seqlen_q: int = 1,
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
            bmm1_scale=1.0 / math.sqrt(q.shape[-1]),
            bmm2_scale=1.0,
            window_left=window_left,
            sinks=sinks,
            out_dtype=q.dtype,
            q_len_per_req=max_seqlen_q,
        )

    @register_kernel(
        "attention",
        "dsa_decode",
        name="flashinfer_trtllm_dsa_decode",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_dsa_decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache is None:
            raise RuntimeError("FlashInfer/TRTLLM sparse MLA requires kv_cache")
        if return_lse:
            raise RuntimeError(
                "FlashInfer/TRTLLM sparse MLA does not support return_lse"
            )
        if logit_cap != 0.0:
            raise RuntimeError(
                "FlashInfer/TRTLLM sparse MLA does not support logit_cap"
            )
        if q.dim() == 3:
            num_tokens = q.shape[0]
            q_kernel = q.view(num_tokens, 1, q.shape[1], q.shape[2])
        elif q.dim() == 4:
            num_tokens = q.shape[0] * q.shape[1]
            q_kernel = q.reshape(num_tokens, 1, q.shape[2], q.shape[3])
        else:
            raise ValueError(f"unsupported q shape {tuple(q.shape)}")
        kv_dtype = q.dtype if q.dtype == torch.float8_e4m3fn else kv_cache.dtype
        kv = _flashinfer_trtllm_mla_kv_cache(kv_cache, page_size, kv_dtype)
        seq_lens = _topk_lens_or_count(topk_slots, topk_lens)
        result = trtllm_batch_decode_with_kv_cache_mla(
            query=q_kernel,
            kv_cache=kv,
            workspace_buffer=_get_dsa_sparse_workspace(q.device),
            qk_nope_head_dim=int(qk_nope_head_dim),
            kv_lora_rank=int(kv_lora_rank),
            qk_rope_head_dim=int(qk_rope_head_dim),
            block_tables=topk_slots.view(num_tokens, 1, -1),
            seq_lens=seq_lens,
            max_seq_len=int(max_seqlen_k),
            sparse_mla_top_k=topk_slots.shape[-1],
            bmm1_scale=float(k_scale) * float(softmax_scale),
            backend="trtllm-gen",
        )
        result = result.reshape(num_tokens, q_kernel.shape[2], int(kv_lora_rank))
        if out is not None:
            out.reshape_as(result).copy_(result)
            return out
        return result

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="flashinfer_trtllm_dsa_prefill",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_dsa_prefill(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return flashinfer_trtllm_dsa_decode(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            page_size=page_size,
            q_len_per_req=q_len_per_req,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
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
