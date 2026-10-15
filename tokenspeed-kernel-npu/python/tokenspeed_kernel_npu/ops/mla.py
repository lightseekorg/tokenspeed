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

"""Ascend Multi-head Latent Attention kernels."""

from __future__ import annotations

import math

import torch
import torch_npu
from tokenspeed_kernel_npu.ops.mla_packed import (
    packed_mla_decode_op,
    packed_mla_prefill_op,
)

_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _causal_mask(device: torch.device) -> torch.Tensor:
    mask = _CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
        )
        _CAUSAL_MASKS[device] = mask
    return mask


def _lse(lse: torch.Tensor, tokens: int, heads: int) -> torch.Tensor:
    return lse.reshape(tokens, heads)


def _result(
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if out is not None:
        out.copy_(output)
        output = out
    return (output, lse) if return_lse else output


def mla_normalize_project_query(
    query: torch.Tensor,
    kv: torch.Tensor,
    query_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    tail_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize Q/KV latents, update KV in place, and project Q."""
    if tail_out is not None:
        raise NotImplementedError("Ascend MLA query projection does not split output")
    query_fp32 = query.float()
    query_norm = query_fp32 * torch.rsqrt(
        query_fp32.square().mean(dim=-1, keepdim=True) + eps
    )
    query_norm = (query_norm * query_norm_weight.float()).to(query.dtype)
    kv_fp32 = kv.float()
    kv.copy_(
        (
            kv_fp32
            * torch.rsqrt(kv_fp32.square().mean(dim=-1, keepdim=True) + eps)
            * kv_norm_weight.float()
        ).to(kv.dtype)
    )
    out.copy_(torch.mm(query_norm, projection_weight.t()))
    return out


def mla_project_value(
    attention: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Project absorbed MLA values and apply the Lite output gate in place."""
    projected = torch.bmm(attention.transpose(0, 1).contiguous(), weight)
    out.view(attention.shape[0], weight.shape[0], weight.shape[2]).copy_(
        projected.transpose(0, 1)
    )
    if gate is not None:
        out.mul_(torch.sigmoid(gate).to(out.dtype))
    return out


def mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    seq_lens_kv: torch.Tensor | None,
    is_causal: bool,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run explicit variable-length MLA Prefill."""
    del max_seqlen_q, max_seqlen_kv, seq_lens_kv
    if logit_cap:
        raise NotImplementedError("Ascend MLA does not support logit caps")
    output, lse = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:],
        actual_seq_lengths_kv=cu_seqlens_kv[1:],
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=softmax_scale,
        input_layout="TND",
        sparse_mode=2 if is_causal else 0,
        softmax_lse_flag=return_lse,
    )
    return _result(
        output,
        _lse(lse, q.shape[0], q.shape[1]) if return_lse else lse,
        return_lse=return_lse,
        out=out,
    )


def mla_extend_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    is_causal: bool,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run absorbed variable-length MLA over the paged latent cache."""
    del cu_seqlens_kv, qk_nope_head_dim
    if logit_cap:
        raise NotImplementedError("Ascend MLA does not support logit caps")
    q_nope, q_aux = q.split((kv_lora_rank, qk_rope_head_dim), dim=-1)
    packed_op = (
        packed_mla_prefill_op(q, kv_cache, page_table, max_seqlen_q, max_seqlen_k)
        if is_causal and (kv_lora_rank, qk_rope_head_dim) == (512, 64)
        else None
    )
    if packed_op is not None:
        lengths = (
            cache_seqlens.to("cpu").tolist()
            if isinstance(cache_seqlens, torch.Tensor)
            else cache_seqlens
        )
        query_lengths = cu_seqlens_q[1:]
        if isinstance(query_lengths, torch.Tensor):
            query_lengths = query_lengths.to("cpu").tolist()
        packed_out = (
            out
            if out is not None and out.is_contiguous()
            else torch.empty(q_nope.shape, dtype=q.dtype, device=q.device)
        )
        packed_lse = torch.empty(
            (q.shape[0], q.shape[1], 1) if return_lse else (0,),
            dtype=torch.float32,
            device=q.device,
        )
        # Only Q is materialized; the persistent cache remains a packed view.
        # Once dispatched, errors propagate without retrying the cache writer.
        output, lse = packed_op(
            q_nope.contiguous(),
            q_aux.contiguous(),
            kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], 576),
            page_table,
            lengths,
            query_lengths,
            _causal_mask(q.device),
            num_heads=q.shape[1],
            scale=softmax_scale,
            return_lse=return_lse,
            out=packed_out,
            lse=packed_lse,
        )
        return _result(
            output,
            _lse(lse, q.shape[0], q.shape[1]) if return_lse else lse,
            return_lse=return_lse,
            out=out,
        )
    k_nope, k_aux = kv_cache.split((kv_lora_rank, qk_rope_head_dim), dim=-1)
    output, lse = torch_npu.npu_fused_infer_attention_score(
        q_nope,
        k_nope.flatten(2),
        k_nope.flatten(2),
        query_rope=q_aux,
        key_rope=k_aux.flatten(2),
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:],
        actual_seq_lengths_kv=cache_seqlens,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=1,
        scale=softmax_scale,
        input_layout="TND",
        block_size=kv_cache.shape[1],
        sparse_mode=3 if is_causal else 0,
        softmax_lse_flag=return_lse,
    )
    return _result(
        output,
        _lse(lse, q.shape[0], q.shape[1]) if return_lse else lse,
        return_lse=return_lse,
        out=out,
    )


def mla_decode_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run graph-capturable absorbed MLA Decode."""
    del qk_nope_head_dim
    if logit_cap:
        raise NotImplementedError("Ascend MLA does not support logit caps")
    batch, q_len, heads, _ = q.shape
    if q_len != 1:
        raise NotImplementedError("Ascend MLA Decode supports one query per request")
    q_nope, q_aux = q.split((kv_lora_rank, qk_rope_head_dim), dim=-1)
    packed_op = (
        packed_mla_decode_op(q, kv_cache, page_table, max_seqlen_k)
        if (kv_lora_rank, qk_rope_head_dim) == (512, 64)
        else None
    )
    live_lengths = (
        [1] * batch if torch.npu.is_current_stream_capturing() else cache_seqlens
    )
    # Capture layout copies as ordinary graph nodes, outside FIA task-update.
    # Replay must refresh them after the packed cache is written.
    q_nope = q_nope.reshape(batch, 1, heads * kv_lora_rank).contiguous()
    q_aux = q_aux.reshape(batch, 1, heads * qk_rope_head_dim).contiguous()
    if packed_op is not None:
        # Allocate outputs outside the task-update group. The registered .out
        # handler reuses these addresses and updates only the Host lengths.
        packed_out = torch.empty_like(q_nope)
        packed_lse = torch.empty(
            (batch, heads, 1) if return_lse else (0,),
            dtype=torch.float32,
            device=q.device,
        )
        lengths = (
            live_lengths.to("cpu").tolist()
            if isinstance(live_lengths, torch.Tensor)
            else live_lengths
        )
        output, lse = packed_op(
            q_nope,
            q_aux,
            kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], 576),
            page_table,
            lengths,
            num_heads=heads,
            scale=softmax_scale,
            return_lse=return_lse,
            out=packed_out,
            lse=packed_lse,
        )
        return _result(
            output.reshape(batch, q_len, heads, kv_lora_rank),
            _lse(lse, batch, heads) if return_lse else lse,
            return_lse=return_lse,
            out=out,
        )
    k_nope, k_aux = kv_cache.split((kv_lora_rank, qk_rope_head_dim), dim=-1)
    k_nope = k_nope.flatten(2).contiguous()
    k_aux = k_aux.flatten(2).contiguous()
    output, lse = torch_npu.npu_fused_infer_attention_score(
        q_nope,
        k_nope,
        k_nope,
        query_rope=q_aux,
        key_rope=k_aux,
        actual_seq_lengths_kv=live_lengths,
        block_table=page_table,
        num_heads=heads,
        num_key_value_heads=1,
        scale=softmax_scale,
        input_layout="BSH",
        block_size=kv_cache.shape[1],
        sparse_mode=0,
        softmax_lse_flag=return_lse,
    )
    output = output.reshape(batch, q_len, heads, kv_lora_rank)
    return _result(
        output,
        _lse(lse, batch, heads) if return_lse else lse,
        return_lse=return_lse,
        out=out,
    )


def attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float,
    inplace: bool = False,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two natural-log partial attention states."""
    del enable_pdl
    if not math.isclose(lse_scale_log2, math.log2(math.e)):
        raise NotImplementedError("Ascend MLA merge requires natural-log LSE")
    merged, _ = torch_npu.npu_attention_update(
        [lse_a.reshape(-1), lse_b.reshape(-1)],
        [out_a.reshape(-1, out_a.shape[-1]), out_b.reshape(-1, out_b.shape[-1])],
        update_type=0,
    )
    merged = merged.reshape_as(out_a)
    merged_lse = torch.logaddexp(lse_a, lse_b)
    if inplace:
        out_a.copy_(merged)
        lse_a.copy_(merged_lse)
        return out_a, lse_a
    return merged, merged_lse


__all__ = [
    "attn_merge_state",
    "mla_decode_with_kvcache",
    "mla_extend_with_kvcache",
    "mla_normalize_project_query",
    "mla_prefill",
    "mla_project_value",
]
