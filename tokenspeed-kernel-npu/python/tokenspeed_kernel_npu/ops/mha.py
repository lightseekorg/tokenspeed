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

"""Ascend multi-head attention kernels."""

from __future__ import annotations

import math

import torch
import torch_npu

_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _causal_mask(device: torch.device) -> torch.Tensor:
    mask = _CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
        )
        _CAUSAL_MASKS[device] = mask
    return mask


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _check_options(
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
) -> None:
    if window_left >= 0:
        raise NotImplementedError("Ascend MHA does not support sliding windows")
    if logit_cap:
        raise NotImplementedError("Ascend MHA does not support logit caps")
    if sinks is not None:
        raise NotImplementedError("Ascend MHA does not support attention sinks")
    if return_lse:
        raise NotImplementedError("Ascend MHA does not return LSE")


def mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run causal variable-length MHA over uncached K/V."""
    del cu_seqlens, max_seqlen
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    # q/k/v usually arrive as views split out of one fused qkv projection, and
    # the non-paged FIA path rejects non-contiguous inputs.
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        atten_mask=_causal_mask(q.device),
        actual_seq_lengths=cu_seqlens_cpu[1:],
        actual_seq_lengths_kv=cu_seqlens_cpu[1:],
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=2,
    )
    return output


def mha_extend_with_kvcache(
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
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run variable-length MHA over a paged K/V cache."""
    del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, enable_pdl
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k_cache.flatten(2),
        v_cache.flatten(2),
        atten_mask=_causal_mask(q.device) if is_causal else None,
        actual_seq_lengths=cu_seqlens_q[1:],
        actual_seq_lengths_kv=cache_seqlens,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="TND",
        sparse_mode=3 if is_causal else 0,
        block_size=k_cache.shape[1],
    )
    return output


def mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    window_sink: int = 0,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run fixed-shape, graph-capturable decode over a paged K/V cache.

    ``max_seqlen_q == 1`` is plain decode. A larger value is a speculative
    verify window: request ``b``'s ``t``-th query attends the cache prefix
    through its own position, i.e. causal right-aligned so the window's last
    row sees every cached key and the first row sees ``seq_len - q_len + 1``.
    FIA expresses that with ``sparse_mode=3`` plus a triangular mask; its
    default (no mask) aligns the window at the start instead, which is the
    wrong span for verify.
    """
    _check_options(
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
    )
    if window_sink:
        raise NotImplementedError("Ascend MHA does not support attention sinks")
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("Ascend MHA does not support scaled FP8 cache")

    del max_seqlen_k, enable_pdl
    batch_size = cache_seqlens.shape[0]
    q_len = max_seqlen_q
    # Every query row needs a non-empty causal span, so the capture stub must
    # reach the window width; a stale length of 1 overflows the earlier rows.
    placeholder = max(1, q_len)
    actual_seq_lengths_kv = (
        [placeholder] * batch_size
        if torch.npu.is_current_stream_capturing()
        else cache_seqlens
    )
    verify_window = q_len > 1
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.reshape(batch_size, q_len, -1),
        k_cache.flatten(2),
        v_cache.flatten(2),
        atten_mask=_causal_mask(q.device) if verify_window else None,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=page_table,
        num_heads=q.shape[1],
        num_key_value_heads=k_cache.shape[2],
        scale=_scale(q, softmax_scale),
        input_layout="BSH",
        block_size=k_cache.shape[1],
        sparse_mode=3 if verify_window else 0,
    )
    return output.reshape_as(q)


__all__ = [
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]
