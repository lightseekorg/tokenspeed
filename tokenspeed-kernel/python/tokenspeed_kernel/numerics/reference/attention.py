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


"""PyTorch ground-truth kernels for attention.

Every kernel here is written for clarity over speed — full ``[heads, q_len,
kv_len]`` score matrices per sequence, or token-by-token recurrences — in
FP32, so they register in the REFERENCE band: never auto-selected, reachable
only through ``solution="reference"`` or ``override=``. Tests and
``verify_kernel`` compare fused kernels against them; they mirror the public
entry points' keyword contracts exactly.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
_ATTENTION_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, *_FP8_DTYPES}
)


def _output_dtype(dtype: torch.dtype) -> torch.dtype:
    # FP8 attention kernels accumulate in FP32 and emit BF16.
    return torch.bfloat16 if dtype in _FP8_DTYPES else dtype


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_start_pos: int,
    is_causal: bool,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend one sequence in FP32.

    Args:
        q: ``[q_len, num_q_heads, head_dim]`` queries.
        k: ``[kv_len, num_kv_heads, head_dim]`` keys.
        v: ``[kv_len, num_kv_heads, head_dim_v]`` values.
        q_start_pos: Position of ``q[0]`` inside the KV sequence.
        is_causal: Mask keys after each query's position.
        window_left: Keys further than this many positions before a query are
            masked; ``-1`` disables the window.
        logit_cap: ``cap * tanh(score / cap)`` soft cap when positive.
        sinks: Optional ``[num_q_heads]`` extra logits that join the softmax
            denominator without a value.
        softmax_scale: Scale applied to the QK scores.

    Returns:
        FP32 ``[q_len, num_q_heads, head_dim_v]`` output and FP32
        ``[q_len, num_q_heads]`` natural-log log-sum-exp; fully masked rows
        produce zeros and ``-inf``.
    """
    q_len, num_q_heads, _ = q.shape
    kv_len, num_kv_heads, _ = k.shape
    group = num_q_heads // num_kv_heads
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * softmax_scale
    if logit_cap > 0.0:
        scores = logit_cap * torch.tanh(scores / logit_cap)

    q_pos = q_start_pos + torch.arange(q_len, device=q.device)
    k_pos = torch.arange(kv_len, device=q.device)
    mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=q.device)
    if is_causal:
        mask &= q_pos[:, None] >= k_pos[None, :]
    if window_left >= 0:
        mask &= q_pos[:, None] <= k_pos[None, :] + window_left
    scores = scores.masked_fill(~mask[None, :, :], float("-inf"))

    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
    probs = torch.exp(scores - row_max)
    denominator = probs.sum(dim=-1, keepdim=True)
    if sinks is not None:
        denominator = denominator + torch.exp(sinks.float()[:, None, None] - row_max)
    safe_denominator = torch.where(
        denominator > 0.0, denominator, torch.ones_like(denominator)
    )
    out = torch.einsum("hqk,khd->hqd", probs, v) / safe_denominator
    lse = torch.where(
        denominator > 0.0,
        torch.log(denominator) + row_max,
        torch.full_like(denominator, float("-inf")),
    )
    return out.permute(1, 0, 2), lse[:, :, 0].transpose(0, 1)


def _finish(
    outputs: list[torch.Tensor],
    lses: list[torch.Tensor],
    *,
    like: torch.Tensor,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = torch.cat(outputs, dim=0).to(_output_dtype(like.dtype))
    if return_lse:
        return out, torch.cat(lses, dim=0)
    return out


def _require_dense(
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise ValueError("the attention reference only covers dense q/k/v")


def _gather_paged_kv(
    cache: torch.Tensor,
    page_table_row: torch.Tensor,
    cache_len: int,
) -> torch.Tensor:
    """Return the first ``cache_len`` tokens of one request from a paged cache."""
    page_size = cache.shape[1]
    num_pages = (cache_len + page_size - 1) // page_size
    pages = page_table_row[:num_pages].long()
    return cache[pages].reshape(-1, cache.shape[2], cache.shape[3])[:cache_len]


@register_kernel(
    "attention",
    "mha_prefill",
    name="torch_mha_prefill",
    solution="reference",
    signatures=format_signatures(("q", "k", "v"), "dense", _ATTENTION_DTYPES),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
    softmax_scale: float | None,
    skip_softmax_threshold: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Causal MHA prefill over ragged sequences.

    Args:
        q: ``[total_q, num_q_heads, head_dim]`` queries.
        k: ``[total_kv, num_kv_heads, head_dim]`` keys.
        v: ``[total_kv, num_kv_heads, head_dim]`` values.
        cu_seqlens: ``[batch + 1]`` cumulative sequence lengths; unused
            because ``cu_seqlens_cpu`` carries the same boundaries.
        cu_seqlens_cpu: Host-side cumulative sequence lengths.
        max_seqlen: Maximum sequence length; unused.
        window_left: Exclusive left sliding-window size; ``-1`` is full.
        logit_cap: Soft cap applied to attention logits when positive.
        sinks: Optional ``[num_q_heads]`` attention sinks.
        return_lse: Also return ``[total_q, num_q_heads]`` FP32 natural-log
            log-sum-exp values.
        softmax_scale: QK scale; ``None`` uses ``1 / sqrt(head_dim)``.
        skip_softmax_threshold: Forwarded only when positive; accepted so
            callers comparing a skip-softmax kernel against exact attention
            can pass it, while the reference always computes exact attention.

    Returns:
        ``[total_q, num_q_heads, head_dim]`` output in ``q``'s dtype (BF16
        for FP8 inputs), plus the LSE when ``return_lse``.
    """
    del cu_seqlens, max_seqlen, skip_softmax_threshold
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for start, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
        out, lse = _attend(
            q[start:end],
            k[start:end],
            v[start:end],
            q_start_pos=0,
            is_causal=True,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            softmax_scale=softmax_scale,
        )
        outputs.append(out)
        lses.append(lse)
    return _finish(outputs, lses, like=q, return_lse=return_lse)


@register_kernel(
    "attention",
    "mha_extend_with_kvcache",
    name="torch_mha_extend_with_kvcache",
    solution="reference",
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", _ATTENTION_DTYPES
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
    softmax_scale: float | None,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA extend over a paged KV cache.

    Each request's queries attend all of its visible cached KV tokens. With
    ``is_causal`` the queries are the causal suffix of that KV sequence:
    query ``i`` sits at position ``cache_len - q_len + i``.

    Args:
        q: ``[total_q, num_q_heads, head_dim]`` queries.
        cu_seqlens_q: ``[batch + 1]`` cumulative query lengths.
        cu_seqlens_kv: ``[batch + 1]`` cumulative KV lengths; unused because
            ``cache_seqlens`` carries the visible lengths.
        k_cache: ``[num_pages, page_size, num_kv_heads, head_dim]`` keys.
        v_cache: ``[num_pages, page_size, num_kv_heads, head_dim]`` values.
        page_table: ``[batch, max_pages_per_seq]`` page indices.
        cache_seqlens: ``[batch]`` visible KV lengths.
        max_seqlen_q: Maximum query length; unused.
        max_seqlen_k: Maximum KV length; unused.
        is_causal: Treat queries as the causal suffix of the KV sequence.
        window_left: Exclusive left sliding-window size; ``-1`` is full.
        logit_cap: Soft cap applied to attention logits when positive.
        sinks: Optional ``[num_q_heads]`` attention sinks.
        return_lse: Also return ``[total_q, num_q_heads]`` FP32 natural-log
            log-sum-exp values.
        softmax_scale: QK scale; ``None`` uses ``1 / sqrt(head_dim)``.
        q_scale: Must be ``None``; block-scaled inputs are not covered.
        k_scale: Must be ``None``; block-scaled inputs are not covered.
        v_scale: Must be ``None``; block-scaled inputs are not covered.
        enable_pdl: Accepted for signature parity with fused kernels; unused.

    Returns:
        ``[total_q, num_q_heads, head_dim]`` output in ``q``'s dtype (BF16
        for FP8 inputs), plus the LSE when ``return_lse``.
    """
    del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, enable_pdl
    _require_dense(q_scale, k_scale, v_scale)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    cu_q = cu_seqlens_q.tolist()
    cache_lens = cache_seqlens.tolist()
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx, (start, end) in enumerate(zip(cu_q[:-1], cu_q[1:])):
        cache_len = int(cache_lens[batch_idx])
        out, lse = _attend(
            q[start:end],
            _gather_paged_kv(k_cache, page_table[batch_idx], cache_len),
            _gather_paged_kv(v_cache, page_table[batch_idx], cache_len),
            q_start_pos=max(cache_len - (end - start), 0),
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            softmax_scale=softmax_scale,
        )
        outputs.append(out)
        lses.append(lse)
    return _finish(outputs, lses, like=q, return_lse=return_lse)


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="torch_mha_decode_with_kvcache",
    solution="reference",
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", _ATTENTION_DTYPES
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
    softmax_scale: float | None,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA decode over a paged KV cache.

    Every request packs ``max_seqlen_q`` query tokens that form the causal
    tail of its ``cache_seqlens`` visible KV tokens.

    Args:
        q: ``[batch * max_seqlen_q, num_q_heads, head_dim]`` queries.
        k_cache: ``[num_pages, page_size, num_kv_heads, head_dim]`` keys.
        v_cache: ``[num_pages, page_size, num_kv_heads, head_dim]`` values.
        page_table: ``[batch, max_pages_per_seq]`` page indices.
        cache_seqlens: ``[batch]`` visible KV lengths including the decode
            tokens.
        max_seqlen_k: Maximum KV length; unused.
        max_seqlen_q: Query tokens packed per request.
        window_left: Exclusive left sliding-window size; ``-1`` is full.
        logit_cap: Soft cap applied to attention logits when positive.
        sinks: Optional ``[num_q_heads]`` attention sinks.
        return_lse: Also return ``[total_q, num_q_heads]`` FP32 natural-log
            log-sum-exp values.
        softmax_scale: QK scale; ``None`` uses ``1 / sqrt(head_dim)``.
        q_scale: Must be ``None``; block-scaled inputs are not covered.
        k_scale: Must be ``None``; block-scaled inputs are not covered.
        v_scale: Must be ``None``; block-scaled inputs are not covered.
        enable_pdl: Accepted for signature parity with fused kernels; unused.

    Returns:
        ``[batch * max_seqlen_q, num_q_heads, head_dim]`` output in ``q``'s
        dtype (BF16 for FP8 inputs), plus the LSE when ``return_lse``.
    """
    del max_seqlen_k, enable_pdl
    _require_dense(q_scale, k_scale, v_scale)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    batch = cache_seqlens.shape[0]
    cache_lens = cache_seqlens.tolist()
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx in range(batch):
        cache_len = int(cache_lens[batch_idx])
        start = batch_idx * max_seqlen_q
        out, lse = _attend(
            q[start : start + max_seqlen_q],
            _gather_paged_kv(k_cache, page_table[batch_idx], cache_len),
            _gather_paged_kv(v_cache, page_table[batch_idx], cache_len),
            q_start_pos=max(cache_len - max_seqlen_q, 0),
            is_causal=True,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            softmax_scale=softmax_scale,
        )
        outputs.append(out)
        lses.append(lse)
    return _finish(outputs, lses, like=q, return_lse=return_lse)


@register_kernel(
    "attention",
    "mla_prefill",
    name="torch_mla_prefill",
    solution="reference",
    signatures=format_signatures(("q", "k", "v"), "dense", _ATTENTION_DTYPES),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None,
    seq_lens_kv: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Non-absorbed MLA prefill over ragged Q and KV sequences.

    With ``is_causal`` the queries are the causal suffix of their KV
    sequence: query ``i`` sits at position ``kv_len - q_len + i``.

    Args:
        q: ``[total_q, num_q_heads, qk_head_dim]`` queries.
        k: ``[total_kv, num_kv_heads, qk_head_dim]`` keys.
        v: ``[total_kv, num_kv_heads, v_head_dim]`` values.
        cu_seqlens_q: ``[batch + 1]`` cumulative query lengths.
        cu_seqlens_kv: ``[batch + 1]`` cumulative KV lengths.
        max_seqlen_q: Maximum query length; unused.
        max_seqlen_kv: Maximum KV length; unused.
        softmax_scale: Scale applied to the QK scores.
        is_causal: Apply the causal-suffix mask.
        logit_cap: Soft cap applied to attention logits when positive.
        return_lse: Also return ``[total_q, num_q_heads]`` FP32 natural-log
            log-sum-exp values.
        out: Optional ``[total_q, num_q_heads, v_head_dim]`` destination.
        seq_lens_kv: Optional per-request KV lengths; unused.

    Returns:
        ``[total_q, num_q_heads, v_head_dim]`` output in ``q``'s dtype (BF16
        for FP8 inputs), plus the LSE when ``return_lse``.
    """
    del max_seqlen_q, max_seqlen_kv, seq_lens_kv
    cu_q = cu_seqlens_q.tolist()
    cu_kv = cu_seqlens_kv.tolist()
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for (q_start, q_end), (kv_start, kv_end) in zip(
        zip(cu_q[:-1], cu_q[1:]), zip(cu_kv[:-1], cu_kv[1:])
    ):
        attended, lse = _attend(
            q[q_start:q_end],
            k[kv_start:kv_end],
            v[kv_start:kv_end],
            q_start_pos=max((kv_end - kv_start) - (q_end - q_start), 0),
            is_causal=is_causal,
            window_left=-1,
            logit_cap=logit_cap,
            sinks=None,
            softmax_scale=softmax_scale,
        )
        outputs.append(attended)
        lses.append(lse)
    result = _finish(outputs, lses, like=q, return_lse=return_lse)
    if out is None:
        return result
    if return_lse:
        out.copy_(result[0])
        return out, result[1]
    out.copy_(result)
    return out


@register_kernel(
    "attention",
    "mla_extend_with_kvcache",
    name="torch_mla_extend_with_kvcache",
    solution="reference",
    signatures=format_signatures(("q", "kv_cache"), "dense", _ATTENTION_DTYPES),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mla_extend_with_kvcache(
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
    *,
    is_causal: bool,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Absorbed MLA extend over the compressed paged cache.

    The current query tokens are already written to the cache, so with
    ``is_causal`` query ``i`` of a request sits at position
    ``cache_len - q_len + i`` and sees everything up to itself.

    Args:
        q: ``[total_q, num_q_heads, kv_lora_rank + qk_rope_head_dim]`` packed
            absorbed queries.
        kv_cache: ``[num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]``
            compressed cache.
        page_table: ``[batch, max_pages_per_seq]`` page indices.
        cache_seqlens: ``[batch]`` visible KV lengths including the queries.
        cu_seqlens_q: ``[batch + 1]`` packed query boundaries.
        cu_seqlens_kv: ``[batch + 1]`` packed KV boundaries; unused.
        max_seqlen_q: Maximum query length; unused.
        max_seqlen_k: Maximum KV length; unused.
        qk_nope_head_dim: Original non-RoPE head dim; unused.
        kv_lora_rank: Latent rank and the output head dim.
        qk_rope_head_dim: RoPE head dim; unused beyond the query width.
        softmax_scale: Scale applied to the QK scores.
        is_causal: Apply the causal-suffix mask.
        logit_cap: Soft cap applied to attention logits when positive.
        return_lse: Also return ``[total_q, num_q_heads]`` FP32 natural-log
            log-sum-exp values.
        out: Optional ``[total_q, num_q_heads, kv_lora_rank]`` destination.

    Returns:
        ``[total_q, num_q_heads, kv_lora_rank]`` latent output in ``q``'s
        dtype (BF16 for FP8 inputs), plus the LSE when ``return_lse``.
    """
    del cu_seqlens_kv, max_seqlen_q, max_seqlen_k, qk_nope_head_dim
    del qk_rope_head_dim
    cu_q = cu_seqlens_q.tolist()
    cache_lens = cache_seqlens.tolist()
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx, (start, end) in enumerate(zip(cu_q[:-1], cu_q[1:])):
        cache_len = int(cache_lens[batch_idx])
        kv = _gather_paged_kv(kv_cache, page_table[batch_idx], cache_len)
        attended, lse = _attend(
            q[start:end],
            kv,
            kv[:, :, :kv_lora_rank],
            q_start_pos=max(cache_len - (end - start), 0),
            is_causal=is_causal,
            window_left=-1,
            logit_cap=logit_cap,
            sinks=None,
            softmax_scale=softmax_scale,
        )
        outputs.append(attended)
        lses.append(lse)
    result = _finish(outputs, lses, like=q, return_lse=return_lse)
    if out is None:
        return result
    if return_lse:
        out.copy_(result[0])
        return out, result[1]
    out.copy_(result)
    return out


def _mla_decode_visible_range(
    *,
    cache_len: int,
    q_len: int,
    query_idx: int,
    batch_idx: int,
    window_left: int,
    noncausal_block_size: int,
) -> tuple[int, int]:
    """Return the ``[start, end)`` KV range one MLA decode query sees.

    A proposal block (``noncausal_block_size > 1``) sees the whole block; the
    block position comes from the query axis when the block sits there and
    from the batch axis when it is flattened. The window bounds the history
    visible to the block's first position. Ordinary decode is the causal
    tail of the cache.
    """
    block = noncausal_block_size > 1 or window_left >= 0
    if not block:
        return 0, cache_len - q_len + query_idx + 1
    if q_len == noncausal_block_size:
        block_position = query_idx
    else:
        block_position = batch_idx % noncausal_block_size
    start = 0
    if window_left >= 0:
        context_len = cache_len - noncausal_block_size
        start = max(0, context_len - window_left + block_position)
    return start, cache_len


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="torch_mla_decode_with_kvcache",
    solution="reference",
    signatures=format_signatures(("q", "kv_cache"), "dense", _ATTENTION_DTYPES),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mla_decode_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None,
    window_left: int = -1,
    noncausal_block_size: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Absorbed MLA decode over the compressed paged cache.

    Args:
        q: ``[batch, q_len, num_q_heads, kv_lora_rank + qk_rope_head_dim]``
            absorbed queries.
        kv_cache: ``[num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]``
            compressed cache; the leading ``kv_lora_rank`` elements are the
            latent value.
        page_table: ``[batch, max_pages_per_seq]`` page indices.
        cache_seqlens: ``[batch]`` visible KV lengths.
        max_seqlen_k: Maximum KV length; unused.
        qk_nope_head_dim: Original non-RoPE head dim; unused.
        kv_lora_rank: Latent rank and the output head dim.
        qk_rope_head_dim: RoPE head dim; unused beyond the query width.
        softmax_scale: Scale applied to the QK scores.
        logit_cap: Soft cap applied to attention logits when positive.
        return_lse: Also return ``[batch, q_len, num_q_heads]`` FP32
            natural-log log-sum-exp values.
        out: Optional ``[batch, q_len, num_q_heads, kv_lora_rank]``
            destination.
        window_left: History visible to a proposal block's first position;
            ``-1`` is full attention. Forwarded only for proposal blocks.
        noncausal_block_size: Proposal rows per request; one for ordinary
            causal decode. Forwarded only for proposal blocks.

    Returns:
        ``[batch, q_len, num_q_heads, kv_lora_rank]`` latent output in
        ``q``'s dtype (BF16 for FP8 inputs), plus the LSE when
        ``return_lse``.
    """
    del max_seqlen_k, qk_nope_head_dim, qk_rope_head_dim
    batch, q_len, num_heads, _ = q.shape
    cache_lens = cache_seqlens.tolist()
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx in range(batch):
        cache_len = int(cache_lens[batch_idx])
        kv = _gather_paged_kv(kv_cache, page_table[batch_idx], cache_len)
        for query_idx in range(q_len):
            start, end = _mla_decode_visible_range(
                cache_len=cache_len,
                q_len=q_len,
                query_idx=query_idx,
                batch_idx=batch_idx,
                window_left=window_left,
                noncausal_block_size=noncausal_block_size,
            )
            attended, lse = _attend(
                q[batch_idx, query_idx : query_idx + 1],
                kv[start:end],
                kv[start:end, :, :kv_lora_rank],
                q_start_pos=0,
                is_causal=False,
                window_left=-1,
                logit_cap=logit_cap,
                sinks=None,
                softmax_scale=softmax_scale,
            )
            outputs.append(attended)
            lses.append(lse)
    output = torch.cat(outputs, dim=0).reshape(batch, q_len, num_heads, kv_lora_rank)
    output = output.to(_output_dtype(q.dtype))
    if out is not None:
        out.copy_(output)
        output = out
    if return_lse:
        return output, torch.cat(lses, dim=0).reshape(batch, q_len, num_heads)
    return output


def _l2norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)


@register_kernel(
    "attention",
    "gdn_chunk_prefill",
    name="torch_gdn_chunk_prefill",
    solution="reference",
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    traits={
        "qk_l2norm": frozenset({False, True}),
        "output_h": frozenset({False}),
    },
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool,
    output_final_state: bool,
    output_h: bool,
) -> GdnChunkPrefillResult:
    """Gated Delta Net prefill as a token-by-token FP32 recurrence.

    Per value head ``h`` (sharing q/k head ``h // group``) and token ``t``::

        S = exp(g[t, h]) * S
        S = S + k[t] (beta[t, h] * (v[t] - k[t] @ S))^T
        out[t, h] = scale * q[t] @ S

    The state follows the public K-last ``[N, Hv, V, K]`` contract; the math
    runs on its ``[K, V]`` transpose.

    Args:
        q: ``[1, total_tokens, num_q_heads, head_dim]`` queries.
        k: ``[1, total_tokens, num_q_heads, head_dim]`` keys.
        v: ``[1, total_tokens, num_v_heads, head_v_dim]`` values.
        g: ``[1, total_tokens, num_v_heads]`` log-space forget gate.
        beta: ``[1, total_tokens, num_v_heads]`` write gate.
        scale: Output scale; ``None`` uses ``head_dim ** -0.5``.
        initial_state: ``[batch, num_v_heads, head_v_dim, head_dim]`` K-last
            recurrent state per sequence.
        cu_seqlens: ``[batch + 1]`` cumulative sequence lengths.
        qk_l2norm: L2-normalize ``q`` and ``k`` (eps ``1e-6``) first.
        output_final_state: Return the final state in ``initial_state``'s
            layout and dtype.
        output_h: Must be false; checkpoints are backend-native.

    Returns:
        ``GdnChunkPrefillResult`` with ``out`` in ``q``'s dtype.
    """
    if output_h:
        raise ValueError("the GDN reference does not emit recurrent checkpoints")
    if q.shape[0] != 1:
        raise ValueError("gdn_chunk_prefill packs every sequence into batch 1")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q_f, k_f = q.float(), k.float()
    if qk_l2norm:
        q_f, k_f = _l2norm(q_f, 1e-6), _l2norm(k_f, 1e-6)
    v_f, g_f, beta_f = v.float(), g.float(), beta.float()
    num_q_heads = q.shape[2]
    num_v_heads = v.shape[2]
    group = num_v_heads // num_q_heads
    # K-last storage -> [K, V] math layout.
    states = initial_state.float().transpose(-2, -1)

    out = torch.empty(v.shape, dtype=torch.float32, device=q.device)
    final_states: list[torch.Tensor] = []
    bounds = cu_seqlens.tolist()
    for seq_idx, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = states[seq_idx].clone()  # [Hv, K, V]
        for token in range(start, end):
            q_t = q_f[0, token].repeat_interleave(group, dim=0)  # [Hv, K]
            k_t = k_f[0, token].repeat_interleave(group, dim=0)
            state = torch.exp(g_f[0, token])[:, None, None] * state
            prediction = torch.einsum("hk,hkv->hv", k_t, state)
            delta = beta_f[0, token][:, None] * (v_f[0, token] - prediction)
            state = state + k_t[:, :, None] * delta[:, None, :]
            out[0, token] = scale * torch.einsum("hk,hkv->hv", q_t, state)
        final_states.append(state)

    final_state = None
    if output_final_state:
        final_state = (
            torch.stack(final_states, dim=0).transpose(-2, -1).to(initial_state.dtype)
        )
    return GdnChunkPrefillResult(out=out.to(q.dtype), final_state=final_state)


__all__ = [
    "torch_gdn_chunk_prefill",
    "torch_mha_decode_with_kvcache",
    "torch_mha_extend_with_kvcache",
    "torch_mha_prefill",
    "torch_mla_decode_with_kvcache",
    "torch_mla_extend_with_kvcache",
    "torch_mla_prefill",
]
