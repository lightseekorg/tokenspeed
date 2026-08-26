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

"""Portable KPool decode and prefill selection entry points."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.cute_dsl.dsa_topk import (
    cute_dsl_decode_topk,
    has_cute_dsl_decode_topk,
)
from tokenspeed_kernel.ops.attention.triton.kpool_expand import (
    expand_kpool_to_flat_kv,
)
from tokenspeed_kernel.ops.attention.triton.kpool_score import (
    score_kpool_dense,
    select_kpool_chunked,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_DEFAULT_CHUNK_POOLS = 8192
_MAX_DENSE_LOGITS_BYTES = 64 * 1024 * 1024
_CUTE_DSL_TOPK_POOLS = 512
_CUTE_DSL_Q_LENS = frozenset({1, 2, 3, 4, 5, 6})

_TRAITS = {
    "head_dim": frozenset({128}),
    "pool_size": frozenset({2, 4, 8, 16}),
    "page_size": frozenset({16, 64}),
    "index_k_format": frozenset({"fp8_scaled"}),
    "score_activation": frozenset({"relu", "none"}),
    "topk_layout": frozenset({"global_slots"}),
}


@triton.jit
def _kpool_decode_metadata_kernel(
    seq_lens,
    req_ids,
    causal_lens,
    num_tokens: tl.constexpr,
    q_len_per_req: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = token < num_tokens
    req = token // q_len_per_req
    q_offset = token - req * q_len_per_req
    seq_len = tl.load(seq_lens + req, mask=mask, other=0).to(tl.int32)
    causal_len = tl.maximum(
        seq_len - (q_len_per_req - 1) + q_offset,
        0,
    )
    tl.store(req_ids + token, req, mask=mask)
    tl.store(causal_lens + token, causal_len, mask=mask)


def _prepare_kpool_decode_metadata(
    seq_lens: torch.Tensor,
    num_tokens: int,
    q_len_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build request ids and visible lengths in one device launch."""
    req_ids = torch.empty(num_tokens, dtype=torch.int32, device=seq_lens.device)
    causal_lens = torch.empty_like(req_ids)
    block = min(triton.next_power_of_2(num_tokens), 1024)
    _kpool_decode_metadata_kernel[(triton.cdiv(num_tokens, block),)](
        seq_lens,
        req_ids,
        causal_lens,
        num_tokens=num_tokens,
        q_len_per_req=q_len_per_req,
        BLOCK=block,
        num_warps=1,
        num_stages=1,
    )
    return req_ids, causal_lens


def _empty_result(
    q: torch.Tensor,
    topk_pools: int,
    pool_size: int,
    append_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = topk_pools * pool_size + (pool_size - 1 if append_tail else 0)
    return (
        torch.empty((0, width), dtype=torch.int32, device=q.device),
        torch.empty((0,), dtype=torch.int32, device=q.device),
    )


def _select_pools_dense(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    topk_pools: int,
    softmax_scale: float,
    apply_relu: bool,
    max_num_pools: int,
    use_cute_dsl_topk: bool,
) -> torch.Tensor:
    num_tokens = q.shape[0]
    if max_num_pools <= topk_pools:
        return torch.empty((num_tokens, topk_pools), dtype=torch.int32, device=q.device)

    row_bytes = max_num_pools * torch.float32.itemsize
    rows_per_tile = min(num_tokens, max(1, _MAX_DENSE_LOGITS_BYTES // row_bytes))
    pool_indices = (
        torch.empty(
            (num_tokens, topk_pools),
            dtype=torch.int32 if use_cute_dsl_topk else torch.int64,
            device=q.device,
        )
        if use_cute_dsl_topk or rows_per_tile < num_tokens
        else None
    )
    logits_workspace = torch.empty(
        (rows_per_tile, max_num_pools), dtype=torch.float32, device=q.device
    )
    pool_lens = None
    if use_cute_dsl_topk:
        pool_lens = torch.div(causal_lens, pool_size, rounding_mode="floor").to(
            torch.int32
        )

    for start in range(0, num_tokens, rows_per_tile):
        end = min(start + rows_per_tile, num_tokens)
        logits = score_kpool_dense(
            q[start:end],
            pooled_k_cache,
            weights[start:end],
            causal_lens[start:end],
            req_ids[start:end],
            index_block_table,
            pool_size=pool_size,
            page_size=page_size,
            softmax_scale=softmax_scale,
            apply_relu=apply_relu,
            max_num_pools=max_num_pools,
            out=logits_workspace[: end - start],
            length_masked_consumer=use_cute_dsl_topk,
        )
        if use_cute_dsl_topk:
            assert pool_indices is not None
            cute_dsl_decode_topk(
                logits,
                pool_lens[start:end],
                topk_pools,
                next_n=1,
                out=pool_indices[start:end],
            )
        else:
            selected = torch.topk(logits, k=topk_pools, dim=-1, sorted=False).indices
            if pool_indices is None:
                return selected
            pool_indices[start:end].copy_(selected)
    assert pool_indices is not None
    return pool_indices


@register_kernel(
    "attention",
    "kpool_decode_topk",
    name="triton_dense_kpool_decode_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
    traits=_TRAITS,
    priority=Priority.PORTABLE,
    tags={"portability", "kpool", "dense-score"},
)
def triton_dense_kpool_decode_topk(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    index_block_table: torch.Tensor,
    kv_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    kv_page_size: int,
    topk_pools: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    apply_relu: bool = True,
    append_tail: bool = True,
    chunk_pools: int = _DEFAULT_CHUNK_POOLS,
    max_seq_len: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select decode pools and expand them to global FlatKV slots.

    Args:
        q: Index queries shaped ``[tokens, heads, head_dim]``.
        pooled_k_cache: Packed paged FP8 KPool cache.
        weights: Per-query head weights.
        seq_lens: Final raw-token length per request.
        index_block_table: Pooled-cache page table.
        kv_block_table: FlatKV page table.
        pool_size: Raw tokens represented by each pool.
        page_size: Pooled rows per index page.
        kv_page_size: Raw tokens per FlatKV page.
        topk_pools: Number of pools to select.
        softmax_scale: Per-head score scale.
        q_len_per_req: Query rows per request.
        apply_relu: Apply the indexer ReLU.
        append_tail: Append the visible partial pool.
        chunk_pools: Accepted for compatibility with the bounded portable path.
        max_seq_len: Optional static context bound for CUDA Graph replay.
        out: Optional expanded-slot output.
        lens_out: Optional valid-count output.

    Returns:
        Global FlatKV slots and valid counts.
    """
    del chunk_pools
    pool_size, topk_pools = int(pool_size), int(topk_pools)
    q_len_per_req = int(q_len_per_req)
    num_tokens = q.shape[0]
    if q_len_per_req < 1 or num_tokens % q_len_per_req:
        raise ValueError(
            f"q_len_per_req={q_len_per_req} must divide tokens={num_tokens}"
        )
    num_reqs = num_tokens // q_len_per_req
    if seq_lens.numel() < num_reqs:
        raise ValueError(
            f"seq_lens must cover {num_reqs} requests, got {seq_lens.numel()}"
        )
    if num_tokens == 0:
        return _empty_result(q, topk_pools, pool_size, append_tail)
    if not q.is_cuda:
        raise RuntimeError("KPool decode top-k requires CUDA tensors")

    device = q.device
    seq_lens = seq_lens.to(device=device).contiguous()
    req_ids, causal_lens = _prepare_kpool_decode_metadata(
        seq_lens,
        num_tokens,
        q_len_per_req,
    )
    use_cute = (
        has_cute_dsl_decode_topk()
        and topk_pools == _CUTE_DSL_TOPK_POOLS
        and q_len_per_req in _CUTE_DSL_Q_LENS
    )
    if use_cute or torch.cuda.is_current_stream_capturing():
        max_num_pools = max(index_block_table.shape[1] * int(page_size), 1)
        if max_seq_len is not None:
            if int(max_seq_len) < 0:
                raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
            max_num_pools = min(
                max_num_pools,
                max(int(max_seq_len) // pool_size, 1),
            )
    else:
        max_num_pools = max(int(causal_lens.max().item()) // pool_size, 1)

    pool_indices = _select_pools_dense(
        q.contiguous(),
        pooled_k_cache,
        weights.contiguous(),
        causal_lens,
        req_ids,
        index_block_table.to(device=device, dtype=torch.int32).contiguous(),
        pool_size=pool_size,
        page_size=int(page_size),
        topk_pools=topk_pools,
        softmax_scale=softmax_scale,
        apply_relu=apply_relu,
        max_num_pools=max_num_pools,
        use_cute_dsl_topk=use_cute,
    )
    return expand_kpool_to_flat_kv(
        pool_indices,
        causal_lens,
        req_ids,
        kv_block_table.to(device=device, dtype=torch.int32).contiguous(),
        pool_size=pool_size,
        kv_page_size=int(kv_page_size),
        append_tail=append_tail,
        out=out,
        lens_out=lens_out,
    )


@register_kernel(
    "attention",
    "kpool_prefill_topk",
    name="triton_kpool_prefill_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
    traits=_TRAITS,
    priority=Priority.PORTABLE,
    tags={"portability", "kpool"},
)
def triton_kpool_prefill_topk(
    q: torch.Tensor,
    pooled_k_cache: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    index_block_table: torch.Tensor,
    kv_block_table: torch.Tensor,
    *,
    pool_size: int,
    page_size: int,
    kv_page_size: int,
    topk_pools: int,
    softmax_scale: float,
    apply_relu: bool = True,
    append_tail: bool = True,
    chunk_pools: int = _DEFAULT_CHUNK_POOLS,
    req_ids: torch.Tensor | None = None,
    causal_lens: torch.Tensor | None = None,
    pool_workspace_slots: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    max_num_pools: int | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ragged-prefill pools and expand them to FlatKV slots.

    Args:
        q: Packed index queries shaped ``[tokens, heads, head_dim]``.
        pooled_k_cache: Packed paged FP8 KPool cache.
        weights: Per-query head weights.
        positions: Absolute query positions.
        query_start_loc: Packed request boundaries.
        index_block_table: Pooled-cache page table.
        kv_block_table: FlatKV page table.
        pool_size: Raw tokens represented by each pool.
        page_size: Pooled rows per index page.
        kv_page_size: Raw tokens per FlatKV page.
        topk_pools: Number of pools to select.
        softmax_scale: Per-head score scale.
        apply_relu: Apply the indexer ReLU.
        append_tail: Append the visible partial pool.
        chunk_pools: Pools scored per bounded window.
        req_ids: Optional precomputed request ids.
        causal_lens: Optional precomputed visible lengths.
        pool_workspace_slots: Optional performant-path pool plan.
        row_starts: Optional performant-path row starts.
        row_ends: Optional performant-path row ends.
        max_num_pools: Optional host-known pool bound.
        max_logits_bytes: Optional cap for portable scoring and sort workspaces.
        out: Optional expanded-slot output.
        lens_out: Optional valid-count output.

    Returns:
        Global FlatKV slots and valid counts.
    """
    pool_size, topk_pools = int(pool_size), int(topk_pools)
    num_tokens = q.shape[0]
    if query_start_loc.dim() != 1 or query_start_loc.numel() < 2:
        raise ValueError("query_start_loc must contain request boundaries")
    if positions.numel() != num_tokens:
        raise ValueError(f"positions must have {num_tokens} entries")
    if num_tokens == 0:
        return _empty_result(q, topk_pools, pool_size, append_tail)
    if not q.is_cuda:
        raise RuntimeError("KPool prefill top-k requires CUDA tensors")

    del pool_workspace_slots, row_starts, row_ends
    device = q.device
    query_start_loc = query_start_loc.to(device=device, dtype=torch.int32).contiguous()
    if req_ids is None:
        query_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.long)
        if int(query_lens.sum().item()) != num_tokens:
            raise ValueError("query_start_loc does not cover every query token")
        req_ids = torch.repeat_interleave(
            torch.arange(query_lens.numel(), device=device, dtype=torch.int32),
            query_lens,
        )
    elif req_ids.numel() != num_tokens:
        raise ValueError(f"req_ids must have {num_tokens} entries")
    else:
        req_ids = req_ids.to(device=device, dtype=torch.int32).contiguous()

    if causal_lens is None:
        causal_lens = (
            positions.to(device=device, dtype=torch.int32).add(1).clamp_min_(0)
        )
    elif causal_lens.numel() != num_tokens:
        raise ValueError(f"causal_lens must have {num_tokens} entries")
    else:
        causal_lens = causal_lens.to(device=device, dtype=torch.int32).contiguous()

    if max_num_pools is None:
        max_num_pools = int(causal_lens.max().item()) // pool_size
    max_num_pools = max(int(max_num_pools), 1)
    pool_indices = torch.empty(
        (num_tokens, topk_pools), dtype=torch.int32, device=device
    )
    if max_num_pools > topk_pools:
        pool_indices = select_kpool_chunked(
            q.contiguous(),
            pooled_k_cache,
            weights.to(torch.float32).contiguous(),
            causal_lens,
            req_ids,
            index_block_table.to(device=device, dtype=torch.int32).contiguous(),
            pool_size=pool_size,
            page_size=int(page_size),
            topk_pools=topk_pools,
            softmax_scale=softmax_scale,
            apply_relu=apply_relu,
            max_num_pools=max_num_pools,
            chunk_pools=chunk_pools,
            max_logits_bytes=max_logits_bytes,
        )
    return expand_kpool_to_flat_kv(
        pool_indices,
        causal_lens,
        req_ids,
        kv_block_table.to(device=device, dtype=torch.int32).contiguous(),
        pool_size=pool_size,
        kv_page_size=int(kv_page_size),
        append_tail=append_tail,
        out=out,
        lens_out=lens_out,
    )


__all__ = ["triton_dense_kpool_decode_topk", "triton_kpool_prefill_topk"]
