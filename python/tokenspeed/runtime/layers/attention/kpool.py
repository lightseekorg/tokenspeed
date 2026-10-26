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

"""KPool cache planning, writes, and sparse-history selection for DSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.ops.attention.kpool import (
    kpool_decode_append,
    kpool_decode_topk,
    kpool_prefill_tail_write,
    kpool_prefill_topk,
    kpool_prefill_write,
)

from tokenspeed.runtime.utils.env import global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext


@dataclass(frozen=True)
class KPoolWritePlan:
    """Physical pooled-index writes and request-local tail updates.

    Every layer writes the same rows from its own keys, so the gather maps are
    built once per forward: a completed pool takes its first
    ``n_from_tail`` keys from the request's tail buffer and the rest from the
    chunk (``pool_from_tail`` selects between the two gathers).
    """

    pool_write_slots: torch.Tensor
    pool_from_tail: torch.Tensor
    pool_tail_slots: torch.Tensor
    pool_tail_rows: torch.Tensor
    pool_chunk_rows: torch.Tensor
    tail_chunk_src: torch.Tensor
    tail_slots: torch.Tensor
    tail_dst_positions: torch.Tensor
    tail_write_counts: torch.Tensor


@dataclass(frozen=True)
class KPoolPrefillPlan:
    """Layer-invariant KPool cache-write and ragged-selection metadata."""

    write: KPoolWritePlan
    num_prefill_tokens: int
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    req_ids: torch.Tensor
    causal_lens: torch.Tensor
    pool_workspace_slots: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    max_num_pools: int
    seq_lens: torch.Tensor
    max_seq_len: int


@dataclass(frozen=True)
class KPoolPrefillTopK:
    """KPool selection output consumed by the generic DSA prefill path."""

    workspace_indices: torch.Tensor
    topk_lens: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_seq_lens: torch.Tensor
    max_seq_len: int
    kv_workspace_slots: torch.Tensor


def _segment_offsets(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each row's owner and its index within that owner.

    Owner ``i`` has ``counts[i]`` consecutive rows.
    """
    owners = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    first_rows = counts.cumsum(0) - counts
    return owners, torch.arange(owners.numel()) - first_rows[owners]


def _upload(device: torch.device, *arrays: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Copy host arrays to ``device`` in one pinned, non-blocking transfer.

    The forward thread must not wait on the stream (docs/design/event-loop.md):
    a copy from pageable memory would block it until the queued layers finish.
    Each array starts on a 16-byte boundary because Triton specializes pointer
    arguments on that alignment: an offset that moved with the batch shape
    would recompile the kernels reading the plan while serving.
    """
    if device.type == "cpu":
        return arrays
    sizes = [array.numel() * array.element_size() for array in arrays]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + (size + 15) // 16 * 16)
    staging = torch.empty(offsets[-1], dtype=torch.uint8, pin_memory=True)
    for array, offset, size in zip(arrays, offsets, sizes):
        staging[offset : offset + size] = array.reshape(-1).view(torch.uint8)
    staging = staging.to(device, non_blocking=True)
    return tuple(
        staging[offset : offset + size].view(array.dtype).view(array.shape)
        for array, offset, size in zip(arrays, offsets, sizes)
    )


def build_kpool_prefill_plan(
    *,
    prefix_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    index_block_table: torch.Tensor,
    request_slots: torch.Tensor,
    kpool: int,
    index_rows_per_page: int,
    tail_rows_per_request: int,
    token_capacity: int | None = None,
) -> KPoolPrefillPlan:
    """Build cache-write and ragged-selection metadata for a prefill batch.

    Args:
        prefix_lens_cpu: Per-request prefix lengths on CPU.
        extend_lens_cpu: Per-request extend lengths on CPU.
        index_block_table: Logical-pool-page to physical-index-page table.
        request_slots: Stable request-pool row for each batch request.
        kpool: Raw tokens represented by one compressed index row.
        index_rows_per_page: Compressed rows stored in one index page.
        tail_rows_per_request: Rows of each request's tail buffer.
        token_capacity: Optional fixed token-row capacity. Rows after the real
            prefill tokens are emitted as inactive padding metadata.

    Returns:
        Cache-write metadata plus the ragged selection workspace mapping. The
        per-token arrays come from host arithmetic over the length mirrors and
        reach the device in one pinned upload: the plan is built at the first
        sparse layer, behind queued GPU work, so it must not synchronize the
        stream.
    """
    if prefix_lens_cpu.device.type != "cpu" or extend_lens_cpu.device.type != "cpu":
        raise RuntimeError("KPool prefill plans require scheduler CPU length mirrors")
    starts = prefix_lens_cpu.to(torch.int64)
    lengths = extend_lens_cpu.to(torch.int64)
    if starts.numel() != lengths.numel():
        raise ValueError(
            "KPool prefix and extend length counts differ: "
            f"{starts.numel()} != {lengths.numel()}"
        )
    if (starts < 0).any() or (lengths < 0).any():
        raise ValueError(
            "KPool prefill lengths must be non-negative, got "
            f"prefix={starts.tolist()}, extend={lengths.tolist()}"
        )
    if request_slots.numel() != starts.numel():
        raise ValueError(
            "KPool request-slot count differs from the request count: "
            f"{request_slots.numel()} != {starts.numel()}"
        )
    num_prefill_tokens = int(lengths.sum())
    if token_capacity is None:
        token_capacity = num_prefill_tokens
    if token_capacity < num_prefill_tokens:
        raise ValueError(
            "KPool token capacity is smaller than the prefill token count: "
            f"capacity={token_capacity}, tokens={num_prefill_tokens}"
        )

    # Selection: each token's causal length and its request's workspace rows,
    # one row per logical pool the request has once this chunk is written.
    query_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    seq_lens = starts + lengths
    final_num_pools = seq_lens // kpool
    workspace_starts = final_num_pools.cumsum(0) - final_num_pools
    workspace_req_ids, workspace_pool_ids = _segment_offsets(final_num_pools)
    token_req_ids, token_offsets = _segment_offsets(lengths)
    token_causal_lens = starts[token_req_ids] + token_offsets + 1
    token_row_starts = workspace_starts[token_req_ids]
    # Rows past the live tokens are padding: an empty selection range and a
    # zero causal length keep selected-slot attention from reading the cache.
    padding = (0, token_capacity - num_prefill_tokens)
    positions, req_ids, causal_lens, row_starts, row_ends = (
        torch.nn.functional.pad(values, padding).to(torch.int32)
        for values in (
            token_causal_lens - 1,
            token_req_ids,
            token_causal_lens,
            token_row_starts,
            token_row_starts + token_causal_lens // kpool,
        )
    )

    # Writes: the pools this chunk completes (the first one finished from the
    # request's tail buffer), then the remainder left in the tail.
    begins = query_offsets[:-1]
    first_slot = starts % kpool
    base_pool = starts // kpool
    num_pools = final_num_pools - base_pool
    consumed = torch.where(num_pools > 0, num_pools * kpool - first_slot, 0)
    tail_counts = lengths - consumed
    pool_req_ids, pool_index = _segment_offsets(num_pools)
    pool_ids = base_pool[pool_req_ids] + pool_index
    first = pool_index == 0
    n_from_tail = torch.where(first, first_slot[pool_req_ids], 0)
    chunk_src = begins[pool_req_ids] + torch.where(
        first, 0, kpool - first_slot[pool_req_ids] + (pool_index - 1) * kpool
    )
    slot = torch.arange(kpool)
    pool_from_tail = slot < n_from_tail[:, None]
    pool_tail_rows = (pool_ids[:, None] * kpool + slot) % tail_rows_per_request
    pool_chunk_rows = (
        chunk_src[:, None] + (slot - n_from_tail[:, None]).clamp(min=0)
    ).reshape(-1)
    # A completed pool is also a selection-workspace row, which already holds
    # its physical index row.
    pool_workspace_rows = workspace_starts[pool_req_ids] + pool_ids
    has_tail = tail_counts > 0
    tail_req_ids = torch.nonzero(has_tail).flatten()
    tail_chunk_src = (begins + lengths - tail_counts)[has_tail]
    tail_dst_positions = (starts + consumed)[has_tail]
    tail_write_counts = tail_counts[has_tail].to(torch.int32)

    query_start_loc = query_offsets.to(torch.int32)
    request_seq_lens = seq_lens.to(torch.int32)
    (
        positions,
        req_ids,
        causal_lens,
        row_starts,
        row_ends,
        query_start_loc,
        request_seq_lens,
        workspace_req_ids,
        workspace_pool_ids,
        pool_workspace_rows,
        pool_req_ids,
        pool_from_tail,
        pool_tail_rows,
        pool_chunk_rows,
        tail_req_ids,
        tail_chunk_src,
        tail_dst_positions,
        tail_write_counts,
    ) = _upload(
        index_block_table.device,
        positions,
        req_ids,
        causal_lens,
        row_starts,
        row_ends,
        query_start_loc,
        request_seq_lens,
        workspace_req_ids,
        workspace_pool_ids,
        pool_workspace_rows,
        pool_req_ids,
        pool_from_tail,
        pool_tail_rows,
        pool_chunk_rows,
        tail_req_ids,
        tail_chunk_src,
        tail_dst_positions,
        tail_write_counts,
    )

    # Resolve request-relative ids against the device-side tables.
    pages = index_block_table[
        workspace_req_ids, workspace_pool_ids // index_rows_per_page
    ].to(torch.int64)
    pool_workspace_slots = (
        pages * index_rows_per_page + workspace_pool_ids % index_rows_per_page
    )
    request_slots = request_slots.to(device=index_block_table.device, dtype=torch.int64)
    return KPoolPrefillPlan(
        write=KPoolWritePlan(
            pool_write_slots=pool_workspace_slots.index_select(0, pool_workspace_rows),
            pool_from_tail=pool_from_tail,
            pool_tail_slots=request_slots.index_select(0, pool_req_ids),
            pool_tail_rows=pool_tail_rows,
            pool_chunk_rows=pool_chunk_rows,
            tail_chunk_src=tail_chunk_src,
            tail_slots=request_slots.index_select(0, tail_req_ids),
            tail_dst_positions=tail_dst_positions,
            tail_write_counts=tail_write_counts,
        ),
        num_prefill_tokens=num_prefill_tokens,
        positions=positions,
        query_start_loc=query_start_loc,
        req_ids=req_ids,
        causal_lens=causal_lens,
        pool_workspace_slots=pool_workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        max_num_pools=int(final_num_pools.max()) if final_num_pools.numel() else 0,
        seq_lens=request_seq_lens,
        max_seq_len=int(seq_lens.max()) if seq_lens.numel() else 0,
    )


class KPoolRuntime:
    """Own KPool-derived state and operations for a DSA backend."""

    def __init__(self, pool_size: int, index_topk: int) -> None:
        self.pool_size = pool_size
        self.index_topk = index_topk
        self.prefill_plan: KPoolPrefillPlan | None = None
        self.req_pool_indices: torch.Tensor | None = None

    def reset_forward(self, req_pool_indices: torch.Tensor | None = None) -> None:
        """Reset KPool state for a new forward."""
        self.prefill_plan = None
        self.req_pool_indices = req_pool_indices

    def ensure_prefill_plan(
        self,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
        token_capacity: int | None = None,
    ) -> None:
        """Build the layer-invariant prefill plan once for the current forward."""
        if ctx.num_extends <= 0 or self.prefill_plan is not None:
            return

        metadata = backend.chunked_prefill_metadata
        index_table = backend.kpool_prefill_page_table(ctx.num_extends)
        index_cache, tail_k, _ = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)
        # The slots come from the backend's per-forward publication, not the
        # prefill metadata: paged leaves' metadata carries no pool indices.
        if self.req_pool_indices is None:
            raise RuntimeError("DSA KPool prefill requires request-pool indices")
        self.prefill_plan = build_kpool_prefill_plan(
            prefix_lens_cpu=metadata.extend_prefix_lens_cpu[: ctx.num_extends],
            extend_lens_cpu=metadata.extend_seq_lens_cpu[: ctx.num_extends],
            index_block_table=index_table,
            request_slots=self.req_pool_indices[: ctx.num_extends],
            kpool=self.pool_size,
            index_rows_per_page=index_cache.shape[1],
            tail_rows_per_request=tail_k.shape[1],
            token_capacity=token_capacity,
        )

    def write_prefill(
        self,
        *,
        key: torch.Tensor,
        gate: torch.Tensor,
        compress_ape: torch.Tensor,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
    ) -> None:
        """Write completed prefill pools and preserve their incomplete tails."""
        pool = ctx.token_to_kv_pool
        index_cache, tail_k, tail_gate = pool.get_kpool_buffers(layer_id)
        shared_plan = self.prefill_plan
        if shared_plan is None:
            raise RuntimeError("DSA KPool prefill plan was not initialized")
        plan = shared_plan.write

        num_pools = plan.pool_write_slots.numel()
        if num_pools > 0:
            tail_slots = plan.pool_tail_slots.unsqueeze(1)
            tail_keys = tail_k[tail_slots, plan.pool_tail_rows]
            tail_scores = tail_gate[tail_slots, plan.pool_tail_rows]
            chunk_keys = key.index_select(0, plan.pool_chunk_rows).view(
                num_pools, self.pool_size, -1
            )
            chunk_scores = gate.index_select(0, plan.pool_chunk_rows).view(
                num_pools, self.pool_size, -1
            )
            from_tail = plan.pool_from_tail.unsqueeze(2)
            slot_keys = torch.where(from_tail, tail_keys, chunk_keys)
            slot_scores = torch.where(from_tail, tail_scores, chunk_scores)
            index_values, index_scales = pool.index_k_block_views(index_cache)
            kpool_prefill_write(
                slot_keys.contiguous(),
                slot_scores.contiguous(),
                plan.pool_write_slots,
                index_values,
                index_scales,
                compress_ape,
            )

        if plan.tail_write_counts.numel() > 0:
            kpool_prefill_tail_write(
                key,
                gate,
                tail_k,
                tail_gate,
                plan.tail_chunk_src,
                plan.tail_slots,
                plan.tail_dst_positions,
                plan.tail_write_counts,
                pool_size=self.pool_size,
            )

    def write_decode(
        self,
        *,
        key: torch.Tensor,
        gate: torch.Tensor,
        compress_ape: torch.Tensor,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
        num_reqs: int,
        q_len_per_req: int,
    ) -> None:
        """Append decode tokens to request tails and flush completed pools."""
        metadata = backend.forward_decode_metadata
        row_start = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[row_start : row_start + num_reqs].to(torch.int32)
        index_table = backend.kpool_decode_page_table(row_start, num_reqs)
        request_slots = self.req_pool_indices
        if request_slots is None:
            raise RuntimeError("DSA KPool decode requires request-pool indices")
        request_slots = request_slots[row_start : row_start + num_reqs].to(torch.int32)

        pool = ctx.token_to_kv_pool
        index_cache, tail_k, tail_gate = pool.get_kpool_buffers(layer_id)
        index_values, index_scales = pool.index_k_block_views(index_cache)
        keys = key.view(num_reqs, q_len_per_req, -1)
        gates = gate.view(num_reqs, q_len_per_req, -1)

        kpool_decode_append(
            keys,
            gates,
            tail_k,
            tail_gate,
            seq_lens,
            request_slots,
            index_table,
            index_values,
            index_scales,
            compress_ape,
        )

    def select_decode(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        softmax_scale: float,
        ctx: ForwardContext,
        layer_id: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        q_len_per_req: int,
        decode_start: int,
        num_decode_tokens: int,
        out: torch.Tensor,
        lens_out: torch.Tensor,
    ) -> None:
        """Select decode history slots into caller-owned reusable workspaces."""
        history_table = page_table[: seq_lens.numel()]
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        kpool_decode_topk(
            query[decode_start : decode_start + num_decode_tokens].contiguous(),
            index_cache,
            weights[decode_start : decode_start + num_decode_tokens],
            seq_lens,
            history_table,
            history_table,
            pool_size=self.pool_size,
            page_size=index_cache.shape[1],
            kv_page_size=ctx.token_to_kv_pool.arena.kv_page_size,
            topk_pools=self.index_topk // self.pool_size,
            softmax_scale=softmax_scale,
            q_len_per_req=q_len_per_req,
            max_seq_len=ctx.attn_backend.max_context_len,
            out=out[decode_start : decode_start + num_decode_tokens],
            lens_out=lens_out[decode_start : decode_start + num_decode_tokens],
        )

    def select_prefill(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        softmax_scale: float,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
        num_prefill_tokens: int,
    ) -> KPoolPrefillTopK | None:
        """Select causal prefill history and return the generic DSA inputs."""
        if ctx.num_extends == 0:
            return None
        history_table = backend.kpool_prefill_page_table(ctx.num_extends)
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        shared_plan = self.prefill_plan
        if shared_plan is None:
            raise RuntimeError("DSA KPool prefill plan was not initialized")
        positions = shared_plan.positions
        if shared_plan.num_prefill_tokens != num_prefill_tokens:
            raise RuntimeError(
                "DSA KPool plan token count mismatch: "
                f"plan={shared_plan.num_prefill_tokens}, tokens={num_prefill_tokens}"
            )
        if positions.numel() != query.shape[0]:
            raise RuntimeError(
                "DSA KPool plan capacity mismatch: "
                f"plan={positions.numel()}, query_rows={query.shape[0]}"
            )
        selected_indices, selected_lens = kpool_prefill_topk(
            query.contiguous(),
            index_cache,
            weights,
            positions,
            shared_plan.query_start_loc,
            history_table,
            history_table,
            pool_size=self.pool_size,
            page_size=index_cache.shape[1],
            kv_page_size=ctx.token_to_kv_pool.arena.kv_page_size,
            topk_pools=self.index_topk // self.pool_size,
            softmax_scale=softmax_scale,
            req_ids=shared_plan.req_ids,
            causal_lens=shared_plan.causal_lens,
            pool_workspace_slots=shared_plan.pool_workspace_slots,
            row_starts=shared_plan.row_starts,
            row_ends=shared_plan.row_ends,
            max_num_pools=shared_plan.max_num_pools,
            max_logits_bytes=max(
                1,
                int(
                    global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"]
                ),
            )
            * 1024
            * 1024,
        )
        workspace_indices = torch.arange(
            selected_indices.numel(),
            dtype=torch.int32,
            device=selected_indices.device,
        ).view_as(selected_indices)
        workspace_indices.masked_fill_(selected_indices < 0, -1)

        return KPoolPrefillTopK(
            workspace_indices=workspace_indices,
            topk_lens=selected_lens,
            page_table=history_table,
            seq_lens=shared_plan.seq_lens,
            kv_seq_lens=shared_plan.causal_lens,
            max_seq_len=shared_plan.max_seq_len,
            kv_workspace_slots=selected_indices.reshape(-1).contiguous(),
        )
