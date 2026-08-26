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
from tokenspeed_kernel.ops.attention import (
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
    """Physical pooled-index writes and request-local tail updates."""

    pool_req_ids: torch.Tensor
    pool_n_from_tail: torch.Tensor
    pool_chunk_src: torch.Tensor
    pool_tail_logical_base: torch.Tensor
    pool_write_slots: torch.Tensor
    tail_req_ids: torch.Tensor
    tail_chunk_src: torch.Tensor
    tail_dst_positions: torch.Tensor
    tail_write_counts: torch.Tensor
    request_slots: torch.Tensor


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


def dsa_prefill_host_lengths(metadata: Any, num_extends: int) -> tuple[int, int]:
    """Return prefill token count and maximum sequence length from CPU mirrors."""
    prefix = metadata.extend_prefix_lens_cpu[:num_extends]
    extend = metadata.extend_seq_lens_cpu[:num_extends]
    if prefix.device.type != "cpu" or extend.device.type != "cpu":
        raise RuntimeError("DSA prefill length mirrors must remain on CPU")
    prefix_lens = [int(value) for value in prefix]
    extend_lens = [int(value) for value in extend]
    return sum(extend_lens), max(
        (p + e for p, e in zip(prefix_lens, extend_lens)), default=0
    )


def kv_page_size(ctx: ForwardContext) -> int:
    """Return the shared cache identity grain from the recipe-planned arena."""
    return ctx.token_to_kv_pool.arena.kv_page_size


def build_kpool_write_plan(
    *,
    req_start_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    index_block_table: torch.Tensor,
    request_slots: torch.Tensor,
    kpool: int,
    index_rows_per_page: int,
) -> KPoolWritePlan:
    """Map completed pools to index pages and remainders to request tail slots."""
    starts = req_start_positions.to(torch.int64).tolist()
    offsets = query_start_loc.to(torch.int64).tolist()
    if request_slots.numel() != len(starts):
        raise ValueError(
            "KPool request-slot count differs from the request count: "
            f"{request_slots.numel()} != {len(starts)}"
        )

    pool_req_ids: list[int] = []
    pool_n_from_tail: list[int] = []
    pool_chunk_src: list[int] = []
    pool_tail_logical_base: list[int] = []
    pool_ids: list[int] = []
    tail_req_ids: list[int] = []
    tail_chunk_src: list[int] = []
    tail_dst_positions: list[int] = []
    tail_write_counts: list[int] = []

    for req_id, (start, chunk_begin, chunk_end) in enumerate(
        zip(starts, offsets, offsets[1:])
    ):
        length = chunk_end - chunk_begin
        if length <= 0:
            continue

        first_slot = start % kpool
        base_pool = start // kpool
        num_pools = (start + length) // kpool - base_pool
        consumed = num_pools * kpool - first_slot if num_pools else 0
        tail_count = length - consumed

        for pool_index in range(num_pools):
            pool_id = base_pool + pool_index
            n_from_tail = first_slot if pool_index == 0 else 0
            src_local = (
                0
                if pool_index == 0
                else (kpool - first_slot) + (pool_index - 1) * kpool
            )
            pool_req_ids.append(req_id)
            pool_n_from_tail.append(n_from_tail)
            pool_chunk_src.append(chunk_begin + src_local)
            pool_tail_logical_base.append(pool_id * kpool)
            pool_ids.append(pool_id)

        if tail_count:
            tail_req_ids.append(req_id)
            tail_chunk_src.append(chunk_begin + length - tail_count)
            tail_dst_positions.append(start + consumed)
            tail_write_counts.append(tail_count)

    device = index_block_table.device
    pool_req_ids_tensor = torch.tensor(pool_req_ids, dtype=torch.int32, device=device)
    pool_ids_tensor = torch.tensor(pool_ids, dtype=torch.int64, device=device)
    if pool_ids_tensor.numel() > 0:
        table_columns = torch.div(
            pool_ids_tensor,
            index_rows_per_page,
            rounding_mode="floor",
        )
        write_pages = index_block_table[
            pool_req_ids_tensor.to(torch.int64), table_columns
        ].to(torch.int64)
        pool_write_slots = write_pages * index_rows_per_page + torch.remainder(
            pool_ids_tensor, index_rows_per_page
        )
    else:
        pool_write_slots = torch.empty(0, dtype=torch.int64, device=device)
    return KPoolWritePlan(
        pool_req_ids=pool_req_ids_tensor,
        pool_n_from_tail=torch.tensor(
            pool_n_from_tail, dtype=torch.int32, device=device
        ),
        pool_chunk_src=torch.tensor(pool_chunk_src, dtype=torch.int64, device=device),
        pool_tail_logical_base=torch.tensor(
            pool_tail_logical_base, dtype=torch.int64, device=device
        ),
        pool_write_slots=pool_write_slots,
        tail_req_ids=torch.tensor(tail_req_ids, dtype=torch.int32, device=device),
        tail_chunk_src=torch.tensor(tail_chunk_src, dtype=torch.int64, device=device),
        tail_dst_positions=torch.tensor(
            tail_dst_positions, dtype=torch.int64, device=device
        ),
        tail_write_counts=torch.tensor(
            tail_write_counts, dtype=torch.int32, device=device
        ),
        request_slots=request_slots.to(device=device, dtype=torch.int64),
    )


def build_kpool_prefill_plan(
    *,
    prefix_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    index_block_table: torch.Tensor,
    request_slots: torch.Tensor,
    kpool: int,
    index_rows_per_page: int,
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
        token_capacity: Optional fixed token-row capacity. Rows after the real
            prefill tokens are emitted as inactive padding metadata.

    Returns:
        Cache-write metadata plus the ragged selection workspace mapping.
    """
    if prefix_lens_cpu.device.type != "cpu" or extend_lens_cpu.device.type != "cpu":
        raise RuntimeError("KPool prefill plans require scheduler CPU length mirrors")
    starts = [int(value) for value in prefix_lens_cpu]
    lengths = [int(value) for value in extend_lens_cpu]
    if len(starts) != len(lengths):
        raise ValueError(
            "KPool prefix and extend length counts differ: "
            f"{len(starts)} != {len(lengths)}"
        )

    query_offsets = [0]
    positions: list[int] = []
    req_ids: list[int] = []
    causal_lens: list[int] = []
    workspace_req_ids: list[int] = []
    workspace_pool_ids: list[int] = []
    row_starts: list[int] = []
    row_ends: list[int] = []
    workspace_start = 0
    max_num_pools = 0

    for req_id, (start, length) in enumerate(zip(starts, lengths)):
        if start < 0 or length < 0:
            raise ValueError(
                "KPool prefill lengths must be non-negative, got "
                f"prefix={start}, extend={length} for request {req_id}"
            )
        query_offsets.append(query_offsets[-1] + length)
        final_num_pools = (start + length) // kpool
        max_num_pools = max(max_num_pools, final_num_pools)
        workspace_req_ids.extend([req_id] * final_num_pools)
        workspace_pool_ids.extend(range(final_num_pools))
        for offset in range(length):
            causal = start + offset + 1
            positions.append(causal - 1)
            req_ids.append(req_id)
            causal_lens.append(causal)
            row_starts.append(workspace_start)
            row_ends.append(workspace_start + causal // kpool)
        workspace_start += final_num_pools

    num_prefill_tokens = query_offsets[-1]
    if token_capacity is None:
        token_capacity = num_prefill_tokens
    token_capacity = int(token_capacity)
    if token_capacity < num_prefill_tokens:
        raise ValueError(
            "KPool token capacity is smaller than the prefill token count: "
            f"capacity={token_capacity}, tokens={num_prefill_tokens}"
        )
    num_padding_tokens = token_capacity - num_prefill_tokens
    if num_padding_tokens:
        # Empty ranges make selection a no-op. A zero causal length also keeps
        # selected-slot attention from reading cache entries for these rows.
        positions.extend([0] * num_padding_tokens)
        req_ids.extend([0] * num_padding_tokens)
        causal_lens.extend([0] * num_padding_tokens)
        row_starts.extend([0] * num_padding_tokens)
        row_ends.extend([0] * num_padding_tokens)

    device = index_block_table.device
    workspace_req_ids_tensor = torch.tensor(
        workspace_req_ids, dtype=torch.int64, device=device
    )
    workspace_pool_ids_tensor = torch.tensor(
        workspace_pool_ids, dtype=torch.int64, device=device
    )
    if workspace_pool_ids_tensor.numel() > 0:
        table_columns = torch.div(
            workspace_pool_ids_tensor,
            index_rows_per_page,
            rounding_mode="floor",
        )
        workspace_pages = index_block_table[workspace_req_ids_tensor, table_columns].to(
            torch.int64
        )
        pool_workspace_slots = workspace_pages * index_rows_per_page + torch.remainder(
            workspace_pool_ids_tensor, index_rows_per_page
        )
    else:
        pool_workspace_slots = torch.empty(0, dtype=torch.int64, device=device)

    query_start_loc_cpu = torch.tensor(query_offsets, dtype=torch.int64)
    write = build_kpool_write_plan(
        req_start_positions=prefix_lens_cpu,
        query_start_loc=query_start_loc_cpu,
        index_block_table=index_block_table,
        request_slots=request_slots,
        kpool=kpool,
        index_rows_per_page=index_rows_per_page,
    )
    return KPoolPrefillPlan(
        write=write,
        num_prefill_tokens=num_prefill_tokens,
        positions=torch.tensor(positions, dtype=torch.int32, device=device),
        query_start_loc=query_start_loc_cpu.to(device=device, dtype=torch.int32),
        req_ids=torch.tensor(req_ids, dtype=torch.int32, device=device),
        causal_lens=torch.tensor(causal_lens, dtype=torch.int32, device=device),
        pool_workspace_slots=pool_workspace_slots.contiguous(),
        row_starts=torch.tensor(row_starts, dtype=torch.int32, device=device),
        row_ends=torch.tensor(row_ends, dtype=torch.int32, device=device),
        max_num_pools=max_num_pools,
    )


class KPoolRuntime:
    """Own KPool-derived state and operations for a DSA backend."""

    def __init__(self, pool_size: int, index_topk: int) -> None:
        self.pool_size = pool_size
        self.index_topk = index_topk

    def ensure_prefill_plan(
        self,
        ctx: ForwardContext,
        backend: Any,
        layer_id: int,
        token_capacity: int | None = None,
    ) -> None:
        """Build the layer-invariant prefill plan once for the current forward."""
        if (
            ctx.num_extends <= 0
            or backend.forward_metadata.kpool_prefill_plan is not None
        ):
            return

        metadata = backend.chunked_prefill_metadata
        index_table = backend.kpool_prefill_page_table(ctx.num_extends)
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        backend.forward_metadata.kpool_prefill_plan = build_kpool_prefill_plan(
            prefix_lens_cpu=metadata.extend_prefix_lens_cpu[: ctx.num_extends],
            extend_lens_cpu=metadata.extend_seq_lens_cpu[: ctx.num_extends],
            index_block_table=index_table,
            request_slots=metadata.req_pool_indices[: ctx.num_extends],
            kpool=self.pool_size,
            index_rows_per_page=index_cache.shape[1],
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
        shared_plan = backend.forward_metadata.kpool_prefill_plan
        if shared_plan is None:
            raise RuntimeError("DSA KPool prefill plan was not initialized")
        plan = shared_plan.write

        num_pools = plan.pool_write_slots.numel()
        if num_pools > 0:
            offsets = torch.arange(
                self.pool_size,
                dtype=torch.int64,
                device=key.device,
            ).unsqueeze(0)
            from_tail = offsets < plan.pool_n_from_tail.to(torch.int64).unsqueeze(1)
            tail_rows = torch.remainder(
                plan.pool_tail_logical_base.unsqueeze(1) + offsets,
                tail_k.shape[1],
            )
            request_slots = plan.request_slots.index_select(
                0, plan.pool_req_ids.to(torch.int64)
            )
            tail_keys = tail_k[request_slots.unsqueeze(1), tail_rows]
            tail_scores = tail_gate[request_slots.unsqueeze(1), tail_rows]
            chunk_rows = plan.pool_chunk_src.unsqueeze(1) + torch.clamp(
                offsets - plan.pool_n_from_tail.to(torch.int64).unsqueeze(1),
                min=0,
            )
            chunk_keys = key.index_select(0, chunk_rows.reshape(-1)).view(
                num_pools, self.pool_size, -1
            )
            chunk_scores = gate.index_select(0, chunk_rows.reshape(-1)).view(
                num_pools, self.pool_size, -1
            )
            slot_keys = torch.where(from_tail.unsqueeze(2), tail_keys, chunk_keys)
            slot_scores = torch.where(from_tail.unsqueeze(2), tail_scores, chunk_scores)
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
            destination_slots = plan.request_slots.index_select(
                0, plan.tail_req_ids.to(torch.int64)
            )
            kpool_prefill_tail_write(
                key,
                gate,
                tail_k,
                tail_gate,
                plan.tail_chunk_src,
                destination_slots,
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
        request_slots = backend.forward_metadata.req_pool_indices
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
            kv_page_size=kv_page_size(ctx),
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
        metadata = backend.chunked_prefill_metadata
        prefix_lens = metadata.extend_prefix_lens[: ctx.num_extends].to(torch.int32)
        extend_lens = metadata.extend_seq_lens[: ctx.num_extends].to(torch.int32)
        seq_lens = prefix_lens + extend_lens
        if seq_lens.numel() == 0:
            return None
        host_token_count, max_seq_len = dsa_prefill_host_lengths(
            metadata, ctx.num_extends
        )
        if host_token_count != num_prefill_tokens:
            raise RuntimeError(
                "DSA KPool prefill token count mismatch: "
                f"metadata={host_token_count}, tokens={num_prefill_tokens}"
            )

        history_table = backend.kpool_prefill_page_table(ctx.num_extends)
        index_cache = ctx.token_to_kv_pool.get_kpool_buffers(layer_id)[0]
        shared_plan = backend.forward_metadata.kpool_prefill_plan
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
            kv_page_size=kv_page_size(ctx),
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
            seq_lens=seq_lens,
            kv_seq_lens=shared_plan.causal_lens,
            max_seq_len=max_seq_len,
            kv_workspace_slots=selected_indices.reshape(-1).contiguous(),
        )
