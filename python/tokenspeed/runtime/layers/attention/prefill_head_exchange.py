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

"""Operator-level QCP via query/head exchange around replicated-KV prefill attention."""

import torch
from tokenspeed_kernel.ops.attention import dsv41

from tokenspeed.runtime.distributed.comm_ops import (
    all_gather,
    all_to_all_head_scatter,
    all_to_all_single,
    all_to_all_transpose,
)
from tokenspeed.runtime.execution.query_shard import scatter_count


def should_exchange_prefill_heads(
    *,
    num_queries: int,
    local_heads: int,
    group_size: int,
    min_queries: int,
    swa_min_queries: int,
    has_global: bool,
) -> bool:
    """Apply an explicit request threshold and the current head-layout contract.

    This is an eligibility gate, not a hardware performance prediction.
    Require one real query per participating rank before exchanging heads.
    """
    threshold = min_queries if has_global else max(min_queries, swa_min_queries)
    return (
        group_size > 1
        and num_queries >= max(threshold, group_size)
        and local_heads * group_size == 64
    )


def selected_prefill_with_head_exchange(
    q: torch.Tensor,
    workspace: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor,
    *,
    softmax_scale: float,
    query_chunk_size: int,
    group: tuple[int, ...],
    rank: int,
) -> torch.Tensor:
    """Return the original TP head layout after attention on local query rows.

    Each rank supplies every query with its own head block. Workspace rows and
    indices must identify the same logical KV values on all ranks; physical
    cache addresses never cross this boundary. Sink padding is not a head.
    """
    tokens, heads, dim = q.shape
    counts = scatter_count(tokens, len(group))
    start = sum(counts[:rank])
    local_q = all_to_all_transpose(
        q.reshape(tokens, heads * dim), group, counts, backend=None
    ).reshape(counts[rank], len(group) * heads, dim)
    sink = all_gather(attn_sink[:heads].contiguous(), group, dim=0, backend=None)
    output = dsv41.selected_attention(
        local_q,
        None,
        None,
        None,
        None,
        None,
        None,
        sink,
        softmax_scale,
        None,
        query_chunk_size,
        None,
        workspace,
        indices[start : start + counts[rank]].contiguous(),
        lse=None,
    )
    return all_to_all_head_scatter(output, group, counts, backend=None)


def exchange_query_rows_padded(
    q: torch.Tensor,
    group: tuple[int, ...],
    rank: int,
    counts: list[int],
    padded_heads: int,
) -> torch.Tensor:
    """Give each rank its own rows with every rank's heads, zero-padded.

    Pure data movement: the query's bytes cross the group whatever its dtype,
    and the received head blocks land in the padded query with one copy.

    Args:
        q: This rank's ``[tokens, heads, dim]`` head block of every row, the
            rows rank-major (``counts[i]`` rows belong to the ``i``-th rank).
        group: The ranks sharing the rows, every one holding the same rows.
        rank: This rank's index in ``group``.
        counts: Rows per rank, in group order; every rank holds the same.
        padded_heads: The core's head count, at least the group's heads.

    Returns:
        ``[counts[rank], padded_heads, dim]``: rank ``i``'s heads at
        ``i * heads`` onward, zeros after the group's heads.
    """
    tokens, heads, dim = q.shape
    world = len(group)
    if padded_heads < world * heads:
        raise ValueError("padded query heads must hold every rank's heads")
    rows = counts[rank]
    width = heads * dim * q.element_size()
    received = torch.empty((world * rows, width), dtype=torch.uint8, device=q.device)
    all_to_all_single(
        received,
        q.contiguous().view(torch.uint8).view(tokens, width),
        group,
        output_split_sizes=[rows] * world,
        input_split_sizes=counts,
    )
    padded = q.new_empty((rows, padded_heads, dim))
    target = padded.view(torch.uint8).view(rows, -1)
    source = received.view(world, rows, width)
    if width % 8 == 0:
        # Copy whole 8-byte words: a strided byte copy of an FP8 query is far slower.
        target, source, width = (
            target.view(torch.int64),
            source.view(torch.int64),
            width // 8,
        )
    target[:, world * width :].zero_()
    target[:, : world * width].view(rows, world, width).copy_(source.transpose(0, 1))
    return padded


def return_head_rows(
    out: torch.Tensor, heads: int, group: tuple[int, ...], counts: list[int]
) -> torch.Tensor:
    """Hand each rank its head block of every row back: the inverse exchange.

    Args:
        out: This rank's ``[counts[rank], padded_heads, dim]`` core output,
            rank ``i``'s heads at ``i * heads`` onward and padding after.
        heads: Heads per rank.
        group: The ranks sharing the rows.
        counts: Rows per rank, in group order.

    Returns:
        ``[sum(counts), heads, dim]``: this rank's heads of every row, the
        rows rank-major.
    """
    rows, _, dim = out.shape
    world = len(group)
    sent = out.new_empty((world, rows, heads, dim))
    sent.copy_(out[:, : world * heads].unflatten(1, (world, heads)).transpose(0, 1))
    received = out.new_empty((sum(counts), heads * dim))
    all_to_all_single(
        received,
        sent.view(world * rows, heads * dim),
        group,
        output_split_sizes=counts,
        input_split_sizes=[rows] * world,
    )
    return received.view(-1, heads, dim)
