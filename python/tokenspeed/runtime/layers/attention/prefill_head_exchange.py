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
